// SPDX-License-Identifier: Apache-2.0
//! Simple utility to stitch together pipeline stages defined in DSLX.

use std::collections::BTreeMap;
use std::path::Path;

use xlsynth::dslx::{Function, MatchableModuleMember};
use xlsynth::{
    DslxCallingConvention, mangle_dslx_name, mangle_dslx_name_with_calling_convention, optimize_ir,
    schedule_and_codegen,
};
use xlsynth_vast::{ModulePortDirection, VastFile, VastFileType};

mod common;
use common::{PipelineCfg, Port, StageInfo};

mod build_pipeline;
use build_pipeline::{
    PipelineConfig as BuildPipelineConfig, StageModuleInterface, StagePortInterface,
    build_pipeline as build_wrapper,
};

pub mod g8r_pipeline;
pub mod validate_constant_flops;

mod equiv;
mod names;
mod spec;
pub use spec::{ResolvedStitchPipeline, StitchPipelineSpec};

use crate::aig::{ClockPort, SequentialPipelineOptions};

use crate::verilog_version::VerilogVersion;

/// Options for generating combinational stages and their pipeline wrapper.
#[derive(Debug)]
pub struct StitchPipelineOptions<'a> {
    pub verilog_version: VerilogVersion,
    pub explicit_stages: Option<Vec<String>>,
    pub stdlib_path: Option<&'a Path>,
    pub search_paths: Vec<&'a Path>,
    pub flop_inputs: bool,
    pub flop_outputs: bool,
    /// Reject registered stage-output bits that are constant over all pipeline
    /// inputs.
    pub validate_no_constant_flops: bool,
    pub input_valid_signal: Option<&'a str>,
    pub output_valid_signal: Option<&'a str>,
    pub reset_signal: Option<&'a str>,
    pub reset_active_low: bool,
    pub add_invariant_assertions: bool,
    pub array_index_bounds_checking: bool,
    /// Optional Verilog template for each stage's gate! operations.
    pub gate_format: Option<&'a str>,
    pub output_module_name: &'a str,
}

/// RTL emission settings for an already resolved pipeline.
#[derive(Clone, Debug)]
pub struct StitchRtlOptions<'a> {
    pub verilog_version: VerilogVersion,
    pub validate_no_constant_flops: bool,
    pub add_invariant_assertions: bool,
    pub array_index_bounds_checking: bool,
    pub gate_format: Option<&'a str>,
}

impl Default for StitchRtlOptions<'_> {
    fn default() -> Self {
        Self {
            verilog_version: VerilogVersion::SystemVerilog,
            validate_no_constant_flops: false,
            add_invariant_assertions: false,
            array_index_bounds_checking: true,
            gate_format: None,
        }
    }
}

impl StitchPipelineOptions<'_> {
    fn boundary_options(&self) -> SequentialPipelineOptions {
        SequentialPipelineOptions {
            name: self.output_module_name.to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: self.flop_inputs,
            flop_outputs: self.flop_outputs,
            input_valid_signal: self.input_valid_signal.map(str::to_string),
            output_valid_signal: self.output_valid_signal.map(str::to_string),
            reset_signal: self.reset_signal.map(str::to_string),
            reset_active_low: self.reset_active_low,
        }
    }
}

fn build_ports_from_ir(
    func: &xlsynth::ir_package::IrFunction,
    fty: &xlsynth::ir_package::IrFunctionType,
) -> Result<(Vec<Port>, u32), xlsynth::XlsynthError> {
    let mut ports = Vec::new();

    // Formal parameters (flattened widths). Combinational stage modules do not
    // have synthetic clock ports, so every input here comes directly from the
    // selected DSLX function.
    for i in 0..fty.param_count() {
        let name = func.param_name(i)?;
        let ty = fty.param_type(i)?;
        let width = ty.get_flat_bit_count() as u32;
        if width == 0 {
            return Err(xlsynth::XlsynthError(format!(
                "stage parameter `{}` has zero-width type `{}`; stitch-pipeline cannot emit it as a Verilog port",
                name, ty
            )));
        }
        ports.push(Port {
            name,
            is_input: true,
            width,
        });
    }

    // Return value -> `out` port.
    let ret_ty = fty.return_type();
    let ret_width = ret_ty.get_flat_bit_count() as u32;
    if ret_width == 0 {
        return Err(xlsynth::XlsynthError(format!(
            "stage return type `{}` is zero-width; stitch-pipeline cannot emit it as Verilog output port `out`",
            ret_ty
        )));
    }
    ports.push(Port {
        name: "out".into(),
        is_input: false,
        width: ret_width,
    });

    Ok((ports, ret_width))
}

/// Resolves every selected DSLX stage to an IR function with a data interface.
fn resolve_stage_ir_names(
    ir: &xlsynth::ir_package::IrPackage,
    module_name: &str,
    names: Vec<String>,
) -> Result<Vec<(String, String)>, xlsynth::XlsynthError> {
    let mut stages = Vec::with_capacity(names.len());
    for name in names {
        let mangled = mangle_dslx_name(module_name, &name)?;
        if ir.get_function(&mangled).is_err() {
            let token_name = mangle_dslx_name_with_calling_convention(
                module_name,
                &name,
                DslxCallingConvention::ImplicitToken,
            )?;
            if ir.get_function(&token_name).is_ok() {
                return Err(xlsynth::XlsynthError(format!(
                    "stage '{name}' uses the implicit-token calling convention and has no data-only IR entry point. Declare the stage pub to emit a wrapper for its packed data ports"
                )));
            }
            return Err(xlsynth::XlsynthError(format!(
                "DSLX stage '{name}' in module '{module_name}' has no IR function '{mangled}'"
            )));
        }
        stages.push((name, mangled));
    }
    Ok(stages)
}

/// Ensure adjacent stage signatures dovetail correctly.
fn verify_stage_signatures(
    ir: &xlsynth::ir_package::IrPackage,
    stages: &[(String, String)],
) -> Result<(), xlsynth::XlsynthError> {
    for pair in stages.windows(2) {
        let (prev_name, prev_mangled) = &pair[0];
        let (next_name, next_mangled) = &pair[1];

        let prev_fn = ir.get_function(prev_mangled)?;
        let prev_ty = prev_fn.get_type()?;
        let prev_ret = prev_ty.return_type();
        let prev_ret_str = format!("{}", prev_ret);

        let next_fn = ir.get_function(next_mangled)?;
        let next_ty = next_fn.get_type()?;
        let next_param_str = if next_ty.param_count() == 1 {
            format!("{}", next_ty.param_type(0)?)
        } else {
            let mut parts = Vec::new();
            for i in 0..next_ty.param_count() {
                parts.push(format!("{}", next_ty.param_type(i)?));
            }
            format!("({})", parts.join(", "))
        };

        if prev_ret_str != next_param_str {
            return Err(xlsynth::XlsynthError(format!(
                "output type of stage '{}' ({}) does not match input{} of stage '{}' ({})",
                prev_name,
                prev_ret_str,
                if next_ty.param_count() == 1 { "" } else { "s" },
                next_name,
                next_param_str
            )));
        }
    }
    Ok(())
}

/// Reject stage interfaces that cannot be represented as Verilog ports.
fn verify_stage_port_widths(
    ir: &xlsynth::ir_package::IrPackage,
    stages: &[(String, String)],
) -> Result<(), xlsynth::XlsynthError> {
    for (stage_name, stage_mangled) in stages {
        let func = ir.get_function(stage_mangled)?;
        let fty = func.get_type()?;
        for i in 0..fty.param_count() {
            let param_name = func.param_name(i)?;
            let param_ty = fty.param_type(i)?;
            if param_ty.get_flat_bit_count() == 0 {
                return Err(xlsynth::XlsynthError(format!(
                    "stage parameter `{}.{}` has zero-width type `{}`; stitch-pipeline cannot emit it as a Verilog port",
                    stage_name, param_name, param_ty
                )));
            }
        }
        let ret_ty = fty.return_type();
        if ret_ty.get_flat_bit_count() == 0 {
            return Err(xlsynth::XlsynthError(format!(
                "stage `{}` has zero-width return type `{}`; stitch-pipeline cannot emit it as Verilog output port `out`",
                stage_name, ret_ty
            )));
        }
    }
    Ok(())
}

/// Optimizes and emits a combinational stage with the requested codegen
/// options.
fn make_stage_info_comb(
    cfg: &PipelineCfg,
    stage_name_unmangled: &str,
    stage_mangled: &str,
    _is_last: bool,
) -> Result<StageInfo, xlsynth::XlsynthError> {
    let opt = optimize_ir(cfg.ir, stage_mangled)?;
    let sched = "delay_model: \"unit\"\npipeline_stages: 1";
    // Use the XLS "combinational" generator so the resulting module has *no*
    // clock.
    let mut codegen = format!(
        r#"register_merge_strategy: STRATEGY_IDENTITY_ONLY
generator: GENERATOR_KIND_COMBINATIONAL
module_name: "{stage}"
use_system_verilog: {sv}"#,
        stage = stage_name_unmangled,
        sv = cfg.verilog_version.is_system_verilog(),
    );
    // Explicitly set the invariant-assertion option regardless of the
    // requested value so that the behaviour is *deterministic* – we do not
    // rely on the generator’s implicit default. When
    // `cfg.add_invariant_assertions` is `false` we now emit
    // `add_invariant_assertions: false`, preventing unexpected assertions
    // from being injected (see xlsynth-driver/tests/invoke_test.rs).
    codegen.push_str(&format!(
        "\nadd_invariant_assertions: {}",
        cfg.add_invariant_assertions
    ));
    // Emit array-index bounds checking option explicitly for determinism.
    codegen.push_str(&format!(
        "\narray_index_bounds_checking: {}",
        cfg.array_index_bounds_checking
    ));
    if let Some(gate_format) = cfg.gate_format {
        codegen.push_str(&format!("\ngate_format: {gate_format:?}"));
    }
    let result = schedule_and_codegen(&opt, sched, &codegen)?;
    let sv_text = result.get_verilog_text()?;

    let func = cfg.ir.get_function(stage_mangled)?;
    let fty = func.get_type()?;
    let (ports, output_width) = build_ports_from_ir(&func, &fty)?;

    Ok(StageInfo {
        sv_text,
        ports,
        output_width,
    })
}

/// Selects every numbered DSLX stage, requiring contiguous canonical indices.
fn implicit_stage_names(
    functions: &BTreeMap<String, Function>,
    top: &str,
) -> Result<Vec<String>, xlsynth::XlsynthError> {
    let mut indices = Vec::new();
    let prefix = format!("{top}_cycle");
    for name in functions.keys() {
        if let Some(digits) = name.strip_prefix(&prefix)
            && !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
        {
            let idx = digits.parse::<usize>().map_err(|_| {
                xlsynth::XlsynthError(format!(
                    "stage '{name}' has an index too large for implicit stage selection"
                ))
            })?;
            if digits.len() > 1 && digits.starts_with('0') {
                return Err(xlsynth::XlsynthError(format!(
                    "stage '{name}' has a leading zero in its index; implicit stage selection expects '{prefix}{idx}'"
                )));
            }
            indices.push(idx);
        }
    }

    if indices.is_empty() {
        return Err(xlsynth::XlsynthError(format!(
            "no DSLX stage functions found for dslx_top '{top}'; expected '{prefix}0'"
        )));
    }

    indices.sort_unstable();
    if indices.iter().copied().ne(0..indices.len()) {
        let found = indices
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(xlsynth::XlsynthError(format!(
            "found stage function(s) named like '{top}_cycleN' but numbering must start at 0 and be contiguous; found indices: {found}"
        )));
    }

    Ok(indices.iter().map(|idx| format!("{prefix}{idx}")).collect())
}

/// Selects concrete functions declared in the source module before IR
/// conversion can introduce generated names for helpers and specializations.
fn select_stage_names(
    tc_module: &xlsynth::dslx::TypecheckedModule,
    top: &str,
    explicit_stages: Option<&[String]>,
) -> Result<Vec<String>, xlsynth::XlsynthError> {
    let module = tc_module.get_module();
    let mut functions = BTreeMap::new();
    for i in 0..module.get_member_count() {
        let member = module.get_member(i);
        if let Some(MatchableModuleMember::Function(func)) = member.to_matchable() {
            functions.insert(func.get_identifier(), func);
        }
    }
    let stages = match explicit_stages {
        Some([]) => return Err(xlsynth::XlsynthError("stages must be nonempty".into())),
        Some(names) => names.to_vec(),
        None => implicit_stage_names(&functions, top)?,
    };
    let mut offending = Vec::new();
    for name in &stages {
        let func = functions.get(name).ok_or_else(|| {
            xlsynth::XlsynthError(format!(
                "selected stage {name:?} is not a declared DSLX function in module '{}'",
                module.get_name()
            ))
        })?;
        if func.is_parametric() {
            offending.push(name.as_str());
        }
    }

    if !offending.is_empty() {
        offending.sort();
        return Err(xlsynth::XlsynthError(format!(
            "parametric stage function(s) detected that cannot be stitched: {}. Provide a concrete (non-parametric) DSLX wrapper before running the stitch tool.",
            offending.join(", ")
        )));
    }
    Ok(stages)
}

/// Generates SystemVerilog for pipeline stages `top_cycle0`, `top_cycle1`, ...
/// in `dslx` and stitches them together into a wrapper module.
///
/// The resulting SystemVerilog contains the stage modules followed by a wrapper
/// module whose name is `opts.output_module_name` when set, otherwise `top`.
/// The `top` parameter is used as a prefix only for implicit stage discovery
/// when `opts.explicit_stages` is `None`.
pub fn stitch_pipeline<'a>(
    dslx: &str,
    path: &Path,
    top: &str,
    opts: &StitchPipelineOptions<'a>,
) -> Result<String, xlsynth::XlsynthError> {
    if opts.output_valid_signal.is_some() && opts.input_valid_signal.is_none() {
        return Err(xlsynth::XlsynthError(
            "output_valid_signal requires input_valid_signal in dslx-stitch-pipeline".to_string(),
        ));
    }
    let resolved = spec::resolve_stages(
        dslx,
        path,
        top,
        opts.explicit_stages.as_deref(),
        spec::StitchImportPaths {
            stdlib: opts.stdlib_path,
            search: &opts.search_paths,
            required_directory: None,
        },
        opts.boundary_options(),
    )?;
    resolved.validate_wrapper_names()?;
    resolved.emit_rtl(&StitchRtlOptions {
        verilog_version: opts.verilog_version,
        validate_no_constant_flops: opts.validate_no_constant_flops,
        add_invariant_assertions: opts.add_invariant_assertions,
        array_index_bounds_checking: opts.array_index_bounds_checking,
        gate_format: opts.gate_format,
    })
}

impl ResolvedStitchPipeline {
    /// Emits combinational stage modules and the clocked wrapper for the
    /// resolved specification.
    pub fn emit_rtl(&self, opts: &StitchRtlOptions<'_>) -> Result<String, xlsynth::XlsynthError> {
        if opts.validate_no_constant_flops && !opts.array_index_bounds_checking {
            return Err(xlsynth::XlsynthError(
                "constant-flop validation requires array_index_bounds_checking=true".to_string(),
            ));
        }
        let cfg = PipelineCfg {
            ir: &self.ir,
            verilog_version: opts.verilog_version,
            add_invariant_assertions: opts.add_invariant_assertions,
            array_index_bounds_checking: opts.array_index_bounds_checking,
            gate_format: opts.gate_format,
        };
        let stages = &self.stages;
        if opts.validate_no_constant_flops {
            let constant_flops = validate_constant_flops::find_constant_flops(
                &self.ir,
                stages,
                self.boundary.flop_outputs,
            )
            .map_err(|error| {
                xlsynth::XlsynthError(format!("constant-flop validation failed: {error}"))
            })?;
            if !constant_flops.is_empty() {
                let details = constant_flops
                    .iter()
                    .map(|flop| {
                        format!(
                            "  stage '{}' (index {}), output bit {} is always {}",
                            flop.stage_name,
                            flop.stage_index,
                            flop.output_bit_index,
                            u8::from(flop.value)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                return Err(xlsynth::XlsynthError(format!(
                    "constant-flop validation failed:\n{details}"
                )));
            }
        }

        // For each stage run codegen immediately so we can parse its port list.
        let mut stage_infos = Vec::with_capacity(stages.len());
        for (stage_unmangled, stage_mangled) in stages {
            stage_infos.push(make_stage_info_comb(
                &cfg,
                stage_unmangled,
                stage_mangled,
                false,
            )?);
        }

        // The wrapper needs only stage interface metadata, not a second VAST
        // file.
        let stage_modules = stage_infos
            .iter()
            .zip(stages)
            .map(|(stage_info, (module_name, _))| StageModuleInterface {
                name: module_name.clone(),
                ports: stage_info
                    .ports
                    .iter()
                    .map(|port| StagePortInterface {
                        name: port.name.clone(),
                        direction: if port.is_input {
                            ModulePortDirection::Input
                        } else {
                            ModulePortDirection::Output
                        },
                        width: i64::from(port.width),
                    })
                    .collect(),
            })
            .collect();

        // Generate wrapper using shared implementation.
        let mut wrapper_file = VastFile::new(match opts.verilog_version {
            VerilogVersion::SystemVerilog => VastFileType::SystemVerilog,
            VerilogVersion::Verilog => VastFileType::Verilog,
        });
        let pipeline_cfg = BuildPipelineConfig {
            top_module_name: self.boundary.name.clone(),
            clk_port_name: self.boundary.clock.name.clone(),
            stage_modules,
            flop_inputs: self.boundary.flop_inputs,
            flop_outputs: self.boundary.flop_outputs,
            input_valid_signal: self.boundary.input_valid_signal.as_deref(),
            output_valid_signal: self.boundary.output_valid_signal.as_deref(),
            reset_signal: self.boundary.reset_signal.as_deref(),
            reset_active_low: self.boundary.reset_active_low,
        };
        let wrapper_sv = build_wrapper(&mut wrapper_file, &pipeline_cfg)?;

        let mut text = stage_infos
            .iter()
            .map(|s| s.sv_text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        text.push_str(&wrapper_sv);
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;
    #[cfg(feature = "iverilog-tests")]
    use xlsynth_pir::IrBits;
    use xlsynth_test_helpers::{self, compare_golden_sv};

    fn default_test_options(output_module_name: &str) -> StitchPipelineOptions<'_> {
        StitchPipelineOptions {
            verilog_version: VerilogVersion::SystemVerilog,
            explicit_stages: None,
            stdlib_path: None,
            search_paths: Vec::new(),
            flop_inputs: true,
            flop_outputs: true,
            validate_no_constant_flops: false,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
            add_invariant_assertions: true,
            array_index_bounds_checking: true,
            gate_format: None,
            output_module_name,
        }
    }

    fn expect_stitch_name_error(
        dslx: &str,
        top: &str,
        opts: StitchPipelineOptions<'_>,
        expected_substrings: &[&str],
    ) {
        let result = stitch_pipeline(dslx, Path::new("test.x"), top, &opts);
        assert!(result.is_err(), "expected stitch pipeline to fail");
        let err = result.unwrap_err().0;
        assert!(
            err.contains("name validation failed"),
            "expected name-validation error, got: {}",
            err
        );
        for expected in expected_substrings {
            assert!(
                err.contains(expected),
                "expected error to contain `{}`, got: {}",
                expected,
                err
            );
        }
    }

    #[test]
    fn test_stitch_pipeline_tuple() {
        let dslx = "fn mul_add_cycle0(x: u32, y: u32, z: u32) -> (u32, u32) { (x * y, z) }\nfn mul_add_cycle1(partial: u32, z: u32) -> u32 { partial + z }";
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "mul_add",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "mul_add",
            },
        )
        .unwrap();
        // Validate generated SV.
        xlsynth_test_helpers::assert_valid_sv(&result);

        compare_golden_sv(&result, "tests/goldens/mul_add.golden.sv");
    }

    #[test]
    fn test_stitch_pipeline_struct() {
        let _ = env_logger::builder().is_test(true).try_init();
        let dslx = r#"struct S { a: u32, b: u32 }
fn foo_cycle0(s: S) -> S { s }
fn foo_cycle1(s: S) -> u32 { s.a + s.b }
"#;
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "foo",
            },
        )
        .unwrap();

        compare_golden_sv(&result, "tests/goldens/foo.golden.sv");
    }

    #[test]
    fn test_stitch_pipeline_single_stage() {
        let dslx = "fn one_cycle0(x: u32, y: u32) -> u32 { x + y }";
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "one",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "one",
            },
        )
        .unwrap();
        compare_golden_sv(&result, "tests/goldens/one.golden.sv");
    }

    #[test]
    fn test_irrelevant_parametrics_are_ignored_by_stitch() {
        let dslx = r#"
fn parmetric_not_stitched0<N: u32>(x: u32) -> u32 { x }
fn parmetric_not_stitched1<N: u32>(x: u32) -> u32 { x }

fn stitchme_cycle0(x: u32) -> u32 { parmetric_not_stitched0<u32:4>(x) }
fn stitchme_cycle1(x: u32) -> u32 { parmetric_not_stitched1<u32:4>(x) }
"#;
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "stitchme",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "stitchme",
            },
        )
        .unwrap();

        xlsynth_test_helpers::assert_valid_sv(&result);
    }

    /// Generates the verilog for two stages composed in DSLX stitching:
    /// * The first stage adds 1
    /// * The second stage adds 2
    fn verilog_for_foo_pipeline_with_valid() -> String {
        let dslx = "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
        stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: Some("input_valid"),
                output_valid_signal: Some("output_valid"),
                reset_signal: Some("rst_n"),
                reset_active_low: true,
                add_invariant_assertions: false,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "foo",
            },
        )
        .unwrap()
    }

    #[test]
    fn test_stitch_pipeline_with_valid() {
        let _ = env_logger::builder().is_test(true).try_init();
        let result = verilog_for_foo_pipeline_with_valid();
        compare_golden_sv(&result, "tests/goldens/foo_with_valid.golden.sv");
    }

    #[cfg(feature = "iverilog-tests")]
    #[test]
    fn test_stitch_pipeline_with_valid_iverilog() {
        let result = verilog_for_foo_pipeline_with_valid();
        let inputs = vec![("x", IrBits::u32(5))];
        let expected = IrBits::u32(8);
        let vcd = xlsynth_test_helpers::simulate_pipeline_single_pulse(
            &result, "foo", &inputs, &expected, 2,
        )
        .expect("simulation succeeds");
        assert!(vcd.contains("$var"));
    }

    /// Same as `test_stitch_pipeline_with_valid` but demonstrates that a reset
    /// signal is no longer required when both input and output valid handshake
    /// signals are used.
    #[test]
    fn test_stitch_pipeline_with_valid_no_reset() {
        let _ = env_logger::builder().is_test(true).try_init();

        // Two-stage pipeline: add 1 then add 2.
        let dslx = "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";

        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: Some("in_valid"),
                output_valid_signal: Some("out_valid"),
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: false,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "foo",
            },
        )
        .unwrap();

        compare_golden_sv(
            &result,
            "tests/goldens/foo_pipeline_with_valid_no_reset.golden.sv",
        );
    }

    #[test]
    fn test_stitch_pipeline_signature_mismatch() {
        let dslx = r#"fn foo_cycle0(x: u32, y: u64) -> (u32, u64) { (x, y) }
fn foo_cycle1(a: u64, b: u32) -> u64 { a + b as u64 }"#;
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            &StitchPipelineOptions {
                verilog_version: VerilogVersion::SystemVerilog,
                explicit_stages: None,
                stdlib_path: None,
                search_paths: Vec::new(),
                flop_inputs: true,
                flop_outputs: true,
                validate_no_constant_flops: false,
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
                gate_format: None,
                output_module_name: "foo",
            },
        );
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(err.contains("does not match"), "unexpected error: {}", err);
    }

    #[test]
    fn test_stitch_pipeline_rejects_unit_stage_output() {
        let dslx = r#"fn foo_cycle0(x: u32) -> () { () }
fn foo_cycle1() -> u32 { u32:1 }"#;
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            &default_test_options("foo"),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(
            err.contains("stage `foo_cycle0` has zero-width return type `()`")
                && err.contains("Verilog output port `out`"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_output_valid_without_input_valid() {
        let mut opts = default_test_options("foo");
        opts.output_valid_signal = Some("output_valid");
        let result = stitch_pipeline(
            "fn foo_cycle0(x: u32) -> u32 { x }\nfn foo_cycle1(x: u32) -> u32 { x }",
            Path::new("test.x"),
            "foo",
            &opts,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(
            err.contains("output_valid_signal requires input_valid_signal"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_first_stage_clk_param() {
        expect_stitch_name_error(
            "fn foo_cycle0(clk: u32) -> u32 { clk }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            default_test_options("foo"),
            &[
                "stage parameter `foo_cycle0.clk`",
                "reserved wrapper/control port `clk`",
            ],
        );
    }

    #[test]
    fn test_stitch_pipeline_allows_later_stage_clk_param() {
        let result = stitch_pipeline(
            "fn foo_cycle0(x: u32) -> u32 { x }\nfn foo_cycle1(clk: u32) -> u32 { clk }",
            Path::new("test.x"),
            "foo",
            &default_test_options("foo"),
        )
        .unwrap();
        assert!(result.contains(".clk(p1_clk)"), "unexpected SV: {}", result);
    }

    #[test]
    fn test_stitch_pipeline_rejects_reset_name_param() {
        let mut opts = default_test_options("foo");
        opts.reset_signal = Some("rst");
        expect_stitch_name_error(
            "fn foo_cycle0(rst: u32) -> u32 { rst }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            opts,
            &[
                "stage parameter `foo_cycle0.rst`",
                "reserved wrapper/control port `rst`",
            ],
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_input_valid_name_param() {
        let mut opts = default_test_options("foo");
        opts.input_valid_signal = Some("input_valid");
        expect_stitch_name_error(
            "fn foo_cycle0(input_valid: u32) -> u32 { input_valid }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            opts,
            &[
                "stage parameter `foo_cycle0.input_valid`",
                "reserved wrapper/control port `input_valid`",
            ],
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_output_valid_name_param() {
        let mut opts = default_test_options("foo");
        opts.input_valid_signal = Some("input_valid");
        opts.output_valid_signal = Some("output_valid");
        expect_stitch_name_error(
            "fn foo_cycle0(output_valid: u32) -> u32 { output_valid }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            opts,
            &[
                "stage parameter `foo_cycle0.output_valid`",
                "reserved wrapper/control port `output_valid`",
            ],
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_out_name_param() {
        let stages = vec![
            names::StageNameInfo {
                stage_name: "foo_cycle0".to_string(),
                params: vec!["out".to_string()],
            },
            names::StageNameInfo {
                stage_name: "foo_cycle1".to_string(),
                params: vec!["x".to_string()],
            },
        ];
        let signatures = vec![
            names::StageSignature {
                output_port_name: "out".to_string(),
            },
            names::StageSignature {
                output_port_name: "out".to_string(),
            },
        ];
        let opts = default_test_options("foo");
        let err =
            names::validate_stitch_pipeline_names(&stages, &signatures, &opts.boundary_options())
                .unwrap_err()
                .0;
        assert!(
            err.contains("stage parameter `foo_cycle0.out`")
                && err.contains("reserved wrapper/output port `out`"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_keyword_param() {
        expect_stitch_name_error(
            "fn foo_cycle0(input: u32) -> u32 { input }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            default_test_options("foo"),
            &[
                "stage parameter `foo_cycle0.input`",
                "SystemVerilog keyword",
            ],
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_system_verilog_keyword_in_verilog_mode() {
        let mut opts = default_test_options("foo");
        opts.verilog_version = VerilogVersion::Verilog;
        expect_stitch_name_error(
            "fn foo_cycle0(logic: u32) -> u32 { logic }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            opts,
            &[
                "stage parameter `foo_cycle0.logic`",
                "SystemVerilog keyword",
            ],
        );
    }

    #[test]
    fn test_stitch_pipeline_rejects_generated_valid_name_collision() {
        let mut opts = default_test_options("foo");
        opts.input_valid_signal = Some("input_valid");
        expect_stitch_name_error(
            "fn foo_cycle0(valid: u32) -> u32 { valid }\nfn foo_cycle1(x: u32) -> u32 { x }",
            "foo",
            opts,
            &[
                "`p0_valid` cannot be emitted as-is",
                "generated valid flop register",
                "generated data flop register for `valid`",
            ],
        );
    }
}
