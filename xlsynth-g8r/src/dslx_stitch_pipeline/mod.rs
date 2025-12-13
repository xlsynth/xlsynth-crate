// SPDX-License-Identifier: Apache-2.0
//! Simple utility to stitch together pipeline stages defined in DSLX.

use std::collections::HashSet;
use std::path::Path;

use xlsynth::dslx::{self, Function, MatchableModuleMember};
use xlsynth::vast::VastDataType;
use xlsynth::vast::VastFile;
use xlsynth::vast::VastModule;
use xlsynth::{
    DslxConvertOptions, convert_dslx_to_ir, mangle_dslx_name, optimize_ir, schedule_and_codegen,
};
use xlsynth::ir_package::IrPackage;

mod common;
use common::{PipelineCfg, Port, StageInfo};

mod build_pipeline;
use build_pipeline::{PipelineConfig as BuildPipelineConfig, build_pipeline as build_wrapper};

use crate::verilog_version::VerilogVersion;

// Insert StitchPipelineOptions struct and Default implementation
#[derive(Debug)]
pub struct StitchPipelineOptions<'a> {
    pub verilog_version: VerilogVersion,
    pub explicit_stages: Option<Vec<String>>,
    pub stdlib_path: Option<&'a Path>,
    pub search_paths: Vec<&'a Path>,
    pub flop_inputs: bool,
    pub flop_outputs: bool,
    pub input_valid_signal: Option<&'a str>,
    pub output_valid_signal: Option<&'a str>,
    pub reset_signal: Option<&'a str>,
    pub reset_active_low: bool,
    pub add_invariant_assertions: bool,
    pub array_index_bounds_checking: bool,
    pub output_module_name: &'a str,
}

#[derive(Debug)]
pub struct StitchPipelineIrPackages {
    pub wrapper_dslx_name: String,
    pub wrapper_ir_name: String,
    pub unopt_ir: String,
    pub opt_ir: String,
}

fn module_function_names(module: &xlsynth::dslx::Module) -> HashSet<String> {
    let mut names = HashSet::new();
    for i in 0..module.get_member_count() {
        if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
            names.insert(f.get_identifier());
        }
    }
    names
}

fn find_module_function_by_name(module: &xlsynth::dslx::Module, name: &str) -> Option<Function> {
    for i in 0..module.get_member_count() {
        if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
            if f.get_identifier() == name {
                return Some(f);
            }
        }
    }
    None
}

fn choose_unique_wrapper_dslx_name(existing: &HashSet<String>, base: &str) -> String {
    let mut sanitized = String::with_capacity(base.len());
    for c in base.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            sanitized.push(c);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        sanitized.push('_');
    }

    let base_name = format!("__xlsynth_stitch_pipeline__{sanitized}");
    if !existing.contains(&base_name) {
        return base_name;
    }
    for i in 0.. {
        let candidate = format!("{base_name}_{i}");
        if !existing.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("infinite loop should always find an unused wrapper name")
}

fn dslx_type_to_str(
    type_info: &xlsynth::dslx::TypeInfo,
    annotation: &xlsynth::dslx::TypeAnnotation,
) -> Result<String, xlsynth::XlsynthError> {
    let ty = type_info.get_type_for_type_annotation(annotation).ok_or_else(|| {
        xlsynth::XlsynthError("could not resolve concrete type for type annotation".into())
    })?;
    ty.to_string()
}

fn make_composed_wrapper_dslx(
    typechecked_module: &xlsynth::dslx::TypecheckedModule,
    stages: &[String],
    wrapper_dslx_name: &str,
) -> Result<String, xlsynth::XlsynthError> {
    let module = typechecked_module.get_module();
    let type_info = typechecked_module.get_type_info();

    let first_stage_name = stages
        .first()
        .expect("validated non-empty stage list before wrapper synthesis");
    let last_stage_name = stages
        .last()
        .expect("validated non-empty stage list before wrapper synthesis");

    let first_stage_fn = find_module_function_by_name(&module, first_stage_name).ok_or_else(|| {
        xlsynth::XlsynthError(format!(
            "could not find stage function `{first_stage_name}` in typechecked module"
        ))
    })?;
    let last_stage_fn = find_module_function_by_name(&module, last_stage_name).ok_or_else(|| {
        xlsynth::XlsynthError(format!(
            "could not find stage function `{last_stage_name}` in typechecked module"
        ))
    })?;

    let mut params = Vec::new();
    let mut arg_names = Vec::new();
    for i in 0..first_stage_fn.get_param_count() {
        let p = first_stage_fn.get_param(i);
        let name = p.get_name();
        let ty = dslx_type_to_str(&type_info, &p.get_type_annotation())?;
        params.push(format!("{name}: {ty}"));
        arg_names.push(name);
    }
    let ret_ty = dslx_type_to_str(&type_info, &last_stage_fn.get_return_type())?;

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "fn {wrapper_dslx_name}({}) -> {ret_ty} {{",
        params.join(", ")
    ));
    let mut prev_var = String::new();
    for (i, stage_name) in stages.iter().enumerate() {
        let stage_fn = find_module_function_by_name(&module, stage_name).ok_or_else(|| {
            xlsynth::XlsynthError(format!(
                "could not find stage function `{stage_name}` in typechecked module"
            ))
        })?;
        let var_name = format!("v{i}");

        let call_expr = if i == 0 {
            format!("{stage_name}({})", arg_names.join(", "))
        } else {
            match stage_fn.get_param_count() {
                0 => format!("{stage_name}()"),
                1 => format!("{stage_name}({prev_var})"),
                n => {
                    let mut args = Vec::with_capacity(n);
                    for idx in 0..n {
                        args.push(format!("{prev_var}.{idx}"));
                    }
                    format!("{stage_name}({})", args.join(", "))
                }
            }
        };

        lines.push(format!("    let {var_name} = {call_expr};"));
        prev_var = var_name;
    }
    lines.push(format!("    {prev_var}"));
    lines.push("}".to_string());

    Ok(lines.join("\n"))
}

fn ir_package_header_prefix(ir_text: &str) -> Result<&str, xlsynth::XlsynthError> {
    // We want everything up to (and including) the blank line after the last
    // `file_number` declaration.
    //
    // Typical structure:
    //   package <name>
    //
    //   file_number 0 "<file>"
    //
    //   fn ...
    //
    let file_number_pos = ir_text.find("\nfile_number ").ok_or_else(|| {
        xlsynth::XlsynthError("could not find `file_number` in IR text".into())
    })?;
    let after_file_number = &ir_text[file_number_pos + 1..];
    let blank_line_pos = after_file_number.find("\n\n").ok_or_else(|| {
        xlsynth::XlsynthError("could not find blank line after `file_number`".into())
    })?;
    let header_end = file_number_pos + 1 + blank_line_pos + 2;
    Ok(&ir_text[..header_end])
}

fn extract_function_def(ir_text: &str, mangled_name: &str) -> Result<String, xlsynth::XlsynthError> {
    let needle_top = format!("\ntop fn {mangled_name}(");
    let needle_fn = format!("\nfn {mangled_name}(");
    let start = if let Some(pos) = ir_text.find(&needle_top) {
        pos + 1
    } else if let Some(pos) = ir_text.find(&needle_fn) {
        pos + 1
    } else {
        return Err(xlsynth::XlsynthError(format!(
            "could not find function `{mangled_name}` in IR text"
        )));
    };

    let func_text = &ir_text[start..];
    let open_brace_pos = func_text.find('{').ok_or_else(|| {
        xlsynth::XlsynthError(format!(
            "could not find opening brace for function `{mangled_name}`"
        ))
    })?;

    let mut depth: i64 = 0;
    let mut end_idx: Option<usize> = None;
    for (i, c) in func_text[open_brace_pos..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(open_brace_pos + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end_idx.ok_or_else(|| {
        xlsynth::XlsynthError(format!(
            "could not find closing brace for function `{mangled_name}`"
        ))
    })?;

    // Include trailing newline if present.
    let mut out = func_text[..end].to_string();
    if func_text[end..].starts_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

fn normalize_stage_def_to_non_top(def: &str) -> String {
    if let Some(rest) = def.strip_prefix("top fn ") {
        format!("fn {rest}")
    } else {
        def.to_string()
    }
}

fn normalize_wrapper_def_to_top(def: &str) -> String {
    if let Some(rest) = def.strip_prefix("fn ") {
        format!("top fn {rest}")
    } else {
        def.to_string()
    }
}

/// Produces IR package texts for the stitched pipeline:
/// - `unopt_ir`: includes all stage functions and the synthesized composed-wrapper
///   function, with `top` set to the wrapper.
/// - `opt_ir`: includes optimized versions of each stage function and the wrapper
///   (optimized per-entity), assembled into a single package text with `top` set
///   to the wrapper.
pub fn stitch_pipeline_ir_packages<'a>(
    dslx: &str,
    path: &Path,
    top: &str,
    opts: &StitchPipelineOptions<'a>,
) -> Result<StitchPipelineIrPackages, xlsynth::XlsynthError> {
    let module_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| xlsynth::XlsynthError("invalid path".into()))?;

    let explicit_stages = opts.explicit_stages.as_ref().map(|v| v.as_slice());

    // Parse/typecheck once for validations and for synthesizing the wrapper DSLX.
    let mut import_data = dslx::ImportData::new(opts.stdlib_path, opts.search_paths.as_slice());
    let typechecked_module =
        dslx::parse_and_typecheck(dslx, path.to_str().unwrap(), module_name, &mut import_data)?;
    check_for_parametric_stages(&typechecked_module, top, explicit_stages)?;
    if explicit_stages.is_none() {
        check_implicit_stage_numbering(&typechecked_module, top)?;
    }

    // Convert original DSLX to IR to discover stage ordering and mangled names.
    let convert_opts = DslxConvertOptions {
        dslx_stdlib_path: opts.stdlib_path,
        additional_search_paths: opts.search_paths.clone(),
        enable_warnings: None,
        disable_warnings: None,
        force_implicit_token_calling_convention: false,
    };
    let conv = convert_dslx_to_ir(dslx, path, &convert_opts)?;
    let ir = conv.ir;
    let discovered = discover_stage_names(&ir, path, top, explicit_stages)?;
    verify_stage_signatures(&ir, &discovered)?;

    let stage_unmangled_names: Vec<String> = discovered.iter().map(|(u, _)| u.clone()).collect();

    let existing_names = module_function_names(&typechecked_module.get_module());
    let wrapper_dslx_name =
        choose_unique_wrapper_dslx_name(&existing_names, opts.output_module_name);

    let wrapper_dslx = make_composed_wrapper_dslx(
        &typechecked_module,
        &stage_unmangled_names,
        &wrapper_dslx_name,
    )?;

    let augmented_dslx = format!(
        r#"{dslx}

// Synthesized by xlsynth-g8r for dslx-stitch-pipeline IR residual generation.
{wrapper_dslx}
"#,
        dslx = dslx,
        wrapper_dslx = wrapper_dslx
    );

    let augmented = convert_dslx_to_ir(&augmented_dslx, path, &convert_opts)?;
    let augmented_ir = augmented.ir;

    let wrapper_ir_name = mangle_dslx_name(module_name, &wrapper_dslx_name)?;

    // -- Unoptimized package with wrapper as top (and all stage functions present).
    let augmented_ir_text = augmented_ir.to_string();
    let mut unopt_pkg = IrPackage::parse_ir(&augmented_ir_text, augmented_ir.filename())?;
    unopt_pkg.set_top_by_name(&wrapper_ir_name)?;
    let unopt_ir = unopt_pkg.to_string();

    // -- Optimized package assembled from per-entity optimized definitions.
    let header = ir_package_header_prefix(&unopt_ir)?.to_string();

    let mut stage_defs: Vec<String> = Vec::with_capacity(discovered.len());
    for (_unmangled, stage_mangled) in &discovered {
        let opt_pkg = optimize_ir(&augmented_ir, stage_mangled)?;
        let opt_text = opt_pkg.to_string();
        let def = extract_function_def(&opt_text, stage_mangled)?;
        stage_defs.push(normalize_stage_def_to_non_top(&def));
    }

    let wrapper_opt_pkg = optimize_ir(&augmented_ir, &wrapper_ir_name)?;
    let wrapper_opt_text = wrapper_opt_pkg.to_string();
    let wrapper_def = extract_function_def(&wrapper_opt_text, &wrapper_ir_name)?;
    let wrapper_def = normalize_wrapper_def_to_top(&wrapper_def);

    let mut opt_ir = header;
    for def in stage_defs {
        opt_ir.push_str(&def);
        if !opt_ir.ends_with("\n\n") {
            if !opt_ir.ends_with('\n') {
                opt_ir.push('\n');
            }
            opt_ir.push('\n');
        }
    }
    opt_ir.push_str(&wrapper_def);

    Ok(StitchPipelineIrPackages {
        wrapper_dslx_name,
        wrapper_ir_name,
        unopt_ir,
        opt_ir,
    })
}

/// Creates a VAST "stub" module from the given `StageInfo`.
///
/// XLS code-gen returns each combinational stage as a *string* of
/// SystemVerilog; it does **not** give us a structured representation we can
/// hand to `build_pipeline`. The wrapper generator, however, only needs each
/// stage's *interface* (port names, directions, bit-widths) so it can declare
/// nets and connect them.
///
/// We therefore synthesize a minimal `VastModule` that mirrors the port list of
/// the real stage. The actual behavior of the design still comes from the
/// original Verilog text that we concatenate onto the output – the stub exists
/// solely to satisfy `build_pipeline`'s API.
fn make_stub_module<'a>(
    file: &'a mut VastFile,
    stage_info: &StageInfo,
    module_name: &str,
    scalar_type: &VastDataType,
    _dynamic_types: &mut Vec<VastDataType>,
) -> VastModule {
    let mut m: VastModule = file.add_module(module_name);

    for port in &stage_info.ports {
        let dt_ref = if port.width == 1 {
            scalar_type.clone()
        } else {
            // Allocate a new type but do *not* keep ownership; this will drop at
            // function exit and later lead to a panic when VAST tries to emit.
            file.make_bit_vector_type(port.width as i64, false)
        };

        if port.is_input {
            m.add_input(&port.name, &dt_ref);
        } else {
            m.add_output(&port.name, &dt_ref);
        }
    }

    m
}

fn build_ports_from_ir(
    func: &xlsynth::ir_package::IrFunction,
    fty: &xlsynth::ir_package::IrFunctionType,
) -> Result<(Vec<Port>, u32), xlsynth::XlsynthError> {
    // 1. Clock
    let mut ports = vec![Port {
        name: "clk".to_string(),
        is_input: true,
        width: 1,
    }];

    // 2. Formal parameters (flattened widths)
    for i in 0..fty.param_count() {
        let name = func.param_name(i)?;
        let ty = fty.param_type(i)?;
        ports.push(Port {
            name,
            is_input: true,
            width: ty.get_flat_bit_count() as u32,
        });
    }

    // 3. Return value -> `out` port
    let ret_ty = fty.return_type();
    let ret_width = ret_ty.get_flat_bit_count() as u32;
    ports.push(Port {
        name: "out".into(),
        is_input: false,
        width: ret_width,
    });

    Ok((ports, ret_width))
}

/// Collect user-specified or auto-discovered stage names and their mangled IR
/// names.
fn discover_stage_names(
    ir: &xlsynth::ir_package::IrPackage,
    path: &std::path::Path,
    top: &str,
    explicit: Option<&[String]>,
) -> Result<Vec<(String, String)>, xlsynth::XlsynthError> {
    let mut stages = Vec::new();
    if let Some(names) = explicit {
        for name in names {
            let mangled = mangle_dslx_name(path.file_stem().unwrap().to_str().unwrap(), name)?;
            if ir.get_function(&mangled).is_err() {
                return Err(xlsynth::XlsynthError(format!(
                    "stage {name} (mangled {mangled}) not found in IR"
                )));
            }
            stages.push((name.clone(), mangled));
        }
    } else {
        for i in 0.. {
            let fn_name = format!("{top}_cycle{i}");
            let mangled =
                match mangle_dslx_name(path.file_stem().unwrap().to_str().unwrap(), &fn_name) {
                    Ok(v) => v,
                    Err(_) => break,
                };
            if ir.get_function(&mangled).is_err() {
                break;
            }
            stages.push((fn_name, mangled));
        }
    }
    if stages.is_empty() {
        return Err(xlsynth::XlsynthError("no pipeline stages found".into()));
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

// Run XLS optimise + schedule + codegen for a stage without inserting any
// flops so the resulting module is purely combinational.
fn make_stage_info_comb(
    cfg: &PipelineCfg,
    stage_name_unmangled: &str,
    stage_mangled: &str,
    _is_last: bool,
) -> Result<StageInfo, xlsynth::XlsynthError> {
    let opt = optimize_ir(cfg.ir, stage_mangled)?;
    let sched = "delay_model: \"unit\"\npipeline_stages: 1";
    // Use the XLS "combinational" generator so the resulting module has *no* clock.
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
    let result = schedule_and_codegen(&opt, sched, &codegen)?;
    let sv_text = result.get_verilog_text()?;

    let func = cfg.ir.get_function(stage_mangled)?;
    let fty = func.get_type()?;
    let (mut ports, output_width) = build_ports_from_ir(&func, &fty)?;
    // Combinational modules do not have a clock port, so drop it from
    // the discovered port list to simplify downstream handling.
    ports.retain(|p| p.name != "clk");

    Ok(StageInfo {
        sv_text,
        ports,
        output_width,
    })
}

fn is_implicit_stage_name(name: &str) -> bool {
    if let Some(idx) = name.rfind("_cycle") {
        let digits = &name[idx + 6..];
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
    } else {
        false
    }
}

/// Scans the typechecked DSLX module for functions named like `<top>_cycle<N>`
/// and enforces that indices start at 0 and are contiguous.
fn check_implicit_stage_numbering(
    tc_module: &xlsynth::dslx::TypecheckedModule,
    top: &str,
) -> Result<(), xlsynth::XlsynthError> {
    let module = tc_module.get_module();
    let mut indices: HashSet<usize> = HashSet::new();
    let prefix = format!("{top}_cycle");
    for i in 0..module.get_member_count() {
        let member = module.get_member(i);
        if let Some(MatchableModuleMember::Function(func)) = member.to_matchable() {
            let name = func.get_identifier();
            if name.starts_with(&prefix) {
                let digits = &name[prefix.len()..];
                if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                    if let Ok(idx) = digits.parse::<usize>() {
                        indices.insert(idx);
                    }
                }
            }
        }
    }

    if indices.is_empty() {
        return Ok(());
    }

    let mut sorted: Vec<usize> = indices.iter().copied().collect();
    sorted.sort_unstable();
    let max = *sorted.last().unwrap();
    let expected_len = max + 1;
    let starts_at_zero = sorted.first().copied().unwrap() == 0;
    let contiguous = indices.len() == expected_len && starts_at_zero;
    if !contiguous {
        let found = sorted
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(xlsynth::XlsynthError(format!(
            "found stage function(s) named like '{top}_cycleN' but numbering must start at 0 and be contiguous; found indices: {found}"
        )));
    }

    Ok(())
}

/// Ensures the stage functions selected for stitching are non-parametric. When
/// `explicit_stages` is provided we check those names directly. Otherwise the
/// scan includes functions named like `<top>_cycle<N>`. This second branch is
/// prone to false positives (parametric functions named like `<top>_cycle<N>`,
/// but not the ones being stitched).
fn check_for_parametric_stages(
    tc_module: &xlsynth::dslx::TypecheckedModule,
    top: &str,
    explicit_stages: Option<&[String]>,
) -> Result<(), xlsynth::XlsynthError> {
    let module = tc_module.get_module();
    let mut offending = HashSet::new();
    match explicit_stages {
        Some(stages) => {
            let stages_set: HashSet<&str> = stages.iter().map(|s| s.as_str()).collect();
            if stages_set.is_empty() {
                return Ok(());
            }
            for i in 0..module.get_member_count() {
                let member = module.get_member(i);
                if let Some(MatchableModuleMember::Function(func)) = member.to_matchable() {
                    let name = func.get_identifier();
                    if stages_set.contains(name.as_str()) && func.is_parametric() {
                        offending.insert(name);
                    }
                }
            }
        }
        None => {
            // Note: even this is more prone to false positives than necessary; we should
            // check for {top}_cycle0, {top}_cycle1, etc.
            let prefix = format!("{top}_cycle");
            for i in 0..module.get_member_count() {
                let member = module.get_member(i);
                if let Some(MatchableModuleMember::Function(func)) = member.to_matchable() {
                    let name = func.get_identifier();
                    if name.starts_with(&prefix)
                        && is_implicit_stage_name(&name)
                        && func.is_parametric()
                    {
                        offending.insert(name);
                    }
                }
            }
        }
    }

    if !offending.is_empty() {
        let mut names: Vec<String> = offending.into_iter().collect();
        names.sort();
        return Err(xlsynth::XlsynthError(format!(
            "parametric stage function(s) detected that cannot be stitched: {}. Provide a concrete (non-parametric) wrapper or specialization before running the stitch tool.",
            names.join(", ")
        )));
    }
    Ok(())
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
    // Extract option fields for backwards-compat ease.
    let verilog_version = opts.verilog_version;
    let explicit_stages = opts.explicit_stages.as_ref().map(|v| v.as_slice());
    let stdlib_path = opts.stdlib_path;
    let search_paths = opts.search_paths.as_slice();
    let flop_inputs = opts.flop_inputs;
    let flop_outputs = opts.flop_outputs;
    let input_valid_signal = opts.input_valid_signal;
    let output_valid_signal = opts.output_valid_signal;
    let reset_signal = opts.reset_signal;
    let reset_active_low = opts.reset_active_low;
    let add_invariant_assertions = opts.add_invariant_assertions;
    let array_index_bounds_checking = opts.array_index_bounds_checking;

    // Parse/typecheck once for pre-codegen validations.
    let module_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| xlsynth::XlsynthError("invalid path".into()))?;
    let mut import_data = dslx::ImportData::new(stdlib_path, search_paths);
    let typechecked_module =
        dslx::parse_and_typecheck(dslx, path.to_str().unwrap(), module_name, &mut import_data)?;

    check_for_parametric_stages(&typechecked_module, top, explicit_stages)?;
    // If relying on implicit stage discovery, ensure any discovered
    // `<top>_cycleN` functions start at 0 and are contiguous.
    if explicit_stages.is_none() {
        check_implicit_stage_numbering(&typechecked_module, top)?;
    }
    let convert_opts = DslxConvertOptions {
        dslx_stdlib_path: stdlib_path,
        additional_search_paths: search_paths.to_vec(),
        enable_warnings: None,
        disable_warnings: None,
        force_implicit_token_calling_convention: false,
    };
    let conv = convert_dslx_to_ir(dslx, path, &convert_opts)?;
    let ir = conv.ir;

    let cfg = PipelineCfg {
        ir: &ir,
        verilog_version,
        add_invariant_assertions,
        array_index_bounds_checking,
    };

    let stages = discover_stage_names(&ir, path, top, explicit_stages)?;
    verify_stage_signatures(&ir, &stages)?;

    // For each stage run codegen immediately so we can parse its port list.
    let mut stage_infos = Vec::with_capacity(stages.len());
    for (stage_unmangled, stage_mangled) in &stages {
        stage_infos.push(make_stage_info_comb(
            &cfg,
            stage_unmangled,
            stage_mangled,
            false,
        )?);
    }

    // Prepare VAST stubs for the new build_pipeline implementation.
    let file_type = match verilog_version {
        VerilogVersion::SystemVerilog => xlsynth::vast::VastFileType::SystemVerilog,
        VerilogVersion::Verilog => xlsynth::vast::VastFileType::Verilog,
    };
    let mut stub_file = xlsynth::vast::VastFile::new(file_type);
    let scalar_type_stub = stub_file.make_scalar_type();
    let mut dynamic_types_stub: Vec<xlsynth::vast::VastDataType> = Vec::new();
    let mut stub_modules: Vec<xlsynth::vast::VastModule> = Vec::with_capacity(stage_infos.len());

    for (idx, stage_info) in stage_infos.iter().enumerate() {
        let module_name = &stages[idx].0;
        let m = make_stub_module(
            &mut stub_file,
            stage_info,
            module_name,
            &scalar_type_stub,
            &mut dynamic_types_stub,
        );
        stub_modules.push(m);
    }

    let stage_module_refs: Vec<&xlsynth::vast::VastModule> = stub_modules.iter().collect();

    // Generate wrapper using shared implementation.
    let mut wrapper_file = xlsynth::vast::VastFile::new(match verilog_version {
        VerilogVersion::SystemVerilog => xlsynth::vast::VastFileType::SystemVerilog,
        VerilogVersion::Verilog => xlsynth::vast::VastFileType::Verilog,
    });
    let pipeline_cfg = BuildPipelineConfig {
        top_module_name: opts.output_module_name.to_string(),
        clk_port_name: "clk".to_string(),
        stage_modules: stage_module_refs,
        flop_inputs,
        flop_outputs,
        input_valid_signal,
        output_valid_signal,
        reset_signal,
        reset_active_low,
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

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;
    use xlsynth::ir_value::IrBits;
    use xlsynth_test_helpers::{self, compare_golden_sv};

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
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
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
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
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
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
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
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
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
                input_valid_signal: Some("input_valid"),
                output_valid_signal: Some("output_valid"),
                reset_signal: Some("rst_n"),
                reset_active_low: true,
                add_invariant_assertions: false,
                array_index_bounds_checking: true,
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

        // Simulation check
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
                input_valid_signal: Some("in_valid"),
                output_valid_signal: Some("out_valid"),
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: false,
                array_index_bounds_checking: true,
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
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
                add_invariant_assertions: true,
                array_index_bounds_checking: true,
                output_module_name: "foo",
            },
        );
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(err.contains("does not match"), "unexpected error: {}", err);
    }
}
