// SPDX-License-Identifier: Apache-2.0
//! Simple utility to stitch together pipeline stages defined in DSLX.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use xlsynth::dslx::{self, MatchableModuleMember};
use xlsynth::{
    DslxConvertOptions, convert_dslx_to_ir, mangle_dslx_name, optimize_ir, schedule_and_codegen,
};

use crate::verilog_version::VerilogVersion;

/// Immutable configuration passed around while stitching a pipeline.
#[derive(Clone)]
struct PipelineCfg<'a> {
    ir: &'a xlsynth::ir_package::IrPackage,
    verilog_version: VerilogVersion,
}

/// One port in a stage module (flattened).
#[derive(Debug, Clone)]
struct Port {
    name: String,
    is_input: bool,
    width: u32,
}

/// Information derived for each stage that the wrapper needs.
#[derive(Debug, Clone)]
struct StageInfo {
    sv_text: String,
    ports: Vec<Port>,
    output_width: u32,
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
    let codegen = format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\ngenerator: GENERATOR_KIND_COMBINATIONAL\nmodule_name: \"{stage}\"\nuse_system_verilog: {sv}",
        stage = stage_name_unmangled,
        sv = cfg.verilog_version.is_system_verilog(),
    );
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

/// Uses the DSLX front-end to identify parametric stage functions following
/// the implicit `<top>_cycle<N>` naming convention. Additional search paths
/// are supplied so imports resolve the same way as during IR conversion.
fn check_for_parametric_stages(
    dslx_text: &str,
    path: &std::path::Path,
    stdlib_path: Option<&std::path::Path>,
    search_paths: &[&std::path::Path],
) -> Result<(), xlsynth::XlsynthError> {
    let module_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| xlsynth::XlsynthError("invalid path".into()))?;

    let mut import_data = dslx::ImportData::new(stdlib_path, search_paths);
    let tc_module = dslx::parse_and_typecheck(
        dslx_text,
        path.to_str().unwrap(),
        module_name,
        &mut import_data,
    )?;

    let module = tc_module.get_module();
    let mut offending = HashSet::new();
    for i in 0..module.get_member_count() {
        let member = module.get_member(i);
        if let Some(MatchableModuleMember::Function(func)) = member.to_matchable() {
            let name = func.get_identifier();
            if is_implicit_stage_name(&name) && func.is_parametric() {
                offending.insert(name);
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
/// module named `<top>_pipeline` that instantiates each stage.
pub fn stitch_pipeline(
    dslx: &str,
    path: &Path,
    top: &str,
    verilog_version: VerilogVersion,
    explicit_stages: Option<&[String]>,
    stdlib_path: Option<&Path>,
    search_paths: &[&Path],
) -> Result<String, xlsynth::XlsynthError> {
    check_for_parametric_stages(dslx, path, stdlib_path, search_paths)?;
    let convert_opts = DslxConvertOptions {
        dslx_stdlib_path: stdlib_path,
        additional_search_paths: search_paths.to_vec(),
        enable_warnings: None,
        disable_warnings: None,
    };
    let conv = convert_dslx_to_ir(dslx, path, &convert_opts)?;
    let ir = conv.ir;

    let cfg = PipelineCfg {
        ir: &ir,
        verilog_version,
    };

    let stages = discover_stage_names(&ir, path, top, explicit_stages)?;
    verify_stage_signatures(&ir, &stages)?;

    // For each stage run codegen immediately so we can parse its port list.
    // Use the combinational generator so the emitted modules have no clock
    // port; this simplifies the stitching wrapper logic.
    let mut stage_infos = Vec::with_capacity(stages.len());
    for (stage_unmangled, stage_mangled) in &stages {
        let info = make_stage_info_comb(&cfg, stage_unmangled, stage_mangled, false)?;
        stage_infos.push(info);
    }

    // Build the wrapper using VAST.
    let file_type = match cfg.verilog_version {
        VerilogVersion::SystemVerilog => xlsynth::vast::VastFileType::SystemVerilog,
        VerilogVersion::Verilog => xlsynth::vast::VastFileType::Verilog,
    };
    let mut file = xlsynth::vast::VastFile::new(file_type);

    let scalar_type = file.make_scalar_type(); // 1-bit default
    // Keep any dynamically created VAST data types alive for the lifetime of
    // this function (until we emit the file) by storing them in this Vec.
    let mut dynamic_types: Vec<xlsynth::vast::VastDataType> = Vec::new();

    // Create the wrapper module. Stage modules generated via the
    // combinational generator have no clock ports, but the wrapper needs a
    // clock to drive the pipeline flops it inserts between stages.
    let mut wrapper = file.add_module(top);
    let clk_port = wrapper.add_input("clk", &scalar_type);

    // Input ports come from first stage inputs excluding clk. These are also
    // captured into the initial pipeline registers named `p0_<port>`.
    let first_stage_ports = &stage_infos[0].ports;
    let mut wrapper_inputs: HashMap<String, xlsynth::vast::LogicRef> = HashMap::new();
    let mut p0_regs: HashMap<String, xlsynth::vast::LogicRef> = HashMap::new();
    for p in first_stage_ports
        .iter()
        .filter(|p| p.is_input && p.name != "clk")
    {
        let dt_ref = if p.width == 1 {
            &scalar_type
        } else {
            // Create and store the bitvector type so its lifetime extends.
            dynamic_types.push(file.make_bit_vector_type(p.width as i64, false));
            dynamic_types.last().unwrap()
        };
        let lr = wrapper.add_input(&p.name, dt_ref);
        wrapper_inputs.insert(p.name.clone(), lr);
        let reg = wrapper.add_reg(&format!("p0_{}", p.name), dt_ref).unwrap();
        p0_regs.insert(p.name.clone(), reg);
    }

    let posedge_clk = file.make_pos_edge(&clk_port.to_expr());
    let always_p0 = if cfg.verilog_version.is_system_verilog() {
        wrapper.add_always_ff(&[&posedge_clk]).unwrap()
    } else {
        wrapper.add_always_at(&[&posedge_clk]).unwrap()
    };
    let mut sb0 = always_p0.get_statement_block();
    // Iterate in a deterministic order to ensure stable Verilog output.
    let mut p0_names: Vec<String> = p0_regs.keys().cloned().collect();
    p0_names.sort();
    for name in p0_names {
        let reg = p0_regs.get(&name).unwrap();
        let src = wrapper_inputs.get(&name).unwrap();
        sb0.add_nonblocking_assignment(&reg.to_expr(), &src.to_expr());
    }

    // Output port derived from final stage out width
    let last_out_width = stage_infos.last().unwrap().output_width;
    let out_dt_ref = if last_out_width == 1 {
        &scalar_type
    } else {
        dynamic_types.push(file.make_bit_vector_type(last_out_width as i64, false));
        dynamic_types.last().unwrap()
    };
    let out_port = wrapper.add_output("out", out_dt_ref);

    // Stage processing: instantiate each combinational stage and insert a
    // pipeline register capturing its output. The register names follow the
    // convention `p<N>` while the combinational result is `p<N>_next`.
    let mut prev_reg: Option<xlsynth::vast::LogicRef> = None;
    let mut prev_width: u32 = 0;

    for (idx, stage_info) in stage_infos.iter().enumerate() {
        let stage_unmangled = &stages[idx].0;

        // Determine output wire for this stage
        let output_width = stage_info.output_width;
        let out_dt_stage_ref = if output_width == 1 {
            &scalar_type
        } else {
            dynamic_types.push(file.make_bit_vector_type(output_width as i64, false));
            dynamic_types.last().unwrap()
        };
        let output_wire = wrapper.add_wire(&format!("p{}_next", idx + 1), out_dt_stage_ref);

        // Build connection names and expressions in order they appear in port list.
        let conn_port_names: Vec<&str> = stage_info
            .ports
            .iter()
            .filter(|p| p.name != "clk")
            .map(|p| p.name.as_str())
            .collect();

        // Keep expression objects alive.
        let mut temp_exprs: Vec<xlsynth::vast::Expr> = Vec::new();
        let mut conn_expr_indices: Vec<Option<usize>> = Vec::new();

        // For slicing we need prev output indexable expr.
        let prev_indexable = prev_reg.as_ref().map(|w| w.to_indexable_expr());

        // For multi-input slicing we accumulate cursor.
        let mut bit_cursor: u32 = 0;

        for port in stage_info.ports.iter().filter(|p| p.name != "clk") {
            if !port.is_input {
                // Output port: connect the wire we just created.
                let e = output_wire.to_expr();
                temp_exprs.push(e);
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
                continue;
            }

            // Input port (non-clk)
            if idx == 0 {
                // Connect from the input pipeline registers.
                if let Some(reg) = p0_regs.get(&port.name) {
                    let e = reg.to_expr();
                    temp_exprs.push(e);
                    conn_expr_indices.push(Some(temp_exprs.len() - 1));
                } else {
                    conn_expr_indices.push(None);
                }
            } else {
                // Connect from prev stage output.
                if stage_info
                    .ports
                    .iter()
                    .filter(|p| p.is_input && p.name != "clk")
                    .count()
                    == 1
                    && port.width == prev_width
                {
                    // Whole vector passthrough.
                    let e = prev_reg.as_ref().unwrap().to_expr();
                    temp_exprs.push(e);
                    conn_expr_indices.push(Some(temp_exprs.len() - 1));
                } else {
                    // Slice prev_out.
                    let hi: i64 = (prev_width - 1 - bit_cursor) as i64;
                    let lo: i64 = (hi - port.width as i64 + 1) as i64;
                    let slice = file
                        .make_slice(&prev_indexable.as_ref().unwrap(), hi, lo)
                        .to_expr();
                    temp_exprs.push(slice);
                    conn_expr_indices.push(Some(temp_exprs.len() - 1));
                    bit_cursor += port.width;
                }
            }
        }

        // Convert indices to references now that `temp_exprs` will no longer
        // be mutated.
        let conn_exprs: Vec<Option<&xlsynth::vast::Expr>> = conn_expr_indices
            .iter()
            .map(|opt_idx| opt_idx.map(|i| &temp_exprs[i]))
            .collect();

        // Build instantiation.
        let instance_name = format!("{}_i", stage_unmangled);
        wrapper.add_member_instantiation(file.make_instantiation(
            stage_unmangled,
            &instance_name,
            &[],
            &[],
            &conn_port_names,
            &conn_exprs,
        ));

        // Create the pipeline register that captures this stage's output.
        let out_reg = wrapper
            .add_reg(&format!("p{}", idx + 1), out_dt_stage_ref)
            .unwrap();
        let always = if cfg.verilog_version.is_system_verilog() {
            wrapper.add_always_ff(&[&posedge_clk]).unwrap()
        } else {
            wrapper.add_always_at(&[&posedge_clk]).unwrap()
        };
        let mut sb = always.get_statement_block();
        sb.add_nonblocking_assignment(&out_reg.to_expr(), &output_wire.to_expr());

        prev_reg = Some(out_reg);
        prev_width = output_width;
    }

    // Assign wrapper.out = last pipeline register
    if let Some(final_reg) = &prev_reg {
        wrapper.add_member_continuous_assignment(
            file.make_continuous_assignment(&out_port.to_expr(), &final_reg.to_expr()),
        );
    }

    // Combine stage module text and wrapper text.
    let mut text = stage_infos
        .iter()
        .map(|s| s.sv_text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    text.push_str(&file.emit());
    Ok(text)
}

/// Like [`stitch_pipeline`] but the stage modules are generated without any
/// flop inputs/outputs and the wrapper inserts register stages with valid
/// handshaking. Each stage's output is captured in a register along with a
/// valid bit that is synchronously cleared when `rst` is low.
pub fn stitch_pipeline_with_valid(
    dslx: &str,
    path: &Path,
    top: &str,
    verilog_version: VerilogVersion,
    explicit_stages: Option<&[String]>,
    stdlib_path: Option<&Path>,
    search_paths: &[&Path],
    input_valid_signal: Option<&str>,
    output_valid_signal: Option<&str>,
    reset: Option<&str>,
    reset_active_low: bool,
) -> Result<String, xlsynth::XlsynthError> {
    // Precondition checks
    if reset.is_none() {
        return Err(xlsynth::XlsynthError(
            "reset signal must be specified when using valid handshaking (with_valid)".to_string(),
        ));
    }
    if output_valid_signal.is_some() && input_valid_signal.is_none() {
        return Err(xlsynth::XlsynthError(
            "--output_valid_signal requires --input_valid_signal to be specified".to_string(),
        ));
    }

    let input_valid_signal = input_valid_signal.unwrap_or("input_valid");
    let reset_signal = reset.unwrap_or("rst");
    let output_valid_requested = output_valid_signal.is_some();
    let output_valid_name = output_valid_signal.unwrap_or("output_valid");
    check_for_parametric_stages(dslx, path, stdlib_path, search_paths)?;
    let convert_opts = DslxConvertOptions {
        dslx_stdlib_path: stdlib_path,
        additional_search_paths: search_paths.to_vec(),
        enable_warnings: None,
        disable_warnings: None,
    };
    let conv = convert_dslx_to_ir(dslx, path, &convert_opts)?;
    let ir = conv.ir;

    let cfg = PipelineCfg {
        ir: &ir,
        verilog_version,
    };

    let stages = discover_stage_names(&ir, path, top, explicit_stages)?;
    verify_stage_signatures(&ir, &stages)?;

    let mut stage_infos = Vec::with_capacity(stages.len());
    for (stage_unmangled, stage_mangled) in &stages {
        stage_infos.push(make_stage_info_comb(
            &cfg,
            stage_unmangled,
            stage_mangled,
            false,
        )?);
    }

    let file_type = match cfg.verilog_version {
        VerilogVersion::SystemVerilog => xlsynth::vast::VastFileType::SystemVerilog,
        VerilogVersion::Verilog => xlsynth::vast::VastFileType::Verilog,
    };
    let mut file = xlsynth::vast::VastFile::new(file_type);

    let scalar_type = file.make_scalar_type();
    let mut dynamic_types: Vec<xlsynth::vast::VastDataType> = Vec::new();

    let mut wrapper = file.add_module(top);
    let clk_port = wrapper.add_input("clk", &scalar_type);
    let rst_port = wrapper.add_input(reset_signal, &scalar_type);
    let input_valid_port = wrapper.add_input(input_valid_signal, &scalar_type);

    let first_stage_ports = &stage_infos[0].ports;
    let mut wrapper_inputs: HashMap<String, xlsynth::vast::LogicRef> = HashMap::new();
    for p in first_stage_ports
        .iter()
        .filter(|p| p.is_input && p.name != "clk")
    {
        let dt_ref = if p.width == 1 {
            &scalar_type
        } else {
            dynamic_types.push(file.make_bit_vector_type(p.width as i64, false));
            dynamic_types.last().unwrap()
        };
        let lr = wrapper.add_input(&p.name, dt_ref);
        wrapper_inputs.insert(p.name.clone(), lr);
    }

    let last_out_width = stage_infos.last().unwrap().output_width;
    let out_dt_ref = if last_out_width == 1 {
        &scalar_type
    } else {
        dynamic_types.push(file.make_bit_vector_type(last_out_width as i64, false));
        dynamic_types.last().unwrap()
    };
    let out_port = wrapper.add_output("out", out_dt_ref);
    let output_valid_port = if output_valid_requested {
        Some(wrapper.add_output(output_valid_name, &scalar_type))
    } else {
        None
    };

    // Input pipeline registers (p0)
    let mut p0_regs = HashMap::new();
    for p in first_stage_ports
        .iter()
        .filter(|p| p.is_input && p.name != "clk")
    {
        let dt_ref = if p.width == 1 {
            &scalar_type
        } else {
            dynamic_types.push(file.make_bit_vector_type(p.width as i64, false));
            dynamic_types.last().unwrap()
        };
        let reg = wrapper.add_reg(&format!("p0_{}", p.name), dt_ref).unwrap();
        p0_regs.insert(p.name.clone(), reg);
    }
    let p0_valid_reg = wrapper.add_reg("p0_valid", &scalar_type).unwrap();

    let posedge_clk = file.make_pos_edge(&clk_port.to_expr());
    let always_p0 = if cfg.verilog_version.is_system_verilog() {
        wrapper.add_always_ff(&[&posedge_clk]).unwrap()
    } else {
        wrapper.add_always_at(&[&posedge_clk]).unwrap()
    };
    let mut sb0 = always_p0.get_statement_block();
    // Iterate in a deterministic order to ensure stable Verilog output.
    let mut p0_names: Vec<String> = p0_regs.keys().cloned().collect();
    p0_names.sort();
    for name in p0_names {
        let reg = p0_regs.get(&name).unwrap();
        let src = wrapper_inputs.get(&name).unwrap();
        sb0.add_nonblocking_assignment(&reg.to_expr(), &src.to_expr());
    }
    let zero = file
        .make_literal("bits[1]:0", &xlsynth::ir_value::IrFormatPreference::Binary)
        .unwrap();
    let rst_expr = if reset_active_low {
        file.make_not(&rst_port.to_expr())
    } else {
        rst_port.to_expr()
    };
    let tern = file.make_ternary(&rst_expr, &input_valid_port.to_expr(), &zero);
    sb0.add_nonblocking_assignment(&p0_valid_reg.to_expr(), &tern);

    // Stage processing
    let mut prev_reg: Option<xlsynth::vast::LogicRef> = None;
    let mut prev_valid = p0_valid_reg;
    let mut prev_width: u32 = 0;
    for (idx, stage_info) in stage_infos.iter().enumerate() {
        let stage_unmangled = &stages[idx].0;
        let output_width = stage_info.output_width;
        let out_dt_stage_ref = if output_width == 1 {
            &scalar_type
        } else {
            dynamic_types.push(file.make_bit_vector_type(output_width as i64, false));
            dynamic_types.last().unwrap()
        };
        let output_wire = wrapper.add_wire(&format!("stage{}_out_comb", idx), out_dt_stage_ref);

        let conn_port_names: Vec<&str> = stage_info
            .ports
            .iter()
            .filter(|p| p.name != "clk")
            .map(|p| p.name.as_str())
            .collect();
        let mut temp_exprs: Vec<xlsynth::vast::Expr> = Vec::new();
        let mut conn_expr_indices: Vec<Option<usize>> = Vec::new();

        let prev_indexable = prev_reg.as_ref().map(|w| w.to_indexable_expr());
        let mut bit_cursor: u32 = 0;
        for port in &stage_info.ports {
            if port.name == "clk" {
                // Skip; combinational stage modules have no clock port.
                continue;
            }
            if !port.is_input {
                let e = output_wire.to_expr();
                temp_exprs.push(e);
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
                continue;
            }
            if idx == 0 {
                let reg = p0_regs.get(&port.name).unwrap();
                temp_exprs.push(reg.to_expr());
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
            } else if stage_info
                .ports
                .iter()
                .filter(|p| p.is_input && p.name != "clk")
                .count()
                == 1
                && port.width == prev_width
            {
                let e = prev_reg.as_ref().unwrap().to_expr();
                temp_exprs.push(e);
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
            } else {
                let hi: i64 = (prev_width - 1 - bit_cursor) as i64;
                let lo: i64 = (hi - port.width as i64 + 1) as i64;
                let slice = file
                    .make_slice(&prev_indexable.as_ref().unwrap(), hi, lo)
                    .to_expr();
                temp_exprs.push(slice);
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
                bit_cursor += port.width;
            }
        }
        let conn_exprs: Vec<Option<&xlsynth::vast::Expr>> = conn_expr_indices
            .iter()
            .map(|opt_idx| opt_idx.map(|i| &temp_exprs[i]))
            .collect();

        let instance_name = format!("{}_i", stage_unmangled);
        wrapper.add_member_instantiation(file.make_instantiation(
            stage_unmangled,
            &instance_name,
            &[],
            &[],
            &conn_port_names,
            &conn_exprs,
        ));

        let out_reg = wrapper
            .add_reg(&format!("p{}_out", idx + 1), out_dt_stage_ref)
            .unwrap();
        let valid_reg = wrapper
            .add_reg(&format!("p{}_valid", idx + 1), &scalar_type)
            .unwrap();
        let always = if cfg.verilog_version.is_system_verilog() {
            wrapper.add_always_ff(&[&posedge_clk]).unwrap()
        } else {
            wrapper.add_always_at(&[&posedge_clk]).unwrap()
        };
        let mut sb = always.get_statement_block();
        sb.add_nonblocking_assignment(&out_reg.to_expr(), &output_wire.to_expr());
        let tern_v = file.make_ternary(&rst_expr, &prev_valid.to_expr(), &zero);
        sb.add_nonblocking_assignment(&valid_reg.to_expr(), &tern_v);

        prev_reg = Some(out_reg);
        prev_valid = valid_reg;
        prev_width = output_width;
    }

    let final_reg = prev_reg.unwrap();
    wrapper.add_member_continuous_assignment(
        file.make_continuous_assignment(&out_port.to_expr(), &final_reg.to_expr()),
    );
    if let Some(vport) = &output_valid_port {
        wrapper.add_member_continuous_assignment(
            file.make_continuous_assignment(&vport.to_expr(), &prev_valid.to_expr()),
        );
    }

    let mut text = stage_infos
        .iter()
        .map(|s| s.sv_text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    text.push_str(&file.emit());
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;
    use pretty_assertions::assert_eq;
    use xlsynth::ir_value::IrBits;
    use xlsynth_test_helpers;

    #[test]
    fn test_stitch_pipeline_tuple() {
        let dslx = "fn mul_add_cycle0(x: u32, y: u32, z: u32) -> (u32, u32) { (x * y, z) }\nfn mul_add_cycle1(partial: u32, z: u32) -> u32 { partial + z }";
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "mul_add",
            VerilogVersion::SystemVerilog,
            None,
            None,
            &[],
        )
        .unwrap();
        // Validate generated SV.
        xlsynth_test_helpers::assert_valid_sv(&result);

        let golden_path = std::path::Path::new("tests/goldens/mul_add.golden.sv");
        if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() || !golden_path.exists() {
            std::fs::write(golden_path, &result).expect("write golden");
        } else if golden_path.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            std::fs::write(golden_path, &result).expect("write golden");
        } else {
            let want = std::fs::read_to_string(golden_path).expect("read golden");
            assert_eq!(
                result.trim(),
                want.trim(),
                "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
            );
        }
    }

    #[test]
    fn test_stitch_pipeline_struct() {
        let dslx = "struct S { a: u32, b: u32 }\nfn foo_cycle0(s: S) -> S { s }\nfn foo_cycle1(s: S) -> u32 { s.a + s.b }";
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            VerilogVersion::SystemVerilog,
            None,
            None,
            &[],
        )
        .unwrap();
        xlsynth_test_helpers::assert_valid_sv(&result);

        let golden_path = std::path::Path::new("tests/goldens/foo.golden.sv");
        if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() || !golden_path.exists() {
            std::fs::write(golden_path, &result).expect("write golden");
        } else if golden_path.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            std::fs::write(golden_path, &result).expect("write golden");
        } else {
            let want = std::fs::read_to_string(golden_path).expect("read golden");
            assert_eq!(
                result.trim(),
                want.trim(),
                "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
            );
        }
    }

    #[test]
    fn test_stitch_pipeline_single_stage() {
        let dslx = "fn one_cycle0(x: u32, y: u32) -> u32 { x + y }";
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "one",
            VerilogVersion::SystemVerilog,
            None,
            None,
            &[],
        )
        .unwrap();
        xlsynth_test_helpers::assert_valid_sv(&result);

        let golden_path = std::path::Path::new("tests/goldens/one.golden.sv");
        if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() || !golden_path.exists() {
            std::fs::write(golden_path, &result).expect("write golden");
        } else if golden_path.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            std::fs::write(golden_path, &result).expect("write golden");
        } else {
            let want = std::fs::read_to_string(golden_path).expect("read golden");
            assert_eq!(
                result.trim(),
                want.trim(),
                "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
            );
        }
    }

    fn verilog_for_foo_pipeline_with_valid() -> String {
        let dslx = "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
        stitch_pipeline_with_valid(
            dslx,
            Path::new("test.x"),
            "foo",
            VerilogVersion::SystemVerilog,
            None,
            None,
            &[],
            Some("input_valid"),
            Some("output_valid"),
            Some("rst"),
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_stitch_pipeline_with_valid() {
        let _ = env_logger::builder().is_test(true).try_init();
        let result = verilog_for_foo_pipeline_with_valid();
        xlsynth_test_helpers::assert_valid_sv(&result);

        // Golden file check
        let golden_path = std::path::Path::new("tests/goldens/foo_with_valid.golden.sv");
        if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() || !golden_path.exists() {
            std::fs::write(golden_path, &result).expect("write golden");
        } else if golden_path.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            std::fs::write(golden_path, &result).expect("write golden");
        } else {
            let want = std::fs::read_to_string(golden_path).expect("read golden");
            assert_eq!(
                result.trim(),
                want.trim(),
                "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
            );
        }

        // Simulation check
        let inputs = vec![("x", IrBits::u32(5))];
        let expected = IrBits::u32(8);
        let vcd = xlsynth_test_helpers::simulate_pipeline_single_pulse(
            &result, "foo", &inputs, &expected, 2,
        )
        .expect("simulation succeeds");
        assert!(vcd.contains("$var"));
    }

    #[test]
    fn test_stitch_pipeline_signature_mismatch() {
        let dslx = r#"fn foo_cycle0(x: u32, y: u64) -> (u32, u64) { (x, y) }
fn foo_cycle1(a: u64, b: u32) -> u64 { a + b as u64 }"#;
        let result = stitch_pipeline(
            dslx,
            Path::new("test.x"),
            "foo",
            VerilogVersion::SystemVerilog,
            None,
            None,
            &[],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(err.contains("does not match"), "unexpected error: {}", err);
    }
}
