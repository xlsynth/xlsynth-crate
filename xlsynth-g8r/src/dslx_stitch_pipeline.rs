// SPDX-License-Identifier: Apache-2.0
//! Simple utility to stitch together pipeline stages defined in DSLX.

use std::collections::HashMap;
use std::path::Path;

use xlsynth::{
    convert_dslx_to_ir, mangle_dslx_name, optimize_ir, schedule_and_codegen, DslxConvertOptions,
};

/// Immutable configuration passed around while stitching a pipeline.
#[derive(Clone)]
struct PipelineCfg<'a> {
    ir: &'a xlsynth::ir_package::IrPackage,
    top: &'a str,
    use_system_verilog: bool,
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

/// Run XLS optimise + schedule + codegen for a stage and gather `StageInfo`.
fn make_stage_info(
    cfg: &PipelineCfg,
    stage_name_unmangled: &str,
    stage_mangled: &str,
    is_last: bool,
) -> Result<StageInfo, xlsynth::XlsynthError> {
    let opt = optimize_ir(cfg.ir, stage_mangled)?;
    let sched = "delay_model: \"unit\"\npipeline_stages: 1";
    let flop_inputs = "flop_inputs: true";
    let flop_outputs = if is_last {
        "flop_outputs: true"
    } else {
        "flop_outputs: false"
    };
    let codegen = format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\ngenerator: GENERATOR_KIND_PIPELINE\nmodule_name: \"{stage}\"\nuse_system_verilog: {sv}\n{fi}\n{fo}",
        stage = stage_name_unmangled,
        sv = cfg.use_system_verilog,
        fi = flop_inputs,
        fo = flop_outputs
    );
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

/// Generates SystemVerilog for pipeline stages `top_cycle0`, `top_cycle1`, ...
/// in `dslx` and stitches them together into a wrapper module.
///
/// The resulting SystemVerilog contains the stage modules followed by a wrapper
/// module named `<top>_pipeline` that instantiates each stage.
pub fn stitch_pipeline(
    dslx: &str,
    path: &Path,
    top: &str,
    use_system_verilog: bool,
    explicit_stages: Option<&[String]>,
) -> Result<String, xlsynth::XlsynthError> {
    let conv = convert_dslx_to_ir(dslx, path, &DslxConvertOptions::default())?;
    let ir = conv.ir;

    let cfg = PipelineCfg {
        ir: &ir,
        top,
        use_system_verilog,
    };

    let stages = discover_stage_names(&ir, path, top, explicit_stages)?;

    // For each stage run codegen immediately so we can parse its port list.
    let mut stage_infos = Vec::with_capacity(stages.len());
    for (idx, (stage_unmangled, stage_mangled)) in stages.iter().enumerate() {
        let info = make_stage_info(
            &cfg,
            stage_unmangled,
            stage_mangled,
            idx + 1 == stages.len(),
        )?;
        stage_infos.push(info);
    }

    // Build the wrapper using VAST.
    let file_type = if use_system_verilog {
        xlsynth::vast::VastFileType::SystemVerilog
    } else {
        xlsynth::vast::VastFileType::Verilog
    };
    let mut file = xlsynth::vast::VastFile::new(file_type);

    let scalar_type = file.make_scalar_type(); // 1-bit default
                                               // Keep any dynamically created VAST data types alive for the lifetime of
                                               // this function (until we emit the file) by storing them in this Vec.
    let mut dynamic_types: Vec<xlsynth::vast::VastDataType> = Vec::new();

    let mut wrapper = file.add_module(&format!("{top}_pipeline"));
    let clk_port = wrapper.add_input("clk", &scalar_type);

    // Input ports come from first stage inputs excluding clk.
    let first_stage_ports = &stage_infos[0].ports;
    let mut wrapper_inputs: HashMap<String, xlsynth::vast::LogicRef> = HashMap::new();
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

    // Wires and instantiations
    let mut prev_wire: Option<xlsynth::vast::LogicRef> = None;
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
        let output_wire_name = if idx + 1 == stage_infos.len() {
            "final_out".to_string()
        } else {
            format!("stage{}_out", idx)
        };
        let output_wire = wrapper.add_wire(&output_wire_name, out_dt_stage_ref);

        // Build connection names and expressions in order they appear in port list.
        let conn_port_names: Vec<&str> = stage_info.ports.iter().map(|p| p.name.as_str()).collect();

        // Keep expression objects alive.
        let mut temp_exprs: Vec<xlsynth::vast::Expr> = Vec::new();
        let mut conn_expr_indices: Vec<Option<usize>> = Vec::new();

        // For slicing we need prev output indexable expr.
        let prev_indexable = prev_wire.as_ref().map(|w| w.to_indexable_expr());

        // For multi-input slicing we accumulate cursor.
        let mut bit_cursor: u32 = 0;

        for port in &stage_info.ports {
            if port.name == "clk" {
                let e = clk_port.to_expr();
                temp_exprs.push(e);
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
                continue;
            }

            if !port.is_input {
                // Output port: connect the wire we just created.
                let e = output_wire.to_expr();
                temp_exprs.push(e);
                conn_expr_indices.push(Some(temp_exprs.len() - 1));
                continue;
            }

            // Input port (non-clk)
            if idx == 0 {
                // Connect from wrapper input ports.
                if let Some(lr) = wrapper_inputs.get(&port.name) {
                    let e = lr.to_expr();
                    temp_exprs.push(e);
                    conn_expr_indices.push(Some(temp_exprs.len() - 1));
                } else {
                    // Should not happen.
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
                    let e = prev_wire.as_ref().unwrap().to_expr();
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

        prev_wire = Some(output_wire);
        prev_width = output_width;
    }

    // Assign wrapper.out = final_out
    if let Some(final_wire) = &prev_wire {
        wrapper.add_member_continuous_assignment(
            file.make_continuous_assignment(&out_port.to_expr(), &final_wire.to_expr()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use xlsynth_test_helpers;

    #[test]
    fn test_stitch_pipeline_tuple() {
        let dslx = "fn mul_add_cycle0(x: u32, y: u32, z: u32) -> (u32, u32) { (x * y, z) }\nfn mul_add_cycle1(partial: u32, z: u32) -> u32 { partial + z }";
        let result = stitch_pipeline(dslx, Path::new("test.x"), "mul_add", true, None).unwrap();
        // Validate generated SV.
        xlsynth_test_helpers::assert_valid_sv(&result);

        let golden_path = std::path::Path::new("tests/goldens/mul_add_pipeline.golden.sv");
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
        let result = stitch_pipeline(dslx, Path::new("test.x"), "foo", true, None).unwrap();
        xlsynth_test_helpers::assert_valid_sv(&result);

        let golden_path = std::path::Path::new("tests/goldens/foo_pipeline.golden.sv");
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
        let result = stitch_pipeline(dslx, Path::new("test.x"), "one", true, None).unwrap();
        xlsynth_test_helpers::assert_valid_sv(&result);

        let golden_path = std::path::Path::new("tests/goldens/one_pipeline.golden.sv");
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
}
