// SPDX-License-Identifier: Apache-2.0
//! Simple utility to stitch together pipeline stages defined in DSLX.

use std::path::Path;

use xlsynth::{
    convert_dslx_to_ir, mangle_dslx_name, optimize_ir, schedule_and_codegen, DslxConvertOptions,
};

/// Generates SystemVerilog for pipeline stages `top_cycle0`, `top_cycle1`, ...
/// in `dslx` and stitches them together into a wrapper module.
///
/// The resulting SystemVerilog contains the stage modules followed by a wrapper
/// module named `<top>_pipeline` that instantiates each stage.
pub fn stitch_pipeline(
    dslx: &str,
    path: &Path,
    top: &str,
) -> Result<String, xlsynth::XlsynthError> {
    let conv = convert_dslx_to_ir(dslx, path, &DslxConvertOptions::default())?;
    let ir = conv.ir;

    // Discover stage functions.
    let mut stages: Vec<String> = Vec::new();
    for i in 0.. {
        let fn_name = format!("{top}_cycle{i}");
        let mangled = match mangle_dslx_name(path.file_stem().unwrap().to_str().unwrap(), &fn_name)
        {
            Ok(v) => v,
            Err(_) => break,
        };
        if ir.get_function(&mangled).is_err() {
            break;
        }
        stages.push(fn_name);
    }
    if stages.is_empty() {
        return Err(xlsynth::XlsynthError("no pipeline stages found".into()));
    }

    let mut file = xlsynth::vast::VastFile::new(xlsynth::vast::VastFileType::SystemVerilog);
    let bit_type = file.make_bit_vector_type(32, false);

    let mut wrapper = file.add_module(&format!("{top}_pipeline"));
    let clk_port = wrapper.add_input("clk", &bit_type);
    let x_port = wrapper.add_input("x", &bit_type);
    let y_port = wrapper.add_input("y", &bit_type);
    let out_port = wrapper.add_output("out", &bit_type);

    let mut prev_wire = Some(wrapper.add_wire("stage0_out", &bit_type));

    let mut sv_modules = Vec::new();
    for (idx, stage) in stages.iter().enumerate() {
        let opt = optimize_ir(&ir, stage)?;
        let sched = "delay_model: \"unit\"\npipeline_stages: 1";
        let flop_inputs = "flop_inputs: true";
        let flop_outputs = if idx + 1 == stages.len() {
            "flop_outputs: true"
        } else {
            "flop_outputs: false"
        };
        let codegen = format!(
            "register_merge_strategy: STRATEGY_IDENTITY_ONLY\ngenerator: GENERATOR_KIND_PIPELINE\nmodule_name: \"{stage}\"\n{flop_inputs}\n{flop_outputs}"
        );
        let result = schedule_and_codegen(&opt, sched, &codegen)?;
        sv_modules.push(result.get_verilog_text()?);

        let instance_name = format!("{stage}_i");
        let output_wire = if idx + 1 == stages.len() {
            wrapper.add_wire("final_out", &bit_type)
        } else {
            wrapper.add_wire(&format!("stage{}_out", idx + 1), &bit_type)
        };
        let conn_names = ["clk", "x", "y", "out"];
        let clk_e = clk_port.to_expr();
        let x_e = x_port.to_expr();
        let y_e = y_port.to_expr();
        let out_e = output_wire.to_expr();
        let conns = [Some(&clk_e), Some(&x_e), Some(&y_e), Some(&out_e)];
        wrapper.add_member_instantiation(file.make_instantiation(
            stage,
            &instance_name,
            &[],
            &[],
            &conn_names,
            &conns,
        ));
        prev_wire = Some(output_wire);
    }
    wrapper.add_member_continuous_assignment(
        file.make_continuous_assignment(&out_port.to_expr(), &prev_wire.unwrap().to_expr()),
    );

    let mut text = sv_modules.join("\n");
    text.push_str(&file.emit());
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stitch_pipeline_tuple() {
        let dslx = "fn mul_add_cycle0(x: u32, y: u32, z: u32) -> (u32, u32) { (x * y, z) }\nfn mul_add_cycle1(partial: u32, z: u32) -> u32 { partial + z }";
        let result = stitch_pipeline(dslx, Path::new("test.x"), "mul_add").unwrap();
        assert!(result.contains("mul_add_cycle0"));
        assert!(result.contains("mul_add_cycle1"));
        assert!(result.contains("mul_add_pipeline"));
    }

    #[test]
    fn test_stitch_pipeline_struct() {
        let dslx = "struct S { a: u32, b: u32 }\nfn foo_cycle0(s: S) -> S { s }\nfn foo_cycle1(s: S) -> u32 { s.a + s.b }";
        let result = stitch_pipeline(dslx, Path::new("test.x"), "foo").unwrap();
        assert!(result.contains("foo_cycle0"));
        assert!(result.contains("foo_cycle1"));
        assert!(result.contains("foo_pipeline"));
    }
}
