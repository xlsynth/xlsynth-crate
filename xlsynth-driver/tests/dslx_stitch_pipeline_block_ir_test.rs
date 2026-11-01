// SPDX-License-Identifier: Apache-2.0

//! Exercise stitched block IR through the same CLI and RTL that users run.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use xlsynth_pir::ir_parser::Parser;
use xlsynth_test_helpers::compare_golden_text;
use xlsynth_vastly::{
    FixtureRunner, LogicBit, Signedness, Value4, compile_combo_module, compile_pipeline_module,
    eval_combo, plan_combo_eval,
};

fn driver(directory: &Path, arguments: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .env("NO_COLOR", "1")
        .args(arguments)
        .output()
}

/// Captures successful stdout and preserves the driver's diagnostic on failure.
fn stdout(directory: &Path, arguments: &[&str]) -> anyhow::Result<String> {
    let output = driver(directory, arguments)?;
    anyhow::ensure!(
        output.status.success(),
        "{arguments:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?)
}

/// Validates the emitted package with pinned XLS before handing it to block2sv.
fn render_block(ir: &str) -> anyhow::Result<String> {
    xlsynth::IrPackage::parse_ir(ir, None)?;
    let directory = tempfile::tempdir()?;
    fs::write(directory.path().join("pipeline.ir"), ir)?;
    stdout(directory.path(), &["block2sv", "pipeline.ir"])
}

#[test]
fn emitted_source_locations_refer_to_the_original_dslx_file() {
    let directory = tempfile::tempdir().unwrap();
    let source = Path::new(r"dir\z").join("alu.pipe.x");
    fs::create_dir_all(directory.path().join(source.parent().unwrap())).unwrap();
    fs::write(
        directory.path().join(&source),
        "fn alu_cycle0(x: u8) -> u8 { x + u8:1 }",
    )
    .unwrap();
    let ir = stdout(
        directory.path(),
        &[
            "dslx-stitch-pipeline",
            "--output-format=block-ir",
            "--dslx_input_file",
            source.to_str().unwrap(),
            "--dslx_top=alu",
        ],
    )
    .unwrap();
    let package = Parser::new(&ir).parse_package().unwrap();
    assert_eq!(package.name, "alu.pipe");
    assert_eq!(package.file_table.id_to_path[&0], source.to_str().unwrap());
    // Native XLS prints the stored filename verbatim. Check both consumers
    // resolve the stage's positions to the same file.
    let native = xlsynth::IrPackage::parse_ir(&ir, None).unwrap().to_string();
    assert_eq!(
        native.lines().find(|line| line.starts_with("file_number ")),
        Some(format!("file_number 0 \"{}\"", source.display()).as_str())
    );
}

#[test]
fn muladd_spec_and_flags_emit_the_same_block_and_take_two_edges() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/stitch-equiv");
    let spec = example.join("muladd.stitch.toml");
    let source = example.join("muladd.x");
    let ir = stdout(
        &example,
        &[
            "dslx-stitch-pipeline",
            "--output-format=block-ir",
            "--spec",
            spec.to_str().unwrap(),
        ],
    )
    .unwrap();
    let from_flags = stdout(
        &example,
        &[
            "dslx-stitch-pipeline",
            "--output-format=block-ir",
            "--dslx_input_file",
            source.to_str().unwrap(),
            "--dslx_top=muladd",
            "--flop_inputs=false",
            "--flop_outputs=true",
        ],
    )
    .unwrap();
    assert_eq!(ir, from_flags);
    // Compare the checkout-independent source name in the golden.
    let mut package = Parser::new(&ir).parse_package().unwrap();
    let filename = package.file_table.id_to_path.get_mut(&0).unwrap();
    assert_eq!(Path::new(filename), source);
    *filename = "muladd.x".into();
    compare_golden_text(
        &package.to_string(),
        "tests/dslx_stitch_pipeline_muladd.golden.ir",
    );

    let rtl = render_block(&ir).unwrap();
    let module = compile_pipeline_module(&rtl).unwrap();
    assert_eq!(module.module_name, "muladd");
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let x = ports.input_with_width("x", 8).unwrap();
    let y = ports.input_with_width("y", 8).unwrap();
    let z = ports.input_with_width("z", 8).unwrap();
    let unknown = Value4::new(8, Signedness::Unsigned, vec![LogicBit::X; 8]);
    let mut before = unknown.clone();
    for (operands, expected) in [
        ([7, 9, 5], unknown),
        ([255, 2, 3], Value4::from_u64(8, Signedness::Unsigned, 68)),
        ([0, 0, 0], Value4::from_u64(8, Signedness::Unsigned, 1)),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_u64(&x, operands[0])?;
                drive.set_u64(&y, operands[1])?;
                drive.set_u64(&z, operands[2])
            })
            .unwrap();
        assert_eq!(edge.pre_outputs["out"], before);
        assert_eq!(edge.outputs["out"], expected);
        before = expected;
    }
}

#[test]
fn packed_stage_arguments_bubbles_and_synchronous_reset() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("pipe.x"),
        r#"fn split(samples: u4[2]) -> (u3, u5) {
    (samples[u32:0] as u3, samples[u32:1] as u5)
}
fn join_bits(lo: u3, hi: u5) -> u8 { lo ++ hi }
fn combine(lo: u3, hi: u5) -> u8 { join_bits(lo, hi) }"#,
    )
    .unwrap();
    fs::write(
        directory.path().join("pipe.toml"),
        r#"dslx_input_file = "pipe.x"
stages = ["split", "combine"]
output_module_name = "repacked"
clock_name = "sample_clk"
input_valid_signal = "present"
output_valid_signal = "ready_out"
reset = "rst_n"
reset_active_low = true"#,
    )
    .unwrap();
    let ir = stdout(
        directory.path(),
        &[
            "dslx-stitch-pipeline",
            "--spec=pipe.toml",
            "--output-format=block-ir",
        ],
    )
    .unwrap();
    let rtl = render_block(&ir).unwrap();
    let module = compile_pipeline_module(&rtl).unwrap();
    assert_eq!(module.module_name, "repacked");
    assert_eq!(module.clk_name, "sample_clk");
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let samples = ports.input_with_width("samples", 8).unwrap();
    let present = ports.input_with_width("present", 1).unwrap();
    let rst_n = ports.input_with_width("rst_n", 1).unwrap();

    // 0xb6 represents array [6, 11]. The tuple (u3:6, u5:11) packs to
    // 0xcb, so the low-first formals get (u3:3, u5:25), yielding 0x79.
    // Input, inter-stage and output registers each consume one edge.
    let mut previous = (None, None);
    for (sample, valid, reset_n, expected_valid, expected_data) in [
        (0, 0, 0, 0, None),
        (0xb6, 1, 1, 0, None),
        (0x18, 1, 1, 0, None),
        (0xff, 0, 1, 1, Some(0x79)),
        // Reset clears every valid register on the edge. The output data
        // register still loads the second sample under its incoming valid.
        (0xaa, 1, 0, 0, Some(0x20)),
        (0x00, 0, 1, 0, Some(0x20)),
        (0x7d, 1, 1, 0, Some(0x20)),
        (0xee, 0, 1, 0, Some(0x20)),
        (0xff, 0, 1, 1, Some(0xf4)),
        (0x00, 0, 1, 0, Some(0xf4)),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_u64(&samples, sample)?;
                drive.set_u64(&present, valid)?;
                drive.set_u64(&rst_n, reset_n)
            })
            .unwrap();
        let before = &edge.pre_outputs;
        assert_eq!(
            (
                before["ready_out"].to_u64_if_known(),
                before["out"].to_u64_if_known()
            ),
            previous,
            "before edge {}",
            edge.cycle
        );
        previous = (Some(expected_valid), expected_data);
        let after = &edge.outputs;
        assert_eq!(
            (
                after["ready_out"].to_u64_if_known(),
                after["out"].to_u64_if_known()
            ),
            previous,
            "after edge {}",
            edge.cycle
        );
    }
}

#[test]
fn packed_tuple_with_an_empty_array_round_trips_through_a_stage() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("empty_array.x"),
        "fn echo(x: (u4[0], u4)) -> (u4[0], u4) { x }",
    )
    .unwrap();
    let ir = stdout(
        directory.path(),
        &[
            "dslx-stitch-pipeline",
            "--output-format=block-ir",
            "--dslx_input_file=empty_array.x",
            "--stages=echo",
            "--output_module_name=echo_pipe",
            "--flop_inputs=false",
            "--flop_outputs=false",
        ],
    )
    .unwrap();
    let rtl = render_block(&ir).unwrap();
    let module = compile_combo_module(&rtl).unwrap();
    let plan = plan_combo_eval(&module).unwrap();
    for nibble in 0..16 {
        let x = Value4::from_u64(4, Signedness::Unsigned, nibble);
        let inputs = [
            ("clk".to_owned(), Value4::zeros(1, Signedness::Unsigned)),
            ("x".to_owned(), x.clone()),
        ]
        .into();
        let outputs = eval_combo(&module, &plan, &inputs).unwrap();
        assert_eq!(outputs["out"], x);
    }
}

#[test]
fn a_nullary_stage_without_valid_needs_no_input_register() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("constant.x"),
        "fn seed() -> u8 { u8:73 }",
    )
    .unwrap();
    let ir = stdout(
        directory.path(),
        &[
            "dslx-stitch-pipeline",
            "--output-format=block-ir",
            "--dslx_input_file=constant.x",
            "--stages=seed",
            "--output_module_name=constant_pipe",
            "--flop_inputs=true",
            "--flop_outputs=false",
        ],
    )
    .unwrap();
    let rtl = render_block(&ir).unwrap();
    let module = compile_combo_module(&rtl).unwrap();
    let plan = plan_combo_eval(&module).unwrap();
    let inputs = [("clk".to_owned(), Value4::zeros(1, Signedness::Unsigned))].into();
    let outputs = eval_combo(&module, &plan, &inputs).unwrap();
    assert_eq!(outputs["out"].to_u64_if_known(), Some(73));
}

#[test]
fn block_output_rejects_xls_keywords_in_wrapper_and_control_names() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/stitch-equiv");
    for (flag, block, name) in [
        (
            "--output_module_name=scheduled_fn",
            "scheduled_fn",
            "scheduled_fn",
        ),
        (
            "--input_valid_signal=active_inputs_valid",
            "muladd",
            "active_inputs_valid",
        ),
    ] {
        let output = driver(
            &example,
            &[
                "dslx-stitch-pipeline",
                "--output-format=block-ir",
                "--dslx_input_file=muladd.x",
                "--dslx_top=muladd",
                flag,
            ],
        )
        .unwrap();
        assert_eq!(output.status.code(), Some(1), "{flag}");
        assert!(output.stdout.is_empty(), "{flag}");
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            format!(
                "xlsynth-driver: could not build pipeline block '{block}': invalid IR identifier: {name:?}: stitch error\n"
            ),
            "{flag}"
        );
    }
}

#[test]
fn block_output_rejects_incompatible_rendering_options() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/stitch-equiv");
    for (flag, diagnostic) in [
        (
            "--gate_format=template",
            "--gate_format requires --output-format=rtl",
        ),
        (
            "--use_system_verilog=false",
            "--use_system_verilog requires --output-format=rtl",
        ),
        (
            "--array_index_bounds_checking=false",
            "--output-format=block-ir uses XLS array indexing semantics and requires --array_index_bounds_checking=true",
        ),
    ] {
        let output = driver(
            &example,
            &[
                "dslx-stitch-pipeline",
                "--output-format=block-ir",
                "--spec=muladd.stitch.toml",
                flag,
            ],
        )
        .unwrap();
        assert_eq!(output.status.code(), Some(1), "{flag}");
        assert!(output.stdout.is_empty(), "{flag}");
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            format!("xlsynth-driver: {diagnostic}\n"),
            "{flag}"
        );
    }
}
