// SPDX-License-Identifier: Apache-2.0

//! End-to-end gate template coverage for DSLX pipeline stitching.

use std::collections::BTreeMap;
use std::process::{Command, Output};

use test_case::test_case;
use xlsynth_test_helpers::compare_golden_sv;
use xlsynth_vastly::{Signedness, Value4, compile_combo_module, eval_combo, plan_combo_eval};

const DSLX: &str = r#"
fn foo_cycle0(x: u8) -> u8 { gate!(x != u8:0, x + u8:1) }
fn foo_cycle1(x: u8) -> u8 { gate!(x != u8:0, x + u8:2) }
"#;

// Quotes, a backslash and a newline exercise the protobuf text string boundary.
const GATE_FORMAT: &str = r#"/* "custom\gate" */
assign {output} = {condition} ? {input} : {width}'d0"#;

/// Runs the actual CLI with independent toolchain and command-line templates.
fn run_stitch(
    toolchain_format: Option<&str>,
    cli_format: Option<&str>,
    explicit_stages: bool,
    system_verilog: bool,
) -> Output {
    let temp = tempfile::tempdir().unwrap();
    let dslx_path = temp.path().join("foo.x");
    std::fs::write(&dslx_path, DSLX).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    // Always supply a config so ambient toolchain settings cannot affect the
    // test.
    let mut codegen = toml::Table::new();
    if let Some(format) = toolchain_format {
        codegen.insert(
            "gate_format".to_string(),
            toml::Value::String(format.to_string()),
        );
    }
    let config = toml::toml! { [toolchain] };
    let mut config = config;
    config
        .get_mut("toolchain")
        .unwrap()
        .as_table_mut()
        .unwrap()
        .insert("codegen".to_string(), toml::Value::Table(codegen));
    let config_path = temp.path().join("toolchain.toml");
    std::fs::write(&config_path, toml::to_string(&config).unwrap()).unwrap();
    command.arg("--toolchain").arg(&config_path);
    command
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(&dslx_path)
        .arg(format!("--use_system_verilog={system_verilog}"));
    if explicit_stages {
        command.args(["--stages=foo_cycle0,foo_cycle1", "--output_module_name=foo"]);
    } else {
        command.args(["--dslx_top=foo"]);
    }
    if let Some(format) = cli_format {
        command.arg("--gate_format").arg(format);
    }
    command.output().unwrap()
}

/// Checks exact emitted text and parses/evaluates each generated stage.
fn check_output(output: Output, golden: &str) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    compare_golden_sv(&text, golden);
    let modules: Vec<_> = text.split_inclusive("endmodule").collect();
    // The final chunk is the trailing newline, after two stages and the
    // wrapper.
    assert_eq!(modules.len(), 4);
    for (index, stage) in modules[..2].iter().enumerate() {
        let module = compile_combo_module(stage).unwrap();
        let plan = plan_combo_eval(&module).unwrap();
        for input in [0, 1, 127, 254, 255] {
            let inputs = BTreeMap::from([(
                "x".to_string(),
                Value4::from_u64(8, Signedness::Unsigned, input),
            )]);
            let outputs = eval_combo(&module, &plan, &inputs).unwrap();
            let expected = if input == 0 {
                0
            } else {
                (input + index as u64 + 1) & 255
            };
            assert_eq!(
                outputs["out"],
                Value4::from_u64(8, Signedness::Unsigned, expected)
            );
        }
    }
}

#[test_case(false, false; "implicit_verilog")]
#[test_case(false, true; "implicit_system_verilog")]
#[test_case(true, false; "explicit_verilog")]
#[test_case(true, true; "explicit_system_verilog")]
fn gate_template_reaches_every_selected_stage(explicit_stages: bool, system_verilog: bool) {
    let golden = if system_verilog {
        "tests/dslx_stitch_pipeline_gate_format.golden.sv"
    } else {
        "tests/dslx_stitch_pipeline_gate_format.golden.v"
    };
    // A bad config template makes a missing CLI override fail visibly.
    check_output(
        run_stitch(
            Some("bad {unknown}"),
            Some(GATE_FORMAT),
            explicit_stages,
            system_verilog,
        ),
        golden,
    );
    check_output(
        run_stitch(Some(GATE_FORMAT), None, explicit_stages, system_verilog),
        golden,
    );
    check_output(
        run_stitch(None, Some(GATE_FORMAT), explicit_stages, system_verilog),
        golden,
    );
}

#[test_case(false; "verilog")]
#[test_case(true; "system_verilog")]
fn absent_or_empty_gate_template_preserves_default(system_verilog: bool) {
    let golden = if system_verilog {
        "tests/dslx_stitch_pipeline_gate_default.golden.sv"
    } else {
        "tests/dslx_stitch_pipeline_gate_default.golden.v"
    };
    check_output(run_stitch(None, None, false, system_verilog), golden);
    check_output(
        run_stitch(Some("bad {unknown}"), Some(""), false, system_verilog),
        golden,
    );
}

#[test_case(false; "toolchain")]
#[test_case(true; "cli")]
fn invalid_gate_template_reports_stitch_error(from_cli: bool) {
    let output = if from_cli {
        run_stitch(None, Some("assign {output} = {unknown}"), false, true)
    } else {
        run_stitch(Some("assign {output} = {unknown}"), None, true, false)
    };
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("stitch error"), "{stderr}");
    assert!(stderr.contains("unknown"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
