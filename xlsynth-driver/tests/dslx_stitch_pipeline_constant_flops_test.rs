// SPDX-License-Identifier: Apache-2.0

//! End-to-end validation of constant data bits in stitched pipeline registers.

use std::error::Error;
use std::process::{Command, Output};

use test_case::test_case;
use xlsynth_test_helpers::compare_golden_text;

const CONSTANT_STAGES: &str = r#"
fn foo_cycle0(x: u1) -> u1 { x ^ x }
fn foo_cycle1(x: u1) -> u1 { !x }
"#;

const CORRELATED_STAGES: &str = r#"
fn foo_cycle0(x: u1) -> (u1, u1) { (x, x) }
fn foo_cycle1(a: u1, b: u1) -> u1 { a ^ b }
"#;

/// Runs stitching with an explicit configuration to isolate ambient settings.
fn run_stitch(dslx: &str, args: &[&str]) -> Result<Output, Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let dslx_path = temp.path().join("foo.x");
    std::fs::write(&dslx_path, dslx)?;
    let config_path = temp.path().join("toolchain.toml");
    std::fs::write(&config_path, "[toolchain]\n")?;
    Ok(Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .env("NO_COLOR", "1")
        .env_remove("RUST_LOG")
        .arg("--toolchain")
        .arg(&config_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(&dslx_path)
        .arg("--dslx_top=foo")
        .args(args)
        .output()?)
}

#[test_case("--validate-no-constant-flops"; "bare_flag")]
#[test_case("--validate-no-constant-flops=true"; "explicit_true")]
fn rejects_constant_bits_at_every_registered_stage(flag: &str) {
    let output = run_stitch(CONSTANT_STAGES, &[flag]).unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    compare_golden_text(
        &String::from_utf8(output.stderr).unwrap(),
        "tests/dslx_stitch_pipeline_constant_flops.golden.txt",
    );
}

#[test]
fn disabled_validation_preserves_constant_pipelines() {
    let default = run_stitch(CONSTANT_STAGES, &[]).unwrap();
    let disabled = run_stitch(CONSTANT_STAGES, &["--validate-no-constant-flops=false"]).unwrap();
    for output in [&default, &disabled] {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stdout.is_empty());
    }
    assert_eq!(default.stdout, disabled.stdout);
}

#[test]
fn accepts_variable_pipeline_without_changing_verilog() {
    let dslx = r#"
fn foo_cycle0(x: u8) -> u8 { x + u8:1 }
fn foo_cycle1(x: u8) -> u8 { !x }
"#;
    let default = run_stitch(dslx, &[]).unwrap();
    let validated = run_stitch(dslx, &["--validate-no-constant-flops"]).unwrap();
    for output in [&default, &validated] {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stdout.is_empty());
    }
    assert_eq!(default.stdout, validated.stdout);
}

#[test]
fn rejects_constant_caused_by_correlated_stage_inputs() {
    let output = run_stitch(CORRELATED_STAGES, &["--validate-no-constant-flops"]).unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    compare_golden_text(
        &String::from_utf8(output.stderr).unwrap(),
        "tests/dslx_stitch_pipeline_correlated_constant_flops.golden.txt",
    );
}

#[test]
fn skips_final_combinational_output() {
    let output = run_stitch(
        CORRELATED_STAGES,
        &["--validate-no-constant-flops", "--flop_outputs=false"],
    )
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.stdout.is_empty());
}
