// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn dslx_block_cli_honors_warnings_as_errors() {
    let temporary = tempfile::tempdir().unwrap();
    let source_path = temporary.path().join("warning_block.x");
    std::fs::write(
        &source_path,
        r#"
pub block warning_block(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let unused = x + u8:1;
  assign y = x;
}

"#,
    )
    .unwrap();
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let rejected = Command::new(driver)
        .arg("dslx-block2ir")
        .arg("--dslx_input_file")
        .arg(&source_path)
        .arg("--warnings_as_errors")
        .arg("true")
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr)
            .contains("warnings found with warnings-as-errors enabled"),
        "{}",
        String::from_utf8_lossy(&rejected.stderr)
    );

    let accepted = Command::new(driver)
        .arg("dslx-block2ir")
        .arg("--dslx_input_file")
        .arg(&source_path)
        .arg("--warnings_as_errors")
        .arg("false")
        .output()
        .unwrap();
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    assert!(String::from_utf8_lossy(&accepted.stdout).contains("top block warning_block"));
}

#[test]
fn dslx_block2sv_rejects_pipeline_mutation_flags_at_argument_parsing() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let rejected = Command::new(driver)
        .arg("dslx-block2sv")
        .arg("--dslx_input_file")
        .arg("unused.x")
        .arg("--flop_inputs")
        .arg("true")
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unexpected argument"));
}

#[test]
fn dslx_block2sv_rejects_reserved_module_name_before_codegen() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source_path = temp_dir.path().join("valid_block.x");
    std::fs::write(
        &source_path,
        r#"
pub block valid_block(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = true;
}
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("dslx-block2sv")
        .arg("--dslx_input_file")
        .arg(source_path)
        .arg("--module_name")
        .arg("module")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not an unreserved SystemVerilog identifier"),
        "{stderr}"
    );
    assert!(!stderr.contains("requires --toolchain"), "{stderr}");
}
