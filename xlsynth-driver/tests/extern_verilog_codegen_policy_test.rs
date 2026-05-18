// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

const IR_WITH_VERILOG_FFI: &str = r#"package extern_verilog_policy

#[ffi_proto("""code_template: "assign {return} = {x};"
""")]
fn verilog_passthrough(x: bits[8] id=1) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}

top fn main(x: bits[8] id=2) -> bits[8] {
  ret x: bits[8] = param(name=x, id=2)
}
"#;

fn write_ir_input(temp_dir: &tempfile::TempDir) -> std::path::PathBuf {
    let ir_path = temp_dir.path().join("extern_verilog.ir");
    std::fs::write(&ir_path, IR_WITH_VERILOG_FFI).expect("write IR input");
    ir_path
}

fn write_toolchain_config(
    temp_dir: &tempfile::TempDir,
    tool_path: &std::path::Path,
) -> std::path::PathBuf {
    let config_path = temp_dir.path().join("xlsynth-toolchain.toml");
    std::fs::write(
        &config_path,
        format!(
            "[toolchain]\ntool_path = {:?}\n",
            tool_path.to_str().expect("tool path should be utf-8")
        ),
    )
    .expect("write toolchain config");
    config_path
}

fn assert_extern_verilog_rejected(output: std::process::Output, subcommand: &str) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "{subcommand} unexpectedly succeeded:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("--allow_extern_verilog=true"),
        "{subcommand} rejection should explain the opt-in flag:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("verilog_passthrough"),
        "{subcommand} rejection should name the FFI function:\nstdout: {stdout}\nstderr: {stderr}"
    );
}

#[test]
fn ir2pipeline_rejects_verilog_ffi_ir_when_disabled() {
    let temp_dir = tempfile::tempdir().expect("make temp dir");
    let ir_path = write_ir_input(&temp_dir);
    let toolchain_path = write_toolchain_config(&temp_dir, temp_dir.path());
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_path)
        .arg("ir2pipeline")
        .arg(ir_path)
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("unit")
        .arg("--allow_extern_verilog=false")
        .output()
        .expect("run ir2pipeline");

    assert_extern_verilog_rejected(output, "ir2pipeline");
}

#[test]
fn ir2combo_rejects_verilog_ffi_ir_when_disabled() {
    let temp_dir = tempfile::tempdir().expect("make temp dir");
    let ir_path = write_ir_input(&temp_dir);
    let toolchain_path = write_toolchain_config(&temp_dir, temp_dir.path());
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_path)
        .arg("ir2combo")
        .arg(ir_path)
        .arg("--delay_model")
        .arg("unit")
        .arg("--allow_extern_verilog=false")
        .output()
        .expect("run ir2combo");

    assert_extern_verilog_rejected(output, "ir2combo");
}

#[cfg(unix)]
fn write_executable_script(path: &std::path::Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;

    std::fs::write(path, text).expect("write tool script");
    let mut permissions = std::fs::metadata(path)
        .expect("stat tool script")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).expect("mark tool script executable");
}

#[cfg(unix)]
#[test]
fn dslx2pipeline_rejects_generated_verilog_ffi_ir_when_disabled() {
    let temp_dir = tempfile::tempdir().expect("make temp dir");
    let fake_tools = temp_dir.path().join("fake-tools");
    std::fs::create_dir(&fake_tools).expect("create fake tool dir");
    write_executable_script(
        &fake_tools.join("ir_converter_main"),
        "#!/bin/sh\ncat \"$FAKE_EXTERN_IR_PATH\"\n",
    );
    write_executable_script(&fake_tools.join("opt_main"), "#!/bin/sh\ncat \"$1\"\n");

    let emitted_ir_path = write_ir_input(&temp_dir);
    let dslx_path = temp_dir.path().join("input.x");
    std::fs::write(&dslx_path, "fn main(x: u8) -> u8 { x }\n").expect("write DSLX input");
    let toolchain_path = write_toolchain_config(&temp_dir, &fake_tools);
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_path)
        .arg("dslx2pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path)
        .arg("--dslx_top")
        .arg("main")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("unit")
        .arg("--allow_extern_verilog=false")
        .env("FAKE_EXTERN_IR_PATH", emitted_ir_path)
        .output()
        .expect("run dslx2pipeline");

    assert_extern_verilog_rejected(output, "dslx2pipeline");
}
