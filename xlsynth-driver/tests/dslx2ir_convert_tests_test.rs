// SPDX-License-Identifier: Apache-2.0

use std::process::Command;
use test_case::test_case;

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS environment variable must be set");
    format!("{}\ntool_path = \"{}\"", toolchain_toml_contents, tool_path)
}

#[test_case(true; "convert_tests_true")]
#[test_case(false; "convert_tests_false")]
fn test_dslx2ir_convert_tests_flag(convert_tests: bool) {
    let _ = env_logger::builder().is_test(true).try_init();

    // Minimal DSLX with a normal function and a #[test] function.
    let dslx_source = r#"
fn inc(x: u32) -> u32 { x + u32:1 }

#[test]
fn my_test() { inc(u32:41); }
"#;

    let temp_dir = tempfile::tempdir().unwrap();

    // Write toolchain TOML that points to the external XLS tools.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml).unwrap();

    // Write the DSLX file.
    let dslx_path = temp_dir.path().join("mod_under_test.x");
    std::fs::write(&dslx_path, dslx_source).unwrap();

    // Invoke the driver.
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--convert_tests")
        .arg(if convert_tests { "true" } else { "false" })
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "dslx2ir failed (status={});\nstdout:{}\nstderr:{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    // Always contains the normal function.
    assert!(
        stdout.contains("fn __mod_under_test__inc("),
        "Expected normal function present in IR.\nIR stdout:\n{}",
        stdout
    );

    // Test function presence depends on --convert_tests flag.
    if convert_tests {
        assert!(
            stdout.contains("fn __itok__mod_under_test__my_test("),
            "Expected test function to be present when --convert_tests=true.\nIR stdout:\n{}",
            stdout
        );
    } else {
        assert!(
            !stdout.contains("fn __itok__mod_under_test__my_test("),
            "Did not expect test function to be present when --convert_tests=false.\nIR stdout:\n{}",
            stdout
        );
    }
}

#[test]
fn test_dslx2ir_error_on_convert_tests_with_top() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx_source = "fn main(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml).unwrap();

    let dslx_path = temp_dir.path().join("mod_under_test.x");
    std::fs::write(&dslx_path, dslx_source).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--convert_tests")
        .arg("true")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "Expected failure when --convert_tests and --dslx_top are both provided. stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot be combined"),
        "Expected helpful error message; got: {}",
        stderr
    );
}
