// SPDX-License-Identifier: Apache-2.0

//! CLI validation tests for `dslx-stitch-pipeline` flags.

#[test]
fn test_mutual_exclusion_dslx_top_and_stages() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Minimal DSLX with one stage function.
    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--stages")
        .arg("foo_cycle0")
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "expected failure for mutually-exclusive flags"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let expected = "the argument '--dslx_top <DSLX_TOP>' cannot be used with '--stages <CSV>'";
    assert!(
        stderr.contains(expected),
        "expected phrase not found. stderr: {}",
        stderr
    );
}

#[test]
fn test_stages_requires_output_module_name() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--stages")
        .arg("foo_cycle0")
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "expected failure when --stages lacks --output_module_name"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--output_module_name is required"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn test_output_module_name_controls_wrapper_name() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--stages")
        .arg("foo_cycle0")
        .arg("--output_module_name")
        .arg("my_wrap")
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "expected success with explicit wrapper name; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Wrapper module name should be the provided value.
    assert!(
        stdout.contains("module my_wrap"),
        "wrapper module name not found in output: {}",
        stdout
    );
}
