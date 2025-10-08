// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn test_dslx_specialize_basic() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx_source = r#"
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

fn helper<M: u32>(x: bits[M]) -> bits[M] { id(x) }

fn call() -> bits[32] { helper(bits[32]:0x0) }
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("parametric.x");
    std::fs::write(&dslx_path, dslx_source).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(driver)
        .arg("dslx-specialize")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("call")
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "dslx-specialize failed (status={});\nstdout:{}\nstderr:{}",
        output.status,
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );

    // Original parametric definitions should be removed.
    assert!(
        !stdout.contains("fn id<N"),
        "Expected parametric definition to be removed.\nSpecialized module:\n{}",
        stdout
    );
    assert!(
        !stdout.contains("fn helper<M"),
        "Expected helper parametric definition to be removed.\nSpecialized module:\n{}",
        stdout
    );

    // Specialized clones should be present.
    assert!(
        stdout.contains("fn id_"),
        "Expected a specialized clone of id().\nSpecialized module:\n{}",
        stdout
    );
    assert!(
        stdout.contains("fn helper_"),
        "Expected a specialized clone of helper().\nSpecialized module:\n{}",
        stdout
    );

    // Top-level call should remain and invoke the specialized helper.
    assert!(stdout.contains("fn call"));
}
