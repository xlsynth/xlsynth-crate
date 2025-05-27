// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn test_invoke_mcmc_driver_with_sample() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let exe = env!("CARGO_BIN_EXE_mcmc-driver");
    let mut cmd = Command::new(exe);
    cmd.arg("sample://ripple_carry_adder:8")
        .arg("-n")
        .arg("10")
        .arg("--paranoid")
        .arg("--output")
        .arg(temp_dir.path());
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        cmd.env("RUST_LOG", rust_log);
    }
    let output = cmd.output().expect("mcmc-driver should run");
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
    assert!(temp_dir.path().join("best.g8r").exists());
}
