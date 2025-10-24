// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn test_dslx_fn_eval_with_assert_itok_pir_interp() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_fn_eval_itok");
    let dir = tmp.path();

    // This function asserts input < 10, which requires itok in IR.
    let dslx_path = dir.join("g.x");
    let dslx_src = r#"
        fn g(x: u32) -> u32 { assert!(x < u32:10, "x_lt_10"); x }
    "#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    // Two inputs: one ok (9), one failing (11).
    let irvals_path = dir.join("inputs.irvals");
    let irvals = "(bits[32]:9)\n(bits[32]:11)\n";
    std::fs::write(&irvals_path, irvals).expect("write irvals");

    // Run with PIR interpreter to observe assertion failure on second line.
    let output = Command::new(driver)
        .arg("dslx-fn-eval")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("g")
        .arg("--input_ir_path")
        .arg(irvals_path.to_str().unwrap())
        .arg("--eval_mode")
        .arg("pir-interp")
        .output()
        .expect("run driver");

    // We expect failure on the second line -> non-zero exit.
    assert!(
        !output.status.success(),
        "expected failure for assertion-triggering input"
    );

    // Show child's outputs to aid debugging on failure.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("child.stdout:\n{}", stdout);
    eprintln!("child.stderr:\n{}", stderr);

    // Ensure stderr mentions assertion.
    assert!(stderr.contains("assertion failure") || stderr.contains("assertion failure(s)"));
}
