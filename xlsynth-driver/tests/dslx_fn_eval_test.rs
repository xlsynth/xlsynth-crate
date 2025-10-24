// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn test_dslx_fn_eval_add() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Create a temporary directory for inputs.
    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_fn_eval_add");
    let dir = tmp.path();

    // Write DSLX file.
    let dslx_path = dir.join("add.x");
    let dslx_src = r#"fn add(a: u32, b: u32) -> u32 { a + b }"#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    // Write .irvals file (each line a tuple of inputs).
    let irvals_path = dir.join("inputs.irvals");
    let irvals = "(bits[32]:0x1, bits[32]:0x2)\n(bits[32]:0x3, bits[32]:0x4)\n";
    std::fs::write(&irvals_path, irvals).expect("write irvals");

    // Invoke the driver.
    let output = Command::new(driver)
        .arg("dslx-fn-eval")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("add")
        .arg("--input_ir_path")
        .arg(irvals_path.to_str().unwrap())
        .output()
        .expect("run driver");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let want = "bits[32]:0x3\nbits[32]:0x7\n";
    assert_eq!(stdout, want);
}

#[test]
fn test_dslx_fn_eval_unary_requires_tuple() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_fn_eval_unary_tuple");
    let dir = tmp.path();

    let dslx_path = dir.join("id.x");
    let dslx_src = r#"fn id(x: u32) -> u32 { x }"#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    // Case 1: bare value should error.
    let irvals_path1 = dir.join("inputs1.irvals");
    std::fs::write(&irvals_path1, "bits[32]:42\n").expect("write irvals1");
    let output1 = Command::new(driver)
        .arg("dslx-fn-eval")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("id")
        .arg("--input_ir_path")
        .arg(irvals_path1.to_str().unwrap())
        .output()
        .expect("run driver");
    assert!(
        !output1.status.success(),
        "expected failure for non-tuple input"
    );

    // Case 2: 1-tuple works.
    let irvals_path2 = dir.join("inputs2.irvals");
    std::fs::write(&irvals_path2, "(bits[32]:42)\n").expect("write irvals2");
    let output2 = Command::new(driver)
        .arg("dslx-fn-eval")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("id")
        .arg("--input_ir_path")
        .arg(irvals_path2.to_str().unwrap())
        .output()
        .expect("run driver");
    assert!(output2.status.success());
    let stdout = String::from_utf8_lossy(&output2.stdout);
    assert_eq!(stdout, "bits[32]:42\n");
}
