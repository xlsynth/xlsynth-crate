// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn test_dslx_fn_eval_float32_add2() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_fn_eval_f32_add2");
    let dir = tmp.path();

    // DSLX module using float32 struct from stdlib.
    let dslx_path = dir.join("f32_add2.x");
    let dslx_src = r#"
        import float32;
        fn add2(f: float32::F32) -> float32::F32 { float32::add(f, f) }
    "#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    // 1.0f tuple: (sign=0, exp=127, frac=0). Expect 2.0f: (0,128,0).
    let irvals_path = dir.join("inputs.irvals");
    let irvals = "((bits[1]:0, bits[8]:127, bits[23]:0))\n";
    std::fs::write(&irvals_path, irvals).expect("write irvals");

    let output = Command::new(driver)
        .arg("dslx-fn-eval")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("add2")
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
    let want = "(bits[1]:0, bits[8]:128, bits[23]:0)\n";
    assert_eq!(stdout, want);
}

#[test]
fn test_dslx_fn_eval_float32_muladd() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_fn_eval_f32_muladd");
    let dir = tmp.path();

    // DSLX module using float32 muladd.
    let dslx_path = dir.join("f32_muladd.x");
    let dslx_src = r#"
        import float32;
        fn muladd(a: float32::F32, b: float32::F32, c: float32::F32) -> float32::F32 {
          float32::add(float32::mul(a, b), c)
        }
    "#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    // 1.0f * 2.0f + 0.0f = 2.0f
    let one = "(bits[1]:0, bits[8]:127, bits[23]:0)";
    let two = "(bits[1]:0, bits[8]:128, bits[23]:0)";
    let zero = "(bits[1]:0, bits[8]:0, bits[23]:0)";
    let irvals_path = dir.join("inputs.irvals");
    let line = format!("({}, {}, {})\n", one, two, zero);
    std::fs::write(&irvals_path, line).expect("write irvals");

    let output = Command::new(driver)
        .arg("dslx-fn-eval")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("muladd")
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
    let want = "(bits[1]:0, bits[8]:128, bits[23]:0)\n";
    assert_eq!(stdout, want);
}
