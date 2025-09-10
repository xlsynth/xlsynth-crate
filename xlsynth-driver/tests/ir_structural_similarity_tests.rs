// SPDX-License-Identifier: Apache-2.0

#[test]
fn test_ir_structural_similarity_equiv_simple() {
    // Verbatim from ~/diff-test/lhs.ir and ~/diff-test/rhs.ir
    let lhs_ir = r#"package test

fn f(a: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  add.3: bits[8] = add(a, one, id=3)
  not.4: bits[8] = not(add.3, id=4)
  two: bits[8] = literal(value=2, id=5)
  not.9: bits[8] = not(two, id=9)
  ret add.6: bits[8] = add(not.4, not.9, id=6)
}
"#;
    let rhs_ir = r#"package test

fn f(a: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  add.3: bits[8] = add(a, one, id=3)
  neg.4: bits[8] = neg(add.3, id=4)
  two: bits[8] = literal(value=2, id=5)
  ret add.6: bits[8] = add(neg.4, two, id=6)
}
"#;

    let tmp = tempfile::tempdir().unwrap();
    let lhs_path = tmp.path().join("lhs.ir");
    let rhs_path = tmp.path().join("rhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("ir-structural-similarity")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--output-dir")
        .arg(tmp.path().to_str().unwrap())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Equiv (lhs_diff ≡ lhs_orig): OK"),
        "stdout=\n{}",
        stdout
    );
    assert!(
        stdout.contains("Equiv (rhs_diff ≡ rhs_orig): OK"),
        "stdout=\n{}",
        stdout
    );
}

#[test]
fn test_ir_structural_similarity_equiv_negation_variant() {
    // Verbatim from ~/diff-test-2/lhs.ir and ~/diff-test-2/rhs.ir
    let lhs_ir = r#"package test

fn f(a: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  add.3: bits[8] = add(a, one, id=3)
  not.4: bits[8] = not(add.3, id=4)
  three: bits[8] = literal(value=3, id=5)
  xor.6: bits[8] = xor(a, three, id=6)
  add.7: bits[8] = add(not.4, xor.6, id=7)
  four: bits[8] = literal(value=4, id=8)
  ret post: bits[8] = add(add.7, four, id=9)
}
"#;
    let rhs_ir = r#"package test

fn f(a: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  add.3: bits[8] = add(a, one, id=3)
  four: bits[8] = literal(value=4, id=4)
  ret post: bits[8] = add(add.3, four, id=5)
}
"#;

    let tmp = tempfile::tempdir().unwrap();
    let lhs_path = tmp.path().join("lhs2.ir");
    let rhs_path = tmp.path().join("rhs2.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("ir-structural-similarity")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--output-dir")
        .arg(tmp.path().to_str().unwrap())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Equiv (lhs_diff ≡ lhs_orig): OK"),
        "stdout=\n{}",
        stdout
    );
    assert!(
        stdout.contains("Equiv (rhs_diff ≡ rhs_orig): OK"),
        "stdout=\n{}",
        stdout
    );
}
