// SPDX-License-Identifier: Apache-2.0

#[test]
fn test_ir_round_trip_standalone_block() {
    let block_ir = r#"block myb(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}"#;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("in.block.ir");
    std::fs::write(&path, block_ir).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout, block_ir);
}

#[test]
fn test_ir_round_trip_package_with_top_block() {
    let pkg_ir = r#"package p

block myb(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}
"#;
    // Input package that includes a top block equivalent to the expected emission.
    let input_ir = r#"package p

top block myb(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}
"#;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("in.ir");
    std::fs::write(&path, input_ir).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout, pkg_ir);
}

#[test]
fn test_ir_round_trip_package_invalid_fn_but_valid_block_recovers_block() {
    // Package starts with an invalid top fn (missing -> ...) but contains a valid
    // block.
    let input_ir = r#"package p

top fn broken() {
}

block myb(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}
"#;
    let expected_block = r#"block myb(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}"#;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("in.ir");
    std::fs::write(&path, input_ir).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout, expected_block);
}

#[test]
fn test_ir_round_trip_multi_output_block_preserves_order_and_ids() {
    let input = r#"block b(a: bits[1], out0: bits[1], out1: bits[1]) {
  a: bits[1] = input_port(name=a, id=10)
  out0: () = output_port(a, name=out0, id=20)
  out1: () = output_port(a, name=out1, id=21)
}"#;
    let want = input; // Canonical emission should match exactly.

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("multi.block.ir");
    std::fs::write(&path, input).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout, want);
}

#[test]
fn test_ir_round_trip_block_with_attributes_and_pos() {
    // Outer attribute and inner attribute should be ignored; pos attributes should
    // be parsed and dropped.
    let input = r#"#[signature("""")]
block attr_test(a: bits[1], out: bits[1]) {
  #![provenance(name="attr_test", kind="function")]
  a: bits[1] = input_port(name=a, id=1, pos=[(0,0,0)])
  out: () = output_port(a, name=out, id=2, pos=[(0,1,0)])
}"#;
    let want = r#"block attr_test(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}"#;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("attrs.block.ir");
    std::fs::write(&path, input).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    // Without strip flag
    let out1 = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(out1.status.success());
    let s1 = String::from_utf8_lossy(&out1.stdout);
    assert_eq!(s1, want);

    // With strip-pos-attrs flag (should be the same for blocks)
    let out2 = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(path.to_str().unwrap())
        .arg("--strip-pos-attrs")
        .arg("true")
        .output()
        .unwrap();
    assert!(out2.status.success());
    let s2 = String::from_utf8_lossy(&out2.stdout);
    assert_eq!(s2, want);
}
