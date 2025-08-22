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
