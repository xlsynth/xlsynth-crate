// SPDX-License-Identifier: Apache-2.0

//! The block2fn command emits a complete package for retained native calls.

use std::process::Command;

use xlsynth_pir::ir_parser::Parser;

#[test]
fn emitted_function_package_includes_its_native_callee() {
    let source = r#"package calls
fn invert(x: bits[4] id=1) -> bits[4] {
  ret result: bits[4] = not(x, id=2)
}
top block top(value: bits[4], result: bits[4]) {
  value: bits[4] = input_port(name=value, id=3)
  called: bits[4] = invoke(value, to_apply=invert, id=4)
  result: () = output_port(called, name=result, id=5)
}
"#;
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.ir");
    std::fs::write(&input, source).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .args(["block2fn", "--block_ir"])
        .arg(input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    let package = Parser::new(&output).parse_and_verify_package().unwrap();
    assert!(package.get_fn("invert").is_some());
    assert_eq!(package.get_top_fn().unwrap().name, "top");
}
