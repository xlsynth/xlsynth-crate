// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_pir::ir::PackageMember;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn block2fn_emits_top_and_its_transitive_callees() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let block_ir = r#"package test_pkg

fn inner(x: bits[8] id=3) -> bits[8] {
  ret n: bits[8] = not(x, id=4)
}
fn outer(x: bits[8] id=5) -> bits[8] {
  ret n: bits[8] = invoke(x, to_apply=inner, id=6)
}
fn unrelated(x: bits[8] id=7) -> bits[8] {
  ret x: bits[8] = param(name=x, id=7)
}
top block top(a: bits[8], out: bits[8]) {
  a: bits[8] = input_port(name=a, id=1)
  result: bits[8] = invoke(a, to_apply=outer, id=8)
  out: () = output_port(result, name=out, id=2)
}
"#;

    let temp_dir = tempfile::tempdir().expect("create tempdir");
    let block_ir_path = temp_dir.path().join("in.block.ir");
    std::fs::write(&block_ir_path, block_ir).expect("write block ir");

    let output = Command::new(driver)
        .arg("block2fn")
        .arg("--block_ir")
        .arg(block_ir_path.as_os_str())
        .output()
        .expect("block2fn invocation should run");

    assert!(
        output.status.success(),
        "block2fn failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.starts_with("package "),
        "block2fn stdout should be package-form IR; got:\n{}",
        stdout
    );

    let mut parser = Parser::new(&stdout);
    let package = parser
        .parse_and_verify_package()
        .expect("valid standalone package");
    let top_fn = package.get_top_fn().expect("top function should exist");
    assert_eq!(top_fn.name, "top");
    assert!(package.get_fn("unrelated").is_none());
    assert_eq!(
        package
            .members
            .iter()
            .filter(|m| matches!(m, PackageMember::Function(_)))
            .count(),
        3
    );
}
