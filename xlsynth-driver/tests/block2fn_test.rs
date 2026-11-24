// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_pir::ir::PackageMember;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn block2fn_emits_ir_package_with_top_fn() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let block_ir = r#"package test_pkg

top block top(a: bits[8], out: bits[8]) {
  a: bits[8] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
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
    let package = parser.parse_package().expect("parse package output");
    let top_fn = package.get_top_fn().expect("top function should exist");
    assert_eq!(top_fn.name, "top");
    assert_eq!(
        package
            .members
            .iter()
            .filter(|m| matches!(m, PackageMember::Function(_)))
            .count(),
        1
    );
}
