// SPDX-License-Identifier: Apache-2.0

//! Tests that invoke the `g8r` binary.

use std::io::Write;
use std::process::Command;
use xlsynth_test_helpers::compare_golden_text;

#[test]
fn test_ir2gates_invoke_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir = r#"package prio_pkg
file_number 0 "foo.x"
file_number 1 "bar.x"

top fn main(sel: bits[1] id=1, a: bits[1] id=2, b: bits[1] id=3) -> bits[1] {
  p: bits[1] = priority_sel(sel, cases=[a], default=b, id=4, pos=[(0,1,0), (1,2,0)])
  ret result: bits[1] = identity(p, id=5)
}
"#;

    let mut temp_file = tempfile::Builder::new().suffix(".ir").tempfile().unwrap();
    write!(temp_file, "{}", ir).unwrap();
    let temp_path = temp_file.into_temp_path();

    let output = Command::new(env!("CARGO_BIN_EXE_g8r"))
        .arg("--check-equivalence=false")
        .arg(temp_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_text(&stdout, "tests/goldens/g8r_invoke_ir2gates.golden.txt");
}
