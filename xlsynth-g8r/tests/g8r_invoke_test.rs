// SPDX-License-Identifier: Apache-2.0

//! Tests that invoke the `g8r` binary.

use std::io::Write;
use std::process::Command;

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
    let golden_path = std::path::Path::new("tests/goldens/g8r_invoke_ir2gates.golden.txt");
    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() {
        println!("INFO: Updating golden file: {}", golden_path.display());
        std::fs::write(golden_path, &stdout).expect("Failed to write golden file");
    } else {
        let golden = std::fs::read_to_string(golden_path).expect("Failed to read golden file");
        assert_eq!(
            stdout, golden,
            "Golden file mismatch. Run with XLSYNTH_UPDATE_GOLDEN=1 to update."
        );
    }
}
