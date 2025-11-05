// SPDX-License-Identifier: Apache-2.0

//! Tests for the gv-instance-csv driver subcommand.

use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::process::Command;

/// Returns decompressed file as string.
fn read_gzipped_csv(path: &std::path::Path) -> String {
    let f = File::open(path).unwrap();
    let mut gz = GzDecoder::new(f);
    let mut s = String::new();
    gz.read_to_string(&mut s).expect("decompress csv.gz");
    s
}

/// Writes the provided content into a fresh temp file and returns its path.
fn write_tempfile(contents: &str) -> std::path::PathBuf {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), contents).unwrap();
    tmp.into_temp_path().to_path_buf()
}

#[test]
fn test_gv_instance_csv_minimal_netlist() {
    let netlist = r#"
module test (input a, output y);
  NAND2X1 inst1 (.A(a), .B(a), .Y(y));
  NOR2X1 inst2 (.A(a), .B(y), .Y(y));
endmodule
"#;
    let in_path = write_tempfile(netlist);
    let out_path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    let exe = env!("CARGO_BIN_EXE_xlsynth-driver");
    let status = Command::new(exe)
        .arg("gv-instance-csv")
        .arg("--input")
        .arg(in_path.as_os_str())
        .arg("--output")
        .arg(out_path.as_os_str())
        .status()
        .expect("run driver");
    assert!(status.success(), "process failed: {status}");
    let s = read_gzipped_csv(&out_path);
    // No header, just rows: instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 2, "should be 2 instances found");
    assert!(lines.iter().any(|&l| l == "inst1,NAND2X1"), "missing inst1");
    assert!(lines.iter().any(|&l| l == "inst2,NOR2X1"), "missing inst2");
}
