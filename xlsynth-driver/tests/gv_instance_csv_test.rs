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

#[test]
fn test_gv_instance_csv_minimal_netlist() {
    let netlist = r#"
module test (a, y);
    input a;
    output y;
    NAND2X1 inst1 (.A(a), .B(a), .Y(y));
    NOR2X1 inst2 (.A(a), .B(y), .Y(y));
endmodule
"#;
    let in_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(in_file.path(), netlist).unwrap();
    let out_file = tempfile::NamedTempFile::new().unwrap();
    let exe = env!("CARGO_BIN_EXE_xlsynth-driver");
    let status = Command::new(exe)
        .arg("gv-instance-csv")
        .arg("--input")
        .arg(in_file.path())
        .arg("--output")
        .arg(out_file.path())
        .status()
        .expect("run driver");
    assert!(status.success(), "process failed: {status}");
    let s = read_gzipped_csv(out_file.path());
    // No header, just rows: instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 2, "should be 2 instances found");
    assert!(lines.iter().any(|&l| l == "inst1,NAND2X1"), "missing inst1");
    assert!(lines.iter().any(|&l| l == "inst2,NOR2X1"), "missing inst2");
}

#[test]
fn test_gv_instance_csv_feedthrough_module() {
    let netlist = r#"
module feedthrough (in, out);
    input in;
    output out;
    assign out = in;
endmodule

module top (a, y);
    input a;
    output y;
    wire w;
    feedthrough u_feed (.in(a), .out(w));
    NAND2X1 inst1 (.A(w), .B(a), .Y(y));
endmodule
"#;
    let in_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(in_file.path(), netlist).unwrap();
    let out_file = tempfile::NamedTempFile::new().unwrap();
    let exe = env!("CARGO_BIN_EXE_xlsynth-driver");
    let status = Command::new(exe)
        .arg("gv-instance-csv")
        .arg("--input")
        .arg(in_file.path())
        .arg("--output")
        .arg(out_file.path())
        .status()
        .expect("run driver");
    assert!(status.success(), "process failed: {status}");
    let s = read_gzipped_csv(out_file.path());
    // No header, just rows: instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 2, "should be 2 instances found");
    assert!(
        lines.iter().any(|&l| l == "u_feed,feedthrough"),
        "missing u_feed"
    );
    assert!(
        lines.iter().any(|&l| l == "inst1,NAND2X1"),
        "missing inst1"
    );
}

#[test]
fn test_gv_instance_csv_feedthrough_module_bus() {
    let netlist = r#"
module feedthrough_bus (in, out);
    input [1:0] in;
    output [1:0] out;
    assign out[0] = in[0];
    assign out[1] = in[1];
endmodule

module top_bus (a, y);
    input [1:0] a;
    output [1:0] y;
    wire [1:0] w;
    feedthrough_bus u_feed (.in(a), .out(w));
    NAND2X1 inst1 (.A(w[0]), .B(a[1]), .Y(y[0]));
endmodule
"#;
    let in_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(in_file.path(), netlist).unwrap();
    let out_file = tempfile::NamedTempFile::new().unwrap();
    let exe = env!("CARGO_BIN_EXE_xlsynth-driver");
    let status = Command::new(exe)
        .arg("gv-instance-csv")
        .arg("--input")
        .arg(in_file.path())
        .arg("--output")
        .arg(out_file.path())
        .status()
        .expect("run driver");
    assert!(status.success(), "process failed: {status}");
    let s = read_gzipped_csv(out_file.path());
    // No header, just rows: instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 2, "should be 2 instances found");
    assert!(
        lines.iter().any(|&l| l == "u_feed,feedthrough_bus"),
        "missing u_feed"
    );
    assert!(
        lines.iter().any(|&l| l == "inst1,NAND2X1"),
        "missing inst1"
    );
}
