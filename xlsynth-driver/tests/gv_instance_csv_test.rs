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
    NAND2 inst1 (.A(a), .B(a), .Y(y));
    NOR2 inst2 (.A(a), .B(y), .Y(y));
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
    // Header + rows: module_name,instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 3, "should be 2 instances found plus header");
    assert_eq!(lines[0], "module_name,instance_name,cell_type");
    assert!(
        lines.iter().any(|&l| l == "test,inst1,NAND2"),
        "missing inst1"
    );
    assert!(
        lines.iter().any(|&l| l == "test,inst2,NOR2"),
        "missing inst2"
    );
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
    NAND2 inst1 (.A(w), .B(a), .Y(y));
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
    // Header + rows: module_name,instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 3, "should be 2 instances found plus header");
    assert_eq!(lines[0], "module_name,instance_name,cell_type");
    assert!(
        lines.iter().any(|&l| l == "top,u_feed,feedthrough"),
        "missing u_feed"
    );
    assert!(
        lines.iter().any(|&l| l == "top,inst1,NAND2"),
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
    NAND2 inst1 (.A(w[0]), .B(a[1]), .Y(y[0]));
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
    // Header + rows: module_name,instance_name,cell_type\n
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 3, "should be 2 instances found plus header");
    assert_eq!(lines[0], "module_name,instance_name,cell_type");
    assert!(
        lines.iter().any(|&l| l == "top_bus,u_feed,feedthrough_bus"),
        "missing u_feed"
    );
    assert!(
        lines.iter().any(|&l| l == "top_bus,inst1,NAND2"),
        "missing inst1"
    );
}
