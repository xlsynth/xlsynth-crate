// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn gv_dump_cone_basic_csv_output() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Minimal Liberty textproto with a single inverter cell.
    let liberty_text = r#"
cells: {
  name: "INVX1"
  pins: {
    name: "A"
    direction: INPUT
  }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "(!A)"
  }
  area: 1.0
}
"#;

    // Minimal netlist: a -> INVX1 u1 -> n1 -> INVX1 u2 -> y
    let netlist_text = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n1;
  INVX1 u1 (.A(a), .Y(n1));
  INVX1 u2 (.A(n1), .Y(y));
endmodule
"#;

    let mut liberty_file = tempfile::NamedTempFile::new().expect("create liberty temp file");
    std::io::Write::write_all(&mut liberty_file, liberty_text.as_bytes())
        .expect("write liberty text");

    let mut netlist_file = tempfile::NamedTempFile::new().expect("create netlist temp file");
    std::io::Write::write_all(&mut netlist_file, netlist_text.as_bytes())
        .expect("write netlist text");

    let output = Command::new(driver)
        .arg("gv-dump-cone")
        .arg(netlist_file.path().as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_file.path().as_os_str())
        .arg("--instance")
        .arg("u1")
        .arg("--traverse")
        .arg("fanout")
        .arg("--stop-at-levels")
        .arg("1")
        .output()
        .expect("gv-dump-cone invocation should run");

    assert!(
        output.status.success(),
        "gv-dump-cone failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let got = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");

    let want = "\
instance_type,instance_name,traversal_pin
INVX1,u1,Y
INVX1,u2,A
";

    assert_eq!(got, want);
}
