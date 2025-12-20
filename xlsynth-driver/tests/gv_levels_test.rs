// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn gv_levels_basic_csv_output() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Minimal Liberty textproto with an inverter and a DFF cell.
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
cells: {
  name: "DFFX1"
  pins: {
    name: "D"
    direction: INPUT
  }
  pins: {
    name: "Q"
    direction: OUTPUT
    function: "IQ"
  }
  area: 1.0
}
"#;

    // Netlist:
    // a -> INVX1 u1 -> n1 -> DFFX1 udff0 -> q0 -> INVX1 u2 -> n2 -> INVX1 u3 -> n3
    // -> DFFX1 udff1 -> q1 -> INVX1 u4 -> y
    let netlist_text = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n1;
  wire q0;
  wire n2;
  wire n3;
  wire q1;
  INVX1 u1 (.A(a), .Y(n1));
  DFFX1 udff0 (.D(n1), .Q(q0));
  INVX1 u2 (.A(q0), .Y(n2));
  INVX1 u3 (.A(n2), .Y(n3));
  DFFX1 udff1 (.D(n3), .Q(q1));
  INVX1 u4 (.A(q1), .Y(y));
endmodule
"#;

    let mut liberty_file = tempfile::NamedTempFile::new().expect("create liberty temp file");
    std::io::Write::write_all(&mut liberty_file, liberty_text.as_bytes())
        .expect("write liberty text");

    let mut netlist_file = tempfile::NamedTempFile::new().expect("create netlist temp file");
    std::io::Write::write_all(&mut netlist_file, netlist_text.as_bytes())
        .expect("write netlist text");

    let output = Command::new(driver)
        .arg("gv-levels")
        .arg(netlist_file.path().as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_file.path().as_os_str())
        .arg("--dff_cells")
        .arg("DFFX1")
        .arg("--format")
        .arg("csv")
        .output()
        .expect("gv-levels invocation should run");

    assert!(
        output.status.success(),
        "gv-levels failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let got = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    let want = "\
category,depth,count,example_path
input-to-reg,1,1,u1(INVX1)
reg-to-reg,2,1,u2(INVX1) -> u3(INVX1)
reg-to-output,1,1,u4(INVX1)
";
    assert_eq!(got, want);
}
