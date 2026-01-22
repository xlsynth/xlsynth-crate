// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;

#[test]
fn gv2aig_emits_parseable_aiger() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Minimal Liberty textproto with INV and AND2.
    let liberty_text = r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "(!A)" }
  area: 1.0
}
cells: {
  name: "AND2"
  pins: { name: "A" direction: INPUT }
  pins: { name: "B" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "(A & B)" }
  area: 1.0
}
"#;

    // Netlist: y = !(a & b)
    let netlist_text = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n1;
  AND2 u1 (.A(a), .B(b), .Y(n1));
  INV u2 (.A(n1), .Y(y));
endmodule
"#;

    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let liberty_path = temp_dir.path().join("lib.textproto");
    let netlist_path = temp_dir.path().join("netlist.v");
    let out_path = temp_dir.path().join("out.aag");

    std::fs::write(&liberty_path, liberty_text).expect("write liberty");
    std::fs::write(&netlist_path, netlist_text).expect("write netlist");

    let output = Command::new(driver)
        .arg("gv2aig")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--aiger-out")
        .arg(out_path.as_os_str())
        .output()
        .expect("gv2aig invocation should run");

    assert!(
        output.status.success(),
        "gv2aig failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let loaded =
        load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt()).expect("load aiger");
    assert!(
        !loaded.gate_fn.gates.is_empty(),
        "expected non-empty GateFn"
    );
}
