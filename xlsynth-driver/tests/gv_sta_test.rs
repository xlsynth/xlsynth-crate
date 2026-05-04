// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn make_timing_enabled_inv_liberty_textproto() -> &'static str {
    r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT capacitance: 1.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!A"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 2.0 }
      tables: { kind: "cell_fall" values: 3.0 }
      tables: { kind: "rise_transition" values: 0.2 }
      tables: { kind: "fall_transition" values: 0.3 }
    }
  }
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
"#
}

#[test]
fn gv_sta_uses_selected_module_net_scope_when_names_repeat() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("multi.gv");
    let liberty_path = temp_dir.path().join("lib.textproto");

    std::fs::write(
        &netlist_path,
        r#"
module fast (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 ( .A(a), .Y(y) );
endmodule

module slow (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  INV u0 ( .A(a), .Y(n0) );
  INV u1 ( .A(n0), .Y(y) );
endmodule
"#,
    )
    .expect("write netlist");
    std::fs::write(&liberty_path, make_timing_enabled_inv_liberty_textproto())
        .expect("write liberty");

    let output = Command::new(driver)
        .arg("gv-sta")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--module_name")
        .arg("fast")
        .output()
        .expect("gv-sta invocation should run");

    assert!(
        output.status.success(),
        "gv-sta failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("worst_output_arrival: 3.000000"),
        "unexpected stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("output y rise_arrival=2.000000 fall_arrival=3.000000"),
        "unexpected stdout: {}",
        stdout
    );
}
