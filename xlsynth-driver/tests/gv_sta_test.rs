// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn make_timing_enabled_inv_liberty_textproto() -> &'static str {
    r#"
format_magic: 5496997758177923663
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1.0 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 3
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 3.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.2 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.3 }
    }
  }
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
interned_strings: ["A", "Y", "!A"]
lut_shapes: {}
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
