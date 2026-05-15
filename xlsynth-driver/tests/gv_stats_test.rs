// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_test_helpers::compare_golden_text;

fn make_timing_enabled_inv_nand2_liberty_textproto() -> &'static str {
    r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT capacitance: 0.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!A"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 1.0 }
      tables: { kind: "cell_fall" values: 1.0 }
      tables: { kind: "rise_transition" values: 0.1 }
      tables: { kind: "fall_transition" values: 0.1 }
    }
  }
  area: 1.0
}
cells: {
  name: "NAND2"
  pins: { name: "A" direction: INPUT capacitance: 0.0 }
  pins: { name: "B" direction: INPUT capacitance: 0.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!(A*B)"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 2.0 }
      tables: { kind: "cell_fall" values: 2.0 }
      tables: { kind: "rise_transition" values: 0.1 }
      tables: { kind: "fall_transition" values: 0.1 }
    }
    timing_arcs: {
      related_pin: "B"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 2.0 }
      tables: { kind: "cell_fall" values: 2.0 }
      tables: { kind: "rise_transition" values: 0.1 }
      tables: { kind: "fall_transition" values: 0.1 }
    }
  }
  area: 2.0
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
"#
}

fn write_fixture() -> tempfile::TempDir {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    std::fs::write(
        temp_dir.path().join("mapped.gv"),
        r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  NAND2 u0 (.A(a), .B(b), .Y(n0));
  INV u1 (.A(n0), .Y(y));
endmodule
"#,
    )
    .expect("write netlist");
    std::fs::write(
        temp_dir.path().join("lib.textproto"),
        make_timing_enabled_inv_nand2_liberty_textproto(),
    )
    .expect("write liberty");
    temp_dir
}

#[test]
fn gv_area_and_gv_stats_emit_expected_text_and_json() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = write_fixture();
    let netlist_path = temp_dir.path().join("mapped.gv");
    let liberty_path = temp_dir.path().join("lib.textproto");
    let area_json_path = temp_dir.path().join("area.json");
    let stats_json_path = temp_dir.path().join("stats.json");

    let area_output = Command::new(driver)
        .arg("gv-area")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--json_out")
        .arg(area_json_path.as_os_str())
        .output()
        .expect("gv-area invocation should run");
    assert!(
        area_output.status.success(),
        "gv-area failed: status={:?}\nstdout={}\nstderr={}",
        area_output.status,
        String::from_utf8_lossy(&area_output.stdout),
        String::from_utf8_lossy(&area_output.stderr)
    );
    compare_golden_text(
        String::from_utf8_lossy(&area_output.stdout).as_ref(),
        "tests/test_gv_area.golden.txt",
    );
    let area_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&area_json_path).expect("read gv-area json"))
            .expect("parse gv-area json");
    assert_eq!(area_json["area"], 3.0);
    assert_eq!(area_json["cell_count"], 2);

    let stats_output = Command::new(driver)
        .arg("gv-stats")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--json_out")
        .arg(stats_json_path.as_os_str())
        .output()
        .expect("gv-stats invocation should run");
    assert!(
        stats_output.status.success(),
        "gv-stats failed: status={:?}\nstdout={}\nstderr={}",
        stats_output.status,
        String::from_utf8_lossy(&stats_output.stdout),
        String::from_utf8_lossy(&stats_output.stderr)
    );
    compare_golden_text(
        String::from_utf8_lossy(&stats_output.stdout).as_ref(),
        "tests/test_gv_stats.golden.txt",
    );
    let stats_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stats_json_path).expect("read gv-stats json"))
            .expect("parse gv-stats json");
    assert_eq!(stats_json["area"], 3.0);
    assert_eq!(stats_json["delay"], 3.0);
    assert_eq!(stats_json["cell_count"], 2);
    assert_eq!(stats_json["cell_levels"], 2);
}
