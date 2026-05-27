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
cells: {
  name: "DFF"
  pins: {
    name: "D"
    direction: INPUT
    capacitance: 0.0
    timing_arcs: {
      related_pin: "CLK"
      timing_type: "setup_rising"
      tables: { kind: "rise_constraint" values: 0.25 }
      tables: { kind: "fall_constraint" values: 0.25 }
    }
  }
  pins: { name: "CLK" direction: INPUT is_clocking_pin: true capacitance: 0.0 }
  pins: {
    name: "Q"
    direction: OUTPUT
    function: "Q"
    timing_arcs: {
      related_pin: "CLK"
      timing_sense: "non_unate"
      timing_type: "rising_edge"
      tables: { kind: "cell_rise" values: 0.5 }
      tables: { kind: "cell_fall" values: 0.5 }
      tables: { kind: "rise_transition" values: 0.1 }
      tables: { kind: "fall_transition" values: 0.1 }
    }
  }
  area: 4.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
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

fn write_pipeline_fixture() -> tempfile::TempDir {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    std::fs::write(
        temp_dir.path().join("mapped.gv"),
        r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  wire d0;
  wire q0;
  wire d1;
  wire q1;
  INV in_logic (.A(a), .Y(d0));
  DFF r0 (.D(d0), .CLK(clk), .Q(q0));
  INV stage_logic (.A(q0), .Y(d1));
  DFF r1 (.D(d1), .CLK(clk), .Q(q1));
  INV out_logic (.A(q1), .Y(y));
endmodule
"#,
    )
    .expect("write registered netlist");
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
    assert_eq!(stats_json["cell_area"], 3.0);
    assert_eq!(stats_json["max_delay"], 3.0);
    assert_eq!(stats_json["cell_count"], 2);
    assert_eq!(stats_json["cell_levels"], 2);
}

#[test]
fn gv_stats_reports_registered_pipeline_stages() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = write_pipeline_fixture();
    let stats_json_path = temp_dir.path().join("stats.json");
    let output = Command::new(driver)
        .arg("gv-stats")
        .arg("--netlist")
        .arg(temp_dir.path().join("mapped.gv"))
        .arg("--liberty_proto")
        .arg(temp_dir.path().join("lib.textproto"))
        .arg("--json_out")
        .arg(stats_json_path.as_os_str())
        .output()
        .expect("registered gv-stats invocation should run");
    assert!(
        output.status.success(),
        "gv-stats failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    compare_golden_text(
        String::from_utf8_lossy(&output.stdout).as_ref(),
        "tests/test_gv_stats_pipeline.golden.txt",
    );
    let stats_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stats_json_path).expect("read gv-stats json"))
            .expect("parse gv-stats json");
    assert_eq!(stats_json["stage_partition_status"], "partitioned");
    assert_eq!(
        stats_json["max_input_to_register_delay_breakdown"]["combinational_delay"],
        1.0
    );
    assert_eq!(
        stats_json["max_input_to_register_delay_breakdown"]["setup_delay"],
        0.25
    );
    assert_eq!(stats_json["max_register_to_register_delay"], 1.75);
    assert_eq!(
        stats_json["max_register_to_register_delay_breakdown"]["clock_to_output_delay"],
        0.5
    );
    assert_eq!(
        stats_json["max_register_to_register_delay_breakdown"]["combinational_delay"],
        1.0
    );
    assert_eq!(
        stats_json["max_register_to_register_delay_breakdown"]["setup_delay"],
        0.25
    );
    assert_eq!(
        stats_json["max_register_to_output_delay_breakdown"]["clock_to_output_delay"],
        0.5
    );
    assert_eq!(
        stats_json["max_register_to_output_delay_breakdown"]["combinational_delay"],
        1.0
    );
    assert_eq!(stats_json["sequential_cell_area"], 8.0);
    assert_eq!(stats_json["non_stage_combinational_cell_area"], 2.0);
    assert_eq!(stats_json["stages"][0]["combinational_cell_area"], 1.0);
    assert_eq!(
        stats_json["stages"][0]["max_delay_breakdown"]["clock_to_output_delay"],
        0.5
    );
    assert_eq!(
        stats_json["stages"][0]["max_delay_breakdown"]["combinational_delay"],
        1.0
    );
    assert_eq!(
        stats_json["stages"][0]["max_delay_breakdown"]["setup_delay"],
        0.25
    );
}
