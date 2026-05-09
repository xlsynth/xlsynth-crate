// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_g8r::aig::AigBitVector;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

fn make_two_input_and_aig() -> String {
    let mut gb = GateBuilder::new("and2_demo".to_string(), GateBuilderOptions::no_opt());
    let a = gb.add_input("a".to_string(), 1);
    let b = gb.add_input("b".to_string(), 1);
    let y = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
    gb.add_output("y".to_string(), AigBitVector::from_bit(y));
    emit_aiger(&gb.build(), true).expect("emit_aiger should succeed")
}

fn make_timing_enabled_inv_nand2_liberty_textproto() -> &'static str {
    r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT capacitance: 0.01 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!A"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 1.0 }
      tables: { kind: "cell_fall" values: 1.1 }
      tables: { kind: "rise_transition" values: 0.3 }
      tables: { kind: "fall_transition" values: 0.4 }
    }
  }
  area: 1.0
}
cells: {
  name: "NAND2"
  pins: { name: "A" direction: INPUT capacitance: 0.02 }
  pins: { name: "B" direction: INPUT capacitance: 0.02 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!(A*B)"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 2.0 }
      tables: { kind: "cell_fall" values: 2.1 }
      tables: { kind: "rise_transition" values: 0.5 }
      tables: { kind: "fall_transition" values: 0.6 }
    }
    timing_arcs: {
      related_pin: "B"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 2.0 }
      tables: { kind: "cell_fall" values: 2.1 }
      tables: { kind: "rise_transition" values: 0.5 }
      tables: { kind: "fall_transition" values: 0.6 }
    }
  }
  area: 2.0
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
"#
}

fn make_no_timing_liberty_textproto() -> &'static str {
    r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "!A" }
}
"#
}

#[test]
fn aig_tech_map_then_gv_sta_smoke() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");

    let aig_path = temp_dir.path().join("in.aag");
    let liberty_path = temp_dir.path().join("lib.textproto");
    let netlist_path = temp_dir.path().join("mapped.gv");

    std::fs::write(&aig_path, make_two_input_and_aig()).expect("write aiger");
    std::fs::write(
        &liberty_path,
        make_timing_enabled_inv_nand2_liberty_textproto(),
    )
    .expect("write liberty");

    let map_output = Command::new(driver)
        .arg("aig-tech-map")
        .arg(aig_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--netlist_out")
        .arg(netlist_path.as_os_str())
        .output()
        .expect("aig-tech-map invocation should run");

    assert!(
        map_output.status.success(),
        "aig-tech-map failed: status={:?}\nstdout={}\nstderr={}",
        map_output.status,
        String::from_utf8_lossy(&map_output.stdout),
        String::from_utf8_lossy(&map_output.stderr)
    );

    let mapped_text = std::fs::read_to_string(&netlist_path).expect("read mapped netlist");
    assert!(mapped_text.contains("NAND2"));
    assert!(mapped_text.contains("INV"));

    let sta_output = Command::new(driver)
        .arg("gv-sta")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .output()
        .expect("gv-sta invocation should run");

    assert!(
        sta_output.status.success(),
        "gv-sta failed: status={:?}\nstdout={}\nstderr={}",
        sta_output.status,
        String::from_utf8_lossy(&sta_output.stdout),
        String::from_utf8_lossy(&sta_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&sta_output.stdout);
    assert!(stdout.contains("worst_output_arrival:"));

    let worst_line = stdout
        .lines()
        .find(|line| line.starts_with("worst_output_arrival:"))
        .expect("expected worst_output_arrival line");
    let value: f64 = worst_line
        .split(':')
        .nth(1)
        .expect("line should have ':'")
        .trim()
        .parse()
        .expect("worst_output_arrival should parse as f64");
    assert!(value > 0.0, "expected positive arrival, got {}", value);
}

#[test]
fn commands_reject_non_timing_liberty_proto() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");

    let aig_path = temp_dir.path().join("in.aag");
    let liberty_path = temp_dir.path().join("lib_no_timing.textproto");
    let netlist_path = temp_dir.path().join("mapped.gv");

    std::fs::write(&aig_path, make_two_input_and_aig()).expect("write aiger");
    std::fs::write(&liberty_path, make_no_timing_liberty_textproto()).expect("write liberty");

    let map_output = Command::new(driver)
        .arg("aig-tech-map")
        .arg(aig_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--netlist_out")
        .arg(netlist_path.as_os_str())
        .output()
        .expect("aig-tech-map invocation should run");

    assert!(
        !map_output.status.success(),
        "aig-tech-map should fail with non-timing liberty"
    );

    let sta_output = Command::new(driver)
        .arg("gv-sta")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .output()
        .expect("gv-sta invocation should run");

    assert!(
        !sta_output.status.success(),
        "gv-sta should fail with non-timing liberty"
    );
}
