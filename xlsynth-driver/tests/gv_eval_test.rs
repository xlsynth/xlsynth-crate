// SPDX-License-Identifier: Apache-2.0

use std::process::{Command, Output};

use xlsynth_test_helpers::compare_golden_text;

const COMBINATIONAL_LIBERTY: &str = r#"
format_magic: 5496997758177923663
units: { time_unit: "ns" capacitance_unit: "pf" voltage_unit: "V" }
nominal_voltage: 1.0
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1.0 }
  pins: { name_string_id: 2 direction: INPUT capacitance: 1.0 }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
}
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1.0 }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 5 }
}
interned_strings: ["A", "B", "Y", "A & B", "!A"]
"#;

const SEQUENTIAL_LIBERTY: &str = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["D", "CLK", "Q", "IQ"]
"#;

const SEQUENTIAL_POWER_LIBERTY: &str = r#"
format_magic: 5496997758177923663
units: { time_unit: "ns" capacitance_unit: "pf" voltage_unit: "V" }
nominal_voltage: 1.0
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1.0 }
  pins: {
    name_string_id: 2
    direction: INPUT
    is_clocking_pin: true
    capacitance: 2.0
    internal_power: {
      tables: {
        transition: POWER_TRANSITION_RISE
        shape_id: 1
        values: 3.0
      }
      tables: {
        transition: POWER_TRANSITION_FALL
        shape_id: 1
        values: 4.0
      }
    }
  }
  pins: {
    name_string_id: 3
    direction: OUTPUT
    function_string_id: 4
    timing_arcs: {
      related_pin_string_id: 2
      timing_type: TIMING_TYPE_RISING_EDGE
      tables: {
        kind: TIMING_TABLE_KIND_RISE_TRANSITION
        shape_id: 1
        values: 1.0
      }
      tables: {
        kind: TIMING_TABLE_KIND_FALL_TRANSITION
        shape_id: 1
        values: 1.0
      }
    }
  }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
lut_shapes: {}
interned_strings: ["D", "CLK", "Q", "IQ"]
"#;

fn write_fixture(netlist: &str, liberty: &str) -> tempfile::TempDir {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    std::fs::write(temp_dir.path().join("design.gv"), netlist).expect("write netlist");
    std::fs::write(temp_dir.path().join("cells.textproto"), liberty).expect("write liberty");
    temp_dir
}

fn run_gv_eval(temp_dir: &tempfile::TempDir, input_args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    command
        .arg("gv-eval")
        .arg("--netlist")
        .arg(temp_dir.path().join("design.gv"))
        .arg("--liberty_proto")
        .arg(temp_dir.path().join("cells.textproto"));
    command.args(input_args);
    command.output().expect("gv-eval invocation should run")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "gv-eval failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn gv_eval_supports_single_and_ordered_batch_inputs() {
    let netlist = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire n;
  AND2 u_and (.A(a), .B(b), .Y(n));
  INV u_inv (.A(n), .Y(y));
endmodule
"#;
    let temp_dir = write_fixture(netlist, COMBINATIONAL_LIBERTY);

    let single = run_gv_eval(&temp_dir, &["(bits[1]:1, bits[1]:1)"]);
    assert_success(&single);
    assert_eq!(String::from_utf8_lossy(&single.stdout), "bits[1]:0\n");

    let irvals_path = temp_dir.path().join("samples.irvals");
    let toggle_json_path = temp_dir.path().join("toggles.json");
    std::fs::write(
        &irvals_path,
        "(bits[1]:0, bits[1]:0)\n(bits[1]:1, bits[1]:0)\n(bits[1]:1, bits[1]:1)\n",
    )
    .expect("write input samples");
    let batch = run_gv_eval(
        &temp_dir,
        &[
            "--input-irvals",
            irvals_path.to_str().unwrap(),
            "--toggle-output-json",
            toggle_json_path.to_str().unwrap(),
        ],
    );
    assert_success(&batch);
    compare_golden_text(
        String::from_utf8_lossy(&batch.stdout).as_ref(),
        "tests/test_gv_eval.golden.txt",
    );
    compare_golden_text(
        &std::fs::read_to_string(toggle_json_path).expect("read toggle JSON"),
        "tests/test_gv_eval_toggles.golden.txt",
    );
}

#[test]
fn gv_eval_writes_dynamic_power_json() {
    let netlist = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire n;
  AND2 u_and (.A(a), .B(b), .Y(n));
  INV u_inv (.A(n), .Y(y));
endmodule
"#;
    let temp_dir = write_fixture(netlist, COMBINATIONAL_LIBERTY);
    let irvals_path = temp_dir.path().join("samples.irvals");
    let power_json_path = temp_dir.path().join("power.json");
    std::fs::write(
        &irvals_path,
        "(bits[1]:0, bits[1]:0)\n(bits[1]:1, bits[1]:0)\n(bits[1]:1, bits[1]:1)\n",
    )
    .expect("write input samples");

    let output = run_gv_eval(
        &temp_dir,
        &[
            "--input-irvals",
            irvals_path.to_str().unwrap(),
            "--power-output-json",
            power_json_path.to_str().unwrap(),
            "--module-output-load",
            "2",
            "--cycle-time",
            "4",
        ],
    );
    assert_success(&output);
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(power_json_path).expect("read power JSON"))
            .expect("parse power JSON");
    assert_eq!(report["slew_buckets"].as_array().unwrap().len(), 32);
    assert_eq!(report["primary_input_transition"], 0.01);
    assert_eq!(report["module_output_load"], 2.0);
    assert_eq!(report["cycle_time"], 4.0);
    assert_eq!(report["cell_internal_energy"], 0.0);
    assert_eq!(report["primary_input_switching_energy"], 1.0);
    assert_eq!(report["cell_output_switching_energy"], 1.5);
    assert_eq!(report["total_dynamic_energy"], 2.5);
    assert_eq!(report["average_dynamic_power"], 0.3125);
}

#[test]
fn gv_eval_accepts_named_irvals_and_rejects_wrong_names() {
    let netlist = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  AND2 u_and (.A(a), .B(b), .Y(y));
endmodule
"#;
    let temp_dir = write_fixture(netlist, COMBINATIONAL_LIBERTY);
    let irvals_path = temp_dir.path().join("named.irvals");
    std::fs::write(
        &irvals_path,
        "{b: bits[1]:0, a: bits[1]:1}\n{a: bits[1]:1, b: bits[1]:1}\n",
    )
    .expect("write named samples");
    let output = run_gv_eval(
        &temp_dir,
        &["--input-irvals", irvals_path.to_str().unwrap()],
    );
    assert_success(&output);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "bits[1]:0\nbits[1]:1\n"
    );

    std::fs::write(&irvals_path, "{a: bits[1]:1, wrong: bits[1]:1}\n")
        .expect("write bad named sample");
    let output = run_gv_eval(
        &temp_dir,
        &["--input-irvals", irvals_path.to_str().unwrap()],
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing [\"b\"]"),
        "unexpected stderr: {stderr}"
    );
    assert!(
        stderr.contains("unknown [\"wrong\"]"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn gv_eval_toggle_output_requires_two_samples() {
    let netlist = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  AND2 u_and (.A(a), .B(b), .Y(y));
endmodule
"#;
    let temp_dir = write_fixture(netlist, COMBINATIONAL_LIBERTY);
    let irvals_path = temp_dir.path().join("one_sample.irvals");
    let toggle_json_path = temp_dir.path().join("toggles.json");
    std::fs::write(&irvals_path, "(bits[1]:0, bits[1]:0)\n").expect("write input sample");

    let output = run_gv_eval(
        &temp_dir,
        &[
            "--input-irvals",
            irvals_path.to_str().unwrap(),
            "--toggle-output-json",
            toggle_json_path.to_str().unwrap(),
        ],
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("requires at least two"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!toggle_json_path.exists());
}

#[test]
fn gv_eval_sequential_named_inputs_final_state_and_toggles() {
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  DFF state (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let temp_dir = write_fixture(netlist, SEQUENTIAL_LIBERTY);
    let irvals_path = temp_dir.path().join("cycles.irvals");
    let final_state_path = temp_dir.path().join("final-state.irvals");
    let toggle_json_path = temp_dir.path().join("toggles.json");
    std::fs::write(
        &irvals_path,
        r#"{d: bits[1]:1}
{d: bits[1]:0}
{d: bits[1]:1}
"#,
    )
    .expect("write input cycles");

    let output = run_gv_eval(
        &temp_dir,
        &[
            "--sequential",
            "--input-irvals",
            irvals_path.to_str().unwrap(),
            "--initial-state-all-zeros",
            "--final-state-irvals",
            final_state_path.to_str().unwrap(),
            "--toggle-output-json",
            toggle_json_path.to_str().unwrap(),
        ],
    );
    assert_success(&output);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "bits[1]:0\nbits[1]:1\nbits[1]:0\n"
    );
    assert_eq!(
        std::fs::read_to_string(final_state_path).unwrap(),
        "{state__IQ: bits[1]:1}\n"
    );
    compare_golden_text(
        &std::fs::read_to_string(toggle_json_path).expect("read sequential toggle JSON"),
        "tests/test_gv_eval_sequential_toggles.golden.txt",
    );
}

#[test]
fn gv_eval_sequential_accepts_file_state_and_requires_state_for_registers() {
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  DFF state (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let temp_dir = write_fixture(netlist, SEQUENTIAL_LIBERTY);
    let state_path = temp_dir.path().join("state.irvals");
    std::fs::write(&state_path, "{state__IQ: bits[1]:1}\n").expect("write state");

    let output = run_gv_eval(
        &temp_dir,
        &[
            "--sequential",
            "--initial-state-file",
            state_path.to_str().unwrap(),
            "(bits[1]:0)",
        ],
    );
    assert_success(&output);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "bits[1]:1\n");

    let missing_state = run_gv_eval(&temp_dir, &["--sequential", "(bits[1]:0)"]);
    assert!(!missing_state.status.success());
    assert!(
        String::from_utf8_lossy(&missing_state.stderr)
            .contains("requires exactly one of --initial-state-all-zeros or --initial-state-file")
    );
}

#[test]
fn gv_eval_sequential_toggle_output_accepts_one_clocked_cycle() {
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  DFF state (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let temp_dir = write_fixture(netlist, SEQUENTIAL_LIBERTY);
    let irvals_path = temp_dir.path().join("one-cycle.irvals");
    let toggle_json_path = temp_dir.path().join("toggles.json");
    std::fs::write(&irvals_path, "(bits[1]:0)\n").expect("write one cycle");

    let output = run_gv_eval(
        &temp_dir,
        &[
            "--sequential",
            "--input-irvals",
            irvals_path.to_str().unwrap(),
            "--initial-state-all-zeros",
            "--toggle-output-json",
            toggle_json_path.to_str().unwrap(),
        ],
    );
    assert_success(&output);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "bits[1]:0\n");
    let activity: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(toggle_json_path).expect("read one-cycle toggle JSON"),
    )
    .expect("parse one-cycle toggle JSON");
    assert_eq!(activity["cycle_count"], 1);
    assert_eq!(activity["logic_transition_count"], 1);
    assert_eq!(activity["clock_transition_count"], 2);
    assert_eq!(activity["clock"]["toggle_count"], 2);
}

#[test]
fn gv_eval_sequential_writes_dynamic_power_json() {
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  DFF state (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let temp_dir = write_fixture(netlist, SEQUENTIAL_POWER_LIBERTY);
    let irvals_path = temp_dir.path().join("cycles.irvals");
    let power_json_path = temp_dir.path().join("power.json");
    std::fs::write(
        &irvals_path,
        r#"{d: bits[1]:1}
{d: bits[1]:0}
"#,
    )
    .expect("write sequential power cycles");

    let output = run_gv_eval(
        &temp_dir,
        &[
            "--sequential",
            "--input-irvals",
            irvals_path.to_str().unwrap(),
            "--initial-state-all-zeros",
            "--power-output-json",
            power_json_path.to_str().unwrap(),
            "--clock-transition",
            "2",
            "--module-output-load",
            "1",
            "--cycle-time",
            "5",
        ],
    );
    assert_success(&output);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "bits[1]:0\nbits[1]:1\n"
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(power_json_path).expect("read power JSON"))
            .expect("parse power JSON");
    assert_eq!(report["cycle_count"], 2);
    assert_eq!(report["phase_transition_count"], 5);
    assert_eq!(report["clock_transition_count"], 4);
    assert_eq!(report["clock_transition"], 2.0);
    assert_eq!(report["primary_input_switching_energy"], 0.5);
    assert_eq!(report["clock_switching_energy"], 4.0);
    assert_eq!(report["cell_output_switching_energy"], 1.0);
    assert_eq!(report["cell_internal_energy"], 14.0);
    assert_eq!(report["total_dynamic_energy"], 19.5);
    assert_eq!(report["average_dynamic_power"], 1.95);
}

#[test]
fn gv_eval_rejects_sequential_cells() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["D", "CLK", "Q", "IQ"]
"#;
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  DFF state (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let temp_dir = write_fixture(netlist, liberty);
    let output = run_gv_eval(&temp_dir, &["(bits[1]:0, bits[1]:0)"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("gv-eval error:"));
    assert!(stderr.contains("sequential cell 'DFF' instance 'state'"));
}
