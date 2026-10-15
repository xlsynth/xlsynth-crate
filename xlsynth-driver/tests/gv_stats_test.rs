// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write;
use std::path::Path;
use std::process::{Command, Output};

use xlsynth_test_helpers::compare_golden_text;

fn make_timing_enabled_inv_nand2_liberty_textproto() -> &'static str {
    r#"
format_magic: 5496997758194700879
default_output_load: 0.004
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT capacitance: 0.0 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 3
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 1.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 1.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
    }
  }
  area: 1.0
}
cells: {
  name: "NAND2"
  pins: { name_string_id: 1 direction: INPUT capacitance: 0.0 }
  pins: { name_string_id: 4 direction: INPUT capacitance: 0.0 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 8
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
    }
    timing_arcs: {
      related_pin_string_id: 4
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
    }
  }
  area: 2.0
}
cells: {
  name: "DFF"
  pins: {
    name_string_id: 6
    direction: INPUT
    capacitance: 0.0
    timing_arcs: {
      related_pin_string_id: 5
      timing_type: TIMING_TYPE_SETUP_RISING
      tables: { kind: TIMING_TABLE_KIND_RISE_CONSTRAINT shape_id: 1 values: 0.25 }
      tables: { kind: TIMING_TABLE_KIND_FALL_CONSTRAINT shape_id: 1 values: 0.25 }
    }
  }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true capacitance: 0.0 }
  pins: {
    name_string_id: 7
    direction: OUTPUT
    function_string_id: 7
    timing_arcs: {
      related_pin_string_id: 5
      timing_sense: TIMING_SENSE_NON_UNATE
      timing_type: TIMING_TYPE_RISING_EDGE
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
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
interned_strings: ["A", "Y", "!A", "B", "CLK", "D", "Q", "!(A*B)"]
lut_shapes: {}
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
    assert_eq!(stats_json["module_output_load"], 0.004);
    assert_eq!(stats_json["max_delay"], 3.0);
    assert_eq!(stats_json["cell_count"], 2);
    assert_eq!(stats_json["cell_levels"], 2);
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["delay_slew_below_min_clamp_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["delay_slew_single_above_max_extrapolation_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["delay_slew_multiple_above_max_clamp_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["setup_below_min_clamp_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["setup_above_max_clamp_count"],
        0
    );
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
    assert_eq!(stats_json["module_output_load"], 0.004);
    assert_eq!(stats_json["max_delay"], serde_json::Value::Null);
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
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["delay_slew_below_min_clamp_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["delay_slew_single_above_max_extrapolation_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["delay_slew_multiple_above_max_clamp_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["setup_below_min_clamp_count"],
        0
    );
    assert_eq!(
        stats_json["timing_query_diagnostic_counts"]["setup_above_max_clamp_count"],
        0
    );
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

/// Runs the report command with optional load overrides and a JSON report.
fn run_stats(directory: &Path, extra_args: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("gv-stats")
        .arg("--netlist")
        .arg(directory.join("mapped.gv"))
        .arg("--liberty_proto")
        .arg(directory.join("lib.textproto"))
        .arg("--json_out")
        .arg(directory.join("stats.json"))
        .args(extra_args)
        .output()
}

#[test]
fn gv_stats_default_and_overrides_affect_timing_in_library_units() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("mapped.gv"),
        "module top(input a, output y); INV u0(.A(a), .Y(y)); endmodule",
    )
    .unwrap();
    // Load-dependent tables catch accidental conversion to fF: all pin loads,
    // table axes, defaults, and overrides must stay in the same library units.
    let liberty = r#"
format_magic: 5496997758194700879
default_output_load: 0.005
units: { time_unit: "1ps" capacitance_unit: "1pf" }
cells: {
  name: "INV"
  area: 1.0
  pins: { name_string_id: 1 direction: INPUT capacitance: 0.001 }
  pins: {
    name_string_id: 2 direction: OUTPUT function_string_id: 3
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: [1.0, 9.0] }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: [1.0, 9.0] }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: [0.1, 0.9] }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: [0.1, 0.9] }
    }
  }
}
interned_strings: ["A", "Y", "!A"]
lut_axes: { values: [0.001, 0.009] }
lu_table_templates: {
  kind: LUT_TEMPLATE_KIND_TIMING
  name: "load"
  variable_1: LUT_VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE
  index_1_id: 1
}
lut_shapes: { template_id: 1 index_1_id: 1 dimensions: 2 }
"#;
    for units in ["1pf", "1ff"] {
        std::fs::write(
            directory.path().join("lib.textproto"),
            liberty.replace("1pf", units),
        )
        .unwrap();
        for (args, expected_load, expected_delay) in [
            (vec![], 0.005, 5.0),
            (vec!["--module_output_load", "0"], 0.0, 1.0),
            (vec!["--module_output_load", "0.009"], 0.009, 9.0),
        ] {
            let output = run_stats(directory.path(), &args).unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(directory.path().join("stats.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(report["module_output_load"], expected_load);
            assert!((report["max_delay"].as_f64().unwrap() - expected_delay).abs() < 1e-9);
        }
    }
}

#[test]
fn gv_stats_output_load_affects_direct_flop_outputs_not_internal_d_pin_loads() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("mapped.gv"),
        r#"module top(input a, input clk, output y);
  wire q;
  DFF r0(.D(a), .CLK(clk), .Q(q));
  DFF r1(.D(q), .CLK(clk), .Q(y));
endmodule
"#,
    )
    .unwrap();
    std::fs::write(
        directory.path().join("lib.textproto"),
        r#"
format_magic: 5496997758194700879
default_output_load: 0.005
units: { time_unit: "1ps" capacitance_unit: "1pf" }
cells: {
  name: "DFF"
  area: 1.0
  pins: {
    name_string_id: 1 direction: INPUT capacitance: 0.002
    timing_arcs: {
      related_pin_string_id: 2
      timing_type: TIMING_TYPE_SETUP_RISING
      tables: { kind: TIMING_TABLE_KIND_RISE_CONSTRAINT shape_id: 2 values: 0.25 }
      tables: { kind: TIMING_TABLE_KIND_FALL_CONSTRAINT shape_id: 2 values: 0.25 }
    }
  }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true capacitance: 0.0 }
  pins: {
    name_string_id: 3 direction: OUTPUT function_string_id: 4
    timing_arcs: {
      related_pin_string_id: 2
      timing_sense: TIMING_SENSE_NON_UNATE
      timing_type: TIMING_TYPE_RISING_EDGE
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: [1.0, 9.0] }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: [1.0, 9.0] }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 2 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 2 values: 0.1 }
    }
  }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["D", "CLK", "Q", "IQ"]
lut_axes: { values: [0.001, 0.009] }
lu_table_templates: {
  kind: LUT_TEMPLATE_KIND_TIMING
  name: "load"
  variable_1: LUT_VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE
  index_1_id: 1
}
lut_shapes: { template_id: 1 index_1_id: 1 dimensions: 2 }
lut_shapes: {}
"#,
    )
    .unwrap();
    for (args, expected_load, expected_output_delay) in [
        (vec![], 0.005, 5.0),
        (vec!["--module_output_load", "0"], 0.0, 1.0),
        (vec!["--module_output_load", "0.009"], 0.009, 9.0),
    ] {
        let output = run_stats(directory.path(), &args).unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("stats.json")).unwrap())
                .unwrap();
        assert_eq!(report["module_output_load"], expected_load);
        // r1 drives the module load directly. r0 always drives just r1.D's
        // 0.002 pF, so its clock-to-Q remains 2 ps regardless of that load.
        for (field, expected) in [
            ("max_register_to_output_delay", expected_output_delay),
            ("max_register_to_register_delay", 2.25),
            ("max_input_to_register_delay", 0.25),
        ] {
            assert!(
                (report[field].as_f64().unwrap() - expected).abs() < 1e-9,
                "{field}: {report}"
            );
        }
        let internal = &report["max_register_to_register_delay_breakdown"];
        assert!((internal["clock_to_output_delay"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(internal["setup_delay"], 0.25);
        assert_eq!(internal["combinational_delay"], 0.0);
    }
}

#[test]
fn gv_stats_missing_or_invalid_defaults_require_an_explicit_override() {
    let directory = write_fixture();
    let base =
        make_timing_enabled_inv_nand2_liberty_textproto().replace("default_output_load: 0.004", "");
    let mut errors = String::new();
    for (name, field) in [
        ("missing", ""),
        ("zero", "default_output_load: 0"),
        ("negative", "default_output_load: -1"),
        ("nan", "default_output_load: nan"),
        ("infinite", "default_output_load: inf"),
    ] {
        std::fs::write(
            directory.path().join("lib.textproto"),
            format!("{base}\n{field}\n"),
        )
        .unwrap();
        let output = run_stats(directory.path(), &[]).unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        writeln!(errors, "{name}:").unwrap();
        errors.push_str(std::str::from_utf8(&output.stderr).unwrap());
        // Explicit loads do not depend on a valid library default, including
        // legacy libraries with no default and an intentional zero-load run.
        for value in ["0", "0.008"] {
            let output = run_stats(directory.path(), &["--module_output_load", value]).unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(directory.path().join("stats.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(report["module_output_load"], value.parse::<f64>().unwrap());
        }
    }
    compare_golden_text(&errors, "tests/test_gv_stats_load_errors.golden.txt");
}
