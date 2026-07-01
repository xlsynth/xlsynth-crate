// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::liberty_model::PinDirection;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, GvToggleAggregate, PinConnection, TogglePinConnection, load_labeled_netlist_aig,
};
use xlsynth_g8r::netlist::parse::PortDirection;
use xlsynth_g8r::netlist::power::{GV_POWER_SLEW_BUCKET_COUNT, GvDynamicPowerOptions};

fn write_fixture(
    netlist: &str,
    liberty: &str,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("design.gv");
    let liberty_path = temp_dir.path().join("cells.textproto");
    std::fs::write(&netlist_path, netlist).expect("write netlist");
    std::fs::write(&liberty_path, liberty).expect("write liberty");
    (temp_dir, netlist_path, liberty_path)
}

#[test]
fn labeled_netlist_aig_evaluates_and_preserves_external_pin_labels() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
}
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 5 }
}
interned_strings: ["A", "B", "Y", "A & B", "!A"]
"#;
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
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(&netlist_path, &liberty_path, &GvEvalOptions::default())
        .expect("build labeled evaluation model");

    assert_eq!(model.module_name, "top");
    assert_eq!(
        model
            .module_ports
            .iter()
            .map(|port| (port.name.as_str(), port.direction.clone()))
            .collect::<Vec<_>>(),
        vec![
            ("a", PortDirection::Input),
            ("b", PortDirection::Input),
            ("y", PortDirection::Output),
        ]
    );
    assert_eq!(model.instances.len(), 2);
    assert_eq!(model.instances[0].instance_name, "u_and");
    assert_eq!(model.instances[0].cell_type, "AND2");
    assert_eq!(
        model.instances[0]
            .pins
            .iter()
            .map(|pin| (pin.pin_name.as_str(), pin.direction))
            .collect::<Vec<_>>(),
        vec![
            ("A", PinDirection::Input),
            ("B", PinDirection::Input),
            ("Y", PinDirection::Output),
        ]
    );
    assert_eq!(
        model.instances[0].pins[0].connection,
        PinConnection::Net {
            net_name: "a".to_string(),
            bit_number: 0,
        }
    );
    assert_eq!(
        model.instances[0].pins[2].connection,
        PinConnection::Net {
            net_name: "n".to_string(),
            bit_number: 0,
        }
    );
    assert_eq!(
        model.instances[0].pins[2].operand,
        model.instances[1].pins[0].operand
    );

    let high = model
        .evaluate_ir_value(&IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap())
        .expect("evaluate true/true");
    let low = model
        .evaluate_ir_value(&IrValue::parse_typed("(bits[1]:1, bits[1]:0)").unwrap())
        .expect("evaluate true/false");
    assert_eq!(high.to_string(), "bits[1]:0");
    assert_eq!(low.to_string(), "bits[1]:1");
}

#[test]
fn module_port_labels_preserve_original_bit_numbers() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "BUF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
}
interned_strings: ["A", "Y"]
"#;
    let netlist = r#"
module top (a, y);
  input [0:1] a;
  output [1:0] y;
  BUF u0 (.A(a[0]), .Y(y[1]));
  BUF u1 (.A(a[1]), .Y(y[0]));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(&netlist_path, &liberty_path, &GvEvalOptions::default())
        .expect("build vector evaluation model");

    assert_eq!(
        model.module_ports[0]
            .bits_lsb_to_msb
            .iter()
            .map(|bit| bit.bit_number)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(
        model.module_ports[1]
            .bits_lsb_to_msb
            .iter()
            .map(|bit| bit.bit_number)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        model
            .evaluate_bits(&[IrBits::make_ubits(2, 0b10).unwrap()])
            .expect("evaluate vector input"),
        vec![IrBits::make_ubits(2, 0b10).unwrap()]
    );
}

#[test]
fn toggle_activity_counts_external_pins_outside_the_primary_output_cone() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "BUF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
}

interned_strings: ["A", "Y"]
"#;
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  BUF live (.A(a), .Y(y));
  BUF dead (.A(a), .Y());
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(&netlist_path, &liberty_path, &GvEvalOptions::default())
        .expect("build toggle evaluation model");
    let samples = [
        IrValue::parse_typed("(bits[1]:0)").unwrap(),
        IrValue::parse_typed("(bits[1]:1)").unwrap(),
        IrValue::parse_typed("(bits[1]:0)").unwrap(),
    ];

    let activity = model
        .count_toggle_activity(&samples)
        .expect("count toggle activity");
    assert_eq!(activity.sample_count, 3);
    assert_eq!(activity.transition_count, 2);
    assert_eq!(
        activity.aggregate,
        GvToggleAggregate {
            module_input_toggles: 2,
            module_output_toggles: 2,
            cell_input_pin_toggles: 4,
            cell_output_pin_toggles: 4,
        }
    );
    assert_eq!(activity.instances[1].instance_name, "dead");
    assert_eq!(
        activity.instances[1].pins[1].connection,
        TogglePinConnection::Unconnected
    );
    assert_eq!(activity.instances[1].pins[1].toggle_count, 2);
    assert_eq!(activity.instances[1].pins[1].toggle_rate, 1.0);
}

#[test]
fn dynamic_power_uses_slew_histograms_loads_and_previous_when_values() {
    let liberty = r#"
format_magic: 5496997758177923663
units: {
  time_unit: "ns"
  capacitance_unit: "pf"
  voltage_unit: "V"
}
nominal_voltage: 2.0
cells: {
  name: "BUF"
  pins: { name_string_id: 1 direction: INPUT capacitance: 2.0 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 1
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_POSITIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: {
        kind: TIMING_TABLE_KIND_RISE_TRANSITION
        shape_id: 1
        values: 2.0
        values: 2.0
        values: 2.0
        values: 2.0
      }
      tables: {
        kind: TIMING_TABLE_KIND_FALL_TRANSITION
        shape_id: 1
        values: 2.0
        values: 2.0
        values: 2.0
        values: 2.0
      }
    }
    internal_power: {
      related_pin_string_ids: 1
      when_string_id: 3
      tables: {
        transition: POWER_TRANSITION_RISE
        shape_id: 2
        values: 10.0
        values: 10.0
        values: 10.0
        values: 10.0
      }
      tables: {
        transition: POWER_TRANSITION_FALL
        shape_id: 2
        values: 20.0
        values: 20.0
        values: 20.0
        values: 20.0
      }
    }
  }
}
lu_table_templates: {
  kind: LUT_TEMPLATE_KIND_TIMING
  name: "timing"
  variable_1: LUT_VARIABLE_INPUT_NET_TRANSITION
  variable_2: LUT_VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE
  index_1_id: 1
  index_2_id: 2
}
lu_table_templates: {
  kind: LUT_TEMPLATE_KIND_POWER
  name: "power"
  variable_1: LUT_VARIABLE_INPUT_TRANSITION_TIME
  variable_2: LUT_VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE
  index_1_id: 1
  index_2_id: 2
}
lut_axes: { values: 1.0 values: 3.0 }
lut_axes: { values: 0.0 values: 4.0 }
lut_shapes: { template_id: 1 dimensions: 2 dimensions: 2 }
lut_shapes: { template_id: 2 dimensions: 2 dimensions: 2 }
interned_strings: ["A", "Y", "!Y"]
"#;
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  BUF u_buf (.A(a), .Y(y));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(&netlist_path, &liberty_path, &GvEvalOptions::default())
        .expect("build power evaluation model");
    let library = xlsynth_g8r::netlist::io::load_liberty_from_path(&liberty_path)
        .expect("reload power library");
    let samples = [
        IrValue::parse_typed("(bits[1]:0)").unwrap(),
        IrValue::parse_typed("(bits[1]:1)").unwrap(),
        IrValue::parse_typed("(bits[1]:0)").unwrap(),
    ];

    let report = model
        .analyze_dynamic_power(
            &library,
            &samples,
            GvDynamicPowerOptions {
                primary_input_transition: 1.0,
                module_output_load: 3.0,
                cycle_time: Some(5.0),
            },
        )
        .expect("analyze dynamic power");

    assert_eq!(report.slew_buckets.len(), GV_POWER_SLEW_BUCKET_COUNT);
    assert_eq!(report.primary_input_switching_energy, 8.0);
    assert_eq!(report.cell_output_switching_energy, 12.0);
    assert_eq!(report.cell_internal_energy, 10.0);
    assert_eq!(report.total_dynamic_energy, 30.0);
    assert_eq!(report.average_energy_per_transition, 15.0);
    assert_eq!(report.average_dynamic_power, Some(3.0));
    assert_eq!(report.instances[0].outputs[0].rise_count, 1);
    assert_eq!(report.instances[0].outputs[0].fall_count, 1);
    assert_eq!(
        report.instances[0].outputs[0]
            .slew_histogram
            .rise
            .iter()
            .sum::<f64>(),
        1.0
    );
    assert_eq!(
        report.instances[0].outputs[0]
            .slew_histogram
            .fall
            .iter()
            .sum::<f64>(),
        1.0
    );
    assert_eq!(report.diagnostics.when_evaluation_count, 2);
    assert_eq!(report.diagnostics.when_changed_during_transition_count, 2);
}

#[test]
fn sequential_cells_are_rejected_before_projection() {
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
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let error = load_labeled_netlist_aig(&netlist_path, &liberty_path, &GvEvalOptions::default())
        .expect_err("sequential cell should be rejected");
    let error = format!("{error:#}");
    assert!(error.contains("sequential cell 'DFF' instance 'state'"));
    assert!(error.contains("kind: ff"));
}
