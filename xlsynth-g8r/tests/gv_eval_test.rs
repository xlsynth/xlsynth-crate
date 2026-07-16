// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig_sim::sequential::{self, SequentialState};
use xlsynth_g8r::liberty::parser::{
    LibertyPayloadOptions, parse_liberty_files_with_payload_options,
};
use xlsynth_g8r::liberty_model::{Library, PinDirection};
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, GvToggleAggregate, PinConnection, SequentialAigSignal, SequentialClockEdge,
    TogglePinConnection, load_labeled_netlist_aig, load_labeled_netlist_aig_with_liberty,
    load_labeled_sequential_netlist_aig, load_sequential_netlist_gate_fn,
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
fn labeled_netlist_aig_accepts_an_in_memory_liberty_model() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("design.gv");
    let liberty_path = temp_dir.path().join("cells.lib");
    std::fs::write(
        &netlist_path,
        r#"
module top (a, y);
  input a;
  output y;
  INV u_inv (.A(a), .Y(y));
endmodule
"#,
    )
    .expect("write netlist");
    std::fs::write(
        &liberty_path,
        r#"
library (test) {
  cell (INV) {
    area: 1;
    pin (A) { direction: input; }
    pin (Y) { direction: output; function: "!A"; }
  }
}
"#,
    )
    .expect("write liberty");
    let liberty = parse_liberty_files_with_payload_options(
        &[&liberty_path],
        LibertyPayloadOptions {
            include_timing: false,
            include_power: false,
        },
    )
    .expect("parse liberty");
    let model =
        load_labeled_netlist_aig_with_liberty(&netlist_path, &liberty, &GvEvalOptions::default())
            .expect("build labeled evaluation model");

    let output = model
        .evaluate_bits(&[IrBits::make_ubits(1, 1).unwrap()])
        .expect("evaluate inverter");
    assert_eq!(output, vec![IrBits::make_ubits(1, 0).unwrap()]);
}

#[test]
fn labeled_netlist_aig_accepts_a_module_without_outputs() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("design.gv");
    std::fs::write(
        &netlist_path,
        r#"
module top (a);
  input a;
endmodule
"#,
    )
    .expect("write netlist");
    let model = load_labeled_netlist_aig_with_liberty(
        &netlist_path,
        &Library::default(),
        &GvEvalOptions::default(),
    )
    .expect("build outputless evaluation model");

    let outputs = model
        .evaluate_bits(&[IrBits::make_ubits(1, 1).unwrap()])
        .expect("evaluate outputless module");
    assert!(outputs.is_empty());
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
fn dynamic_power_uses_owner_output_direction_for_negative_unate_cells() {
    let liberty = r#"
format_magic: 5496997758177923663
units: {
  time_unit: "ns"
  capacitance_unit: "pf"
  voltage_unit: "V"
}
nominal_voltage: 1.0
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 3
    internal_power: {
      related_pin_string_ids: 1
      tables: {
        transition: POWER_TRANSITION_RISE
        shape_id: 1
        values: 10.0
      }
      tables: {
        transition: POWER_TRANSITION_FALL
        shape_id: 1
        values: 20.0
      }
    }
  }
}
lut_shapes: {}
interned_strings: ["A", "Y", "!A"]
"#;
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  INV u_inv (.A(a), .Y(y));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(&netlist_path, &liberty_path, &GvEvalOptions::default())
        .expect("build negative-unate power evaluation model");
    let library = xlsynth_g8r::netlist::io::load_liberty_from_path(&liberty_path)
        .expect("reload negative-unate power library");
    let options = GvDynamicPowerOptions {
        primary_input_transition: 1.0,
        module_output_load: 0.0,
        cycle_time: None,
    };

    let output_fall = model
        .analyze_dynamic_power(
            &library,
            &[
                IrValue::parse_typed("(bits[1]:0)").unwrap(),
                IrValue::parse_typed("(bits[1]:1)").unwrap(),
            ],
            options,
        )
        .expect("analyze inverter output fall");
    assert_eq!(output_fall.cell_internal_energy, 20.0);
    assert_eq!(
        output_fall.instances[0].pin_internal_energy[0].internal_energy,
        20.0
    );

    let output_rise = model
        .analyze_dynamic_power(
            &library,
            &[
                IrValue::parse_typed("(bits[1]:1)").unwrap(),
                IrValue::parse_typed("(bits[1]:0)").unwrap(),
            ],
            options,
        )
        .expect("analyze inverter output rise");
    assert_eq!(output_rise.cell_internal_energy, 10.0);
    assert_eq!(
        output_rise.instances[0].pin_internal_energy[0].internal_energy,
        10.0
    );
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

#[test]
fn sequential_netlist_gate_fn_simulates_ff_state() {
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
    let model = load_labeled_sequential_netlist_aig(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions::default(),
    )
    .expect("load labeled sequential evaluation model");
    let design = &model.sequential_gate_fn;

    assert_eq!(
        design.clock.as_ref().map(|clock| clock.name.as_str()),
        Some("clk")
    );
    assert_eq!(
        model.clock.as_ref().and_then(|clock| clock.active_edge),
        Some(SequentialClockEdge::Rising)
    );
    assert_eq!(
        model.module_ports[1].bits_lsb_to_msb[0].signal,
        SequentialAigSignal::Clock
    );
    assert_eq!(model.instances[0].state_register_index, Some(0));
    assert_eq!(
        model.instances[0].pins[1].signal,
        SequentialAigSignal::Clock
    );
    assert_eq!(
        model.instances[0].pins[1].connection,
        PinConnection::Net {
            net_name: "clk".to_string(),
            bit_number: 0,
        }
    );
    assert_eq!(design.registers.len(), 1);
    assert_eq!(design.inputs.len(), 1);
    assert_eq!(design.transition.inputs[design.inputs[0].index()].name, "d");
    assert_eq!(design.outputs.len(), 1);
    assert_eq!(
        design.transition.outputs[design.outputs[0].index()].name,
        "q"
    );

    let trace = sequential::simulate(
        design,
        &[
            vec![IrBits::make_ubits(1, 1).unwrap()],
            vec![IrBits::make_ubits(1, 0).unwrap()],
            vec![IrBits::make_ubits(1, 0).unwrap()],
        ],
        SequentialState::all_zeros(design),
    )
    .expect("simulate DFF");
    assert_eq!(
        trace.external_outputs(),
        &[
            vec![IrBits::make_ubits(1, 0).unwrap()],
            vec![IrBits::make_ubits(1, 1).unwrap()],
            vec![IrBits::make_ubits(1, 0).unwrap()],
        ]
    );
}

#[test]
fn sequential_netlist_gate_fn_supports_feedback_and_qn() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFFE"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 4 direction: OUTPUT function_string_id: 6 }
  pins: { name_string_id: 5 direction: OUTPUT function_string_id: 7 }
  sequential: {
    state_var: "IQ"
    complementary_state_var: "IQN"
    next_state: "(!EN * IQ) + (EN * D)"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["D", "EN", "CLK", "Q", "QN", "IQ", "IQN"]
"#;
    let netlist = r#"
module top (d, en, clk, q, qn);
  input d;
  input en;
  input clk;
  output q;
  output qn;
  DFFE state (.D(d), .EN(en), .CLK(clk), .Q(q), .QN(qn));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let design =
        load_sequential_netlist_gate_fn(&netlist_path, &liberty_path, &GvEvalOptions::default())
            .expect("load feedback sequential evaluation model");

    let trace = sequential::simulate(
        &design,
        &[
            vec![
                IrBits::make_ubits(1, 1).unwrap(),
                IrBits::make_ubits(1, 1).unwrap(),
            ],
            vec![
                IrBits::make_ubits(1, 0).unwrap(),
                IrBits::make_ubits(1, 0).unwrap(),
            ],
            vec![
                IrBits::make_ubits(1, 0).unwrap(),
                IrBits::make_ubits(1, 1).unwrap(),
            ],
            vec![
                IrBits::make_ubits(1, 0).unwrap(),
                IrBits::make_ubits(1, 0).unwrap(),
            ],
        ],
        SequentialState::all_zeros(&design),
    )
    .expect("simulate feedback DFF");
    assert_eq!(
        trace.external_outputs(),
        &[
            vec![
                IrBits::make_ubits(1, 0).unwrap(),
                IrBits::make_ubits(1, 1).unwrap(),
            ],
            vec![
                IrBits::make_ubits(1, 1).unwrap(),
                IrBits::make_ubits(1, 0).unwrap(),
            ],
            vec![
                IrBits::make_ubits(1, 1).unwrap(),
                IrBits::make_ubits(1, 0).unwrap(),
            ],
            vec![
                IrBits::make_ubits(1, 0).unwrap(),
                IrBits::make_ubits(1, 1).unwrap(),
            ],
        ]
    );
}

#[test]
fn labeled_sequential_netlist_aig_preserves_negative_edge_clock() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFFN"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "!CLK"
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
  DFFN state (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_sequential_netlist_aig(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions::default(),
    )
    .expect("load negative-edge sequential model");
    assert_eq!(
        model.clock.as_ref().and_then(|clock| clock.active_edge),
        Some(SequentialClockEdge::Falling)
    );
}

#[test]
fn sequential_netlist_gate_fn_accepts_a_clock_wiring_alias() {
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
  wire clk_alias;
  assign clk_alias = clk;
  DFF state (.D(d), .CLK(clk_alias), .Q(q));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let design =
        load_sequential_netlist_gate_fn(&netlist_path, &liberty_path, &GvEvalOptions::default())
            .expect("load sequential model with clock alias");
    assert_eq!(
        design.clock.as_ref().map(|clock| clock.name.as_str()),
        Some("clk")
    );
}

#[test]
fn sequential_netlist_gate_fn_rejects_mixed_clock_edges() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFFP"
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
cells: {
  name: "DFFN"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "!CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["D", "CLK", "Q", "IQ"]
"#;
    let netlist = r#"
module top (d, clk, q0, q1);
  input d;
  input clk;
  output q0;
  output q1;
  DFFP state0 (.D(d), .CLK(clk), .Q(q0));
  DFFN state1 (.D(d), .CLK(clk), .Q(q1));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let error =
        load_sequential_netlist_gate_fn(&netlist_path, &liberty_path, &GvEvalOptions::default())
            .expect_err("mixed clock edges should be rejected");
    assert!(format!("{error:#}").contains("mixed positive-edge and negative-edge"));
}

#[test]
fn sequential_netlist_gate_fn_rejects_a_derived_clock() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 3 }
}
cells: {
  name: "DFF"
  pins: { name_string_id: 4 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 6 direction: OUTPUT function_string_id: 7 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["A", "Y", "!A", "D", "CLK", "Q", "IQ"]
"#;
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  wire derived_clk;
  INV clock_inv (.A(clk), .Y(derived_clk));
  DFF state (.D(d), .CLK(derived_clk), .Q(q));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let error =
        load_sequential_netlist_gate_fn(&netlist_path, &liberty_path, &GvEvalOptions::default())
            .expect_err("derived clock should be rejected");
    assert!(format!("{error:#}").contains("derived clock"));
}

#[test]
fn sequential_netlist_gate_fn_rejects_async_reset_cells() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFFCLR"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 4 direction: OUTPUT function_string_id: 5 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    clear_expr: "RST"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["D", "RST", "CLK", "Q", "IQ"]
"#;
    let netlist = r#"
module top (d, rst, clk, q);
  input d;
  input rst;
  input clk;
  output q;
  DFFCLR state (.D(d), .RST(rst), .CLK(clk), .Q(q));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let error =
        load_sequential_netlist_gate_fn(&netlist_path, &liberty_path, &GvEvalOptions::default())
            .expect_err("asynchronous reset should be rejected");
    assert!(format!("{error:#}").contains("does not support asynchronous reset cell 'DFFCLR'"));
}

#[test]
fn sequential_netlist_gate_fn_accepts_an_outputless_module() {
    let liberty = "format_magic: 5496997758177923663\n";
    let netlist = r#"
module top (clk, a);
  input clk;
  input a;
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let design = load_sequential_netlist_gate_fn(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions {
            clock_port_name: Some("clk".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect("load outputless sequential evaluation model");
    assert_eq!(
        design.clock.as_ref().map(|clock| clock.name.as_str()),
        Some("clk")
    );
    assert_eq!(design.inputs.len(), 1);
    assert_eq!(design.transition.inputs[design.inputs[0].index()].name, "a");
    let trace = sequential::simulate(
        &design,
        &[vec![IrBits::make_ubits(1, 1).unwrap()]],
        SequentialState::all_zeros(&design),
    )
    .expect("simulate outputless module");
    assert_eq!(trace.external_outputs(), &[Vec::<IrBits>::new()]);
}

#[test]
fn labeled_netlist_aig_flattens_structural_hierarchy_and_keeps_boundaries() {
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
module child (a, y);
  input [1:0] a;
  output [1:0] y;
  BUF u_buf (.A(a[0]), .Y(y[0]));
endmodule

module top (a, y);
  input [1:0] a;
  output y;
  wire [1:0] child_y;
  child u_child (.a(a), .y(child_y));
  BUF u_top (.A(child_y[0]), .Y(y));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions {
            module_name: Some("top".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect("build hierarchical evaluation model");

    assert_eq!(
        model
            .instances
            .iter()
            .map(|instance| instance.instance_name.as_str())
            .collect::<Vec<_>>(),
        vec!["u_child/u_buf", "u_top"]
    );
    assert_eq!(model.module_boundaries.len(), 1);
    let boundary = &model.module_boundaries[0];
    assert_eq!(boundary.instance_path, "u_child");
    assert_eq!(boundary.module_name, "child");
    assert_eq!(
        boundary
            .ports
            .iter()
            .map(|port| {
                (
                    port.name.as_str(),
                    port.bits_lsb_to_msb
                        .iter()
                        .map(|bit| bit.bit_number)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        vec![("a", vec![0, 1]), ("y", vec![0])]
    );

    let low = model
        .evaluate_ir_value(&IrValue::parse_typed("(bits[2]:0)").unwrap())
        .expect("evaluate low child input");
    let high = model
        .evaluate_ir_value(&IrValue::parse_typed("(bits[2]:1)").unwrap())
        .expect("evaluate high child input");
    assert_eq!(low.to_string(), "bits[1]:0");
    assert_eq!(high.to_string(), "bits[1]:1");

    let activity = model
        .count_toggle_activity(&[
            IrValue::parse_typed("(bits[2]:0)").unwrap(),
            IrValue::parse_typed("(bits[2]:1)").unwrap(),
            IrValue::parse_typed("(bits[2]:0)").unwrap(),
        ])
        .expect("count hierarchical toggle activity");
    assert_eq!(activity.module_boundaries.len(), 1);
    assert_eq!(activity.module_boundaries[0].instance_path, "u_child");
    assert_eq!(
        activity.module_boundaries[0].ports[1].bits_lsb_to_msb[0].toggle_count,
        2
    );
}

#[test]
fn labeled_netlist_aig_sizes_hierarchical_port_connections_directionally() {
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
module copy1 (a, y);
  input a;
  output y;
  BUF u_buf (.A(a), .Y(y));
endmodule

module copy2 (a, y);
  input [1:0] a;
  output [1:0] y;
  BUF u_buf0 (.A(a[0]), .Y(y[0]));
  BUF u_buf1 (.A(a[1]), .Y(y[1]));
endmodule

module top (scalar, wide, input_pad, input_trunc, output_pad, output_trunc);
  input scalar;
  input [1:0] wide;
  output [1:0] input_pad;
  output input_trunc;
  output [1:0] output_pad;
  output output_trunc;

  copy2 u_input_pad (.a(wide[1]), .y(input_pad));
  copy1 u_input_trunc (.a({wide[1], scalar}), .y(input_trunc));
  copy1 u_output_pad (.a(scalar), .y({output_pad[1], output_pad[0]}));
  copy2 u_output_trunc (.a(wide), .y(output_trunc));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions {
            module_name: Some("top".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect("build hierarchy with width-mismatched ports");

    let outputs = model
        .evaluate_bits(&[
            IrBits::make_ubits(1, 1).unwrap(),
            IrBits::make_ubits(2, 0b10).unwrap(),
        ])
        .expect("evaluate directionally sized ports");
    assert_eq!(
        outputs,
        vec![
            IrBits::make_ubits(2, 0b01).unwrap(),
            IrBits::make_ubits(1, 1).unwrap(),
            IrBits::make_ubits(2, 0b01).unwrap(),
            IrBits::make_ubits(1, 0).unwrap(),
        ]
    );
}

#[test]
fn labeled_netlist_aig_reports_unknown_child_ports_in_source_order() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("design.gv");
    std::fs::write(
        &netlist_path,
        r#"
module child (a, y);
  input a;
  output y;
endmodule

module top (a, y);
  input a;
  output y;
  child u_child (.zeta(a), .alpha(a), .a(a), .y(y));
endmodule
"#,
    )
    .expect("write netlist");
    let error = load_labeled_netlist_aig_with_liberty(
        &netlist_path,
        &Library::default(),
        &GvEvalOptions {
            module_name: Some("top".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect_err("unknown child ports should be rejected");
    assert_eq!(
        format!("{error:#}"),
        "module instance 'u_child' of 'child' connects unknown port 'zeta'"
    );
}

#[test]
fn labeled_netlist_aig_rejects_unconnected_child_inout_ports() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("design.gv");
    std::fs::write(
        &netlist_path,
        r#"
module child (io, y);
  inout io;
  output y;
  assign y = 1'b0;
endmodule

module top (y);
  output y;
  child u_child (.io(), .y(y));
endmodule
"#,
    )
    .expect("write netlist");
    let error = load_labeled_netlist_aig_with_liberty(
        &netlist_path,
        &Library::default(),
        &GvEvalOptions {
            module_name: Some("top".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect_err("unconnected child inout ports should be rejected");
    assert_eq!(
        format!("{error:#}"),
        "module instance 'u_child' of 'child' has inout port 'io'; hierarchical gv-eval supports only input and output module ports"
    );
}

#[test]
fn labeled_netlist_aig_flattens_nested_structural_hierarchy() {
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
module leaf (a, y);
  input a;
  output y;
  BUF u_buf (.A(a), .Y(y));
endmodule

module wrapper (a, y);
  input a;
  output y;
  leaf u_leaf (.a(a), .y(y));
endmodule

module top (a, y);
  input a;
  output y;
  wrapper u_wrapper (.a(a), .y(y));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_netlist_aig(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions {
            module_name: Some("top".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect("build nested hierarchical evaluation model");

    assert_eq!(model.instances.len(), 1);
    assert_eq!(model.instances[0].instance_name, "u_wrapper/u_leaf/u_buf");
    assert_eq!(
        model
            .module_boundaries
            .iter()
            .map(|boundary| boundary.instance_path.as_str())
            .collect::<Vec<_>>(),
        vec!["u_wrapper", "u_wrapper/u_leaf"]
    );
    let output = model
        .evaluate_ir_value(&IrValue::parse_typed("(bits[1]:1)").unwrap())
        .expect("evaluate nested hierarchy");
    assert_eq!(output.to_string(), "bits[1]:1");
}

#[test]
fn labeled_sequential_netlist_aig_flattens_structural_hierarchy() {
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
module child (clk, d, q);
  input clk;
  input d;
  output q;
  DFF state (.D(d), .CLK(clk), .Q(q));
endmodule

module top (clk, d, q);
  input clk;
  input d;
  output q;
  child u_child (.clk(clk), .d(d), .q(q));
endmodule
"#;
    let (_temp_dir, netlist_path, liberty_path) = write_fixture(netlist, liberty);
    let model = load_labeled_sequential_netlist_aig(
        &netlist_path,
        &liberty_path,
        &GvEvalOptions {
            module_name: Some("top".to_string()),
            ..GvEvalOptions::default()
        },
    )
    .expect("build hierarchical sequential evaluation model");

    assert_eq!(model.instances.len(), 1);
    assert_eq!(model.instances[0].instance_name, "u_child/state");
    assert_eq!(model.module_boundaries.len(), 1);
    let boundary = &model.module_boundaries[0];
    assert_eq!(boundary.instance_path, "u_child");
    assert_eq!(boundary.module_name, "child");
    assert!(matches!(
        boundary.ports[0].bits_lsb_to_msb[0].signal,
        SequentialAigSignal::Clock
    ));

    let design = &model.sequential_gate_fn;
    let trace = sequential::simulate(
        design,
        &[
            vec![IrBits::make_ubits(1, 1).unwrap()],
            vec![IrBits::make_ubits(1, 0).unwrap()],
        ],
        SequentialState::all_zeros(design),
    )
    .expect("simulate hierarchical DFF");
    assert_eq!(
        trace
            .external_outputs()
            .iter()
            .map(|outputs| outputs[0].to_string())
            .collect::<Vec<_>>(),
        vec!["bits[1]:0", "bits[1]:1"]
    );
}

#[test]
fn labeled_netlist_aig_rejects_recursive_module_hierarchy() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("design.gv");
    std::fs::write(
        &netlist_path,
        r#"
module top (a, y);
  input a;
  output y;
  top recurse (.a(a), .y(y));
endmodule
"#,
    )
    .expect("write recursive netlist");
    let error = load_labeled_netlist_aig_with_liberty(
        &netlist_path,
        &Library::default(),
        &GvEvalOptions::default(),
    )
    .expect_err("recursive hierarchy should be rejected");
    assert!(format!("{error:#}").contains("recursive module instantiation"));
}
