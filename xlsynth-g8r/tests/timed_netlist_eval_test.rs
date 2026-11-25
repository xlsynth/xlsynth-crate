// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, TimedLabeledNetlist, load_timed_netlist_with_liberty,
};
use xlsynth_g8r::netlist::io::load_liberty_from_path;
use xlsynth_g8r::netlist::power::timed::{
    TimedInputEvent, TimedNetlistEvaluator, TimedRun, TimedSignal,
};
use xlsynth_pir::IrValue;

const XOR_NETLIST: &str = r#"
module top(input a, input b, output y);
  XOR2 sum (.A(a), .B(b), .Y(y));
endmodule
"#;

const RECONVERGENT_NETLIST: &str = r#"
module top(input a, output y);
  wire n1, n2;
  XOR2 stage1 (.A(a), .B(1'b0), .Y(n1));
  XOR2 stage2 (.A(n1), .B(1'b0), .Y(n2));
  XOR2 sum (.A(a), .B(n2), .Y(y));
endmodule
"#;

/// A mapped test circuit and the Liberty characterization used to simulate it.
struct TestNetlist {
    model: TimedLabeledNetlist,
    library: Library,
}

impl TestNetlist {
    /// All test circuits drive a 3 pF external load.
    fn evaluator(&self) -> Result<TimedNetlistEvaluator<'_>, String> {
        TimedNetlistEvaluator::new(&self.model, &self.library, 3.0, Default::default())
    }
}

/// The XOR has distinct input capacitances and a slew-dependent input power
/// group on A.
fn xor_liberty() -> String {
    let arc = r#"
    timing_arcs: {
      related_pin_string_id: PIN
      timing_sense: TIMING_SENSE_NON_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 3 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.75 }
    }
    internal_power: {
      related_pin_string_ids: PIN
      tables: { transition: POWER_TRANSITION_RISE shape_id: 1 values: 10 }
      tables: { transition: POWER_TRANSITION_FALL shape_id: 1 values: 20 }
    }
"#;
    r#"
format_magic: 5496997758228255311
units: { time_unit: "ns" capacitance_unit: "pf" voltage_unit: "V" }
nominal_voltage: 2
cells: {
  name: "XOR2"
  pins: {
    name_string_id: 1 direction: INPUT capacitance: 1
    internal_power: {
      when_string_id: 2
      tables: { transition: POWER_TRANSITION_RISE shape_id: 2 values: [6, 10] }
      tables: { transition: POWER_TRANSITION_FALL shape_id: 2 values: [8, 12] }
    }
  }
  pins: { name_string_id: 2 direction: INPUT capacitance: 2 }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4
    ARC_A
    ARC_B
  }
}
lu_table_templates: {
  kind: LUT_TEMPLATE_KIND_POWER name: "input_slew"
  variable_1: LUT_VARIABLE_INPUT_TRANSITION_TIME index_1_id: 1
}
lut_axes: { values: [0.25, 1.25] }
lut_shapes: {}
lut_shapes: { template_id: 1 dimensions: 2 }
interned_strings: ["A", "B", "Y", "A ^ B"]
"#
    .replace("ARC_A", &arc.replace("PIN", "1"))
    .replace("ARC_B", &arc.replace("PIN", "2"))
}

/// Loads a flat test circuit with the supplied Liberty characterization.
fn fixture_with_liberty(netlist: &str, liberty: &str) -> anyhow::Result<TestNetlist> {
    let dir = tempfile::tempdir()?;
    let lib_path = dir.path().join("xor.textproto");
    let gv_path = dir.path().join("design.gv");
    std::fs::write(&lib_path, liberty)?;
    std::fs::write(&gv_path, netlist)?;
    let library = load_liberty_from_path(&lib_path)?;
    let model = load_timed_netlist_with_liberty(&gv_path, &library, &GvEvalOptions::default())?;
    Ok(TestNetlist { model, library })
}

fn fixture(netlist: &str) -> anyhow::Result<TestNetlist> {
    fixture_with_liberty(netlist, &xor_liberty())
}

fn low_xor_inputs() -> IrValue {
    IrValue::make_tuple(&[IrValue::bool(false), IrValue::bool(false)])
}

fn event(port: &str, time: f64, value: bool) -> TimedInputEvent {
    TimedInputEvent {
        port: port.to_owned(),
        bit: 0,
        time,
        value,
        slew: 0.25,
    }
}

/// Select the observed crossings of a physical cell output from the full event
/// stream.
fn output_edges(run: &TimedRun, instance_name: &str) -> Vec<(f64, bool, f64, f64)> {
    let signal = TimedSignal::CellOutput {
        instance: instance_name.to_owned(),
        pin: "Y".to_owned(),
    };
    run.events
        .iter()
        .filter(|edge| edge.signal == signal)
        .map(|edge| (edge.time, edge.value, edge.slew, edge.switching_energy))
        .collect()
}

/// Preserve every field and the order of the cell-internal energy records.
fn internal_edges(run: &TimedRun) -> Vec<(&str, &str, f64, bool, f64)> {
    run.internal_events
        .iter()
        .map(|edge| {
            (
                edge.instance.as_str(),
                edge.pin.as_str(),
                edge.time,
                edge.value,
                edge.energy,
            )
        })
        .collect()
}

#[test]
fn reconvergent_paths_emit_a_pulse_on_a_settled_constant_output() {
    let circuit = fixture(RECONVERGENT_NETLIST).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = IrValue::make_tuple(&[IrValue::bool(false)]);
    let final_input = IrValue::make_tuple(&[IrValue::bool(true)]);
    let run = simulator
        .simulate(&initial, &[event("a", 1.0, true)], 10.0)
        .unwrap();

    assert_eq!(output_edges(&run, "stage1"), [(3.0, true, 0.5, 2.0)]);
    assert_eq!(output_edges(&run, "stage2"), [(5.0, true, 0.5, 4.0)]);
    assert_eq!(
        output_edges(&run, "sum"),
        [(3.0, true, 0.5, 6.0), (8.0, false, 0.75, 6.0)]
    );
    let settled = circuit.model.settled_model();
    for inputs in [&initial, &final_input] {
        assert_eq!(
            settled.evaluate_ir_value(inputs).unwrap(),
            IrValue::bool(false)
        );
    }

    // The primary input drives two 1 pF pins. Three cell rises and one fall
    // contribute 3 * 10 + 20 pJ of internal energy.
    assert_eq!(run.events.len(), 5);
    let first = &run.events[0];
    assert!(matches!(&first.signal, TimedSignal::Input { port, bit: 0 } if port == "a"));
    assert_eq!(
        (first.time, first.value, first.slew, first.switching_energy),
        (1.0, true, 0.25, 4.0)
    );
    assert_eq!(run.switching_energy, 22.0);
    assert_eq!(run.internal_energy, 50.0);
    assert_eq!(run.total_dynamic_energy, 72.0);
    assert_eq!(run.average_dynamic_power, 7.2);
    assert_eq!(run.energy_unit, "V^2*pf");
    assert_eq!(run.power_unit, "V^2*pf/ns");
}

#[test]
fn simultaneous_pin_changes_are_applied_as_one_xor_input_vector() {
    let circuit = fixture(XOR_NETLIST).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = low_xor_inputs();
    for (first, second) in [("a", "b"), ("b", "a")] {
        let inputs = [event(first, 1.0, true), event(second, 1.0, true)];
        let run = simulator.simulate(&initial, &inputs, 10.0).unwrap();
        assert!(output_edges(&run, "sum").is_empty());
        assert_eq!(run.events.len(), 2);
        assert_eq!(run.switching_energy, 6.0);
        assert_eq!(run.internal_energy, 0.0);
    }
}

#[test]
fn simultaneous_changes_sensitize_timing_and_power_on_the_same_assignment() {
    let netlist = r#"
module top(input a, input b, output y);
  AND2 product (.A(a), .B(b), .Y(y));
endmodule
"#;
    let conditioned_arc = |pin, when, delay, energy| {
        format!(
            r#"
    timing_arcs: {{
      related_pin_string_id: {pin} when_string_id: {when}
      timing_sense: TIMING_SENSE_POSITIVE_UNATE
      tables: {{ kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: {delay} }}
      tables: {{ kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.5 }}
    }}
    internal_power: {{
      related_pin_string_ids: {pin} when_string_id: {when}
      tables: {{ transition: POWER_TRANSITION_RISE shape_id: 1 values: {energy} }}
    }}
"#
        )
    };
    let liberty = r#"
format_magic: 5496997758228255311
units: { time_unit: "ns" capacitance_unit: "pf" voltage_unit: "V" }
nominal_voltage: 2
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1 }
  pins: { name_string_id: 2 direction: INPUT capacitance: 1 }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4
    A_WHEN_B
    A_WHEN_NOT_B
    B_WHEN_A
    internal_power: {
      when_string_id: 2
      tables: { transition: POWER_TRANSITION_RISE shape_id: 1 values: 24 }
    }
  }
}
lut_shapes: {}
interned_strings: ["A", "B", "Y", "A & B", "!B"]
"#
    .replace("A_WHEN_B", &conditioned_arc(1, 2, 2, 10))
    .replace("A_WHEN_NOT_B", &conditioned_arc(1, 5, 1, 90))
    .replace("B_WHEN_A", &conditioned_arc(2, 1, 5, 14));
    let circuit = fixture_with_liberty(netlist, &liberty).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = low_xor_inputs();
    for (first, second) in [("a", "b"), ("b", "a")] {
        let inputs = [event(first, 1.0, true), event(second, 1.0, true)];
        let run = simulator.simulate(&initial, &inputs, 10.0).unwrap();
        // A can cause a rise only after B has risen, excluding A's faster
        // !B arc and its 90 pJ power group. B can cause the rise after A.
        assert_eq!(output_edges(&run, "product"), [(3.0, true, 0.5, 6.0)]);
        assert_eq!(internal_edges(&run), [("product", "Y", 3.0, true, 16.0)]);
    }
}

#[test]
fn coincident_opposite_crossings_reach_the_sink_in_order() {
    let netlist = r#"
module top(input a, output y);
  wire pulse;
  XOR2 driver (.A(a), .B(1'b0), .Y(pulse));
  XOR2 sink (.A(pulse), .B(1'b1), .Y(y));
endmodule
"#;
    let liberty = xor_liberty().replace(
        "kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 3",
        "kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 0",
    );
    let circuit = fixture_with_liberty(netlist, &liberty).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = IrValue::make_tuple(&[IrValue::bool(false)]);
    let run = simulator
        .simulate(
            &initial,
            &[event("a", 1.0, true), event("a", 3.0, false)],
            10.0,
        )
        .unwrap();
    assert_eq!(
        output_edges(&run, "driver"),
        [(3.0, true, 0.5, 2.0), (3.0, false, 0.75, 2.0)]
    );
    assert_eq!(
        output_edges(&run, "sink"),
        [(3.0, false, 0.75, 6.0), (5.0, true, 0.5, 6.0)]
    );
    assert_eq!(
        internal_edges(&run),
        [
            ("driver", "Y", 3.0, true, 10.0),
            ("driver", "Y", 3.0, false, 20.0),
            ("sink", "A", 3.0, true, 7.0),
            ("sink", "A", 3.0, false, 10.0),
            ("sink", "Y", 3.0, false, 20.0),
            ("sink", "Y", 5.0, true, 10.0),
        ]
    );
}

#[test]
fn output_crossings_are_inertial_and_the_observation_window_is_half_open() {
    let circuit = fixture(XOR_NETLIST).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = low_xor_inputs();
    let short = [event("a", 1.0, true), event("a", 2.0, false)];
    let filtered = simulator.simulate(&initial, &short, 10.0).unwrap();
    assert!(output_edges(&filtered, "sum").is_empty());
    assert_eq!(filtered.internal_energy, 0.0);
    assert_eq!(filtered.switching_energy, 4.0);

    // At 3 ns the rising output crosses just as the input reverses.
    let boundary = [event("a", 1.0, true), event("a", 3.0, false)];
    let clipped = simulator.simulate(&initial, &boundary, 6.0).unwrap();
    assert_eq!(output_edges(&clipped, "sum"), [(3.0, true, 0.5, 6.0)]);
    assert_eq!(clipped.total_dynamic_energy, 20.0);
    assert_eq!(clipped.average_dynamic_power, 20.0 / 6.0);
    let full = simulator.simulate(&initial, &boundary, 8.0).unwrap();
    assert_eq!(
        output_edges(&full, "sum"),
        [(3.0, true, 0.5, 6.0), (6.0, false, 0.75, 6.0)]
    );
    assert_eq!(full.total_dynamic_energy, 46.0);
    assert_eq!(full.average_dynamic_power, 46.0 / 8.0);
}

#[test]
fn a_filtered_input_pulse_consumes_conditioned_internal_energy_at_its_actual_slews() {
    let netlist = r#"
module top(input [5:4] a, output y);
  XOR2 sum (.A(a[5]), .B(a[4]), .Y(y));
endmodule
"#;
    let circuit = fixture(netlist).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = IrValue::parse_typed("(bits[2]:1)").unwrap();
    let inputs = [
        // B is already high in the settled initial state and this emits no
        // edge.
        TimedInputEvent {
            bit: 4,
            ..event("a", 0.0, true)
        },
        TimedInputEvent {
            bit: 5,
            slew: 0.5,
            ..event("a", 1.0, true)
        },
        TimedInputEvent {
            bit: 5,
            slew: 1.0,
            ..event("a", 2.0, false)
        },
        TimedInputEvent {
            bit: 4,
            ..event("a", 5.0, false)
        },
    ];
    let run = simulator.simulate(&initial, &inputs, 10.0).unwrap();
    assert_eq!(run.events.len(), 4);
    assert_eq!(output_edges(&run, "sum"), [(8.0, false, 0.75, 6.0)]);
    assert_eq!(
        internal_edges(&run),
        [
            ("sum", "A", 1.0, true, 7.0),
            ("sum", "A", 2.0, false, 11.0),
            ("sum", "Y", 8.0, false, 20.0),
        ]
    );
    assert_eq!(run.switching_energy, 14.0);
    assert_eq!(run.internal_energy, 38.0);
    assert_eq!(run.total_dynamic_energy, 52.0);
    assert_eq!(run.average_dynamic_power, 5.2);
}

#[test]
fn invalid_event_coordinates_and_observation_windows_are_rejected() {
    let circuit = fixture(XOR_NETLIST).unwrap();
    let simulator = circuit.evaluator().unwrap();
    let initial = low_xor_inputs();
    for duration in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(simulator.simulate(&initial, &[], duration).is_err());
    }
    for invalid in [
        event("a", -1.0, true),
        event("a", f64::NAN, true),
        event("a", 10.0, true),
        event("a", 11.0, true),
        event("missing", 1.0, true),
        event("y", 1.0, true),
        TimedInputEvent {
            bit: 1,
            ..event("a", 1.0, true)
        },
        TimedInputEvent {
            slew: -1.0,
            ..event("a", 1.0, true)
        },
    ] {
        assert!(simulator.simulate(&initial, &[invalid], 10.0).is_err());
    }
    let wrong_arity = IrValue::make_tuple(&[IrValue::bool(false)]);
    assert!(simulator.simulate(&wrong_arity, &[], 10.0).is_err());
}

#[test]
fn a_stable_input_owner_charges_rise_and_fall_energy_from_its_related_pin() {
    let mut circuit = fixture(XOR_NETLIST).unwrap();
    // Place a group characterized on A under pin B. Its two table entries
    // distinguish the direction of the trigger even though B stays low.
    let pins = &mut circuit.library.cells[0].pins;
    let related_a = pins[2].internal_power[0].clone();
    pins[1].internal_power.push(related_a);
    let simulator = circuit.evaluator().unwrap();
    let initial = low_xor_inputs();
    let inputs = [event("a", 1.0, true), event("a", 2.0, false)];
    let run = simulator.simulate(&initial, &inputs, 10.0).unwrap();

    assert!(output_edges(&run, "sum").is_empty());
    assert_eq!(run.events.len(), 2);
    assert_eq!(
        internal_edges(&run),
        [
            ("sum", "B", 1.0, false, 10.0),
            ("sum", "B", 2.0, false, 20.0)
        ]
    );
    assert_eq!(run.switching_energy, 4.0);
    assert_eq!(run.internal_energy, 30.0);
    assert_eq!(run.total_dynamic_energy, 34.0);
}
