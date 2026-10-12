// SPDX-License-Identifier: Apache-2.0

//! Shared-generator properties for external AIG serialization formats.

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::{AigNode, SequentialGateFn};
use xlsynth_g8r::aig_serdes::blif::{emit_blif, parse_blif};
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::gate2ir::{
    GateFnInterfaceSchema, repack_gate_fn_interface_with_schema,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto;
use xlsynth_g8r::aig_serdes::load_aiger_binary::load_aiger_binary;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, eval};
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gate_random::{RandomGateOptions, generate_gate_design};
use xlsynth_g8r::test_utils::structurally_equivalent;
use xlsynth_pir::ir_random::DepletableBytes;
use xlsynth_pir::random_inputs::generate_mixed_irbits_with_rng;

use crate::direct_g8r::generate_combinational;

/// Semantic port identity, independent of reconstructed transition endpoint
/// IDs.
#[derive(Debug, PartialEq, Eq)]
struct PortShape {
    name: String,
    width: usize,
}

/// Observable interface and state ordering that BLIF must preserve.
#[derive(Debug, PartialEq, Eq)]
struct SequentialInterface {
    inputs: Vec<PortShape>,
    outputs: Vec<PortShape>,
    registers: Vec<PortShape>,
}

impl SequentialInterface {
    /// Extracts packed boundaries without requiring internal graph identity.
    fn from_design(design: &SequentialGateFn) -> Self {
        Self {
            inputs: design
                .inputs
                .iter()
                .map(|id| {
                    let port = &design.transition.inputs[id.index()];
                    PortShape {
                        name: port.name.clone(),
                        width: port.get_bit_count(),
                    }
                })
                .collect(),
            outputs: design
                .outputs
                .iter()
                .map(|id| {
                    let port = &design.transition.outputs[id.index()];
                    PortShape {
                        name: port.name.clone(),
                        width: port.get_bit_count(),
                    }
                })
                .collect(),
            registers: design
                .registers
                .iter()
                .map(|register| PortShape {
                    name: register.name.clone(),
                    width: design.transition.inputs[register.q.index()].get_bit_count(),
                })
                .collect(),
        }
    }
}

/// Coverage measured by the BLIF smoke test, never used to bias generation.
#[derive(Debug, Default)]
pub struct BlifCoverage {
    pub registers: usize,
    pub enables: usize,
    pub resets: usize,
    pub active_high_controls: usize,
    pub active_low_controls: usize,
    pub packed_ports: usize,
}

/// Round-trips combinational and legalized sequential BLIF from common state.
pub fn check_blif_roundtrip(data: &[u8]) -> BlifCoverage {
    // Plain BLIF cannot recover zero-width ports/registers. Constrain the
    // supported profile up front rather than silently skipping emit failures.
    let options = RandomGateOptions {
        input_width: 1..=8,
        output_ports: 0..=4,
        output_width: 1..=8,
        registers: 0..=6,
        register_width: 1..=16,
        ..RandomGateOptions::default()
    };
    let source = generate_gate_design(&mut DepletableBytes::new(data), &options)
        .expect("valid BLIF generation options")
        .design;
    let legalized = source.legalize_register_controls().unwrap();
    let text = emit_blif(&legalized).unwrap();
    let decoded = parse_blif(&text)
        .unwrap_or_else(|error| panic!("emitted BLIF did not parse: {error}\n{text}"));
    let redecoded = parse_blif(&emit_blif(&decoded).unwrap()).unwrap();
    let interface = SequentialInterface::from_design(&source);
    for design in [&legalized, &decoded, &redecoded] {
        design.validate_legalized().unwrap();
        assert_eq!(
            SequentialInterface::from_design(design),
            interface,
            "{text}"
        );
        assert_eq!(design.clock, source.clock, "{text}");
        assert_eq!(design.name, source.name, "{text}");
    }
    let input_widths = interface.inputs.iter().map(|p| p.width).collect::<Vec<_>>();
    let register_widths = interface
        .registers
        .iter()
        .map(|p| p.width)
        .collect::<Vec<_>>();
    let variants = [&source, &legalized, &decoded, &redecoded];
    let mut rng = StdRng::from_seed(*blake3::hash(data).as_bytes());
    for trial in 0..4 {
        let values = generate_mixed_irbits_with_rng(&mut rng, &register_widths);
        let mut states = variants
            .iter()
            .map(|design| SequentialState::from_register_values(design, values.clone()).unwrap())
            .collect::<Vec<_>>();
        for cycle in 0..12 {
            // No reset warm-up: compare cycle zero and arbitrary initial state.
            let inputs = vec![generate_mixed_irbits_with_rng(&mut rng, &input_widths)];
            let traces = variants
                .iter()
                .zip(&states)
                .map(|(design, state)| simulate(design, &inputs, state.clone()).unwrap())
                .collect::<Vec<_>>();
            for trace in traces.iter().skip(1) {
                assert_eq!(
                    trace.external_outputs(),
                    traces[0].external_outputs(),
                    "outputs: trial {trial}, cycle {cycle}\n{text}"
                );
                assert_eq!(
                    trace.final_state(),
                    traces[0].final_state(),
                    "state: trial {trial}, cycle {cycle}\n{text}"
                );
            }
            states = traces
                .iter()
                .map(|trace| trace.final_state().clone())
                .collect();
        }
    }
    let mut coverage = BlifCoverage {
        registers: source.registers.len(),
        packed_ports: interface
            .inputs
            .iter()
            .chain(&interface.outputs)
            .chain(&interface.registers)
            .filter(|p| p.width > 1)
            .count(),
        ..BlifCoverage::default()
    };
    for register in &source.registers {
        coverage.enables += usize::from(register.enable.is_some());
        coverage.resets += usize::from(register.reset.is_some());
        for control in register
            .enable
            .iter()
            .chain(register.reset.iter().map(|r| &r.control))
        {
            coverage.active_low_controls += usize::from(control.active_low);
            coverage.active_high_controls += usize::from(!control.active_low);
        }
    }
    coverage
}

/// Coverage measured by the binary-AIGER smoke test.
#[derive(Debug, Default)]
pub struct BinaryAigerCoverage {
    pub zero_width_ports: usize,
    pub packed_ports: usize,
    pub and_nodes: usize,
}

/// Checks binary AIGER with/without symbols, automatic detection and
/// regrouping.
pub fn check_binary_aiger_roundtrip(data: &[u8]) -> BinaryAigerCoverage {
    let options = RandomGateOptions {
        input_width: 0..=16,
        output_ports: 0..=4,
        output_width: 0..=16,
        ..RandomGateOptions::default()
    };
    let graph = generate_combinational(data, &options).unwrap().graph;
    let schema = GateFnInterfaceSchema::from_gate_fn(&graph).unwrap();
    let input_widths = graph
        .inputs
        .iter()
        .map(|p| p.get_bit_count())
        .collect::<Vec<_>>();
    let mut rng = StdRng::from_seed(*blake3::hash(data).as_bytes());
    for include_symbols in [false, true] {
        let bytes = emit_aiger_binary(&graph, include_symbols).unwrap();
        let raw = load_aiger_binary(&bytes, GateBuilderOptions::no_opt())
            .unwrap()
            .gate_fn;
        let detected = load_aiger_auto(&bytes, GateBuilderOptions::no_opt())
            .unwrap()
            .gate_fn;
        assert!(structurally_equivalent(&raw, &detected));
        assert!(raw.inputs.iter().all(|p| p.get_bit_count() == 1));
        assert!(raw.outputs.iter().all(|p| p.get_bit_count() == 1));
        let decoded = repack_gate_fn_interface_with_schema(raw, &schema).unwrap();
        assert_eq!(
            GateFnInterfaceSchema::from_gate_fn(&decoded).unwrap(),
            schema
        );
        assert!(structurally_equivalent(&graph, &decoded));
        assert_eq!(emit_aiger_binary(&graph, include_symbols).unwrap(), bytes);
        // AND operand ordering may change the postorder numbering after a
        // round trip. Re-emission must preserve the graph, not identical bytes.
        let reencoded = emit_aiger_binary(&decoded, include_symbols).unwrap();
        let redecoded = load_aiger_binary(&reencoded, GateBuilderOptions::no_opt()).unwrap();
        let redecoded = repack_gate_fn_interface_with_schema(redecoded.gate_fn, &schema).unwrap();
        assert!(structurally_equivalent(&graph, &redecoded));
        for _ in 0..16 {
            let inputs = generate_mixed_irbits_with_rng(&mut rng, &input_widths);
            assert_eq!(
                eval(&graph, &inputs, Collect::None).outputs,
                eval(&decoded, &inputs, Collect::None).outputs
            );
            assert_eq!(
                eval(&graph, &inputs, Collect::None).outputs,
                eval(&redecoded, &inputs, Collect::None).outputs
            );
        }
    }
    let widths = graph
        .inputs
        .iter()
        .map(|p| p.get_bit_count())
        .chain(graph.outputs.iter().map(|p| p.get_bit_count()))
        .collect::<Vec<_>>();
    BinaryAigerCoverage {
        zero_width_ports: widths.iter().filter(|width| **width == 0).count(),
        packed_ports: widths.iter().filter(|width| **width > 1).count(),
        and_nodes: graph
            .gates
            .iter()
            .filter(|node| matches!(node, AigNode::And2 { .. }))
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn blif_smoke_covers_combinational_state_and_controls() {
        let mut rng = StdRng::seed_from_u64(0xb11f);
        let mut total = BlifCoverage::default();
        let mut combinational = 0;
        for length in 0..128 {
            let mut bytes = vec![0; length * 16];
            rng.fill_bytes(&mut bytes);
            let seen = check_blif_roundtrip(&bytes);
            combinational += usize::from(seen.registers == 0);
            total.registers += seen.registers;
            total.enables += seen.enables;
            total.resets += seen.resets;
            total.active_high_controls += seen.active_high_controls;
            total.active_low_controls += seen.active_low_controls;
            total.packed_ports += seen.packed_ports;
        }
        assert!(combinational > 0 && total.registers > 0, "{total:?}");
        assert!(total.enables > 0 && total.resets > 0, "{total:?}");
        assert!(
            total.active_high_controls > 0 && total.active_low_controls > 0,
            "{total:?}"
        );
        assert!(total.packed_ports > 0, "{total:?}");
    }

    #[test]
    fn binary_aiger_smoke_covers_zero_width_and_packed_ports() {
        let mut rng = StdRng::seed_from_u64(0xa19e2);
        let mut total = BinaryAigerCoverage::default();
        for length in 0..128 {
            let mut bytes = vec![0; length * 16];
            rng.fill_bytes(&mut bytes);
            let seen = check_binary_aiger_roundtrip(&bytes);
            total.zero_width_ports += seen.zero_width_ports;
            total.packed_ports += seen.packed_ports;
            total.and_nodes += seen.and_nodes;
        }
        assert!(
            total.zero_width_ports > 0 && total.packed_ports > 0,
            "{total:?}"
        );
        assert!(total.and_nodes > 0, "{total:?}");
    }
}
