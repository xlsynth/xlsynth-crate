// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use rand::{RngCore, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::{AigNode, AigOperand, GateFn};
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, parse_g8r};
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gate_random::{
    RandomControlPolarity, RandomGateBuilderMode, RandomGateOperation, RandomGateOptions,
    RandomGateShapes, generate_gate_design,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_random::{DepletableBytes, RngEntropy};

#[test]
fn seeded_profiles_are_valid_deterministic_and_bounded() {
    for seed in 0..128 {
        for mode in [
            RandomGateBuilderMode::Optimized,
            RandomGateBuilderMode::Unoptimized,
        ] {
            let options = RandomGateOptions {
                input_ports: 0..=6,
                input_width: 0..=16,
                output_ports: 0..=8,
                output_width: 0..=8,
                registers: 0..=6,
                register_width: 0..=32,
                max_aig_nodes: 128,
                builder_mode: mode,
                shapes: RandomGateShapes {
                    max_chain_length: 8,
                    max_redundant_pairs: 8,
                    max_fanout_branches: 5,
                },
                ..RandomGateOptions::default()
            };
            let a =
                generate_gate_design(&mut RngEntropy::new(StdRng::seed_from_u64(seed)), &options)
                    .unwrap();
            let b =
                generate_gate_design(&mut RngEntropy::new(StdRng::seed_from_u64(seed)), &options)
                    .unwrap();
            a.design.validate().unwrap();
            assert!(a.design.transition.gates.len() <= options.max_aig_nodes);
            assert!(a.design.inputs.len() <= 6);
            assert!(a.design.registers.len() <= 6);
            assert_eq!(a.design.clock.is_some(), !a.design.registers.is_empty());
            assert_eq!(emit_g8r(&a.design).unwrap(), emit_g8r(&b.design).unwrap());
            for (id, node) in a.design.transition.gates.iter().enumerate() {
                if let AigNode::And2 { a, b, .. } = node {
                    assert!(a.node.id < id && b.node.id < id);
                }
            }
            let text = emit_g8r(&a.design).unwrap();
            assert_eq!(emit_g8r(&parse_g8r(&text).unwrap()).unwrap(), text);
        }
    }
}

#[test]
fn truncated_entropy_never_produces_invalid_graphs() {
    let mut bytes = [0; 320];
    StdRng::seed_from_u64(31).fill_bytes(&mut bytes);
    let options = RandomGateOptions {
        registers: 1..=4,
        input_width: 0..=8,
        register_width: 0..=8,
        max_aig_nodes: 64,
        ..RandomGateOptions::default()
    };
    for length in 0..=bytes.len() {
        let generated =
            generate_gate_design(&mut DepletableBytes::new(&bytes[..length]), &options).unwrap();
        generated.design.validate().unwrap();
        assert!(generated.design.transition.gates.len() <= 64);
        assert!(!generated.design.registers.is_empty());
    }
}

#[test]
fn zero_width_empty_interfaces_and_single_node_budget_work() {
    let options = RandomGateOptions {
        input_ports: 2..=2,
        input_width: 0..=0,
        output_ports: 0..=0,
        registers: 2..=2,
        register_width: 0..=0,
        max_aig_nodes: 1,
        enable: None,
        reset: None,
        ..RandomGateOptions::default()
    };
    let generated = generate_gate_design(&mut DepletableBytes::new(&[]), &options).unwrap();
    assert_eq!(generated.design.transition.gates.len(), 1);
    assert_eq!(generated.design.registers.len(), 2);
    assert!(generated.design.outputs.is_empty());
    generated.design.validate_legalized().unwrap();
    let options = RandomGateOptions {
        input_ports: 0..=0,
        output_ports: 0..=0,
        max_aig_nodes: 1,
        ..RandomGateOptions::default()
    };
    let generated = generate_gate_design(&mut DepletableBytes::new(&[]), &options).unwrap();
    assert!(
        generated
            .design
            .try_into_gate_fn()
            .unwrap()
            .outputs
            .is_empty()
    );
}

#[test]
fn invalid_profiles_are_errors_not_generator_misses() {
    for options in [
        RandomGateOptions {
            max_aig_nodes: 0,
            ..RandomGateOptions::default()
        },
        RandomGateOptions {
            registers: 1..=1,
            register_width: 8..=8,
            max_aig_nodes: 8,
            ..RandomGateOptions::default()
        },
        RandomGateOptions {
            input_ports: 4..=2,
            ..RandomGateOptions::default()
        },
        RandomGateOptions {
            operations: vec![],
            ..RandomGateOptions::default()
        },
        RandomGateOptions {
            output_width: 0..=usize::MAX,
            ..RandomGateOptions::default()
        },
    ] {
        assert!(generate_gate_design(&mut DepletableBytes::new(&[]), &options).is_err());
    }
}

#[test]
fn control_options_and_wide_reset_values_are_respected() {
    let mut seen = BTreeSet::new();
    for polarity in [
        RandomControlPolarity::ActiveHigh,
        RandomControlPolarity::ActiveLow,
        RandomControlPolarity::Either,
    ] {
        for seed in 0..64 {
            let options = RandomGateOptions {
                registers: 2..=2,
                register_width: 129..=129,
                max_aig_nodes: 512,
                reset: Some(polarity),
                enable: Some(polarity),
                ..RandomGateOptions::default()
            };
            let design =
                generate_gate_design(&mut RngEntropy::new(StdRng::seed_from_u64(seed)), &options)
                    .unwrap()
                    .design;
            for r in &design.registers {
                if let Some(reset) = &r.reset {
                    assert_eq!(reset.value.get_bit_count(), 129);
                }
                for c in r.enable.iter().chain(r.reset.iter().map(|r| &r.control)) {
                    seen.insert(c.active_low);
                    match polarity {
                        RandomControlPolarity::ActiveHigh => assert!(!c.active_low),
                        RandomControlPolarity::ActiveLow => assert!(c.active_low),
                        RandomControlPolarity::Either => { /* Both polarities are intentionally allowed. */
                        }
                    }
                }
            }
        }
    }
    assert_eq!(seen, BTreeSet::from([false, true]));
    let options = RandomGateOptions {
        registers: 2..=2,
        reset: None,
        enable: None,
        ..RandomGateOptions::default()
    };
    let design = generate_gate_design(&mut RngEntropy::new(StdRng::seed_from_u64(9)), &options)
        .unwrap()
        .design;
    design.validate_legalized().unwrap();
}

/// Packs explicit entropy words for focused construction characterization.
fn bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[test]
fn boolean_operation_profiles_have_expected_truth_tables() {
    for op in [
        RandomGateOperation::And,
        RandomGateOperation::Or,
        RandomGateOperation::Xor,
        RandomGateOperation::Xnor,
        RandomGateOperation::Mux,
    ] {
        let options = RandomGateOptions {
            input_ports: 3..=3,
            input_width: 1..=1,
            output_ports: 1..=1,
            max_operations: 1,
            allow_constants: false,
            operations: vec![op],
            builder_mode: RandomGateBuilderMode::Unoptimized,
            ..RandomGateOptions::default()
        };
        // Header (seeds/mode/counts/widths), then operation and three operands.
        let data = bytes(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4]);
        let graph = generate_gate_design(&mut DepletableBytes::new(&data), &options)
            .unwrap()
            .design
            .try_into_gate_fn()
            .unwrap();
        let AigOperand { node, .. } = *graph.outputs[0].bit_vector.get_lsb(0);
        assert!(matches!(graph.gates[node.id], AigNode::And2 { .. }));
        let mut actual = Vec::new();
        let mut expected = Vec::new();
        for a in [false, true] {
            for b in [false, true] {
                for c in [false, true] {
                    let result = gate_sim::eval(
                        &graph,
                        &[IrBits::bool(a), IrBits::bool(b), IrBits::bool(c)],
                        Collect::None,
                    );
                    actual.push(result.outputs[0].get_bit(0).unwrap());
                    expected.push(match op {
                        RandomGateOperation::And => a & b,
                        RandomGateOperation::Or => a | b,
                        RandomGateOperation::Xor => a ^ b,
                        RandomGateOperation::Xnor => !(a ^ b),
                        RandomGateOperation::Mux => {
                            if a {
                                b
                            } else {
                                c
                            }
                        }
                    });
                }
            }
        }
        // Output wiring may complement the whole generated function.
        let complement = expected.iter().map(|value| !value).collect::<Vec<_>>();
        assert!(
            actual == expected || actual == complement,
            "{op:?}: {actual:?}"
        );
    }
}

#[test]
fn output_selection_observes_logic_and_varies_independently() {
    let options = RandomGateOptions {
        input_ports: 2..=2,
        input_width: 1..=1,
        output_ports: 2..=2,
        max_operations: 2,
        allow_constants: false,
        operations: vec![RandomGateOperation::And],
        builder_mode: RandomGateBuilderMode::Unoptimized,
        ..RandomGateOptions::default()
    };
    let mut first = BTreeSet::new();
    let mut second = BTreeSet::new();
    for seed in 0..64 {
        let data = bytes(&[seed, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3]);
        let graph = generate_gate_design(&mut DepletableBytes::new(&data), &options)
            .unwrap()
            .design
            .try_into_gate_fn()
            .unwrap();
        let a = *graph.outputs[0].bit_vector.get_lsb(0);
        assert!(matches!(graph.gates[a.node.id], AigNode::And2 { .. }));
        first.insert((a.node.id, a.negated));
        let b = *graph.outputs[1].bit_vector.get_lsb(0);
        second.insert((b.node.id, b.negated));
    }
    assert!(first.len() >= 2);
    assert!(second.len() > first.len());
}

#[test]
fn endpoint_reuse_preserves_distinct_external_ports_and_shared_register_roles() {
    for reuse in [false, true] {
        let options = RandomGateOptions {
            input_ports: 0..=0,
            output_ports: 2..=2,
            output_width: 0..=0,
            registers: 3..=3,
            register_width: 0..=0,
            max_aig_nodes: 1,
            enable: None,
            reset: None,
            reuse_output_endpoints: reuse,
            ..RandomGateOptions::default()
        };
        let design = generate_gate_design(&mut DepletableBytes::new(&[]), &options)
            .unwrap()
            .design;
        assert_eq!(design.outputs.len(), 2);
        assert_ne!(design.outputs[0], design.outputs[1]);
        assert_eq!(design.transition.outputs.len(), if reuse { 2 } else { 5 });
        if reuse {
            assert!(design.registers.iter().all(|r| r.d == design.outputs[0]));
        }
        design.validate().unwrap();
    }
}

#[test]
fn structural_stress_is_opt_in_and_bounded() {
    let base = RandomGateOptions {
        input_ports: 2..=2,
        input_width: 1..=1,
        max_operations: 0,
        builder_mode: RandomGateBuilderMode::Unoptimized,
        allow_constants: false,
        max_aig_nodes: 24,
        ..RandomGateOptions::default()
    };
    for shapes in [
        RandomGateShapes {
            max_chain_length: 8,
            ..RandomGateShapes::default()
        },
        RandomGateShapes {
            max_redundant_pairs: 8,
            ..RandomGateShapes::default()
        },
        RandomGateShapes {
            max_fanout_branches: 5,
            ..RandomGateShapes::default()
        },
    ] {
        let options = RandomGateOptions {
            shapes,
            ..base.clone()
        };
        let mut exercised = false;
        for seed in 0..32 {
            let plain =
                generate_gate_design(&mut RngEntropy::new(StdRng::seed_from_u64(seed)), &base)
                    .unwrap();
            assert_eq!(plain.design.transition.gates.len(), 3);
            let stressed =
                generate_gate_design(&mut RngEntropy::new(StdRng::seed_from_u64(seed)), &options)
                    .unwrap();
            let count = stressed.design.transition.gates.len();
            assert!(count <= 24);
            exercised |= count > 3;
        }
        assert!(exercised);
    }
}

/// Returns dependency node IDs for a set of roots without assuming storage
/// order.
fn dependency_cone(graph: &GateFn, roots: impl IntoIterator<Item = AigOperand>) -> BTreeSet<usize> {
    let mut pending = roots
        .into_iter()
        .map(|root| root.node.id)
        .collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if seen.insert(id) {
            pending.extend(graph.gates[id].operands().map(|edge| edge.node.id));
        }
    }
    seen
}

#[test]
fn rng_sweep_covers_boolean_operations_liveness_and_graph_shapes() {
    let options = RandomGateOptions::default();
    let mut entropy = RngEntropy::new(StdRng::seed_from_u64(0x91a7_c0de));
    let mut seen = BTreeSet::new();
    let mut emitted = BTreeSet::new();
    let mut live_operations = BTreeSet::new();
    let mut widths = BTreeSet::new();
    for _ in 0..2_000 {
        let generated = generate_gate_design(&mut entropy, &options).unwrap();
        let design = &generated.design;
        let graph = &design.transition;
        design.validate_legalized().unwrap();
        emitted.extend(generated.stats.emitted_operations.keys().copied());
        live_operations.extend(generated.stats.live_operations.keys().copied());
        widths.extend(graph.inputs.iter().map(|port| port.get_bit_count()));
        if design.inputs.is_empty() {
            seen.insert("no inputs");
        }
        if design.inputs.len() == *options.input_ports.end() {
            seen.insert("max inputs");
        }
        if design.outputs.len() == *options.output_ports.start() {
            seen.insert("min outputs");
        }
        if design.outputs.len() == *options.output_ports.end() {
            seen.insert("max outputs");
        }
        seen.insert(if generated.builder_options.fold {
            "optimized"
        } else {
            "unoptimized"
        });
        let roots = graph
            .outputs
            .iter()
            .flat_map(|o| o.bit_vector.iter_lsb_to_msb().copied());
        let live = dependency_cone(graph, roots);
        let mut uses = vec![0; graph.gates.len()];
        for (index, node) in graph.gates.iter().enumerate() {
            if let AigNode::And2 { a, b, .. } = node {
                uses[a.node.id] += 1;
                uses[b.node.id] += 1;
                if !live.contains(&index) {
                    seen.insert("dead logic");
                }
                if live.contains(&index) {
                    if a.negated || b.negated {
                        seen.insert("complemented fanin");
                    }
                    if a.node.id == 0 || b.node.id == 0 {
                        seen.insert("constant fanin");
                    }
                    let lhs = dependency_cone(graph, [*a]);
                    let rhs = dependency_cone(graph, [*b]);
                    if lhs.intersection(&rhs).any(|id| *id != 0) {
                        seen.insert("reconvergence");
                    }
                }
            }
        }
        if uses.iter().any(|count| *count >= 3) {
            seen.insert("high fanout");
        }
        if live
            .iter()
            .filter(|id| matches!(graph.gates[**id], AigNode::And2 { .. }))
            .count()
            >= 6
        {
            seen.insert("nontrivial live cone");
        }
        for output in &graph.outputs {
            for bit in output.bit_vector.iter_lsb_to_msb() {
                if bit.negated {
                    seen.insert("complemented output");
                }
                if bit.node.id == 0 {
                    seen.insert("constant output");
                }
            }
        }
    }
    for operation in &options.operations {
        assert!(emitted.contains(operation), "never emitted {operation:?}");
        assert!(
            live_operations.contains(operation),
            "never emitted live {operation:?}"
        );
    }
    assert_eq!(widths, BTreeSet::from([1, 2, 3, 4]));
    for category in [
        "no inputs",
        "max inputs",
        "min outputs",
        "max outputs",
        "optimized",
        "unoptimized",
        "dead logic",
        "complemented fanin",
        "constant fanin",
        "reconvergence",
        "high fanout",
        "nontrivial live cone",
        "complemented output",
        "constant output",
    ] {
        assert!(seen.contains(category), "missing {category}; saw {seen:?}");
    }
}

#[test]
fn rng_sweep_covers_state_controls_endpoints_and_packed_widths() {
    let options = RandomGateOptions {
        input_ports: 0..=4,
        input_width: 0..=8,
        output_ports: 0..=4,
        output_width: 0..=8,
        registers: 0..=4,
        register_width: 0..=16,
        ..RandomGateOptions::default()
    };
    let mut entropy = RngEntropy::new(StdRng::seed_from_u64(0x5e0_c0de));
    let mut seen = BTreeSet::new();
    let mut input_widths = BTreeSet::new();
    let mut output_widths = BTreeSet::new();
    let mut register_widths = BTreeSet::new();
    let mut combinations = BTreeSet::new();
    let mut enable_polarities = BTreeSet::new();
    let mut reset_polarities = BTreeSet::new();
    for _ in 0..2_000 {
        let design = generate_gate_design(&mut entropy, &options).unwrap().design;
        let graph = &design.transition;
        design.validate().unwrap();
        if design.registers.is_empty() {
            seen.insert("no registers");
        }
        if design.registers.len() == *options.registers.end() {
            seen.insert("max registers");
        }
        if design.inputs.is_empty() {
            seen.insert("no inputs");
        }
        if design.inputs.len() == *options.input_ports.end() {
            seen.insert("max inputs");
        }
        if design.outputs.is_empty() {
            seen.insert("no outputs");
        }
        if design.outputs.len() == *options.output_ports.end() {
            seen.insert("max outputs");
        }
        input_widths.extend(
            design
                .inputs
                .iter()
                .map(|id| graph.inputs[id.index()].get_bit_count()),
        );
        output_widths.extend(
            design
                .outputs
                .iter()
                .map(|id| graph.outputs[id.index()].get_bit_count()),
        );
        let mut roles = BTreeSet::new();
        for register in &design.registers {
            let q = &graph.inputs[register.q.index()].bit_vector;
            let d = &graph.outputs[register.d.index()].bit_vector;
            register_widths.insert(q.get_bit_count());
            combinations.insert((register.enable.is_some(), register.reset.is_some()));
            if !roles.insert(register.d) && q.get_bit_count() > 0 {
                seen.insert("shared D");
            }
            if design.outputs.contains(&register.d) && q.get_bit_count() > 0 {
                seen.insert("external D alias");
            }
            if q.get_bit_count() != 0 {
                let cone = dependency_cone(graph, d.iter_lsb_to_msb().copied());
                if q.iter_lsb_to_msb().any(|bit| cone.contains(&bit.node.id)) {
                    seen.insert("feedback");
                } else {
                    seen.insert("feedforward");
                }
            }
            if let Some(enable) = register.enable {
                enable_polarities.insert(enable.active_low);
            }
            if let Some(reset) = &register.reset {
                reset_polarities.insert(reset.control.active_low);
                if reset.value.get_bit_count() > 0 && !reset.value.is_zero() {
                    seen.insert("nonzero reset");
                }
            }
            for control in register
                .enable
                .iter()
                .chain(register.reset.iter().map(|r| &r.control))
            {
                if !roles.insert(control.signal) {
                    seen.insert("shared control endpoint");
                }
                let bit = *graph.outputs[control.signal.index()].bit_vector.get_lsb(0);
                if bit.node.id == 0 {
                    seen.insert("constant control");
                }
                if matches!(graph.gates[bit.node.id], AigNode::And2 { .. }) {
                    seen.insert("computed control");
                }
            }
        }
    }
    assert_eq!(input_widths, (0..=8).collect());
    assert_eq!(output_widths, (0..=8).collect());
    assert_eq!(register_widths, (0..=16).collect());
    assert_eq!(
        combinations,
        BTreeSet::from([(false, false), (false, true), (true, false), (true, true)])
    );
    assert_eq!(enable_polarities, BTreeSet::from([false, true]));
    assert_eq!(reset_polarities, BTreeSet::from([false, true]));
    for category in [
        "no registers",
        "max registers",
        "no inputs",
        "max inputs",
        "no outputs",
        "max outputs",
        "shared D",
        "external D alias",
        "feedback",
        "feedforward",
        "nonzero reset",
        "shared control endpoint",
        "constant control",
        "computed control",
    ] {
        assert!(seen.contains(category), "missing {category}; saw {seen:?}");
    }
}
