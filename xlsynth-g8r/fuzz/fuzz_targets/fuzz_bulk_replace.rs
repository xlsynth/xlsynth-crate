// SPDX-License-Identifier: Apache-2.0

#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rand::Rng;
use std::collections::HashMap;
use xlsynth::IrBits;
use xlsynth_g8r::bulk_replace::bulk_replace;
use xlsynth_g8r::check_equivalence::validate_same_gate_fn;
use xlsynth_g8r::dce::dce;
use xlsynth_g8r::gate::{AigBitVector, AigOperand, AigRef, GateFn};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::gate_sim::{eval, Collect};

#[derive(Debug, Clone, Arbitrary)]
enum FuzzGateOp {
    And {
        lhs: u16,
        rhs: u16,
        lhs_neg: bool,
        rhs_neg: bool,
    },
    Or {
        lhs: u16,
        rhs: u16,
        lhs_neg: bool,
        rhs_neg: bool,
    },
    Xor {
        lhs: u16,
        rhs: u16,
        lhs_neg: bool,
        rhs_neg: bool,
    },
    Xnor {
        lhs: u16,
        rhs: u16,
        lhs_neg: bool,
        rhs_neg: bool,
    },
    Mux2 {
        sel: u16,
        on_true: u16,
        on_false: u16,
        sel_neg: bool,
        t_neg: bool,
        f_neg: bool,
    },
}

#[derive(Debug, Clone, Arbitrary)]
struct FuzzRedundantPair {
    a: u16,
    b: u16,
}

#[derive(Debug, Clone, Arbitrary)]
struct FuzzGateGraph {
    num_inputs: u8,
    input_width: u8,
    num_ops: u8,
    num_outputs: u8,
    ops: Vec<FuzzGateOp>,                    // various gate operations
    constants: Vec<bool>,                    // true/false constants to add
    redundant_pairs: Vec<FuzzRedundantPair>, // pairs of ANDs that should be identical
    chain_depth: u8,                         // for deep chains
    fanout_node: Option<u16>,                // node to use for high fanout
    use_opt: bool,                           // whether to use optimizing builder options
}

fn build_gate_graph(sample: &FuzzGateGraph) -> (GateFn, GateBuilderOptions) {
    let opts = if sample.use_opt {
        GateBuilderOptions::opt()
    } else {
        GateBuilderOptions::no_opt()
    };
    let mut builder = GateBuilder::new("fuzz_bulk_replace".to_string(), opts);
    let mut nodes = Vec::new();
    // Add inputs
    for i in 0..sample.num_inputs {
        let bv = builder.add_input(format!("in{}", i), sample.input_width as usize);
        for j in 0..sample.input_width {
            nodes.push(*bv.get_lsb(j as usize));
        }
    }
    // Add constants
    let mut const_nodes = Vec::new();
    for &val in &sample.constants {
        let op = if val {
            builder.get_true()
        } else {
            builder.get_false()
        };
        const_nodes.push(op);
        nodes.push(op);
    }
    // Defensive: If no nodes, skip AND and output generation
    if nodes.is_empty() {
        return (builder.build(), opts);
    }
    // Add deep AND chain if requested
    if sample.chain_depth > 0 && !nodes.is_empty() {
        let mut chain = nodes[0];
        for i in 1..(sample.chain_depth as usize).min(nodes.len()) {
            chain = builder.add_and_binary(chain, nodes[i]);
        }
        nodes.push(chain);
    }
    // Add a variety of gate operations
    for op in sample.ops.iter().take(sample.num_ops as usize) {
        if nodes.is_empty() {
            break;
        }
        let new_node = match op {
            FuzzGateOp::And {
                lhs,
                rhs,
                lhs_neg,
                rhs_neg,
            } => {
                let a = nodes
                    .get((*lhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let b = nodes
                    .get((*rhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let a = if *lhs_neg { builder.add_not(a) } else { a };
                let b = if *rhs_neg { builder.add_not(b) } else { b };
                builder.add_and_binary(a, b)
            }
            FuzzGateOp::Or {
                lhs,
                rhs,
                lhs_neg,
                rhs_neg,
            } => {
                let a = nodes
                    .get((*lhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let b = nodes
                    .get((*rhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let a = if *lhs_neg { builder.add_not(a) } else { a };
                let b = if *rhs_neg { builder.add_not(b) } else { b };
                builder.add_or_binary(a, b)
            }
            FuzzGateOp::Xor {
                lhs,
                rhs,
                lhs_neg,
                rhs_neg,
            } => {
                let a = nodes
                    .get((*lhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let b = nodes
                    .get((*rhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let a = if *lhs_neg { builder.add_not(a) } else { a };
                let b = if *rhs_neg { builder.add_not(b) } else { b };
                builder.add_xor_binary(a, b)
            }
            FuzzGateOp::Xnor {
                lhs,
                rhs,
                lhs_neg,
                rhs_neg,
            } => {
                let a = nodes
                    .get((*lhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let b = nodes
                    .get((*rhs as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let a = if *lhs_neg { builder.add_not(a) } else { a };
                let b = if *rhs_neg { builder.add_not(b) } else { b };
                builder.add_xnor(a, b)
            }
            FuzzGateOp::Mux2 {
                sel,
                on_true,
                on_false,
                sel_neg,
                t_neg,
                f_neg,
            } => {
                let s = nodes
                    .get((*sel as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let t = nodes
                    .get((*on_true as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let f = nodes
                    .get((*on_false as usize) % nodes.len())
                    .copied()
                    .unwrap_or_else(|| builder.get_false());
                let s = if *sel_neg { builder.add_not(s) } else { s };
                let t = if *t_neg { builder.add_not(t) } else { t };
                let f = if *f_neg { builder.add_not(f) } else { f };
                builder.add_mux2(s, t, f)
            }
        };
        nodes.push(new_node);
    }
    // Add redundant AND pairs
    for pair in &sample.redundant_pairs {
        if nodes.is_empty() {
            break;
        }
        let a = nodes
            .get((pair.a as usize) % nodes.len())
            .copied()
            .unwrap_or_else(|| builder.get_false());
        let b = nodes
            .get((pair.b as usize) % nodes.len())
            .copied()
            .unwrap_or_else(|| builder.get_false());
        let and1 = builder.add_and_binary(a, b);
        let and2 = builder.add_and_binary(a, b);
        nodes.push(and1);
        nodes.push(and2);
    }
    // Add high fanout ANDs if requested
    if let Some(fanout_idx) = sample.fanout_node {
        if !nodes.is_empty() {
            let fanout = nodes
                .get((fanout_idx as usize) % nodes.len())
                .copied()
                .unwrap_or_else(|| builder.get_false());
            for i in 0..5.min(nodes.len()) {
                let and = builder.add_and_binary(fanout, nodes[i]);
                nodes.push(and);
            }
        }
    }
    // Defensive: If no nodes, skip output generation
    if nodes.is_empty() {
        return (builder.build(), opts);
    }
    // Add outputs
    let max_outputs = nodes.len().min(sample.num_outputs as usize);
    for i in 0..max_outputs {
        builder.add_output(format!("out{}", i), AigBitVector::from_bit(nodes[i]));
    }
    (builder.build(), opts)
}

#[derive(Debug, Clone, Arbitrary)]
struct FuzzSubstitutions {
    // Indices into the node list to substitute, and their replacements
    subs: Vec<(u16, u16, bool)>, // (from_idx, to_idx, negated)
}

fuzz_target!(|data: (FuzzGateGraph, FuzzSubstitutions)| {
    // Ensure XLSYNTH_TOOLS is set up front for equivalence checking
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for equivalence checking in this fuzz target");
    }
    let _ = env_logger::builder().is_test(true).try_init();
    let (gate_fn, opts) = build_gate_graph(&data.0);
    let mut subs_map = HashMap::new();
    let node_count = gate_fn.gates.len();
    // Collect all input node IDs
    let mut input_node_ids = std::collections::HashSet::new();
    for input in &gate_fn.inputs {
        for i in 0..input.get_bit_count() {
            input_node_ids.insert(input.bit_vector.get_lsb(i).node.id);
        }
    }
    // Build a set of substitutions, avoiding chains and input nodes
    let mut targets = std::collections::HashSet::new();
    let mut any_valid_sub = false;
    for (from_idx, to_idx, negated) in &data.1.subs {
        if *from_idx as usize >= node_count || *to_idx as usize >= node_count {
            continue;
        }
        if input_node_ids.contains(&(*from_idx as usize)) {
            continue; // Don't substitute input nodes!
        }
        if *from_idx as usize == 0 {
            continue; // Don't substitute the constant literal node!
        }
        if from_idx == to_idx {
            continue;
        }
        if targets.contains(from_idx) {
            continue;
        }
        // Avoid chaining: don't substitute to a node that's also a target
        if targets.contains(to_idx) {
            continue;
        }
        let from_ref = AigRef {
            id: *from_idx as usize,
        };
        let to_op = AigOperand {
            node: AigRef {
                id: *to_idx as usize,
            },
            negated: *negated,
        };
        subs_map.insert(from_ref, to_op);
        targets.insert(from_idx);
        any_valid_sub = true;
    }
    // If all substitutions were vetoed, skip this fuzz case
    if !any_valid_sub && !subs_map.is_empty() {
        return;
    }
    // Run bulk_replace
    let (new_fn, _map) = bulk_replace(&gate_fn, &subs_map, opts);
    // Check invariants and for cycles
    new_fn.check_invariants_with_debug_assert();
    let _dce_fn = dce(&new_fn);
    // If we get here, the graph is valid and passes invariants.

    // 1. Structural checks: output count and widths
    assert_eq!(
        gate_fn.outputs.len(),
        new_fn.outputs.len(),
        "Output count changed after bulk_replace"
    );
    for (orig, new) in gate_fn.outputs.iter().zip(new_fn.outputs.iter()) {
        assert_eq!(
            orig.get_bit_count(),
            new.get_bit_count(),
            "Output width changed for {}",
            orig.name
        );
    }

    // 2. Idempotence: running bulk_replace again should yield the same result
    //    (structurally)
    let (new_fn2, _map2) = bulk_replace(&new_fn, &subs_map, opts);
    assert_eq!(
        new_fn.outputs.len(),
        new_fn2.outputs.len(),
        "Idempotence: output count changed"
    );
    for (a, b) in new_fn.outputs.iter().zip(new_fn2.outputs.iter()) {
        assert_eq!(
            a.get_bit_count(),
            b.get_bit_count(),
            "Idempotence: output width changed"
        );
    }

    // 3. No-op substitution: empty map should be a no-op (modulo DCE renumbering)
    let (noop_fn, _noop_map) = bulk_replace(&gate_fn, &HashMap::new(), opts);
    assert_eq!(
        gate_fn.outputs.len(),
        noop_fn.outputs.len(),
        "No-op: output count changed"
    );
    for (orig, new) in gate_fn.outputs.iter().zip(noop_fn.outputs.iter()) {
        assert_eq!(
            orig.get_bit_count(),
            new.get_bit_count(),
            "No-op: output width changed"
        );
    }

    // 4. Randomized simulation and equivalence check for small graphs
    if gate_fn.gates.len() < 32 {
        let mut rng = rand::thread_rng();
        let mut input_vecs = Vec::new();
        for input in &gate_fn.inputs {
            let width = input.get_bit_count();
            let value = rng.gen_range(0..(1u64 << width));
            input_vecs.push(IrBits::make_ubits(width, value).unwrap());
        }
        let orig_sim = eval(&gate_fn, &input_vecs, Collect::None);
        let new_sim = eval(&new_fn, &input_vecs, Collect::None);
        assert_eq!(
            orig_sim.outputs, new_sim.outputs,
            "Simulation outputs differ after bulk_replace"
        );
        // Equivalence check
        validate_same_gate_fn(&gate_fn, &new_fn)
            .expect("GateFns should be equivalent after bulk_replace");
    }
});
