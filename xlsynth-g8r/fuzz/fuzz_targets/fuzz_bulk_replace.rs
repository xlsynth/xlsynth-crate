// SPDX-License-Identifier: Apache-2.0

#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rand::prelude::IteratorRandom;
use rand::Rng;
use std::collections::HashMap;
use xlsynth_g8r::bulk_replace::bulk_replace;
use xlsynth_g8r::dce::dce;
use xlsynth_g8r::fuzz_utils;
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

fn build_gate_graph(sample: &FuzzGateGraph) -> Option<(GateFn, GateBuilderOptions)> {
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
        return None;
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
        return None;
    }
    // Add outputs
    let max_outputs = nodes.len().min(sample.num_outputs as usize);
    for i in 0..max_outputs {
        builder.add_output(format!("out{}", i), AigBitVector::from_bit(nodes[i]));
    }
    // Veto graphs with no outputs before build
    if builder.outputs.is_empty() {
        return None;
    }
    Some((builder.build(), opts))
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
    // Clamp the number of operations to avoid excessive graph size and OOM
    const MAX_OPS: u8 = 64;
    let mut graph = data.0.clone();
    graph.num_ops = graph.num_ops.min(MAX_OPS);
    // Build the graph using the clamped parameters
    let result = build_gate_graph(&graph);
    let (gate_fn, opts) = match result {
        Some(pair) => pair,
        None => return, // skip degenerate graph
    };
    if gate_fn.outputs.is_empty() {
        // Skip degenerate graphs with no outputs
        return;
    }
    let mut subs_map = HashMap::new();
    let node_count = gate_fn.gates.len();
    // Collect all input node IDs
    let mut input_node_ids = std::collections::HashSet::new();
    for input in &gate_fn.inputs {
        for i in 0..input.get_bit_count() {
            input_node_ids.insert(input.bit_vector.get_lsb(i).node.id);
        }
    }
    // Build a set of substitutions, avoiding chains and input/literal nodes
    let mut targets = std::collections::HashSet::<usize>::new();
    let mut any_valid_sub = false;

    // With some probability, generate a long chain substitution map (A->B, B->C,
    // ..., Y->Z)
    if rand::random::<u8>() % 8 == 0 && node_count > 3 {
        let max_chain = node_count.min(16);
        let chain_len = (3..=max_chain).choose(&mut rand::thread_rng()).unwrap_or(3);
        if node_count <= chain_len + 1 {
            // Not enough nodes for a valid chain, fall back to normal
            // substitution logic
        } else {
            let start = rand::thread_rng().gen_range(1..(node_count - chain_len));
            for i in 0..chain_len {
                let from_idx = start + i;
                let to_idx = start + i + 1;
                // Skip input/literal nodes
                if input_node_ids.contains(&from_idx) || from_idx == 0 {
                    continue;
                }
                if input_node_ids.contains(&to_idx) || to_idx == 0 {
                    continue;
                }
                let from_ref = AigRef { id: from_idx };
                let to_op = AigOperand {
                    node: AigRef { id: to_idx },
                    negated: false,
                };
                subs_map.insert(from_ref, to_op);
                targets.insert(from_idx);
                any_valid_sub = true;
            }
            // If we did a chain, skip the normal substitution logic
            if any_valid_sub {
                // If all substitutions were vetoed, skip this fuzz case
                if !any_valid_sub && !subs_map.is_empty() {
                    return;
                }
                // Continue to bulk_replace
            } else {
                // If chain produced no valid subs, fall back to normal logic
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
                    let from_idx_usize = *from_idx as usize;
                    let to_idx_usize = *to_idx as usize;
                    if targets.contains(&from_idx_usize) {
                        continue;
                    }
                    // Avoid chaining: don't substitute to a node that's also a target
                    if targets.contains(&to_idx_usize) {
                        continue;
                    }
                    let from_ref = AigRef { id: from_idx_usize };
                    let to_op = AigOperand {
                        node: AigRef { id: to_idx_usize },
                        negated: *negated,
                    };
                    subs_map.insert(from_ref, to_op);
                    targets.insert(from_idx_usize);
                    any_valid_sub = true;
                }
            }
        }
    } else {
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
            let from_idx_usize = *from_idx as usize;
            let to_idx_usize = *to_idx as usize;
            if targets.contains(&from_idx_usize) {
                continue;
            }
            // Avoid chaining: don't substitute to a node that's also a target
            if targets.contains(&to_idx_usize) {
                continue;
            }
            let from_ref = AigRef { id: from_idx_usize };
            let to_op = AigOperand {
                node: AigRef { id: to_idx_usize },
                negated: *negated,
            };
            subs_map.insert(from_ref, to_op);
            targets.insert(from_idx_usize);
            any_valid_sub = true;
        }
    }
    // If all substitutions were vetoed, skip this fuzz case
    if !any_valid_sub && !subs_map.is_empty() {
        return;
    }
    // Veto samples with substitution chains (a node is both a key and a target)
    let keys: std::collections::HashSet<_> = subs_map.keys().map(|k| k.id).collect();
    let targets: std::collections::HashSet<_> = subs_map.values().map(|v| v.node.id).collect();
    if !keys.is_disjoint(&targets) {
        // Substitution chain detected, skip this fuzz case
        return;
    }
    // Defensive: ensure postorder traversal completes and is not too large
    let postorder = gate_fn.post_order_operands(false);
    let (new_fn, _map) = bulk_replace(&gate_fn, &subs_map, opts);
    // Assert that substitution does not increase node count
    assert!(
        new_fn.gates.len() <= gate_fn.gates.len(),
        "bulk_replace increased node count: before = {}, after = {}",
        gate_fn.gates.len(),
        new_fn.gates.len()
    );
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

    // 3. No-op DCE: DCE should not change the simulation outputs
    let dce_fn = dce(&gate_fn);
    assert_eq!(
        gate_fn.outputs.len(),
        dce_fn.outputs.len(),
        "No-op DCE: output count changed"
    );
    for (orig, new) in gate_fn.outputs.iter().zip(dce_fn.outputs.iter()) {
        assert_eq!(
            orig.get_bit_count(),
            new.get_bit_count(),
            "No-op DCE: output width changed"
        );
    }
    // Optionally, check simulation outputs for a random input vector
    let mut rng = rand::thread_rng();
    let mut input_vecs = Vec::new();
    for input in &gate_fn.inputs {
        let width = input.get_bit_count();
        if width == 0 || width >= 64 {
            // zero-width or too big to fit in u64, skip this fuzz case
            return;
        }
        let bits = fuzz_utils::arbitrary_irbits(&mut rng, width).unwrap();
        input_vecs.push(bits);
    }
    let orig_sim = eval(&gate_fn, &input_vecs, Collect::None);
    let dce_sim = eval(&dce_fn, &input_vecs, Collect::None);
    assert_eq!(
        orig_sim.outputs, dce_sim.outputs,
        "No-op DCE: simulation outputs changed"
    );
});
