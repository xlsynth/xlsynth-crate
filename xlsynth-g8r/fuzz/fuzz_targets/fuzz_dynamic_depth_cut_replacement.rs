// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::collections::BTreeSet;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig::dynamic_depth::DynamicDepthState;
use xlsynth_g8r::aig::dynamic_structural_hash::DynamicStructuralHash;
use xlsynth_g8r::aig::{AigNode, AigOperand, AigRef};
use xlsynth_g8r_fuzz::{FuzzGraph, build_graph};

const MAX_STEPS: usize = 32;
const MAX_CUT_EXPANSIONS: usize = 8;
const MAX_CUT_LEAVES: usize = 8;
const MAX_FRAGMENT_OPS: usize = 8;

#[derive(Debug, Clone, Arbitrary)]
struct CutReplacementSample {
    graph: FuzzGraph,
    steps: Vec<CutReplacementStep>,
}

#[derive(Debug, Clone, Arbitrary)]
struct CutReplacementStep {
    root: u16,
    leaf_limit: u8,
    expand_choices: Vec<u16>,
    fragment_ops: Vec<FragmentOp>,
    output_choice: u16,
}

#[derive(Debug, Clone, Arbitrary)]
struct FragmentOp {
    lhs: u16,
    rhs: u16,
    lhs_neg: bool,
    rhs_neg: bool,
}

fn maybe_negate(op: AigOperand, negate: bool) -> AigOperand {
    if negate { op.negate() } else { op }
}

fn expand_leaf_once(
    state: &DynamicStructuralHash,
    leaves: &mut Vec<AigOperand>,
    leaf_index: usize,
) -> bool {
    let Some(leaf) = leaves.get(leaf_index).copied() else {
        return false;
    };
    if leaf.negated {
        return false;
    }
    let AigNode::And2 { a, b, .. } = state.gate_fn().gates[leaf.node.id] else {
        return false;
    };
    leaves.swap_remove(leaf_index);
    leaves.push(a);
    leaves.push(b);
    true
}

fn cleanup_dangling_new_nodes(
    state: &mut DynamicStructuralHash,
    first_new_id: usize,
    dirty_nodes: &mut BTreeSet<AigRef>,
) {
    for id in (first_new_id..state.gate_fn().gates.len()).rev() {
        let node = AigRef { id };
        dirty_nodes.insert(node);
        if state.is_live(node)
            && matches!(state.gate_fn().gates[id], AigNode::And2 { .. })
            && state.use_count(node) == 0
        {
            let _ = state.delete_node(node);
        }
    }
}

fn add_node_and_fanins(nodes: &mut BTreeSet<AigRef>, id: usize, node: &AigNode) {
    let node_ref = AigRef { id };
    nodes.insert(node_ref);
    if let AigNode::And2 { a, b, .. } = node {
        nodes.insert(a.node);
        nodes.insert(b.node);
    }
}

fn collect_changed_nodes(
    before_gates: &[AigNode],
    before_live: &[bool],
    state: &DynamicStructuralHash,
    mut dirty_nodes: BTreeSet<AigRef>,
) -> Vec<AigRef> {
    let after_gates = &state.gate_fn().gates;
    let after_live = state.live_mask();
    let node_count = before_gates.len().max(after_gates.len());
    for id in 0..node_count {
        let live_changed =
            before_live.get(id).copied().unwrap_or(false) != after_live.get(id).copied().unwrap_or(false);
        let gate_changed = before_gates.get(id) != after_gates.get(id);
        if !live_changed && !gate_changed {
            continue;
        }
        if let Some(node) = before_gates.get(id) {
            add_node_and_fanins(&mut dirty_nodes, id, node);
        }
        if let Some(node) = after_gates.get(id) {
            add_node_and_fanins(&mut dirty_nodes, id, node);
        }
    }
    for output in &state.gate_fn().outputs {
        for op in output.bit_vector.iter_lsb_to_msb() {
            dirty_nodes.insert(op.node);
        }
    }
    dirty_nodes.into_iter().collect()
}

fuzz_target!(|sample: CutReplacementSample| {
    let _ = env_logger::Builder::from_env(env_logger::Env::default())
        .is_test(true)
        .try_init();

    // Some arbitrary graph descriptions are outside the fuzz graph builder's
    // supported shape; those are generator misses, not depth-state failures.
    let Some(g) = build_graph(&sample.graph) else {
        return;
    };
    let mut hash =
        DynamicStructuralHash::new(g).expect("initial dynamic structural hash should build");
    let mut depth =
        DynamicDepthState::new(&hash).expect("initial dynamic depth state should build");
    hash.check_invariants()
        .expect("initial dynamic structural hash should be coherent");
    depth
        .check_invariants(&hash)
        .expect("initial dynamic depth state should be coherent");

    for step in sample.steps.iter().take(MAX_STEPS) {
        let and_nodes = hash.live_and_nodes();
        if and_nodes.is_empty() {
            continue;
        }
        let root = and_nodes[step.root as usize % and_nodes.len()];
        if !hash.is_live(root) {
            continue;
        }

        let mut leaves = vec![AigOperand {
            node: root,
            negated: false,
        }];
        let mut cut_nodes = BTreeSet::new();
        cut_nodes.insert(root);
        if !expand_leaf_once(&hash, &mut leaves, 0) {
            continue;
        }
        let leaf_limit = usize::from(step.leaf_limit).min(MAX_CUT_LEAVES).max(2);
        for choice in step.expand_choices.iter().take(MAX_CUT_EXPANSIONS) {
            if leaves.len() >= leaf_limit {
                break;
            }
            let expandable = leaves
                .iter()
                .enumerate()
                .filter_map(|(index, leaf)| {
                    if !leaf.negated
                        && matches!(hash.gate_fn().gates[leaf.node.id], AigNode::And2 { .. })
                    {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if expandable.is_empty() {
                break;
            }
            let leaf_index = expandable[*choice as usize % expandable.len()];
            cut_nodes.insert(leaves[leaf_index].node);
            expand_leaf_once(&hash, &mut leaves, leaf_index);
        }

        let mut dirty_nodes = BTreeSet::new();
        dirty_nodes.extend(cut_nodes);
        for fanout in hash.fanout_nodes(root) {
            dirty_nodes.insert(fanout);
        }
        for leaf in &leaves {
            dirty_nodes.insert(leaf.node);
        }

        let before_gates = hash.gate_fn().gates.clone();
        let before_live = hash.live_mask().to_vec();
        let first_new_id = hash.gate_fn().gates.len();
        let mut operands = leaves.clone();
        for op in step.fragment_ops.iter().take(MAX_FRAGMENT_OPS) {
            if operands.is_empty() {
                break;
            }
            let lhs = maybe_negate(operands[op.lhs as usize % operands.len()], op.lhs_neg);
            let rhs = maybe_negate(operands[op.rhs as usize % operands.len()], op.rhs_neg);
            let Ok(result) = hash.add_and(lhs, rhs) else {
                break;
            };
            dirty_nodes.insert(result.node);
            operands.push(result);
        }

        if operands.is_empty() {
            continue;
        }
        let replacement = operands[step.output_choice as usize % operands.len()];
        dirty_nodes.insert(replacement.node);

        if hash.replace_node_with_operand(root, replacement).is_err() {
            cleanup_dangling_new_nodes(&mut hash, first_new_id, &mut dirty_nodes);
            let dirty_nodes =
                collect_changed_nodes(&before_gates, &before_live, &hash, dirty_nodes);
            depth
                .refresh_from_changed_nodes(&hash, &dirty_nodes)
                .expect("incremental dynamic depth update should succeed after failed replacement");
            hash.check_invariants()
                .expect("failed replacement left dynamic structural hash incoherent");
            depth
                .check_invariants(&hash)
                .expect("failed replacement changed dynamic depth state");
            continue;
        }
        cleanup_dangling_new_nodes(&mut hash, first_new_id, &mut dirty_nodes);

        let dirty_nodes = collect_changed_nodes(&before_gates, &before_live, &hash, dirty_nodes);
        depth
            .refresh_from_changed_nodes(&hash, &dirty_nodes)
            .expect("incremental dynamic depth update should succeed");
        hash.check_invariants()
            .expect("cut replacement left dynamic structural hash incoherent");
        depth
            .check_invariants(&hash)
            .expect("incremental dynamic depth update diverged from full recompute");
    }
});
