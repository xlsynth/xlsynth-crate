// SPDX-License-Identifier: Apache-2.0

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig::dynamic_depth::DynamicDepthState;
use xlsynth_g8r::aig::dynamic_structural_hash::{DynamicStructuralHash, EdgeRef};
use xlsynth_g8r::aig::{AigNode, AigOperand, AigRef};
use xlsynth_g8r_fuzz::{FuzzGraph, build_graph};

const MAX_STEPS: usize = 128;

#[derive(Debug, Clone, Arbitrary)]
struct EditSample {
    graph: FuzzGraph,
    ops: Vec<EditOp>,
}

#[derive(Debug, Clone, Arbitrary)]
enum EditOp {
    AddAnd {
        lhs: u16,
        rhs: u16,
        lhs_neg: bool,
        rhs_neg: bool,
    },
    MoveFanin {
        node: u16,
        input_index: bool,
        new_operand: u16,
        new_neg: bool,
    },
    MoveOutput {
        output: u16,
        bit: u16,
        new_operand: u16,
        new_neg: bool,
    },
    ReplaceNode {
        node: u16,
        replacement: u16,
        replacement_neg: bool,
    },
    DeleteNode {
        node: u16,
    },
}

fn indexed_operand(nodes: &[AigRef], index: u16, negated: bool) -> Option<AigOperand> {
    if nodes.is_empty() {
        return None;
    }
    Some(AigOperand {
        node: nodes[index as usize % nodes.len()],
        negated,
    })
}

fn operand_depends_on_node(
    hash: &DynamicStructuralHash,
    operand: AigOperand,
    target: AigRef,
) -> bool {
    let g = hash.gate_fn();
    if operand.node.id >= g.gates.len() || target.id >= g.gates.len() {
        return false;
    }

    let mut visited = vec![false; g.gates.len()];
    let mut stack = vec![operand.node];
    while let Some(node) = stack.pop() {
        if node == target {
            return true;
        }
        if node.id >= g.gates.len() || visited[node.id] || !hash.is_live(node) {
            continue;
        }
        visited[node.id] = true;
        if let AigNode::And2 { a, b, .. } = g.gates[node.id] {
            stack.push(a.node);
            stack.push(b.node);
        }
    }
    false
}

fuzz_target!(|sample: EditSample| {
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
    let mut state =
        DynamicDepthState::new(&hash).expect("initial dynamic depth state should build");
    hash.check_invariants()
        .expect("initial dynamic structural hash should be coherent");
    state
        .check_invariants(&hash)
        .expect("initial dynamic depth state should be coherent");

    for op in sample.ops.iter().take(MAX_STEPS) {
        let live_nodes = hash.live_nodes();
        match *op {
            EditOp::AddAnd {
                lhs,
                rhs,
                lhs_neg,
                rhs_neg,
            } => {
                let Some(lhs) = indexed_operand(&live_nodes, lhs, lhs_neg) else {
                    continue;
                };
                let Some(rhs) = indexed_operand(&live_nodes, rhs, rhs_neg) else {
                    continue;
                };
                hash.add_and(lhs, rhs).expect("live operands should add");
            }
            EditOp::MoveFanin {
                node,
                input_index,
                new_operand,
                new_neg,
            } => {
                let and_nodes = hash.live_and_nodes();
                let Some(new_operand) = indexed_operand(&live_nodes, new_operand, new_neg) else {
                    continue;
                };
                if and_nodes.is_empty() {
                    continue;
                }
                let node = and_nodes[node as usize % and_nodes.len()];
                // The dynamic hash API intentionally does not pay for cycle
                // checks on every edit; production cut replacement constructs
                // acyclic edits by construction. This generic fuzzer can pick
                // arbitrary descendants as new fanins, so skip those samples.
                if operand_depends_on_node(&hash, new_operand, node) {
                    continue;
                }
                hash.move_edge(
                    EdgeRef::AndFanin {
                        node,
                        input_index: usize::from(input_index),
                    },
                    new_operand,
                )
                .expect("cycle-free live fanin edit should move");
            }
            EditOp::MoveOutput {
                output,
                bit,
                new_operand,
                new_neg,
            } => {
                let Some(new_operand) = indexed_operand(&live_nodes, new_operand, new_neg) else {
                    continue;
                };
                let outputs = &hash.gate_fn().outputs;
                if outputs.is_empty() {
                    continue;
                }
                let output_index = output as usize % outputs.len();
                let width = outputs[output_index].get_bit_count();
                if width == 0 {
                    continue;
                }
                hash.move_edge(
                    EdgeRef::OutputBit {
                        output_index,
                        bit_index: bit as usize % width,
                    },
                    new_operand,
                )
                .expect("checked output edge should move");
            }
            EditOp::ReplaceNode {
                node,
                replacement,
                replacement_neg,
            } => {
                let and_nodes = hash.live_and_nodes();
                let Some(replacement) =
                    indexed_operand(&live_nodes, replacement, replacement_neg)
                else {
                    continue;
                };
                if and_nodes.is_empty() {
                    continue;
                }
                let node = and_nodes[node as usize % and_nodes.len()];
                if hash.use_count(node) == 0 {
                    continue;
                }
                // Replacing a node with one of its descendants would turn the
                // descendant's old reference to the node into a cycle.
                if operand_depends_on_node(&hash, replacement, node) {
                    continue;
                }
                hash.replace_node_with_operand(node, replacement)
                    .expect("production-like live replacement should apply");
            }
            EditOp::DeleteNode { node } => {
                let and_nodes = hash.live_and_nodes();
                if and_nodes.is_empty() {
                    continue;
                }
                let node = and_nodes[node as usize % and_nodes.len()];
                if hash.use_count(node) != 0 {
                    continue;
                }
                hash.delete_node(node)
                    .expect("unused live AND should delete");
            }
        }

        state
            .refresh_depths(&hash)
            .expect("dynamic depth state should rebuild after edit");
        hash.check_invariants()
            .expect("dynamic structural hash became incoherent after edit");
        state
            .check_invariants(&hash)
            .expect("dynamic depth state became incoherent after edit");
    }
});
