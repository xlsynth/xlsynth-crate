// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::gate;

/// Computes a use count for the nodes in the gate function.
///
/// This indicates both which nodes are unused and indicates which nodes have
/// high fanout.
pub fn get_id_to_use_count(gate_fn: &gate::GateFn) -> HashMap<gate::AigRef, usize> {
    let mut id_to_use_count: HashMap<gate::AigRef, usize> = HashMap::new();
    let mut bump_use_count = |node: gate::AigRef| {
        *id_to_use_count.entry(node).or_insert(0) += 1;
    };
    let mut processed_nodes = HashSet::new();

    // The worklist represents the set of nodes for which we observed that they were
    // used but we haven't yet traversed to their arguments. If a node is in
    // "processed" that means we have already traversed to its arguments so we
    // should not do it again.
    let mut worklist = Vec::new();

    // Start from outputs - each output use counts as 1
    for output in gate_fn.outputs.iter() {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            bump_use_count(operand.node);
            worklist.push(operand.node);
        }
    }

    // Process the worklist - each time a node is used as an argument, increment its
    // use count
    while let Some(node) = worklist.pop() {
        // Skip if we've already processed this node's arguments
        let first_time_seen = processed_nodes.insert(node);
        if !first_time_seen {
            continue;
        }

        // Get the node's arguments and process them
        let gate: &gate::AigNode = &gate_fn.gates[node.id];
        for arg in gate.get_args() {
            bump_use_count(arg);
            worklist.push(arg);
        }
    }

    id_to_use_count
}
