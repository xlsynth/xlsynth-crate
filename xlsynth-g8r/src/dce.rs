// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Input, Output};
use std::collections::{HashMap, HashSet};

/// Worklist-based DCE: removes unreachable nodes from a GateFn.
pub fn dce(orig_fn: &GateFn) -> GateFn {
    let mut reachable = HashSet::new();
    let mut worklist = Vec::new();
    for output in &orig_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            worklist.push(*bit);
        }
    }
    for input in &orig_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            reachable.insert(bit.node);
        }
    }

    while let Some(current) = worklist.pop() {
        if !reachable.insert(current.node) {
            continue;
        }
        let node = &orig_fn.gates[current.node.id];
        for op in node.get_operands() {
            worklist.push(op);
        }
    }
    // Rebuild the GateFn with only reachable nodes
    let mut new_gates = Vec::new();
    let mut old_to_new = HashMap::new();
    for (i, node) in orig_fn.gates.iter().enumerate() {
        let aref = AigRef { id: i };
        if reachable.contains(&aref) {
            let new_id = new_gates.len();
            let mut new_node = node.clone();
            if let AigNode::And2 { a, b, .. } = &mut new_node {
                let a_new = old_to_new[&a.node.id];
                let b_new = old_to_new[&b.node.id];
                assert!(
                    a_new < new_id,
                    "DCE remapping: 'a' operand (id {}) is not less than new node id {}",
                    a_new,
                    new_id
                );
                assert!(
                    b_new < new_id,
                    "DCE remapping: 'b' operand (id {}) is not less than new node id {}",
                    b_new,
                    new_id
                );
                a.node.id = a_new;
                b.node.id = b_new;
            }
            old_to_new.insert(i, new_id);
            new_gates.push(new_node);
        }
    }
    // Remap all AigRefs in outputs
    let mut new_outputs = Vec::new();
    for output in &orig_fn.outputs {
        let mut new_bits = Vec::new();
        for bit in output.bit_vector.iter_lsb_to_msb() {
            let new_id = old_to_new[&bit.node.id];
            new_bits.push(AigOperand {
                node: AigRef { id: new_id },
                negated: bit.negated,
            });
        }
        new_outputs.push(Output {
            name: output.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(&new_bits),
        });
    }
    // Remap all AigRefs in inputs
    let mut new_inputs = Vec::new();
    for input in &orig_fn.inputs {
        let mut new_bits = Vec::new();
        for bit in input.bit_vector.iter_lsb_to_msb() {
            let new_id = old_to_new[&bit.node.id];
            new_bits.push(AigOperand {
                node: AigRef { id: new_id },
                negated: bit.negated,
            });
        }
        new_inputs.push(Input {
            name: input.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(&new_bits),
        });
    }
    let result = GateFn {
        name: orig_fn.name.clone(),
        inputs: new_inputs,
        outputs: new_outputs,
        gates: new_gates,
    };
    #[cfg(debug_assertions)]
    {
        // Postcondition: all nodes in result.gates are reachable from outputs
        let mut reachable = std::collections::HashSet::new();
        let mut worklist = Vec::new();
        for output in &result.outputs {
            for bit in output.bit_vector.iter_lsb_to_msb() {
                worklist.push(bit.node.id);
            }
        }
        while let Some(id) = worklist.pop() {
            if !reachable.insert(id) {
                continue;
            }
            let node = &result.gates[id];
            for op in node.get_operands() {
                worklist.push(op.node.id);
            }
        }
        if reachable.len() != result.gates.len() {
            let all_ids: std::collections::HashSet<_> = (0..result.gates.len()).collect();
            let unreachable: Vec<_> = all_ids.difference(&reachable).cloned().collect();
            // Only consider non-input nodes for the assertion and debug output
            let unreachable_non_inputs: Vec<_> = unreachable
                .iter()
                .cloned()
                .filter(|&id| !matches!(result.gates[id], crate::gate::AigNode::Input { .. }))
                .collect();
            // Do some trace logging before our assertion so we can easily get some more
            // context if it ever fails.
            log::trace!(
                "[DCE debug] Unreachable non-input node IDs: {:?}",
                unreachable_non_inputs
            );
            for id in &unreachable_non_inputs {
                log::trace!("[DCE debug] Node {}: {:?}", id, result.gates[*id]);
            }
            assert_eq!(
                unreachable_non_inputs.len(),
                0,
                "DCE postcondition failed: not all non-input nodes in result.gates are reachable from outputs (unreachable: {:?})",
                unreachable_non_inputs
            );
        }
    }
    result
}
