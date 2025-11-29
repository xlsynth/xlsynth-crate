// SPDX-License-Identifier: Apache-2.0

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Input, Output};
use crate::aig::topo::topo_sort_refs;
use std::collections::{HashMap, HashSet};

/// Dead-code elimination that is robust to *any* node ordering.
///
/// Properties ("safe"):
///   • Accepts a `GateFn` whose `gates` vector may be in arbitrary order (even
///     with parents before children).
///   • Produces a new `GateFn` where:
///       – Every gate is reachable from at least one output.
///       – The `gates` list is in topological order (children precede
///         parents), so later passes relying on that invariant won't panic.
///   • Never panics on well-formed DAGs (cycles are already impossible in
///     `GateFn` by construction).
///
/// Internally it first figures out reachability, then *rebuilds* the graph in
/// topological order before remapping inputs/outputs.
pub fn dce_safe(orig_fn: &GateFn) -> GateFn {
    // 1. Mark reachable nodes starting from outputs (and always include the bits of
    //    inputs so they stay alive).
    let mut reachable = HashSet::new();
    let mut stack = Vec::new();
    for output in &orig_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            stack.push(*bit);
        }
    }
    for input in &orig_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            reachable.insert(bit.node);
        }
    }

    while let Some(current) = stack.pop() {
        if !reachable.insert(current.node) {
            continue;
        }
        for op in orig_fn.gates[current.node.id].get_operands() {
            stack.push(op);
        }
    }

    // 2. Build mapping old_id -> new_id using a topological ordering to ensure
    //    children appear before parents.
    let topo_order = topo_sort_refs(&orig_fn.gates);
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    let mut new_gates: Vec<AigNode> = Vec::with_capacity(reachable.len() + 1);

    // Always keep the original node 0 (constant FALSE) at position 0 so that
    // downstream passes that rely on this invariant (e.g. InsertTrueAnd) stay
    // correct even if the constant is otherwise unused.
    old_to_new.insert(0, 0);
    new_gates.push(orig_fn.gates[0].clone());

    for aref in topo_order {
        if !reachable.contains(&aref) || aref.id == 0 {
            continue;
        }
        let old_id = aref.id;
        let new_id = new_gates.len();
        old_to_new.insert(old_id, new_id);

        let mut new_node = orig_fn.gates[old_id].clone();
        if let AigNode::And2 { a, b, .. } = &mut new_node {
            a.node.id = *old_to_new
                .get(&a.node.id)
                .expect("operand 'a' should have been remapped earlier");
            b.node.id = *old_to_new
                .get(&b.node.id)
                .expect("operand 'b' should have been remapped earlier");
        }
        new_gates.push(new_node);
    }

    // Remap outputs
    let mut new_outputs = Vec::with_capacity(orig_fn.outputs.len());
    for output in &orig_fn.outputs {
        let mut bits = Vec::new();
        for bit in output.bit_vector.iter_lsb_to_msb() {
            let new_id = old_to_new[&bit.node.id];
            bits.push(AigOperand {
                node: AigRef { id: new_id },
                negated: bit.negated,
            });
        }
        new_outputs.push(Output {
            name: output.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(&bits),
        });
    }

    // Remap inputs
    let mut new_inputs = Vec::with_capacity(orig_fn.inputs.len());
    for input in &orig_fn.inputs {
        let mut bits = Vec::new();
        for bit in input.bit_vector.iter_lsb_to_msb() {
            let new_id = old_to_new[&bit.node.id];
            bits.push(AigOperand {
                node: AigRef { id: new_id },
                negated: bit.negated,
            });
        }
        new_inputs.push(Input {
            name: input.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(&bits),
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
                .filter(|&id| {
                    !matches!(
                        result.gates[id],
                        AigNode::Input { .. } | AigNode::Literal(_)
                    )
                })
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

/// A *fast* DCE that assumes the `gates` vector is already in topological
/// order (children first). This is essentially the original implementation
/// before we made it panic-safe.  It is kept around because it is ~15-20 %
/// faster when the precondition holds.
#[allow(unused)]
fn dce_simple(orig_fn: &GateFn) -> GateFn {
    let mut reachable = HashSet::new();
    let mut stack = Vec::new();
    for output in &orig_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            stack.push(*bit);
        }
    }
    for input in &orig_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            reachable.insert(bit.node);
        }
    }
    while let Some(current) = stack.pop() {
        if !reachable.insert(current.node) {
            continue;
        }
        for op in orig_fn.gates[current.node.id].get_operands() {
            stack.push(op);
        }
    }

    let mut new_gates = Vec::with_capacity(reachable.len());
    let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(reachable.len());

    for (old_id, node) in orig_fn.gates.iter().enumerate() {
        if !reachable.contains(&AigRef { id: old_id }) {
            continue;
        }
        let new_id = new_gates.len();
        old_to_new.insert(old_id, new_id);

        let mut new_node = node.clone();
        if let AigNode::And2 { a, b, .. } = &mut new_node {
            a.node.id = *old_to_new.get(&a.node.id).expect("topo order violated (a)");
            b.node.id = *old_to_new.get(&b.node.id).expect("topo order violated (b)");
        }
        new_gates.push(new_node);
    }

    // Remap outputs
    let mut new_outputs = Vec::with_capacity(orig_fn.outputs.len());
    for output in &orig_fn.outputs {
        let mut bits = Vec::new();
        for bit in output.bit_vector.iter_lsb_to_msb() {
            let new_id = old_to_new[&bit.node.id];
            bits.push(AigOperand {
                node: AigRef { id: new_id },
                negated: bit.negated,
            });
        }
        new_outputs.push(Output {
            name: output.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(&bits),
        });
    }

    // Remap inputs
    let mut new_inputs = Vec::with_capacity(orig_fn.inputs.len());
    for input in &orig_fn.inputs {
        let mut bits = Vec::new();
        for bit in input.bit_vector.iter_lsb_to_msb() {
            let new_id = old_to_new[&bit.node.id];
            bits.push(AigOperand {
                node: AigRef { id: new_id },
                negated: bit.negated,
            });
        }
        new_inputs.push(Input {
            name: input.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(&bits),
        });
    }

    GateFn {
        name: orig_fn.name.clone(),
        inputs: new_inputs,
        outputs: new_outputs,
        gates: new_gates,
    }
}

/// Public entry point: use the safe variant; switch to `dce_simple` if you
/// know your `GateFn`'s `gates` are already topologically ordered.
pub fn dce(orig_fn: &GateFn) -> GateFn {
    dce_safe(orig_fn)
}
