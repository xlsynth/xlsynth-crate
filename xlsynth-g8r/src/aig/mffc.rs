// SPDX-License-Identifier: Apache-2.0

//! A disjoint MFFC cover of the output-reachable AND2 gates in a combinational
//! AIG.

use std::collections::{BTreeMap, BTreeSet};

use crate::aig::{AigNode, AigOperand, GateFn};

/// One maximal cone of AND2 gates with a single output and signed input pins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AigMffc {
    pub root_node_id: usize,
    pub internal_and2_count: usize,
    /// Distinct frontier signals; opposite polarities are distinct pins.
    pub input_pins: Vec<AigOperand>,
}

#[derive(Default)]
struct MffcBuilder {
    internal_and2_count: usize,
    input_pins: BTreeSet<AigOperand>,
}

/// Partitions live AND2 gates into MFFCs, including reconvergent fanout within
/// a cone when all consumers of a node belong to that same cone.
pub fn enumerate_mffc_cover(gate_fn: &GateFn) -> Vec<AigMffc> {
    let node_count = gate_fn.gates.len();
    let mut live = vec![false; node_count];
    let mut postorder = Vec::new();
    for operand in gate_fn.post_order_operands(false) {
        if !live[operand.node.id] {
            live[operand.node.id] = true;
            postorder.push(operand.node.id);
        }
    }

    let mut users = vec![Vec::<usize>::new(); node_count];
    for &user_id in &postorder {
        if let AigNode::And2 { a, b, .. } = &gate_fn.gates[user_id] {
            users[a.node.id].push(user_id);
            users[b.node.id].push(user_id);
        }
    }
    for node_users in &mut users {
        node_users.sort_unstable();
        node_users.dedup();
    }

    let mut is_primary_output = vec![false; node_count];
    for output in &gate_fn.outputs {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            is_primary_output[operand.node.id] = true;
        }
    }

    // In reverse topological order, all users of a node already have an owner.
    // A node joins its users' cone only if all users share that owner and the
    // node is not independently observable as a primary output.
    let mut owner = vec![None; node_count];
    for &node_id in postorder.iter().rev() {
        if !matches!(gate_fn.gates[node_id], AigNode::And2 { .. }) {
            continue;
        }
        let first_owner = users[node_id].first().and_then(|&user| owner[user]);
        let shares_one_owner = first_owner.is_some()
            && users[node_id]
                .iter()
                .all(|&user| owner[user] == first_owner);
        owner[node_id] = if is_primary_output[node_id] || !shares_one_owner {
            Some(node_id)
        } else {
            first_owner
        };
    }

    let mut cones = BTreeMap::<usize, MffcBuilder>::new();
    for &node_id in &postorder {
        let AigNode::And2 { a, b, .. } = &gate_fn.gates[node_id] else {
            continue;
        };
        let root = owner[node_id].expect("every live AND2 has an MFFC owner");
        let cone = cones.entry(root).or_default();
        cone.internal_and2_count += 1;
        for pin in [a, b] {
            if owner[pin.node.id] != Some(root) {
                cone.input_pins.insert(*pin);
            }
        }
    }

    cones
        .into_iter()
        .map(|(root_node_id, cone)| AigMffc {
            root_node_id,
            internal_and2_count: cone.internal_and2_count,
            input_pins: cone.input_pins.into_iter().collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::AigBitVector;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn reconvergent_fanout_stays_in_one_mffc() {
        let mut gb = GateBuilder::new("reconvergent".to_string(), GateBuilderOptions::no_opt());
        let x = gb.add_input("x".to_string(), 3);
        let shared = gb.add_and_binary(*x.get_lsb(0), *x.get_lsb(1));
        let left = gb.add_and_binary(shared, *x.get_lsb(2));
        let right = gb.add_and_binary(shared, *x.get_lsb(0));
        let root = gb.add_and_binary(left, right);
        gb.add_output("out".to_string(), AigBitVector::from_bit(root));

        let cones = enumerate_mffc_cover(&gb.build());
        assert_eq!(cones.len(), 1);
        assert_eq!(cones[0].root_node_id, root.node.id);
        assert_eq!(cones[0].internal_and2_count, 4);
        assert_eq!(cones[0].input_pins.len(), 3);
    }

    #[test]
    fn shared_and_primary_output_nodes_form_boundaries() {
        let mut gb = GateBuilder::new("boundaries".to_string(), GateBuilderOptions::no_opt());
        let x = gb.add_input("x".to_string(), 3);
        let shared = gb.add_and_binary(*x.get_lsb(0), *x.get_lsb(1));
        let a = gb.add_and_binary(shared, *x.get_lsb(2));
        let b = gb.add_and_binary(shared.negate(), *x.get_lsb(0));
        gb.add_output("a".to_string(), AigBitVector::from_bit(a));
        gb.add_output("b".to_string(), AigBitVector::from_bit(b));
        gb.add_output("shared".to_string(), AigBitVector::from_bit(shared));

        let cones = enumerate_mffc_cover(&gb.build());
        assert_eq!(
            cones.iter().map(|c| c.root_node_id).collect::<Vec<_>>(),
            vec![shared.node.id, a.node.id, b.node.id,]
        );
        assert_eq!(
            cones.iter().map(|c| c.internal_and2_count).sum::<usize>(),
            3
        );
        assert!(cones[1].input_pins.contains(&shared));
        assert!(cones[2].input_pins.contains(&shared.negate()));
    }
}
