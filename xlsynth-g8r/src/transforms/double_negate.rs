// SPDX-License-Identifier: Apache-2.0

use rand::seq::SliceRandom;
use rand::Rng;

use crate::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleNegated {
    /// (output index, bit index within that output)
    Output { out_idx: usize, bit_idx: usize },
    /// (parent AND node, is_rhs = true if it's the second operand b)
    AndFanIn { node: AigRef, is_rhs: bool },
}

/// Push/pull a double-negation through an AND2 edge.
/// * `g` – gate graph to mutate
/// * `op` – the operand feeding an AND2 whose negated flag we want to toggle
/// Returns the *new* operand (with its `negated` bit flipped) if successful.
/// Fails if the operand feeds a primary input or literal.
pub fn double_negate(g: &mut GateFn, op: AigOperand) -> Result<AigOperand, &'static str> {
    match &mut g.gates[op.node.id] {
        // Cannot push negation into inputs or literals.
        AigNode::Input { .. } | AigNode::Literal(_) => {
            return Err("cannot double-negate input/literal")
        }
        AigNode::And2 { a, b, .. } => {
            // Flip the negation flag on *this* operand.
            let mut new_op = op;
            new_op.negated = !new_op.negated;

            // Also flip both child edges of the AND2 it feeds.
            a.negated = !a.negated;
            b.negated = !b.negated;
            Ok(new_op)
        }
    }
}

/// Randomly selects an operand eligible for double-negation and applies it.
/// Returns Ok(()) if a move was made, Err if no eligible operand exists.
pub fn double_negate_rand<R: Rng + ?Sized>(
    g: &mut GateFn,
    rng: &mut R,
) -> Result<DoubleNegated, &'static str> {
    // Collect (operand, location) pairs.
    let mut candidates: Vec<(AigOperand, DoubleNegated)> = Vec::new();

    // 1. Primary outputs
    for (out_idx, out) in g.outputs.iter().enumerate() {
        for (bit_idx, bit) in out.bit_vector.iter_lsb_to_msb().enumerate() {
            if matches!(g.gates[bit.node.id], AigNode::And2 { .. }) {
                candidates.push((*bit, DoubleNegated::Output { out_idx, bit_idx }));
            }
        }
    }
    // 2. Internal And2 fan-ins
    for (idx, node) in g.gates.iter().enumerate() {
        if let AigNode::And2 { a, b, .. } = node {
            if matches!(g.gates[a.node.id], AigNode::And2 { .. }) {
                candidates.push((
                    *a,
                    DoubleNegated::AndFanIn {
                        node: AigRef { id: idx },
                        is_rhs: false,
                    },
                ));
            }
            if matches!(g.gates[b.node.id], AigNode::And2 { .. }) {
                candidates.push((
                    *b,
                    DoubleNegated::AndFanIn {
                        node: AigRef { id: idx },
                        is_rhs: true,
                    },
                ));
            }
        }
    }

    if candidates.is_empty() {
        return Err("no eligible operand for double-negate");
    }

    let (chosen_op, loc) = *candidates.choose(rng).unwrap();
    let new_op = double_negate(g, chosen_op)?;

    match loc {
        DoubleNegated::Output { out_idx, bit_idx } => {
            let out = &mut g.outputs[out_idx];
            let replaced: Vec<AigOperand> = out
                .bit_vector
                .iter_lsb_to_msb()
                .enumerate()
                .map(|(i, op)| if i == bit_idx { new_op } else { *op })
                .collect();
            out.bit_vector = AigBitVector::from_lsb_is_index_0(&replaced);
        }
        DoubleNegated::AndFanIn { node, is_rhs } => {
            if let AigNode::And2 { a, b, .. } = &mut g.gates[node.id] {
                if is_rhs {
                    *b = new_op;
                } else {
                    *a = new_op;
                }
            }
        }
    }

    Ok(loc)
}

/// Re-apply a double-negation at a known location.
///
/// Layering:
/// 1. `double_negate` – primitive: flips one operand + its two fan-ins.
/// 2. `double_negate_rand` – picks a random eligible operand, calls (1),
///    patches the graph, and returns a `DoubleNegated` descriptor.
/// 3. `apply_double_negate_at_location` (this fn) – given that descriptor,
///    looks up the current operand, calls (1) again, and rewires the result.
pub fn apply_double_negate_at_location(g: &mut GateFn, loc: DoubleNegated) {
    let op = match loc {
        DoubleNegated::Output { out_idx, bit_idx } => {
            *g.outputs[out_idx].bit_vector.get_lsb(bit_idx)
        }
        DoubleNegated::AndFanIn { node, is_rhs } => {
            if let AigNode::And2 { a, b, .. } = g.gates[node.id] {
                if is_rhs {
                    b
                } else {
                    a
                }
            } else {
                panic!("location refers to non-And2 node");
            }
        }
    };
    let new_op = double_negate(g, op).expect("double_negate should succeed");
    // splice back similar to logic in rand version
    match loc {
        DoubleNegated::Output { out_idx, bit_idx } => {
            let out = &mut g.outputs[out_idx];
            let replaced: Vec<AigOperand> = out
                .bit_vector
                .iter_lsb_to_msb()
                .enumerate()
                .map(|(i, p)| if i == bit_idx { new_op } else { *p })
                .collect();
            out.bit_vector = AigBitVector::from_lsb_is_index_0(&replaced);
        }
        DoubleNegated::AndFanIn { node, is_rhs } => {
            if let AigNode::And2 { a, b, .. } = &mut g.gates[node.id] {
                if is_rhs {
                    *b = new_op;
                } else {
                    *a = new_op;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_double_negate_self_inverse() {
        // Build o = AND(i0, i1)
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let g1 = gb.build();

        // Pick that single output operand.
        let op = g1.outputs[0].bit_vector.get_lsb(0).clone();
        let mut g2 = g1.clone();
        double_negate(&mut g2, op).unwrap();
        double_negate(&mut g2, op).unwrap(); // apply twice
        assert_eq!(g1.to_string(), g2.to_string());
    }

    #[test]
    fn test_double_negate_rand_progress() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let mut g = gb.build();
        let pre = g.to_string();
        let mut rng = StdRng::seed_from_u64(123);
        let loc = double_negate_rand(&mut g, &mut rng).unwrap();
        apply_double_negate_at_location(&mut g, loc);
        let post_twice = g.to_string();
        assert_eq!(pre, post_twice);
    }
}
