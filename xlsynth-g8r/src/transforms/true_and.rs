// SPDX-License-Identifier: Apache-2.0

use rand::seq::SliceRandom;
use rand::Rng;

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};

/// Creates a new AND gate that ANDs the given operand with the constant TRUE.
///
/// Returns the `AigRef` of the newly created gate.
pub fn insert_true_and(g: &mut GateFn, op: AigOperand) -> AigRef {
    let true_op = AigOperand {
        node: AigRef { id: 0 },
        negated: true,
    };
    let new_gate = AigNode::And2 {
        a: op,
        b: true_op,
        tags: None,
    };
    let new_ref = AigRef { id: g.gates.len() };
    g.gates.push(new_gate);
    new_ref
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperandLoc {
    Output { out_idx: usize, bit_idx: usize },
    AndFanIn { node: AigRef, is_rhs: bool },
}

/// Randomly chooses an operand in the graph and wraps it with an AND gate
/// feeding TRUE. The chosen location is rewired to point to the new gate.
///
/// Returns the `AigRef` of the inserted gate on success.
pub fn insert_true_and_rand<R: Rng + ?Sized>(
    g: &mut GateFn,
    rng: &mut R,
) -> Result<AigRef, &'static str> {
    let mut candidates: Vec<(AigOperand, OperandLoc)> = Vec::new();

    // 1. Primary outputs
    for (out_idx, out) in g.outputs.iter().enumerate() {
        for (bit_idx, bit) in out.bit_vector.iter_lsb_to_msb().enumerate() {
            candidates.push((*bit, OperandLoc::Output { out_idx, bit_idx }));
        }
    }

    // 2. Internal And2 fan-ins
    for (idx, node) in g.gates.iter().enumerate() {
        if let AigNode::And2 { a, b, .. } = node {
            candidates.push((
                *a,
                OperandLoc::AndFanIn {
                    node: AigRef { id: idx },
                    is_rhs: false,
                },
            ));
            candidates.push((
                *b,
                OperandLoc::AndFanIn {
                    node: AigRef { id: idx },
                    is_rhs: true,
                },
            ));
        }
    }

    if candidates.is_empty() {
        return Err("insert_true_and_rand: no operands");
    }

    let (op, loc) = *candidates.choose(rng).unwrap();
    let new_ref = insert_true_and(g, op);
    let new_op = AigOperand {
        node: new_ref,
        negated: false,
    };

    match loc {
        OperandLoc::Output { out_idx, bit_idx } => {
            g.outputs[out_idx].bit_vector.set_lsb(bit_idx, new_op);
        }
        OperandLoc::AndFanIn { node, is_rhs } => {
            if let AigNode::And2 { a, b, .. } = &mut g.gates[node.id] {
                if is_rhs {
                    *b = new_op;
                } else {
                    *a = new_op;
                }
            }
        }
    }

    Ok(new_ref)
}

/// Collapses an `AND(x, true)` gate. All references to the gate are rewritten
/// to use `x` directly.
pub fn remove_true_and(g: &mut GateFn, node: AigRef) -> Result<(), &'static str> {
    let true_op = AigOperand {
        node: AigRef { id: 0 },
        negated: true,
    };
    let inner = match g.gates[node.id] {
        AigNode::And2 { a, b, .. } => {
            if b == true_op {
                a
            } else if a == true_op {
                b
            } else {
                return Err("remove_true_and: node is not AND(x,true)");
            }
        }
        _ => return Err("remove_true_and: node is not And2"),
    };

    // Rewrite fan-ins of all gates
    for gate in &mut g.gates {
        if let AigNode::And2 { a, b, .. } = gate {
            if a.node == node {
                *a = AigOperand {
                    node: inner.node,
                    negated: a.negated ^ inner.negated,
                };
            }
            if b.node == node {
                *b = AigOperand {
                    node: inner.node,
                    negated: b.negated ^ inner.negated,
                };
            }
        }
    }

    // Rewrite outputs
    for out in &mut g.outputs {
        for idx in 0..out.get_bit_count() {
            let bit = *out.bit_vector.get_lsb(idx);
            if bit.node == node {
                out.bit_vector.set_lsb(
                    idx,
                    AigOperand {
                        node: inner.node,
                        negated: bit.negated ^ inner.negated,
                    },
                );
            }
        }
    }

    Ok(())
}

/// Picks a random `AND(x, true)` gate in the graph and collapses it.
/// Returns the `AigRef` of the collapsed gate, or `None` if no such gate
/// exists.
pub fn remove_true_and_rand<R: Rng + ?Sized>(g: &mut GateFn, rng: &mut R) -> Option<AigRef> {
    let true_op = AigOperand {
        node: AigRef { id: 0 },
        negated: true,
    };
    let candidates: Vec<AigRef> = g
        .gates
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| match node {
            AigNode::And2 { a, b, .. } if *a == true_op || *b == true_op => {
                Some(AigRef { id: idx })
            }
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let chosen = *candidates.choose(rng).unwrap();
    remove_true_and(g, chosen).unwrap();
    Some(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_insert_and_remove_self_inverse() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let g1 = gb.build();

        let op = g1.outputs[0].bit_vector.get_lsb(0).clone();
        let mut g2 = g1.clone();
        let new_ref = insert_true_and(&mut g2, op);
        g2.outputs[0].bit_vector.set_lsb(
            0,
            AigOperand {
                node: new_ref,
                negated: false,
            },
        );
        remove_true_and(&mut g2, new_ref).unwrap();
        assert_eq!(g1.to_string(), g2.to_string());
    }

    #[test]
    fn test_insert_true_and_rand_round_trip() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let mut g = gb.build();
        let pre = g.to_string();
        let mut rng = StdRng::seed_from_u64(123);
        let new_ref = insert_true_and_rand(&mut g, &mut rng).unwrap();
        remove_true_and(&mut g, new_ref).unwrap();
        let post = g.to_string();
        assert_eq!(pre, post);
    }

    #[test]
    fn test_remove_true_and_rand_none() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let mut g = gb.build();
        let mut rng = StdRng::seed_from_u64(42);
        let res = remove_true_and_rand(&mut g, &mut rng);
        assert!(res.is_none());
    }
}
