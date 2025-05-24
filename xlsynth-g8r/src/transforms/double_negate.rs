// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Primitive: Applies double negation to an operand `op`.
/// `op.node` MUST be an And2 gate.
/// Flips `op.negated`, and flips `negated` on both children of `op.node`.
/// Returns the new value for `op`.
fn do_double_negate_on_gates(
    gates: &mut Vec<AigNode>,
    op: AigOperand,
) -> Result<AigOperand, &'static str> {
    if op.node.id >= gates.len() {
        return Err("Operand node ID out of bounds in do_double_negate");
    }
    match &mut gates[op.node.id] {
        AigNode::And2 { a, b, .. } => {
            let mut new_op = op;
            new_op.negated = !new_op.negated; // Flip negation on the operand itself
            a.negated = !a.negated; // Flip negation on child a
            b.negated = !b.negated; // Flip negation on child b
            Ok(new_op)
        }
        _ => Err("Operand node for do_double_negate must be an And2 gate"),
    }
}

#[derive(Debug)]
pub struct DoubleNegateTransform;

impl DoubleNegateTransform {
    pub fn new() -> Self {
        DoubleNegateTransform
    }
}

impl Transform for DoubleNegateTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::DoubleNegate
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection, // Self-inverse, direction doesn't change candidates
    ) -> Vec<TransformLocation> {
        let mut candidates = Vec::new();

        // 1. Primary outputs whose node is an And2
        for (out_idx, output_spec) in g.outputs.iter().enumerate() {
            for (bit_idx, output_op) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
                if output_op.node.id < g.gates.len()
                    && matches!(g.gates[output_op.node.id], AigNode::And2 { .. })
                {
                    candidates.push(TransformLocation::OutputPortBit {
                        output_idx: out_idx,
                        bit_idx,
                    });
                }
            }
        }

        // 2. Operands of And2 gates, where the operand's node is also an And2
        for (parent_idx, parent_gate) in g.gates.iter().enumerate() {
            if let AigNode::And2 {
                a: op_a, b: op_b, ..
            } = parent_gate
            {
                let parent_ref = AigRef { id: parent_idx };
                // Check operand a
                if op_a.node.id < g.gates.len()
                    && matches!(g.gates[op_a.node.id], AigNode::And2 { .. })
                {
                    candidates.push(TransformLocation::Operand(parent_ref, false));
                    // false for LHS
                }
                // Check operand b
                if op_b.node.id < g.gates.len()
                    && matches!(g.gates[op_b.node.id], AigNode::And2 { .. })
                {
                    candidates.push(TransformLocation::Operand(parent_ref, true));
                    // true for RHS
                }
            }
        }
        candidates
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        _direction: TransformDirection, // Self-inverse, direction doesn't change behavior
    ) -> Result<()> {
        match candidate_location {
            TransformLocation::OutputPortBit {
                output_idx,
                bit_idx,
            } => {
                if *output_idx >= g.outputs.len()
                    || *bit_idx >= g.outputs[*output_idx].bit_vector.get_bit_count()
                {
                    return Err(anyhow!(
                        "OutputPortBit location out of bounds: {:?}",
                        candidate_location
                    ));
                }
                let current_op_val = *g.outputs[*output_idx].bit_vector.get_lsb(*bit_idx);
                let new_op_val = do_double_negate_on_gates(&mut g.gates, current_op_val)
                    .map_err(anyhow::Error::msg)?;

                let output_bv_mut = &mut g.outputs[*output_idx].bit_vector;
                let mut ops: Vec<AigOperand> = output_bv_mut.iter_lsb_to_msb().copied().collect();
                if *bit_idx < ops.len() {
                    ops[*bit_idx] = new_op_val;
                    *output_bv_mut = crate::gate::AigBitVector::from_lsb_is_index_0(&ops);
                    Ok(())
                } else {
                    Err(anyhow!(
                        "OutputPortBit bit_idx out of bounds during splice: {:?}",
                        candidate_location
                    ))
                }
            }
            TransformLocation::Operand(parent_ref, is_rhs) => {
                if parent_ref.id >= g.gates.len() {
                    return Err(anyhow!(
                        "Operand location parent_ref out of bounds: {:?}",
                        parent_ref
                    ));
                }
                let target_op_val = match &g.gates[parent_ref.id] {
                    AigNode::And2 { a, b, .. } => {
                        if *is_rhs {
                            *b
                        } else {
                            *a
                        }
                    }
                    _ => {
                        return Err(anyhow!(
                            "Operand location parent_ref {:?} is not an And2 gate internally",
                            parent_ref
                        ))
                    }
                };

                let new_op_val = do_double_negate_on_gates(&mut g.gates, target_op_val)
                    .map_err(anyhow::Error::msg)?;

                match &mut g.gates[parent_ref.id] {
                    AigNode::And2 { a, b, .. } => {
                        if *is_rhs {
                            *b = new_op_val;
                        } else {
                            *a = new_op_val;
                        }
                        Ok(())
                    }
                    _ => Err(anyhow!(
                        "Operand location parent_ref {:?} became not And2 gate during update",
                        parent_ref
                    )),
                }
            }
            _ => Err(anyhow!(
                "Invalid location type for DoubleNegateTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{AigNode, AigOperand, AigRef};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn test_do_double_negate_primitive_direct(
        gates: &mut Vec<AigNode>,
        op: AigOperand,
    ) -> Result<AigOperand, &'static str> {
        do_double_negate_on_gates(gates, op)
    }

    #[test]
    fn test_double_negate_on_output() {
        let mut gb = GateBuilder::new("f_out".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_op.into());
        let g_original = gb.build();
        let mut g_transformed = g_original.clone();

        let mut transform = DoubleNegateTransform::new();
        let candidates = transform.find_candidates(&g_transformed, TransformDirection::Forward);

        let target_loc = candidates
            .iter()
            .find(|loc| {
                matches!(
                    loc,
                    TransformLocation::OutputPortBit {
                        output_idx: 0,
                        bit_idx: 0
                    }
                )
            })
            .expect("Candidate not found");

        transform
            .apply(&mut g_transformed, target_loc, TransformDirection::Forward)
            .unwrap();
        assert_ne!(g_original.to_string(), g_transformed.to_string());

        transform
            .apply(&mut g_transformed, target_loc, TransformDirection::Backward)
            .unwrap();
        assert_eq!(g_original.to_string(), g_transformed.to_string());
    }

    #[test]
    fn test_double_negate_on_and_operand() {
        let mut gb = GateBuilder::new("f_fanin".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let inner_and_op = gb.add_and_binary(i0, i1);
        let root_and_op = gb.add_and_binary(inner_and_op, i2);
        gb.add_output("o".to_string(), root_and_op.into());
        let g_original = gb.build();
        let mut g_transformed = g_original.clone();

        let mut transform = DoubleNegateTransform::new();
        let candidates = transform.find_candidates(&g_transformed, TransformDirection::Forward);

        let target_loc = candidates
            .iter()
            .find(
                |loc| matches!(loc, TransformLocation::Operand(r, false) if *r == root_and_op.node),
            )
            .expect("Candidate not found");

        transform
            .apply(&mut g_transformed, target_loc, TransformDirection::Forward)
            .unwrap();
        assert_ne!(g_original.to_string(), g_transformed.to_string());

        transform
            .apply(&mut g_transformed, target_loc, TransformDirection::Backward)
            .unwrap();
        assert_eq!(g_original.to_string(), g_transformed.to_string());
    }

    #[test]
    fn test_do_double_negate_primitive() {
        // Manually construct gates for primitive test
        // Nodes: 0: Literal(false), 1: Input(i0), 2: Input(i1), 3: And2(i0, i1)
        let mut gates_vec = vec![
            AigNode::Literal(false),
            AigNode::Input {
                name: "i0".to_string(),
                lsb_index: 0,
            },
            AigNode::Input {
                name: "i1".to_string(),
                lsb_index: 0,
            },
            AigNode::And2 {
                a: AigOperand {
                    node: AigRef { id: 1 },
                    negated: false,
                },
                b: AigOperand {
                    node: AigRef { id: 2 },
                    negated: false,
                },
                tags: None,
            },
        ];
        let initial_op = AigOperand {
            node: AigRef { id: 3 },
            negated: false,
        }; // Targeting the And2 gate

        // Original state: And2(i0, i1) (negated=false)
        // Children: i0 (negated=false), i1 (negated=false)
        match &gates_vec[initial_op.node.id] {
            AigNode::And2 { a, b, .. } => {
                assert!(!initial_op.negated);
                assert!(!a.negated);
                assert!(!b.negated);
            }
            _ => panic!("Expected And2 gate"),
        }

        let result_op = test_do_double_negate_primitive_direct(&mut gates_vec, initial_op).unwrap();

        // After double negation on the And2 node itself:
        // Operand to And2 is now negated: !And2(...)
        // Children of And2 are now negated: And2(!i0, !i1)
        assert!(result_op.negated, "Outer operand should be negated");
        match &gates_vec[result_op.node.id] {
            // result_op.node is same as initial_op.node
            AigNode::And2 { a, b, .. } => {
                assert!(a.negated, "Child a should be negated");
                assert!(b.negated, "Child b should be negated");
            }
            _ => panic!("Expected And2 gate after transform"),
        }

        // Apply again to reverse
        let final_op = test_do_double_negate_primitive_direct(&mut gates_vec, result_op).unwrap();
        assert!(
            !final_op.negated,
            "Outer operand should be un-negated after second application"
        );
        match &gates_vec[final_op.node.id] {
            AigNode::And2 { a, b, .. } => {
                assert!(!a.negated, "Child a should be un-negated");
                assert!(!b.negated, "Child b should be un-negated");
            }
            _ => panic!("Expected And2 gate after second transform"),
        }
    }

    #[test]
    #[should_panic(expected = "Operand node for do_double_negate must be an And2 gate")]
    fn test_do_double_negate_on_input_node_fails() {
        // Manually construct: 0: Literal(false), 1: Input(i0)
        let mut gates_vec = vec![
            AigNode::Literal(false),
            AigNode::Input {
                name: "i0".to_string(),
                lsb_index: 0,
            },
        ];
        let op_on_input = AigOperand {
            node: AigRef { id: 1 },
            negated: false,
        }; // Targeting the input node
           // This should panic because do_double_negate expects an And2 gate
        test_do_double_negate_primitive_direct(&mut gates_vec, op_on_input).unwrap();
    }

    #[test]
    #[should_panic(expected = "Operand node ID out of bounds")]
    fn test_do_double_negate_out_of_bounds() {
        let mut gates_vec = vec![AigNode::Literal(false)]; // Only literal false
        let op_out_of_bounds = AigOperand {
            node: AigRef { id: 100 },
            negated: false,
        }; // ID 100 is out of bounds
           // This should panic
        test_do_double_negate_primitive_direct(&mut gates_vec, op_out_of_bounds).unwrap();
    }
}
