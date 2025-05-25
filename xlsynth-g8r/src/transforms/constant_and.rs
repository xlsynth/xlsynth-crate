// SPDX-License-Identifier: Apache-2.0

//! Fold an `And2` whose *both* operands are literals into a single literal.
//!
//! Variants handled (using FALSE = node0,TRUE = !node0):
//!   * AND(false, false)  → false
//!   * AND(false, true)   → false
//!   * AND(true,  true)   → true
//!
//! The transform rewires all fan-outs of the matched gate to the computed
//! literal; the original gate becomes dead and will be removed by later DCE.
//!
//! This operation is semantics-preserving and one-way (no inverse).

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Primitive: replace an AND of two literals at `node` with its constant
/// result.
fn remove_constant_and_primitive(g: &mut GateFn, node: AigRef) -> Result<(), &'static str> {
    if node.id >= g.gates.len() {
        return Err("remove_constant_and_primitive: AigRef oob");
    }
    let (a, b) = match g.gates[node.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("remove_constant_and_primitive: node not And2"),
    };
    // Both operands must be literals (node id == 0)
    if a.node.id != 0 || b.node.id != 0 {
        return Err("remove_constant_and_primitive: operands not both literals");
    }

    // Compute result: AND(x,y)
    let a_val = !a.negated; // node0 is FALSE; negated -> TRUE
    let b_val = !b.negated;
    let result_val = a_val & b_val;
    let replacement = AigOperand {
        node: AigRef { id: 0 },
        negated: result_val, // false => 0, true => negated to represent TRUE
    };

    // Rewire gates.
    for gate in &mut g.gates {
        if let AigNode::And2 {
            a: op_a, b: op_b, ..
        } = gate
        {
            if op_a.node == node {
                *op_a = AigOperand {
                    node: replacement.node,
                    negated: op_a.negated ^ replacement.negated,
                };
            }
            if op_b.node == node {
                *op_b = AigOperand {
                    node: replacement.node,
                    negated: op_b.negated ^ replacement.negated,
                };
            }
        }
    }
    for out in &mut g.outputs {
        for idx in 0..out.bit_vector.get_bit_count() {
            let bit = *out.bit_vector.get_lsb(idx);
            if bit.node == node {
                out.bit_vector.set_lsb(
                    idx,
                    AigOperand {
                        node: replacement.node,
                        negated: bit.negated ^ replacement.negated,
                    },
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct RemoveConstantAndTransform;

impl RemoveConstantAndTransform {
    pub fn new() -> Self {
        Self
    }
}

impl Transform for RemoveConstantAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RemoveConstantAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| match node {
                AigNode::And2 { a, b, .. } if a.node.id == 0 && b.node.id == 0 => {
                    Some(TransformLocation::Node(AigRef { id: idx }))
                }
                _ => None,
            })
            .collect()
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Backward {
            return Err(anyhow!("Backward not supported for RemoveConstantAnd"));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                remove_constant_and_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!("Invalid location type for RemoveConstantAnd")),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn setup_and_const(a_true: bool, b_true: bool) -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let lit = |val: bool| AigOperand {
            node: AigRef { id: 0 },
            negated: !val, // val=true => TRUE = !false literal
        };
        let and_op = gb.add_and_binary(lit(a_true), lit(b_true));
        gb.add_output("o".to_string(), and_op.into());
        (gb.build(), and_op.node)
    }

    #[test]
    fn test_remove_constant_and_folds_to_false() {
        let (mut g, and_ref) = setup_and_const(false, true);
        let mut t = RemoveConstantAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == and_ref));
        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        let out = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(out.node.id, 0);
        assert!(!out.negated); // false literal
    }

    #[test]
    fn test_remove_constant_and_folds_to_true() {
        let (mut g, _and_ref) = setup_and_const(true, true);
        let mut t = RemoveConstantAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        let out = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(out.node.id, 0);
        assert!(out.negated); // true literal
    }
}
