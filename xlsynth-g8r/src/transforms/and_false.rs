// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Primitive: collapses an AND gate that has a constant FALSE operand.
///
/// Any gate of the form `AND(FALSE, x)` or `AND(x, FALSE)` can be replaced by
/// the constant FALSE literal (node 0, not-negated).  All fan-outs of the gate
/// are rewritten to point to the literal.
///
/// The original gate is *not* removed; it merely becomes dead and can later be
/// cleaned up by DCE.
pub fn remove_and_false_operand_primitive(
    g: &mut GateFn,
    node: AigRef,
) -> Result<(), &'static str> {
    if node.id >= g.gates.len() {
        return Err("remove_and_false_operand_primitive: AigRef out of bounds");
    }
    // Constant FALSE operand.
    let const_false = AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    };

    let (a, b) = match g.gates[node.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("remove_and_false_operand_primitive: node is not And2"),
    };

    if a != const_false && b != const_false {
        return Err("remove_and_false_operand_primitive: gate has no FALSE operand");
    }

    // Replacement operand is the literal FALSE.
    let replacement = const_false;

    // Re-wire all gate operands that reference `node`.
    for gate_idx in 0..g.gates.len() {
        if gate_idx == node.id {
            continue; // skip the gate itself
        }
        if let AigNode::And2 { a, b, .. } = &mut g.gates[gate_idx] {
            if a.node == node {
                *a = AigOperand {
                    node: replacement.node,
                    negated: a.negated ^ replacement.negated,
                };
            }
            if b.node == node {
                *b = AigOperand {
                    node: replacement.node,
                    negated: b.negated ^ replacement.negated,
                };
            }
        }
    }

    // Re-wire all outputs.
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
pub struct RemoveFalseOperandAndTransform;

impl RemoveFalseOperandAndTransform {
    pub fn new() -> Self {
        RemoveFalseOperandAndTransform
    }
}

impl Transform for RemoveFalseOperandAndTransform {
    // This transform *removes* information by replacing a whole gate with the
    // literal `FALSE`.  After the re-write all fan-outs of the original AND
    // are redirected to node-0 and the original operands are no longer
    // recoverable from the modified graph.  That means there is no
    // deterministic inverse operation we could apply to "undo" the change
    // later in an MCMC walk—once the original fan-ins are gone we cannot
    // reconstruct them without external bookkeeping.  Consequently the
    // sampler must treat this transform as *one-way* and we simply decline
    // to provide a Backward implementation (the call sites already fall back
    // to picking another move when Backward is unsupported).
    fn kind(&self) -> TransformKind {
        TransformKind::RemoveFalseOperandAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new(); // no backward direction
        }
        let const_false = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| match node {
                AigNode::And2 { a, b, .. } if *a == const_false || *b == const_false => {
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
            return Err(anyhow!(
                "Backward direction not supported for RemoveFalseOperandAndTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                remove_and_false_operand_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for RemoveFalseOperandAndTransform: {:?}",
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
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn test_remove_false_operand_and_candidates() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let const_false = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        let and_op = gb.add_and_binary(i0, const_false); // AND(i0, FALSE)
        gb.add_output("o".to_string(), and_op.into());
        let g = gb.build();

        let mut t = RemoveFalseOperandAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == and_op.node));
    }

    #[test]
    fn test_remove_false_operand_and_apply() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let const_false = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        let and_op = gb.add_and_binary(i0, const_false);
        gb.add_output("o".to_string(), and_op.into());
        let mut g = gb.build();

        let mut t = RemoveFalseOperandAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();

        // After transform, output should reference constant FALSE.
        let out_op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(out_op.node.id, 0);
        assert!(!out_op.negated);
    }

    #[test]
    fn test_not_candidate_without_false_operand() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_op.into());
        let g = gb.build();

        let mut t = RemoveFalseOperandAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert!(cands.is_empty());
    }
}
