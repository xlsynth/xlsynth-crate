// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Constant FALSE operand (node 0, not negated).
#[inline]
fn const_false() -> AigOperand {
    AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    }
}

/// Constant TRUE operand (node 0, negated=true).
#[inline]
fn const_true() -> AigOperand {
    AigOperand {
        node: AigRef { id: 0 },
        negated: true,
    }
}

/// Primitive: if `outer` is an `And2` of the form
///   `op   &  ( !op  &  y )`  or  `(!op) & ( op  & y)`
/// collapses it to constant false by rewriting all uses of `outer` to the
/// literal‐false node (ID 0).
pub fn complement_absorb_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    if outer.id >= g.gates.len() {
        return Err("outer ref out of bounds");
    }
    // Extract outer operands.
    let (lhs, rhs) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("outer is not And2"),
    };

    // Helper to test pattern where main operand is `main` and inner is `inner_ref`.
    fn matches_pattern(g: &GateFn, main: AigOperand, inner_ref: AigRef) -> bool {
        if inner_ref.id >= g.gates.len() {
            return false;
        }
        match g.gates[inner_ref.id] {
            AigNode::And2 { a: ia, b: ib, .. } => {
                (ia.node == main.node && ia.negated != main.negated)
                    || (ib.node == main.node && ib.negated != main.negated)
            }
            _ => false,
        }
    }

    let pattern_ok = if matches!(g.gates[rhs.node.id], AigNode::And2 { .. }) {
        matches_pattern(g, lhs, rhs.node)
    } else if matches!(g.gates[lhs.node.id], AigNode::And2 { .. }) {
        matches_pattern(g, rhs, lhs.node)
    } else {
        false
    };

    if !pattern_ok {
        return Err("pattern not matched");
    }

    // Rewrite fan-ins of every gate.
    for node in &mut g.gates {
        if let AigNode::And2 { a, b, .. } = node {
            if a.node == outer {
                *a = if a.negated {
                    const_true()
                } else {
                    const_false()
                };
            }
            if b.node == outer {
                *b = if b.negated {
                    const_true()
                } else {
                    const_false()
                };
            }
        }
    }

    // Rewrite primary outputs.
    for output in &mut g.outputs {
        for idx in 0..output.bit_vector.get_bit_count() {
            let op = *output.bit_vector.get_lsb(idx);
            if op.node == outer {
                let replacement = if op.negated {
                    const_true()
                } else {
                    const_false()
                };
                output.bit_vector.set_lsb(idx, replacement);
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
pub struct ComplementAbsorbTransform;

impl ComplementAbsorbTransform {
    pub fn new() -> Self {
        ComplementAbsorbTransform
    }
}

impl Transform for ComplementAbsorbTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::ComplementAbsorb
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        let mut cands = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { a: lhs, b: rhs, .. } = node {
                // Check lhs as main, rhs as inner candidate.
                let outer_ref = AigRef { id: idx };
                if matches!(g.gates[rhs.node.id], AigNode::And2 { .. }) {
                    if pattern_matches(g, *lhs, rhs.node) {
                        cands.push(TransformLocation::Node(outer_ref));
                        continue;
                    }
                }
                if matches!(g.gates[lhs.node.id], AigNode::And2 { .. }) {
                    if pattern_matches(g, *rhs, lhs.node) {
                        cands.push(TransformLocation::Node(outer_ref));
                        continue;
                    }
                }
            }
        }
        cands
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Backward {
            return Err(anyhow!(
                "Backward direction not supported for ComplementAbsorbTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                complement_absorb_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for ComplementAbsorbTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        false // until proven otherwise
    }
}

/// Helper used in find_candidates – no mutation.
fn pattern_matches(g: &GateFn, main: AigOperand, inner_ref: AigRef) -> bool {
    if inner_ref.id >= g.gates.len() {
        return false;
    }
    match g.gates[inner_ref.id] {
        AigNode::And2 { a: ia, b: ib, .. } => {
            (ia.node == main.node && ia.negated != main.negated)
                || (ib.node == main.node && ib.negated != main.negated)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    // Build a graph containing pattern a & (!a & b)
    fn build_pattern_graph() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1).get_lsb(0).clone();
        let b = gb.add_input("b".to_string(), 1).get_lsb(0).clone();
        let not_a = a.negate();
        let inner = gb.add_and_binary(not_a, b);
        let outer = gb.add_and_binary(a, inner);
        gb.add_output("o".to_string(), outer.into());
        (gb.build(), outer.node)
    }

    #[test]
    fn test_complement_absorb_primitive() {
        let (mut g, outer_ref) = build_pattern_graph();
        complement_absorb_primitive(&mut g, outer_ref).unwrap();
        // Output should now be constant false.
        let op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(op.node.id, 0);
        assert!(!op.negated);
    }

    #[test]
    fn test_transform_find_and_apply() {
        let (mut g, outer_ref) = build_pattern_graph();
        let mut t = ComplementAbsorbTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == outer_ref));
        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        let op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(op.node.id, 0);
        assert!(!op.negated);
    }
}
