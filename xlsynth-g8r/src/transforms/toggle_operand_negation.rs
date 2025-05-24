// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Primitive that toggles the negation flag on one operand of an `And2` gate.
///
/// * `node` - reference to the `And2` gate
/// * `is_rhs` - `false` to toggle the left operand, `true` to toggle the right
///   operand
pub fn toggle_operand_negation_primitive(
    g: &mut GateFn,
    node: AigRef,
    is_rhs: bool,
) -> Result<(), &'static str> {
    if node.id >= g.gates.len() {
        return Err("toggle_operand_negation_primitive: AigRef out of bounds");
    }
    match &mut g.gates[node.id] {
        AigNode::And2 { a, b, .. } => {
            if is_rhs {
                b.negated = !b.negated;
            } else {
                a.negated = !a.negated;
            }
            Ok(())
        }
        _ => Err("toggle_operand_negation_primitive: node is not And2"),
    }
}

#[derive(Debug)]
pub struct ToggleOperandNegationTransform;

impl ToggleOperandNegationTransform {
    pub fn new() -> Self {
        ToggleOperandNegationTransform
    }
}

impl Transform for ToggleOperandNegationTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::ToggleOperandNegation
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        let mut candidates = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if matches!(node, AigNode::And2 { .. }) {
                let r = AigRef { id: idx };
                candidates.push(TransformLocation::Operand(r, false));
                candidates.push(TransformLocation::Operand(r, true));
            }
        }
        candidates
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        _direction: TransformDirection,
    ) -> Result<()> {
        match candidate_location {
            TransformLocation::Operand(node_ref, is_rhs) => {
                toggle_operand_negation_primitive(g, *node_ref, *is_rhs).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid candidate location for ToggleOperandNegationTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::AigRef;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::setup_simple_graph;

    #[test]
    fn test_toggle_operand_negation_round_trip() {
        let test = setup_simple_graph();
        let mut g = test.g.clone();
        let target_ref = test.o.node; // use root AND gate
        let mut t = ToggleOperandNegationTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        let cand = cands
            .iter()
            .find(|loc| matches!(loc, TransformLocation::Operand(r, false) if *r == target_ref))
            .expect("candidate not found");
        // Apply forward
        t.apply(&mut g, cand, TransformDirection::Forward).unwrap();
        // Ensure operand changed
        let after_first = match &g.gates[target_ref.id] {
            AigNode::And2 { a, .. } => a.negated,
            _ => panic!("not an And2"),
        };
        assert!(after_first, "operand should be negated after first toggle");
        // Apply backward
        t.apply(&mut g, cand, TransformDirection::Backward).unwrap();
        assert_eq!(g.to_string(), test.g.to_string());
    }

    #[test]
    fn test_toggle_operand_negation_invalid_location() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let t = ToggleOperandNegationTransform::new();
        let invalid = TransformLocation::Node(AigRef { id: 0 });
        assert!(t
            .apply(&mut g, &invalid, TransformDirection::Forward)
            .is_err());
    }

    #[test]
    fn test_toggle_operand_negation_primitive_non_and() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        assert!(toggle_operand_negation_primitive(&mut g, i0.node, false).is_err());
    }
}
