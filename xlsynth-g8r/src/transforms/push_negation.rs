// SPDX-License-Identifier: Apache-2.0

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use crate::aig::topo;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{Result, anyhow};

/// Primitive: complements the negation on both operands of `node` and all
/// fanouts of `node`.
///
/// Effectively toggles a negation through the AND gate, similar to applying
/// DeMorgan's law. The operation is its own inverse.
pub fn push_negation_primitive(g: &mut GateFn, node: AigRef) -> Result<(), &'static str> {
    if node.id >= g.gates.len() {
        return Err("push_negation_primitive: AigRef out of bounds");
    }
    match &mut g.gates[node.id] {
        AigNode::And2 { a, b, .. } => {
            a.negated = !a.negated;
            b.negated = !b.negated;
        }
        _ => return Err("push_negation_primitive: node is not And2"),
    }

    // Update all fanouts of the node by toggling their negation bit.
    for gate in &mut g.gates {
        if let AigNode::And2 { a, b, .. } = gate {
            if a.node == node {
                a.negated = !a.negated;
            }
            if b.node == node {
                b.negated = !b.negated;
            }
        }
    }
    for output in &mut g.outputs {
        let mut ops: Vec<AigOperand> = output.bit_vector.iter_lsb_to_msb().copied().collect();
        let mut changed = false;
        for op in &mut ops {
            if op.node == node {
                op.negated = !op.negated;
                changed = true;
            }
        }
        if changed {
            output.bit_vector = AigBitVector::from_lsb_is_index_0(&ops);
        }
    }

    topo::debug_assert_no_cycles(&g.gates, "push_negation_primitive");
    Ok(())
}

#[derive(Debug)]
pub struct PushNegationTransform;

impl PushNegationTransform {
    pub fn new() -> Self {
        PushNegationTransform
    }
}

impl Transform for PushNegationTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::PushNegation
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        g.gates
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node, AigNode::And2 { .. }))
            .map(|(idx, _)| TransformLocation::Node(AigRef { id: idx }))
            .collect()
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        _direction: TransformDirection,
    ) -> Result<()> {
        match candidate_location {
            TransformLocation::Node(target_ref) => {
                push_negation_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location type for PushNegationTransform: {:?}",
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
    use crate::aig::gate::AigRef;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn test_push_negation_round_trip() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_op.into());
        let g_original = gb.build();
        let mut g_transformed = g_original.clone();

        let mut t = PushNegationTransform::new();
        let cands = t.find_candidates(&g_transformed, TransformDirection::Forward);
        let cand = cands
            .iter()
            .find(|loc| matches!(loc, TransformLocation::Node(r) if *r == and_op.node))
            .expect("candidate not found");
        t.apply(&mut g_transformed, cand, TransformDirection::Forward)
            .unwrap();
        assert_ne!(g_transformed.to_string(), g_original.to_string());
        t.apply(&mut g_transformed, cand, TransformDirection::Backward)
            .unwrap();
        assert_eq!(g_transformed.to_string(), g_original.to_string());
    }

    #[test]
    fn test_push_negation_invalid_location() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let t = PushNegationTransform::new();
        let invalid = TransformLocation::Operand(AigRef { id: 0 }, false);
        assert!(
            t.apply(&mut g, &invalid, TransformDirection::Forward)
                .is_err()
        );
    }
}
