// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigRef, GateFn};
use crate::topo::reaches_target as node_reaches_target;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{Result, anyhow};

/// Primitive for collapsing `((a & b) & a)` into `(a & b)`.
///
/// `outer` must be an `And2` node whose left operand is a non-negated
/// `And2` node (`inner`) with fanout 1. The right operand of `outer` must be
/// identical to the left operand of `inner`.
pub fn and_absorb_right_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left_op_of_outer, right_op_of_outer) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("and_absorb_right_primitive: outer is not And2"),
    };

    if left_op_of_outer.negated {
        return Err("and_absorb_right_primitive: left operand of outer is negated");
    }

    let inner_ref = left_op_of_outer.node;
    let (inner_a, inner_b) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("and_absorb_right_primitive: inner (left op of outer) is not And2"),
    };

    if right_op_of_outer != inner_a {
        return Err("and_absorb_right_primitive: right operand does not match inner left");
    }

    // Ensure inner has fanout 1
    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&inner_ref).unwrap_or(&0) != 1 {
        return Err("and_absorb_right_primitive: inner (left op of outer) has fanout > 1");
    }

    if node_reaches_target(&g.gates, inner_b.node, outer) {
        return Err("and_absorb_right_primitive: would create cycle");
    }

    if let AigNode::And2 { a, b, .. } = &mut g.gates[outer.id] {
        *a = inner_a;
        *b = inner_b;
    }

    crate::topo::debug_assert_no_cycles(&g.gates, "and_absorb_right_primitive");
    Ok(())
}

/// Mirror primitive collapsing `(a & (a & b))` into `(a & b)`.
///
/// `outer` must be an `And2` whose right operand is a non-negated `And2`
/// with fanout 1. The left operand of `outer` must equal the left operand of
/// the inner gate.
pub fn and_absorb_left_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left_op_of_outer, right_op_of_outer) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("and_absorb_left_primitive: outer is not And2"),
    };

    if right_op_of_outer.negated {
        return Err("and_absorb_left_primitive: right operand of outer is negated");
    }

    let inner_ref = right_op_of_outer.node;
    let (inner_a, inner_b) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("and_absorb_left_primitive: inner (right op of outer) is not And2"),
    };

    if left_op_of_outer != inner_a {
        return Err("and_absorb_left_primitive: left operand does not match inner left");
    }

    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&inner_ref).unwrap_or(&0) != 1 {
        return Err("and_absorb_left_primitive: inner (right op of outer) has fanout > 1");
    }

    if node_reaches_target(&g.gates, inner_b.node, outer) {
        return Err("and_absorb_left_primitive: would create cycle");
    }

    if let AigNode::And2 { a, b, .. } = &mut g.gates[outer.id] {
        *a = inner_a;
        *b = inner_b;
    }

    crate::topo::debug_assert_no_cycles(&g.gates, "and_absorb_left_primitive");
    Ok(())
}

#[derive(Debug)]
pub struct AndAbsorbRightTransform;

impl AndAbsorbRightTransform {
    pub fn new() -> Self {
        AndAbsorbRightTransform
    }
}

impl Transform for AndAbsorbRightTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::AndAbsorbRight
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        let use_counts = get_id_to_use_count(g);
        let mut candidates = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 {
                a: outer_left,
                b: outer_right,
                ..
            } = node
            {
                if outer_left.negated {
                    continue;
                }
                if let AigNode::And2 { a: inner_a, .. } = g.gates[outer_left.node.id] {
                    if *use_counts.get(&outer_left.node).unwrap_or(&0) == 1
                        && inner_a == *outer_right
                    {
                        candidates.push(TransformLocation::Node(AigRef { id: idx }));
                    }
                }
            }
        }
        candidates
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Backward {
            return Err(anyhow!(
                "Backward direction not supported for AndAbsorbRightTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(target_ref) => {
                and_absorb_right_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for AndAbsorbRightTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct AndAbsorbLeftTransform;

impl AndAbsorbLeftTransform {
    pub fn new() -> Self {
        AndAbsorbLeftTransform
    }
}

impl Transform for AndAbsorbLeftTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::AndAbsorbLeft
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        let use_counts = get_id_to_use_count(g);
        let mut candidates = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 {
                a: outer_left,
                b: outer_right,
                ..
            } = node
            {
                if outer_right.negated {
                    continue;
                }
                if let AigNode::And2 { a: inner_a, .. } = g.gates[outer_right.node.id] {
                    if *use_counts.get(&outer_right.node).unwrap_or(&0) == 1
                        && inner_a == *outer_left
                    {
                        candidates.push(TransformLocation::Node(AigRef { id: idx }));
                    }
                }
            }
        }
        candidates
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Backward {
            return Err(anyhow!(
                "Backward direction not supported for AndAbsorbLeftTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(target_ref) => {
                and_absorb_left_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for AndAbsorbLeftTransform: {:?}",
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

    fn setup_absorb_right_graph() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let inner = gb.add_and_binary(i0.clone(), i1.clone());
        let outer = gb.add_and_binary(inner, i0.clone());
        gb.add_output("o".to_string(), outer.into());
        (gb.build(), outer.node)
    }

    fn setup_absorb_left_graph() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let inner = gb.add_and_binary(i0.clone(), i1.clone());
        let outer = gb.add_and_binary(i0.clone(), inner);
        gb.add_output("o".to_string(), outer.into());
        (gb.build(), outer.node)
    }

    #[test]
    fn test_and_absorb_right_primitive() {
        let (mut g, outer_ref) = setup_absorb_right_graph();
        let original_inner = match g.gates[outer_ref.id] {
            AigNode::And2 { a, .. } => a.node,
            _ => panic!("Outer not And2"),
        };
        and_absorb_right_primitive(&mut g, outer_ref).unwrap();
        match &g.gates[outer_ref.id] {
            AigNode::And2 { a, b, .. } => {
                let inner_gate = match &g.gates[original_inner.id] {
                    AigNode::And2 { a, b, .. } => (*a, *b),
                    _ => panic!("Inner not And2"),
                };
                assert_eq!(*a, inner_gate.0);
                assert_eq!(*b, inner_gate.1);
            }
            _ => panic!("Outer not And2 after absorb"),
        }
    }

    #[test]
    fn test_and_absorb_left_primitive() {
        let (mut g, outer_ref) = setup_absorb_left_graph();
        let original_inner = match g.gates[outer_ref.id] {
            AigNode::And2 { b, .. } => b.node,
            _ => panic!("Outer not And2"),
        };
        and_absorb_left_primitive(&mut g, outer_ref).unwrap();
        match &g.gates[outer_ref.id] {
            AigNode::And2 { a, b, .. } => {
                let inner_gate = match &g.gates[original_inner.id] {
                    AigNode::And2 { a, b, .. } => (*a, *b),
                    _ => panic!("Inner not And2"),
                };
                assert_eq!(*a, inner_gate.0);
                assert_eq!(*b, inner_gate.1);
            }
            _ => panic!("Outer not And2 after absorb"),
        }
    }

    #[test]
    fn test_and_absorb_right_transform_finds_and_applies() {
        let (mut g, outer_ref) = setup_absorb_right_graph();
        let mut transform = AndAbsorbRightTransform::new();
        let cands = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == outer_ref));
        transform
            .apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        match &g.gates[outer_ref.id] {
            AigNode::And2 { .. } => (),
            _ => panic!("Outer not And2 after transform"),
        }
    }

    #[test]
    fn test_and_absorb_left_transform_finds_and_applies() {
        let (mut g, outer_ref) = setup_absorb_left_graph();
        let mut transform = AndAbsorbLeftTransform::new();
        let cands = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == outer_ref));
        transform
            .apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        match &g.gates[outer_ref.id] {
            AigNode::And2 { .. } => (),
            _ => panic!("Outer not And2 after transform"),
        }
    }
}
