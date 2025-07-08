// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::topo::reaches_target as node_reaches_target;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{Result, anyhow};

// --- Primitives ---

/// Rotates an AND tree to the right: `((a & b) & c) -> (a & (b & c))`.
///
/// Requires that the left operand of `outer` is a non-negated `And2` node
/// used only by `outer` itself. Returns `Ok(())` on success.
pub fn rotate_and_right_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left_op_of_outer, right_op_of_outer) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("rotate_and_right_primitive: outer is not And2"),
    };

    if left_op_of_outer.negated {
        return Err("rotate_and_right_primitive: left operand of outer is negated");
    }

    let inner_ref = left_op_of_outer.node; // This is the ((a & b)) part, which becomes the new (b & c)
    let (a_op, b_op) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b), // a_op is 'a', b_op is 'b' from ((a & b) & c)
        _ => return Err("rotate_and_right_primitive: inner (left op of outer) is not And2"),
    };

    // Ensure that `right_op_of_outer` does NOT (transitively) depend on
    // `inner_ref`.
    if node_reaches_target(&g.gates, right_op_of_outer.node, inner_ref) {
        return Err(
            "rotate_and_right_primitive: right operand depends on inner; rotation would create a cycle",
        );
    }

    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&inner_ref).unwrap_or(&0) != 1 {
        return Err("rotate_and_right_primitive: inner (left op of outer) has fanout > 1");
    }

    // Modify inner_ref (was (a&b)) to become (b&c) where c is right_op_of_outer
    if let AigNode::And2 {
        a: inner_lhs,
        b: inner_rhs,
        ..
    } = &mut g.gates[inner_ref.id]
    {
        *inner_lhs = b_op; // b
        *inner_rhs = right_op_of_outer; // c
    }
    // Modify outer_ref (was ((a&b)&c)) to become (a & (new_inner_ref))
    if let AigNode::And2 {
        a: outer_lhs,
        b: outer_rhs,
        ..
    } = &mut g.gates[outer.id]
    {
        *outer_lhs = a_op; // a
        *outer_rhs = AigOperand {
            node: inner_ref, // now (b&c)
            negated: false,  // inner_ref itself is not negated as an operand
        };
    }

    // Sanity-check: ensure we did not introduce cycles.
    crate::topo::debug_assert_no_cycles(&g.gates, "rotate_and_right_primitive");
    Ok(())
}

/// Inverse of [`rotate_and_right_primitive`]: transforms `(a & (b & c)) -> ((a
/// & b) & c)`. The right operand of `outer` must be a non-negated `And2` node
/// with fanout 1.
pub fn rotate_and_left_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left_op_of_outer, right_op_of_outer) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("rotate_and_left_primitive: outer is not And2"),
    };

    if right_op_of_outer.negated {
        return Err("rotate_and_left_primitive: right operand of outer is negated");
    }

    let inner_ref = right_op_of_outer.node; // This is the (b & c) part, which becomes the new (a & b)
    let (b_op, c_op) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b), // b_op is 'b', c_op is 'c' from (a & (b & c))
        _ => return Err("rotate_and_left_primitive: inner (right op of outer) is not And2"),
    };

    // Ensure that `left_op_of_outer` does NOT (transitively) depend on `inner_ref`.
    if node_reaches_target(&g.gates, left_op_of_outer.node, inner_ref) {
        return Err(
            "rotate_and_left_primitive: left operand depends on inner; rotation would create a cycle",
        );
    }

    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&inner_ref).unwrap_or(&0) != 1 {
        return Err("rotate_and_left_primitive: inner (right op of outer) has fanout > 1");
    }

    // Modify inner_ref (was (b&c)) to become (a&b) where a is left_op_of_outer
    if let AigNode::And2 {
        a: inner_lhs,
        b: inner_rhs,
        ..
    } = &mut g.gates[inner_ref.id]
    {
        *inner_lhs = left_op_of_outer; // a
        *inner_rhs = b_op; // b
    }
    // Modify outer_ref (was (a & (b&c))) to become ((new_inner_ref) & c)
    if let AigNode::And2 {
        a: outer_lhs,
        b: outer_rhs,
        ..
    } = &mut g.gates[outer.id]
    {
        *outer_lhs = AigOperand {
            node: inner_ref, // now (a&b)
            negated: false,  // inner_ref itself is not negated as an operand
        };
        *outer_rhs = c_op; // c
    }

    // Sanity-check that we didn't create a cycle.
    crate::topo::debug_assert_no_cycles(&g.gates, "rotate_and_left_primitive");
    Ok(())
}

// --- RotateAndRightTransform ---
#[derive(Debug)]
pub struct RotateAndRightTransform;

impl RotateAndRightTransform {
    pub fn new() -> Self {
        RotateAndRightTransform
    }
}

impl Transform for RotateAndRightTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RotateAndRight
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            // This could delegate to RotateAndLeftTransform.find_candidates
            return Vec::new();
        }
        let use_counts = get_id_to_use_count(g);
        let mut candidates = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 {
                a: outer_left_op, ..
            } = node
            {
                if outer_left_op.negated {
                    continue;
                }
                // Check if the left operand is an AND gate and has fanout 1
                if let AigNode::And2 { .. } = g.gates[outer_left_op.node.id] {
                    if *use_counts.get(&outer_left_op.node).unwrap_or(&0) == 1 {
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
            // This could call rotate_and_left_primitive
            return Err(anyhow!(
                "Backward direction not supported directly by RotateAndRightTransform, use RotateAndLeftTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(target_ref) => {
                rotate_and_right_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location type for RotateAndRightTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

// --- RotateAndLeftTransform ---
#[derive(Debug)]
pub struct RotateAndLeftTransform;

impl RotateAndLeftTransform {
    pub fn new() -> Self {
        RotateAndLeftTransform
    }
}

impl Transform for RotateAndLeftTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RotateAndLeft
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            // This could delegate to RotateAndRightTransform.find_candidates
            return Vec::new();
        }
        let use_counts = get_id_to_use_count(g);
        let mut candidates = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 {
                b: outer_right_op, ..
            } = node
            {
                if outer_right_op.negated {
                    continue;
                }
                // Check if the right operand is an AND gate and has fanout 1
                if let AigNode::And2 { .. } = g.gates[outer_right_op.node.id] {
                    if *use_counts.get(&outer_right_op.node).unwrap_or(&0) == 1 {
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
            // This could call rotate_and_right_primitive
            return Err(anyhow!(
                "Backward direction not supported directly by RotateAndLeftTransform, use RotateAndRightTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(target_ref) => {
                rotate_and_left_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location type for RotateAndLeftTransform: {:?}",
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
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn setup_test_graph_for_rotate_right() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let inner_and = gb.add_and_binary(i0, i1); // (i0 & i1)
        let outer_and = gb.add_and_binary(inner_and, i2); // ((i0 & i1) & i2)
        gb.add_output("o".to_string(), outer_and.into());
        (gb.build(), outer_and.node)
    }

    fn setup_test_graph_for_rotate_left() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let inner_and = gb.add_and_binary(i1, i2); // (i1 & i2)
        let outer_and = gb.add_and_binary(i0, inner_and); // (i0 & (i1 & i2))
        gb.add_output("o".to_string(), outer_and.into());
        (gb.build(), outer_and.node)
    }

    #[test]
    fn test_rotate_and_primitives_round_trip() {
        let (mut g, outer_ref) = setup_test_graph_for_rotate_right();
        let original_g_string = g.to_string();

        rotate_and_right_primitive(&mut g, outer_ref).unwrap();
        assert_ne!(
            g.to_string(),
            original_g_string,
            "Graph should change after rotate_right"
        );

        // After rotate_right, outer_ref is now the top of (i0 & (i1 & i2))
        rotate_and_left_primitive(&mut g, outer_ref).unwrap();
        assert_eq!(
            g.to_string(),
            original_g_string,
            "Graph should be original after rotate_left"
        );
    }

    #[test]
    fn test_rotate_and_right_transform_finds_and_applies() {
        let (mut g, outer_ref) = setup_test_graph_for_rotate_right();
        let original_g_string = g.to_string();

        let mut transform = RotateAndRightTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0], TransformLocation::Node(r) if r.id == outer_ref.id));

        transform
            .apply(&mut g, &candidates[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original_g_string);
        // Further checks could verify the structure: i0 & (i1 & i2)
    }

    #[test]
    fn test_rotate_and_left_transform_finds_and_applies() {
        let (mut g, outer_ref) = setup_test_graph_for_rotate_left();
        let original_g_string = g.to_string();

        let mut transform = RotateAndLeftTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0], TransformLocation::Node(r) if r.id == outer_ref.id));

        transform
            .apply(&mut g, &candidates[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original_g_string);
        // Further checks could verify the structure: ((i0 & i1) & i2)
    }

    #[test]
    fn test_rotate_and_right_primitive_fanout_fail() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let i3 = gb.add_input("i3".to_string(), 1).get_lsb(0).clone();
        let inner_and = gb.add_and_binary(i0, i1); // (i0 & i1)
        let outer_and1 = gb.add_and_binary(inner_and, i2); // ((i0 & i1) & i2)
        let outer_and2 = gb.add_and_binary(inner_and, i3); // ((i0 & i1) & i3) - inner_and has fanout 2
        gb.add_output("o1".to_string(), outer_and1.into());
        gb.add_output("o2".to_string(), outer_and2.into()); // Add second output to ensure inner_and has fanout 2
        let mut g = gb.build();
        let res = rotate_and_right_primitive(&mut g, outer_and1.node);
        assert!(res.is_err());
        assert_eq!(
            res.err().unwrap(),
            "rotate_and_right_primitive: inner (left op of outer) has fanout > 1"
        );
    }

    #[test]
    fn test_rotate_and_left_primitive_fanout_fail() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let i3 = gb.add_input("i3".to_string(), 1).get_lsb(0).clone();
        let inner_and = gb.add_and_binary(i1, i2); // (i1 & i2)
        let outer_and1 = gb.add_and_binary(i0, inner_and); // (i0 & (i1 & i2))
        let outer_and2 = gb.add_and_binary(i3, inner_and); // (i3 & (i1 & i2)) - inner_and has fanout 2
        gb.add_output("o1".to_string(), outer_and1.into());
        gb.add_output("o2".to_string(), outer_and2.into()); // Add second output to ensure inner_and has fanout 2
        let mut g = gb.build();
        let res = rotate_and_left_primitive(&mut g, outer_and1.node);
        assert!(res.is_err());
        assert_eq!(
            res.err().unwrap(),
            "rotate_and_left_primitive: inner (right op of outer) has fanout > 1"
        );
    }
}
