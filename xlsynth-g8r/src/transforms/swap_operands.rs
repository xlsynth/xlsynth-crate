// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::Result;
use std::fmt::Debug;

/// Swaps the left and right operands of an `And2` gate.
/// This remains the core primitive.
fn do_swap_operands(g: &mut GateFn, node_ref: AigRef) -> Result<(), anyhow::Error> {
    // Ensure the node_ref.id is within bounds.
    if node_ref.id >= g.gates.len() {
        return Err(anyhow::anyhow!(
            "swap_operands: node ID {} out of bounds ({} gates)",
            node_ref.id,
            g.gates.len()
        ));
    }
    match &mut g.gates[node_ref.id] {
        AigNode::And2 { a, b, .. } => {
            core::mem::swap(a, b);
            Ok(())
        }
        _ => Err(anyhow::anyhow!(
            "swap_operands: node {:?} is not And2",
            node_ref
        )),
    }
}

#[derive(Debug)]
pub struct SwapOperandsTransform;

impl SwapOperandsTransform {
    pub fn new() -> Self {
        SwapOperandsTransform
    }
}

impl Transform for SwapOperandsTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::SwapOperands
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection, // Direction doesn't change candidates for this transform
    ) -> Vec<TransformLocation> {
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| match node {
                AigNode::And2 { .. } => Some(TransformLocation::Node(AigRef { id: idx })),
                _ => None,
            })
            .collect()
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        _direction: TransformDirection, // Direction doesn't change behavior for self-inverse op
    ) -> Result<()> {
        match candidate_location {
            TransformLocation::Node(node_ref) => do_swap_operands(g, *node_ref),
            _ => Err(anyhow::anyhow!(
                "Invalid candidate location for SwapOperandsTransform: {:?}",
                candidate_location
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::transforms::transform_trait::TransformDirection; // For tests

    #[test]
    fn test_swap_operands_transform_application() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_gate_op = gb.add_and_binary(i0, i1); // This is an AigOperand pointing to the And2 gate
        gb.add_output("o".to_string(), and_gate_op.into());
        let g_original = gb.build();
        let mut g_transformed = g_original.clone();

        let mut transform = SwapOperandsTransform::new();

        // Find candidates
        let candidates = transform.find_candidates(&g_transformed, TransformDirection::Forward);
        assert!(!candidates.is_empty(), "No candidates found for swap");

        // Find the specific candidate for our AND gate
        let target_and_ref = and_gate_op.node;
        let target_location = candidates
            .iter()
            .find(|loc| match loc {
                TransformLocation::Node(r) => *r == target_and_ref,
                _ => false,
            })
            .expect("Target AND gate not found in candidates");

        // Apply forward (swap)
        transform
            .apply(
                &mut g_transformed,
                target_location,
                TransformDirection::Forward,
            )
            .unwrap();
        let g_after_forward_str = g_transformed.to_string();
        assert_ne!(
            g_original.to_string(),
            g_after_forward_str,
            "Graph should have changed after forward apply (operands swapped)"
        );

        // Apply backward (swap again)
        transform
            .apply(
                &mut g_transformed,
                target_location,
                TransformDirection::Backward,
            )
            .unwrap();
        assert_eq!(
            g_original.to_string(),
            g_transformed.to_string(),
            "Graph should revert to original after backward apply"
        );
    }

    #[test]
    fn test_swap_operands_no_and_gates() {
        let mut gb = GateBuilder::new("f_no_and".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into()); // No And gate in this graph
        let g = gb.build();

        let mut transform = SwapOperandsTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert!(
            candidates.is_empty(),
            "Expected no candidates in a graph with no And2 gates"
        );
    }

    #[test]
    fn test_do_swap_operands_on_non_and_node() {
        let mut gb = GateBuilder::new("f_non_and".to_string(), GateBuilderOptions::no_opt());
        let i0_op = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0_op.into());
        let mut g = gb.build();

        // i0_op.node points to an Input node, not an And2 node.
        let result = do_swap_operands(&mut g, i0_op.node);
        assert!(
            result.is_err(),
            "Expected error when trying to swap operands of a non-And2 node"
        );
        if let Err(e) = result {
            assert!(e.to_string().contains("is not And2"));
        }
    }

    #[test]
    fn test_do_swap_operands_out_of_bounds() {
        let mut gb = GateBuilder::new("f_oob".to_string(), GateBuilderOptions::no_opt());
        let i0_bv = gb.add_input("i0".to_string(), 1);
        // Add a dummy output to allow GateFn to build
        gb.add_output("dummy_out".to_string(), i0_bv.get_lsb(0).clone().into());
        let mut g = gb.build(); // Graph has one input node (id 1, if literal is 0)

        let invalid_ref = AigRef { id: g.gates.len() }; // ID definitely out of bounds
        let result = do_swap_operands(&mut g, invalid_ref);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("out of bounds"));
        }
    }
}
