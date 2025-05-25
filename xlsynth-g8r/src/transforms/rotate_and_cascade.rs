// SPDX-License-Identifier: Apache-2.0

//! Cascade versions of rotate‐AND transforms that keep rotating until the
//! inner node's fan-out is greater than one (or the operand is no longer an
//! `And2`).  This collapses entire skinny spines in a single edit so that the
//! sampler pays the Metropolis/oracle cost only once instead of once per link.

use crate::gate::{AigNode, AigRef, GateFn};
use crate::transforms::rotate_and::{rotate_and_left_primitive, rotate_and_right_primitive};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{anyhow, Result};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
fn rotate_right_cascade(g: &mut GateFn, root: AigRef) -> Result<usize> {
    let mut count = 0;
    loop {
        // Attempt one step rotate.  If it fails we stop.
        match rotate_and_right_primitive(g, root) {
            Ok(()) => count += 1,
            Err(_) => break,
        }
        // After rotation, root is still the outer gate; check whether pattern
        // still matches (left child is non-negated And2 with fan-out 1).
        let (left_op, _) = match g.gates[root.id] {
            AigNode::And2 { a, b, .. } => (a, b),
            _ => break, // should not happen
        };
        if left_op.negated {
            break;
        }
        if !matches!(g.gates[left_op.node.id], AigNode::And2 { .. }) {
            break;
        }
        let use_counts = get_id_to_use_count(g);
        if *use_counts.get(&left_op.node).unwrap_or(&0) != 1 {
            break;
        }
        // else loop continues.
    }
    if count == 0 {
        Err(anyhow!("rotate_right_cascade: nothing to do"))
    } else {
        Ok(count)
    }
}

fn rotate_left_cascade(g: &mut GateFn, root: AigRef) -> Result<usize> {
    let mut count = 0;
    loop {
        match rotate_and_left_primitive(g, root) {
            Ok(()) => count += 1,
            Err(_) => break,
        }
        let (_, right_op) = match g.gates[root.id] {
            AigNode::And2 { a, b, .. } => (a, b),
            _ => break,
        };
        if right_op.negated {
            break;
        }
        if !matches!(g.gates[right_op.node.id], AigNode::And2 { .. }) {
            break;
        }
        let use_counts = get_id_to_use_count(g);
        if *use_counts.get(&right_op.node).unwrap_or(&0) != 1 {
            break;
        }
    }
    if count == 0 {
        Err(anyhow!("rotate_left_cascade: nothing to do"))
    } else {
        Ok(count)
    }
}

// -----------------------------------------------------------------------------
// RotateAndRightCascadeTransform
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct RotateAndRightCascadeTransform;

impl RotateAndRightCascadeTransform {
    pub fn new() -> Self {
        Self
    }
}

impl Transform for RotateAndRightCascadeTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RotateAndRightCascade
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        // Same predicate as one-step RotateAndRight.
        let use_counts = get_id_to_use_count(g);
        let mut cands = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { a: left_op, .. } = node {
                if left_op.negated {
                    continue;
                }
                if !matches!(g.gates[left_op.node.id], AigNode::And2 { .. }) {
                    continue;
                }
                if *use_counts.get(&left_op.node).unwrap_or(&0) != 1 {
                    continue;
                }
                cands.push(TransformLocation::Node(AigRef { id: idx }));
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
                "Backward direction not supported for RotateAndRightCascadeTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => rotate_right_cascade(g, *r)
                .map(|_| ())
                .map_err(anyhow::Error::msg),
            _ => Err(anyhow!(
                "Invalid location for RotateAndRightCascadeTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

// -----------------------------------------------------------------------------
// RotateAndLeftCascadeTransform
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct RotateAndLeftCascadeTransform;

impl RotateAndLeftCascadeTransform {
    pub fn new() -> Self {
        Self
    }
}

impl Transform for RotateAndLeftCascadeTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RotateAndLeftCascade
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
        let mut cands = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { b: right_op, .. } = node {
                if right_op.negated {
                    continue;
                }
                if !matches!(g.gates[right_op.node.id], AigNode::And2 { .. }) {
                    continue;
                }
                if *use_counts.get(&right_op.node).unwrap_or(&0) != 1 {
                    continue;
                }
                cands.push(TransformLocation::Node(AigRef { id: idx }));
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
                "Backward direction not supported for RotateAndLeftCascadeTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => rotate_left_cascade(g, *r)
                .map(|_| ())
                .map_err(anyhow::Error::msg),
            _ => Err(anyhow!(
                "Invalid location for RotateAndLeftCascadeTransform: {:?}",
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
    use crate::transforms::transform_trait::TransformDirection;

    fn setup_right_chain(depth: usize) -> (GateFn, AigRef) {
        assert!(depth >= 3);
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let mut inputs = Vec::new();
        for idx in 0..depth {
            inputs.push(gb.add_input(format!("i{idx}"), 1).get_lsb(0).clone());
        }
        let mut acc = gb.add_and_binary(inputs[0], inputs[1]);
        for inp in inputs.iter().skip(2) {
            acc = gb.add_and_binary(acc, *inp);
        }
        gb.add_output("o".to_string(), acc.into());
        (gb.build(), acc.node)
    }

    #[test]
    fn test_rotate_right_cascade_progresses() {
        let (mut g, root_ref) = setup_right_chain(5); // depth 5 chain on left side
        let mut t = RotateAndRightCascadeTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert!(!cands.is_empty());
        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        // No panic means cascade executed without error; further property
        // checks are left to integration tests.
    }
}
