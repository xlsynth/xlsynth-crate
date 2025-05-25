// SPDX-License-Identifier: Apache-2.0

//! Flatten a two-level branching AND tree into a monotonic chain so that
//! subsequent `BalanceAndTreeTransform` can re-balance it efficiently.
//!
//! Pattern recognised (symmetrically):
//!
//! ```text
//! root  = AND(left, right)
//! left  = AND(a, b)      // fan-out == 1, not negated
//! right = AND(c, d)      // fan-out == 1, not negated
//! ```
//!
//! After transform:
//!
//! ```text
//! new_1 = AND(left /*(a&b)*/, c)
//! root  = AND(new_1, d)
//! ```
//!
//! This forms the chain `root → new_1 → left`.
//! The original `right` node becomes dead and will be deleted by later DCE.
//! Node count increases by +1, depth of one leaf may rise, but the structure is
//! now a *monotonic* chain suitable for balancing.

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::topo::reaches_target as node_reaches_target;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{anyhow, Result};

/// Primitive helper that rewrites `root` which matches the pattern described
/// above.  Caller must ensure pre-conditions.
fn flatten_branching_and_primitive(g: &mut GateFn, root: AigRef) -> Result<(), &'static str> {
    // Extract operands of root.
    let (left_op, right_op) = match g.gates[root.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("flatten_branching_and: root not And2"),
    };

    // Preconditions (caller enforced): both operands are non-negated AND2 with
    // fan-out 1.
    if left_op.negated || right_op.negated {
        return Err("flatten_branching_and: operand negated");
    }
    let left_ref = left_op.node;
    let right_ref = right_op.node;

    // Get inner operands.
    let (l_a, l_b) = match g.gates[left_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => unreachable!(),
    };
    let (r_a, r_b) = match g.gates[right_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => unreachable!(),
    };

    // Sanity: ensure we don't introduce cycles by wiring upward references.
    // new_1 will reference `left_ref` and `r_a`.  Ensure neither can reach `root`.
    if node_reaches_target(&g.gates, left_ref, root)
        || node_reaches_target(&g.gates, r_a.node, root)
    {
        return Err("flatten_branching_and: would create cycle");
    }

    // Build the new middle gate: AND(left_ref, r_a)
    let new_gate = AigNode::And2 {
        a: AigOperand {
            node: left_ref,
            negated: false,
        },
        b: r_a,
        tags: None,
    };
    let new_ref = AigRef { id: g.gates.len() };
    g.gates.push(new_gate);

    // Rewrite root: (new_ref, r_b)
    if let AigNode::And2 { a, b, .. } = &mut g.gates[root.id] {
        *a = AigOperand {
            node: new_ref,
            negated: false,
        };
        *b = r_b;
    }

    // Strong invariant: graph remains acyclic.
    crate::topo::debug_assert_no_cycles(&g.gates, "flatten_branching_and");
    Ok(())
}

#[derive(Debug)]
pub struct FlattenBranchingAndTransform;

impl FlattenBranchingAndTransform {
    pub fn new() -> Self {
        Self
    }
}

impl Transform for FlattenBranchingAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::FlattenBranchingAnd
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
            if let AigNode::And2 {
                a: left_op,
                b: right_op,
                ..
            } = node
            {
                if left_op.negated || right_op.negated {
                    continue;
                }
                let left_ref = left_op.node;
                let right_ref = right_op.node;
                if !matches!(g.gates[left_ref.id], AigNode::And2 { .. })
                    || !matches!(g.gates[right_ref.id], AigNode::And2 { .. })
                {
                    continue;
                }
                if *use_counts.get(&left_ref).unwrap_or(&0) != 1
                    || *use_counts.get(&right_ref).unwrap_or(&0) != 1
                {
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
                "Backward direction not supported for FlattenBranchingAndTransform",
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                flatten_branching_and_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for FlattenBranchingAndTransform: {:?}",
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

    fn setup_branching_tree() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1).get_lsb(0).clone();
        let b = gb.add_input("b".to_string(), 1).get_lsb(0).clone();
        let c = gb.add_input("c".to_string(), 1).get_lsb(0).clone();
        let d = gb.add_input("d".to_string(), 1).get_lsb(0).clone();
        let left = gb.add_and_binary(a, b);
        let right = gb.add_and_binary(c, d);
        let root = gb.add_and_binary(left, right);
        gb.add_output("o".to_string(), root.into());
        (gb.build(), root.node)
    }

    #[test]
    fn test_flatten_branching_and_transform() {
        let (mut g, root_ref) = setup_branching_tree();
        let original = g.to_string();

        let mut t = FlattenBranchingAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == root_ref));

        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original);

        // Verify new structure is chain (root.left is And2, not negated)
        let (new_left, new_right) = match g.gates[root_ref.id] {
            AigNode::And2 { a, b, .. } => (a, b),
            _ => panic!("root not And2"),
        };
        assert!(!new_left.negated);
        assert!(matches!(g.gates[new_left.node.id], AigNode::And2 { .. }));
        // depth check not strict, but structure changed as intended.
    }
}
