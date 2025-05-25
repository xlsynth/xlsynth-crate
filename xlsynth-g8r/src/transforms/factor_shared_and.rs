// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::test_utils::structurally_equivalent;
use crate::topo::reaches_target as node_reaches_target;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{anyhow, Result};

/// Collapses `((a & b) & (a & c))` into `(a & (b & c))`.
///
/// `outer` must be an `And2` gate whose children are both non-negated `And2`
/// nodes with fanout 1 and exactly one shared operand.
pub fn factor_shared_and_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left, right) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("factor_shared_and_primitive: outer is not And2"),
    };

    if left.negated || right.negated {
        return Err("factor_shared_and_primitive: child operand negated");
    }

    let (ll_a, ll_b) = match g.gates[left.node.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("factor_shared_and_primitive: left child not And2"),
    };
    let (rl_a, rl_b) = match g.gates[right.node.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("factor_shared_and_primitive: right child not And2"),
    };

    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&left.node).unwrap_or(&0) != 1 {
        return Err("factor_shared_and_primitive: left child fanout != 1");
    }
    if *use_counts.get(&right.node).unwrap_or(&0) != 1 {
        return Err("factor_shared_and_primitive: right child fanout != 1");
    }

    let mut shared = None;
    let mut unique_l = None;
    let mut unique_r = None;

    if ll_a == rl_a {
        shared = Some(ll_a);
        unique_l = Some(ll_b);
        unique_r = Some(rl_b);
    }
    if ll_a == rl_b {
        if shared.is_some() {
            return Err("factor_shared_and_primitive: multiple shared operands");
        }
        shared = Some(ll_a);
        unique_l = Some(ll_b);
        unique_r = Some(rl_a);
    }
    if ll_b == rl_a {
        if shared.is_some() {
            return Err("factor_shared_and_primitive: multiple shared operands");
        }
        shared = Some(ll_b);
        unique_l = Some(ll_a);
        unique_r = Some(rl_b);
    }
    if ll_b == rl_b {
        if shared.is_some() {
            return Err("factor_shared_and_primitive: multiple shared operands");
        }
        shared = Some(ll_b);
        unique_l = Some(ll_a);
        unique_r = Some(rl_a);
    }

    let (shared, unique_l, unique_r) = match (shared, unique_l, unique_r) {
        (Some(s), Some(l), Some(r)) => (s, l, r),
        _ => return Err("factor_shared_and_primitive: no single shared operand"),
    };

    if node_reaches_target(&g.gates, unique_l.node, outer)
        || node_reaches_target(&g.gates, unique_r.node, outer)
    {
        return Err("factor_shared_and_primitive: would create cycle");
    }

    if let AigNode::And2 { a, b, .. } = &mut g.gates[left.node.id] {
        *a = unique_l;
        *b = unique_r;
    }
    if let AigNode::And2 { a, b, .. } = &mut g.gates[outer.id] {
        *a = shared;
        *b = AigOperand {
            node: left.node,
            negated: false,
        };
    }

    crate::topo::debug_assert_no_cycles(&g.gates, "factor_shared_and_primitive");
    Ok(())
}

/// Expands `(a & (b & c))` into `((a & b) & (a & c))`.
/// The operand that is not the `And2` child becomes the common factor.
pub fn unfactor_shared_and_primitive(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left, right) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("unfactor_shared_and_primitive: outer is not And2"),
    };

    let use_counts = get_id_to_use_count(g);

    let (inner_is_rhs, inner_ref, common_op) = if !right.negated
        && matches!(g.gates[right.node.id], AigNode::And2 { .. })
        && *use_counts.get(&right.node).unwrap_or(&0) == 1
    {
        (true, right.node, left)
    } else if !left.negated
        && matches!(g.gates[left.node.id], AigNode::And2 { .. })
        && *use_counts.get(&left.node).unwrap_or(&0) == 1
    {
        (false, left.node, right)
    } else {
        return Err("unfactor_shared_and_primitive: pattern not matched");
    };

    let (inner_a, inner_b) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => unreachable!(),
    };

    // Reject degenerate inner gate where both operands identical.
    if inner_a == inner_b {
        return Err("unfactor_shared_and_primitive: inner gate is AND(x,x); cannot unfactor");
    }

    // Ensure we do not combine operands of opposite polarity referring to the same
    // node.
    let same_node_diff_pol =
        |x: AigOperand| x.node == common_op.node && x.negated != common_op.negated;
    if same_node_diff_pol(inner_a) || same_node_diff_pol(inner_b) {
        return Err("unfactor_shared_and_primitive: polarity mismatch between common operand and inner gate");
    }

    // Choose unique operand as inner_a; keep ordering deterministic.
    let unique_op = inner_a;

    if node_reaches_target(&g.gates, common_op.node, outer)
        || node_reaches_target(&g.gates, unique_op.node, outer)
    {
        return Err("unfactor_shared_and_primitive: would create cycle");
    }

    // Reject patterns that would create self-referential loops
    if common_op.node == inner_ref || unique_op.node == inner_ref {
        return Err("unfactor_shared_and_primitive: would create self-loop");
    }

    let new_gate = AigNode::And2 {
        a: common_op,
        b: unique_op,
        tags: None,
    };
    let new_ref = AigRef { id: g.gates.len() };
    // Extra safety: neither operand of the new gate may reference new_ref itself.
    debug_assert!(common_op.node != new_ref && unique_op.node != new_ref);
    g.gates.push(new_gate);

    if let AigNode::And2 { a, b, .. } = &mut g.gates[inner_ref.id] {
        *a = common_op;
        *b = inner_b;
    }

    if let AigNode::And2 { a, b, .. } = &mut g.gates[outer.id] {
        if inner_is_rhs {
            *a = AigOperand {
                node: new_ref,
                negated: false,
            };
            *b = AigOperand {
                node: inner_ref,
                negated: false,
            };
        } else {
            *a = AigOperand {
                node: inner_ref,
                negated: false,
            };
            *b = AigOperand {
                node: new_ref,
                negated: false,
            };
        }
    }

    crate::topo::debug_assert_no_cycles(&g.gates, "unfactor_shared_and_primitive");
    Ok(())
}

#[derive(Debug)]
pub struct FactorSharedAndTransform;

impl FactorSharedAndTransform {
    pub fn new() -> Self {
        FactorSharedAndTransform
    }
}

impl Transform for FactorSharedAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::FactorSharedAnd
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
            if let AigNode::And2 { a: l, b: r, .. } = node {
                if l.node == r.node && l.negated == r.negated {
                    continue;
                }
                if l.negated || r.negated {
                    continue;
                }
                if let (
                    AigNode::And2 {
                        a: ll_a, b: ll_b, ..
                    },
                    AigNode::And2 {
                        a: rl_a, b: rl_b, ..
                    },
                ) = (&g.gates[l.node.id], &g.gates[r.node.id])
                {
                    if *use_counts.get(&l.node).unwrap_or(&0) != 1
                        || *use_counts.get(&r.node).unwrap_or(&0) != 1
                    {
                        continue;
                    }
                    let mut eq_count = 0;
                    if *ll_a == *rl_a {
                        eq_count += 1;
                    }
                    if *ll_a == *rl_b {
                        eq_count += 1;
                    }
                    if *ll_b == *rl_a {
                        eq_count += 1;
                    }
                    if *ll_b == *rl_b {
                        eq_count += 1;
                    }
                    if eq_count == 1 {
                        cands.push(TransformLocation::Node(AigRef { id: idx }));
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
                "Backward direction not supported for FactorSharedAndTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                factor_shared_and_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for FactorSharedAndTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct UnfactorSharedAndTransform;

impl UnfactorSharedAndTransform {
    pub fn new() -> Self {
        UnfactorSharedAndTransform
    }
}

impl Transform for UnfactorSharedAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::UnfactorSharedAnd
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
            if let AigNode::And2 { a, b, .. } = node {
                let a_is_inner = !a.negated
                    && matches!(g.gates[a.node.id], AigNode::And2 { .. })
                    && *use_counts.get(&a.node).unwrap_or(&0) == 1;
                let b_is_inner = !b.negated
                    && matches!(g.gates[b.node.id], AigNode::And2 { .. })
                    && *use_counts.get(&b.node).unwrap_or(&0) == 1;
                if a_is_inner ^ b_is_inner {
                    cands.push(TransformLocation::Node(AigRef { id: idx }));
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
                "Backward direction not supported for UnfactorSharedAndTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                unfactor_shared_and_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for UnfactorSharedAndTransform: {:?}",
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

    fn setup_factor_graph() -> (GateFn, AigRef) {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1).get_lsb(0).clone();
        let b = gb.add_input("b".to_string(), 1).get_lsb(0).clone();
        let c = gb.add_input("c".to_string(), 1).get_lsb(0).clone();
        let left = gb.add_and_binary(a.clone(), b);
        let right = gb.add_and_binary(a.clone(), c);
        let outer = gb.add_and_binary(left, right);
        gb.add_output("o".to_string(), outer.into());
        (gb.build(), outer.node)
    }

    #[test]
    fn test_factor_and_unfactor_round_trip() {
        let (mut g, outer) = setup_factor_graph();
        let orig = g.clone();
        let mut f = FactorSharedAndTransform::new();
        let cands = f.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        assert!(matches!(cands[0], TransformLocation::Node(r) if r == outer));
        f.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        assert!(!structurally_equivalent(&g, &orig));
        let mut u = UnfactorSharedAndTransform::new();
        let c2 = u.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(c2.len(), 1);
        u.apply(&mut g, &c2[0], TransformDirection::Forward)
            .unwrap();
        assert!(structurally_equivalent(&g, &orig));
    }

    #[test]
    fn test_factor_shared_and_no_candidate_when_identical() {
        let (mut g, outer) = setup_factor_graph();
        // Make both children of the outer node point to the same node (id 1)
        if let AigNode::And2 { a, b, .. } = &mut g.gates[outer.id] {
            *a = AigOperand {
                node: AigRef { id: 1 },
                negated: false,
            };
            *b = AigOperand {
                node: AigRef { id: 1 },
                negated: false,
            };
        }
        let mut f = FactorSharedAndTransform::new();
        let c = f.find_candidates(&g, TransformDirection::Forward);
        assert!(c.is_empty());
    }
}
