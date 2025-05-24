// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

const TAG_LEFT: &str = "balanced_chain_left";
const TAG_RIGHT: &str = "balanced_chain_right";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Left,
    Right,
}

fn collect_chain(
    g: &GateFn,
    root: AigRef,
    use_counts: &HashMap<AigRef, usize>,
) -> Option<(Orientation, Vec<AigRef>, Vec<AigOperand>)> {
    let mut nodes = Vec::new();
    let mut ops_rev = Vec::new();
    let mut cur = root;
    let mut orient = None;
    loop {
        nodes.push(cur);
        let (a, b) = match g.gates[cur.id] {
            AigNode::And2 { a, b, .. } => (a, b),
            _ => return None,
        };
        if orient.is_none() {
            let left_chain = !a.negated
                && matches!(g.gates[a.node.id], AigNode::And2 { .. })
                && *use_counts.get(&a.node).unwrap_or(&0) == 1;
            let right_chain = !b.negated
                && matches!(g.gates[b.node.id], AigNode::And2 { .. })
                && *use_counts.get(&b.node).unwrap_or(&0) == 1;
            if left_chain ^ right_chain {
                orient = Some(if left_chain {
                    Orientation::Left
                } else {
                    Orientation::Right
                });
            } else {
                return None;
            }
        }
        match orient.unwrap() {
            Orientation::Left => {
                if !a.negated
                    && matches!(g.gates[a.node.id], AigNode::And2 { .. })
                    && *use_counts.get(&a.node).unwrap_or(&0) == 1
                {
                    // Ensure the sibling `b` is a true leaf; if it also looks
                    // like the next link of a chain we would be balancing a
                    // *branching* tree which is not semantics-preserving.
                    if !b.negated
                        && matches!(g.gates[b.node.id], AigNode::And2 { .. })
                        && *use_counts.get(&b.node).unwrap_or(&0) == 1
                    {
                        return None; // branching, abort
                    }
                    ops_rev.push(b);
                    cur = a.node;
                    continue;
                } else {
                    ops_rev.push(b);
                    ops_rev.push(a);
                    break;
                }
            }
            Orientation::Right => {
                if !b.negated
                    && matches!(g.gates[b.node.id], AigNode::And2 { .. })
                    && *use_counts.get(&b.node).unwrap_or(&0) == 1
                {
                    // sibling `a` must be a leaf, otherwise branching
                    if !a.negated
                        && matches!(g.gates[a.node.id], AigNode::And2 { .. })
                        && *use_counts.get(&a.node).unwrap_or(&0) == 1
                    {
                        return None; // branching
                    }
                    ops_rev.push(a);
                    nodes.push(cur);
                    cur = b.node;
                    continue;
                } else {
                    ops_rev.push(a);
                    ops_rev.push(b);
                    break;
                }
            }
        }
    }
    let mut ops = ops_rev;
    if orient == Some(Orientation::Left) {
        ops.reverse();
    }
    if nodes.len() >= 2 {
        Some((orient.unwrap(), nodes, ops))
    } else {
        None
    }
}

fn build_balanced(g: &mut GateFn, nodes: &[AigRef], ops: &[AigOperand]) -> Result<()> {
    fn helper(
        g: &mut GateFn,
        nodes: &[AigRef],
        ops: &[AigOperand],
        i: usize,
        j: usize,
        next: &mut usize,
    ) -> Result<AigOperand> {
        if i == j {
            return Ok(ops[i]);
        }
        let node = nodes[*next];
        *next += 1;
        let m = (i + j) / 2;
        let left = helper(g, nodes, ops, i, m, next)?;
        let right = helper(g, nodes, ops, m + 1, j, next)?;
        if let AigNode::And2 { a, b, .. } = &mut g.gates[node.id] {
            *a = left;
            *b = right;
        } else {
            return Err(anyhow!("Node {:?} not And2", node));
        }
        Ok(AigOperand {
            node,
            negated: false,
        })
    }
    let mut idx = 0;
    helper(g, nodes, ops, 0, ops.len() - 1, &mut idx)?;
    debug_assert_eq!(idx, nodes.len());
    crate::topo::debug_assert_no_cycles(&g.gates, "balance_and_tree");
    Ok(())
}

fn build_chain(
    g: &mut GateFn,
    orient: Orientation,
    nodes_preorder: &[AigRef],
    ops_inorder: &[AigOperand],
) -> Result<()> {
    if nodes_preorder.is_empty() || ops_inorder.len() != nodes_preorder.len() + 1 {
        return Err(anyhow!("Invalid sizes for chain reconstruction"));
    }
    let mut current = *nodes_preorder.last().unwrap();
    match orient {
        Orientation::Left => {
            if let AigNode::And2 { a, b, .. } = &mut g.gates[current.id] {
                *a = ops_inorder[0];
                *b = ops_inorder[1];
            }
            let mut op_idx = 2;
            for node in nodes_preorder.iter().rev().skip(1) {
                if let AigNode::And2 { a, b, .. } = &mut g.gates[node.id] {
                    *a = AigOperand {
                        node: current,
                        negated: false,
                    };
                    *b = ops_inorder[op_idx];
                }
                current = *node;
                op_idx += 1;
            }
        }
        Orientation::Right => {
            if ops_inorder.len() < 3 {
                return Err(anyhow!(
                    "build_chain expected at least 3 operands for Right orientation, got {}",
                    ops_inorder.len()
                ));
            }
            if let AigNode::And2 { a, b, .. } = &mut g.gates[current.id] {
                *a = ops_inorder[ops_inorder.len() - 2];
                *b = ops_inorder[ops_inorder.len() - 1];
            }
            let mut op_idx = ops_inorder.len() - 3;
            for node in nodes_preorder.iter().rev().skip(1) {
                if let AigNode::And2 { a, b, .. } = &mut g.gates[node.id] {
                    *a = ops_inorder[op_idx];
                    *b = AigOperand {
                        node: current,
                        negated: false,
                    };
                }
                current = *node;
                if op_idx == 0 {
                    break;
                }
                op_idx -= 1;
            }
        }
    }
    crate::topo::debug_assert_no_cycles(&g.gates, "unbalance_and_tree");
    Ok(())
}

/// Starting from `root`, follow a monotonic chain along the given `orient`
/// direction, collecting:
///   * `nodes`: the `AigRef`s that form the internal AND gates of the chain in
///     root-to-leaf order.
///   * `ops`:  the operands that appear *between* the chain links, given in
///     inorder such that `ops.len() == nodes.len() + 1` – exactly what
///     `build_chain` expects.
///
/// Returns `None` if `root` is not an `And2` or does not have a monotonic
/// chain along the requested orientation.
fn collect_chain_linear(
    g: &GateFn,
    root: AigRef,
    orient: Orientation,
    use_counts: &HashMap<AigRef, usize>,
) -> Option<(Vec<AigRef>, Vec<AigOperand>)> {
    let mut nodes = Vec::new();
    let mut ops_rev = Vec::new();

    let mut cur = root;
    loop {
        // Current node must be And2.
        let (a, b) = match g.gates[cur.id] {
            AigNode::And2 { a, b, .. } => (a, b),
            _ => return None,
        };

        match orient {
            Orientation::Left => {
                // Chain grows on the *left* input.
                if !a.negated
                    && matches!(g.gates[a.node.id], AigNode::And2 { .. })
                    && *use_counts.get(&a.node).unwrap_or(&0) == 1
                {
                    // Still in the chain. The *other* operand (b) is a leaf at
                    // this level.
                    ops_rev.push(b);
                    nodes.push(cur);
                    cur = a.node;
                    continue;
                } else {
                    // Final link – both a & b are leaves.
                    ops_rev.push(b);
                    ops_rev.push(a);
                    nodes.push(cur);
                    break;
                }
            }
            Orientation::Right => {
                // Chain grows on the *right* input.
                if !b.negated
                    && matches!(g.gates[b.node.id], AigNode::And2 { .. })
                    && *use_counts.get(&b.node).unwrap_or(&0) == 1
                {
                    ops_rev.push(a);
                    nodes.push(cur);
                    cur = b.node;
                    continue;
                } else {
                    ops_rev.push(a);
                    ops_rev.push(b);
                    nodes.push(cur);
                    break;
                }
            }
        }
    }

    let mut ops = ops_rev;
    if orient == Orientation::Left {
        ops.reverse();
    }
    Some((nodes, ops))
}

#[derive(Debug)]
pub struct BalanceAndTreeTransform;

impl BalanceAndTreeTransform {
    pub fn new() -> Self {
        BalanceAndTreeTransform
    }
}

impl Transform for BalanceAndTreeTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::BalanceAndTree
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
            if let AigNode::And2 { .. } = node {
                if let Some((_o, nodes, _ops)) = collect_chain(g, AigRef { id: idx }, &use_counts) {
                    if nodes.len() >= 2 {
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
            return Err(anyhow!("Use UnbalanceAndTreeTransform for backward"));
        }
        let use_counts = get_id_to_use_count(g);
        match candidate_location {
            TransformLocation::Node(root) => {
                if let Some((orient, nodes, ops)) = collect_chain(g, *root, &use_counts) {
                    if nodes.len() + 1 != ops.len() {
                        return Err(anyhow!(
                            "collect_chain produced inconsistent sizes: nodes={}, ops={}",
                            nodes.len(),
                            ops.len()
                        ));
                    }
                    build_balanced(g, &nodes, &ops)?;
                    if let AigNode::And2 { tags, .. } = &mut g.gates[root.id] {
                        let tag = match orient {
                            Orientation::Left => TAG_LEFT,
                            Orientation::Right => TAG_RIGHT,
                        };
                        if let Some(ts) = tags {
                            ts.push(tag.to_string());
                        } else {
                            *tags = Some(vec![tag.to_string()]);
                        }
                    }
                    Ok(())
                } else {
                    Err(anyhow!("Candidate does not form monotonic chain"))
                }
            }
            _ => Err(anyhow!(
                "Invalid location type for BalanceAndTreeTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct UnbalanceAndTreeTransform;

impl UnbalanceAndTreeTransform {
    pub fn new() -> Self {
        UnbalanceAndTreeTransform
    }
}

impl Transform for UnbalanceAndTreeTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::UnbalanceAndTree
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        let mut candidates = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { tags, .. } = node {
                if let Some(ts) = tags {
                    if ts.iter().any(|t| t == TAG_LEFT || t == TAG_RIGHT) {
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
            return Err(anyhow!("Use BalanceAndTreeTransform for backward"));
        }
        match candidate_location {
            TransformLocation::Node(root) => {
                let orient = match &g.gates[root.id] {
                    AigNode::And2 { tags: Some(ts), .. } => {
                        if ts.iter().any(|t| t == TAG_LEFT) {
                            Orientation::Left
                        } else if ts.iter().any(|t| t == TAG_RIGHT) {
                            Orientation::Right
                        } else {
                            return Err(anyhow!("No orientation tag found"));
                        }
                    }
                    _ => return Err(anyhow!("Root is not And2 with tags")),
                };
                let use_counts = get_id_to_use_count(g);
                let (nodes, ops) = collect_chain_linear(g, *root, orient, &use_counts)
                    .ok_or_else(|| anyhow!("Failed to collect chain for unbalance"))?;
                if nodes.len() < 2 {
                    return Err(anyhow!(
                        "UnbalanceAndTree requires at least two nodes in chain; found {}",
                        nodes.len()
                    ));
                }
                build_chain(g, orient, &nodes, &ops)?;
                // Assert strong invariant: UnbalanceAndTree must not create cycles.
                crate::topo::debug_assert_no_cycles(&g.gates, "unbalance_and_tree");
                if let AigNode::And2 { tags: Some(ts), .. } = &mut g.gates[root.id] {
                    ts.retain(|t| t != TAG_LEFT && t != TAG_RIGHT);
                    if ts.is_empty() {
                        g.gates[root.id] = match &g.gates[root.id] {
                            AigNode::And2 { a, b, .. } => AigNode::And2 {
                                a: *a,
                                b: *b,
                                tags: None,
                            },
                            _ => unreachable!(),
                        };
                    }
                }
                Ok(())
            }
            _ => Err(anyhow!(
                "Invalid location type for UnbalanceAndTreeTransform: {:?}",
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
    use crate::transforms::transform_trait::{Transform, TransformLocation};

    fn setup_left_chain() -> GateFn {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let i3 = gb.add_input("i3".to_string(), 1).get_lsb(0).clone();
        let n1 = gb.add_and_binary(i0, i1);
        let n2 = gb.add_and_binary(n1, i2);
        let n3 = gb.add_and_binary(n2, i3);
        gb.add_output("o".to_string(), n3.into());
        gb.build()
    }

    #[test]
    fn test_balance_and_unbalance_roundtrip() {
        let mut g = setup_left_chain();
        let original = g.to_string();
        let mut bal = BalanceAndTreeTransform::new();
        let candidates = bal.find_candidates(&g, TransformDirection::Forward);
        assert!(!candidates.is_empty());
        bal.apply(&mut g, &candidates[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original);
        let mut unbal = UnbalanceAndTreeTransform::new();
        let c2 = unbal.find_candidates(&g, TransformDirection::Forward);
        assert!(!c2.is_empty());
        unbal
            .apply(&mut g, &c2[0], TransformDirection::Forward)
            .unwrap();
        assert_eq!(g.to_string(), original);
    }

    #[test]
    fn test_balance_and_unbalance_roundtrip_right() {
        // Construct a right-leaning AND chain: o = ((((i0 & i1) & i2) & i3)) with chain
        // on the right.
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let i3 = gb.add_input("i3".to_string(), 1).get_lsb(0).clone();
        let n1 = gb.add_and_binary(i0, i1); // depth 1
        let n2 = gb.add_and_binary(n1, i2); // depth 2 (chain on RHS of root will start here)
        let n3 = gb.add_and_binary(n2, i3); // depth 3, right-leaning chain root
        let root_op = gb.add_and_binary(n2, n3);
        gb.add_output("o".to_string(), root_op.into());
        let mut g = gb.build();
        let _root = root_op.node; // AigRef of the root AND gate
        let original = g.to_string();

        // Balance, then unbalance, and ensure we round-trip.
        let mut bal = BalanceAndTreeTransform::new();
        let candidates = bal.find_candidates(&g, TransformDirection::Forward);
        assert!(!candidates.is_empty());
        bal.apply(&mut g, &candidates[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original);

        let mut unbal = UnbalanceAndTreeTransform::new();
        let c2 = unbal.find_candidates(&g, TransformDirection::Forward);
        assert!(!c2.is_empty());
        unbal
            .apply(&mut g, &c2[0], TransformDirection::Forward)
            .unwrap();
        assert_eq!(g.to_string(), original);
    }

    #[test]
    fn test_branching_chain_not_balanced() {
        // Build a root AND with two AND children (branching tree). This SHOULD
        // NOT be considered a valid candidate for BalanceAndTreeTransform.
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let i3 = gb.add_input("i3".to_string(), 1).get_lsb(0).clone();

        let left_chain = gb.add_and_binary(i0, i1); // left child AND
        let right_chain = gb.add_and_binary(i2, i3); // right child AND

        let root_op = gb.add_and_binary(left_chain, right_chain);
        gb.add_output("o".to_string(), root_op.into());
        let mut g = gb.build();
        let root = root_op.node; // AigRef of the root AND gate

        let mut bal = BalanceAndTreeTransform::new();
        let cands = bal.find_candidates(&g, TransformDirection::Forward);
        assert!(
            cands.is_empty(),
            "Branching chain should not be a candidate for balancing"
        );

        // For good measure attempt to apply directly to root and expect Err.
        let res = bal.apply(
            &mut g,
            &TransformLocation::Node(root),
            TransformDirection::Forward,
        );
        assert!(res.is_err(), "Balancing branching chain should fail");
    }
}
