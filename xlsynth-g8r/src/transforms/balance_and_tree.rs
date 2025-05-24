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
                    ops_rev.push(b);
                    cur = a.node;
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
                    ops_rev.push(a);
                    cur = b.node;
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

fn preorder_collect(g: &GateFn, root: AigRef, nodes: &mut Vec<AigRef>) {
    nodes.push(root);
    if let AigNode::And2 { a, b, .. } = g.gates[root.id] {
        if let AigNode::And2 { .. } = g.gates[a.node.id] {
            preorder_collect(g, a.node, nodes);
        }
        if let AigNode::And2 { .. } = g.gates[b.node.id] {
            preorder_collect(g, b.node, nodes);
        }
    }
}

fn inorder_collect(g: &GateFn, root: AigRef, ops: &mut Vec<AigOperand>) {
    if let AigNode::And2 { a, b, .. } = g.gates[root.id] {
        if let AigNode::And2 { .. } = g.gates[a.node.id] {
            inorder_collect(g, a.node, ops);
        } else {
            ops.push(a);
        }
        if let AigNode::And2 { .. } = g.gates[b.node.id] {
            inorder_collect(g, b.node, ops);
        } else {
            ops.push(b);
        }
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
                let mut nodes = Vec::new();
                preorder_collect(g, *root, &mut nodes);
                let mut ops = Vec::new();
                inorder_collect(g, *root, &mut ops);
                build_chain(g, orient, &nodes, &ops)?;
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
    use crate::transforms::transform_trait::Transform;

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
}
