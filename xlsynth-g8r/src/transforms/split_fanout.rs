// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::duplicate::duplicate;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use crate::use_count::get_id_to_use_count;
use anyhow::{Result, anyhow};

/// Primitive that duplicates `parent` and rewires one fanout edge to the
/// duplicate. `child` must reference `parent` as one of its operands.
pub fn split_fanout_primitive(
    g: &mut GateFn,
    parent: AigRef,
    child: AigRef,
) -> Result<(), &'static str> {
    if parent.id >= g.gates.len() || child.id >= g.gates.len() {
        return Err("AigRef out of bounds");
    }
    if !matches!(g.gates[parent.id], AigNode::And2 { .. }) {
        return Err("parent not And2");
    }
    let new_ref = duplicate(g, parent)?;
    match &mut g.gates[child.id] {
        AigNode::And2 { a, b, .. } => {
            if a.node != parent && b.node != parent {
                return Err("child does not use parent");
            }
            if a.node == parent {
                *a = AigOperand {
                    node: new_ref,
                    negated: a.negated,
                };
            }
            if b.node == parent {
                *b = AigOperand {
                    node: new_ref,
                    negated: b.negated,
                };
            }
            Ok(())
        }
        _ => Err("child not And2"),
    }
}

#[derive(Debug)]
pub struct SplitFanoutTransform;

impl SplitFanoutTransform {
    pub fn new() -> Self {
        SplitFanoutTransform
    }
}

impl Transform for SplitFanoutTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::SplitFanout
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
            if matches!(node, AigNode::And2 { .. }) {
                let parent_ref = AigRef { id: idx };
                if *use_counts.get(&parent_ref).unwrap_or(&0) > 1 {
                    for (cid, cnode) in g.gates.iter().enumerate() {
                        if let AigNode::And2 { a, b, .. } = cnode {
                            if a.node == parent_ref || b.node == parent_ref {
                                cands.push(TransformLocation::FanoutEdge {
                                    parent: parent_ref,
                                    child: AigRef { id: cid },
                                });
                            }
                        }
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
            return Err(anyhow!("Backward direction not supported"));
        }
        match candidate_location {
            TransformLocation::FanoutEdge { parent, child } => {
                split_fanout_primitive(g, *parent, *child).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for SplitFanoutTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// Primitive that merges `duplicate` back into `original` by redirecting all
/// fanouts of `duplicate` to `original`.
pub fn merge_fanout_primitive(
    g: &mut GateFn,
    original: AigRef,
    duplicate: AigRef,
) -> Result<(), &'static str> {
    if original.id >= g.gates.len() || duplicate.id >= g.gates.len() {
        return Err("AigRef out of bounds");
    }
    let (orig_a, orig_b) = match g.gates[original.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("original not And2"),
    };
    let (dup_a, dup_b) = match g.gates[duplicate.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("duplicate not And2"),
    };
    if orig_a != dup_a || orig_b != dup_b {
        return Err("gates are not structurally identical");
    }
    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&duplicate).unwrap_or(&0) != 1 {
        return Err("duplicate fanout != 1");
    }
    for gate in &mut g.gates {
        if let AigNode::And2 { a, b, .. } = gate {
            if a.node == duplicate {
                a.node = original;
            }
            if b.node == duplicate {
                b.node = original;
            }
        }
    }
    for output in &mut g.outputs {
        let mut ops: Vec<AigOperand> = output.bit_vector.iter_lsb_to_msb().copied().collect();
        let mut changed = false;
        for op in &mut ops {
            if op.node == duplicate {
                op.node = original;
                changed = true;
            }
        }
        if changed {
            output.bit_vector = crate::gate::AigBitVector::from_lsb_is_index_0(&ops);
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct MergeFanoutTransform;

impl MergeFanoutTransform {
    pub fn new() -> Self {
        MergeFanoutTransform
    }
}

impl Transform for MergeFanoutTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::MergeFanout
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
        for (dup_idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 {
                a: dup_a, b: dup_b, ..
            } = node
            {
                let dup_ref = AigRef { id: dup_idx };
                if *use_counts.get(&dup_ref).unwrap_or(&0) != 1 {
                    continue;
                }
                for (orig_idx, onode) in g.gates.iter().enumerate() {
                    if dup_idx == orig_idx {
                        continue;
                    }
                    if let AigNode::And2 { a: oa, b: ob, .. } = onode {
                        if oa == dup_a && ob == dup_b {
                            cands.push(TransformLocation::FanoutEdge {
                                parent: AigRef { id: orig_idx },
                                child: dup_ref,
                            });
                            break;
                        }
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
            return Err(anyhow!("Backward direction not supported"));
        }
        match candidate_location {
            TransformLocation::FanoutEdge { parent, child } => {
                merge_fanout_primitive(g, *parent, *child).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for MergeFanoutTransform: {:?}",
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

    #[test]
    fn test_split_fanout_forward() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let mid = gb.add_and_binary(i0, i1);
        let use_a = gb.add_and_binary(mid, i2);
        let use_b = gb.add_and_binary(mid, i2);
        gb.add_output("o0".to_string(), use_a.into());
        gb.add_output("o1".to_string(), use_b.into());
        let mut g = gb.build();

        let mut t = SplitFanoutTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 2);
        let cand = cands
            .iter()
            .find(|loc| matches!(loc, TransformLocation::FanoutEdge { child, .. } if *child == use_a.node))
            .unwrap();
        let prev_gate_count = g.gates.len();
        t.apply(&mut g, cand, TransformDirection::Forward).unwrap();
        assert_eq!(g.gates.len(), prev_gate_count + 1);
        if let AigNode::And2 { a, .. } = &g.gates[use_a.node.id] {
            assert_ne!(a.node, mid.node);
        } else {
            panic!("use_a not And2");
        }
    }

    #[test]
    fn test_merge_fanout_forward() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let orig = gb.add_and_binary(i0, i1);
        let dup = gb.add_and_binary(i0, i1);
        let u0 = gb.add_and_binary(orig, i2);
        let u1 = gb.add_and_binary(dup, i2);
        gb.add_output("o0".to_string(), u0.into());
        gb.add_output("o1".to_string(), u1.into());
        let mut g = gb.build();

        let mut t = MergeFanoutTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert!(!cands.is_empty());
        let cand = cands
            .iter()
            .find(|loc| matches!(loc, TransformLocation::FanoutEdge { child, .. } if *child == dup.node))
            .unwrap();
        t.apply(&mut g, cand, TransformDirection::Forward).unwrap();
        if let AigNode::And2 { a, .. } = &g.gates[u1.node.id] {
            assert_eq!(a.node, orig.node);
        } else {
            panic!("u1 not And2");
        }
    }
}
