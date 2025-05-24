// SPDX-License-Identifier: Apache-2.0

use crate::aig_hasher::AigHasher;
use crate::gate::{AigNode, AigRef, GateFn};
use crate::transforms::rewire_operand::rewire_operand_primitive;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

#[derive(Debug)]
pub struct MergeEquivLeavesTransform;

impl MergeEquivLeavesTransform {
    pub fn new() -> Self {
        MergeEquivLeavesTransform
    }
}

impl Transform for MergeEquivLeavesTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::MergeEquivLeaves
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        let mut hasher = AigHasher::new();
        let mut cands = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { a, b, .. } = node {
                if a.node == b.node && a.negated == b.negated {
                    continue;
                }
                let hash_a = hasher.get_hash(&a.node, &g.gates);
                let hash_b = hasher.get_hash(&b.node, &g.gates);
                if hash_a == hash_b && a.negated == b.negated {
                    let parent = AigRef { id: idx };
                    cands.push(TransformLocation::OperandTarget {
                        parent,
                        is_rhs: true,
                        old_op: *b,
                    });
                    cands.push(TransformLocation::OperandTarget {
                        parent,
                        is_rhs: false,
                        old_op: *a,
                    });
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
        let (parent, is_rhs, old_op) = match candidate_location {
            TransformLocation::OperandTarget {
                parent,
                is_rhs,
                old_op,
            } => (parent, is_rhs, old_op),
            _ => {
                return Err(anyhow!(
                    "Invalid location type for MergeEquivLeavesTransform: {:?}",
                    candidate_location
                ))
            }
        };
        if parent.id >= g.gates.len() {
            return Err(anyhow!("Parent ref out of bounds"));
        }
        match g.gates[parent.id] {
            AigNode::And2 { .. } => {}
            _ => return Err(anyhow!("Parent node is not And2")),
        }
        let target_op = if direction == TransformDirection::Forward {
            match g.gates[parent.id] {
                AigNode::And2 { a, b, .. } => {
                    if *is_rhs {
                        a
                    } else {
                        b
                    }
                }
                _ => unreachable!(),
            }
        } else {
            *old_op
        };
        rewire_operand_primitive(g, parent, *is_rhs, &target_op)
            .map(|_| ())
            .map_err(anyhow::Error::msg)
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::transforms::transform_trait::{Transform, TransformDirection};

    #[test]
    fn test_merge_equiv_leaves_round_trip() {
        let mut gb = GateBuilder::new("m".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        let b = gb.add_and_binary(i0, i1);
        let top = gb.add_and_binary(a, b);
        gb.add_output("o".to_string(), top.into());
        let mut g = gb.build();
        let orig = g.clone();

        let mut t = MergeEquivLeavesTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 2);
        let cand = cands[0].clone();

        t.apply(&mut g, &cand, TransformDirection::Forward).unwrap();
        match &g.gates[top.node.id] {
            AigNode::And2 {
                a: op_a, b: op_b, ..
            } => assert_eq!(op_a, op_b),
            _ => panic!("top not And2"),
        }
        t.apply(&mut g, &cand, TransformDirection::Backward)
            .unwrap();
        assert_eq!(g.to_string(), orig.to_string());
    }

    #[test]
    fn test_merge_equiv_leaves_no_candidates() {
        let mut gb = GateBuilder::new("n".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        let c = gb.add_and_binary(i0, i2);
        let top = gb.add_and_binary(a, c);
        gb.add_output("o".to_string(), top.into());
        let g = gb.build();
        let mut t = MergeEquivLeavesTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert!(cands.is_empty());
    }
}
