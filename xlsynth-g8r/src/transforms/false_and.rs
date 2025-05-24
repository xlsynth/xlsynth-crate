// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Creates a new AND gate of the form `AND(x, !x)`, which always evaluates
/// to false. The new gate is appended to the list of gates and returned.
pub fn insert_false_and_primitive(g: &mut GateFn, x: AigRef) -> Result<AigRef, &'static str> {
    if x.id >= g.gates.len() {
        return Err("insert_false_and_primitive: AigRef out of bounds");
    }
    let new_gate = AigNode::And2 {
        a: AigOperand {
            node: x,
            negated: false,
        },
        b: AigOperand {
            node: x,
            negated: true,
        },
        tags: None,
    };
    let new_ref = AigRef { id: g.gates.len() };
    g.gates.push(new_gate);
    Ok(new_ref)
}

/// Replaces all uses of `node` with constant false when `node` is of the form
/// `AND(x, !x)`. The node itself is left in place but becomes dead.
pub fn remove_false_and_primitive(g: &mut GateFn, node: AigRef) -> Result<(), &'static str> {
    if node.id >= g.gates.len() {
        return Err("remove_false_and_primitive: AigRef out of bounds");
    }
    let (a, b) = match g.gates[node.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("remove_false_and_primitive: node is not And2"),
    };
    if a.node != b.node || a.negated == b.negated {
        return Err("remove_false_and_primitive: node is not AND(x, !x)");
    }
    let false_op = AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    };
    for gate in &mut g.gates {
        if let AigNode::And2 {
            a: ref mut op_a,
            b: ref mut op_b,
            ..
        } = gate
        {
            if op_a.node == node {
                *op_a = false_op;
            }
            if op_b.node == node {
                *op_b = false_op;
            }
        }
    }
    for output in &mut g.outputs {
        for idx in 0..output.bit_vector.get_bit_count() {
            let op = *output.bit_vector.get_lsb(idx);
            if op.node == node {
                output.bit_vector.set_lsb(idx, false_op);
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct InsertFalseAndTransform;

impl InsertFalseAndTransform {
    pub fn new() -> Self {
        InsertFalseAndTransform
    }
}

impl Transform for InsertFalseAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::InsertFalseAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        // Any node except the constant false can be used to form AND(x, !x)
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, _)| {
                if idx == 0 {
                    None
                } else {
                    Some(TransformLocation::Node(AigRef { id: idx }))
                }
            })
            .collect()
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Backward {
            return Err(anyhow!(
                "Backward direction not supported for InsertFalseAndTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => insert_false_and_primitive(g, *r)
                .map(|_| ())
                .map_err(anyhow::Error::msg),
            _ => Err(anyhow!(
                "Invalid location for InsertFalseAndTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct RemoveFalseAndTransform;

impl RemoveFalseAndTransform {
    pub fn new() -> Self {
        RemoveFalseAndTransform
    }
}

impl Transform for RemoveFalseAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RemoveFalseAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new();
        }
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| match node {
                AigNode::And2 { a, b, .. } if a.node == b.node && a.negated != b.negated => {
                    Some(TransformLocation::Node(AigRef { id: idx }))
                }
                _ => None,
            })
            .collect()
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Backward {
            return Err(anyhow!(
                "Backward direction not supported for RemoveFalseAndTransform"
            ));
        }
        match candidate_location {
            TransformLocation::Node(r) => {
                remove_false_and_primitive(g, *r).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location for RemoveFalseAndTransform: {:?}",
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
    use crate::gate::AigNode;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::setup_simple_graph;

    #[test]
    fn test_insert_and_remove_false_and_primitive_round_trip() {
        let test = setup_simple_graph();
        let mut g = test.g.clone();
        let target = test.i0.node;
        let new_ref = insert_false_and_primitive(&mut g, target).unwrap();
        assert!(matches!(g.gates[new_ref.id], AigNode::And2 { .. }));
        remove_false_and_primitive(&mut g, new_ref).unwrap();
        assert_eq!(g.to_string(), test.g.to_string());
    }

    #[test]
    fn test_remove_false_and_primitive_rewrites_uses() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i0.negate());
        gb.add_output("o".to_string(), and_op.into());
        let mut g = gb.build();
        remove_false_and_primitive(&mut g, and_op.node).unwrap();
        let out_op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(out_op.node.id, 0);
        assert!(!out_op.negated);
    }

    #[test]
    fn test_insert_false_and_transform() {
        let test = setup_simple_graph();
        let mut g = test.g.clone();
        let mut t = InsertFalseAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert!(!cands.is_empty());
        let cand = &cands[0];
        let old_len = g.gates.len();
        t.apply(&mut g, cand, TransformDirection::Forward).unwrap();
        assert_eq!(g.gates.len(), old_len + 1);
    }

    #[test]
    fn test_remove_false_and_transform() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i0.negate());
        gb.add_output("o".to_string(), and_op.into());
        let mut g = gb.build();
        let mut t = RemoveFalseAndTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(cands.len(), 1);
        t.apply(&mut g, &cands[0], TransformDirection::Forward)
            .unwrap();
        let out_op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(out_op.node.id, 0);
        assert!(!out_op.negated);
    }
}
