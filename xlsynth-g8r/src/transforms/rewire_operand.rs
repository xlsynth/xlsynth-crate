// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};
use rand::Rng;

/// Primitive: rewires one operand of an `And2` gate to `new_op`.
/// Returns the previous operand value.
pub fn rewire_operand_primitive(
    g: &mut GateFn,
    parent: AigRef,
    is_rhs: bool,
    new_op: AigOperand,
) -> Result<AigOperand, &'static str> {
    if parent.id >= g.gates.len() {
        return Err("Parent ref out of bounds in rewire_operand_primitive");
    }
    match &mut g.gates[parent.id] {
        AigNode::And2 { a, b, .. } => {
            let old = if is_rhs { *b } else { *a };
            if is_rhs {
                *b = new_op;
            } else {
                *a = new_op;
            }
            Ok(old)
        }
        _ => Err("Parent is not And2 in rewire_operand_primitive"),
    }
}

#[derive(Debug, Clone)]
pub struct RewireOperandLocation {
    pub parent: AigRef,
    pub is_rhs: bool,
    pub old_op: AigOperand,
    pub new_op: AigOperand,
}

#[derive(Debug)]
pub struct RewireOperandTransform;

impl RewireOperandTransform {
    pub fn new() -> Self {
        RewireOperandTransform
    }
}

impl Transform for RewireOperandTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RewireOperand
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        let mut rng = rand::thread_rng();
        let mut cands = Vec::new();
        for (idx, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { a, b, .. } = node {
                let parent = AigRef { id: idx };
                let new_op_lhs = AigOperand {
                    node: AigRef {
                        id: rng.gen_range(0..g.gates.len()),
                    },
                    negated: rng.gen(),
                };
                cands.push(TransformLocation::Custom(Box::new(RewireOperandLocation {
                    parent,
                    is_rhs: false,
                    old_op: *a,
                    new_op: new_op_lhs,
                })));
                let new_op_rhs = AigOperand {
                    node: AigRef {
                        id: rng.gen_range(0..g.gates.len()),
                    },
                    negated: rng.gen(),
                };
                cands.push(TransformLocation::Custom(Box::new(RewireOperandLocation {
                    parent,
                    is_rhs: true,
                    old_op: *b,
                    new_op: new_op_rhs,
                })));
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
        let loc = match candidate_location {
            TransformLocation::Custom(b) => b
                .downcast_ref::<RewireOperandLocation>()
                .ok_or_else(|| anyhow!("Invalid location type for RewireOperandTransform"))?,
            _ => {
                return Err(anyhow!(
                    "Invalid candidate location for RewireOperandTransform: {:?}",
                    candidate_location
                ))
            }
        };
        let target_op = if direction == TransformDirection::Forward {
            loc.new_op
        } else {
            loc.old_op
        };
        rewire_operand_primitive(g, loc.parent, loc.is_rhs, target_op)
            .map(|_| ())
            .map_err(anyhow::Error::msg)
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn test_rewire_operand_primitive_round_trip() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_op.into());
        let mut g = gb.build();
        let new_op = i2;
        let old = rewire_operand_primitive(&mut g, and_op.node, false, new_op).unwrap();
        match &g.gates[and_op.node.id] {
            AigNode::And2 { a, .. } => {
                assert_eq!(*a, new_op);
            }
            _ => panic!("and_op not And2"),
        }
        // revert
        rewire_operand_primitive(&mut g, and_op.node, false, old).unwrap();
        assert_eq!(g.outputs[0].bit_vector.get_lsb(0).node, and_op.node);
    }

    #[test]
    fn test_rewire_operand_transform_apply_forward_backward() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_op.into());
        let mut g = gb.build();
        let loc = RewireOperandLocation {
            parent: and_op.node,
            is_rhs: true,
            old_op: i1,
            new_op: i2,
        };
        let tloc = TransformLocation::Custom(Box::new(loc.clone()));
        let t = RewireOperandTransform::new();
        t.apply(&mut g, &tloc, TransformDirection::Forward).unwrap();
        match &g.gates[and_op.node.id] {
            AigNode::And2 { b, .. } => assert_eq!(*b, i2),
            _ => panic!("not and2"),
        }
        t.apply(&mut g, &tloc, TransformDirection::Backward)
            .unwrap();
        match &g.gates[and_op.node.id] {
            AigNode::And2 { b, .. } => assert_eq!(*b, i1),
            _ => panic!("not and2"),
        }
    }
}
