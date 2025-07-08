// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{Result, anyhow};

/// Creates a new AND gate that ANDs the given operand with the constant TRUE.
///
/// Returns the `AigRef` of the newly created gate.
pub fn insert_true_and_primitive(g: &mut GateFn, op: AigOperand) -> AigRef {
    let true_op = AigOperand {
        node: AigRef { id: 0 }, // Node 0 is constant false
        negated: true,          // so negated it becomes constant true
    };
    let new_gate = AigNode::And2 {
        a: op,
        b: true_op,
        tags: None,
    };
    let new_ref = AigRef { id: g.gates.len() };
    g.gates.push(new_gate);
    new_ref
}

/// Collapses an `AND(x, true)` gate. All references to the gate are rewritten
/// to use `x` directly.
pub fn remove_true_and_primitive(g: &mut GateFn, node_ref: AigRef) -> Result<(), &'static str> {
    let true_op = AigOperand {
        node: AigRef { id: 0 },
        negated: true,
    };
    let inner_operand = match g.gates[node_ref.id] {
        AigNode::And2 { a, b, .. } => {
            if b == true_op {
                a
            } else if a == true_op {
                b
            } else {
                return Err("remove_true_and_primitive: node is not AND(x,true)");
            }
        }
        _ => return Err("remove_true_and_primitive: node is not And2"),
    };

    // Rewrite fan-ins of all gates that use `node_ref`
    for gate_idx in 0..g.gates.len() {
        if gate_idx == node_ref.id {
            continue;
        } // Don't modify the node being removed
        if let AigNode::And2 { a, b, .. } = &mut g.gates[gate_idx] {
            if a.node == node_ref {
                *a = AigOperand {
                    node: inner_operand.node,
                    negated: a.negated ^ inner_operand.negated,
                };
            }
            if b.node == node_ref {
                *b = AigOperand {
                    node: inner_operand.node,
                    negated: b.negated ^ inner_operand.negated,
                };
            }
        }
    }

    // Rewrite outputs that use `node_ref`
    for out in &mut g.outputs {
        for idx in 0..out.get_bit_count() {
            let bit = *out.bit_vector.get_lsb(idx);
            if bit.node == node_ref {
                out.bit_vector.set_lsb(
                    idx,
                    AigOperand {
                        node: inner_operand.node,
                        negated: bit.negated ^ inner_operand.negated,
                    },
                );
            }
        }
    }
    // Note: The original node `node_ref` is now dead. DCE will clean it up.
    Ok(())
}

// --- InsertTrueAnd Transform ---

#[derive(Debug)]
pub struct InsertTrueAndTransform;

impl InsertTrueAndTransform {
    pub fn new() -> Self {
        InsertTrueAndTransform
    }
}

impl Transform for InsertTrueAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::InsertTrueAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new(); // This transform is for insertion
        }

        let mut candidates = Vec::new();
        // 1. Primary outputs (any bit can be wrapped)
        for (out_idx, output_spec) in g.outputs.iter().enumerate() {
            for bit_idx in 0..output_spec.bit_vector.get_bit_count() {
                candidates.push(TransformLocation::OutputPortBit {
                    output_idx: out_idx,
                    bit_idx,
                });
            }
        }

        // 2. Internal And2 fan-ins (any operand of an And2 can be wrapped)
        for (parent_idx, node) in g.gates.iter().enumerate() {
            if matches!(node, AigNode::And2 { .. }) {
                let parent_ref = AigRef { id: parent_idx };
                // Only consider non-constant inputs for wrapping
                if let AigNode::And2 { a, b, .. } = node {
                    if a.node.id != 0 {
                        // Node 0 is constant
                        candidates.push(TransformLocation::Operand(parent_ref, false));
                        // LHS
                    }
                    if b.node.id != 0 {
                        candidates.push(TransformLocation::Operand(parent_ref, true));
                        // RHS
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
            return Err(anyhow!(
                "Backward direction not supported for InsertTrueAndTransform"
            ));
        }

        match candidate_location {
            TransformLocation::OutputPortBit {
                output_idx,
                bit_idx,
            } => {
                if *output_idx >= g.outputs.len()
                    || *bit_idx >= g.outputs[*output_idx].bit_vector.get_bit_count()
                {
                    return Err(anyhow!(
                        "OutputPortBit location out of bounds: {:?}",
                        candidate_location
                    ));
                }
                let operand_to_wrap = *g.outputs[*output_idx].bit_vector.get_lsb(*bit_idx);
                let new_and_ref = insert_true_and_primitive(g, operand_to_wrap);
                let new_op = AigOperand {
                    node: new_and_ref,
                    negated: false,
                };
                g.outputs[*output_idx].bit_vector.set_lsb(*bit_idx, new_op);
                Ok(())
            }
            TransformLocation::Operand(parent_ref, is_rhs) => {
                if parent_ref.id >= g.gates.len() {
                    return Err(anyhow!(
                        "Operand parent_ref out of bounds: {:?}",
                        parent_ref
                    ));
                }
                let operand_to_wrap = match &g.gates[parent_ref.id] {
                    AigNode::And2 { a, b, .. } => {
                        if *is_rhs {
                            *b
                        } else {
                            *a
                        }
                    }
                    _ => {
                        return Err(anyhow!(
                            "Operand parent_ref {:?} is not an And2 gate",
                            parent_ref
                        ));
                    }
                };
                if operand_to_wrap.node.id == 0 {
                    // Do not wrap constant false/true
                    return Err(anyhow!(
                        "Attempted to wrap a constant operand with TrueAnd: {:?}",
                        operand_to_wrap
                    ));
                }
                let new_and_ref = insert_true_and_primitive(g, operand_to_wrap);
                let new_op = AigOperand {
                    node: new_and_ref,
                    negated: false,
                };
                match &mut g.gates[parent_ref.id] {
                    AigNode::And2 { a, b, .. } => {
                        if *is_rhs {
                            *b = new_op;
                        } else {
                            *a = new_op;
                        }
                        Ok(())
                    }
                    _ => unreachable!(), // Should have been caught above
                }
            }
            _ => Err(anyhow!(
                "Invalid location type for InsertTrueAndTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

// --- RemoveTrueAnd Transform ---

#[derive(Debug)]
pub struct RemoveTrueAndTransform;

impl RemoveTrueAndTransform {
    pub fn new() -> Self {
        RemoveTrueAndTransform
    }
}

impl Transform for RemoveTrueAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RemoveTrueAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new(); // This transform is for removal
        }
        let true_op = AigOperand {
            node: AigRef { id: 0 },
            negated: true,
        };
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| match node {
                AigNode::And2 { a, b, .. } if *a == true_op || *b == true_op => {
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
                "Backward direction not supported for RemoveTrueAndTransform"
            ));
        }

        match candidate_location {
            TransformLocation::Node(target_ref) => {
                remove_true_and_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location type for RemoveTrueAndTransform: {:?}",
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
    // use crate::test_utils::simplify_aig_string_via_dce; // Removed for now

    #[test]
    fn test_insert_true_and_primitive_on_output() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let original_g_string = g.to_string(); // Compare raw strings for now

        let output_operand = g.outputs[0].bit_vector.get_lsb(0).clone();
        let new_and_ref = insert_true_and_primitive(&mut g, output_operand);
        g.outputs[0].bit_vector.set_lsb(
            0,
            AigOperand {
                node: new_and_ref,
                negated: false,
            },
        );

        assert_ne!(g.to_string(), original_g_string);
        match g.gates.last().unwrap() {
            AigNode::And2 { a, b, .. } => {
                assert_eq!(*a, output_operand);
                assert_eq!(
                    *b,
                    AigOperand {
                        node: AigRef { id: 0 },
                        negated: true
                    }
                );
            }
            _ => panic!("Last gate was not an And2"),
        }
        assert_eq!(g.outputs[0].bit_vector.get_lsb(0).node, new_and_ref);
    }

    #[test]
    fn test_remove_true_and_primitive_simple() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let true_val = AigOperand {
            node: AigRef { id: 0 },
            negated: true,
        };
        let and_true_op = gb.add_and_binary(i0, true_val);
        gb.add_output("o".to_string(), and_true_op.into());
        let mut g = gb.build();
        let original_output_op_before_removal = g.outputs[0].bit_vector.get_lsb(0).clone();
        assert_eq!(original_output_op_before_removal.node, and_true_op.node);

        remove_true_and_primitive(&mut g, and_true_op.node).unwrap();
        let final_output_op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(final_output_op.node, i0.node);
        assert_eq!(final_output_op.negated, i0.negated ^ and_true_op.negated); // RHS was true, so effective negation comes from and_true_op
    }

    #[test]
    fn test_insert_true_and_transform_applies_to_output() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let original_g_string = g.to_string();

        let mut transform = InsertTrueAndTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert!(!candidates.is_empty());

        let output_candidate = candidates
            .iter()
            .find(|loc| {
                matches!(
                    loc,
                    TransformLocation::OutputPortBit {
                        output_idx: 0,
                        bit_idx: 0
                    }
                )
            })
            .unwrap();
        transform
            .apply(&mut g, output_candidate, TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original_g_string);
        // Check output points to AND(i0, TRUE)
        let output_op = g.outputs[0].bit_vector.get_lsb(0);
        match g.gates[output_op.node.id] {
            AigNode::And2 { a, b, .. } => {
                assert_eq!(a.node, i0.node); // Assuming i0 was the original output operand
                assert_eq!(b.node.id, 0); // Constant node
                assert!(b.negated); // Negated to be TRUE
            }
            _ => panic!("Output does not point to an And2 gate after transform"),
        }
    }

    #[test]
    fn test_remove_true_and_transform_applies() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let true_val = AigOperand {
            node: AigRef { id: 0 },
            negated: true,
        };
        let and_true_op = gb.add_and_binary(i0, true_val.clone());
        gb.add_output("o".to_string(), and_true_op.into());
        let mut g = gb.build();
        let original_g_string = g.to_string();
        let and_true_ref = and_true_op.node;

        let mut transform = RemoveTrueAndTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0], TransformLocation::Node(r) if r.id == and_true_ref.id));

        transform
            .apply(&mut g, &candidates[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(g.to_string(), original_g_string);
        let final_output_op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(final_output_op.node, i0.node);
    }

    #[test]
    fn test_insert_true_and_on_constant_input_fails_gracefully_in_apply() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let const_false_ref = AigRef { id: 0 };
        // Create AND(i0, FALSE)
        let and_gate_op = gb.add_and_binary(
            i0,
            AigOperand {
                node: const_false_ref,
                negated: false,
            },
        );
        gb.add_output("o".to_string(), and_gate_op.into());
        let mut g = gb.build();

        let transform = InsertTrueAndTransform::new();
        // Candidate should be Operand(and_gate_ref, is_rhs=true) which refers to
        // const_false_ref
        let candidate_loc = TransformLocation::Operand(and_gate_op.node, true);
        let result = transform.apply(&mut g, &candidate_loc, TransformDirection::Forward);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Attempted to wrap a constant operand")
        );
    }

    #[test]
    fn test_insert_true_and_find_candidates_avoids_constants() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let const_false_op = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        // Create AND(i0, FALSE)
        let and_gate_op = gb.add_and_binary(i0, const_false_op);
        gb.add_output("o".to_string(), and_gate_op.into());
        let g = gb.build();

        let mut transform = InsertTrueAndTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        // Expected candidates:
        // 1. OutputPortBit for o[0] (refers to and_gate_op)
        // 2. Operand for and_gate_op's LHS (i0)
        // *NOT* Operand for and_gate_op's RHS (const_false_op)
        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .any(|loc| matches!(loc, TransformLocation::OutputPortBit { .. }))
        );
        assert!(candidates.iter().any(|loc| match loc {
            TransformLocation::Operand(parent_ref, is_rhs) =>
                parent_ref.id == and_gate_op.node.id && !*is_rhs, // Should be LHS (i0)
            _ => false,
        }));
        assert!(!candidates.iter().any(|loc| match loc {
            // RHS (const_false_op) should not be a candidate
            TransformLocation::Operand(parent_ref, is_rhs) =>
                parent_ref.id == and_gate_op.node.id && *is_rhs,
            _ => false,
        }));
    }
}
