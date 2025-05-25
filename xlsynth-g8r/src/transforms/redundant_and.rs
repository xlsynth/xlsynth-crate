// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Creates a new AND gate that feeds the given operand twice.
///
/// Returns the `AigRef` of the newly created gate.
pub fn insert_redundant_and_primitive(g: &mut GateFn, op: AigOperand) -> AigRef {
    let new_gate = AigNode::And2 {
        a: op,
        b: op,
        tags: None,
    };
    let new_ref = AigRef { id: g.gates.len() };
    g.gates.push(new_gate);
    // Strong post-condition: inserting a fresh AND(x,x) must never introduce
    // cycles.
    crate::topo::debug_assert_no_cycles(&g.gates, "insert_redundant_and_primitive");
    new_ref
}

/// Collapses a redundant AND gate of the form `AND(x, x)`.
/// All references to the gate are rewritten to use `x` directly.
pub fn remove_redundant_and_primitive(g: &mut GateFn, node: AigRef) -> Result<(), &'static str> {
    let (inner, _) = match g.gates[node.id] {
        AigNode::And2 { a, b, .. } => {
            if a != b {
                return Err("remove_redundant_and_primitive: node is not AND(x,x)");
            }
            (a, b)
        }
        _ => return Err("remove_redundant_and_primitive: node is not And2"),
    };

    // Rewrite fan-ins of all gates
    for gate_idx in 0..g.gates.len() {
        if gate_idx == node.id {
            continue;
        } // Don't modify the node being removed
        if let AigNode::And2 { a, b, .. } = &mut g.gates[gate_idx] {
            if a.node == node {
                *a = AigOperand {
                    node: inner.node,
                    negated: a.negated ^ inner.negated,
                };
            }
            if b.node == node {
                *b = AigOperand {
                    node: inner.node,
                    negated: b.negated ^ inner.negated,
                };
            }
        }
    }

    // Rewrite outputs
    for out in &mut g.outputs {
        for idx in 0..out.get_bit_count() {
            let bit = *out.bit_vector.get_lsb(idx);
            if bit.node == node {
                out.bit_vector.set_lsb(
                    idx,
                    AigOperand {
                        node: inner.node,
                        negated: bit.negated ^ inner.negated,
                    },
                );
            }
        }
    }
    // Note: The original node `node` is now dead. DCE will clean it up.
    Ok(())
}

#[derive(Debug)]
pub struct InsertRedundantAndTransform;

impl InsertRedundantAndTransform {
    pub fn new() -> Self {
        InsertRedundantAndTransform
    }
}

impl Transform for InsertRedundantAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::InsertRedundantAnd
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
                candidates.push(TransformLocation::Operand(parent_ref, false)); // LHS
                candidates.push(TransformLocation::Operand(parent_ref, true)); // RHS
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
                "Backward direction not supported for InsertRedundantAndTransform"
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
                let new_and_ref = insert_redundant_and_primitive(g, operand_to_wrap);
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
                        ))
                    }
                };
                let new_and_ref = insert_redundant_and_primitive(g, operand_to_wrap);
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
                "Invalid location type for InsertRedundantAndTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct RemoveRedundantAndTransform;

impl RemoveRedundantAndTransform {
    pub fn new() -> Self {
        RemoveRedundantAndTransform
    }
}

impl Transform for RemoveRedundantAndTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::RemoveRedundantAnd
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Backward {
            return Vec::new(); // Or delegate to
                               // InsertRedundantAnd.find_candidates
        }
        g.gates
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| match node {
                AigNode::And2 { a, b, .. } if a == b => {
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
                "Backward direction not supported for RemoveRedundantAndTransform"
            ));
        }

        match candidate_location {
            TransformLocation::Node(target_ref) => {
                remove_redundant_and_primitive(g, *target_ref).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location type for RemoveRedundantAndTransform: {:?}",
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
    // use rand::rngs::StdRng; // No longer needed if the specific test is removed
    // use rand::SeedableRng; // No longer needed
    use crate::transforms::transform_trait::Transform; // For new trait tests

    #[test]
    fn test_insert_and_remove_self_inverse_primitive() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a_op.into());
        let g1 = gb.build();

        let mut g2 = g1.clone();
        // Directly test primitive: insert redundant AND for the output operand
        let output_operand = g2.outputs[0].bit_vector.get_lsb(0).clone();
        let new_ref = insert_redundant_and_primitive(&mut g2, output_operand);

        // Manually rewire the output to point to the new redundant AND
        g2.outputs[0].bit_vector.set_lsb(
            0,
            AigOperand {
                node: new_ref,
                negated: false,
            },
        );

        remove_redundant_and_primitive(&mut g2, new_ref).unwrap();
        assert_eq!(g1.to_string(), g2.to_string());
    }

    // test_insert_redundant_and_rand_round_trip is removed as its logic is covered
    // by MCMC testing of the Transform trait implementations, or should be
    // tested via direct calls to the new Transform trait methods if specific
    // scenarios are desired.

    #[test]
    fn test_remove_redundant_and_primitive_on_non_redundant_fails() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a_op.into());
        let mut g = gb.build();
        let result = remove_redundant_and_primitive(&mut g, a_op.node);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "remove_redundant_and_primitive: node is not AND(x,x)"
        );
    }

    #[test]
    fn test_remove_redundant_and_primitive_on_non_and_fails() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0_op = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0_op.into());
        let mut g = gb.build();
        let result = remove_redundant_and_primitive(&mut g, i0_op.node);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "remove_redundant_and_primitive: node is not And2"
        );
    }

    #[test]
    fn test_insert_redundant_and_transform_finds_candidates_and_applies() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_op.into()); // Output 0, bit 0 refers to and_op
        let mut g = gb.build();
        let original_g_string = g.to_string();

        let mut transform = InsertRedundantAndTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);

        // Expected candidates:
        // 1. OutputPortBit for o[0]
        // 2. Operand for and_op's LHS (i0)
        // 3. Operand for and_op's RHS (i1)
        assert_eq!(candidates.len(), 3, "Expected 3 candidates for insertion");
        assert!(candidates.iter().any(|loc| matches!(
            loc,
            TransformLocation::OutputPortBit {
                output_idx: 0,
                bit_idx: 0
            }
        )));
        assert!(candidates.iter().any(
            |loc| matches!(loc, TransformLocation::Operand(r, false) if r.id == and_op.node.id)
        ));
        assert!(candidates.iter().any(
            |loc| matches!(loc, TransformLocation::Operand(r, true) if r.id == and_op.node.id)
        ));

        // Test applying to the output port bit
        let output_candidate = candidates
            .iter()
            .find(|loc| matches!(loc, TransformLocation::OutputPortBit { .. }))
            .unwrap();
        transform
            .apply(&mut g, output_candidate, TransformDirection::Forward)
            .unwrap();

        assert_ne!(
            g.to_string(),
            original_g_string,
            "Graph should change after inserting redundant AND"
        );
        // Verify the output now points to a new AND(x,x) gate
        let new_output_op = g.outputs[0].bit_vector.get_lsb(0);
        match &g.gates[new_output_op.node.id] {
            AigNode::And2 { a, b, .. } => {
                assert_eq!(a, b, "Inserted AND should be AND(x,x)");
                assert_eq!(
                    a.node, and_op.node,
                    "Inner operand of new AND should be original output operand"
                );
            }
            _ => panic!("Output does not point to an And2 gate after transform"),
        }
    }

    #[test]
    fn test_remove_redundant_and_transform_finds_candidates_and_applies() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        // Create AND(i0, i0)
        let redundant_and_op = gb.add_and_binary(i0, i0.clone());
        gb.add_output("o".to_string(), redundant_and_op.into());
        let mut g = gb.build();
        let original_g_string = g.to_string();
        let redundant_and_ref = redundant_and_op.node;

        let mut transform = RemoveRedundantAndTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(candidates.len(), 1, "Expected 1 candidate for removal");
        assert!(
            matches!(candidates[0], TransformLocation::Node(r) if r.id == redundant_and_ref.id)
        );

        transform
            .apply(&mut g, &candidates[0], TransformDirection::Forward)
            .unwrap();
        assert_ne!(
            g.to_string(),
            original_g_string,
            "Graph should change after removing redundant AND"
        );
        // Verify the output now points to i0 directly
        let final_output_op = g.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(final_output_op.node, i0.node);
        // The negation might change depending on how remove_redundant_and_primitive
        // handles it. If redundant_and_op was Op(redundant_ref, false), and
        // inner was i0 (false), then output is Op(i0_ref, false ^ false)
        assert_eq!(
            final_output_op.negated,
            i0.negated ^ redundant_and_op.negated
        );
    }

    #[test]
    fn test_remove_redundant_and_transform_no_candidates() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_op = gb.add_and_binary(i0, i1); // Not redundant
        gb.add_output("o".to_string(), and_op.into());
        let g = gb.build();

        let mut transform = RemoveRedundantAndTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert!(
            candidates.is_empty(),
            "Expected no candidates for removal in a graph with no redundant ANDs"
        );
    }
}
