// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigRef, GateFn};
use crate::topo::topo_sort_refs;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use std::collections::HashMap;

// Renamed from duplicate_internal to avoid conflict with pub fn duplicate
// This is the version used by DuplicateGateTransform
fn duplicate_gate_transform_internal(
    g: &mut GateFn,
    which: AigRef,
) -> Result<AigRef, &'static str> {
    if which.id >= g.gates.len() {
        return Err("cannot duplicate: AigRef out of bounds");
    }
    match &g.gates[which.id] {
        AigNode::Literal(_) => return Err("cannot duplicate literal"),
        AigNode::Input { .. } => return Err("cannot duplicate input"),
        AigNode::And2 { a, b, .. } => {
            let new_gate = AigNode::And2 {
                a: *a,
                b: *b,
                tags: None, // Duplicated gates don't inherit tags by default
            };
            let new_ref = AigRef { id: g.gates.len() };
            g.gates.push(new_gate);
            Ok(new_ref)
        }
    }
}

/// Duplicate (replicate) a gate inside a `GateFn`.
///
/// The newly created gate is structurally *identical* to the one referenced by
/// `which`, i.e. it is created with the same kind and **re-uses** the very same
/// fan-ins.  The function deliberately leaves the list of designated outputs
/// untouched so that callers can decide on their own whether or not the fresh
/// gate should become observable.
///
/// Attempting to duplicate primary inputs or constants has no semantic value
/// and is therefore rejected with an error.
pub fn duplicate(g: &mut GateFn, which: AigRef) -> Result<AigRef, &'static str> {
    if which.id >= g.gates.len() {
        return Err("cannot duplicate: AigRef out of bounds");
    }
    match &g.gates[which.id] {
        AigNode::Literal(_) => return Err("cannot duplicate literal"),
        AigNode::Input { .. } => return Err("cannot duplicate input"),
        AigNode::And2 { a, b, .. } => {
            let new_gate = AigNode::And2 {
                a: *a,
                b: *b,
                tags: None,
            };
            let new_ref = AigRef { id: g.gates.len() };
            g.gates.push(new_gate);
            Ok(new_ref)
        }
    }
}

/// Attempts to unduplicate a structurally redundant node in the graph.
/// Returns the AigRef that should now be dead (all references replaced), or
/// None if no unduplication was possible.
pub fn unduplicate<R: rand::Rng + ?Sized>(g: &mut GateFn, rng: &mut R) -> Option<AigRef> {
    let topo = topo_sort_refs(&g.gates);
    let mut buckets: HashMap<String, Vec<AigRef>> = HashMap::new();
    for &node_ref in &topo {
        match &g.gates[node_ref.id] {
            AigNode::Input { .. } => continue,
            AigNode::Literal(val) => {
                let key = format!("Literal({})", val);
                buckets.entry(key).or_default().push(node_ref);
            }
            AigNode::And2 { a, b, .. } => {
                let key = format!("And2({:?},{:?})", a, b);
                buckets.entry(key).or_default().push(node_ref);
            }
        }
    }
    let candidates: Vec<_> = buckets.values().filter(|v| v.len() >= 2).collect();
    if candidates.is_empty() {
        return None;
    }
    let bucket = candidates.choose(rng).unwrap();
    debug_assert!(bucket.len() >= 2, "Bucket should have at least 2 nodes");
    let mut pair = bucket.iter().copied().collect::<Vec<_>>();
    pair.shuffle(rng);
    let (keep, kill) = (pair[0], pair[1]);
    for node_idx in 0..g.gates.len() {
        if node_idx == kill.id {
            continue;
        }
        if let AigNode::And2 { a, b, .. } = &mut g.gates[node_idx] {
            if a.node == kill {
                a.node = keep;
            }
            if b.node == kill {
                b.node = keep;
            }
        }
    }
    for output_spec in &mut g.outputs {
        let mut modified_ops = false;
        let mut current_ops: Vec<_> = output_spec.bit_vector.iter_lsb_to_msb().copied().collect();
        for op_idx in 0..current_ops.len() {
            if current_ops[op_idx].node == kill {
                current_ops[op_idx].node = keep;
                modified_ops = true;
            }
        }
        if modified_ops {
            output_spec.bit_vector = crate::gate::AigBitVector::from_lsb_is_index_0(&current_ops);
        }
    }
    Some(kill)
}

fn get_and2_key(g: &GateFn, node_ref: AigRef) -> Option<String> {
    if node_ref.id >= g.gates.len() {
        return None;
    }
    match &g.gates[node_ref.id] {
        AigNode::And2 { a, b, .. } => {
            if (a.node.id < b.node.id) || (a.node.id == b.node.id && !a.negated && b.negated) {
                Some(format!("And2({:?},{:?})", a, b))
            } else if (b.node.id < a.node.id) || (a.node.id == b.node.id && !b.negated && a.negated)
            {
                Some(format!("And2({:?},{:?})", b, a))
            } else {
                Some(format!("And2({:?},{:?})", a, b))
            }
        }
        _ => None,
    }
}

#[derive(Debug)]
pub struct DuplicateGateTransform;

impl DuplicateGateTransform {
    pub fn new() -> Self {
        DuplicateGateTransform
    }
}

impl Transform for DuplicateGateTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::DuplicateGate
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Forward {
            g.gates
                .iter()
                .enumerate()
                .filter_map(|(idx, node)| match node {
                    AigNode::And2 { .. } => Some(TransformLocation::Node(AigRef { id: idx })),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Forward {
            match candidate_location {
                TransformLocation::Node(target_ref) => {
                    duplicate_gate_transform_internal(g, *target_ref)
                        .map(|_| ())
                        .map_err(anyhow::Error::msg)
                }
                _ => Err(anyhow!(
                    "Invalid candidate location for DuplicateGateTransform: {:?}",
                    candidate_location
                )),
            }
        } else {
            Err(anyhow!(
                "Backward direction not supported for DuplicateGateTransform"
            ))
        }
    }
}

#[derive(Debug)]
pub struct UnduplicateGateTransform;

impl UnduplicateGateTransform {
    pub fn new() -> Self {
        UnduplicateGateTransform
    }
}

impl Transform for UnduplicateGateTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::UnduplicateGate
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        if direction == TransformDirection::Forward {
            let mut candidates = Vec::new();
            let mut buckets: HashMap<String, Vec<AigRef>> = HashMap::new();

            for (idx, node) in g.gates.iter().enumerate() {
                if matches!(node, AigNode::And2 { .. }) {
                    if let Some(key) = get_and2_key(g, AigRef { id: idx }) {
                        buckets.entry(key).or_default().push(AigRef { id: idx });
                    }
                }
            }
            for bucket in buckets.values() {
                if bucket.len() >= 2 {
                    for node_ref in bucket {
                        candidates.push(TransformLocation::Node(*node_ref));
                    }
                }
            }
            candidates
        } else {
            Vec::new()
        }
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()> {
        if direction == TransformDirection::Forward {
            match candidate_location {
                TransformLocation::Node(potential_kill_ref) => {
                    if potential_kill_ref.id >= g.gates.len()
                        || !matches!(g.gates[potential_kill_ref.id], AigNode::And2 { .. })
                    {
                        return Err(anyhow!(
                            "Candidate to kill {:?} is out of bounds or not an And2 gate",
                            potential_kill_ref
                        ));
                    }

                    let kill_key = get_and2_key(g, *potential_kill_ref).ok_or_else(|| {
                        anyhow!(
                            "Could not get key for candidate to kill: {:?}",
                            potential_kill_ref
                        )
                    })?;

                    let mut potential_keep_ref: Option<AigRef> = None;
                    for (idx, _) in g.gates.iter().enumerate() {
                        let current_gate_ref = AigRef { id: idx };
                        if current_gate_ref == *potential_kill_ref {
                            continue;
                        }
                        if matches!(g.gates[idx], AigNode::And2 { .. }) {
                            if let Some(key) = get_and2_key(g, current_gate_ref) {
                                if key == kill_key {
                                    potential_keep_ref = Some(current_gate_ref);
                                    break;
                                }
                            }
                        }
                    }

                    if let Some(keep_ref) = potential_keep_ref {
                        for node_idx_iter in 0..g.gates.len() {
                            if node_idx_iter == potential_kill_ref.id
                                || node_idx_iter == keep_ref.id
                            {
                                continue;
                            }

                            if let AigNode::And2 { a, b, .. } = &mut g.gates[node_idx_iter] {
                                if a.node == *potential_kill_ref {
                                    a.node = keep_ref;
                                }
                                if b.node == *potential_kill_ref {
                                    b.node = keep_ref;
                                }
                            }
                        }
                        for output_spec in &mut g.outputs {
                            let mut modified_ops = false;
                            let mut current_ops: Vec<_> =
                                output_spec.bit_vector.iter_lsb_to_msb().copied().collect();
                            for op_idx in 0..current_ops.len() {
                                if current_ops[op_idx].node == *potential_kill_ref {
                                    current_ops[op_idx].node = keep_ref;
                                    modified_ops = true;
                                }
                            }
                            if modified_ops {
                                output_spec.bit_vector =
                                    crate::gate::AigBitVector::from_lsb_is_index_0(&current_ops);
                            }
                        }
                        Ok(())
                    } else {
                        Err(anyhow!("No suitable gate found to merge with {:?}. It might be unique or an error occurred.", potential_kill_ref))
                    }
                }
                _ => Err(anyhow!(
                    "Invalid candidate location for UnduplicateGateTransform: {:?}",
                    candidate_location
                )),
            }
        } else {
            Err(anyhow!(
                "Backward direction not supported for UnduplicateGateTransform"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::AigRef;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::setup_simple_graph;

    #[test]
    fn test_duplicate_gate_transform_forward() {
        let test_graph_setup = setup_simple_graph();
        let mut g = test_graph_setup.g;
        let and_gate_ref = test_graph_setup.a.node;

        let mut transform = DuplicateGateTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        let target_loc = candidates
            .iter()
            .find(|loc| match loc {
                TransformLocation::Node(r) => *r == and_gate_ref,
                _ => false,
            })
            .expect("Target AND gate not found in candidates");

        let original_gate_count = g.gates.len();
        transform
            .apply(&mut g, &target_loc, TransformDirection::Forward)
            .unwrap();
        assert_eq!(
            g.gates.len(),
            original_gate_count + 1,
            "A gate should have been added"
        );

        let new_gate_ref = AigRef {
            id: original_gate_count,
        };
        match (&g.gates[and_gate_ref.id], &g.gates[new_gate_ref.id]) {
            (AigNode::And2 { a: a1, b: b1, .. }, AigNode::And2 { a: a2, b: b2, .. }) => {
                assert_eq!(a1.node, a2.node, "Fanin a node of duplicated gate mismatch");
                assert_eq!(
                    a1.negated, a2.negated,
                    "Fanin a negation of duplicated gate mismatch"
                );
                assert_eq!(b1.node, b2.node, "Fanin b node of duplicated gate mismatch");
                assert_eq!(
                    b1.negated, b2.negated,
                    "Fanin b negation of duplicated gate mismatch"
                );
            }
            _ => panic!(
                "Expected And2 gates, found: {:?} and {:?}",
                g.gates[and_gate_ref.id], g.gates[new_gate_ref.id]
            ),
        }
    }

    #[test]
    fn test_duplicate_gate_transform_no_candidates_for_input_graph() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0_bv = gb.add_input("i0".to_string(), 1);
        // Add a dummy output to allow GateFn to build
        // The test logic depends on the content of the graph (no And2s), not this
        // specific output.
        gb.add_output("dummy_out".to_string(), i0_bv.get_lsb(0).clone().into());
        let g = gb.build();
        let mut transform = DuplicateGateTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert!(
            candidates.is_empty(),
            "Should be no candidates for duplication in a graph with only inputs/literals"
        );
    }

    #[test]
    fn test_unduplicate_gate_transform_merges_identical_gates() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and1_op = gb.add_and_binary(i0, i1);
        let and2_op = gb.add_and_binary(i0, i1);
        let and3_op = gb.add_and_binary(and1_op, and2_op);
        gb.add_output("o1".to_string(), and1_op.into());
        gb.add_output("o2".to_string(), and2_op.into());
        gb.add_output("o3".to_string(), and3_op.into());
        let mut g = gb.build();

        let and1_ref = and1_op.node;
        let and2_ref = and2_op.node;
        let and3_ref = and3_op.node;

        let mut transform = UnduplicateGateTransform::new();

        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert_eq!(
            candidates.len(),
            2,
            "Expected two candidates for unduplication (and1, and2)"
        );
        assert!(candidates
            .iter()
            .any(|loc| matches!(loc, TransformLocation::Node(r) if *r == and1_ref)));
        assert!(candidates
            .iter()
            .any(|loc| matches!(loc, TransformLocation::Node(r) if *r == and2_ref)));

        let kill_loc = TransformLocation::Node(and2_ref);
        transform
            .apply(&mut g, &kill_loc, TransformDirection::Forward)
            .unwrap();

        let o2_output_spec = g.outputs.iter().find(|o| o.name == "o2").unwrap();
        assert_eq!(
            o2_output_spec.bit_vector.get_lsb(0).node,
            and1_ref,
            "Output o2 was not rewired to and1_ref"
        );

        match &g.gates[and3_ref.id] {
            AigNode::And2 { a, b, .. } => {
                assert_eq!(
                    a.node, and1_ref,
                    "Operand 'a' of and3_ref not rewired to and1_ref"
                );
                assert_eq!(
                    b.node, and1_ref,
                    "Operand 'b' of and3_ref not rewired to and1_ref"
                );
            }
            _ => panic!("and3_ref is not an And2 gate after transform"),
        }

        for (idx, gate_node) in g.gates.iter().enumerate() {
            if idx == and2_ref.id {
                continue;
            }
            if let AigNode::And2 { a, b, .. } = gate_node {
                assert_ne!(
                    a.node,
                    and2_ref,
                    "Gate {:?} still refers to killed node {:?} in operand a",
                    AigRef { id: idx },
                    and2_ref
                );
                assert_ne!(
                    b.node,
                    and2_ref,
                    "Gate {:?} still refers to killed node {:?} in operand b",
                    AigRef { id: idx },
                    and2_ref
                );
            }
        }
        for output_spec in &g.outputs {
            for op_in_output in output_spec.bit_vector.iter_lsb_to_msb() {
                assert_ne!(
                    op_in_output.node, and2_ref,
                    "Output {} still refers to killed node {:?}",
                    output_spec.name, and2_ref
                );
            }
        }
    }

    #[test]
    fn test_unduplicate_no_candidates_when_gates_differ_by_negation() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and1 = gb.add_and_binary(i0, i1);
        let and2 = gb.add_and_binary(i0, i1.negate()); // Different due to negation
        gb.add_output("o1".to_string(), and1.into());
        gb.add_output("o2".to_string(), and2.into());
        let g = gb.build();
        let mut transform = UnduplicateGateTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert!(
            candidates.is_empty(),
            "Expected no candidates as gates are not structurally identical"
        );
    }

    #[test]
    fn test_unduplicate_no_candidates_for_single_gate() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let and_gate = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), and_gate.into());
        let g = gb.build();
        let mut transform = UnduplicateGateTransform::new();
        let candidates = transform.find_candidates(&g, TransformDirection::Forward);
        assert!(
            candidates.is_empty(),
            "Expected no candidates for a single And2 gate"
        );
    }
}
