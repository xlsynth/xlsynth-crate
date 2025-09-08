// SPDX-License-Identifier: Apache-2.0

//! Functionality for proposing equivalence classes via concrete simulation.

use crate::gate::{AigNode, AigRef, GateFn};
use crate::gate_sim::{self, Collect, GateSimResult};
use bitvec::vec::BitVec;
use xlsynth::IrBits;
use xlsynth_pir::fuzz_utils::arbitrary_irbits;

use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Represents a node within a proposed equivalence class, indicating whether
/// its simulation history matches the class directly (Normal) or inversely
/// (Inverted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquivNode {
    Normal(AigRef),
    Inverted(AigRef),
}

impl EquivNode {
    /// Returns the underlying AigRef.
    pub fn aig_ref(&self) -> AigRef {
        match self {
            EquivNode::Normal(r) => *r,
            EquivNode::Inverted(r) => *r,
        }
    }

    /// Returns true if the node represents an inverted history relationship.
    pub fn is_inverted(&self) -> bool {
        matches!(self, EquivNode::Inverted(_))
    }
}

// Implement ordering for EquivNode: Normal comes before Inverted, then compare
// AigRef.
impl PartialOrd for EquivNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EquivNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (EquivNode::Normal(a), EquivNode::Normal(b)) => a.cmp(b),
            (EquivNode::Inverted(a), EquivNode::Inverted(b)) => a.cmp(b),
            (EquivNode::Normal(_), EquivNode::Inverted(_)) => std::cmp::Ordering::Less,
            (EquivNode::Inverted(_), EquivNode::Normal(_)) => std::cmp::Ordering::Greater,
        }
    }
}

fn gen_random_inputs(gate_fn: &GateFn, rng: &mut impl Rng) -> Vec<IrBits> {
    gate_fn
        .inputs
        .iter()
        .map(|input| arbitrary_irbits(rng, input.bit_vector.get_bit_count()))
        .collect()
}

/// Returns a mapping from hash value (a hash over the history for a given gate
/// as it's fed random samples) to a sequence of the nodes that had the same
/// history.
pub fn propose_equivalence_classes(
    gate_fn: &GateFn,
    input_sample_count: usize,
    rng: &mut impl Rng,
    counterexamples: &HashSet<Vec<IrBits>>,
) -> HashMap<u64, Vec<EquivNode>> {
    // samples x gate values -- would be nicer to have a BitMatrix
    let gate_count = gate_fn.gates.len();
    let mut history: Vec<BitVec> = Vec::with_capacity(input_sample_count);

    // Push `input_sample_count` random samples through the gate function and
    // collect the history of all the nodes.
    for _ in 0..input_sample_count {
        let inputs: Vec<IrBits> = gen_random_inputs(gate_fn, rng);
        let result: GateSimResult = gate_sim::eval(gate_fn, &inputs, Collect::All);
        history.push(result.all_values.unwrap());
    }

    // Now do it for all explicitly-provided counterexamples.
    for counterexample in counterexamples {
        let result: GateSimResult = gate_sim::eval(gate_fn, counterexample, Collect::All);
        history.push(result.all_values.unwrap());
    }

    // Collects the history for the given `gate_index` across all samples.
    let collect_across_samples = |gate_index: usize| -> BitVec {
        history.iter().map(|h| -> bool { h[gate_index] }).collect()
    };

    // Calculate normal and inverse hashes in a single pass.
    let node_hashes: Vec<(u64, u64)> = (0..gate_count)
        .map(|i| {
            let mut gate_history = collect_across_samples(i);

            // Calculate normal hash
            let mut normal_hasher = DefaultHasher::new();
            gate_history.hash(&mut normal_hasher);
            let normal_hash = normal_hasher.finish();

            // Calculate inverse hash
            gate_history.iter_mut().for_each(|mut bit| *bit = !*bit);
            let mut inverse_hasher = DefaultHasher::new();
            gate_history.hash(&mut inverse_hasher);
            let inverse_hash = inverse_hasher.finish();

            (normal_hash, inverse_hash)
        })
        .collect();

    // Collect all potential hash entries (normal and inverted) for
    // non-input/literal nodes.
    let all_potential_equivs: Vec<(u64, EquivNode)> = (0..gate_count)
        .filter_map(|node_index| {
            let node = &gate_fn.gates[node_index];
            if matches!(node, AigNode::Input { .. } | AigNode::Literal(..)) {
                None // Skip inputs and literals
            } else {
                let (normal_hash, inverse_hash) = node_hashes[node_index];
                let node_ref = AigRef { id: node_index };
                Some(vec![
                    (normal_hash, EquivNode::Normal(node_ref)),
                    (inverse_hash, EquivNode::Inverted(node_ref)),
                ])
            }
        })
        .flatten()
        .collect();

    // Group by hash
    let mut grouped_classes: HashMap<u64, Vec<EquivNode>> = HashMap::new();
    for (hash, equiv_node) in all_potential_equivs {
        grouped_classes
            .entry(hash)
            .or_insert_with(Vec::new)
            .push(equiv_node);
    }

    // Filter out singleton classes and sort remaining classes.
    let equiv_classes: HashMap<u64, Vec<EquivNode>> = grouped_classes
        .into_iter()
        .filter(|(_, nodes)| nodes.len() > 1)
        .map(|(hash, mut nodes)| {
            nodes.sort_unstable(); // Sort for deterministic output within the class list
            (hash, nodes)
        })
        .collect();

    equiv_classes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{setup_graph_with_redundancies, setup_simple_graph};
    use rand::SeedableRng;

    #[test]
    fn test_propose_equiv_simple_graph() {
        let _ = env_logger::builder().is_test(true).try_init();
        let graph = setup_simple_graph();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let counterexamples = HashSet::new();
        let equiv_classes =
            propose_equivalence_classes(&graph.g, 4096, &mut seeded_rng, &counterexamples);
        // There are no redundancies in this graph, so we should not find any
        // equivalence classes.
        assert!(equiv_classes.is_empty());
    }

    #[test]
    fn test_propose_equiv_graph_with_redundancies() {
        let _ = env_logger::builder().is_test(true).try_init();
        let graph = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let counterexamples = HashSet::new();
        let equiv_classes =
            propose_equivalence_classes(&graph.g, 4096, &mut seeded_rng, &counterexamples);
        log::info!("equiv_classes: {:?}", equiv_classes);
        assert_eq!(equiv_classes.len(), 4); // Expect 4 classes: normal pairs and inverted pairs
        let mut values = equiv_classes.values().collect::<Vec<_>>();
        // Sort them so we can do stable tests.
        values.sort();
        let want = vec![
            vec![
                // Normal equivalence inner0 == inner1
                EquivNode::Normal(graph.inner0.node),
                EquivNode::Normal(graph.inner1.node),
            ],
            vec![
                // Normal equivalence outer0 == outer1
                EquivNode::Normal(graph.outer0.node),
                EquivNode::Normal(graph.outer1.node),
            ],
            vec![
                // Inverted equivalence !inner0 == !inner1
                EquivNode::Inverted(graph.inner0.node),
                EquivNode::Inverted(graph.inner1.node),
            ],
            vec![
                // Inverted equivalence !outer0 == !outer1
                EquivNode::Inverted(graph.outer0.node),
                EquivNode::Inverted(graph.outer1.node),
            ],
        ];
        assert_eq!(
            values
                .iter()
                .map(|v| {
                    let mut sorted_v = (*v).clone();
                    sorted_v.sort();
                    sorted_v
                })
                .collect::<Vec<_>>(),
            want
        );
    }
}
