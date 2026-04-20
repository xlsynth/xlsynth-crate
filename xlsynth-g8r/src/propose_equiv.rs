// SPDX-License-Identifier: Apache-2.0

//! Functionality for proposing equivalence classes via concrete simulation.

use crate::aig::{AigNode, AigRef, GateFn};
use crate::aig_sim::gate_simd::{self, Vec256};
use xlsynth::IrBits;
use xlsynth_pir::fuzz_utils::arbitrary_irbits;

use rand::Rng;
use std::cmp::min;
use std::collections::{HashMap, HashSet};

const SIMD_LANES: usize = 256;
// Domain-separation labels for the BLAKE3 simulation signature hash chain.
const SIGNATURE_INIT_DOMAIN: &[u8] = b"xlsynth-g8r/fraig/simulation-signature/v1/init";
const SIGNATURE_UPDATE_DOMAIN: &[u8] = b"xlsynth-g8r/fraig/simulation-signature/v1/update";
const SIGNATURE_FINAL_DOMAIN: &[u8] = b"xlsynth-g8r/fraig/simulation-signature/v1/final";

/// A streaming 256-bit simulation signature for one node.
///
/// SAT validation is still the authority, so a hash collision can only create
/// extra candidates that are later rejected. Keeping only the BLAKE3 digest
/// avoids materializing the old samples-by-nodes history matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SimulationSignature([u8; 32]);

impl SimulationSignature {
    #[inline]
    fn new() -> Self {
        Self(*blake3::hash(SIGNATURE_INIT_DOMAIN).as_bytes())
    }

    #[inline]
    fn words_to_le_bytes(words: [u64; 4]) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (i, word) in words.into_iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    #[inline]
    fn update_words(&mut self, words: [u64; 4], batch_index: u64) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SIGNATURE_UPDATE_DOMAIN);
        hasher.update(&self.0);
        hasher.update(&batch_index.to_le_bytes());
        hasher.update(&Self::words_to_le_bytes(words));
        self.0 = *hasher.finalize().as_bytes();
    }

    #[inline]
    fn finalize(mut self, total_samples: usize) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SIGNATURE_FINAL_DOMAIN);
        hasher.update(&self.0);
        hasher.update(&total_samples.to_le_bytes());
        self.0 = *hasher.finalize().as_bytes();
        self
    }
}

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

fn total_input_bits(gate_fn: &GateFn) -> usize {
    gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum()
}

fn set_lane_inputs(
    gate_fn: &GateFn,
    lane: usize,
    inputs: &[IrBits],
    words_per_input_bit: &mut [[u64; 4]],
) {
    assert!(lane < SIMD_LANES);
    assert_eq!(inputs.len(), gate_fn.inputs.len());
    let limb = lane / 64;
    let bit_mask = 1u64 << (lane % 64);
    let mut bit_cursor = 0;
    for (input_value, input) in inputs.iter().zip(gate_fn.inputs.iter()) {
        for bit_idx in 0..input.bit_vector.get_bit_count() {
            if input_value.get_bit(bit_idx).unwrap() {
                words_per_input_bit[bit_cursor + bit_idx][limb] |= bit_mask;
            }
        }
        bit_cursor += input.bit_vector.get_bit_count();
    }
}

fn random_simd_inputs(gate_fn: &GateFn, lane_count: usize, rng: &mut impl Rng) -> Vec<Vec256> {
    assert!(lane_count <= SIMD_LANES);
    let mut words_per_input_bit = vec![[0u64; 4]; total_input_bits(gate_fn)];
    for lane in 0..lane_count {
        let inputs = gen_random_inputs(gate_fn, rng);
        set_lane_inputs(gate_fn, lane, &inputs, &mut words_per_input_bit);
    }
    words_per_input_bit
        .into_iter()
        .map(Vec256::from_words)
        .collect()
}

fn counterexample_simd_inputs(gate_fn: &GateFn, counterexamples: &[&Vec<IrBits>]) -> Vec<Vec256> {
    assert!(counterexamples.len() <= SIMD_LANES);
    let mut words_per_input_bit = vec![[0u64; 4]; total_input_bits(gate_fn)];
    for (lane, inputs) in counterexamples.iter().enumerate() {
        set_lane_inputs(gate_fn, lane, inputs, &mut words_per_input_bit);
    }
    words_per_input_bit
        .into_iter()
        .map(Vec256::from_words)
        .collect()
}

fn lane_mask(valid_lanes: usize) -> [u64; 4] {
    assert!(valid_lanes <= SIMD_LANES);
    let mut mask = [0u64; 4];
    let full_limbs = valid_lanes / 64;
    let remainder = valid_lanes % 64;
    for word in mask.iter_mut().take(full_limbs) {
        *word = u64::MAX;
    }
    if remainder > 0 {
        mask[full_limbs] = (1u64 << remainder) - 1;
    }
    mask
}

fn masked_words(value: Vec256, mask: [u64; 4], inverted: bool) -> [u64; 4] {
    let words = value.to_array();
    [
        (if inverted { !words[0] } else { words[0] }) & mask[0],
        (if inverted { !words[1] } else { words[1] }) & mask[1],
        (if inverted { !words[2] } else { words[2] }) & mask[2],
        (if inverted { !words[3] } else { words[3] }) & mask[3],
    ]
}

fn update_signatures(
    gate_fn: &GateFn,
    candidate_node_indices: &[usize],
    normal_signatures: &mut [SimulationSignature],
    inverse_signatures: &mut [SimulationSignature],
    inputs: &[Vec256],
    valid_lanes: usize,
    batch_index: u64,
) {
    let node_values = gate_simd::eval_all_node_values(gate_fn, inputs);
    let mask = lane_mask(valid_lanes);
    for &node_index in candidate_node_indices {
        let value = node_values[node_index];
        normal_signatures[node_index].update_words(masked_words(value, mask, false), batch_index);
        inverse_signatures[node_index].update_words(masked_words(value, mask, true), batch_index);
    }
}

/// Returns a mapping from hash value (a hash over the history for a given gate
/// as it's fed random samples) to a sequence of the nodes that had the same
/// history.
pub fn propose_equivalence_classes(
    gate_fn: &GateFn,
    input_sample_count: usize,
    rng: &mut impl Rng,
    counterexamples: &HashSet<Vec<IrBits>>,
) -> HashMap<SimulationSignature, Vec<EquivNode>> {
    let gate_count = gate_fn.gates.len();
    let candidate_node_indices: Vec<usize> = gate_fn
        .gates
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| {
            if matches!(node, AigNode::Input { .. } | AigNode::Literal { .. }) {
                None
            } else {
                Some(node_index)
            }
        })
        .collect();
    let mut normal_signatures = vec![SimulationSignature::new(); gate_count];
    let mut inverse_signatures = vec![SimulationSignature::new(); gate_count];
    let mut batch_index = 0u64;

    // Push `input_sample_count` random samples through the gate function in
    // 256-wide batches and stream each node's sampled value into a signature.
    let mut remaining_random_samples = input_sample_count;
    while remaining_random_samples > 0 {
        let lane_count = min(remaining_random_samples, SIMD_LANES);
        let inputs = random_simd_inputs(gate_fn, lane_count, rng);
        update_signatures(
            gate_fn,
            &candidate_node_indices,
            &mut normal_signatures,
            &mut inverse_signatures,
            &inputs,
            lane_count,
            batch_index,
        );
        batch_index += 1;
        remaining_random_samples -= lane_count;
    }

    // Now do it for all explicitly-provided counterexamples.
    let counterexample_refs: Vec<&Vec<IrBits>> = counterexamples.iter().collect();
    for batch in counterexample_refs.chunks(SIMD_LANES) {
        let inputs = counterexample_simd_inputs(gate_fn, batch);
        update_signatures(
            gate_fn,
            &candidate_node_indices,
            &mut normal_signatures,
            &mut inverse_signatures,
            &inputs,
            batch.len(),
            batch_index,
        );
        batch_index += 1;
    }

    // Group by hash
    let total_samples = input_sample_count + counterexamples.len();
    let mut grouped_classes: HashMap<SimulationSignature, Vec<EquivNode>> = HashMap::new();
    for node_index in candidate_node_indices {
        let node_ref = AigRef { id: node_index };
        let entries = [
            (
                normal_signatures[node_index].finalize(total_samples),
                EquivNode::Normal(node_ref),
            ),
            (
                inverse_signatures[node_index].finalize(total_samples),
                EquivNode::Inverted(node_ref),
            ),
        ];
        for (hash, equiv_node) in entries {
            grouped_classes
                .entry(hash)
                .or_insert_with(Vec::new)
                .push(equiv_node);
        }
    }

    for nodes in grouped_classes.values_mut() {
        nodes.sort_unstable();
    }

    grouped_classes.retain(|_, nodes| nodes.len() > 1);
    grouped_classes
}

#[cfg(test)]
fn propose_equivalence_classes_scalar_for_test(
    gate_fn: &GateFn,
    input_samples: &[Vec<IrBits>],
    counterexamples: &[Vec<IrBits>],
) -> HashMap<Vec<bool>, Vec<EquivNode>> {
    use crate::aig_sim::gate_sim::{self, Collect, GateSimResult};

    let gate_count = gate_fn.gates.len();
    let mut history = Vec::with_capacity(input_samples.len() + counterexamples.len());
    for sample in input_samples {
        let result: GateSimResult = gate_sim::eval(gate_fn, sample, Collect::All);
        history.push(result.all_values.unwrap());
    }
    for sample in counterexamples {
        let result: GateSimResult = gate_sim::eval(gate_fn, sample, Collect::All);
        history.push(result.all_values.unwrap());
    }

    let mut grouped_classes = HashMap::new();
    for node_index in 0..gate_count {
        let node = &gate_fn.gates[node_index];
        if matches!(node, AigNode::Input { .. } | AigNode::Literal { .. }) {
            continue;
        }
        let normal_history: Vec<bool> = history.iter().map(|h| h[node_index]).collect();
        let inverse_history: Vec<bool> = normal_history.iter().map(|bit| !*bit).collect();
        let node_ref = AigRef { id: node_index };
        grouped_classes
            .entry(normal_history)
            .or_insert_with(Vec::new)
            .push(EquivNode::Normal(node_ref));
        grouped_classes
            .entry(inverse_history)
            .or_insert_with(Vec::new)
            .push(EquivNode::Inverted(node_ref));
    }

    for nodes in grouped_classes.values_mut() {
        nodes.sort_unstable();
    }

    grouped_classes.retain(|_, nodes| nodes.len() > 1);
    grouped_classes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{setup_graph_with_redundancies, setup_simple_graph};
    use rand::SeedableRng;

    fn sorted_classes<K>(classes: &HashMap<K, Vec<EquivNode>>) -> Vec<Vec<EquivNode>>
    where
        K: Eq + std::hash::Hash,
    {
        let mut values = classes.values().cloned().collect::<Vec<_>>();
        for value in values.iter_mut() {
            value.sort_unstable();
        }
        values.sort();
        values
    }

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

    #[test]
    fn test_propose_equiv_simd_hash_matches_scalar_multiple_batches() {
        let graph = setup_graph_with_redundancies();
        let input_sample_count = 513;
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(7);
        let input_samples = (0..input_sample_count)
            .map(|_| gen_random_inputs(&graph.g, &mut seeded_rng))
            .collect::<Vec<_>>();
        let counterexample_samples = vec![
            vec![
                IrBits::bool(false),
                IrBits::bool(false),
                IrBits::bool(false),
            ],
            vec![IrBits::bool(true), IrBits::bool(false), IrBits::bool(true)],
            vec![IrBits::bool(true), IrBits::bool(true), IrBits::bool(false)],
        ];
        let scalar_classes = propose_equivalence_classes_scalar_for_test(
            &graph.g,
            &input_samples,
            &counterexample_samples,
        );

        let counterexamples = counterexample_samples.into_iter().collect::<HashSet<_>>();
        let mut simd_rng = rand::rngs::StdRng::seed_from_u64(7);
        let simd_classes = propose_equivalence_classes(
            &graph.g,
            input_sample_count,
            &mut simd_rng,
            &counterexamples,
        );

        assert_eq!(
            sorted_classes(&simd_classes),
            sorted_classes(&scalar_classes)
        );
    }
}
