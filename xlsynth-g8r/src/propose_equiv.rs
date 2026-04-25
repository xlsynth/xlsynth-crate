// SPDX-License-Identifier: Apache-2.0

//! Functionality for proposing equivalence classes via concrete simulation.

use crate::aig::{AigNode, AigRef, GateFn};
use crate::aig_sim::gate_simd::{self, Vec256};
use xlsynth::IrBits;

use rand::Rng;
use std::cmp::min;
use std::collections::{HashMap, HashSet};

const SIMD_LANES: usize = 256;
const SIGNATURE_INIT_STATE: [u64; 2] = [0x7266_6169_672f_7369, 0x6d75_6c61_7469_6f6e];
const SIGNATURE_UPDATE_TAG: u64 = 0x1d8e_4e27_c47d_124f;
const SIGNATURE_FINAL_TAG: u64 = 0x9c6d_62a8_7f5b_2d31;
const SIGNATURE_WORD_TAGS: [u64; 4] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
];

/// A streaming 128-bit simulation signature for one node.
///
/// SAT validation is still the authority, so a hash collision can only create
/// extra candidates that are later rejected. Keeping only a compact fingerprint
/// avoids materializing the old samples-by-nodes history matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SimulationSignature([u64; 2]);

impl SimulationSignature {
    #[inline]
    fn new() -> Self {
        Self(SIGNATURE_INIT_STATE)
    }

    #[inline]
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    #[inline]
    fn update_words(&mut self, words: [u64; 4], batch_index: u64) {
        let mut state = self.0;
        let batch_mix = Self::splitmix64(batch_index ^ SIGNATURE_UPDATE_TAG);
        state[0] = Self::splitmix64(state[0] ^ batch_mix);
        state[1] = Self::splitmix64(state[1] ^ batch_mix.rotate_left(31));
        for (i, word) in words.into_iter().enumerate() {
            let tag = SIGNATURE_WORD_TAGS[i];
            state[0] = Self::splitmix64(state[0] ^ word ^ tag);
            state[1] =
                Self::splitmix64(state[1] ^ word.rotate_left(32) ^ tag.rotate_left(17) ^ state[0]);
        }
        self.0 = state;
    }

    #[inline]
    fn finalize(mut self, total_samples: usize) -> Self {
        let samples = total_samples as u64;
        self.0[0] = Self::splitmix64(self.0[0] ^ SIGNATURE_FINAL_TAG ^ samples);
        self.0[1] = Self::splitmix64(
            self.0[1] ^ SIGNATURE_FINAL_TAG.rotate_left(29) ^ samples.rotate_left(17) ^ self.0[0],
        );
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

fn input_widths(gate_fn: &GateFn) -> Vec<usize> {
    gate_fn
        .inputs
        .iter()
        .map(|input| input.get_bit_count())
        .collect()
}

fn total_input_bits(input_widths: &[usize]) -> usize {
    input_widths.iter().sum()
}

#[inline]
fn byte_bit(bytes: &[u8], bit_idx: usize) -> bool {
    ((bytes[bit_idx / 8] >> (bit_idx % 8)) & 1) != 0
}

fn set_lane_counterexample_inputs(
    input_widths: &[usize],
    lane: usize,
    inputs: &[IrBits],
    words_per_input_bit: &mut [[u64; 4]],
) {
    assert!(lane < SIMD_LANES);
    assert_eq!(inputs.len(), input_widths.len());
    let limb = lane / 64;
    let bit_mask = 1u64 << (lane % 64);
    let mut bit_cursor = 0;
    for (input_value, &input_width) in inputs.iter().zip(input_widths.iter()) {
        let bytes = input_value.to_le_bytes().unwrap();
        for bit_idx in 0..input_width {
            if byte_bit(&bytes, bit_idx) {
                words_per_input_bit[bit_cursor + bit_idx][limb] |= bit_mask;
            }
        }
        bit_cursor += input_width;
    }
}

fn random_simd_inputs(
    input_widths: &[usize],
    lane_count: usize,
    rng: &mut impl Rng,
) -> Vec<Vec256> {
    assert!(lane_count <= SIMD_LANES);
    let valid_lane_mask = lane_mask(lane_count);
    let mut result = Vec::with_capacity(total_input_bits(input_widths));
    for &input_width in input_widths {
        for _ in 0..input_width {
            result.push(Vec256::from_words([
                rng.r#gen::<u64>() & valid_lane_mask[0],
                rng.r#gen::<u64>() & valid_lane_mask[1],
                rng.r#gen::<u64>() & valid_lane_mask[2],
                rng.r#gen::<u64>() & valid_lane_mask[3],
            ]));
        }
    }
    result
}

fn counterexample_simd_inputs(
    input_widths: &[usize],
    counterexamples: &[&Vec<IrBits>],
) -> Vec<Vec256> {
    assert!(counterexamples.len() <= SIMD_LANES);
    let mut words_per_input_bit = vec![[0u64; 4]; total_input_bits(input_widths)];
    for (lane, inputs) in counterexamples.iter().enumerate() {
        set_lane_counterexample_inputs(input_widths, lane, inputs, &mut words_per_input_bit);
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

fn output_reachable_nodes(gate_fn: &GateFn) -> Vec<bool> {
    let mut live_nodes = vec![false; gate_fn.gates.len()];
    let mut stack = Vec::new();
    for output in &gate_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            stack.push(bit.node);
        }
    }
    while let Some(node_ref) = stack.pop() {
        if live_nodes[node_ref.id] {
            continue;
        }
        live_nodes[node_ref.id] = true;
        if let AigNode::And2 { a, b, .. } = &gate_fn.gates[node_ref.id] {
            stack.push(a.node);
            stack.push(b.node);
        }
    }
    live_nodes
}

fn update_signatures(
    gate_fn: &GateFn,
    live_nodes: &[bool],
    candidate_node_indices: &[usize],
    normal_signatures: &mut [SimulationSignature],
    inverse_signatures: &mut [SimulationSignature],
    inputs: &[Vec256],
    valid_lanes: usize,
    batch_index: u64,
    node_values: &mut Vec<Vec256>,
) {
    gate_simd::eval_live_node_values_dense_into(gate_fn, inputs, live_nodes, node_values);
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
    let live_nodes = output_reachable_nodes(gate_fn);
    let input_widths = input_widths(gate_fn);
    let candidate_node_indices: Vec<usize> = gate_fn
        .gates
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| {
            if !live_nodes[node_index]
                || matches!(node, AigNode::Input { .. } | AigNode::Literal { .. })
            {
                None
            } else {
                Some(node_index)
            }
        })
        .collect();
    let mut normal_signatures = vec![SimulationSignature::new(); gate_count];
    let mut inverse_signatures = vec![SimulationSignature::new(); gate_count];
    let mut node_values = Vec::with_capacity(gate_count);
    let mut batch_index = 0u64;

    // Push `input_sample_count` random samples through the gate function in
    // 256-wide batches and stream each node's sampled value into a signature.
    let mut remaining_random_samples = input_sample_count;
    while remaining_random_samples > 0 {
        let lane_count = min(remaining_random_samples, SIMD_LANES);
        let inputs = random_simd_inputs(&input_widths, lane_count, rng);
        update_signatures(
            gate_fn,
            &live_nodes,
            &candidate_node_indices,
            &mut normal_signatures,
            &mut inverse_signatures,
            &inputs,
            lane_count,
            batch_index,
            &mut node_values,
        );
        batch_index += 1;
        remaining_random_samples -= lane_count;
    }

    // Now do it for all explicitly-provided counterexamples.
    let counterexample_refs: Vec<&Vec<IrBits>> = counterexamples.iter().collect();
    for batch in counterexample_refs.chunks(SIMD_LANES) {
        let inputs = counterexample_simd_inputs(&input_widths, batch);
        update_signatures(
            gate_fn,
            &live_nodes,
            &candidate_node_indices,
            &mut normal_signatures,
            &mut inverse_signatures,
            &inputs,
            batch.len(),
            batch_index,
            &mut node_values,
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
    fn test_propose_equiv_counterexample_hash_matches_scalar() {
        let graph = setup_graph_with_redundancies();
        let counterexample_samples = vec![
            vec![
                IrBits::bool(false),
                IrBits::bool(false),
                IrBits::bool(false),
            ],
            vec![IrBits::bool(true), IrBits::bool(false), IrBits::bool(true)],
            vec![IrBits::bool(true), IrBits::bool(true), IrBits::bool(false)],
        ];
        let scalar_classes =
            propose_equivalence_classes_scalar_for_test(&graph.g, &[], &counterexample_samples);

        let counterexamples = counterexample_samples.into_iter().collect::<HashSet<_>>();
        let mut simd_rng = rand::rngs::StdRng::seed_from_u64(7);
        let simd_classes =
            propose_equivalence_classes(&graph.g, 0, &mut simd_rng, &counterexamples);

        assert_eq!(
            sorted_classes(&simd_classes),
            sorted_classes(&scalar_classes)
        );
    }
}
