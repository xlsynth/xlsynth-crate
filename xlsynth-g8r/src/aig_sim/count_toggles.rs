// SPDX-License-Identifier: Apache-2.0

use crate::aig::{AigNode, GateFn};
use crate::aig_sim::gate_simd::{self, Vec256};
use serde::Serialize;
use xlsynth::IrBits;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToggleStats {
    /// The number of toggles at all gate outputs, where a gate is an AND2 node
    /// in the graph.
    pub gate_output_toggles: usize,

    /// The number of toggles at all gate inputs, where a gate is an AND2 node
    /// in the graph.
    pub gate_input_toggles: usize,

    /// The number of toggles in the raw batch input vectors (primary inputs).
    pub primary_input_toggles: usize,

    /// The number of toggles at the circuit's output pins only.
    pub primary_output_toggles: usize,
}

impl Into<crate::result_proto::ToggleStats> for ToggleStats {
    fn into(self) -> crate::result_proto::ToggleStats {
        crate::result_proto::ToggleStats {
            gate_output_toggles: self.gate_output_toggles as u64,
            gate_input_toggles: self.gate_input_toggles as u64,
            primary_input_toggles: self.primary_input_toggles as u64,
            primary_output_toggles: self.primary_output_toggles as u64,
        }
    }
}

/// Parameters for load-weighted switching estimation.
///
/// A node with effective fanout/load `f` receives weight:
/// `beta1 * f + beta2 * f^2`.
///
/// The optional quadratic term lets us penalize high-fanout hotspots more
/// aggressively than a purely linear model. This can better approximate cases
/// where buffering/wiring overhead grows super-linearly with fanout. Defaults
/// keep this disabled (`beta2=0.0`) for a simple linear load model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedSwitchingOptions {
    pub beta1: f64,
    pub beta2: f64,
    /// Additional load per primary-output use.
    pub primary_output_load: f64,
}

impl Default for WeightedSwitchingOptions {
    fn default() -> Self {
        Self {
            beta1: 1.0,
            beta2: 0.0,
            primary_output_load: 1.0,
        }
    }
}

/// Load-weighted switching statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WeightedSwitchingStats {
    /// Number of interior gate-output toggles (AND2 nodes only).
    pub gate_output_toggles: usize,
    /// Sum over nodes of `toggles(node) * weight(node)`, scaled by 1e3.
    pub weighted_switching_milli: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToggleNodeKind {
    Input,
    Literal,
    And2,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NodeToggleStats {
    /// Stable AIG node id within the `GateFn`.
    pub node_id: usize,
    pub node_kind: ToggleNodeKind,
    /// Number of observed sample-to-sample transitions for this node.
    pub toggle_count: usize,
    /// Fraction of ordered stimulus transitions on which this node toggled.
    pub toggle_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToggleActivityStats {
    pub sample_count: usize,
    pub transition_count: usize,
    pub aggregate: ToggleStats,
    /// Output-reachable AIG nodes in stable node-id order.
    pub nodes: Vec<NodeToggleStats>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveNodeToggleCounts {
    live_nodes: Vec<bool>,
    per_node_toggles: Vec<usize>,
}

fn collect_and2_indices(gate_fn: &GateFn) -> Vec<usize> {
    gate_fn
        .gates
        .iter()
        .enumerate()
        .filter_map(|(idx, gate)| {
            if matches!(gate, AigNode::And2 { .. }) {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

fn collect_and2_input_uses(gate_fn: &GateFn) -> Vec<usize> {
    let mut use_counts = vec![0usize; gate_fn.gates.len()];
    for gate in &gate_fn.gates {
        for operand in gate.get_operands() {
            use_counts[operand.node.id] += 1;
        }
    }
    use_counts
}

fn collect_primary_output_uses(gate_fn: &GateFn) -> Vec<usize> {
    let mut use_counts = vec![0usize; gate_fn.gates.len()];
    for output in &gate_fn.outputs {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            use_counts[operand.node.id] += 1;
        }
    }
    use_counts
}

fn collect_output_reachable_nodes(gate_fn: &GateFn) -> Vec<bool> {
    let mut live_nodes = vec![false; gate_fn.gates.len()];
    for operand in gate_fn.post_order_operands(/* discard_inputs= */ false) {
        live_nodes[operand.node.id] = true;
    }
    live_nodes
}

fn node_kind(node: &AigNode) -> ToggleNodeKind {
    match node {
        AigNode::Input { .. } => ToggleNodeKind::Input,
        AigNode::Literal { .. } => ToggleNodeKind::Literal,
        AigNode::And2 { .. } => ToggleNodeKind::And2,
    }
}

fn count_low_bits_set(words: [u64; 4], bit_count: usize) -> usize {
    assert!(
        bit_count <= 256,
        "bit_count must be <= 256; got {bit_count}"
    );
    let mut remaining = bit_count;
    let mut masked_words = [0u64; 4];
    for (word_index, word) in words.into_iter().enumerate() {
        if remaining == 0 {
            break;
        }
        let bits_here = remaining.min(64);
        let mask = if bits_here == 64 {
            u64::MAX
        } else {
            (1u64 << bits_here) - 1
        };
        masked_words[word_index] = word & mask;
        remaining -= bits_here;
    }
    Vec256::from_words(masked_words).popcount()
}

/// Counts sample-to-sample transitions within one packed 256-lane trace.
fn count_adjacent_toggles(value: Vec256, valid_samples: usize) -> usize {
    assert!(
        (1..=256).contains(&valid_samples),
        "valid sample count must be in 1..=256; got {valid_samples}"
    );
    let transition_count = valid_samples - 1;
    if transition_count == 0 {
        return 0;
    }
    let words = value.to_array();
    let adjacent_diffs = [
        words[0] ^ ((words[0] >> 1) | (words[1] << 63)),
        words[1] ^ ((words[1] >> 1) | (words[2] << 63)),
        words[2] ^ ((words[2] >> 1) | (words[3] << 63)),
        words[3] ^ (words[3] >> 1),
    ];
    count_low_bits_set(adjacent_diffs, transition_count)
}

fn validate_toggle_batch_inputs(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
) -> Result<(), String> {
    if batch_inputs.len() < 2 {
        return Err(format!(
            "toggle stimulus must contain at least two samples; got {}",
            batch_inputs.len()
        ));
    }
    gate_simd::validate_ordered_batch_inputs(gate_fn, batch_inputs)
        .map_err(|e| format!("invalid toggle stimulus: {e}"))
}

fn count_primary_input_toggles(batch_inputs: &[Vec<IrBits>]) -> usize {
    let mut primary_input_toggles = 0;
    for pair in batch_inputs.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        assert_eq!(prev.len(), next.len());
        for (prev_bits, next_bits) in prev.iter().zip(next.iter()) {
            assert_eq!(prev_bits.get_bit_count(), next_bits.get_bit_count());
            for i in 0..prev_bits.get_bit_count() {
                let a = prev_bits.get_bit(i).unwrap();
                let b = next_bits.get_bit(i).unwrap();
                if a != b {
                    primary_input_toggles += 1;
                }
            }
        }
    }
    primary_input_toggles
}

/// Counts output-reachable node transitions across an ordered stimulus batch.
fn count_live_node_toggles_simd(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
) -> Result<LiveNodeToggleCounts, String> {
    validate_toggle_batch_inputs(gate_fn, batch_inputs)?;
    let live_nodes = collect_output_reachable_nodes(gate_fn);
    let mut per_node_toggles = vec![0usize; gate_fn.gates.len()];
    let mut previous_chunk_last_values = vec![false; gate_fn.gates.len()];
    let mut has_previous_chunk = false;

    for (chunk_index, chunk) in batch_inputs.chunks(256).enumerate() {
        let chunk_start = chunk_index * 256;
        let packed_inputs =
            gate_simd::pack_ordered_input_chunk(gate_fn, batch_inputs, chunk_start, chunk.len());
        let all_values = gate_simd::eval_all_node_values(gate_fn, &packed_inputs);

        for (node_index, &is_live) in live_nodes.iter().enumerate() {
            if !is_live {
                continue;
            }
            let value = all_values[node_index];
            if has_previous_chunk && previous_chunk_last_values[node_index] != value.get_lane(0) {
                per_node_toggles[node_index] += 1;
            }
            per_node_toggles[node_index] += count_adjacent_toggles(value, chunk.len());
            previous_chunk_last_values[node_index] = value.get_lane(chunk.len() - 1);
        }
        has_previous_chunk = true;
    }

    Ok(LiveNodeToggleCounts {
        live_nodes,
        per_node_toggles,
    })
}

fn aggregate_toggle_stats(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
    per_node_toggles: &[usize],
) -> ToggleStats {
    let and2_indices = collect_and2_indices(gate_fn);
    let gate_output_toggles = and2_indices.iter().map(|&idx| per_node_toggles[idx]).sum();
    let and2_input_uses = collect_and2_input_uses(gate_fn);
    let gate_input_toggles = per_node_toggles
        .iter()
        .zip(and2_input_uses.iter())
        .map(|(&toggles, &uses)| toggles * uses)
        .sum();
    let primary_input_toggles = count_primary_input_toggles(batch_inputs);
    let primary_output_uses = collect_primary_output_uses(gate_fn);
    let primary_output_toggles = per_node_toggles
        .iter()
        .zip(primary_output_uses.iter())
        .map(|(&toggles, &uses)| toggles * uses)
        .sum();
    ToggleStats {
        gate_output_toggles,
        gate_input_toggles,
        primary_input_toggles,
        primary_output_toggles,
    }
}

fn f64_to_u128_round_saturating(v: f64) -> u128 {
    // We use f64 because the load coefficients (`beta1`, `beta2`,
    // `primary_output_load`) are runtime-configurable floats.
    // Keep clamping sign-aware for non-finite values:
    // - `NaN` / `-inf` => 0
    // - `+inf` => u128::MAX
    // This avoids poisoning metrics with maximal costs for malformed negatives.
    if v.is_nan() {
        return 0;
    }
    if v == f64::NEG_INFINITY {
        return 0;
    }
    if v == f64::INFINITY {
        return u128::MAX;
    }
    if v <= 0.0 {
        return 0;
    }
    if v >= u128::MAX as f64 {
        u128::MAX
    } else {
        v.round() as u128
    }
}

/// Estimates load-weighted switching activity over a batch.
///
/// For each interior AND2 node `u`, we compute:
/// - `toggles(u)`: output transitions observed across consecutive samples
/// - `f(u)`: effective load = internal fanout + primary_output_load *
///   output_uses
/// - `weight(u)`: beta1 * f(u) + beta2 * f(u)^2
///
/// The final score is `sum_u toggles(u) * weight(u)`, reported in milli-units.
pub fn count_weighted_switching(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
    options: &WeightedSwitchingOptions,
) -> WeightedSwitchingStats {
    let live_node_toggles =
        count_live_node_toggles_simd(gate_fn, batch_inputs).unwrap_or_else(|e| panic!("{e}"));
    let per_node_output_toggles = &live_node_toggles.per_node_toggles;
    let and2_indices = collect_and2_indices(gate_fn);
    let gate_output_toggles = and2_indices
        .iter()
        .map(|&idx| per_node_output_toggles[idx])
        .sum();

    let mut internal_fanout = vec![0usize; gate_fn.gates.len()];
    for gate in gate_fn.gates.iter() {
        if let AigNode::And2 { a, b, .. } = gate {
            internal_fanout[a.node.id] += 1;
            internal_fanout[b.node.id] += 1;
        }
    }
    let primary_output_uses = collect_primary_output_uses(gate_fn);

    let mut weighted_switching_milli: u128 = 0;
    for &idx in &and2_indices {
        let toggles = per_node_output_toggles[idx] as u128;
        if toggles == 0 {
            continue;
        }
        // We use f64 only for this per-node load/weight computation (not for
        // the accumulated corpus metric). So the magnitude is tied to one
        // gate's fanout and the beta coefficients. We immediately convert this
        // local value into clamped u128 milli-units before accumulation.
        let effective_load = (internal_fanout[idx] as f64)
            + options.primary_output_load * (primary_output_uses[idx] as f64);
        let weight = options.beta1 * effective_load + options.beta2 * effective_load.powi(2);
        let weight_milli = f64_to_u128_round_saturating(weight * 1000.0);
        let contribution_milli = toggles.saturating_mul(weight_milli);
        weighted_switching_milli = weighted_switching_milli.saturating_add(contribution_milli);
    }

    WeightedSwitchingStats {
        gate_output_toggles,
        weighted_switching_milli,
    }
}

/// Counts toggles at gate outputs, primary inputs, and in the raw batch input
/// vectors for a batch of input vectors.
///
/// # Arguments
/// * `gate_fn` - The gate function to simulate.
/// * `batch_inputs` - A batch of input vectors, each is a vector of IrBits (one
///   per input port).
///
/// # Returns
/// ToggleStats: gate_output_toggles, gate_input_toggles, primary_input_toggles,
/// primary_output_toggles
pub fn count_toggles(gate_fn: &GateFn, batch_inputs: &[Vec<IrBits>]) -> ToggleStats {
    // Debug: print first 3 input vectors
    for (i, input_vec) in batch_inputs.iter().take(3).enumerate() {
        log::debug!("batch_inputs[{}]: {:?}", i, input_vec);
    }
    let live_node_toggles =
        count_live_node_toggles_simd(gate_fn, batch_inputs).unwrap_or_else(|e| panic!("{e}"));
    aggregate_toggle_stats(gate_fn, batch_inputs, &live_node_toggles.per_node_toggles)
}

/// Counts ordered-stimulus toggle activity for output-reachable AIG nodes.
pub fn count_toggle_activity(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
) -> ToggleActivityStats {
    let live_node_toggles =
        count_live_node_toggles_simd(gate_fn, batch_inputs).unwrap_or_else(|e| panic!("{e}"));
    let transition_count = batch_inputs.len() - 1;
    let nodes = gate_fn
        .gates
        .iter()
        .enumerate()
        .filter(|(node_index, _)| live_node_toggles.live_nodes[*node_index])
        .map(|(node_id, node)| {
            let toggle_count = live_node_toggles.per_node_toggles[node_id];
            NodeToggleStats {
                node_id,
                node_kind: node_kind(node),
                toggle_count,
                toggle_rate: toggle_count as f64 / transition_count as f64,
            }
        })
        .collect();
    ToggleActivityStats {
        sample_count: batch_inputs.len(),
        transition_count,
        aggregate: aggregate_toggle_stats(
            gate_fn,
            batch_inputs,
            &live_node_toggles.per_node_toggles,
        ),
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aig::gate::AigBitVector,
        aig_sim::gate_sim::{self, Collect},
        gate_builder::{GateBuilder, GateBuilderOptions},
    };
    use bitvec::vec::BitVec;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use xlsynth::IrBits;

    fn scalar_collect_all_values(gate_fn: &GateFn, batch_inputs: &[Vec<IrBits>]) -> Vec<BitVec> {
        batch_inputs
            .iter()
            .map(|input_vec| {
                gate_sim::eval(gate_fn, input_vec, Collect::AllWithInputs)
                    .all_values
                    .expect("Collect::AllWithInputs should produce all_values")
            })
            .collect()
    }

    fn scalar_count_node_toggles(gate_fn: &GateFn, batch_inputs: &[Vec<IrBits>]) -> Vec<usize> {
        let all_values = scalar_collect_all_values(gate_fn, batch_inputs);
        let mut per_node_toggles = vec![0usize; gate_fn.gates.len()];
        for pair in all_values.windows(2) {
            let (prev, next) = (&pair[0], &pair[1]);
            for node_index in 0..gate_fn.gates.len() {
                if prev[node_index] != next[node_index] {
                    per_node_toggles[node_index] += 1;
                }
            }
        }
        per_node_toggles
    }

    fn scalar_count_toggles(gate_fn: &GateFn, batch_inputs: &[Vec<IrBits>]) -> ToggleStats {
        let per_node_toggles = scalar_count_node_toggles(gate_fn, batch_inputs);
        let gate_output_toggles = collect_and2_indices(gate_fn)
            .iter()
            .map(|&idx| per_node_toggles[idx])
            .sum();
        let and2_input_uses = collect_and2_input_uses(gate_fn);
        let gate_input_toggles = per_node_toggles
            .iter()
            .zip(and2_input_uses.iter())
            .map(|(&toggles, &uses)| toggles * uses)
            .sum();
        let primary_output_uses = collect_primary_output_uses(gate_fn);
        let primary_output_toggles = per_node_toggles
            .iter()
            .zip(primary_output_uses.iter())
            .map(|(&toggles, &uses)| toggles * uses)
            .sum();
        ToggleStats {
            gate_output_toggles,
            gate_input_toggles,
            primary_input_toggles: count_primary_input_toggles(batch_inputs),
            primary_output_toggles,
        }
    }

    fn scalar_count_weighted_switching(
        gate_fn: &GateFn,
        batch_inputs: &[Vec<IrBits>],
        options: &WeightedSwitchingOptions,
    ) -> WeightedSwitchingStats {
        let per_node_output_toggles = scalar_count_node_toggles(gate_fn, batch_inputs);
        let gate_output_toggles = collect_and2_indices(gate_fn)
            .iter()
            .map(|&idx| per_node_output_toggles[idx])
            .sum();
        let mut internal_fanout = vec![0usize; gate_fn.gates.len()];
        for gate in &gate_fn.gates {
            if let AigNode::And2 { a, b, .. } = gate {
                internal_fanout[a.node.id] += 1;
                internal_fanout[b.node.id] += 1;
            }
        }
        let primary_output_uses = collect_primary_output_uses(gate_fn);
        let mut weighted_switching_milli = 0u128;
        for idx in collect_and2_indices(gate_fn) {
            let toggles = per_node_output_toggles[idx] as u128;
            if toggles == 0 {
                continue;
            }
            let effective_load = (internal_fanout[idx] as f64)
                + options.primary_output_load * (primary_output_uses[idx] as f64);
            let weight = options.beta1 * effective_load + options.beta2 * effective_load.powi(2);
            let weight_milli = f64_to_u128_round_saturating(weight * 1000.0);
            weighted_switching_milli =
                weighted_switching_milli.saturating_add(toggles.saturating_mul(weight_milli));
        }
        WeightedSwitchingStats {
            gate_output_toggles,
            weighted_switching_milli,
        }
    }

    #[test]
    fn test_count_toggles_simple_and() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut gb = GateBuilder::new("simple_and".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 2);
        let input_b = gb.add_input("b".to_string(), 2);
        let and_node = gb.add_and_vec(&input_a, &input_b);
        gb.add_output("out".to_string(), and_node);
        let gate_fn = gb.build();

        // Batch: 00, 01, 10, 11 for both inputs
        let batch_inputs = vec![
            vec![
                IrBits::make_ubits(2, 0b00).unwrap(),
                IrBits::make_ubits(2, 0b00).unwrap(),
            ],
            vec![
                IrBits::make_ubits(2, 0b01).unwrap(),
                IrBits::make_ubits(2, 0b01).unwrap(),
            ],
            vec![
                IrBits::make_ubits(2, 0b10).unwrap(),
                IrBits::make_ubits(2, 0b10).unwrap(),
            ],
            vec![
                IrBits::make_ubits(2, 0b11).unwrap(),
                IrBits::make_ubits(2, 0b11).unwrap(),
            ],
        ];
        let stats = count_toggles(&gate_fn, &batch_inputs);
        // gate_output_toggles: toggles at all gate outputs (AND2 nodes only)
        // gate_input_toggles: toggles at all gate input pins (inputs to AND2 nodes)
        //   Note: In this test, gate input toggles and primary input toggles are equal
        // because each AND2 input is directly wired to a primary input bit.
        //   Not every input bit toggles every transition, so the count is 8, not 12.
        assert_eq!(
            stats.gate_output_toggles, 4,
            "Expected 4 gate output toggles (AND2 nodes only)"
        );
        assert_eq!(
            stats.gate_input_toggles, 8,
            "Expected 8 gate input toggles (see comment above)"
        );
        assert_eq!(
            stats.primary_input_toggles, 8,
            "Expected 8 primary input toggles (see comment above)"
        );
        assert_eq!(
            stats.primary_output_toggles, 4,
            "Expected 4 primary output toggles (same as gate outputs in this test)"
        );
    }

    #[test]
    fn test_count_toggles_and_reduce_4() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut gb = GateBuilder::new("and_reduce_4".to_string(), GateBuilderOptions::opt());
        let input = gb.add_input("in".to_string(), 4);
        // AND-reduce: ((in[0] & in[1]) & (in[2] & in[3]))
        let and01 = gb.add_and_binary(*input.get_lsb(0), *input.get_lsb(1));
        let and23 = gb.add_and_binary(*input.get_lsb(2), *input.get_lsb(3));
        let and_reduce = gb.add_and_binary(and01, and23);
        gb.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[and_reduce]),
        );
        let gate_fn = gb.build();

        // Batch: all 16 possible 4-bit input patterns
        let batch_inputs: Vec<Vec<IrBits>> = (0..16)
            .map(|v| vec![IrBits::make_ubits(4, v).unwrap()])
            .collect();
        let stats = count_toggles(&gate_fn, &batch_inputs);
        println!(
            "and_reduce_4: gate_output_toggles = {}, gate_input_toggles = {}, primary_input_toggles = {}, primary_output_toggles = {}",
            stats.gate_output_toggles,
            stats.gate_input_toggles,
            stats.primary_input_toggles,
            stats.primary_output_toggles
        );

        // Note: The AND function masks many toggles — a toggle on one input does not
        // propagate if the other input is 0. These values are observed from simulation.
        assert_eq!(
            stats.primary_input_toggles, 26,
            "Expected 26 toggles for 4 input bits across 16 binary patterns"
        );
        assert_eq!(
            stats.gate_output_toggles, 9,
            "Observe 9 toggles at all AND2 nodes due to masking"
        );
        assert_eq!(
            stats.gate_input_toggles, 34,
            "Observe 34 toggles at all AND2 input pins due to masking"
        );
        assert_eq!(
            stats.primary_output_toggles, 1,
            "Observe 1 toggle at the output pin (from 0 to 1 at the last transition)"
        );
    }

    #[test]
    fn test_weighted_switching_defaults_and_monotonicity() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut gb = GateBuilder::new(
            "weighted_switching_case".to_string(),
            GateBuilderOptions::opt(),
        );
        let input = gb.add_input("in".to_string(), 4);
        let and01 = gb.add_and_binary(*input.get_lsb(0), *input.get_lsb(1));
        let and23 = gb.add_and_binary(*input.get_lsb(2), *input.get_lsb(3));
        let and_reduce = gb.add_and_binary(and01, and23);
        gb.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[and_reduce]),
        );
        let gate_fn = gb.build();

        let batch_inputs: Vec<Vec<IrBits>> = (0..16)
            .map(|v| vec![IrBits::make_ubits(4, v).unwrap()])
            .collect();

        let defaults = count_weighted_switching(
            &gate_fn,
            &batch_inputs,
            &WeightedSwitchingOptions::default(),
        );
        let explicit_defaults = count_weighted_switching(
            &gate_fn,
            &batch_inputs,
            &WeightedSwitchingOptions {
                beta1: 1.0,
                beta2: 0.0,
                primary_output_load: 1.0,
            },
        );
        assert_eq!(defaults, explicit_defaults);

        let quadratic = count_weighted_switching(
            &gate_fn,
            &batch_inputs,
            &WeightedSwitchingOptions {
                beta1: 1.0,
                beta2: 1.0,
                primary_output_load: 1.0,
            },
        );
        assert!(
            quadratic.weighted_switching_milli >= defaults.weighted_switching_milli,
            "quadratic load weighting should not reduce weighted switching"
        );

        let heavier_output = count_weighted_switching(
            &gate_fn,
            &batch_inputs,
            &WeightedSwitchingOptions {
                beta1: 1.0,
                beta2: 0.0,
                primary_output_load: 4.0,
            },
        );
        assert!(
            heavier_output.weighted_switching_milli >= defaults.weighted_switching_milli,
            "larger primary-output load should not reduce weighted switching"
        );

        let toggle_stats = count_toggles(&gate_fn, &batch_inputs);
        assert_eq!(
            defaults.gate_output_toggles, toggle_stats.gate_output_toggles,
            "weighted switching should report the same raw interior gate-output toggles"
        );
    }

    #[test]
    fn test_count_toggles_counts_simd_chunk_boundary_transition() {
        let mut gb = GateBuilder::new("chunk_boundary".to_string(), GateBuilderOptions::no_opt());
        let input = gb.add_input("in".to_string(), 1);
        let true_op = gb.get_true();
        let passthrough = gb.add_and_binary(*input.get_lsb(0), true_op);
        gb.add_output("out".to_string(), AigBitVector::from_bit(passthrough));
        let gate_fn = gb.build();

        let mut batch_inputs = vec![vec![IrBits::make_ubits(1, 0).unwrap()]; 256];
        batch_inputs.push(vec![IrBits::make_ubits(1, 1).unwrap()]);

        assert_eq!(
            count_toggles(&gate_fn, &batch_inputs),
            ToggleStats {
                gate_output_toggles: 1,
                gate_input_toggles: 1,
                primary_input_toggles: 1,
                primary_output_toggles: 1,
            }
        );
        assert_eq!(
            count_weighted_switching(
                &gate_fn,
                &batch_inputs,
                &WeightedSwitchingOptions::default(),
            ),
            WeightedSwitchingStats {
                gate_output_toggles: 1,
                weighted_switching_milli: 1000,
            }
        );
        assert_eq!(
            count_toggle_activity(&gate_fn, &batch_inputs),
            ToggleActivityStats {
                sample_count: 257,
                transition_count: 256,
                aggregate: ToggleStats {
                    gate_output_toggles: 1,
                    gate_input_toggles: 1,
                    primary_input_toggles: 1,
                    primary_output_toggles: 1,
                },
                nodes: vec![
                    NodeToggleStats {
                        node_id: 0,
                        node_kind: ToggleNodeKind::Literal,
                        toggle_count: 0,
                        toggle_rate: 0.0,
                    },
                    NodeToggleStats {
                        node_id: 1,
                        node_kind: ToggleNodeKind::Input,
                        toggle_count: 1,
                        toggle_rate: 1.0 / 256.0,
                    },
                    NodeToggleStats {
                        node_id: 2,
                        node_kind: ToggleNodeKind::And2,
                        toggle_count: 1,
                        toggle_rate: 1.0 / 256.0,
                    },
                ],
            }
        );
    }

    #[test]
    fn test_simd_toggle_metrics_match_scalar_reference_across_chunks() {
        let mut gb = GateBuilder::new(
            "simd_toggle_equivalence".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a = gb.add_input("a".to_string(), 3);
        let b = gb.add_input("b".to_string(), 2);
        let live0 = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let not_b1 = gb.add_not(*b.get_lsb(1));
        let live1 = gb.add_and_binary(*a.get_lsb(1), not_b1);
        let out = gb.add_and_binary(live0, live1);
        let _dead = gb.add_and_binary(*a.get_lsb(0), *a.get_lsb(2));
        gb.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[out, *a.get_lsb(0)]),
        );
        let gate_fn = gb.build();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let batch_inputs: Vec<Vec<IrBits>> = (0..513)
            .map(|_| {
                vec![
                    IrBits::make_ubits(3, rng.r#gen_range(0u64..8u64)).unwrap(),
                    IrBits::make_ubits(2, rng.r#gen_range(0u64..4u64)).unwrap(),
                ]
            })
            .collect();
        let options = WeightedSwitchingOptions {
            beta1: 1.5,
            beta2: 0.25,
            primary_output_load: 2.0,
        };

        assert_eq!(
            count_toggles(&gate_fn, &batch_inputs),
            scalar_count_toggles(&gate_fn, &batch_inputs)
        );
        assert_eq!(
            count_weighted_switching(&gate_fn, &batch_inputs, &options),
            scalar_count_weighted_switching(&gate_fn, &batch_inputs, &options)
        );
    }

    #[test]
    fn test_f64_to_u128_round_saturating_non_finite_sign_aware() {
        assert_eq!(f64_to_u128_round_saturating(f64::NAN), 0);
        assert_eq!(f64_to_u128_round_saturating(f64::NEG_INFINITY), 0);
        assert_eq!(f64_to_u128_round_saturating(f64::INFINITY), u128::MAX);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::test_utils::{Opt, ir_value_bf16_to_flat_ir_bits, load_bf16_add_sample, make_bf16};
    use half::bf16;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_bf16_adder_toggle_counting() {
        // Load the bf16 adder circuit
        let loaded = load_bf16_add_sample(Opt::Yes);
        let gate_fn = &loaded.gate_fn;

        // Set up RNG
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut batch_inputs = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let a = bf16::from_f32(rng.r#gen::<f32>());
            let b = bf16::from_f32(rng.r#gen::<f32>());
            let a_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(a));
            let b_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(b));
            batch_inputs.push(vec![a_bits.clone(), b_bits.clone()]);
        }
        // count_toggles expects at least 2 inputs
        assert!(batch_inputs.len() >= 2);
        let stats = count_toggles(gate_fn, &batch_inputs);
        println!(
            "bf16 adder: gate_output_toggles = {}, gate_input_toggles = {}, primary_input_toggles = {}, primary_output_toggles = {}",
            stats.gate_output_toggles,
            stats.gate_input_toggles,
            stats.primary_input_toggles,
            stats.primary_output_toggles
        );
        assert!(
            stats.gate_output_toggles > 0,
            "Should see gate output toggles"
        );
        assert!(
            stats.gate_input_toggles > 0,
            "Should see gate input toggles"
        );
        assert!(
            stats.primary_input_toggles > 0,
            "Should see primary input toggles"
        );
        assert!(
            stats.primary_output_toggles > 0,
            "Should see primary output toggles"
        );
    }

    #[test]
    fn test_bf16_mul_toggle_counting() {
        // Load the bf16 multiplier circuit
        let loaded = crate::test_utils::load_bf16_mul_sample(Opt::Yes);
        let gate_fn = &loaded.gate_fn;

        // Set up RNG
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut batch_inputs = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let a = bf16::from_f32(rng.r#gen::<f32>());
            let b = bf16::from_f32(rng.r#gen::<f32>());
            let a_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(a));
            let b_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(b));
            batch_inputs.push(vec![a_bits.clone(), b_bits.clone()]);
        }
        assert!(batch_inputs.len() >= 2);
        let stats = count_toggles(gate_fn, &batch_inputs);
        println!(
            "bf16 mul: gate_output_toggles = {}, gate_input_toggles = {}, primary_input_toggles = {}, primary_output_toggles = {}",
            stats.gate_output_toggles,
            stats.gate_input_toggles,
            stats.primary_input_toggles,
            stats.primary_output_toggles
        );
        assert!(
            stats.gate_output_toggles > 0,
            "Should see gate output toggles"
        );
        assert!(
            stats.gate_input_toggles > 0,
            "Should see gate input toggles"
        );
        assert!(
            stats.primary_input_toggles > 0,
            "Should see primary input toggles"
        );
        assert!(
            stats.primary_output_toggles > 0,
            "Should see primary output toggles"
        );
    }
}
