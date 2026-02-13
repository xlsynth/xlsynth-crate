// SPDX-License-Identifier: Apache-2.0

use crate::aig::{AigNode, GateFn};
use crate::aig_sim::gate_sim::{Collect, eval};
use bitvec::vec::BitVec;
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

fn collect_all_values(gate_fn: &GateFn, batch_inputs: &[Vec<IrBits>]) -> Vec<BitVec> {
    let mut all_values_vec: Vec<BitVec> = Vec::with_capacity(batch_inputs.len());
    for input_vec in batch_inputs {
        let result = eval(gate_fn, input_vec, Collect::AllWithInputs);
        let all_values = result
            .all_values
            .expect("Collect::AllWithInputs should produce all_values");
        all_values_vec.push(all_values);
    }
    all_values_vec
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
    assert!(
        batch_inputs.len() >= 2,
        "Need at least two input vectors to count toggles"
    );

    let all_values_vec = collect_all_values(gate_fn, batch_inputs);
    let and2_indices = collect_and2_indices(gate_fn);

    let mut per_node_output_toggles = vec![0usize; gate_fn.gates.len()];
    let mut gate_output_toggles = 0usize;
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        for &idx in &and2_indices {
            if prev[idx] != next[idx] {
                per_node_output_toggles[idx] += 1;
                gate_output_toggles += 1;
            }
        }
    }

    let mut internal_fanout = vec![0usize; gate_fn.gates.len()];
    for gate in gate_fn.gates.iter() {
        if let AigNode::And2 { a, b, .. } = gate {
            internal_fanout[a.node.id] += 1;
            internal_fanout[b.node.id] += 1;
        }
    }
    let mut primary_output_uses = vec![0usize; gate_fn.gates.len()];
    for output in gate_fn.outputs.iter() {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            primary_output_uses[operand.node.id] += 1;
        }
    }

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
    assert!(
        batch_inputs.len() >= 2,
        "Need at least two input vectors to count toggles"
    );
    // Debug: print first 3 input vectors
    for (i, input_vec) in batch_inputs.iter().take(3).enumerate() {
        log::debug!("batch_inputs[{}]: {:?}", i, input_vec);
    }
    let all_values_vec = collect_all_values(gate_fn, batch_inputs);

    // For each consecutive pair, count toggles at all gate outputs (AND2 nodes
    // only)
    let and2_indices = collect_and2_indices(gate_fn);
    let mut gate_output_toggles = 0;
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        for &idx in &and2_indices {
            if prev[idx] != next[idx] {
                gate_output_toggles += 1;
            }
        }
    }
    // For each consecutive pair, count toggles at all gate inputs
    let mut gate_input_toggles = 0;
    // For each gate in the circuit
    for (gate_idx, gate) in gate_fn.gates.iter().enumerate() {
        // For each input operand to the gate
        for operand in gate.get_operands() {
            // For the first 3 transitions, print the values
            for (trans_idx, pair) in all_values_vec.windows(2).enumerate() {
                let (prev, next) = (&pair[0], &pair[1]);
                let prev_val = prev[operand.node.id] ^ operand.negated;
                let next_val = next[operand.node.id] ^ operand.negated;
                if trans_idx < 3 {
                    log::debug!(
                        "Gate {} operand {}: prev={} next={} (toggle={})",
                        gate_idx,
                        operand.node.id,
                        prev_val,
                        next_val,
                        prev_val != next_val
                    );
                }
                if prev_val != next_val {
                    gate_input_toggles += 1;
                }
            }
        }
    }
    // Count bit toggles in the raw batch input vectors (primary inputs)
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
    // Count toggles at the circuit's output pins only
    let mut primary_output_toggles = 0;
    // For each output bit in the circuit
    let output_bit_indices: Vec<usize> = gate_fn
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb().map(|bit| bit.node.id))
        .collect();
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        for &idx in &output_bit_indices {
            if prev[idx] != next[idx] {
                primary_output_toggles += 1;
            }
        }
    }
    ToggleStats {
        gate_output_toggles,
        gate_input_toggles,
        primary_input_toggles,
        primary_output_toggles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aig::gate::AigBitVector,
        gate_builder::{GateBuilder, GateBuilderOptions},
    };
    use xlsynth::IrBits;

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
