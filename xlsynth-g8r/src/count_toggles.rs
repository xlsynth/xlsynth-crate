// SPDX-License-Identifier: Apache-2.0

use crate::gate::GateFn;
use crate::gate_sim::{eval, Collect};
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
    // Run eval for each input vector, collect all_values (gate outputs)
    let mut all_values_vec: Vec<BitVec> = Vec::with_capacity(batch_inputs.len());
    for input_vec in batch_inputs {
        let result = eval(gate_fn, input_vec, Collect::AllWithInputs);
        let all_values = result
            .all_values
            .expect("Collect::All should produce all_values");
        all_values_vec.push(all_values);
    }
    // After collecting all_values_vec, print the values at all primary input gates
    // for the first 3 transitions
    let num_input_gates = gate_fn
        .inputs
        .iter()
        .map(|input| input.bit_vector.get_bit_count())
        .sum::<usize>();
    for (i, values) in all_values_vec.iter().enumerate().take(3) {
        let mut input_bits = Vec::new();
        for gate_idx in 0..num_input_gates {
            input_bits.push(if values[gate_idx] { '1' } else { '0' });
        }
        log::debug!(
            "Transition {}: input gate values: {}",
            i,
            input_bits.iter().collect::<String>()
        );
    }
    // For each gate, print its type and input operand info
    for (gate_idx, gate) in gate_fn.gates.iter().enumerate() {
        log::debug!("Gate {}: {:?}", gate_idx, gate);
        for operand in gate.get_operands() {
            let node = &gate_fn.gates[operand.node.id];
            log::debug!("  Input operand node {}: {:?}", operand.node.id, node);
        }
    }
    // For each consecutive pair, count toggles at all gate outputs (AND2 nodes
    // only)
    let and2_indices: Vec<usize> = gate_fn
        .gates
        .iter()
        .enumerate()
        .filter_map(|(idx, gate)| {
            if matches!(gate, crate::gate::AigNode::And2 { .. }) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
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
        gate::AigBitVector,
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
            stats.gate_output_toggles, stats.gate_input_toggles, stats.primary_input_toggles, stats.primary_output_toggles
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
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::test_utils::{ir_value_bf16_to_flat_ir_bits, load_bf16_add_sample, make_bf16, Opt};
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
            let a = bf16::from_f32(rng.gen::<f32>());
            let b = bf16::from_f32(rng.gen::<f32>());
            let a_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(a));
            let b_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(b));
            batch_inputs.push(vec![a_bits.clone(), b_bits.clone()]);
        }
        // count_toggles expects at least 2 inputs
        assert!(batch_inputs.len() >= 2);
        let stats = count_toggles(gate_fn, &batch_inputs);
        println!(
            "bf16 adder: gate_output_toggles = {}, gate_input_toggles = {}, primary_input_toggles = {}, primary_output_toggles = {}",
            stats.gate_output_toggles, stats.gate_input_toggles, stats.primary_input_toggles, stats.primary_output_toggles
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
            let a = bf16::from_f32(rng.gen::<f32>());
            let b = bf16::from_f32(rng.gen::<f32>());
            let a_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(a));
            let b_bits = ir_value_bf16_to_flat_ir_bits(&make_bf16(b));
            batch_inputs.push(vec![a_bits.clone(), b_bits.clone()]);
        }
        assert!(batch_inputs.len() >= 2);
        let stats = count_toggles(gate_fn, &batch_inputs);
        println!(
            "bf16 mul: gate_output_toggles = {}, gate_input_toggles = {}, primary_input_toggles = {}, primary_output_toggles = {}",
            stats.gate_output_toggles, stats.gate_input_toggles, stats.primary_input_toggles, stats.primary_output_toggles
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
