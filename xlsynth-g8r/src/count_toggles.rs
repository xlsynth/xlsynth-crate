// SPDX-License-Identifier: Apache-2.0

use crate::gate::GateFn;
use crate::gate_sim::{eval, Collect};
use bitvec::vec::BitVec;
use std::iter::zip;
use xlsynth::IrBits;

/// Counts toggles at gate outputs and primary inputs for a batch of input
/// vectors.
///
/// # Arguments
/// * `gate_fn` - The gate function to simulate.
/// * `batch_inputs` - A batch of input vectors, each is a vector of IrBits (one
///   per input port).
///
/// # Returns
/// (output_toggles, input_toggles):
///   - output_toggles: total number of toggles at all gate outputs across the
///     batch
///   - input_toggles: total number of toggles at all primary inputs across the
///     batch
pub fn count_toggles(gate_fn: &GateFn, batch_inputs: &[Vec<IrBits>]) -> (usize, usize) {
    assert!(
        batch_inputs.len() >= 2,
        "Need at least two input vectors to count toggles"
    );
    // Run eval for each input vector, collect all_values (gate outputs)
    let mut all_values_vec: Vec<BitVec> = Vec::with_capacity(batch_inputs.len());
    for input_vec in batch_inputs {
        let result = eval(gate_fn, input_vec, Collect::All);
        let all_values = result
            .all_values
            .expect("Collect::All should produce all_values");
        all_values_vec.push(all_values);
    }
    // For each consecutive pair, count toggles at all gate outputs (all nodes)
    let mut output_toggles = 0;
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        assert_eq!(prev.len(), next.len());
        output_toggles += prev.iter().zip(next.iter()).filter(|(a, b)| a != b).count();
    }
    // For each consecutive pair, count toggles at all gate inputs
    let mut input_toggles = 0;
    // For each gate in the circuit
    for (gate_idx, gate) in gate_fn.gates.iter().enumerate() {
        // For each input operand to the gate
        for operand in gate.get_operands() {
            // For each consecutive pair of input vectors
            for pair in all_values_vec.windows(2) {
                let (prev, next) = (&pair[0], &pair[1]);
                let prev_val = prev[operand.node.id] ^ operand.negated;
                let next_val = next[operand.node.id] ^ operand.negated;
                if prev_val != next_val {
                    input_toggles += 1;
                }
            }
        }
    }
    (output_toggles, input_toggles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use xlsynth::IrBits;

    #[test]
    fn test_count_toggles_simple_and() {
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
        let (output_toggles, input_toggles) = count_toggles(&gate_fn, &batch_inputs);
        assert!(output_toggles > 0);
        assert!(input_toggles > 0);
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
        let (output_toggles, input_toggles) = count_toggles(gate_fn, &batch_inputs);
        println!("bf16 adder: output_toggles = {output_toggles}, input_toggles = {input_toggles}");
        assert!(output_toggles > 0, "Should see output toggles");
        assert!(input_toggles > 0, "Should see input toggles");
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
        let (output_toggles, input_toggles) = count_toggles(gate_fn, &batch_inputs);
        println!("bf16 mul: output_toggles = {output_toggles}, input_toggles = {input_toggles}");
        assert!(output_toggles > 0, "Should see output toggles");
        assert!(input_toggles > 0, "Should see input toggles");
    }
}
