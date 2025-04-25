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
    // For each consecutive pair, count toggles at gate outputs
    let mut output_toggles = 0;
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        assert_eq!(prev.len(), next.len());
        output_toggles += prev.iter().zip(next.iter()).filter(|(a, b)| a != b).count();
    }
    // For each consecutive pair, count toggles at primary inputs
    let mut input_toggles = 0;
    for input_pair in batch_inputs.windows(2) {
        let (prev_inputs, next_inputs) = (&input_pair[0], &input_pair[1]);
        assert_eq!(prev_inputs.len(), next_inputs.len());
        for (prev_bits, next_bits) in zip(prev_inputs, next_inputs) {
            assert_eq!(prev_bits.get_bit_count(), next_bits.get_bit_count());
            for i in 0..prev_bits.get_bit_count() {
                let a = prev_bits.get_bit(i).unwrap();
                let b = next_bits.get_bit(i).unwrap();
                if a != b {
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
