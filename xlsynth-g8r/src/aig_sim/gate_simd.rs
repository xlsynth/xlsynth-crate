// SPDX-License-Identifier: Apache-2.0

//! SIMD gate-level interpreter specialised for 256-wide batches.
//!
//! The intent is to exploit the fact that, for our purposes, a *single* operand
//! value in the AIG can be represented as a 256-bit boolean vector where the
//! Nth bit corresponds to the value of that operand in sample N (0≤N<256).
//!
//! All computations therefore boil down to bit-wise operations on these
//! 256-bit vectors.

use crate::aig::gate::{AigNode, GateFn};
use core::simd::u64x4;
use std::ops::{BitAnd, Not};

/// A fixed-width boolean vector with 256 lanes.
///
/// Internally this is represented as 4 × 64-bit limbs laid out least
/// significant limb first – lane *i* lives in bit *(i % 64)* of
/// `words[i / 64]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vec256(pub u64x4);

impl Vec256 {
    #[inline]
    pub const fn zero() -> Self {
        Self(u64x4::from_array([0; 4]))
    }

    #[inline]
    pub const fn splat(bit: bool) -> Self {
        Self(u64x4::from_array(if bit { [u64::MAX; 4] } else { [0; 4] }))
    }

    #[inline]
    pub fn apply_neg(self, negated: bool) -> Self {
        if negated { !self } else { self }
    }

    /// Packs 256 boolean samples into a `Vec256`.
    pub fn from_samples(samples: &[bool]) -> Self {
        assert_eq!(samples.len(), 256, "from_samples() requires 256 bits");
        let mut words = [0u64; 4];
        for (i, &bit) in samples.iter().enumerate() {
            if bit {
                let lane = i / 64;
                let offset = i % 64;
                words[lane] |= 1u64 << offset;
            }
        }
        Self(u64x4::from_array(words))
    }

    #[inline]
    pub fn to_array(self) -> [u64; 4] {
        self.0.to_array()
    }
}

impl Not for Vec256 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl BitAnd for Vec256 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// The result of a SIMD simulation – just the output bit-vectors.
#[derive(Debug, Clone)]
pub struct GateSimdResult {
    pub outputs: Vec<Vec256>,
}

/// Evaluates `gate_fn` on a *batch* of 256 input samples supplied in SIMD
/// form.
///
/// The caller must flatten all input bits (across all declared inputs and their
/// constituent bit-vectors) into the `inputs` slice in the same order that
/// `gate_fn.inputs` enumerates them **from LSb→MSb**.
///
/// Panics if the batch size is not exactly 256 or if the number of supplied
/// input vectors does not match the total input bit-count.
pub fn eval(gate_fn: &GateFn, inputs: &[Vec256]) -> GateSimdResult {
    // Sanity-check that the batch size is 256 – this is a *fixed* requirement
    // for this interpreter variant.
    debug_assert_eq!(core::mem::size_of::<Vec256>(), 32);

    // Seed non-negated input values.
    let total_input_bits: usize = gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum();
    assert_eq!(
        inputs.len(),
        total_input_bits,
        "input vector length mismatch"
    );

    let mut env: Vec<Vec256> = vec![Vec256::zero(); gate_fn.gates.len()];
    let mut next_input_index = 0;
    for input in &gate_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            env[bit.node.id] = inputs[next_input_index];
            next_input_index += 1;
        }
    }
    assert_eq!(next_input_index, total_input_bits);

    // Evaluate gates in topological order.
    for aig_ref in gate_fn.post_order_refs() {
        match gate_fn.get(aig_ref) {
            AigNode::Input { .. } => {
                // Already seeded above.
            }
            AigNode::Literal(value) => {
                env[aig_ref.id] = Vec256::splat(*value);
            }
            AigNode::And2 { a, b, .. } => {
                let a_val = env[a.node.id].apply_neg(a.negated);
                let b_val = env[b.node.id].apply_neg(b.negated);
                env[aig_ref.id] = a_val & b_val;
            }
        }
    }

    // Collect outputs.
    let mut outputs: Vec<Vec256> = Vec::new();
    for output in &gate_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            let value = env[bit.node.id].apply_neg(bit.negated);
            outputs.push(value);
        }
    }

    GateSimdResult { outputs }
}

/// Evaluates `gate_fn` on `inputs` and returns the total Hamming distance
/// between the produced outputs and `target_outputs`.
///
/// `inputs` and `target_outputs` must be batches of 256 samples flattened in
/// the same format required by [`eval`]. Each output bit-vector in
/// `target_outputs` corresponds to the matching output bit-vector produced by
/// `gate_fn`.
///
/// # Panics
/// Panics if `target_outputs.len()` does not match the number of output bits of
/// `gate_fn`.
pub fn eval_correctness_distance(
    gate_fn: &GateFn,
    inputs: &[Vec256],
    target_outputs: &[Vec256],
) -> usize {
    let result = eval(gate_fn, inputs);
    assert_eq!(
        result.outputs.len(),
        target_outputs.len(),
        "mismatching number of output vectors"
    );

    let mut distance = 0usize;
    for (got, want) in result.outputs.iter().zip(target_outputs.iter()) {
        let got_words = got.to_array();
        let want_words = want.to_array();
        for (a, b) in got_words.iter().zip(want_words.iter()) {
            distance += (a ^ b).count_ones() as usize;
        }
    }

    distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Packs an array of `bool` into a `Vec256` – thin wrapper over
    /// `Vec256::from_samples` for test ergonomics.
    fn pack(samples: &[bool]) -> Vec256 {
        Vec256::from_samples(samples)
    }

    #[test]
    fn test_simd_bitwise_and_1bit() {
        let mut gb = GateBuilder::new("simd_and".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 1);
        let input_b = gb.add_input("b".to_string(), 1);
        let and_node = gb.add_and_vec(&input_a, &input_b);
        gb.add_output("out".to_string(), and_node);
        let gate_fn = gb.build();

        // Generate deterministic random inputs for 256 samples.
        let mut rng = StdRng::seed_from_u64(42);
        let mut a_samples = [false; 256];
        let mut b_samples = [false; 256];
        for i in 0..256 {
            a_samples[i] = rng.r#gen();
            b_samples[i] = rng.r#gen();
        }

        let simd_inputs = vec![pack(&a_samples), pack(&b_samples)];
        let result = eval(&gate_fn, &simd_inputs);
        assert_eq!(result.outputs.len(), 1);

        // Build expected output vector.
        let mut expected = [false; 256];
        for i in 0..256 {
            expected[i] = a_samples[i] & b_samples[i];
        }
        let expected_vec = pack(&expected);
        assert_eq!(result.outputs[0], expected_vec);
    }

    #[test]
    fn test_eval_correctness_distance() {
        let mut gb_and = GateBuilder::new("and_fn".to_string(), GateBuilderOptions::opt());
        let input_a = gb_and.add_input("a".to_string(), 1);
        let input_b = gb_and.add_input("b".to_string(), 1);
        let and_node = gb_and.add_and_vec(&input_a, &input_b);
        gb_and.add_output("out".to_string(), and_node);
        let gfn_and = gb_and.build();

        let mut gb_passthrough = GateBuilder::new("a_fn".to_string(), GateBuilderOptions::opt());
        let input_a_pt = gb_passthrough.add_input("a".to_string(), 1);
        let _input_b_pt = gb_passthrough.add_input("b".to_string(), 1);
        gb_passthrough.add_output("out".to_string(), input_a_pt.clone());
        let gfn_a = gb_passthrough.build();

        let mut rng = StdRng::seed_from_u64(123);
        let mut a_samples = [false; 256];
        let mut b_samples = [false; 256];
        for i in 0..256 {
            a_samples[i] = rng.r#gen();
            b_samples[i] = rng.r#gen();
        }

        let simd_inputs = vec![pack(&a_samples), pack(&b_samples)];
        let target_outputs = eval(&gfn_and, &simd_inputs).outputs;

        // Distance to the correct gate should be zero.
        assert_eq!(
            eval_correctness_distance(&gfn_and, &simd_inputs, &target_outputs),
            0
        );

        // Manually compute expected distance when using the passthrough gate.
        let mut expected_distance = 0usize;
        for i in 0..256 {
            let correct = a_samples[i] & b_samples[i];
            let cand = a_samples[i];
            if correct != cand {
                expected_distance += 1;
            }
        }
        assert_eq!(
            eval_correctness_distance(&gfn_a, &simd_inputs, &target_outputs),
            expected_distance
        );
    }
}
