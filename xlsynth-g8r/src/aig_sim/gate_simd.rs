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
use xlsynth::IrBits;

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
    pub const fn from_words(words: [u64; 4]) -> Self {
        Self(u64x4::from_array(words))
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

    /// Returns the boolean value stored in `lane`.
    #[inline]
    pub fn get_lane(self, lane: usize) -> bool {
        assert!(lane < 256, "Vec256 lane out of bounds: {lane}");
        let words = self.to_array();
        ((words[lane / 64] >> (lane % 64)) & 1) != 0
    }

    /// Returns the number of set lanes in this packed boolean vector.
    #[inline]
    pub fn popcount(self) -> usize {
        self.to_array()
            .into_iter()
            .map(|word| word.count_ones() as usize)
            .sum()
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

/// Validates one ordered batch of GateFn input vectors.
///
/// Each sample must provide one `IrBits` value per declared GateFn input port,
/// with a width matching that port.
pub fn validate_ordered_batch_inputs(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
) -> Result<(), String> {
    let expected_input_count = gate_fn.inputs.len();
    for (sample_index, sample) in batch_inputs.iter().enumerate() {
        if sample.len() != expected_input_count {
            return Err(format!(
                "sample {} has {} inputs, expected {}",
                sample_index + 1,
                sample.len(),
                expected_input_count
            ));
        }
        for (input_index, (bits, gate_input)) in
            sample.iter().zip(gate_fn.inputs.iter()).enumerate()
        {
            let expected_width = gate_input.get_bit_count();
            if bits.get_bit_count() != expected_width {
                return Err(format!(
                    "sample {} input {} has width {}, expected {}",
                    sample_index + 1,
                    input_index,
                    bits.get_bit_count(),
                    expected_width
                ));
            }
        }
    }
    Ok(())
}

fn check_simd_input_shape(gate_fn: &GateFn, inputs: &[Vec256]) -> usize {
    // Sanity-check that the batch size is 256 – this is a *fixed* requirement
    // for this interpreter variant.
    debug_assert_eq!(core::mem::size_of::<Vec256>(), 32);

    let total_input_bits: usize = gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum();
    assert_eq!(
        inputs.len(),
        total_input_bits,
        "input vector length mismatch"
    );
    total_input_bits
}

pub(crate) fn pack_ordered_input_chunk(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
    chunk_start: usize,
    chunk_len: usize,
) -> Vec<Vec256> {
    assert!(
        (1..=256).contains(&chunk_len),
        "chunk length must be in 1..=256; got {chunk_len}"
    );
    assert!(
        chunk_start + chunk_len <= batch_inputs.len(),
        "input chunk exceeds ordered batch: start {chunk_start}, len {chunk_len}, batch len {}",
        batch_inputs.len()
    );
    let total_input_bits: usize = gate_fn
        .inputs
        .iter()
        .map(|input| input.get_bit_count())
        .sum();
    let mut packed_inputs = Vec::with_capacity(total_input_bits);
    for (input_index, gate_input) in gate_fn.inputs.iter().enumerate() {
        for bit_index in 0..gate_input.get_bit_count() {
            let mut words = [0u64; 4];
            for lane in 0..chunk_len {
                let bit = batch_inputs[chunk_start + lane][input_index]
                    .get_bit(bit_index)
                    .unwrap();
                if bit {
                    words[lane / 64] |= 1u64 << (lane % 64);
                }
            }
            packed_inputs.push(Vec256::from_words(words));
        }
    }
    packed_inputs
}

fn seed_input_values(gate_fn: &GateFn, inputs: &[Vec256], env: &mut [Vec256]) {
    let total_input_bits = check_simd_input_shape(gate_fn, inputs);
    let mut next_input_index = 0;
    for input in &gate_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            env[bit.node.id] = inputs[next_input_index];
            next_input_index += 1;
        }
    }
    assert_eq!(next_input_index, total_input_bits);
}

fn debug_assert_dense_topological_order(gate_fn: &GateFn, live_nodes: &[bool]) {
    if !cfg!(debug_assertions) {
        return;
    }
    for (node_index, node) in gate_fn.gates.iter().enumerate() {
        if !live_nodes[node_index] {
            continue;
        }
        if let AigNode::And2 { a, b, .. } = node {
            debug_assert!(
                live_nodes[a.node.id],
                "live AND node %{} depends on non-live operand %{}",
                node_index, a.node.id
            );
            debug_assert!(
                live_nodes[b.node.id],
                "live AND node %{} depends on non-live operand %{}",
                node_index, b.node.id
            );
            debug_assert!(
                a.node.id < node_index,
                "dense SIMD evaluation requires topological gate order: node %{} depends on later/equal operand %{}",
                node_index,
                a.node.id
            );
            debug_assert!(
                b.node.id < node_index,
                "dense SIMD evaluation requires topological gate order: node %{} depends on later/equal operand %{}",
                node_index,
                b.node.id
            );
        }
    }
}

/// Evaluates `gate_fn` and returns every node's 256-wide value.
///
/// The returned vector is indexed by `AigRef::id`. Nodes outside the output
/// cone are left at zero, matching the scalar simulator's `Collect::All`
/// behavior for unvisited nodes.
pub fn eval_all_node_values(gate_fn: &GateFn, inputs: &[Vec256]) -> Vec<Vec256> {
    let mut env: Vec<Vec256> = vec![Vec256::zero(); gate_fn.gates.len()];
    seed_input_values(gate_fn, inputs, &mut env);

    // Evaluate gates in topological order.
    for aig_ref in gate_fn.post_order_refs() {
        match gate_fn.get(aig_ref) {
            AigNode::Input { .. } => {
                // Already seeded above.
            }
            AigNode::Literal { value, .. } => {
                env[aig_ref.id] = Vec256::splat(*value);
            }
            AigNode::And2 { a, b, .. } => {
                let a_val = env[a.node.id].apply_neg(a.negated);
                let b_val = env[b.node.id].apply_neg(b.negated);
                env[aig_ref.id] = a_val & b_val;
            }
        }
    }

    env
}

/// Evaluates output-reachable nodes in dense node-id order into `env`.
///
/// This variant is for hot loops that repeatedly simulate the same graph. The
/// caller owns `env`, so its allocation is reused across batches. Entries for
/// non-live nodes are intentionally left unspecified; callers must only read
/// nodes whose `live_nodes` bit is set.
///
/// The graph's dense `gates` order must be topological: every live `And2`
/// operand must have a smaller node id than the gate that uses it. This is the
/// order produced by `GateBuilder` and by `dce_safe` after `bulk_replace`.
pub fn eval_live_node_values_dense_into(
    gate_fn: &GateFn,
    inputs: &[Vec256],
    live_nodes: &[bool],
    env: &mut Vec<Vec256>,
) {
    assert_eq!(
        live_nodes.len(),
        gate_fn.gates.len(),
        "live node mask length mismatch"
    );
    if env.len() != gate_fn.gates.len() {
        env.resize(gate_fn.gates.len(), Vec256::zero());
    }
    seed_input_values(gate_fn, inputs, env.as_mut_slice());
    debug_assert_dense_topological_order(gate_fn, live_nodes);

    for (node_index, node) in gate_fn.gates.iter().enumerate() {
        if !live_nodes[node_index] {
            continue;
        }
        match node {
            AigNode::Input { .. } => {
                // Already seeded above.
            }
            AigNode::Literal { value, .. } => {
                env[node_index] = Vec256::splat(*value);
            }
            AigNode::And2 { a, b, .. } => {
                let a_val = env[a.node.id].apply_neg(a.negated);
                let b_val = env[b.node.id].apply_neg(b.negated);
                env[node_index] = a_val & b_val;
            }
        }
    }
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
    let env = eval_all_node_values(gate_fn, inputs);

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

fn unpack_output_chunk(
    gate_fn: &GateFn,
    flat_outputs: &[Vec256],
    chunk_len: usize,
) -> Vec<Vec<IrBits>> {
    let expected_output_bits: usize = gate_fn
        .outputs
        .iter()
        .map(|output| output.get_bit_count())
        .sum();
    assert_eq!(
        flat_outputs.len(),
        expected_output_bits,
        "SIMD output vector length mismatch"
    );
    let mut chunk_outputs = Vec::with_capacity(chunk_len);
    for lane in 0..chunk_len {
        let mut sample_outputs = Vec::with_capacity(gate_fn.outputs.len());
        let mut flat_output_index = 0usize;
        for output in &gate_fn.outputs {
            let output_width = output.get_bit_count();
            let bits_lsb_first = (0..output_width)
                .map(|bit_index| flat_outputs[flat_output_index + bit_index].get_lane(lane))
                .collect::<Vec<bool>>();
            sample_outputs.push(IrBits::from_lsb_is_0(&bits_lsb_first));
            flat_output_index += output_width;
        }
        chunk_outputs.push(sample_outputs);
    }
    chunk_outputs
}

/// Evaluates an ordered GateFn input batch with the 256-lane SIMD evaluator.
///
/// Samples are chunked internally when the batch is larger than 256 entries.
/// The returned outer vector is sample-major and preserves the input order.
pub fn eval_ordered_batch(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
) -> Result<Vec<Vec<IrBits>>, String> {
    validate_ordered_batch_inputs(gate_fn, batch_inputs)?;
    let mut batch_outputs = Vec::with_capacity(batch_inputs.len());
    for (chunk_index, chunk) in batch_inputs.chunks(256).enumerate() {
        let chunk_start = chunk_index * 256;
        let packed_inputs =
            pack_ordered_input_chunk(gate_fn, batch_inputs, chunk_start, chunk.len());
        let flat_outputs = eval(gate_fn, &packed_inputs).outputs;
        batch_outputs.extend(unpack_output_chunk(gate_fn, &flat_outputs, chunk.len()));
    }
    Ok(batch_outputs)
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
        distance += Vec256::from_words([
            got_words[0] ^ want_words[0],
            got_words[1] ^ want_words[1],
            got_words[2] ^ want_words[2],
            got_words[3] ^ want_words[3],
        ])
        .popcount();
    }

    distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::gate::AigBitVector;
    use crate::aig_sim::gate_sim::{self, Collect};
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

    #[test]
    fn test_dense_eval_reuses_env_for_live_nodes() {
        let mut gb = GateBuilder::new("dense_eval".to_string(), GateBuilderOptions::no_opt());
        let input_a: crate::aig::AigOperand = gb.add_input("a".to_string(), 1).try_into().unwrap();
        let input_b: crate::aig::AigOperand = gb.add_input("b".to_string(), 1).try_into().unwrap();
        let input_c: crate::aig::AigOperand = gb.add_input("c".to_string(), 1).try_into().unwrap();
        let live_and = gb.add_and_binary(input_a, input_b);
        let _dead_and = gb.add_and_binary(input_b, input_c);
        gb.add_output("out".to_string(), live_and.into());
        let gate_fn = gb.build();

        let mut rng = StdRng::seed_from_u64(456);
        let mut a_samples = [false; 256];
        let mut b_samples = [false; 256];
        let mut c_samples = [false; 256];
        for i in 0..256 {
            a_samples[i] = rng.r#gen();
            b_samples[i] = rng.r#gen();
            c_samples[i] = rng.r#gen();
        }
        let simd_inputs = vec![pack(&a_samples), pack(&b_samples), pack(&c_samples)];

        let all_values = eval_all_node_values(&gate_fn, &simd_inputs);
        let mut live_nodes = vec![false; gate_fn.gates.len()];
        live_nodes[input_a.node.id] = true;
        live_nodes[input_b.node.id] = true;
        live_nodes[live_and.node.id] = true;

        let mut dense_values = Vec::new();
        eval_live_node_values_dense_into(&gate_fn, &simd_inputs, &live_nodes, &mut dense_values);
        let reused_ptr = dense_values.as_ptr();
        for (node_index, is_live) in live_nodes.iter().copied().enumerate() {
            if is_live {
                assert_eq!(dense_values[node_index], all_values[node_index]);
            }
        }

        eval_live_node_values_dense_into(&gate_fn, &simd_inputs, &live_nodes, &mut dense_values);
        assert_eq!(dense_values.as_ptr(), reused_ptr);
    }

    #[test]
    fn test_eval_ordered_batch_matches_scalar_across_chunks() {
        let mut gb = GateBuilder::new("simd_batch".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 3);
        let input_b = gb.add_input("b".to_string(), 2);
        let xor_a = gb.add_xor_vec(&input_a, &input_a);
        let and_low = gb.add_and_binary(*input_a.get_lsb(0), *input_b.get_lsb(0));
        let out1 = gb.add_or_binary(*input_a.get_lsb(1), *input_b.get_lsb(1));
        gb.add_output("out0".to_string(), xor_a);
        gb.add_output(
            "out1".to_string(),
            AigBitVector::from_lsb_is_index_0(&[and_low, out1]),
        );
        let gate_fn = gb.build();
        let batch_inputs = (0..513)
            .map(|sample_index| {
                vec![
                    IrBits::make_ubits(3, (sample_index % 8) as u64).unwrap(),
                    IrBits::make_ubits(2, ((sample_index / 3) % 4) as u64).unwrap(),
                ]
            })
            .collect::<Vec<Vec<IrBits>>>();

        let simd_outputs = eval_ordered_batch(&gate_fn, &batch_inputs).unwrap();
        let scalar_outputs = batch_inputs
            .iter()
            .map(|sample| gate_sim::eval(&gate_fn, sample, Collect::None).outputs)
            .collect::<Vec<Vec<IrBits>>>();
        assert_eq!(simd_outputs, scalar_outputs);
    }
}
