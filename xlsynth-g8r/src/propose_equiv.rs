// SPDX-License-Identifier: Apache-2.0

//! Functionality for proposing equivalence classes via concrete simulation.

use crate::gate::{AigRef, GateFn};
use crate::gate_sim::{self, Collect, GateSimResult};
use bitvec::vec::BitVec;
use xlsynth::IrBits;

use rand::Rng;
use std::collections::HashMap;
use std::hash::DefaultHasher;
use std::hash::{Hash, Hasher};

fn gen_random_input_bits(bit_count: usize, rng: &mut impl Rng) -> IrBits {
    let value = rng.gen::<u64>();
    let value_masked = value & ((1 << bit_count) - 1);
    IrBits::make_ubits(bit_count, value_masked).unwrap()
}

fn gen_random_inputs(gate_fn: &GateFn, rng: &mut impl Rng) -> Vec<IrBits> {
    gate_fn
        .inputs
        .iter()
        .map(|input| gen_random_input_bits(input.bit_vector.get_bit_count(), rng))
        .collect()
}

/// Returns a mapping from hash value (a hash over the history for a given gate
/// as it's fed random samples) to a sequence of the nodes that had the same
/// history.
pub fn propose_equiv(
    gate_fn: &GateFn,
    input_sample_count: usize,
    rng: &mut impl Rng,
) -> HashMap<u64, Vec<AigRef>> {
    let mut history: Vec<BitVec> = Vec::with_capacity(input_sample_count);
    for _ in 0..input_sample_count {
        let inputs: Vec<IrBits> = gen_random_inputs(gate_fn, rng);
        let result: GateSimResult = gate_sim::eval(gate_fn, &inputs, Collect::All);
        history.push(result.all_values.unwrap());
    }

    let history_hashes: Vec<u64> = history
        .iter()
        .map(|bit_vec| {
            let mut hasher = DefaultHasher::new();
            bit_vec.hash(&mut hasher);
            hasher.finish()
        })
        .collect();

    let mut equiv_classes: HashMap<u64, Vec<AigRef>> = HashMap::new();
    for (node_index, hash) in history_hashes.iter().enumerate() {
        equiv_classes
            .entry(*hash)
            .or_insert_with(Vec::new)
            .push(AigRef { id: node_index });
    }

    equiv_classes
}
