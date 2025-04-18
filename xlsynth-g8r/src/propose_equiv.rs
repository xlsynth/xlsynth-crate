// SPDX-License-Identifier: Apache-2.0

//! Functionality for proposing equivalence classes via concrete simulation.

use crate::gate::{AigNode, AigRef, GateFn};
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
    // samples x gate values -- would be nicer to have a BitMatrix
    let gate_count = gate_fn.gates.len();
    let mut history: Vec<BitVec> = Vec::with_capacity(input_sample_count);

    // Push `input_sample_count` random samples through the gate function and
    // collect the history of all the nodes.
    for _ in 0..input_sample_count {
        let inputs: Vec<IrBits> = gen_random_inputs(gate_fn, rng);
        let result: GateSimResult = gate_sim::eval(gate_fn, &inputs, Collect::All);
        history.push(result.all_values.unwrap());
    }

    // Collects the history for the given `gate_index` across all samples.
    let collect_across_samples = |gate_index: usize| -> BitVec {
        history.iter().map(|h| -> bool { h[gate_index] }).collect()
    };

    let history_hashes: Vec<u64> = (0..gate_count)
        .map(|i| {
            let gate_history = collect_across_samples(i);
            let mut hasher = DefaultHasher::new();
            gate_history.hash(&mut hasher);
            hasher.finish()
        })
        .collect();

    assert_eq!(
        history_hashes.len(),
        gate_count,
        "We should have a history hash for each gate"
    );

    let mut equiv_classes: HashMap<u64, Vec<AigRef>> = HashMap::new();
    for (node_index, hash) in history_hashes.iter().enumerate() {
        let node = &gate_fn.gates[node_index];
        if matches!(node, AigNode::Input { .. } | AigNode::Literal(..)) {
            continue;
        }
        equiv_classes
            .entry(*hash)
            .or_insert_with(Vec::new)
            .push(AigRef { id: node_index });
    }

    equiv_classes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{setup_graph_with_redundancies, setup_simple_graph};
    use rand::SeedableRng;

    #[test]
    fn test_propose_equiv_simple_graph() {
        let _ = env_logger::builder().is_test(true).try_init();
        let graph = setup_simple_graph();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let equiv_classes = propose_equiv(&graph.g, 4096, &mut seeded_rng);
        assert_eq!(equiv_classes.len(), 4);
        log::info!("equiv_classes: {:?}", equiv_classes);
        let mut values = equiv_classes.values().collect::<Vec<_>>();
        // Sort them so we can do stable tests.
        values.sort();
        assert_eq!(
            *values,
            vec![
                &vec![graph.a.node],
                &vec![graph.b.node],
                &vec![graph.c.node],
                &vec![graph.o.node],
            ]
        );
    }

    #[test]
    fn test_propose_equiv_graph_with_redundancies() {
        let _ = env_logger::builder().is_test(true).try_init();
        let graph = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let equiv_classes = propose_equiv(&graph.g, 4096, &mut seeded_rng);
        log::info!("equiv_classes: {:?}", equiv_classes);
        assert_eq!(equiv_classes.len(), 2);
        let mut values = equiv_classes.values().collect::<Vec<_>>();
        // Sort them so we can do stable tests.
        values.sort();
        assert_eq!(
            *values,
            vec![
                &vec![graph.inner0.node, graph.inner1.node],
                &vec![graph.outer0.node, graph.outer1.node],
            ]
        );
    }
}
