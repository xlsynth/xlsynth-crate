// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmark that exercises `count_toggles` on a moderately-sized
//! circuit and stimulus batch.  The goal is to provide a quick, reproducible
//! way to observe the memory-usage and runtime improvements of the streaming
//! implementation introduced in PR #XXX.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::RngCore;
use xlsynth::IrBits;
use bitvec::vec::BitVec;
use xlsynth_g8r::{count_toggles, gate_sim::{eval, Collect}};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

/// Number of input vectors in the batch.  Large enough to show a difference
/// between the old and new algorithms but still quick to run inside CI.
const BATCH: usize = 1 << 9; // 512 transitions => 513 vectors

/// Width (in bits) of each primary input.
const WIDTH: usize = 4096;

fn build_circuit() -> xlsynth_g8r::gate::GateFn {
    // A simple but sizeable design: bit-wise AND of two 4096-bit inputs.
    let mut gb = GateBuilder::new("toggle_demo".into(), GateBuilderOptions::opt());
    let a = gb.add_input("a".into(), WIDTH);
    let b = gb.add_input("b".into(), WIDTH);
    let _and = gb.add_and_vec(&a, &b);
    gb.add_output("out".into(), b); // arbitrary choice
    gb.build()
}

fn generate_batch() -> Vec<Vec<IrBits>> {
    let mut rng = rand::thread_rng();
    (0..=BATCH) // BATCH + 1 vectors → BATCH transitions
        .map(|_| {
            vec![
                IrBits::make_ubits(WIDTH, rng.next_u64()).unwrap(),
                IrBits::make_ubits(WIDTH, rng.next_u64()).unwrap(),
            ]
        })
        .collect()
}

fn bench_count_toggles(c: &mut Criterion) {
    let gate_fn = build_circuit();
    let batch = generate_batch();

    // Benchmark the current (streaming) implementation.
    c.bench_function("count_toggles_stream", |b| {
        b.iter(|| count_toggles(&gate_fn, &batch))
    });

    // Benchmark a faithful re-implementation of the legacy algorithm that kept
    // every BitVec alive and did multiple passes.  Implemented locally to avoid
    // polluting the library code.
    c.bench_function("count_toggles_legacy", |b| {
        b.iter(|| legacy_count_toggles(&gate_fn, &batch))
    });
}

/// A local, intentionally slow copy of the pre-refactor algorithm kept only for
/// performance comparison in the benchmark.
fn legacy_count_toggles(
    gate_fn: &xlsynth_g8r::gate::GateFn,
    batch_inputs: &[Vec<IrBits>],
) -> xlsynth_g8r::count_toggles::ToggleStats {
    assert!(batch_inputs.len() >= 2);

    // Evaluate every vector and store all BitVecs.
    let mut all_values_vec: Vec<BitVec> = Vec::with_capacity(batch_inputs.len());
    for input_vec in batch_inputs {
        let result = eval(gate_fn, input_vec, Collect::AllWithInputs);
        all_values_vec.push(result.all_values.unwrap());
    }

    // Precompute indices.
    let and2_indices: Vec<usize> = gate_fn
        .gates
        .iter()
        .enumerate()
        .filter_map(|(idx, g)| {
            if matches!(g, xlsynth_g8r::gate::AigNode::And2 { .. }) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    let output_bit_indices: Vec<usize> = gate_fn
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb().map(|b| b.node.id))
        .collect();

    let mut gate_output_toggles = 0;
    let mut gate_input_toggles = 0;
    let mut primary_input_toggles = 0;
    let mut primary_output_toggles = 0;

    // gate_input toggle needs operands list.
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);

        // Gate outputs.
        for &idx in &and2_indices {
            if prev[idx] != next[idx] {
                gate_output_toggles += 1;
            }
        }

        // Gate inputs.
        for gate in gate_fn.gates.iter() {
            for operand in gate.get_operands() {
                let prev_val = prev[operand.node.id] ^ operand.negated;
                let next_val = next[operand.node.id] ^ operand.negated;
                if prev_val != next_val {
                    gate_input_toggles += 1;
                }
            }
        }

        // Primary inputs.
        // Need the raw stimulus vectors; compute index of pair in batch_inputs.
    }

    // Primary input toggles – do another pass over raw inputs.
    for pair in batch_inputs.windows(2) {
        let (prev_v, next_v) = (&pair[0], &pair[1]);
        for (prev_bits, next_bits) in prev_v.iter().zip(next_v.iter()) {
            for i in 0..prev_bits.get_bit_count() {
                if prev_bits.get_bit(i).unwrap() != next_bits.get_bit(i).unwrap() {
                    primary_input_toggles += 1;
                }
            }
        }
    }

    // Primary output toggles.
    for pair in all_values_vec.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        for &idx in &output_bit_indices {
            if prev[idx] != next[idx] {
                primary_output_toggles += 1;
            }
        }
    }

    xlsynth_g8r::count_toggles::ToggleStats {
        gate_output_toggles,
        gate_input_toggles,
        primary_input_toggles,
        primary_output_toggles,
    }
}

criterion_group!(benches, bench_count_toggles);
criterion_main!(benches);
