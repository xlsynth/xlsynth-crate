// SPDX-License-Identifier: Apache-2.0
#![feature(portable_simd)]

//! Benchmarks scalar vs SIMD gate evaluators on the bf16_add circuit.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth::IrBits;
use xlsynth_g8r::{
    gate_sim::{self, Collect},
    gate_simd,
    test_utils::{Opt, load_bf16_add_sample},
};

/// Number of samples that the SIMD evaluator expects (one per bit lane).
const BATCH: usize = 256;

/// Prepare the same set of random inputs in two representations:
/// 1. `scalar_inputs` – a Vec of length BATCH, each element is a slice of
///    `IrBits` (`&[IrBits; 2]`) suitable for `gate_sim::eval`.
/// 2. `simd_inputs` – a Vec of `Vec256`, each representing one input bit over
///    the 256-wide batch, suitable for `gate_simd::eval`.
fn prepare_inputs() -> (Vec<[IrBits; 2]>, Vec<xlsynth_g8r::gate_simd::Vec256>) {
    let mut rng = StdRng::seed_from_u64(0x5eed);

    // First gather the scalar representation.
    let mut scalar_inputs: Vec<[IrBits; 2]> = Vec::with_capacity(BATCH);
    let mut bit_accumulators = vec![[0u64; 4]; 32]; // 32 input bits → 4×64-bit limbs

    for lane in 0..BATCH {
        let x: u16 = rng.r#gen();
        let y: u16 = rng.r#gen();

        let x_bits = IrBits::make_ubits(16, x as u64).unwrap();
        let y_bits = IrBits::make_ubits(16, y as u64).unwrap();
        scalar_inputs.push([x_bits.clone(), y_bits.clone()]);

        let limb = (lane / 64) as usize;
        let offset = (lane % 64) as u64;

        // Scatter the bits into the accumulators.
        for bit in 0..16 {
            if x & (1 << bit) != 0 {
                bit_accumulators[bit][limb] |= 1u64 << offset;
            }
            if y & (1 << bit) != 0 {
                bit_accumulators[16 + bit][limb] |= 1u64 << offset;
            }
        }
    }

    // Convert accumulators into Vec256s.
    let simd_inputs: Vec<xlsynth_g8r::gate_simd::Vec256> = bit_accumulators
        .into_iter()
        .map(|words| xlsynth_g8r::gate_simd::Vec256(core::simd::u64x4::from_array(words)))
        .collect();

    (scalar_inputs, simd_inputs)
}

fn sim_vs_simd_benchmark(c: &mut Criterion) {
    let sample = load_bf16_add_sample(Opt::Yes);
    let gate_fn = &sample.gate_fn;

    let (scalar_inputs, simd_inputs) = prepare_inputs();

    let mut group = c.benchmark_group("gate_sim_vs_simd_bf16_add");

    group.bench_function("scalar_gate_sim", |b| {
        b.iter(|| {
            for pair in &scalar_inputs {
                // gate_sim::eval allocates a Vec internally; keep it in scope.
                let _ = gate_sim::eval(black_box(gate_fn), black_box(&pair[..]), Collect::None);
            }
        })
    });

    group.bench_function("simd_gate_simd", |b| {
        b.iter(|| {
            let _ = gate_simd::eval(black_box(gate_fn), black_box(&simd_inputs));
        })
    });

    group.finish();
}

criterion_group!(benches, sim_vs_simd_benchmark);
criterion_main!(benches);
