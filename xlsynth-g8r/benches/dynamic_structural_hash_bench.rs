// SPDX-License-Identifier: Apache-2.0
//! Benchmarks for the dynamic structural hash used by cut-db rewrites.

use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use xlsynth_g8r::aig::dynamic_structural_hash::DynamicStructuralHash;
use xlsynth_g8r::aig::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

#[derive(Clone)]
struct RandomGraphWorkload {
    gate_fn: GateFn,
    hit_queries: Vec<(AigOperand, AigOperand)>,
    miss_queries: Vec<(AigOperand, AigOperand)>,
    fanout_queries: Vec<AigRef>,
}

#[derive(Clone)]
struct EditGadget {
    a: AigOperand,
    b: AigOperand,
    c: AigOperand,
    d: AigOperand,
    e: AigOperand,
    g: AigOperand,
    ab: AigOperand,
    cd: AigOperand,
    root: AigOperand,
    root_duplicate: AigOperand,
    move_node: AigOperand,
    output_new: AigOperand,
    output_bit_index: usize,
}

#[derive(Clone)]
struct CutdbLikeWorkload {
    gate_fn: GateFn,
    gadgets: Vec<EditGadget>,
}

fn maybe_negate(rng: &mut Pcg64Mcg, operand: AigOperand) -> AigOperand {
    if rng.gen_bool(0.2) {
        operand.negate()
    } else {
        operand
    }
}

fn choose_biased_operand(
    rng: &mut Pcg64Mcg,
    operands: &[AigOperand],
    recent_window: usize,
) -> AigOperand {
    let recent = operands.len().min(recent_window);
    let index = if recent != 0 && rng.gen_bool(0.75) {
        operands.len() - recent + rng.gen_range(0..recent)
    } else {
        rng.gen_range(0..operands.len())
    };
    maybe_negate(rng, operands[index])
}

fn and_operands(gate_fn: &GateFn, node: AigRef) -> Option<(AigOperand, AigOperand)> {
    match gate_fn.gates[node.id] {
        AigNode::And2 { a, b, .. } => Some((a, b)),
        _ => None,
    }
}

fn build_random_graph_workload(input_count: usize, and_count: usize) -> RandomGraphWorkload {
    let mut rng = Pcg64Mcg::seed_from_u64(0x51a5_5eED_d15c_a11d);
    let mut builder = GateBuilder::new(
        format!("dynamic_structural_hash_random_{and_count}"),
        GateBuilderOptions::no_opt(),
    );

    let inputs = builder.add_input("in".to_string(), input_count);
    let mut operands = inputs
        .iter_lsb_to_msb()
        .copied()
        .collect::<Vec<AigOperand>>();
    let mut emitted_pairs = Vec::<(AigOperand, AigOperand)>::with_capacity(and_count);
    let mut and_nodes = Vec::<AigRef>::with_capacity(and_count);

    for _ in 0..and_count {
        let (lhs, rhs) = if !emitted_pairs.is_empty() && rng.gen_bool(0.08) {
            emitted_pairs[rng.gen_range(0..emitted_pairs.len())]
        } else {
            (
                choose_biased_operand(&mut rng, &operands, 256),
                choose_biased_operand(&mut rng, &operands, 256),
            )
        };
        let result = builder.add_and_binary(lhs, rhs);
        emitted_pairs.push((lhs, rhs));
        and_nodes.push(result.node);
        operands.push(result);
    }

    let output_bits = (0..512)
        .map(|_| choose_biased_operand(&mut rng, &operands, 2048))
        .collect::<Vec<_>>();
    builder.add_output(
        "out".to_string(),
        AigBitVector::from_lsb_is_index_0(&output_bits),
    );
    let gate_fn = builder.build();

    let hit_queries = and_nodes
        .iter()
        .step_by((and_nodes.len() / 2048).max(1))
        .filter_map(|node| and_operands(&gate_fn, *node))
        .take(2048)
        .collect::<Vec<_>>();

    let miss_queries = (0..2048)
        .map(|_| {
            (
                choose_biased_operand(&mut rng, &operands, 4096).negate(),
                choose_biased_operand(&mut rng, &operands, 4096),
            )
        })
        .collect::<Vec<_>>();

    let fanout_queries = and_nodes
        .iter()
        .step_by((and_nodes.len() / 2048).max(1))
        .copied()
        .take(2048)
        .collect::<Vec<_>>();

    RandomGraphWorkload {
        gate_fn,
        hit_queries,
        miss_queries,
        fanout_queries,
    }
}

fn build_cutdb_like_workload(gadget_count: usize) -> CutdbLikeWorkload {
    let mut rng = Pcg64Mcg::seed_from_u64(0xc07d_db17_5a57_eD17);
    let mut builder = GateBuilder::new(
        format!("dynamic_structural_hash_cutdb_like_{gadget_count}"),
        GateBuilderOptions::no_opt(),
    );
    let inputs = builder.add_input("in".to_string(), 512);
    let input_operands = inputs
        .iter_lsb_to_msb()
        .copied()
        .collect::<Vec<AigOperand>>();
    let mut output_bits = Vec::with_capacity(gadget_count);
    let mut gadgets = Vec::with_capacity(gadget_count);

    for output_bit_index in 0..gadget_count {
        let choose_input = |rng: &mut Pcg64Mcg| {
            let index = rng.gen_range(0..input_operands.len());
            maybe_negate(rng, input_operands[index])
        };
        let a = choose_input(&mut rng);
        let b = choose_input(&mut rng);
        let c = choose_input(&mut rng);
        let d = choose_input(&mut rng);
        let e = choose_input(&mut rng);
        let f = choose_input(&mut rng);
        let g = choose_input(&mut rng);

        let ab = builder.add_and_binary(a, b);
        let cd = builder.add_and_binary(c, d);
        let root = builder.add_and_binary(ab, cd);
        let root_duplicate = builder.add_and_binary(cd, ab);
        let _user = builder.add_and_binary(root, e);
        let _user_duplicate = builder.add_and_binary(root_duplicate, e);
        let move_node = builder.add_and_binary(a, f);
        let _move_target_duplicate = builder.add_and_binary(a, g);
        let output_old = builder.add_and_binary(b, f);
        let output_new = builder.add_and_binary(b, g);

        output_bits.push(output_old);
        gadgets.push(EditGadget {
            a,
            b,
            c,
            d,
            e,
            g,
            ab,
            cd,
            root,
            root_duplicate,
            move_node,
            output_new,
            output_bit_index,
        });
    }

    builder.add_output(
        "out".to_string(),
        AigBitVector::from_lsb_is_index_0(&output_bits),
    );

    CutdbLikeWorkload {
        gate_fn: builder.build(),
        gadgets,
    }
}

fn lookup_queries(state: &DynamicStructuralHash, queries: &[(AigOperand, AigOperand)]) -> usize {
    queries
        .iter()
        .filter(|(lhs, rhs)| state.lookup_and(*lhs, *rhs).is_some())
        .count()
}

fn fanout_queries(state: &DynamicStructuralHash, queries: &[AigRef]) -> usize {
    queries
        .iter()
        .map(|node| state.fanout_count(*node))
        .sum::<usize>()
}

fn run_cutdb_like_trace(state: &mut DynamicStructuralHash, gadgets: &[EditGadget]) {
    for gadget in gadgets {
        black_box(state.lookup_and(gadget.b, gadget.a));
        black_box(state.lookup_and(gadget.ab, gadget.cd));
        black_box(state.add_and(gadget.d, gadget.c).unwrap());
        black_box(state.add_and(gadget.ab.negate(), gadget.e).unwrap());
        black_box(state.add_and(gadget.c.negate(), gadget.g).unwrap());
        state
            .move_fanin_edge(gadget.move_node.node, 1, gadget.g)
            .unwrap();
        state
            .move_output_edge(0, gadget.output_bit_index, gadget.output_new)
            .unwrap();
        state
            .replace_node_with_operand(gadget.root.node, gadget.root_duplicate)
            .unwrap();
    }
    black_box(state.live_and_count());
}

fn dynamic_structural_hash_large_random_graph_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_structural_hash_large_random_graph");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for (input_count, and_count) in [(256usize, 10_000usize), (512, 100_000)] {
        let workload = build_random_graph_workload(input_count, and_count);
        group.bench_with_input(
            BenchmarkId::new("build_index", and_count),
            &workload,
            |b, w| {
                b.iter_batched(
                    || w.gate_fn.clone(),
                    |gate_fn| {
                        let state = DynamicStructuralHash::new(black_box(gate_fn)).unwrap();
                        black_box(state.live_and_count());
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        let state = DynamicStructuralHash::new(workload.gate_fn.clone()).unwrap();
        group.bench_with_input(
            BenchmarkId::new("lookup_hits", and_count),
            &workload,
            |b, w| {
                b.iter(|| black_box(lookup_queries(&state, &w.hit_queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lookup_misses", and_count),
            &workload,
            |b, w| {
                b.iter(|| black_box(lookup_queries(&state, &w.miss_queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fanout_count", and_count),
            &workload,
            |b, w| {
                b.iter(|| black_box(fanout_queries(&state, &w.fanout_queries)));
            },
        );
    }

    group.finish();
}

fn dynamic_structural_hash_cutdb_like_edit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_structural_hash_cutdb_like_edits");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for gadget_count in [1_000usize, 10_000] {
        let workload = build_cutdb_like_workload(gadget_count);
        group.bench_with_input(
            BenchmarkId::new("edit_trace", gadget_count),
            &workload,
            |b, w| {
                b.iter_batched(
                    || DynamicStructuralHash::new(w.gate_fn.clone()).unwrap(),
                    |mut state| run_cutdb_like_trace(&mut state, &w.gadgets),
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    dynamic_structural_hash_large_random_graph_benchmark,
    dynamic_structural_hash_cutdb_like_edit_benchmark
);
criterion_main!(benches);
