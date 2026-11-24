// SPDX-License-Identifier: Apache-2.0

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use xlsynth_pir::BlockBuilder;
use xlsynth_pir::ir::{FileTable, Package, Type};
use xlsynth_pir::ir_builder::FunctionBatch;

/// Measures insertion of distinct definitions independently of builder setup.
fn bench_distinct_blocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_distinct_blocks");
    for count in [128, 256, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter_batched(
                || (0..count).map(distinct_block).collect::<Vec<_>>(),
                |builders| {
                    let mut package = empty_package();
                    let mut batch = FunctionBatch::new(&mut package).unwrap();
                    for builder in builders {
                        batch.append_block(builder).unwrap();
                    }
                    black_box(package)
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Exercises dependency summaries retained across unrelated child-parent pairs.
fn bench_disjoint_hierarchy(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_disjoint_hierarchy");
    for (pairs, ports) in [(8, 128), (32, 256)] {
        group.bench_with_input(
            BenchmarkId::new("pairs_and_ports", format!("{pairs}x{ports}")),
            &(pairs, ports),
            |b, &(pairs, ports)| {
                b.iter(|| {
                    let mut package = empty_package();
                    let mut batch = FunctionBatch::new(&mut package).unwrap();
                    for pair in 0..pairs {
                        let name = format!("leaf_{pair}");
                        batch.append_block(dense_leaf(&name, ports)).unwrap();
                        let mut parent = BlockBuilder::new(&format!("parent_{pair}"));
                        let input = parent.input_port("x", Type::Bits(1)).unwrap();
                        let instance = parent
                            .instantiate("child", batch.package().get_block(&name).unwrap())
                            .unwrap();
                        for port in 0..ports {
                            parent
                                .instantiation_input(instance, &format!("input_{port}"), input)
                                .unwrap();
                            parent
                                .instantiation_output(instance, &format!("output_{port}"))
                                .unwrap();
                        }
                        batch.append_block(parent).unwrap();
                    }
                    black_box(package)
                });
            },
        );
    }
    group.finish();
}

/// Creates one small combination of slices for the distinct-definition sweep.
fn distinct_block(index: usize) -> BlockBuilder {
    let mut builder = BlockBuilder::new(&format!("lane_{index}"));
    let input = builder.input_port("a", Type::Bits(32)).unwrap();
    let lo = builder.bit_slice(input, 0, 16).unwrap();
    let hi = builder.bit_slice(input, 16, 16).unwrap();
    let swapped = builder.concat(&[lo, hi]).unwrap();
    builder.output_port("y", swapped).unwrap();
    builder
}

/// Creates a child whose every output depends on every input.
fn dense_leaf(name: &str, ports: usize) -> BlockBuilder {
    let mut builder = BlockBuilder::new(name);
    let mut sum = builder.input_port("input_0", Type::Bits(1)).unwrap();
    for port in 1..ports {
        let input = builder
            .input_port(&format!("input_{port}"), Type::Bits(1))
            .unwrap();
        sum = builder.add(sum, input).unwrap();
    }
    for port in 0..ports {
        builder.output_port(&format!("output_{port}"), sum).unwrap();
    }
    builder
}

/// Creates the verified empty destination for each measured batch.
fn empty_package() -> Package {
    Package {
        name: "batch_scaling".into(),
        file_table: FileTable::new(),
        members: Vec::new(),
        top: None,
    }
}

criterion_group!(
    block_batch_scaling,
    bench_distinct_blocks,
    bench_disjoint_hierarchy
);
criterion_main!(block_batch_scaling);
