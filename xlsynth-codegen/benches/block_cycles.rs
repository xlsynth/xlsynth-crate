// SPDX-License-Identifier: Apache-2.0

//! Benchmarks strict validation, structural refinement, and explicit
//! simplification and whole-node legalization.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::BlockBuilder;
use xlsynth_pir::block_inline::flatten_block_hierarchy;
use xlsynth_pir::block_legalize::legalize_block_hierarchy;
use xlsynth_pir::block_simplify::simplify_block_package;
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir::{Package, Type};
use xlsynth_pir::ir_builder::{verify_block_combinational_cycles, verify_block_wiring_cycles};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::verify_package;

/// Creates many instances of the same wide, combinational leaf.
fn acyclic_hierarchy(width: usize, count: usize) -> Package {
    let mut leaf = BlockBuilder::new("leaf");
    let data = leaf.input_port("data", Type::Bits(width)).unwrap();
    leaf.output_port("result", data).unwrap();
    let mut package = leaf.build_package("benchmark").unwrap();
    let mut top = BlockBuilder::new("root");
    let data = top.input_port("data", Type::Bits(width)).unwrap();
    let mut result = data;
    for i in 0..count {
        let instance = top
            .instantiate(&format!("u_{i}"), package.get_block("leaf").unwrap())
            .unwrap();
        top.instantiation_input(instance, "data", result).unwrap();
        result = top.instantiation_output(instance, "result").unwrap();
    }
    top.output_port("result", result).unwrap();
    top.build_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();
    package
}

/// Creates a whole-node cycle whose individual wires form an acyclic shift.
fn shift_feedback() -> Package {
    Parser::new(
        r#"package feedback
block leaf(d: bits[64], q: bits[64]) {
  d: bits[64] = input_port(name=d, id=1)
  one: bits[1] = literal(value=1, id=2)
  shifted: bits[64] = shll(d, one, id=3)
  q: () = output_port(shifted, name=q, id=4)
}
top block root(result: bits[64]) {
  instantiation u(block=leaf, kind=block)
  q: bits[64] = instantiation_output(instantiation=u, port_name=q, id=5)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=6)
  result: () = output_port(q, name=result, id=7)
}"#,
    )
    .parse_and_verify_package()
    .unwrap()
}

/// Creates a hierarchy with sliced resolved-net feedback at the deepest leaf.
fn inout_hierarchy(depth: usize) -> Package {
    assert!(depth > 1);
    let mut source = String::from(
        "package chain\nblock layer0(seed: bits[1], result: bits[2]) {\n  ext_inout_net bus(bits[2])\n  seed: bits[1] = input_port(name=seed, id=1)\n  one: bits[1] = literal(value=1, id=2)\n  sample: bits[2] = ext_inout_read(net=bus, id=3)\n  lower: bits[1] = bit_slice(sample, start=0, width=1, id=4)\n  data: bits[2] = concat(lower, seed, id=5)\n  drive: () = ext_inout_drive(data, one, net=bus, id=6)\n  result: () = output_port(sample, name=result, id=7)\n}\n",
    );
    let mut id = 8;
    for layer in 1..depth {
        let top = if layer + 1 == depth { "top " } else { "" };
        source.push_str(&format!(
            "{top}block layer{layer}(seed: bits[1], result: bits[2]) {{\n  instantiation u(block=layer{}, kind=block)\n  seed: bits[1] = input_port(name=seed, id={id})\n  q: bits[2] = instantiation_output(instantiation=u, port_name=result, id={})\n  bind: () = instantiation_input(seed, instantiation=u, port_name=seed, id={})\n  result: () = output_port(q, name=result, id={})\n}}\n",
            layer - 1,
            id + 1,
            id + 2,
            id + 3,
        ));
        id += 4;
    }
    Parser::new(&source).parse_and_verify_package().unwrap()
}

fn block_cycles(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_cycles");
    for width in [8, 4096] {
        let package = acyclic_hierarchy(width, 128);
        let top = package.get_top_block().unwrap();
        group.bench_with_input(BenchmarkId::new("strict", width), &width, |b, _| {
            b.iter(|| {
                verify_block_combinational_cycles(black_box(top), black_box(&package)).unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("wiring", width), &width, |b, _| {
            b.iter(|| verify_block_wiring_cycles(black_box(top), black_box(&package)).unwrap())
        });
        group.bench_with_input(
            BenchmarkId::new("flatten_acyclic", width),
            &width,
            |b, _| b.iter(|| flatten_block_hierarchy(black_box(&package), "root").unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("block2fn_acyclic", width),
            &width,
            |b, _| {
                b.iter(|| {
                    block_package_to_fn(black_box(&package), &Block2FnOptions::default()).unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("legalize_acyclic", width),
            &width,
            |b, _| b.iter(|| legalize_block_hierarchy(black_box(&package), "root").unwrap()),
        );
    }
    for depth in [8, 64] {
        let package = inout_hierarchy(depth);
        group.bench_with_input(
            BenchmarkId::new("verify_inout_chain", depth),
            &depth,
            |b, _| b.iter(|| verify_package(black_box(&package)).unwrap()),
        );
    }
    let original = shift_feedback();
    let simplified = simplify_block_package(&original).unwrap();
    group.bench_function("simplify_shift", |b| {
        b.iter(|| simplify_block_package(black_box(&original)).unwrap())
    });
    group.bench_function("refine_feedback", |b| {
        b.iter(|| {
            verify_block_wiring_cycles(
                black_box(simplified.get_top_block().unwrap()),
                black_box(&simplified),
            )
            .unwrap()
        })
    });
    group.bench_function("block2fn_feedback", |b| {
        b.iter(|| block_package_to_fn(black_box(&simplified), &Block2FnOptions::default()).unwrap())
    });
    group.bench_function("legalize_feedback", |b| {
        b.iter(|| legalize_block_hierarchy(black_box(&simplified), "root").unwrap())
    });
    group.bench_function("emit_feedback", |b| {
        b.iter(|| {
            emit_system_verilog(black_box(&simplified), &BlockCodegenOptions::default()).unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, block_cycles);
criterion_main!(benches);
