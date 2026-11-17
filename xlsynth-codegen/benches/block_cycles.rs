// SPDX-License-Identifier: Apache-2.0

//! Release benchmarks for whole-port validation, range refinement, and the
//! native emitters.

use criterion::{Criterion, criterion_group, criterion_main};
use std::fmt::Write;
use std::hint::black_box;
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir::{Package, Type};
use xlsynth_pir::ir_builder::verify_block_combinational_cycles;
use xlsynth_pir::{BlockBuilder, FnBuilder};

/// Each output has its own driver and shares the entire upstream chain.
fn shared_cone(outputs: usize, unused: usize) -> Package {
    let mut leaf = BlockBuilder::new("leaf");
    let mut value = leaf.input_port("data", Type::Bits(1)).unwrap();
    for i in 0..unused {
        leaf.input_port(&format!("unused_{i}"), Type::Bits(1))
            .unwrap();
    }
    for _ in 0..4000 {
        value = leaf.not(value).unwrap();
    }
    for index in 0..outputs {
        let last = leaf.not(value).unwrap();
        leaf.output_port(&format!("q_{index}"), last).unwrap();
    }
    observe_leaf(leaf)
}

enum ReductionOutputs {
    DirectTaps(usize),
    Dense,
}

/// All inputs feed a common reduction. Direct taps have short independent
/// paths; dense outputs each observe the entire chain and a distinct input.
fn shared_reduction(inputs: usize, depth: usize, outputs: ReductionOutputs) -> Package {
    let mut leaf = BlockBuilder::new("leaf");
    let mut ports = Vec::new();
    for i in 0..inputs {
        ports.push(
            leaf.input_port(&format!("data_{i}"), Type::Bits(1))
                .unwrap(),
        );
    }
    let word = leaf.concat(&ports).unwrap();
    let mut value = leaf.xor_reduce(word).unwrap();
    for _ in 0..depth {
        value = leaf.not(value).unwrap();
    }
    leaf.output_port("reduced", value).unwrap();
    match outputs {
        ReductionOutputs::DirectTaps(count) => {
            for (i, &input) in ports.iter().take(count).enumerate() {
                leaf.output_port(&format!("direct_{i}"), input).unwrap();
            }
        }
        ReductionOutputs::Dense => {
            for (i, &input) in ports.iter().enumerate() {
                let q = leaf.xor(value, input).unwrap();
                leaf.output_port(&format!("q_{i}"), q).unwrap();
            }
        }
    }
    observe_leaf(leaf)
}

/// Instantiates a one-bit-port leaf and makes every output observable.
fn observe_leaf(leaf: BlockBuilder) -> Package {
    let mut package = leaf.build_package("cones").unwrap();
    let leaf = package.get_block("leaf").unwrap();
    let mut parent = BlockBuilder::new("wrapper");
    let seed = parent.input_port("seed", Type::Bits(1)).unwrap();
    let inst = parent.instantiate("u", leaf).unwrap();
    for port in leaf.input_ports() {
        parent
            .instantiation_input(inst, leaf.port_name(port), seed)
            .unwrap();
    }
    let ports = leaf
        .output_ports()
        .map(|port| {
            parent
                .instantiation_output(inst, leaf.port_name(port))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let observed = parent.concat(&ports).unwrap();
    parent.output_port("result", observed).unwrap();
    parent.build_into_package(&mut package).unwrap();
    package.set_top_block("wrapper").unwrap();
    package
}

/// Feeds the low output bits back above an independent seed. Without feedback
/// both halves of the input come directly from the seed.
fn lanes(width: usize, feedback_bits: Option<usize>, helper_depth: usize) -> Package {
    let seed_width = feedback_bits.map_or(width / 2, |bits| width - bits);
    let mut leaf = BlockBuilder::new("leaf");
    let data = leaf.input_port("data", Type::Bits(width)).unwrap();
    leaf.output_port("q", data).unwrap();
    let mut package = leaf.build_package("lanes").unwrap();
    let mut parent = BlockBuilder::new("wrapper");
    let seed = parent.input_port("seed", Type::Bits(seed_width)).unwrap();
    let inst = parent
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = parent.instantiation_output(inst, "q").unwrap();
    let upper = if let Some(bits) = feedback_bits {
        parent.bit_slice(q, 0, bits).unwrap()
    } else {
        seed
    };
    let data = parent.concat(&[upper, seed]).unwrap();
    parent.instantiation_input(inst, "data", data).unwrap();
    let mut helper_name = None;
    for level in 0..helper_depth {
        let name = format!("helper_{level}");
        let mut helper = FnBuilder::new(&name);
        let arg = helper.param("x", Type::Bits(width)).unwrap();
        let value = match helper_name.as_deref() {
            None => helper.not(arg).unwrap(),
            Some(previous) => {
                let previous = package.get_fn(previous).unwrap();
                let first = helper.invoke(previous, &[arg]).unwrap();
                helper.invoke(previous, &[first]).unwrap()
            }
        };
        helper.build_into_package(value, &mut package).unwrap();
        helper_name = Some(name);
    }
    let result = match helper_name {
        Some(name) => parent.invoke(package.get_fn(&name).unwrap(), &[q]).unwrap(),
        None => q,
    };
    parent.output_port("result", result).unwrap();
    parent.build_into_package(&mut package).unwrap();
    package.set_top_block("wrapper").unwrap();
    package
}

/// Measures repeated blocked calls with many unrelated package members.
fn blocked_invokes(count: usize, unrelated: usize) -> Package {
    let mut function = xlsynth_pir::FnBuilder::new("pass");
    let arg = function.param("x", Type::Bits(2)).unwrap();
    let mut package = function.build_package(arg, "calls").unwrap();
    for i in 0..unrelated {
        let mut f = xlsynth_pir::FnBuilder::new(&format!("unrelated_{i}"));
        let arg = f.param("x", Type::Bits(2)).unwrap();
        f.build_into_package(arg, &mut package).unwrap();
    }
    let mut child = BlockBuilder::new("leaf");
    let d = child.input_port("d", Type::Bits(2)).unwrap();
    let q = child.invoke(package.get_fn("pass").unwrap(), &[d]).unwrap();
    child.output_port("q", q).unwrap();
    child.build_into_package(&mut package).unwrap();
    let mut top = BlockBuilder::new("wrapper");
    let seed = top.input_port("seed", Type::Bits(1)).unwrap();
    let mut outputs = Vec::new();
    for i in 0..count {
        let instance = top
            .instantiate(&format!("u_{i}"), package.get_block("leaf").unwrap())
            .unwrap();
        let q = top.instantiation_output(instance, "q").unwrap();
        let lo = top.bit_slice(q, 0, 1).unwrap();
        let hi = top.bit_slice(q, 1, 1).unwrap();
        let d = top.concat(&[lo, seed]).unwrap();
        top.instantiation_input(instance, "d", d).unwrap();
        outputs.push(hi);
    }
    let output = top.concat(&outputs).unwrap();
    top.output_port("result", output).unwrap();
    top.build_into_package(&mut package).unwrap();
    package.set_top_block("wrapper").unwrap();
    package
}

/// A chain of forwarding functions sits inside a three-bit feedback bus.
/// Resolving the bus needs the innermost function's implementation.
fn forwarding_calls(depth: usize) -> Package {
    let mut ir = String::from("package forwarding\n");
    for level in 0..depth {
        let param = 10 + 2 * level;
        let result = param + 1;
        let op = if level == 0 {
            format!("identity(x, id={result})")
        } else {
            format!("invoke(x, to_apply=f{}, id={result})", level - 1)
        };
        writeln!(
            ir,
            "fn f{level}(x: bits[3] id={param}) -> bits[3] {{ ret q: bits[3] = {op} }}"
        )
        .unwrap();
    }
    writeln!(
        ir,
        r#"block leaf(d: bits[3], q: bits[3]) {{
  d: bits[3] = input_port(name=d, id=1)
  forwarded: bits[3] = invoke(d, to_apply=f{}, id=2)
  q: () = output_port(forwarded, name=q, id=3)
}}
top block wrapper(seed: bits[1], q: bits[3]) {{
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=4)
  lanes: bits[3] = instantiation_output(instantiation=u, port_name=q, id=5)
  low: bits[2] = bit_slice(lanes, start=0, width=2, id=6)
  bus: bits[3] = concat(low, seed, id=7)
  drive: () = instantiation_input(bus, instantiation=u, port_name=d, id=8)
  q: () = output_port(lanes, name=q, id=9)
}}"#,
        depth - 1
    )
    .unwrap();
    xlsynth_pir::ir_parser::Parser::new(&ir)
        .parse_and_verify_package()
        .unwrap()
}

/// Each shifter has an independent formal control. Only the enclosing block
/// ties those ports to one, making every two-bit feedback chain acyclic.
fn independent_controls(count: usize) -> Package {
    let mut ir = String::from(
        r#"package controls
block shift(x: bits[2], amount: bits[1], y: bits[2]) {
  x: bits[2] = input_port(name=x, id=1)
  amount: bits[1] = input_port(name=amount, id=2)
  shifted: bits[2] = shll(x, amount, id=3)
  y: () = output_port(shifted, name=y, id=4)
}
"#,
    );
    let ports = (0..count)
        .map(|i| format!("c{i}: bits[1]"))
        .collect::<Vec<_>>();
    writeln!(ir, "block bank({}) {{", ports.join(", ")).unwrap();
    for i in 0..count {
        writeln!(ir, "  instantiation u{i}(block=shift, kind=block)").unwrap();
    }
    for i in 0..count {
        let id = 5 + 4 * i;
        writeln!(ir, "  c{i}: bits[1] = input_port(name=c{i}, id={id})").unwrap();
        writeln!(
            ir,
            "  q{i}: bits[2] = instantiation_output(instantiation=u{i}, port_name=y, id={})",
            id + 1
        )
        .unwrap();
        writeln!(
            ir,
            "  drive{i}: () = instantiation_input(q{i}, instantiation=u{i}, port_name=x, id={})",
            id + 2
        )
        .unwrap();
        writeln!(ir, "  control{i}: () = instantiation_input(c{i}, instantiation=u{i}, port_name=amount, id={})", id + 3).unwrap();
    }
    ir.push_str("}\ntop block wrapper() {\n  instantiation b(block=bank, kind=block)\n");
    let one = 5 + 4 * count;
    writeln!(ir, "  one: bits[1] = literal(value=1, id={one})").unwrap();
    for i in 0..count {
        writeln!(
            ir,
            "  bind{i}: () = instantiation_input(one, instantiation=b, port_name=c{i}, id={})",
            one + i + 1
        )
        .unwrap();
    }
    ir.push_str("}\n");
    xlsynth_pir::ir_parser::Parser::new(&ir)
        .parse_and_verify_package()
        .unwrap()
}

fn block_cycles(c: &mut Criterion) {
    for count in [32, 64, 128, 256] {
        let package = independent_controls(count);
        let top = package.get_top_block().unwrap();
        c.bench_function(&format!("block_cycles/independent_controls/{count}"), |b| {
            b.iter(|| verify_block_combinational_cycles(black_box(top), &package).unwrap());
        });
    }

    for (count, unused) in [(1, 0), (512, 0), (512, 512)] {
        let package = shared_cone(count, unused);
        let top = package.get_top_block().unwrap();
        c.bench_function(
            &format!("block_cycles/shared_cone/{count}_outputs_{unused}_unused"),
            |b| {
                b.iter(|| verify_block_combinational_cycles(black_box(top), &package).unwrap());
            },
        );
        c.bench_function(
            &format!("block2sv/shared_cone/{count}_outputs_{unused}_unused"),
            |b| {
                b.iter(|| {
                    emit_system_verilog(black_box(&package), &BlockCodegenOptions::default())
                        .unwrap()
                });
            },
        );
    }
    for (name, inputs, depth, outputs) in [
        ("511_direct", 512, 4000, ReductionOutputs::DirectTaps(511)),
        ("512_direct", 512, 4000, ReductionOutputs::DirectTaps(512)),
        ("short_dense", 192, 0, ReductionOutputs::Dense),
        ("long_dense", 256, 4000, ReductionOutputs::Dense),
    ] {
        let package = shared_reduction(inputs, depth, outputs);
        let top = package.get_top_block().unwrap();
        c.bench_function(&format!("block_cycles/shared_reduction/{name}"), |b| {
            b.iter(|| verify_block_combinational_cycles(black_box(top), &package).unwrap());
        });
    }
    for depth in [500, 1000, 2000] {
        let package = forwarding_calls(depth);
        c.bench_function(&format!("block2fn/forwarding_calls/{depth}"), |b| {
            b.iter(|| {
                block_package_to_fn(black_box(&package), &Block2FnOptions::default()).unwrap()
            });
        });
    }
    for count in [1, 32] {
        let package = blocked_invokes(count, 1000);
        c.bench_function(&format!("block2fn/blocked_invokes/{count}"), |b| {
            b.iter(|| {
                block_package_to_fn(black_box(&package), &Block2FnOptions::default()).unwrap()
            });
        });
    }
    // Only the two-bit wiring participates in feedback. Each additional
    // helper level doubles the amount of work if a consumer is expanded.
    for depth in [1, 5, 11] {
        let package = lanes(2, Some(1), depth);
        c.bench_function(&format!("block2fn/downstream_calls/depth_{depth}"), |b| {
            b.iter(|| {
                block_package_to_fn(black_box(&package), &Block2FnOptions::default()).unwrap()
            });
        });
    }
    for width in [8, 4096] {
        for (name, bits) in [
            ("feed_forward", None),
            ("disjoint_halves", Some(width / 2)),
            ("shift_by_one", Some(width - 1)),
        ] {
            let package = lanes(width, bits, 0);
            let top = package.get_top_block().unwrap();
            c.bench_function(&format!("block_cycles/{name}/{width}"), |b| {
                b.iter(|| verify_block_combinational_cycles(black_box(top), &package).unwrap());
            });
            if bits.is_some() {
                c.bench_function(&format!("block2fn/{name}/{width}"), |b| {
                    b.iter(|| {
                        block_package_to_fn(black_box(&package), &Block2FnOptions::default())
                            .unwrap()
                    });
                });
            }
        }
    }
}

criterion_group!(benches, block_cycles);
criterion_main!(benches);
