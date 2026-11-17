// SPDX-License-Identifier: Apache-2.0

//! Independent exhaustive oracle for three-lane hierarchical wiring.

use std::fmt::Write;
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_builder::verify_block_combinational_cycles;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{BuilderError, IrValue};

/// Enumerates three choices from the three feedback lanes or an independent
/// input.
fn lanes(n: usize) -> [usize; 3] {
    std::array::from_fn(|i| (n >> (2 * i)) & 3)
}

/// Solves each output lane by following its independent scalar dependency
/// graph. `None` means a real loop; otherwise each lane names a seed or
/// independent bit and its inversion.
fn solve(
    input: [usize; 3],
    output: [usize; 3],
    invert_lanes: u64,
) -> Option<[(bool, usize, u64); 3]> {
    let mut result = [(false, 0, 0); 3];
    for start in 0..3 {
        let mut seen = [false; 3];
        let mut k = start;
        let mut invert = 0;
        loop {
            if std::mem::replace(&mut seen[k], true) {
                return None;
            }
            invert ^= (invert_lanes >> k) & 1;
            let d = output[k];
            if d == 3 {
                result[start] = (true, k, invert);
                break;
            }
            if input[d] == 3 {
                result[start] = (false, d, invert);
                break;
            }
            k = input[d];
        }
    }
    Some(result)
}

#[derive(Clone, Copy)]
enum Wiring {
    SliceConcat,
    ReverseSliceUpdate,
}

/// Builds the vector IR independently from the scalar oracle.
fn design(input: [usize; 3], output: [usize; 3], wiring: Wiring) -> String {
    let mut ir = String::from(
        "package lanes\nblock leaf(d: bits[3], cs: bits[3], q: bits[3]) {\nd: bits[3] = input_port(name=d, id=1)\ncs: bits[3] = input_port(name=cs, id=2)\n",
    );
    if let Wiring::ReverseSliceUpdate = wiring {
        ir.push_str("rd: bits[3] = reverse(d, id=20)\nrcs: bits[3] = reverse(cs, id=21)\n");
        for k in 0..3 {
            writeln!(ir, "i{k}: bits[2] = literal(value={k}, id={})", 22 + k).unwrap();
        }
    }
    for (k, &v) in output.iter().enumerate() {
        let (source, offset) = if v == 3 { ("cs", k) } else { ("d", v) };
        match wiring {
            Wiring::SliceConcat => writeln!(
                ir,
                "b{k}: bits[1] = bit_slice({source}, start={offset}, width=1, id={})",
                3 + k
            ),
            Wiring::ReverseSliceUpdate => writeln!(
                ir,
                "b{k}: bits[1] = dynamic_bit_slice(r{source}, i{}, width=1, id={})",
                2 - offset,
                3 + k
            ),
        }
        .unwrap();
    }
    match wiring {
        Wiring::SliceConcat => ir.push_str("packed_lanes: bits[3] = concat(b2, b1, b0, id=6)\n"),
        Wiring::ReverseSliceUpdate => ir.push_str("u0: bits[3] = bit_slice_update(d, i0, b0, id=25)\nu1: bits[3] = bit_slice_update(u0, i1, b1, id=26)\npacked_lanes: bits[3] = bit_slice_update(u1, i2, b2, id=6)\n"),
    }
    ir.push_str("mask: bits[3] = literal(value=5, id=7)\nflipped: bits[3] = xor(packed_lanes, mask, id=8)\nq: () = output_port(flipped, name=q, id=9)\n}\ntop block wrapper(seed: bits[3], cs: bits[3], result: bits[3]) {\ninstantiation u(block=leaf, kind=block)\nseed: bits[3] = input_port(name=seed, id=10)\ncs: bits[3] = input_port(name=cs, id=11)\noutp: bits[3] = instantiation_output(instantiation=u, port_name=q, id=12)\n");
    for (k, &v) in input.iter().enumerate() {
        let (source, offset) = if v == 3 { ("seed", k) } else { ("outp", v) };
        writeln!(
            ir,
            "p{k}: bits[1] = bit_slice({source}, start={offset}, width=1, id={})",
            13 + k
        )
        .unwrap();
    }
    ir.push_str("packed_data: bits[3] = concat(p2, p1, p0, id=16)\ndrive: () = instantiation_input(packed_data, instantiation=u, port_name=d, id=17)\nconstants: () = instantiation_input(cs, instantiation=u, port_name=cs, id=18)\nresult: () = output_port(outp, name=result, id=19)\n}\n");
    ir
}

/// Generates the same scalar wiring with array-valued ports and single-element
/// slices. An output-to-input feedback path stays entirely in array operations.
fn array_design(input: [usize; 3], output: [usize; 3]) -> String {
    let mut ir = String::from(
        r#"package arrays
block leaf(d: bits[1][3], cs: bits[1][3], q: bits[1][3]) {
  d: bits[1][3] = input_port(name=d, id=1)
  cs: bits[1][3] = input_port(name=cs, id=2)
"#,
    );
    for (lane, &source) in output.iter().enumerate() {
        let (array, offset) = if source == 3 {
            ("cs", lane)
        } else {
            ("d", source)
        };
        writeln!(
            ir,
            "  i{lane}: bits[2] = literal(value={offset}, id={})",
            3 + lane
        )
        .unwrap();
        writeln!(
            ir,
            "  b{lane}: bits[1][1] = array_slice({array}, i{lane}, width=1, id={})",
            6 + lane
        )
        .unwrap();
    }
    ir.push_str(
        r#"  bus: bits[1][3] = array_concat(b0, b1, b2, id=9)
  q: () = output_port(bus, name=q, id=10)
}
top block wrapper(seed: bits[3], cs: bits[3], result: bits[3]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[3] = input_port(name=seed, id=11)
  cs: bits[3] = input_port(name=cs, id=12)
  q: bits[1][3] = instantiation_output(instantiation=u, port_name=q, id=13)
"#,
    );
    for lane in 0..3 {
        writeln!(
            ir,
            "  s{lane}: bits[1] = bit_slice(seed, start={lane}, width=1, id={})",
            14 + lane
        )
        .unwrap();
        writeln!(
            ir,
            "  c{lane}: bits[1] = bit_slice(cs, start={lane}, width=1, id={})",
            17 + lane
        )
        .unwrap();
        writeln!(
            ir,
            "  i{lane}: bits[2] = literal(value={lane}, id={})",
            20 + lane
        )
        .unwrap();
        writeln!(
            ir,
            "  y{lane}: bits[1] = array_index(q, indices=[i{lane}], id={})",
            23 + lane
        )
        .unwrap();
    }
    ir.push_str("  seeds: bits[1][3] = array(s0, s1, s2, id=26)\n  constants: bits[1][3] = array(c0, c1, c2, id=27)\n");
    for (lane, &source) in input.iter().enumerate() {
        let (array, offset) = if source == 3 {
            ("seeds", lane)
        } else {
            ("q", source)
        };
        writeln!(
            ir,
            "  b{lane}: bits[1][1] = array_slice({array}, i{offset}, width=1, id={})",
            28 + lane
        )
        .unwrap();
    }
    ir.push_str(
        r#"  data: bits[1][3] = array_concat(b0, b1, b2, id=31)
  drive: () = instantiation_input(data, instantiation=u, port_name=d, id=32)
  cs_in: () = instantiation_input(constants, instantiation=u, port_name=cs, id=33)
  packed_result: bits[3] = concat(y2, y1, y0, id=34)
  result: () = output_port(packed_result, name=result, id=35)
}
"#,
    );
    ir
}

#[test]
fn exhaustive_array_feedback_wiring() {
    check_wirings(array_design, 0);
}

#[test]
fn exhaustive_slice_concat_wiring() {
    check_wirings(|i, o| design(i, o, Wiring::SliceConcat), 5);
}

#[test]
fn exhaustive_reverse_slice_update_wiring() {
    check_wirings(|i, o| design(i, o, Wiring::ReverseSliceUpdate), 5);
}

/// Checks every three-lane wiring against the same independent scalar model.
fn check_wirings(wiring: impl Fn([usize; 3], [usize; 3]) -> String, invert_lanes: u64) {
    for i in 0..64 {
        for j in 0..64 {
            let (input, output) = (lanes(i), lanes(j));
            let ir = wiring(input, output);
            let package = Parser::new(&ir).parse_and_verify_package().unwrap();
            let check =
                verify_block_combinational_cycles(package.get_top_block().unwrap(), &package);
            let projected = block_package_to_fn(&package, &Block2FnOptions::default());
            let Some(expected) = solve(input, output, invert_lanes) else {
                assert!(
                    matches!(check, Err(BuilderError::CombinationalCycle { .. })),
                    "{ir}"
                );
                assert!(projected.unwrap_err().contains("cycle"), "{ir}");
                continue;
            };
            check.unwrap_or_else(|e| panic!("{e}\n{ir}"));
            let function = projected.unwrap_or_else(|e| panic!("{e}\n{ir}")).function;
            xlsynth_pir::ir_verify::verify_function(&function).unwrap();
            for seed in 0..8 {
                for cs in 0..8 {
                    let answer: u64 = expected
                        .iter()
                        .enumerate()
                        .map(|(k, &(child, offset, invert))| {
                            ((((if child { cs } else { seed }) >> offset) & 1) ^ invert) << k
                        })
                        .sum();
                    let args = [
                        IrValue::make_ubits(3, seed).unwrap(),
                        IrValue::make_ubits(3, cs).unwrap(),
                    ];
                    let FnEvalResult::Success(actual) =
                        eval_fn_in_package(&package, &function, &args)
                    else {
                        panic!("evaluation failed\n{ir}");
                    };
                    assert_eq!(
                        actual.value,
                        IrValue::make_ubits(3, answer).unwrap(),
                        "{ir}"
                    );
                }
            }
        }
    }
}

#[test]
fn downstream_call_stays_shared_as_its_body_grows() {
    let layouts = [
        "low: bits[1] = bit_slice(q, start=0, width=1, id=5)",
        // The call is on a whole-word cycle, but its output occupies only
        // bits [2:1] and the feedback reads bit 0.
        "packed: bits[3] = concat(consumer, seed, id=10)\n  low: bits[1] = bit_slice(packed, start=0, width=1, id=5)",
    ];
    for wiring in layouts {
        let mut previous_node_count = None;
        for depth in [1, 4, 8] {
            // Each level invokes its predecessor twice. Expanding the output
            // helper in the projected function would grow its body
            // exponentially.
            let mut source = String::from(
                "package downstream\nfn helper_0(x: bits[2] id=100) -> bits[2] {\n  ret y: bits[2] = not(x, id=101)\n}\n",
            );
            for level in 1..=depth {
                let param = 100 + level * 3;
                let callee = level - 1;
                writeln!(
                source,
                "fn helper_{level}(x: bits[2] id={param}) -> bits[2] {{\n  a: bits[2] = invoke(x, to_apply=helper_{callee}, id={})\n  ret b: bits[2] = invoke(a, to_apply=helper_{callee}, id={})\n}}",
                param + 1, param + 2
            )
            .unwrap();
            }
            let callee = format!("helper_{depth}");
            writeln!(
                source,
                r#"block leaf(d: bits[2], q: bits[2]) {{
  d: bits[2] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}}
top block wrapper(seed: bits[1], result: bits[2]) {{
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[2] = instantiation_output(instantiation=u, port_name=q, id=4)
  consumer: bits[2] = invoke(q, to_apply={callee}, id=8)
  {wiring}
  bus: bits[2] = concat(low, seed, id=6)
  drive: () = instantiation_input(bus, instantiation=u, port_name=d, id=7)
  result: () = output_port(consumer, name=result, id=9)
}}"#
            )
            .unwrap();

            let package = Parser::new(&source).parse_and_verify_package().unwrap();
            let function = block_package_to_fn(&package, &Block2FnOptions::default())
                .unwrap()
                .function;
            let calls = function
                .nodes
                .iter()
                .filter_map(|node| match &node.payload {
                    NodePayload::Invoke { to_apply, .. } => Some(to_apply.as_str()),
                    _ => None,
                });
            assert_eq!(calls.collect::<Vec<_>>(), [callee.as_str()]);
            if let Some(count) = previous_node_count.replace(function.nodes.len()) {
                assert_eq!(function.nodes.len(), count, "helper depth {depth}");
            }
            for seed in [0, 1] {
                let FnEvalResult::Success(actual) = eval_fn_in_package(
                    &package,
                    &function,
                    &[IrValue::make_ubits(1, seed).unwrap()],
                ) else {
                    panic!("evaluation failed at helper depth {depth}");
                };
                assert_eq!(actual.value, IrValue::make_ubits(2, 3 * seed).unwrap());
            }
        }
    }
}
