// SPDX-License-Identifier: Apache-2.0

//! Checks static wiring feedback against independent expected values.

#[cfg(feature = "iverilog-tests")]
mod support;
#[cfg(feature = "iverilog-tests")]
use std::collections::BTreeMap;
#[cfg(feature = "iverilog-tests")]
use xlsynth_test_helpers::rtl_sim::LogicValue;

use xlsynth_codegen::{BlockCodegenOptions, Layout, emit_system_verilog};
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir_builder::verify_block_combinational_cycles;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{BuilderError, IrValue};

/// A child transforms a two-bit input; its parent feeds one selected output
/// bit into either input lane and observes the entire child result.
fn design(extra: &str, expression: &str, width: usize, lane: usize, feedback_high: bool) -> String {
    let data = if feedback_high {
        "feedback, seed"
    } else {
        "seed, feedback"
    };
    format!(
        r#"package wiring
block leaf(d: bits[2], q: bits[{width}]) {{
  d: bits[2] = input_port(name=d, id=1)
{extra}
  value: bits[{width}] = {expression}
  q: () = output_port(value, name=q, id=9)
}}
top block wrapper(seed: bits[1], result: bits[{width}]) {{
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=10)
  q: bits[{width}] = instantiation_output(instantiation=u, port_name=q, id=11)
  feedback: bits[1] = bit_slice(q, start={lane}, width=1, id=12)
  d: bits[2] = concat({data}, id=13)
  drive: () = instantiation_input(d, instantiation=u, port_name=d, id=14)
  result: () = output_port(q, name=result, id=15)
}}
"#
    )
}

/// Checks both SV layouts, PIR classification and projected function values.
fn check(source: &str, expected: Option<[u64; 2]>, width: usize) {
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    let check = verify_block_combinational_cycles(package.get_top_block().unwrap(), &package);
    let projected = block_package_to_fn(&package, &Block2FnOptions::default());
    if let Some(expected) = expected {
        check.unwrap_or_else(|error| panic!("{error}\n{source}"));
        let function = projected
            .unwrap_or_else(|error| panic!("{error}\n{source}"))
            .function;
        xlsynth_pir::ir_verify::verify_function_in_package(&function, &package).unwrap();
        for (seed, expected) in expected.into_iter().enumerate() {
            let FnEvalResult::Success(actual) = eval_fn_in_package(
                &package,
                &function,
                &[IrValue::make_ubits(1, seed as u64).unwrap()],
            ) else {
                panic!("evaluation failed");
            };
            assert_eq!(
                actual.value,
                IrValue::make_ubits(width, expected).unwrap(),
                "{source}"
            );
        }
        for layout in [Layout::None, Layout::Pipeline] {
            let options = BlockCodegenOptions {
                layout,
                ..Default::default()
            };
            emit_system_verilog(&package, &options).unwrap();
            #[cfg(feature = "iverilog-tests")]
            {
                let rtl = support::TestRtl::emit(&package, &options);
                for (seed, expected) in expected.into_iter().enumerate() {
                    let inputs =
                        BTreeMap::from([("seed".to_owned(), LogicValue::from_u64(1, seed as u64))]);
                    let outputs = rtl.evaluate(&inputs).unwrap();
                    assert_eq!(
                        outputs["result"],
                        LogicValue::from_u64(width as u32, expected),
                        "{source}"
                    );
                }
            }
        }
    } else {
        assert!(
            matches!(check, Err(BuilderError::CombinationalCycle { .. })),
            "{source}"
        );
        let projected_error = projected.unwrap_err();
        assert!(
            projected_error.contains("cycle"),
            "{projected_error}\n{source}"
        );
        let sv_error = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap_err();
        assert!(
            sv_error.to_string().contains("cycle"),
            "{sv_error}\n{source}"
        );
    }
}

#[test]
fn extensions_and_constant_shifts_preserve_feedback_dependencies() {
    // q[3] is padding, while q[1] depends on the high input lane.
    check(
        &design("", "zero_ext(d, new_bit_count=4, id=8)", 4, 3, true),
        Some([0, 1]),
        4,
    );
    check(
        &design("", "zero_ext(d, new_bit_count=4, id=8)", 4, 1, true),
        None,
        4,
    );
    // Sign bits depend on d[1], which is driven independently in the first
    // case.
    check(
        &design("", "sign_ext(d, new_bit_count=4, id=8)", 4, 3, false),
        Some([0, 15]),
        4,
    );
    check(
        &design("", "sign_ext(d, new_bit_count=4, id=8)", 4, 3, true),
        None,
        4,
    );
    let amount = "  amount: bits[128] = literal(value=1, id=2)";
    for (op, lane, high, expected) in [
        ("shll", 0, true, [0, 2]),
        ("shrl", 1, false, [0, 1]),
        ("shra", 1, false, [0, 3]),
    ] {
        check(
            &design(amount, &format!("{op}(d, amount, id=8)"), 2, lane, high),
            Some(expected),
            2,
        );
    }
    check(
        &design(amount, "shra(d, amount, id=8)", 2, 1, true),
        None,
        2,
    );
}

#[test]
fn aggregate_selections_ignore_unselected_elements() {
    let tuple = "  t: (bits[2], bits[2]) = tuple(d, d, id=2)";
    check(
        &design(tuple, "tuple_index(t, index=1, id=8)", 2, 0, true),
        Some([0, 3]),
        2,
    );
    check(
        &design(tuple, "tuple_index(t, index=1, id=8)", 2, 1, true),
        None,
        2,
    );
    let array = "  a: bits[2][2] = array(d, d, id=2)\n  index: bits[128] = literal(value=1, id=3)";
    check(
        &design(array, "array_index(a, indices=[index], id=8)", 2, 0, true),
        Some([0, 3]),
        2,
    );
    check(
        &design(array, "array_index(a, indices=[index], id=8)", 2, 1, true),
        None,
        2,
    );
}

#[test]
fn reading_a_port_sink_result_does_not_read_its_driver() {
    // Each sink observes q, but using the sink as a tuple field contributes
    // only its unit result. Feeding back q's data field still closes a loop.
    let source = r#"package sink_results
block pass(d: ((), bits[1]), observe: ((), bits[1]), q: ((), bits[1])) {
  d: ((), bits[1]) = input_port(name=d, id=1)
  observe: ((), bits[1]) = input_port(name=observe, id=2)
  q: () = output_port(d, name=q, id=3)
}
top block wrapper(seed: bits[1], result: bits[1]) {
  instantiation u(block=pass, kind=block)
  seed: bits[1] = input_port(name=seed, id=4)
  q: ((), bits[1]) = instantiation_output(instantiation=u, port_name=q, id=5)
  seen: () = instantiation_input(q, instantiation=u, port_name=observe, id=6)
  bit: bits[1] = tuple_index(q, index=1, id=7)
  result: () = output_port(bit, name=result, id=8)
  data_in: ((), bits[1]) = tuple(SINK, DATA, id=9)
  drive: () = instantiation_input(data_in, instantiation=u, port_name=d, id=10)
}
"#;
    for sink in ["seen", "result"] {
        let source = source.replace("SINK", sink);
        check(&source.replace("DATA", "seed"), Some([0, 1]), 1);
        check(&source.replace("DATA", "bit"), None, 1);
    }
}

#[test]
fn empty_data_feedback_contains_no_physical_cycle() {
    let source = r#"package empty
block leaf(d: bits[0], q: bits[0]) {
  d: bits[0] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block wrapper(result: bits[0]) {
  instantiation u(block=leaf, kind=block)
  q: bits[0] = instantiation_output(instantiation=u, port_name=q, id=3)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=4)
  result: () = output_port(q, name=result, id=5)
}
"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    verify_block_combinational_cycles(package.get_top_block().unwrap(), &package).unwrap();
    let projected = block_package_to_fn(&package, &Block2FnOptions::default()).unwrap();
    xlsynth_pir::ir_verify::verify_function(&projected.function).unwrap();
    let FnEvalResult::Success(result) = eval_fn_in_package(&package, &projected.function, &[])
    else {
        panic!("evaluation failed");
    };
    assert_eq!(result.value, IrValue::make_ubits(0, 0).unwrap());
    for layout in [Layout::None, Layout::Pipeline] {
        emit_system_verilog(
            &package,
            &BlockCodegenOptions {
                layout,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn selected_aggregate_can_ignore_direct_feedback() {
    for (extra, expression) in [
        (
            "  t: (bits[2], bits[2]) = tuple(d, seed, id=3)",
            "tuple_index(t, index=1, id=4)",
        ),
        (
            "  t: bits[2][2] = array(d, seed, id=3)\n  i: bits[1] = literal(value=1, id=16)",
            "array_index(t, indices=[i], id=4)",
        ),
    ] {
        let source = format!(
            r#"package aggregate
block leaf(d: bits[2], seed: bits[2], q: bits[2]) {{
  d: bits[2] = input_port(name=d, id=1)
  seed: bits[2] = input_port(name=seed, id=2)
{extra}
  v: bits[2] = {expression}
  q: () = output_port(v, name=q, id=5)
}}
top block wrapper(seed: bits[2], result: bits[2]) {{
  instantiation u(block=leaf, kind=block)
  seed: bits[2] = input_port(name=seed, id=6)
  q: bits[2] = instantiation_output(instantiation=u, port_name=q, id=7)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=8)
  input_seed: () = instantiation_input(seed, instantiation=u, port_name=seed, id=9)
  result: () = output_port(q, name=result, id=10)
}}
"#
        );
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        verify_block_combinational_cycles(package.get_top_block().unwrap(), &package).unwrap();
        let f = block_package_to_fn(&package, &Block2FnOptions::default())
            .unwrap()
            .function;
        for v in 0..4 {
            let input = IrValue::make_ubits(2, v).unwrap();
            let FnEvalResult::Success(actual) =
                eval_fn_in_package(&package, &f, std::slice::from_ref(&input))
            else {
                panic!("evaluation failed")
            };
            assert_eq!(actual.value, input);
        }
        for layout in [Layout::None, Layout::Pipeline] {
            emit_system_verilog(
                &package,
                &BlockCodegenOptions {
                    layout,
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
}

#[test]
fn extension_and_shift_projections_cover_crossing_ranges() {
    let source = r#"package crossing
block leaf(d: bits[2], q: bits[4]) {
  d: bits[2] = input_port(name=d, id=1)
  value: bits[4] = zero_ext(d, new_bit_count=4, id=2)
  q: () = output_port(value, name=q, id=3)
}
top block wrapper(result: bits[4]) {
  instantiation u(block=leaf, kind=block)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  slice: bits[2] = bit_slice(q, start=1, width=2, id=5)
  drive: () = instantiation_input(slice, instantiation=u, port_name=d, id=6)
  result: () = output_port(q, name=result, id=7)
}
"#;
    for op in ["zero_ext", "sign_ext"] {
        let source = source.replace("zero_ext", op);
        let p = Parser::new(&source).parse_and_verify_package().unwrap();
        let check = verify_block_combinational_cycles(p.get_top_block().unwrap(), &p);
        if op == "sign_ext" {
            assert!(matches!(
                check,
                Err(BuilderError::CombinationalCycle { .. })
            ));
            continue;
        }
        check.unwrap();
        let f = block_package_to_fn(&p, &Block2FnOptions::default())
            .unwrap()
            .function;
        let FnEvalResult::Success(actual) = eval_fn_in_package(&p, &f, &[]) else {
            panic!("evaluation failed")
        };
        assert_eq!(actual.value, IrValue::make_ubits(4, 0).unwrap());
    }
    // A left shift inserts a low zero while preserving the selected live input.
    let shifted = source.replace("value: bits[4] = zero_ext(d, new_bit_count=4, id=2)", "ext: bits[4] = zero_ext(d, new_bit_count=4, id=16)\n  amount: bits[128] = literal(value=1, id=17)\n  value: bits[4] = shll(ext, amount, id=2)").replace("start=1, width=2", "start=0, width=2");
    let p = Parser::new(&shifted).parse_and_verify_package().unwrap();
    verify_block_combinational_cycles(p.get_top_block().unwrap(), &p).unwrap();
    let f = block_package_to_fn(&p, &Block2FnOptions::default())
        .unwrap()
        .function;
    let FnEvalResult::Success(actual) = eval_fn_in_package(&p, &f, &[]) else {
        panic!("evaluation failed")
    };
    assert_eq!(actual.value, IrValue::make_ubits(4, 0).unwrap());
}

/// Connects the child's entire output back to its input, with a separate seed.
/// Any cut in the resulting cycle must come from the child's own operations.
fn whole_word_feedback(body: &str) -> String {
    format!(
        r#"package whole
block leaf(d: bits[3], seed: bits[1], q: bits[3]) {{
  d: bits[3] = input_port(name=d, id=1)
  seed: bits[1] = input_port(name=seed, id=2)
{body}
  q: () = output_port(v, name=q, id=20)
}}
top block wrapper(seed: bits[1], result: bits[3]) {{
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=21)
  q: bits[3] = instantiation_output(instantiation=u, port_name=q, id=22)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=23)
  source_seed: () = instantiation_input(seed, instantiation=u, port_name=seed, id=24)
  result: () = output_port(q, name=result, id=25)
}}
"#
    )
}

#[test]
fn whole_port_shifts_and_rotations() {
    let amounts = "one: bits[2] = literal(value=1, id=3)\ntwo: bits[2] = literal(value=2, id=4)\nzero: bits[1] = literal(value=0, id=9)";
    for (expr, expected) in [
        ("shll(d, one, id=8)", Some([0, 0])),
        ("shrl(d, one, id=8)", Some([0, 0])),
        ("shra(d, one, id=8)", None),
        ("shll(d, zero, id=8)", None),
        // With seed=0 the shift is an identity.
        ("shll(d, seed, id=8)", None),
        ("shrl(d, seed, id=8)", None),
        ("dynamic_bit_slice(d, seed, width=3, id=8)", None),
        ("bit_slice_update(d, seed, seed, id=8)", None),
    ] {
        check(
            &whole_word_feedback(&format!("{amounts}\nv: bits[3] = {expr}")),
            expected,
            3,
        );
    }
    for rotate in [
        "low: bits[2] = bit_slice(d, start=0, width=2, id=5)\nhigh: bits[1] = bit_slice(d, start=2, width=1, id=6)\nrot: bits[3] = concat(low, high, id=7)",
        "left: bits[3] = shll(d, one, id=5)\nright: bits[3] = shrl(d, two, id=6)\nrot: bits[3] = or(left, right, id=7)",
    ] {
        // A pure rotation feeds every lane back. Replacing the middle bit
        // with the seed breaks the loop and copies the seed to all three lanes.
        for (expr, expected) in [
            ("identity(rot, id=8)", None),
            ("bit_slice_update(rot, one, seed, id=8)", Some([0, 7])),
        ] {
            check(
                &whole_word_feedback(&format!("{amounts}\n{rotate}\nv: bits[3] = {expr}")),
                expected,
                3,
            );
        }
    }
}

#[test]
fn equivalent_two_bit_swaps_and_replication() {
    let one = "one: bits[2] = literal(value=1, id=2)";
    for (extra, expr) in [
        ("", "reverse(d, id=8)"),
        (
            "lo: bits[1] = bit_slice(d, start=0, width=1, id=4)\nhi: bits[1] = bit_slice(d, start=1, width=1, id=5)",
            "concat(lo, hi, id=8)",
        ),
        (
            "zero: bits[2] = literal(value=0, id=2)\nlo: bits[1] = dynamic_bit_slice(d, zero, width=1, id=4)\nhi: bits[1] = bit_slice(d, start=1, width=1, id=5)",
            "concat(lo, hi, id=8)",
        ),
        (
            "one: bits[2] = literal(value=1, id=2)\nl: bits[2] = shll(d, one, id=4)\nr: bits[2] = shrl(d, one, id=5)",
            "or(l, r, id=8)",
        ),
    ] {
        check(&design(extra, expr, 2, 1, true), Some([0, 3]), 2);
        check(&design(extra, expr, 2, 0, true), None, 2);
    }
    let extra = format!("{one}\nlo: bits[1] = bit_slice(d, start=0, width=1, id=4)");
    for expr in [
        "concat(lo, lo, id=8)",
        "sign_ext(lo, new_bit_count=2, id=8)",
        "bit_slice_update(d, one, lo, id=8)",
    ] {
        check(&design(&extra, expr, 2, 1, true), Some([0, 3]), 2);
        check(&design(&extra, expr, 2, 1, false), None, 2);
    }
}

#[test]
fn fixed_fields_pad_truncate_and_ignore_out_of_bounds_updates() {
    let huge = "huge: bits[128] = literal(value=0x10000000000000000, id=3)";
    for (body, expected) in [
        (format!("{huge}\nv: bits[3] = dynamic_bit_slice(d, huge, width=3, id=4)"), Some([0, 0])),
        (format!("{huge}\nv: bits[3] = bit_slice_update(d, huge, seed, id=4)"), None),
        ("one: bits[2] = literal(value=1, id=3)\nv: bits[3] = dynamic_bit_slice(d, one, width=3, id=4)".to_owned(), Some([0, 0])),
        // The top bit of the update is truncated. q[0] is tied to seed and
        // q[2:1] reads q[1:0], so all three bits settle to seed.
        ("one: bits[2] = literal(value=1, id=3)\nbase: bits[3] = zero_ext(seed, new_bit_count=3, id=4)\nv: bits[3] = bit_slice_update(base, one, d, id=5)".to_owned(), Some([0, 7])),
    ] {
        check(&whole_word_feedback(&body), expected, 3);
    }
}

#[test]
fn local_calls_preserve_literal_offsets_and_distinguish_bindings() {
    let functions = r#"fn move_bits(x: bits[3] id=40, amount: bits[2] id=41) -> bits[3] {
  ret shifted: bits[3] = shll(x, amount, id=42)
}
fn forward(n: bits[2] id=43, data: bits[3] id=44) -> bits[3] {
  ret moved: bits[3] = invoke(data, n, to_apply=move_bits, id=45)
}
"#;
    for (left, right, expected) in [
        ("one", "two", Some([0, 7])),
        ("one", "zero", None),
        ("zero", "one", None),
        ("one", "runtime", None),
    ] {
        let body = format!(
            r#"  zero: bits[2] = literal(value=0, id=3)
  one: bits[2] = literal(value=1, id=4)
  two: bits[2] = literal(value=2, id=5)
  runtime: bits[2] = zero_ext(seed, new_bit_count=2, id=6)
  a: bits[3] = invoke({left}, d, to_apply=forward, id=7)
  b: bits[3] = invoke({right}, d, to_apply=forward, id=8)
  low: bits[3] = zero_ext(seed, new_bit_count=3, id=9)
  v: bits[3] = or(a, b, low, id=10)"#
        );
        let source =
            whole_word_feedback(&body).replace("block leaf(", &format!("{functions}block leaf("));
        check(&source, expected, 3);
    }
}

#[test]
fn local_calls_use_literal_offsets_for_fields_and_array_indices() {
    for (body, expected) in [
        (
            "ret v: bits[3] = dynamic_bit_slice(x, i, width=3, id=43)",
            [0, 0],
        ),
        (
            "base: bits[3] = zero_ext(s, new_bit_count=3, id=44)\n  ret v: bits[3] = bit_slice_update(base, i, x, id=43)",
            [0, 7],
        ),
        (
            "base: bits[3] = zero_ext(s, new_bit_count=3, id=44)\n  a: bits[3][2] = array(x, base, id=45)\n  ret v: bits[3] = array_index(a, indices=[i], id=43)",
            [0, 1],
        ),
    ] {
        let function = format!(
            "fn select_bits(x: bits[3] id=40, i: bits[1] id=41, s: bits[1] id=42) -> bits[3] {{\n  {body}\n}}\n"
        );
        for (offset, result) in [(1, Some(expected)), (0, None)] {
            let source = whole_word_feedback(&format!(
                "i: bits[1] = literal(value={offset}, id=3)\n  v: bits[3] = invoke(d, i, seed, to_apply=select_bits, id=4)"
            ))
            .replace("block leaf(", &format!("{function}block leaf("));
            check(&source, result, 3);
        }
    }
}

#[test]
fn literal_child_ports_keep_instances_distinct() {
    for (left, right, expected) in [
        ("one", "two", Some([0, 7])),
        ("one", "zero", None),
        ("zero", "one", None),
        ("one", "runtime", None),
    ] {
        let source = format!(
            r#"package shifts
fn move_bits(x: bits[3] id=1, n: bits[2] id=2) -> bits[3] {{
  ret value: bits[3] = shll(x, n, id=3)
}}
block shift(d: bits[3], amount: bits[2], q: bits[3]) {{
  d: bits[3] = input_port(name=d, id=4)
  amount: bits[2] = input_port(name=amount, id=5)
  value: bits[3] = invoke(d, amount, to_apply=move_bits, id=6)
  q: () = output_port(value, name=q, id=7)
}}
block stage(d: bits[3], amount: bits[2], q: bits[3]) {{
  instantiation inner(block=shift, kind=block)
  d: bits[3] = input_port(name=d, id=8)
  amount: bits[2] = input_port(name=amount, id=9)
  data_in: () = instantiation_input(d, instantiation=inner, port_name=d, id=10)
  amount_in: () = instantiation_input(amount, instantiation=inner, port_name=amount, id=11)
  value: bits[3] = instantiation_output(instantiation=inner, port_name=q, id=12)
  q: () = output_port(value, name=q, id=13)
}}
top block wrapper(seed: bits[1], result: bits[3]) {{
  instantiation a(block=stage, kind=block)
  instantiation b(block=stage, kind=block)
  seed: bits[1] = input_port(name=seed, id=14)
  one: bits[2] = literal(value=1, id=15)
  two: bits[2] = literal(value=2, id=16)
  zero: bits[2] = literal(value=0, id=17)
  runtime: bits[2] = zero_ext(seed, new_bit_count=2, id=18)
  a_out: bits[3] = instantiation_output(instantiation=a, port_name=q, id=19)
  b_out: bits[3] = instantiation_output(instantiation=b, port_name=q, id=20)
  seed_ext: bits[3] = zero_ext(seed, new_bit_count=3, id=21)
  v: bits[3] = or(a_out, b_out, seed_ext, id=22)
  a_data: () = instantiation_input(v, instantiation=a, port_name=d, id=23)
  b_data: () = instantiation_input(v, instantiation=b, port_name=d, id=24)
  a_amount: () = instantiation_input({left}, instantiation=a, port_name=amount, id=25)
  b_amount: () = instantiation_input({right}, instantiation=b, port_name=amount, id=26)
  result: () = output_port(v, name=result, id=27)
}}
"#
        );
        check(&source, expected, 3);
    }
}

#[test]
fn selects_only_the_requested_element_of_a_callers_aggregate() {
    let functions = r#"fn pick(rows: bits[3][2][2] id=40, i: bits[2] id=41, j: bits[2] id=42) -> bits[3] {
  ret selected: bits[3] = array_index(rows, indices=[i, j], id=43)
}
"#;
    for (i, j, expected) in [
        (0, 0, None),
        (0, 1, Some([0, 1])),
        (1, 0, Some([1, 0])),
        (1, 1, None),
        (3, 0, Some([1, 0])),
    ] {
        let body = format!(
            r#"  i: bits[2] = literal(value={i}, id=3)
  j: bits[2] = literal(value={j}, id=4)
  s: bits[3] = zero_ext(seed, new_bit_count=3, id=5)
  n: bits[1] = not(seed, id=6)
  ns: bits[3] = zero_ext(n, new_bit_count=3, id=7)
  a: bits[3][2] = array(d, s, id=8)
  b: bits[3][2] = array(ns, d, id=9)
  rows: bits[3][2][2] = array(a, b, id=10)
  v: bits[3] = invoke(rows, i, j, to_apply=pick, id=11)"#
        );
        let source =
            whole_word_feedback(&body).replace("block leaf(", &format!("{functions}block leaf("));
        check(&source, expected, 3);
    }
}

#[test]
fn aggregate_results_preserve_nested_tuple_and_array_selections() {
    let helpers = r#"fn get_row(fields: (bits[3], bits[3][2]) id=40) -> bits[3][2] {
  ret row: bits[3][2] = tuple_index(fields, index=1, id=41)
}
fn pick(row: bits[3][2] id=42, i: bits[1] id=43) -> bits[3] {
  ret selected: bits[3] = array_index(row, indices=[i], id=44)
}
"#;
    for (index, expected) in [(1, Some([0, 1])), (0, None)] {
        let body = format!(
            r#"  i: bits[1] = literal(value={index}, id=3)
  s: bits[3] = zero_ext(seed, new_bit_count=3, id=4)
  a: bits[3][2] = array(d, s, id=5)
  t: (bits[3], bits[3][2]) = tuple(d, a, id=6)
  row: bits[3][2] = invoke(t, to_apply=get_row, id=7)
  v: bits[3] = invoke(row, i, to_apply=pick, id=8)"#
        );
        let source =
            whole_word_feedback(&body).replace("block leaf(", &format!("{helpers}block leaf("));
        check(&source, expected, 3);
    }
}

/// Drives a shift amount from a fixed value, an independent input or the loop.
#[derive(Clone, Copy)]
enum ShiftControl {
    Fixed(usize),
    Input,
    Feedback,
}

/// Moves the same feedback connection between the top and a child and permutes
/// sibling declarations without changing any of the drivers.
fn sibling_shift(control: ShiftControl, nested_feedback: bool, source_first: bool) -> String {
    let source_body = match control {
        ShiftControl::Fixed(n) => format!("literal(value={n}, id=5)"),
        _ => "zero_ext(arg, new_bit_count=3, id=5)".to_owned(),
    };
    let source_input = match control {
        ShiftControl::Feedback => "low",
        _ => "seed",
    };
    let source_instance = "  instantiation src(block=source, kind=block)";
    let (component, feedback) = if nested_feedback {
        ("stage", "")
    } else {
        (
            "shift",
            "drive: () = instantiation_input(q, instantiation=u, port_name=d, id=32)",
        )
    };
    let shift_instance = format!("  instantiation u(block={component}, kind=block)");
    let declarations = if source_first {
        format!("{source_instance}\n{shift_instance}")
    } else {
        format!("{shift_instance}\n{source_instance}")
    };
    let stage = if nested_feedback {
        r#"block stage(seed: bits[1], amount: bits[3], q: bits[3]) {
  instantiation inner(block=shift, kind=block)
  seed: bits[1] = input_port(name=seed, id=18)
  amount: bits[3] = input_port(name=amount, id=19)
  q_value: bits[3] = instantiation_output(instantiation=inner, port_name=q, id=20)
  feedback: () = instantiation_input(q_value, instantiation=inner, port_name=d, id=21)
  seed_in: () = instantiation_input(seed, instantiation=inner, port_name=seed, id=22)
  amount_in: () = instantiation_input(amount, instantiation=inner, port_name=amount, id=23)
  q: () = output_port(q_value, name=q, id=24)
}
"#
    } else {
        ""
    };
    format!(
        r#"package shifted_feedback
fn forward(x: bits[3] id=1) -> bits[3] {{
  ret alias: bits[3] = identity(x, id=2)
}}
block source(arg: bits[1], n: bits[3]) {{
  arg: bits[1] = input_port(name=arg, id=4)
  raw: bits[3] = {source_body}
  n_value: bits[3] = invoke(raw, to_apply=forward, id=6)
  n: () = output_port(n_value, name=n, id=7)
}}
block shift(d: bits[3], seed: bits[1], amount: bits[3], q: bits[3]) {{
  d: bits[3] = input_port(name=d, id=8)
  seed: bits[1] = input_port(name=seed, id=9)
  amount: bits[3] = input_port(name=amount, id=10)
  shifted: bits[3] = shll(d, amount, id=11)
  low: bits[3] = zero_ext(seed, new_bit_count=3, id=12)
  value: bits[3] = or(shifted, low, id=13)
  q: () = output_port(value, name=q, id=14)
}}
{stage}top block wrapper(seed: bits[1], result: bits[3]) {{
{declarations}
  seed: bits[1] = input_port(name=seed, id=26)
  q: bits[3] = instantiation_output(instantiation=u, port_name=q, id=27)
  low: bits[1] = bit_slice(q, start=0, width=1, id=28)
  arg_in: () = instantiation_input({source_input}, instantiation=src, port_name=arg, id=29)
  n: bits[3] = instantiation_output(instantiation=src, port_name=n, id=30)
  amount_in: () = instantiation_input(n, instantiation=u, port_name=amount, id=31)
  {feedback}
  seed_in: () = instantiation_input(seed, instantiation=u, port_name=seed, id=33)
  result: () = output_port(q, name=result, id=34)
}}
"#
    )
}

#[test]
fn feedback_is_invariant_to_sibling_order_and_hierarchy() {
    for control in [
        ShiftControl::Fixed(0),
        ShiftControl::Fixed(1),
        ShiftControl::Fixed(2),
        ShiftControl::Fixed(3),
        ShiftControl::Fixed(5),
        ShiftControl::Input,
        ShiftControl::Feedback,
    ] {
        // q[0] gets seed and each higher lane reads the lane n positions below.
        // A runtime amount can be zero, leaving a lane dependent on itself.
        let expected = match control {
            ShiftControl::Fixed(n) if n > 0 => {
                Some([0, (0..3).step_by(n).map(|lane| 1u64 << lane).sum()])
            }
            _ => None,
        };
        for nested_feedback in [false, true] {
            for source_first in [false, true] {
                check(
                    &sibling_shift(control, nested_feedback, source_first),
                    expected,
                    3,
                );
            }
        }
    }
}

#[test]
fn static_array_reads_and_updates_select_only_their_drivers() {
    let fields = r#"  s: bits[3] = zero_ext(seed, new_bit_count=3, id=3)
  x: bits[3][1] = array(d, id=4)
  y: bits[3][1] = array(s, id=5)
  row: bits[3][2] = array_concat(x, y, id=6)
  matrix: bits[3][2][2] = array(row, row, id=7)
  zero: bits[2] = literal(value=0, id=8)
  one: bits[2] = literal(value=1, id=9)
  huge: bits[128] = literal(value=0x10000000000000000, id=10)"#;
    for (ops, expected) in [
        (
            "v: bits[3] = array_index(row, indices=[one], id=11)",
            Some([0, 1]),
        ),
        ("v: bits[3] = array_index(row, indices=[zero], id=11)", None),
        // A huge start clamps to the last source element, including the second
        // read.
        (
            "part: bits[3][2] = array_slice(row, huge, width=2, id=11)\nv: bits[3] = array_index(part, indices=[one], id=12)",
            Some([0, 1]),
        ),
        (
            "changed: bits[3][2][2] = array_update(matrix, s, indices=[zero, zero], id=11)\nv: bits[3] = array_index(changed, indices=[zero, zero], id=12)",
            Some([0, 1]),
        ),
        (
            "changed: bits[3][2][2] = array_update(matrix, s, indices=[zero, zero], id=11)\nv: bits[3] = array_index(changed, indices=[one, zero], id=12)",
            None,
        ),
        (
            "changed: bits[3][2][2] = array_update(matrix, d, indices=[one, huge], id=11)\nv: bits[3] = array_index(changed, indices=[one, one], id=12)",
            Some([0, 1]),
        ),
        (
            "changed: bits[3][2][2] = array_update(matrix, d, indices=[one, huge], id=11)\nv: bits[3] = array_index(changed, indices=[one, zero], id=12)",
            None,
        ),
        (
            "v: bits[3] = sel(one, cases=[d, s], default=d, id=11)",
            Some([0, 1]),
        ),
        (
            "v: bits[3] = sel(huge, cases=[d], default=s, id=11)",
            Some([0, 1]),
        ),
        (
            "v: bits[3] = sel(zero, cases=[d, s], default=s, id=11)",
            None,
        ),
        ("v: bits[3] = sel(seed, cases=[d, s], id=11)", None),
    ] {
        check(
            &whole_word_feedback(&format!("{fields}\n  {ops}")),
            expected,
            3,
        );
    }
}

#[test]
fn whole_array_feedback_distinguishes_shifting_from_recirculation() {
    for (parts, expected) in [("new, head", Some([0, 7])), ("head, new", None)] {
        let source = format!(
            r#"package array_shift
block shift(d: bits[1][3], seed: bits[1], q: bits[1][3]) {{
  d: bits[1][3] = input_port(name=d, id=1)
  seed: bits[1] = input_port(name=seed, id=2)
  zero: bits[1] = literal(value=0, id=3)
  new: bits[1][1] = array(seed, id=4)
  head: bits[1][2] = array_slice(d, zero, width=2, id=5)
  moved: bits[1][3] = array_concat({parts}, id=6)
  q: () = output_port(moved, name=q, id=7)
}}
top block wrapper(seed: bits[1], result: bits[3]) {{
  instantiation u(block=shift, kind=block)
  seed: bits[1] = input_port(name=seed, id=8)
  q: bits[1][3] = instantiation_output(instantiation=u, port_name=q, id=9)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=10)
  input_seed: () = instantiation_input(seed, instantiation=u, port_name=seed, id=11)
  zero: bits[2] = literal(value=0, id=12)
  one: bits[2] = literal(value=1, id=13)
  two: bits[2] = literal(value=2, id=14)
  a: bits[1] = array_index(q, indices=[zero], id=15)
  b: bits[1] = array_index(q, indices=[one], id=16)
  c: bits[1] = array_index(q, indices=[two], id=17)
  word: bits[3] = concat(c, b, a, id=18)
  result: () = output_port(word, name=result, id=19)
}}"#,
        );
        check(&source, expected, 3);
    }
}

/// Routes a pass-through child's feedback into selected or unselected cases.
/// A live feedback arm recirculates, while "lo" and "hi" carry distinct seed
/// bits.
fn fixed_mux(op: &str, mask: u128, cases: &[&str], default: &str, aggregate: bool) -> String {
    let width = cases.len();
    let ty = if aggregate {
        "(bits[3], bits[3][2])"
    } else {
        "bits[3]"
    };
    let fields = if aggregate {
        r#"
  low_row: bits[3][2] = array(lo_bits, hi_bits, id=16)
  high_row: bits[3][2] = array(hi_bits, lo_bits, id=17)
  cyclic_row: bits[3][2] = array(q, q, id=18)
  lo: (bits[3], bits[3][2]) = tuple(q, low_row, id=19)
  hi: (bits[3], bits[3][2]) = tuple(q, high_row, id=20)
  feedback: (bits[3], bits[3][2]) = tuple(q, cyclic_row, id=21)"#
    } else {
        r#"
  lo: bits[3] = identity(lo_bits, id=19)
  hi: bits[3] = identity(hi_bits, id=20)
  feedback: bits[3] = identity(q, id=21)"#
    };
    let selection = if aggregate {
        r#"
  row: bits[3][2] = tuple_index(selected, index=1, id=23)
  index: bits[1] = literal(value=0, id=24)
  next: bits[3] = array_index(row, indices=[index], id=25)"#
    } else {
        "  next: bits[3] = identity(selected, id=25)"
    };
    let fallback = if op == "priority_sel" {
        format!(", default={default}")
    } else {
        String::new()
    };
    let cases = cases.join(", ");
    format!(
        r#"package fixed_mux
fn route_mask(x: bits[{width}] id=1) -> bits[{width}] {{
  ret routed: bits[{width}] = identity(x, id=2)
}}
block pass(d: bits[3], y: bits[3]) {{
  d: bits[3] = input_port(name=d, id=3)
  y: () = output_port(d, name=y, id=4)
}}
top block wrapper(seed: bits[1], result: bits[3]) {{
  instantiation u(block=pass, kind=block)
  seed: bits[1] = input_port(name=seed, id=5)
  raw_mask: bits[{width}] = literal(value={mask:#x}, id=6)
  mask: bits[{width}] = invoke(raw_mask, to_apply=route_mask, id=7)
  pad: bits[2] = literal(value=0, id=8)
  lo_bits: bits[3] = concat(pad, seed, id=9)
  hi_bits: bits[3] = concat(seed, pad, id=10)
  q: bits[3] = instantiation_output(instantiation=u, port_name=y, id=11)
{fields}
  selected: {ty} = {op}(mask, cases=[{cases}]{fallback}, id=22)
{selection}
  drive: () = instantiation_input(next, instantiation=u, port_name=d, id=26)
  result: () = output_port(q, name=result, id=27)
}}
"#
    )
}

#[test]
fn constant_priority_and_one_hot_masks_trace_only_enabled_arms() {
    let cases = ["lo", "feedback", "hi"];
    for aggregate in [false, true] {
        for (op, mask, default, expected) in [
            ("priority_sel", 0, "hi", Some(4)),
            ("priority_sel", 0, "feedback", None),
            ("priority_sel", 0b111, "feedback", Some(1)),
            ("priority_sel", 0b110, "lo", None),
            ("one_hot_sel", 0, "lo", Some(0)),
            ("one_hot_sel", 0b100, "lo", Some(4)),
            ("one_hot_sel", 0b101, "lo", Some(5)),
            ("one_hot_sel", 0b111, "lo", None),
        ] {
            check(
                &fixed_mux(op, mask, &cases, default, aggregate),
                expected.map(|value| [0, value]),
                3,
            );
        }
    }
}

#[test]
fn selector_masks_keep_case_bits_above_the_host_word() {
    let mut cases = vec!["feedback"; 100];
    cases[0] = "lo";
    cases[99] = "hi";
    let high = 1u128 << 99;
    for (op, mask, expected) in [
        ("priority_sel", high, Some(4)),
        ("priority_sel", high | 1, Some(1)),
        ("one_hot_sel", high | 1, Some(5)),
        ("one_hot_sel", high | (1 << 98), None),
    ] {
        check(
            &fixed_mux(op, mask, &cases, "feedback", false),
            expected.map(|value| [0, value]),
            3,
        );
    }
}

#[test]
fn projected_aggregate_mux_can_itself_supply_a_static_control() {
    // The shift reads (1 | 2) from the second tuple field. The first field
    // recirculates and the third mux arm is entirely disabled.
    let source = r#"package masked_shift
block pass(d: bits[3], y: bits[3]) {
  d: bits[3] = input_port(name=d, id=1)
  y: () = output_port(d, name=y, id=2)
}
top block wrapper(seed: bits[1], result: bits[3]) {
  instantiation u(block=pass, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[3] = instantiation_output(instantiation=u, port_name=y, id=4)
  one: bits[3] = literal(value=1, id=5)
  two: bits[3] = literal(value=2, id=6)
  mask: bits[3] = literal(value=3, id=7)
  left: (bits[3], bits[3]) = tuple(q, one, id=8)
  right: (bits[3], bits[3]) = tuple(q, two, id=9)
  disabled: (bits[3], bits[3]) = tuple(q, q, id=10)
  mux: (bits[3], bits[3]) = one_hot_sel(mask, cases=[left, right, disabled], id=11)
  amount: bits[3] = tuple_index(mux, index=1, id=12)
  d: bits[3] = shll(q, amount, id=13)
  from_seed: bits[3] = zero_ext(seed, new_bit_count=3, id=14)
  next: bits[3] = or(d, from_seed, id=15)
  drive: () = instantiation_input(next, instantiation=u, port_name=d, id=16)
  result: () = output_port(q, name=result, id=17)
}
"#;
    check(source, Some([0, 1]), 3);
}
