// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
#[allow(dead_code)]
#[path = "support/cases.rs"]
mod cases;
#[path = "support/snapshots.rs"]
mod snapshots;
mod support;
use cases::*;
use snapshots::assert_golden_sv;
use std::collections::BTreeMap;
use support::{ClockedInputs, CycleInputs, TestRtl, run_icarus_cycles};
use xlsynth::{IrBits, IrValue};
use xlsynth_codegen::{BlockCodegenOptions, Layout};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::math::ceil_log2;
use xlsynth_test_helpers::rtl_sim::LogicValue;

#[test]
fn reserved_internal_names_are_legalized_without_changing_behavior() {
    let ir = r#"package public_names

top block legalize_internal(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  initial: bits[8] = identity(value, id=2)
  result: () = output_port(initial, name=result, id=3)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let declarations = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("logic "))
        .collect::<Vec<_>>();
    assert_eq!(declarations, ["  logic [7:0] initial_;"]);
    assert_eq!(
        output(&evaluate(&generated, &[("value", 8, 0xd7)]), "result"),
        0xd7
    );
}

#[test]
fn zero_extending_an_empty_input_produces_a_nonempty_zero() {
    let ir = r#"package public_empty

top block zero_extend_empty(empty: bits[0], result: bits[8]) {
  empty: bits[0] = input_port(name=empty, id=1)
  expanded: bits[8] = zero_ext(empty, new_bit_count=8, id=2)
  result: () = output_port(expanded, name=result, id=3)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[]);
    assert_eq!(output(&values, "result"), 0);
}

#[test]
fn reductions_of_zero_width_values_preserve_their_identity_elements() {
    let ir = r#"package public_empty

top block reduce_empty(empty: bits[0], all_set: bits[1], any_set: bits[1], odd_set: bits[1]) {
  empty: bits[0] = input_port(name=empty, id=1)
  and_value: bits[1] = and_reduce(empty, id=2)
  or_value: bits[1] = or_reduce(empty, id=3)
  xor_value: bits[1] = xor_reduce(empty, id=4)
  all_set: () = output_port(and_value, name=all_set, id=5)
  any_set: () = output_port(or_value, name=any_set, id=6)
  odd_set: () = output_port(xor_value, name=odd_set, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[]);
    assert_eq!(output(&values, "all_set"), 1);
    assert_eq!(output(&values, "any_set"), 0);
    assert_eq!(output(&values, "odd_set"), 0);
}

#[test]
fn zero_width_selector_selects_its_only_case() {
    let ir = r#"package public_empty

top block select_empty(selector: bits[0], value: bits[8], result: bits[8]) {
  selector: bits[0] = input_port(name=selector, id=1)
  value: bits[8] = input_port(name=value, id=2)
  selected: bits[8] = sel(selector, cases=[value], id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("value", 8, 0xa5)]);
    assert_eq!(output(&values, "result"), 0xa5);
}

#[test]
fn zero_width_array_index_selects_element_zero() {
    let ir = r#"package public_empty

top block select_first(values: bits[8][3], index: bits[0], result: bits[8]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[0] = input_port(name=index, id=2)
  selected: bits[8] = array_index(values, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("values", 24, 0x33_22_11)]);
    assert_eq!(output(&values, "result"), 0x11);
}

#[test]
fn one_hot_functions_are_shared_and_avoid_fixed_name_collisions() {
    let ir = r#"package public_hot_helpers

top block hot_helpers(value: bits[4], one_hot_lsb_4: bits[4], first: bits[5], second: bits[5], high: bits[5]) {
  value: bits[4] = input_port(name=value, id=1)
  one_hot_lsb_4: bits[4] = input_port(name=one_hot_lsb_4, id=2)
  a: bits[5] = one_hot(value, lsb_prio=true, id=3)
  b: bits[5] = one_hot(one_hot_lsb_4, lsb_prio=true, id=4)
  h: bits[5] = one_hot(value, lsb_prio=false, id=5)
  first: () = output_port(a, name=first, id=6)
  second: () = output_port(b, name=second, id=7)
  high: () = output_port(h, name=high, id=8)
}
"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_golden_sv(&generated, "tests/testdata/one_hot_functions.svtxt");
    for value in 0_u64..16 {
        for second in 0_u64..16 {
            let outputs = evaluate(
                &generated,
                &[("value", 4, value), ("one_hot_lsb_4", 4, second)],
            );
            let low = |x: u64| if x == 0 { 16 } else { x & x.wrapping_neg() };
            let high = if value == 0 {
                16
            } else {
                1 << (63 - value.leading_zeros())
            };
            assert_eq!(output(&outputs, "first"), low(value));
            assert_eq!(output(&outputs, "second"), low(second));
            assert_eq!(output(&outputs, "high"), high);
        }
    }
}

#[test]
fn one_hot_case_functions_match_both_priorities_at_scalar_and_wide_boundaries() {
    for width in [0, 1, 2, 3, 4, 7, 8, 63, 64, 65, 129, 255, 256] {
        let ir = format!(
            r#"package public_hot_width

top block hot_width(value: bits[{width}], lsb: bits[{out_width}], msb: bits[{out_width}]) {{
  value: bits[{width}] = input_port(name=value, id=1)
  low: bits[{out_width}] = one_hot(value, lsb_prio=true, id=2)
  high: bits[{out_width}] = one_hot(value, lsb_prio=false, id=3)
  lsb: () = output_port(low, name=lsb, id=4)
  msb: () = output_port(high, name=msb, id=5)
}}
"#,
            out_width = width + 1
        );
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let samples = if width <= 8 {
            (0..1_usize << width)
                .map(|value| {
                    (0..width)
                        .map(|bit| value & (1 << bit) != 0)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            let mut samples = vec![vec![false; width], vec![true; width]];
            for selected in 0..width {
                samples.push((0..width).map(|bit| bit == selected).collect());
                samples.push(
                    (0..width)
                        .map(|bit| bit == selected || bit == width - 1 - selected)
                        .collect(),
                );
            }
            samples
        };
        for input in samples {
            let bindings = if width == 0 {
                BTreeMap::new()
            } else {
                BTreeMap::from([(
                    "value".to_owned(),
                    LogicValue::from_bits(&IrBits::from_lsb_is_0(&input)),
                )])
            };
            let actual = generated.evaluate(&bindings).unwrap();
            for (name, index) in [
                ("lsb", input.iter().position(|bit| *bit)),
                ("msb", input.iter().rposition(|bit| *bit)),
            ] {
                let selected = index.unwrap_or(width);
                let expected = IrBits::from_lsb_is_0(
                    &(0..=width).map(|bit| bit == selected).collect::<Vec<_>>(),
                );
                assert_eq!(
                    actual[name],
                    LogicValue::from_bits(&expected),
                    "{name}, width={width}, input={input:?}"
                );
            }
        }
    }
}

#[test]
fn empty_one_hot_input_produces_only_the_no_active_bit() {
    let ir = r#"package public_empty

top block one_hot_empty(empty: bits[0], result: bits[1]) {
  empty: bits[0] = input_port(name=empty, id=1)
  selected: bits[1] = one_hot(empty, lsb_prio=true, id=2)
  result: () = output_port(selected, name=result, id=3)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(output(&evaluate(&generated, &[]), "result"), 1);
}

#[test]
fn empty_decode_input_selects_bit_zero() {
    let ir = r#"package public_empty

top block decode_empty(empty: bits[0], result: bits[1]) {
  empty: bits[0] = input_port(name=empty, id=1)
  decoded: bits[1] = decode(empty, width=1, id=2)
  result: () = output_port(decoded, name=result, id=3)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(output(&evaluate(&generated, &[]), "result"), 1);
}

#[test]
fn encoding_a_single_input_bit_produces_an_empty_value() {
    let ir = r#"package public_empty

top block encode_single(value: bits[1], result: bits[8]) {
  value: bits[1] = input_port(name=value, id=1)
  encoded: bits[0] = encode(value, id=2)
  expanded: bits[8] = zero_ext(encoded, new_bit_count=8, id=3)
  result: () = output_port(expanded, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    for value in [0, 1] {
        let values = evaluate(&generated, &[("value", 1, value)]);
        assert_eq!(output(&values, "result"), 0);
    }
}

#[test]
fn nested_empty_tuple_and_array_components_preserve_live_packing() {
    let ir = r#"package public_empty

top block mixed_nested(payload: (bits[0], (bits[8], bits[0][2]), bits[0]), result: bits[8]) {
  payload: (bits[0], (bits[8], bits[0][2]), bits[0]) = input_port(name=payload, id=1)
  nested: (bits[8], bits[0][2]) = tuple_index(payload, index=1, id=2)
  value: bits[8] = tuple_index(nested, index=0, id=3)
  empty: bits[0][2] = tuple_index(nested, index=1, id=4)
  zero: bits[0] = array_index(empty, indices=[value], id=5)
  combined: bits[8] = concat(zero, value, id=6)
  result: () = output_port(combined, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(
        output(&evaluate(&generated, &[("payload", 8, 0xa6)]), "result"),
        0xa6
    );
}

#[test]
fn concatenating_a_zero_length_array_preserves_nonempty_elements() {
    let ir = r#"package public_empty

top block empty_concat(empty: bits[8][0], value: bits[8][2], result: bits[8][2]) {
  empty: bits[8][0] = input_port(name=empty, id=1)
  value: bits[8][2] = input_port(name=value, id=2)
  combined: bits[8][2] = array_concat(empty, value, id=3)
  result: () = output_port(combined, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(
        output(&evaluate(&generated, &[("value", 16, 0x33_22)]), "result"),
        0x33_22
    );
}

#[test]
fn empty_tuple_component_can_feed_a_nonempty_concatenation() {
    let ir = r#"package public_empty

top block empty_component(pair: (bits[0], bits[8]), result: bits[8]) {
  pair: (bits[0], bits[8]) = input_port(name=pair, id=1)
  empty: bits[0] = tuple_index(pair, index=0, id=2)
  value: bits[8] = tuple_index(pair, index=1, id=3)
  combined: bits[8] = concat(empty, value, id=4)
  result: () = output_port(combined, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("pair", 8, 0x5a)]);
    assert_eq!(output(&values, "result"), 0x5a);
}

#[test]
fn scalar_arithmetic_and_comparisons_match_known_results() {
    let cases = [
        ("add", 8, 250, 9, 3),
        ("sub", 8, 2, 5, 253),
        ("umul", 8, 17, 19, 67),
        ("smul", 8, 0xfe, 3, 250),
        ("and", 8, 0xf0, 0x3c, 0x30),
        ("or", 8, 0xf0, 0x3c, 0xfc),
        ("xor", 8, 0xf0, 0x3c, 0xcc),
        ("shll", 8, 3, 2, 12),
        ("shrl", 8, 0x80, 3, 0x10),
        ("shra", 8, 0x80, 3, 0xf0),
        ("eq", 1, 12, 12, 1),
        ("ne", 1, 12, 13, 1),
        ("ult", 1, 0x7f, 0x80, 1),
        ("slt", 1, 0x80, 0x7f, 1),
        ("ugt", 1, 0x80, 0x7f, 1),
        ("sgt", 1, 0x7f, 0x80, 1),
    ];
    for (operation, width, lhs, rhs, expected) in cases {
        let generated = emit(
            &binary_ir(operation, width),
            &BlockCodegenOptions::default(),
        );
        let values = evaluate(&generated, &[("lhs", 8, lhs), ("rhs", 8, rhs)]);
        assert_eq!(
            output(&values, "result"),
            expected,
            "operation {operation} produced the wrong result"
        );
    }
}

#[test]
fn unary_operations_and_reductions_preserve_bit_vector_semantics() {
    let ir = r#"package public_unary

top block unary(value: bits[4], inverted: bits[4], negated: bits[4], reversed: bits[4], allbits: bits[1], anybit: bits[1], parity: bits[1]) {
  value: bits[4] = input_port(name=value, id=1)
  complement: bits[4] = not(value, id=2)
  negative: bits[4] = neg(value, id=3)
  backward: bits[4] = reverse(value, id=4)
  all_set: bits[1] = and_reduce(value, id=5)
  any_set: bits[1] = or_reduce(value, id=6)
  odd_set: bits[1] = xor_reduce(value, id=7)
  inverted: () = output_port(complement, name=inverted, id=8)
  negated: () = output_port(negative, name=negated, id=9)
  reversed: () = output_port(backward, name=reversed, id=10)
  allbits: () = output_port(all_set, name=allbits, id=11)
  anybit: () = output_port(any_set, name=anybit, id=12)
  parity: () = output_port(odd_set, name=parity, id=13)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    for (value, inverted, negated, reversed, allbits, anybit, parity) in [
        (0b0000, 0b1111, 0b0000, 0b0000, 0, 0, 0),
        (0b0101, 0b1010, 0b1011, 0b1010, 0, 1, 0),
        (0b0111, 0b1000, 0b1001, 0b1110, 0, 1, 1),
        (0b1111, 0b0000, 0b0001, 0b1111, 1, 1, 0),
    ] {
        let values = evaluate(&generated, &[("value", 4, value)]);
        assert_eq!(output(&values, "inverted"), inverted);
        assert_eq!(output(&values, "negated"), negated);
        assert_eq!(output(&values, "reversed"), reversed);
        assert_eq!(output(&values, "allbits"), allbits);
        assert_eq!(output(&values, "anybit"), anybit);
        assert_eq!(output(&values, "parity"), parity);
    }
}

#[test]
fn priority_select_helpers_preserve_scalar_and_aggregate_cases() {
    for (ty, width) in [
        ("bits[1]", 1),
        ("bits[8]", 8),
        ("(bits[3], bits[5])", 8),
        ("bits[4][2]", 8),
    ] {
        let ir = format!(
            r#"package priority_cases
top block choose(selector: bits[2], first: {ty}, second: {ty}, fallback: {ty}, out: {ty}) {{
  selector: bits[2] = input_port(name=selector, id=1)
  first: {ty} = input_port(name=first, id=2)
  second: {ty} = input_port(name=second, id=3)
  fallback: {ty} = input_port(name=fallback, id=4)
  selected: {ty} = priority_sel(selector, cases=[first, second], default=fallback, id=5)
  out: () = output_port(selected, name=out, id=6)
}}
"#
        );
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let mask = (1 << width) - 1;
        for (selector, expected) in [(0, 0xee), (1, 0x35), (2, 0xc8), (3, 0x35)] {
            let values = evaluate(
                &generated,
                &[
                    ("selector", 2, selector),
                    ("first", width, 0x35 & mask),
                    ("second", width, 0xc8 & mask),
                    ("fallback", width, 0xee & mask),
                ],
            );
            assert_eq!(
                output(&values, "out"),
                expected & mask,
                "{ty}, selector={selector}"
            );
        }
    }
}

#[test]
fn encode_combines_position_bits_for_all_small_input_patterns() {
    for width in 2..=8 {
        let result_width = ceil_log2(width);
        let ir = format!(
            r#"package encode_patterns
top block encode_patterns(data: bits[{width}], out: bits[{result_width}]) {{
  data: bits[{width}] = input_port(name=data, id=1)
  result: bits[{result_width}] = encode(data, id=2)
  out: () = output_port(result, name=out, id=3)
}}
"#
        );
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let module = generated.prepare().unwrap();
        for data in 0..(1 << width) {
            let inputs = [("data".to_owned(), LogicValue::from_u64(width as u32, data))]
                .into_iter()
                .collect();
            let actual = module.evaluate(&inputs).unwrap();
            let expected = (0..width)
                .filter(|index| data & (1 << index) != 0)
                .fold(0, |acc, index| acc | index as u64);
            assert_eq!(
                output(&actual, "out"),
                expected,
                "width={width} data={data}"
            );
        }
    }
}

#[test]
fn selectors_priority_one_hot_encoding_and_gate_follow_xls_semantics() {
    let ir = r#"package public_selectors

top block selector_modes(mask: bits[3], predicate: bits[1], a: bits[8], b: bits[8], c: bits[8], fallback: bits[8], chosen: bits[8], prioritized: bits[8], combined: bits[8], hotlow: bits[4], hothigh: bits[4], encoded: bits[2], decoded: bits[5], gated: bits[8]) {
  mask: bits[3] = input_port(name=mask, id=1)
  predicate: bits[1] = input_port(name=predicate, id=2)
  a: bits[8] = input_port(name=a, id=3)
  b: bits[8] = input_port(name=b, id=4)
  c: bits[8] = input_port(name=c, id=5)
  fallback: bits[8] = input_port(name=fallback, id=6)
  exact: bits[8] = sel(mask, cases=[a, b, c], default=fallback, id=7)
  priority_value: bits[8] = priority_sel(mask, cases=[a, b, c], default=fallback, id=8)
  combined_value: bits[8] = one_hot_sel(mask, cases=[a, b, c], id=9)
  low_hot: bits[4] = one_hot(mask, lsb_prio=true, id=10)
  high_hot: bits[4] = one_hot(mask, lsb_prio=false, id=11)
  index: bits[2] = encode(mask, id=12)
  one_position: bits[5] = decode(mask, width=5, id=13)
  gated_value: bits[8] = gate(predicate, a, id=14)
  chosen: () = output_port(exact, name=chosen, id=15)
  prioritized: () = output_port(priority_value, name=prioritized, id=16)
  combined: () = output_port(combined_value, name=combined, id=17)
  hotlow: () = output_port(low_hot, name=hotlow, id=18)
  hothigh: () = output_port(high_hot, name=hothigh, id=19)
  encoded: () = output_port(index, name=encoded, id=20)
  decoded: () = output_port(one_position, name=decoded, id=21)
  gated: () = output_port(gated_value, name=gated, id=22)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let cases = [
        (0, 0x11, 0xee, 0x00, 8, 8, 0, 1),
        (1, 0x22, 0x11, 0x11, 1, 1, 0, 2),
        (2, 0x44, 0x22, 0x22, 2, 2, 1, 4),
        (3, 0xee, 0x11, 0x33, 1, 2, 1, 8),
        (4, 0xee, 0x44, 0x44, 4, 4, 2, 16),
        (5, 0xee, 0x11, 0x55, 1, 4, 2, 0),
        (7, 0xee, 0x11, 0x77, 1, 4, 3, 0),
    ];
    for (mask, chosen, prioritized, combined, hotlow, hothigh, encoded, decoded) in cases {
        let predicate = mask & 1;
        let values = evaluate(
            &generated,
            &[
                ("mask", 3, mask),
                ("predicate", 1, predicate),
                ("a", 8, 0x11),
                ("b", 8, 0x22),
                ("c", 8, 0x44),
                ("fallback", 8, 0xee),
            ],
        );
        for (name, expected) in [
            ("chosen", chosen),
            ("prioritized", prioritized),
            ("combined", combined),
            ("hotlow", hotlow),
            ("hothigh", hothigh),
            ("encoded", encoded),
            ("decoded", decoded),
            ("gated", if predicate == 1 { 0x11 } else { 0 }),
        ] {
            assert_eq!(
                output(&values, name),
                expected,
                "mask={mask}, output={name}"
            );
        }
    }
}

#[test]
fn signed_and_unsigned_division_corner_cases_follow_xls_semantics() {
    let cases = [
        ("udiv", 0x7f, 0x00, 0xff),
        ("udiv", 0x81, 0x03, 0x2b),
        ("sdiv", 0x7f, 0x00, 0x7f),
        ("sdiv", 0x80, 0x00, 0x80),
        ("sdiv", 0x80, 0xff, 0x80),
        ("sdiv", 0xfe, 0x02, 0xff),
        ("umod", 0x7f, 0x00, 0x00),
        ("umod", 0x81, 0x03, 0x00),
        ("smod", 0xfe, 0x03, 0xfe),
    ];
    for (operation, lhs, rhs, expected) in cases {
        let generated = emit(&binary_ir(operation, 8), &BlockCodegenOptions::default());
        let values = evaluate(&generated, &[("lhs", 8, lhs), ("rhs", 8, rhs)]);
        assert_eq!(
            output(&values, "result"),
            expected,
            "operation {operation}({lhs:#x}, {rhs:#x}) produced the wrong result:\n{generated}"
        );
    }
}

#[test]
fn arbitrary_width_arithmetic_and_literals_match_native_xls_bits() {
    let ir = r#"package public_wide

top block wide_arithmetic(lhs: bits[129], rhs: bits[129], narrow: bits[96], factor: bits[33], amount: bits[8], sum: bits[129], difference: bits[129], uproduct: bits[129], sproduct: bits[129], squotient: bits[129], sremainder: bits[129], uquotient: bits[129], uremainder: bits[129], shifted: bits[129], constant96: bits[96], constant129: bits[129]) {
  lhs: bits[129] = input_port(name=lhs, id=1)
  rhs: bits[129] = input_port(name=rhs, id=2)
  narrow: bits[96] = input_port(name=narrow, id=3)
  factor: bits[33] = input_port(name=factor, id=4)
  amount: bits[8] = input_port(name=amount, id=5)
  added: bits[129] = add(lhs, rhs, id=6)
  subtracted: bits[129] = sub(lhs, rhs, id=7)
  unsigned_product: bits[129] = umul(narrow, factor, id=8)
  signed_product: bits[129] = smul(narrow, factor, id=9)
  signed_quotient: bits[129] = sdiv(lhs, rhs, id=10)
  signed_remainder: bits[129] = smod(lhs, rhs, id=11)
  unsigned_quotient: bits[129] = udiv(lhs, rhs, id=12)
  unsigned_remainder: bits[129] = umod(lhs, rhs, id=13)
  arithmetic_shift: bits[129] = shra(lhs, amount, id=14)
  literal96: bits[96] = literal(value=0xfedcba987654321001234567, id=15)
  literal129: bits[129] = literal(value=0x10123456789abcdeffedcba9876543210, id=16)
  sum: () = output_port(added, name=sum, id=17)
  difference: () = output_port(subtracted, name=difference, id=18)
  uproduct: () = output_port(unsigned_product, name=uproduct, id=19)
  sproduct: () = output_port(signed_product, name=sproduct, id=20)
  squotient: () = output_port(signed_quotient, name=squotient, id=21)
  sremainder: () = output_port(signed_remainder, name=sremainder, id=22)
  uquotient: () = output_port(unsigned_quotient, name=uquotient, id=23)
  uremainder: () = output_port(unsigned_remainder, name=uremainder, id=24)
  shifted: () = output_port(arithmetic_shift, name=shifted, id=25)
  constant96: () = output_port(literal96, name=constant96, id=26)
  constant129: () = output_port(literal129, name=constant129, id=27)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let compiled = generated
        .prepare()
        .unwrap_or_else(|error| panic!("wide RTL did not compile:\n{generated}\n{error:?}"));

    let wide_literal = IrValue::parse_typed("bits[129]:0x10123456789abcdeffedcba9876543210")
        .unwrap()
        .to_bits()
        .unwrap();
    let narrow = IrValue::parse_typed("bits[96]:0xfedcba987654321001234567")
        .unwrap()
        .to_bits()
        .unwrap();
    let factor = IrValue::parse_typed("bits[33]:0x1fffffffd")
        .unwrap()
        .to_bits()
        .unwrap();
    let positive = IrBits::make_ubits(129, 3).unwrap();
    let zero = IrBits::zero(129);
    let negative_one = IrBits::all_ones(129);
    let minimum = IrBits::signed_min_value(129);
    let maximum = IrBits::signed_max_value(129);
    let cases = [
        (&wide_literal, &positive, 65),
        (&minimum, &negative_one, 128),
        (&minimum, &zero, 1),
        (&maximum, &zero, 96),
    ];

    for (lhs, rhs, shift) in cases {
        let inputs = [
            ("lhs".to_owned(), logic_from_ir_bits(lhs)),
            ("rhs".to_owned(), logic_from_ir_bits(rhs)),
            ("narrow".to_owned(), logic_from_ir_bits(&narrow)),
            ("factor".to_owned(), logic_from_ir_bits(&factor)),
            ("amount".to_owned(), LogicValue::from_u64(8, shift)),
        ]
        .into_iter()
        .collect();
        let actual = compiled
            .evaluate(&inputs)
            .unwrap_or_else(|error| panic!("wide RTL evaluation failed: {error:?}"));
        let expected = [
            ("sum", lhs.add(rhs)),
            ("difference", lhs.sub(rhs)),
            ("uproduct", narrow.umul(&factor)),
            ("sproduct", narrow.smul(&factor)),
            ("squotient", lhs.sdiv(rhs)),
            ("sremainder", lhs.smod(rhs)),
            ("uquotient", lhs.udiv(rhs)),
            ("uremainder", lhs.umod(rhs)),
            ("shifted", lhs.shra(shift as i64)),
            ("constant96", narrow.clone()),
            ("constant129", wide_literal.clone()),
        ];
        for (name, expected_bits) in expected {
            assert_eq!(
                ir_bits_from_logic(&actual[name]),
                expected_bits,
                "wide output={name}, lhs={lhs}, rhs={rhs}, shift={shift}"
            );
        }
    }
}

#[test]
fn arbitrary_width_multipliers_and_literals_match_icarus() {
    let ir = r#"package public_wide

top block wide_oracle(left_operand: bits[96], right_operand: bits[33], unsigned_product: bits[129], signed_product: bits[129], constant_value: bits[129]) {
  left_operand: bits[96] = input_port(name=left_operand, id=1)
  right_operand: bits[33] = input_port(name=right_operand, id=2)
  unsigned_value: bits[129] = umul(left_operand, right_operand, id=3)
  signed_value: bits[129] = smul(left_operand, right_operand, id=4)
  literal_value: bits[129] = literal(value=0x10123456789abcdeffedcba9876543210, id=5)
  unsigned_product: () = output_port(unsigned_value, name=unsigned_product, id=6)
  signed_product: () = output_port(signed_value, name=signed_product, id=7)
  constant_value: () = output_port(literal_value, name=constant_value, id=8)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let left = IrValue::parse_typed("bits[96]:0xfedcba987654321001234567")
        .unwrap()
        .to_bits()
        .unwrap();
    let right = IrValue::parse_typed("bits[33]:0x1fffffffd")
        .unwrap()
        .to_bits()
        .unwrap();
    let literal = IrValue::parse_typed("bits[129]:0x10123456789abcdeffedcba9876543210")
        .unwrap()
        .to_bits()
        .unwrap();
    let inputs = [
        ("left_operand".to_owned(), logic_from_ir_bits(&left)),
        ("right_operand".to_owned(), logic_from_ir_bits(&right)),
    ]
    .into_iter()
    .collect();
    let actual = generated.evaluate(&inputs).unwrap_or_else(|error| {
        panic!("Icarus rejected arbitrary-width arithmetic:\n{generated}\n{error:?}")
    });
    assert_eq!(
        ir_bits_from_logic(&actual["unsigned_product"]),
        left.umul(&right)
    );
    assert_eq!(
        ir_bits_from_logic(&actual["signed_product"]),
        left.smul(&right)
    );
    assert_eq!(ir_bits_from_logic(&actual["constant_value"]), literal);
}

#[test]
fn mixed_width_multipliers_and_partial_products_preserve_modular_results() {
    for (lhs_width, rhs_width, width) in [
        (1, 1, 1),
        (1, 5, 2),
        (3, 5, 1),
        (3, 5, 4),
        (3, 5, 12),
        (5, 3, 8),
        (0, 5, 8),
        (5, 0, 8),
        (65, 7, 80),
        (7, 65, 80),
    ] {
        let ir = format!(
            r#"package mixed_products
top block products(lhs: bits[{lhs_width}], rhs: bits[{rhs_width}], unsigned_product: bits[{width}], signed_product: bits[{width}], unsigned_first: bits[{width}], unsigned_second: bits[{width}], signed_first: bits[{width}], signed_second: bits[{width}]) {{
  lhs: bits[{lhs_width}] = input_port(name=lhs, id=1)
  rhs: bits[{rhs_width}] = input_port(name=rhs, id=2)
  up: bits[{width}] = umul(lhs, rhs, id=3)
  sp: bits[{width}] = smul(lhs, rhs, id=4)
  ups: (bits[{width}], bits[{width}]) = umulp(lhs, rhs, id=5)
  sps: (bits[{width}], bits[{width}]) = smulp(lhs, rhs, id=6)
  ua: bits[{width}] = tuple_index(ups, index=0, id=7)
  ub: bits[{width}] = tuple_index(ups, index=1, id=8)
  sa: bits[{width}] = tuple_index(sps, index=0, id=9)
  sb: bits[{width}] = tuple_index(sps, index=1, id=10)
  unsigned_product: () = output_port(up, name=unsigned_product, id=11)
  signed_product: () = output_port(sp, name=signed_product, id=12)
  unsigned_first: () = output_port(ua, name=unsigned_first, id=13)
  unsigned_second: () = output_port(ub, name=unsigned_second, id=14)
  signed_first: () = output_port(sa, name=signed_first, id=15)
  signed_second: () = output_port(sb, name=signed_second, id=16)
}}
"#
        );
        let lhs_samples = multiplier_input_samples(lhs_width);
        let rhs_samples = multiplier_input_samples(rhs_width);
        for layout in [Layout::None, Layout::Pipeline] {
            let generated = emit(
                &ir,
                &BlockCodegenOptions {
                    layout,
                    ..BlockCodegenOptions::default()
                },
            );
            let module = generated.prepare().unwrap();
            for lhs in &lhs_samples {
                for rhs in &rhs_samples {
                    let inputs = [("lhs", lhs), ("rhs", rhs)]
                        .into_iter()
                        .filter(|(_, bits)| bits.get_bit_count() != 0)
                        .map(|(name, bits)| (name.to_owned(), logic_from_ir_bits(bits)))
                        .collect();
                    let actual = module.evaluate(&inputs).unwrap();
                    for (signed, prefix) in [(false, "unsigned"), (true, "signed")] {
                        let resize = |bits: &IrBits| {
                            let count = bits.get_bit_count();
                            let sign = signed && count > 0 && bits.get_bit(count - 1).unwrap();
                            IrBits::from_lsb_is_0(
                                &(0..width)
                                    .map(|bit| {
                                        if bit < count {
                                            bits.get_bit(bit).unwrap()
                                        } else {
                                            sign
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        };
                        let left = resize(lhs);
                        let right = resize(rhs);
                        let expected = if signed {
                            left.smul(&right)
                        } else {
                            left.umul(&right)
                        };
                        let expected = resize(&expected);
                        let observed = ir_bits_from_logic(&actual[&format!("{prefix}_product")]);
                        let first = ir_bits_from_logic(&actual[&format!("{prefix}_first")]);
                        let second = ir_bits_from_logic(&actual[&format!("{prefix}_second")]);
                        assert_eq!(
                            observed, expected,
                            "{prefix}: {lhs_width}x{rhs_width}->{width}; lhs={lhs}; rhs={rhs}"
                        );
                        assert_eq!(
                            first.add(&second),
                            expected,
                            "{prefix} partials: {lhs_width}x{rhs_width}->{width}; lhs={lhs}; rhs={rhs}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn signed_division_matches_icarus_for_negative_operands() {
    let generated = emit(&binary_ir("sdiv", 8), &BlockCodegenOptions::default());
    let inputs = [
        ("lhs".to_owned(), LogicValue::from_u64(8, 0xfe)),
        ("rhs".to_owned(), LogicValue::from_u64(8, 2)),
    ]
    .into_iter()
    .collect();
    let reference = generated.evaluate(&inputs).unwrap_or_else(|error| {
        panic!("Icarus rejected generated division:\n{generated}\n{error:?}")
    });
    assert_eq!(
        output(&reference, "result"),
        0xff,
        "Icarus confirms the emitted RTL implements the wrong signed division:\n{generated}"
    );
}

#[test]
fn signed_operations_match_icarus_across_widths_and_conditional_contexts() {
    let ir = r#"package public_signed

top block signed_operations(lhs: bits[8], rhs: bits[4], divisor: bits[8], shift: bits[4], choose: bits[1], wide: bits[12], narrow: bits[5], quotient: bits[8], remainder: bits[8], shifted: bits[8], selected: bits[8]) {
  lhs: bits[8] = input_port(name=lhs, id=1)
  rhs: bits[4] = input_port(name=rhs, id=2)
  divisor: bits[8] = input_port(name=divisor, id=3)
  shift: bits[4] = input_port(name=shift, id=4)
  choose: bits[1] = input_port(name=choose, id=5)
  multiplied_wide: bits[12] = smul(lhs, rhs, id=6)
  multiplied_narrow: bits[5] = smul(lhs, rhs, id=7)
  divided: bits[8] = sdiv(lhs, divisor, id=8)
  modulo: bits[8] = smod(lhs, divisor, id=9)
  arithmetic_shift: bits[8] = shra(lhs, shift, id=10)
  muxed: bits[8] = sel(choose, cases=[divided, modulo], id=11)
  wide: () = output_port(multiplied_wide, name=wide, id=12)
  narrow: () = output_port(multiplied_narrow, name=narrow, id=13)
  quotient: () = output_port(divided, name=quotient, id=14)
  remainder: () = output_port(modulo, name=remainder, id=15)
  shifted: () = output_port(arithmetic_shift, name=shifted, id=16)
  selected: () = output_port(muxed, name=selected, id=17)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let arguments = [
        ("lhs", 8, 0xf9),
        ("rhs", 4, 0xd),
        ("divisor", 8, 3),
        ("shift", 4, 2),
        ("choose", 1, 1),
    ];
    let inputs = arguments
        .into_iter()
        .map(|(name, width, value)| (name.to_owned(), LogicValue::from_u64(width, value)))
        .collect();
    let reference = generated.evaluate(&inputs).unwrap_or_else(|error| {
        panic!("Icarus rejected signed operations:\n{generated}\n{error:?}")
    });
    let expected = [
        ("wide", 21),
        ("narrow", 21),
        ("quotient", 0xfe),
        ("remainder", 0xff),
        ("shifted", 0xfe),
        ("selected", 0xff),
    ];
    for (name, expected_value) in expected {
        assert_eq!(
            output(&reference, name),
            expected_value,
            "Icarus computed an incorrect signed result for `{name}`:\n{generated}"
        );
    }
}

#[test]
fn signed_expressions_remain_signed_inside_unsigned_comparisons() {
    let ir = r#"package public_signed

top block signed_context(lhs: bits[8], rhs: bits[4], shift: bits[4], bound: bits[8], limit: bits[12], shracmp: bits[1], mulcmp: bits[1]) {
  lhs: bits[8] = input_port(name=lhs, id=1)
  rhs: bits[4] = input_port(name=rhs, id=2)
  shift: bits[4] = input_port(name=shift, id=3)
  bound: bits[8] = input_port(name=bound, id=4)
  limit: bits[12] = input_port(name=limit, id=5)
  shra.6: bits[8] = shra(lhs, shift, id=6)
  smul.7: bits[12] = smul(lhs, rhs, id=7)
  ule.8: bits[1] = ule(shra.6, bound, id=8)
  ule.9: bits[1] = ule(smul.7, limit, id=9)
  shracmp: () = output_port(ule.8, name=shracmp, id=10)
  mulcmp: () = output_port(ule.9, name=mulcmp, id=11)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let arguments = [
        ("lhs", 8, 0x80),
        ("rhs", 4, 3),
        ("shift", 4, 3),
        ("bound", 8, 0x10),
        ("limit", 12, 0x100),
    ];
    let inputs = arguments
        .into_iter()
        .map(|(name, width, value)| (name.to_owned(), LogicValue::from_u64(width, value)))
        .collect();
    let reference = generated.evaluate(&inputs).unwrap_or_else(|error| {
        panic!("Icarus rejected signed comparisons:\n{generated}\n{error:?}")
    });
    assert_eq!(
        output(&reference, "shracmp"),
        0,
        "arithmetic shift was interpreted as unsigned in its comparison:\n{generated}"
    );
    assert_eq!(
        output(&reference, "mulcmp"),
        0,
        "signed multiplication was interpreted as unsigned in its comparison:\n{generated}"
    );
}

#[test]
fn unsigned_comparisons_of_signed_products_preserve_unsigned_bit_patterns() {
    for (operation, expected_negative_first, expected_positive_first) in
        [("ult", 0, 1), ("ule", 0, 1), ("ugt", 1, 0), ("uge", 1, 0)]
    {
        let ir = format!(
            r#"package public_unsigned_comparison

top block compare_products(x: bits[8], y: bits[8], z: bits[8], w: bits[8], result: bits[1]) {{
  x: bits[8] = input_port(name=x, id=1)
  y: bits[8] = input_port(name=y, id=2)
  z: bits[8] = input_port(name=z, id=3)
  w: bits[8] = input_port(name=w, id=4)
  smul.5: bits[8] = smul(x, y, id=5)
  smul.6: bits[8] = smul(z, w, id=6)
  {operation}.7: bits[1] = {operation}(smul.5, smul.6, id=7)
  result: () = output_port({operation}.7, name=result, id=8)
}}
"#
        );
        assert_stock_xls_accepts(&ir);
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let negative_first = evaluate(
            &generated,
            &[("x", 8, 0xff), ("y", 8, 1), ("z", 8, 1), ("w", 8, 1)],
        );
        let positive_first = evaluate(
            &generated,
            &[("x", 8, 1), ("y", 8, 1), ("z", 8, 0xff), ("w", 8, 1)],
        );
        assert_eq!(
            output(&negative_first, "result"),
            expected_negative_first,
            "{operation} must compare 0xff and 0x01 as unsigned:\n{generated}"
        );
        assert_eq!(
            output(&positive_first, "result"),
            expected_positive_first,
            "{operation} must compare 0x01 and 0xff as unsigned:\n{generated}"
        );
    }
}

#[test]
fn literal_and_nested_one_hot_selectors_preserve_results() {
    let ir = r#"package public_selectors

top block selectors(mask: bits[2], a: bits[8], b: bits[8], c: bits[8], d: bits[8], literalresult: bits[8], nestedresult: bits[8]) {
  mask: bits[2] = input_port(name=mask, id=1)
  a: bits[8] = input_port(name=a, id=2)
  b: bits[8] = input_port(name=b, id=3)
  c: bits[8] = input_port(name=c, id=4)
  d: bits[8] = input_port(name=d, id=5)
  literal.6: bits[4] = literal(value=6, id=6)
  one_hot_sel.7: bits[8] = one_hot_sel(literal.6, cases=[a, b, c, d], id=7)
  one_hot.8: bits[3] = one_hot(mask, lsb_prio=true, id=8)
  one_hot.9: bits[4] = one_hot(one_hot.8, lsb_prio=true, id=9)
  one_hot_sel.10: bits[8] = one_hot_sel(one_hot.9, cases=[a, b, c, d], id=10)
  literalresult: () = output_port(one_hot_sel.7, name=literalresult, id=11)
  nestedresult: () = output_port(one_hot_sel.10, name=nestedresult, id=12)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let arguments = [
        ("mask", 2, 0),
        ("a", 8, 0x11),
        ("b", 8, 0x22),
        ("c", 8, 0x44),
        ("d", 8, 0x88),
    ];
    let inputs = arguments
        .into_iter()
        .map(|(name, width, value)| (name.to_owned(), LogicValue::from_u64(width, value)))
        .collect();
    let reference = generated.evaluate(&inputs).unwrap_or_else(|error| {
        panic!("Icarus rejected literal or nested selector indexing:\n{generated}\n{error:?}")
    });
    assert_eq!(output(&reference, "literalresult"), 0x66);
    assert_eq!(output(&reference, "nestedresult"), 0x44);

    for (mask, expected) in [(0, 0x44), (1, 0x11), (2, 0x22), (3, 0x11)] {
        let values = evaluate(
            &generated,
            &[
                ("mask", 2, mask),
                ("a", 8, 0x11),
                ("b", 8, 0x22),
                ("c", 8, 0x44),
                ("d", 8, 0x88),
            ],
        );
        assert_eq!(output(&values, "literalresult"), 0x66);
        assert_eq!(output(&values, "nestedresult"), expected, "mask={mask}");
    }
}

#[test]
fn deep_one_hot_chains_materialize_compact_reusable_expression_nodes() {
    let ir = r#"package public_selectors

top block deep_one_hot(mask: bits[4], result: bits[4]) {
  mask: bits[4] = input_port(name=mask, id=1)
  one_hot.2: bits[5] = one_hot(mask, lsb_prio=true, id=2)
  one_hot.3: bits[6] = one_hot(one_hot.2, lsb_prio=true, id=3)
  one_hot.4: bits[7] = one_hot(one_hot.3, lsb_prio=true, id=4)
  one_hot.5: bits[8] = one_hot(one_hot.4, lsb_prio=true, id=5)
  one_hot.6: bits[9] = one_hot(one_hot.5, lsb_prio=true, id=6)
  encode.7: bits[4] = encode(one_hot.6, id=7)
  xor.8: bits[4] = xor(encode.7, mask, id=8)
  result: () = output_port(xor.8, name=result, id=9)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(
        ir,
        &BlockCodegenOptions {
            max_inline_depth: 100,
            ..BlockCodegenOptions::default()
        },
    );
    assert!(
        generated.len() < 12_000,
        "expression-expanding nodes must not duplicate their input DAG: {} bytes\n{generated}",
        generated.len()
    );
    let materialized = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("logic [") && line.contains("one_hot_"))
        .collect::<Vec<_>>();
    assert_eq!(
        materialized,
        [
            "  logic [4:0] one_hot_2;",
            "  logic [5:0] one_hot_3;",
            "  logic [6:0] one_hot_4;",
            "  logic [7:0] one_hot_5;",
            "  logic [8:0] one_hot_6;",
        ]
    );
    for mask in 0u64..16 {
        let expected_index = if mask == 0 {
            4
        } else {
            u64::from(mask.trailing_zeros())
        };
        let values = evaluate(&generated, &[("mask", 4, mask)]);
        assert_eq!(
            output(&values, "result"),
            expected_index ^ mask,
            "mask={mask}"
        );
    }
}

#[test]
fn array_indexing_clamps_out_of_bounds_indices_by_default() {
    let ir = r#"package public_arrays

top block lookup(values: bits[8][3], index: bits[2], result: bits[8]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[2] = input_port(name=index, id=2)
  selected: bits[8] = array_index(values, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    for (index, expected) in [(0, 0x11), (1, 0x22), (2, 0x33), (3, 0x33)] {
        let values = evaluate(
            &generated,
            &[("values", 24, 0x33_22_11), ("index", 2, index)],
        );
        assert_eq!(output(&values, "result"), expected);
    }
}

#[test]
fn array_index_guards_are_omitted_only_when_the_index_width_is_safe() {
    for (count, index_width, access) in [
        (1usize, 0usize, "data[1'h0]"),
        (1, 1, "data[index < 1'h1 ? index : 1'h0]"),
        (2, 1, "data[index]"),
        (3, 1, "data[index]"),
        (3, 2, "data[index < 2'h3 ? index : 2'h2]"),
        (4, 2, "data[index]"),
        (4, 3, "data[index < 3'h4 ? index : 3'h3]"),
        (5, 2, "data[index]"),
        (8, 2, "data[index]"),
        (8, 3, "data[index]"),
    ] {
        let ir = format!(
            r#"package public_index_width

top block width_index(data: bits[8][{count}], index: bits[{index_width}], result: bits[8]) {{
  data: bits[8][{count}] = input_port(name=data, id=1)
  index: bits[{index_width}] = input_port(name=index, id=2)
  selected: bits[8] = array_index(data, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}}
"#
        );
        assert_stock_xls_accepts(&ir);
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let index_port = match index_width {
            0 => String::new(),
            1 => "  input logic index,\n".to_owned(),
            width => format!("  input logic [{}:0] index,\n", width - 1),
        };
        assert_eq!(
            generated,
            format!(
                r#"module width_index(
  input logic [{last}:0][7:0] data,
{index_port}  output logic [7:0] result
);
  logic [7:0] selected;
  assign selected = {access};
  assign result = selected;
endmodule
"#,
                last = count - 1
            )
        );
        let data = (0..count).fold(0u64, |packed, i| packed | ((0x10 + i as u64) << (8 * i)));
        for index in 0..(1usize << index_width) {
            let mut inputs = vec![("data", (count * 8) as u32, data)];
            if index_width != 0 {
                inputs.push(("index", index_width as u32, index as u64));
            }
            let observed = evaluate(&generated, &inputs);
            assert_eq!(
                output(&observed, "result"),
                0x10 + index.min(count - 1) as u64
            );
        }
    }
}

#[test]
fn nested_array_index_width_checks_are_per_dimension() {
    let ir = r#"package public_nested_index_width

top block nested_index_width(data: bits[8][3][2], row: bits[1], column: bits[2], result: bits[8]) {
  data: bits[8][3][2] = input_port(name=data, id=1)
  row: bits[1] = input_port(name=row, id=2)
  column: bits[2] = input_port(name=column, id=3)
  selected: bits[8] = array_index(data, indices=[row, column], id=4)
  result: () = output_port(selected, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(
        generated,
        r#"module nested_index_width(
  input logic [1:0][2:0][7:0] data,
  input logic row,
  input logic [1:0] column,
  output logic [7:0] result
);
  logic [2:0][7:0] selected__dim0;
  logic [7:0] selected;
  assign selected__dim0 = data[row];
  assign selected = selected__dim0[column < 2'h3 ? column : 2'h2];
  assign result = selected;
endmodule
"#
    );
    for row in 0..2 {
        for column in 0..4 {
            let observed = evaluate(
                &generated,
                &[
                    ("data", 48, 0x0605_0403_0201),
                    ("row", 1, row),
                    ("column", 2, column),
                ],
            );
            assert_eq!(output(&observed, "result"), row * 3 + column.min(2) + 1);
        }
    }
}

#[test]
fn array_index_bounds_checks_cover_every_small_index() {
    for count in [1usize, 2, 3, 4, 5, 8] {
        let ir = format!(
            r#"package public_bounds_matrix

top block bounds_matrix(data: bits[8][{count}], index: bits[4], result: bits[8]) {{
  data: bits[8][{count}] = input_port(name=data, id=1)
  index: bits[4] = input_port(name=index, id=2)
  selected: bits[8] = array_index(data, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}}
"#
        );
        let data = (0..count).fold(0u64, |packed, i| packed | ((0x10 + i as u64) << (8 * i)));
        for checked in [true, false] {
            let generated = emit(
                &ir,
                &BlockCodegenOptions {
                    array_index_bounds_checking: checked,
                    ..BlockCodegenOptions::default()
                },
            );
            let compiled = generated.prepare().unwrap();
            for index in 0..16usize {
                let inputs = [
                    (
                        "data".to_owned(),
                        LogicValue::from_u64((8 * count) as u32, data),
                    ),
                    ("index".to_owned(), LogicValue::from_u64(4, index as u64)),
                ]
                .into_iter()
                .collect();
                let observed = compiled.evaluate(&inputs).unwrap();
                let effective = if checked {
                    index.min(count - 1)
                } else {
                    index & ((1usize << ceil_log2(count).max(1)) - 1)
                };
                let expected = if effective < count {
                    LogicValue::from_u64(8, 0x10 + effective as u64)
                } else {
                    LogicValue::unknown(8)
                };
                assert_eq!(
                    observed["result"], expected,
                    "count={count}, checked={checked}, index={index}"
                );
            }
        }
    }
}

#[test]
fn array_index_bounds_checks_inspect_high_bits_of_wide_indices() {
    let ir = r#"package public_wide_array_index

top block wide_index(data: bits[8][4], index: bits[80], result: bits[8]) {
  data: bits[8][4] = input_port(name=data, id=1)
  index: bits[80] = input_port(name=index, id=2)
  selected: bits[8] = array_index(data, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    let index = IrValue::parse_typed("bits[80]:0x8000_0000_0000_0000_0002")
        .unwrap()
        .to_bits()
        .unwrap();
    let inputs = [
        ("data".to_owned(), LogicValue::from_u64(32, 0x4433_2211)),
        ("index".to_owned(), logic_from_ir_bits(&index)),
    ]
    .into_iter()
    .collect();
    for (checked, expected) in [(true, 0x44), (false, 0x33)] {
        let generated = emit(
            ir,
            &BlockCodegenOptions {
                array_index_bounds_checking: checked,
                ..BlockCodegenOptions::default()
            },
        );
        let compiled = generated.prepare().unwrap();
        assert_eq!(
            output(&compiled.evaluate(&inputs).unwrap(), "result"),
            expected
        );
    }
}

#[test]
fn unchecked_array_indexing_exposes_unknown_out_of_bounds_values() {
    let ir = r#"package public_arrays

top block lookup(values: bits[8][3], index: bits[2], result: bits[8]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[2] = input_port(name=index, id=2)
  selected: bits[8] = array_index(values, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    let options = BlockCodegenOptions {
        array_index_bounds_checking: false,
        ..BlockCodegenOptions::default()
    };
    let generated = emit(ir, &options);
    let values = evaluate(&generated, &[("values", 24, 0x33_22_11), ("index", 2, 3)]);
    assert!(values["result"].has_unknown());
}

#[test]
fn unchecked_power_of_two_array_indices_truncate_to_address_width() {
    let ir = r#"package public_arrays

top block unchecked_lookup(values: bits[8][4], index: bits[3], result: bits[8]) {
  values: bits[8][4] = input_port(name=values, id=1)
  index: bits[3] = input_port(name=index, id=2)
  selected: bits[8] = array_index(values, indices=[index], id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let options = BlockCodegenOptions {
        array_index_bounds_checking: false,
        ..BlockCodegenOptions::default()
    };
    let generated = emit(ir, &options);
    for (index, expected) in [(0, 0x11), (3, 0x44), (4, 0x11), (5, 0x22), (7, 0x44)] {
        let values = evaluate(
            &generated,
            &[("values", 32, 0x44_33_22_11), ("index", 3, index)],
        );
        assert_eq!(output(&values, "result"), expected, "index={index}");
    }
}

#[test]
fn array_slice_repeats_the_last_element_without_index_overflow() {
    let ir = r#"package public_arrays

top block slice(values: bits[8][4], start: bits[3], result: bits[8][2]) {
  values: bits[8][4] = input_port(name=values, id=1)
  start: bits[3] = input_port(name=start, id=2)
  selected: bits[8][2] = array_slice(values, start, width=2, id=3)
  result: () = output_port(selected, name=result, id=4)
}
    "#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert!(
        generated.contains("for (genvar __i0 = 0; __i0 < 2;"),
        "array slices should use a compact generated element loop:\n{generated}"
    );
    for (start, expected) in [
        (0, 0x22_11),
        (1, 0x33_22),
        (2, 0x44_33),
        (3, 0x44_44),
        (4, 0x44_44),
        (7, 0x44_44),
    ] {
        let values = evaluate(
            &generated,
            &[("values", 32, 0x44_33_22_11), ("start", 3, start)],
        );
        assert_eq!(output(&values, "result"), expected, "start={start}");
    }
}

#[test]
fn array_slices_clamp_even_when_array_index_bounds_checks_are_disabled() {
    let ir = r#"package public_arrays

top block unchecked_slice(values: bits[8][4], start: bits[3], result: bits[8][2]) {
  values: bits[8][4] = input_port(name=values, id=1)
  start: bits[3] = input_port(name=start, id=2)
  selected: bits[8][2] = array_slice(values, start, width=2, id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(
        ir,
        &BlockCodegenOptions {
            array_index_bounds_checking: false,
            ..BlockCodegenOptions::default()
        },
    );
    for start in [3, 4, 7] {
        let values = evaluate(
            &generated,
            &[("values", 32, 0x44_33_22_11), ("start", 3, start)],
        );
        assert_eq!(output(&values, "result"), 0x44_44, "start={start}");
    }
}

#[test]
fn narrowing_boundaries_preserve_concatenation_width() {
    let cases = [
        (
            "dynamic",
            "  dynamic_bit_slice.3: bits[1] = dynamic_bit_slice(value, start, width=1, id=3)",
            "dynamic_bit_slice.3",
        ),
        (
            "static",
            "  not.3: bits[3] = not(value, id=3)\n  bit_slice.4: bits[1] = bit_slice(not.3, start=1, width=1, id=4)",
            "bit_slice.4",
        ),
        (
            "tuple",
            "  xor_reduce.3: bits[1] = xor_reduce(value, id=3)\n  tuple.4: (bits[3], bits[1]) = tuple(value, xor_reduce.3, id=4)\n  tuple_index.5: bits[1] = tuple_index(tuple.4, index=1, id=5)",
            "tuple_index.5",
        ),
        (
            "carry",
            "  literal.3: bits[1] = literal(value=0, id=3)\n  ext_carry_out.4: bits[1] = ext_carry_out(value, value, literal.3, id=4)",
            "ext_carry_out.4",
        ),
        (
            "update",
            "  not.3: bits[3] = not(value, id=3)\n  literal.4: bits[1] = literal(value=0, id=4)\n  bit_slice_update.5: bits[1] = bit_slice_update(literal.4, literal.4, not.3, id=5)",
            "bit_slice_update.5",
        ),
    ];
    for (kind, body, selected) in cases {
        let ir = format!(
            r#"package public_narrowing
top block narrowing(value: bits[3], start: bits[3], result: bits[3]) {{
  value: bits[3] = input_port(name=value, id=1)
  start: bits[3] = input_port(name=start, id=2)
{body}
  zero: bits[1] = literal(value=0, id=10)
  prefix: bits[2] = literal(value=2, id=11)
  or.12: bits[1] = or({selected}, zero, id=12)
  concat.13: bits[3] = concat(prefix, or.12, id=13)
  result: () = output_port(concat.13, name=result, id=14)
}}
"#
        );
        let generated = emit(
            &ir,
            &BlockCodegenOptions {
                max_inline_depth: usize::MAX,
                ..BlockCodegenOptions::default()
            },
        );
        for value in 0_u64..8 {
            for start in 0..8 {
                let bit = match kind {
                    "dynamic" => (value >> start) & 1,
                    "static" => (!value >> 1) & 1,
                    "tuple" => u64::from(value.count_ones() & 1),
                    "carry" => (2 * value) >> 3,
                    "update" => !value & 1,
                    _ => unreachable!(),
                };
                let inputs = BTreeMap::from([
                    ("value".into(), LogicValue::from_u64(3, value)),
                    ("start".into(), LogicValue::from_u64(3, start)),
                ]);
                let expected = LogicValue::from_u64(3, 4 | bit);
                assert_eq!(
                    generated.evaluate(&inputs).unwrap()["result"],
                    expected,
                    "{kind}, value={value}, start={start}"
                );
            }
        }
    }
}

#[test]
fn unary_dynamic_slices_match_icarus_without_cast_width_ambiguity() {
    for width in [1, 4] {
        for op in ["not", "neg", "and_reduce", "or_reduce", "xor_reduce"] {
            let result_width = if op.ends_with("reduce") { 1 } else { width };
            let ir = format!(
                r#"package public_cast_unary

top block cast_unary(value: bits[8], start: bits[3], result: bits[{result_width}]) {{
  value: bits[8] = input_port(name=value, id=1)
  start: bits[3] = input_port(name=start, id=2)
  selected: bits[{width}] = dynamic_bit_slice(value, start, width={width}, id=3)
  unary: bits[{result_width}] = {op}(selected, id=4)
  result: () = output_port(unary, name=result, id=5)
}}
"#
            );
            let generated = emit(&ir, &BlockCodegenOptions::default());
            let inputs = BTreeMap::from([
                ("value".into(), LogicValue::from_u64(8, 0xd3)),
                ("start".into(), LogicValue::from_u64(3, 0)),
            ]);
            let mask = (1_u64 << width) - 1;
            let selected = 3 & mask;
            let expected = match op {
                "not" => !selected & mask,
                "neg" => selected.wrapping_neg() & mask,
                "and_reduce" => u64::from(selected == mask),
                "or_reduce" => u64::from(selected != 0),
                "xor_reduce" => u64::from(selected.count_ones() % 2),
                _ => unreachable!(),
            };
            let actual = generated.evaluate(&inputs).unwrap();
            assert_eq!(
                actual["result"],
                LogicValue::from_u64(result_width, expected)
            );
        }
    }
}

#[test]
fn dynamic_bit_slices_zero_fill_indices_past_the_input_width() {
    let ir = r#"package public_slices

top block dynamic_slice(value: bits[8], start: bits[4], result: bits[4]) {
  value: bits[8] = input_port(name=value, id=1)
  start: bits[4] = input_port(name=start, id=2)
  selected: bits[4] = dynamic_bit_slice(value, start, width=4, id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    for (start, expected) in [(0, 3), (4, 0xd), (5, 6), (7, 1), (8, 0), (15, 0)] {
        let values = evaluate(&generated, &[("value", 8, 0xd3), ("start", 4, start)]);
        assert_eq!(output(&values, "result"), expected, "start={start}");
    }
}

#[test]
fn slice_helpers_cover_width_changes_and_wide_out_of_range_indices() {
    for (width, start_width, slice_width, update_width) in [
        (1_usize, 1, 1, 1),
        (1, 4, 1, 13),
        (3, 1, 3, 1),
        (8, 3, 4, 8),
        (8, 0, 8, 33),
        (8, 4, 4, 0),
        (0, 4, 0, 0),
        (33, 129, 33, 65),
        (129, 129, 65, 256),
    ] {
        let ir = format!(
            r#"package public_slice_widths
top block slice_widths(data: bits[{width}], start: bits[{start_width}], update: bits[{update_width}], sliced: bits[{slice_width}], updated: bits[{width}]) {{
  data: bits[{width}] = input_port(name=data, id=1)
  start: bits[{start_width}] = input_port(name=start, id=2)
  update: bits[{update_width}] = input_port(name=update, id=3)
  dynamic_bit_slice.4: bits[{slice_width}] = dynamic_bit_slice(data, start, width={slice_width}, id=4)
  bit_slice_update.5: bits[{width}] = bit_slice_update(data, start, update, id=5)
  sliced: () = output_port(dynamic_bit_slice.4, name=sliced, id=6)
  updated: () = output_port(bit_slice_update.5, name=updated, id=7)
}}
"#
        );
        assert_stock_xls_accepts(&ir);
        let generated = emit(
            &ir,
            &BlockCodegenOptions {
                max_inline_depth: usize::MAX,
                ..Default::default()
            },
        );
        let data = IrBits::from_lsb_is_0(&(0..width).map(|bit| bit % 3 != 1).collect::<Vec<_>>());
        let update = IrBits::from_lsb_is_0(
            &(0..update_width)
                .map(|bit| bit % 3 == 1)
                .collect::<Vec<_>>(),
        );
        let mut starts = [0, 1, width.saturating_sub(1), width, width + 1]
            .into_iter()
            .filter_map(|start| {
                IrBits::make_ubits(start_width, start as u64)
                    .ok()
                    .map(|bits| (bits, start))
            })
            .collect::<Vec<_>>();
        if start_width > 64 {
            starts.push((IrBits::all_ones(start_width), usize::MAX));
        }
        for (start_bits, start) in &starts {
            let inputs = [
                ("data".to_owned(), LogicValue::from_bits(&data)),
                ("start".to_owned(), LogicValue::from_bits(start_bits)),
                ("update".to_owned(), LogicValue::from_bits(&update)),
            ]
            .into_iter()
            .filter(|(_, value)| value.width() != 0)
            .collect();
            let actual = generated.evaluate(&inputs).unwrap();
            let selected = IrBits::from_lsb_is_0(
                &(0..slice_width)
                    .map(|bit| {
                        start
                            .checked_add(bit)
                            .is_some_and(|source| source < width && data.get_bit(source).unwrap())
                    })
                    .collect::<Vec<_>>(),
            );
            if slice_width != 0 {
                assert_eq!(actual["sliced"], LogicValue::from_bits(&selected));
            }
            if width != 0 {
                let replaced = IrBits::from_lsb_is_0(
                    &(0..width)
                        .map(|bit| {
                            if bit >= *start && bit - start < update_width {
                                update.get_bit(bit - start).unwrap()
                            } else {
                                data.get_bit(bit).unwrap()
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                assert_eq!(actual["updated"], LogicValue::from_bits(&replaced));
            }
        }
    }
}

#[test]
fn zero_width_dynamic_slice_index_selects_the_least_significant_bits() {
    let ir = r#"package public_slices

top block dynamic_slice_empty(value: bits[8], start: bits[0], result: bits[4]) {
  value: bits[8] = input_port(name=value, id=1)
  start: bits[0] = input_port(name=start, id=2)
  selected: bits[4] = dynamic_bit_slice(value, start, width=4, id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("value", 8, 0xd3)]);
    assert_eq!(output(&values, "result"), 3);
}

#[test]
fn bit_slice_updates_clip_at_the_input_boundary() {
    let ir = r#"package public_slices

top block update_slice(value: bits[8], start: bits[4], replacement: bits[4], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  start: bits[4] = input_port(name=start, id=2)
  replacement: bits[4] = input_port(name=replacement, id=3)
  updated: bits[8] = bit_slice_update(value, start, replacement, id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    for (start, expected) in [(0, 0xaf), (4, 0xfb), (6, 0xeb), (8, 0xab), (15, 0xab)] {
        let values = evaluate(
            &generated,
            &[
                ("value", 8, 0xab),
                ("start", 4, start),
                ("replacement", 4, 0xf),
            ],
        );
        assert_eq!(output(&values, "result"), expected, "start={start}");
    }
}

#[test]
fn zero_width_bit_slice_update_index_replaces_low_bits() {
    let ir = r#"package public_slices

top block update_slice_empty(value: bits[8], start: bits[0], replacement: bits[4], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  start: bits[0] = input_port(name=start, id=2)
  replacement: bits[4] = input_port(name=replacement, id=3)
  updated: bits[8] = bit_slice_update(value, start, replacement, id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("value", 8, 0xab), ("replacement", 4, 0xf)]);
    assert_eq!(output(&values, "result"), 0xaf);
}

#[test]
fn zero_width_bit_slice_replacement_preserves_the_input() {
    let ir = r#"package public_slices

top block empty_update(value: bits[8], start: bits[4], replacement: bits[0], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  start: bits[4] = input_port(name=start, id=2)
  replacement: bits[0] = input_port(name=replacement, id=3)
  updated: bits[8] = bit_slice_update(value, start, replacement, id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("value", 8, 0xab), ("start", 4, 15)]);
    assert_eq!(output(&values, "result"), 0xab);
}

#[test]
fn array_update_indices_materialize_even_with_aggressive_inlining() {
    let ir = r#"package public_update_indices
top block update_indices(values: bits[8][7], data_bit: bits[1], replacement: bits[8], result: bits[8][7]) {
  values: bits[8][7] = input_port(name=values, id=1)
  data_bit: bits[1] = input_port(name=data_bit, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  nor.4: bits[1] = nor(data_bit, data_bit, data_bit, data_bit, id=4)
  updated: bits[8][7] = array_update(values, replacement, indices=[nor.4], assumed_in_bounds=true, id=5)
  result: () = output_port(updated, name=result, id=6)
}
"#;
    assert_stock_xls_accepts(ir);
    for max_inline_depth in [5, usize::MAX] {
        let generated = emit(
            ir,
            &BlockCodegenOptions {
                max_inline_depth,
                ..BlockCodegenOptions::default()
            },
        );
        assert_golden_sv(&generated, "tests/testdata/array_update_indices.svtxt");
        for (data_bit, expected) in [(0, 0x07_06_05_04_03_a9_01), (1, 0x07_06_05_04_03_02_a9)] {
            let inputs = [
                ("values", 56, 0x07_06_05_04_03_02_01),
                ("data_bit", 1, data_bit),
                ("replacement", 8, 0xa9),
            ];
            assert_eq!(output(&evaluate(&generated, &inputs), "result"), expected);
        }
    }
}

#[test]
fn computed_array_update_indices_preserve_width_and_out_of_bounds_semantics() {
    for width in [1, 2, 3, 8] {
        let ir = format!(
            r#"package public_update_indices
top block update_indices(values: bits[8][7], index_source: bits[{width}], replacement: bits[8], result: bits[8][7]) {{
  values: bits[8][7] = input_port(name=values, id=1)
  index_source: bits[{width}] = input_port(name=index_source, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  not.4: bits[{width}] = not(index_source, id=4)
  updated: bits[8][7] = array_update(values, replacement, indices=[not.4], id=5)
  result: () = output_port(updated, name=result, id=6)
}}
"#
        );
        assert_stock_xls_accepts(&ir);
        for separate_lines in [false, true] {
            let generated = emit(
                &ir,
                &BlockCodegenOptions {
                    separate_lines,
                    max_inline_depth: usize::MAX,
                    ..BlockCodegenOptions::default()
                },
            );
            let input_array = 0x07_06_05_04_03_02_01_u64;
            for index_source in 0..(1_u64 << width) {
                let index = !index_source & ((1 << width) - 1);
                let expected = if index < 7 {
                    (input_array & !(0xff << (8 * index))) | (0xa9 << (8 * index))
                } else {
                    input_array
                };
                let actual = evaluate(
                    &generated,
                    &[
                        ("values", 56, input_array),
                        ("index_source", width, index_source),
                        ("replacement", 8, 0xa9),
                    ],
                );
                assert_eq!(
                    output(&actual, "result"),
                    expected,
                    "width={width} index={index}"
                );
            }
        }
    }
}

#[test]
fn nested_array_updates_materialize_computed_and_literal_indices() {
    let ir = r#"package public_nested_update
top block nested_update(values: bits[1][3][7], row_source: bits[1], replacement: bits[1], result: bits[1][3][7]) {
  values: bits[1][3][7] = input_port(name=values, id=1)
  row_source: bits[1] = input_port(name=row_source, id=2)
  replacement: bits[1] = input_port(name=replacement, id=3)
  not.4: bits[1] = not(row_source, id=4)
  literal.5: bits[2] = literal(value=2, id=5)
  updated: bits[1][3][7] = array_update(values, replacement, indices=[not.4, literal.5], assumed_in_bounds=true, id=6)
  result: () = output_port(updated, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(
        ir,
        &BlockCodegenOptions {
            max_inline_depth: usize::MAX,
            ..BlockCodegenOptions::default()
        },
    );
    assert_golden_sv(
        &generated,
        "tests/testdata/nested_array_update_indices.svtxt",
    );
    for row_source in 0..2 {
        let expected = 1 << ((1 - row_source) * 3 + 2);
        assert_eq!(
            output(
                &evaluate(
                    &generated,
                    &[
                        ("values", 21, 0),
                        ("row_source", 1, row_source),
                        ("replacement", 1, 1)
                    ]
                ),
                "result"
            ),
            expected
        );
    }
}

#[test]
fn array_updates_preserve_the_original_array_for_invalid_indices() {
    let ir = r#"package public_arrays

top block update(values: bits[8][3], index: bits[2], replacement: bits[8], result: bits[8][3]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[2] = input_port(name=index, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  updated: bits[8][3] = array_update(values, replacement, indices=[index], id=4)
  result: () = output_port(updated, name=result, id=5)
}
    "#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert!(
        generated.contains("for (genvar __i0 = 0; __i0 < 3;"),
        "one-dimensional array updates should use an XLS-style generate loop:\n{generated}"
    );
    for (index, expected) in [
        (0, 0x33_22_aa),
        (1, 0x33_aa_11),
        (2, 0xaa_22_11),
        (3, 0x33_22_11),
    ] {
        let values = evaluate(
            &generated,
            &[
                ("values", 24, 0x33_22_11),
                ("index", 2, index),
                ("replacement", 8, 0xaa),
            ],
        );
        assert_eq!(output(&values, "result"), expected);
    }
}

#[test]
fn partial_array_updates_replace_entire_packed_subarrays() {
    let ir = r#"package public_arrays

top block update_row(values: bits[8][3][2], row: bits[2], replacement: bits[8][3], result: bits[8][3][2]) {
  values: bits[8][3][2] = input_port(name=values, id=1)
  row: bits[2] = input_port(name=row, id=2)
  replacement: bits[8][3] = input_port(name=replacement, id=3)
  updated: bits[8][3][2] = array_update(values, replacement, indices=[row], id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(generated.matches("for (genvar ").count(), 1);
    for (row, expected) in [
        (0, 0x66_55_44_cc_bb_aa),
        (1, 0xcc_bb_aa_33_22_11),
        (2, 0x66_55_44_33_22_11),
        (3, 0x66_55_44_33_22_11),
    ] {
        let values = evaluate(
            &generated,
            &[
                ("values", 48, 0x66_55_44_33_22_11),
                ("row", 2, row),
                ("replacement", 24, 0xcc_bb_aa),
            ],
        );
        assert_eq!(output(&values, "result"), expected, "row={row}");
    }
}

#[test]
fn generated_array_updates_support_packed_tuple_elements() {
    let ir = r#"package public_arrays

top block update_tuple(values: (bits[4], bits[8])[2], index: bits[2], replacement: (bits[4], bits[8]), result: (bits[4], bits[8])[2]) {
  values: (bits[4], bits[8])[2] = input_port(name=values, id=1)
  index: bits[2] = input_port(name=index, id=2)
  replacement: (bits[4], bits[8]) = input_port(name=replacement, id=3)
  updated: (bits[4], bits[8])[2] = array_update(values, replacement, indices=[index], id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(generated.matches("for (genvar ").count(), 1);
    for (index, expected) in [(0, 0xb22_c33), (1, 0xc33_a11), (2, 0xb22_a11)] {
        let values = evaluate(
            &generated,
            &[
                ("values", 24, 0xb22_a11),
                ("index", 2, index),
                ("replacement", 12, 0xc33),
            ],
        );
        assert_eq!(output(&values, "result"), expected, "index={index}");
    }
}

#[test]
fn generated_array_updates_support_single_bit_elements() {
    let ir = r#"package public_arrays

top block update_bit(values: bits[1][4], index: bits[3], replacement: bits[1], result: bits[1][4]) {
  values: bits[1][4] = input_port(name=values, id=1)
  index: bits[3] = input_port(name=index, id=2)
  replacement: bits[1] = input_port(name=replacement, id=3)
  updated: bits[1][4] = array_update(values, replacement, indices=[index], id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert!(generated.contains("assign updated[__i0] ="));
    for (index, expected) in [(0, 0b0101), (1, 0b0110), (3, 0b1100), (4, 0b0100)] {
        let values = evaluate(
            &generated,
            &[
                ("values", 4, 0b0100),
                ("index", 3, index),
                ("replacement", 1, 1),
            ],
        );
        assert_eq!(output(&values, "result"), expected, "index={index}");
    }
}

#[test]
fn short_genvars_are_reused_without_shadowing_ports_or_later_signals() {
    let ir = r#"package public_genvar_names

top block genvar_names(values: bits[8][2], __i0: bits[1], gen__updated_0: bits[8], result: bits[8][2]) {
  values: bits[8][2] = input_port(name=values, id=1)
  __i0: bits[1] = input_port(name=__i0, id=2)
  gen__updated_0: bits[8] = input_port(name=gen__updated_0, id=3)
  updated: bits[8][2] = array_update(values, gen__updated_0, indices=[__i0], id=4)
  __i0__1: bits[8] = array_index(updated, indices=[__i0], id=5)
  updated_again: bits[8][2] = array_update(updated, __i0__1, indices=[__i0], id=6)
  result: () = output_port(updated_again, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(
        generated,
        r#"module genvar_names(
  input logic [1:0][7:0] values,
  input logic __i0,
  input logic [7:0] gen__updated_0,
  output logic [1:0][7:0] result
);
  logic [1:0][7:0] updated;
  logic [7:0] __i0__1__1;
  logic [1:0][7:0] updated_again;
  for (genvar __i0__1 = 0; __i0__1 < 2; __i0__1 = __i0__1 + 1) begin : gen__updated_0__1
    assign updated[__i0__1] = __i0 == __i0__1 ? gen__updated_0 : values[__i0__1];
  end
  assign __i0__1__1 = updated[__i0];
  for (genvar __i0__1 = 0; __i0__1 < 2; __i0__1 = __i0__1 + 1) begin : gen__updated_again_0
    assign updated_again[__i0__1] = __i0 == __i0__1 ? __i0__1__1 : updated[__i0__1];
  end
  assign result = updated_again;
endmodule
"#
    );
    for (index, expected) in [(0, 0x22aa), (1, 0xaa11)] {
        let inputs = [
            ("values", 16, 0x2211),
            ("__i0", 1, index),
            ("gen__updated_0", 8, 0xaa),
        ];
        assert_eq!(output(&evaluate(&generated, &inputs), "result"), expected);
    }
}

#[test]
fn chained_array_updates_preserve_named_intermediates_and_bounded_output() {
    const UPDATE_COUNT: usize = 18;
    let mut ir = String::from(
        r#"package public_array_chain

top block array_chain(values: bits[8][4], index: bits[2], replacement: bits[8], result: bits[8][4]) {
  values: bits[8][4] = input_port(name=values, id=1)
  index: bits[2] = input_port(name=index, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
"#,
    );
    for offset in 0..UPDATE_COUNT {
        let id = offset + 4;
        let previous = if offset == 0 {
            "values".to_owned()
        } else {
            format!("array_update.{}", id - 1)
        };
        ir.push_str(&format!(
            "  array_update.{id}: bits[8][4] = array_update({previous}, replacement, indices=[index], id={id})\n"
        ));
    }
    ir.push_str(&format!(
        "  result: () = output_port(array_update.{}, name=result, id={})\n}}\n",
        UPDATE_COUNT + 3,
        UPDATE_COUNT + 4
    ));

    assert_stock_xls_accepts(&ir);
    let generated = emit(&ir, &BlockCodegenOptions::default());
    assert!(
        generated.len() < 12_000,
        "array updates must retain graph sharing, not expand exponentially: {} bytes",
        generated.len()
    );
    assert_eq!(
        generated
            .lines()
            .filter(|line| line.starts_with("  logic [3:0][7:0] array_update_"))
            .count(),
        UPDATE_COUNT,
        "array-valued intermediates must be materialized:\n{generated}"
    );
    assert_eq!(
        generated.matches("for (genvar __i0 =").count(),
        UPDATE_COUNT,
        "each array update should retain its own generated element loop:\n{generated}"
    );
    let actual = evaluate(
        &generated,
        &[
            ("values", 32, 0x44_33_22_11),
            ("index", 2, 2),
            ("replacement", 8, 0xaa),
        ],
    );
    assert_eq!(output(&actual, "result"), 0x44_aa_22_11);
}

#[test]
fn zero_width_array_update_index_updates_only_the_first_element() {
    let ir = r#"package public_arrays

top block update_first(values: bits[8][3], index: bits[0], replacement: bits[8], result: bits[8][3]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[0] = input_port(name=index, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  updated: bits[8][3] = array_update(values, replacement, indices=[index], id=4)
  result: () = output_port(updated, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(
        &generated,
        &[("values", 24, 0x33_22_11), ("replacement", 8, 0xaa)],
    );
    assert_eq!(output(&values, "result"), 0x33_22_aa);
}

#[test]
fn nested_array_indices_clamp_each_dimension_independently() {
    let ir = r#"package public_arrays

top block nested_lookup(values: bits[8][2][2], row: bits[2], column: bits[2], result: bits[8]) {
  values: bits[8][2][2] = input_port(name=values, id=1)
  row: bits[2] = input_port(name=row, id=2)
  column: bits[2] = input_port(name=column, id=3)
  selected: bits[8] = array_index(values, indices=[row, column], id=4)
  result: () = output_port(selected, name=result, id=5)
}
    "#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    for (row, column, expected) in [
        (0, 0, 0x11),
        (0, 1, 0x22),
        (1, 0, 0x33),
        (1, 1, 0x44),
        (2, 0, 0x33),
        (0, 3, 0x22),
        (3, 3, 0x44),
    ] {
        let values = evaluate(
            &generated,
            &[
                ("values", 32, 0x44_33_22_11),
                ("row", 2, row),
                ("column", 2, column),
            ],
        );
        assert_eq!(
            output(&values, "result"),
            expected,
            "row={row}, column={column}"
        );
    }
}

#[test]
fn singleton_packed_dimensions_preserve_scalar_leaf_indexing() {
    let ir = r#"package public_singleton_arrays

top block singleton_lookup(data: bits[1][1][2], row: bits[2], column: bits[2], result: bits[1]) {
  data: bits[1][1][2] = input_port(name=data, id=1)
  row: bits[2] = input_port(name=row, id=2)
  column: bits[2] = input_port(name=column, id=3)
  selected: bits[1] = array_index(data, indices=[row, column], id=4)
  result: () = output_port(selected, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    for data in 0..4 {
        for row in 0..4 {
            for column in 0..4 {
                let values = evaluate(
                    &generated,
                    &[("data", 2, data), ("row", 2, row), ("column", 2, column)],
                );
                assert_eq!(output(&values, "result"), (data >> row.min(1)) & 1);
            }
        }
    }
}

#[test]
fn packed_tuple_leaves_preserve_nested_arrays_and_bit_slices() {
    let ir = r#"package public_array_tuple_leaves

top block tuple_leaf(data: (bits[0], bits[4], bits[8][2])[3], index: bits[2], result: bits[3]) {
  data: (bits[0], bits[4], bits[8][2])[3] = input_port(name=data, id=1)
  index: bits[2] = input_port(name=index, id=2)
  array_index.3: (bits[0], bits[4], bits[8][2]) = array_index(data, indices=[index], id=3)
  tuple_index.4: bits[8][2] = tuple_index(array_index.3, index=2, id=4)
  one: bits[1] = literal(value=1, id=5)
  array_index.6: bits[8] = array_index(tuple_index.4, indices=[one], id=6)
  bit_slice.7: bits[4] = bit_slice(array_index.6, start=2, width=4, id=7)
  bit_slice.8: bits[3] = bit_slice(bit_slice.7, start=1, width=3, id=8)
  result: () = output_port(bit_slice.8, name=result, id=9)
}
"#;
    assert_stock_xls_accepts(ir);
    for separate_lines in [false, true] {
        let generated = emit(
            ir,
            &BlockCodegenOptions {
                separate_lines,
                ..BlockCodegenOptions::default()
            },
        );
        for (index, expected) in [(0, 2), (1, 6), (2, 2), (3, 2)] {
            let inputs = [("data", 60, 0xc5566_b3344_a1122), ("index", 2, index)];
            assert_eq!(output(&evaluate(&generated, &inputs), "result"), expected);
        }
    }
}

#[test]
fn nested_array_updates_require_every_index_to_be_in_bounds() {
    let ir = r#"package public_arrays

top block nested_update(values: bits[8][2][2], row: bits[2], column: bits[2], replacement: bits[8], result: bits[8][2][2]) {
  values: bits[8][2][2] = input_port(name=values, id=1)
  row: bits[2] = input_port(name=row, id=2)
  column: bits[2] = input_port(name=column, id=3)
  replacement: bits[8] = input_port(name=replacement, id=4)
  updated: bits[8][2][2] = array_update(values, replacement, indices=[row, column], id=5)
  result: () = output_port(updated, name=result, id=6)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(
        generated.matches("for (genvar __i").count(),
        2,
        "multidimensional array updates should use nested generate loops:\n{generated}"
    );
    for (row, column, expected) in [
        (0, 0, 0x44_33_22_aa),
        (0, 1, 0x44_33_aa_11),
        (1, 0, 0x44_aa_22_11),
        (1, 1, 0xaa_33_22_11),
        (2, 0, 0x44_33_22_11),
        (0, 2, 0x44_33_22_11),
    ] {
        let values = evaluate(
            &generated,
            &[
                ("values", 32, 0x44_33_22_11),
                ("row", 2, row),
                ("column", 2, column),
                ("replacement", 8, 0xaa),
            ],
        );
        assert_eq!(
            output(&values, "result"),
            expected,
            "row={row}, column={column}"
        );
    }
}

#[test]
fn array_concatenation_preserves_element_order() {
    let ir = r#"package public_arrays

top block join_arrays(first: bits[8][2], second: bits[8][1], result: bits[8][3]) {
  first: bits[8][2] = input_port(name=first, id=1)
  second: bits[8][1] = input_port(name=second, id=2)
  combined: bits[8][3] = array_concat(first, second, id=3)
  result: () = output_port(combined, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("first", 16, 0x22_11), ("second", 8, 0x33)]);
    assert_eq!(output(&values, "result"), 0x33_22_11);
}

#[test]
fn tuple_inputs_preserve_most_significant_first_element_order() {
    let ir = r#"package public_tuples

top block choose(pair: (bits[8], bits[8]), first: bits[8], second: bits[8]) {
  pair: (bits[8], bits[8]) = input_port(name=pair, id=1)
  upper: bits[8] = tuple_index(pair, index=0, id=2)
  lower: bits[8] = tuple_index(pair, index=1, id=3)
  first: () = output_port(upper, name=first, id=4)
  second: () = output_port(lower, name=second, id=5)
}
"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("pair", 16, 0x12_34)]);
    assert_eq!(output(&values, "first"), 0x12);
    assert_eq!(output(&values, "second"), 0x34);
}

#[test]
fn nested_array_tuple_values_keep_consistent_packed_layout() {
    let ir = r#"package public_nested

top block nested(values: (bits[4], bits[8][2]), element: bits[8]) {
  values: (bits[4], bits[8][2]) = input_port(name=values, id=1)
  array: bits[8][2] = tuple_index(values, index=1, id=2)
  zero: bits[1] = literal(value=0, id=3)
  selected: bits[8] = array_index(array, indices=[zero], id=4)
  element: () = output_port(selected, name=element, id=5)
}
"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    let values = evaluate(&generated, &[("values", 20, 0xa_33_22)]);
    assert_eq!(output(&values, "element"), 0x22);
}

#[test]
fn nested_tuple_and_bit_slices_preserve_results() {
    let ir = r#"package public_nested

top block nested_slices(pair: (bits[12], bits[20]), nested: (bits[4], (bits[16], bits[8])), first: bits[3], second: bits[4]) {
  pair: (bits[12], bits[20]) = input_port(name=pair, id=1)
  nested: (bits[4], (bits[16], bits[8])) = input_port(name=nested, id=2)
  tuple_index.3: bits[12] = tuple_index(pair, index=0, id=3)
  bit_slice.4: bits[5] = bit_slice(tuple_index.3, start=3, width=5, id=4)
  bit_slice.5: bits[3] = bit_slice(bit_slice.4, start=1, width=3, id=5)
  tuple_index.6: (bits[16], bits[8]) = tuple_index(nested, index=1, id=6)
  tuple_index.7: bits[16] = tuple_index(tuple_index.6, index=0, id=7)
  bit_slice.8: bits[4] = bit_slice(tuple_index.7, start=5, width=4, id=8)
  first: () = output_port(bit_slice.5, name=first, id=9)
  second: () = output_port(bit_slice.8, name=second, id=10)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let arguments = [("pair", 32, 0xabc1_2345), ("nested", 28, 0x0dbe_ef5a)];
    let inputs = arguments
        .into_iter()
        .map(|(name, width, value)| (name.to_owned(), LogicValue::from_u64(width, value)))
        .collect();
    let reference = generated
        .evaluate(&inputs)
        .unwrap_or_else(|error| panic!("Icarus rejected nested slices:\n{generated}\n{error:?}"));
    assert_eq!(output(&reference, "first"), 3);
    assert_eq!(output(&reference, "second"), 7);
}

#[test]
fn partial_products_preserve_the_modular_product_invariant() {
    for operation in ["umulp", "smulp"] {
        let ir = format!(
            r#"package public_partial_products

top block partial_product(lhs: bits[8], rhs: bits[8], first: bits[8], second: bits[8]) {{
  lhs: bits[8] = input_port(name=lhs, id=1)
  rhs: bits[8] = input_port(name=rhs, id=2)
  pair: (bits[8], bits[8]) = {operation}(lhs, rhs, id=3)
  low: bits[8] = tuple_index(pair, index=0, id=4)
  high: bits[8] = tuple_index(pair, index=1, id=5)
  first: () = output_port(low, name=first, id=6)
  second: () = output_port(high, name=second, id=7)
}}
"#
        );
        let generated = emit(&ir, &BlockCodegenOptions::default());
        for lhs in [0, 1, 0x7f, 0x80, 0xfe, 0xff] {
            for rhs in [0, 1, 3, 0x80, 0xff] {
                let values = evaluate(&generated, &[("lhs", 8, lhs), ("rhs", 8, rhs)]);
                let modular_sum = (output(&values, "first") + output(&values, "second")) & 0xff;
                let expected = (lhs * rhs) & 0xff;
                assert_eq!(
                    modular_sum, expected,
                    "{operation}({lhs:#x}, {rhs:#x}) violated the modular product invariant"
                );
            }
        }
    }
}

#[test]
fn public_extension_operations_match_icarus() {
    let ir = r#"package public_extensions

top block extensions(data: bits[8], lhs: bits[8], rhs: bits[8], carry: bits[1], count: bits[4], signedterm: bits[8], carryout: bits[1], priorityindex: bits[4], leading: bits[4], lowmask: bits[8], summed: bits[10], normal: bits[8], zeros: bits[4]) {
  data: bits[8] = input_port(name=data, id=1)
  lhs: bits[8] = input_port(name=lhs, id=2)
  rhs: bits[8] = input_port(name=rhs, id=3)
  carry: bits[1] = input_port(name=carry, id=4)
  count: bits[4] = input_port(name=count, id=5)
  signedterm: bits[8] = input_port(name=signedterm, id=6)
  carry_value: bits[1] = ext_carry_out(lhs, rhs, carry, id=7)
  priority_value: bits[4] = ext_prio_encode(data, lsb_prio=true, id=8)
  leading_value: bits[4] = ext_clz(data, offset=1, new_bit_count=4, id=9)
  mask_value: bits[8] = ext_mask_low(count, id=10)
  sum_value: bits[10] = ext_nary_add(lhs, signedterm, rhs, signed=[false, true, false], negated=[false, false, true], arch=kogge_stone, id=11)
  pair: (bits[8], bits[4]) = ext_normalize_left(data, shift_offset=1, normalized_bit_count=8, clz_bit_count=4, id=12)
  normalized_value: bits[8] = tuple_index(pair, index=0, id=13)
  zero_count: bits[4] = tuple_index(pair, index=1, id=14)
  carryout: () = output_port(carry_value, name=carryout, id=15)
  priorityindex: () = output_port(priority_value, name=priorityindex, id=16)
  leading: () = output_port(leading_value, name=leading, id=17)
  lowmask: () = output_port(mask_value, name=lowmask, id=18)
  summed: () = output_port(sum_value, name=summed, id=19)
  normal: () = output_port(normalized_value, name=normal, id=20)
  zeros: () = output_port(zero_count, name=zeros, id=21)
}

"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    let arguments = [
        ("data", 8, 0x18),
        ("lhs", 8, 250),
        ("rhs", 8, 10),
        ("carry", 1, 1),
        ("count", 4, 3),
        ("signedterm", 8, 0xfe),
    ];
    let interpreted = evaluate(&generated, &arguments);
    let expected = [
        ("carryout", 1),
        ("priorityindex", 3),
        ("leading", 4),
        ("lowmask", 7),
        ("summed", 238),
        ("normal", 0x80),
        ("zeros", 3),
    ];
    for (name, value) in expected {
        assert_eq!(output(&interpreted, name), value, "extension output={name}");
    }
}

#[test]
fn priority_helpers_match_pir_at_width_offset_and_zero_boundaries() {
    struct Case {
        ty: String,
        operation: String,
        unpack: String,
        output_width: usize,
        output_node: &'static str,
    }
    for input_width in [0usize, 1, 5, 8, 65] {
        let mut cases = Vec::new();
        for lsb_prio in [false, true] {
            let width = ceil_log2(input_width + 1);
            cases.push(Case {
                ty: format!("bits[{width}]"),
                operation: format!("ext_prio_encode(data, lsb_prio={lsb_prio}, id=2)"),
                unpack: String::new(),
                output_width: width,
                output_node: "result",
            });
        }
        for (width, offset) in [(1, 0), (4, 3), (4, 15), (80, usize::MAX)] {
            cases.push(Case {
                ty: format!("bits[{width}]"),
                operation: format!("ext_clz(data, offset={offset}, new_bit_count={width}, id=2)"),
                unpack: String::new(),
                output_width: width,
                output_node: "result",
            });
        }
        for (normalized_width, shift_offset, clz_width) in [
            (0, 0, Some(4)),
            (1, 0, None),
            (4, 1, Some(2)),
            (8, 0, Some(0)),
            (16, 1, Some(4)),
            (80, 0, Some(80)),
            (8, 8, None),
            (8, usize::MAX, Some(4)),
        ] {
            let output_width = normalized_width + clz_width.unwrap_or(0);
            let (ty, attribute, unpack, output_node) = if let Some(width) = clz_width {
                (
                    format!("(bits[{normalized_width}], bits[{width}])"),
                    format!(", clz_bit_count={width}"),
                    format!(
                        "  normalized: bits[{normalized_width}] = tuple_index(result, index=0, id=3)\n  count: bits[{width}] = tuple_index(result, index=1, id=4)\n  packed_result: bits[{output_width}] = concat(normalized, count, id=5)\n"
                    ),
                    "packed_result",
                )
            } else {
                (
                    format!("bits[{normalized_width}]"),
                    String::new(),
                    String::new(),
                    "result",
                )
            };
            cases.push(Case {
                ty, unpack, output_width, output_node,
                operation: format!("ext_normalize_left(data, shift_offset={shift_offset}, normalized_bit_count={normalized_width}{attribute}, id=2)"),
            });
        }
        let samples = if input_width <= 8 {
            (0..(1 << input_width))
                .map(|value| IrBits::make_ubits(input_width, value).unwrap())
                .collect::<Vec<_>>()
        } else {
            // Every priority position, adjacent set bits, all-zero, and
            // all-one.
            let mut samples = vec![
                IrBits::make_ubits(input_width, 0).unwrap(),
                IrBits::from_lsb_is_0(&vec![true; input_width]),
            ];
            for index in 0..input_width {
                let mut bits = vec![false; input_width];
                bits[index] = true;
                samples.push(IrBits::from_lsb_is_0(&bits));
                bits[(index + 1) % input_width] = true;
                samples.push(IrBits::from_lsb_is_0(&bits));
            }
            samples
        };
        for case in cases {
            let ir = format!(
                r#"package priority_boundary
top block classify(data: bits[{input_width}], out: bits[{output_width}]) {{
  data: bits[{input_width}] = input_port(name=data, id=1)
  result: {ty} = {operation}
{unpack}  out: () = output_port({output_node}, name=out, id=6)
}}
"#,
                output_width = case.output_width,
                ty = case.ty,
                operation = case.operation,
                unpack = case.unpack,
                output_node = case.output_node
            );
            let package = package(&ir);
            let xlsynth_pir::ir::PackageMember::Block { func, .. } =
                package.get_top_block().unwrap()
            else {
                unreachable!()
            };
            for layout in [Layout::None, Layout::Pipeline] {
                let generated = TestRtl::emit(
                    &package,
                    &BlockCodegenOptions {
                        layout,
                        ..BlockCodegenOptions::default()
                    },
                );
                let module = generated
                    .prepare()
                    .unwrap_or_else(|error| panic!("{ir}\n{generated}\n{error:?}"));
                for bits in &samples {
                    let expected =
                        match eval_fn_in_package(&package, func, &[IrValue::from_bits(bits)]) {
                            FnEvalResult::Success(result) => result.value.to_bits().unwrap(),
                            failure => panic!("{failure:?}"),
                        };
                    let inputs = if input_width == 0 {
                        BTreeMap::new()
                    } else {
                        [("data".to_owned(), logic_from_ir_bits(bits))]
                            .into_iter()
                            .collect()
                    };
                    let actual = module.evaluate(&inputs).unwrap();
                    if case.output_width == 0 {
                        assert!(!actual.contains_key("out"));
                    } else {
                        assert_eq!(
                            ir_bits_from_logic(&actual["out"]),
                            expected,
                            "{ir}\ninput={bits}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn synchronous_register_reset_and_enable_are_cycle_accurate() {
    for active_low in [false, true] {
        let ir = register_ir(false, active_low, true);
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let compiled = generated
            .prepare()
            .unwrap_or_else(|error| panic!("generated register RTL did not compile: {error:?}"));
        let reset = if active_low { "rst_n" } else { "rst" };
        let asserted = u64::from(!active_low);
        let released = u64::from(active_low);
        let stimulus = ClockedInputs {
            cycles: [
                (asserted, 99, 0),
                (released, 4, 1),
                (released, 9, 0),
                (released, 2, 1),
                (asserted, 255, 1),
            ]
            .into_iter()
            .map(|(reset_value, data, enable)| CycleInputs {
                inputs: [
                    (reset.to_owned(), LogicValue::from_u64(1, reset_value)),
                    ("data".to_owned(), LogicValue::from_u64(8, data)),
                    ("enable".to_owned(), LogicValue::from_u64(1, enable)),
                ]
                .into_iter()
                .collect(),
            })
            .collect(),
        };
        let observed = run_icarus_cycles(&compiled, &stimulus, &compiled.initial_state_x())
            .unwrap_or_else(|error| panic!("generated register RTL failed to execute: {error:?}"));
        let values = observed
            .iter()
            .map(|cycle| output(cycle, "result"))
            .collect::<Vec<_>>();
        assert_eq!(values, [3, 7, 7, 9, 3], "active_low={active_low}");
    }
}

#[test]
fn pipeline_and_flat_layouts_preserve_identical_cycle_behavior() {
    let flat = emit(PIPELINE, &BlockCodegenOptions::default());
    let pipelined = emit(
        PIPELINE,
        &BlockCodegenOptions {
            layout: Layout::Pipeline,
            ..BlockCodegenOptions::default()
        },
    );
    let flat = flat.prepare().unwrap();
    let pipelined = pipelined.prepare().unwrap();
    let stimulus = ClockedInputs {
        cycles: [2, 5, 9, 0xff, 0]
            .into_iter()
            .map(|data| CycleInputs {
                inputs: [("data".to_owned(), LogicValue::from_u64(8, data))]
                    .into_iter()
                    .collect(),
            })
            .collect(),
    };
    let initial = [
        ("first".to_owned(), LogicValue::from_u64(8, 0)),
        ("second".to_owned(), LogicValue::from_u64(8, 0)),
    ]
    .into_iter()
    .collect();
    let flat_outputs = run_icarus_cycles(&flat, &stimulus, &initial).unwrap();
    let pipelined_outputs = run_icarus_cycles(&pipelined, &stimulus, &initial).unwrap();
    assert_eq!(flat_outputs, pipelined_outputs);
    assert_eq!(
        flat_outputs
            .iter()
            .map(|cycle| output(cycle, "result"))
            .collect::<Vec<_>>(),
        [1, 3, 6, 10, 0]
    );
}

#[test]
fn constant_data_can_be_shared_across_pipeline_stages_without_changing_cycles() {
    let definitions = [
        ("  shared: bits[8] = literal(value=90, id=5)", 90),
        (
            r#"  empty: bits[0] = concat(id=5)
  compared: bits[1] = ne(empty, empty, id=6)
  shared: bits[8] = zero_ext(compared, new_bit_count=8, id=7)"#,
            0,
        ),
        (
            r#"  first: bits[8] = literal(value=90, id=5)
  second: bits[8] = literal(value=60, id=6)
  pair: (bits[8], bits[8]) = tuple(first, second, id=7)
  pairs: (bits[8], bits[8])[2] = array(pair, pair, id=8)
  index: bits[1] = literal(value=1, id=9)
  element: (bits[8], bits[8]) = array_index(pairs, indices=[index], id=10)
  shared: bits[8] = tuple_index(element, index=1, id=11)"#,
            60,
        ),
    ];
    for (definition, constant) in definitions {
        for early_use in ["none", "dead", "live"] {
            let ir = pipeline_with_shared_data_ir(definition, early_use);
            assert_stock_xls_accepts(&ir);
            let stimulus = ClockedInputs {
                cycles: [(0, 1), (0, 0), (0, 3), (0, 2), (1, 0), (0, 1), (0, 0)]
                    .into_iter()
                    .map(|(rst, data)| CycleInputs {
                        inputs: [
                            ("rst".to_owned(), LogicValue::from_u64(1, rst)),
                            ("data".to_owned(), LogicValue::from_u64(8, data)),
                        ]
                        .into_iter()
                        .collect(),
                    })
                    .collect(),
            };
            let initial = [
                ("r0".to_owned(), LogicValue::from_u64(8, 0)),
                ("r1".to_owned(), LogicValue::from_u64(8, 231)),
            ]
            .into_iter()
            .collect();
            let expected = [231, constant, constant, constant, 0, 0, constant];
            let mut flat_outputs = None;
            for layout in [Layout::None, Layout::Pipeline] {
                for separate_lines in [false, true] {
                    let rtl = emit(
                        &ir,
                        &BlockCodegenOptions {
                            layout,
                            separate_lines,
                            max_inline_depth: usize::MAX,
                            ..Default::default()
                        },
                    );
                    let observed = run_icarus_cycles(&rtl, &stimulus, &initial).unwrap();
                    assert_eq!(
                        observed
                            .iter()
                            .map(|cycle| output(cycle, "result"))
                            .collect::<Vec<_>>(),
                        expected
                    );
                    if let Some(flat) = &flat_outputs {
                        assert_eq!(&observed, flat);
                    } else {
                        flat_outputs = Some(observed);
                    }
                }
            }
        }
    }
}

#[test]
fn generated_array_updates_preserve_pipeline_register_boundaries() {
    let ir = r#"package public_array_pipeline

top block array_pipeline(clk: clock, values: bits[8][3], index: bits[2], replacement: bits[8], result: bits[8][3]) {
  reg state(bits[8][3])
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[2] = input_port(name=index, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  updated: bits[8][3] = array_update(values, replacement, indices=[index], id=4)
  state_d: () = register_write(updated, register=state, id=5)
  state_q: bits[8][3] = register_read(register=state, id=6)
  result: () = output_port(state_q, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    let initial = [("state".to_owned(), LogicValue::from_u64(24, 0))]
        .into_iter()
        .collect();
    let stimulus = ClockedInputs {
        cycles: [(0, 0xaa), (2, 0xbb), (3, 0xcc)]
            .into_iter()
            .map(|(index, replacement)| CycleInputs {
                inputs: [
                    ("values".to_owned(), LogicValue::from_u64(24, 0x33_22_11)),
                    ("index".to_owned(), LogicValue::from_u64(2, index)),
                    (
                        "replacement".to_owned(),
                        LogicValue::from_u64(8, replacement),
                    ),
                ]
                .into_iter()
                .collect(),
            })
            .collect(),
    };

    for layout in [Layout::None, Layout::Pipeline] {
        let generated = emit(
            ir,
            &BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            },
        );
        assert_eq!(generated.matches("for (genvar ").count(), 1);
        if layout == Layout::Pipeline {
            let loop_offset = generated.find("for (genvar ").unwrap();
            let register_offset = generated.find("// Registers for pipe stage 0:").unwrap();
            assert!(loop_offset < register_offset);
        }
        let compiled = generated
            .prepare()
            .unwrap_or_else(|error| panic!("array pipeline failed to compile: {error:?}"));
        let outputs = run_icarus_cycles(&compiled, &stimulus, &initial)
            .unwrap_or_else(|error| panic!("array pipeline failed to execute: {error:?}"));
        assert_eq!(
            outputs
                .iter()
                .map(|cycle| output(cycle, "result"))
                .collect::<Vec<_>>(),
            [0x33_22_aa, 0xbb_22_11, 0x33_22_11],
            "layout={layout:?}"
        );
    }
}

#[test]
fn nested_packed_registers_preserve_reset_enable_and_flat_bit_layout() {
    let ir = r#"package public_nested_register

top block nested_register(clk: clock, rst: bits[1], enable: bits[1], data: bits[8][2][2], result: bits[8][2][2]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg state(bits[8][2][2], reset_value=[[17, 34], [51, 68]])
  rst: bits[1] = input_port(name=rst, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  data: bits[8][2][2] = input_port(name=data, id=3)
  current: bits[8][2][2] = register_read(register=state, id=4)
  state_write: () = register_write(data, register=state, load_enable=enable, reset=rst, id=5)
  result: () = output_port(current, name=result, id=6)
}
"#;
    assert_stock_xls_accepts(ir);
    let stimulus = ClockedInputs {
        cycles: [
            (1, 1, 0),
            (0, 1, 0xdead_beef),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0x1234_5678),
        ]
        .into_iter()
        .map(|(rst, enable, data)| CycleInputs {
            inputs: [("rst", 1, rst), ("enable", 1, enable), ("data", 32, data)]
                .into_iter()
                .map(|(name, width, value)| (name.to_owned(), LogicValue::from_u64(width, value)))
                .collect(),
        })
        .collect(),
    };
    for layout in [Layout::None, Layout::Pipeline] {
        let generated = emit(
            ir,
            &BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            },
        );
        let compiled = generated.prepare().unwrap();
        let observed =
            run_icarus_cycles(&compiled, &stimulus, &compiled.initial_state_x()).unwrap();
        assert_eq!(
            observed
                .iter()
                .map(|cycle| output(cycle, "result"))
                .collect::<Vec<_>>(),
            [
                0x4433_2211,
                0xdead_beef,
                0xdead_beef,
                0x4433_2211,
                0x1234_5678
            ]
        );
    }
}

#[test]
fn pipeline_layout_preserves_reset_and_load_enable_controls() {
    let ir = register_ir(false, false, false);
    assert_stock_xls_accepts(&ir);
    let generated = emit(
        &ir,
        &BlockCodegenOptions {
            layout: Layout::Pipeline,
            ..BlockCodegenOptions::default()
        },
    );
    let stages = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("// ===== Pipe stage"))
        .collect::<Vec<_>>();
    assert_eq!(stages, ["  // ===== Pipe stage 0:"]);

    let compiled = generated.prepare().unwrap();
    let stimulus = ClockedInputs {
        cycles: [(1, 99, 0), (0, 9, 1), (0, 12, 0), (0, 6, 1)]
            .into_iter()
            .map(|(reset, data, enable)| CycleInputs {
                inputs: [
                    ("rst".to_owned(), LogicValue::from_u64(1, reset)),
                    ("data".to_owned(), LogicValue::from_u64(8, data)),
                    ("enable".to_owned(), LogicValue::from_u64(1, enable)),
                ]
                .into_iter()
                .collect(),
            })
            .collect(),
    };
    let observed = run_icarus_cycles(&compiled, &stimulus, &compiled.initial_state_x()).unwrap();
    assert_eq!(
        observed
            .iter()
            .map(|cycle| output(cycle, "result"))
            .collect::<Vec<_>>(),
        [3, 9, 9, 6]
    );
}

#[test]
fn separate_lines_and_inline_depth_preserve_observable_results() {
    let ir = r#"package public_options

top block arithmetic(a: bits[8], b: bits[8], c: bits[8], result: bits[8]) {
  a: bits[8] = input_port(name=a, id=1)
  b: bits[8] = input_port(name=b, id=2)
  c: bits[8] = input_port(name=c, id=3)
  add.4: bits[8] = add(a, b, id=4)
  xor.5: bits[8] = xor(add.4, c, id=5)
  result: () = output_port(xor.5, name=result, id=6)
}
"#;
    let inline = emit(ir, &BlockCodegenOptions::default());
    let separate = emit(
        ir,
        &BlockCodegenOptions {
            separate_lines: true,
            max_inline_depth: 0,
            ..BlockCodegenOptions::default()
        },
    );
    for (a, b, c) in [(0, 0, 0), (1, 2, 3), (0xff, 1, 0x55), (0x80, 0x80, 0x12)] {
        let inputs = [("a", 8, a), ("b", 8, b), ("c", 8, c)];
        assert_eq!(
            output(&evaluate(&inline, &inputs), "result"),
            output(&evaluate(&separate, &inputs), "result")
        );
    }
}
