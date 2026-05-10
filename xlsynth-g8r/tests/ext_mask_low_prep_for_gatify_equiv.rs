// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser;
use xlsynth_pir::math::ceil_log2;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn assert_ir_fns_equivalent(orig_fn: &ir::Fn, prepared_fn: &ir::Fn) {
    let mut orig_desugared = orig_fn.clone();
    desugar_extensions_in_fn(&mut orig_desugared).expect("desugar original PIR");
    let mut prepared_desugared = prepared_fn.clone();
    desugar_extensions_in_fn(&mut prepared_desugared).expect("desugar prepared PIR");
    let orig_pkg_text = format!("package orig\n\ntop {}", orig_desugared);
    let prepared_pkg_text = format!("package prepared\n\ntop {}", prepared_desugared);
    check_equivalence::check_equivalence(&orig_pkg_text, &prepared_pkg_text)
        .expect("prepared PIR should be equivalent to original PIR");
}

fn build_sub_mask_low_ir_text() -> String {
    r#"package sample

top fn f(count: bits[4] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  sh: bits[8] = shll(one, count, id=3)
  ret mask: bits[8] = sub(sh, one, id=4)
}
"#
    .to_string()
}

fn build_add_mask_low_ir_text() -> String {
    r#"package sample

top fn f(count: bits[4] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  all_ones: bits[8] = literal(value=255, id=3)
  sh: bits[8] = shll(one, count, id=4)
  ret mask: bits[8] = add(sh, all_ones, id=5)
}
"#
    .to_string()
}

fn build_shifted_all_ones_ir_text(output_width: usize, count_width: usize) -> String {
    let all_ones = all_ones_value(output_width);
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  ret sh: bits[{output_width}] = shll(all_ones, count, id=3)
}}
"
    )
}

fn build_sliced_shifted_all_ones_ir_text(
    source_width: usize,
    count_width: usize,
    start: usize,
    width: usize,
) -> String {
    let all_ones = all_ones_value(source_width);
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{width}] {{
  all_ones: bits[{source_width}] = literal(value={all_ones}, id=2)
  sh: bits[{source_width}] = shll(all_ones, count, id=3)
  ret slice: bits[{width}] = bit_slice(sh, start={start}, width={width}, id=4)
}}
"
    )
}

fn build_not_shifted_all_ones_ir_text(output_width: usize, count_width: usize) -> String {
    let all_ones = all_ones_value(output_width);
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  high: bits[{output_width}] = shll(all_ones, count, id=3)
  ret low: bits[{output_width}] = not(high, id=4)
}}
"
    )
}

fn build_shifted_shrl_all_ones_ir_text(output_width: usize, count_width: usize) -> String {
    let all_ones = all_ones_value(output_width);
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  low_from_top: bits[{output_width}] = shrl(all_ones, count, id=3)
  ret high: bits[{output_width}] = shll(low_from_top, count, id=4)
}}
"
    )
}

fn build_shrl_all_ones_ir_text(output_width: usize, count_width: usize) -> String {
    let all_ones = all_ones_value(output_width);
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  ret low_from_top: bits[{output_width}] = shrl(all_ones, count, id=3)
}}
"
    )
}

fn build_zero_concat_not_shifted_all_ones_ir_text(
    prefix_width: usize,
    low_width: usize,
    count_width: usize,
) -> String {
    let all_ones = all_ones_value(low_width);
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{}] {{
  zero: bits[{prefix_width}] = literal(value=0, id=2)
  all_ones: bits[{low_width}] = literal(value={all_ones}, id=3)
  high: bits[{low_width}] = shll(all_ones, count, id=4)
  low: bits[{low_width}] = not(high, id=5)
  ret out: bits[{}] = concat(zero, low, id=6)
}}
",
        prefix_width + low_width,
        prefix_width + low_width
    )
}

fn all_ones_value(width: usize) -> u128 {
    if width == 0 { 0 } else { (1u128 << width) - 1 }
}

fn bounded_count_width_for_width(width: usize) -> usize {
    let mut count_width = ceil_log2(width.saturating_add(1));
    if count_width > 0 && ((1usize << count_width) - 1) > width {
        count_width -= 1;
    }
    count_width
}

fn build_mask_low_idiom_ir_text(output_width: usize, count_width: usize, op_kind: &str) -> String {
    let all_ones = all_ones_value(output_width);
    let ret_expr = match op_kind {
        "sub" => "sub(sh, one, id=4)".to_string(),
        "add_rhs" => "add(sh, all_ones, id=5)".to_string(),
        "add_lhs" => "add(all_ones, sh, id=5)".to_string(),
        _ => panic!("unknown op kind {op_kind}"),
    };
    let all_ones_node = if op_kind == "sub" {
        String::new()
    } else {
        format!("  all_ones: bits[{output_width}] = literal(value={all_ones}, id=3)\n")
    };
    format!(
        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
{all_ones_node}  sh: bits[{output_width}] = shll(one, count, id=10)
  ret mask: bits[{output_width}] = {ret_expr}
}}
"
    )
}

fn prep_with_mask_low(f: &ir::Fn) -> ir::Fn {
    prep_for_gatify(
        f,
        None,
        PrepForGatifyOptions {
            enable_rewrite_mask_low: true,
            enable_rewrite_nary_add: false,
            ..PrepForGatifyOptions::default()
        },
    )
}

fn gatify_for_test(pir_fn: &ir::Fn, enable_rewrite_mask_low: bool) -> xlsynth_g8r::aig::GateFn {
    gatify(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low,
            array_index_lowering_strategy: Default::default(),
            unsafe_gatify_gate_operation: false,
        },
    )
    .expect("gatify")
    .gate_fn
}

fn eval_success(f: &ir::Fn, count_width: usize, count: u64) -> IrValue {
    let args = [IrValue::make_ubits(count_width, count).expect("count literal")];
    match eval_fn(f, &args) {
        FnEvalResult::Success(s) => s.value,
        FnEvalResult::Failure(failure) => panic!("unexpected eval failure: {failure:?}"),
    }
}

#[test]
fn prep_rewrites_sub_shift_one_minus_one_to_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_sub_mask_low_ir_text());
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(count"),
        "expected ext_mask_low rewrite; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_add_shift_one_all_ones_to_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_add_mask_low_ir_text());
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(count"),
        "expected ext_mask_low rewrite; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_shifted_all_ones_to_not_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_shifted_all_ones_ir_text(8, 4));
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(count"),
        "expected ext_mask_low rewrite; got:\n{prepared_text}"
    );
    assert!(
        prepared_text.contains("not("),
        "expected not(ext_mask_low) rewrite; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_bf16_sticky_mask_shape_to_not_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_sliced_shifted_all_ones_ir_text(12, 8, 3, 8));
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(count"),
        "expected ext_mask_low rewrite for bf16 sticky-mask shape; got:\n{prepared_text}"
    );
    assert!(
        prepared_text.contains("not("),
        "expected shifted all-ones mask to become not(ext_mask_low); got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_not_shifted_all_ones_to_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_not_shifted_all_ones_ir_text(8, 4));
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(count") && !prepared_text.contains("not("),
        "expected direct ext_mask_low rewrite; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_shifted_shrl_all_ones_to_not_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_shifted_shrl_all_ones_ir_text(8, 4));
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(count") && prepared_text.contains("not("),
        "expected not(ext_mask_low) rewrite; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_return_shrl_all_ones_to_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_shrl_all_ones_ir_text(8, 4));
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_mask_low(") && !prepared_text.contains("shrl("),
        "expected shrl(all_ones, count) to become ext_mask_low over remaining count; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_zero_concat_low_mask_to_wide_ext_mask_low() {
    let pir_fn = parse_top_fn(&build_zero_concat_not_shifted_all_ones_ir_text(4, 8, 3));
    let prepared = prep_with_mask_low(&pir_fn);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("bits[12] = ext_mask_low(count")
            && !prepared_text.contains("concat("),
        "expected zero-extended low mask to become wide ext_mask_low; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);
}

#[test]
fn prep_rewrites_mask_low_idioms_across_width_sweep() {
    for output_width in 1usize..=16 {
        let natural_count_width = ceil_log2(output_width.saturating_add(1));
        for count_width in [0usize, natural_count_width, natural_count_width + 2] {
            let max_count = if count_width == 0 {
                0
            } else {
                (1u64 << count_width) - 1
            };
            for op_kind in ["sub", "add_rhs", "add_lhs"] {
                let pir_fn = parse_top_fn(&build_mask_low_idiom_ir_text(
                    output_width,
                    count_width,
                    op_kind,
                ));
                let prepared = prep_with_mask_low(&pir_fn);
                let prepared_text = prepared.to_string();
                assert!(
                    prepared_text.contains("ext_mask_low(count"),
                    "expected ext_mask_low rewrite for output_width={output_width} count_width={count_width} op_kind={op_kind}; got:\n{prepared_text}"
                );
                for count in 0..=max_count {
                    assert_eq!(
                        eval_success(&prepared, count_width, count),
                        eval_success(&pir_fn, count_width, count),
                        "prepared mismatch for output_width={output_width} count_width={count_width} op_kind={op_kind} count={count}"
                    );
                }
            }
        }
    }
}

#[test]
fn prep_rewrites_shifted_all_ones_across_width_sweep() {
    for source_width in 1usize..=16 {
        let natural_count_width = ceil_log2(source_width.saturating_add(1));
        for count_width in [0usize, natural_count_width, natural_count_width + 2] {
            let max_count = if count_width == 0 {
                0
            } else {
                (1u64 << count_width) - 1
            };
            for start in [0usize, source_width / 2] {
                let width = source_width - start;
                let pir_fn = parse_top_fn(&build_sliced_shifted_all_ones_ir_text(
                    source_width,
                    count_width,
                    start,
                    width,
                ));
                let prepared = prep_with_mask_low(&pir_fn);
                let prepared_text = prepared.to_string();
                assert!(
                    prepared_text.contains("ext_mask_low(count"),
                    "expected shifted all-ones rewrite for source_width={source_width} count_width={count_width} start={start} width={width}; got:\n{prepared_text}"
                );
                assert!(
                    prepared_text.contains("not("),
                    "expected not(ext_mask_low) for source_width={source_width} count_width={count_width} start={start} width={width}; got:\n{prepared_text}"
                );
                for count in 0..=max_count {
                    assert_eq!(
                        eval_success(&prepared, count_width, count),
                        eval_success(&pir_fn, count_width, count),
                        "prepared mismatch for source_width={source_width} count_width={count_width} start={start} width={width} count={count}"
                    );
                }
            }
        }
    }
}

#[test]
fn prep_rewrites_additional_mask_low_families_across_width_sweep() {
    for output_width in 1usize..=16 {
        let natural_count_width = ceil_log2(output_width.saturating_add(1));
        for count_width in [0usize, natural_count_width, natural_count_width + 2] {
            let max_count = if count_width == 0 {
                0
            } else {
                (1u64 << count_width) - 1
            };
            for (family, ir_text) in [
                (
                    "not_shifted_all_ones",
                    build_not_shifted_all_ones_ir_text(output_width, count_width),
                ),
                (
                    "shifted_shrl_all_ones",
                    build_shifted_shrl_all_ones_ir_text(output_width, count_width),
                ),
                (
                    "shrl_all_ones",
                    build_shrl_all_ones_ir_text(output_width, count_width),
                ),
            ] {
                let pir_fn = parse_top_fn(&ir_text);
                let prepared = prep_with_mask_low(&pir_fn);
                let prepared_text = prepared.to_string();
                assert!(
                    prepared_text.contains("ext_mask_low("),
                    "expected ext_mask_low rewrite for output_width={output_width} count_width={count_width} family={family}; got:\n{prepared_text}"
                );
                for count in 0..=max_count {
                    assert_eq!(
                        eval_success(&prepared, count_width, count),
                        eval_success(&pir_fn, count_width, count),
                        "prepared mismatch for output_width={output_width} count_width={count_width} family={family} count={count}"
                    );
                }
            }
        }
    }
}

#[test]
fn prep_rewrites_zero_concat_low_mask_across_width_sweep() {
    for low_width in 1usize..=16 {
        let count_width = bounded_count_width_for_width(low_width);
        let max_count = (1u64 << count_width) - 1;
        for prefix_width in [1usize, 3usize] {
            let pir_fn = parse_top_fn(&build_zero_concat_not_shifted_all_ones_ir_text(
                prefix_width,
                low_width,
                count_width,
            ));
            let prepared = prep_with_mask_low(&pir_fn);
            let prepared_text = prepared.to_string();
            assert!(
                prepared_text.contains("ext_mask_low(") && !prepared_text.contains("concat("),
                "expected zero concat low mask rewrite for prefix_width={prefix_width} low_width={low_width} count_width={count_width}; got:\n{prepared_text}"
            );
            for count in 0..=max_count {
                assert_eq!(
                    eval_success(&prepared, count_width, count),
                    eval_success(&pir_fn, count_width, count),
                    "prepared mismatch for prefix_width={prefix_width} low_width={low_width} count_width={count_width} count={count}"
                );
            }
        }
    }
}

#[test]
fn prep_does_not_rewrite_wrong_literals_or_shift_direction_or_mismatched_widths() {
    for ir_text in [
        r#"package sample

top fn wrong_one(count: bits[4] id=1) -> bits[8] {
  two: bits[8] = literal(value=2, id=2)
  one: bits[8] = literal(value=1, id=3)
  sh: bits[8] = shll(two, count, id=4)
  ret mask: bits[8] = sub(sh, one, id=5)
}
"#,
        r#"package sample

top fn wrong_one_1b(count: bits[1] id=1) -> bits[1] {
  zero: bits[1] = literal(value=0, id=2)
  one: bits[1] = literal(value=1, id=3)
  sh: bits[1] = shll(zero, count, id=4)
  ret mask: bits[1] = sub(sh, one, id=5)
}
"#,
        r#"package sample

top fn wrong_shift(count: bits[4] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  sh: bits[8] = shrl(one, count, id=3)
  ret mask: bits[8] = sub(sh, one, id=4)
}
"#,
        r#"package sample

top fn mismatched_width(count: bits[4] id=1) -> bits[9] {
  one8: bits[8] = literal(value=1, id=2)
  one9: bits[9] = literal(value=1, id=3)
  sh: bits[8] = shll(one8, count, id=4)
  zext: bits[9] = zero_ext(sh, new_bit_count=9, id=5)
  ret mask: bits[9] = sub(zext, one9, id=6)
}
"#,
    ] {
        let pir_fn = parse_top_fn(ir_text);
        let prepared = prep_with_mask_low(&pir_fn);
        let prepared_text = prepared.to_string();
        assert!(
            !prepared_text.contains("ext_mask_low("),
            "unexpected ext_mask_low rewrite; got:\n{prepared_text}"
        );
    }
}

#[test]
fn prep_does_not_rewrite_additional_mask_low_near_misses() {
    for (case_name, ir_text, forbidden) in [
        (
            "shifted_shrl_mismatched_counts",
            r#"package sample

top fn shifted_shrl_mismatched_counts(count0: bits[4] id=1, count1: bits[4] id=2) -> bits[8] {
  all_ones: bits[8] = literal(value=255, id=3)
  low_from_top: bits[8] = shrl(all_ones, count0, id=4)
  ret high: bits[8] = shll(low_from_top, count1, id=5)
}
"#,
            "ext_mask_low(",
        ),
        (
            "embedded_shrl_all_ones",
            r#"package sample

top fn embedded_shrl_all_ones(count: bits[4] id=1) -> bits[8] {
  all_ones: bits[8] = literal(value=255, id=2)
  low_from_top: bits[8] = shrl(all_ones, count, id=3)
  ret out: bits[8] = and(low_from_top, all_ones, id=4)
}
"#,
            "ext_mask_low(",
        ),
        (
            "nonzero_concat_prefix",
            r#"package sample

top fn nonzero_concat_prefix(count: bits[3] id=1) -> bits[12] {
  prefix: bits[4] = literal(value=1, id=2)
  all_ones: bits[8] = literal(value=255, id=3)
  high: bits[8] = shll(all_ones, count, id=4)
  low: bits[8] = not(high, id=5)
  ret out: bits[12] = concat(prefix, low, id=6)
}
"#,
            "ret out: bits[12] = ext_mask_low",
        ),
        (
            "unbounded_concat_count",
            r#"package sample

top fn unbounded_concat_count(count: bits[4] id=1) -> bits[12] {
  prefix: bits[4] = literal(value=0, id=2)
  all_ones: bits[8] = literal(value=255, id=3)
  high: bits[8] = shll(all_ones, count, id=4)
  low: bits[8] = not(high, id=5)
  ret out: bits[12] = concat(prefix, low, id=6)
}
"#,
            "ret out: bits[12] = ext_mask_low",
        ),
    ] {
        let pir_fn = parse_top_fn(ir_text);
        let prepared = prep_with_mask_low(&pir_fn);
        let prepared_text = prepared.to_string();
        assert!(
            !prepared_text.contains(forbidden),
            "unexpected rewrite for {case_name}; got:\n{prepared_text}"
        );
    }
}

#[test]
fn ext_mask_low_rejects_non_bits_count() {
    let ir_text = r#"package sample

top fn non_bits_count(x: bits[4] id=1) -> bits[8] {
  tup: (bits[4]) = tuple(x, id=2)
  ret mask: bits[8] = ext_mask_low(tup, id=3)
}
"#;
    let mut parser = ir_parser::Parser::new(ir_text);
    let err = parser.parse_and_validate_package().unwrap_err();
    assert!(
        err.to_string().contains("ext_mask_low"),
        "expected ext_mask_low validation error, got {err}"
    );
}

#[test]
fn gate_graph_equivalence_old_vs_mask_low_rewrite() {
    for ir_text in [
        build_sub_mask_low_ir_text(),
        build_add_mask_low_ir_text(),
        build_shifted_all_ones_ir_text(8, 4),
        build_sliced_shifted_all_ones_ir_text(12, 8, 3, 8),
        build_not_shifted_all_ones_ir_text(8, 4),
        build_shifted_shrl_all_ones_ir_text(8, 4),
        build_shrl_all_ones_ir_text(8, 4),
        build_zero_concat_not_shifted_all_ones_ir_text(4, 8, 3),
    ] {
        let pir_fn = parse_top_fn(&ir_text);
        let gate_old = gatify_for_test(&pir_fn, /* enable_rewrite_mask_low= */ false);
        let gate_new = gatify_for_test(&pir_fn, /* enable_rewrite_mask_low= */ true);
        check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
            .expect("expected old vs rewritten mask-low lowering to be equivalent");
    }
}

#[test]
fn gate_graph_equivalence_shifted_all_ones_rewrite_sweep() {
    for (source_width, count_width, start, width) in [
        (1usize, 0usize, 0usize, 1usize),
        (1, 3, 0, 1),
        (2, 1, 0, 2),
        (3, 2, 0, 3),
        (3, 5, 1, 2),
        (5, 3, 0, 5),
        (5, 5, 2, 3),
        (7, 3, 0, 7),
        (8, 4, 0, 8),
        (9, 4, 3, 4),
        (12, 8, 3, 8),
        (16, 5, 4, 8),
        (17, 7, 8, 9),
        (31, 6, 15, 16),
        (32, 8, 8, 16),
        (64, 8, 16, 32),
    ] {
        let ir_text =
            build_sliced_shifted_all_ones_ir_text(source_width, count_width, start, width);
        let pir_fn = parse_top_fn(&ir_text);
        let gate_old = gatify_for_test(&pir_fn, /* enable_rewrite_mask_low= */ false);
        let gate_new = gatify_for_test(&pir_fn, /* enable_rewrite_mask_low= */ true);
        check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new).unwrap_or_else(|e| {
            panic!(
                "expected shifted-all-ones rewrite to be equivalent for source_width={source_width} count_width={count_width} start={start} width={width}: {e}"
            )
        });
    }
}
