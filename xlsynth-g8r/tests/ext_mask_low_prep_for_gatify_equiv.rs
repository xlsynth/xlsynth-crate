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

fn all_ones_value(width: usize) -> u128 {
    if width == 0 { 0 } else { (1u128 << width) - 1 }
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
    for ir_text in [build_sub_mask_low_ir_text(), build_add_mask_low_ir_text()] {
        let pir_fn = parse_top_fn(&ir_text);
        let gate_old = gatify_for_test(&pir_fn, /* enable_rewrite_mask_low= */ false);
        let gate_new = gatify_for_test(&pir_fn, /* enable_rewrite_mask_low= */ true);
        check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
            .expect("expected old vs rewritten mask-low lowering to be equivalent");
    }
}
