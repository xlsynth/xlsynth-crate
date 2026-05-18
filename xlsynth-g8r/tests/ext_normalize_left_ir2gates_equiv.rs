// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn gatify_for_test(pir_fn: &ir::Fn) -> xlsynth_g8r::aig::GateFn {
    gatify(pir_fn, GatifyOptions::all_opts_disabled())
        .expect("gatify")
        .gate_fn
}

#[test]
fn direct_ext_normalize_left_matches_desugared_semantics() {
    let pir_fn = parse_top_fn(
        r#"package sample

top fn ext_normalize_left_4b(input: bits[4] id=1) -> (bits[8], bits[3]) {
  ret normalized: (bits[8], bits[3]) = ext_normalize_left(input, shift_offset=1, normalized_bit_count=8, clz_bit_count=3, id=2)
}
"#,
    );
    let mut desugared_fn = pir_fn.clone();
    desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_normalize_left");

    let gate_ext = gatify_for_test(&pir_fn);
    let gate_desugared = gatify_for_test(&desugared_fn);
    check_equivalence::prove_same_gate_fn_via_ir(&gate_ext, &gate_desugared)
        .expect("direct ext_normalize_left lowering should match desugared semantics");
}

#[test]
fn direct_ext_normalize_left_non_power_of_two_matches_desugared_semantics() {
    let pir_fn = parse_top_fn(
        r#"package sample

top fn ext_normalize_left_5b(input: bits[5] id=1) -> (bits[7], bits[4]) {
  ret normalized: (bits[7], bits[4]) = ext_normalize_left(input, shift_offset=2, normalized_bit_count=7, clz_bit_count=4, id=2)
}
"#,
    );
    let mut desugared_fn = pir_fn.clone();
    desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_normalize_left");

    let gate_ext = gatify_for_test(&pir_fn);
    let gate_desugared = gatify_for_test(&desugared_fn);
    check_equivalence::prove_same_gate_fn_via_ir(&gate_ext, &gate_desugared)
        .expect("direct ext_normalize_left lowering should match desugared semantics");
}

#[test]
fn prep_rewrites_normalize_shift_and_shares_widened_raw_clz() {
    let pir_fn = parse_top_fn(
        r#"package sample

top fn normalize_with_wide_clz(x: bits[7] id=1) -> (bits[8], bits[8]) {
  clz: bits[3] = ext_clz(x, offset=0, new_bit_count=3, id=2)
  x_zero: bits[1] = literal(value=0, id=3)
  x_ext: bits[8] = concat(x_zero, x, id=4)
  shifted: bits[8] = shll(x_ext, clz, id=5)
  clz_zero: bits[5] = literal(value=0, id=6)
  wide_clz: bits[8] = concat(clz_zero, clz, id=7)
  ret out: (bits[8], bits[8]) = tuple(shifted, wide_clz, id=8)
}
"#,
    );
    let prepared = prep_for_gatify(
        &pir_fn,
        None,
        PrepForGatifyOptions {
            enable_rewrite_prio_encode: true,
            enable_rewrite_normalize_left: true,
            ..PrepForGatifyOptions::default()
        },
    );
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains(
            "ext_normalize_left(x, shift_offset=0, normalized_bit_count=8, clz_bit_count=3"
        ),
        "expected normalize rewrite to keep the narrow raw clz output, got:\n{prepared_text}"
    );
    assert!(
        !prepared_text.contains("shll("),
        "did not expect generic shll to remain, got:\n{prepared_text}"
    );
    assert!(
        prepared_text.contains("wide_clz: bits[8] = zero_ext(tuple_index."),
        "expected widened raw clz consumer to zero-extend the shared narrow clz, got:\n{prepared_text}"
    );
}
