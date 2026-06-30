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

fn assert_ir_fns_equivalent(orig_fn: &ir::Fn, prepared_fn: &ir::Fn) {
    let mut orig_desugared = orig_fn.clone();
    desugar_extensions_in_fn(&mut orig_desugared).expect("desugar original PIR");
    let mut prepared_desugared = prepared_fn.clone();
    desugar_extensions_in_fn(&mut prepared_desugared).expect("desugar prepared PIR");
    let orig_pkg_text = format!("package orig\n\ntop {}", orig_desugared);
    let prepared_pkg_text = format!("package prepared\n\ntop {}", prepared_desugared);
    check_equivalence::check_equivalence_via_toolchain(&orig_pkg_text, &prepared_pkg_text)
        .expect("prepared PIR should be equivalent to original PIR");
}

fn build_adjusted_clz_shift_ir(offset: usize) -> String {
    format!(
        r#"package sample

top fn adjusted_clz_shift(input: bits[20] id=1) -> bits[20] {{
  reversed: bits[20] = reverse(input, id=2)
  one_hot: bits[21] = one_hot(reversed, lsb_prio=true, id=3)
  clz: bits[5] = encode(one_hot, id=4)
  offset: bits[5] = literal(value={offset}, id=5)
  amount: bits[5] = add(clz, offset, id=6)
  ret shifted: bits[20] = shll(input, amount, id=7)
}}
"#
    )
}

fn prep_adjusted_clz_shift(offset: usize) -> (ir::Fn, ir::Fn) {
    let original = parse_top_fn(&build_adjusted_clz_shift_ir(offset));
    let prepared = prep_for_gatify(
        &original,
        None,
        PrepForGatifyOptions {
            enable_rewrite_prio_encode: true,
            enable_rewrite_normalize_left: true,
            ..PrepForGatifyOptions::default()
        },
    );
    (original, prepared)
}

fn build_zero_extended_adjusted_clz_shift_ir(offset: usize) -> String {
    format!(
        r#"package sample

top fn zero_extended_adjusted_clz_shift(input: bits[20] id=1) -> bits[20] {{
  reversed: bits[20] = reverse(input, id=2)
  one_hot: bits[21] = one_hot(reversed, lsb_prio=true, id=3)
  clz: bits[5] = encode(one_hot, id=4)
  offset: bits[5] = literal(value={offset}, id=5)
  wrapped_amount: bits[5] = add(clz, offset, id=6)
  amount: bits[6] = zero_ext(wrapped_amount, new_bit_count=6, id=7)
  ret shifted: bits[20] = shll(input, amount, id=8)
}}
"#
    )
}

#[test]
fn prep_rewrites_nonwrapping_adjusted_clz_shift() {
    // For a 20-bit input and 5-bit shift amount, 20 + 11 = 31 is the
    // largest adjusted CLZ value that remains representable without wrapping.
    let (original, prepared) = prep_adjusted_clz_shift(11);
    assert!(
        prepared.nodes.iter().any(|node| matches!(
            node.payload,
            ir::NodePayload::ExtNormalizeLeft {
                shift_offset: 11,
                normalized_bit_count: 20,
                ..
            }
        )),
        "expected lossless adjusted CLZ shift to become ext_normalize_left:\n{prepared}"
    );
    assert!(
        !prepared
            .nodes
            .iter()
            .any(|node| matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Shll, _, _))),
        "did not expect generic shll to remain:\n{prepared}"
    );
    assert_ir_fns_equivalent(&original, &prepared);
}

#[test]
fn prep_keeps_wrapping_adjusted_clz_shift() {
    // The 5-bit amount (clz(input) + 29) mod 32 can wrap, so replacing it
    // with an unbounded ext_normalize_left offset would change semantics.
    let (original, prepared) = prep_adjusted_clz_shift(29);
    assert!(
        prepared.nodes.iter().any(|node| matches!(
            node.payload,
            ir::NodePayload::ExtClz {
                offset: 29,
                new_bit_count: 5,
                ..
            }
        )),
        "expected adjusted ext_clz to remain as the shift amount:\n{prepared}"
    );
    assert!(
        prepared
            .nodes
            .iter()
            .any(|node| matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Shll, _, _))),
        "expected generic shll to remain for wrapping shift amount:\n{prepared}"
    );
    assert!(
        !prepared
            .nodes
            .iter()
            .any(|node| matches!(node.payload, ir::NodePayload::ExtNormalizeLeft { .. })),
        "did not expect wrapping shift to become ext_normalize_left:\n{prepared}"
    );
    assert_ir_fns_equivalent(&original, &prepared);
}

#[test]
fn prep_rewrites_zero_extended_nonwrapping_adjusted_clz_shift() {
    let original = parse_top_fn(&build_zero_extended_adjusted_clz_shift_ir(11));
    let prepared = prep_for_gatify(
        &original,
        None,
        PrepForGatifyOptions {
            enable_rewrite_prio_encode: true,
            enable_rewrite_normalize_left: true,
            ..PrepForGatifyOptions::default()
        },
    );
    assert!(
        prepared.nodes.iter().any(|node| matches!(
            node.payload,
            ir::NodePayload::ExtNormalizeLeft {
                shift_offset: 11,
                normalized_bit_count: 20,
                ..
            }
        )),
        "expected zero-extended lossless amount to become ext_normalize_left:\n{prepared}"
    );
    assert!(
        !prepared
            .nodes
            .iter()
            .any(|node| matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Shll, _, _))),
        "did not expect generic shll to remain:\n{prepared}"
    );
    assert_ir_fns_equivalent(&original, &prepared);
}

#[test]
fn prep_keeps_zero_extended_wrapping_adjusted_clz_shift() {
    // Widening the result of the original 5-bit add must not expose the
    // unwrapped value: for clz(input) == 3, (3 + 29) mod 32 is zero.
    let original = parse_top_fn(&build_zero_extended_adjusted_clz_shift_ir(29));
    let prepared = prep_for_gatify(
        &original,
        None,
        PrepForGatifyOptions {
            enable_rewrite_prio_encode: true,
            enable_rewrite_normalize_left: true,
            ..PrepForGatifyOptions::default()
        },
    );
    assert!(
        prepared.nodes.iter().any(|node| matches!(
            node.payload,
            ir::NodePayload::ExtClz {
                offset: 29,
                new_bit_count: 5,
                ..
            }
        )),
        "expected the wrapped 5-bit ext_clz to remain:\n{prepared}"
    );
    assert!(
        prepared.nodes.iter().any(|node| matches!(
            node.payload,
            ir::NodePayload::ZeroExt {
                new_bit_count: 6,
                ..
            }
        )),
        "expected zero-extension after the wrapped amount to remain:\n{prepared}"
    );
    assert!(
        prepared
            .nodes
            .iter()
            .any(|node| matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Shll, _, _))),
        "expected generic shll to remain for the wrapped shift amount:\n{prepared}"
    );
    assert!(
        !prepared
            .nodes
            .iter()
            .any(|node| matches!(node.payload, ir::NodePayload::ExtNormalizeLeft { .. })),
        "did not expect wrapped shift to become ext_normalize_left:\n{prepared}"
    );
    assert_ir_fns_equivalent(&original, &prepared);
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
    check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&gate_ext, &gate_desugared)
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
    check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&gate_ext, &gate_desugared)
        .expect("direct ext_normalize_left lowering should match desugared semantics");
}

#[test]
fn direct_ext_normalize_left_large_shift_offset_matches_desugared_semantics() {
    let pir_fn = parse_top_fn(
        r#"package sample

top fn ext_normalize_left_zeroed(input: bits[4] id=1) -> bits[8] {
  ret normalized: bits[8] = ext_normalize_left(input, shift_offset=8, normalized_bit_count=8, id=2)
}
"#,
    );
    let mut desugared_fn = pir_fn.clone();
    desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_normalize_left");

    let gate_ext = gatify_for_test(&pir_fn);
    let gate_desugared = gatify_for_test(&desugared_fn);
    check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&gate_ext, &gate_desugared)
        .expect("large-offset ext_normalize_left lowering should match desugared semantics");
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
