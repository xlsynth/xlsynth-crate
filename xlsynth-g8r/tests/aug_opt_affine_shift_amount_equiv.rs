// SPDX-License-Identifier: Apache-2.0

//! Equivalence sweep for aug-opt's "affine shift amount" rewrite.
//!
//! In this test, "affine" means the unsigned natural-number shift amount is
//! `K + 1*flag`, where `flag: bits[1]` is either used directly as a one-bit
//! amount or zero-extended into the amount type. The shift amount is therefore
//! exactly one of two constants, `K` or `(K + 1) mod 2^amount_w`, and aug-opt
//! can expose that as a `sel(flag, ...)` around constant shifts. The wrapping
//! case is included because a one-bit increment can only wrap to zero.

use std::collections::BTreeSet;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::math::ceil_log2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ShiftOp {
    Shll,
    Shrl,
    Shra,
}

impl ShiftOp {
    fn as_str(self) -> &'static str {
        match self {
            ShiftOp::Shll => "shll",
            ShiftOp::Shrl => "shrl",
            ShiftOp::Shra => "shra",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FlagExtKind {
    Direct,
    ZeroExt,
    SingleZeroConcat,
    MultiZeroConcat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ConsumerKind {
    WholeShift,
    Identity,
    FullSlice,
    LowSlice,
    MiddleSlice,
    HighSlice,
    ZeroWidthSlice,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AffineShiftCase {
    op: ShiftOp,
    source_width: usize,
    amount_width: usize,
    k: usize,
    ext_kind: FlagExtKind,
    consumer: ConsumerKind,
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn aug_opt_for_test(ir_text: &str) -> ir::Fn {
    let result = run_aug_opt_over_ir_text_with_stats(
        ir_text,
        Some("affine_shift"),
        AugOptOptions {
            enable: true,
            rounds: 1,
            mode: AugOptMode::PirOnly,
        },
    )
    .expect("run aug-opt");
    parse_top_fn(&result.output_text)
}

fn gatify_without_prep(pir_fn: &ir::Fn) -> GateFn {
    gatify_prepared_fn(
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
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
            unsafe_gatify_gate_operation: false,
        },
    )
    .expect("gatify_prepared_fn")
    .gate_fn
}

fn assert_ir_fns_equivalent(orig_fn: &ir::Fn, prepared_fn: &ir::Fn, label: &str) {
    let orig_pkg_text = format!("package orig\n\ntop {}", orig_fn);
    let prepared_pkg_text = format!("package prepared\n\ntop {}", prepared_fn);
    check_equivalence::check_equivalence(&orig_pkg_text, &prepared_pkg_text)
        .unwrap_or_else(|e| panic!("PIR equivalence failed for {label}: {e}"));
}

fn host_usize_value_fits_in_bits_width(value: usize, width: usize) -> bool {
    if width == 0 {
        return value == 0;
    }
    u32::try_from(width)
        .ok()
        .and_then(|w| 1usize.checked_shl(w))
        .map(|limit| value < limit)
        .unwrap_or(true)
}

fn legal_ext_kinds(amount_width: usize) -> Vec<FlagExtKind> {
    let mut kinds = vec![FlagExtKind::ZeroExt];
    if amount_width == 1 {
        kinds.push(FlagExtKind::Direct);
    }
    if amount_width >= 2 {
        kinds.push(FlagExtKind::SingleZeroConcat);
    }
    if amount_width >= 3 {
        kinds.push(FlagExtKind::MultiZeroConcat);
    }
    kinds
}

fn amount_widths(source_width: usize) -> Vec<usize> {
    let natural = ceil_log2(source_width.saturating_add(1)).max(1);
    let mut widths = BTreeSet::new();
    widths.insert(1);
    widths.insert(natural);
    widths.insert(natural + 2);
    widths.into_iter().collect()
}

fn constants(source_width: usize, amount_width: usize) -> Vec<usize> {
    let mut values = BTreeSet::new();
    for value in [
        0,
        1,
        source_width / 2,
        source_width.saturating_sub(1),
        source_width,
    ] {
        if value
            .checked_add(1)
            .is_some_and(|v| host_usize_value_fits_in_bits_width(v, amount_width))
        {
            values.insert(value);
        }
    }
    if amount_width < usize::BITS as usize {
        let max_non_wrapping = (1usize << amount_width).saturating_sub(2);
        values.insert(max_non_wrapping);
        let max_wrapping = (1usize << amount_width).saturating_sub(1);
        values.insert(max_wrapping);
    }
    values.into_iter().collect()
}

fn consumer_start_width(consumer: ConsumerKind, source_width: usize) -> (usize, usize) {
    match consumer {
        ConsumerKind::WholeShift | ConsumerKind::Identity | ConsumerKind::FullSlice => {
            (0, source_width)
        }
        ConsumerKind::LowSlice => (0, source_width.div_ceil(2)),
        ConsumerKind::MiddleSlice => {
            let start = source_width / 3;
            let width = (source_width - start).div_ceil(2);
            (start, width)
        }
        ConsumerKind::HighSlice => (source_width - 1, 1),
        ConsumerKind::ZeroWidthSlice => (0, 0),
    }
}

fn build_affine_shift_ir(case: &AffineShiftCase) -> String {
    let op = case.op.as_str();
    let source_width = case.source_width;
    let amount_width = case.amount_width;
    let (slice_start, out_width) = consumer_start_width(case.consumer, source_width);

    let mut lines = Vec::new();
    lines.push("package sample".to_string());
    lines.push(String::new());
    lines.push(format!(
        "top fn affine_shift(x: bits[{source_width}] id=1, flag: bits[1] id=2) -> bits[{out_width}] {{"
    ));

    let mut next_id = 3usize;
    let amount_operand = match case.ext_kind {
        FlagExtKind::Direct => {
            assert_eq!(amount_width, 1);
            "flag".to_string()
        }
        FlagExtKind::ZeroExt => {
            lines.push(format!(
                "  flag_ext: bits[{amount_width}] = zero_ext(flag, new_bit_count={amount_width}, id={next_id})"
            ));
            next_id += 1;
            "flag_ext".to_string()
        }
        FlagExtKind::SingleZeroConcat => {
            assert!(amount_width >= 2);
            let high_width = amount_width - 1;
            lines.push(format!(
                "  zero_hi: bits[{high_width}] = literal(value=0, id={next_id})"
            ));
            next_id += 1;
            lines.push(format!(
                "  flag_ext: bits[{amount_width}] = concat(zero_hi, flag, id={next_id})"
            ));
            next_id += 1;
            "flag_ext".to_string()
        }
        FlagExtKind::MultiZeroConcat => {
            assert!(amount_width >= 3);
            let high_width = amount_width - 2;
            lines.push(format!(
                "  zero_hi: bits[{high_width}] = literal(value=0, id={next_id})"
            ));
            next_id += 1;
            lines.push(format!(
                "  zero_mid: bits[1] = literal(value=0, id={next_id})"
            ));
            next_id += 1;
            lines.push(format!(
                "  flag_ext: bits[{amount_width}] = concat(zero_hi, zero_mid, flag, id={next_id})"
            ));
            next_id += 1;
            "flag_ext".to_string()
        }
    };

    lines.push(format!(
        "  k: bits[{amount_width}] = literal(value={}, id={next_id})",
        case.k
    ));
    next_id += 1;
    lines.push(format!(
        "  amount: bits[{amount_width}] = add({amount_operand}, k, id={next_id})"
    ));
    next_id += 1;
    match case.consumer {
        ConsumerKind::WholeShift => {
            lines.push(format!(
                "  ret shifted: bits[{source_width}] = {op}(x, amount, id={next_id})"
            ));
        }
        ConsumerKind::Identity => {
            lines.push(format!(
                "  shifted: bits[{source_width}] = {op}(x, amount, id={next_id})"
            ));
            next_id += 1;
            lines.push(format!(
                "  ret out: bits[{source_width}] = identity(shifted, id={next_id})"
            ));
        }
        ConsumerKind::FullSlice
        | ConsumerKind::LowSlice
        | ConsumerKind::MiddleSlice
        | ConsumerKind::HighSlice
        | ConsumerKind::ZeroWidthSlice => {
            lines.push(format!(
                "  shifted: bits[{source_width}] = {op}(x, amount, id={next_id})"
            ));
            next_id += 1;
            lines.push(format!(
                "  ret out: bits[{out_width}] = bit_slice(shifted, start={slice_start}, width={out_width}, id={next_id})"
            ));
        }
    }
    lines.push("}".to_string());
    lines.push(String::new());
    lines.join("\n")
}

fn label(case: &AffineShiftCase) -> String {
    format!(
        "op={} source_width={} amount_width={} k={} ext_kind={:?} consumer={:?}",
        case.op.as_str(),
        case.source_width,
        case.amount_width,
        case.k,
        case.ext_kind,
        case.consumer
    )
}

fn assert_rewrites_and_proves(case: &AffineShiftCase, prove_gates: bool) {
    let label = label(case);
    let ir_text = build_affine_shift_ir(case);
    let pir_fn = parse_top_fn(&ir_text);
    let prepared = aug_opt_for_test(&ir_text);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("sel(flag"),
        "expected affine shift rewrite for {label}; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared, &label);

    if prove_gates {
        let gate_old = gatify_without_prep(&pir_fn);
        let gate_new = gatify_without_prep(&prepared);
        check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
            .unwrap_or_else(|e| panic!("gate equivalence failed for {label}: {e}"));
        check_equivalence::validate_same_fn(&pir_fn, &gate_new)
            .unwrap_or_else(|e| panic!("rewritten gate validation failed for {label}: {e}"));
    }
}

fn width_constant_cases() -> Vec<AffineShiftCase> {
    let mut cases = Vec::new();
    for op in [ShiftOp::Shrl, ShiftOp::Shll, ShiftOp::Shra] {
        for source_width in [1usize, 2, 3, 4, 5, 8, 13, 16] {
            for amount_width in amount_widths(source_width) {
                for k in constants(source_width, amount_width) {
                    let ext_kind = if amount_width == 1 {
                        FlagExtKind::Direct
                    } else {
                        FlagExtKind::SingleZeroConcat
                    };
                    let consumer = match (source_width + amount_width + k) % 4 {
                        0 => ConsumerKind::FullSlice,
                        1 => ConsumerKind::LowSlice,
                        2 => ConsumerKind::MiddleSlice,
                        _ => ConsumerKind::HighSlice,
                    };
                    cases.push(AffineShiftCase {
                        op,
                        source_width,
                        amount_width,
                        k,
                        ext_kind,
                        consumer,
                    });
                }
            }
        }
    }
    cases
}

fn extension_consumer_cases() -> Vec<AffineShiftCase> {
    let mut cases = Vec::new();
    for op in [ShiftOp::Shrl, ShiftOp::Shll, ShiftOp::Shra] {
        for ext_kind in legal_ext_kinds(4) {
            for consumer in [
                ConsumerKind::WholeShift,
                ConsumerKind::Identity,
                ConsumerKind::FullSlice,
                ConsumerKind::LowSlice,
                ConsumerKind::MiddleSlice,
                ConsumerKind::HighSlice,
            ] {
                cases.push(AffineShiftCase {
                    op,
                    source_width: 8,
                    amount_width: 4,
                    k: 2,
                    ext_kind,
                    consumer,
                });
            }
        }
    }
    cases
}

fn should_gate_prove_width_constant(case: &AffineShiftCase) -> bool {
    matches!(case.source_width, 1 | 2 | 4 | 8 | 13)
        && matches!(case.k, 0 | 1)
        && matches!(
            case.consumer,
            ConsumerKind::FullSlice | ConsumerKind::HighSlice
        )
}

#[test]
fn affine_shift_rewrite_width_and_constant_proof_sweep() {
    let mut seen = BTreeSet::new();
    for case in width_constant_cases() {
        if !seen.insert(case.clone()) {
            continue;
        }
        let prove_gates = should_gate_prove_width_constant(&case);
        assert_rewrites_and_proves(&case, prove_gates);
    }
}

#[test]
fn affine_shift_rewrite_extension_and_consumer_proof_sweep() {
    for case in extension_consumer_cases() {
        assert_rewrites_and_proves(&case, true);
    }
}

fn assert_does_not_rewrite(ir_text: &str, label: &str) {
    let pir_fn = parse_top_fn(ir_text);
    let result = run_aug_opt_over_ir_text_with_stats(
        ir_text,
        Some(&pir_fn.name),
        AugOptOptions {
            enable: true,
            rounds: 1,
            mode: AugOptMode::PirOnly,
        },
    )
    .expect("run aug-opt");
    let prepared = parse_top_fn(&result.output_text);
    let prepared_text = prepared.to_string();
    assert!(
        !prepared_text.contains("sel(flag"),
        "unexpected affine shift rewrite for {label}; got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared, label);
}

#[test]
fn affine_shift_rewrite_handles_wraparound_amount() {
    let case = AffineShiftCase {
        op: ShiftOp::Shrl,
        source_width: 8,
        amount_width: 2,
        k: 3,
        ext_kind: FlagExtKind::ZeroExt,
        consumer: ConsumerKind::WholeShift,
    };
    assert_rewrites_and_proves(&case, true);
}

#[test]
fn affine_shift_rewrite_handles_zero_width_bit_slice() {
    let case = AffineShiftCase {
        op: ShiftOp::Shra,
        source_width: 8,
        amount_width: 4,
        k: 3,
        ext_kind: FlagExtKind::SingleZeroConcat,
        consumer: ConsumerKind::ZeroWidthSlice,
    };
    assert_rewrites_and_proves(&case, false);
}

#[test]
fn affine_shift_rewrite_rejects_non_one_bit_flag() {
    assert_does_not_rewrite(
        r#"package sample

top fn flag_too_wide(flag: bits[2] id=1, x: bits[8] id=2) -> bits[8] {
  flag_ext: bits[3] = zero_ext(flag, new_bit_count=3, id=3)
  k: bits[3] = literal(value=1, id=4)
  amount: bits[3] = add(flag_ext, k, id=5)
  ret shifted: bits[8] = shll(x, amount, id=6)
}
"#,
        "non-one-bit flag",
    );
}

#[test]
fn affine_shift_rewrite_rejects_nonzero_concat_high_bits() {
    assert_does_not_rewrite(
        r#"package sample

top fn nonzero_concat(flag: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  one_hi: bits[2] = literal(value=1, id=3)
  flag_ext: bits[3] = concat(one_hi, flag, id=4)
  k: bits[3] = literal(value=1, id=5)
  amount: bits[3] = add(flag_ext, k, id=6)
  ret shifted: bits[8] = shra(x, amount, id=7)
}
"#,
        "nonzero concat high bits",
    );
}

#[test]
fn affine_shift_rewrite_handles_shared_non_return_cone() {
    let ir_text = r#"package sample

top fn affine_shift(flag: bits[1] id=1, x: bits[8] id=2) -> (bits[8], bits[8]) {
  flag_ext: bits[3] = zero_ext(flag, new_bit_count=3, id=3)
  k: bits[3] = literal(value=1, id=4)
  amount: bits[3] = add(flag_ext, k, id=5)
  shifted: bits[8] = shrl(x, amount, id=6)
  id_shifted: bits[8] = identity(shifted, id=7)
  ret pair: (bits[8], bits[8]) = tuple(shifted, id_shifted, id=8)
}
"#;
    let pir_fn = parse_top_fn(ir_text);
    let prepared = aug_opt_for_test(ir_text);
    assert!(
        prepared.to_string().contains("sel(flag"),
        "expected affine shift rewrite for shared cone; got:\n{prepared}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared, "shared non-return cone");
}

#[test]
fn affine_shift_rewrite_rejects_mismatched_extension_width() {
    assert_does_not_rewrite(
        r#"package sample

top fn mismatched_width(flag: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  flag_ext: bits[2] = zero_ext(flag, new_bit_count=2, id=3)
  flag_ext_wide: bits[3] = zero_ext(flag_ext, new_bit_count=3, id=4)
  k: bits[3] = literal(value=1, id=5)
  amount: bits[3] = add(flag_ext_wide, k, id=6)
  ret shifted: bits[8] = shrl(x, amount, id=7)
}
"#,
        "mismatched extension width",
    );
}
