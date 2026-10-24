// SPDX-License-Identifier: Apache-2.0

//! Targeted, non-vacuous coverage of projected positive-mask preparation.
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir::{Binop, NodePayload};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult, ProverFn};
use xlsynth_prover::solver::bitwuzla::Bitwuzla;

pub const KINDS: &[&str] = &[
    "match",
    "shared_and",
    "shared_sle",
    "shared_not",
    "shared_mask",
    "one",
    "minus_one",
    "other_source",
    "reverse_compare",
    "slt",
    "ule",
    "no_not",
    "zero_ext",
    "low_start",
    "source_width",
    "projection_start",
    "projection_width",
    "three_operands",
    "or",
];

/// All cases are valid IR. Width three is the minimum needed by the negative
/// slice shapes; width two is additionally exercised for matching/sharing
/// cases.
pub fn check_case(width: usize, kind: usize, reversed: bool, expression: bool) {
    assert!(width >= 1);
    let kind = KINDS[kind];
    let low_width = if kind == "source_width" {
        width
    } else {
        width - 1
    };
    let source = if expression {
        "xor(x, y, id=3)"
    } else {
        "identity(x, id=3)"
    };
    let constant = match kind {
        "one" => "1".to_string(),
        "minus_one" => format!("0b{}", "1".repeat(width)),
        _ => "0".to_string(),
    };
    let compare_source = if kind == "other_source" { "y" } else { "src" };
    let (lhs, rhs) = if kind == "reverse_compare" {
        ("zero", "src")
    } else {
        (compare_source, "zero")
    };
    let compare = match kind {
        "slt" => "slt",
        "ule" => "ule",
        _ => "sle",
    };
    let not = if kind == "no_not" { "identity" } else { "not" };
    let ext = if kind == "zero_ext" {
        "zero_ext"
    } else {
        "sign_ext"
    };
    let low_start = usize::from(kind == "low_start");
    let projection_width = if kind == "projection_width" { 2 } else { 1 };
    let projection_start = if kind == "projection_start" {
        0
    } else {
        low_width - projection_width
    };
    let operands = if reversed { "mask, low" } else { "low, mask" };
    let operands = if kind == "three_operands" {
        format!("{operands}, low")
    } else {
        operands.to_string()
    };
    let and = if kind == "or" { "or" } else { "and" };
    let (extra, extra_width) = match kind {
        "shared_sle" => ("cmp", 1),
        "shared_not" => ("positive", 1),
        "shared_mask" => ("mask", low_width),
        "shared_and" => ("masked", low_width),
        _ => ("out", projection_width),
    };
    let text = format!("package sample
top fn main(x: bits[{width}] id=1, y: bits[{width}] id=2) -> (bits[{projection_width}], bits[{extra_width}]) {{
 src: bits[{width}] = {source}
 zero: bits[{width}] = literal(value={constant}, id=4)
 cmp: bits[1] = {compare}({lhs}, {rhs}, id=5)
 positive: bits[1] = {not}(cmp, id=6)
 mask: bits[{low_width}] = {ext}(positive, new_bit_count={low_width}, id=7)
 low: bits[{low_width}] = bit_slice(src, start={low_start}, width={low_width}, id=8)
 masked: bits[{low_width}] = {and}({operands}, id=9)
 out: bits[{projection_width}] = bit_slice(masked, start={projection_start}, width={projection_width}, id=10)
 ret result: (bits[{projection_width}], bits[{extra_width}]) = tuple(out, {extra}, id=11)
}}");
    let pkg = Parser::new(&text)
        .parse_and_validate_package()
        .expect("valid generated IR");
    let original = pkg.get_top_fn().unwrap();
    let prepared = prep_for_gatify(original, None, PrepForGatifyOptions::all_opts_disabled());
    // Check payloads, not rendered spelling or node indices (compaction changes
    // indices). The comparison's stable text ID identifies the mutated node.
    let cmp = prepared
        .nodes
        .iter()
        .find(|n| n.text_id == 5)
        .expect("live comparison");
    let should_fire = matches!(kind, "match" | "shared_and");
    assert_eq!(
        matches!(cmp.payload, NodePayload::BitSlice { start, width: 1, .. } if start == width - 1),
        should_fire,
        "fire/decline: {kind}, width={width}, reversed={reversed}, expression={expression}\n{prepared}"
    );
    if !should_fire {
        assert!(
            matches!(
                cmp.payload,
                NodePayload::Binop(Binop::Sle | Binop::Slt | Binop::Ule, _, _)
            ),
            "declining must preserve the comparison: {kind}"
        );
    }
    let result = prove_ir_fn_equiv::<Bitwuzla>(
        &crate::fuzz_bitwuzla_options(),
        &ProverFn::new(original, None),
        &ProverFn::new(&prepared, None),
        AssertionSemantics::Same,
        None,
        false,
    );
    assert!(
        matches!(result, EquivResult::Proved),
        "{kind} width={width}: {result:?}\n{text}"
    );
}

/// A deterministic matrix covers every category independently of fuzz
/// discovery. Solver proofs quantify all inputs, including zero and signed
/// extrema.
pub fn coverage_audit() -> usize {
    let mut proofs = 0;
    for width in [3, 4, 8, 16, 32, 64, 65, 129] {
        for kind in 0..KINDS.len() {
            for reversed in [false, true] {
                for expression in [false, true] {
                    check_case(width, kind, reversed, expression);
                    proofs += 1;
                }
            }
        }
    }
    for name in [
        "match",
        "shared_and",
        "shared_sle",
        "shared_not",
        "shared_mask",
    ] {
        let kind = KINDS.iter().position(|k| *k == name).unwrap();
        for reversed in [false, true] {
            for expression in [false, true] {
                check_case(2, kind, reversed, expression);
                proofs += 1;
            }
        }
    }
    for reversed in [false, true] {
        for expression in [false, true] {
            let kind = KINDS.iter().position(|k| *k == "source_width").unwrap();
            check_case(1, kind, reversed, expression);
            proofs += 1;
        }
    }
    proofs
}
