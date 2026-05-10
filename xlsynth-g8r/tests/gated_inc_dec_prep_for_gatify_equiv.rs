// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GatedIncDecKind {
    Increment,
    Decrement,
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn direct_inc_dec_ir(width: usize, kind: GatedIncDecKind) -> String {
    let op = match kind {
        GatedIncDecKind::Increment => "add",
        GatedIncDecKind::Decrement => "sub",
    };
    format!(
        r#"package sample

top fn f(x: bits[{width}] id=1, en: bits[1] id=2) -> bits[{width}] {{
  en_ext: bits[{width}] = zero_ext(en, new_bit_count={width}, id=3)
  ret out: bits[{width}] = {op}(x, en_ext, id=4)
}}
"#
    )
}

fn helper_inc_dec_factor_name(kind: GatedIncDecKind, index: usize) -> String {
    match kind {
        GatedIncDecKind::Increment => format!("x{index}"),
        GatedIncDecKind::Decrement => format!("nx{index}"),
    }
}

fn prefix_helper_inc_dec_ir(width: usize, kind: GatedIncDecKind) -> String {
    assert!(width > 1);
    let mut ir_text = format!(
        "package sample\n\ntop fn f(x: bits[{width}] id=1, en: bits[1] id=2) -> bits[{width}] {{\n"
    );
    let mut next_id = 3usize;

    for i in 0..width {
        ir_text.push_str(&format!(
            "  x{i}: bits[1] = bit_slice(x, start={i}, width=1, id={next_id})\n"
        ));
        next_id += 1;
    }
    if kind == GatedIncDecKind::Decrement {
        for i in 0..width {
            ir_text.push_str(&format!("  nx{i}: bits[1] = not(x{i}, id={next_id})\n"));
            next_id += 1;
        }
    }

    ir_text.push_str(&format!("  y0: bits[1] = xor(en, x0, id={next_id})\n"));
    next_id += 1;

    let mut previous_carry: Option<String> = None;
    for i in 1..width {
        let carry = format!("c{i}");
        let factor = helper_inc_dec_factor_name(kind, i - 1);
        let lhs = factor;
        let rhs = previous_carry.clone().unwrap_or_else(|| "en".to_string());
        ir_text.push_str(&format!(
            "  {carry}: bits[1] = and({lhs}, {rhs}, id={next_id})\n"
        ));
        next_id += 1;
        previous_carry = Some(carry.clone());

        ir_text.push_str(&format!(
            "  y{i}: bits[1] = xor({carry}, x{i}, id={next_id})\n"
        ));
        next_id += 1;
    }

    let outputs = (0..width)
        .rev()
        .map(|i| format!("y{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    ir_text.push_str(&format!(
        "  ret out: bits[{width}] = concat({outputs}, id={next_id})\n}}\n"
    ));
    ir_text
}

fn gatify_with_nary_rewrite(pir_fn: &ir::Fn) -> (GateFn, SummaryStats) {
    let out = gatify(
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
            enable_rewrite_nary_add: true,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
            unsafe_gatify_gate_operation: false,
        },
    )
    .expect("gatify");
    let stats = get_summary_stats(&out.gate_fn);
    (out.gate_fn, stats)
}

fn assert_direct_and_helper_match(kind: GatedIncDecKind) {
    let width = 8usize;
    let direct_fn = parse_top_fn(&direct_inc_dec_ir(width, kind));
    let helper_fn = parse_top_fn(&prefix_helper_inc_dec_ir(width, kind));

    let (direct_gate, direct_stats) = gatify_with_nary_rewrite(&direct_fn);
    let (helper_gate, helper_stats) = gatify_with_nary_rewrite(&helper_fn);

    assert_eq!(
        direct_stats, helper_stats,
        "expected direct and helper {kind:?} forms to have identical gate stats"
    );
    check_equivalence::prove_same_gate_fn_via_ir(&direct_gate, &helper_gate)
        .expect("direct and helper gates should be equivalent");
    check_equivalence::validate_same_fn(&direct_fn, &helper_gate)
        .expect("helper gate should match direct PIR");
}

#[test]
fn gated_increment_direct_and_helper_forms_have_same_qor() {
    assert_direct_and_helper_match(GatedIncDecKind::Increment);
}

#[test]
fn gated_decrement_direct_and_helper_forms_have_same_qor() {
    assert_direct_and_helper_match(GatedIncDecKind::Decrement);
}
