// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn assert_ir_fns_equivalent(orig_fn: &ir::Fn, prepared_fn: &ir::Fn) {
    let orig_pkg_text = format!("package orig\n\ntop {}", orig_fn);
    let prepared_pkg_text = format!("package prepared\n\ntop {}", prepared_fn);
    check_equivalence::check_equivalence(&orig_pkg_text, &prepared_pkg_text)
        .expect("prepared PIR should be equivalent to original PIR");
}

fn gatify_without_prep(pir_fn: &ir::Fn) -> (GateFn, SummaryStats) {
    let out = gatify_prepared_fn(
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
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify_prepared_fn");
    let stats = get_summary_stats(&out.gate_fn);
    (out.gate_fn, stats)
}

fn prep_for_test(pir_fn: &ir::Fn, enable_small_shift_choices: bool) -> ir::Fn {
    prep_for_gatify(
        pir_fn,
        None,
        PrepForGatifyOptions {
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_small_shift_choices: enable_small_shift_choices,
        },
    )
}

#[test]
fn sel_shift_choice_rewrite_is_equivalent_and_reduces_qor_cost() {
    let ir_text = "package sample

top fn shift_sel(data_hi: bits[9] id=1, data_lo: bits[3] id=2, pred: bits[1] id=3) -> bits[7] {
  three: bits[3] = literal(value=3, id=4)
  four: bits[3] = literal(value=4, id=5)
  amt: bits[3] = sel(pred, cases=[three, four], id=6)
  cat: bits[12] = concat(data_hi, data_lo, id=7)
  shifted: bits[12] = shrl(cat, amt, id=8)
  ret out: bits[7] = bit_slice(shifted, start=0, width=7, id=9)
}
";
    let pir_fn = parse_top_fn(ir_text);
    let prepared = prep_for_test(&pir_fn, /* enable_small_shift_choices= */ true);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ret sel.")
            && prepared_text.contains("bit_slice(cat, start=3")
            && prepared_text.contains("bit_slice(cat, start=4")
            && !prepared_text.contains("shrl("),
        "expected select-like bit-slice rewrite, got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);

    let (gate_old, stats_old) = gatify_without_prep(&pir_fn);
    let (gate_new, stats_new) = gatify_without_prep(&prepared);

    check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
        .expect("old vs rewritten shift_sel lowerings should be equivalent");
    check_equivalence::validate_same_fn(&pir_fn, &gate_new)
        .expect("rewritten shift_sel lowering should match source PIR");

    assert_eq!(
        stats_old.live_nodes, 99,
        "unexpected old shift_sel live_nodes"
    );
    assert_eq!(stats_old.deepest_path, 7, "unexpected old shift_sel depth");
    assert_eq!(
        stats_new.live_nodes, 30,
        "unexpected new shift_sel live_nodes"
    );
    assert_eq!(stats_new.deepest_path, 3, "unexpected new shift_sel depth");
    let old_product = stats_old.live_nodes * stats_old.deepest_path;
    let new_product = stats_new.live_nodes * stats_new.deepest_path;
    assert!(
        stats_new.live_nodes < stats_old.live_nodes && new_product < old_product,
        "expected shift_sel rewrite to reduce QoR: old={stats_old:?} new={stats_new:?}"
    );
}

#[test]
fn priority_sel_shift_choice_rewrite_is_equivalent_and_reduces_qor_cost() {
    let ir_text = "package sample

top fn shift_priority(data_hi: bits[8] id=1, data_lo: bits[4] id=2, p0: bits[1] id=3, p1: bits[1] id=4) -> bits[6] {
  selector: bits[2] = concat(p1, p0, id=5)
  two: bits[4] = literal(value=2, id=6)
  five: bits[4] = literal(value=5, id=7)
  seven: bits[4] = literal(value=7, id=8)
  amt: bits[4] = priority_sel(selector, cases=[two, five], default=seven, id=9)
  cat: bits[12] = concat(data_hi, data_lo, id=10)
  shifted: bits[12] = shrl(cat, amt, id=11)
  ret out: bits[6] = bit_slice(shifted, start=1, width=6, id=12)
}
";
    let pir_fn = parse_top_fn(ir_text);
    let prepared = prep_for_test(&pir_fn, /* enable_small_shift_choices= */ true);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("priority_sel(") && !prepared_text.contains("shrl("),
        "expected priority-select shift rewrite, got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);

    let (gate_old, stats_old) = gatify_without_prep(&pir_fn);
    let (gate_new, stats_new) = gatify_without_prep(&prepared);

    check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
        .expect("old vs rewritten shift_priority lowerings should be equivalent");
    check_equivalence::validate_same_fn(&pir_fn, &gate_new)
        .expect("rewritten shift_priority lowering should match source PIR");

    assert_eq!(
        stats_old.live_nodes, 94,
        "unexpected old shift_priority live_nodes"
    );
    assert_eq!(
        stats_old.deepest_path, 10,
        "unexpected old shift_priority depth"
    );
    assert_eq!(
        stats_new.live_nodes, 39,
        "unexpected new shift_priority live_nodes"
    );
    assert_eq!(
        stats_new.deepest_path, 6,
        "unexpected new shift_priority depth"
    );
    let old_product = stats_old.live_nodes * stats_old.deepest_path;
    let new_product = stats_new.live_nodes * stats_new.deepest_path;
    assert!(
        stats_new.live_nodes < stats_old.live_nodes && new_product < old_product,
        "expected shift_priority rewrite to reduce QoR: old={stats_old:?} new={stats_new:?}"
    );
}

#[test]
fn add_xor_and_rewrite_is_equivalent_and_reduces_qor_cost() {
    let ir_text = "package sample

top fn add_xor_and(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  xor.3: bits[8] = xor(a, b, id=3)
  and.4: bits[8] = and(a, b, id=4)
  ret add.5: bits[8] = add(xor.3, and.4, id=5)
}
";
    let pir_fn = parse_top_fn(ir_text);
    let prepared = prep_for_test(&pir_fn, /* enable_small_shift_choices= */ false);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("or(a, b, id=5)") && !prepared_text.contains("add("),
        "expected add(xor, and) rewrite to or(a, b), got:\n{prepared_text}"
    );
    assert_ir_fns_equivalent(&pir_fn, &prepared);

    let (gate_old, stats_old) = gatify_without_prep(&pir_fn);
    let (gate_new, stats_new) = gatify_without_prep(&prepared);

    check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
        .expect("old vs rewritten add_xor_and lowerings should be equivalent");
    check_equivalence::validate_same_fn(&pir_fn, &gate_new)
        .expect("rewritten add_xor_and lowering should match source PIR");

    assert_eq!(
        stats_old.live_nodes, 118,
        "unexpected old add_xor_and live_nodes"
    );
    assert_eq!(
        stats_old.deepest_path, 15,
        "unexpected old add_xor_and depth"
    );
    assert_eq!(
        stats_new.live_nodes, 24,
        "unexpected new add_xor_and live_nodes"
    );
    assert_eq!(
        stats_new.deepest_path, 2,
        "unexpected new add_xor_and depth"
    );
    let old_product = stats_old.live_nodes * stats_old.deepest_path;
    let new_product = stats_new.live_nodes * stats_new.deepest_path;
    assert!(
        stats_new.live_nodes < stats_old.live_nodes && new_product < old_product,
        "expected add(xor, and) rewrite to reduce QoR: old={stats_old:?} new={stats_new:?}"
    );
}
