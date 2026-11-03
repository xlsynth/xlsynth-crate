// SPDX-License-Identifier: Apache-2.0

use test_case::test_matrix;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, PreparedGateSim};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_node_as_fn, gatify_prepared_fn};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

#[test_matrix([1, 2, 3, 9, 48, 65], ["shrl", "shll"], ["or_reduce", "eq", "ne"])]
fn shift_predicates_preserve_all_amounts(width: usize, direction: &str, predicate: &str) {
    for amount_width in [1, ceil_log2(width) + 3] {
        let ir = predicate_ir(
            width,
            amount_width,
            direction,
            predicate,
            None,
            "predicate",
            false,
        );
        lower_and_prove(&ir, true, true);
    }
}

#[test_matrix(["shrl", "shll"], ["eq", "ne"], [true, false], [true, false])]
fn wide_controls_and_commuted_zero_comparisons(
    direction: &str,
    predicate: &str,
    fold: bool,
    hash: bool,
) {
    let ir = predicate_ir(9, 65, direction, predicate, None, "predicate", true);
    lower_and_prove(&ir, fold, hash);
}

#[test_matrix([0, 1], [0, 3], ["eq", "ne"])]
fn literal_selection_arms_reduce_to_a_boolean_guard(
    shift_case: usize,
    literal: usize,
    predicate: &str,
) {
    let ir = predicate_ir(
        17,
        8,
        "shrl",
        predicate,
        Some((shift_case, literal)),
        "predicate",
        false,
    );
    lower_and_prove(&ir, true, true);
}

#[test_matrix(["shifted", "selected"], [true, false])]
fn shared_words_keep_all_consumers(exposure: &str, fold: bool) {
    let ir = predicate_ir(17, 8, "shll", "eq", Some((0, 3)), exposure, false);
    lower_and_prove(&ir, fold, true);
}

#[test]
fn shift_predicate_charges_a_newly_live_source() {
    let ir = r#"package sample

top fn main(a: bits[1], b_input: bits[64], h_input: bits[64]) -> bits[1] {
  b: bits[1] = or_reduce(b_input, id=4)
  h: bits[1] = or_reduce(h_input, id=5)
  data: bits[4] = concat(b, b, a, h, id=6)
  na: bits[1] = not(a, id=7)
  s: bits[2] = concat(na, a, id=8)
  shifted: bits[4] = shrl(data, s, id=9)
  ret predicate: bits[1] = or_reduce(shifted, id=10)
}
"#;
    let stats = get_aig_stats(&lower_and_prove(ir, true, true));
    // The ordinary cone needs 63 ANDs for b and 10 for the shift and
    // reduction. The shift always discards h.
    assert!(stats.and_nodes <= 73, "{stats:?}");
}

#[test]
fn shift_predicate_charges_gates_retained_by_a_wider_shift() {
    let ir = r#"package sample

top fn main(x: bits[8], s: bits[3], y: bits[1]) -> (bits[1], bits[9]) {
  shifted: bits[8] = shrl(x, s, id=4)
  predicate: bits[1] = or_reduce(shifted, id=5)
  data: bits[9] = concat(x, y, id=6)
  wide: bits[9] = shrl(data, s, id=7)
  ret result: (bits[1], bits[9]) = tuple(predicate, wide, id=8)
}
"#;
    let stats = get_aig_stats(&lower_and_prove(ir, true, true));
    // All eight narrow shift bits are also outputs of the wider barrel.
    // The ordinary predicate adds only a seven-AND reduction to its 67 ANDs.
    assert!(stats.and_nodes <= 74, "{stats:?}");
}

#[test]
fn sharing_in_one_shift_preserves_an_independent_predicate_win() {
    let ir = r#"package sample

top fn main(x: bits[8], s: bits[3], y: bits[1], z: bits[8], t: bits[3]) -> (bits[1], bits[9], bits[1]) {
  shifted: bits[8] = shrl(x, s, id=6)
  predicate: bits[1] = or_reduce(shifted, id=7)
  data: bits[9] = concat(x, y, id=8)
  wide: bits[9] = shrl(data, s, id=9)
  independent: bits[8] = shrl(z, t, id=10)
  nonzero: bits[1] = or_reduce(independent, id=11)
  ret result: (bits[1], bits[9], bits[1]) = tuple(predicate, wide, nonzero, id=12)
}
"#;
    let stats = get_aig_stats(&lower_and_prove(ir, true, true));
    // Keep the 74-AND shared barrel and reduction, and spend at most 21
    // ANDs on the independent suffix predicate.
    assert!(stats.and_nodes <= 74 + 21, "{stats:?}");
}

#[test]
fn another_output_can_retain_the_entire_ordinary_predicate() {
    let ir = r#"package sample

top fn main(x: bits[8], s: bits[3], y: bits[1], z: bits[3], t: bits[2]) -> (bits[1], bits[1], bits[1]) {
  shifted: bits[8] = shrl(x, s, id=6)
  predicate: bits[1] = or_reduce(shifted, id=7)
  data: bits[9] = concat(x, y, id=8)
  wide: bits[9] = shrl(data, s, id=9)
  upper: bits[8] = bit_slice(wide, start=1, width=8, id=10)
  other: bits[1] = or_reduce(upper, id=11)
  independent: bits[3] = shrl(z, t, id=12)
  nonzero: bits[1] = or_reduce(independent, id=13)
  ret result: (bits[1], bits[1], bits[1]) = tuple(predicate, other, nonzero, id=14)
}
"#;
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let lowered = gatify(function, GatifyOptions::all_opts_disabled()).unwrap();
    check_equivalence::validate_same_fn_via_toolchain(function, &lowered.gate_fn).unwrap();
    // The first two outputs share all 65 ANDs. Rewriting only the third
    // output reduces its cost from 14 ANDs to 6.
    let stats = get_aig_stats(&lowered.gate_fn);
    assert!(stats.and_nodes <= 65 + 6, "{stats:?}");
    let result = &lowered.lowering_map[&function.ret_node_ref.unwrap()];
    assert!(
        result
            .iter_lsb_to_msb()
            .eq(lowered.gate_fn.outputs[0].bit_vector.iter_lsb_to_msb())
    );
}

#[test]
fn repeated_outputs_of_one_predicate_share_the_rewrite() {
    let ir = r#"package sample

top fn main(x: bits[8], s: bits[3]) -> (bits[1], bits[1]) {
  shifted: bits[8] = shrl(x, s, id=3)
  predicate: bits[1] = or_reduce(shifted, id=4)
  ret result: (bits[1], bits[1]) = tuple(predicate, predicate, id=5)
}
"#;
    let stats = get_aig_stats(&lower_and_prove(ir, true, true));
    assert!(stats.and_nodes <= 21, "{stats:?}");
}

#[test_matrix([true, false])]
fn late_computed_selector_can_share_the_source(fold: bool) {
    let ir = predicate_ir(17, 8, "shrl", "ne", Some((1, 3)), "predicate", false)
        .replace(", selector: bits[1] id=3", "")
        .replace(
            "  literal:",
            "  selector: bits[1] = or_reduce(x, id=3)\n  literal:",
        );
    lower_and_prove(&ir, fold, true);
}

#[test_matrix([0, 1, 8, 9, 16], ["shrl", "shll"])]
fn literal_amounts_preserve_zero_and_overshift_behavior(amount: usize, direction: &str) {
    let ir = predicate_ir(9, 5, direction, "eq", None, "predicate", false)
        .replace("amount: bits[5] id=2, ", "")
        .replace(
            "  zero:",
            &format!("  amount: bits[5] = literal(value={amount}, id=2)\n  zero:"),
        );
    lower_and_prove(&ir, true, true);
}

#[test_matrix([9, 48, 65], ["shrl", "shll"])]
fn canonical_shift_predicate_has_linear_gate_cost(width: usize, direction: &str) {
    let amount_width = ceil_log2(width) + 3;
    let ir = predicate_ir(
        width,
        amount_width,
        direction,
        "eq",
        None,
        "predicate",
        false,
    );
    let lowered = canonical_ir_text_to_g8r_lowering_artifacts(
        &ir,
        Some("main"),
        &CanonicalG8rOptions {
            enable_formal_array_alias_analysis: false,
            fraig: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            ..CanonicalG8rOptions::default()
        },
    )
    .unwrap();
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    check_equivalence::validate_same_fn_via_toolchain(
        package.get_top_fn().unwrap(),
        &lowered.gate_fn,
    )
    .unwrap();
    let stats = get_aig_stats(&lowered.gate_fn);
    // Each binary tree node needs at most one unconditional OR and three
    // predicate gates; the remaining controls form one overshift guard.
    assert!(
        stats.and_nodes <= 4 * width + amount_width,
        "unexpected predicate size: {stats:?}"
    );
    assert!(
        stats.max_depth <= 3 * (ceil_log2(width) + 1),
        "unexpected predicate depth: {stats:?}"
    );
}

#[test]
fn priority_window_preserves_selector_bit_32() {
    let mut arms = ["zero"; 33];
    arms[0] = "y";
    arms[32] = "x";
    let arms = arms.join(", ");
    let ir = format!(
        r#"package p
top fn f(k: bits[1], x: bits[2], y: bits[2]) -> bits[1] {{
  low: bits[32] = literal(value=0, id=4)
  s: bits[33] = concat(k, low, id=5)
  zero: bits[2] = literal(value=0, id=6)
  word: bits[2] = priority_sel(s, cases=[{arms}], default=zero, id=7)
  ret pred: bits[1] = or_reduce(word, id=8)
}}"#
    );
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let gates = gatify_prepared_fn(
        package.get_top_fn().unwrap(),
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap()
    .gate_fn;
    assert_eq!(get_aig_stats(&gates).and_nodes, 2);
    let mut sim = PreparedGateSim::new(&gates);
    for k in [false, true] {
        for x in 0..4 {
            for y in 0..4 {
                let inputs = [
                    IrBits::bool(k),
                    IrBits::make_ubits(2, x).unwrap(),
                    IrBits::make_ubits(2, y).unwrap(),
                ];
                assert_eq!(sim.eval_outputs(&inputs), [IrBits::bool(k && x != 0)]);
            }
        }
    }
}

#[test]
fn window_prices_shared_selector_fanin_with_the_predicate() {
    // A window can replace the balanced four-input AND with a chain when it
    // is private. Returning the balanced tree as well would cost an extra AND.
    for (ret_type, outputs, expected_ands) in [
        ("bits[1]", "identity(pred, id=8)", 5),
        ("(bits[1], bits[1])", "tuple(p, pred, id=8)", 6),
    ] {
        let ir = format!(
            r#"package p
top fn f(k: bits[4], x: bits[2]) -> {ret_type} {{
  p: bits[1] = and_reduce(k, id=3)
  s: bits[1] = identity(p, id=4)
  zero: bits[2] = literal(value=0, id=5)
  word: bits[2] = sel(s, cases=[zero, x], id=6)
  pred: bits[1] = or_reduce(word, id=7)
  ret result: {ret_type} = {outputs}
}}"#
        );
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let gates = gatify_prepared_fn(
            package.get_top_fn().unwrap(),
            GatifyOptions::all_opts_disabled(),
        )
        .unwrap()
        .gate_fn;
        let stats = get_aig_stats(&gates);
        assert_eq!((stats.and_nodes, stats.max_depth), (expected_ands, 4));
    }
}

#[test]
fn hashed_selector_alias_keeps_an_independent_window_win() {
    let ir = r#"package p
top fn f(k: bits[4], x: bits[2], y: bits[2], t: bits[1]) -> (bits[1], bits[1], bits[1]) {
  p: bits[1] = and_reduce(k, id=5)
  q: bits[1] = and_reduce(k, id=6)
  zero: bits[2] = literal(value=0, id=7)
  a: bits[2] = sel(p, cases=[zero, x], id=8)
  pa: bits[1] = or_reduce(a, id=9)
  b: bits[2] = sel(t, cases=[zero, y], id=10)
  pb: bits[1] = or_reduce(b, id=11)
  ret result: (bits[1], bits[1], bits[1]) = tuple(q, pa, pb, id=12)
}"#;
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let gates = gatify_prepared_fn(
        package.get_top_fn().unwrap(),
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(package.get_top_fn().unwrap(), &gates)
        .unwrap();
    // q retains p's balanced tree. The first predicate costs six ANDs with
    // that tree, and the independent guarded reduction costs two.
    let stats = get_aig_stats(&gates);
    assert_eq!((stats.and_nodes, stats.max_depth), (8, 4));
}

#[test_matrix([false, true])]
fn winning_window_keeps_its_observable_tag(identity: bool) {
    for (operation, tag) in [
        ("or_reduce(word, id=7)", "or_reduce_7"),
        ("eq(word, zero, id=7)", "eq_7"),
        ("ne(word, zero, id=7)", "ne_7"),
        ("sel(s, cases=[off, nz], id=7)", "sel_7_output_bit_0"),
    ] {
        let result = if identity {
            format!("pred: bits[1] = {operation}\n  ret result: bits[1] = identity(pred, id=8)")
        } else {
            format!("ret pred: bits[1] = {operation}")
        };
        let ir = format!(
            r#"package p
top fn f(s: bits[1], x: bits[2]) -> bits[1] {{
  zero: bits[2] = literal(value=0, id=3)
  word: bits[2] = sel(s, cases=[zero, x], id=4)
  nz: bits[1] = or_reduce(word, id=5)
  off: bits[1] = literal(value=0, id=6)
  {result}
}}"#
        );
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let gates = gatify_prepared_fn(
            package.get_top_fn().unwrap(),
            GatifyOptions::all_opts_disabled(),
        )
        .unwrap()
        .gate_fn;
        assert_eq!(get_aig_stats(&gates).and_nodes, 2);
        let root = gates.outputs[0].bit_vector.get_lsb(0);
        let mut sim = PreparedGateSim::new(&gates);
        for s in [false, true] {
            let result = sim.eval(
                &[IrBits::bool(s), IrBits::make_ubits(2, 1).unwrap()],
                Collect::Tagged,
            );
            // Tags identify AIG nodes and the output edge may invert that node.
            let canonical = result.outputs[0].get_bit(0).unwrap() ^ root.negated;
            assert_eq!(
                result.tagged_values.unwrap().get(tag),
                Some(&canonical),
                "{tag}"
            );
        }
    }
}

#[test]
fn standalone_reduction_observes_an_independent_word() {
    let ir = predicate_ir(3, 2, "shrl", "or_reduce", None, "predicate", false);
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let gates = gatify_node_as_fn(
        function,
        function.ret_node_ref.unwrap(),
        &GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let mut simulation = PreparedGateSim::new(&gates);
    for value in 0..8 {
        assert_eq!(
            simulation.eval_outputs(&[IrBits::make_ubits(3, value).unwrap()]),
            vec![IrBits::bool(value != 0)]
        );
    }
}

/// Constructs an observation with optional literal selection and shared words.
fn predicate_ir(
    width: usize,
    amount_width: usize,
    direction: &str,
    predicate: &str,
    selection: Option<(usize, usize)>,
    exposure: &str,
    zero_first: bool,
) -> String {
    let (selection_lines, observed) = if let Some((shift_case, literal)) = selection {
        let cases = if shift_case == 0 {
            "shifted, literal"
        } else {
            "literal, shifted"
        };
        (
            format!(
                "  literal: bits[{width}] = literal(value={literal}, id=6)\n  selected: bits[{width}] = sel(selector, cases=[{cases}], id=7)\n"
            ),
            "selected",
        )
    } else {
        (String::new(), "shifted")
    };
    let operation = if predicate == "or_reduce" {
        format!("or_reduce({observed}, id=8)")
    } else if zero_first {
        format!("{predicate}(zero, {observed}, id=8)")
    } else {
        format!("{predicate}({observed}, zero, id=8)")
    };
    let (return_type, return_lines) = if exposure == "predicate" {
        (
            "bits[1]".to_string(),
            format!("  ret predicate: bits[1] = {operation}\n"),
        )
    } else {
        (
            format!("(bits[{width}], bits[1])"),
            format!(
                "  predicate: bits[1] = {operation}\n  ret result: (bits[{width}], bits[1]) = tuple({exposure}, predicate, id=9)\n"
            ),
        )
    };
    format!(
        "package sample\n\n top fn main(x: bits[{width}] id=1, amount: bits[{amount_width}] id=2, selector: bits[1] id=3) -> {return_type} {{\n  zero: bits[{width}] = literal(value=0, id=4)\n  shifted: bits[{width}] = {direction}(x, amount, id=5)\n{selection_lines}{return_lines}}}\n"
    )
}

/// Proves the complete function, including shared word outputs, against native
/// XLS.
fn lower_and_prove(ir: &str, fold: bool, hash: bool) -> GateFn {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let gates = gatify(
        function,
        GatifyOptions {
            fold,
            hash,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
    gates
}
