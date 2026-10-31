// SPDX-License-Identifier: Apache-2.0

use test_case::test_matrix;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;

#[test_matrix([2, 3, 17, 65, 129], [true, false])]
fn scalar_sum_preserves_unsigned_extension_and_shared_consumers(width: usize, fold: bool) {
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let text = scalar_sum_ir(width, [3, 1, 9], order, true);
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let prepared = prep_for_gatify(
            function,
            None,
            PrepForGatifyOptions {
                enable_rewrite_nary_add: true,
                ..PrepForGatifyOptions::all_opts_disabled()
            },
        );
        let gates = gatify_prepared_fn(
            &prepared,
            GatifyOptions {
                fold,
                hash: true,
                enable_rewrite_nary_add: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap()
        .gate_fn;
        check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
    }
}

#[test]
fn scalar_sum_places_the_late_input_after_the_first_xor() {
    let text = scalar_sum_ir(2, [3, 1, 17], [0, 1, 2], false);
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(
        function,
        None,
        PrepForGatifyOptions {
            enable_rewrite_nary_add: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    let mut measurements = Vec::new();
    for enable_rewrite_nary_add in [false, true] {
        let gates = gatify_prepared_fn(
            &prepared,
            GatifyOptions {
                enable_rewrite_nary_add,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap()
        .gate_fn;
        check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
        measurements.push(get_aig_stats(&gates));
    }
    let carry_in = &measurements[0];
    let full_adder = &measurements[1];
    assert!(
        full_adder.max_depth < carry_in.max_depth,
        "late scalar term retained the CPA carry-in placement: {measurements:?}"
    );
}

#[test]
fn scalar_sum_keeps_an_already_late_or_tied_carry_input() {
    for sizes in [[17, 1, 3], [3, 1, 3], [1, 1, 1]] {
        let text = scalar_sum_ir(2, sizes, [0, 1, 2], false);
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let prepared = prep_for_gatify(
            function,
            None,
            PrepForGatifyOptions {
                enable_rewrite_nary_add: true,
                ..PrepForGatifyOptions::all_opts_disabled()
            },
        );
        let mut graphs = Vec::new();
        for enable_rewrite_nary_add in [false, true] {
            let gates = gatify_prepared_fn(
                &prepared,
                GatifyOptions {
                    enable_rewrite_nary_add,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap()
            .gate_fn;
            check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
            graphs.push(gates.to_string());
        }
        assert_eq!(
            graphs[0], graphs[1],
            "already-late carry-in placement changed"
        );
    }
}

#[test_matrix([1, 2, 65], [true, false], [true, false])]
fn scalar_sum_preserves_signed_and_subtracted_terms(width: usize, signed: bool, subtract: bool) {
    let extension = if signed { "sign_ext" } else { "zero_ext" };
    let operation = if subtract { "sub" } else { "add" };
    let text = format!(
        r#"package scalar_sum

top fn main(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3) -> bits[{width}] {{
  a_wide: bits[{width}] = {extension}(a, new_bit_count={width}, id=4)
  b_wide: bits[{width}] = {extension}(b, new_bit_count={width}, id=5)
  c_wide: bits[{width}] = {extension}(c, new_bit_count={width}, id=6)
  ab: bits[{width}] = add(a_wide, b_wide, id=7)
  ret result: bits[{width}] = {operation}(ab, c_wide, id=8)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(
        function,
        None,
        PrepForGatifyOptions {
            enable_rewrite_nary_add: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    let gates = gatify_prepared_fn(
        &prepared,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
}

#[test_matrix(["umul", "smul"])]
fn scalar_sum_preserves_deferred_product_operands(operation: &str) {
    let text = format!(
        r#"package scalar_sum

top fn main(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3,
            d: bits[1] id=4, e: bits[1] id=5, f: bits[1] id=6) -> bits[2] {{
  p: bits[1] = {operation}(a, b, id=7)
  q: bits[1] = {operation}(c, d, id=8)
  r: bits[1] = {operation}(e, f, id=9)
  p_wide: bits[2] = zero_ext(p, new_bit_count=2, id=10)
  q_wide: bits[2] = zero_ext(q, new_bit_count=2, id=11)
  r_wide: bits[2] = zero_ext(r, new_bit_count=2, id=12)
  pq: bits[2] = add(p_wide, q_wide, id=13)
  ret result: bits[2] = add(pq, r_wide, id=14)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(
        function,
        None,
        PrepForGatifyOptions {
            enable_rewrite_nary_add: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    let gates = gatify_prepared_fn(
        &prepared,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
}

#[test_matrix([true, false])]
fn scalar_sum_preserves_constants_aliases_and_complements(fold: bool) {
    for terms in [
        ["a", "a", "a"],
        ["a", "a", "na"],
        ["a", "zero", "one"],
        ["one", "one", "one"],
        ["zero", "zero", "zero"],
    ] {
        let [a, b, c] = terms;
        let text = format!(
            r#"package scalar_sum

top fn main(a: bits[1] id=1) -> bits[9] {{
  na: bits[1] = not(a, id=2)
  zero: bits[1] = literal(value=0, id=3)
  one: bits[1] = literal(value=1, id=4)
  a_wide: bits[9] = zero_ext({a}, new_bit_count=9, id=5)
  b_wide: bits[9] = zero_ext({b}, new_bit_count=9, id=6)
  c_wide: bits[9] = zero_ext({c}, new_bit_count=9, id=7)
  ab: bits[9] = add(a_wide, b_wide, id=8)
  ret result: bits[9] = add(ab, c_wide, id=9)
}}
"#
        );
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let prepared = prep_for_gatify(
            function,
            None,
            PrepForGatifyOptions {
                enable_rewrite_nary_add: true,
                ..PrepForGatifyOptions::all_opts_disabled()
            },
        );
        let gates = gatify_prepared_fn(
            &prepared,
            GatifyOptions {
                fold,
                hash: true,
                enable_rewrite_nary_add: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap()
        .gate_fn;
        check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
    }
}

#[test]
fn scalar_sum_respects_explicit_architecture_and_unhashed_mapping() {
    let text = scalar_sum_ir(2, [3, 1, 9], [0, 1, 2], false);
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(
        function,
        None,
        PrepForGatifyOptions {
            enable_rewrite_nary_add: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    for arch in [
        None,
        Some(ir::ExtNaryAddArchitecture::RippleCarry),
        Some(ir::ExtNaryAddArchitecture::BrentKung),
        Some(ir::ExtNaryAddArchitecture::KoggeStone),
    ] {
        let mut explicit = prepared.clone();
        for node in &mut explicit.nodes {
            if let ir::NodePayload::ExtNaryAdd { arch: actual, .. } = &mut node.payload {
                *actual = arch;
            }
        }
        let hash = arch.is_some();
        let mut graphs = Vec::new();
        for enable_rewrite_nary_add in [false, true] {
            let gates = gatify_prepared_fn(
                &explicit,
                GatifyOptions {
                    hash,
                    enable_rewrite_nary_add,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap()
            .gate_fn;
            check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
            graphs.push(gates.to_string());
        }
        assert_eq!(
            graphs[0], graphs[1],
            "explicit architecture or unhashed lowering changed"
        );
    }
}

/// Makes three independently delayed scalar terms with an optional shared
/// observation.
fn scalar_sum_ir(width: usize, sizes: [usize; 3], order: [usize; 3], shared: bool) -> String {
    let [aw, bw, cw] = sizes;
    let names = ["a_wide", "b_wide", "c_wide"];
    let [a, b, c] = order.map(|i| names[i]);
    let (result_type, return_line) = if shared {
        (
            format!("(bits[{width}], bits[1], bits[1], bits[1])"),
            format!(
                "  ret observed: (bits[{width}], bits[1], bits[1], bits[1]) = tuple(result, a, b, c, id=12)"
            ),
        )
    } else {
        (
            format!("bits[{width}]"),
            format!("  ret observed: bits[{width}] = identity(result, id=12)"),
        )
    };
    format!(
        r#"package scalar_sum

top fn main(a_in: bits[{aw}] id=1, b_in: bits[{bw}] id=2, c_in: bits[{cw}] id=3) -> {result_type} {{
  a: bits[1] = and_reduce(a_in, id=4)
  b: bits[1] = and_reduce(b_in, id=5)
  c: bits[1] = and_reduce(c_in, id=6)
  a_wide: bits[{width}] = zero_ext(a, new_bit_count={width}, id=7)
  b_wide: bits[{width}] = zero_ext(b, new_bit_count={width}, id=8)
  c_wide: bits[{width}] = zero_ext(c, new_bit_count={width}, id=9)
  ab: bits[{width}] = add({a}, {b}, id=10)
  result: bits[{width}] = add(ab, {c}, id=11)
{return_line}
}}
"#
    )
}
