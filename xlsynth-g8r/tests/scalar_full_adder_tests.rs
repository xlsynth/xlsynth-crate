// SPDX-License-Identifier: Apache-2.0

use test_case::test_matrix;
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, parse_g8r};
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::IrBits;
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
fn scalar_sum_shares_parity_when_the_two_latest_inputs_tie() {
    let text = r#"package scalar_sum

top fn main(a: bits[1] id=1, x: bits[1] id=2, y: bits[1] id=3, u: bits[1] id=4, v: bits[1] id=5) -> bits[2] {
  b: bits[1] = and(x, y, id=6)
  c: bits[1] = and(u, v, id=7)
  a_wide: bits[2] = zero_ext(a, new_bit_count=2, id=8)
  b_wide: bits[2] = zero_ext(b, new_bit_count=2, id=9)
  c_wide: bits[2] = zero_ext(c, new_bit_count=2, id=10)
  ab: bits[2] = add(a_wide, b_wide, id=11)
  sum: bits[2] = add(ab, c_wide, id=12)
  low: bits[1] = bit_slice(sum, start=0, width=1, id=13)
  bc: bits[1] = xor(b, c, id=14)
  parity: bits[1] = xor(a, bc, id=15)
  ret observed: bits[2] = concat(low, parity, id=16)
}
"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let gates = gatify(
        function,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
    let outputs = &gates.outputs[0].bit_vector;
    assert_eq!(
        outputs.get_lsb(0),
        outputs.get_lsb(1),
        "the explicit parity and sum[0] should share a gate"
    );
}

#[test]
fn scalar_sum_folds_a_repeated_or_complementary_late_gate() {
    for (operation, expected_live_ands) in [("and", 1), ("nand", 0)] {
        let text = format!(
            r#"package scalar_sum

top fn main(a: bits[1] id=1, x: bits[1] id=2, y: bits[1] id=3) -> bits[2] {{
  b: bits[1] = and(x, y, id=4)
  c: bits[1] = {operation}(y, x, id=5)
  a_wide: bits[2] = zero_ext(a, new_bit_count=2, id=6)
  b_wide: bits[2] = zero_ext(b, new_bit_count=2, id=7)
  c_wide: bits[2] = zero_ext(c, new_bit_count=2, id=8)
  ab: bits[2] = add(a_wide, b_wide, id=9)
  ret result: bits[2] = add(ab, c_wide, id=10)
}}
"#
        );
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let gates = gatify(
            function,
            GatifyOptions {
                enable_rewrite_nary_add: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap()
        .gate_fn;
        check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
        // b + b = 2*b, while b + !b = 1. Neither sum needs any adder gates.
        assert_eq!(get_aig_stats(&gates).and_nodes, expected_live_ands);
    }
}

#[test]
fn scalar_add_chain_uses_the_callers_adder_mapping() {
    let text = scalar_sum_ir(2, [1, 1, 2], [0, 1, 2], false);
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(function, None, PrepForGatifyOptions::all_opts_enabled());
    for adder_mapping in [
        AdderMapping::RippleCarry,
        AdderMapping::BrentKung,
        AdderMapping::KoggeStone,
    ] {
        let options = GatifyOptions {
            adder_mapping,
            ..GatifyOptions::all_opts_disabled()
        };
        // Use the same prepared n-ary sum with the scalar rewrite disabled as
        // the structural reference for this topology.
        let reference = gatify_prepared_fn(&prepared, options.clone()).unwrap();
        let actual = gatify(
            function,
            GatifyOptions {
                enable_rewrite_nary_add: true,
                ..options
            },
        )
        .unwrap();
        assert_eq!(
            emit_aiger(&actual.gate_fn, false).unwrap(),
            emit_aiger(&reference.gate_fn, false).unwrap(),
            "{adder_mapping}"
        );
    }
}

#[test]
fn scalar_sum_serializes_both_output_tags_for_simulation() {
    let text = scalar_sum_ir(2, [1, 1, 2], [0, 1, 2], false);
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let gates = gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions {
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let serialized = emit_g8r(&SequentialGateFn::from_gate_fn(gates)).unwrap();
    let gates = parse_g8r(&serialized).unwrap().try_into_gate_fn().unwrap();
    for (a, b, c) in [(0, 0, 0), (0, 1, 0), (0, 1, 3), (1, 1, 3)] {
        let inputs = [
            IrBits::make_ubits(1, a).unwrap(),
            IrBits::make_ubits(1, b).unwrap(),
            IrBits::make_ubits(2, c).unwrap(),
        ];
        let sim = gate_sim::eval(&gates, &inputs, Collect::Tagged);
        let tags = sim.tagged_values.unwrap();
        let sum = a + b + u64::from(c == 3);
        for (i, output) in gates.outputs[0].bit_vector.iter_lsb_to_msb().enumerate() {
            // Tags belong to the underlying AND node, before output inversion.
            let node_value = ((sum >> i) & 1 != 0) ^ output.negated;
            assert_eq!(
                tags.get(&format!("ext_nary_add_11_brent_kung_output_bit_{i}")),
                Some(&node_value)
            );
        }
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
