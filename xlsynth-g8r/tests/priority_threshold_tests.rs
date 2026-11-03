// SPDX-License-Identifier: Apache-2.0

use test_case::test_matrix;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_cone_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_node_as_fn};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

#[test_matrix([false, true])]
fn masked_affine_priority_bit_forwards_the_winning_input(output_inverted: bool) {
    for (operation, constant, invert) in [("sub", 16, false), ("add", 15, true)] {
        let returned = if output_inverted {
            "bit: bits[1] = bit_slice(masked, start=4, width=1, id=9)\n  ret flag: bits[1] = not(bit, id=10)"
        } else {
            "ret bit: bits[1] = bit_slice(masked, start=4, width=1, id=9)"
        };
        let ir = format!(
            r#"package affine
top fn main(word: bits[13], mask: bits[5]) -> bits[1] {{
  hot: bits[14] = one_hot(word, lsb_prio=true, id=3)
  index: bits[4] = encode(hot, id=4)
  extended: bits[5] = zero_ext(index, new_bit_count=5, id=5)
  k: bits[5] = literal(value={constant}, id=6)
  value: bits[5] = {operation}(k, extended, id=7)
  masked: bits[5] = and(mask, value, id=8)
  {returned}
}}"#
        );
        let gates = canonical_ir_text_to_g8r_lowering_artifacts(
            &ir,
            Some("main"),
            &CanonicalG8rOptions {
                enable_formal_array_alias_analysis: false,
                fraig: false,
                ..Default::default()
            },
        )
        .unwrap()
        .gate_fn;
        let root = *gates.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(get_aig_cone_stats(&gates.gates, &[root]).and_nodes, 1);
        let mut sim = PreparedGateSim::new(&gates);
        for word in 0..8192 {
            for mask in [15, 16] {
                let inputs = [
                    IrBits::make_ubits(13, word).unwrap(),
                    IrBits::make_ubits(5, mask).unwrap(),
                ];
                assert_eq!(
                    sim.eval_outputs(&inputs),
                    vec![IrBits::bool(
                        (mask == 16 && ((word & 1 != 0) ^ invert)) ^ output_inverted
                    )]
                );
            }
        }
    }
}

#[test]
fn affine_priority_preserves_an_independent_output_through_rewriting() {
    use xlsynth_g8r::gate_fn_optimize::{GateFnOptimizeOptions, optimize_gate_fn};
    use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};
    use xlsynth_g8r::process_ir_path::canonical_ir_text_to_g8r_artifacts;

    let ir = r#"package dispatch
top fn main(request: bits[13] id=1, bank: bits[3] id=2) -> (bits[1], bits[1]) {
  hot: bits[14] = one_hot(request, lsb_prio=true, id=3)
  ordinal: bits[4] = encode(hot, id=4)
  wide: bits[6] = zero_ext(ordinal, new_bit_count=6, id=5)
  bias: bits[6] = literal(value=31, id=6)
  stamp: bits[6] = add(wide, bias, id=7)
  defer: bits[1] = bit_slice(stamp, start=5, width=1, id=8)
  lanes: bits[7] = bit_slice(request, start=4, width=7, id=9)
  high: bits[8] = one_hot(lanes, lsb_prio=false, id=10)
  idx: bits[3] = encode(high, id=11)
  matches: bits[1] = eq(idx, bank, id=12)
  ret result: (bits[1], bits[1]) = tuple(defer, matches, id=13)
}"#;
    let settings = CanonicalG8rOptions {
        enable_formal_array_alias_analysis: false,
        fraig: false,
        enable_rewrite_prio_encode: false,
        enable_rewrite_carry_out: false,
        ..Default::default()
    };
    let options = settings.to_process_ir_path_options(Some("main"), true, false, false, None);
    let ordinary =
        ir2gates_from_ir_text(ir, Some("main"), Ir2GatesOptions::from(&options)).unwrap();
    let finished = optimize_gate_fn(
        ordinary.gatify_output.gate_fn,
        &GateFnOptimizeOptions::from(&options),
    )
    .unwrap()
    .gate_fn;
    let actual = canonical_ir_text_to_g8r_artifacts(ir, Some("main"), &settings).unwrap();
    let direct = *finished.inputs[0].bit_vector.get_lsb(0);
    assert_eq!(*finished.outputs[0].bit_vector.get_lsb(1), direct.negate());
    assert_eq!(
        actual.binary_aiger,
        xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary(&finished, true).unwrap()
    );
}

#[test_matrix([("hot", 9), ("index", 4), ("extended", 5), ("difference", 5)])]
fn shared_affine_ordinal_intermediates_preserve_the_word(exposure: (&str, usize)) {
    let (other, width) = exposure;
    let ir = format!(
        r#"package shared
top fn main(x: bits[8], mask: bits[5]) -> (bits[1], bits[{width}]) {{
  hot: bits[9] = one_hot(x, lsb_prio=true, id=3)
  index: bits[4] = encode(hot, id=4)
  extended: bits[5] = zero_ext(index, new_bit_count=5, id=5)
  k: bits[5] = literal(value=16, id=6)
  difference: bits[5] = sub(k, extended, id=7)
  masked: bits[5] = and(mask, difference, id=8)
  bit: bits[1] = bit_slice(masked, start=4, width=1, id=9)
  ret result: (bits[1], bits[{width}]) = tuple(bit, {other}, id=10)
}}"#
    );
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let gates = gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let root = *gates.outputs[0].bit_vector.get_lsb(width);
    assert!(
        get_aig_cone_stats(&gates.gates, &[root]).and_nodes > 1,
        "the observed word keeps the priority arithmetic for {other}"
    );
    let mut sim = PreparedGateSim::new(&gates);
    for x in 0u64..256 {
        let ordinal = u64::from(x.trailing_zeros().min(8));
        let retained = match other {
            "hot" => 1 << ordinal,
            "difference" => 16 - ordinal,
            _ => ordinal,
        };
        for mask in [0, 16] {
            let expected = (u64::from(mask != 0 && x & 1 != 0) << width) | retained;
            let inputs = [
                IrBits::make_ubits(8, x).unwrap(),
                IrBits::make_ubits(5, mask).unwrap(),
            ];
            assert_eq!(
                sim.eval_outputs(&inputs),
                vec![IrBits::make_ubits(width + 1, expected).unwrap()]
            );
        }
    }
}

#[test_matrix([false, true])]
fn computed_priority_or_mask_bit_retains_the_affine_cone(compute_mask: bool) {
    let (source, mask) = if compute_mask {
        ("x", "lt")
    } else {
        ("sum", "m")
    };
    let mut bare =
        String::from("package computed\n\ntop fn main(x: bits[4], y: bits[4], m: bits[1])");
    let body = format!(
        r#"
  sum: bits[4] = xor(x, y, id=4)
  lt: bits[1] = ult(x, y, id=5)
  hot: bits[5] = one_hot({source}, lsb_prio=true, id=6)
  index: bits[3] = encode(hot, id=7)
  wide: bits[4] = zero_ext(index, new_bit_count=4, id=8)
  k: bits[4] = literal(value=8, id=9)
  difference: bits[4] = sub(k, wide, id=10)
  mask: bits[4] = sign_ext({mask}, new_bit_count=4, id=11)
  masked: bits[4] = xor(difference, mask, id=12)
  flag: bits[1] = bit_slice(masked, start=3, width=1, id=13)"#
    );
    let retained = format!(
        "{bare} -> (bits[1], bits[4]) {{{body}\n  ret out: (bits[1], bits[4]) = tuple(flag, difference, id=14)\n}}"
    );
    bare.push_str(&format!(
        " -> bits[1] {{{}\n}}",
        body.replace("  flag: ", "  ret flag: ")
    ));

    let mut sizes = Vec::new();
    for (ir, retained_width) in [(&bare, 0), (&retained, 4)] {
        let package = Parser::new(ir).parse_and_validate_package().unwrap();
        let gates = gatify(
            package.get_top_fn().unwrap(),
            GatifyOptions {
                fold: true,
                hash: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap()
        .gate_fn;
        let flag = *gates.outputs[0].bit_vector.get_lsb(retained_width);
        sizes.push(get_aig_cone_stats(&gates.gates, &[flag]).and_nodes);
        let mut sim = PreparedGateSim::new(&gates);
        for x in 0u64..16 {
            for y in 0u64..16 {
                for m in [false, true] {
                    let source = if compute_mask { x } else { x ^ y };
                    let mask = if compute_mask { x < y } else { m };
                    let difference = 8 - u64::from(source.trailing_zeros().min(4));
                    let value = u64::from((source & 1 != 0) ^ mask);
                    let expected = if retained_width == 0 {
                        value
                    } else {
                        (value << 4) | difference
                    };
                    let inputs = [
                        IrBits::make_ubits(4, x).unwrap(),
                        IrBits::make_ubits(4, y).unwrap(),
                        IrBits::bool(m),
                    ];
                    assert_eq!(
                        sim.eval_outputs(&inputs),
                        vec![IrBits::make_ubits(retained_width + 1, expected).unwrap()]
                    );
                }
            }
        }
    }
    assert_eq!(sizes[0], sizes[1]);
}

#[test_matrix([1, 2, 3, 4, 8, 9, 33, 65, 161], [true, false], [true, false])]
fn priority_threshold_preserves_ordinal_semantics(width: usize, lsb_prio: bool, fold: bool) {
    let encoded_width = ceil_log2(width + 1);
    for start in 0..encoded_width {
        let ir = priority_ir(width, lsb_prio, Some((start, encoded_width - start)));
        lower_and_prove(&ir, fold);
    }
    lower_and_prove(&priority_ir(width, lsb_prio, None), fold);
}

#[test_matrix([true, false], [true, false])]
fn partial_ordinal_slices_preserve_bit_tests(lsb_prio: bool, fold: bool) {
    // Ordinal bit one is periodic, unlike a test of all bits from one upward.
    lower_and_prove(&priority_ir(16, lsb_prio, Some((1, 1))), fold);
}

#[test_matrix([true, false])]
fn shared_priority_ordinal_preserves_both_outputs(fold: bool) {
    let ir = r#"package sample
 top fn main(x: bits[65]) -> (bits[7], bits[1]) {
  hot: bits[66] = one_hot(x, lsb_prio=true, id=2)
  ordinal: bits[7] = encode(hot, id=3)
  high: bits[6] = bit_slice(ordinal, start=1, width=6, id=4)
  predicate: bits[1] = or_reduce(high, id=5)
  ret result: (bits[7], bits[1]) = tuple(ordinal, predicate, id=6)
 }
"#;
    lower_and_prove(ir, fold);
}

#[test]
fn standalone_reduction_treats_its_operand_as_independent() {
    let ir = priority_ir(8, true, Some((1, 3)));
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

#[test_matrix([true, false])]
fn conditional_sum_predicate_avoids_serial_carries(nary_prep: bool) {
    let options = CanonicalG8rOptions {
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
        enable_rewrite_nary_add: nary_prep,
        ..CanonicalG8rOptions::default()
    };
    let observed_high_bits = predicate_depth(&sum_predicate_ir(65, 1, false, "high-bit"), &options);
    let materialized_word =
        predicate_depth(&sum_predicate_ir(65, 1, false, "full-selection"), &options);
    assert!(
        observed_high_bits < materialized_word,
        "predicate should arrive earlier when the whole word is unobserved: {observed_high_bits} vs {materialized_word}"
    );
}

#[test]
fn prepared_clz_preserves_priority_predicate_depth() {
    let ir = sum_predicate_ir(65, 1, false, "high-bit");
    let raw_depth = predicate_depth(
        &ir,
        &CanonicalG8rOptions {
            enable_rewrite_prio_encode: false,
            ..CanonicalG8rOptions::default()
        },
    );
    let prepared_depth = predicate_depth(&ir, &CanonicalG8rOptions::default());
    assert!(
        prepared_depth <= raw_depth,
        "prepared CLZ depth {prepared_depth} exceeds encoder depth {raw_depth}"
    );
}

#[test_matrix(["and", "nand"])]
fn threshold_conjunction_folds_only_for_tested_bits(op: &str) {
    let options = CanonicalG8rOptions::default();
    for (source, bit, constant) in [
        ("selected", 4, true),
        ("selected", 3, false),
        ("a", 4, false),
    ] {
        let ir = sum_predicate_ir(8, 2, false, "high-bit")
            .replace(
                "bit_slice(selected, start=7",
                &format!("bit_slice({source}, start={bit}"),
            )
            .replace(
                "  ret result:",
                &format!(
                    "  combined: bits[1] = {op}(high, control, predicate, id=15)\n  ret result:"
                ),
            )
            .replace("tuple(predicate, high,", "tuple(combined, high,");
        assert_eq!(
            predicate_depth(&ir, &options) == 0,
            constant,
            "{source}[{bit}]"
        );
    }
}

#[test]
fn adjusted_clz_predicates_preserve_offset_and_truncation() {
    for (offset, count_width) in [(1, 4), (0, 3)] {
        let ir = format!(
            r#"package sample
top fn main(a: bits[8], b: bits[8], control: bits[1]) -> bits[1] {{
  sum: bits[8] = add(a, b, id=4)
  negative: bits[8] = neg(sum, id=5)
  selected: bits[8] = sel(control, cases=[sum, negative], id=6)
  count: bits[{count_width}] = ext_clz(selected, offset={offset}, new_bit_count={count_width}, id=7)
  ret predicate: bits[1] = or_reduce(count, id=8)
}}
"#
        );
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let gates = gatify(function, GatifyOptions::all_opts_enabled())
            .unwrap()
            .gate_fn;
        let mut oracle = function.clone();
        desugar_extensions_in_fn(&mut oracle).unwrap();
        check_equivalence::validate_same_fn_via_toolchain(&oracle, &gates).unwrap();
    }
}

#[test_matrix([false, true])]
fn full_width_priority_predicate_has_zero_test_depth(prio_prep: bool) {
    let ir = r#"package sample
top fn main(a: bits[64], control: bits[1]) -> bits[1] {
  three: bits[64] = literal(value=3, id=3)
  sum: bits[64] = add(a, three, id=4)
  negative: bits[64] = neg(sum, id=5)
  selected: bits[64] = sel(control, cases=[sum, negative], id=6)
  reversed: bits[64] = reverse(selected, id=7)
  hot: bits[65] = one_hot(reversed, lsb_prio=true, id=8)
  ordinal: bits[7] = encode(hot, id=9)
  high: bits[1] = bit_slice(ordinal, start=6, width=1, id=10)
  ret predicate: bits[1] = or_reduce(high, id=11)
}
"#;
    let depth = predicate_depth(
        ir,
        &CanonicalG8rOptions {
            enable_rewrite_prio_encode: prio_prep,
            ..CanonicalG8rOptions::default()
        },
    );
    // Each operand mismatch needs at most two XORs before the balanced zero
    // test.
    assert!(
        depth <= 4 + ceil_log2(64),
        "full-width predicate depth {depth}"
    );
}

#[test_matrix([1, 2, 3, 4, 8, 9, 33, 65, 161], [true, false], [true, false], [true, false])]
fn conditional_sum_priority_predicates_are_equivalent(
    width: usize,
    fold: bool,
    truncated: bool,
    nary_prep: bool,
) {
    for start in 0..ceil_log2(width + 1) {
        lower_and_prove_with_prep(
            &sum_predicate_ir(width, start, truncated, "high-bit"),
            fold,
            nary_prep,
        );
    }
}

#[test_matrix(["full-selection", "shared-ordinal", "partial-ordinal"], [true, false])]
fn shared_conditional_sum_outputs_are_preserved(exposure: &str, fold: bool) {
    lower_and_prove(&sum_predicate_ir(65, 1, true, exposure), fold);
}

#[test_matrix([false, true], ["high-bit", "full-sum"])]
fn priority_prefix_preserves_explicit_adder_architectures(truncated: bool, exposure: &str) {
    use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
    use xlsynth_g8r::ir2gate_utils::AdderMapping;

    // Only the low nine bits of the widened sum can vary. Without a separately
    // observed sum, bypassing its low prefix can eliminate the entire mapped
    // adder.
    let width = 16 + usize::from(truncated);
    let ir = sum_predicate_ir(16, 1, truncated, exposure).replace(
        &format!("a: bits[{width}], b: bits[{width}]"),
        "a: bits[8], b: bits[8]",
    );
    let oracle = ir.replace(
        "  added:",
        &format!(
            "  ax: bits[{width}] = zero_ext(a, new_bit_count={width}, id=20)\n  bx: bits[{width}] = zero_ext(b, new_bit_count={width}, id=21)\n  added:"
        ),
    ).replace("add(a, b,", "add(ax, bx,");
    let package = Parser::new(&oracle).parse_and_validate_package().unwrap();
    let lower = |arch: Option<&str>, default_mapping| {
        let attribute = arch.map(|a| format!(", arch={a}")).unwrap_or_default();
        let ir = ir.replace(
            "add(a, b,",
            &format!(
                "ext_nary_add(a, b, signed=[false, false], negated=[false, false]{attribute},"
            ),
        );
        let lowered = canonical_ir_text_to_g8r_lowering_artifacts(
            &ir,
            Some("main"),
            &CanonicalG8rOptions {
                adder_mapping: default_mapping,
                fraig: false,
                enable_formal_array_alias_analysis: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_carry_out: false,
                ..CanonicalG8rOptions::default()
            },
        )
        .unwrap();
        check_equivalence::validate_same_fn_via_toolchain(
            package.get_top_fn().unwrap(),
            &lowered.gate_fn,
        )
        .unwrap();
        emit_aiger(&lowered.gate_fn, false).unwrap()
    };
    let mut explicit = std::collections::HashSet::new();
    let mut inferred = std::collections::HashSet::new();
    for (arch, mapping) in [
        ("ripple_carry", AdderMapping::RippleCarry),
        ("brent_kung", AdderMapping::BrentKung),
        ("kogge_stone", AdderMapping::KoggeStone),
    ] {
        assert!(
            explicit.insert(lower(Some(arch), AdderMapping::BrentKung)),
            "{arch}"
        );
        if exposure == "high-bit" {
            inferred.insert(lower(None, mapping));
        }
    }
    if exposure == "high-bit" {
        assert_eq!(
            inferred.len(),
            1,
            "unannotated predicates use the carry-free prefix"
        );
    }
}

#[test_matrix([8, 65], ["sign_ext", "zero_ext"], [true, false])]
fn prepared_sum_extensions_are_preserved(width: usize, extension: &str, fold: bool) {
    let ir = sum_predicate_ir(width, 1, false, "high-bit")
        .replace(&format!("a: bits[{width}]"), "a: bits[3]")
        .replace(
            "  added:",
            &format!(
                "  extended: bits[{width}] = {extension}(a, new_bit_count={width}, id=20)\n  added:"
            ),
        )
        .replace("add(a, b,", "add(extended, b,");
    lower_and_prove_with_prep(&ir, fold, true);
}

/// Proves the mapped circuit and measures the predicate at its most significant
/// output bit.
fn predicate_depth(ir: &str, options: &CanonicalG8rOptions) -> usize {
    let lowered = canonical_ir_text_to_g8r_lowering_artifacts(
        ir,
        Some("main"),
        &CanonicalG8rOptions {
            fraig: false,
            enable_formal_array_alias_analysis: false,
            ..options.clone()
        },
    )
    .unwrap();
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let gates = &lowered.gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(package.get_top_fn().unwrap(), gates)
        .unwrap();
    let root = *gates.outputs[0].bit_vector.get_msb(0);
    get_aig_cone_stats(&gates.gates, &[root]).root_depths[0]
}

fn sum_predicate_ir(width: usize, start: usize, truncated: bool, exposure: &str) -> String {
    let input_width = width + usize::from(truncated);
    let encoded_width = ceil_log2(width + 1);
    let slice_width = if exposure == "partial-ordinal" {
        1
    } else {
        encoded_width - start
    };
    // Direct sums exercise the matcher without an intervening slice.
    let sum_line = if truncated {
        format!("  sum: bits[{width}] = bit_slice(added, start=0, width={width}, id=5)\n")
    } else {
        String::new()
    };
    let sum = if truncated { "sum" } else { "added" };
    let (other, other_width) = match exposure {
        "full-selection" => ("selected", width),
        "full-sum" => ("added", input_width),
        "shared-ordinal" => ("ordinal", encoded_width),
        _ => ("high", 1),
    };
    let hot_width = width + 1;
    let high_bit = width - 1;
    format!(
        r#"package sample
 top fn main(a: bits[{input_width}], b: bits[{input_width}], control: bits[1]) -> (bits[1], bits[{other_width}]) {{
  added: bits[{input_width}] = add(a, b, id=4)
{sum_line}  negative: bits[{width}] = neg({sum}, id=6)
  selected: bits[{width}] = sel(control, cases=[{sum}, negative], id=7)
  reversed: bits[{width}] = reverse(selected, id=8)
  hot: bits[{hot_width}] = one_hot(reversed, lsb_prio=true, id=9)
  ordinal: bits[{encoded_width}] = encode(hot, id=10)
  sliced: bits[{slice_width}] = bit_slice(ordinal, start={start}, width={slice_width}, id=11)
  predicate: bits[1] = or_reduce(sliced, id=12)
  high: bits[1] = bit_slice(selected, start={high_bit}, width=1, id=13)
  ret result: (bits[1], bits[{other_width}]) = tuple(predicate, {other}, id=14)
 }}
"#
    )
}

/// Builds either a full ordinal reduction or a reduction of an ordinal slice.
fn priority_ir(width: usize, lsb_prio: bool, slice: Option<(usize, usize)>) -> String {
    let hot_width = width + 1;
    let encoded_width = ceil_log2(hot_width);
    let reduction = if let Some((start, count)) = slice {
        format!(
            r#"  high: bits[{count}] = bit_slice(ordinal, start={start}, width={count}, id=4)
  ret result: bits[1] = or_reduce(high, id=5)"#
        )
    } else {
        "  ret result: bits[1] = or_reduce(ordinal, id=5)".to_string()
    };
    format!(
        r#"package sample
 top fn main(x: bits[{width}]) -> bits[1] {{
  hot: bits[{hot_width}] = one_hot(x, lsb_prio={lsb_prio}, id=2)
  ordinal: bits[{encoded_width}] = encode(hot, id=3)
{reduction}
 }}
"#
    )
}

/// Proves the mapped gates against the original IR using the XLS toolchain.
fn lower_and_prove(ir: &str, fold: bool) -> GateFn {
    lower_and_prove_with_prep(ir, fold, false)
}

fn lower_and_prove_with_prep(ir: &str, fold: bool, nary_prep: bool) -> GateFn {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let gates = gatify(
        function,
        GatifyOptions {
            fold,
            hash: fold,
            enable_rewrite_nary_add: nary_prep,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_with_solver(
        function,
        &gates,
        xlsynth_g8r::test_utils::equivalence_test_solver(),
        None,
    )
    .unwrap();
    gates
}
