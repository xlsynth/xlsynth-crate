// SPDX-License-Identifier: Apache-2.0

use test_case::test_matrix;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

#[test_matrix([1, 2, 3, 8, 9, 65, 161], [true, false])]
fn plain_priority_predicate_preserves_values_and_zero_ordinal(width: usize, fold: bool) {
    for start in 0..ceil_log2(width + 1) {
        let ir = predicate_ir(width, start, false, "predicate");
        let gates = lower_and_prove(&ir, fold);
        let mut simulation = PreparedGateSim::new(&gates);
        for control in [false, true] {
            // The no-set-bit ordinal must not be confused with ordinal zero.
            assert_eq!(
                simulation
                    .eval_outputs(&[IrBits::make_ubits(width, 0).unwrap(), IrBits::bool(control),]),
                vec![IrBits::bool(false)]
            );
        }
    }
}

#[test_matrix([2, 3, 8, 9, 65, 161], [true, false])]
fn high_zero_predicate_preserves_shared_observations(width: usize, fold: bool) {
    for exposure in ["zero", "nonzero", "zero-shared-high", "high-set"] {
        for start in 0..ceil_log2(width + 1) {
            let gates = lower_and_prove(&predicate_ir(width, start, false, exposure), fold);
            let mut simulation = PreparedGateSim::new(&gates);
            for control in [false, true] {
                assert_eq!(
                    simulation.eval_outputs(&[
                        IrBits::make_ubits(width, 0).unwrap(),
                        IrBits::bool(control),
                    ]),
                    vec![IrBits::bool(matches!(
                        exposure,
                        "zero" | "zero-shared-high"
                    ))]
                );
            }
        }
    }
}

#[test_matrix([9, 26, 65, 161], [false, true])]
fn direct_high_zero_predicate_shares_its_implied_high_bit(width: usize, packed: bool) {
    for exposure in ["zero", "zero-shared-high"] {
        let gates = lower_and_prove(&predicate_ir(width, 1, packed, exposure), true);
        let stats = get_aig_stats(&gates);
        // The shared-bit test is implied by the predicate and needs no extra
        // gates.
        assert!(
            stats.and_nodes <= width + 3 && stats.max_depth <= ceil_log2(width - 2) + 3,
            "high-zero observation retained redundant shared-bit logic: {stats:?}"
        );
    }
}

#[test_matrix([true, false])]
fn constant_priority_selections_preserve_observations(fold: bool) {
    let ir = predicate_ir(9, 1, false, "zero-shared-high");
    let constant_value = ir.replace("x: bits[9] id=1, ", "").replace(
        "  negative:",
        "  x: bits[9] = literal(value=0, id=1)\n  negative:",
    );
    lower_and_prove(&constant_value, fold);
    let constant_control = ir.replace(", control: bits[1] id=2", "").replace(
        "  negative:",
        "  control: bits[1] = literal(value=1, id=2)\n  negative:",
    );
    lower_and_prove(&constant_control, fold);
}

#[test_matrix([65, 161], [true, false])]
fn canonical_plain_priority_predicate_uses_observed_prefixes(width: usize, packed: bool) {
    let ir = predicate_ir(width, 1, packed, "predicate");
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
    // Two observed bits need one reduction tree and constant-sized selection
    // logic.
    assert!(
        stats.and_nodes <= 2 * width + 12 && stats.max_depth <= ceil_log2(width) + 8,
        "plain priority predicate retains unnecessary prefix logic: {stats:?}"
    );
}

#[test_matrix(
    ["full-selection", "shared-ordinal", "shared-negation", "low-bit", "partial-ordinal"],
    [true, false]
)]
fn shared_plain_priority_consumers_are_preserved(exposure: &str, fold: bool) {
    lower_and_prove(&predicate_ir(65, 1, false, exposure), fold);
}

#[test_matrix([true, false])]
fn opposite_priority_direction_preserves_ordinal_semantics(fold: bool) {
    let ir = predicate_ir(9, 1, false, "predicate").replace("lsb_prio=true", "lsb_prio=false");
    lower_and_prove(&ir, fold);
}

/// Builds the predicate with independent or packed operand and selector inputs.
fn predicate_ir(width: usize, start: usize, packed: bool, exposure: &str) -> String {
    let encoded_width = ceil_log2(width + 1);
    let hot_width = width + 1;
    let high_bit = width - 1;
    let slice_width = if exposure == "partial-ordinal" {
        1
    } else {
        encoded_width - start
    };
    let (params, input_lines) = if packed {
        (
            format!("packed: bits[{hot_width}] id=1"),
            format!(
                "  x: bits[{width}] = bit_slice(packed, start=0, width={width}, id=2)\n  control: bits[1] = bit_slice(packed, start={width}, width=1, id=3)\n"
            ),
        )
    } else {
        (
            format!("x: bits[{width}] id=1, control: bits[1] id=2"),
            String::new(),
        )
    };
    let (other, other_width) = match exposure {
        "full-selection" => ("selected", width),
        "shared-ordinal" => ("ordinal", encoded_width),
        "shared-negation" => ("negative", width),
        "low-bit" => ("low", 1),
        _ => ("", 0),
    };
    let low_line = if exposure == "low-bit" {
        "  low: bits[1] = bit_slice(selected, start=0, width=1, id=13)\n"
    } else {
        ""
    };
    let (return_ty, return_line) = if other.is_empty() {
        (
            "bits[1]".to_string(),
            match exposure {
                "zero" => "  ret result: bits[1] = identity(predicate, id=12)",
                "nonzero" => "  ret result: bits[1] = not(predicate, id=12)",
                "zero-shared-high" => "  inverse: bits[1] = not(predicate, id=15)\n  ret result: bits[1] = nor(high, inverse, id=12)",
                "high-set" => "  inverse: bits[1] = not(high, id=15)\n  ret result: bits[1] = nor(inverse, predicate, id=12)",
                _ => "  ret result: bits[1] = nor(high, predicate, id=12)",
            }.to_string(),
        )
    } else {
        (
            format!("(bits[1], bits[{other_width}])"),
            format!(
                "  result: bits[1] = nor(high, predicate, id=12)\n{low_line}  ret pair: (bits[1], bits[{other_width}]) = tuple(result, {other}, id=14)"
            ),
        )
    };
    format!(
        r#"package sample
top fn main({params}) -> {return_ty} {{
{input_lines}  negative: bits[{width}] = neg(x, id=4)
  selected: bits[{width}] = sel(control, cases=[x, negative], id=5)
  reversed: bits[{width}] = reverse(selected, id=6)
  hot: bits[{hot_width}] = one_hot(reversed, lsb_prio=true, id=7)
  ordinal: bits[{encoded_width}] = encode(hot, id=8)
  sliced: bits[{slice_width}] = bit_slice(ordinal, start={start}, width={slice_width}, id=9)
  predicate: bits[1] = or_reduce(sliced, id=10)
  high: bits[1] = bit_slice(selected, start={high_bit}, width=1, id=11)
{return_line}
}}
"#
    )
}

/// Proves the mapped gates against the original IR using the XLS toolchain.
fn lower_and_prove(ir: &str, fold: bool) -> GateFn {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let gates = gatify(
        function,
        GatifyOptions {
            fold,
            hash: fold,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
    gates
}
