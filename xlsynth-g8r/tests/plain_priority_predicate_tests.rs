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

#[test_matrix([3, 8, 9, 65, 161], [true, false], [true, false])]
fn ordinal_threshold_uses_only_the_observed_region(width: usize, lowest: bool, reverse: bool) {
    let encoded = ceil_log2(width + 1);
    let input = if reverse { "reversed" } else { "x" };
    for start in 0..encoded {
        let cutoff = 1 << start;
        let hot_width = width + 1;
        let slice_width = encoded - start;
        let ir = format!(
            r#"package sample
top fn main(x: bits[{width}] id=1) -> bits[1] {{
  reversed: bits[{width}] = reverse(x, id=2)
  hot: bits[{hot_width}] = one_hot({input}, lsb_prio={lowest}, id=3)
  ordinal: bits[{encoded}] = encode(hot, id=4)
  high: bits[{slice_width}] = bit_slice(ordinal, start={start}, width={slice_width}, id=5)
  ret predicate: bits[1] = or_reduce(high, id=6)
}}
"#
        );
        let gates = lower_and_prove(&ir, true);
        let stats = get_aig_stats(&gates);
        let bound = if lowest { cutoff - 1 } else { width - 1 };
        assert!(
            stats.and_nodes <= bound,
            "width={width} start={start} {stats:?}"
        );
        // All-zero input has ordinal N under either priority convention.
        assert_eq!(
            PreparedGateSim::new(&gates).eval_outputs(&[IrBits::make_ubits(width, 0).unwrap()]),
            vec![IrBits::bool(true)]
        );
    }
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

/// A true leading/trailing count threshold rules out each set bit in its zero
/// region.
#[test_matrix([1, 3, 5, 8], [false, true])]
fn ordinal_zero_region_excludes_its_required_bits(width: usize, leading: bool) {
    for start in 0..ceil_log2(width + 1) {
        let cutoff = 1 << start;
        for bit in 0..width {
            let inside = if leading {
                bit >= width - cutoff
            } else {
                bit < cutoff
            };
            for nand in [false, true] {
                let ir = required_bit_ir(width, start, bit, leading, true, "observed", nand);
                let gates = lower_and_prove(&ir, true);
                if inside {
                    assert_eq!(get_aig_stats(&gates).and_nodes, 0);
                }
                let value = IrBits::make_ubits(width, 1 << bit).unwrap();
                let mut sim = PreparedGateSim::new(&gates);
                let outputs = sim.eval_outputs(&[value.clone(), value, IrBits::bool(true)]);
                assert_eq!(outputs, vec![IrBits::bool(inside == nand)]);
            }
        }
    }
}

#[test_matrix([false, true])]
fn ordinal_zero_region_preserves_unrelated_observations(nand: bool) {
    let width = 5;
    for (leading, lowest, name, bit, x, y) in [
        (true, true, "other", 4, 0, 16),
        (true, true, "inverted", 4, 0, 0),
        (false, false, "observed", 4, 31, 0),
        (false, false, "observed", 0, 31, 0),
    ] {
        let ir = required_bit_ir(width, 1, bit, leading, lowest, name, nand);
        let gates = lower_and_prove(&ir, true);
        let mut sim = PreparedGateSim::new(&gates);
        let inputs = [
            IrBits::make_ubits(width, x).unwrap(),
            IrBits::make_ubits(width, y).unwrap(),
            IrBits::bool(true),
        ];
        assert_eq!(sim.eval_outputs(&inputs), vec![IrBits::bool(!nand)]);
    }
    // Truncating the zero-input ordinal 5 to two bits yields 1, below the
    // cutoff 2.
    let truncated = required_bit_ir(width, 1, 4, true, true, "other", nand).replace(
        "high: bits[2] = bit_slice(ordinal, start=1, width=2, id=7)",
        "short: bits[2] = bit_slice(ordinal, start=0, width=2, id=13)\n  high: bits[1] = bit_slice(short, start=1, width=1, id=7)",
    );
    let gates = lower_and_prove(&truncated, true);
    let mut sim = PreparedGateSim::new(&gates);
    assert_eq!(
        sim.eval_outputs(&[
            IrBits::make_ubits(width, 0).unwrap(),
            IrBits::make_ubits(width, 16).unwrap(),
            IrBits::bool(true)
        ]),
        vec![IrBits::bool(nand)],
    );
    let ir = required_bit_ir(width, 1, 4, true, true, "observed", nand);
    assert!(get_aig_stats(&lower_and_prove(&ir, false)).and_nodes > 0);
}

fn required_bit_ir(
    width: usize,
    start: usize,
    bit: usize,
    leading: bool,
    lowest: bool,
    observation: &str,
    nand: bool,
) -> String {
    let encoded = ceil_log2(width + 1);
    let hot = width + 1;
    let slice = encoded - start;
    let input = if leading { "rev" } else { "x" };
    let observer = if width == 1 && observation == "observed" {
        "x"
    } else {
        observation
    };
    let op = if nand { "nand" } else { "and" };
    let args = if nand {
        format!("{observer}, enable, threshold")
    } else {
        format!("threshold, enable, {observer}")
    };
    format!(
        r#"package sample
top fn main(x: bits[{width}] id=1, y: bits[{width}] id=2, enable: bits[1] id=3) -> bits[1] {{
  rev: bits[{width}] = reverse(x, id=4)
  hot: bits[{hot}] = one_hot({input}, lsb_prio={lowest}, id=5)
  ordinal: bits[{encoded}] = encode(hot, id=6)
  high: bits[{slice}] = bit_slice(ordinal, start={start}, width={slice}, id=7)
  threshold: bits[1] = or_reduce(high, id=8)
  observed: bits[1] = bit_slice(x, start={bit}, width=1, id=9)
  other: bits[1] = bit_slice(y, start={bit}, width=1, id=10)
  inverted: bits[1] = not(observed, id=11)
  ret output: bits[1] = {op}({args}, id=12)
}}
"#
    )
}
