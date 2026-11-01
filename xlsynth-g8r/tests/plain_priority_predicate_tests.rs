// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use test_case::test_matrix;
use xlsynth_g8r::aig::gate::{AigNode, GateFn};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::known_bits;
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

#[test_matrix([0, 1])]
fn implied_bit_folds_at_boolean_consumers_of_shared_wires(start: usize) {
    let ir = predicate_ir(9, start, false, "zero")
        .replace("-> bits[1]", "-> bits[6]")
        .replace(
            "  ret result: bits[1] = identity(predicate, id=12)",
            r#"  nz: bits[1] = not(predicate, id=12)
  clear: bits[1] = not(high, id=13)
  a: bits[1] = and(high, nz, id=14)
  b: bits[1] = nand(clear, predicate, id=15)
  c: bits[1] = or(nz, high, id=16)
  d: bits[1] = nor(predicate, clear, id=17)
  ret result: bits[6] = concat(high, predicate, a, b, c, d, id=18)"#,
        );
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let gates = gatify(
        function,
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
    let output = &gates.outputs[0].bit_vector;
    let high = *output.get_msb(0);
    let nonzero = output.get_msb(1).negate();
    assert_eq!(
        output
            .iter_msb_to_lsb()
            .skip(2)
            .copied()
            .collect::<Vec<_>>(),
        [high, nonzero, nonzero, high]
    );
    for (i, id) in (14..=17).enumerate() {
        let gate = output.get_msb(i + 2);
        assert!(
            gates.gates[gate.node.id].get_pir_node_ids().contains(&id),
            "missing consumer {id}"
        );
    }
    if start == 0 {
        assert_eq!(nonzero, high);
        assert!(
            gates.gates[high.node.id].get_pir_node_ids().contains(&10),
            "missing sign-bit reduction"
        );
    }
}

#[test_matrix([true, false])]
fn full_width_zero_test_keeps_independent_slice_wires(variable_selector: bool) {
    let selector = if variable_selector { "control" } else { "off" };
    let ir = format!(
        r#"package sample
top fn main(x: bits[256], control: bits[1]) -> (bits[1], bits[255]) {{
  off: bits[1] = literal(value=0, id=3)
  negative: bits[256] = neg(x, id=4)
  selected: bits[256] = sel({selector}, cases=[x, negative], id=5)
  count: bits[9] = ext_clz(selected, offset=0, new_bit_count=9, id=6)
  threshold: bits[1] = bit_slice(count, start=8, width=1, id=7)
  zero: bits[1] = or_reduce(threshold, id=8)
  slice: bits[255] = bit_slice(selected, start=0, width=255, id=9)
  ret pair: (bits[1], bits[255]) = tuple(zero, slice, id=10)
}}
"#
    );
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let gates = gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap()
    .gate_fn;
    let output = &gates.outputs[0].bit_vector;
    let x = &gates.inputs[0].bit_vector;
    // Conditional negation always preserves the low bit.
    assert_eq!(output.get_lsb(0), x.get_lsb(0));
    if !variable_selector {
        for i in 0..255 {
            assert_eq!(output.get_lsb(i), x.get_lsb(i));
        }
        let stats = get_aig_stats(&gates);
        assert_eq!((stats.and_nodes, stats.max_depth), (255, 8));
    }
}

#[test_matrix([7, 12])]
fn priority_word_constructs_only_observed_prefixes(start: usize) {
    let threshold = 1 << start;
    let width = 2 * threshold;
    let count_width = ceil_log2(width + 1);
    let lower = |output_ty: &str, output_nodes: &str| {
        let ir = format!(
            r#"package sample
top fn main(x: bits[{width}], control: bits[1]) -> {output_ty} {{
  negative: bits[{width}] = neg(x, id=3)
  selected: bits[{width}] = sel(control, cases=[x, negative], id=4)
  count: bits[{count_width}] = ext_clz(selected, offset=0, new_bit_count={count_width}, id=5)
  high: bits[2] = bit_slice(count, start={start}, width=2, id=6)
  {output_nodes}
}}
"#
        );
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        // Keep every binary operation visible in the stored gate count.
        gatify(
            package.get_top_fn().unwrap(),
            GatifyOptions {
                fold: false,
                hash: false,
                ..GatifyOptions::all_opts_disabled()
            },
        )
    };
    let ordinary = lower("bits[2]", "ret result: bits[2] = identity(high, id=7)")
        .unwrap()
        .gate_fn;
    let predicate = lower("bits[1]", "ret result: bits[1] = or_reduce(high, id=7)")
        .unwrap()
        .gate_fn;
    // Three half-word reductions (low nonzero, high zero, high ones), one AND
    // for the selector and a three-AND mux suffice beyond the common lowering.
    let direct_gates = 3 * (threshold - 1) + 1 + 3;
    assert!(
        predicate.gates.len() <= ordinary.gates.len() + direct_gates,
        "stored {} gates beyond the ordinary lowering (budget {direct_gates})",
        predicate.gates.len() - ordinary.gates.len()
    );
    assert!(get_aig_stats(&predicate).and_nodes <= direct_gates);

    for (bit, count) in [(width - 1, 0), (0, 0), (width - 1, 1), (threshold, 1)] {
        let output_ty = format!("(bits[1], bits[{count}])");
        let output_nodes = format!(
            r#"result: bits[1] = or_reduce(high, id=7)
  slice: bits[{count}] = bit_slice(selected, start={bit}, width={count}, id=8)
  ret pair: {output_ty} = tuple(result, slice, id=9)"#
        );
        let sliced = lower(&output_ty, &output_nodes).unwrap().gate_fn;
        // A single observed bit extends the shared low reduction through the
        // gap below it, then uses one AND and one XOR for conditional negation.
        let extra = if count == 0 { 0 } else { bit - threshold + 4 };
        assert!(
            sliced.gates.len() <= predicate.gates.len() + extra,
            "slice [{bit} +: {count}] stored {} gates beyond the predicate (budget {extra})",
            sliced.gates.len() - predicate.gates.len()
        );
        assert!(get_aig_stats(&sliced).and_nodes <= direct_gates + extra);
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

#[test_matrix([63, 64], [true, false])]
fn priority_index_of_exactly_one_word_uses_only_high_bits(z_width: usize, lsb_prio: bool) {
    let width = z_width + 1;
    let hot_width = width + 1;
    let ir = format!(
        r#"package sample
top fn main(z: bits[{z_width}] id=1) -> bits[1] {{
  one: bits[{width}] = one_hot(z, lsb_prio=true, id=2)
  x: bits[{width}] = reverse(one, id=3)
  hot: bits[{hot_width}] = one_hot(x, lsb_prio={lsb_prio}, id=4)
  ordinal: bits[7] = encode(hot, id=5)
  high: bits[1] = bit_slice(ordinal, start=6, width=1, id=6)
  ret predicate: bits[1] = or_reduce(high, id=7)
}}
"#
    );
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let facts = known_bits::analyze_fn(f).unwrap();
    let gates = gatify(
        f,
        GatifyOptions {
            range_info: Some(Arc::new(IrRangeInfo::from_known_bits(&facts))),
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    // The sentinel is unreachable. Only x[64], when present, reaches 64.
    assert_eq!(get_aig_stats(&gates).and_nodes, 0);
    let output = gates.outputs[0].bit_vector.get_lsb(0);
    if width == 64 {
        assert!(
            matches!(gates.gates[output.node.id], AigNode::Literal { value, .. } if value == output.negated)
        );
    } else {
        assert_eq!(output, gates.inputs[0].bit_vector.get_lsb(0));
    }
}

#[test]
fn full_width_nonzero_count_cutoff_retains_reduction_id() {
    let ir = r#"package sample
top fn main(e: bits[1] id=1, z: bits[2] id=2) -> bits[1] {
  one: bits[3] = one_hot(z, lsb_prio=true, id=3)
  x: bits[4] = concat(e, one, id=4)
  hot: bits[5] = one_hot(x, lsb_prio=true, id=5)
  ordinal: bits[3] = encode(hot, id=6)
  high: bits[1] = bit_slice(ordinal, start=2, width=1, id=7)
  ret predicate: bits[1] = or_reduce(high, id=8)
}
"#;
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let facts = known_bits::analyze_fn(f).unwrap();
    let gates = gatify(
        f,
        GatifyOptions {
            range_info: Some(Arc::new(IrRangeInfo::from_known_bits(&facts))),
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let output = gates.outputs[0].bit_vector.get_lsb(0);
    let gate = &gates.gates[output.node.id];
    assert!(matches!(gate, AigNode::Literal { value, .. } if *value == output.negated));
    assert!(gate.get_pir_node_ids().contains(&8));
}

#[test_matrix([false, true])]
fn out_of_range_clz_cutoff_retains_reduction_id(fold_and_hash: bool) {
    let ir = r#"package sample
top fn main(x: bits[8] id=1) -> bits[1] {
  count: bits[5] = ext_clz(x, offset=0, new_bit_count=5, id=2)
  high: bits[1] = bit_slice(count, start=4, width=1, id=3)
  ret predicate: bits[1] = or_reduce(high, id=4)
}
"#;
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let gates = gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions {
            fold: fold_and_hash,
            hash: fold_and_hash,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let output = gates.outputs[0].bit_vector.get_lsb(0);
    let gate = &gates.gates[output.node.id];
    assert!(matches!(gate, AigNode::Literal { value, .. } if *value == output.negated));
    assert!(gate.get_pir_node_ids().contains(&4));
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
