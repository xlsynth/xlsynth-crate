// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{Binop, NodePayload};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::prove_ir_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};

/// Exercises full/interior windows, overshifts, selector polarity and zero
/// tests.
#[test]
fn shift_survival_predicates_prove_symbolic_data_and_amounts() {
    for (n, aw, start, width) in [
        (2, 1, 0, 2),
        (3, 1, 0, 3),
        (7, 2, 0, 7),
        (8, 3, 0, 8),
        (23, 3, 0, 23),
        (24, 5, 0, 23),
        (32, 3, 4, 23),
        (65, 5, 0, 65),
        (2, 65, 0, 2),
        (3, 6, 0, 3),
        (16, 9, 0, 16),
        (22, 32, 0, 22),
        (23, 129, 0, 23),
        (31, 6, 0, 31),
        (32, 65, 0, 32),
    ] {
        for op in ["shll", "shrl"] {
            for shape in 0..4 {
                let selection = match shape {
                    0 => format!("identity(shifted, id=5)"),
                    1 => format!("sel(s, cases=[shifted, y], id=5)"),
                    2 => format!("priority_sel(s, cases=[y], default=shifted, id=5)"),
                    _ => format!("priority_sel(s, cases=[shifted], default=y, id=5)"),
                };
                let predicate = match shape {
                    0 => "or_reduce(window, id=8)",
                    1 => "ne(window, zero, id=8)",
                    2 => "eq(zero, window, id=8)",
                    _ => "ne(zero, window, id=8)",
                };
                // The direct case deliberately avoids an identity wrapper.
                let selection = if shape == 0 {
                    String::new()
                } else {
                    format!("chosen: bits[{n}] = {selection}")
                };
                let source = if shape == 0 { "shifted" } else { "chosen" };
                let text = format!(
                    r#"package survival
top fn main(x: bits[{n}] id=1, k: bits[{aw}] id=2, y: bits[{n}] id=3, s: bits[1] id=4) -> bits[1] {{
 shifted: bits[{n}] = {op}(x, k, id=9)
 {selection}
 window: bits[{width}] = bit_slice({source}, start={start}, width={width}, id=6)
 zero: bits[{width}] = literal(value=0, id=7)
 ret result: bits[1] = {predicate}
}}"#
                );
                let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
                let source = pkg.get_top_fn().unwrap();
                let prepared =
                    prep_for_gatify(source, None, PrepForGatifyOptions::all_opts_disabled());
                assert!(
                    !prepared.nodes.iter().any(|n| matches!(
                        n.payload,
                        NodePayload::Binop(Binop::Shll | Binop::Shrl, _, _)
                    )),
                    "{text}"
                );
                let result = prove_ir_equiv(
                    &ProverFn::new(source, None),
                    &ProverFn::new(&prepared, None),
                    EquivParallelism::SingleThreaded,
                    AssertionSemantics::Ignore,
                    None,
                    false,
                );
                assert!(matches!(result, EquivResult::Proved), "{text}\n{result:?}");
            }
        }
    }
}

/// Keeps the large select-of-shifts opportunity small and reproducible.
#[test]
fn selected_shift_nonzero_fixture_has_small_native_graph() {
    let text = include_str!("data/selected_shift_nonzero.ir");
    let artifacts = xlsynth_g8r::process_ir_path::canonical_ir_text_to_g8r_lowering_artifacts(
        text,
        Some("main"),
        &xlsynth_g8r::process_ir_path::CanonicalG8rOptions {
            fraig: false,
            ..Default::default()
        },
    )
    .unwrap();
    let stats = xlsynth_g8r::aig::get_summary_stats::get_aig_stats(&artifacts.gate_fn);
    assert_eq!((stats.and_nodes, stats.max_depth), (139, 10));
    let le = xlsynth_g8r::aig::graph_logical_effort::score_graph_logical_effort(
        &artifacts.gate_fn,
        &xlsynth_g8r::aig::graph_logical_effort::GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        },
    )
    .delay;
    assert!(le < 43.0, "{le}");
}

/// Checks the data-width boundary separately from the low/high amount split.
#[test]
fn wide_shift_amount_boundaries_return_the_expected_nonzero_bit() {
    for n in [3usize, 22, 32] {
        for op in ["shll", "shrl"] {
            let text = format!(
                r#"package boundaries
top fn main(x: bits[{n}] id=1, k: bits[65] id=2) -> bits[1] {{
 shifted: bits[{n}] = {op}(x, k, id=3)
 ret out: bits[1] = or_reduce(shifted, id=4)
}}"#
            );
            let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
            let prepared = prep_for_gatify(
                pkg.get_top_fn().unwrap(),
                None,
                PrepForGatifyOptions::all_opts_disabled(),
            );
            assert!(!prepared.nodes.iter().any(|node| matches!(
                node.payload,
                NodePayload::Binop(Binop::Shll | Binop::Shrl, _, _)
            )));
            let mask = (1u64 << n) - 1;
            for x in [0, 1, 1u64 << (n - 1), mask] {
                for k in [
                    0,
                    n as u128 - 1,
                    n as u128,
                    n as u128 + 1,
                    31,
                    32,
                    1u128 << 64,
                ] {
                    let args = [
                        IrValue::make_ubits(n, x).unwrap(),
                        IrValue::parse_typed(&format!("bits[65]:{k}")).unwrap(),
                    ];
                    let expected = k < n as u128
                        && if op == "shll" {
                            ((x << k) & mask) != 0
                        } else {
                            (x >> k) != 0
                        };
                    let FnEvalResult::Success(result) = eval_fn(&prepared, &args) else {
                        panic!("evaluation failed: {op}, width={n}, x={x}, k={k}");
                    };
                    assert_eq!(
                        result.value.to_bool().unwrap(),
                        expected,
                        "{op}, width={n}, x={x}, k={k}"
                    );
                }
            }
        }
    }
}

/// Covers the useful wide-amount match and a native fold it must leave alone.
#[test]
fn wide_amount_native_graphs_preserve_profitability_boundaries() {
    for (text, max_ands, max_depth) in [
        (
            r#"package wide_selected
top fn main(s: bits[1] id=1, x: bits[22] id=2, k: bits[6] id=3) -> bits[1] {
 shifted: bits[22] = shrl(x, k, id=4)
 nonzero: bits[22] = literal(value=3145728, id=5)
 selected: bits[22] = sel(s, cases=[shifted, nonzero], id=6)
 zero: bits[22] = literal(value=0, id=7)
 ret result: bits[1] = eq(selected, zero, id=8)
}"#,
            100,
            10,
        ),
        (
            r#"package onehot_fold
top fn main(x: bits[15] id=1, k: bits[1] id=2) -> bits[1] {
 raw_data: bits[16] = one_hot(x, lsb_prio=true, id=3)
 data: bits[16] = bit_slice(raw_data, start=0, width=16, id=4)
 amount: bits[6] = zero_ext(k, new_bit_count=6, id=5)
 shifted: bits[16] = shrl(data, amount, id=6)
 ret result: bits[1] = or_reduce(shifted, id=7)
}"#,
            1,
            1,
        ),
        (
            r#"package fixed_left
top fn main(x: bits[16] id=1, k: bits[2] id=2) -> bits[1] {
 high: bits[3] = literal(value=1, id=3)
 low: bits[1] = literal(value=0, id=4)
 amount: bits[6] = concat(high, k, low, id=5)
 shifted: bits[16] = shll(x, amount, id=6)
 ret result: bits[1] = or_reduce(shifted, id=7)
}"#,
            11,
            5,
        ),
        (
            r#"package fixed_right
top fn main(x: bits[22] id=1, k: bits[2] id=2) -> bits[1] {
 high: bits[3] = literal(value=1, id=3)
 low: bits[1] = literal(value=1, id=4)
 amount: bits[6] = concat(high, k, low, id=5)
 shifted: bits[22] = shrl(x, amount, id=6)
 ret result: bits[1] = or_reduce(shifted, id=7)
}"#,
            16,
            5,
        ),
    ] {
        let artifacts = xlsynth_g8r::process_ir_path::canonical_ir_text_to_g8r_lowering_artifacts(
            text,
            Some("main"),
            &xlsynth_g8r::process_ir_path::CanonicalG8rOptions {
                fraig: false,
                ..Default::default()
            },
        )
        .unwrap();
        let stats = xlsynth_g8r::aig::get_summary_stats::get_aig_stats(&artifacts.gate_fn);
        assert!(
            stats.and_nodes <= max_ands && stats.max_depth <= max_depth,
            "{stats:?}"
        );
    }
}
