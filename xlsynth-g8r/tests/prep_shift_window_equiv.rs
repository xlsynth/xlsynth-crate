// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, score_graph_logical_effort,
};
use xlsynth_g8r::gate_fn_optimize::{GateFnOptimizeOptions, optimize_gate_fn};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;

/// Builds symbolic affine shifts while varying width, threshold and spelling.
fn case(
    base: usize,
    data: usize,
    low: usize,
    offset: usize,
    start: usize,
    width: usize,
    spelling: usize,
) -> String {
    let wide = base + 1;
    let amount_width = wide + low;
    let extension = if spelling % 2 == 0 {
        format!("zero_ext(a, new_bit_count={wide}, id=5)")
    } else {
        "concat(z, a, id=5)".to_string()
    };
    let add = if spelling % 2 == 0 {
        "add(wide, c, id=7)"
    } else {
        "add(c, wide, id=7)"
    };
    let amount = if low == 0 {
        String::new()
    } else {
        format!("amount: bits[{amount_width}] = concat(sum, b, id=8)")
    };
    let shift_arg = if low == 0 { "sum" } else { "amount" };
    let predicate = match spelling % 3 {
        0 => "or_reduce(slice, id=12)",
        1 => "ne(zero, slice, id=12)",
        _ => "eq(slice, zero, id=12)",
    };
    format!(
        r#"package threshold
top fn f(a: bits[{base}] id=1, x: bits[{data}] id=2, b: bits[{low}] id=3) -> bits[1] {{
 z: bits[1] = literal(value=0, id=4)
 wide: bits[{wide}] = {extension}
 c: bits[{wide}] = literal(value={offset}, id=6)
 sum: bits[{wide}] = {add}
 {amount}
 shift: bits[{data}] = shll(x, {shift_arg}, id=9)
 slice: bits[{width}] = bit_slice(shift, start={start}, width={width}, id=10)
 zero: bits[{width}] = literal(value=0, id=11)
 ret result: bits[1] = {predicate}
}}
"#
    )
}

#[test]
fn bounded_shift_windows_are_equivalent_for_symbolic_inputs() {
    for (base, data) in [(4, 8), (5, 33), (65, 65), (129, 129)] {
        for (low, offset, start, width) in [
            (0, 1, 0, 3),
            (0, 1, 0, 5),
            (0, 8, 1, 7),
            (1, 1, 2, 5),
            (1, 2, 0, 8),
            (1, 4, 0, 8),
            (2, 1, 4, 4),
            (2, 2, 0, 8),
        ] {
            for spelling in 0..3 {
                let text = case(base, data, low, offset, start, width, spelling);
                let mut parser = Parser::new(&text);
                let pkg = parser.parse_and_validate_package().expect("valid test IR");
                let source = pkg.get_top_fn().unwrap();
                let prepared =
                    prep_for_gatify(source, None, PrepForGatifyOptions::all_opts_disabled());
                assert!(
                    !prepared.nodes.iter().any(|n| matches!(
                        n.payload,
                        xlsynth_pir::ir::NodePayload::Binop(xlsynth_pir::ir::Binop::Shll, _, _)
                    )),
                    "expected rewrite: {text}"
                );
                let result = prove_prepared_ir_equiv(source, &prepared);
                assert!(
                    matches!(result, EquivResult::Proved),
                    "base={base} data={data} low={low} offset={offset} start={start} width={width} spelling={spelling}: {result:?}"
                );
            }
        }
    }
}

#[test]
fn bounded_shift_fixture_improves_native_aig_without_fraig() {
    let pkg = Parser::new(include_str!("data/shift_window_threshold.ir"))
        .parse_and_validate_package()
        .expect("fixture parses");
    let source = pkg.get_top_fn().unwrap();
    let emitted = gatify(source, GatifyOptions::all_opts_enabled())
        .expect("gatify")
        .gate_fn;
    let native = optimize_gate_fn(
        emitted,
        &GateFnOptimizeOptions {
            fraig: false,
            ..GateFnOptimizeOptions::default()
        },
    )
    .expect("native optimize")
    .gate_fn;
    let stats = get_aig_stats(&native);
    let le = score_graph_logical_effort(
        &native,
        &GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        },
    )
    .delay;
    assert_eq!((stats.and_nodes, stats.max_depth), (19, 6));
    assert!((le - 22.079368399158987).abs() < 1e-8, "graph LE: {le}");
}

#[test]
fn guarded_shift_uses_modular_product() {
    use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
    use xlsynth_pir::IrBits;

    let package = Parser::new(
        "package p
        top fn f(x: bits[17] id=1, k: bits[3] id=2) -> bits[1] {
            zero: bits[3] = literal(value=0, id=3)
            enabled: bits[1] = ne(k, zero, id=4)
            three: bits[3] = literal(value=3, id=5)
            amount: bits[3] = umul(k, three, id=6)
            shifted: bits[17] = shrl(x, amount, id=7)
            chosen: bits[17] = sel(enabled, cases=[x, shifted], id=8)
            observed: bits[7] = bit_slice(chosen, start=2, width=7, id=9)
            ret result: bits[1] = or_reduce(observed, id=10)
        }",
    )
    .parse_and_validate_package()
    .unwrap();
    let gates = gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions::all_opts_enabled(),
    )
    .unwrap();
    for k in 0u64..8 {
        for bit in 0..17 {
            let x = 1u64 << bit;
            let args = [
                IrBits::make_ubits(17, x).unwrap(),
                IrBits::make_ubits(3, k).unwrap(),
            ];
            let actual = gate_sim::eval(&gates.gate_fn, &args, Collect::None);
            assert_eq!(
                actual.outputs[0].get_bit(0).unwrap(),
                ((x >> (3 * k % 8)) >> 2) & 127 != 0,
                "x={x}, k={k}"
            );
        }
    }
}
