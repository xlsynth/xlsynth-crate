// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtNaryAddSubRow {
    bit_count: usize,
    live_nodes: usize,
    deepest_path: usize,
}

fn build_sub_ir_text(bit_count: usize) -> String {
    format!(
        r#"package test

top fn f(lhs: bits[{bit_count}] id=1, rhs: bits[{bit_count}] id=2) -> bits[{bit_count}] {{
  ret sub.3: bits[{bit_count}] = sub(lhs, rhs, id=3)
}}
"#
    )
}

fn build_nary_sub_ir_text(bit_count: usize) -> String {
    format!(
        r#"package test

top fn f(lhs: bits[{bit_count}] id=1, rhs: bits[{bit_count}] id=2) -> bits[{bit_count}] {{
  ret ext_nary_add.3: bits[{bit_count}] = ext_nary_add(lhs, rhs, signed=[false, false], negated=[false, true], arch=ripple_carry, id=3)
}}
"#
    )
}

fn get_ir_gate_stats(ir_text: &str) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            adder_mapping: AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    let stats = get_summary_stats(&out.gatify_output.gate_fn);
    (stats.live_nodes, stats.deepest_path)
}

fn gather_ext_nary_add_sub_rows() -> Vec<ExtNaryAddSubRow> {
    let mut got = Vec::new();
    for bit_count in 1..=16 {
        let sub_stats = get_ir_gate_stats(&build_sub_ir_text(bit_count));
        let nary_sub_stats = get_ir_gate_stats(&build_nary_sub_ir_text(bit_count));
        assert_eq!(
            sub_stats, nary_sub_stats,
            "expected explicit ext_nary_add(lhs, -rhs) to match dedicated sub lowering for {bit_count}-bit operands"
        );
        got.push(ExtNaryAddSubRow {
            bit_count,
            live_nodes: sub_stats.0,
            deepest_path: sub_stats.1,
        });
    }
    got
}

#[test]
fn test_ext_nary_add_sub_gate_stats_sweep_1_to_16() {
    let got = gather_ext_nary_add_sub_rows();

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth for `sub` / `ext_nary_add(lhs, -rhs)`, so we can notice
    // regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddSubRow] = &[
        ExtNaryAddSubRow { bit_count: 1, live_nodes: 5, deepest_path: 3 },
        ExtNaryAddSubRow { bit_count: 2, live_nodes: 14, deepest_path: 5 },
        ExtNaryAddSubRow { bit_count: 3, live_nodes: 27, deepest_path: 7 },
        ExtNaryAddSubRow { bit_count: 4, live_nodes: 40, deepest_path: 10 },
        ExtNaryAddSubRow { bit_count: 5, live_nodes: 53, deepest_path: 13 },
        ExtNaryAddSubRow { bit_count: 6, live_nodes: 66, deepest_path: 16 },
        ExtNaryAddSubRow { bit_count: 7, live_nodes: 79, deepest_path: 19 },
        ExtNaryAddSubRow { bit_count: 8, live_nodes: 92, deepest_path: 22 },
        ExtNaryAddSubRow { bit_count: 9, live_nodes: 105, deepest_path: 25 },
        ExtNaryAddSubRow { bit_count: 10, live_nodes: 118, deepest_path: 28 },
        ExtNaryAddSubRow { bit_count: 11, live_nodes: 131, deepest_path: 31 },
        ExtNaryAddSubRow { bit_count: 12, live_nodes: 144, deepest_path: 34 },
        ExtNaryAddSubRow { bit_count: 13, live_nodes: 157, deepest_path: 37 },
        ExtNaryAddSubRow { bit_count: 14, live_nodes: 170, deepest_path: 40 },
        ExtNaryAddSubRow { bit_count: 15, live_nodes: 183, deepest_path: 43 },
        ExtNaryAddSubRow { bit_count: 16, live_nodes: 196, deepest_path: 46 },
    ];

    assert_eq!(got.as_slice(), want);
}
