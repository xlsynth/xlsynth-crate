// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth::IrValue;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::test_utils::Opt;
use xlsynth_pir::ir_parser;

#[derive(Clone, Debug, PartialEq, Eq)]
struct AddPow2Minus1Row {
    bit_count: u32,
    live_nodes_baseline: usize,
    deepest_path_baseline: usize,
    live_nodes_pow2m1: usize,
    deepest_path_pow2m1: usize,
}

fn stats_for_ir_text(ir_text: &str, opt: Opt) -> SummaryStats {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().expect("parse package");
    let ir_fn = ir_package.get_top_fn().expect("top fn");
    let out = gatify(
        &ir_fn,
        GatifyOptions {
            fold: opt == Opt::Yes,
            check_equivalence: false,
            hash: opt == Opt::Yes,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    )
    .expect("gatify");
    get_summary_stats(&out.gate_fn)
}

fn build_add_ir_text(bit_count: u32, rhs_literal: Option<u64>) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = match rhs_literal {
        Some(rhs) => format!("add_{}b_rhs_{rhs}", bit_count),
        None => format!("add_{}b_rhs_param", bit_count),
    };
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_bits = package.get_bits_type(u64::from(bit_count));
    let lhs = fb.param("lhs", &ty_bits);
    let rhs = match rhs_literal {
        Some(rhs) => {
            let v = IrValue::make_ubits(bit_count as usize, rhs).expect("make ubits");
            fb.literal(&v, Some("rhs"))
        }
        None => fb.param("rhs", &ty_bits),
    };

    let out = fb.add(&lhs, &rhs, Some("sum"));
    let _ = fb.build_with_return_value(&out).expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    package.to_string()
}

fn gather_add_pow2m1_rows() -> Vec<AddPow2Minus1Row> {
    let mut got = Vec::new();

    for bit_count in 1..=8 {
        let baseline_ir_text = build_add_ir_text(bit_count, None);
        let baseline_stats = stats_for_ir_text(&baseline_ir_text, Opt::Yes);

        let rhs_k = bit_count - 1;
        let rhs_literal = if rhs_k == 0 { 0 } else { (1u64 << rhs_k) - 1 };
        let pow2m1_ir_text = build_add_ir_text(bit_count, Some(rhs_literal));
        let pow2m1_stats = stats_for_ir_text(&pow2m1_ir_text, Opt::Yes);

        got.push(AddPow2Minus1Row {
            bit_count,
            live_nodes_baseline: baseline_stats.live_nodes,
            deepest_path_baseline: baseline_stats.deepest_path,
            live_nodes_pow2m1: pow2m1_stats.live_nodes,
            deepest_path_pow2m1: pow2m1_stats.deepest_path,
        });
    }

    got
}

#[test]
fn test_add_const_pow2m1_gate_stats_sweep_1_to_8() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_add_pow2m1_rows();

    for row in &got {
        assert!(
            row.live_nodes_pow2m1 <= row.live_nodes_baseline,
            "expected <= baseline live_nodes for {}b, got baseline={} pow2m1={}",
            row.bit_count,
            row.live_nodes_baseline,
            row.live_nodes_pow2m1
        );
    }

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[AddPow2Minus1Row] = &[
        AddPow2Minus1Row { bit_count: 1, live_nodes_baseline: 5, deepest_path_baseline: 3, live_nodes_pow2m1: 1, deepest_path_pow2m1: 1 },
        AddPow2Minus1Row { bit_count: 2, live_nodes_baseline: 14, deepest_path_baseline: 5, live_nodes_pow2m1: 5, deepest_path_pow2m1: 3 },
        AddPow2Minus1Row { bit_count: 3, live_nodes_baseline: 27, deepest_path_baseline: 7, live_nodes_pow2m1: 10, deepest_path_pow2m1: 4 },
        AddPow2Minus1Row { bit_count: 4, live_nodes_baseline: 40, deepest_path_baseline: 10, live_nodes_pow2m1: 15, deepest_path_pow2m1: 5 },
        AddPow2Minus1Row { bit_count: 5, live_nodes_baseline: 53, deepest_path_baseline: 13, live_nodes_pow2m1: 20, deepest_path_pow2m1: 6 },
        AddPow2Minus1Row { bit_count: 6, live_nodes_baseline: 66, deepest_path_baseline: 16, live_nodes_pow2m1: 25, deepest_path_pow2m1: 7 },
        AddPow2Minus1Row { bit_count: 7, live_nodes_baseline: 79, deepest_path_baseline: 19, live_nodes_pow2m1: 30, deepest_path_pow2m1: 8 },
        AddPow2Minus1Row { bit_count: 8, live_nodes_baseline: 92, deepest_path_baseline: 22, live_nodes_pow2m1: 35, deepest_path_pow2m1: 9 },
    ];

    assert_eq!(got.as_slice(), want);
}
