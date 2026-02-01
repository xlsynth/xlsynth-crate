// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::test_utils::Opt;
use xlsynth_pir::ir_parser;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ShraRow {
    width: u32,
    amount_width: u32,
    live_nodes: usize,
    deepest_path: usize,
}

fn build_shra_ir_text(width: u32, amount_width: u32) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = format!("shra_w{width}_amt{amount_width}");
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_x = package.get_bits_type(u64::from(width));
    let ty_amt = package.get_bits_type(u64::from(amount_width));

    let x = fb.param("x", &ty_x);
    let amt = fb.param("amt", &ty_amt);

    let out = fb.shra(&x, &amt, Some("shra"));
    let _ = fb.build_with_return_value(&out).expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    package.to_string()
}

fn stats_for_ir_text(ir_text: &str, opt: Opt) -> SummaryStats {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().unwrap();
    let ir_fn = ir_package.get_top_fn().unwrap();
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
    .unwrap();
    get_summary_stats(&out.gate_fn)
}

fn gather_shra_rows() -> Vec<ShraRow> {
    let mut got: Vec<ShraRow> = Vec::new();
    for width in 1..=8 {
        for amount_width in 1..=4 {
            let ir_text = build_shra_ir_text(width, amount_width);
            let stats = stats_for_ir_text(&ir_text, Opt::Yes);
            got.push(ShraRow {
                width,
                amount_width,
                live_nodes: stats.live_nodes,
                deepest_path: stats.deepest_path,
            });
        }
    }
    got
}

#[test]
fn test_shra_gate_stats_sweep_w1_to_w8_amt1_to_amt4() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_shra_rows();

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ShraRow] = &[
        ShraRow { width: 1, amount_width: 1, live_nodes: 5, deepest_path: 3 },
        ShraRow { width: 1, amount_width: 2, live_nodes: 7, deepest_path: 4 },
        ShraRow { width: 1, amount_width: 3, live_nodes: 9, deepest_path: 5 },
        ShraRow { width: 1, amount_width: 4, live_nodes: 11, deepest_path: 5 },

        ShraRow { width: 2, amount_width: 1, live_nodes: 8, deepest_path: 3 },
        ShraRow { width: 2, amount_width: 2, live_nodes: 14, deepest_path: 5 },
        ShraRow { width: 2, amount_width: 3, live_nodes: 16, deepest_path: 5 },
        ShraRow { width: 2, amount_width: 4, live_nodes: 18, deepest_path: 5 },

        ShraRow { width: 3, amount_width: 1, live_nodes: 12, deepest_path: 3 },
        ShraRow { width: 3, amount_width: 2, live_nodes: 28, deepest_path: 7 },
        ShraRow { width: 3, amount_width: 3, live_nodes: 30, deepest_path: 7 },
        ShraRow { width: 3, amount_width: 4, live_nodes: 32, deepest_path: 7 },

        ShraRow { width: 4, amount_width: 1, live_nodes: 16, deepest_path: 3 },
        ShraRow { width: 4, amount_width: 2, live_nodes: 27, deepest_path: 5 },
        ShraRow { width: 4, amount_width: 3, live_nodes: 37, deepest_path: 7 },
        ShraRow { width: 4, amount_width: 4, live_nodes: 39, deepest_path: 7 },

        ShraRow { width: 5, amount_width: 1, live_nodes: 20, deepest_path: 3 },
        ShraRow { width: 5, amount_width: 2, live_nodes: 34, deepest_path: 5 },
        ShraRow { width: 5, amount_width: 3, live_nodes: 65, deepest_path: 9 },
        ShraRow { width: 5, amount_width: 4, live_nodes: 67, deepest_path: 9 },

        ShraRow { width: 6, amount_width: 1, live_nodes: 24, deepest_path: 3 },
        ShraRow { width: 6, amount_width: 2, live_nodes: 41, deepest_path: 5 },
        ShraRow { width: 6, amount_width: 3, live_nodes: 76, deepest_path: 9 },
        ShraRow { width: 6, amount_width: 4, live_nodes: 78, deepest_path: 9 },

        ShraRow { width: 7, amount_width: 1, live_nodes: 28, deepest_path: 3 },
        ShraRow { width: 7, amount_width: 2, live_nodes: 48, deepest_path: 5 },
        ShraRow { width: 7, amount_width: 3, live_nodes: 83, deepest_path: 9 },
        ShraRow { width: 7, amount_width: 4, live_nodes: 85, deepest_path: 9 },

        ShraRow { width: 8, amount_width: 1, live_nodes: 32, deepest_path: 3 },
        ShraRow { width: 8, amount_width: 2, live_nodes: 55, deepest_path: 5 },
        ShraRow { width: 8, amount_width: 3, live_nodes: 76, deepest_path: 7 },
        ShraRow { width: 8, amount_width: 4, live_nodes: 94, deepest_path: 9 },
    ];

    assert_eq!(got.as_slice(), want);
}
