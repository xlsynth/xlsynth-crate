// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::aig_serdes::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::test_utils::Opt;
use xlsynth_pir::ir_parser;

#[derive(Clone, Debug, PartialEq, Eq)]
struct OneHotRow {
    bit_count: u32,
    prio: &'static str,
    live_nodes: usize,
    deepest_path: usize,
}

fn build_one_hot_ir_text(bit_count: u32, lsb_prio: bool) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = match lsb_prio {
        true => format!("one_hot_lsb_{bit_count}b"),
        false => format!("one_hot_msb_{bit_count}b"),
    };
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_bits = package.get_bits_type(u64::from(bit_count));
    let input = fb.param("input", &ty_bits);
    let out = fb.one_hot(&input, lsb_prio, Some("one_hot"));

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
        },
    )
    .unwrap();
    get_summary_stats(&out.gate_fn)
}

fn gather_one_hot_rows() -> Vec<OneHotRow> {
    let mut got: Vec<OneHotRow> = Vec::new();
    for bit_count in 1..=8 {
        for lsb_prio in [true, false] {
            let prio = match lsb_prio {
                true => "lsb",
                false => "msb",
            };
            let ir_text = build_one_hot_ir_text(bit_count, lsb_prio);
            let stats = stats_for_ir_text(&ir_text, Opt::Yes);
            got.push(OneHotRow {
                bit_count,
                prio,
                live_nodes: stats.live_nodes,
                deepest_path: stats.deepest_path,
            });
        }
    }
    got
}

#[test]
fn test_one_hot_gate_stats_sweep_1_to_8() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_one_hot_rows();

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[OneHotRow] = &[
        OneHotRow { bit_count: 1, prio: "lsb", live_nodes: 1, deepest_path: 1 },
        OneHotRow { bit_count: 1, prio: "msb", live_nodes: 1, deepest_path: 1 },
        OneHotRow { bit_count: 2, prio: "lsb", live_nodes: 4, deepest_path: 2 },
        OneHotRow { bit_count: 2, prio: "msb", live_nodes: 4, deepest_path: 2 },
        OneHotRow { bit_count: 3, prio: "lsb", live_nodes: 8, deepest_path: 3 },
        OneHotRow { bit_count: 3, prio: "msb", live_nodes: 8, deepest_path: 3 },
        OneHotRow { bit_count: 4, prio: "lsb", live_nodes: 12, deepest_path: 4 },
        OneHotRow { bit_count: 4, prio: "msb", live_nodes: 12, deepest_path: 4 },
        OneHotRow { bit_count: 5, prio: "lsb", live_nodes: 17, deepest_path: 4 },
        OneHotRow { bit_count: 5, prio: "msb", live_nodes: 17, deepest_path: 4 },
        OneHotRow { bit_count: 6, prio: "lsb", live_nodes: 22, deepest_path: 5 },
        OneHotRow { bit_count: 6, prio: "msb", live_nodes: 22, deepest_path: 5 },
        OneHotRow { bit_count: 7, prio: "lsb", live_nodes: 27, deepest_path: 5 },
        OneHotRow { bit_count: 7, prio: "msb", live_nodes: 27, deepest_path: 5 },
        OneHotRow { bit_count: 8, prio: "lsb", live_nodes: 32, deepest_path: 5 },
        OneHotRow { bit_count: 8, prio: "msb", live_nodes: 32, deepest_path: 5 },
    ];

    assert_eq!(got.as_slice(), want);
}
