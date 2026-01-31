// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::test_utils::Opt;
use xlsynth_pir::ir_parser;

#[derive(Clone, Debug, PartialEq, Eq)]
struct EncodeOneHotRow {
    bit_count: u32,
    prio: &'static str,
    live_nodes: usize,
    deepest_path: usize,
}

fn build_encode_one_hot_ir_text(bit_count: u32, lsb_prio: bool) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = match lsb_prio {
        true => format!("encode_one_hot_lsb_{bit_count}b"),
        false => format!("encode_one_hot_msb_{bit_count}b"),
    };
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_bits = package.get_bits_type(u64::from(bit_count));
    let input = fb.param("input", &ty_bits);
    let one_hot = fb.one_hot(&input, lsb_prio, Some("one_hot"));
    let encoded = fb.encode(&one_hot, Some("encode"));

    let _ = fb
        .build_with_return_value(&encoded)
        .expect("build function");
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

fn gather_encode_one_hot_rows() -> Vec<EncodeOneHotRow> {
    let mut got: Vec<EncodeOneHotRow> = Vec::new();
    for bit_count in 1..=16 {
        for lsb_prio in [true, false] {
            let prio = match lsb_prio {
                true => "lsb",
                false => "msb",
            };
            let ir_text = build_encode_one_hot_ir_text(bit_count, lsb_prio);
            let stats = stats_for_ir_text(&ir_text, Opt::Yes);
            got.push(EncodeOneHotRow {
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
fn test_encode_one_hot_gate_stats_sweep_1_to_16() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_encode_one_hot_rows();

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[EncodeOneHotRow] = &[
        EncodeOneHotRow { bit_count: 1, prio: "lsb", live_nodes: 1, deepest_path: 1 },
        EncodeOneHotRow { bit_count: 1, prio: "msb", live_nodes: 1, deepest_path: 1 },
        EncodeOneHotRow { bit_count: 2, prio: "lsb", live_nodes: 4, deepest_path: 2 },
        EncodeOneHotRow { bit_count: 2, prio: "msb", live_nodes: 3, deepest_path: 2 },
        EncodeOneHotRow { bit_count: 3, prio: "lsb", live_nodes: 9, deepest_path: 4 },
        EncodeOneHotRow { bit_count: 3, prio: "msb", live_nodes: 8, deepest_path: 4 },
        EncodeOneHotRow { bit_count: 4, prio: "lsb", live_nodes: 13, deepest_path: 5 },
        EncodeOneHotRow { bit_count: 4, prio: "msb", live_nodes: 11, deepest_path: 4 },
        EncodeOneHotRow { bit_count: 5, prio: "lsb", live_nodes: 18, deepest_path: 6 },
        EncodeOneHotRow { bit_count: 5, prio: "msb", live_nodes: 17, deepest_path: 6 },
        EncodeOneHotRow { bit_count: 6, prio: "lsb", live_nodes: 24, deepest_path: 7 },
        EncodeOneHotRow { bit_count: 6, prio: "msb", live_nodes: 22, deepest_path: 6 },
        EncodeOneHotRow { bit_count: 7, prio: "lsb", live_nodes: 29, deepest_path: 7 },
        EncodeOneHotRow { bit_count: 7, prio: "msb", live_nodes: 28, deepest_path: 7 },
        EncodeOneHotRow { bit_count: 8, prio: "lsb", live_nodes: 34, deepest_path: 8 },
        EncodeOneHotRow { bit_count: 8, prio: "msb", live_nodes: 32, deepest_path: 7 },
        EncodeOneHotRow { bit_count: 9, prio: "lsb", live_nodes: 39, deepest_path: 9 },
        EncodeOneHotRow { bit_count: 9, prio: "msb", live_nodes: 38, deepest_path: 8 },
        EncodeOneHotRow { bit_count: 10, prio: "lsb", live_nodes: 46, deepest_path: 9 },
        EncodeOneHotRow { bit_count: 10, prio: "msb", live_nodes: 44, deepest_path: 8 },
        EncodeOneHotRow { bit_count: 11, prio: "lsb", live_nodes: 51, deepest_path: 9 },
        EncodeOneHotRow { bit_count: 11, prio: "msb", live_nodes: 50, deepest_path: 9 },
        EncodeOneHotRow { bit_count: 12, prio: "lsb", live_nodes: 59, deepest_path: 10 },
        EncodeOneHotRow { bit_count: 12, prio: "msb", live_nodes: 57, deepest_path: 8 },
        EncodeOneHotRow { bit_count: 13, prio: "lsb", live_nodes: 64, deepest_path: 10 },
        EncodeOneHotRow { bit_count: 13, prio: "msb", live_nodes: 63, deepest_path: 9 },
        EncodeOneHotRow { bit_count: 14, prio: "lsb", live_nodes: 71, deepest_path: 10 },
        EncodeOneHotRow { bit_count: 14, prio: "msb", live_nodes: 69, deepest_path: 9 },
        EncodeOneHotRow { bit_count: 15, prio: "lsb", live_nodes: 73, deepest_path: 10 },
        EncodeOneHotRow { bit_count: 15, prio: "msb", live_nodes: 72, deepest_path: 10 },
        EncodeOneHotRow { bit_count: 16, prio: "lsb", live_nodes: 79, deepest_path: 11 },
        EncodeOneHotRow { bit_count: 16, prio: "msb", live_nodes: 77, deepest_path: 10 },
    ];

    assert_eq!(got.as_slice(), want);
}
