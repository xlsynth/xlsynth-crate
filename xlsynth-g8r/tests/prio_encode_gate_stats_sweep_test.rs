// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::ir2gate_utils::gatify_prio_encode;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PrioEncodeRow {
    bit_count: u32,
    prio: &'static str,
    live_nodes: usize,
    deepest_path: usize,
}

fn stats_for_prio_encode(bit_count: usize, lsb_prio: bool) -> SummaryStats {
    let mut gb = GateBuilder::new("prio_encode".to_string(), GateBuilderOptions::opt());
    let input = gb.add_input("input".to_string(), bit_count);
    let (any, idx_bits) = gatify_prio_encode(&mut gb, &input, lsb_prio);
    gb.add_output("any".to_string(), any.into());
    gb.add_output("idx".to_string(), idx_bits);
    let gate_fn = gb.build();
    get_summary_stats(&gate_fn)
}

fn gather_prio_encode_rows() -> Vec<PrioEncodeRow> {
    let mut got: Vec<PrioEncodeRow> = Vec::new();
    for bit_count in [1usize, 2, 4, 8, 16] {
        for lsb_prio in [true, false] {
            let prio = match lsb_prio {
                true => "lsb",
                false => "msb",
            };
            let stats = stats_for_prio_encode(bit_count, lsb_prio);
            got.push(PrioEncodeRow {
                bit_count: bit_count as u32,
                prio,
                live_nodes: stats.live_nodes,
                deepest_path: stats.deepest_path,
            });
        }
    }
    got
}

#[test]
fn test_prio_encode_gate_stats_pow2_sweep_1_to_16() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_prio_encode_rows();

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[PrioEncodeRow] = &[
        PrioEncodeRow { bit_count: 1, prio: "lsb", live_nodes: 1, deepest_path: 1 },
        PrioEncodeRow { bit_count: 1, prio: "msb", live_nodes: 1, deepest_path: 1 },
        PrioEncodeRow { bit_count: 2, prio: "lsb", live_nodes: 4, deepest_path: 3 },
        PrioEncodeRow { bit_count: 2, prio: "msb", live_nodes: 3, deepest_path: 2 },
        PrioEncodeRow { bit_count: 4, prio: "lsb", live_nodes: 11, deepest_path: 5 },
        PrioEncodeRow { bit_count: 4, prio: "msb", live_nodes: 9, deepest_path: 4 },
        PrioEncodeRow { bit_count: 8, prio: "lsb", live_nodes: 27, deepest_path: 7 },
        PrioEncodeRow { bit_count: 8, prio: "msb", live_nodes: 24, deepest_path: 6 },
        PrioEncodeRow { bit_count: 16, prio: "lsb", live_nodes: 61, deepest_path: 9 },
        PrioEncodeRow { bit_count: 16, prio: "msb", live_nodes: 57, deepest_path: 8 },
    ];

    assert_eq!(got.as_slice(), want);
}
