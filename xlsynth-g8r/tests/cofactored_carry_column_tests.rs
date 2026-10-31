// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;

#[derive(Clone, Copy, Debug)]
enum Arrival {
    LateUpperBit,
    IndependentBit,
    WordAndItsUpperBit,
}

#[test]
fn pair_sum_uses_the_late_bit_only_when_it_arrives_last() {
    for arrival in [
        Arrival::LateUpperBit,
        Arrival::IndependentBit,
        Arrival::WordAndItsUpperBit,
    ] {
        for upper_first in [false, true] {
            let source = pair_sum_ir(arrival, upper_first);
            let package = Parser::new(&source).parse_and_validate_package().unwrap();
            let function = package.get_top_fn().unwrap();
            let prepared = prep_for_gatify(
                function,
                None,
                PrepForGatifyOptions {
                    enable_rewrite_nary_add: true,
                    ..PrepForGatifyOptions::all_opts_disabled()
                },
            );
            let mut graphs = Vec::new();
            for enable_rewrite_nary_add in [false, true] {
                let gates = gatify_prepared_fn(
                    &prepared,
                    GatifyOptions {
                        enable_rewrite_nary_add,
                        ..GatifyOptions::all_opts_disabled()
                    },
                )
                .unwrap()
                .gate_fn;
                check_equivalence::validate_same_fn_via_toolchain(function, &gates).unwrap();
                let mut sim = PreparedGateSim::new(&gates);
                for stimulus in 0..256u64 {
                    let x = stimulus & 3;
                    let y = (stimulus >> 2) & 3;
                    let tick = (stimulus >> 4) & 1;
                    let base = (stimulus >> 5) & 3;
                    let vote = stimulus >> 7;
                    let pair = (x + y + tick) & 3;
                    let (word, bit) = match arrival {
                        Arrival::LateUpperBit => (base, pair >> 1),
                        Arrival::IndependentBit => (base, tick),
                        Arrival::WordAndItsUpperBit => (pair, pair >> 1),
                    };
                    let result = (word + bit + vote) & 3;
                    let inputs = [(2, x), (2, y), (1, tick), (2, base), (1, vote)]
                        .map(|(width, n)| IrBits::make_ubits(width, n).unwrap());
                    let expected = IrBits::make_ubits(4, (result << 2) | pair).unwrap();
                    assert_eq!(sim.eval_outputs(&inputs), vec![expected]);
                }
                graphs.push(gates);
            }
            if matches!(arrival, Arrival::LateUpperBit) {
                let baseline = get_aig_stats(&graphs[0]).max_depth;
                let cofactored = get_aig_stats(&graphs[1]).max_depth;
                assert!(
                    cofactored < baseline,
                    "{arrival:?}, {upper_first}: {baseline} -> {cofactored}"
                );
            } else {
                assert_eq!(graphs[0].to_string(), graphs[1].to_string(), "{arrival:?}");
            }
        }
    }
}

fn pair_sum_ir(arrival: Arrival, upper_first: bool) -> String {
    let (word, bit) = match arrival {
        Arrival::LateUpperBit => ("base", "upper"),
        Arrival::IndependentBit => ("base", "tick"),
        Arrival::WordAndItsUpperBit => ("pair", "upper"),
    };
    let (first, second) = if upper_first {
        ("bit_ext", "vote_ext")
    } else {
        ("vote_ext", "bit_ext")
    };
    format!(
        "package pair_sum
top fn output(x: bits[2] id=1, y: bits[2] id=2, tick: bits[1] id=3, base: bits[2] id=4, vote: bits[1] id=5) -> bits[4] {{
  tick_ext: bits[2] = zero_ext(tick, new_bit_count=2, id=6)
  xy: bits[2] = add(x, y, id=7)
  pair: bits[2] = add(xy, tick_ext, id=8)
  upper: bits[1] = bit_slice(pair, start=1, width=1, id=9)
  vote_ext: bits[2] = zero_ext(vote, new_bit_count=2, id=10)
  bit_ext: bits[2] = zero_ext({bit}, new_bit_count=2, id=11)
  partial: bits[2] = add({word}, {first}, id=12)
  value: bits[2] = add({second}, partial, id=13)
  ret both: bits[4] = concat(value, pair, id=14)
}}"
    )
}
