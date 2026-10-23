// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::SolverChoice;

#[test]
fn shared_select_mask_lowering() {
    for width in [7, 10, 23, 52, 65] {
        for masked_on_true in [false, true] {
            for constant in [0, 1, 2] {
                let arms = if masked_on_true {
                    "c, masked"
                } else {
                    "masked, c"
                };
                let text = format!(
                    r#"fn f(x: bits[{width}] id=1, en: bits[1] id=2, s: bits[1] id=3) -> bits[{width}] {{
  mask: bits[{width}] = sign_ext(en, new_bit_count={width}, id=4)
  masked: bits[{width}] = and(x, mask, id=5)
  c: bits[{width}] = literal(value={constant}, id=6)
  ret result: bits[{width}] = sel(s, cases=[{arms}], id=7)
}}"#
                );
                let f = Parser::new(&text).parse_fn().unwrap();
                let result = gatify_prepared_fn(
                    &f,
                    GatifyOptions {
                        check_equivalence: cfg!(feature = "has-bitwuzla"),
                        equivalence_solver: SolverChoice::Bitwuzla,
                        ..GatifyOptions::all_opts_disabled()
                    },
                )
                .unwrap();
                // One shared enable, one AND per data bit, and at most one
                // OR and its original masked value for the constant-one bit.
                let stats = get_aig_stats(&dce(&result.gate_fn));
                assert!(stats.and_nodes <= width + 3, "{stats:?}");
            }
        }
    }
}

#[test]
fn shared_select_mask_preserves_provenance_after_dce() {
    for tracking in [false, true] {
        for other_consumer in [false, true] {
            let ret = if other_consumer {
                "ret pair: (bits[4], bits[4]) = tuple(result, masked, id=8)"
            } else {
                "ret out: bits[4] = identity(result, id=8)"
            };
            let ty = if other_consumer {
                "(bits[4], bits[4])"
            } else {
                "bits[4]"
            };
            let text = format!(
                r#"fn f(x: bits[4] id=1, en: bits[1] id=2, s: bits[1] id=3) -> {ty} {{
  mask: bits[4] = sign_ext(en, new_bit_count=4, id=4)
  masked: bits[4] = and(x, mask, id=5)
  c: bits[4] = literal(value=0, id=6)
  result: bits[4] = sel(s, cases=[masked, c], id=7)
  {ret}
}}"#
            );
            let f = Parser::new(&text).parse_fn().unwrap();
            let result = gatify_prepared_fn(
                &f,
                GatifyOptions {
                    track_pir_node_ids: tracking,
                    check_equivalence: cfg!(feature = "has-bitwuzla"),
                    equivalence_solver: SolverChoice::Bitwuzla,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            let live = dce(&result.gate_fn);
            let ands = live
                .gates
                .iter()
                .filter(|node| matches!(node, xlsynth_g8r::aig::AigNode::And2 { .. }))
                .collect::<Vec<_>>();
            assert_eq!(ands.len(), if other_consumer { 9 } else { 5 });
            if tracking {
                // The shared enable and four replacement ANDs carry both
                // operations, even when all original mask ANDs become dead.
                assert_eq!(
                    ands.iter()
                        .filter(|node| {
                            let ids = node.get_pir_node_ids();
                            ids.contains(&5) && ids.contains(&7)
                        })
                        .count(),
                    5
                );
                assert_eq!(
                    ands.iter()
                        .filter(|node| {
                            let ids = node.get_pir_node_ids();
                            ids.contains(&5) && !ids.contains(&7)
                        })
                        .count(),
                    if other_consumer { 4 } else { 0 }
                );
            } else {
                assert!(
                    live.gates
                        .iter()
                        .all(|node| node.get_pir_node_ids().is_empty())
                );
            }
        }
    }
}
