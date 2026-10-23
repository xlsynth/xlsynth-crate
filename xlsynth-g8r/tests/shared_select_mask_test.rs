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
