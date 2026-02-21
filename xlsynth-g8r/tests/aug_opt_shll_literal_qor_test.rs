// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions};

fn sample_ir_text() -> &'static str {
    r#"package shll_qor

top fn cone(leaf_168: bits[10] id=1, leaf_236: bits[4] id=2) -> bits[1] {
  bit_slice.3: bits[9] = bit_slice(leaf_168, start=0, width=9, id=3)
  shll.4: bits[9] = shll(bit_slice.3, leaf_236, id=4)
  bit_slice.5: bits[7] = bit_slice(shll.4, start=1, width=7, id=5)
  literal.6: bits[7] = literal(value=127, id=6)
  ret eq.7: bits[1] = eq(bit_slice.5, literal.6, id=7)
}
"#
}

fn stats_for_aug_opt_mode(enable_aug_opt: bool) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        sample_ir_text(),
        Some("cone"),
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: AugOptOptions {
                enable: enable_aug_opt,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        },
    )
    .expect("ir2gates");

    let s = get_summary_stats(&out.gatify_output.gate_fn);
    (s.live_nodes, s.deepest_path)
}

#[test]
fn ir2gates_aug_opt_improves_shll_slice_literal_qor() {
    let (nodes_before, levels_before) = stats_for_aug_opt_mode(false);
    let (nodes_after, levels_after) = stats_for_aug_opt_mode(true);

    eprintln!(
        "shll-slice-literal QoR: before nodes={} levels={}, after nodes={} levels={}, delta nodes={} levels={}",
        nodes_before,
        levels_before,
        nodes_after,
        levels_after,
        nodes_after as isize - nodes_before as isize,
        levels_after as isize - levels_before as isize
    );

    assert!(
        nodes_after < nodes_before,
        "expected aug-opt to reduce nodes; before={} after={} (levels before={} after={})",
        nodes_before,
        nodes_after,
        levels_before,
        levels_after
    );
    assert!(
        levels_after <= levels_before,
        "expected aug-opt to avoid worsening levels; before={} after={} (nodes before={} after={})",
        levels_before,
        levels_after,
        nodes_before,
        nodes_after
    );
}
