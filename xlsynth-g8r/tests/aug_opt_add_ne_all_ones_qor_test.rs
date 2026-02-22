// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions};

fn sample_ir_text() -> &'static str {
    r#"package add_ne_all_ones_qor

top fn cone(leaf_303: bits[8] id=1, leaf_304: bits[8] id=2) -> bits[1] {
  literal.3: bits[8] = literal(value=255, id=3)
  add.4: bits[8] = add(leaf_303, leaf_304, id=4)
  ret ne.5: bits[1] = ne(add.4, literal.3, id=5)
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
fn ir2gates_aug_opt_improves_add_ne_all_ones_qor_to_match_abc() {
    let (nodes_before, levels_before) = stats_for_aug_opt_mode(false);
    let (nodes_after, levels_after) = stats_for_aug_opt_mode(true);

    eprintln!(
        "add-ne-all-ones QoR: before nodes={} levels={}, after nodes={} levels={}, delta nodes={} levels={}",
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

    // End-to-end reference target from yosys+abc for this cone.
    // We require aug-opt to meet or beat that baseline.
    assert!(
        nodes_after <= 58,
        "expected nodes_after <= 58; got {}",
        nodes_after
    );
    assert!(
        levels_after <= 10,
        "expected levels_after <= 10; got {}",
        levels_after
    );
}
