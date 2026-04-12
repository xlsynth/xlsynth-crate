// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions};

fn priority_sel_sample_ir_text() -> &'static str {
    r#"package neg_prio_sel_qor

top fn cone(leaf_25: bits[7] id=1, leaf_44: bits[1] id=2) -> bits[7] {
  literal.3: bits[1] = literal(value=0, id=3)
  concat.4: bits[8] = concat(literal.3, leaf_25, id=4)
  neg.5: bits[8] = neg(concat.4, id=5)
  literal.6: bits[8] = literal(value=246, id=6)
  priority_sel.7: bits[8] = priority_sel(leaf_44, cases=[neg.5], default=literal.6, id=7)
  neg.8: bits[8] = neg(priority_sel.7, id=8)
  bit_slice.9: bits[6] = bit_slice(neg.8, start=1, width=6, id=9)
  literal.10: bits[1] = literal(value=1, id=10)
  ret concat.11: bits[7] = concat(bit_slice.9, literal.10, id=11)
}
"#
}

fn sel_sample_ir_text() -> &'static str {
    r#"package neg_sel_qor

top fn cone(leaf_35: bits[7] id=1, leaf_50: bits[1] id=2) -> bits[6] {
  literal.3: bits[1] = literal(value=0, id=3)
  concat.4: bits[8] = concat(literal.3, leaf_35, id=4)
  literal.5: bits[8] = literal(value=246, id=5)
  neg.6: bits[8] = neg(concat.4, id=6)
  sel.7: bits[8] = sel(leaf_50, cases=[literal.5, neg.6], id=7)
  neg.8: bits[8] = neg(sel.7, id=8)
  ret bit_slice.9: bits[6] = bit_slice(neg.8, start=1, width=6, id=9)
}
"#
}

fn stats_for_aug_opt_mode(ir_text: &str, enable_aug_opt: bool) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        Some("cone"),
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
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
fn ir2gates_aug_opt_improves_neg_priority_sel_denegate_qor() {
    let (nodes_before, levels_before) =
        stats_for_aug_opt_mode(priority_sel_sample_ir_text(), false);
    let (nodes_after, levels_after) = stats_for_aug_opt_mode(priority_sel_sample_ir_text(), true);

    eprintln!(
        "neg-priority-sel QoR: before nodes={} levels={}, after nodes={} levels={}, delta nodes={} levels={}",
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

#[test]
fn ir2gates_aug_opt_improves_neg_sel_denegate_qor() {
    let (nodes_before, levels_before) = stats_for_aug_opt_mode(sel_sample_ir_text(), false);
    let (nodes_after, levels_after) = stats_for_aug_opt_mode(sel_sample_ir_text(), true);

    eprintln!(
        "neg-sel QoR: before nodes={} levels={}, after nodes={} levels={}, delta nodes={} levels={}",
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
