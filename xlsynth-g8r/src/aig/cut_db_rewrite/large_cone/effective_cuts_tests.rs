// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::aig::AigBitVector;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

fn leaves(ids: &[usize]) -> Leaves {
    ids.iter().map(|&id| AigRef { id }).collect()
}

#[test]
fn ranked_and_unranked_cuts_carry_their_truth_and_enclosed_count() {
    for (width, budget, expected) in [
        (3, 64, 1),
        (7, 1, 0),
        (7, 4, 2),
        (7, 64, MAX_FACTORED),
        (7, 0, MAX_FACTORED),
    ] {
        let mut builder = GateBuilder::new("selected-truth".into(), GateBuilderOptions::no_opt());
        let leaves = builder.add_input("inputs".into(), width);
        let mut root = *leaves.get_lsb(0);
        for i in 1..width {
            root = builder.add_and_binary(root, *leaves.get_lsb(i));
        }
        builder.add_output("out".into(), AigBitVector::from_bit(root));
        let state = DynamicStructuralHash::new(builder.build()).unwrap();
        let g = state.gate_fn();
        let mut cuts = EffectiveCuts::new(budget);
        let selected = cuts.select(root.node, &state, &vec![0; g.gates.len()]);
        assert!(
            budget == 0
                || cuts
                    .by_node
                    .values()
                    .all(|boundaries| boundaries.len() <= budget)
        );
        assert_eq!(selected.len(), expected);
        for input in selected {
            assert_eq!(
                input.mffc.len(),
                collect_mffc_nodes_under_cut(&state, root.node, &input.leaves).len()
            );
            let cone = input.cone;
            for row in 0..1 << input.leaves.len() {
                assert_eq!(
                    cone.table.get_bit(row),
                    cut_truth::eval_cut_row(g, root.node, &input.leaves, row)
                );
            }
            assert_eq!(
                cone.enclosed,
                crate::aig::cut_db_rewrite::collect_internal_and_nodes_under_cut(
                    g,
                    root.node,
                    &input.leaves
                )
                .len()
            );
        }
    }
}

#[test]
fn effective_density_ties_keep_physical_density_enclosed_and_leaf_order() {
    fn score(ids: &[usize], removable: usize, support: usize, enclosed: usize) -> ScoredBoundary {
        ScoredBoundary {
            leaves: leaves(ids).into_iter().map(positive_operand).collect(),
            cone: ConeTruth {
                table: super::super::LargeTruthTable::const0(ids.len()),
                enclosed,
            },
            removable,
            support,
        }
    }
    // Equal effective density for the last four. A higher physical density wins
    // before enclosed size, with leaf order resolving the last tie.
    let mut cuts = [
        score(&[2, 4, 5, 6], 4, 2, 9),
        score(&[4, 5], 4, 2, 3),
        score(&[3, 5], 4, 2, 3),
        score(&[2, 5], 4, 2, 7),
        score(&[7, 8, 9, 10], 3, 1, 2),
    ];
    cuts.sort_by(ScoredBoundary::factor_order);
    let first: Vec<_> = cuts.iter().map(|c| c.leaves[0].node.id).collect();
    assert_eq!(first, [7, 2, 3, 4, 2]);
}

#[test]
fn an_edit_invalidates_cached_fanout_boundaries_without_touching_independent_ones() {
    let mut builder =
        GateBuilder::new("effective-cut-requeue".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let c = *builder.add_input("c".into(), 1).get_lsb(0);
    let old = builder.add_and_binary(a, b);
    let root = builder.add_and_binary(old, c);
    builder.add_output("root".into(), AigBitVector::from_bit(root));
    let mut state = DynamicStructuralHash::new(builder.build()).unwrap();
    let mut cache = EffectiveCuts::default();
    let depths = vec![0; state.gate_fn().gates.len()];
    cache.fill(state.gate_fn(), root.node, &depths);
    let before = cache.by_node[&root.node]
        .iter()
        .any(|cut| cut.leaves.contains(&b.node));
    assert!(before);
    state.replace_node_with_operand(old.node, a).unwrap();
    cache.invalidate([old.node, root.node].into_iter());
    assert!(cache.by_node.contains_key(&c.node));
    cache.fill(state.gate_fn(), root.node, &depths);
    let mut fresh = EffectiveCuts::default();
    fresh.fill(state.gate_fn(), root.node, &depths);
    let boundaries = |c: &EffectiveCuts| {
        c.by_node[&root.node]
            .iter()
            .map(|cut| {
                (
                    cut.leaves.clone(),
                    (
                        cut.priority.latest,
                        cut.priority.total_depth,
                        cut.priority.tree_ands,
                        cut.priority.overlap,
                    ),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(boundaries(&cache), boundaries(&fresh));
    assert!(
        cache.by_node[&root.node]
            .iter()
            .all(|cut| !cut.leaves.contains(&b.node))
    );
}
