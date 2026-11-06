// SPDX-License-Identifier: Apache-2.0

use super::super::{Replacement, ReplacementImpl};
use super::{gate_count_diff_with_protected, materialize_replacement};
use crate::aig::AigBitVector;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigOperand, AigRef};
use crate::cut_db::fragment::{FIRST_NODE_ID, FragmentNode, GateFnFragment, Lit};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

fn input(g: &mut GateBuilder, name: &str) -> AigOperand {
    *g.add_input(name.to_owned(), 1).get_lsb(0)
}

fn output(g: &mut GateBuilder, name: &str, op: AigOperand) {
    g.add_output(name.to_owned(), AigBitVector::from_bit(op));
}

fn replacement(
    root: AigOperand,
    leaves: &[AigOperand],
    implementation: ReplacementImpl,
) -> Replacement {
    Replacement {
        root: root.node,
        leaf_ops: leaves.to_vec(),
        implementation,
        score_depth: 0,
        score_ands: 0,
        raw_score_ands: 0,
        structural_hash_only_area_win: false,
    }
}

fn conjunction(root: AigOperand, leaves: &[AigOperand]) -> Replacement {
    let a = Lit::new(0, false);
    let (nodes, output) = match leaves.len() {
        1 => (Vec::new(), a),
        2 => (
            vec![FragmentNode::And2 {
                a,
                b: Lit::new(1, false),
            }],
            Lit::new(FIRST_NODE_ID, false),
        ),
        _ => panic!("test expects one or two leaves"),
    };
    let frag = GateFnFragment { nodes, output };
    replacement(
        root,
        leaves,
        ReplacementImpl::Fragment {
            input_depths: frag.input_depths(),
            frag,
        },
    )
}

/// Price against the pristine graph, then run the reference guard from that
/// graph.
fn compare_guard(
    state: &DynamicStructuralHash,
    repl: &Replacement,
    protected: &[AigRef],
    expected_live: bool,
    expected_certificate: Option<bool>,
) {
    assert!(protected.is_sorted());
    let diff = gate_count_diff_with_protected(state, repl, protected)
        .unwrap()
        .unwrap();
    let mut trial = state.clone();
    materialize_replacement(&mut trial, repl).unwrap().unwrap();
    trial.check_invariants().unwrap();

    let real_live = protected.iter().all(|&gate| trial.is_live(gate));
    assert_eq!(real_live, expected_live);
    assert!(diff.live_and_delta > 0, "exercise the positive-gain guard");
    assert_eq!(diff.after_live_ands, trial.live_and_count());
    assert_eq!(diff.protected_live, expected_certificate);
    assert_eq!(diff.protected_live.unwrap_or(real_live), real_live);
}

#[test]
fn certifies_output_uses_and_deleted_ids_with_a_new_replacement_node() {
    let mut g = GateBuilder::new("output_uses".into(), GateBuilderOptions::no_opt());
    let a = input(&mut g, "a");
    let b = input(&mut g, "b");
    let c = input(&mut g, "c");
    let aa = g.add_and_binary(a, a);
    let bb = g.add_and_binary(b, b);
    let aab = g.add_and_binary(aa, b);
    let root = g.add_and_binary(aab, bb);
    let parent = g.add_and_binary(root, c);
    output(&mut g, "shared", aa);
    output(&mut g, "unchanged_parent_id", parent);
    output(&mut g, "direct", root);
    output(&mut g, "complement", root.negate());
    let state = DynamicStructuralHash::new(g.build()).unwrap();
    let edit = conjunction(root, &[a, b]);

    // The new (a & b) has no existing representative. Rewiring two outputs
    // and one direct fanout releases bb, aab, and root. The old aa remains
    // live due solely to its output; the parent's fanins change in place.
    let mut trial = state.clone();
    let applied = materialize_replacement(&mut trial, &edit).unwrap().unwrap();
    assert!(applied.replacement_op.node.id >= applied.first_new_id);
    assert_eq!(trial.output_use_count(applied.replacement_op.node), 2);
    compare_guard(&state, &edit, &[aa.node, parent.node], true, Some(true));
    for lost in [bb.node, aab.node, root.node] {
        compare_guard(
            &state,
            &edit,
            &[aa.node, lost, parent.node],
            false,
            Some(false),
        );
    }
}

#[test]
fn fanout_merge_defers_for_both_the_representative_and_the_merged_id() {
    let mut g = GateBuilder::new("fanout_merge".into(), GateBuilderOptions::no_opt());
    let a = input(&mut g, "a");
    let b = input(&mut g, "b");
    let root = g.add_and_binary(a, a);
    let representative = g.add_and_binary(a, b);
    let merged = g.add_and_binary(root, b);
    output(&mut g, "rep", representative);
    output(&mut g, "fanout", merged);
    let state = DynamicStructuralHash::new(g.build()).unwrap();
    let edit = conjunction(root, &[a]);

    compare_guard(&state, &edit, &[representative.node], true, None);
    compare_guard(&state, &edit, &[merged.node], false, None);
}

#[test]
fn recreating_the_excluded_root_defers_even_with_positive_area_gain() {
    let mut g = GateBuilder::new("excluded_root".into(), GateBuilderOptions::no_opt());
    let a = input(&mut g, "a");
    let root = g.add_and_binary(a, a);
    output(&mut g, "root", root);
    let state = DynamicStructuralHash::new(g.build()).unwrap();

    // The unused fragment AND has the old root's key. Real construction
    // excludes that ID and allocates a duplicate, then cleanup deletes it.
    // The cost overlay can reuse the old ID during construction.
    let var = Lit::new(0, false);
    let frag = GateFnFragment {
        nodes: vec![FragmentNode::And2 { a: var, b: var }],
        output: var,
    };
    let edit = replacement(
        root,
        &[a],
        ReplacementImpl::Fragment {
            input_depths: frag.input_depths(),
            frag,
        },
    );
    compare_guard(&state, &edit, &[root.node], false, None);
}

#[test]
fn effective_area_price_keeps_duplicate_fanout_gains_and_protection_results() {
    use super::super::large_cone::{LargeTruthTable, sop_replacements_for_test};
    use super::super::{CutMffc, LargeConeCandidate, collect_mffc_nodes_under_cut};
    use super::{AreaCost, EffectiveCutPricing, price_area_candidate};

    for duplicate in [false, true] {
        let mut g = GateBuilder::new("duplicate_fanouts".into(), GateBuilderOptions::no_opt());
        let a = input(&mut g, "a");
        let b = input(&mut g, "b");
        let root = g.add_and_binary(a, b);
        let user = g.add_and_binary(root, root.negate());
        output(&mut g, "y", user);
        if duplicate {
            let other = g.add_and_binary(root, root.negate());
            output(&mut g, "z", other);
        }
        let state = DynamicStructuralHash::new(g.build()).unwrap();
        let mut table = LargeTruthTable::const0(2);
        for assignment in 1..4 {
            table.set_bit(assignment, true);
        }
        let sop = sop_replacements_for_test(&table, &[0, 0]).remove(0);
        let cand = LargeConeCandidate {
            replacement: replacement(root, &[a, b], ReplacementImpl::Sop(sop)),
            mffc: CutMffc::Members(collect_mffc_nodes_under_cut(&state, root.node, &[a, b])),
            raw_and_count: 1,
            new_root_depth: 1,
            variant_order: 0,
        };
        let protected = [user.node];
        let exact = gate_count_diff_with_protected(&state, &cand.replacement, &protected)
            .unwrap()
            .unwrap();
        assert_eq!(exact.live_and_delta, isize::from(duplicate));
        let mut pricing = EffectiveCutPricing::new(1, std::slice::from_ref(&cand));
        let bounded = price_area_candidate(&state, &cand, &protected, Some(&mut pricing)).unwrap();
        match bounded {
            AreaCost::Profitable(cost) => {
                assert!(duplicate);
                assert_eq!(cost.live_and_delta, exact.live_and_delta);
                assert_eq!(cost.protected_live, exact.protected_live);
            }
            AreaCost::NoAreaGain => assert!(!duplicate),
            AreaCost::Unavailable => panic!("expected a live root"),
        }
    }
}

#[test]
fn effective_area_price_preserves_a_deeper_sops_root_reuse_error() {
    use super::super::large_cone::{LargeTruthTable, sop_replacements_for_test};
    use super::super::{CutMffc, LargeConeCandidate, collect_mffc_nodes_under_cut};
    use super::{EffectiveCutPricing, price_area_candidate};

    let mut and = LargeTruthTable::const0(3);
    and.set_bit(7, true);
    let variants = sop_replacements_for_test(&and, &[0; 3]);
    let mut checked = 0;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let mut g = GateBuilder::new("root_use".into(), GateBuilderOptions::no_opt());
        let leaves = [input(&mut g, "a"), input(&mut g, "b"), input(&mut g, "c")];
        let root = g.add_and_binary(leaves[i], leaves[j]);
        output(&mut g, "root", root);
        let state = DynamicStructuralHash::new(g.build()).unwrap();
        for variant in &variants {
            let cand = LargeConeCandidate {
                replacement: replacement(root, &leaves, ReplacementImpl::Sop(variant.clone())),
                mffc: CutMffc::Members(collect_mffc_nodes_under_cut(&state, root.node, &leaves)),
                raw_and_count: 2,
                new_root_depth: 2,
                variant_order: 0,
            };
            if let Err(expected) = gate_count_diff_with_protected(&state, &cand.replacement, &[]) {
                let mut pricing = EffectiveCutPricing::new(1, std::slice::from_ref(&cand));
                let actual = price_area_candidate(&state, &cand, &[], Some(&mut pricing)).err();
                assert_eq!(actual, Some(expected));
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}
