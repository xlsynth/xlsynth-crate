// SPDX-License-Identifier: Apache-2.0

use super::BatchedMffc;
use crate::aig::cut_db_rewrite::collect_mffc_nodes_under_cut;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigBitVector, AigOperand, AigRef};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

fn check(
    scratch: &mut BatchedMffc,
    state: &DynamicStructuralHash,
    root: AigRef,
    cuts: &[&[AigRef]],
) -> Vec<usize> {
    let expected: Vec<_> = cuts
        .iter()
        .map(|cut| {
            let leaves: Vec<_> = cut.iter().copied().map(AigOperand::from).collect();
            collect_mffc_nodes_under_cut(state, root, &leaves).len()
        })
        .collect();
    let before = state.use_counts().to_vec();
    let actual = scratch.counts(state, root, cuts);
    assert_eq!(actual.as_slice(), expected.as_slice());
    assert_eq!(state.use_counts(), before.as_slice());
    expected
}

fn diamond(pin_shared: bool) -> (DynamicStructuralHash, [AigRef; 7]) {
    let mut builder = GateBuilder::new("batched".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let shared = builder.add_and_binary(a, b);
    let duplicate = builder.add_and_binary(shared, shared.negate());
    let left = builder.add_and_binary(duplicate, a);
    let right = builder.add_and_binary(duplicate.negate(), b);
    let root = builder.add_and_binary(left, right);
    let unused = builder.add_and_binary(a.negate(), b);
    builder.add_output("root".into(), AigBitVector::from_bit(root));
    builder.add_output("root2".into(), AigBitVector::from_bit(root.negate()));
    if pin_shared {
        builder.add_output("shared".into(), AigBitVector::from_bit(shared));
    }
    (
        DynamicStructuralHash::new(builder.build()).unwrap(),
        [
            root.node,
            left.node,
            right.node,
            duplicate.node,
            shared.node,
            a.node,
            unused.node,
        ],
    )
}

#[test]
fn lane_order_and_shared_edge_intersections_match_the_set_oracle() {
    let mut scratch = BatchedMffc::default();
    for pin_shared in [false, true] {
        let (state, [root, left, right, duplicate, shared, input, unused]) = diamond(pin_shared);
        let missing = AigRef { id: usize::MAX };
        let cuts: &[&[AigRef]] = &[
            &[],
            &[left],
            &[right],
            &[duplicate],
            &[shared, shared],
            &[missing, input],
            &[root],
        ];
        let whole = if pin_shared { 4 } else { 5 };
        assert_eq!(
            check(&mut scratch, &state, root, cuts),
            [whole, 2, 2, 3, 4, whole, 0]
        );
        // Blocking opposite branches yields tied counts. Their masks at the
        // shared child have empty intersection, while the unpinned cut in a
        // third lane must still release it.
        assert_eq!(
            check(&mut scratch, &state, root, &[&[left], &[right], &[]]),
            [2, 2, whole]
        );
        for width in [0, 1, 4, 16, 31, 32, 33, 63, 64, 65, 129] {
            let mut lanes: Vec<_> = cuts.iter().copied().cycle().take(width).collect();
            let expected = check(&mut scratch, &state, root, &lanes);
            lanes.reverse();
            assert_eq!(
                check(&mut scratch, &state, root, &lanes),
                expected.into_iter().rev().collect::<Vec<_>>()
            );
        }
        assert_eq!(
            check(&mut scratch, &state, duplicate, &[&[], &[shared]]),
            [whole - 3, 1]
        );
        for terminal in [input, unused] {
            assert_eq!(
                check(&mut scratch, &state, terminal, &[&[], &[missing]]),
                [0, 0]
            );
        }
    }
}

#[test]
fn generation_wrap_and_smaller_graph_cannot_revive_previous_marks() {
    let mut scratch = BatchedMffc::default();
    let mut builder = GateBuilder::new("large".into(), GateBuilderOptions::no_opt());
    let base = *builder.add_input("base".into(), 1).get_lsb(0);
    let mut root = base;
    for i in 0..64 {
        let leaf = *builder.add_input(format!("i{i}"), 1).get_lsb(0);
        let side = builder.add_and_binary(leaf, base);
        root = builder.add_and_binary(side, root);
    }
    builder.add_output("out".into(), AigBitVector::from_bit(root));
    let large = DynamicStructuralHash::new(builder.build()).unwrap();
    let high = root.node;
    let (mut small, [low, _, _, _, shared, input, _]) = diamond(false);
    // Plant a lane-two root-boundary bit at an ID reused by the smaller graph.
    let prior = check(&mut scratch, &large, high, &[&[], &[high], &[low]]);
    assert_eq!(&prior[..2], &[128, 0]);

    // Marks from epoch one in the large allocation must be cleared before
    // that generation is reused for the smaller graph with overlapping IDs.
    scratch.epoch = u32::MAX;
    assert_eq!(
        check(&mut scratch, &small, low, &[&[shared], &[high], &[]]),
        [4, 5, 5]
    );
    small
        .replace_node_with_operand(low, AigOperand::from(input))
        .unwrap();
    assert!(!small.is_live(low));
    assert_eq!(check(&mut scratch, &small, low, &[&[], &[low]]), [0, 0]);
    assert_eq!(check(&mut scratch, &large, high, &[&[high], &[]]), [0, 128]);
}
