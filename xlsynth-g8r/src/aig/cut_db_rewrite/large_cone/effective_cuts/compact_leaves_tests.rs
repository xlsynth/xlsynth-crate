// SPDX-License-Identifier: Apache-2.0
use super::*;
use std::cmp::Reverse;
use std::collections::BTreeSet;

fn candidate(mask: u64, latest: usize, total_depth: usize, tree_ands: usize) -> MaskedBoundary<1> {
    MaskedBoundary {
        mask: [mask],
        priority: Priority {
            latest,
            total_depth,
            tree_ands,
            overlap: 0,
        },
    }
}

#[test]
fn retention_prioritizes_arrival_before_size_and_keeps_the_best_duplicate() {
    let local: Vec<_> = (0..16).map(|id| (AigRef { id }, 0)).collect();
    let mut choices = vec![
        candidate(3, 1, 3, usize::MAX), // Dominated by the narrower mask.
        candidate(12, 4, 7, 5),
        candidate(48, 4, 7, 8),
        candidate(12, 4, 7, usize::MAX),
        candidate(192, 4, 7, 9),
        candidate(12, 4, 7, 10),
        candidate(768, 4, 8, usize::MAX),
        candidate(3072, 5, 6, usize::MAX),
    ];
    let winners = MaskedBoundary::best_at_width(&mut choices, &[[1]], &local);
    assert_eq!(winners[0].mask, [12]);
    assert_eq!(winners[0].priority.tree_ands, usize::MAX);
    assert_eq!(winners[1].mask, [192]);
}

#[test]
fn every_mask_word_and_reused_scratch_match_sorted_physical_sets() {
    let mut depths = vec![0; 149];
    // Repeated in the second and third requests. Child totals must avoid
    // overflowing before a shared boundary leaf's depth is subtracted.
    depths[0] = usize::MAX - 1;
    let bag = |mut n: usize| {
        let mut cuts = Vec::new();
        // One unit plus two per width gives 73 distinct leaves per child;
        // descending widths put narrower right-hand cuts in the third word.
        for width in std::iter::once(1).chain((1..=WIDTH).rev().flat_map(|w| [w, w])) {
            let mut leaves: super::super::Leaves = (n..n + width)
                .map(|i| AigRef { id: (53 * i) % 149 })
                .collect();
            leaves.sort_unstable();
            let arrival = leaves.iter().map(|node| depths[node.id]).sum();
            cuts.push(Boundary {
                leaves,
                priority: Priority {
                    latest: arrival,
                    total_depth: arrival,
                    tree_ands: n + 1,
                    overlap: 0,
                },
            });
            n += width;
        }
        cuts
    };
    let left = bag(0);
    let disjoint = bag(73);
    let common = bag(68);
    let mut scratch = CompactLeaves::default();
    for (lhs, rhs) in [
        (&left[..], &disjoint[..]), // 146 distinct IDs
        (&left[..], &common[..]),   // three words with an overlap
        (&left[..3], &left[..3]),   // one word in the same workspace
    ] {
        let mut all = Vec::new();
        for a in lhs {
            for b in rhs {
                let leaves: BTreeSet<_> = a.leaves.iter().chain(&b.leaves).copied().collect();
                if leaves.len() > WIDTH {
                    continue;
                }
                let latest = leaves.iter().map(|n| depths[n.id]).max().unwrap();
                let total: usize = leaves.iter().map(|n| depths[n.id]).sum();
                let size = a
                    .priority
                    .tree_ands
                    .saturating_add(b.priority.tree_ands)
                    .saturating_add(1);
                let overlap = a.leaves.len() + b.leaves.len() - leaves.len();
                all.push((
                    leaves.len(),
                    latest,
                    total,
                    Reverse(size),
                    Reverse(overlap),
                    leaves,
                ));
            }
        }
        all.sort();
        let mut expected: Vec<(
            usize,
            usize,
            usize,
            Reverse<usize>,
            Reverse<usize>,
            BTreeSet<AigRef>,
        )> = Vec::new();
        let mut width_counts = [0; WIDTH + 1];
        for entry in all {
            if width_counts[entry.0] < PER_WIDTH
                && !expected.iter().any(|e| e.5.is_subset(&entry.5))
            {
                width_counts[entry.0] += 1;
                expected.push(entry);
            }
        }
        let mut retained = Vec::new();
        scratch.retain(lhs, rhs, &depths, &mut retained);
        assert_eq!(expected.len(), retained.len());
        for (cut, (width, latest, total, Reverse(size), Reverse(overlap), leaves)) in
            retained.iter().zip(expected)
        {
            assert_eq!(cut.leaves.len(), width);
            assert!(cut.leaves.iter().eq(leaves.iter()));
            assert_eq!(
                (
                    cut.priority.latest,
                    cut.priority.total_depth,
                    cut.priority.tree_ands,
                    cut.priority.overlap
                ),
                (latest, total, size, overlap)
            );
        }
    }
}
