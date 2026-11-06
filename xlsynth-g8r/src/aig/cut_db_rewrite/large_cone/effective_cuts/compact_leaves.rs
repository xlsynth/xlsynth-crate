// SPDX-License-Identifier: Apache-2.0

//! Join child cuts without allocating a list of gate IDs for every pair.
//! Assign each distinct leaf a bit position local to this gate. Combining
//! boundaries such as `{a,b}` and `{b,c}` is then a bitwise OR with three
//! set bits. Only retained boundaries need sorted physical gate IDs again.
use super::{AigRef, Boundary, LARGE_CONE_MAX_LEAVES, PER_WIDTH, Priority};
use smallvec::SmallVec;
use std::cmp::Ordering;

const WIDTH: usize = LARGE_CONE_MAX_LEAVES;
// Bound the number of distinct leaves across both child lists. Each has one
// trivial cut and at most PER_WIDTH cuts of each width from one to WIDTH.
const MAX_LEAVES: usize = 2 * (1 + PER_WIDTH * WIDTH * (WIDTH + 1) / 2);
const MAX_WORDS: usize = MAX_LEAVES.div_ceil(64);
const _: () = assert!(MAX_LEAVES <= u8::MAX as usize);

fn set_bits<const N: usize>(words: [u64; N]) -> impl Iterator<Item = usize> {
    words.into_iter().enumerate().flat_map(|(word, mut bits)| {
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let slot = 64 * word + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some(slot)
        })
    })
}

/// An exact candidate over the current gate's local leaves.
#[derive(Clone, Copy)]
struct MaskedBoundary<const N: usize> {
    mask: [u64; N],
    priority: Priority,
}

impl<const N: usize> MaskedBoundary<N> {
    fn order(&self, other: &Self, leaves: &[(AigRef, usize)]) -> Ordering {
        self.priority.order(&other.priority).then_with(|| {
            let diff = std::array::from_fn::<_, N, _>(|i| self.mask[i] ^ other.mask[i]);
            // The first differing physical leaf breaks the equal-width tie.
            set_bits(diff)
                .min_by_key(|&i| leaves[i].0)
                .map_or(Ordering::Equal, |i| {
                    let bit = 1 << (i % 64);
                    (other.mask[i / 64] & bit).cmp(&(self.mask[i / 64] & bit))
                })
        })
    }

    /// Keep the best distinct boundaries at this width. A candidate that
    /// contains an already retained narrower cut spends extra input slots
    /// to separate the same root, so give the slot to another candidate.
    fn best_at_width(
        choices: &mut Vec<Self>,
        narrower: &[[u64; N]],
        leaves: &[(AigRef, usize)],
    ) -> SmallVec<[Self; PER_WIDTH]> {
        let mut winners = SmallVec::<[Self; PER_WIDTH]>::new();
        for candidate in choices.drain(..) {
            let position = winners.partition_point(|old| old.order(&candidate, leaves).is_le());
            if position >= PER_WIDTH {
                continue;
            }
            let duplicate = winners.iter().position(|old| old.mask == candidate.mask);
            if duplicate.is_some_and(|index| index < position)
                || narrower
                    .iter()
                    .any(|old| (0..N).all(|i| old[i] & !candidate.mask[i] == 0))
            {
                continue;
            }
            if let Some(index) = duplicate {
                winners.remove(index);
            } else if winners.len() == PER_WIDTH {
                winners.pop();
            }
            winners.insert(position, candidate);
        }
        winners
    }

    fn physical(&self, local: &[(AigRef, usize)]) -> Boundary {
        let mut leaves: super::Leaves = set_bits(self.mask).map(|i| local[i].0).collect();
        leaves.sort_unstable();
        Boundary {
            leaves,
            priority: self.priority,
        }
    }
}

/// A physical-to-local map shared by all candidate pairs of this gate.
#[derive(Default)]
struct Basis {
    slots: Vec<u8>,
    leaves: Vec<(AigRef, usize)>,
}

/// Scratch for joining cuts at one gate. Each mask has `N` words to address
/// the distinct leaves seen across *all* its child cuts. An individual cut
/// still has at most `WIDTH` leaves.
#[derive(Default)]
struct MaskWork<const N: usize> {
    masks: Vec<[u64; N]>,
    choices: [Vec<MaskedBoundary<N>>; WIDTH + 1],
}

impl<const N: usize> MaskWork<N> {
    /// Append the parent boundaries worth passing to its users. Try every
    /// pair of retained child cuts, count shared leaves once, and discard
    /// unions wider than the limit. Keep up to two at each width. A narrow
    /// cut leaves more room for the other input of a later join, while a
    /// wider one can expose more logic to simplify at this gate.
    ///
    /// Return false before appending anything if this gate's local leaf
    /// numbering needs more than `N` words. The caller retries the same
    /// joins with a larger mask and the same cut-width limit.
    fn retain(
        &mut self,
        basis: &mut Basis,
        left: &[Boundary],
        right: &[Boundary],
        depths: &[usize],
        retained: &mut Vec<Boundary>,
    ) -> bool {
        for (node, _) in basis.leaves.drain(..) {
            basis.slots[node.id] = 0;
        }
        basis.slots.resize(depths.len(), 0);
        self.masks.clear();
        for cut in left.iter().chain(right) {
            let mut mask = [0; N];
            for &node in &cut.leaves {
                let slot = &mut basis.slots[node.id];
                if *slot == 0 {
                    if basis.leaves.len() == N * 64 {
                        return false;
                    }
                    basis.leaves.push((node, depths[node.id]));
                    *slot = basis.leaves.len() as u8;
                }
                let pos = usize::from(*slot - 1);
                mask[pos / 64] |= 1 << (pos % 64);
            }
            self.masks.push(mask);
        }

        let (lhs, rhs) = self.masks.split_at(left.len());
        for (l, lm) in left.iter().zip(lhs) {
            for (r, rm) in right.iter().zip(rhs) {
                let mask = std::array::from_fn(|i| lm[i] | rm[i]);
                let width: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
                if width == 0 || width > WIDTH {
                    continue;
                }
                let common = std::array::from_fn::<_, N, _>(|i| lm[i] & rm[i]);
                // A shared signal has one arrival depth in the parent cut,
                // although both child scores already include it.
                let shared: usize = set_bits(common).map(|i| basis.leaves[i].1).sum();
                let total = l.priority.total_depth - shared + r.priority.total_depth;
                self.choices[width].push(MaskedBoundary {
                    mask,
                    priority: Priority::merged(l, r, width, total),
                });
            }
        }
        let mut narrower = SmallVec::<[[u64; N]; PER_WIDTH * WIDTH]>::new();
        // Process smaller widths first so their retained boundaries can
        // eliminate wider candidates that contain all the same leaves.
        for choices in &mut self.choices {
            let winners = MaskedBoundary::best_at_width(choices, &narrower, &basis.leaves);
            for cut in winners {
                narrower.push(cut.mask);
                retained.push(cut.physical(&basis.leaves));
            }
        }
        true
    }
}

/// Round-owned workspace for exact bottom-up cut retention.
#[derive(Default)]
pub(super) struct CompactLeaves {
    basis: Basis,
    common: MaskWork<1>,
    wide: MaskWork<MAX_WORDS>,
}

impl CompactLeaves {
    /// Most child lists fit all their distinct gate IDs in one machine word.
    /// Use several words only for the gates that need them.
    pub(super) fn retain(
        &mut self,
        left: &[Boundary],
        right: &[Boundary],
        depths: &[usize],
        retained: &mut Vec<Boundary>,
    ) {
        if !self
            .common
            .retain(&mut self.basis, left, right, depths, retained)
        {
            assert!(
                self.wide
                    .retain(&mut self.basis, left, right, depths, retained)
            );
        }
    }
}

#[cfg(test)]
#[path = "compact_leaves_tests.rs"]
mod tests;
