// SPDX-License-Identifier: Apache-2.0

use smallvec::{SmallVec, smallvec};

use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigRef};

/// Tracks which cuts could release this AND and how many uses have been
/// accounted for in the current traversal.
#[derive(Clone, Copy, Default)]
struct Entry {
    epoch: u32,
    allowed: u32,
    arrived: usize,
}

/// Counts removable ANDs for several candidate boundaries in one traversal.
/// One bit per cut records whether an AND can disappear for that cut.
#[derive(Default)]
pub(super) struct BatchedMffc {
    epoch: u32,
    entries: Vec<Entry>,
    pending: Vec<(AigRef, u32)>,
}

impl BatchedMffc {
    fn begin(&mut self, nodes: usize) {
        if self.entries.len() < nodes {
            self.entries.resize(nodes, Entry::default());
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.entries.fill(Entry::default());
            self.epoch = 1;
        }
        self.pending.clear();
    }

    fn entry(&mut self, node: AigRef) -> &mut Entry {
        let entry = &mut self.entries[node.id];
        if entry.epoch != self.epoch {
            *entry = Entry {
                epoch: self.epoch,
                allowed: u32::MAX,
                arrived: 0,
            };
        }
        entry
    }

    /// Count the gates that would lose their last use if the root were
    /// removed and each cut's own boundary signals were kept live.
    pub(super) fn counts(
        &mut self,
        state: &DynamicStructuralHash,
        root: AigRef,
        boundaries: &[&[AigRef]],
    ) -> SmallVec<[usize; 16]> {
        let mut counts = smallvec![0; boundaries.len()];
        if boundaries.is_empty() || state.use_count(root) == 0 {
            return counts;
        }
        let gates = &state.gate_fn().gates;
        let AigNode::And2 { a, b, .. } = gates[root.id] else {
            return counts;
        };

        // If releasing the root cannot release either immediate AND, each
        // boundary either removes the root alone or pins the root itself.
        let uses_from_root = if a.node == b.node { 2 } else { 1 };
        if ![a.node, b.node].iter().any(|&child| {
            matches!(gates[child.id], AigNode::And2 { .. })
                && state.use_count(child) == uses_from_root
        }) {
            for (answer, leaves) in counts.iter_mut().zip(boundaries) {
                *answer = usize::from(!leaves.contains(&root));
            }
            return counts;
        }

        let width = u32::BITS as usize;
        for (group, answers) in boundaries.chunks(width).zip(counts.chunks_mut(width)) {
            self.begin(gates.len());
            for (i, &leaves) in group.iter().enumerate() {
                for &leaf in leaves {
                    if leaf.id < gates.len() {
                        self.entry(leaf).allowed &= !(1 << i);
                    }
                }
            }
            let root_mask = (u32::MAX >> (width - group.len())) & self.entry(root).allowed;
            if root_mask != 0 {
                self.pending.push((root, root_mask));
            }
            while let Some((node, mask)) = self.pending.pop() {
                let mut credit = mask;
                while credit != 0 {
                    answers[credit.trailing_zeros() as usize] += 1;
                    credit &= credit - 1;
                }
                let AigNode::And2 { a, b, .. } = gates[node.id] else {
                    unreachable!()
                };
                // Each removed input edge releases one use of its source.
                // For a cut to remove the child, every use must be released
                // and the child must not be a leaf of that cut. An outside
                // user keeps it live. Removing x & x releases two uses of x.
                for child in [a.node, b.node] {
                    if !matches!(gates[child.id], AigNode::And2 { .. }) {
                        continue;
                    }
                    let entry = self.entry(child);
                    entry.allowed &= mask;
                    entry.arrived += 1;
                    if entry.arrived == state.use_count(child) && entry.allowed != 0 {
                        let child_mask = entry.allowed;
                        self.pending.push((child, child_mask));
                    }
                }
            }
        }
        counts
    }
}

#[cfg(test)]
#[path = "batched_mffc_tests.rs"]
mod tests;
