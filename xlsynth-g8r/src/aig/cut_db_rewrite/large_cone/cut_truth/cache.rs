// SPDX-License-Identifier: Apache-2.0

//! Many outputs can share a long cone, even when each output adds a different
//! input. Remember truth tables from expensive walks so later cuts can reuse
//! a root or a fanin. Both the Boolean function and the number of distinct
//! enclosed ANDs are needed to keep the same factoring order.

use super::super::LARGE_CONE_MAX_LEAVES;
use super::{AigNode, AigOperand, AigRef, ConeTruth, CutEvaluator, GateFn, LargeTruthTable};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::VecDeque;

// Only consider caching when the root lies this many levels above the latest
// leaf.
pub(in super::super) const MIN_SPAN: usize = 256;
// Promote the cache once long walks alone have revisited one graph's worth
// of gates sixteen times in this round.
const SCANS_BEFORE_CACHING: usize = 16;
const MAX_SAVED_WALKS: usize = 16;

/// The first reachable AND and the first AND of a dense suffix, in local
/// postorder. Equal indices certify that every AND between this node and its
/// first ancestor is inside its cone. Otherwise the node is unavailable.
struct Slot {
    first: usize,
    suffix: usize,
    // Boundary positions reached in the graph, including functionally redundant
    // inputs.
    used: u8,
    table: LargeTruthTable,
}

/// Postorder positions are local to a particular graph and ordered boundary.
/// Interior membership stays exact regardless of the ordering of physical IDs.
struct SavedWalk {
    leaves: Vec<AigOperand>,
    order: FxHashMap<AigRef, usize>,
    slots: Vec<Slot>,
}

impl SavedWalk {
    fn contains(&self, first: usize, end: usize, node: AigRef) -> bool {
        self.order
            .get(&node)
            .is_some_and(|i| (first..=end).contains(i))
    }

    /// The measured truth table can be reused if every reached leaf is still
    /// in its old position and any new boundary wire lies outside this cone.
    fn on_basis(&self, end: usize, leaves: &[AigOperand]) -> Option<ConeTruth> {
        let slot = &self.slots[end];
        if slot.first != slot.suffix || end + 1 - slot.first < MIN_SPAN {
            return None;
        }
        if leaves.len() != self.leaves.len() {
            return None;
        }
        for (i, (new, old)) in leaves.iter().zip(&self.leaves).enumerate() {
            if new.node != old.node
                && (slot.used & (1 << i) != 0 || self.contains(slot.first, end, new.node))
            {
                return None;
            }
        }
        Some(ConeTruth {
            table: slot.table,
            enclosed: end + 1 - slot.first,
        })
    }
}

/// The evaluator reuses several recent leaf sets. Their combined postorders
/// contain at most twice the number of physical gates in the current graph.
#[derive(Default)]
pub(in super::super) struct ConeCache {
    walks: VecDeque<SavedWalk>,
    slots: usize,
    deep_visits: usize,
}

impl ConeCache {
    /// Graph edits invalidate saved measurements. Work accumulated in this
    /// round still determines whether to cache the next deep walk.
    pub(in super::super) fn clear(&mut self) {
        self.walks.clear();
        self.slots = 0;
    }

    fn find(&self, node: AigRef, leaves: &[AigOperand]) -> Option<(usize, usize, ConeTruth)> {
        self.walks.iter().enumerate().rev().find_map(|(i, walk)| {
            let end = *walk.order.get(&node)?;
            Some((i, end, walk.on_basis(end, leaves)?))
        })
    }
}

impl CutEvaluator {
    /// Reuse the root or a measured fanin. If the other side reconverges
    /// inside that fanin, subtract those shared ANDs from the combined count.
    pub(in super::super) fn collapse_reusing(
        &mut self,
        cache: &mut ConeCache,
        g: &GateFn,
        root: AigRef,
        leaves: &[AigOperand],
    ) -> ConeTruth {
        if cache.deep_visits < g.gates.len().saturating_mul(SCANS_BEFORE_CACHING) {
            let cone = self.collapse(g, root, leaves);
            if cone.enclosed >= MIN_SPAN {
                cache.deep_visits = cache.deep_visits.saturating_add(cone.enclosed);
            }
            return cone;
        }
        if !cache.walks.is_empty() && !leaves.iter().any(|leaf| leaf.node == root) {
            if let Some((_, _, cone)) = cache.find(root, leaves) {
                return cone;
            }
            if let AigNode::And2 { a, b, .. } = g.gates[root.id] {
                for child in [a.node, b.node] {
                    if leaves.iter().any(|leaf| leaf.node == child) {
                        continue;
                    }
                    if let Some((i, end, cone)) = cache.find(child, leaves) {
                        let mut boundary =
                            SmallVec::<[AigOperand; LARGE_CONE_MAX_LEAVES + 1]>::from_slice(leaves);
                        boundary.push(child.into());
                        let literal = self.prepare(g, root, &boundary);
                        let width = leaves.len();
                        let walk = &cache.walks[i];
                        let outside = self
                            .touched
                            .iter()
                            .filter(|&&id| {
                                (self.local[id] - 1) / 2 > width + 1
                                    && !walk.contains(walk.slots[end].first, end, AigRef { id })
                            })
                            .count();
                        let table = LargeTruthTable::from_words(width, |block| {
                            self.word::<true>(width, block, literal, cone.table.words[block])
                        });
                        return ConeTruth {
                            table,
                            enclosed: cone.enclosed + outside,
                        };
                    }
                }
            }
        }

        let literal = self.prepare(g, root, leaves);
        let width = leaves.len();
        if self.ands.len() < MIN_SPAN {
            return ConeTruth {
                table: LargeTruthTable::from_words(width, |b| {
                    self.word::<false>(width, b, literal, 0)
                }),
                enclosed: self.ands.len(),
            };
        }
        self.save_subcones(cache, g.gates.len(), leaves)
    }

    fn save_subcones(
        &mut self,
        cache: &mut ConeCache,
        gate_count: usize,
        leaves: &[AigOperand],
    ) -> ConeTruth {
        let width = leaves.len();
        let base = width + 1;
        let count = self.ands.len();
        while cache.walks.len() >= MAX_SAVED_WALKS || cache.slots + count > 2 * gate_count {
            let oldest = cache.walks.pop_front().unwrap();
            cache.slots -= oldest.slots.len();
        }
        let mut slots: Vec<Slot> = Vec::with_capacity(count);
        let order = self
            .touched
            .iter()
            .filter_map(|&id| {
                let slot = (self.local[id] - 1) / 2;
                (slot >= base).then(|| (AigRef { id }, slot - base))
            })
            .collect();
        for (i, &[a, b]) in self.ands.iter().enumerate() {
            let mut first = i;
            let mut suffix = i;
            let mut used = 0;
            for child in [a.max(b) >> 1, a.min(b) >> 1] {
                if child >= base {
                    let j = child - base;
                    let c = &slots[j];
                    first = first.min(c.first);
                    // The later child can bridge the gap to the earlier one.
                    if j + 1 >= suffix {
                        suffix = suffix.min(c.suffix);
                    }
                    used |= c.used;
                } else if child != 0 {
                    used |= 1 << (child - 1);
                }
            }
            slots.push(Slot {
                first,
                suffix,
                used,
                table: LargeTruthTable::const0(width),
            });
        }
        // Postorder places the root in the last AND slot.
        let literal = 2 * (base + count - 1);
        let table = LargeTruthTable::from_words(width, |block| {
            let word = self.word::<false>(width, block, literal, 0);
            for (slot, &word) in slots.iter_mut().zip(&self.words[base..]) {
                slot.table.words[block] = word;
            }
            word
        });
        if width < LargeTruthTable::WITHIN_WORD_VARIABLES {
            for slot in &mut slots {
                slot.table.mask_unused_bits();
            }
        }
        cache.walks.push_back(SavedWalk {
            leaves: leaves.to_vec(),
            order,
            slots,
        });
        cache.slots += count;
        ConeTruth {
            table,
            enclosed: count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::AigBitVector;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn deep_prefixes_and_shared_fanins_reuse_exact_cones() {
        const N: usize = 512;
        let mut builder = GateBuilder::new("shared_cones".into(), GateBuilderOptions::no_opt());
        let inputs: Vec<_> = (0..3)
            .map(|i| *builder.add_input(format!("x{i}"), 1).get_lsb(0))
            .collect();
        let mut previous = inputs[0];
        let mut chain = Vec::new();
        for i in 0..N {
            previous = builder.add_and_binary(previous.negate(), inputs[i % inputs.len()]);
            chain.push(previous);
        }
        let branches: Vec<_> = (0..N)
            .map(|i| {
                let a = *builder.add_input(format!("a{i}"), 1).get_lsb(0);
                let b = *builder.add_input(format!("b{i}"), 1).get_lsb(0);
                let side = builder.add_and_binary(a, b);
                let root = builder.add_and_binary(previous, side);
                (root, a, b)
            })
            .collect();
        let halfway = chain[N / 2];
        let side = builder.add_and_binary(halfway, inputs[1]);
        let overlap = builder.add_and_binary(previous, side);
        builder.add_output("out".into(), AigBitVector::from_bit(branches[0].0));
        let graph = builder.build();
        let mut cached = CutEvaluator::default();
        let mut reference = CutEvaluator::default();
        let mut cache = ConeCache::default();
        let mut compare = |root, leaves: &[AigOperand]| {
            let expected = reference.collapse(&graph, root, leaves);
            let actual = cached.collapse_reusing(&mut cache, &graph, root, leaves);
            assert_eq!(
                (actual.table, actual.enclosed),
                (expected.table, expected.enclosed)
            );
        };
        for root in chain.iter().rev() {
            compare(root.node, &inputs);
        }
        for (root, a, b) in &branches {
            compare(root.node, &[*a, inputs[2], inputs[0], *b, inputs[1]]);
        }
        // Introducing an internal wire into the boundary splits the saved
        // cone, even when all of its former inputs still appear in the cut.
        let (root, a, b) = branches[0];
        compare(root.node, &[a, b, halfway, inputs[0], inputs[1], inputs[2]]);
        // Both sides contain the first half of the chain. Combining their
        // enclosed counts would count the shared ANDs twice.
        compare(overlap.node, &inputs);
        assert!(reference.and_visits > N * N);
        assert!(
            cached.and_visits
                < SCANS_BEFORE_CACHING * graph.gates.len() + MIN_SPAN * MIN_SPAN + 8 * N
        );

        // The same physical indices can name different gates after an edit.
        cache.clear();
        let mut edited = graph;
        let AigNode::And2 { a, .. } = &mut edited.gates[halfway.node.id] else {
            unreachable!();
        };
        *a = inputs[0];
        let actual = cached.collapse_reusing(&mut cache, &edited, previous.node, &inputs);
        let expected = reference.collapse(&edited, previous.node, &inputs);
        assert_eq!(
            (actual.table, actual.enclosed),
            (expected.table, expected.enclosed)
        );
        assert!(actual.enclosed < N);
    }
}
