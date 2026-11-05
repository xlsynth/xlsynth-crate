// SPDX-License-Identifier: Apache-2.0

use std::hash::{DefaultHasher, Hash, Hasher};

use rustc_hash::FxHashMap;

use super::{AigNode, AigOperand, AigRef, DynamicStructuralHash};

/// One word simulates 64 whole-graph input assignments. An unequal word rules
/// out functional equivalence; a collision simply leaves the root searchable.
/// A round owns physical IDs. Rewrites preserve each surviving ID's function,
/// so only newly allocated IDs need simulation. Deletions retain their words
/// and may conservatively prevent a certificate until the next round.
#[derive(Default)]
pub(super) struct SingletonSignatures {
    words: Vec<u64>,
    unique: FxHashMap<u64, bool>,
    out_of_order: bool,
}

impl SingletonSignatures {
    fn operand_word(&self, operand: AigOperand) -> Option<u64> {
        let word = *self.words.get(operand.node.id)?;
        Some(if operand.negated { !word } else { word })
    }

    fn extend(&mut self, gates: &[AigNode]) -> Option<()> {
        for gate in &gates[self.words.len()..] {
            let word = match gate {
                AigNode::Literal { value, .. } => {
                    if *value {
                        u64::MAX
                    } else {
                        0
                    }
                }
                AigNode::Input {
                    name, lsb_index, ..
                } => {
                    let mut hasher = DefaultHasher::new();
                    (name, lsb_index).hash(&mut hasher);
                    hasher.finish()
                }
                AigNode::And2 { a, b, .. } => self.operand_word(*a)? & self.operand_word(*b)?,
            };
            self.words.push(word);
            self.unique
                .entry(word.min(!word))
                .and_modify(|unique| *unique = false)
                .or_insert(true);
        }
        Some(())
    }

    pub(super) fn cannot_save(&mut self, state: &DynamicStructuralHash, root: AigRef) -> bool {
        let gates = &state.gate_fn().gates;
        let AigNode::And2 { a, b, .. } = &gates[root.id] else {
            return false;
        };
        if self.out_of_order
            || state.use_count(root) == 0
            || (!state.has_unique_and_keys() && state.fanout_count(root) > 1)
        {
            return false;
        }
        // Both slots count when the same AND is a child twice. Every child AND
        // must keep a use outside the root, so only the root itself could die.
        let removed_edges = 1 + usize::from(a.node == b.node);
        if [a.node, b.node].into_iter().any(|child| {
            matches!(gates[child.id], AigNode::And2 { .. })
                && state.use_count(child) <= removed_edges
        }) {
            return false;
        }
        if self.extend(gates).is_none() {
            // A forward reference in the append-only graph needs a different
            // evaluation order. Resume ordinary search for the rest of the
            // round.
            self.out_of_order = true;
            return false;
        }
        // An equivalent compact SOP must return the old root or a fresh AND
        // when no other signed literal (including constants) matches. The
        // fresh result has no users before substitution. Re-keying its users
        // cannot collide under the fanout condition above, so it costs at
        // least one AND to remove at most one.
        let word = self.words[root.id];
        let class = word.min(!word);
        class != 0 && self.unique[&class]
    }
}

#[cfg(test)]
mod tests;
