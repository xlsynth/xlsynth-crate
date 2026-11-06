// SPDX-License-Identifier: Apache-2.0

//! Evaluate exact cut truth in a compact, cut-local gate order.

use super::{AigNode, AigOperand, AigRef, ConeTruth, GateFn, LargeTruthTable};

mod cache;
pub(super) use cache::{ConeCache, MIN_SPAN};

/// Reusable truth evaluation scratch for one large-cone rewrite round.
///
/// This walk reuses its buffers and records the operand slots of each AND.
/// An earlier evaluator used `postorder_for_aig_ref` and a fresh hash map
/// of truth tables per cut. Both produced identical AIGs on six public
/// whole functions. With bottom-up rewriting enabled, compiling IR to optimized
/// AIGs took 6.95 s with the compact program and 8.52 s with the generic
/// walk (+22.53%).
///
/// A local literal is twice its word slot plus its complemented-edge bit.
/// Slot zero always contains Boolean false. Slots 1..=width are the leaves.
/// A reused fanin, when present, occupies the next slot. The AND list follows
/// in postorder so each instruction reads earlier slots.
#[derive(Default)]
pub(in crate::aig::cut_db_rewrite) struct CutEvaluator {
    #[cfg(test)]
    and_visits: usize,
    local: Vec<usize>,
    touched: Vec<usize>,
    pending: Vec<AigRef>,
    ands: Vec<[usize; 2]>,
    words: Vec<u64>,
}

impl CutEvaluator {
    fn remember(&mut self, node: usize, literal: usize) {
        self.local[node] = literal + 1;
        self.touched.push(node);
    }

    fn root_literal(&mut self, g: &GateFn, root: AigRef, width: usize) -> usize {
        // A narrow cut can span arbitrarily many gates. Keep the DFS path
        // in reusable heap storage and emit an AND only after both fanins.
        self.pending.clear();
        self.pending.push(root);
        while let Some(&node) = self.pending.last() {
            if self.local[node.id] == 0 {
                let literal = match g.gates[node.id] {
                    AigNode::Literal { value, .. } => usize::from(value),
                    AigNode::Input { .. } => panic!("unbound cut input {node:?}"),
                    AigNode::And2 { a, b, .. } => {
                        let lhs = self.local[a.node.id];
                        let rhs = self.local[b.node.id];
                        if lhs == 0 || rhs == 0 {
                            self.pending.push(if lhs == 0 { a.node } else { b.node });
                            continue;
                        }
                        self.ands.push([
                            (lhs - 1) ^ usize::from(a.negated),
                            (rhs - 1) ^ usize::from(b.negated),
                        ]);
                        2 * (width + self.ands.len())
                    }
                };
                self.remember(node.id, literal);
            }
            self.pending.pop();
        }
        self.local[root.id] - 1
    }

    fn prepare(&mut self, g: &GateFn, root: AigRef, leaves: &[AigOperand]) -> usize {
        // Clear only nodes used on the preceding cut. Node numbers may be
        // reused after graph edits within the same round.
        for node in self.touched.drain(..) {
            self.local[node] = 0;
        }
        self.local.resize(g.gates.len(), 0);
        self.ands.clear();
        for (variable, leaf) in leaves.iter().enumerate() {
            assert_eq!(self.local[leaf.node.id], 0, "duplicate cut boundary");
            self.remember(leaf.node.id, 2 * (variable + 1));
        }
        let root_literal = self.root_literal(g, root, leaves.len());
        #[cfg(test)]
        {
            self.and_visits += self.ands.len();
        }
        self.words.resize(1 + leaves.len() + self.ands.len(), 0);
        root_literal
    }

    fn word<const CACHED: bool>(
        &mut self,
        width: usize,
        block: usize,
        root_literal: usize,
        cached_word: u64,
    ) -> u64 {
        let words = &mut self.words;
        for (var, slot) in words[1..1 + width].iter_mut().enumerate() {
            *slot = LargeTruthTable::variable_word(var, block);
        }
        if CACHED {
            words[width + 1] = cached_word;
        }
        for (i, &[a, b]) in self.ands.iter().enumerate() {
            let lhs = words[a >> 1] ^ 0u64.wrapping_sub((a & 1) as u64);
            let rhs = words[b >> 1] ^ 0u64.wrapping_sub((b & 1) as u64);
            words[width + 1 + usize::from(CACHED) + i] = lhs & rhs;
        }
        words[root_literal >> 1] ^ 0u64.wrapping_sub((root_literal & 1) as u64)
    }
}

impl CutEvaluator {
    /// Treat the boundary signals as independent variables and compute the
    /// root's function over them. Internal wires work just like primary
    /// inputs here. Reuse this table for support ranking and for factoring.
    pub(super) fn collapse(
        &mut self,
        g: &GateFn,
        root: AigRef,
        leaves: &[AigOperand],
    ) -> ConeTruth {
        let width = leaves.len();
        let literal = self.prepare(g, root, leaves);
        let table = LargeTruthTable::from_words(width, |block| {
            self.word::<false>(width, block, literal, 0)
        });
        ConeTruth {
            table,
            enclosed: self.ands.len(),
        }
    }
}

/// Independent scalar evaluation of one assignment to the ordered cut leaves.
#[cfg(test)]
pub(super) fn eval_cut_row(g: &GateFn, node: AigRef, leaves: &[AigOperand], row: usize) -> bool {
    if let Some(i) = leaves.iter().position(|leaf| leaf.node == node) {
        return row & (1 << i) != 0;
    }
    match g.gates[node.id] {
        AigNode::Input { .. } => panic!("unbound test input"),
        AigNode::Literal { value, .. } => value,
        AigNode::And2 { a, b, .. } => {
            (eval_cut_row(g, a.node, leaves, row) ^ a.negated)
                & (eval_cut_row(g, b.node, leaves, row) ^ b.negated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::AigBitVector;
    use crate::aig::cut_db_rewrite::collect_internal_and_nodes_under_cut;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn packed_truth_and_enclosed_count_match_independent_oracles() {
        let mut gb = GateBuilder::new("fused_cut".into(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".into(), 1).get_lsb(0);
        let b = *gb.add_input("b".into(), 1).get_lsb(0);
        let c = *gb.add_input("c".into(), 1).get_lsb(0);
        let zero = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        let shared = gb.add_and_binary(a, b.negate());
        let dup = gb.add_and_binary(shared, shared);
        let with_literal = gb.add_and_binary(dup, zero.negate());
        let side = gb.add_and_binary(shared.negate(), c);
        let root = gb.add_and_binary(with_literal, side.negate());
        gb.add_output("out".into(), AigBitVector::from_bit(root));
        let g = gb.build();
        let optional = [shared, dup, with_literal, side, root];
        let mut evaluator = CutEvaluator::default();
        for mask in 0..1usize << optional.len() {
            let mut leaves = vec![c, a, b];
            for (i, leaf) in optional.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    leaves.push(*leaf);
                }
            }
            let cone = evaluator.collapse(&g, root.node, &leaves);
            let enclosed = collect_internal_and_nodes_under_cut(&g, root.node, &leaves);
            assert_eq!(cone.enclosed, enclosed.len(), "mask={mask}");
            for row in 0..1usize << leaves.len() {
                assert_eq!(
                    cone.table.get_bit(row),
                    eval_cut_row(&g, root.node, &leaves, row),
                    "mask={mask} row={row}"
                );
            }
        }
    }
    #[test]
    fn narrow_cut_with_deep_interior() {
        let mut gb = GateBuilder::new("deep_cut".into(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".into(), 1).get_lsb(0);
        let b = *gb.add_input("b".into(), 1).get_lsb(0);
        let mut root = a;
        for _ in 0..20_000 {
            root = gb.add_and_binary(root, b);
        }
        gb.add_output("out".into(), AigBitVector::from_bit(root));
        let graph = gb.build();
        let result = CutEvaluator::default().collapse(&graph, root.node, &[a, b]);
        assert_eq!(result.enclosed, 20_000);
        assert_eq!(result.table.active_words(), &[0b1000]);
    }
}
