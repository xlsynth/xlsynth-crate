// SPDX-License-Identifier: Apache-2.0

//! 4-input cut enumeration and cut-db candidate construction.

use std::collections::BTreeSet;
use std::time::Instant;

use crate::aig::dynamic_depth::DynamicDepthState;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::cut_db::loader::CutDb;
use crate::cut_db::tt16::TruthTable16;

use super::{Replacement, ReplacementImpl, live_forward_depth};

const EMPTY_LEAF_OPERAND: AigOperand = AigOperand {
    node: AigRef { id: 0 },
    negated: false,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LeafSet {
    len: u8,
    leaves: [AigOperand; 4],
}

impl LeafSet {
    fn empty() -> Self {
        Self {
            len: 0,
            leaves: [EMPTY_LEAF_OPERAND; 4],
        }
    }

    fn singleton(leaf: AigOperand) -> Self {
        let mut out = Self::empty();
        out.leaves[0] = leaf;
        out.len = 1;
        out
    }

    pub(super) fn len(self) -> usize {
        self.len as usize
    }

    pub(super) fn as_slice(&self) -> &[AigOperand] {
        &self.leaves[..self.len()]
    }

    fn iter(&self) -> impl Iterator<Item = &AigOperand> {
        self.as_slice().iter()
    }

    pub(super) fn to_vec(self) -> Vec<AigOperand> {
        self.as_slice().to_vec()
    }

    fn union(lhs: Self, rhs: Self) -> Option<Self> {
        let lhs = lhs.as_slice();
        let rhs = rhs.as_slice();
        let mut out = [EMPTY_LEAF_OPERAND; 4];
        let mut out_len = 0usize;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < lhs.len() || j < rhs.len() {
            let next = if i == lhs.len() {
                let value = rhs[j];
                j += 1;
                value
            } else if j == rhs.len() {
                let value = lhs[i];
                i += 1;
                value
            } else if lhs[i] < rhs[j] {
                let value = lhs[i];
                i += 1;
                value
            } else if rhs[j] < lhs[i] {
                let value = rhs[j];
                j += 1;
                value
            } else {
                let value = lhs[i];
                i += 1;
                j += 1;
                value
            };

            if out_len != 0 && out[out_len - 1] == next {
                continue;
            }
            if out_len == out.len() {
                return None;
            }
            out[out_len] = next;
            out_len += 1;
        }

        Some(Self {
            len: out_len as u8,
            leaves: out,
        })
    }
}

impl PartialOrd for LeafSet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeafSet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Cut {
    pub(super) leaves: LeafSet,
    pub(super) tt: TruthTable16,
}

fn push_unique_cut(cuts: &mut Vec<Cut>, cut: Cut) -> bool {
    if cuts.iter().any(|existing| existing == &cut) {
        false
    } else {
        cuts.push(cut);
        true
    }
}

fn union_cut_leaves(lhs: LeafSet, rhs: LeafSet) -> Option<LeafSet> {
    LeafSet::union(lhs, rhs)
}

fn sort_and_prune_cuts(mut cuts: Vec<Cut>, max_cuts_per_node: usize) -> Vec<Cut> {
    cuts.sort();
    cuts.dedup();
    if max_cuts_per_node != 0 && cuts.len() > max_cuts_per_node {
        cuts.truncate(max_cuts_per_node);
    }
    cuts
}

fn negate_tt(tt: TruthTable16) -> TruthTable16 {
    tt.not()
}

/// Remaps `tt` from `old_leaves` order into the larger `union_leaves` order.
fn embed_tt_into_union(
    tt: TruthTable16,
    old_leaves: LeafSet,
    union_leaves: LeafSet,
) -> TruthTable16 {
    if old_leaves.len() == 0 {
        return tt;
    }
    let mut map: [usize; 4] = [0; 4];
    for (i, leaf) in old_leaves.iter().enumerate() {
        let j = union_leaves
            .as_slice()
            .binary_search(leaf)
            .expect("old leaf must appear in union");
        map[i] = j;
    }
    let old_len = old_leaves.len();
    let mut out = TruthTable16::const0();
    for assign in 0u8..16 {
        let mut old_assign: u8 = 0;
        for i in 0..old_len {
            let bit = ((assign >> map[i]) & 1) != 0;
            old_assign |= (bit as u8) << i;
        }
        let bit = tt.get_bit(old_assign);
        out.set_bit(assign, bit);
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CutEnumerationStats {
    pub(super) computed_nodes: usize,
    pub(super) total_cuts: usize,
    pub(super) truncated_nodes: usize,
    pub(super) elapsed_ms: u128,
}

impl CutEnumerationStats {
    pub(super) fn add(&mut self, other: CutEnumerationStats) {
        self.computed_nodes += other.computed_nodes;
        self.total_cuts += other.total_cuts;
        self.truncated_nodes += other.truncated_nodes;
        self.elapsed_ms += other.elapsed_ms;
    }
}

/// Enumerates and prunes 4-feasible cuts for one graph snapshot.
///
/// The memo is shared across roots in a rewrite sweep. Callers invalidate the
/// rewritten fanout cone after each committed replacement so unrelated
/// descendant cut sets remain cached. Cut sets are populated iteratively from
/// fanins to roots, avoiding recursion on deep AIGs.
pub(super) struct CutEnumerator {
    max_cuts_per_node: usize,
    memo: Vec<Option<Vec<Cut>>>,
    stats: CutEnumerationStats,
}

impl CutEnumerator {
    pub(super) fn new(g: &GateFn, max_cuts_per_node: usize) -> Self {
        Self {
            max_cuts_per_node,
            memo: vec![None; g.gates.len()],
            stats: CutEnumerationStats::default(),
        }
    }

    pub(super) fn stats(&self) -> CutEnumerationStats {
        self.stats
    }

    pub(super) fn sync_len(&mut self, g: &GateFn) {
        if self.memo.len() < g.gates.len() {
            self.memo.resize(g.gates.len(), None);
        }
    }

    fn invalidate_node(&mut self, node: AigRef) {
        if let Some(entry) = self.memo.get_mut(node.id) {
            *entry = None;
        }
    }

    pub(super) fn invalidate_nodes(&mut self, nodes: &BTreeSet<AigRef>) {
        for node in nodes {
            self.invalidate_node(*node);
        }
    }

    pub(super) fn cuts_for_root(&mut self, g: &GateFn, root: AigRef) -> Vec<Cut> {
        let t0 = Instant::now();
        self.ensure_cuts_for_node(g, root);
        let cuts = self.memo[root.id]
            .as_ref()
            .expect("root cuts should be computed")
            .clone();
        self.stats.elapsed_ms += t0.elapsed().as_millis();
        cuts
    }

    fn cuts_for_operand_from_memo(&self, op: AigOperand) -> Vec<Cut> {
        let mut cuts = self.memo[op.node.id]
            .as_ref()
            .expect("operand cuts should be computed before parent")
            .clone();
        if op.negated {
            for c in &mut cuts {
                c.tt = negate_tt(c.tt);
            }
        }
        cuts
    }

    fn ensure_cuts_for_node(&mut self, g: &GateFn, root: AigRef) {
        self.sync_len(g);
        if self.memo[root.id].is_some() {
            return;
        }

        let mut stack = vec![(root, false)];
        while let Some((node, expanded)) = stack.pop() {
            if self.memo[node.id].is_some() {
                continue;
            }
            let fanins = match &g.gates[node.id] {
                AigNode::And2 { a, b, .. } => Some((*a, *b)),
                AigNode::Input { .. } | AigNode::Literal { .. } => None,
            };
            if !expanded {
                stack.push((node, true));
                if let Some((a, b)) = fanins {
                    if self.memo[b.node.id].is_none() {
                        stack.push((b.node, false));
                    }
                    if self.memo[a.node.id].is_none() {
                        stack.push((a.node, false));
                    }
                }
                continue;
            }

            if let Some((a, b)) = fanins {
                if self.memo[a.node.id].is_none() || self.memo[b.node.id].is_none() {
                    stack.push((node, true));
                    if self.memo[b.node.id].is_none() {
                        stack.push((b.node, false));
                    }
                    if self.memo[a.node.id].is_none() {
                        stack.push((a.node, false));
                    }
                    continue;
                }
            }

            let cuts = self.compute_cuts_for_node_from_memo(g, node);
            self.memo[node.id] = Some(cuts);
        }
    }

    fn compute_cuts_for_node_from_memo(&mut self, g: &GateFn, r: AigRef) -> Vec<Cut> {
        let mut cuts: Vec<Cut> = Vec::new();

        // Trivial self-cut: allow this node to be used as a leaf for its fanout.
        push_unique_cut(
            &mut cuts,
            Cut {
                leaves: LeafSet::singleton(AigOperand {
                    node: r,
                    negated: false,
                }),
                tt: TruthTable16::var(0),
            },
        );

        let node_kind = match &g.gates[r.id] {
            AigNode::Input { .. } => None,
            AigNode::Literal { value: v, .. } => {
                // Constant cut with no leaves.
                push_unique_cut(
                    &mut cuts,
                    Cut {
                        leaves: LeafSet::empty(),
                        tt: if *v {
                            TruthTable16::const1()
                        } else {
                            TruthTable16::const0()
                        },
                    },
                );
                None
            }
            AigNode::And2 { a, b, .. } => Some((*a, *b)),
        };

        if let Some((a, b)) = node_kind {
            let a_cuts = self.cuts_for_operand_from_memo(a);
            let b_cuts = self.cuts_for_operand_from_memo(b);
            'pairs: for ca in &a_cuts {
                for cb in &b_cuts {
                    let Some(union_leaves) = union_cut_leaves(ca.leaves, cb.leaves) else {
                        continue;
                    };
                    let ca_tt = embed_tt_into_union(ca.tt, ca.leaves, union_leaves);
                    let cb_tt = embed_tt_into_union(cb.tt, cb.leaves, union_leaves);
                    let tt = ca_tt.and(cb_tt);
                    push_unique_cut(
                        &mut cuts,
                        Cut {
                            leaves: union_leaves,
                            tt,
                        },
                    );
                    if self.max_cuts_per_node != 0 && cuts.len() >= self.max_cuts_per_node {
                        self.stats.truncated_nodes += 1;
                        break 'pairs;
                    }
                }
            }
        }

        let v = sort_and_prune_cuts(cuts, self.max_cuts_per_node);
        self.stats.computed_nodes += 1;
        self.stats.total_cuts += v.len();
        v
    }
}

/// Picks a small, deterministically ordered set of candidate replacements for
/// `root` using `db` and the current depth map.
pub(super) fn choose_candidate_replacements_for_root(
    root: AigRef,
    root_cuts: &[Cut],
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
    db: &CutDb,
) -> Vec<Replacement> {
    let mut cands: Vec<Replacement> = Vec::new();

    for cut in root_cuts {
        let cut_leaves = cut.leaves.as_slice();
        // Skip trivial self-cut.
        if cut_leaves.len() == 1 && cut_leaves[0].node == root && !cut_leaves[0].negated {
            continue;
        }
        if cut.leaves.len() > 4 {
            continue;
        }

        let (xform, pareto) = db.lookup(cut.tt.0);
        for p in pareto {
            let frag = p.frag.apply_npn(xform);
            let input_depths = frag.input_depths();
            let mut new_depth_at_root: usize = 0;
            for (i, leaf) in cut_leaves.iter().enumerate() {
                let leaf_depth = live_forward_depth(depth_state, structural_hash_state, leaf.node);
                let cand = leaf_depth + (input_depths[i] as usize);
                new_depth_at_root = core::cmp::max(new_depth_at_root, cand);
            }

            let score_depth = new_depth_at_root;
            let score_ands = p.ands;

            cands.push(Replacement {
                root,
                leaf_ops: cut.leaves.to_vec(),
                implementation: ReplacementImpl::Fragment { frag, input_depths },
                score_depth,
                score_ands: score_ands as usize,
                raw_score_ands: score_ands as usize,
                structural_hash_only_area_win: false,
            });
        }
    }

    // Deterministic ordering + cap for performance. `sort_by` is stable, so
    // equal-score candidates keep the deterministic generation order.
    cands.sort_by(|a, b| {
        (a.score_depth, a.score_ands, a.root.id, &a.leaf_ops).cmp(&(
            b.score_depth,
            b.score_ands,
            b.root.id,
            &b.leaf_ops,
        ))
    });
    cands.truncate(16);
    cands
}
