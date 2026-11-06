// SPDX-License-Identifier: Apache-2.0

//! Choose cuts for the bottom-up large-cone area rewriter in two steps.
//! First retain a few boundaries per width for use by each gate's parents.
//! Then, at the root being rewritten, choose up to four to factor. Retention
//! uses cheap structural estimates. Factoring uses the current truth table
//! and the number of ANDs the root could release.

mod batched_mffc;
mod compact_leaves;
mod singleton;

use self::compact_leaves::CompactLeaves;
use super::cut_truth::{self, ConeCache, CutEvaluator};
use crate::aig::cut_db_rewrite::CutFaninWalk;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

use super::{
    AigNode, AigOperand, AigRef, ConeTruth, CutForFactoring, CutMffc, CutsForFactoring,
    DynamicStructuralHash, GateFn, LARGE_CONE_MAX_LEAVES, collect_mffc_nodes_under_cut,
    positive_operand,
};
use smallvec::{SmallVec, smallvec};

// Two equal-width cuts can have different leaves. Either may overlap better
// with the other input at a later gate. Keep a second choice at each width
// without multiplying the number of child pairs as much as a larger list.
const PER_WIDTH: usize = 2;
// Factoring constructs several circuits per cut. Limit that work at this
// root independently of how many boundaries it retains for its parents.
pub(super) const MAX_FACTORED: usize = 4;
type Leaves = SmallVec<[AigRef; LARGE_CONE_MAX_LEAVES]>;

/// Structural priority inherited by parent cuts. The unfolded tree size can
/// saturate even on a small reconvergent DAG.
#[derive(Clone, Copy)]
struct Priority {
    tree_ands: usize,
    overlap: usize,
    latest: usize,
    total_depth: usize,
}

impl Priority {
    fn merged(left: &Boundary, right: &Boundary, width: usize, total_depth: usize) -> Self {
        Self {
            tree_ands: left
                .priority
                .tree_ands
                .saturating_add(right.priority.tree_ands)
                .saturating_add(1),
            overlap: left.leaves.len() + right.leaves.len() - width,
            latest: left.priority.latest.max(right.priority.latest),
            total_depth,
        }
    }

    /// A replacement that uses a late leaf must wait for it. Prefer an
    /// earlier latest leaf, then a lower sum of arrivals to leave more room
    /// on other paths. On a tie, a larger unfolded tree or more overlap can
    /// expose more logic or sharing. The tree estimate can count a gate
    /// reached along two paths twice.
    fn order(&self, other: &Self) -> Ordering {
        self.latest
            .cmp(&other.latest)
            .then_with(|| self.total_depth.cmp(&other.total_depth))
            .then_with(|| other.tree_ands.cmp(&self.tree_ands))
            .then_with(|| other.overlap.cmp(&self.overlap))
    }
}

/// One retained cut, using physical gate IDs in ascending order.
#[derive(Clone)]
struct Boundary {
    leaves: Leaves,
    priority: Priority,
}

impl Boundary {
    fn unit(node: AigRef, depth: usize) -> Self {
        Self {
            leaves: smallvec![node],
            priority: Priority {
                tree_ands: 0,
                overlap: 0,
                latest: depth,
                total_depth: depth,
            },
        }
    }

    fn operands(&self) -> Vec<AigOperand> {
        self.leaves.iter().copied().map(positive_operand).collect()
    }
}

/// Current graph measurements used to choose the four cuts to factor.
struct ScoredBoundary {
    leaves: Vec<AigOperand>,
    cone: ConeTruth,
    removable: usize,
    support: usize,
}

impl ScoredBoundary {
    fn new(leaves: Vec<AigOperand>, cone: ConeTruth, removable: usize) -> Self {
        let support = cone.table.effective_input_count();
        Self {
            leaves,
            cone,
            removable,
            support,
        }
    }

    /// Spend factoring time where many gates might be replaced by a function
    /// of few inputs. An input is essential if flipping it can flip the
    /// output with all other leaves fixed. This ratio ranks opportunities.
    /// The later cost calculation decides whether a replacement saves gates.
    fn factor_order(&self, other: &Self) -> Ordering {
        // Compare the ratios without rounding. Use one as the denominator
        // for constants. For equal ratios prefer less physical boundary per
        // removable AND, then more enclosed gates, then stable leaf order.
        let density_order = |width, other_width| {
            ((other.removable as u128) * (width as u128))
                .cmp(&((self.removable as u128) * (other_width as u128)))
        };
        density_order(self.support.max(1), other.support.max(1))
            .then_with(|| density_order(self.leaves.len(), other.leaves.len()))
            .then_with(|| other.cone.enclosed.cmp(&self.cone.enclosed))
            .then_with(|| self.leaves.cmp(&other.leaves))
    }
}

/// Boundaries and deep-cut measurements owned by an area-rewrite round.
/// Each root visit measures removability and candidate costs on the current
/// graph.
#[derive(Default)]
pub(in crate::aig::cut_db_rewrite) struct EffectiveCuts {
    max_cuts_per_node: usize,
    by_node: FxHashMap<AigRef, Vec<Boundary>>,
    signatures: singleton::SingletonSignatures,
    evaluator: CutEvaluator,
    deep_truth: ConeCache,
    mffc: batched_mffc::BatchedMffc,
    compact: CompactLeaves,
    walk: CutFaninWalk,
}

impl EffectiveCuts {
    pub(in crate::aig::cut_db_rewrite) fn new(max_cuts_per_node: usize) -> Self {
        Self {
            max_cuts_per_node,
            ..Self::default()
        }
    }

    pub(in crate::aig::cut_db_rewrite) fn cannot_save_singleton(
        &mut self,
        state: &DynamicStructuralHash,
        root: AigRef,
    ) -> bool {
        self.signatures.cannot_save(state, root)
    }

    pub(in crate::aig::cut_db_rewrite) fn invalidate(
        &mut self,
        nodes: impl Iterator<Item = AigRef>,
    ) {
        self.deep_truth.clear();
        for node in nodes {
            self.by_node.remove(&node);
        }
    }

    /// Fill missing boundaries in fanin order. A shared gate's retained cuts
    /// can serve several parents until a graph edit invalidates them.
    fn fill(&mut self, g: &GateFn, root: AigRef, depths: &[usize]) {
        self.walk.start(root);
        while let Some(node) = self.walk.next_ready(g, |n| self.by_node.contains_key(&n)) {
            let gate = &g.gates[node.id];
            let mut unit = Boundary::unit(node, depths[node.id]);
            if matches!(gate, AigNode::Literal { .. }) {
                unit.leaves.clear();
                unit.priority.latest = 0;
                unit.priority.total_depth = 0;
            }
            let mut retained = vec![unit];
            if let AigNode::And2 { a, b, .. } = gate {
                let left = &self.by_node[&a.node];
                let right = &self.by_node[&b.node];
                self.compact.retain(left, right, depths, &mut retained);
                // Keep the unit cut so a parent can stop here. If the caller's
                // cap binds, narrower cuts leave room for the other fanin.
                if self.max_cuts_per_node != 0 {
                    retained.truncate(self.max_cuts_per_node);
                }
            }
            self.by_node.insert(node, retained);
        }
    }

    /// Offer every eligible cut if it fits the factoring budget. Otherwise
    /// measure the cuts on this graph and take the four highest ranked.
    pub(super) fn select(
        &mut self,
        root: AigRef,
        state: &DynamicStructuralHash,
        depths: &[usize],
    ) -> CutsForFactoring {
        self.fill(state.gate_fn(), root, depths);
        if depths[root.id] < cut_truth::MIN_SPAN {
            self.select_from_retained::<false>(root, state, depths[root.id])
        } else {
            self.select_from_retained::<true>(root, state, depths[root.id])
        }
    }

    fn select_from_retained<const DEEP: bool>(
        &mut self,
        root: AigRef,
        state: &DynamicStructuralHash,
        root_depth: usize,
    ) -> CutsForFactoring {
        let eligible = self.by_node[&root]
            .iter()
            .filter(|cut| cut.priority.tree_ands > 1 && cut.leaves.len() >= 2);
        let cut_leaves = eligible.clone().map(|cut| cut.leaves.as_slice());
        let evaluated = eligible.map(|cut| {
            let leaves = cut.operands();
            let cone =
                if DEEP && root_depth.saturating_sub(cut.priority.latest) >= cut_truth::MIN_SPAN {
                    self.evaluator.collapse_reusing(
                        &mut self.deep_truth,
                        state.gate_fn(),
                        root,
                        &leaves,
                    )
                } else {
                    self.evaluator.collapse(state.gate_fn(), root, &leaves)
                };
            (leaves, cone)
        });
        let mut chosen: CutsForFactoring = if cut_leaves.clone().count() > MAX_FACTORED {
            let boundaries: SmallVec<[&[AigRef]; 16]> = cut_leaves.collect();
            let counts = self.mffc.counts(state, root, &boundaries);
            let mut scored: Vec<_> = evaluated
                .zip(counts)
                .map(|((leaves, cone), removable)| ScoredBoundary::new(leaves, cone, removable))
                .collect();
            scored.sort_by(ScoredBoundary::factor_order);
            scored
                .into_iter()
                .take(MAX_FACTORED)
                .map(|cut| CutForFactoring {
                    leaves: cut.leaves,
                    cone: cut.cone,
                    mffc: CutMffc::Count(cut.removable),
                })
                .collect()
        } else {
            evaluated
                .map(|(leaves, cone)| {
                    let removable = collect_mffc_nodes_under_cut(state, root, &leaves).len();
                    CutForFactoring {
                        leaves,
                        cone,
                        mffc: CutMffc::Count(removable),
                    }
                })
                .collect()
        };
        // Candidate generation uses the same stable leaf order for both paths.
        chosen.sort_by(|a, b| a.leaves.cmp(&b.leaves));
        chosen
    }
}

#[cfg(test)]
#[path = "effective_cuts_tests.rs"]
mod tests;
