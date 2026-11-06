// SPDX-License-Identifier: Apache-2.0

//! Reject sum-of-products (SOP) replacements early in the bottom-up large-cone
//! area rewriter when they cannot reduce the number of live AND gates.
//!
//! For example, if at most three old ANDs can disappear, a trial that creates
//! a third AND can at best break even. Reusing an existing AND costs nothing.
//! The SOP builder keeps only gates needed by its result, so each new AND
//! counts toward the cost of the replacement.
//!
//! The removal bound counts the root and the ANDs feeding it that would lose
//! their last use with it. Reuse that count across candidates for the same
//! root. If rewiring could also merge gates downstream of the root, use the
//! exact cost calculation because those merges could save additional ANDs.
//! A replacement no deeper than the root cannot contain the old root as a
//! proper subexpression, so construction can stop as soon as it reaches the
//! bound. For a deeper replacement, finish construction and check for reuse
//! of the old root first. Candidates that might save area still get the exact
//! cost and protected-gate checks.
//!
//! In a matched comparison on six public BVC whole functions, this policy
//! reduced total IR-to-optimized-AIG time from 7.11 s to 6.82 s (4.1%).
//! All six AIGs were byte-identical with and without the early rejection.

use super::super::large_cone::SopReplacement;
use super::super::{LargeConeCandidate, ReplacementImpl, collect_mffc_nodes_under_cut_up_to};
use super::{ReplacementGateCountDiff, VirtualCostOverlay, gate_count_diff_with_protected};
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::AigRef;

pub(in crate::aig::cut_db_rewrite) enum AreaCost {
    Profitable(ReplacementGateCountDiff),
    NoAreaGain,
    Unavailable,
}

impl AreaCost {
    fn from_diff(cost: Option<ReplacementGateCountDiff>) -> Self {
        match cost {
            Some(diff) if diff.live_and_delta > 0 => Self::Profitable(diff),
            Some(_) => Self::NoAreaGain,
            None => Self::Unavailable,
        }
    }
}

/// Caches the removal bound for one root while its candidates are priced.
pub(in crate::aig::cut_db_rewrite) struct EffectiveCutPricing {
    root_depth: usize,
    search_limit: usize,
    removable: Option<usize>,
}

impl EffectiveCutPricing {
    pub(in crate::aig::cut_db_rewrite) fn new(
        root_depth: usize,
        candidates: &[LargeConeCandidate],
    ) -> Self {
        let max_raw = candidates
            .iter()
            .map(|c| c.raw_and_count)
            .max()
            .unwrap_or(0);
        Self {
            root_depth,
            search_limit: max_raw.saturating_add(1),
            removable: None,
        }
    }

    fn sop_limit<'a>(
        &mut self,
        state: &DynamicStructuralHash,
        candidate: &'a LargeConeCandidate,
    ) -> Option<(&'a SopReplacement, usize)> {
        let replacement = &candidate.replacement;
        let root = replacement.root;
        let ReplacementImpl::Sop(sop) = &replacement.implementation else {
            return None;
        };
        if candidate.raw_and_count < candidate.mffc.len()
            || !state.is_live(root)
            || replacement
                .leaf_ops
                .iter()
                .any(|op| op.node.id == 0 || op.node == root || !state.is_live(op.node))
        {
            return None;
        }
        let limit = *self.removable.get_or_insert_with(|| {
            if state.use_count(root) == 0
                || (!state.has_unique_and_keys() && state.fanout_count(root) > 1)
            {
                return self.search_limit;
            }
            // A fresh result cannot match physical keys. Relabeling the root
            // preserves distinct fanout keys; at most the whole root cone dies.
            collect_mffc_nodes_under_cut_up_to(state, root, &[], self.search_limit).len()
        });
        (limit > 0 && limit <= candidate.raw_and_count).then_some((sop, limit))
    }
}

pub(in crate::aig::cut_db_rewrite) fn price_area_candidate(
    state: &DynamicStructuralHash,
    candidate: &LargeConeCandidate,
    protected: &[AigRef],
    effective: Option<&mut EffectiveCutPricing>,
) -> Result<AreaCost, String> {
    let replacement = &candidate.replacement;
    let bounded = effective.and_then(|pricing| {
        let shallow = candidate.new_root_depth <= pricing.root_depth;
        pricing
            .sop_limit(state, candidate)
            .map(|(sop, limit)| (sop, limit, shallow))
    });
    let Some((sop, limit, shallow)) = bounded else {
        return gate_count_diff_with_protected(state, replacement, protected)
            .map(AreaCost::from_diff);
    };

    // The SOP builder folds constants and compacts to the root cone. Every
    // fresh AND remains live through the fresh result. In a shallow SOP no
    // proper subexpression can reconstruct the old root at its physical depth.
    let mut overlay = VirtualCostOverlay::new(state, Some(replacement.root));
    let output = if shallow {
        let Some(output) =
            overlay.instantiate_sop_until(sop, &replacement.leaf_ops, |fresh| fresh >= limit)?
        else {
            return Ok(AreaCost::NoAreaGain);
        };
        output
    } else {
        let output = overlay.instantiate_sop(sop, &replacement.leaf_ops)?;
        // A deeper SOP can reuse the physical root. Preserve the ordinary
        // numerical cost and its errors when that happens.
        if overlay.tracking_root.is_some() && overlay.virtual_ands.len() >= limit {
            return Ok(AreaCost::NoAreaGain);
        }
        output
    };
    if protected.is_empty() {
        overlay.tracking_root = None;
    }
    overlay
        .finish_cost(replacement, output, protected)
        .map(|cost| AreaCost::from_diff(Some(cost)))
}
