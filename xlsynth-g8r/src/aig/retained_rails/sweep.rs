// SPDX-License-Identifier: Apache-2.0

use super::{CutBit, CutIndex, RailBudget};
use crate::aig::bulk_replace::SubstitutionMap;
use crate::aig::dce::dce;
use crate::aig::get_summary_stats::get_output_cone_stats;
use crate::aig::{AigNode, AigOperand, AigRef, GateFn};
use std::collections::BTreeSet;

/// Redirect equivalent signed edges at outputs and exits from the nominated
/// full-care cut. Return a replacement only if all output deadlines and live
/// signal tags are preserved and the union of live AND gates shrinks.
pub(crate) fn coalesce_cut_boundary(
    stock: &GateFn,
    bits: &[CutBit],
    budget: &mut RailBudget,
) -> Option<GateFn> {
    let index = CutIndex::build(stock, bits, budget)?;
    let work = stock.gates.len().saturating_add(
        stock
            .outputs
            .iter()
            .map(|p| p.get_bit_count())
            .sum::<usize>(),
    );
    if !budget.take(work) {
        return None;
    }
    let mut frontier = vec![false; stock.gates.len()];
    let mut tagged = BTreeSet::new();
    for edge in stock.post_order_operands(false) {
        if stock.gates[edge.node.id]
            .get_tags()
            .is_some_and(|tags| !tags.is_empty())
        {
            tagged.insert(edge.node.id);
        }
        if let AigNode::And2 { a, b, .. } = &stock.gates[edge.node.id] {
            if index.signature(edge).is_none() {
                frontier[a.node.id] = true;
                frontier[b.node.id] = true;
            }
        }
    }
    for output in &stock.outputs {
        for edge in output.bit_vector.iter_lsb_to_msb() {
            frontier[edge.node.id] = true;
        }
    }
    let mut substitutions = SubstitutionMap::new();
    for (id, on_boundary) in frontier.into_iter().enumerate() {
        if !on_boundary || !matches!(stock.gates[id], AigNode::And2 { .. }) {
            continue;
        }
        let old = AigRef { id };
        // Every signed class selects one lowest-level member, so substitutions
        // cannot chain or introduce a cycle.
        if let Some(best) = index.best_equivalent(old.into()) {
            if best.node != old {
                substitutions.add(old, best);
            }
        }
    }
    if substitutions.len() == 0 || !budget.take(work.saturating_mul(4)) {
        return None;
    }
    let redirect = |edge: AigOperand| {
        substitutions.get(&edge.node).map_or(edge, |&value| {
            if edge.negated { value.negate() } else { value }
        })
    };
    let mut trial = stock.clone();
    // An inverted edge toggles on exactly the same transitions. Only the
    // substituted signal shares this attribution; its abandoned fanins do not.
    for (old, replacement) in substitutions.iter() {
        trial.gates[replacement.node.id].union_pir_node_ids_from_node(&stock.gates[old.id]);
    }
    for gate in &mut trial.gates {
        if let AigNode::And2 { a, b, .. } = gate {
            *a = redirect(*a);
            *b = redirect(*b);
        }
    }
    for output in &mut trial.outputs {
        for bit in 0..output.get_bit_count() {
            let replacement = redirect(*output.bit_vector.get_lsb(bit));
            output.bit_vector.set_lsb(bit, replacement);
        }
    }
    // Existing tags observe the positive phase at their original node,
    // including internal ancestors that lose their last use when a boundary is
    // replaced.
    if !tagged.is_empty() {
        for live in trial.post_order_operands(false) {
            tagged.remove(&live.node.id);
        }
        if !tagged.is_empty() {
            return None;
        }
    }
    let old = get_output_cone_stats(stock);
    let new = get_output_cone_stats(&trial);
    if new.and_nodes < old.and_nodes
        && new
            .root_depths
            .iter()
            .zip(old.root_depths)
            .all(|(&n, p)| n <= p)
    {
        log::debug!(
            "retained cut coalesced={} live_ands={}->{}",
            substitutions.len(),
            old.and_nodes,
            new.and_nodes
        );
        Some(dce(&trial))
    } else {
        None
    }
}
