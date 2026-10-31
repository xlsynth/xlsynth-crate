// SPDX-License-Identifier: Apache-2.0

//! Lower priority predicates using only the observed conditional-negation bits.

use super::GateEnv;
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::{GateBuilder, ReductionKind};
use crate::prefix_scan_utils::{PrefixScanStrategy, prefix_scan_inclusive};
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_utils;

/// Plans priority thresholds whose selected high bits avoid serial carries.
#[derive(Default)]
pub(super) struct PriorityPredicatePlan {
    selections: HashMap<NodeRef, Selection>,
    reductions: HashMap<NodeRef, (NodeRef, usize)>,
}

/// Captures a conditional negation and the first high bit its consumers
/// observe.
struct Selection {
    value: NodeRef,
    prefix_source: PrefixSource,
    selector: NodeRef,
    /// First bit tested by the ordinal threshold. Zero means a sum-zero test.
    first_tested_bit: usize,
    /// Lowest selected bit needed by any consumer, or the word width if none.
    first_selected_bit: usize,
    high_zero: OnceCell<AigOperand>,
}

/// Supplies nonzero prefixes directly or without an addition's carry chain.
enum PrefixSource {
    Operand,
    Sum {
        lhs: ExtNaryAddTerm,
        rhs: ExtNaryAddTerm,
    },
}

impl PriorityPredicatePlan {
    /// Finds eligible chains once using consumer counts from prepared IR.
    pub(super) fn extract(f: &ir::Fn, env: &GateEnv) -> Self {
        let mut users = vec![Vec::new(); f.nodes.len()];
        for (index, node) in f.nodes.iter().enumerate() {
            for operand in ir_utils::operands(&node.payload) {
                users[operand.index].push(NodeRef { index });
            }
        }
        let mut result = Self::default();
        let mut candidates = 0usize;
        for (index, node) in f.nodes.iter().enumerate() {
            let NodePayload::Unop(Unop::OrReduce, arg) = node.payload else {
                continue;
            };
            candidates += 1;
            let Some((selected, selection)) = match_predicate(f, env, arg, &users) else {
                continue;
            };
            result
                .reductions
                .insert(NodeRef { index }, (selected, selection.first_tested_bit));
            result.selections.insert(selected, selection);
        }
        log::debug!(
            "priority_predicate summary function={} candidates={} accepted={} rejected={} selections={}",
            f.name,
            candidates,
            result.reductions.len(),
            candidates - result.reductions.len(),
            result.selections.len()
        );
        result
    }

    /// A true threshold requires every tested high bit of its selection to be
    /// zero.
    pub(super) fn contradicts_selected_bit(&self, f: &ir::Fn, args: &[NodeRef]) -> bool {
        if self.reductions.is_empty() {
            return false;
        }
        let tested: HashSet<_> = args
            .iter()
            .filter_map(|arg| self.reductions.get(arg).map(|&(selected, _)| selected))
            .collect();
        !tested.is_empty() && args.iter().any(|arg| {
            matches!(
                f.get_node(*arg).payload,
                NodePayload::BitSlice { arg: selected, start, width: 1 }
                    if tested.contains(&selected) && start >= self.selections[&selected].first_tested_bit
            )
        })
    }

    /// Derives observed selection bits from carry-free operand prefixes.
    pub(super) fn lower_selection(
        &self,
        node: NodeRef,
        ordinary: AigBitVector,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> AigBitVector {
        let Some(plan) = self.selections.get(&node) else {
            return ordinary;
        };
        if plan.first_selected_bit == ordinary.get_bit_count() {
            return ordinary;
        }
        let value = env
            .get_bit_vector(plan.value)
            .expect("planned value is lowered");
        let selector = *env
            .get_bit_vector(plan.selector)
            .expect("planned selector is lowered")
            .get_lsb(0);
        let width = value.get_bit_count();
        let prefix_bits = plan.prefix_bits(width - 1, env, gb);
        // Only the high selected bits need negation carries. Share one tree
        // reduction of the lower region instead of building its unused
        // prefixes.
        let lower_nonzero = if plan.first_selected_bit == 0 {
            gb.get_false()
        } else {
            gb.add_or_reduce(
                &AigBitVector::from_lsb_is_index_0(&prefix_bits[..plan.first_selected_bit]),
                ReductionKind::Tree,
            )
        };
        let mut needed = vec![lower_nonzero];
        needed.extend_from_slice(&prefix_bits[plan.first_selected_bit..]);
        let prefixes = prefix_scan_inclusive(
            gb,
            &needed,
            PrefixScanStrategy::KoggeStone,
            gb.get_false(),
            |gb, a, b| gb.add_or_binary(a, b),
        );
        let mut bits = ordinary.iter_lsb_to_msb().copied().collect::<Vec<_>>();
        for i in plan.first_selected_bit..width {
            let lower_nonzero = prefixes[i - plan.first_selected_bit];
            let flip = gb.add_and_binary(selector, lower_nonzero);
            bits[i] = gb.add_xor_binary(*value.get_lsb(i), flip);
        }
        if matches!(plan.prefix_source, PrefixSource::Operand) {
            let high =
                value.get_lsb_slice(plan.first_selected_bit, width - plan.first_selected_bit);
            let zero = if plan.first_tested_bit == 0 {
                gb.add_or_reduce(&value, ReductionKind::Tree).negate()
            } else if high.get_bit_count() == 1 {
                bits[plan.first_selected_bit].negate()
            } else {
                // With a nonzero low region, negation complements the high
                // word; otherwise it preserves whether that word is zero.
                let all_zero = gb.add_or_reduce(&high, ReductionKind::Tree).negate();
                let all_ones = gb.add_and_reduce(&high, ReductionKind::Tree);
                let complement = gb.add_and_binary(selector, lower_nonzero);
                gb.add_mux2(complement, all_ones, all_zero)
            };
            // Expose the implication to shared high-bit consumers. This lets
            // Boolean folding remove tests already implied by the zero
            // predicate.
            for bit in &mut bits[plan.first_selected_bit..] {
                *bit = gb.add_and_binary(*bit, zero.negate());
            }
            plan.high_zero
                .set(zero)
                .expect("planned selection is lowered once");
            log::debug!(
                "priority_predicate direct_zero selected_index={} first_bit={} high_bits={}",
                node.index,
                plan.first_selected_bit,
                width - plan.first_selected_bit
            );
        }
        AigBitVector::from_lsb_is_index_0(&bits)
    }

    /// Tests selected high bits while preserving the zero-input ordinal.
    pub(super) fn lower_reduction(
        &self,
        node: NodeRef,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> Option<AigOperand> {
        let &(selected, first) = self.reductions.get(&node)?;
        if let Some(zero) = self.selections.get(&selected)?.high_zero.get() {
            return Some(*zero);
        }
        let bits = env.get_bit_vector(selected).ok()?;
        if first == 0 {
            let prefix = self.selections[&selected].prefix_bits(bits.get_bit_count(), env, gb);
            return Some(gb.add_or_nary(&prefix, ReductionKind::Tree).negate());
        }
        let high = bits.get_lsb_slice(first, bits.get_bit_count() - first);
        Some(gb.add_or_reduce(&high, ReductionKind::Tree).negate())
    }
}

impl Selection {
    /// A prefix is nonzero if any of these bits is set. For sums, use the
    /// operand mismatches that witness a nonzero prefix without a carry chain.
    fn prefix_bits(
        &self,
        bit_count: usize,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> Vec<AigOperand> {
        let PrefixSource::Sum { lhs, rhs } = &self.prefix_source else {
            return env
                .get_bit_vector(self.value)
                .expect("planned operand is lowered")
                .iter_lsb_to_msb()
                .take(bit_count)
                .copied()
                .collect();
        };
        let lhs_bits = env
            .get_bit_vector(lhs.operand)
            .expect("planned sum operand is lowered");
        let rhs_bits = env
            .get_bit_vector(rhs.operand)
            .expect("planned sum operand is lowered");
        // A sum prefix is zero iff p[0] = 0 and p[i] = a[i-1] | b[i-1].
        // Detect mismatches directly, without waiting for the addition carry.
        let mut mismatches = Vec::with_capacity(bit_count);
        let zero = gb.get_false();
        let extended_bit = |bits: &AigBitVector, signed: bool, i: usize| {
            if i < bits.get_bit_count() {
                *bits.get_lsb(i)
            } else if signed && bits.get_bit_count() > 0 {
                *bits.get_lsb(bits.get_bit_count() - 1)
            } else {
                zero
            }
        };
        let mut previous_either = zero;
        for i in 0..bit_count {
            let a = extended_bit(&lhs_bits, lhs.signed, i);
            let b = extended_bit(&rhs_bits, rhs.signed, i);
            let propagate = gb.add_xor_binary(a, b);
            mismatches.push(gb.add_xor_binary(propagate, previous_either));
            previous_either = gb.add_or_binary(a, b);
        }
        mismatches
    }
}

/// Matches complete high-ordinal tests without full-vector selection consumers.
fn match_predicate(
    f: &ir::Fn,
    env: &GateEnv,
    arg: NodeRef,
    users: &[Vec<NodeRef>],
) -> Option<(NodeRef, Selection)> {
    let reject = |reason: &str| -> Option<(NodeRef, Selection)> {
        log::debug!(
            "priority_predicate reject function={} input_text_id={} reason={}",
            f.name,
            f.get_node(arg).text_id,
            reason
        );
        None
    };
    let (encoded, start) = match f.get_node(arg).payload {
        NodePayload::BitSlice {
            arg: encoded,
            start,
            width,
        } => {
            if !env.has_single_use(arg) || start + width != f.get_node_ty(encoded).bit_count() {
                return reject("shared_or_partial_ordinal_slice");
            }
            (encoded, start)
        }
        _ => (arg, 0),
    };
    if !env.has_single_use(encoded) {
        return reject("shared_priority_chain");
    }
    let (selected, priority_user) = match f.get_node(encoded).payload {
        NodePayload::ExtClz { arg, offset: 0, .. } => (arg, encoded),
        NodePayload::Encode { arg: hot } => {
            let NodePayload::OneHot {
                arg: reversed,
                lsb_prio: true,
            } = f.get_node(hot).payload
            else {
                return reject("shared_priority_chain");
            };
            let NodePayload::Unop(Unop::Reverse, selected) = f.get_node(reversed).payload else {
                return reject("not_priority_ordinal");
            };
            if !env.has_single_use(hot) || !env.has_single_use(reversed) {
                return reject("not_priority_ordinal");
            }
            (selected, reversed)
        }
        _ => return reject("not_priority_ordinal"),
    };
    let width = f.get_node_ty(selected).bit_count();
    if f.get_node_ty(encoded).bit_count() < xlsynth_pir::math::ceil_log2(width.checked_add(1)?) {
        return reject("truncated_zero_ordinal");
    }
    let NodePayload::Sel {
        selector,
        cases,
        default: None,
    } = &f.get_node(selected).payload
    else {
        return reject("not_selection_without_default");
    };
    if cases.len() != 2
        || f.get_node_ty(*selector).bit_count() != 1
        || !env.has_single_use(cases[1])
    {
        return reject("selection_shape_or_shared_negation");
    }
    let NodePayload::Unop(Unop::Neg, value) = f.get_node(cases[1]).payload else {
        return reject("not_negation");
    };
    if cases[0] != value {
        return reject("different_positive_operand");
    }
    let Some(threshold) = u32::try_from(start)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
    else {
        return reject("threshold_overflow");
    };
    if threshold > width || f.ret_node_ref == Some(selected) {
        return reject("threshold_or_returned_selection");
    }
    let first_tested_bit = width - threshold;
    // Negation preserves zero, so a full-width test needs no selected bits.
    let mut first_selected_bit = if first_tested_bit == 0 {
        width
    } else {
        first_tested_bit
    };
    for &user in &users[selected.index] {
        if user == priority_user {
            continue;
        }
        let NodePayload::BitSlice { start, .. } = f.get_node(user).payload else {
            return reject("other_selection_consumer");
        };
        if start < first_tested_bit {
            return reject("other_selection_consumer");
        }
        first_selected_bit = first_selected_bit.min(start);
    }
    let added = match f.get_node(value).payload {
        NodePayload::BitSlice { arg, start: 0, .. } => arg,
        _ => value,
    };
    let prefix_source = match &f.get_node(added).payload {
        NodePayload::Binop(Binop::Add, lhs, rhs) => PrefixSource::Sum {
            lhs: ExtNaryAddTerm {
                operand: *lhs,
                signed: false,
                negated: false,
            },
            rhs: ExtNaryAddTerm {
                operand: *rhs,
                signed: false,
                negated: false,
            },
        },
        // An explicitly mapped sum stays opaque to the carry-free prefix.
        NodePayload::ExtNaryAdd { terms, arch: None }
            if terms.len() == 2 && terms.iter().all(|term| !term.negated) =>
        {
            PrefixSource::Sum {
                lhs: terms[0],
                rhs: terms[1],
            }
        }
        _ => PrefixSource::Operand,
    };
    match &prefix_source {
        PrefixSource::Sum { lhs, rhs } => {
            if env.product_is_deferred(lhs.operand) || env.product_is_deferred(rhs.operand) {
                return reject("deferred_sum_operand");
            }
        }
        PrefixSource::Operand => {
            if env.product_is_deferred(value) {
                return reject("deferred_operand");
            }
        }
    }
    log::debug!(
        "priority_predicate accept function={} input_text_id={} selected_text_id={} value_text_id={} width={} first_bit={} source={}",
        f.name,
        f.get_node(arg).text_id,
        f.get_node(selected).text_id,
        f.get_node(value).text_id,
        width,
        first_tested_bit,
        match &prefix_source {
            PrefixSource::Operand => "operand",
            PrefixSource::Sum { .. } => "sum",
        }
    );
    Some((
        selected,
        Selection {
            value,
            prefix_source,
            selector: *selector,
            first_tested_bit,
            first_selected_bit,
            high_zero: OnceCell::new(),
        },
    ))
}
