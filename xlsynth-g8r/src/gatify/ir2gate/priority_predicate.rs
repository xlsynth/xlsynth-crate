// SPDX-License-Identifier: Apache-2.0

//! Lower priority-ordinal thresholds using the input bits they observe.

use super::GateEnv;
use super::priority_ordinal::{self, Order};
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::{GateBuilder, ReductionKind};
use crate::prefix_scan_utils::{PrefixScanStrategy, prefix_scan_inclusive};
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_utils;

/// Plans priority thresholds and specializes eligible conditional negations.
#[derive(Default)]
pub(super) struct PriorityPredicatePlan {
    selections: HashMap<NodeRef, Selection>,
    reductions: HashMap<NodeRef, Threshold>,
    pub(super) zero_conjunctions: HashSet<NodeRef>,
}

/// The complete high part of an ordinal is nonzero iff it reaches the cutoff.
struct Threshold {
    input: NodeRef,
    cutoff: usize,
    order: Order,
}

impl Threshold {
    /// Bits that must all be zero when this ordinal predicate is true.
    fn required_zero_bits(&self, width: usize) -> Option<std::ops::Range<usize>> {
        if self.cutoff > width {
            return None;
        }
        match self.order {
            Order::LeadingZeros => Some(width - self.cutoff..width),
            Order::TrailingZeros => Some(0..self.cutoff),
            Order::HighestSetIndex => None,
        }
    }
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
    pub(super) fn extract(f: &ir::Fn, env: &GateEnv, fold: bool) -> Self {
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
            let Some((threshold, leading_user)) = match_threshold(f, env, arg) else {
                continue;
            };
            let selection = leading_user.and_then(|user| {
                match_selection(f, env, arg, threshold.input, user, threshold.cutoff, &users)
            });
            // Specialized conditional selections already fold contradictory bit
            // tests. Enable the broader ordinal folds only when general folding
            // is requested.
            if (fold || selection.is_some())
                && let Some(region) =
                    threshold.required_zero_bits(f.get_node_ty(threshold.input).bit_count())
            {
                for &user in &users[index] {
                    let NodePayload::Nary(NaryOp::And | NaryOp::Nand, terms) =
                        &f.get_node(user).payload
                    else {
                        continue;
                    };
                    if terms.iter().any(|&term| {
                        (fold && term == threshold.input)
                            || matches!(
                                f.get_node(term).payload,
                                NodePayload::BitSlice { arg, start, width: 1 }
                                    if arg == threshold.input && region.contains(&start)
                            )
                    }) {
                        result.zero_conjunctions.insert(user);
                    }
                }
            }
            if let Some(selection) = selection {
                result.selections.insert(threshold.input, selection);
            }
            log::debug!(
                "priority_predicate threshold function={} reduction_text_id={} input_text_id={} cutoff={}",
                f.name,
                node.text_id,
                f.get_node(threshold.input).text_id,
                threshold.cutoff
            );
            result.reductions.insert(NodeRef { index }, threshold);
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
        let threshold = self.reductions.get(&node)?;
        let bits = env.get_bit_vector(threshold.input).ok()?;
        let width = bits.get_bit_count();
        let count = threshold.cutoff;
        if count > width {
            return Some(gb.get_false());
        }
        let first = if matches!(threshold.order, Order::LeadingZeros) {
            let first = width - count;
            if let Some(selection) = self.selections.get(&threshold.input)
                && selection.first_tested_bit == first
            {
                if let Some(zero) = selection.high_zero.get() {
                    return Some(*zero);
                }
                if first == 0 {
                    let prefix = selection.prefix_bits(width, env, gb);
                    return Some(gb.add_or_nary(&prefix, ReductionKind::Tree).negate());
                }
            }
            first
        } else {
            0
        };
        let region = bits.get_lsb_slice(first, count);
        let region_zero = gb.add_or_reduce(&region, ReductionKind::Tree).negate();
        if matches!(threshold.order, Order::HighestSetIndex) && count < width {
            // A high hit qualifies regardless of the low region. With no high
            // hit, only the all-zero word qualifies via its sentinel ordinal N.
            let high = bits.get_lsb_slice(count, width - count);
            let high_hit = gb.add_or_reduce(&high, ReductionKind::Tree);
            return Some(gb.add_or_binary(high_hit, region_zero));
        }
        Some(region_zero)
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

/// Matches an exclusive priority encoder observed only above a bit boundary.
/// The optional leading-count consumer allows specializing its input selection.
fn match_threshold(
    f: &ir::Fn,
    env: &GateEnv,
    arg: NodeRef,
) -> Option<(Threshold, Option<NodeRef>)> {
    let (encoded, start) = match f.get_node(arg).payload {
        NodePayload::BitSlice {
            arg: encoded,
            start,
            width,
        } => {
            if !env.has_single_use(arg) || start + width != f.get_node_ty(encoded).bit_count() {
                return None;
            }
            (encoded, start)
        }
        _ => (arg, 0),
    };
    let ordinal = priority_ordinal::owned(f, env, encoded)?;
    let input = ordinal.input;
    let order = ordinal.order;
    let leading_user = ordinal.leading_user;
    // A truncated sentinel can wrap to zero, invalidating the region test.
    let width = f.get_node_ty(input).bit_count();
    if f.get_node_ty(encoded).bit_count() < xlsynth_pir::math::ceil_log2(width.checked_add(1)?) {
        return None;
    }
    let cutoff = 1usize.checked_shl(u32::try_from(start).ok()?)?;
    Some((
        Threshold {
            input,
            cutoff,
            order,
        },
        leading_user,
    ))
}

/// Matches conditional negations whose other uses observe the same high bits.
fn match_selection(
    f: &ir::Fn,
    env: &GateEnv,
    arg: NodeRef,
    selected: NodeRef,
    leading_user: NodeRef,
    threshold: usize,
    users: &[Vec<NodeRef>],
) -> Option<Selection> {
    let reject = |reason: &str| {
        log::debug!(
            "priority_predicate selection reject function={} input_text_id={} reason={}",
            f.name,
            f.get_node(arg).text_id,
            reason
        );
        None
    };
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
    let width = f.get_node_ty(value).bit_count();
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
        if user == leading_user {
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
    Some(Selection {
        value,
        prefix_source,
        selector: *selector,
        first_tested_bit,
        first_selected_bit,
        high_zero: OnceCell::new(),
    })
}
