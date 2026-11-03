// SPDX-License-Identifier: Apache-2.0

//! Lower priority predicates using only the observed conditional-negation bits.

use super::GateEnv;
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::{GateBuilder, ReductionKind};
use crate::prefix_scan_utils::{PrefixScanStrategy, prefix_scan_inclusive};
use bitvec::vec::BitVec;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_utils;

/// Plans priority thresholds whose selected high bits avoid serial carries.
#[derive(Default)]
pub(super) struct PriorityPredicatePlan {
    selections: HashMap<NodeRef, Selection>,
    reductions: HashMap<NodeRef, (NodeRef, usize)>,
}

/// Captures a conditional negation and the bits its consumers observe.
struct Selection {
    value: NodeRef,
    prefix_source: PrefixSource,
    selector: NodeRef,
    /// First bit tested by the ordinal threshold. Zero means a word-zero test.
    first_tested_bit: usize,
    /// Selected bits needed by slices or a threshold that reduces a sum.
    selected_bits: BitVec,
    /// Low nonzero reduction shared by rewritten slice bits and the threshold.
    lower_nonzero: OnceCell<AigOperand>,
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

    /// Folds a binary consumer of mutually exclusive bits when one is inverted.
    pub(super) fn fold_implied_operand(
        &self,
        f: &ir::Fn,
        op: NaryOp,
        args: &[NodeRef],
        env: &GateEnv,
    ) -> Option<AigOperand> {
        if self.reductions.is_empty() {
            return None;
        }
        let &[a, b] = args else { return None };
        let (positive, inner) = match (&f.get_node(a).payload, &f.get_node(b).payload) {
            (NodePayload::Unop(Unop::Not, inner), _) => (b, *inner),
            (_, NodePayload::Unop(Unop::Not, inner)) => (a, *inner),
            _ => return None,
        };
        if !self.contradicts_selected_bit(f, &[positive, inner]) {
            return None;
        }
        // For disjoint a and b, a & !b = a and a | !b = !b.
        let (kept, invert) = match op {
            NaryOp::And => (positive, false),
            NaryOp::Nand => (positive, true),
            NaryOp::Or => (inner, true),
            NaryOp::Nor => (inner, false),
            _ => return None,
        };
        let bit = *env.get_bit_vector(kept).ok()?.get_lsb(0);
        Some(if invert { bit.negate() } else { bit })
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
        let Some(last_bit) = plan.selected_bits.last_one() else {
            return ordinary;
        };
        let value = env
            .get_bit_vector(plan.value)
            .expect("planned value is lowered");
        let selector = *env
            .get_bit_vector(plan.selector)
            .expect("planned selector is lowered")
            .get_lsb(0);
        let prefix_bits = plan.prefix_bits(last_bit, env, gb);
        let mut prefix_ends = plan.selected_bits.clone();
        let share_low =
            matches!(plan.prefix_source, PrefixSource::Operand) && plan.first_tested_bit != 0;
        if share_low {
            prefix_ends.set(plan.first_tested_bit, true);
        }
        // The full-word zero test reduces a power-of-two number of bits.
        // Match its subtree boundaries when hashing can share them. Keep the
        // n-ary reduction's literal folding for gaps with constant inputs.
        let share_zero_tree = plan.first_tested_bit == 0
            && gb.options.hash
            && !(gb.options.fold
                && prefix_bits
                    .iter()
                    .any(|&bit| gb.is_known_false(bit) || gb.is_known_true(bit)));
        let prefixes = if share_zero_tree {
            zero_test_slice_prefixes(gb, &prefix_bits, &prefix_ends)
        } else {
            // Reduce each gap once. An inclusive scan of these groups gives
            // the exclusive operand prefix at each requested bit.
            let mut groups = Vec::new();
            let mut start = 0;
            for end in prefix_ends.iter_ones() {
                groups.push(match &prefix_bits[start..end] {
                    [] => gb.get_false(),
                    [bit] => *bit,
                    gap => gb.add_or_nary(gap, ReductionKind::Tree),
                });
                start = end;
            }
            prefix_scan_inclusive(
                gb,
                &groups,
                PrefixScanStrategy::KoggeStone,
                gb.get_false(),
                |gb, a, b| gb.add_or_binary(a, b),
            )
        };
        let mut bits = ordinary.iter_lsb_to_msb().copied().collect::<Vec<_>>();
        for (bit, prefix) in prefix_ends.iter_ones().zip(prefixes) {
            if share_low && bit == plan.first_tested_bit {
                plan.lower_nonzero
                    .set(prefix)
                    .expect("selection lowered once");
            }
            if plan.selected_bits[bit] {
                let flip = gb.add_and_binary(selector, prefix);
                bits[bit] = gb.add_xor_binary(*value.get_lsb(bit), flip);
            }
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
        let plan = &self.selections[&selected];
        if matches!(plan.prefix_source, PrefixSource::Sum { .. }) {
            let bits = env.get_bit_vector(selected).ok()?;
            let width = bits.get_bit_count();
            if first == 0 {
                let prefix = plan.prefix_bits(width, env, gb);
                return Some(gb.add_or_nary(&prefix, ReductionKind::Tree).negate());
            }
            let high = bits.get_lsb_slice(first, width - first);
            return Some(gb.add_or_reduce(&high, ReductionKind::Tree).negate());
        }
        let value = env.get_bit_vector(plan.value).ok()?;
        let width = value.get_bit_count();
        if first == 0 {
            return Some(gb.add_or_reduce(&value, ReductionKind::Tree).negate());
        }
        if width - first == 1 && plan.selected_bits[first] {
            // A slice already requested the carry-free sign bit.
            let sign = *env.get_bit_vector(selected).ok()?.get_msb(0);
            return Some(gb.union_current_pir_node_id_into_operand(sign.negate()));
        }
        let selector = *env.get_bit_vector(plan.selector).ok()?.get_lsb(0);
        let lower_nonzero = plan.lower_nonzero.get().copied().unwrap_or_else(|| {
            gb.add_or_reduce(&value.get_lsb_slice(0, first), ReductionKind::Tree)
        });
        let complement = gb.add_and_binary(selector, lower_nonzero);
        let high = value.get_lsb_slice(first, width - first);
        log::debug!(
            "priority_predicate direct_zero selected_index={} first_bit={} high_bits={}",
            selected.index,
            first,
            high.get_bit_count()
        );
        if high.get_bit_count() == 1 {
            return Some(gb.add_xor_binary(*high.get_lsb(0), complement).negate());
        }
        // With a nonzero low region, negation complements the high word.
        // Otherwise negation preserves whether that word is zero.
        let all_zero = gb.add_or_reduce(&high, ReductionKind::Tree).negate();
        let all_ones = gb.add_and_reduce(&high, ReductionKind::Tree);
        Some(gb.add_mux2(complement, all_ones, all_zero))
    }
}

/// Builds requested prefixes from aligned subtrees of the balanced zero test.
fn zero_test_slice_prefixes(
    gb: &mut GateBuilder,
    bits: &[AigOperand],
    ends: &BitVec,
) -> Vec<AigOperand> {
    let mut blocks = Vec::new();
    let mut prefixes = Vec::new();
    let mut start = 0;
    for end in ends.iter_ones() {
        for i in start..end {
            // The set bits of i give the sizes of the completed blocks,
            // largest first. Each trailing one closes a pair of equal blocks.
            let mut block = bits[i];
            for _ in 0..i.trailing_ones() {
                block = gb.add_or_binary(blocks.pop().expect("completed left block"), block);
            }
            blocks.push(block);
        }
        // Combine smaller blocks first to keep each prefix balanced.
        let prefix = blocks
            .iter()
            .rev()
            .copied()
            .reduce(|a, b| gb.add_or_binary(b, a));
        prefixes.push(prefix.unwrap_or_else(|| gb.get_false()));
        start = end;
    }
    prefixes
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
    let mut selected_bits = BitVec::repeat(false, width);
    for &user in &users[selected.index] {
        if user == priority_user {
            continue;
        }
        let NodePayload::BitSlice { start, width, .. } = f.get_node(user).payload else {
            return reject("other_selection_consumer");
        };
        if width == 0 {
            continue;
        }
        if start < first_tested_bit {
            return reject("other_selection_consumer");
        }
        selected_bits[start..start + width].fill(true);
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
            // Non-full sum thresholds reduce the selected high bits directly.
            if first_tested_bit != 0 {
                selected_bits[first_tested_bit..].fill(true);
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
            selected_bits,
            lower_nonzero: OnceCell::new(),
        },
    ))
}
