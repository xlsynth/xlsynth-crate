// SPDX-License-Identifier: Apache-2.0

//! Lower priority-ordinal thresholds using the input bits they observe.

use super::priority_ordinal::{self, Order};
use super::{GateEnv, GatifyOptions};
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::{GateBuilder, ReductionKind};
use crate::prefix_scan_utils::{PrefixScanStrategy, prefix_scan_inclusive};
use bitvec::vec::BitVec;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils;

/// Plans priority thresholds and specializes eligible conditional negations.
#[derive(Default)]
pub(super) struct PriorityPredicatePlan {
    selections: HashMap<NodeRef, Selection>,
    reductions: HashMap<NodeRef, Threshold>,
    pub(super) zero_conjunctions: HashSet<NodeRef>,
    #[cfg(test)]
    conjunction_terms_indexed: usize,
    #[cfg(test)]
    selection_users_examined: usize,
}

/// The complete high part of an ordinal is nonzero iff it reaches the cutoff.
struct Threshold {
    input: NodeRef,
    cutoff: usize,
    order: Order,
    /// Test whether any bit outside the region is set.
    use_outside_hit: bool,
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

/// Captures a conditional negation and the bits its consumers observe.
struct Selection {
    value: NodeRef,
    prefix_source: PrefixSource,
    selector: NodeRef,
    /// First bit tested by the ordinal threshold. Zero means a word-zero test.
    first_tested_bit: usize,
    /// Selected bits needed by slices or a threshold that reduces a sum.
    selected_bits: BitVec,
    /// Each mapping lowers the selection before its threshold, refreshing this
    /// shared prefix with the current builder's operand.
    lower_nonzero: Cell<Option<AigOperand>>,
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
    pub(super) fn extract(f: &ir::Fn, env: &GateEnv, options: &GatifyOptions) -> Self {
        let fold = options.fold;
        let mut users = vec![Vec::new(); f.nodes.len()];
        for (index, node) in f.nodes.iter().enumerate() {
            for operand in ir_utils::operands(&node.payload) {
                users[operand.index].push(NodeRef { index });
            }
        }
        let mut result = Self::default();
        let mut candidates = 0usize;
        let mut observed_by_conjunction = HashMap::new();
        let mut checked_selections = HashSet::new();
        for (index, node) in f.nodes.iter().enumerate() {
            let NodePayload::Unop(Unop::OrReduce, arg) = node.payload else {
                continue;
            };
            candidates += 1;
            let Some((threshold, leading_user)) =
                match_threshold(f, env, arg, options.range_info.as_deref())
            else {
                continue;
            };
            let selection = leading_user.and_then(|user| {
                // Each matched chain has a distinct, single-use non-slice
                // consumer of its selection. Two such chains disqualify each
                // other, so only the first needs to inspect the selection.
                if !checked_selections.insert(threshold.input) {
                    return None;
                }
                let users = users[threshold.input.index].iter().copied();
                #[cfg(test)]
                let users = users.inspect(|_| result.selection_users_examined += 1);
                match_selection(f, env, arg, threshold.input, user, threshold.cutoff, users)
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
                    let observed = observed_by_conjunction.entry(user).or_insert_with(|| {
                        let terms = terms.iter().copied();
                        #[cfg(test)]
                        let terms = terms.inspect(|_| result.conjunction_terms_indexed += 1);
                        observed_bit_bounds(f, terms, fold)
                    });
                    // A required-zero region touches one end of the word, so
                    // any overlap includes the lowest or highest observed bit.
                    if let Some((lowest, highest)) = observed.get(&threshold.input)
                        && (region.contains(lowest) || region.contains(highest))
                    {
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

    /// Returns the conditional negation specialized for this leading threshold.
    fn leading_selection(&self, threshold: &Threshold, width: usize) -> Option<&Selection> {
        let selection = self.selections.get(&threshold.input)?;
        (matches!(threshold.order, Order::LeadingZeros)
            && width.checked_sub(threshold.cutoff) == Some(selection.first_tested_bit))
        .then_some(selection)
    }

    /// Folds a binary consumer of mutually exclusive bits when one is inverted.
    pub(super) fn fold_implied_operand(
        &self,
        f: &ir::Fn,
        op: NaryOp,
        args: &[NodeRef],
        env: &GateEnv,
    ) -> Option<AigOperand> {
        if self.selections.is_empty() {
            return None;
        }
        let &[a, b] = args else { return None };
        let (positive, inner) = match (&f.get_node(a).payload, &f.get_node(b).payload) {
            (NodePayload::Unop(Unop::Not, inner), _) => (b, *inner),
            (_, NodePayload::Unop(Unop::Not, inner)) => (a, *inner),
            _ => return None,
        };
        let (threshold, observed) = if let Some(threshold) = self.reductions.get(&positive) {
            (threshold, inner)
        } else {
            (self.reductions.get(&inner)?, positive)
        };
        let selection =
            self.leading_selection(threshold, f.get_node_ty(threshold.input).bit_count())?;
        if !matches!(
            f.get_node(observed).payload,
            NodePayload::BitSlice { arg, start, width: 1 }
                if arg == threshold.input && start >= selection.first_tested_bit
        ) {
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
                plan.lower_nonzero.set(Some(prefix));
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
        let threshold = self.reductions.get(&node)?;
        let bits = env.get_bit_vector(threshold.input).ok()?;
        let width = bits.get_bit_count();
        let count = threshold.cutoff;
        if count > width || (threshold.use_outside_hit && count == width) {
            return Some(gb.union_current_pir_node_id_into_operand(gb.get_false()));
        }
        let outside_hit = |gb: &mut GateBuilder| {
            let start = if matches!(threshold.order, Order::LeadingZeros) {
                0
            } else {
                count
            };
            gb.add_or_reduce(
                &bits.get_lsb_slice(start, width - count),
                ReductionKind::Tree,
            )
        };
        if threshold.use_outside_hit {
            return Some(outside_hit(gb));
        }
        let first = if matches!(threshold.order, Order::LeadingZeros) {
            if let Some(selection) = self.leading_selection(threshold, width) {
                if matches!(selection.prefix_source, PrefixSource::Operand) {
                    return selection.lower_operand_high_zero(threshold.input, &bits, env, gb);
                }
                if selection.first_tested_bit == 0 {
                    let prefix = selection.prefix_bits(width, env, gb);
                    return Some(gb.add_or_nary(&prefix, ReductionKind::Tree).negate());
                }
            }
            width - count
        } else {
            0
        };
        let region = bits.get_lsb_slice(first, count);
        let region_zero = gb.add_or_reduce(&region, ReductionKind::Tree).negate();
        if matches!(threshold.order, Order::HighestSetIndex) && count < width {
            // Without a high hit, only the all-zero word qualifies via its
            // sentinel.
            let hit = outside_hit(gb);
            return Some(gb.add_or_binary(hit, region_zero));
        }
        Some(region_zero)
    }
}

/// Indexes the lowest and highest bit observed from each word in a conjunction.
fn observed_bit_bounds(
    f: &ir::Fn,
    terms: impl Iterator<Item = NodeRef>,
    fold: bool,
) -> HashMap<NodeRef, (usize, usize)> {
    let mut bounds = HashMap::new();
    let mut observe = |word, bit: usize| {
        let (lowest, highest) = bounds.entry(word).or_insert((bit, bit));
        *lowest = bit.min(*lowest);
        *highest = bit.max(*highest);
    };
    for term in terms {
        if fold {
            // This conjunction consumes a scalar threshold, so every bare
            // operand is itself a one-bit word.
            observe(term, 0);
        }
        if let NodePayload::BitSlice {
            arg,
            start,
            width: 1,
        } = f.get_node(term).payload
        {
            observe(arg, start);
        }
    }
    bounds
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
    /// Tests the selected high bits using the operand's low prefix.
    fn lower_operand_high_zero(
        &self,
        selected: NodeRef,
        selected_bits: &AigBitVector,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> Option<AigOperand> {
        let first = self.first_tested_bit;
        let width = selected_bits.get_bit_count();
        let value = env.get_bit_vector(self.value).ok()?;
        if first == 0 {
            return Some(gb.add_or_reduce(&value, ReductionKind::Tree).negate());
        }
        if width - first == 1 && self.selected_bits[first] {
            // A slice already requested the carry-free sign bit.
            let zero = selected_bits.get_msb(0).negate();
            return Some(gb.union_current_pir_node_id_into_operand(zero));
        }
        let selector = *env.get_bit_vector(self.selector).ok()?.get_lsb(0);
        let lower_nonzero = self.lower_nonzero.get().unwrap_or_else(|| {
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
    range_info: Option<&IrRangeInfo>,
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
    let priority_ordinal::Ordinal {
        input,
        order,
        leading_user,
    } = priority_ordinal::owned(f, env, encoded)?;
    // A truncated sentinel can wrap to zero, invalidating the region test.
    let width = f.get_node_ty(input).bit_count();
    if f.get_node_ty(encoded).bit_count() < xlsynth_pir::math::ceil_log2(width.checked_add(1)?) {
        return None;
    }
    let cutoff = 1usize.checked_shl(u32::try_from(start).ok()?)?;
    let use_outside_hit = range_info.is_some_and(|ri| {
        // The recognizer can look through a reverse. Population is unchanged,
        // but range facts may be recorded only on the encoder's actual operand.
        let encoder = match f.get_node(encoded).payload {
            NodePayload::Encode { arg } => arg,
            _ => encoded,
        };
        let fact_input = match f.get_node(encoder).payload {
            NodePayload::OneHot { arg, .. } | NodePayload::ExtPrioEncode { arg, .. } => arg,
            _ => input,
        };
        let text_id = f.get_node(fact_input).text_id;
        // A full-width count tests only the zero sentinel. When just one
        // bit lies outside a count's zero region on an exactly-one word,
        // that bit is the predicate.
        let single_outside_bit = width.checked_sub(cutoff) == Some(1)
            && ri
                .get(text_id)
                .and_then(|info| info.known_bits.as_ref())
                .is_some_and(|k| k.bit_count() == width && k.max_ones() <= 1);
        (matches!(order, Order::HighestSetIndex) || cutoff == width || single_outside_bit)
            && ri.proves_nonzero(text_id)
    });
    Some((
        Threshold {
            input,
            cutoff,
            order,
            use_outside_hit,
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
    users: impl Iterator<Item = NodeRef>,
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
    let mut selected_bits = BitVec::repeat(false, width);
    for user in users {
        if user == leading_user {
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
    Some(Selection {
        value,
        prefix_source,
        selector: *selector,
        first_tested_bit,
        selected_bits,
        lower_nonzero: Cell::new(None),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn reversed_encoders_examine_their_shared_selection_once() {
        const N: usize = 4096;
        let mut ir = format!(
            r#"package sample
top fn main(c: bits[1] id=1, v: bits[2] id=2) -> bits[{}] {{
  negated: bits[2] = neg(v, id=3)
  s: bits[2] = sel(c, cases=[v, negated], id=4)
"#,
            2 * N
        );
        let mut outputs = Vec::new();
        for i in 0..N {
            writeln!(
                ir,
                "  b{i}: bits[1] = bit_slice(s, start=1, width=1, id={})",
                5 + i
            )
            .unwrap();
            outputs.push(format!("b{i}"));
        }
        for i in 0..N {
            let id = 5 + N + 3 * i;
            writeln!(ir, "  r{i}: bits[2] = reverse(s, id={id})").unwrap();
            writeln!(
                ir,
                "  o{i}: bits[2] = ext_prio_encode(r{i}, lsb_prio=true, id={})",
                id + 1
            )
            .unwrap();
            writeln!(ir, "  p{i}: bits[1] = or_reduce(o{i}, id={})", id + 2).unwrap();
            outputs.push(format!("p{i}"));
        }
        writeln!(
            ir,
            "  ret out: bits[{}] = concat({}, id={})\n}}",
            2 * N,
            outputs.join(", "),
            5 + 4 * N
        )
        .unwrap();
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        let plan = PriorityPredicatePlan::extract(
            f,
            &GateEnv::new(f),
            &GatifyOptions::all_opts_disabled(),
        );
        assert_eq!(plan.reductions.len(), N);
        assert!(plan.selections.is_empty());
        assert!(plan.selection_users_examined <= 2 * N);
    }

    #[test]
    fn independent_thresholds_index_their_shared_nand_once() {
        const N: usize = 4096;
        let mut ir = format!(
            "package sample\ntop fn main(x: bits[{}] id=1) -> bits[1] {{\n",
            2 * N
        );
        let mut predicates = Vec::with_capacity(N);
        for i in 0..N {
            writeln!(
                ir,
                "  x{i}: bits[2] = bit_slice(x, start={}, width=2, id={})",
                2 * i,
                3 * i + 2
            )
            .unwrap();
            writeln!(
                ir,
                "  o{i}: bits[2] = ext_prio_encode(x{i}, lsb_prio=true, id={})",
                3 * i + 3
            )
            .unwrap();
            writeln!(ir, "  p{i}: bits[1] = or_reduce(o{i}, id={})", 3 * i + 4).unwrap();
            predicates.push(format!("p{i}"));
        }
        writeln!(
            ir,
            "  ret out: bits[1] = nand({}, id={})\n}}",
            predicates.join(", "),
            3 * N + 2
        )
        .unwrap();
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        let plan = PriorityPredicatePlan::extract(
            f,
            &GateEnv::new(f),
            &GatifyOptions::all_opts_disabled(),
        );
        assert_eq!(plan.reductions.len(), N);
        assert!(plan.zero_conjunctions.is_empty());
        assert_eq!(plan.conjunction_terms_indexed, N);
    }
}
