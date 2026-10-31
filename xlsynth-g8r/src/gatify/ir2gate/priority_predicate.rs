// SPDX-License-Identifier: Apache-2.0

//! Lower priority predicates on conditionally negated sums without serial
//! carries.

use super::GateEnv;
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::{GateBuilder, ReductionKind};
use crate::prefix_scan_utils::{PrefixScanStrategy, prefix_scan_inclusive};
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_utils;

/// Plans priority thresholds whose selected high bits avoid serial carries.
#[derive(Default)]
pub(super) struct PriorityPredicatePlan {
    selections: HashMap<NodeRef, Selection>,
    reductions: HashMap<NodeRef, NodeRef>,
}

/// Captures a conditional sum and the first high bit its consumers observe.
struct Selection {
    sum: NodeRef,
    lhs: ExtNaryAddTerm,
    rhs: ExtNaryAddTerm,
    selector: NodeRef,
    /// First bit tested by the ordinal threshold. Zero means a sum-zero test.
    first_tested_bit: usize,
    /// Lowest selected bit needed by any consumer, or the word width if none.
    first_selected_bit: usize,
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
        for (index, node) in f.nodes.iter().enumerate() {
            let NodePayload::Unop(Unop::OrReduce, arg) = node.payload else {
                continue;
            };
            let Some((selected, selection)) = match_predicate(f, env, arg, &users) else {
                continue;
            };
            result.reductions.insert(NodeRef { index }, selected);
            result.selections.insert(selected, selection);
        }
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
            .filter_map(|arg| self.reductions.get(arg).copied())
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
        let sum = env
            .get_bit_vector(plan.sum)
            .expect("planned sum is lowered");
        let selector = *env
            .get_bit_vector(plan.selector)
            .expect("planned selector is lowered")
            .get_lsb(0);
        let width = sum.get_bit_count();
        let mismatches = plan.sum_prefix_mismatches(width - 1, env, gb);
        // Only the high selected bits need negation carries. Share one tree
        // reduction of the lower region instead of building its unused
        // prefixes.
        let lower_nonzero = if plan.first_selected_bit == 0 {
            gb.get_false()
        } else {
            gb.add_or_nary(&mismatches[..plan.first_selected_bit], ReductionKind::Tree)
        };
        let mut needed = vec![lower_nonzero];
        needed.extend_from_slice(&mismatches[plan.first_selected_bit..]);
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
            bits[i] = gb.add_xor_binary(*sum.get_lsb(i), flip);
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
        let selected = self.reductions.get(&node)?;
        let plan = &self.selections[selected];
        let first = plan.first_tested_bit;
        let bits = env.get_bit_vector(*selected).ok()?;
        if first == 0 {
            let mismatches = plan.sum_prefix_mismatches(bits.get_bit_count(), env, gb);
            return Some(gb.add_or_nary(&mismatches, ReductionKind::Tree).negate());
        }
        let high = bits.get_lsb_slice(first, bits.get_bit_count() - first);
        Some(gb.add_or_reduce(&high, ReductionKind::Tree).negate())
    }
}

impl Selection {
    /// A prefix of the sum is nonzero if any of its operand bits mismatch the
    /// zero-sum relation.
    fn sum_prefix_mismatches(
        &self,
        bit_count: usize,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> Vec<AigOperand> {
        let lhs = env
            .get_bit_vector(self.lhs.operand)
            .expect("planned sum operand is lowered");
        let rhs = env
            .get_bit_vector(self.rhs.operand)
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
            let a = extended_bit(&lhs, self.lhs.signed, i);
            let b = extended_bit(&rhs, self.rhs.signed, i);
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
    if !env.has_single_use(encoded) {
        return None;
    }
    let (selected, priority_user) = match f.get_node(encoded).payload {
        NodePayload::ExtClz { arg, offset: 0, .. } => (arg, encoded),
        NodePayload::Encode { arg: hot } => {
            let NodePayload::OneHot {
                arg: reversed,
                lsb_prio: true,
            } = f.get_node(hot).payload
            else {
                return None;
            };
            let NodePayload::Unop(Unop::Reverse, selected) = f.get_node(reversed).payload else {
                return None;
            };
            if !env.has_single_use(hot) || !env.has_single_use(reversed) {
                return None;
            }
            (selected, reversed)
        }
        _ => return None,
    };
    let width = f.get_node_ty(selected).bit_count();
    if f.get_node_ty(encoded).bit_count() < xlsynth_pir::math::ceil_log2(width.checked_add(1)?) {
        return None;
    }
    let NodePayload::Sel {
        selector,
        cases,
        default: None,
    } = &f.get_node(selected).payload
    else {
        return None;
    };
    if cases.len() != 2
        || f.get_node_ty(*selector).bit_count() != 1
        || !env.has_single_use(cases[1])
    {
        return None;
    }
    let NodePayload::Unop(Unop::Neg, sum) = f.get_node(cases[1]).payload else {
        return None;
    };
    if cases[0] != sum {
        return None;
    }
    let threshold = 1usize.checked_shl(u32::try_from(start).ok()?)?;
    if threshold > width || f.ret_node_ref == Some(selected) {
        return None;
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
            return None;
        };
        if start < first_tested_bit {
            return None;
        }
        first_selected_bit = first_selected_bit.min(start);
    }
    let added = match f.get_node(sum).payload {
        NodePayload::BitSlice { arg, start: 0, .. } => arg,
        _ => sum,
    };
    let (lhs, rhs) = match &f.get_node(added).payload {
        NodePayload::Binop(Binop::Add, lhs, rhs) => (
            ExtNaryAddTerm {
                operand: *lhs,
                signed: false,
                negated: false,
            },
            ExtNaryAddTerm {
                operand: *rhs,
                signed: false,
                negated: false,
            },
        ),
        // An explicitly mapped sum stays opaque to the carry-free prefix.
        NodePayload::ExtNaryAdd { terms, arch: None }
            if terms.len() == 2 && terms.iter().all(|term| !term.negated) =>
        {
            (terms[0], terms[1])
        }
        _ => return None,
    };
    if env.product_is_deferred(lhs.operand) || env.product_is_deferred(rhs.operand) {
        return None;
    }
    Some((
        selected,
        Selection {
            sum,
            lhs,
            rhs,
            selector: *selector,
            first_tested_bit,
            first_selected_bit,
        },
    ))
}
