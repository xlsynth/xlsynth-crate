// SPDX-License-Identifier: Apache-2.0

//! Lower bit-range observations of shifts and selected words to scalar gates.

mod window;
mod window_gate;

use super::GateEnv;
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::{GateBuilder, ReductionKind};
use std::collections::HashMap;
use std::time::Instant;
use xlsynth_pir::ir::{self, Binop, NodePayload, NodeRef, Unop};
use xlsynth_pir::math::ceil_log2;

/// Whole-function plans for scalar observations with replaceable interior
/// words.
#[derive(Default)]
pub(super) struct ShiftReductionPredicatePlan {
    predicates: HashMap<NodeRef, Predicate>,
    windows: HashMap<NodeRef, window::Plan>,
}

/// A zero/nonzero observation and the word cone it can replace.
struct Predicate {
    data: NodeRef,
    amount: NodeRef,
    left: bool,
    inverted: bool,
    selection: Option<Selection>,
}

/// A two-case selection with one shifted word and one literal word.
struct Selection {
    selector: NodeRef,
    shift_case: usize,
    other_nonzero: bool,
}

impl ShiftReductionPredicatePlan {
    /// Plans only predicates whose intermediate words have no other consumers.
    pub(super) fn extract(f: &ir::Fn, env: &GateEnv) -> Self {
        let mut result = Self::default();
        for index in 0..f.nodes.len() {
            let node = NodeRef { index };
            if let Some((word, inverted)) = observed(f, node)
                && let Some(predicate) = match_shift(f, env, word, inverted)
            {
                result.predicates.insert(node, predicate);
            }
        }
        log::debug!(
            "shift_reduction_predicate summary function={} planned={}",
            f.name,
            result.predicates.len()
        );
        let observed = window::analyze(f, true);
        log::debug!(
            "shift_window scan function={} visits={} valuations={} reasons={:?}",
            f.name,
            observed.visits,
            observed.valuations,
            observed.reasons
        );
        result.windows = observed.plans.into_iter().map(|p| (p.root, p)).collect();
        log::debug!(
            "shift_window summary function={} planned={}",
            f.name,
            result.windows.len()
        );
        result
    }

    /// Keep source bits and both controls on the region boundary; either
    /// control can be computed after some of the ordinary shift gates.
    pub(super) fn lower_window(
        &self,
        node: NodeRef,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> Option<AigOperand> {
        let plan = self.windows.get(&node)?;
        let ordinary = *env.get_bit_vector(node).ok()?.get_lsb(0);
        window_gate::lower(plan, ordinary, env, gb)
    }

    /// Tries a direct shifted-word predicate. A rejected region trial restores
    /// structural hashing and provenance. An accepted trial leaves the unused
    /// gates for dead-code removal.
    pub(super) fn lower(
        &self,
        node: NodeRef,
        ordinary: AigOperand,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> AigOperand {
        let Some(plan) = self.predicates.get(&node) else {
            return ordinary;
        };
        let started = log::log_enabled!(log::Level::Debug).then(Instant::now);
        let data = env
            .get_bit_vector(plan.data)
            .expect("planned data is lowered");
        let amount = env
            .get_bit_vector(plan.amount)
            .expect("planned amount is lowered");
        let selector = plan.selection.as_ref().map(|selection| {
            *env.get_bit_vector(selection.selector)
                .expect("planned selector is lowered")
                .get_lsb(0)
        });
        // All source/amount fanins precede this frontier in the append-only
        // builder. Exclude that shared prefix from both node counts, and stop
        // at the selector even when it was lowered after the shift.
        let prefix_end = data
            .iter_lsb_to_msb()
            .chain(amount.iter_lsb_to_msb())
            .map(|bit| bit.node.id)
            .max()
            .unwrap_or(0)
            + 1;
        let (result, trial) =
            gb.try_region_alternative(ordinary.into(), prefix_end, selector.as_slice(), |gb| {
                let mut bits: Vec<_> = data.iter_lsb_to_msb().copied().collect();
                if plan.left {
                    bits.reverse();
                }
                let mut predicate = shifted_nonzero(&bits, &amount, gb);
                if let Some(selection) = &plan.selection {
                    let selector = selector.expect("selection has a selector");
                    let shift_selected = if selection.shift_case == 1 {
                        selector
                    } else {
                        selector.negate()
                    };
                    predicate = if selection.other_nonzero {
                        gb.add_or_binary(shift_selected.negate(), predicate)
                    } else {
                        gb.add_and_binary(shift_selected, predicate)
                    };
                }
                if plan.inverted {
                    predicate = predicate.negate();
                }
                predicate.into()
            });
        log::debug!(
            "shift_reduction_predicate trial node={node:?} width={} control_width={} left={} cost={trial:?} elapsed_us={}",
            data.get_bit_count(),
            amount.get_bit_count(),
            plan.left,
            started
                .map(|start| start.elapsed().as_micros())
                .unwrap_or(0)
        );
        *result.get_lsb(0)
    }
}

/// Matches a direct observation or a single-use selection with a literal arm.
fn match_shift(f: &ir::Fn, env: &GateEnv, value: NodeRef, inverted: bool) -> Option<Predicate> {
    let (shift, selection) = match &f.get_node(value).payload {
        NodePayload::Sel {
            selector,
            cases,
            default: None,
        } if cases.len() == 2 && f.get_node_ty(*selector) == &ir::Type::Bits(1) => {
            if !env.has_single_use(value) {
                log::debug!(
                    "shift_reduction_predicate skip node={value:?} reason=shared_selection"
                );
                return None;
            }
            let (shift_case, other_nonzero) = if let Some(nonzero) = constant_nonzero(f, cases[0]) {
                (1, nonzero)
            } else {
                (0, constant_nonzero(f, cases[1])?)
            };
            (
                cases[shift_case],
                Some(Selection {
                    selector: *selector,
                    shift_case,
                    other_nonzero,
                }),
            )
        }
        _ => (value, None),
    };
    let (left, data, amount) = match f.get_node(shift).payload {
        NodePayload::Binop(Binop::Shrl, data, amount) => (false, data, amount),
        NodePayload::Binop(Binop::Shll, data, amount) => (true, data, amount),
        _ => return None,
    };
    if !env.has_single_use(shift) {
        log::debug!("shift_reduction_predicate skip node={shift:?} reason=shared_shift");
        return None;
    }
    if env.product_is_deferred(data) || env.product_is_deferred(amount) {
        log::debug!("shift_reduction_predicate skip node={shift:?} reason=deferred_operand");
        return None;
    }
    Some(Predicate {
        data,
        amount,
        left,
        inverted,
        selection,
    })
}

/// Classifies a literal word without converting its width to a host integer.
/// Return the observed word and whether its nonzero result is inverted.
fn observed(f: &ir::Fn, node: NodeRef) -> Option<(NodeRef, bool)> {
    match f.get_node(node).payload {
        NodePayload::Unop(Unop::OrReduce, word) => Some((word, false)),
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) => {
            let word = if constant_nonzero(f, lhs) == Some(false) {
                rhs
            } else if constant_nonzero(f, rhs) == Some(false) {
                lhs
            } else {
                return None;
            };
            Some((word, op == Binop::Eq))
        }
        _ => None,
    }
}

fn constant_nonzero(f: &ir::Fn, value: NodeRef) -> Option<bool> {
    if !matches!(f.get_node_ty(value), ir::Type::Bits(_)) {
        return None;
    }
    let NodePayload::Literal(value) = &f.get_node(value).payload else {
        return None;
    };
    Some(!value.bits_equals_u64_value(0))
}

/// A left shift's reduction is a right shift's reduction after reversing the
/// data. Padding the high end with zeros makes the same tree handle
/// non-power-of-two widths.
fn shifted_nonzero(bits: &[AigOperand], amount: &AigBitVector, gb: &mut GateBuilder) -> AigOperand {
    if bits.is_empty() {
        return gb.get_false();
    }
    let levels = ceil_log2(bits.len());
    let (_, mut predicate) = suffix_tree(bits, levels, amount, gb);
    if amount.get_bit_count() > levels {
        let high = amount.get_lsb_slice(levels, amount.get_bit_count() - levels);
        let oversized = gb.add_or_reduce(&high, ReductionKind::Tree);
        predicate = gb.add_and_binary(predicate, oversized.negate());
    }
    predicate
}

/// Returns the unconditional reduction U and shifted reduction P for this
/// block. For equal high/low halves H/L and amount b++t:
/// P(H++L, b++t) = P(H,t) | (!b & (U(H) | P(L,t))).
/// Each unconditional reduction is shared by its ancestors.
fn suffix_tree(
    bits: &[AigOperand],
    levels: usize,
    amount: &AigBitVector,
    gb: &mut GateBuilder,
) -> (AigOperand, AigOperand) {
    if bits.is_empty() {
        return (gb.get_false(), gb.get_false());
    }
    if levels == 0 {
        return (bits[0], bits[0]);
    }
    let split = (1usize << (levels - 1)).min(bits.len());
    let (low_total, low_predicate) = suffix_tree(&bits[..split], levels - 1, amount, gb);
    let (high_total, high_predicate) = suffix_tree(&bits[split..], levels - 1, amount, gb);
    let total = gb.add_or_binary(high_total, low_total);
    let control = if levels <= amount.get_bit_count() {
        *amount.get_lsb(levels - 1)
    } else {
        gb.get_false()
    };
    let predicate = if control == gb.get_true() {
        high_predicate
    } else if control == gb.get_false() {
        gb.add_or_binary(high_total, low_predicate)
    } else if levels == 1 {
        // At the leaves, U(H) equals P(H), so absorption removes one gate.
        let lower_half = gb.add_and_binary(control.negate(), low_predicate);
        gb.add_or_binary(high_predicate, lower_half)
    } else {
        let either = gb.add_or_binary(high_total, low_predicate);
        let lower_half = gb.add_and_binary(control.negate(), either);
        gb.add_or_binary(high_predicate, lower_half)
    };
    (total, predicate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::GateBuilderOptions;
    use crate::gatify::ir2gate::GateOrVec;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn equal_cost_trial_restores_gates_and_provenance() {
        let package = Parser::new(
            r#"package sample
 top fn main(x: bits[1], amount: bits[1]) -> bits[1] {
  shifted: bits[1] = shrl(x, amount, id=3)
  ret predicate: bits[1] = or_reduce(shifted, id=4)
 }
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let f = package.get_top_fn().unwrap();
        for hash in [true, false] {
            let mut env = GateEnv::new(f);
            let plan = ShiftReductionPredicatePlan::extract(f, &env);
            let mut gb =
                GateBuilder::new("trial".to_string(), GateBuilderOptions { hash, fold: true });
            let x = gb.add_input("x".to_string(), 1);
            let amount = gb.add_input("amount".to_string(), 1);
            let x_bit = *x.get_lsb(0);
            let amount_bit = *amount.get_lsb(0);
            let ordinary = gb.add_and_binary(x_bit, amount_bit.negate());
            env.add(f.params[0], GateOrVec::BitVector(x));
            env.add(f.params[1], GateOrVec::BitVector(amount));
            gb.add_tag(ordinary.node, "ordinary".to_string());
            gb.set_current_pir_node_id(Some(99));
            let before = gb.gates.clone();
            let result = plan.lower(f.ret_node_ref.unwrap(), ordinary, &env, &mut gb);
            assert_eq!(result, ordinary);
            assert_eq!(gb.gates, before);
            if hash {
                gb.set_current_pir_node_id(None);
                assert_eq!(gb.add_and_binary(x_bit, amount_bit.negate()), ordinary);
                assert_eq!(gb.aig_depth(ordinary), Some(1));
            }
        }
    }
}
