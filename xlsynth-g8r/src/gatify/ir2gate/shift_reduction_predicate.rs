// SPDX-License-Identifier: Apache-2.0

//! Lower bit-range observations of shifts and selected words to scalar gates.

mod window;
mod window_gate;

use super::GateEnv;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef};
use crate::gate_builder::{GateBuilder, ReductionKind};
use std::cell::LazyCell;
use std::collections::{HashMap, HashSet};
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

/// The lowered inputs of a shifted-word predicate in one mapping.
struct PredicateInputs {
    data: AigBitVector,
    amount: AigBitVector,
    selector: Option<AigOperand>,
}

impl ShiftReductionPredicatePlan {
    /// Plans only predicates whose intermediate words have no other consumers.
    pub(super) fn extract(f: &ir::Fn, env: &GateEnv) -> Self {
        let mut result = Self::default();
        for index in 0..f.nodes.len() {
            let node = NodeRef { index };
            if let Some((word, inverted)) = observed(&f.get_node(node).payload, |a| {
                constant_nonzero(f, a) == Some(false)
            }) && let Some(predicate) = match_shift(f, env, word, inverted)
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

    /// Watch a root created by this predicate. If another lowering rebuilds
    /// its AND, structural hashing invalidates the claim to all its fanouts.
    pub(super) fn observe_ordinary(
        &self,
        node: NodeRef,
        first_gate: usize,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) {
        if self.predicates.contains_key(&node) || self.windows.contains_key(&node) {
            let bits = env.get_bit_vector(node).expect("predicate is lowered");
            let root = bits.get_lsb(0).node;
            if root.id >= first_gate {
                gb.sole_and_producers.insert(root, node.index);
            }
        }
    }

    /// Screens predicates and windows against the complete ordinary mapping.
    /// Returns its live area if any replacement qualifies for a second mapping.
    pub(super) fn retain_profitable(
        &mut self,
        f: &ir::Fn,
        env: &GateEnv,
        gb: &mut GateBuilder,
    ) -> Option<usize> {
        if self.predicates.is_empty() && self.windows.is_empty() {
            return None;
        }
        let mut live = LiveUses::from_outputs(gb);
        let sole_producers = std::mem::take(&mut gb.sole_and_producers);
        // A returned predicate redirects the output edge even if a dead IR
        // node also built its AND. Identity and NOT forward the same AIG node.
        let returned = f.ret_node_ref.map(|mut root| {
            while let NodePayload::Unop(Unop::Identity | Unop::Not, arg) = f.get_node(root).payload
            {
                root = arg;
            }
            root
        });
        let redirects_root = |node: NodeRef, root: AigRef| {
            returned == Some(node) || sole_producers.get(&root) == Some(&node.index)
        };
        self.predicates.retain(|&node, plan| {
            let Some(ordinary) = live.predicate_root(node, env, gb) else {
                return false;
            };
            let inputs = plan.operands(env);
            let sources = LazyCell::new(|| inputs.source_gates());
            live.profitable(
                node,
                ordinary,
                gb,
                redirects_root,
                &sources,
                |gb| plan.emit(&inputs, gb),
                |_| {},
            )
        });
        self.windows.retain(|&node, plan| {
            let Some(ordinary) = live.predicate_root(node, env, gb) else {
                return false;
            };
            let Some(inputs) = window_gate::WindowInputs::collect(plan, env) else {
                return false;
            };
            let sources = LazyCell::new(|| inputs.source_gates());
            let mut selector_retained = false;
            let mut observe_remaining = |remaining: &LiveUses| {
                selector_retained = inputs
                    .mapped_selector()
                    .is_some_and(|gate| remaining.counts[gate.id] != 0 && !sources.contains(&gate));
            };
            let profitable = inputs.with_candidates(
                plan,
                |bit| bit,
                |preferred, last| {
                    preferred.iter().copied().chain([last]).any(|emit| {
                        live.profitable(
                            node,
                            ordinary,
                            gb,
                            redirects_root,
                            &sources,
                            emit,
                            &mut observe_remaining,
                        )
                    })
                },
            );
            // A distinct IR value can hash to the same selector AND. Carry
            // the IR reference across mappings so the region prices its fanin
            // with the predicate, just as for a shared IR control.
            if profitable
                && selector_retained
                && let Some(guard) = &plan.guard
                && !plan.retained_controls.contains(&guard.selector)
            {
                plan.retained_controls.push(guard.selector);
            }
            profitable
        });
        (!self.predicates.is_empty() || !self.windows.is_empty()).then_some(live.and_nodes)
    }

    /// Trial the window with its source bits and controls on the boundary.
    /// A control can be computed after some of the ordinary shift gates.
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

    /// Recheck the local trial with the actual operands selected by earlier
    /// lowerings. The completed mapping also gets a joint output check.
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
        let inputs = plan.operands(env);
        // This cut is only a local prefilter. Its cost omits sharing outside
        // the region and source gates that folding can make dead.
        let prefix_end = inputs
            .data
            .iter_lsb_to_msb()
            .chain(inputs.amount.iter_lsb_to_msb())
            .map(|bit| bit.node.id)
            .max()
            .unwrap_or(0)
            + 1;
        let (result, trial) = gb.try_region_alternative(
            ordinary.into(),
            prefix_end,
            inputs.selector.as_slice(),
            |gb| plan.emit(&inputs, gb).into(),
        );
        log::debug!(
            "shift_reduction_predicate trial node={node:?} width={} control_width={} left={} cost={trial:?} elapsed_us={}",
            inputs.data.get_bit_count(),
            inputs.amount.get_bit_count(),
            plan.left,
            started
                .map(|start| start.elapsed().as_micros())
                .unwrap_or(0)
        );
        *result.get_lsb(0)
    }
}

/// Counts uses by live AIG fanins and primary outputs in the ordinary mapping.
pub(super) struct LiveUses {
    counts: Vec<usize>,
    pub(super) and_nodes: usize,
}

/// ANDs released by a probe and newly reachable ANDs counted before rejection.
#[derive(Debug)]
struct ReplacementCost {
    removed: usize,
    added: usize,
}

impl LiveUses {
    /// Skip dead roots and wires before collecting or emitting alternatives.
    fn predicate_root(&self, node: NodeRef, env: &GateEnv, gb: &GateBuilder) -> Option<AigOperand> {
        let ordinary = *env
            .get_bit_vector(node)
            .expect("predicate is lowered")
            .get_lsb(0);
        (self.counts[ordinary.node.id] != 0 && gb.aig_depth(ordinary)? != 0).then_some(ordinary)
    }

    /// Probe on the complete ordinary mapping and restore it for the next root.
    /// Collect the source gates only if the depth warrants an area comparison.
    fn profitable(
        &mut self,
        node: NodeRef,
        ordinary: AigOperand,
        gb: &mut GateBuilder,
        redirects_root: impl FnOnce(NodeRef, AigRef) -> bool,
        source_gates: &LazyCell<HashSet<AigRef>, impl FnOnce() -> HashSet<AigRef>>,
        emit: impl FnOnce(&mut GateBuilder) -> AigOperand,
        observe_remaining: impl FnOnce(&Self),
    ) -> bool {
        let ordinary_depth = gb.aig_depth(ordinary).expect("hashing enabled");
        let checkpoint = gb.begin_append_checkpoint();
        let alternative = emit(gb);
        let depth = gb.aig_depth(alternative).expect("hashing enabled");
        let profitable = if alternative.node == ordinary.node || depth > ordinary_depth {
            false
        } else {
            // A sole producer redirects every fanout, including repeated
            // output bits. An independent alias can retain even a root
            // with just one fanout if its consumer is shared too.
            let redirected = if redirects_root(node, ordinary.node) {
                self.counts[ordinary.node.id]
            } else {
                0
            };
            let cost = self.replacement_cost(
                gb,
                ordinary.node,
                redirected,
                alternative.node,
                source_gates,
                observe_remaining,
            );
            log::debug!(
                "shift_reduction_predicate live trial node={node:?} cost={cost:?} ordinary_depth={ordinary_depth} alternative_depth={depth}"
            );
            cost.added < cost.removed || (cost.added == cost.removed && depth < ordinary_depth)
        };
        gb.rollback_append_checkpoint(checkpoint);
        profitable
    }

    /// Seed every output edge, then propagate uses through live gates in
    /// reverse builder order, where every fanout precedes its fanins.
    pub(super) fn from_outputs(gb: &GateBuilder) -> Self {
        let mut counts = vec![0; gb.gates.len()];
        for bit in output_bits(gb) {
            counts[bit.node.id] += 1;
        }
        let mut and_nodes = 0;
        for (id, gate) in gb.gates.iter().enumerate().rev() {
            if counts[id] != 0
                && let AigNode::And2 { a, b, .. } = gate
            {
                and_nodes += 1;
                counts[a.node.id] += 1;
                counts[b.node.id] += 1;
            }
        }
        Self { counts, and_nodes }
    }

    /// Tests whether the released ANDs pay for both the replacement roots and
    /// their downstream predicates. Detach users before fanins for nested
    /// roots, then restore the original counts for the next read.
    pub(super) fn replacing_saves_ands(
        &mut self,
        gb: &GateBuilder,
        roots_descending: &[AigRef],
        replacements: impl Iterator<Item = AigRef>,
        predicate_ands: usize,
    ) -> bool {
        let mut removed = 0;
        let mut uses = Vec::with_capacity(roots_descending.len());
        for &root in roots_descending {
            let count = self.counts[root.id] as isize;
            uses.push(count);
            removed += self.adjust(gb, root, -count);
        }
        let budget = removed.saturating_sub(predicate_ands);
        let profitable = self.new_ands_up_to(gb, replacements, budget) < budget;
        for (&root, count) in roots_descending.iter().zip(uses).rev() {
            self.adjust(gb, root, count);
        }
        profitable
    }

    /// Counts ANDs that cross the live boundary after changing a root's use
    /// count. Each live AND contributes one reference to each of its fanins.
    fn adjust(&mut self, gb: &GateBuilder, root: AigRef, delta: isize) -> usize {
        let mut pending = vec![(root, delta)];
        let mut changed = 0;
        while let Some((node, delta)) = pending.pop() {
            let uses = &mut self.counts[node.id];
            let was_live = *uses > 0;
            *uses = uses
                .checked_add_signed(delta)
                .expect("invalid gate use count");
            if was_live != (*uses > 0)
                && let AigNode::And2 { a, b, .. } = gb.gates[node.id]
            {
                changed += 1;
                let fanin_delta = delta.signum();
                pending.extend([(a.node, fanin_delta), (b.node, fanin_delta)]);
            }
        }
        changed
    }

    /// Measures replacement area and exposes remaining live uses while the
    /// redirected edges are detached. The source cut stays live during the
    /// probe.
    fn replacement_cost(
        &mut self,
        gb: &GateBuilder,
        ordinary: AigRef,
        redirected: usize,
        alternative: AigRef,
        sources: &HashSet<AigRef>,
        observe_remaining: impl FnOnce(&Self),
    ) -> ReplacementCost {
        // Only spend savings from the replaced region. Holding its
        // ordinary sources live bounds the removal walk to the lowering.
        let mut detached = Vec::new();
        let mut pending = vec![(ordinary, redirected)];
        let mut removed = 0;
        while let Some((node, uses)) = pending.pop() {
            if !sources.contains(&node)
                && let AigNode::And2 { a, b, .. } = gb.gates[node.id]
            {
                self.counts[node.id] -= uses;
                detached.push((node, uses));
                if self.counts[node.id] == 0 {
                    removed += 1;
                    pending.extend([(a.node, 1), (b.node, 1)]);
                }
            }
        }
        // The first excess AND rules out the candidate, even if a newly
        // reachable source has a much larger dead cone below it.
        let added = self.new_ands_up_to(gb, [alternative], removed + 1);
        observe_remaining(self);
        for (node, uses) in detached {
            self.counts[node.id] += uses;
        }
        ReplacementCost { removed, added }
    }

    /// Counts dead ANDs once across all replacement roots. Stop as soon as
    /// the cost rules out the trial, even partway through a single dead cone.
    fn new_ands_up_to(
        &self,
        gb: &GateBuilder,
        roots: impl IntoIterator<Item = AigRef>,
        limit: usize,
    ) -> usize {
        let mut roots = roots.into_iter();
        let mut added = HashSet::new();
        let mut pending = Vec::new();
        while added.len() < limit
            && let Some(node) = pending.pop().or_else(|| roots.next())
        {
            if self.counts.get(node.id).copied().unwrap_or(0) == 0
                && let AigNode::And2 { a, b, .. } = gb.gates[node.id]
                && added.insert(node)
            {
                pending.extend([a.node, b.node]);
            }
        }
        added.len()
    }
}

fn output_bits(gb: &GateBuilder) -> impl Iterator<Item = &AigOperand> {
    gb.outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
}

/// Independent probes can compete for savings or change downstream folding.
/// Check the combined area and each output bit's depth on the finished map.
pub(super) fn outputs_no_worse(
    ordinary: &GateBuilder,
    ordinary_area: usize,
    trial: &GateBuilder,
) -> bool {
    output_bits(ordinary)
        .zip(output_bits(trial))
        .all(|(old, new)| {
            trial.aig_depth(*new).expect("hashing enabled")
                <= ordinary.aig_depth(*old).expect("hashing enabled")
        })
        && LiveUses::from_outputs(trial).and_nodes <= ordinary_area
}

impl PredicateInputs {
    /// Gates at the data, shift amount and selector boundary of the trial.
    fn source_gates(&self) -> HashSet<AigRef> {
        #[cfg(test)]
        tests::SOURCE_GATE_COLLECTIONS.set(tests::SOURCE_GATE_COLLECTIONS.get() + 1);
        self.data
            .iter_lsb_to_msb()
            .chain(self.amount.iter_lsb_to_msb())
            .chain(self.selector.iter())
            .map(|bit| bit.node)
            .collect()
    }
}

impl Predicate {
    fn operands(&self, env: &GateEnv) -> PredicateInputs {
        let data = env
            .get_bit_vector(self.data)
            .expect("planned data is lowered");
        let amount = env
            .get_bit_vector(self.amount)
            .expect("planned amount is lowered");
        let selector = self.selection.as_ref().map(|selection| {
            *env.get_bit_vector(selection.selector)
                .expect("planned selector is lowered")
                .get_lsb(0)
        });
        PredicateInputs {
            data,
            amount,
            selector,
        }
    }

    /// Builds the shift predicate and applies its optional selection and
    /// polarity.
    fn emit(&self, inputs: &PredicateInputs, gb: &mut GateBuilder) -> AigOperand {
        let mut bits: Vec<_> = inputs.data.iter_lsb_to_msb().copied().collect();
        if self.left {
            bits.reverse();
        }
        let mut predicate = shifted_nonzero(&bits, &inputs.amount, gb);
        if let Some(selection) = &self.selection {
            let selector = inputs.selector.expect("selection has a selector");
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
        if self.inverted {
            predicate.negate()
        } else {
            predicate
        }
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

/// Return the observed word and whether its nonzero result is inverted.
fn observed(
    payload: &NodePayload,
    mut is_zero: impl FnMut(NodeRef) -> bool,
) -> Option<(NodeRef, bool)> {
    match *payload {
        NodePayload::Unop(Unop::OrReduce, word) => Some((word, false)),
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) => {
            let word = if is_zero(lhs) {
                rhs
            } else if is_zero(rhs) {
                lhs
            } else {
                return None;
            };
            Some((word, op == Binop::Eq))
        }
        _ => None,
    }
}

/// Classifies a literal word without converting its width to a host integer.
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
    use crate::aig::get_summary_stats::get_aig_cone_stats;
    use crate::gate_builder::GateBuilderOptions;
    use crate::gatify::ir2gate::{GateOrVec, GatifyOptions, gatify, gatify_prepared_fn};
    use crate::ir2gate_utils::{Direction, gatify_barrel_shifter};
    use std::cell::Cell;
    use xlsynth_pir::ir_parser::Parser;

    thread_local! {
        pub(super) static SOURCE_GATE_COLLECTIONS: Cell<usize> = const { Cell::new(0) };
    }

    #[test]
    fn identical_shift_predicate_skips_source_gate_collection() {
        let package = Parser::new(
            r#"package sample
top fn main(x: bits[1], k: bits[65536]) -> bits[1] {
  shifted: bits[1] = shrl(x, k, id=3)
  ret predicate: bits[1] = or_reduce(shifted, id=4)
}"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let function = package.get_top_fn().unwrap();
        for lower in [gatify_prepared_fn, gatify] {
            SOURCE_GATE_COLLECTIONS.set(0);
            let gates = lower(function, GatifyOptions::all_opts_disabled())
                .unwrap()
                .gate_fn;
            // Both lowerings build x & !or_reduce(k) with the same balanced
            // reduction.
            let root = gates.outputs[0].bit_vector.get_lsb(0);
            let stats = get_aig_cone_stats(&gates.gates, std::slice::from_ref(root));
            assert_eq!(stats.and_nodes, 65_536);
            assert_eq!(stats.root_depths, [17]);
            assert_eq!(SOURCE_GATE_COLLECTIONS.get(), 0);
        }
    }

    #[test]
    fn probes_stop_before_walking_a_shared_dead_source() {
        let mut gb = GateBuilder::new("shared".into(), GateBuilderOptions::opt());
        let controls = gb.add_input("a".into(), 512);
        let b_input = gb.add_input("b".into(), 512);
        let h_input = gb.add_input("h".into(), 512);
        let b = gb.add_or_reduce(&b_input, ReductionKind::Tree);
        let h = gb.add_or_reduce(&h_input, ReductionKind::Tree);
        let mut probes = Vec::new();
        for &a in controls.iter_lsb_to_msb() {
            let data = AigBitVector::from_lsb_is_index_0(&[h, a, b, b]);
            let amount = AigBitVector::from_lsb_is_index_0(&[a, a.negate()]);
            let shifted = gatify_barrel_shifter(&data, &amount, Direction::Right, "shift", &mut gb);
            let ordinary = gb.add_or_reduce(&shifted, ReductionKind::Tree);
            gb.add_output(format!("p{}", probes.len()), ordinary.into());
            probes.push((
                ordinary,
                PredicateInputs {
                    data,
                    amount,
                    selector: None,
                },
            ));
        }
        let mut live = LiveUses::from_outputs(&gb);
        assert_eq!(live.counts[h.node.id], 0);
        let initial = live.counts.clone();
        for (ordinary, inputs) in probes {
            let checkpoint = gb.begin_append_checkpoint();
            let data: Vec<_> = inputs.data.iter_lsb_to_msb().copied().collect();
            let alternative = shifted_nonzero(&data, &inputs.amount, &mut gb);
            let sources = inputs.source_gates();
            let cost =
                live.replacement_cost(&gb, ordinary.node, 1, alternative.node, &sources, |_| {});
            // Only the local four-bit shift and reduction can fund this probe.
            assert!(cost.removed <= 10, "{cost:?}");
            assert_eq!(cost.added, cost.removed + 1);
            gb.rollback_append_checkpoint(checkpoint);
        }
        assert_eq!(live.counts, initial);
    }

    #[test]
    fn read_retry_only_inspects_columns_within_the_released_area() {
        let mut gb = GateBuilder::new("columns".into(), GateBuilderOptions::opt());
        let data = gb.add_input("data".into(), 4096);
        let dead_column = gb.add_and_reduce(&data, ReductionKind::Tree);
        let pair = gb.add_and_binary(*data.get_lsb(0), *data.get_lsb(1));
        let pred_bits = gb.add_input("predicate".into(), 4);
        let predicate = gb.add_and_reduce(&pred_bits, ReductionKind::Tree);
        let zero = gb.get_false();
        gb.add_output("folded".into(), zero.into());
        gb.add_output("predicate".into(), predicate.into());
        let mut live = LiveUses::from_outputs(&gb);
        let initial = live.counts.clone();
        let mut columns = [dead_column.node; 2].into_iter();

        // Folded predicates release no ANDs. Three released ANDs minus one
        // downstream AND let us inspect at most two dead ANDs.
        assert!(!live.replacing_saves_ands(&gb, &[zero.node], columns.by_ref(), 2));
        assert_eq!(columns.len(), 2);
        assert!(!live.replacing_saves_ands(&gb, &[predicate.node], columns.by_ref(), 1));
        assert_eq!(columns.len(), 1);
        assert_eq!(live.new_ands_up_to(&gb, columns, 2), 2);
        // Two columns sharing a single dead AND still fit that budget.
        assert!(live.replacing_saves_ands(&gb, &[predicate.node], [pair.node; 2].into_iter(), 1));
        assert_eq!(live.counts, initial);
    }

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
