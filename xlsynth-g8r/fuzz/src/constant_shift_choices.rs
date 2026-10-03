// SPDX-License-Identifier: Apache-2.0

//! Structural and independent semantic checks for targeted shift-choice
//! fuzzing.

use std::collections::{HashMap, HashSet};

use xlsynth_g8r::gatify::ir2gate::GateBuilderCostEvaluator;
use xlsynth_pir::IrValue;
use xlsynth_pir::constant_shift_choices::{
    ShiftChoiceCostGraph, rewrite_constant_shift_choices_with_evaluator,
};
use xlsynth_pir::dce::get_dead_nodes;
use xlsynth_pir::ir::{self, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_cost::{IrCost, ShiftChoiceCostEvaluator};
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_utils::{operands, push_node, remap_payload_with};
use xlsynth_pir::ir_verify::verify_function;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice_with_limits};

use crate::constant_shift_choices_sample::{Expectation, Sample, generate_sample};
use crate::fuzz_solver_limits;

const INJECTED_ERROR: &str = "deliberate fuzz cost-evaluator failure";

/// Counts exercised paths separately from completed semantic proofs.
#[derive(Clone, Copy, Debug, Default)]
pub struct FuzzStats {
    pub samples: u64,
    pub modes: [u64; 8],
    pub changed: u64,
    pub unchanged: u64,
    pub rewrites: u64,
    pub evaluator_calls: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub expected_errors: u64,
    pub rollback_after_accept: u64,
    pub local_smt_proofs: u64,
    pub full_smt_proofs: u64,
    pub local_exhaustive_proofs: u64,
    pub full_exhaustive_proofs: u64,
    pub exhaustive_assignments: u64,
    pub inconclusive_proofs: u64,
}

impl FuzzStats {
    /// Adds a completed sample's evidence to a worker's campaign totals.
    pub fn accumulate(&mut self, other: Self) {
        self.samples += other.samples;
        for (total, value) in self.modes.iter_mut().zip(other.modes) {
            *total += value;
        }
        self.changed += other.changed;
        self.unchanged += other.unchanged;
        self.rewrites += other.rewrites;
        self.evaluator_calls += other.evaluator_calls;
        self.accepted += other.accepted;
        self.rejected += other.rejected;
        self.expected_errors += other.expected_errors;
        self.rollback_after_accept += other.rollback_after_accept;
        self.local_smt_proofs += other.local_smt_proofs;
        self.full_smt_proofs += other.full_smt_proofs;
        self.local_exhaustive_proofs += other.local_exhaustive_proofs;
        self.full_exhaustive_proofs += other.full_exhaustive_proofs;
        self.exhaustive_assignments += other.exhaustive_assignments;
        self.inconclusive_proofs += other.inconclusive_proofs;
    }
}

/// Controls profitability and transaction failures without production hooks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Accept,
    Real,
    Reject,
    Alternating,
    ErrorOnCall(usize),
}

impl Mode {
    fn from_byte(byte: u8) -> Self {
        match byte % 8 {
            0 => Self::Accept,
            1 => Self::Real,
            2 => Self::Reject,
            3 => Self::Alternating,
            n => Self::ErrorOnCall(usize::from(n - 3)),
        }
    }
}

/// Owns one callback's region so subsequent mutation cannot invalidate
/// evidence.
struct RegionSnapshot {
    function: ir::Fn,
    inputs: Vec<NodeRef>,
    retained_outputs: Vec<NodeRef>,
}

/// Requires incumbent interior values that escape to live users to be retained.
fn check_retained_outputs(
    function: &ir::Fn,
    nodes: &[NodeRef],
    outputs: &[NodeRef],
) -> Result<(), String> {
    let (&primary, retained) = outputs
        .split_first()
        .ok_or("cost graph has no primary result")?;
    let interior: HashSet<_> = nodes.iter().copied().collect();
    let missing = |reference: NodeRef| {
        reference != primary && interior.contains(&reference) && !retained.contains(&reference)
    };
    if function.ret_node_ref.is_some_and(missing) {
        return Err("cost graph omitted an interior value returned by the function".to_string());
    }
    // New candidate nodes have no live users yet. Ignoring dead users also
    // avoids requiring retained outputs solely for that speculative suffix.
    let dead: HashSet<_> = get_dead_nodes(function).into_iter().collect();
    for (index, node) in function.nodes.iter().enumerate() {
        let user = NodeRef { index };
        if interior.contains(&user) || dead.contains(&user) {
            continue;
        }
        for operand in operands(&node.payload) {
            if missing(operand) {
                return Err(format!(
                    "cost graph omitted interior node {} used by live external node {index}",
                    operand.index
                ));
            }
        }
    }
    Ok(())
}

/// Checks the advertised cut and copies only its explicitly listed nodes.
fn snapshot(graph: &ShiftChoiceCostGraph<'_>) -> Result<RegionSnapshot, String> {
    let source = graph.function();
    let mut function = ir::Fn {
        graph: ir::NodeGraph::new("cost_region"),
        params: Vec::new(),
        ret_ty: Type::Tuple(Vec::new()),
        ret_node_ref: None,
    };
    let mut mapping = HashMap::new();
    for (index, &input) in graph.inputs().iter().enumerate() {
        let node = source
            .nodes
            .get(input.index)
            .ok_or("invalid boundary reference")?;
        let param = push_node(&mut function, node.ty.clone(), NodePayload::Param);
        function.get_node_mut(param).name = Some(format!("input_{index}"));
        function.params.push(param);
        if mapping.insert(input, param).is_some() {
            return Err("duplicate boundary input".to_string());
        }
    }
    for &reference in graph.nodes() {
        let node = source
            .nodes
            .get(reference.index)
            .ok_or("invalid interior reference")?;
        if operands(&node.payload)
            .iter()
            .any(|operand| !mapping.contains_key(operand))
        {
            return Err("region is not closed and dependency ordered".to_string());
        }
        let payload = remap_payload_with(&node.payload, |(_, operand)| mapping[&operand]);
        let copied = push_node(&mut function, node.ty.clone(), payload);
        if mapping.insert(reference, copied).is_some() {
            return Err("duplicate interior node or overlap with boundary".to_string());
        }
    }
    if graph.outputs().is_empty() {
        return Err("cost graph has no primary result".to_string());
    }
    let outputs = graph
        .outputs()
        .iter()
        .map(|output| {
            mapping
                .get(output)
                .copied()
                .ok_or("output is outside the region".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let interior: HashSet<_> = graph.nodes().iter().copied().collect();
    let mut reachable = HashSet::new();
    let mut pending = graph.outputs().to_vec();
    while let Some(reference) = pending.pop() {
        if interior.contains(&reference) && reachable.insert(reference) {
            pending.extend(operands(&source.get_node(reference).payload));
        }
    }
    if reachable != interior {
        return Err("cost graph includes unreachable interior nodes".to_string());
    }
    function.ret_ty = Type::Tuple(
        outputs
            .iter()
            .map(|reference| Box::new(function.get_node(*reference).ty.clone()))
            .collect(),
    );
    let return_type = function.ret_ty.clone();
    function.ret_node_ref = Some(push_node(
        &mut function,
        return_type,
        NodePayload::Tuple(outputs),
    ));
    verify_function(&function).map_err(|error| format!("invalid materialized region: {error}"))?;
    Ok(RegionSnapshot {
        function,
        inputs: graph.inputs().to_vec(),
        retained_outputs: graph.outputs()[1..].to_vec(),
    })
}

/// Proves PIR directly with Bitwuzla; g8r's lowering is not part of the oracle.
fn smt_equivalence(lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
    prover_for_choice_with_limits(SolverChoice::Bitwuzla, None, fuzz_solver_limits())
        .prove_ir_fn_equiv(lhs, rhs)
}

/// Enumerates independent parameter bits and compares return values and
/// effects.
fn exhaustive_equivalence(lhs: &ir::Fn, rhs: &ir::Fn, max_bits: usize) -> Result<u64, String> {
    let bit_count: usize = lhs.param_nodes().map(|node| node.ty.bit_count()).sum();
    if bit_count > max_bits {
        return Err(format!(
            "exhaustive sample has {bit_count} input bits; limit is {max_bits}"
        ));
    }
    let assignments = 1u64 << bit_count;
    for assignment in 0..assignments {
        let mut remaining = assignment;
        let args = lhs
            .param_nodes()
            .map(|node| match node.ty {
                Type::Bits(width) => {
                    let value = remaining & ((1u64 << width) - 1);
                    remaining >>= width;
                    Ok(IrValue::make_ubits(width, value).expect("bounded value fits its width"))
                }
                Type::Token => Ok(IrValue::make_token()),
                _ => Err("generator introduced a non-scalar parameter".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lhs_result = eval_fn(lhs, &args);
        let rhs_result = eval_fn(rhs, &args);
        if lhs_result != rhs_result {
            return Err(format!(
                "exhaustive counterexample {args:?}:\nlhs={lhs_result:?}\nrhs={rhs_result:?}"
            ));
        }
    }
    Ok(assignments)
}

/// Records only completed proofs; a solver limit never increments a proof
/// count.
fn check_equivalence(
    lhs: &ir::Fn,
    rhs: &ir::Fn,
    exhaustive: bool,
    local: bool,
    stats: &mut FuzzStats,
) -> Result<(), String> {
    if lhs.get_type() != rhs.get_type() {
        return Err("equivalence pair has different signatures".to_string());
    }
    if exhaustive {
        stats.exhaustive_assignments +=
            exhaustive_equivalence(lhs, rhs, if local { 16 } else { 10 })?;
        if local {
            stats.local_exhaustive_proofs += 1;
        } else {
            stats.full_exhaustive_proofs += 1;
        }
    } else {
        match smt_equivalence(lhs, rhs) {
            EquivResult::Proved => {
                if local {
                    stats.local_smt_proofs += 1;
                } else {
                    stats.full_smt_proofs += 1;
                }
            }
            EquivResult::Inconclusive(reason) => {
                // Resource limits are tracked separately; this is not a proof.
                stats.inconclusive_proofs += 1;
                log::debug!("shift-choice proof inconclusive: {reason}");
            }
            result => return Err(format!("independent PIR proof failed: {result:?}")),
        }
    }
    Ok(())
}

/// Captures both alternatives before answering each profitability query.
struct RecordingEvaluator {
    mode: Mode,
    exhaustive: bool,
    incumbent: Option<(RegionSnapshot, IrCost)>,
    real: GateBuilderCostEvaluator,
    stats: FuzzStats,
    injected_error: bool,
    failure: Option<String>,
}

impl RecordingEvaluator {
    fn new(mode: Mode, exhaustive: bool) -> Self {
        Self {
            mode,
            exhaustive,
            incumbent: None,
            real: GateBuilderCostEvaluator::default(),
            stats: FuzzStats::default(),
            injected_error: false,
            failure: None,
        }
    }

    /// Checks completed pairs even if their cost will reject or abort the
    /// rewrite.
    fn record(&mut self, graph: &ShiftChoiceCostGraph<'_>) -> Result<IrCost, String> {
        self.stats.evaluator_calls += 1;
        let call = self.stats.evaluator_calls as usize;
        let candidate_call = call % 2 == 0;
        let captured = snapshot(graph)?;
        if !candidate_call {
            // Only the incumbent is connected to current function users;
            // candidate nodes are still an uncommitted alternative.
            check_retained_outputs(graph.function(), graph.nodes(), graph.outputs())?;
        }
        let accept = match self.mode {
            Mode::Reject => false,
            Mode::Alternating => ((call - 1) / 2) % 2 == 0,
            _ => true,
        };
        let cost = if self.mode == Mode::Real {
            self.real.estimate(graph)?
        } else {
            IrCost {
                area: if candidate_call && accept { 0 } else { 1 },
                delay: 1.0,
            }
        };
        let improvement = if candidate_call {
            let (incumbent, incumbent_cost) =
                self.incumbent.take().ok_or("candidate without incumbent")?;
            if incumbent.inputs != captured.inputs
                || incumbent.retained_outputs != captured.retained_outputs
            {
                return Err(
                    "alternatives changed boundary input or retained-output order".to_string(),
                );
            }
            check_equivalence(
                &incumbent.function,
                &captured.function,
                self.exhaustive,
                true,
                &mut self.stats,
            )
            .map_err(|error| {
                format!(
                    "{error}\nIncumbent local IR:\n{}\nCandidate local IR:\n{}",
                    incumbent.function, captured.function
                )
            })?;
            Some(cost.is_pareto_improvement_on(incumbent_cost))
        } else {
            self.incumbent = Some((captured, cost));
            None
        };
        if self.mode == Mode::ErrorOnCall(call) {
            self.injected_error = true;
            return Err(INJECTED_ERROR.to_string());
        }
        if let Some(improvement) = improvement {
            if improvement {
                self.stats.accepted += 1;
            } else {
                self.stats.rejected += 1;
            }
        }
        Ok(cost)
    }
}

impl ShiftChoiceCostEvaluator for RecordingEvaluator {
    fn estimate(&mut self, graph: &ShiftChoiceCostGraph<'_>) -> Result<IrCost, String> {
        let result = self.record(graph);
        if let Err(error) = &result {
            if error != INJECTED_ERROR {
                self.failure = Some(error.clone());
            }
        }
        result
    }
}

/// Checks one transaction, including exact rollback and its complete semantics.
fn check_sample(sample: &Sample, mode: Mode, rewritten: &mut ir::Fn) -> Result<FuzzStats, String> {
    let original = &sample.function;
    verify_function(original).map_err(|error| format!("invalid generated function: {error}"))?;
    let mut evaluator = RecordingEvaluator::new(mode, sample.exhaustive_inputs);
    let result =
        rewrite_constant_shift_choices_with_evaluator(rewritten, sample.limits, &mut evaluator);
    if let Some(error) = evaluator.failure {
        return Err(error);
    }
    let rewrites = match result {
        Ok(count) if !evaluator.injected_error => count,
        Err(error) if evaluator.injected_error && error == INJECTED_ERROR => {
            evaluator.stats.expected_errors += 1;
            // Calls 3 and 4 abort after the first pair was accepted and
            // applied.
            if evaluator.stats.evaluator_calls > 2 && evaluator.stats.accepted > 0 {
                evaluator.stats.rollback_after_accept += 1;
            }
            0
        }
        result => return Err(format!("unexpected rewrite result: {result:?}")),
    };
    if !evaluator.injected_error && evaluator.incumbent.is_some() {
        return Err("successful rewrite stopped halfway through a comparison".to_string());
    }
    // Only limit-constrained samples may accept a site and later discard the
    // complete transaction. Any committed transaction must apply every
    // accepted choice, including in the alternating and real-cost modes.
    if !evaluator.injected_error
        && (rewrites != 0 || sample.expectation == Expectation::GuaranteedCandidate)
        && rewrites as u64 != evaluator.stats.accepted
    {
        return Err(format!(
            "rewrite committed {rewrites} sites after accepting {} choices",
            evaluator.stats.accepted
        ));
    }
    if sample.expectation == Expectation::GuaranteedRejected && evaluator.stats.evaluator_calls != 0
    {
        return Err("an ineligible amount reached costing".to_string());
    }
    if sample.expectation == Expectation::GuaranteedCandidate {
        if evaluator.stats.evaluator_calls == 0 {
            return Err("guaranteed candidate never reached costing".to_string());
        }
        if mode == Mode::Accept && rewrites == 0 {
            return Err("forced acceptance failed to apply a guaranteed rewrite".to_string());
        }
    }
    if mode == Mode::Reject && rewrites != 0 {
        return Err("equal-cost candidates were applied".to_string());
    }
    if rewrites == 0 && format!("{original:?}") != format!("{rewritten:?}") {
        return Err("rejection or error changed the input function".to_string());
    }
    verify_function(rewritten).map_err(|error| format!("invalid rewritten function: {error}"))?;
    let original_params = original
        .param_nodes()
        .map(|node| &node.name)
        .collect::<Vec<_>>();
    let rewritten_params = rewritten
        .param_nodes()
        .map(|node| &node.name)
        .collect::<Vec<_>>();
    if original.name != rewritten.name || original_params != rewritten_params {
        return Err("rewrite changed function or parameter names".to_string());
    }
    check_equivalence(
        original,
        rewritten,
        sample.exhaustive_inputs,
        false,
        &mut evaluator.stats,
    )?;
    evaluator.stats.samples = 1;
    evaluator.stats.rewrites = rewrites as u64;
    if rewrites == 0 {
        evaluator.stats.unchanged = 1;
    } else {
        evaluator.stats.changed = 1;
    }
    Ok(evaluator.stats)
}

/// Decodes one replayable fuzz input and checks the requested cost decision
/// path.
pub fn check_input(data: &[u8]) -> FuzzStats {
    let (mode_byte, graph_bytes) = data
        .split_first()
        .map_or((0, &[][..]), |(&mode, rest)| (mode, rest));
    let mode = Mode::from_byte(mode_byte);
    let sample = generate_sample(graph_bytes);
    let mut rewritten = sample.function.clone();
    let mut stats = check_sample(&sample, mode, &mut rewritten).unwrap_or_else(|error| {
        panic!("constant-shift-choice failure: {error}\nMode: {mode:?}\nBytes: {data:02x?}\nLimits: {:?}\nOriginal IR:\n{}\nResult IR:\n{}", sample.limits, sample.function, rewritten)
    });
    stats.modes[usize::from(mode_byte % 8)] = 1;
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::FnBuilder;
    use xlsynth_pir::constant_shift_choices::{
        ConstantShiftChoiceLimits, constant_shift_choice_candidate,
    };

    /// Two independent shifts ensure errors can follow an already applied site.
    fn two_sites() -> Sample {
        let mut builder = FnBuilder::new("two_sites");
        let x = builder.param("x", Type::Bits(4)).unwrap();
        let selector = builder.param("p", Type::Bits(1)).unwrap();
        let zero = builder.literal(IrValue::make_ubits(3, 0).unwrap()).unwrap();
        let one = builder.literal(IrValue::make_ubits(3, 1).unwrap()).unwrap();
        let amount = builder.select(selector, &[zero, one], None).unwrap();
        let left = builder.shll(x, amount).unwrap();
        let right = builder.shrl(x, amount).unwrap();
        let result = builder.tuple(&[left, right]).unwrap();
        Sample {
            function: builder.build(result).unwrap(),
            limits: ConstantShiftChoiceLimits::default(),
            expectation: Expectation::GuaranteedCandidate,
            exhaustive_inputs: false,
        }
    }

    #[test]
    fn decision_modes_and_late_rollback_are_exercised() {
        for byte in 0..8 {
            let sample = two_sites();
            let mut rewritten = sample.function.clone();
            let stats = check_sample(&sample, Mode::from_byte(byte), &mut rewritten).unwrap();
            assert_eq!(stats.full_smt_proofs, 1);
            match byte {
                0 => assert_eq!(stats.rewrites, 2),
                1 => assert_eq!(stats.evaluator_calls, 4),
                2 => assert_eq!(stats.rejected, 2),
                3 => {
                    assert_eq!(stats.rewrites, 1);
                    assert_eq!(stats.rejected, 1);
                }
                4 | 5 => assert_eq!(stats.expected_errors, 1),
                6 | 7 => assert_eq!(stats.rollback_after_accept, 1),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn exhaustive_lanes_check_special_selects_and_observable_effects() {
        for lane in [6, 7] {
            for variant in 0..8 {
                let stats = check_input(&[0, lane, 3, 1, 1, 1, variant]);
                assert_eq!(stats.full_exhaustive_proofs, 1);
                assert_eq!(stats.inconclusive_proofs, 0);
            }
        }
    }

    #[test]
    fn accepted_choices_must_commit_unless_the_total_limit_can_abort() {
        let mut sample = two_sites();
        // One choice needs a bound of seven emitted nodes. The next site
        // exceeds this total after the first has already been accepted.
        sample.limits.max_emitted_nodes = 7;
        sample.expectation = Expectation::Unspecified;
        let mut rewritten = sample.function.clone();
        let stats = check_sample(&sample, Mode::Alternating, &mut rewritten).unwrap();
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.rewrites, 0);

        // A no-op after acceptance must fail the harness when the generator
        // promised enough budget for its candidates, even in alternating mode.
        sample.expectation = Expectation::GuaranteedCandidate;
        assert!(check_sample(&sample, Mode::Alternating, &mut rewritten).is_err());
    }

    #[test]
    fn retained_outputs_cover_live_users_returns_and_effects() {
        let mut function = two_sites().function;
        let NodePayload::Tuple(results) =
            &function.get_node(function.ret_node_ref.unwrap()).payload
        else {
            panic!("expected two shift results");
        };
        let primary = results[0];
        let NodePayload::Binop(_, _, amount) = function.get_node(primary).payload else {
            panic!("expected a shift");
        };
        let nodes = [amount, primary];
        assert!(check_retained_outputs(&function, &nodes, &[primary]).is_err());
        check_retained_outputs(&function, &nodes, &[primary, amount]).unwrap();

        // The second shift and tuple now have no live users, just like an
        // uncommitted candidate suffix. Extra retained outputs remain legal.
        function.ret_node_ref = Some(primary);
        function.ret_ty = function.get_node(primary).ty.clone();
        check_retained_outputs(&function, &nodes, &[primary]).unwrap();
        check_retained_outputs(&function, &nodes, &[primary, amount]).unwrap();

        function.ret_node_ref = Some(amount);
        function.ret_ty = function.get_node(amount).ty.clone();
        assert!(check_retained_outputs(&function, &nodes, &[primary]).is_err());
        check_retained_outputs(&function, &nodes, &[primary, amount]).unwrap();

        function.ret_node_ref = Some(primary);
        function.ret_ty = function.get_node(primary).ty.clone();
        let token = push_node(
            &mut function,
            Type::Token,
            NodePayload::AfterAll(Vec::new()),
        );
        let activated = function.params[1];
        push_node(
            &mut function,
            Type::Token,
            NodePayload::Trace {
                token,
                activated,
                format: "amount={}".to_string(),
                verbosity: 0,
                operands: vec![amount],
            },
        );
        verify_function(&function).unwrap();
        assert!(check_retained_outputs(&function, &nodes, &[primary]).is_err());
        check_retained_outputs(&function, &nodes, &[primary, amount]).unwrap();
    }

    #[test]
    fn independent_oracle_detects_swapped_branches() {
        let source = two_sites().function;
        let mut candidate =
            constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default()).unwrap();
        let cases = candidate
            .nodes
            .iter_mut()
            .find_map(|node| match &mut node.payload {
                NodePayload::Sel { cases, .. } => Some(cases),
                _ => None,
            })
            .unwrap();
        cases.swap(0, 1);
        verify_function(&candidate).unwrap();
        assert!(matches!(
            smt_equivalence(&source, &candidate),
            EquivResult::Disproved { .. }
        ));
    }

    #[test]
    fn independent_oracle_detects_wrong_slice_offset() {
        let source = two_sites().function;
        let mut candidate =
            constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default()).unwrap();
        let start = candidate
            .nodes
            .iter_mut()
            .find_map(|node| match &mut node.payload {
                NodePayload::BitSlice {
                    start, width: 3, ..
                } => Some(start),
                _ => None,
            })
            .unwrap();
        *start = 1 - *start;
        verify_function(&candidate).unwrap();
        assert!(matches!(
            smt_equivalence(&source, &candidate),
            EquivResult::Disproved { .. }
        ));
    }
}
