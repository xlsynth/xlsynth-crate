// SPDX-License-Identifier: Apache-2.0

//! Bounded recognition and construction of constant logical-shift choices.

use std::collections::{HashMap, HashSet};

use crate::IrBits;
use crate::dce::get_dead_nodes;
use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use crate::ir_cost::{IrCost, ShiftChoiceCostEvaluator};
use crate::ir_match::MatchCtx;
use crate::ir_utils;
use crate::ir_utils::{make_constant_shift_expr, make_constant_shrl_bit_slice_expr, push_node};
use crate::ir_value_utils::ir_bits_to_usize;

/// Bounds for recognizing and emitting one function's constant-shift choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstantShiftChoiceLimits {
    /// Distinct effective amounts per shift, with all overshifts counted once.
    pub max_distinct_shifts: usize,
    /// Inspected amount nodes per shift, also bounding flattened AND operands
    /// and the nodes in each local cost graph before projection expansion.
    pub max_visited_nodes: usize,
    /// Conservative upper bound on newly emitted IR nodes for the function.
    pub max_emitted_nodes: usize,
    /// Largest data or amount width considered, bounding temporary bitvectors
    /// and projection literals before they are allocated.
    pub max_bit_width: usize,
}

impl Default for ConstantShiftChoiceLimits {
    fn default() -> Self {
        Self {
            max_distinct_shifts: 4,
            max_visited_nodes: 64,
            max_emitted_nodes: 128,
            max_bit_width: 16_384,
        }
    }
}

/// Borrows one small IR region supplied by the constant-shift-choice rewrite.
///
/// Each comparison uses the same boundary inputs and returns the shifted
/// value followed by any interior values needed by users outside the region.
/// Only matched amount operations, logical shifts, and transparent wiring or
/// inversion occur inside the region; other computations are inputs. The
/// containing function also holds unrelated nodes and speculative alternatives;
/// evaluators must only lower the listed nodes and treat the inputs as leaves.
#[derive(Clone, Copy, Debug)]
pub struct ShiftChoiceCostGraph<'a> {
    function: &'a ir::Fn,
    nodes: &'a [NodeRef],
    inputs: &'a [NodeRef],
    outputs: &'a [NodeRef],
}

impl ShiftChoiceCostGraph<'_> {
    /// Borrows the containing function to look up the region's node payloads.
    pub fn function(&self) -> &ir::Fn {
        self.function
    }

    /// Lists reachable interior nodes in dependency order, excluding inputs.
    pub fn nodes(&self) -> &[NodeRef] {
        self.nodes
    }

    /// Lists the shared ordered boundary, including inputs unused by this side.
    pub fn inputs(&self) -> &[NodeRef] {
        self.inputs
    }

    /// Lists the primary result followed by retained values in comparison
    /// order.
    pub fn outputs(&self) -> &[NodeRef] {
        self.outputs
    }
}

/// A choice refers only to earlier entries, so repeated subexpressions stay
/// shared.
#[derive(Debug, PartialEq, Eq)]
enum Choice {
    Shift(usize),
    Select {
        selector: NodeRef,
        cases: Vec<usize>,
        default: Option<usize>,
        priority: bool,
    },
}

/// Recognized shift-amount DAG used to emit and cost constant-shift fusion.
///
/// Leaves hold effective constant shifts; select entries reference earlier
/// entries, preserving shared branches. `root` identifies the amount's result,
/// and `amount_nodes` records the original IR nodes needed for local costing.
struct ChoiceDag {
    choices: Vec<Choice>,
    root: usize,
    amount_nodes: HashSet<NodeRef>,
}

/// Traversal state for recognizing one shift amount without modifying its IR.
///
/// Memoizes recognized IR expressions and interns equivalent choices to
/// preserve sharing, while tracking recognition limits and normalizing
/// oversized shifts to the data width. A successful traversal yields a
/// `ChoiceDag`.
struct Recognizer<'a> {
    f: &'a ir::Fn,
    data_width: usize,
    limits: ConstantShiftChoiceLimits,
    visited: HashSet<NodeRef>,
    memo: HashMap<NodeRef, usize>,
    effective_shifts: HashSet<usize>,
    choices: Vec<Choice>,
    saw_choice: bool,
}

impl<'a> Recognizer<'a> {
    /// Reads a complete amount DAG without mutating the function.
    fn recognize(
        f: &'a ir::Fn,
        amount: NodeRef,
        data_width: usize,
        limits: ConstantShiftChoiceLimits,
    ) -> Option<ChoiceDag> {
        if data_width > limits.max_bit_width {
            return None;
        }
        let mut recognizer = Self {
            f,
            data_width,
            limits,
            visited: HashSet::new(),
            memo: HashMap::new(),
            effective_shifts: HashSet::new(),
            choices: Vec::new(),
            saw_choice: false,
        };
        let root = recognizer.read(amount)?;
        recognizer.saw_choice.then_some(ChoiceDag {
            choices: recognizer.choices,
            root,
            amount_nodes: recognizer.visited,
        })
    }

    fn visit(&mut self, nr: NodeRef) -> Option<()> {
        self.visited.insert(nr);
        (self.visited.len() <= self.limits.max_visited_nodes).then_some(())
    }

    /// Interning both leaves and decisions preserves sharing and equal
    /// branches.
    fn intern(&mut self, choice: Choice) -> Option<usize> {
        if let Some(index) = self.choices.iter().position(|existing| *existing == choice) {
            return Some(index);
        }
        if self.choices.len() >= self.limits.max_emitted_nodes {
            return None;
        }
        let index = self.choices.len();
        self.choices.push(choice);
        Some(index)
    }

    fn shift(&mut self, shift: usize) -> Option<usize> {
        self.effective_shifts.insert(shift);
        if self.effective_shifts.len() > self.limits.max_distinct_shifts {
            return None;
        }
        self.intern(Choice::Shift(shift))
    }

    fn select(
        &mut self,
        selector: NodeRef,
        cases: Vec<usize>,
        default: Option<usize>,
        priority: bool,
    ) -> Option<usize> {
        self.saw_choice = true;
        let first = *cases.first()?;
        if cases.iter().all(|case| *case == first) && default.is_none_or(|default| default == first)
        {
            return Some(first);
        }
        self.intern(Choice::Select {
            selector,
            cases,
            default,
            priority,
        })
    }

    /// Compares at the literal's full width before converting an in-range
    /// value.
    fn effective_shift(&self, bits: &IrBits) -> usize {
        if self.data_width == 0 {
            return 0;
        }
        let required_width = (usize::BITS - self.data_width.leading_zeros()) as usize;
        if bits.get_bit_count() >= required_width {
            let limit = IrBits::make_ubits(bits.get_bit_count(), self.data_width as u64)
                .expect("the data width fits in the comparison literal");
            if bits.uge(&limit) {
                return self.data_width;
            }
        }
        // If the amount has fewer bits than the bound, it is also in range.
        ir_bits_to_usize(bits).expect("a shift less than the data width fits in usize")
    }

    /// Preserves selector values and priority order; no selector correlation is
    /// assumed.
    fn read_select(
        &mut self,
        selector: NodeRef,
        cases: &[NodeRef],
        default: Option<NodeRef>,
        priority: bool,
    ) -> Option<usize> {
        if cases.is_empty()
            || cases.len() > self.limits.max_visited_nodes
            || (priority && default.is_none())
        {
            return None;
        }
        let choices = cases
            .iter()
            .map(|case| self.read(*case))
            .collect::<Option<Vec<_>>>()?;
        let default = match default {
            Some(nr) => Some(self.read(nr)?),
            None => None,
        };
        self.select(selector, choices, default, priority)
    }

    /// Bounds flattening before using the associative/commutative matcher.
    fn bounded_and_operands(&mut self, nr: NodeRef) -> Option<Vec<NodeRef>> {
        let mut stack = vec![nr];
        let mut expanded = 0usize;
        while let Some(nr) = stack.pop() {
            expanded += 1;
            if expanded > self.limits.max_visited_nodes {
                return None;
            }
            self.visit(nr)?;
            if let NodePayload::Nary(NaryOp::And, args) = &self.f.get_node(nr).payload {
                if args.len() > self.limits.max_visited_nodes - expanded {
                    return None;
                }
                stack.extend(args.iter().copied());
            }
        }
        MatchCtx::new(self.f).flattened_nary_operands(nr, NaryOp::And)
    }

    fn mask_predicate(&self, nr: NodeRef, amount_width: usize) -> Option<NodeRef> {
        match self.f.get_node(nr).payload {
            NodePayload::SignExt { arg, new_bit_count }
                if new_bit_count == amount_width
                    && MatchCtx::new(self.f).bits_width(arg) == Some(1) =>
            {
                Some(arg)
            }
            _ => None,
        }
    }

    /// Only exact replicated Boolean masks may gate an otherwise constant
    /// choice.
    fn read_mask(&mut self, nr: NodeRef, amount_width: usize) -> Option<usize> {
        let operands = self.bounded_and_operands(nr)?;
        let mut masks = Vec::new();
        let mut amounts = Vec::new();
        for operand in operands {
            if let Some(predicate) = self.mask_predicate(operand, amount_width) {
                masks.push((operand, predicate));
            } else {
                amounts.push(operand);
            }
        }
        if masks.is_empty() || amounts.len() > 1 {
            return None;
        }
        masks.sort_by_key(|(mask, _)| mask.index);
        let mut body = if let Some(&amount) = amounts.first() {
            self.read(amount)?
        } else {
            let all_ones = IrBits::all_ones(amount_width);
            self.shift(self.effective_shift(&all_ones))?
        };
        let zero = self.shift(0)?;
        for (_, predicate) in masks.into_iter().rev() {
            body = self.select(predicate, vec![zero, body], None, false)?;
        }
        Some(body)
    }

    /// The low bit is a selector; every prefix bit must come from a literal.
    fn read_concat(&mut self, operands: &[NodeRef]) -> Option<usize> {
        if operands.len() < 2 || operands.len() > self.limits.max_visited_nodes {
            return None;
        }
        let (&selector, prefixes) = operands.split_last()?;
        if MatchCtx::new(self.f).bits_width(selector) != Some(1) {
            return None;
        }
        let mut values = vec![false];
        for prefix in prefixes.iter().rev() {
            self.visit(*prefix)?;
            let NodePayload::Literal(value) = &self.f.get_node(*prefix).payload else {
                return None;
            };
            let bits = value.to_bits().ok()?;
            values.extend((0..bits.get_bit_count()).map(|i| bits.get_bit(i).unwrap()));
        }
        let zero = self.shift(self.effective_shift(&IrBits::from_lsb_is_0(&values)))?;
        values[0] = true;
        let one = self.shift(self.effective_shift(&IrBits::from_lsb_is_0(&values)))?;
        self.select(selector, vec![zero, one], None, false)
    }

    /// Rejects any unrecognized leaf, even a variable with a tiny numeric
    /// range.
    fn read(&mut self, nr: NodeRef) -> Option<usize> {
        if let Some(choice) = self.memo.get(&nr) {
            return Some(*choice);
        }
        self.visit(nr)?;
        let width = MatchCtx::new(self.f).bits_width(nr)?;
        if width > self.limits.max_bit_width {
            return None;
        }
        let choice = match self.f.get_node(nr).payload.clone() {
            NodePayload::Literal(value) => self.shift(self.effective_shift(&value.to_bits().ok()?)),
            NodePayload::Unop(Unop::Identity, arg) => self.read(arg),
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => self.read_select(selector, &cases, default, false),
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => self.read_select(selector, &cases, default, true),
            NodePayload::Nary(NaryOp::And, _) => self.read_mask(nr, width),
            NodePayload::Nary(NaryOp::Concat, operands) => self.read_concat(&operands),
            NodePayload::SignExt { .. } => {
                let predicate = self.mask_predicate(nr, width)?;
                let zero = self.shift(0)?;
                let one = self.shift(self.effective_shift(&IrBits::all_ones(width)))?;
                self.select(predicate, vec![zero, one], None, false)
            }
            _ => None,
        }?;
        self.memo.insert(nr, choice);
        Some(choice)
    }
}

impl ChoiceDag {
    /// Bounds all projections and decisions before any IR node is constructed.
    fn emitted_node_bound(&self) -> Option<usize> {
        self.choices.iter().try_fold(0usize, |count, choice| {
            count.checked_add(match choice {
                Choice::Shift(_) => 3, // A slice, zero literal, and concatenation.
                Choice::Select { .. } => 1,
            })
        })
    }

    /// Reconstructs choices once each and reuses every distinct shift
    /// projection.
    fn emit(
        &self,
        f: &mut ir::Fn,
        ty: &Type,
        mut build_case: impl FnMut(&mut ir::Fn, usize) -> NodeRef,
    ) -> NodeRef {
        let mut emitted = Vec::with_capacity(self.choices.len());
        for choice in &self.choices {
            let nr = match choice {
                Choice::Shift(shift) => build_case(f, *shift),
                Choice::Select {
                    selector,
                    cases,
                    default,
                    priority,
                } => {
                    let cases = cases.iter().map(|index| emitted[*index]).collect();
                    let default = default.map(|index| emitted[index]);
                    let payload = if *priority {
                        NodePayload::PrioritySel {
                            selector: *selector,
                            cases,
                            default,
                        }
                    } else {
                        NodePayload::Sel {
                            selector: *selector,
                            cases,
                            default,
                        }
                    };
                    push_node(f, ty.clone(), payload)
                }
            };
            emitted.push(nr);
        }
        emitted[self.root]
    }
}

/// One logical shift, optionally followed by a slice, considered for fusion.
struct ShiftChoiceSite {
    target: NodeRef,
    shift: NodeRef,
    op: Binop,
    data: NodeRef,
    amount: NodeRef,
    slice: Option<(usize, usize)>,
}

impl ShiftChoiceSite {
    /// Recognizes a logical shift or a slice of a logical right shift.
    fn at(f: &ir::Fn, target: NodeRef, slice_phase: bool) -> Option<Self> {
        let (shift, slice) = if slice_phase {
            let NodePayload::BitSlice { arg, start, width } = f.get_node(target).payload else {
                return None;
            };
            (arg, Some((start, width)))
        } else {
            (target, None)
        };
        let NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl), data, amount) =
            f.get_node(shift).payload
        else {
            return None;
        };
        if slice.is_some() && op != Binop::Shrl {
            return None;
        }
        Some(Self {
            target,
            shift,
            op,
            data,
            amount,
            slice,
        })
    }

    /// Appends a speculative replacement without redirecting existing users.
    fn emit(&self, f: &mut ir::Fn, dag: &ChoiceDag) -> NodeRef {
        let data = self.data;
        let ty = f.get_node(self.target).ty.clone();
        dag.emit(f, &ty, |f, shift| match self.slice {
            Some((start, width)) => make_constant_shrl_bit_slice_expr(f, data, shift, start, width),
            None => make_constant_shift_expr(f, self.op, data, shift),
        })
    }
}

/// Owns the recognized recipe while the outer rewrite appends and costs it.
struct ShiftChoiceProposal {
    site: ShiftChoiceSite,
    dag: ChoiceDag,
}

impl ShiftChoiceProposal {
    /// Analyzes one site without changing its function or constructing IR
    /// nodes.
    fn recognize(
        f: &ir::Fn,
        target: NodeRef,
        slice_phase: bool,
        limits: ConstantShiftChoiceLimits,
    ) -> Option<Self> {
        let site = ShiftChoiceSite::at(f, target, slice_phase)?;
        let data_width = MatchCtx::new(f).bits_width(site.data)?;
        let dag = Recognizer::recognize(f, site.amount, data_width, limits)?;
        Some(Self { site, dag })
    }
}

/// Fixes the inputs and retained outputs shared by both alternatives at a site.
struct ShiftChoiceCostBoundary {
    inputs: Vec<NodeRef>,
    retained_outputs: Vec<NodeRef>,
}

impl ShiftChoiceCostBoundary {
    /// Costs only the region reachable from this result and the retained
    /// values.
    fn estimate(
        &self,
        f: &ir::Fn,
        result: NodeRef,
        evaluator: &mut dyn ShiftChoiceCostEvaluator,
    ) -> Result<IrCost, String> {
        let outputs: Vec<_> = std::iter::once(result)
            .chain(self.retained_outputs.iter().copied())
            .collect();
        let nodes = cost_region_order(f, &self.inputs, &outputs);
        evaluator.estimate(&ShiftChoiceCostGraph {
            function: f,
            nodes: &nodes,
            inputs: &self.inputs,
            outputs: &outputs,
        })
    }
}

/// Includes only wiring and inversion around the exact matched amount graph.
fn cost_region_nodes(
    f: &ir::Fn,
    site: &ShiftChoiceSite,
    dag: &ChoiceDag,
    limits: ConstantShiftChoiceLimits,
) -> Option<HashSet<NodeRef>> {
    let mut included = dag.amount_nodes.clone();
    included.extend([site.shift, site.target]);
    if included.len() > limits.max_visited_nodes {
        return None;
    }
    let mut boundary: HashSet<_> = included
        .iter()
        .flat_map(|nr| ir_utils::operands(&f.get_node(*nr).payload))
        .filter(|nr| !included.contains(nr))
        .collect();
    if included.len().checked_add(boundary.len())? > limits.max_visited_nodes
        || !included.iter().chain(&boundary).all(
            |nr| matches!(f.get_node(*nr).ty, Type::Bits(width) if width <= limits.max_bit_width),
        )
    {
        return None;
    }
    let mut pending: Vec<_> = boundary.iter().copied().collect();
    pending.sort_by_key(|nr| std::cmp::Reverse(nr.index));
    while let Some(nr) = pending.pop() {
        let transparent = match &f.get_node(nr).payload {
            NodePayload::Literal(_)
            | NodePayload::Unop(Unop::Identity | Unop::Not, _)
            | NodePayload::BitSlice { .. }
            | NodePayload::ZeroExt { .. }
            | NodePayload::SignExt { .. } => true,
            NodePayload::Nary(NaryOp::Concat, args) => args.len() <= limits.max_visited_nodes,
            _ => false, // Other computations remain independent boundary inputs.
        };
        if !transparent {
            continue;
        }
        let mut args = ir_utils::operands(&f.get_node(nr).payload);
        args.sort_by_key(|arg| std::cmp::Reverse(arg.index));
        args.dedup();
        let new_inputs = args
            .iter()
            .filter(|arg| !included.contains(*arg) && !boundary.contains(*arg))
            .count();
        // Moving this node inside the region must leave room for its inputs.
        // If it cannot fit, its result remains a perfectly usable boundary.
        if new_inputs > limits.max_visited_nodes - included.len() - boundary.len()
            || !args.iter().all(|arg| {
                matches!(f.get_node(*arg).ty, Type::Bits(width) if width <= limits.max_bit_width)
            })
        {
            continue;
        }
        boundary.remove(&nr);
        included.insert(nr);
        for arg in args {
            if !included.contains(&arg) && boundary.insert(arg) {
                pending.push(arg);
            }
        }
    }
    Some(included)
}

/// Records the original cut before speculative nodes can change use counts.
fn cost_boundary(
    f: &ir::Fn,
    site: &ShiftChoiceSite,
    dag: &ChoiceDag,
    users: &ir_utils::Users,
    limits: ConstantShiftChoiceLimits,
) -> Option<ShiftChoiceCostBoundary> {
    let included = cost_region_nodes(f, site, dag, limits)?;
    let mut nodes: Vec<_> = included.iter().copied().collect();
    nodes.sort_by_key(|nr| nr.index);
    let mut inputs: Vec<_> = nodes
        .iter()
        .flat_map(|nr| ir_utils::operands(&f.get_node(*nr).payload))
        .filter(|nr| !included.contains(nr))
        .collect();
    inputs.sort_by_key(|nr| nr.index);
    inputs.dedup();
    if included.len().checked_add(inputs.len())? > limits.max_visited_nodes {
        return None;
    }
    let mut retained_outputs = Vec::new();
    for nr in nodes {
        if nr != site.target
            && (f.ret_node_ref == Some(nr)
                || users.get(&nr)?.iter().any(|user| !included.contains(user)))
        {
            retained_outputs.push(nr);
        }
    }
    // The backend concatenates the outputs; reject an unrepresentable width.
    std::iter::once(site.target)
        .chain(retained_outputs.iter().copied())
        .try_fold(0usize, |width, nr| {
            width.checked_add(f.get_node(nr).ty.bit_count())
        })?;
    Some(ShiftChoiceCostBoundary {
        inputs,
        retained_outputs,
    })
}

/// Walks only a closed region, stopping at its inputs and preserving sharing.
fn cost_region_order(f: &ir::Fn, inputs: &[NodeRef], outputs: &[NodeRef]) -> Vec<NodeRef> {
    let mut seen: HashSet<_> = inputs.iter().copied().collect();
    let mut pending: Vec<_> = outputs.iter().rev().map(|nr| (*nr, false)).collect();
    let mut nodes = Vec::new();
    while let Some((nr, expanded)) = pending.pop() {
        if expanded {
            nodes.push(nr);
        } else if seen.insert(nr) {
            pending.push((nr, true));
            pending.extend(
                ir_utils::operands(&f.get_node(nr).payload)
                    .into_iter()
                    .rev()
                    .map(|operand| (operand, false)),
            );
        }
    }
    nodes
}

/// Builds a bounded constant-choice candidate, leaving `f` unchanged on
/// rejection.
///
/// Callers choose whether the candidate is profitable for their optimization
/// pipeline. Every recognized shift is validated before it is emitted.
/// Exceeding the total emission budget discards the entire candidate, including
/// earlier rewrites.
pub fn constant_shift_choice_candidate(
    f: &ir::Fn,
    limits: ConstantShiftChoiceLimits,
) -> Option<ir::Fn> {
    build_candidate(f, limits, None)
        .expect("unconditional construction does not evaluate costs")
        .map(|candidate| candidate.function)
}

/// Applies a complete bounded candidate and returns the number of rewritten
/// sites.
pub fn rewrite_constant_shift_choices(f: &mut ir::Fn, limits: ConstantShiftChoiceLimits) -> usize {
    let Some(candidate) = build_candidate(f, limits, None)
        .expect("unconditional construction does not evaluate costs")
    else {
        return 0;
    };
    *f = candidate.function;
    candidate.rewrites
}

/// Applies each bounded choice whose local area/delay estimate improves.
///
/// Cost tradeoffs retain that site. Evaluation errors leave the entire input
/// unchanged, including any earlier sites accepted during this call.
pub fn rewrite_constant_shift_choices_with_evaluator(
    f: &mut ir::Fn,
    limits: ConstantShiftChoiceLimits,
    evaluator: &mut dyn ShiftChoiceCostEvaluator,
) -> Result<usize, String> {
    let Some(candidate) = build_candidate(f, limits, Some(evaluator))? else {
        return Ok(0);
    };
    *f = candidate.function;
    Ok(candidate.rewrites)
}

/// Holds a complete speculative function until every selected site succeeds.
struct ConstantShiftChoiceCandidate {
    function: ir::Fn,
    rewrites: usize,
}

/// Builds all rewrites atomically, preserving parameter and observable-effect
/// roots.
fn build_candidate(
    f: &ir::Fn,
    limits: ConstantShiftChoiceLimits,
    mut evaluator: Option<&mut dyn ShiftChoiceCostEvaluator>,
) -> Result<Option<ConstantShiftChoiceCandidate>, String> {
    let mut candidate = f.clone();
    let original_len = candidate.nodes.len();
    let mut emitted_bound = 0usize;
    let mut rewrites = 0;
    for slice_phase in [true, false] {
        let mut users = ir_utils::compute_users(&candidate);
        for index in 0..original_len {
            let target = NodeRef { index };
            if candidate.ret_node_ref != Some(target)
                && users
                    .get(&target)
                    .expect("candidate node has a users entry")
                    .is_empty()
            {
                continue;
            }
            let Some(proposal) =
                ShiftChoiceProposal::recognize(&candidate, target, slice_phase, limits)
            else {
                continue;
            };
            let Some(next_emitted_bound) = proposal
                .dag
                .emitted_node_bound()
                .and_then(|n| emitted_bound.checked_add(n))
            else {
                return Ok(None);
            };
            if next_emitted_bound > limits.max_emitted_nodes {
                return Ok(None);
            }
            let boundary = if evaluator.is_some() {
                let Some(boundary) =
                    cost_boundary(&candidate, &proposal.site, &proposal.dag, &users, limits)
                else {
                    continue;
                };
                Some(boundary)
            } else {
                None
            };
            // Emission only appends nodes. Both alternatives can then borrow
            // this function, and rejection can discard the appended suffix.
            let first_new_node = candidate.nodes.len();
            let replacement = proposal.site.emit(&mut candidate, &proposal.dag);
            if let Some(evaluator) = evaluator.as_deref_mut() {
                let boundary = boundary.expect("costed proposals have a boundary");
                let incumbent = boundary.estimate(&candidate, target, evaluator)?;
                let projected = boundary.estimate(&candidate, replacement, evaluator)?;
                if !projected.is_pareto_improvement_on(incumbent) {
                    candidate.nodes.truncate(first_new_node);
                    continue;
                }
            }
            emitted_bound = next_emitted_bound;
            ir_utils::replace_node_with_ref(&mut candidate, target, replacement)
                .expect("constant-shift choices preserve the target type");
            rewrites += 1;
            users = ir_utils::compute_users(&candidate);
        }
        for dead in get_dead_nodes(&candidate) {
            if dead.index == 0 || candidate.params.contains(&dead) {
                continue;
            }
            let node = candidate.get_node_mut(dead);
            node.payload = NodePayload::Nil;
            node.ty = Type::nil();
        }
    }
    if rewrites == 0 {
        return Ok(None);
    }
    ir_utils::compact_and_toposort_in_place(&mut candidate)
        .expect("constant-shift choices remain acyclic");
    Ok(Some(ConstantShiftChoiceCandidate {
        function: candidate,
        rewrites,
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::IrValue;
    use crate::ir_cost::IrCost;
    use crate::ir_eval::eval_fn;
    use crate::ir_parser::Parser;
    use crate::ir_verify::verify_function;

    /// Records isolated test functions while controlling profitability answers.
    struct ScriptedEvaluator {
        answers: VecDeque<Result<IrCost, String>>,
        inputs: Vec<ir::Fn>,
        containing_node_counts: Vec<usize>,
    }

    impl ShiftChoiceCostEvaluator for ScriptedEvaluator {
        fn estimate(&mut self, graph: &ShiftChoiceCostGraph<'_>) -> Result<IrCost, String> {
            self.containing_node_counts
                .push(graph.function().nodes.len());
            self.inputs.push(materialize_cost_graph(graph));
            self.answers
                .pop_front()
                .expect("unexpected cost evaluation")
        }
    }

    fn scripted(areas: &[usize]) -> ScriptedEvaluator {
        ScriptedEvaluator {
            answers: areas
                .iter()
                .map(|area| {
                    Ok(IrCost {
                        area: *area,
                        delay: 5.0,
                    })
                })
                .collect(),
            inputs: Vec::new(),
            containing_node_counts: Vec::new(),
        }
    }

    /// Materializes a borrowed region only for structural and equivalence
    /// checks.
    fn materialize_cost_graph(graph: &ShiftChoiceCostGraph<'_>) -> ir::Fn {
        let source = graph.function();
        let mut function = ir::Fn {
            graph: ir::NodeGraph::new("cost_region"),
            params: Vec::new(),
            ret_ty: Type::Bits(0),
            ret_node_ref: None,
        };
        let mut mapping = HashMap::new();
        for (index, &input) in graph.inputs().iter().enumerate() {
            let param = push_node(
                &mut function,
                source.get_node(input).ty.clone(),
                NodePayload::Param,
            );
            function.get_node_mut(param).name = Some(format!("input_{index}"));
            function.params.push(param);
            mapping.insert(input, param);
        }
        for &nr in graph.nodes() {
            let node = source.get_node(nr);
            let payload = ir_utils::remap_payload_with(&node.payload, |(_, arg)| mapping[&arg]);
            let mapped = push_node(&mut function, node.ty.clone(), payload);
            assert!(mapping.insert(nr, mapped).is_none());
        }
        let outputs: Vec<_> = graph.outputs().iter().map(|nr| mapping[nr]).collect();
        function.ret_ty = Type::Bits(
            outputs
                .iter()
                .map(|nr| function.get_node(*nr).ty.bit_count())
                .sum(),
        );
        function.ret_node_ref = Some(if outputs.len() == 1 {
            outputs[0]
        } else {
            let ty = function.ret_ty.clone();
            push_node(
                &mut function,
                ty,
                NodePayload::Nary(NaryOp::Concat, outputs),
            )
        });
        verify_function(&function).unwrap();
        function
    }

    fn masked_choice(width: usize, op: &str) -> ir::Fn {
        Parser::new(&format!(
            r#"package constant_choices

top fn main(x: bits[{width}] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4) -> bits[{width}] {{
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  ret result: bits[{width}] = {op}(x, amount, id=11)
}}
"#,
        ))
        .parse_and_validate_package()
        .unwrap()
        .get_top_fn()
        .unwrap()
        .clone()
    }

    #[test]
    fn injected_cost_model_controls_candidate_acceptance() {
        let mut function = masked_choice(16, "shll");
        let limits = ConstantShiftChoiceLimits::default();
        let candidate = constant_shift_choice_candidate(&function, limits).unwrap();
        let mut evaluator = scripted(&[10, 9]);
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(&mut function, limits, &mut evaluator),
            Ok(1),
        );
        assert_eq!(evaluator.inputs.len(), 2);
        assert_eq!(
            evaluator.inputs[0].get_type(),
            evaluator.inputs[1].get_type()
        );
        assert_eq!(function.to_string(), candidate.to_string());
        assert!(evaluator.answers.is_empty());
    }

    #[test]
    fn injected_cost_failure_preserves_input() {
        for failed_call in [0, 1] {
            let mut function = masked_choice(4, "shrl");
            let original = function.to_string();
            let mut evaluator = scripted(&[10][..failed_call]);
            evaluator
                .answers
                .push_back(Err("cost backend failed".to_string()));
            assert_eq!(
                rewrite_constant_shift_choices_with_evaluator(
                    &mut function,
                    ConstantShiftChoiceLimits::default(),
                    &mut evaluator,
                ),
                Err("cost backend failed".to_string()),
            );
            assert_eq!(function.to_string(), original);
            assert_eq!(evaluator.inputs.len(), failed_call + 1);
            assert!(evaluator.answers.is_empty());
        }
    }

    #[test]
    fn unrelated_live_arithmetic_does_not_enter_cost_graphs() {
        let mut isolated = masked_choice(4, "shll");
        let mut extended = isolated.clone();
        let wide_ty = Type::Bits(256);
        let unrelated = push_node(&mut extended, wide_ty.clone(), NodePayload::Param);
        extended.get_node_mut(unrelated).name = Some("unrelated".to_string());
        extended.params.push(unrelated);
        let mut sum = unrelated;
        for _ in 0..64 {
            sum = push_node(
                &mut extended,
                wide_ty.clone(),
                NodePayload::Binop(Binop::Add, sum, unrelated),
            );
        }
        let original_result = extended.ret_node_ref.unwrap();
        let ret_ty = Type::Tuple(vec![Box::new(extended.ret_ty.clone()), Box::new(wide_ty)]);
        extended.ret_node_ref = Some(push_node(
            &mut extended,
            ret_ty.clone(),
            NodePayload::Tuple(vec![original_result, sum]),
        ));
        extended.ret_ty = ret_ty;
        verify_function(&extended).unwrap();
        let mut isolated_costs = scripted(&[10, 9]);
        let mut extended_costs = scripted(&[10, 9]);
        let limits = ConstantShiftChoiceLimits::default();
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut isolated,
                limits,
                &mut isolated_costs
            ),
            Ok(1)
        );
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut extended,
                limits,
                &mut extended_costs
            ),
            Ok(1)
        );
        assert_eq!(
            isolated_costs
                .inputs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            extended_costs
                .inputs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        );
        // The evaluator borrows the containing function; only its explicit
        // region is copied into the test functions above.
        for (isolated_count, extended_count) in isolated_costs
            .containing_node_counts
            .iter()
            .zip(&extended_costs.containing_node_counts)
        {
            assert!(extended_count >= &(isolated_count + 64));
        }
    }

    #[test]
    fn long_wiring_context_stops_at_a_boundary_without_discarding_the_choice() {
        let limits = ConstantShiftChoiceLimits::default();
        for chain_length in [54, 200] {
            let mut function = masked_choice(4, "shll");
            let shifted = function.ret_node_ref.unwrap();
            let NodePayload::Binop(_, mut data, amount) = function.get_node(shifted).payload else {
                unreachable!("masked_choice returns a shift");
            };
            for _ in 0..chain_length {
                data = push_node(
                    &mut function,
                    Type::Bits(4),
                    NodePayload::Unop(Unop::Identity, data),
                );
            }
            function.get_node_mut(shifted).payload = NodePayload::Binop(Binop::Shll, data, amount);
            let mut evaluator = scripted(&[10, 9]);
            assert_eq!(
                rewrite_constant_shift_choices_with_evaluator(
                    &mut function,
                    limits,
                    &mut evaluator
                ),
                Ok(1)
            );
            assert_eq!(evaluator.inputs.len(), 2);
            // Allow the reserved Nil node and the bounded replacement
            // expansion.
            assert!(evaluator.inputs[0].nodes.len() <= limits.max_visited_nodes + 1);
            assert!(
                evaluator.inputs[1].nodes.len()
                    <= limits.max_visited_nodes + limits.max_emitted_nodes + 1
            );
        }
    }

    fn two_shifts() -> ir::Fn {
        let mut function = masked_choice(4, "shll");
        let left = function.ret_node_ref.unwrap();
        let NodePayload::Binop(_, data, amount) = function.get_node(left).payload else {
            unreachable!("masked_choice returns a shift");
        };
        let right = push_node(
            &mut function,
            Type::Bits(4),
            NodePayload::Binop(Binop::Shrl, data, amount),
        );
        let ret_ty = Type::Tuple(vec![Box::new(Type::Bits(4)), Box::new(Type::Bits(4))]);
        function.ret_node_ref = Some(push_node(
            &mut function,
            ret_ty.clone(),
            NodePayload::Tuple(vec![left, right]),
        ));
        function.ret_ty = ret_ty;
        function
    }

    #[test]
    fn choices_are_costed_independently_and_errors_roll_back_all_sites() {
        let original = two_shifts();
        let limits = ConstantShiftChoiceLimits::default();
        let mut function = original.clone();
        let mut evaluator = scripted(&[10, 9, 10, 11]);
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(&mut function, limits, &mut evaluator),
            Ok(1)
        );
        assert_eq!(evaluator.inputs.len(), 4);
        assert!(
            !function
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Binop(Binop::Shll, ..)))
        );
        assert!(
            function
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Binop(Binop::Shrl, ..)))
        );
        // The first choice must retain the amount used by the second shift.
        // After that replacement the second shift is its only remaining user.
        assert_eq!(evaluator.inputs[0].ret_ty, Type::Bits(6));
        assert_eq!(evaluator.inputs[2].ret_ty, Type::Bits(4));

        let mut function = original.clone();
        let mut evaluator = scripted(&[10, 9]);
        evaluator
            .answers
            .push_back(Err("second site failed".to_string()));
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(&mut function, limits, &mut evaluator),
            Err("second site failed".to_string())
        );
        assert_eq!(function.to_string(), original.to_string());
    }

    #[test]
    fn rejected_speculation_is_removed_before_the_next_site() {
        let mut function = two_shifts();
        let mut evaluator = scripted(&[10, 11, 10, 9]);
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut function,
                ConstantShiftChoiceLimits::default(),
                &mut evaluator,
            ),
            Ok(1),
        );
        assert_eq!(evaluator.containing_node_counts.len(), 4);
        // Left and right projections append the same number of nodes. The
        // rejected left-shift proposal must not remain in the second attempt.
        assert_eq!(
            evaluator.containing_node_counts[0],
            evaluator.containing_node_counts[2],
        );
        assert!(
            function
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Binop(Binop::Shll, ..)))
        );
        assert!(
            !function
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Binop(Binop::Shrl, ..)))
        );
        verify_function(&function).unwrap();
    }

    #[test]
    fn externally_used_amount_is_an_output_of_both_cost_graphs() {
        let mut function = masked_choice(4, "shll");
        let shifted = function.ret_node_ref.unwrap();
        let NodePayload::Binop(_, _, amount) = function.get_node(shifted).payload else {
            unreachable!("masked_choice returns a shift");
        };
        let ret_ty = Type::Tuple(vec![Box::new(Type::Bits(4)), Box::new(Type::Bits(2))]);
        function.ret_node_ref = Some(push_node(
            &mut function,
            ret_ty.clone(),
            NodePayload::Tuple(vec![shifted, amount]),
        ));
        function.ret_ty = ret_ty;
        let mut evaluator = scripted(&[10, 9]);
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut function,
                ConstantShiftChoiceLimits::default(),
                &mut evaluator
            ),
            Ok(1)
        );
        for graph in &evaluator.inputs {
            assert_eq!(graph.ret_ty, Type::Bits(6));
        }
        for x in 0..16 {
            for controls in 0..8 {
                let en = controls & 1;
                let p = (controls >> 1) & 1;
                let q = (controls >> 2) & 1;
                let args = [
                    IrValue::make_ubits(4, x).unwrap(),
                    IrValue::make_ubits(1, en).unwrap(),
                    IrValue::make_ubits(1, p).unwrap(),
                    IrValue::make_ubits(1, q).unwrap(),
                ];
                let before = eval_fn(&evaluator.inputs[0], &args);
                let after = eval_fn(&evaluator.inputs[1], &args);
                assert_eq!(before, after);
            }
        }
    }

    #[test]
    fn unrecognized_shift_amount_never_calls_evaluator() {
        let mut function = Parser::new(
            r#"package variable_shift
top fn main(x: bits[4] id=1, amount: bits[2] id=2) -> bits[4] {
  ret out: bits[4] = shll(x, amount, id=3)
}"#,
        )
        .parse_and_validate_package()
        .unwrap()
        .get_top_fn()
        .unwrap()
        .clone();
        let mut evaluator = scripted(&[]);
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut function,
                ConstantShiftChoiceLimits::default(),
                &mut evaluator
            ),
            Ok(0)
        );
        assert!(evaluator.inputs.is_empty());
    }

    #[test]
    fn type_only_wide_inputs_are_rejected_before_materializing_bits() {
        for (data_width, amount_width) in [(4, 1_000_000_000), (1_000_000_000, 32)] {
            let mut function = Parser::new(&format!(
                r#"package wide_choices

top fn main(x: bits[{data_width}] id=1, p: bits[1] id=2) -> bits[{data_width}] {{
  amount: bits[{amount_width}] = sign_ext(p, new_bit_count={amount_width}, id=3)
  ret result: bits[{data_width}] = shll(x, amount, id=4)
}}
"#,
            ))
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone();
            let original = function.to_string();
            let limits = ConstantShiftChoiceLimits::default();
            assert!(constant_shift_choice_candidate(&function, limits).is_none());
            assert_eq!(rewrite_constant_shift_choices(&mut function, limits), 0);
            let mut evaluator = scripted(&[]);
            assert_eq!(
                rewrite_constant_shift_choices_with_evaluator(
                    &mut function,
                    limits,
                    &mut evaluator
                ),
                Ok(0)
            );
            assert!(evaluator.inputs.is_empty());
            assert_eq!(function.to_string(), original);
        }
    }

    #[test]
    fn rewrite_preserves_effects_and_shift_used_only_by_trace() {
        let package = Parser::new(
            r#"package effects

top fn main(x: bits[4] id=1, selector: bits[1] id=2, enabled: bits[1] id=3) -> bits[1] {
  one: bits[2] = literal(value=1, id=4)
  two: bits[2] = literal(value=2, id=5)
  amount: bits[2] = sel(selector, cases=[one, two], id=6)
  shifted: bits[4] = shll(x, amount, id=7)
  tok: token = after_all(id=8)
  traced: token = trace(tok, enabled, format="shifted={}", data_operands=[shifted], verbosity=0, id=9)
  covered: () = cover(enabled, label="enabled", id=10)
  checked: token = assert(traced, enabled, message="disabled", label="check", id=11)
  ret out: bits[1] = literal(value=0, id=12)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let original = package.get_top_fn().unwrap();
        let mut rewritten = original.clone();
        assert_eq!(
            rewrite_constant_shift_choices(&mut rewritten, ConstantShiftChoiceLimits::default()),
            1
        );
        verify_function(&rewritten).unwrap();

        // Equality includes trace messages, cover counts, and assertion
        // failures, even though none of their nodes contribute to the
        // returned value.
        for x in 0..16 {
            for selector in 0..2 {
                for enabled in 0..2 {
                    let args = [
                        IrValue::make_ubits(4, x).unwrap(),
                        IrValue::make_ubits(1, selector).unwrap(),
                        IrValue::make_ubits(1, enabled).unwrap(),
                    ];
                    assert_eq!(eval_fn(&rewritten, &args), eval_fn(original, &args));
                }
            }
        }
    }
}
