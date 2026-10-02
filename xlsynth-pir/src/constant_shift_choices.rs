// SPDX-License-Identifier: Apache-2.0

//! Bounded recognition and construction of constant logical-shift choices.

use std::collections::{HashMap, HashSet};

use crate::dce::get_dead_nodes;
use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use crate::ir_match::MatchCtx;
use crate::ir_utils;
use crate::ir_value_utils::ir_bits_to_usize;
use crate::ir_verify::verify_function;
use crate::local_cost::estimate_local_cost;
use crate::{IrBits, IrValue};

/// Bounds for recognizing and emitting one function's constant-shift choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstantShiftChoiceLimits {
    /// Distinct effective amounts per shift, with all overshifts counted once.
    pub max_distinct_shifts: usize,
    /// Inspected amount nodes per shift, also bounding flattened AND operands.
    pub max_visited_nodes: usize,
    /// Conservative upper bound on newly emitted IR nodes for the function.
    pub max_emitted_nodes: usize,
}

impl Default for ConstantShiftChoiceLimits {
    fn default() -> Self {
        Self {
            max_distinct_shifts: 4,
            max_visited_nodes: 64,
            max_emitted_nodes: 128,
        }
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

struct ChoiceDag {
    choices: Vec<Choice>,
    root: usize,
}

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

/// Appends a helper node with a fresh text ID and no source location.
fn push_node(f: &mut ir::Fn, ty: Type, payload: NodePayload) -> NodeRef {
    let text_id = f
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let index = f.nodes.len();
    f.nodes.push(ir::Node {
        text_id,
        name: None,
        ty,
        payload,
        pos: None,
    });
    NodeRef { index }
}

/// Reuses an existing literal before appending zero padding for a projection.
fn get_or_insert_ubits_literal(f: &mut ir::Fn, bit_count: usize, value: u64) -> NodeRef {
    for (index, node) in f.nodes.iter().enumerate() {
        if let NodePayload::Literal(literal) = &node.payload {
            if node.ty.bit_count() == bit_count && literal.bits_equals_u64_value(value) {
                return NodeRef { index };
            }
        }
    }
    let literal = IrValue::make_ubits(bit_count, value).expect("ubits literal");
    push_node(f, Type::Bits(bit_count), NodePayload::Literal(literal))
}

/// Builds a logical constant shift from an unchanged data operand and wiring.
fn make_constant_shift_expr(f: &mut ir::Fn, op: Binop, arg: NodeRef, shift: usize) -> NodeRef {
    let arg_width = f.get_node(arg).ty.bit_count();
    if shift == 0 {
        return arg;
    }
    if arg_width == 0 || shift >= arg_width {
        return get_or_insert_ubits_literal(f, arg_width, 0);
    }

    let shifted_width = arg_width - shift;
    let shifted_slice = push_node(
        f,
        Type::Bits(shifted_width),
        NodePayload::BitSlice {
            arg,
            start: if op == Binop::Shrl { shift } else { 0 },
            width: shifted_width,
        },
    );
    let zero_padding = get_or_insert_ubits_literal(f, shift, 0);
    let operands = match op {
        Binop::Shrl => vec![zero_padding, shifted_slice],
        Binop::Shll => vec![shifted_slice, zero_padding],
        _ => unreachable!("constant-shift projection requires a logical shift"),
    };
    push_node(
        f,
        Type::Bits(arg_width),
        NodePayload::Nary(NaryOp::Concat, operands),
    )
}

/// Builds a slice of a logical right shift using only source bits and zeros.
fn make_constant_shrl_bit_slice_expr(
    f: &mut ir::Fn,
    arg: NodeRef,
    shift: usize,
    start: usize,
    width: usize,
) -> NodeRef {
    let arg_width = f.get_node(arg).ty.bit_count();
    if width == 0 {
        return get_or_insert_ubits_literal(f, 0, 0);
    }
    let Some(source_start) = start.checked_add(shift) else {
        return get_or_insert_ubits_literal(f, width, 0);
    };
    if source_start >= arg_width {
        return get_or_insert_ubits_literal(f, width, 0);
    }

    let valid_width = std::cmp::min(width, arg_width - source_start);
    if valid_width == 0 {
        return get_or_insert_ubits_literal(f, width, 0);
    }
    if valid_width == width {
        if source_start == 0 && width == arg_width {
            return arg;
        }
        return push_node(
            f,
            Type::Bits(width),
            NodePayload::BitSlice {
                arg,
                start: source_start,
                width,
            },
        );
    }

    let payload_bits = push_node(
        f,
        Type::Bits(valid_width),
        NodePayload::BitSlice {
            arg,
            start: source_start,
            width: valid_width,
        },
    );
    let zero_prefix = get_or_insert_ubits_literal(f, width - valid_width, 0);
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Nary(NaryOp::Concat, vec![zero_prefix, payload_bits]),
    )
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
    build_candidate(f, limits).map(|candidate| candidate.function)
}

/// Applies a complete bounded candidate and returns the number of rewritten
/// sites.
pub fn rewrite_constant_shift_choices(f: &mut ir::Fn, limits: ConstantShiftChoiceLimits) -> usize {
    let Some(candidate) = build_candidate(f, limits) else {
        return 0;
    };
    *f = candidate.function;
    candidate.rewrites
}

/// Applies a candidate only when the bounded PIR-local area/depth heuristic
/// improves.
///
/// Unknown estimates and cost tradeoffs retain the input. This estimate does
/// not guarantee an improvement after a backend maps and optimizes the IR.
pub fn rewrite_profitable_constant_shift_choices(
    f: &mut ir::Fn,
    limits: ConstantShiftChoiceLimits,
) -> usize {
    let Some(candidate) = build_candidate(f, limits) else {
        return 0;
    };
    let (Some(incumbent_cost), Some(candidate_cost)) = (
        estimate_local_cost(f),
        estimate_local_cost(&candidate.function),
    ) else {
        return 0;
    };
    if !candidate_cost.is_pareto_improvement_on(incumbent_cost) {
        return 0;
    }
    *f = candidate.function;
    candidate.rewrites
}

struct ConstantShiftChoiceCandidate {
    function: ir::Fn,
    rewrites: usize,
}

/// Builds all rewrites atomically, preserving parameter and observable-effect
/// roots.
fn build_candidate(
    f: &ir::Fn,
    limits: ConstantShiftChoiceLimits,
) -> Option<ConstantShiftChoiceCandidate> {
    let mut candidate = f.clone();
    let original_len = candidate.nodes.len();
    let mut emitted_bound = 0usize;
    let mut rewrites = 0;
    for slice_phase in [true, false] {
        let users = ir_utils::compute_users(&candidate);
        for index in 0..original_len {
            let target = NodeRef { index };
            if candidate.ret_node_ref != Some(target) && users.get(&target)?.is_empty() {
                continue;
            }
            let payload = candidate.nodes[index].payload.clone();
            let (op, data, amount, slice) = match (slice_phase, payload) {
                (true, NodePayload::BitSlice { arg, start, width }) => {
                    let NodePayload::Binop(Binop::Shrl, data, amount) =
                        candidate.get_node(arg).payload
                    else {
                        continue;
                    };
                    (Binop::Shrl, data, amount, Some((start, width)))
                }
                (false, NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl), data, amount)) => {
                    (op, data, amount, None)
                }
                _ => continue,
            };
            let data_width = MatchCtx::new(&candidate).bits_width(data)?;
            let Some(dag) = Recognizer::recognize(&candidate, amount, data_width, limits) else {
                continue;
            };
            emitted_bound = emitted_bound.checked_add(dag.emitted_node_bound()?)?;
            if emitted_bound > limits.max_emitted_nodes {
                return None;
            }
            let ty = candidate.get_node(target).ty.clone();
            let replacement = dag.emit(&mut candidate, &ty, |f, shift| match slice {
                Some((start, width)) => {
                    make_constant_shrl_bit_slice_expr(f, data, shift, start, width)
                }
                None => make_constant_shift_expr(f, op, data, shift),
            });
            ir_utils::replace_node_with_ref(&mut candidate, target, replacement)
                .expect("constant-shift choices preserve the target type");
            rewrites += 1;
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
        return None;
    }
    ir_utils::compact_and_toposort_in_place(&mut candidate)
        .expect("constant-shift choices remain acyclic");
    verify_function(&candidate).expect("constant-shift choices preserve well-formed IR");
    Some(ConstantShiftChoiceCandidate {
        function: candidate,
        rewrites,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_eval::eval_fn;
    use crate::ir_parser::Parser;

    fn masked_choice(width: usize, op: &str, return_amount: bool) -> ir::Fn {
        let return_type = if return_amount {
            format!("(bits[{width}], bits[2])")
        } else {
            format!("bits[{width}]")
        };
        let result = if return_amount {
            "tuple(shifted, amount, id=12)"
        } else {
            "identity(shifted, id=12)"
        };
        Parser::new(&format!(
            r#"package constant_choices

top fn main(x: bits[{width}] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4) -> {return_type} {{
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  shifted: bits[{width}] = {op}(x, amount, id=11)
  ret result: {return_type} = {result}
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
    fn profitability_rejects_area_tradeoffs_and_preserves_shared_amounts() {
        for width in [4, 8, 16] {
            for op in ["shll", "shrl"] {
                for return_amount in [false, true] {
                    let mut function = masked_choice(width, op, return_amount);
                    let original = function.to_string();
                    let limits = ConstantShiftChoiceLimits::default();
                    assert!(constant_shift_choice_candidate(&function, limits).is_some());
                    let expected = usize::from(width == 4 && !return_amount);
                    assert_eq!(
                        rewrite_profitable_constant_shift_choices(&mut function, limits),
                        expected,
                        "width={width}, op={op}, return_amount={return_amount}"
                    );
                    if expected == 0 {
                        assert_eq!(function.to_string(), original);
                    }
                    assert_eq!(
                        rewrite_profitable_constant_shift_choices(&mut function, limits),
                        0
                    );
                }
            }
        }
    }

    #[test]
    fn profitability_keeps_input_on_equal_cost_or_exhausted_estimate() {
        for width in [1, 16_384] {
            let mut function = Parser::new(&format!(
                r#"package guarded_choices

top fn main(x: bits[{width}] id=1, p: bits[1] id=2) -> bits[{width}] {{
  zero: bits[1] = literal(value=0, id=3)
  one: bits[1] = literal(value=1, id=4)
  amount: bits[1] = sel(p, cases=[zero, one], id=5)
  ret result: bits[{width}] = shll(x, amount, id=6)
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
            let candidate = constant_shift_choice_candidate(&function, limits)
                .expect("eligibility must not depend on the cost estimate");
            if width == 1 {
                let cost = estimate_local_cost(&function).expect("small estimate completes");
                assert_eq!(estimate_local_cost(&candidate), Some(cost));
            } else {
                assert_eq!(estimate_local_cost(&function), None);
            }
            assert_eq!(
                rewrite_profitable_constant_shift_choices(&mut function, limits),
                0
            );
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
        for rewrite in [
            rewrite_constant_shift_choices,
            rewrite_profitable_constant_shift_choices,
        ] {
            let mut rewritten = original.clone();
            assert_eq!(
                rewrite(&mut rewritten, ConstantShiftChoiceLimits::default()),
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
}
