// SPDX-License-Identifier: Apache-2.0

//! Nonzero tests of bounded logical shifts, without materializing shifted
//! words.

use super::helpers::{NodeAppender, match_zero_test, next_text_id};
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_match::MatchCtx;
use xlsynth_pir::ir_utils;
use xlsynth_pir::math::ceil_log2;

/// A contiguous input region sharing the same inclusive shift-amount bounds.
struct Term {
    start: usize,
    width: usize,
    low: usize,
    high: usize,
}

/// An immutable scalar replacement; unrecognized leaves retain their reduction.
enum Plan {
    Reduce {
        value: NodeRef,
        start: usize,
        width: usize,
    },
    Shift {
        data: NodeRef,
        amount: NodeRef,
        low_width: usize,
        max: usize,
        terms: Vec<Term>,
    },
    Select {
        selector: NodeRef,
        cases: [Box<Plan>; 2],
    },
}

/// Bounds both recursive select/slice traversal and emitted boundary
/// predicates.
#[derive(Default)]
struct Budget {
    visits: usize,
    terms: usize,
    shifts: usize,
}

/// Finds one-hot producers through wiring and polarity wrappers. Exhausting the
/// shared admission budget conservatively declines the new wide-amount match.
fn has_one_hot_source(f: &ir::Fn, value: NodeRef, budget: &mut usize) -> bool {
    if *budget == 0 {
        return true;
    }
    *budget -= 1;
    match &f.get_node(value).payload {
        NodePayload::OneHot { .. } => true,
        NodePayload::Unop(Unop::Identity | Unop::Not, arg)
        | NodePayload::BitSlice { arg, .. }
        | NodePayload::ZeroExt { arg, .. }
        | NodePayload::SignExt { arg, .. } => has_one_hot_source(f, *arg, budget),
        NodePayload::Nary(NaryOp::Concat, args) => {
            args.iter().any(|&arg| has_one_hot_source(f, arg, budget))
        }
        _ => false,
    }
}

/// Conservatively detects constant wiring or literal-fed bitwise controls in
/// a low amount bit. Arithmetic is opaque; this is a bounded admission check,
/// not a general known-bits analysis or a change to the narrow-path estimate.
fn amount_bit_has_constant(f: &ir::Fn, value: NodeRef, bit: usize, budget: &mut usize) -> bool {
    if *budget == 0 {
        return true;
    }
    *budget -= 1;
    match &f.get_node(value).payload {
        NodePayload::Literal(_) => true,
        NodePayload::Unop(Unop::Identity | Unop::Not, arg) => {
            amount_bit_has_constant(f, *arg, bit, budget)
        }
        NodePayload::BitSlice { arg, start, .. } => {
            amount_bit_has_constant(f, *arg, start + bit, budget)
        }
        NodePayload::ZeroExt { arg, .. } => {
            bit >= f.get_node(*arg).ty.bit_count() || amount_bit_has_constant(f, *arg, bit, budget)
        }
        NodePayload::SignExt { arg, .. } => {
            let width = f.get_node(*arg).ty.bit_count();
            width == 0 || amount_bit_has_constant(f, *arg, bit.min(width - 1), budget)
        }
        NodePayload::Nary(NaryOp::Concat, args) => {
            let mut offset = bit;
            for &arg in args.iter().rev() {
                if *budget == 0 {
                    return true;
                }
                *budget -= 1;
                let width = f.get_node(arg).ty.bit_count();
                if offset < width {
                    return amount_bit_has_constant(f, arg, offset, budget);
                }
                offset -= width;
            }
            true
        }
        NodePayload::Nary(
            NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor,
            args,
        ) => args
            .iter()
            .any(|&arg| amount_bit_has_constant(f, arg, bit, budget)),
        _ => false,
    }
}

/// Plans a survival predicate using a bounded structural gate estimate.
///
/// For a left shift, input bit i contributes iff start <= i + amount < end;
/// for a right shift, iff start <= i - amount < end. Intersecting that interval
/// with the amount's unsigned range handles zero fill and overshifts exactly.
/// Common unconditional bits become one reduction. At most 64 groups, 32 plan
/// nodes, 256 source bits and five effective amount bits are considered per
/// predicate. Wider amounts use a high-bits-zero guard for whole-word tests
/// when the source width needs at most five low bits. The intervals still
/// reject low-bit overshifts for non-power-of-two widths.
/// The estimate compares boundary comparisons/reductions with the demanded
/// barrel-shifter muxes; it never constructs or scores competing AIGs.
fn shift_plan(
    f: &ir::Fn,
    op: Binop,
    data: NodeRef,
    amount: NodeRef,
    start: usize,
    width: usize,
    budget: &mut Budget,
) -> Option<Plan> {
    let ctx = MatchCtx::new(f);
    let n = ctx.bits_width(data)?;
    let aw = ctx.bits_width(amount)?;
    let low_width = if aw > 5 { ceil_log2(n) } else { aw };
    // Keep newly supported matches within the structural estimate's scope.
    // All checks share a small budget and leave the existing narrow path alone.
    if aw > 5 {
        let mut admission_budget = 32;
        if start != 0
            || width != n
            || has_one_hot_source(f, data, &mut admission_budget)
            || (0..low_width)
                .any(|bit| amount_bit_has_constant(f, amount, bit, &mut admission_budget))
        {
            return None;
        }
    }
    if n > 256
        || !(1..=5).contains(&low_width)
        || matches!(f.get_node(data).payload, NodePayload::Literal(_))
        || matches!(f.get_node(amount).payload, NodePayload::Literal(_))
    {
        return None;
    }
    let max = (1usize << low_width) - 1;
    let end = start.checked_add(width)?;
    let mut terms: Vec<Term> = Vec::new();
    for i in 0..n {
        let (low, high) = if op == Binop::Shll {
            if i >= end {
                continue;
            }
            (start.saturating_sub(i), (end - 1 - i).min(max))
        } else {
            if i < start {
                continue;
            }
            (i.saturating_sub(end - 1), (i - start).min(max))
        };
        if low > high {
            continue;
        }
        if let Some(last) = terms.last_mut()
            && last.start + last.width == i
            && last.low == low
            && last.high == high
        {
            last.width += 1;
        } else {
            terms.push(Term {
                start: i,
                width: 1,
                low,
                high,
            });
        }
    }
    if terms.is_empty() || budget.terms + terms.len() > 64 {
        return None;
    }
    let replacement: usize = terms
        .iter()
        .map(|term| {
            term.width - 1
                + usize::from(term.low > 0) * 2 * low_width
                + usize::from(term.high < max) * 2 * low_width
                + usize::from(term.low > 0 || term.high < max)
        })
        .sum::<usize>()
        + terms.len()
        - 1
        + usize::from(low_width < aw);
    // Both forms reduce the high amount bits to detect overshifts. Cancel that
    // common cost, charging the replacement's final guard AND without taking
    // credit for the original shifter's per-output overflow masking.
    let original = 3 * width * low_width.min(ceil_log2(n)) + width - 1;
    if replacement >= original {
        return None;
    }
    budget.terms += terms.len();
    budget.shifts += 1;
    Some(Plan::Shift {
        data,
        amount,
        low_width,
        max,
        terms,
    })
}

/// Walks only exclusive data paths, distributing the scalar test over binary
/// selects. Shared data nodes remain leaves so their numeric users are
/// retained.
fn plan(
    f: &ir::Fn,
    uses: &[usize],
    value: NodeRef,
    start: usize,
    width: usize,
    budget: &mut Budget,
) -> Option<Plan> {
    budget.visits += 1;
    if budget.visits > 32 || width == 0 {
        return None;
    }
    let fallback = || Plan::Reduce {
        value,
        start,
        width,
    };
    if uses[value.index] != 1 {
        return Some(fallback());
    }
    let payload = &f.get_node(value).payload;
    match payload {
        NodePayload::BitSlice {
            arg, start: offset, ..
        } => plan(f, uses, *arg, start.checked_add(*offset)?, width, budget),
        NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl), data, amount) => {
            Some(shift_plan(f, *op, *data, *amount, start, width, budget).unwrap_or_else(fallback))
        }
        NodePayload::Sel {
            selector,
            cases,
            default: None,
        } if f.get_node(*selector).ty == Type::Bits(1) && cases.len() == 2 => {
            let lhs = plan(f, uses, cases[0], start, width, budget)?;
            let rhs = plan(f, uses, cases[1], start, width, budget)?;
            Some(Plan::Select {
                selector: *selector,
                cases: [Box::new(lhs), Box::new(rhs)],
            })
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default: Some(default),
        } if f.get_node(*selector).ty == Type::Bits(1) && cases.len() == 1 => {
            let lhs = plan(f, uses, *default, start, width, budget)?;
            let rhs = plan(f, uses, cases[0], start, width, budget)?;
            Some(Plan::Select {
                selector: *selector,
                cases: [Box::new(lhs), Box::new(rhs)],
            })
        }
        _ => Some(fallback()),
    }
}

/// A root and its pre-mutation plan; boolean users of the root may be shared.
struct Rewrite {
    root: NodeRef,
    invert: bool,
    plan: Plan,
}

/// Recognizes OR reductions and comparisons with zero, independent of operand
/// order.
fn recognize(f: &ir::Fn, uses: &[usize], root: NodeRef) -> Option<Rewrite> {
    let (value, invert) = match_zero_test(f, root)?;
    let width = MatchCtx::new(f).bits_width(value)?;
    let mut budget = Budget::default();
    let plan = plan(f, uses, value, 0, width, &mut budget)?;
    (budget.shifts > 0).then_some(Rewrite { root, invert, plan })
}

impl NodeAppender<'_> {
    fn reduce(&mut self, value: NodeRef, start: usize, width: usize) -> NodeRef {
        let slice = if start == 0 && self.f.get_node(value).ty == Type::Bits(width) {
            value
        } else {
            self.push(
                width,
                NodePayload::BitSlice {
                    arg: value,
                    start,
                    width,
                },
            )
        };
        if width == 1 {
            slice
        } else {
            self.push(1, NodePayload::Unop(Unop::OrReduce, slice))
        }
    }

    fn combine(&mut self, op: NaryOp, terms: Vec<NodeRef>) -> NodeRef {
        if terms.len() == 1 {
            terms[0]
        } else {
            self.push(1, NodePayload::Nary(op, terms))
        }
    }

    /// Builds each survival interval and preserves select polarity.
    fn emit(&mut self, plan: Plan) -> NodeRef {
        match plan {
            Plan::Reduce {
                value,
                start,
                width,
            } => self.reduce(value, start, width),
            Plan::Select {
                selector,
                cases: [lhs, rhs],
            } => {
                let a = self.emit(*lhs);
                let b = self.emit(*rhs);
                self.push(
                    1,
                    NodePayload::Sel {
                        selector,
                        cases: vec![a, b],
                        default: None,
                    },
                )
            }
            Plan::Shift {
                data,
                amount,
                low_width,
                max,
                terms,
            } => {
                let aw = self.f.get_node(amount).ty.bit_count();
                let low_amount = if low_width < aw {
                    self.push(
                        low_width,
                        NodePayload::BitSlice {
                            arg: amount,
                            start: 0,
                            width: low_width,
                        },
                    )
                } else {
                    amount
                };
                let mut outputs = Vec::new();
                for term in terms {
                    let mut factors = vec![self.reduce(data, term.start, term.width)];
                    if term.low > 0 {
                        let bound = self.literal(low_width, term.low);
                        factors
                            .push(self.push(1, NodePayload::Binop(Binop::Uge, low_amount, bound)));
                    }
                    if term.high < max {
                        let bound = self.literal(low_width, term.high);
                        factors
                            .push(self.push(1, NodePayload::Binop(Binop::Ule, low_amount, bound)));
                    }
                    outputs.push(self.combine(NaryOp::And, factors));
                }
                let survives = self.combine(NaryOp::Or, outputs);
                if low_width < aw {
                    let high_nonzero = self.reduce(amount, low_width, aw - low_width);
                    let in_range = self.push(1, NodePayload::Unop(Unop::Not, high_nonzero));
                    self.combine(NaryOp::And, vec![in_range, survives])
                } else {
                    survives
                }
            }
        }
    }
}

/// Plans against a single live-use snapshot, then rewrites only scalar roots.
/// Interior data nodes are untouched, so previously planned independent or
/// nested predicates keep their meaning. Dead consumers cannot veto removal.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    // Most functions have no logical shift. Avoid liveness/user allocation
    // entirely when this rewrite cannot apply.
    if !f.nodes.iter().any(|node| {
        matches!(
            node.payload,
            NodePayload::Binop(Binop::Shll | Binop::Shrl, _, _)
        )
    }) {
        return 0;
    }
    let mut live = vec![true; f.nodes.len()];
    for n in xlsynth_pir::dce::get_dead_nodes(f) {
        live[n.index] = false;
    }
    let mut uses = vec![0; f.nodes.len()];
    for (index, node) in f.nodes.iter().enumerate() {
        if live[index] {
            for operand in ir_utils::operands(&node.payload) {
                uses[operand.index] += 1;
            }
        }
    }
    if let Some(root) = f.ret_node_ref {
        uses[root.index] += 1;
    }
    let rewrites: Vec<_> = (0..f.nodes.len())
        .filter(|&index| live[index])
        .filter_map(|index| recognize(f, &uses, NodeRef { index }))
        .collect();
    let count = rewrites.len();
    let next_id = next_text_id(f);
    let mut builder = NodeAppender { f, next_id };
    for rewrite in rewrites {
        let result = builder.emit(rewrite.plan);
        builder.f.nodes[rewrite.root.index].payload = if rewrite.invert {
            NodePayload::Unop(Unop::Not, result)
        } else {
            builder.f.get_node(result).payload.clone()
        };
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    /// Builds a direct reduction with optional numeric sharing or a dead user.
    fn case(
        n: usize,
        aw: usize,
        op: &str,
        start: usize,
        width: usize,
        shared: bool,
        dead: bool,
    ) -> ir::Fn {
        let (ret_ty, ret) = if shared {
            (
                format!("(bits[1], bits[{n}])"),
                format!("ret result: (bits[1], bits[{n}]) = tuple(pred, shifted, id=9)"),
            )
        } else {
            (
                "bits[1]".into(),
                "ret result: bits[1] = identity(pred, id=9)".into(),
            )
        };
        let unused = if dead {
            format!("unused: bits[{n}] = not(shifted, id=8)")
        } else {
            String::new()
        };
        let text = format!(
            r#"package example
top fn main(x: bits[{n}] id=1, k: bits[{aw}] id=2) -> {ret_ty} {{
 shifted: bits[{n}] = {op}(x, k, id=3)
 window: bits[{width}] = bit_slice(shifted, start={start}, width={width}, id=4)
 pred: bits[1] = or_reduce(window, id=5)
 {unused}
 {ret}
}}"#
        );
        Parser::new(&text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone()
    }

    #[test]
    fn survival_plan_fires_and_declines_with_structural_guards() {
        for op in ["shll", "shrl"] {
            for (n, aw, start, width, expected) in [
                (1, 1, 0, 1, 0),
                (2, 1, 0, 2, 1),
                (3, 1, 0, 3, 1),
                (23, 3, 0, 23, 1),
                (24, 5, 0, 23, 1),
                (32, 3, 4, 23, 1),
                (65, 3, 0, 65, 1),
                (256, 3, 0, 256, 1),
                (257, 3, 0, 257, 0),
                (23, 6, 0, 23, 1),
                (2, 65, 0, 2, 1),
                (22, 32, 0, 22, 1),
                (31, 129, 0, 31, 1),
                (32, 32, 0, 32, 1),
                (32, 65, 1, 31, 0),
                (7, 9, 0, 6, 0),
                (33, 6, 0, 33, 0),
                (65, 65, 0, 65, 0),
                (32, 5, 0, 1, usize::from(op == "shll")),
            ] {
                let mut f = case(n, aw, op, start, width, false, false);
                assert_eq!(rewrite(&mut f), expected, "{op} {n} {aw} {start} {width}");
                let mut shared = case(n, aw, op, start, width, true, false);
                assert_eq!(rewrite(&mut shared), 0, "numeric consumer must survive");
            }
            let mut dead = case(23, 3, op, 0, 23, false, true);
            assert_eq!(rewrite(&mut dead), 1, "dead consumers do not veto");
        }
        assert_eq!(rewrite(&mut case(23, 3, "shra", 0, 23, false, false)), 0);
    }

    #[test]
    fn constants_keep_the_existing_specialized_lowerings() {
        for parameter in [0, 1] {
            for value in [0, 1] {
                let mut f = case(23, 3, "shll", 0, 23, false, false);
                let arg = f.params[parameter];
                let width = f.get_node(arg).ty.bit_count();
                // Add a literal rather than replacing a signature parameter.
                let next_id = next_text_id(&f);
                let mut builder = NodeAppender { f: &mut f, next_id };
                let literal = builder.literal(width, value);
                let shifted = builder
                    .f
                    .nodes
                    .iter()
                    .position(|n| n.name.as_deref() == Some("shifted"))
                    .unwrap();
                if let NodePayload::Binop(_, data, amount) = &mut builder.f.nodes[shifted].payload {
                    if parameter == 0 {
                        *data = literal;
                    } else {
                        *amount = literal;
                    }
                }
                assert_eq!(rewrite(&mut f), 0);
            }
        }
    }

    #[test]
    fn scalarization_is_bounded_and_requires_a_shift() {
        let mut f = case(23, 3, "shll", 0, 23, false, false);
        let value = f
            .nodes
            .iter()
            .position(|n| n.name.as_deref() == Some("shifted"))
            .unwrap();
        f.nodes[value].payload = NodePayload::Unop(Unop::Identity, f.params[0]);
        assert_eq!(rewrite(&mut f), 0);
    }
}
