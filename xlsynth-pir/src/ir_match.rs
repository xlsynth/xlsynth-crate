// SPDX-License-Identifier: Apache-2.0

//! Programmatic structural matching helpers for PIR.
//!
//! This module complements, rather than replaces, `crate::ir_query`.
//! `ir_query` is a textual query DSL intended for fixed, user-facing patterns
//! and rewrite rules. The helpers here are for Rust code that needs to build
//! shape-dependent matchers from local context, inspect typed `NodeRef`s, and
//! apply controlled structural normalization such as commutative operand
//! matching and associative flattening.
//!
//! This module is not a solver-backed equivalence engine and does not attempt
//! general algebraic simplification. It only provides small, explicit
//! combinators for matching PIR DAG structure in ways that are tedious or
//! brittle to express as text queries, such as comparing an `and` operand
//! multiset independent of operand order and nesting.

use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use std::collections::HashMap;

/// Match-time value bound to a placeholder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchValue {
    Node(NodeRef),
    Nodes(Vec<NodeRef>),
}

/// Placeholder bindings accumulated while matching a pattern.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bindings {
    values: HashMap<&'static str, MatchValue>,
}

impl Bindings {
    /// Returns the node bound to `name`, if any.
    pub fn get_node(&self, name: &'static str) -> Option<NodeRef> {
        match self.values.get(name) {
            Some(MatchValue::Node(node_ref)) => Some(*node_ref),
            _ => None,
        }
    }

    /// Returns the node list bound to `name`, if any.
    pub fn get_nodes(&self, name: &'static str) -> Option<&[NodeRef]> {
        match self.values.get(name) {
            Some(MatchValue::Nodes(node_refs)) => Some(node_refs),
            _ => None,
        }
    }

    fn bind(&self, name: &'static str, value: MatchValue) -> Option<Self> {
        if name == "_" {
            return Some(self.clone());
        }
        match self.values.get(name) {
            Some(existing) if existing == &value => Some(self.clone()),
            Some(_) => None,
            None => {
                let mut next = self.clone();
                next.values.insert(name, value);
                Some(next)
            }
        }
    }
}

/// Context for matching nodes in one PIR function.
pub struct MatchCtx<'a> {
    pub f: &'a ir::Fn,
}

impl<'a> MatchCtx<'a> {
    /// Creates a matcher context for `f`.
    pub fn new(f: &'a ir::Fn) -> Self {
        Self { f }
    }

    /// Returns the bit width of `node` when it has bits type.
    pub fn bits_width(&self, node: NodeRef) -> Option<usize> {
        match self.f.get_node(node).ty {
            Type::Bits(width) => Some(width),
            _ => None,
        }
    }

    /// Returns operands of `node` for `op`, flattening nested uses of `op`.
    pub fn flattened_nary_operands(&self, node: NodeRef, op: NaryOp) -> Option<Vec<NodeRef>> {
        let NodePayload::Nary(node_op, operands) = &self.f.get_node(node).payload else {
            return None;
        };
        if *node_op != op {
            return None;
        }
        let mut flattened = Vec::new();
        self.flatten_nary_operands_into(*node_op, operands, &mut flattened);
        Some(flattened)
    }

    fn flatten_nary_operands_into(
        &self,
        op: NaryOp,
        operands: &[NodeRef],
        flattened: &mut Vec<NodeRef>,
    ) {
        for operand in operands {
            if nary_op_is_associative(op)
                && matches!(
                    &self.f.get_node(*operand).payload,
                    NodePayload::Nary(child_op, _) if *child_op == op
                )
            {
                let NodePayload::Nary(_, child_operands) = &self.f.get_node(*operand).payload
                else {
                    unreachable!("payload was checked as nary")
                };
                self.flatten_nary_operands_into(op, child_operands, flattened);
            } else {
                flattened.push(*operand);
            }
        }
    }

    /// Returns true if flattened `node` operands contain every required ref.
    pub fn commutative_contains_all(
        &self,
        node: NodeRef,
        op: NaryOp,
        required: &[NodeRef],
    ) -> bool {
        if !nary_op_is_commutative(op) {
            return false;
        }
        let Some(operands) = self.flattened_nary_operands(node, op) else {
            return false;
        };
        let mut used = vec![false; operands.len()];
        for required_ref in required {
            let Some(index) = operands
                .iter()
                .enumerate()
                .position(|(i, operand)| !used[i] && operand == required_ref)
            else {
                return false;
            };
            used[index] = true;
        }
        true
    }

    /// Returns true iff `node` is exactly `bit_slice(arg, start, width)`.
    pub fn bit_slice_of(&self, node: NodeRef, arg: NodeRef, start: usize, width: usize) -> bool {
        matches!(
            &self.f.get_node(node).payload,
            NodePayload::BitSlice {
                arg: node_arg,
                start: node_start,
                width: node_width,
            } if *node_arg == arg && *node_start == start && *node_width == width
        )
    }

    /// Returns true iff `node` is exactly `not(arg)`.
    pub fn not_of(&self, node: NodeRef, arg: NodeRef) -> bool {
        matches!(
            &self.f.get_node(node).payload,
            NodePayload::Unop(Unop::Not, node_arg) if *node_arg == arg
        )
    }

    /// Matches `pattern` against `node`, returning bindings on success.
    pub fn matches<P: IntoPattern>(&self, node: NodeRef, pattern: P) -> Option<Bindings> {
        pattern
            .into_pattern()
            .match_node(self, node, &Bindings::default())
    }

    /// Matches a two-operand commutative operator in either operand order.
    pub fn commutative_pair<L: IntoPattern, R: IntoPattern>(
        &self,
        node: NodeRef,
        op: NaryOp,
        lhs: L,
        rhs: R,
    ) -> Option<Bindings> {
        self.matches(
            node,
            commutative(op, vec![lhs.into_pattern(), rhs.into_pattern()]),
        )
    }

    /// Matches a two-operand commutative binary operator in either operand
    /// order.
    pub fn commutative_binop_pair<L: IntoPattern, R: IntoPattern>(
        &self,
        node: NodeRef,
        op: Binop,
        lhs: L,
        rhs: R,
    ) -> Option<Bindings> {
        self.matches(node, commutative_binop(op, lhs, rhs))
    }
}

/// Programmatic node pattern.
#[derive(Clone, Debug, PartialEq)]
pub enum PatternExpr {
    Any(&'static str),
    Exact(NodeRef),
    BitSlice {
        arg: Box<PatternExpr>,
        start: usize,
        width: usize,
    },
    Binop {
        op: Binop,
        lhs: Box<PatternExpr>,
        rhs: Box<PatternExpr>,
    },
    CommutativeBinop {
        op: Binop,
        lhs: Box<PatternExpr>,
        rhs: Box<PatternExpr>,
    },
    Not(Box<PatternExpr>),
    Commutative {
        op: NaryOp,
        args: Vec<PatternExpr>,
    },
}

/// Converts values into a pattern expression.
pub trait IntoPattern {
    fn into_pattern(self) -> PatternExpr;
}

impl IntoPattern for PatternExpr {
    fn into_pattern(self) -> PatternExpr {
        self
    }
}

impl IntoPattern for NodeRef {
    fn into_pattern(self) -> PatternExpr {
        PatternExpr::Exact(self)
    }
}

/// Matchable PIR pattern.
pub trait Pattern {
    fn match_node(&self, ctx: &MatchCtx, node: NodeRef, bindings: &Bindings) -> Option<Bindings>;
}

impl Pattern for PatternExpr {
    fn match_node(&self, ctx: &MatchCtx, node: NodeRef, bindings: &Bindings) -> Option<Bindings> {
        match self {
            PatternExpr::Any(name) => bindings.bind(name, MatchValue::Node(node)),
            PatternExpr::Exact(expected) => (node == *expected).then(|| bindings.clone()),
            PatternExpr::BitSlice { arg, start, width } => {
                let NodePayload::BitSlice {
                    arg: node_arg,
                    start: node_start,
                    width: node_width,
                } = &ctx.f.get_node(node).payload
                else {
                    return None;
                };
                if *node_start != *start || *node_width != *width {
                    return None;
                }
                arg.match_node(ctx, *node_arg, bindings)
            }
            PatternExpr::Binop { op, lhs, rhs } => {
                let NodePayload::Binop(node_op, node_lhs, node_rhs) = &ctx.f.get_node(node).payload
                else {
                    return None;
                };
                if *node_op != *op {
                    return None;
                }
                let bindings = lhs.match_node(ctx, *node_lhs, bindings)?;
                rhs.match_node(ctx, *node_rhs, &bindings)
            }
            PatternExpr::CommutativeBinop { op, lhs, rhs } => {
                if !binop_is_commutative(*op) {
                    return None;
                }
                let NodePayload::Binop(node_op, node_lhs, node_rhs) = &ctx.f.get_node(node).payload
                else {
                    return None;
                };
                if *node_op != *op {
                    return None;
                }
                if let Some(lhs_first) = lhs.match_node(ctx, *node_lhs, bindings) {
                    if let Some(result) = rhs.match_node(ctx, *node_rhs, &lhs_first) {
                        return Some(result);
                    }
                }
                let rhs_first = rhs.match_node(ctx, *node_lhs, bindings)?;
                lhs.match_node(ctx, *node_rhs, &rhs_first)
            }
            PatternExpr::Not(arg) => {
                let NodePayload::Unop(Unop::Not, node_arg) = &ctx.f.get_node(node).payload else {
                    return None;
                };
                arg.match_node(ctx, *node_arg, bindings)
            }
            PatternExpr::Commutative { op, args } => {
                if !nary_op_is_commutative(*op) {
                    return None;
                }
                let operands = ctx.flattened_nary_operands(node, *op)?;
                match_commutative_args(ctx, &operands, args, bindings)
            }
        }
    }
}

/// Binds any node to `name`; `_` is treated as an unbound wildcard.
pub fn any(name: &'static str) -> PatternExpr {
    PatternExpr::Any(name)
}

/// Matches exactly `node`.
pub fn exact(node: NodeRef) -> PatternExpr {
    PatternExpr::Exact(node)
}

/// Matches a static bit slice.
pub fn bit_slice<P: IntoPattern>(arg: P, start: usize, width: usize) -> PatternExpr {
    PatternExpr::BitSlice {
        arg: Box::new(arg.into_pattern()),
        start,
        width,
    }
}

/// Matches an exact-order binary op.
pub fn binop<L: IntoPattern, R: IntoPattern>(op: Binop, lhs: L, rhs: R) -> PatternExpr {
    PatternExpr::Binop {
        op,
        lhs: Box::new(lhs.into_pattern()),
        rhs: Box::new(rhs.into_pattern()),
    }
}

/// Matches a commutative binary op independent of operand order.
pub fn commutative_binop<L: IntoPattern, R: IntoPattern>(op: Binop, lhs: L, rhs: R) -> PatternExpr {
    PatternExpr::CommutativeBinop {
        op,
        lhs: Box::new(lhs.into_pattern()),
        rhs: Box::new(rhs.into_pattern()),
    }
}

/// Matches a unary `not`.
pub fn not<P: IntoPattern>(arg: P) -> PatternExpr {
    PatternExpr::Not(Box::new(arg.into_pattern()))
}

/// Matches a commutative n-ary op independent of operand order and nesting.
pub fn commutative<P: IntoPattern>(op: NaryOp, args: Vec<P>) -> PatternExpr {
    PatternExpr::Commutative {
        op,
        args: args.into_iter().map(IntoPattern::into_pattern).collect(),
    }
}

fn nary_op_is_associative(op: NaryOp) -> bool {
    matches!(op, NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Concat)
}

fn nary_op_is_commutative(op: NaryOp) -> bool {
    matches!(op, NaryOp::And | NaryOp::Or | NaryOp::Xor)
}

fn binop_is_commutative(op: Binop) -> bool {
    matches!(
        op,
        Binop::Add | Binop::Eq | Binop::Ne | Binop::Umul | Binop::Smul
    )
}

fn match_commutative_args(
    ctx: &MatchCtx,
    operands: &[NodeRef],
    patterns: &[PatternExpr],
    bindings: &Bindings,
) -> Option<Bindings> {
    if operands.len() != patterns.len() {
        return None;
    }
    let mut used = vec![false; operands.len()];
    match_commutative_args_from(ctx, operands, patterns, bindings, &mut used, 0)
}

fn match_commutative_args_from(
    ctx: &MatchCtx,
    operands: &[NodeRef],
    patterns: &[PatternExpr],
    bindings: &Bindings,
    used: &mut [bool],
    pattern_index: usize,
) -> Option<Bindings> {
    if pattern_index == patterns.len() {
        return Some(bindings.clone());
    }
    let pattern = &patterns[pattern_index];
    for operand_index in 0..operands.len() {
        if used[operand_index] {
            continue;
        }
        let Some(next_bindings) = pattern.match_node(ctx, operands[operand_index], bindings) else {
            continue;
        };
        used[operand_index] = true;
        if let Some(result) = match_commutative_args_from(
            ctx,
            operands,
            patterns,
            &next_bindings,
            used,
            pattern_index + 1,
        ) {
            return Some(result);
        }
        used[operand_index] = false;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    fn parse_top_fn(ir_text: &str) -> ir::Fn {
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().expect("parse/validate");
        pkg.get_top_fn().expect("top fn").clone()
    }

    fn node_by_name(f: &ir::Fn, name: &str) -> NodeRef {
        f.nodes
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                (node.name.as_deref() == Some(name)).then_some(NodeRef { index })
            })
            .unwrap_or_else(|| panic!("missing node named {name}"))
    }

    #[test]
    fn ir_match_commutative_pair_matches_both_operand_orders() {
        let f = parse_top_fn(
            r#"package sample

top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  x0: bits[1] = xor(a, b, id=3)
  ret x1: bits[1] = xor(b, a, id=4)
}
"#,
        );
        let a = node_by_name(&f, "a");
        let b = node_by_name(&f, "b");
        let x0 = node_by_name(&f, "x0");
        let x1 = node_by_name(&f, "x1");
        let ctx = MatchCtx::new(&f);

        assert!(
            ctx.commutative_pair(x0, NaryOp::Xor, exact(a), exact(b))
                .is_some()
        );
        assert!(
            ctx.commutative_pair(x1, NaryOp::Xor, exact(a), exact(b))
                .is_some()
        );
    }

    #[test]
    fn ir_match_nested_and_flattening_handles_left_and_right_association() {
        let f = parse_top_fn(
            r#"package sample

top fn f(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3) -> bits[1] {
  ab: bits[1] = and(a, b, id=4)
  left: bits[1] = and(ab, c, id=5)
  bc: bits[1] = and(b, c, id=6)
  ret right: bits[1] = and(a, bc, id=7)
}
"#,
        );
        let a = node_by_name(&f, "a");
        let b = node_by_name(&f, "b");
        let c = node_by_name(&f, "c");
        let left = node_by_name(&f, "left");
        let right = node_by_name(&f, "right");
        let ctx = MatchCtx::new(&f);
        let pattern = || commutative(NaryOp::And, vec![exact(a), exact(b), exact(c)]);

        assert!(ctx.matches(left, pattern()).is_some());
        assert!(ctx.matches(right, pattern()).is_some());
    }

    #[test]
    fn ir_match_duplicate_operands_are_respected() {
        let f = parse_top_fn(
            r#"package sample

top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  aa: bits[1] = and(a, a, id=3)
  ret ab: bits[1] = and(a, b, id=4)
}
"#,
        );
        let a = node_by_name(&f, "a");
        let aa = node_by_name(&f, "aa");
        let ab = node_by_name(&f, "ab");
        let ctx = MatchCtx::new(&f);
        let pattern = || commutative(NaryOp::And, vec![exact(a), exact(a)]);

        assert!(ctx.matches(aa, pattern()).is_some());
        assert!(ctx.matches(ab, pattern()).is_none());
    }

    #[test]
    fn ir_match_repeated_bindings_must_be_consistent() {
        let f = parse_top_fn(
            r#"package sample

top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  aa: bits[1] = xor(a, a, id=3)
  ret ab: bits[1] = xor(a, b, id=4)
}
"#,
        );
        let aa = node_by_name(&f, "aa");
        let ab = node_by_name(&f, "ab");
        let ctx = MatchCtx::new(&f);
        let pattern = || commutative(NaryOp::Xor, vec![any("x"), any("x")]);

        let bindings = ctx.matches(aa, pattern()).expect("aa should match");
        assert_eq!(bindings.get_node("x"), Some(node_by_name(&f, "a")));
        assert!(ctx.matches(ab, pattern()).is_none());
    }

    #[test]
    fn ir_match_commutative_binop_matches_both_operand_orders() {
        let f = parse_top_fn(
            r#"package sample

top fn f(a: bits[4] id=1, b: bits[4] id=2) -> bits[4] {
  ab: bits[4] = add(a, b, id=3)
  ret ba: bits[4] = add(b, a, id=4)
}
"#,
        );
        let a = node_by_name(&f, "a");
        let b = node_by_name(&f, "b");
        let ab = node_by_name(&f, "ab");
        let ba = node_by_name(&f, "ba");
        let ctx = MatchCtx::new(&f);

        assert!(
            ctx.commutative_binop_pair(ab, Binop::Add, exact(a), exact(b))
                .is_some()
        );
        assert!(
            ctx.commutative_binop_pair(ba, Binop::Add, exact(a), exact(b))
                .is_some()
        );
    }

    #[test]
    fn ir_match_exact_binop_preserves_operand_order() {
        let f = parse_top_fn(
            r#"package sample

top fn f(a: bits[4] id=1, b: bits[4] id=2) -> bits[4] {
  ab: bits[4] = shll(a, b, id=3)
  ret ba: bits[4] = shll(b, a, id=4)
}
"#,
        );
        let a = node_by_name(&f, "a");
        let b = node_by_name(&f, "b");
        let ab = node_by_name(&f, "ab");
        let ba = node_by_name(&f, "ba");
        let ctx = MatchCtx::new(&f);
        let pattern = || binop(Binop::Shll, exact(a), exact(b));

        assert!(ctx.matches(ab, pattern()).is_some());
        assert!(ctx.matches(ba, pattern()).is_none());
    }

    #[test]
    fn ir_match_repeated_binop_bindings_must_be_consistent() {
        let f = parse_top_fn(
            r#"package sample

top fn f(ones: bits[4] id=1, count0: bits[4] id=2, count1: bits[4] id=3) -> bits[4] {
  shr: bits[4] = shrl(ones, count0, id=4)
  same: bits[4] = shll(shr, count0, id=5)
  ret different: bits[4] = shll(shr, count1, id=6)
}
"#,
        );
        let ones = node_by_name(&f, "ones");
        let same = node_by_name(&f, "same");
        let different = node_by_name(&f, "different");
        let ctx = MatchCtx::new(&f);
        let pattern = || {
            binop(
                Binop::Shll,
                binop(Binop::Shrl, exact(ones), any("count")),
                any("count"),
            )
        };

        let bindings = ctx.matches(same, pattern()).expect("same should match");
        assert_eq!(bindings.get_node("count"), Some(node_by_name(&f, "count0")));
        assert!(ctx.matches(different, pattern()).is_none());
    }

    #[test]
    fn ir_match_repeated_commutative_binop_bindings_must_be_consistent() {
        let f = parse_top_fn(
            r#"package sample

top fn f(count0: bits[4] id=1, count1: bits[4] id=2) -> bits[4] {
  same: bits[4] = add(count0, count0, id=3)
  ret different: bits[4] = add(count0, count1, id=4)
}
"#,
        );
        let same = node_by_name(&f, "same");
        let different = node_by_name(&f, "different");
        let ctx = MatchCtx::new(&f);
        let pattern = || commutative_binop(Binop::Add, any("count"), any("count"));

        let bindings = ctx.matches(same, pattern()).expect("same should match");
        assert_eq!(bindings.get_node("count"), Some(node_by_name(&f, "count0")));
        assert!(ctx.matches(different, pattern()).is_none());
    }

    #[test]
    fn ir_match_width_and_bit_slice_guards_reject_mismatches() {
        let f = parse_top_fn(
            r#"package sample

top fn f(x: bits[4] id=1) -> bits[1] {
  s0: bits[1] = bit_slice(x, start=0, width=1, id=2)
  ret s1: bits[1] = bit_slice(x, start=1, width=1, id=3)
}
"#,
        );
        let x = node_by_name(&f, "x");
        let s0 = node_by_name(&f, "s0");
        let s1 = node_by_name(&f, "s1");
        let ctx = MatchCtx::new(&f);

        assert_eq!(ctx.bits_width(x), Some(4));
        assert_eq!(ctx.bits_width(s0), Some(1));
        assert!(ctx.bit_slice_of(s0, x, 0, 1));
        assert!(!ctx.bit_slice_of(s1, x, 0, 1));
        assert!(ctx.matches(s0, bit_slice(exact(x), 0, 1)).is_some());
        assert!(ctx.matches(s1, bit_slice(exact(x), 0, 1)).is_none());
    }
}
