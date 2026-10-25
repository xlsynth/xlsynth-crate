// SPDX-License-Identifier: Apache-2.0

//! Bounded predicates over the low bits of an affine left shift.

use std::collections::BTreeMap;
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_match::{self, MatchCtx};

/// A single-use shift cone whose observed source windows are short prefixes.
struct Window {
    root: NodeRef,
    data: NodeRef,
    base: NodeRef,
    low: Option<NodeRef>,
    offset: usize,
    stride: usize,
    end: usize,
    cases: usize,
    invert: bool,
}

/// Recognizes bounded literals without restricting the IR's bit width.
fn bounded_literal(f: &ir::Fn, n: NodeRef, bound: usize) -> Option<usize> {
    let NodePayload::Literal(value) = &f.get_node(n).payload else {
        return None;
    };
    (0..=bound).find(|&i| value.bits_equals_u64_value(i as u64))
}

/// Finds a nonzero/zero predicate with a provably nonwrapping affine shift.
///
/// For `amount = (zext(base) + offset) ++ low`, only a few small base values
/// can contribute to a low output window. Requiring the minimum shift to reach
/// the window's start makes all contributing input windows prefixes. The bounds
/// limit expansion to eight (base, low) choices, independent of operand width.
/// Single-use interior nodes ensure the rewrite can remove the barrel shifter.
fn recognize(f: &ir::Fn, uses: &[usize], root: NodeRef) -> Option<Window> {
    let ctx = MatchCtx::new(f);
    let (slice, invert) = match f.get_node(root).payload {
        NodePayload::Unop(Unop::OrReduce, arg) => (arg, false),
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) => {
            let zero = [lhs, rhs]
                .into_iter()
                .find(|&n| bounded_literal(f, n, 0) == Some(0))?;
            let bindings = ctx.commutative_binop_pair(
                root,
                op,
                ir_match::any("slice"),
                ir_match::exact(zero),
            )?;
            (bindings.get_node("slice")?, op == Binop::Eq)
        }
        _ => return None,
    };
    let NodePayload::BitSlice {
        arg: shift,
        start,
        width,
    } = f.get_node(slice).payload
    else {
        return None;
    };
    let end = start.checked_add(width)?;
    if width < 3 || end > 8 {
        return None;
    }
    let NodePayload::Binop(Binop::Shll, data, amount) = f.get_node(shift).payload else {
        return None;
    };
    let (sum, low) = match &f.get_node(amount).payload {
        NodePayload::Nary(NaryOp::Concat, args) if args.len() == 2 => (args[0], Some(args[1])),
        _ => (amount, None),
    };
    let low_width = low.map(|n| ctx.bits_width(n)).unwrap_or(Some(0))?;
    if low_width > 2 {
        return None;
    }
    let stride = 1usize << low_width;
    let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(sum).payload else {
        return None;
    };
    let literal = [lhs, rhs]
        .into_iter()
        .find(|&n| bounded_literal(f, n, 8).is_some())?;
    let offset = bounded_literal(f, literal, 8)?;
    if offset == 0 || offset * stride < start {
        return None;
    }
    let bindings = ctx.commutative_binop_pair(
        sum,
        Binop::Add,
        ir_match::any("wide"),
        ir_match::exact(literal),
    )?;
    let wide = bindings.get_node("wide")?;
    let base = match &f.get_node(wide).payload {
        NodePayload::ZeroExt { arg, .. } => *arg,
        NodePayload::Nary(NaryOp::Concat, args)
            if args.len() == 2 && bounded_literal(f, args[0], 0) == Some(0) =>
        {
            args[1]
        }
        _ => return None,
    };
    // With base width >= 4, offset <= 8 fits in the extension bit even at
    // base's maximum value. Thus neither addition nor concatenation wraps.
    if ctx.bits_width(base)? < 4 || ctx.bits_width(wide)? <= ctx.bits_width(base)? {
        return None;
    }
    let cases = ((end - 1) / stride + 1).saturating_sub(offset);
    if cases > 4 || cases * stride > 8 {
        return None;
    }
    if [slice, shift, amount, sum, wide]
        .into_iter()
        .any(|n| uses[n.index] != 1)
    {
        return None;
    }
    Some(Window {
        root,
        data,
        base,
        low,
        offset,
        stride,
        end,
        cases,
        invert,
    })
}

/// Appends a bounded replacement without rescanning the function for text IDs.
struct Builder<'a> {
    f: &'a mut ir::Fn,
    next_id: usize,
}

impl Builder<'_> {
    fn push(&mut self, width: usize, payload: NodePayload) -> NodeRef {
        let result = NodeRef {
            index: self.f.nodes.len(),
        };
        self.f.nodes.push(ir::Node {
            text_id: self.next_id,
            name: None,
            ty: Type::Bits(width),
            payload,
            pos: None,
        });
        self.next_id += 1;
        result
    }

    fn literal(&mut self, width: usize, value: usize) -> NodeRef {
        self.push(
            width,
            NodePayload::Literal(
                IrValue::make_ubits(width, value as u64)
                    .expect("bounded literal fits its IR width"),
            ),
        )
    }

    fn or(&mut self, terms: Vec<NodeRef>) -> NodeRef {
        match terms.as_slice() {
            [] => self.literal(1, 0),
            [one] => *one,
            _ => self.push(1, NodePayload::Nary(NaryOp::Or, terms)),
        }
    }

    /// Emits prefix reductions and low-control thresholds shared across cases.
    fn replace(&mut self, w: Window) {
        let mut prefixes = BTreeMap::new();
        let mut thresholds = BTreeMap::new();
        let mut cases = Vec::new();
        for value in 0..w.cases {
            let count = w.end - (value + w.offset) * w.stride;
            let always = count.saturating_sub(w.stride - 1);
            let mut terms = Vec::new();
            if always > 0 {
                let prefix = *prefixes.entry(always).or_insert_with(|| {
                    let slice = self.push(
                        always,
                        NodePayload::BitSlice {
                            arg: w.data,
                            start: 0,
                            width: always,
                        },
                    );
                    if always == 1 {
                        slice
                    } else {
                        self.push(1, NodePayload::Unop(Unop::OrReduce, slice))
                    }
                });
                terms.push(prefix);
            }
            for bit in always..count {
                let threshold = count - bit;
                let predicate = *thresholds.entry(threshold).or_insert_with(|| {
                    let low = w.low.expect("partial prefix requires low control bits");
                    let limit = self.literal(self.f.get_node(low).ty.bit_count(), threshold);
                    self.push(1, NodePayload::Binop(Binop::Ult, low, limit))
                });
                let input = self.push(
                    1,
                    NodePayload::BitSlice {
                        arg: w.data,
                        start: bit,
                        width: 1,
                    },
                );
                terms.push(self.push(1, NodePayload::Nary(NaryOp::And, vec![input, predicate])));
            }
            let any = self.or(terms);
            let literal = self.literal(self.f.get_node(w.base).ty.bit_count(), value);
            let select = self.push(1, NodePayload::Binop(Binop::Eq, w.base, literal));
            cases.push(self.push(1, NodePayload::Nary(NaryOp::And, vec![select, any])));
        }
        let result = self.or(cases);
        self.f.nodes[w.root.index].payload = if w.invert {
            NodePayload::Unop(Unop::Not, result)
        } else {
            self.f.get_node(result).payload.clone()
        };
    }
}

/// Analyzes one immutable graph snapshot, then replaces disjoint single-use
/// cones.
///
/// Uses and text IDs are computed once. Replacements only change predicate
/// roots and append helpers; the recognized interiors are never modified.
/// Existing prep-for-gatify cleanup removes dead shift cones and unused helper
/// roots.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    let uses = super::get_use_counts(f);
    let candidates: Vec<_> = (0..f.nodes.len())
        .filter_map(|index| recognize(f, &uses, NodeRef { index }))
        .collect();
    let count = candidates.len();
    let next_id = super::next_text_id(f);
    let mut builder = Builder { f, next_id };
    for candidate in candidates {
        builder.replace(candidate);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    fn fixture() -> ir::Fn {
        let mut parser = Parser::new(include_str!(
            "../../../tests/data/shift_window_threshold.ir"
        ));
        parser
            .parse_and_validate_package()
            .expect("valid fixture")
            .get_top_fn()
            .expect("top function")
            .clone()
    }

    #[test]
    fn recognizes_fixture_and_shared_predicate_root() {
        let mut f = fixture();
        assert_eq!(rewrite(&mut f), 1);
        let mut f = fixture();
        let root = f.ret_node_ref.unwrap();
        let mut b = Builder {
            next_id: super::super::next_text_id(&f),
            f: &mut f,
        };
        let tuple = b.push(1, NodePayload::Nary(NaryOp::And, vec![root, root]));
        b.f.ret_node_ref = Some(tuple);
        assert_eq!(rewrite(&mut f), 1);
    }

    #[test]
    fn rejects_shared_interior_nodes() {
        for name in ["wide", "sum", "amount", "shift", "slice"] {
            let mut f = fixture();
            let n = f
                .nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(name))
                .expect(name);
            let root = f.ret_node_ref.unwrap();
            let mut b = Builder {
                next_id: super::super::next_text_id(&f),
                f: &mut f,
            };
            let shared = b.push(1, NodePayload::Unop(Unop::OrReduce, NodeRef { index: n }));
            let ret = b.push(1, NodePayload::Nary(NaryOp::Or, vec![root, shared]));
            b.f.ret_node_ref = Some(ret);
            assert_eq!(rewrite(&mut f), 0, "{name}");
        }
    }

    #[test]
    fn rejects_unbounded_wrapping_and_nonprefix_shapes() {
        let text = include_str!("../../../tests/data/shift_window_threshold.ir");
        for (from, to) in [
            ("value=1", "value=0"),
            ("value=1", "value=9"),
            ("start=2, width=5", "start=3, width=5"),
            ("start=2, width=5", "start=5, width=5"),
            ("shll(x, amount", "shrl(x, amount"),
            (
                "zero_ext(a, new_bit_count=6,",
                "sign_ext(a, new_bit_count=6,",
            ),
        ] {
            // The wider observed window needs a corresponding result type.
            let altered = text.replace(from, to);
            let mut parser = Parser::new(&altered);
            let mut f = parser
                .parse_and_validate_package()
                .expect("valid negative case")
                .get_top_fn()
                .unwrap()
                .clone();
            assert_eq!(rewrite(&mut f), 0, "{from} -> {to}");
        }
        let same_width = text.replace("a: bits[5]", "a: bits[6]");
        let mut parser = Parser::new(&same_width);
        let mut f = parser
            .parse_and_validate_package()
            .expect("valid wrapping addition")
            .get_top_fn()
            .unwrap()
            .clone();
        assert_eq!(
            rewrite(&mut f),
            0,
            "same-width extension does not prevent wrap"
        );
    }
    #[test]
    fn respects_case_budget_boundary() {
        for (width, expected) in [(5, 1), (6, 0)] {
            let text = format!(
                r#"package bounded
top fn f(a: bits[4] id=1, x: bits[8] id=2) -> bits[1] {{
 wide: bits[5] = zero_ext(a, new_bit_count=5, id=3)
 one: bits[5] = literal(value=1, id=4)
 amount: bits[5] = add(wide, one, id=5)
 shift: bits[8] = shll(x, amount, id=6)
 slice: bits[{width}] = bit_slice(shift, start=0, width={width}, id=7)
 ret result: bits[1] = or_reduce(slice, id=8)
}}"#
            );
            let mut parser = Parser::new(&text);
            let mut f = parser
                .parse_and_validate_package()
                .expect("valid boundary case")
                .get_top_fn()
                .unwrap()
                .clone();
            assert_eq!(rewrite(&mut f), expected, "width={width}");
        }
    }
}
