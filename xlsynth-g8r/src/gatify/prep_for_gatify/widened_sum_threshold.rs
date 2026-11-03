// SPDX-License-Identifier: Apache-2.0

//! Compare exact and modular sums at an operand’s range boundary.

use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::{IrBits, IrValue};

/// For a k-bit unsigned b, x + b > 2^k-1 iff x > ~b. For signed b, bias it
/// by its sign bit first. In either case the resulting threshold is unsigned
/// and fits k bits. A full-width modular sum also needs explicit wrap tests.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    let mut uses = None;
    let mut plans = Vec::new();
    for i in 0..f.nodes.len() {
        if let Some(plan) = Plan::recognize(f, i)
            && uses.get_or_insert_with(|| super::get_use_counts(f))[plan.sum.index] == 1
        {
            plans.push(plan);
        }
    }
    log::debug!(
        "modular_sum_threshold rewrites={}",
        plans
            .iter()
            .filter(|p| p.modular_signed_order.is_some())
            .count()
    );
    if plans.is_empty() {
        return 0;
    }
    let count = plans.len();
    let mut builder = Builder {
        next_id: super::next_text_id(f),
        f,
    };
    for plan in plans {
        builder.emit(plan);
    }
    // The old wide sum is now dead. The enclosing final compaction restores
    // topological order after inserting the narrow predicate.
    super::mark_dead_nodes_as_nil(builder.f);
    count
}

struct Plan {
    root: usize,
    sum: NodeRef,
    wide: ExtNaryAddTerm,
    narrow: ExtNaryAddTerm,
    invert: bool,
    modular_signed_order: Option<bool>,
}

impl Plan {
    fn recognize(f: &ir::Fn, root: usize) -> Option<Self> {
        use Binop::*;
        let NodePayload::Binop(op, a, b) = f.nodes[root].payload else {
            return None;
        };
        let (sum, literal, swapped) = if matches!(f.get_node(b).payload, NodePayload::Literal(_)) {
            (a, b, false)
        } else {
            (b, a, true)
        };
        let NodePayload::Literal(ref endpoint) = f.get_node(literal).payload else {
            return None;
        };
        let NodePayload::ExtNaryAdd {
            ref terms,
            arch: None,
        } = f.get_node(sum).payload
        else {
            return None;
        };
        let [left, right] = terms.as_slice() else {
            return None;
        };
        if left.negated
            || right.negated
            || [left, right]
                .iter()
                .any(|t| matches!(f.get_node(t.operand).payload, NodePayload::Literal(_)))
        {
            return None;
        }
        let width = |t: &ExtNaryAddTerm| f.get_node(t.operand).ty.bit_count();
        let (wide, narrow) = if width(left) > width(right) {
            (left, right)
        } else {
            (right, left)
        };
        let (w, k) = (width(wide), width(narrow));
        let result_width = f.get_node(sum).ty.bit_count();
        let signed = matches!(op, Sgt | Sge | Slt | Sle);
        if k == 0
            || k == w
            || result_width < w
            || (result_width > w && !signed && (wide.signed || narrow.signed))
            || (result_width > w && signed && !wide.signed && result_width - w < 2)
        {
            return None;
        }
        // Express every polarity and commuting order as GT, or its complement.
        // Inclusive lower bounds use the next literal above the endpoint.
        let (invert, next) = match (op, swapped) {
            (Sgt | Ugt, false) | (Slt | Ult, true) => (false, false),
            (Sle | Ule, false) | (Sge | Uge, true) => (true, false),
            (Sge | Uge, false) | (Sle | Ule, true) => (false, true),
            (Slt | Ult, false) | (Sgt | Ugt, true) => (true, true),
            _ => return None,
        };
        // At signed MAX there is no adjacent greater signed literal.
        if next && signed && !narrow.signed && result_width - 1 == k {
            return None;
        }
        let ones = k - usize::from(narrow.signed);
        let literal_bits = endpoint.to_bits().ok()?;
        if !(0..result_width)
            .all(|i| literal_bits.get_bit(i).unwrap() == if next { i == ones } else { i < ones })
        {
            return None;
        }
        Some(Self {
            root,
            sum,
            wide: *wide,
            narrow: *narrow,
            invert,
            modular_signed_order: (result_width == w).then_some(signed),
        })
    }
}

struct Builder<'a> {
    f: &'a mut ir::Fn,
    next_id: usize,
}

impl Builder<'_> {
    fn emit(&mut self, p: Plan) {
        let result = match p.modular_signed_order {
            Some(signed_order) => self.modular_threshold(&p, signed_order),
            None => self.exact_threshold(&p),
        };
        self.f.nodes[p.root].payload = if p.invert {
            NodePayload::Unop(Unop::Not, result)
        } else {
            self.f.get_node(result).payload.clone()
        };
    }

    fn exact_threshold(&mut self, p: &Plan) -> NodeRef {
        let x = p.wide.operand;
        let y = p.narrow.operand;
        let w = self.f.get_node(x).ty.bit_count();
        let k = self.f.get_node(y).ty.bit_count();
        let bias = if p.narrow.signed {
            IrBits::signed_min_value(k)
        } else {
            IrBits::zero(k)
        };
        let sign = p.wide.signed.then(|| {
            let sign = self.slice(x, w - 1, 1);
            self.push(1, NodePayload::Unop(Unop::Not, sign))
        });
        let high = w - k - usize::from(p.wide.signed);
        let any_high = (high > 0).then(|| {
            let bits = self.slice(x, k, high);
            self.push(1, NodePayload::Unop(Unop::OrReduce, bits))
        });
        let low = self.slice(x, 0, k);
        let (biased, increment) = match self.unit_translation(y, k, &bias) {
            Some(pair) => pair,
            None if p.narrow.signed => {
                let mask = self.push(k, NodePayload::Literal(IrValue::from_bits(&bias)));
                (
                    self.push(k, NodePayload::Nary(NaryOp::Xor, vec![y, mask])),
                    false,
                )
            }
            None => (y, false),
        };
        let threshold = self.push(k, NodePayload::Unop(Unop::Not, biased));
        let mut result = self.push(
            1,
            NodePayload::Binop(
                if increment { Binop::Uge } else { Binop::Ugt },
                low,
                threshold,
            ),
        );
        if increment {
            // x > ~(y+1 mod 2^k) is x >= ~y, except when the increment wraps.
            let wrapped = self.push(1, NodePayload::Unop(Unop::AndReduce, biased));
            let ok = self.push(1, NodePayload::Unop(Unop::Not, wrapped));
            result = self.push(1, NodePayload::Nary(NaryOp::And, vec![result, ok]));
        }
        if let Some(high) = any_high {
            result = self.push(1, NodePayload::Nary(NaryOp::Or, vec![high, result]));
        }
        if let Some(nonnegative) = sign {
            result = self.push(1, NodePayload::Nary(NaryOp::And, vec![nonnegative, result]));
        }
        result
    }

    /// Partition x into a narrow low digit and its upper quotient. Only the
    /// upper quotient's two endpoints can wrap across the comparison boundary.
    fn modular_threshold(&mut self, p: &Plan, signed_order: bool) -> NodeRef {
        let x = p.wide.operand;
        let y = p.narrow.operand;
        let w = self.f.get_node(x).ty.bit_count();
        let k = self.f.get_node(y).ty.bit_count();
        let mid_width = w - k - usize::from(signed_order);
        let sign = signed_order.then(|| self.slice(x, w - 1, 1));
        let (mid_any, mid_all) = if mid_width == 0 {
            (self.bit(false), self.bit(true))
        } else {
            let middle = self.slice(x, k, mid_width);
            let any = self.push(1, NodePayload::Unop(Unop::OrReduce, middle));
            let all = self.push(1, NodePayload::Unop(Unop::AndReduce, middle));
            (any, all)
        };
        // u = b + 2^(k-1) for a signed addend, or b for an unsigned addend.
        // x_low + u > 2^k-1 is the majority of their high bits and lower carry.
        let bias = if p.narrow.signed {
            IrBits::signed_min_value(k)
        } else {
            IrBits::zero(k)
        };
        let translation = self.unit_translation(y, k, &bias);
        let (operand, increment) = translation.unwrap_or((y, false));
        let a = self.slice(x, k - 1, 1);
        let mut b = self.slice(operand, k - 1, 1);
        if translation.is_none() && p.narrow.signed {
            b = self.not(b);
        }
        let carry = if k == 1 {
            if increment {
                b = self.not(b);
            }
            self.bit(false)
        } else {
            let xlo = self.slice(x, 0, k - 1);
            let ylo = self.slice(operand, 0, k - 1);
            let not_ylo = self.not(ylo);
            if increment {
                let roll = self.push(1, NodePayload::Unop(Unop::AndReduce, ylo));
                b = self.push(1, NodePayload::Nary(NaryOp::Xor, vec![b, roll]));
                let ge = self.push(1, NodePayload::Binop(Binop::Uge, xlo, not_ylo));
                let no_roll = self.not(roll);
                self.and(ge, no_roll)
            } else {
                self.push(1, NodePayload::Binop(Binop::Ugt, xlo, not_ylo))
            }
        };
        let both = self.and(a, b);
        let either = self.or(a, b);
        let propagate = self.and(either, carry);
        let above = self.or(both, propagate);
        // For a signed addend, crossing the low digit's upper/lower boundary
        // requires all three carry inputs to be one/zero respectively.
        let up = if p.narrow.signed {
            self.and(both, carry)
        } else {
            above
        };
        let wrap_above = self.and(mid_all, up);
        let not_wrapped = self.not(wrap_above);
        let usual = self.or(mid_any, above);
        let positive = self.and(usual, not_wrapped);
        let wrapped_below = if p.narrow.signed {
            let nonzero = self.or(either, carry);
            let down = self.not(nonzero);
            let mid_none = self.not(mid_any);
            self.and(mid_none, down)
        } else {
            self.bit(false)
        };
        match sign {
            Some(selector) => self.push(
                1,
                NodePayload::Sel {
                    selector,
                    cases: vec![positive, wrapped_below],
                    default: None,
                },
            ),
            None => self.or(positive, wrapped_below),
        }
    }

    fn bit(&mut self, value: bool) -> NodeRef {
        self.push(
            1,
            NodePayload::Literal(IrValue::make_ubits(1, u64::from(value)).unwrap()),
        )
    }

    fn not(&mut self, value: NodeRef) -> NodeRef {
        self.push(
            self.f.get_node(value).ty.bit_count(),
            NodePayload::Unop(Unop::Not, value),
        )
    }

    fn and(&mut self, a: NodeRef, b: NodeRef) -> NodeRef {
        self.push(1, NodePayload::Nary(NaryOp::And, vec![a, b]))
    }

    fn or(&mut self, a: NodeRef, b: NodeRef) -> NodeRef {
        self.push(1, NodePayload::Nary(NaryOp::Or, vec![a, b]))
    }

    /// Absorb the sign bias into a same-width modular offset. A surviving
    /// unit offset can move across a strict inequality with a wrap guard.
    fn unit_translation(&self, y: NodeRef, k: usize, bias: &IrBits) -> Option<(NodeRef, bool)> {
        let NodePayload::ExtNaryAdd {
            ref terms,
            arch: None,
        } = self.f.get_node(y).payload
        else {
            return None;
        };
        let [a, b] = terms.as_slice() else {
            return None;
        };
        if a.negated
            || b.negated
            || [a, b]
                .iter()
                .any(|t| self.f.get_node(t.operand).ty.bit_count() != k)
        {
            return None;
        }
        [(a, b), (b, a)].into_iter().find_map(|(value, constant)| {
            let NodePayload::Literal(ref literal) = self.f.get_node(constant.operand).payload
            else {
                return None;
            };
            let offset = literal.to_bits().ok()?.add(bias);
            if offset.is_zero() {
                Some((value.operand, false))
            } else if offset.to_u64().ok() == Some(1) {
                Some((value.operand, true))
            } else {
                None
            }
        })
    }

    fn slice(&mut self, arg: NodeRef, start: usize, width: usize) -> NodeRef {
        self.push(width, NodePayload::BitSlice { arg, start, width })
    }

    fn push(&mut self, width: usize, payload: NodePayload) -> NodeRef {
        let index = self.f.nodes.len();
        self.f.nodes.push(ir::Node {
            text_id: self.next_id,
            name: None,
            ty: Type::Bits(width),
            payload,
            pos: None,
        });
        self.next_id += 1;
        NodeRef { index }
    }
}
