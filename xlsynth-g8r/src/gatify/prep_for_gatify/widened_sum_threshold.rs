// SPDX-License-Identifier: Apache-2.0

//! Compare an exact widened sum at an operand's range boundary.

use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::{IrBits, IrValue};

/// For a k-bit unsigned b, x + b > 2^k-1 iff x > ~b. For signed b, bias it
/// by its sign bit first. In either case the resulting threshold is unsigned
/// and fits k bits. An extra result bit must make the outer addition exact.
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
            || result_width <= w
            || (!signed && (wide.signed || narrow.signed))
            || (signed && !wide.signed && result_width - w < 2)
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
        })
    }
}

struct Builder<'a> {
    f: &'a mut ir::Fn,
    next_id: usize,
}

impl Builder<'_> {
    fn emit(&mut self, p: Plan) {
        let x = p.wide.operand;
        let y = p.narrow.operand;
        let w = self.f.get_node(x).ty.bit_count();
        let k = self.f.get_node(y).ty.bit_count();
        let mut bias = vec![false; k];
        bias[k - 1] = p.narrow.signed;
        let bias = IrBits::from_lsb_is_0(&bias);
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
        self.f.nodes[p.root].payload = if p.invert {
            NodePayload::Unop(Unop::Not, result)
        } else {
            self.f.get_node(result).payload.clone()
        };
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
