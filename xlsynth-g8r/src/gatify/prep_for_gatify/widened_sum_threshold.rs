// SPDX-License-Identifier: Apache-2.0

//! Compare an exact widened sum at an operand's range boundary.

use crate::aig::gate::PirNodeIds;
use std::collections::HashMap;
use xlsynth_pir::ir::Binop::*;
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::{IrBits, IrValue};

/// For a k-bit unsigned b, x + b > 2^k-1 iff x > ~b. For signed b, bias it
/// by its sign bit first. In either case the resulting threshold is unsigned
/// and fits k bits. An extra result bit must make the outer addition exact.
pub(super) fn rewrite(
    f: &mut ir::Fn,
    mut origins: Option<&mut HashMap<usize, PirNodeIds>>,
) -> usize {
    let mut uses = None;
    let plans: Vec<_> = (0..f.nodes.len())
        .filter_map(|i| Plan::recognize(f, i, &mut uses))
        .collect();
    if plans.is_empty() {
        return 0;
    }
    let mut max_id = f.nodes.iter().map(|n| n.text_id).max().unwrap();
    let mut count = 0;
    for plan in plans {
        let root = plan.root;
        let mut builder = Builder {
            f,
            helpers: Vec::new(),
        };
        let replacement = builder.emit(plan);
        let helpers = builder.helpers;
        // Reserve every helper ID before changing the predicate or its
        // provenance.
        let Some(last_id) = max_id.checked_add(helpers.len()) else {
            continue;
        };
        let ids = max_id + 1..=last_id;
        if let Some(origins) = origins.as_deref_mut() {
            let observer = u32::try_from(f.nodes[root].text_id).expect("node id too large for u32");
            origins.extend(
                ids.clone()
                    .map(|id| (id, PirNodeIds::from_slice(&[observer]))),
            );
        }
        for (text_id, (width, payload)) in ids.zip(helpers) {
            f.nodes.push(ir::Node {
                text_id,
                name: None,
                ty: Type::Bits(width),
                payload,
                pos: None,
            });
        }
        let predicate = &mut f.nodes[root];
        // PIR reserves names ending in a numeric suffix for <opcode>.<id>.
        if let Some((_, suffix)) = predicate.name.as_deref().and_then(|s| s.rsplit_once('.'))
            && !suffix.is_empty()
            && suffix.bytes().all(|c| c.is_ascii_digit())
        {
            predicate.name = None;
        }
        predicate.payload = replacement;
        max_id = last_id;
        count += 1;
    }
    // The old wide sum is now dead. The enclosing final compaction restores
    // topological order after inserting the narrow predicate.
    if count > 0 {
        super::mark_dead_nodes_as_nil(f);
    }
    count
}

/// Operands and polarity of a single-use sum comparison that can be narrowed.
struct Plan {
    root: usize,
    wide: ExtNaryAddTerm,
    narrow: ExtNaryAddTerm,
    invert: bool,
}

impl Plan {
    /// Matches on the unchanged graph and shares its use counts across
    /// candidates.
    fn recognize(f: &ir::Fn, root: usize, uses: &mut Option<Vec<usize>>) -> Option<Self> {
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
        // A shared sum can have many observers of the same wide literal.
        if uses.get_or_insert_with(|| super::get_use_counts(f))[sum.index] != 1 {
            return None;
        }
        let ones = k - usize::from(narrow.signed);
        let literal_bits = endpoint.as_bits().ok()?;
        if !(0..result_width)
            .all(|i| literal_bits.get_bit(i).unwrap() == if next { i == ones } else { i < ones })
        {
            return None;
        }
        Some(Self {
            root,
            wide: *wide,
            narrow: *narrow,
            invert,
        })
    }
}

/// Stages one predicate's helpers at their eventual node indices.
struct Builder<'a> {
    f: &'a ir::Fn,
    helpers: Vec<(usize, NodePayload)>,
}

impl Builder<'_> {
    /// Stages the narrower comparison and returns the original predicate's
    /// replacement.
    fn emit(&mut self, p: Plan) -> NodePayload {
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
        if p.invert {
            NodePayload::Unop(Unop::Not, result)
        } else {
            // The last emitted operation fits directly in the predicate's slot.
            self.helpers.pop().expect("emitted the result last").1
        }
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
        let index = self.f.nodes.len() + self.helpers.len();
        self.helpers.push((width, payload));
        NodeRef { index }
    }
}
