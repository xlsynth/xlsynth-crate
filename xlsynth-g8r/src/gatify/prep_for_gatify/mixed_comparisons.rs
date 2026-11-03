// SPDX-License-Identifier: Apache-2.0

//! Share lower-bit ordering across signed and unsigned comparisons of the same
//! words.

use std::collections::BTreeMap;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};

/// Recognize only live comparisons on the original operand pair.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    use Binop::*;

    if !f.nodes.iter().any(|node| {
        matches!(
            node.payload,
            NodePayload::Binop(Slt | Sle | Sgt | Sge, _, _)
        )
    }) {
        return 0;
    }
    let mut groups = BTreeMap::<(usize, usize), Vec<(usize, Binop, bool)>>::new();
    for (index, node) in f.nodes.iter().enumerate() {
        let NodePayload::Binop(
            op @ (Eq | Ne | Ult | Ule | Ugt | Uge | Slt | Sle | Sgt | Sge),
            a,
            b,
        ) = node.payload
        else {
            continue;
        };
        if a == b
            || !matches!(f.get_node(a).ty, Type::Bits(w) if w >= 2)
            || matches!(f.get_node(a).payload, NodePayload::Literal(_))
            || matches!(f.get_node(b).payload, NodePayload::Literal(_))
        {
            continue;
        }
        let swapped = a.index > b.index;
        groups
            .entry((a.index.min(b.index), a.index.max(b.index)))
            .or_default()
            .push((index, op, swapped));
    }

    let complete = |relations: &[(usize, Binop, bool)]| {
        relations.iter().any(|&(_, op, _)| matches!(op, Eq | Ne))
            && relations
                .iter()
                .any(|&(_, op, _)| matches!(op, Ult | Ule | Ugt | Uge))
            && relations
                .iter()
                .any(|&(_, op, _)| matches!(op, Slt | Sle | Sgt | Sge))
    };
    groups.retain(|_, relations| complete(relations));
    if groups.is_empty() {
        return 0;
    }

    // Only families that can save comparison trees pay for a liveness walk.
    let mut live = vec![false; f.nodes.len()];
    let mut pending: Vec<_> = f.ret_node_ref.into_iter().collect();
    while let Some(nr) = pending.pop() {
        if !std::mem::replace(&mut live[nr.index], true) {
            pending.extend(xlsynth_pir::ir_utils::operands(&f.get_node(nr).payload));
        }
    }
    let mut next_id = super::next_text_id(f);
    let mut rewritten = 0;
    for ((a, b), mut relations) in groups {
        relations.retain(|&(index, _, _)| live[index]);
        if !complete(&relations) {
            continue;
        }
        let (a, b) = (NodeRef { index: a }, NodeRef { index: b });
        let low_width = f.get_node(a).ty.bit_count() - 1;
        let mut push = |width: usize, payload| {
            let nr = NodeRef {
                index: f.nodes.len(),
            };
            f.nodes.push(ir::Node {
                text_id: next_id,
                name: None,
                ty: Type::Bits(width),
                payload,
                pos: None,
            });
            next_id += 1;
            nr
        };
        let sx = push(
            1,
            NodePayload::BitSlice {
                arg: a,
                start: low_width,
                width: 1,
            },
        );
        let sy = push(
            1,
            NodePayload::BitSlice {
                arg: b,
                start: low_width,
                width: 1,
            },
        );
        let sign_diff = push(1, NodePayload::Nary(NaryOp::Xor, vec![sx, sy]));
        let same = push(1, NodePayload::Unop(Unop::Not, sign_diff));
        let lo_a = push(
            low_width,
            NodePayload::BitSlice {
                arg: a,
                start: 0,
                width: low_width,
            },
        );
        let lo_b = push(
            low_width,
            NodePayload::BitSlice {
                arg: b,
                start: 0,
                width: low_width,
            },
        );
        let low_lt = push(1, NodePayload::Binop(Ult, lo_a, lo_b));
        let low_eq = push(1, NodePayload::Binop(Eq, lo_a, lo_b));
        let carry = push(1, NodePayload::Nary(NaryOp::And, vec![same, low_lt]));

        // Equal signs admit the same lower-bit decision. Opposite signs reverse
        // the high bit's contribution to signed versus unsigned ordering.
        let not_x = push(1, NodePayload::Unop(Unop::Not, sx));
        let unsigned_high = push(1, NodePayload::Nary(NaryOp::And, vec![not_x, sy]));
        let ult = push(1, NodePayload::Nary(NaryOp::Or, vec![unsigned_high, carry]));
        let not_y = push(1, NodePayload::Unop(Unop::Not, sy));
        let signed_high = push(1, NodePayload::Nary(NaryOp::And, vec![sx, not_y]));
        let slt = push(1, NodePayload::Nary(NaryOp::Or, vec![signed_high, carry]));
        let eq = push(1, NodePayload::Nary(NaryOp::And, vec![same, low_eq]));

        for (index, op, swapped) in relations {
            let lt = if matches!(op, Slt | Sle | Sgt | Sge) {
                slt
            } else {
                ult
            };
            // Express each observed truth set using the mutually exclusive
            // less and equal results, retaining the original result's users.
            f.nodes[index].payload = match (op, swapped) {
                (Eq, _) => NodePayload::Unop(Unop::Identity, eq),
                (Ne, _) => NodePayload::Unop(Unop::Not, eq),
                (Ult | Slt, false) | (Ugt | Sgt, true) => NodePayload::Unop(Unop::Identity, lt),
                (Uge | Sge, false) | (Ule | Sle, true) => NodePayload::Unop(Unop::Not, lt),
                (Ugt | Sgt, false) | (Ult | Slt, true) => {
                    NodePayload::Nary(NaryOp::Nor, vec![eq, lt])
                }
                (Ule | Sle, false) | (Uge | Sge, true) => {
                    NodePayload::Nary(NaryOp::Or, vec![eq, lt])
                }
                _ => unreachable!("only comparisons are grouped"),
            };
            rewritten += 1;
        }
    }
    rewritten
}
