// SPDX-License-Identifier: Apache-2.0

//! Share lower-bit ordering across signed and unsigned comparisons of the same
//! words.

use crate::aig::gate::PirNodeIds;
use std::collections::{BTreeMap, HashMap};
use xlsynth_pir::ir::Binop::*;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils;

/// The shared operand order and the equality merges needed by its observers.
struct OrderPlan {
    reverse: bool,
    unsigned_merge: bool,
    signed_merge: bool,
}

/// Chooses the shared strict order that needs fewer distinct merges with
/// equality.
fn plan_order(relations: &[(usize, Binop, bool)]) -> OrderPlan {
    let mut forward = [false; 2];
    let mut reverse = [false; 2];
    for &(_, op, swapped) in relations {
        // A non-strict order uses the complement of the opposite strict order.
        let (signed, wants_reverse) = match op {
            Ult | Uge => (false, false),
            Ugt | Ule => (false, true),
            Slt | Sge => (true, false),
            Sgt | Sle => (true, true),
            Eq | Ne => continue,
            _ => unreachable!("only comparisons are grouped"),
        };
        if swapped ^ wants_reverse {
            reverse[usize::from(signed)] = true;
        } else {
            forward[usize::from(signed)] = true;
        }
    }
    // Keep operand index order on ties. Each opposite order needs one merge
    // in PIR, regardless of fanout or gate hashing.
    let prefer_reverse =
        reverse.into_iter().filter(|&x| x).count() > forward.into_iter().filter(|&x| x).count();
    let [unsigned_merge, signed_merge] = if prefer_reverse { forward } else { reverse };
    OrderPlan {
        reverse: prefer_reverse,
        unsigned_merge,
        signed_merge,
    }
}

/// Attributes only this group's helpers to the observer, stopping at the words.
fn record_helper_origins(
    f: &ir::Fn,
    first_helper: usize,
    observer: &ir::Node,
    origins: &mut HashMap<usize, PirNodeIds>,
) {
    let id = u32::try_from(observer.text_id).expect("node id too large for u32");
    let mut pending = ir_utils::operands(&observer.payload);
    while let Some(nr) = pending.pop() {
        if nr.index < first_helper {
            continue;
        }
        let node = f.get_node(nr);
        let ids = origins.entry(node.text_id).or_default();
        if ids.last() != Some(&id) {
            ids.push(id);
            pending.extend(ir_utils::operands(&node.payload));
        }
    }
}

/// Recognize only live comparisons on the original operand pair.
pub(super) fn rewrite(
    f: &mut ir::Fn,
    mut origins: Option<&mut HashMap<usize, PirNodeIds>>,
) -> usize {
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
            pending.extend(ir_utils::operands(&f.get_node(nr).payload));
        }
    }
    let mut max_id = super::max_text_id(f);
    let mut rewritten = 0;
    for ((a, b), mut relations) in groups {
        relations.retain(|&(index, _, _)| live[index]);
        if !complete(&relations) {
            continue;
        }
        // Reserve the entire helper group before touching the function.
        let plan = plan_order(&relations);
        let helper_count = 16 + usize::from(plan.unsigned_merge) + usize::from(plan.signed_merge);
        if helper_count > super::remaining_text_ids(max_id) {
            // A later group may need fewer order merges.
            continue;
        }
        let last_id = max_id + helper_count;
        let mut ids = max_id + 1..=last_id;
        max_id = last_id;
        let (a, b) = if plan.reverse { (b, a) } else { (a, b) };
        let (a, b) = (NodeRef { index: a }, NodeRef { index: b });
        let low_width = f.get_node(a).ty.bit_count() - 1;
        let first_helper = f.nodes.len();
        let mut push = |width: usize, payload| {
            let nr = NodeRef {
                index: f.nodes.len(),
            };
            f.nodes.push(ir::Node {
                text_id: ids.next().expect("reserved a text ID for each helper"),
                name: None,
                ty: Type::Bits(width),
                payload,
                pos: None,
            });
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
        let ule = plan
            .unsigned_merge
            .then(|| push(1, NodePayload::Nary(NaryOp::Or, vec![eq, ult])));
        let sle = plan
            .signed_merge
            .then(|| push(1, NodePayload::Nary(NaryOp::Or, vec![eq, slt])));
        debug_assert!(ids.next().is_none());

        if origins.is_some() {
            // Visit in text-ID order so every helper accumulates a sorted set.
            relations.sort_unstable_by_key(|&(index, _, _)| f.nodes[index].text_id);
        }
        for (index, op, swapped) in relations {
            let (lt, le) = if matches!(op, Slt | Sle | Sgt | Sge) {
                (slt, sle)
            } else {
                (ult, ule)
            };
            // Express each observed truth set using the mutually exclusive
            // less and equal results, retaining the original result's users.
            let (unop, arg) = match (op, swapped ^ plan.reverse) {
                (Eq, _) => (Unop::Identity, eq),
                (Ne, _) => (Unop::Not, eq),
                (Ult | Slt, false) | (Ugt | Sgt, true) => (Unop::Identity, lt),
                (Uge | Sge, false) | (Ule | Sle, true) => (Unop::Not, lt),
                (Ugt | Sgt, false) | (Ult | Slt, true) => {
                    (Unop::Not, le.expect("planned opposite order"))
                }
                (Ule | Sle, false) | (Uge | Sge, true) => {
                    (Unop::Identity, le.expect("planned opposite order"))
                }
                _ => unreachable!("only comparisons are grouped"),
            };
            let observer = &mut f.nodes[index];
            if observer
                .name
                .as_deref()
                .and_then(|name| name.rsplit_once('.'))
                .is_some_and(|(_, digits)| {
                    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
                })
            {
                // A validated op.<digits> name belongs to the old operator.
                observer.name = None;
            }
            observer.payload = NodePayload::Unop(unop, arg);
            if let Some(origins) = origins.as_deref_mut() {
                record_helper_origins(f, first_helper, &f.nodes[index], origins);
            }
            rewritten += 1;
        }
    }
    rewritten
}
