// SPDX-License-Identifier: Apache-2.0

//! Structural IR isomorphism checking utilities.

use std::collections::HashMap;

use crate::ir::Fn;
use crate::ir_utils::operands;

/// Returns true if two functions are structurally isomorphic when traversed
/// from their return nodes. Names/ids may differ; operators, attributes,
/// types, and ordered operand relationships must match.
pub fn is_ir_isomorphic(lhs: &Fn, rhs: &Fn) -> bool {
    if lhs.get_type() != rhs.get_type() {
        return false;
    }
    let lret = match lhs.ret_node_ref {
        Some(nr) => nr,
        None => return false,
    };
    let rret = match rhs.ret_node_ref {
        Some(nr) => nr,
        None => return false,
    };

    let mut l2r: HashMap<usize, usize> = HashMap::new();
    let mut r2l: HashMap<usize, usize> = HashMap::new();
    let mut stack: Vec<(usize, usize)> = vec![(lret.index, rret.index)];

    while let Some((li, ri)) = stack.pop() {
        if let Some(&mapped) = l2r.get(&li) {
            if mapped != ri {
                return false;
            }
            continue;
        }
        if let Some(&mapped) = r2l.get(&ri) {
            if mapped != li {
                return false;
            }
        }

        let ln = &lhs.nodes[li];
        let rn = &rhs.nodes[ri];
        if ln.ty != rn.ty {
            return false;
        }
        if !payload_structurally_equal(&ln.payload, &rn.payload) {
            return false;
        }

        let lops = operands(&ln.payload);
        let rops = operands(&rn.payload);
        if lops.len() != rops.len() {
            return false;
        }

        l2r.insert(li, ri);
        r2l.insert(ri, li);
        for k in 0..lops.len() {
            stack.push((lops[k].index, rops[k].index));
        }
    }
    true
}

fn payload_structurally_equal(lp: &crate::ir::NodePayload, rp: &crate::ir::NodePayload) -> bool {
    use crate::ir::NodePayload as P;
    match (lp, rp) {
        (P::Nil, P::Nil) => true,
        (P::GetParam(_), P::GetParam(_)) => true,
        (P::Tuple(a), P::Tuple(b)) => a.len() == b.len(),
        (P::Array(a), P::Array(b)) => a.len() == b.len(),
        (P::TupleIndex { index: ia, .. }, P::TupleIndex { index: ib, .. }) => ia == ib,
        (P::Binop(oa, _, _), P::Binop(ob, _, _)) => oa == ob,
        (P::Unop(oa, _), P::Unop(ob, _)) => oa == ob,
        (P::Literal(va), P::Literal(vb)) => va == vb,
        (
            P::SignExt {
                new_bit_count: a, ..
            },
            P::SignExt {
                new_bit_count: b, ..
            },
        ) => a == b,
        (
            P::ZeroExt {
                new_bit_count: a, ..
            },
            P::ZeroExt {
                new_bit_count: b, ..
            },
        ) => a == b,
        (
            P::ArrayUpdate {
                assumed_in_bounds: a,
                indices: ia,
                ..
            },
            P::ArrayUpdate {
                assumed_in_bounds: b,
                indices: ib,
                ..
            },
        ) => a == b && ia.len() == ib.len(),
        (
            P::ArrayIndex {
                assumed_in_bounds: a,
                indices: ia,
                ..
            },
            P::ArrayIndex {
                assumed_in_bounds: b,
                indices: ib,
                ..
            },
        ) => a == b && ia.len() == ib.len(),
        (P::DynamicBitSlice { width: wa, .. }, P::DynamicBitSlice { width: wb, .. }) => wa == wb,
        (
            P::BitSlice {
                start: sa,
                width: wa,
                ..
            },
            P::BitSlice {
                start: sb,
                width: wb,
                ..
            },
        ) => sa == sb && wa == wb,
        (P::BitSliceUpdate { .. }, P::BitSliceUpdate { .. }) => true,
        (
            P::Assert {
                message: ma,
                label: la,
                ..
            },
            P::Assert {
                message: mb,
                label: lb,
                ..
            },
        ) => ma == mb && la == lb,
        (
            P::Trace {
                format: fa,
                operands: oa,
                ..
            },
            P::Trace {
                format: fb,
                operands: ob,
                ..
            },
        ) => fa == fb && oa.len() == ob.len(),
        (P::AfterAll(a), P::AfterAll(b)) => a.len() == b.len(),
        (P::Nary(oa, a), P::Nary(ob, b)) => oa == ob && a.len() == b.len(),
        (
            P::Invoke {
                to_apply: ta,
                operands: oa,
            },
            P::Invoke {
                to_apply: tb,
                operands: ob,
            },
        ) => ta == tb && oa.len() == ob.len(),
        (
            P::PrioritySel {
                cases: ca,
                default: da,
                ..
            },
            P::PrioritySel {
                cases: cb,
                default: db,
                ..
            },
        ) => ca.len() == cb.len() && da.is_some() == db.is_some(),
        (P::OneHotSel { cases: ca, .. }, P::OneHotSel { cases: cb, .. }) => ca.len() == cb.len(),
        (P::OneHot { lsb_prio: la, .. }, P::OneHot { lsb_prio: lb, .. }) => la == lb,
        (
            P::Sel {
                cases: ca,
                default: da,
                ..
            },
            P::Sel {
                cases: cb,
                default: db,
                ..
            },
        ) => ca.len() == cb.len() && da.is_some() == db.is_some(),
        (P::Cover { label: la, .. }, P::Cover { label: lb, .. }) => la == lb,
        (P::Decode { width: wa, .. }, P::Decode { width: wb, .. }) => wa == wb,
        (P::Encode { .. }, P::Encode { .. }) => true,
        (
            P::CountedFor {
                trip_count: ta,
                stride: sa,
                body: ba,
                invariant_args: ia,
                ..
            },
            P::CountedFor {
                trip_count: tb,
                stride: sb,
                body: bb,
                invariant_args: ib,
                ..
            },
        ) => ta == tb && sa == sb && ba == bb && ia.len() == ib.len(),
        _ => false,
    }
}
