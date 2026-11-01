// SPDX-License-Identifier: Apache-2.0

//! Flattened bit dependencies shared by gate lowering and PIR preparation.

use std::ops::Range;
use xlsynth_pir::ir::{Fn, NaryOp, NodePayload, NodeRef, Unop};

/// Result bit i depends on operand bit start + i. Inversion preserves demand.
pub(super) fn slice_source(f: &Fn, payload: &NodePayload) -> Option<(NodeRef, usize)> {
    match payload {
        NodePayload::Unop(Unop::Identity | Unop::Not, arg) => Some((*arg, 0)),
        NodePayload::BitSlice { arg, start, .. } => Some((*arg, *start)),
        NodePayload::TupleIndex { tuple, index } => {
            let field = f
                .get_node_ty(*tuple)
                .tuple_get_flat_bit_slice_for_index(*index)
                .expect("validated tuple index");
            Some((*tuple, field.start))
        }
        _ => None,
    }
}

/// Yields whole operands and their bit ranges in the result, least significant
/// first.
pub(super) fn concat_sources<'a>(
    f: &'a Fn,
    payload: &'a NodePayload,
) -> Option<impl Iterator<Item = (NodeRef, Range<usize>)> + 'a> {
    let (NodePayload::Tuple(args) | NodePayload::Nary(NaryOp::Concat, args)) = payload else {
        return None;
    };
    Some(args.iter().rev().scan(0, |offset, &arg| {
        let start = *offset;
        *offset += f.get_node_ty(arg).bit_count();
        Some((arg, start..*offset))
    }))
}
