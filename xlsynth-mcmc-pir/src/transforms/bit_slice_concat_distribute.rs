// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform distributing `bit_slice` over 2-operand
/// concat:
///
/// `bit_slice(concat(a,b), start=s, width=w) ↔ ...` (in-a / in-b / straddle)
///
/// Note: concat is interpreted as `concat(msb=a, lsb=b)` (so `b` occupies the
/// low bits).
#[derive(Debug)]
pub struct BitSliceConcatDistributeTransform;

impl BitSliceConcatDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_bit_slice_node(
        f: &mut IrFn,
        out_w: usize,
        arg: NodeRef,
        start: usize,
        width: usize,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_concat_node(f: &mut IrFn, out_w: usize, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::Nary(NaryOp::Concat, vec![a, b]),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for BitSliceConcatDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceConcatDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if let NodePayload::BitSlice { arg, .. } = &f.get_node(nr).payload {
                if matches!(
                    f.get_node(*arg).payload,
                    NodePayload::Nary(NaryOp::Concat, _)
                ) {
                    out.push(TransformLocation::Node(nr));
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BitSliceConcatDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::BitSlice { arg, start, width } = f.get_node(target_ref).payload.clone()
        else {
            return Err(
                "BitSliceConcatDistributeTransform: expected bit_slice payload".to_string(),
            );
        };
        if Self::bits_width(f, target_ref) != Some(width) {
            return Err(
                "BitSliceConcatDistributeTransform: output must be bits[width]".to_string(),
            );
        }

        let NodePayload::Nary(NaryOp::Concat, ops) = f.get_node(arg).payload.clone() else {
            return Err("BitSliceConcatDistributeTransform: expected concat arg".to_string());
        };
        if ops.len() != 2 {
            return Err(
                "BitSliceConcatDistributeTransform: only supports 2-operand concat".to_string(),
            );
        }
        let a = ops[0];
        let b = ops[1];
        let wa = Self::bits_width(f, a)
            .ok_or_else(|| "BitSliceConcatDistributeTransform: a must be bits[wa]".to_string())?;
        let wb = Self::bits_width(f, b)
            .ok_or_else(|| "BitSliceConcatDistributeTransform: b must be bits[wb]".to_string())?;
        let total = wa.saturating_add(wb);
        if start.saturating_add(width) > total {
            return Err("BitSliceConcatDistributeTransform: slice out of bounds".to_string());
        }

        // b occupies low bits [0..wb), a occupies high bits [wb..wb+wa).
        if start.saturating_add(width) <= wb {
            // entirely within b
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: b,
                start,
                width,
            };
            return Ok(());
        }
        if start >= wb {
            // entirely within a
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: a,
                start: start - wb,
                width,
            };
            return Ok(());
        }

        // Straddle: low part from b[start..wb), high part from a[0..(start+width-wb)).
        let width_b = wb - start;
        let width_a = width - width_b;
        let bs_b = Self::mk_bit_slice_node(f, width_b, b, start, width_b);
        let bs_a = Self::mk_bit_slice_node(f, width_a, a, 0, width_a);
        let cat = Self::mk_concat_node(f, width, bs_a, bs_b);
        f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, cat);
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
