// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `add(x, y)` (bits[w]) ↔ `concat(sum_hi, sum_lo)` where:
/// - sum_lo comes from adding the low k bits with one extra carry bit
/// - sum_hi adds the high halves plus the carry-out from the low half
///
/// This is intentionally a structure-changing move aimed at g8r-nodes×depth.
#[derive(Debug)]
pub struct CarrySplitAddTransform;

impl CarrySplitAddTransform {
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

    fn mk_bit_slice_node(f: &mut IrFn, width: usize, arg: NodeRef, start: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(width),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_zero_ext_node(f: &mut IrFn, new_bit_count: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(new_bit_count),
            payload: NodePayload::ZeroExt { arg, new_bit_count },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_binop_bits_node(f: &mut IrFn, op: Binop, w: usize, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Binop(op, a, b),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn choose_k(w: usize) -> usize {
        // A simple deterministic split point: lower half.
        w / 2
    }
}

impl PirTransform for CarrySplitAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CarrySplitAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Add, _, _) => {
                    if let Some(w) = Self::bits_width(f, nr) {
                        if w >= 2 {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                    if matches!(
                        f.get_node(ops[0]).payload,
                        NodePayload::Binop(Binop::Add, _, _)
                    ) || matches!(
                        f.get_node(ops[0]).payload,
                        NodePayload::Binop(Binop::Sub, _, _)
                    ) || matches!(
                        f.get_node(ops[0]).payload,
                        NodePayload::Binop(Binop::Add, _, _)
                    ) {
                        out.push(TransformLocation::Node(nr));
                    } else {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "CarrySplitAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "CarrySplitAddTransform: output must be bits[w]".to_string())?;
        if w < 2 {
            return Err("CarrySplitAddTransform: requires w >= 2".to_string());
        }
        let k = Self::choose_k(w);
        if k == 0 || k >= w {
            return Err("CarrySplitAddTransform: invalid split point".to_string());
        }

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            // Expand: add(x,y) -> concat(sum_hi,sum_lo) with explicit carry.
            NodePayload::Binop(Binop::Add, x, y) => {
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, y) != Some(w) {
                    return Err("CarrySplitAddTransform: operands must be bits[w]".to_string());
                }

                let hi_w = w - k;
                let lo_w = k;

                // Low half with extra carry bit.
                let x_lo = Self::mk_bit_slice_node(f, lo_w, x, 0);
                let y_lo = Self::mk_bit_slice_node(f, lo_w, y, 0);
                let x_lo_ext = Self::mk_zero_ext_node(f, lo_w + 1, x_lo);
                let y_lo_ext = Self::mk_zero_ext_node(f, lo_w + 1, y_lo);
                let sum_lo_ext =
                    Self::mk_binop_bits_node(f, Binop::Add, lo_w + 1, x_lo_ext, y_lo_ext);
                let sum_lo = Self::mk_bit_slice_node(f, lo_w, sum_lo_ext, 0);
                let carry = Self::mk_bit_slice_node(f, 1, sum_lo_ext, lo_w);

                // High half and carry.
                let x_hi = Self::mk_bit_slice_node(f, hi_w, x, lo_w);
                let y_hi = Self::mk_bit_slice_node(f, hi_w, y, lo_w);
                let sum_hi0 = Self::mk_binop_bits_node(f, Binop::Add, hi_w, x_hi, y_hi);
                let carry_ext = Self::mk_zero_ext_node(f, hi_w, carry);
                let sum_hi = Self::mk_binop_bits_node(f, Binop::Add, hi_w, sum_hi0, carry_ext);

                // Concatenate (hi is more-significant bits).
                f.get_node_mut(target_ref).payload =
                    NodePayload::Nary(NaryOp::Concat, vec![sum_hi, sum_lo]);
                Ok(())
            }

            // Fold: concat(sum_hi,sum_lo) matching our expansion pattern -> add(x,y).
            NodePayload::Nary(NaryOp::Concat, ops) => {
                if ops.len() != 2 {
                    return Err(
                        "CarrySplitAddTransform: only supports 2-operand concat".to_string()
                    );
                }
                let sum_hi = ops[0];
                let sum_lo = ops[1];

                let hi_w = w - k;
                let lo_w = k;

                let NodePayload::BitSlice {
                    arg: sum_lo_ext,
                    start: lo_start,
                    width: lo_width,
                } = f.get_node(sum_lo).payload
                else {
                    return Err(
                        "CarrySplitAddTransform: expected sum_lo to be bit_slice".to_string()
                    );
                };
                if lo_start != 0 || lo_width != lo_w {
                    return Err("CarrySplitAddTransform: sum_lo slice mismatch".to_string());
                }
                if Self::bits_width(f, sum_lo_ext) != Some(lo_w + 1) {
                    return Err("CarrySplitAddTransform: expected sum_lo_ext bits[k+1]".to_string());
                }
                let NodePayload::Binop(Binop::Add, x_lo_ext, y_lo_ext) =
                    f.get_node(sum_lo_ext).payload
                else {
                    return Err("CarrySplitAddTransform: expected sum_lo_ext to be add".to_string());
                };
                let NodePayload::ZeroExt {
                    arg: x_lo,
                    new_bit_count: xlo_n,
                } = f.get_node(x_lo_ext).payload
                else {
                    return Err(
                        "CarrySplitAddTransform: expected x_lo_ext to be zero_ext".to_string()
                    );
                };
                let NodePayload::ZeroExt {
                    arg: y_lo,
                    new_bit_count: ylo_n,
                } = f.get_node(y_lo_ext).payload
                else {
                    return Err(
                        "CarrySplitAddTransform: expected y_lo_ext to be zero_ext".to_string()
                    );
                };
                if xlo_n != lo_w + 1 || ylo_n != lo_w + 1 {
                    return Err("CarrySplitAddTransform: low zero_ext width mismatch".to_string());
                }
                let NodePayload::BitSlice {
                    arg: x,
                    start: xlo_s,
                    width: xlo_w,
                } = f.get_node(x_lo).payload
                else {
                    return Err("CarrySplitAddTransform: expected x_lo to be bit_slice".to_string());
                };
                let NodePayload::BitSlice {
                    arg: y,
                    start: ylo_s,
                    width: ylo_w,
                } = f.get_node(y_lo).payload
                else {
                    return Err("CarrySplitAddTransform: expected y_lo to be bit_slice".to_string());
                };
                if xlo_s != 0 || ylo_s != 0 || xlo_w != lo_w || ylo_w != lo_w {
                    return Err("CarrySplitAddTransform: low slice mismatch".to_string());
                }
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, y) != Some(w) {
                    return Err("CarrySplitAddTransform: x/y must be bits[w]".to_string());
                }

                // Carry must come from bit_slice(sum_lo_ext, start=k, width=1), and be
                // zero-extended.
                let NodePayload::Binop(Binop::Add, sum_hi0, carry_ext) = f.get_node(sum_hi).payload
                else {
                    return Err("CarrySplitAddTransform: expected sum_hi to be add".to_string());
                };
                if Self::bits_width(f, sum_hi0) != Some(hi_w)
                    || Self::bits_width(f, carry_ext) != Some(hi_w)
                {
                    return Err(
                        "CarrySplitAddTransform: sum_hi operands width mismatch".to_string()
                    );
                }
                let NodePayload::ZeroExt {
                    arg: carry,
                    new_bit_count: carry_n,
                } = f.get_node(carry_ext).payload
                else {
                    return Err(
                        "CarrySplitAddTransform: expected carry_ext to be zero_ext".to_string()
                    );
                };
                if carry_n != hi_w {
                    return Err("CarrySplitAddTransform: carry_ext width mismatch".to_string());
                }
                let NodePayload::BitSlice {
                    arg: carry_src,
                    start: carry_s,
                    width: carry_w,
                } = f.get_node(carry).payload
                else {
                    return Err(
                        "CarrySplitAddTransform: expected carry to be bit_slice".to_string()
                    );
                };
                if carry_src != sum_lo_ext || carry_s != lo_w || carry_w != 1 {
                    return Err("CarrySplitAddTransform: carry slice mismatch".to_string());
                }

                let NodePayload::Binop(Binop::Add, x_hi, y_hi) = f.get_node(sum_hi0).payload else {
                    return Err("CarrySplitAddTransform: expected sum_hi0 to be add".to_string());
                };
                let NodePayload::BitSlice {
                    arg: x2,
                    start: xhi_s,
                    width: xhi_w,
                } = f.get_node(x_hi).payload
                else {
                    return Err("CarrySplitAddTransform: expected x_hi to be bit_slice".to_string());
                };
                let NodePayload::BitSlice {
                    arg: y2,
                    start: yhi_s,
                    width: yhi_w,
                } = f.get_node(y_hi).payload
                else {
                    return Err("CarrySplitAddTransform: expected y_hi to be bit_slice".to_string());
                };
                if x2 != x || y2 != y {
                    return Err(
                        "CarrySplitAddTransform: hi/lo slices must refer to same x/y".to_string(),
                    );
                }
                if xhi_s != lo_w || yhi_s != lo_w || xhi_w != hi_w || yhi_w != hi_w {
                    return Err("CarrySplitAddTransform: high slice mismatch".to_string());
                }

                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, x, y);
                Ok(())
            }

            _ => Err("CarrySplitAddTransform: expected add(..) or concat(..)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
