// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Binop, Fn as IrFn, Node, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformCandidate, TransformLocation};

/// Distributes low-bit truncation over add/sub (and folds back).
///
/// Supported reversible forms:
/// - `bit_slice(add(x, y), start=0, width=k) ↔ add(bit_slice(x,0,k),
///   bit_slice(y,0,k))`
/// - `bit_slice(sub(x, y), start=0, width=k) ↔ sub(bit_slice(x,0,k),
///   bit_slice(y,0,k))`
#[derive(Debug)]
pub struct BitSliceAddSubDistributeTransform;

impl BitSliceAddSubDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(ty: &Type) -> Option<usize> {
        match ty {
            Type::Bits(w) => Some(*w),
            _ => None,
        }
    }

    fn mk_bit_slice_node(f: &mut IrFn, arg: NodeRef, start: usize, width: usize) -> NodeRef {
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

    fn mk_binop_node(f: &mut IrFn, op: Binop, ty: Type, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn add_sub_op(payload: &NodePayload) -> Option<(Binop, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(op @ Binop::Add, lhs, rhs)
            | NodePayload::Binop(op @ Binop::Sub, lhs, rhs) => Some((*op, *lhs, *rhs)),
            _ => None,
        }
    }
}

impl PirTransform for BitSliceAddSubDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceAddSubDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::BitSlice { arg, start, width } if *start == 0 => {
                    let Some((_, lhs, rhs)) = Self::add_sub_op(&f.get_node(*arg).payload) else {
                        continue;
                    };
                    let Some(k) = Self::bits_width(&node.ty) else {
                        continue;
                    };
                    if k != *width {
                        continue;
                    }
                    let Some(wl) = Self::bits_width(&f.get_node(lhs).ty) else {
                        continue;
                    };
                    let Some(wr) = Self::bits_width(&f.get_node(rhs).ty) else {
                        continue;
                    };
                    if wl == wr && k <= wl {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
                    }
                }
                NodePayload::Binop(op @ Binop::Add, lhs, rhs)
                | NodePayload::Binop(op @ Binop::Sub, lhs, rhs) => {
                    let Some(k) = Self::bits_width(&node.ty) else {
                        continue;
                    };
                    let lhs_node = f.get_node(*lhs);
                    let rhs_node = f.get_node(*rhs);
                    let NodePayload::BitSlice {
                        arg: x,
                        start: sx,
                        width: wx,
                    } = lhs_node.payload.clone()
                    else {
                        continue;
                    };
                    let NodePayload::BitSlice {
                        arg: y,
                        start: sy,
                        width: wy,
                    } = rhs_node.payload.clone()
                    else {
                        continue;
                    };
                    if *op != Binop::Add && *op != Binop::Sub {
                        continue;
                    }
                    if sx != 0 || sy != 0 || wx != k || wy != k {
                        continue;
                    }
                    let Some(wx_src) = Self::bits_width(&f.get_node(x).ty) else {
                        continue;
                    };
                    let Some(wy_src) = Self::bits_width(&f.get_node(y).ty) else {
                        continue;
                    };
                    if wx_src == wy_src && k <= wx_src {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
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
                    "BitSliceAddSubDistributeTransform: expected TransformLocation::Node"
                        .to_string(),
                );
            }
        };

        let target_ty = f.get_node(target_ref).ty.clone();
        let Some(k) = Self::bits_width(&target_ty) else {
            return Err("BitSliceAddSubDistributeTransform: target must be bits[k]".to_string());
        };
        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::BitSlice { arg, start, width } => {
                if start != 0 || width != k {
                    return Err(
                        "BitSliceAddSubDistributeTransform: expected low bit_slice width=k"
                            .to_string(),
                    );
                }
                let (op, lhs, rhs) =
                    Self::add_sub_op(&f.get_node(arg).payload).ok_or_else(|| {
                        "BitSliceAddSubDistributeTransform: expected add/sub".to_string()
                    })?;
                let lhs_slice = Self::mk_bit_slice_node(f, lhs, 0, k);
                let rhs_slice = Self::mk_bit_slice_node(f, rhs, 0, k);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(op, lhs_slice, rhs_slice);
                Ok(())
            }
            NodePayload::Binop(op @ Binop::Add, lhs, rhs)
            | NodePayload::Binop(op @ Binop::Sub, lhs, rhs) => {
                let lhs_node = f.get_node(lhs).payload.clone();
                let rhs_node = f.get_node(rhs).payload.clone();
                let NodePayload::BitSlice {
                    arg: x,
                    start: sx,
                    width: wx,
                } = lhs_node
                else {
                    return Err(
                        "BitSliceAddSubDistributeTransform: lhs must be low bit_slice".to_string(),
                    );
                };
                let NodePayload::BitSlice {
                    arg: y,
                    start: sy,
                    width: wy,
                } = rhs_node
                else {
                    return Err(
                        "BitSliceAddSubDistributeTransform: rhs must be low bit_slice".to_string(),
                    );
                };
                if sx != 0 || sy != 0 || wx != k || wy != k {
                    return Err(
                        "BitSliceAddSubDistributeTransform: expected matching low slices"
                            .to_string(),
                    );
                }
                let w = Self::bits_width(&f.get_node(x).ty).ok_or_else(|| {
                    "BitSliceAddSubDistributeTransform: x must be bits[w]".to_string()
                })?;
                let wy_src = Self::bits_width(&f.get_node(y).ty).ok_or_else(|| {
                    "BitSliceAddSubDistributeTransform: y must be bits[w]".to_string()
                })?;
                if w != wy_src || k > w {
                    return Err(
                        "BitSliceAddSubDistributeTransform: source widths must match".to_string(),
                    );
                }
                let wide_ref = Self::mk_binop_node(f, op, Type::Bits(w), x, y);
                f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                    arg: wide_ref,
                    start: 0,
                    width: k,
                };
                Ok(())
            }
            _ => Err("BitSliceAddSubDistributeTransform: unsupported target".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitSliceAddSubDistributeTransform;
    use crate::transforms::{PirTransform, TransformLocation};
    use xlsynth_pir::ir::{Binop, NodePayload};
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_low_slice_of_add() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[4] {
  add.3: bits[8] = add(x, y, id=3)
  ret bit_slice.4: bits[4] = bit_slice(add.3, start=0, width=4, id=4)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let mut t = BitSliceAddSubDistributeTransform;
        let cand = t.find_candidates(&f).pop().expect("candidate");
        let target = match cand.location.clone() {
            TransformLocation::Node(nr) => nr,
            _ => unreachable!(),
        };
        t.apply(&mut f, &cand.location).expect("apply");
        match &f.get_node(target).payload {
            NodePayload::Binop(Binop::Add, _, _) => {}
            other => panic!("unexpected payload: {:?}", other),
        }
    }

    #[test]
    fn folds_narrow_sub_back_to_wide_slice() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[4] {
  bit_slice.3: bits[4] = bit_slice(x, start=0, width=4, id=3)
  bit_slice.4: bits[4] = bit_slice(y, start=0, width=4, id=4)
  ret sub.5: bits[4] = sub(bit_slice.3, bit_slice.4, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let mut t = BitSliceAddSubDistributeTransform;
        let cand = t.find_candidates(&f).pop().expect("candidate");
        let target = match cand.location.clone() {
            TransformLocation::Node(nr) => nr,
            _ => unreachable!(),
        };
        t.apply(&mut f, &cand.location).expect("apply");
        match &f.get_node(target).payload {
            NodePayload::BitSlice { start, width, .. } => {
                assert_eq!(*start, 0);
                assert_eq!(*width, 4);
            }
            other => panic!("unexpected payload: {:?}", other),
        }
    }
}
