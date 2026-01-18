// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Binop, Fn as IrFn, NaryOp, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformLocation, compute_users, remap_payload_with};

/// A non-always-equivalent transform that rewires users between sibling adds:
/// - wide: add(zext(a), zext(b)) (zext via zero_ext or concat(0_k, x))
/// - narrow: add(a, b)
#[derive(Debug)]
pub struct RewireUsersToSiblingAddTransform;

#[derive(Clone, Copy)]
struct ExtendInfo {
    base: NodeRef,
    wide_width: usize,
}

#[derive(Clone, Copy)]
struct AddPair {
    narrow: NodeRef,
    wide: NodeRef,
    narrow_width: usize,
    wide_width: usize,
}

impl RewireUsersToSiblingAddTransform {
    fn bits_width(ty: &Type) -> Option<usize> {
        match ty {
            Type::Bits(w) => Some(*w),
            _ => None,
        }
    }

    fn add_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(Binop::Add, lhs, rhs) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn zero_extend_info(f: &IrFn, r: NodeRef) -> Option<ExtendInfo> {
        match &f.get_node(r).payload {
            NodePayload::ZeroExt { arg, new_bit_count } => Some(ExtendInfo {
                base: *arg,
                wide_width: *new_bit_count,
            }),
            NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                let zero = ops[0];
                let base = ops[1];
                let zero_w = Self::bits_width(&f.get_node(zero).ty)?;
                let base_w = Self::bits_width(&f.get_node(base).ty)?;
                let concat_w = Self::bits_width(&f.get_node(r).ty)?;
                if zero_w + base_w != concat_w {
                    return None;
                }
                if !Self::is_zero_literal_node(f, zero, zero_w) {
                    return None;
                }
                Some(ExtendInfo {
                    base,
                    wide_width: concat_w,
                })
            }
            _ => None,
        }
    }

    fn find_pair_for_add(f: &IrFn, add_ref: NodeRef) -> Option<AddPair> {
        let add_node = f.get_node(add_ref);
        let (lhs, rhs) = Self::add_parts(&add_node.payload)?;
        let narrow_width = Self::bits_width(&add_node.ty)?;
        if Self::bits_width(&f.get_node(lhs).ty) != Some(narrow_width)
            || Self::bits_width(&f.get_node(rhs).ty) != Some(narrow_width)
        {
            return None;
        }

        for nr in f.node_refs() {
            if nr == add_ref {
                continue;
            }
            let node = f.get_node(nr);
            let (w_lhs, w_rhs) = match Self::add_parts(&node.payload) {
                Some(v) => v,
                None => continue,
            };
            let Some(w_lhs_ext) = Self::zero_extend_info(f, w_lhs) else {
                continue;
            };
            let Some(w_rhs_ext) = Self::zero_extend_info(f, w_rhs) else {
                continue;
            };
            if w_lhs_ext.wide_width != w_rhs_ext.wide_width {
                continue;
            }
            let wide_width = w_lhs_ext.wide_width;
            if wide_width != narrow_width.saturating_add(1) {
                continue;
            }
            if Self::bits_width(&node.ty) != Some(wide_width) {
                continue;
            }
            let matches_direct = w_lhs_ext.base == lhs && w_rhs_ext.base == rhs;
            let matches_swapped = w_lhs_ext.base == rhs && w_rhs_ext.base == lhs;
            if !matches_direct && !matches_swapped {
                continue;
            }
            return Some(AddPair {
                narrow: add_ref,
                wide: nr,
                narrow_width,
                wide_width,
            });
        }
        None
    }

    fn mk_bit_slice_payload(arg: NodeRef, width: usize) -> NodePayload {
        NodePayload::BitSlice {
            arg,
            start: 0,
            width,
        }
    }

    fn mk_zero_ext_payload(arg: NodeRef, width: usize) -> NodePayload {
        NodePayload::ZeroExt {
            arg,
            new_bit_count: width,
        }
    }

    fn rewire_users(f: &mut IrFn, from: NodeRef, to: NodeRef) {
        let users_map = compute_users(f);
        let Some(users) = users_map.get(&from) else {
            return;
        };
        for user in users {
            let old_payload = f.get_node(*user).payload.clone();
            let new_payload = remap_payload_with(
                &old_payload,
                |(_slot, dep)| {
                    if dep == from { to } else { dep }
                },
            );
            f.get_node_mut(*user).payload = new_payload;
        }
    }
}

impl PirTransform for RewireUsersToSiblingAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::RewireUsersToSiblingAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if Self::find_pair_for_add(f, nr).is_some() {
                out.push(TransformLocation::Node(nr));
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "RewireUsersToSiblingAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let pair = Self::find_pair_for_add(f, target_ref)
            .ok_or_else(|| "RewireUsersToSiblingAddTransform: no matching add pair".to_string())?;

        let users_map = compute_users(f);
        let narrow_users = users_map.get(&pair.narrow).map_or(0, |u| u.len());
        let wide_users = users_map.get(&pair.wide).map_or(0, |u| u.len());

        if narrow_users <= wide_users {
            // Prefer rewiring narrow users to slice of wide.
            let slice_ref = {
                let payload = Self::mk_bit_slice_payload(pair.wide, pair.narrow_width);
                let node = xlsynth_pir::ir::Node {
                    text_id: f.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1,
                    name: None,
                    ty: Type::Bits(pair.narrow_width),
                    payload,
                    pos: None,
                };
                let idx = f.nodes.len();
                f.nodes.push(node);
                NodeRef { index: idx }
            };
            Self::rewire_users(f, pair.narrow, slice_ref);
        } else {
            // Rewire wide users to zero_ext of narrow.
            let ext_ref = {
                let payload = Self::mk_zero_ext_payload(pair.narrow, pair.wide_width);
                let node = xlsynth_pir::ir::Node {
                    text_id: f.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1,
                    name: None,
                    ty: Type::Bits(pair.wide_width),
                    payload,
                    pos: None,
                };
                let idx = f.nodes.len();
                f.nodes.push(node);
                NodeRef { index: idx }
            };
            Self::rewire_users(f, pair.wide, ext_ref);
        }

        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef, Type};
    use xlsynth_pir::ir_parser;

    use super::RewireUsersToSiblingAddTransform;
    use crate::transforms::{PirTransform, TransformLocation};

    fn find_add_node(f: &xlsynth_pir::ir::Fn, width: usize) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _))
                && matches!(f.get_node(nr).ty, Type::Bits(w) if w == width)
            {
                return nr;
            }
        }
        panic!("expected add node");
    }

    #[test]
    fn rewire_narrow_users_to_wide_slice() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  zero_ext.10: bits[9] = zero_ext(a, new_bit_count=9, id=10)
  zero_ext.11: bits[9] = zero_ext(b, new_bit_count=9, id=11)
  add.12: bits[9] = add(zero_ext.10, zero_ext.11, id=12)
  add.20: bits[8] = add(a, b, id=20)
  add.21: bits[8] = add(add.20, a, id=21)
  bit_slice.30: bits[8] = bit_slice(add.12, start=0, width=8, id=30)
  ret add.31: bits[8] = add(bit_slice.30, a, id=31)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let narrow_ref = find_add_node(&f, 8);
        let t = RewireUsersToSiblingAddTransform;
        t.apply(&mut f, &TransformLocation::Node(narrow_ref))
            .expect("apply");

        let mut user_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _))
                && nr != narrow_ref
                && matches!(f.get_node(nr).ty, Type::Bits(8))
            {
                user_ref = Some(nr);
            }
        }
        let user_ref = user_ref.expect("expected user node");
        let NodePayload::Binop(Binop::Add, lhs, _) = f.get_node(user_ref).payload else {
            panic!("expected add user");
        };
        assert!(matches!(
            f.get_node(lhs).payload,
            NodePayload::BitSlice { .. }
        ));
    }

    #[test]
    fn rewire_wide_users_to_narrow_ext() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[9] {
  zero_ext.10: bits[9] = zero_ext(a, new_bit_count=9, id=10)
  zero_ext.11: bits[9] = zero_ext(b, new_bit_count=9, id=11)
  add.12: bits[9] = add(zero_ext.10, zero_ext.11, id=12)
  add.20: bits[8] = add(a, b, id=20)
  add.21: bits[8] = add(add.20, a, id=21)
  add.22: bits[8] = add(add.20, b, id=22)
  ret add.30: bits[9] = add(add.12, zero_ext.10, id=30)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut wide_ref: Option<NodeRef> = None;
        let mut user_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(nr).payload {
                if matches!(f.get_node(nr).ty, Type::Bits(9)) {
                    let lhs_is_zext =
                        matches!(f.get_node(lhs).payload, NodePayload::ZeroExt { .. });
                    let rhs_is_zext =
                        matches!(f.get_node(rhs).payload, NodePayload::ZeroExt { .. });
                    if lhs_is_zext && rhs_is_zext {
                        wide_ref = Some(nr);
                    } else {
                        user_ref = Some(nr);
                    }
                }
            }
        }
        let _wide_ref = wide_ref.expect("expected wide add");
        let user_ref = user_ref.expect("expected wide user");
        let narrow_ref = find_add_node(&f, 8);
        let t = RewireUsersToSiblingAddTransform;
        t.apply(&mut f, &TransformLocation::Node(narrow_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Add, lhs, _) = f.get_node(user_ref).payload else {
            panic!("expected add user");
        };
        assert!(matches!(
            f.get_node(lhs).payload,
            NodePayload::ZeroExt { .. }
        ));
    }
}
