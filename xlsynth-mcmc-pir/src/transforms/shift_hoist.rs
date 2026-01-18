// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Binop, Fn as IrFn, Node, NodePayload, NodeRef, Type, Unop};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A non-always-equivalent transform that hoists ops across shifts.
///
/// - `unop(shift(x,k)) ↔ shift(unop(x),k)`
/// - `binop(shift(x,k), t) ↔ shift(binop(x,t), k)`
/// - `binop(t, shift(x,k)) ↔ shift(binop(t,x), k)`
#[derive(Debug)]
pub struct ShiftHoistTransform;

impl ShiftHoistTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_bits_type(ty: &Type) -> Option<usize> {
        match ty {
            Type::Bits(w) => Some(*w),
            _ => None,
        }
    }

    fn shift_parts(payload: &NodePayload) -> Option<(Binop, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(op @ Binop::Shll, x, k)
            | NodePayload::Binop(op @ Binop::Shrl, x, k)
            | NodePayload::Binop(op @ Binop::Shra, x, k) => Some((*op, *x, *k)),
            _ => None,
        }
    }

    fn mk_unop_node(f: &mut IrFn, op: Unop, ty: Type, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Unop(op, arg),
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

    fn mk_shift_node(f: &mut IrFn, op: Binop, ty: Type, x: NodeRef, k: NodeRef) -> NodeRef {
        Self::mk_binop_node(f, op, ty, x, k)
    }
}

impl PirTransform for ShiftHoistTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ShiftHoist
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::Unop(_, arg) => {
                    let arg_node = f.get_node(*arg);
                    let Some((_op, x, _k)) = Self::shift_parts(&arg_node.payload) else {
                        continue;
                    };
                    let Some(w) = Self::is_bits_type(&node.ty) else {
                        continue;
                    };
                    if Self::is_bits_type(&arg_node.ty) != Some(w) {
                        continue;
                    }
                    if Self::is_bits_type(&f.get_node(x).ty) != Some(w) {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Binop(_, lhs, rhs) => {
                    let lhs_node = f.get_node(*lhs);
                    let rhs_node = f.get_node(*rhs);
                    let lhs_shift = Self::shift_parts(&lhs_node.payload);
                    let rhs_shift = Self::shift_parts(&rhs_node.payload);
                    if lhs_shift.is_some() == rhs_shift.is_some() {
                        continue;
                    }
                    let Some(w) = Self::is_bits_type(&node.ty) else {
                        continue;
                    };
                    if Self::is_bits_type(&lhs_node.ty) != Some(w)
                        || Self::is_bits_type(&rhs_node.ty) != Some(w)
                    {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }
                payload => {
                    let Some((_, x, _k)) = Self::shift_parts(payload) else {
                        continue;
                    };
                    let Some(w) = Self::is_bits_type(&node.ty) else {
                        continue;
                    };
                    if Self::is_bits_type(&f.get_node(x).ty) != Some(w) {
                        continue;
                    }
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
                    "ShiftHoistTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        let target_ty = f.get_node(target_ref).ty.clone();
        let Some(w) = Self::is_bits_type(&target_ty) else {
            return Err("ShiftHoistTransform: target must be bits[w]".to_string());
        };

        if let Some((shift_op, x, k)) = Self::shift_parts(&target_payload) {
            let x_ty = f.get_node(x).ty.clone();
            if Self::is_bits_type(&x_ty) != Some(w) {
                return Err("ShiftHoistTransform: shift arg must be bits[w]".to_string());
            }
            let x_payload = f.get_node(x).payload.clone();
            match x_payload {
                NodePayload::Unop(op, inner) => {
                    let inner_ty = f.get_node(inner).ty.clone();
                    if Self::is_bits_type(&inner_ty) != Some(w) {
                        return Err("ShiftHoistTransform: unop input must be bits[w]".to_string());
                    }
                    let new_shift = Self::mk_shift_node(f, shift_op, target_ty.clone(), inner, k);
                    f.get_node_mut(target_ref).payload = NodePayload::Unop(op, new_shift);
                    Ok(())
                }
                NodePayload::Binop(op, a, b) => {
                    let a_ty = f.get_node(a).ty.clone();
                    let b_ty = f.get_node(b).ty.clone();
                    if Self::is_bits_type(&a_ty) != Some(w) || Self::is_bits_type(&b_ty) != Some(w)
                    {
                        return Err(
                            "ShiftHoistTransform: binop operands must be bits[w]".to_string()
                        );
                    }
                    let new_shift = Self::mk_shift_node(f, shift_op, target_ty.clone(), a, k);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(op, new_shift, b);
                    Ok(())
                }
                _ => Err("ShiftHoistTransform: shift arg must be unop/binop".to_string()),
            }
        } else {
            match target_payload {
                NodePayload::Unop(op, arg) => {
                    let arg_node = f.get_node(arg);
                    let Some((shift_op, x, k)) = Self::shift_parts(&arg_node.payload) else {
                        return Err("ShiftHoistTransform: expected unop(shift(x,k))".to_string());
                    };
                    if Self::is_bits_type(&arg_node.ty) != Some(w)
                        || Self::is_bits_type(&f.get_node(x).ty) != Some(w)
                    {
                        return Err(
                            "ShiftHoistTransform: unop and shift must preserve bits width"
                                .to_string(),
                        );
                    }
                    let new_unop = Self::mk_unop_node(f, op, target_ty.clone(), x);
                    let new_shift = Self::mk_shift_node(f, shift_op, target_ty, new_unop, k);
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Unop(Unop::Identity, new_shift);
                    Ok(())
                }
                NodePayload::Binop(op, lhs, rhs) => {
                    let lhs_payload = f.get_node(lhs).payload.clone();
                    let rhs_payload = f.get_node(rhs).payload.clone();
                    let lhs_ty = f.get_node(lhs).ty.clone();
                    let rhs_ty = f.get_node(rhs).ty.clone();
                    let lhs_shift = Self::shift_parts(&lhs_payload);
                    let rhs_shift = Self::shift_parts(&rhs_payload);
                    if lhs_shift.is_some() == rhs_shift.is_some() {
                        return Err(
                            "ShiftHoistTransform: expected exactly one shift operand".to_string()
                        );
                    }
                    if Self::is_bits_type(&lhs_ty) != Some(w)
                        || Self::is_bits_type(&rhs_ty) != Some(w)
                    {
                        return Err(
                            "ShiftHoistTransform: binop operands must match bits width".to_string()
                        );
                    }
                    if let Some((shift_op, x, k)) = lhs_shift {
                        let new_binop = Self::mk_binop_node(f, op, target_ty.clone(), x, rhs);
                        let new_shift = Self::mk_shift_node(f, shift_op, target_ty, new_binop, k);
                        f.get_node_mut(target_ref).payload =
                            NodePayload::Unop(Unop::Identity, new_shift);
                        Ok(())
                    } else if let Some((shift_op, x, k)) = rhs_shift {
                        let new_binop = Self::mk_binop_node(f, op, target_ty.clone(), lhs, x);
                        let new_shift = Self::mk_shift_node(f, shift_op, target_ty, new_binop, k);
                        f.get_node_mut(target_ref).payload =
                            NodePayload::Unop(Unop::Identity, new_shift);
                        Ok(())
                    } else {
                        Err("ShiftHoistTransform: missing shift operand".to_string())
                    }
                }
                _ => Err("ShiftHoistTransform: expected unop or binop target".to_string()),
            }
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef, Unop};
    use xlsynth_pir::ir_parser;

    use super::ShiftHoistTransform;
    use crate::transforms::{PirTransform, TransformLocation};

    fn find_unop_node(f: &xlsynth_pir::ir::Fn, op: Unop) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(o, _) if o == op) {
                return nr;
            }
        }
        panic!("expected unop node");
    }

    fn find_binop_node(f: &xlsynth_pir::ir::Fn, op: Binop) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(o, _, _) if o == op) {
                return nr;
            }
        }
        panic!("expected binop node");
    }

    #[test]
    fn shift_hoist_unop_forward() {
        let ir_text = r#"fn t(x: bits[8] id=1, k: bits[3] id=2) -> bits[8] {
  shll.10: bits[8] = shll(x, k, id=10)
  ret not.11: bits[8] = not(shll.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let not_ref = find_unop_node(&f, Unop::Not);
        let t = ShiftHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Identity, inner) = f.get_node(not_ref).payload else {
            panic!("expected identity wrapper");
        };
        assert!(matches!(
            f.get_node(inner).payload,
            NodePayload::Binop(Binop::Shll, _, _)
        ));
    }

    #[test]
    fn shift_hoist_unop_reverse() {
        let ir_text = r#"fn t(x: bits[8] id=1, k: bits[3] id=2) -> bits[8] {
  not.10: bits[8] = not(x, id=10)
  ret shll.11: bits[8] = shll(not.10, k, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let shll_ref = find_binop_node(&f, Binop::Shll);
        let t = ShiftHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(shll_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(shll_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn shift_hoist_binop_lhs() {
        let ir_text = r#"fn t(x: bits[8] id=1, k: bits[3] id=2, y: bits[8] id=3) -> bits[8] {
  shll.10: bits[8] = shll(x, k, id=10)
  ret add.11: bits[8] = add(shll.10, y, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = find_binop_node(&f, Binop::Add);
        let t = ShiftHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Identity, inner) = f.get_node(add_ref).payload else {
            panic!("expected identity wrapper");
        };
        assert!(matches!(
            f.get_node(inner).payload,
            NodePayload::Binop(Binop::Shll, _, _)
        ));
    }

    #[test]
    fn shift_hoist_binop_rhs_reverse() {
        let ir_text = r#"fn t(x: bits[8] id=1, k: bits[3] id=2, y: bits[8] id=3) -> bits[8] {
  add.10: bits[8] = add(y, x, id=10)
  ret shll.11: bits[8] = shll(add.10, k, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let shll_ref = find_binop_node(&f, Binop::Shll);
        let t = ShiftHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(shll_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(shll_ref).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }
}
