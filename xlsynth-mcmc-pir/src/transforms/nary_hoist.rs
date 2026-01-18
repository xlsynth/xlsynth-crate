// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Binop, Fn as IrFn, NaryOp, Node, NodePayload, NodeRef, Type, Unop};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A non-always-equivalent transform that hoists unary/binary ops over n-ary
/// ops.
///
/// - `unop(naryop(a,b,c)) -> naryop(unop(a), unop(b), unop(c))`
/// - `binop(naryop(a,b,c), t) -> naryop(binop(a,t), binop(b,t), binop(c,t))`
/// - `binop(t, naryop(a,b,c)) -> naryop(binop(t,a), binop(t,b), binop(t,c))`
#[derive(Debug)]
pub struct NaryHoistTransform;

impl NaryHoistTransform {
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

    fn nary_parts(payload: &NodePayload) -> Option<(NaryOp, Vec<NodeRef>)> {
        match payload {
            NodePayload::Nary(op, operands) => Some((*op, operands.clone())),
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

    fn mk_nary_node(f: &mut IrFn, op: NaryOp, ty: Type, operands: Vec<NodeRef>) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Nary(op, operands),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NaryHoistTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NaryHoist
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::Unop(_, arg) => {
                    let arg_node = f.get_node(*arg);
                    let Some((_op, operands)) = Self::nary_parts(&arg_node.payload) else {
                        continue;
                    };
                    let Some(w) = Self::bits_width(&node.ty) else {
                        continue;
                    };
                    if Self::bits_width(&arg_node.ty) != Some(w) {
                        continue;
                    }
                    if operands
                        .iter()
                        .all(|op| Self::bits_width(&f.get_node(*op).ty) == Some(w))
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Binop(_, lhs, rhs) => {
                    let lhs_payload = f.get_node(*lhs).payload.clone();
                    let rhs_payload = f.get_node(*rhs).payload.clone();
                    let lhs_ty = f.get_node(*lhs).ty.clone();
                    let rhs_ty = f.get_node(*rhs).ty.clone();
                    let lhs_nary = Self::nary_parts(&lhs_payload);
                    let rhs_nary = Self::nary_parts(&rhs_payload);
                    if lhs_nary.is_some() == rhs_nary.is_some() {
                        continue;
                    }
                    let Some(w) = Self::bits_width(&node.ty) else {
                        continue;
                    };
                    if Self::bits_width(&lhs_ty) != Some(w) || Self::bits_width(&rhs_ty) != Some(w)
                    {
                        continue;
                    }
                    if let Some((_op, operands)) = lhs_nary.as_ref() {
                        if operands
                            .iter()
                            .any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w))
                        {
                            continue;
                        }
                    }
                    if let Some((_op, operands)) = rhs_nary.as_ref() {
                        if operands
                            .iter()
                            .any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w))
                        {
                            continue;
                        }
                    }
                    out.push(TransformLocation::Node(nr));
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
                    "NaryHoistTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        let target_ty = f.get_node(target_ref).ty.clone();
        let Some(w) = Self::bits_width(&target_ty) else {
            return Err("NaryHoistTransform: target must be bits[w]".to_string());
        };

        match target_payload {
            NodePayload::Unop(op, arg) => {
                let arg_node = f.get_node(arg);
                let Some((nary_op, operands)) = Self::nary_parts(&arg_node.payload) else {
                    return Err("NaryHoistTransform: expected unop(nary(...))".to_string());
                };
                if Self::bits_width(&arg_node.ty) != Some(w)
                    || operands
                        .iter()
                        .any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w))
                {
                    return Err("NaryHoistTransform: nary operands must be bits[w]".to_string());
                }
                let mut new_operands: Vec<NodeRef> = Vec::with_capacity(operands.len());
                for operand in operands {
                    new_operands.push(Self::mk_unop_node(f, op, target_ty.clone(), operand));
                }
                let new_nary = Self::mk_nary_node(f, nary_op, target_ty, new_operands);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, new_nary);
                Ok(())
            }
            NodePayload::Binop(op, lhs, rhs) => {
                let lhs_payload = f.get_node(lhs).payload.clone();
                let rhs_payload = f.get_node(rhs).payload.clone();
                let lhs_ty = f.get_node(lhs).ty.clone();
                let rhs_ty = f.get_node(rhs).ty.clone();
                let lhs_nary = Self::nary_parts(&lhs_payload);
                let rhs_nary = Self::nary_parts(&rhs_payload);
                if lhs_nary.is_some() == rhs_nary.is_some() {
                    return Err("NaryHoistTransform: expected exactly one nary operand".to_string());
                }
                if Self::bits_width(&lhs_ty) != Some(w) || Self::bits_width(&rhs_ty) != Some(w) {
                    return Err(
                        "NaryHoistTransform: binop operands must match bits width".to_string()
                    );
                }
                if let Some((nary_op, operands)) = lhs_nary {
                    if operands
                        .iter()
                        .any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w))
                    {
                        return Err("NaryHoistTransform: nary operands must be bits[w]".to_string());
                    }
                    let mut new_operands: Vec<NodeRef> = Vec::with_capacity(operands.len());
                    for operand in operands {
                        new_operands.push(Self::mk_binop_node(
                            f,
                            op,
                            target_ty.clone(),
                            operand,
                            rhs,
                        ));
                    }
                    let new_nary = Self::mk_nary_node(f, nary_op, target_ty, new_operands);
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Unop(Unop::Identity, new_nary);
                    Ok(())
                } else if let Some((nary_op, operands)) = rhs_nary {
                    if operands
                        .iter()
                        .any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w))
                    {
                        return Err("NaryHoistTransform: nary operands must be bits[w]".to_string());
                    }
                    let mut new_operands: Vec<NodeRef> = Vec::with_capacity(operands.len());
                    for operand in operands {
                        new_operands.push(Self::mk_binop_node(
                            f,
                            op,
                            target_ty.clone(),
                            lhs,
                            operand,
                        ));
                    }
                    let new_nary = Self::mk_nary_node(f, nary_op, target_ty, new_operands);
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Unop(Unop::Identity, new_nary);
                    Ok(())
                } else {
                    Err("NaryHoistTransform: missing nary operand".to_string())
                }
            }
            _ => Err("NaryHoistTransform: expected unop or binop target".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
    use xlsynth_pir::ir_parser;

    use super::NaryHoistTransform;
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

    fn find_param_named(f: &xlsynth_pir::ir::Fn, name: &str) -> NodeRef {
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            if matches!(node.payload, NodePayload::GetParam(_))
                && node.name.as_deref() == Some(name)
            {
                return nr;
            }
        }
        panic!("expected param");
    }

    #[test]
    fn nary_hoist_unop_forward() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  and.10: bits[8] = and(a, b, c, id=10)
  ret not.11: bits[8] = not(and.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let not_ref = find_unop_node(&f, Unop::Not);
        let t = NaryHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Identity, inner) = f.get_node(not_ref).payload else {
            panic!("expected identity wrapper");
        };
        assert!(matches!(
            f.get_node(inner).payload,
            NodePayload::Nary(NaryOp::And, _)
        ));
    }

    #[test]
    fn nary_hoist_binop_lhs() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3, t: bits[8] id=4) -> bits[8] {
  or.10: bits[8] = or(a, b, c, id=10)
  ret add.11: bits[8] = add(or.10, t, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = find_binop_node(&f, Binop::Add);
        let t = NaryHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Identity, inner) = f.get_node(add_ref).payload else {
            panic!("expected identity wrapper");
        };
        assert!(matches!(
            f.get_node(inner).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
    }

    #[test]
    fn nary_hoist_binop_rhs() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3, t: bits[8] id=4) -> bits[8] {
  or.10: bits[8] = or(a, b, c, id=10)
  ret add.11: bits[8] = add(t, or.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = find_binop_node(&f, Binop::Add);
        let t = NaryHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Identity, inner) = f.get_node(add_ref).payload else {
            panic!("expected identity wrapper");
        };
        assert!(matches!(
            f.get_node(inner).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
    }

    #[test]
    fn nary_hoist_rejects_operand_mismatch() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3, t: bits[8] id=4) -> bits[8] {
  or.10: bits[8] = or(a, b, c, id=10)
  ret add.11: bits[8] = add(or.10, t, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let b_ref = find_param_named(&f, "b");
        f.get_node_mut(b_ref).ty = Type::Bits(4);

        let add_ref = find_binop_node(&f, Binop::Add);
        let t = NaryHoistTransform;
        let err = t
            .apply(&mut f, &TransformLocation::Node(add_ref))
            .unwrap_err();
        assert!(err.contains("nary operands must be bits"));
    }
}
