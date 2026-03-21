// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Binop, Fn as IrFn, Node, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// Hoists/folds a binary op through two 2-case sels sharing the same selector.
///
/// `binop(sel(p,[a0,a1]), sel(p,[b0,b1])) ↔ sel(p,[binop(a0,b0), binop(a1,b1)])`
#[derive(Debug)]
pub struct DualSelHoistTransform;

impl DualSelHoistTransform {
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

    fn mk_sel_node(f: &mut IrFn, selector: NodeRef, ty: Type, cases: Vec<NodeRef>) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Sel {
                selector,
                cases,
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for DualSelHoistTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::DualSelHoist
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::Binop(_, lhs, rhs) => {
                    let Some(w) = Self::bits_width(&node.ty) else {
                        continue;
                    };
                    let lhs_node = f.get_node(*lhs);
                    let rhs_node = f.get_node(*rhs);
                    let NodePayload::Sel {
                        selector: ls,
                        cases: lc,
                        default: ld,
                    } = &lhs_node.payload
                    else {
                        continue;
                    };
                    let NodePayload::Sel {
                        selector: rs,
                        cases: rc,
                        default: rd,
                    } = &rhs_node.payload
                    else {
                        continue;
                    };
                    if ls != rs || ld.is_some() || rd.is_some() || lc.len() != 2 || rc.len() != 2 {
                        continue;
                    }
                    if Self::bits_width(&lhs_node.ty) != Some(w)
                        || Self::bits_width(&rhs_node.ty) != Some(w)
                    {
                        continue;
                    }
                    if Self::bits_width(&f.get_node(*ls).ty) != Some(1) {
                        continue;
                    }
                    if lc.iter().chain(rc.iter()).any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w)) {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if default.is_some() || cases.len() != 2 {
                        continue;
                    }
                    if Self::bits_width(&f.get_node(*selector).ty) != Some(1) {
                        continue;
                    }
                    let case0 = f.get_node(cases[0]).payload.clone();
                    let case1 = f.get_node(cases[1]).payload.clone();
                    let NodePayload::Binop(op0, a0, b0) = case0 else {
                        continue;
                    };
                    let NodePayload::Binop(op1, a1, b1) = case1 else {
                        continue;
                    };
                    if op0 != op1 {
                        continue;
                    }
                    let Some(w) = Self::bits_width(&f.get_node(nr).ty) else {
                        continue;
                    };
                    if [a0, b0, a1, b1]
                        .iter()
                        .any(|op| Self::bits_width(&f.get_node(*op).ty) != Some(w))
                    {
                        continue;
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
                return Err("DualSelHoistTransform: expected TransformLocation::Node".to_string())
            }
        };
        let target_ty = f.get_node(target_ref).ty.clone();
        let target_payload = f.get_node(target_ref).payload.clone();

        match target_payload {
            NodePayload::Binop(op, lhs, rhs) => {
                let lhs_sel = f.get_node(lhs).payload.clone();
                let rhs_sel = f.get_node(rhs).payload.clone();
                let NodePayload::Sel {
                    selector,
                    cases: lhs_cases,
                    default: None,
                } = lhs_sel
                else {
                    return Err("DualSelHoistTransform: lhs must be 2-case sel".to_string());
                };
                let NodePayload::Sel {
                    selector: rhs_selector,
                    cases: rhs_cases,
                    default: None,
                } = rhs_sel
                else {
                    return Err("DualSelHoistTransform: rhs must be 2-case sel".to_string());
                };
                if selector != rhs_selector || lhs_cases.len() != 2 || rhs_cases.len() != 2 {
                    return Err("DualSelHoistTransform: sels must align".to_string());
                }
                let case0 = Self::mk_binop_node(f, op, target_ty.clone(), lhs_cases[0], rhs_cases[0]);
                let case1 = Self::mk_binop_node(f, op, target_ty.clone(), lhs_cases[1], rhs_cases[1]);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector,
                    cases: vec![case0, case1],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default: None,
            } => {
                if cases.len() != 2 {
                    return Err("DualSelHoistTransform: sel must have exactly 2 cases".to_string());
                }
                let case0 = f.get_node(cases[0]).payload.clone();
                let case1 = f.get_node(cases[1]).payload.clone();
                let NodePayload::Binop(op0, a0, b0) = case0 else {
                    return Err("DualSelHoistTransform: case0 must be binop".to_string());
                };
                let NodePayload::Binop(op1, a1, b1) = case1 else {
                    return Err("DualSelHoistTransform: case1 must be binop".to_string());
                };
                if op0 != op1 {
                    return Err("DualSelHoistTransform: case binops must match".to_string());
                }
                let lhs_sel = Self::mk_sel_node(f, selector, target_ty.clone(), vec![a0, a1]);
                let rhs_sel = Self::mk_sel_node(f, selector, target_ty.clone(), vec![b0, b1]);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(op0, lhs_sel, rhs_sel);
                Ok(())
            }
            _ => Err("DualSelHoistTransform: unsupported target".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DualSelHoistTransform;
    use crate::transforms::{PirTransform, TransformLocation};
    use xlsynth_pir::ir::{Binop, NodePayload};
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_add_of_aligned_sels() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4, d: bits[8] id=5) -> bits[8] {
  s0: bits[8] = sel(p, cases=[a, b], id=10)
  s1: bits[8] = sel(p, cases=[c, d], id=11)
  ret out: bits[8] = add(s0, s1, id=12)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let mut t = DualSelHoistTransform;
        let cand = t.find_candidates(&f).pop().expect("candidate");
        let target = match cand.clone() { TransformLocation::Node(nr) => nr, _ => unreachable!() };
        t.apply(&mut f, &cand).expect("apply");
        assert!(matches!(f.get_node(target).payload, NodePayload::Sel { .. }));
    }

    #[test]
    fn folds_sel_of_same_binop_cases() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4, d: bits[8] id=5) -> bits[8] {
  ac: bits[8] = add(a, c, id=10)
  bd: bits[8] = add(b, d, id=11)
  ret out: bits[8] = sel(p, cases=[ac, bd], id=12)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let mut t = DualSelHoistTransform;
        let cand = t.find_candidates(&f).pop().expect("candidate");
        let target = match cand.clone() { TransformLocation::Node(nr) => nr, _ => unreachable!() };
        t.apply(&mut f, &cand).expect("apply");
        assert!(matches!(f.get_node(target).payload, NodePayload::Binop(Binop::Add, _, _)));
    }
}
