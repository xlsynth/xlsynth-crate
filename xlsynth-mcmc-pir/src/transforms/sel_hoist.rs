// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Binop, Fn as IrFn, Node, NodePayload, NodeRef, Type, Unop};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A semantics-preserving transform that hoists unary/binary ops over `sel`.
///
/// - `unop(sel(s, cases)) -> sel(s, map(unop, cases))`
/// - `binop(sel(s, cases), t) -> sel(s, map(|c| binop(c, t), cases))`
/// - `binop(t, sel(s, cases)) -> sel(s, map(|c| binop(t, c), cases))`
#[derive(Debug)]
pub struct SelHoistTransform;

impl SelHoistTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_bits_type(ty: &Type) -> bool {
        matches!(ty, Type::Bits(_))
    }

    fn sel_cases_match_type(
        f: &IrFn,
        cases: &[NodeRef],
        default: &Option<NodeRef>,
        ty: &Type,
    ) -> bool {
        if cases.is_empty() {
            return false;
        }
        for case in cases {
            if f.get_node(*case).ty != *ty {
                return false;
            }
        }
        if let Some(d) = default {
            if f.get_node(*d).ty != *ty {
                return false;
            }
        }
        true
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
}

impl PirTransform for SelHoistTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelHoist
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::Unop(_, arg) => {
                    if !Self::is_bits_type(&node.ty) {
                        continue;
                    }
                    let sel_node = f.get_node(*arg);
                    let NodePayload::Sel { cases, default, .. } = &sel_node.payload else {
                        continue;
                    };
                    if !Self::is_bits_type(&sel_node.ty) {
                        continue;
                    }
                    if !Self::sel_cases_match_type(f, cases, default, &sel_node.ty) {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Binop(_, lhs, rhs) => {
                    if !Self::is_bits_type(&node.ty) {
                        continue;
                    }
                    let lhs_node = f.get_node(*lhs);
                    let rhs_node = f.get_node(*rhs);
                    let lhs_sel = matches!(lhs_node.payload, NodePayload::Sel { .. });
                    let rhs_sel = matches!(rhs_node.payload, NodePayload::Sel { .. });
                    if lhs_sel == rhs_sel {
                        continue;
                    }
                    if lhs_sel {
                        let NodePayload::Sel { cases, default, .. } = &lhs_node.payload else {
                            continue;
                        };
                        if !Self::is_bits_type(&lhs_node.ty) {
                            continue;
                        }
                        if !Self::sel_cases_match_type(f, cases, default, &lhs_node.ty) {
                            continue;
                        }
                        out.push(TransformLocation::Node(nr));
                    } else {
                        let NodePayload::Sel { cases, default, .. } = &rhs_node.payload else {
                            continue;
                        };
                        if !Self::is_bits_type(&rhs_node.ty) {
                            continue;
                        }
                        if !Self::sel_cases_match_type(f, cases, default, &rhs_node.ty) {
                            continue;
                        }
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
                    "SelHoistTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        let target_ty = f.get_node(target_ref).ty.clone();
        if !Self::is_bits_type(&target_ty) {
            return Err("SelHoistTransform: target output must be bits".to_string());
        }

        match target_payload {
            NodePayload::Unop(op, arg_sel_ref) => {
                let sel_node = f.get_node(arg_sel_ref);
                let sel_ty = sel_node.ty.clone();
                let NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } = sel_node.payload.clone()
                else {
                    return Err("SelHoistTransform: expected unop(sel(...))".to_string());
                };
                if !Self::is_bits_type(&sel_ty) {
                    return Err("SelHoistTransform: sel cases must be bits".to_string());
                }
                if !Self::sel_cases_match_type(f, &cases, &default, &sel_ty) {
                    return Err(
                        "SelHoistTransform: sel cases/default must match sel type".to_string()
                    );
                }

                let mut new_cases: Vec<NodeRef> = Vec::with_capacity(cases.len());
                for case in cases {
                    new_cases.push(Self::mk_unop_node(f, op, target_ty.clone(), case));
                }
                let new_default = default.map(|d| Self::mk_unop_node(f, op, target_ty.clone(), d));
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector,
                    cases: new_cases,
                    default: new_default,
                };
                Ok(())
            }
            NodePayload::Binop(op, lhs, rhs) => {
                let lhs_node = f.get_node(lhs);
                let rhs_node = f.get_node(rhs);
                let lhs_sel = matches!(lhs_node.payload, NodePayload::Sel { .. });
                let rhs_sel = matches!(rhs_node.payload, NodePayload::Sel { .. });
                if lhs_sel == rhs_sel {
                    return Err("SelHoistTransform: expected exactly one sel operand".to_string());
                }

                if lhs_sel {
                    let sel_ty = lhs_node.ty.clone();
                    let NodePayload::Sel {
                        selector,
                        cases,
                        default,
                    } = lhs_node.payload.clone()
                    else {
                        return Err("SelHoistTransform: expected sel on lhs".to_string());
                    };
                    if !Self::is_bits_type(&sel_ty) {
                        return Err("SelHoistTransform: sel cases must be bits".to_string());
                    }
                    if !Self::sel_cases_match_type(f, &cases, &default, &sel_ty) {
                        return Err(
                            "SelHoistTransform: sel cases/default must match lhs type".to_string()
                        );
                    }

                    let mut new_cases: Vec<NodeRef> = Vec::with_capacity(cases.len());
                    for case in cases {
                        new_cases.push(Self::mk_binop_node(f, op, target_ty.clone(), case, rhs));
                    }
                    let new_default =
                        default.map(|d| Self::mk_binop_node(f, op, target_ty.clone(), d, rhs));
                    f.get_node_mut(target_ref).payload = NodePayload::Sel {
                        selector,
                        cases: new_cases,
                        default: new_default,
                    };
                } else {
                    let sel_ty = rhs_node.ty.clone();
                    let NodePayload::Sel {
                        selector,
                        cases,
                        default,
                    } = rhs_node.payload.clone()
                    else {
                        return Err("SelHoistTransform: expected sel on rhs".to_string());
                    };
                    if !Self::is_bits_type(&sel_ty) {
                        return Err("SelHoistTransform: sel cases must be bits".to_string());
                    }
                    if !Self::sel_cases_match_type(f, &cases, &default, &sel_ty) {
                        return Err(
                            "SelHoistTransform: sel cases/default must match rhs type".to_string()
                        );
                    }

                    let mut new_cases: Vec<NodeRef> = Vec::with_capacity(cases.len());
                    for case in cases {
                        new_cases.push(Self::mk_binop_node(f, op, target_ty.clone(), lhs, case));
                    }
                    let new_default =
                        default.map(|d| Self::mk_binop_node(f, op, target_ty.clone(), lhs, d));
                    f.get_node_mut(target_ref).payload = NodePayload::Sel {
                        selector,
                        cases: new_cases,
                        default: new_default,
                    };
                }
                Ok(())
            }
            _ => Err("SelHoistTransform: expected unop or binop target".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef, Unop};
    use xlsynth_pir::ir_parser;

    use super::SelHoistTransform;
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
    fn sel_hoist_unop_with_default() {
        let ir_text = r#"fn t(p: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4, d: bits[8] id=5) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b, c], default=d, id=10)
  ret not.11: bits[8] = not(sel.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let not_ref = find_unop_node(&f, Unop::Not);
        let t = SelHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        let NodePayload::Sel {
            ref cases, default, ..
        } = f.get_node(not_ref).payload
        else {
            panic!("expected sel after hoist");
        };
        assert_eq!(cases.len(), 3);
        for case in cases {
            assert!(matches!(
                f.get_node(*case).payload,
                NodePayload::Unop(Unop::Not, _)
            ));
        }
        let default = default.expect("expected default");
        assert!(matches!(
            f.get_node(default).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn sel_hoist_binop_lhs() {
        let ir_text = r#"fn t(p: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4, t: bits[8] id=5) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b, c], id=10)
  ret add.11: bits[8] = add(sel.10, t, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = find_binop_node(&f, Binop::Add);
        let t = SelHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Sel { ref cases, .. } = f.get_node(add_ref).payload else {
            panic!("expected sel after hoist");
        };
        assert_eq!(cases.len(), 3);
        for case in cases {
            assert!(matches!(
                f.get_node(*case).payload,
                NodePayload::Binop(Binop::Add, _, _)
            ));
        }
    }

    #[test]
    fn sel_hoist_binop_rhs() {
        let ir_text = r#"fn t(p: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4, t: bits[8] id=5) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b, c], id=10)
  ret add.11: bits[8] = add(t, sel.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = find_binop_node(&f, Binop::Add);
        let t = SelHoistTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Sel { ref cases, .. } = f.get_node(add_ref).payload else {
            panic!("expected sel after hoist");
        };
        assert_eq!(cases.len(), 3);
        for case in cases {
            assert!(matches!(
                f.get_node(*case).payload,
                NodePayload::Binop(Binop::Add, _, _)
            ));
        }
    }
}
