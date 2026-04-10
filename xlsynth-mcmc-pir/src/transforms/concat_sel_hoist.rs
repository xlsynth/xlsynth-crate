// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Fn as IrFn, NaryOp, Node, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformCandidate, TransformLocation};

/// Hoists a single 2-case sel operand through concat and folds the narrow
/// reverse form back.
#[derive(Debug)]
pub struct ConcatSelHoistTransform;

impl ConcatSelHoistTransform {
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

impl PirTransform for ConcatSelHoistTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ConcatSelHoist
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Nary(NaryOp::Concat, ops) => {
                    let sel_positions: Vec<usize> = ops
                        .iter()
                        .enumerate()
                        .filter_map(|(i, op)| {
                            matches!(f.get_node(*op).payload, NodePayload::Sel { .. }).then_some(i)
                        })
                        .collect();
                    if sel_positions.len() != 1 {
                        continue;
                    }
                    let sel_ref = ops[sel_positions[0]];
                    let NodePayload::Sel {
                        selector,
                        cases,
                        default,
                    } = &f.get_node(sel_ref).payload
                    else {
                        continue;
                    };
                    if default.is_some()
                        || cases.len() != 2
                        || Self::bits_width(&f.get_node(*selector).ty) != Some(1)
                    {
                        continue;
                    }
                    out.push(TransformCandidate {
                        location: TransformLocation::Node(nr),
                        always_equivalent,
                    });
                }
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if default.is_some()
                        || cases.len() != 2
                        || Self::bits_width(&f.get_node(*selector).ty) != Some(1)
                    {
                        continue;
                    }
                    let NodePayload::Nary(NaryOp::Concat, ops0) =
                        f.get_node(cases[0]).payload.clone()
                    else {
                        continue;
                    };
                    let NodePayload::Nary(NaryOp::Concat, ops1) =
                        f.get_node(cases[1]).payload.clone()
                    else {
                        continue;
                    };
                    if ops0.len() != ops1.len() {
                        continue;
                    }
                    let diff_positions: Vec<usize> =
                        (0..ops0.len()).filter(|i| ops0[*i] != ops1[*i]).collect();
                    if diff_positions.len() == 1 {
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
                return Err("ConcatSelHoistTransform: expected TransformLocation::Node".to_string());
            }
        };
        let target_ty = f.get_node(target_ref).ty.clone();
        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Nary(NaryOp::Concat, ops) => {
                let sel_positions: Vec<usize> = ops
                    .iter()
                    .enumerate()
                    .filter_map(|(i, op)| {
                        matches!(f.get_node(*op).payload, NodePayload::Sel { .. }).then_some(i)
                    })
                    .collect();
                if sel_positions.len() != 1 {
                    return Err(
                        "ConcatSelHoistTransform: expected exactly one sel operand".to_string()
                    );
                }
                let idx = sel_positions[0];
                let sel_ref = ops[idx];
                let NodePayload::Sel {
                    selector,
                    cases,
                    default: None,
                } = f.get_node(sel_ref).payload.clone()
                else {
                    return Err(
                        "ConcatSelHoistTransform: sel operand must be 2-case/no-default"
                            .to_string(),
                    );
                };
                if cases.len() != 2 {
                    return Err(
                        "ConcatSelHoistTransform: sel operand must have 2 cases".to_string()
                    );
                }
                let mut arm0 = ops.clone();
                arm0[idx] = cases[0];
                let mut arm1 = ops.clone();
                arm1[idx] = cases[1];
                let concat0 = Self::mk_nary_node(f, NaryOp::Concat, target_ty.clone(), arm0);
                let concat1 = Self::mk_nary_node(f, NaryOp::Concat, target_ty.clone(), arm1);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector,
                    cases: vec![concat0, concat1],
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
                    return Err("ConcatSelHoistTransform: expected 2-case sel".to_string());
                }
                let NodePayload::Nary(NaryOp::Concat, ops0) = f.get_node(cases[0]).payload.clone()
                else {
                    return Err("ConcatSelHoistTransform: case0 must be concat".to_string());
                };
                let NodePayload::Nary(NaryOp::Concat, ops1) = f.get_node(cases[1]).payload.clone()
                else {
                    return Err("ConcatSelHoistTransform: case1 must be concat".to_string());
                };
                if ops0.len() != ops1.len() {
                    return Err("ConcatSelHoistTransform: concat arity mismatch".to_string());
                }
                let diff_positions: Vec<usize> =
                    (0..ops0.len()).filter(|i| ops0[*i] != ops1[*i]).collect();
                if diff_positions.len() != 1 {
                    return Err(
                        "ConcatSelHoistTransform: expected exactly one varying concat slot"
                            .to_string(),
                    );
                }
                let idx = diff_positions[0];
                let sel_ty = f.get_node(ops0[idx]).ty.clone();
                let inner_sel = Self::mk_sel_node(f, selector, sel_ty, vec![ops0[idx], ops1[idx]]);
                let mut new_ops = ops0.clone();
                new_ops[idx] = inner_sel;
                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Concat, new_ops);
                Ok(())
            }
            _ => Err("ConcatSelHoistTransform: unsupported target".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ConcatSelHoistTransform;
    use crate::transforms::{PirTransform, TransformLocation};
    use xlsynth_pir::ir::{NaryOp, NodePayload};
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_concat_with_single_sel_operand() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[4] id=2, b: bits[4] id=3, x: bits[4] id=4) -> bits[8] {
  s: bits[4] = sel(p, cases=[a, b], id=10)
  ret out: bits[8] = concat(x, s, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let mut t = ConcatSelHoistTransform;
        let cand = t.find_candidates(&f).pop().expect("candidate");
        let target = match cand.location.clone() {
            TransformLocation::Node(nr) => nr,
            _ => unreachable!(),
        };
        t.apply(&mut f, &cand.location).expect("apply");
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Sel { .. }
        ));
    }

    #[test]
    fn folds_sel_of_concats_back() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[4] id=2, b: bits[4] id=3, x: bits[4] id=4) -> bits[8] {
  c0: bits[8] = concat(x, a, id=10)
  c1: bits[8] = concat(x, b, id=11)
  ret out: bits[8] = sel(p, cases=[c0, c1], id=12)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let mut t = ConcatSelHoistTransform;
        let cand = t.find_candidates(&f).pop().expect("candidate");
        let target = match cand.location.clone() {
            TransformLocation::Node(nr) => nr,
            _ => unreachable!(),
        };
        t.apply(&mut f, &cand.location).expect("apply");
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Concat, _)
        ));
    }
}
