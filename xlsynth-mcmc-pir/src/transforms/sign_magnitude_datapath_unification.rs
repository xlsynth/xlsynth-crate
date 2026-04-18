// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use super::macro_utils as mu;
use super::selected_add_sub_unification::SelectedAddSubUnificationTransform;
use super::*;

const MAX_SIGN_MAG_CONE_NODES: usize = 64;

/// Oracle-backed recursive sharing for two-arm sign/magnitude datapaths.
#[derive(Debug)]
pub struct SignMagnitudeDatapathUnificationTransform;

impl SignMagnitudeDatapathUnificationTransform {
    fn cone_size(f: &IrFn, roots: &[NodeRef], max_nodes: usize) -> Option<usize> {
        let mut seen = HashSet::new();
        let mut stack = roots.to_vec();
        while let Some(r) = stack.pop() {
            if !seen.insert(r.index) {
                continue;
            }
            if seen.len() > max_nodes {
                return None;
            }
            match &f.get_node(r).payload {
                NodePayload::Binop(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                NodePayload::Unop(_, a)
                | NodePayload::SignExt { arg: a, .. }
                | NodePayload::ZeroExt { arg: a, .. }
                | NodePayload::Encode { arg: a }
                | NodePayload::ExtClz { arg: a }
                | NodePayload::Decode { arg: a, .. } => stack.push(*a),
                NodePayload::Nary(_, ops)
                | NodePayload::Tuple(ops)
                | NodePayload::Array(ops)
                | NodePayload::AfterAll(ops) => stack.extend(ops.iter().copied()),
                NodePayload::BitSlice { arg, .. } => stack.push(*arg),
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                }
                | NodePayload::PrioritySel {
                    selector,
                    cases,
                    default,
                } => {
                    stack.push(*selector);
                    stack.extend(cases.iter().copied());
                    if let Some(d) = default {
                        stack.push(*d);
                    }
                }
                NodePayload::OneHotSel { selector, cases } => {
                    stack.push(*selector);
                    stack.extend(cases.iter().copied());
                }
                _ => {}
            }
        }
        Some(seen.len())
    }

    fn cone_contains_add_sub(f: &IrFn, root: NodeRef, max_nodes: usize) -> bool {
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(r) = stack.pop() {
            if !seen.insert(r.index) || seen.len() > max_nodes {
                continue;
            }
            match &f.get_node(r).payload {
                NodePayload::Binop(op @ (Binop::Add | Binop::Sub), a, b) => {
                    let _ = op;
                    stack.push(*a);
                    stack.push(*b);
                    return true;
                }
                NodePayload::Binop(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                NodePayload::Unop(_, a) => stack.push(*a),
                NodePayload::Nary(_, ops) => stack.extend(ops.iter().copied()),
                _ => {}
            }
        }
        false
    }

    fn mk_sel2(f: &mut IrFn, selector: NodeRef, ty: Type, a: NodeRef, b: NodeRef) -> NodeRef {
        mu::push_node(
            f,
            ty,
            NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
        )
    }

    fn unify_pair(
        f: &mut IrFn,
        selector: NodeRef,
        left: NodeRef,
        right: NodeRef,
        depth: usize,
    ) -> Result<NodeRef, String> {
        if left == right {
            return Ok(left);
        }
        if depth > 16 {
            return Ok(Self::mk_sel2(
                f,
                selector,
                f.get_node(left).ty.clone(),
                left,
                right,
            ));
        }
        let ty = f.get_node(left).ty.clone();
        if f.get_node(right).ty != ty {
            return Ok(Self::mk_sel2(f, selector, ty, left, right));
        }
        match (
            f.get_node(left).payload.clone(),
            f.get_node(right).payload.clone(),
        ) {
            (NodePayload::Unop(op_l, arg_l), NodePayload::Unop(op_r, arg_r)) if op_l == op_r => {
                let arg = Self::unify_pair(f, selector, arg_l, arg_r, depth + 1)?;
                Ok(mu::mk_unop(f, op_l, ty, arg))
            }
            (
                NodePayload::Binop(Binop::Add, a_l, b_l),
                NodePayload::Binop(Binop::Sub, a_r, b_r),
            ) if a_l == a_r && b_l == b_r && matches!(ty, Type::Bits(_)) => {
                let Type::Bits(width) = ty else {
                    unreachable!();
                };
                Ok(SelectedAddSubUnificationTransform::build_unified_add_sub(
                    f, selector, a_l, b_l, width,
                ))
            }
            (
                NodePayload::Binop(Binop::Sub, a_l, b_l),
                NodePayload::Binop(Binop::Add, a_r, b_r),
            ) if a_l == a_r && b_l == b_r && matches!(ty, Type::Bits(_)) => {
                let Type::Bits(width) = ty else {
                    unreachable!();
                };
                let pred_sub = mu::mk_unop(f, Unop::Not, Type::Bits(1), selector);
                Ok(SelectedAddSubUnificationTransform::build_unified_add_sub(
                    f, pred_sub, a_l, b_l, width,
                ))
            }
            (NodePayload::Binop(op_l, a_l, b_l), NodePayload::Binop(op_r, a_r, b_r))
                if op_l == op_r =>
            {
                let a = Self::unify_pair(f, selector, a_l, a_r, depth + 1)?;
                let b = Self::unify_pair(f, selector, b_l, b_r, depth + 1)?;
                Ok(mu::mk_binop(f, op_l, ty, a, b))
            }
            (NodePayload::Nary(op_l, ops_l), NodePayload::Nary(op_r, ops_r))
                if op_l == op_r && ops_l.len() == ops_r.len() =>
            {
                let mut ops = Vec::with_capacity(ops_l.len());
                for (l, r) in ops_l.into_iter().zip(ops_r.into_iter()) {
                    ops.push(Self::unify_pair(f, selector, l, r, depth + 1)?);
                }
                Ok(mu::push_node(f, ty, NodePayload::Nary(op_l, ops)))
            }
            _ => Ok(Self::mk_sel2(f, selector, ty, left, right)),
        }
    }
}

impl PirTransform for SignMagnitudeDatapathUnificationTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SignMagnitudeDatapathUnification
    }

    fn can_emit_always_equivalent_candidates(&self) -> bool {
        false
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            let NodePayload::Sel {
                selector,
                cases,
                default,
            } = &f.get_node(nr).payload
            else {
                continue;
            };
            if default.is_some() || cases.len() != 2 || !mu::is_u1(f, *selector) {
                continue;
            }
            if !matches!(f.get_node(nr).ty, Type::Bits(_)) {
                continue;
            }
            if f.get_node(cases[0]).ty != f.get_node(nr).ty
                || f.get_node(cases[1]).ty != f.get_node(nr).ty
            {
                continue;
            }
            if Self::cone_size(f, cases, MAX_SIGN_MAG_CONE_NODES).is_none() {
                continue;
            }
            if Self::cone_contains_add_sub(f, cases[0], MAX_SIGN_MAG_CONE_NODES)
                && Self::cone_contains_add_sub(f, cases[1], MAX_SIGN_MAG_CONE_NODES)
            {
                out.push(TransformCandidate {
                    location: TransformLocation::Node(nr),
                    always_equivalent: false,
                });
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SignMagnitudeDatapathUnificationTransform: expected node location".to_string(),
                );
            }
        };
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = f.get_node(target).payload.clone()
        else {
            return Err("SignMagnitudeDatapathUnificationTransform: expected sel".to_string());
        };
        if default.is_some() || cases.len() != 2 || !mu::is_u1(f, selector) {
            return Err("SignMagnitudeDatapathUnificationTransform: invalid sel shape".to_string());
        }
        if Self::cone_size(f, &cases, MAX_SIGN_MAG_CONE_NODES).is_none() {
            return Err("SignMagnitudeDatapathUnificationTransform: cone too large".to_string());
        }
        let unified = Self::unify_pair(f, selector, cases[0], cases[1], 0)?;
        f.get_node_mut(target).payload = NodePayload::Unop(Unop::Identity, unified);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn emits_oracle_backed_candidate_for_add_sub_cone() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[8] {
  add.5: bits[8] = add(a, b, id=5)
  l: bits[8] = add(add.5, c, id=6)
  sub.7: bits[8] = sub(a, b, id=7)
  r: bits[8] = add(sub.7, c, id=8)
  ret out: bits[8] = sel(p, cases=[l, r], id=9)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let mut t = SignMagnitudeDatapathUnificationTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(!candidates[0].always_equivalent);
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }
}
