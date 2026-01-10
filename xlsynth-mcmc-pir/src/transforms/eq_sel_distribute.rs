// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing the equivalence:
///
/// - `eq(sel(p,a,b),c) ↔ sel(p,eq(a,c),eq(b,c))`
/// - `ne(sel(p,a,b),c) ↔ sel(p,ne(a,c),ne(b,c))`
///
/// This works for any value type `T` where `eq(T,T) -> bits[1]` is defined.
#[derive(Debug)]
pub struct EqSelDistributeTransform;

impl EqSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn eq_operands(payload: &NodePayload) -> Option<(NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(Binop::Eq, lhs, rhs) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    fn ne_operands(payload: &NodePayload) -> Option<(NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(Binop::Ne, lhs, rhs) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    fn cmp_operands(payload: &NodePayload, op: Binop) -> Option<(NodeRef, NodeRef)> {
        match op {
            Binop::Eq => Self::eq_operands(payload),
            Binop::Ne => Self::ne_operands(payload),
            _ => None,
        }
    }

    fn type_of(f: &IrFn, r: NodeRef) -> Type {
        f.get_node(r).ty.clone()
    }

    fn choose_fold_candidate(
        f: &IrFn,
        op: Binop,
        c1: NodeRef,
        c2: NodeRef,
    ) -> Option<(NodeRef, NodeRef, NodeRef)> {
        let (u, v) = Self::cmp_operands(&f.get_node(c1).payload, op)?;
        let (s, t) = Self::cmp_operands(&f.get_node(c2).payload, op)?;

        let mut candidates: Vec<(NodeRef, NodeRef, NodeRef)> = Vec::new();
        // If u is common
        if u == s {
            candidates.push((v, t, u));
        }
        if u == t {
            candidates.push((v, s, u));
        }
        // If v is common
        if v == s {
            candidates.push((u, t, v));
        }
        if v == t {
            candidates.push((u, s, v));
        }

        candidates.retain(|(a, b, c)| {
            let ta = Self::type_of(f, *a);
            ta == Self::type_of(f, *b) && ta == Self::type_of(f, *c)
        });

        candidates.sort_by_key(|(a, b, c)| (c.index, a.index, b.index));
        candidates.into_iter().next()
    }
}

impl PirTransform for EqSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::EqSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::Binop(Binop::Eq, lhs, rhs)
                | NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                    // eq/ne(sel(p,a,b), c) or eq/ne(c, sel(p,a,b))
                    let lhs_node = f.get_node(*lhs);
                    let rhs_node = f.get_node(*rhs);

                    if let Some((p, a, b)) = Self::sel2_parts(&lhs_node.payload) {
                        if Self::is_u1_selector(f, p)
                            && Self::type_of(f, a) == Self::type_of(f, *rhs)
                            && Self::type_of(f, b) == Self::type_of(f, *rhs)
                        {
                            out.push(TransformLocation::Node(nr));
                            continue;
                        }
                    }
                    if let Some((p, a, b)) = Self::sel2_parts(&rhs_node.payload) {
                        if Self::is_u1_selector(f, p)
                            && Self::type_of(f, a) == Self::type_of(f, *lhs)
                            && Self::type_of(f, b) == Self::type_of(f, *lhs)
                        {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    // sel(p, eq(a,c), eq(b,c)) or sel(p, ne(a,c), ne(b,c))
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    if Self::choose_fold_candidate(f, Binop::Eq, cases[0], cases[1]).is_some()
                        || Self::choose_fold_candidate(f, Binop::Ne, cases[0], cases[1]).is_some()
                    {
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
                    "EqSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Binop(Binop::Eq, lhs, rhs) | NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                let op = match &target_payload {
                    NodePayload::Binop(op, _, _) => *op,
                    _ => unreachable!(),
                };
                let (sel_ref, c_ref) = match Self::sel2_parts(&f.get_node(lhs).payload) {
                    Some(_) => (lhs, rhs),
                    None => match Self::sel2_parts(&f.get_node(rhs).payload) {
                        Some(_) => (rhs, lhs),
                        None => {
                            return Err(
                                "EqSelDistributeTransform: cmp node did not match {eq,ne}(sel(...), c)"
                                    .to_string(),
                            );
                        }
                    },
                };

                let (p, a, b) = Self::sel2_parts(&f.get_node(sel_ref).payload)
                    .expect("sel_ref should refer to sel with 2 cases");
                if !Self::is_u1_selector(f, p) {
                    return Err("EqSelDistributeTransform: sel selector is not bits[1]".to_string());
                }
                let c_ty = Self::type_of(f, c_ref);
                if Self::type_of(f, a) != c_ty || Self::type_of(f, b) != c_ty {
                    return Err(
                        "EqSelDistributeTransform: sel cases and c must have the same type"
                            .to_string(),
                    );
                }

                let mut next_text_id = Self::next_text_id(f);
                let eq_ty = Type::Bits(1);

                let eq_a_c_ref = {
                    let new_index = f.nodes.len();
                    f.nodes.push(xlsynth_pir::ir::Node {
                        text_id: next_text_id,
                        name: None,
                        ty: eq_ty.clone(),
                        payload: NodePayload::Binop(op, a, c_ref),
                        pos: None,
                    });
                    next_text_id = next_text_id.saturating_add(1);
                    NodeRef { index: new_index }
                };

                let eq_b_c_ref = {
                    let new_index = f.nodes.len();
                    f.nodes.push(xlsynth_pir::ir::Node {
                        text_id: next_text_id,
                        name: None,
                        ty: eq_ty,
                        payload: NodePayload::Binop(op, b, c_ref),
                        pos: None,
                    });
                    NodeRef { index: new_index }
                };

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![eq_a_c_ref, eq_b_c_ref],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "EqSelDistributeTransform: sel node did not match 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("EqSelDistributeTransform: sel selector is not bits[1]".to_string());
                }

                let (op, a, b, c) = if let Some((a, b, c)) =
                    Self::choose_fold_candidate(f, Binop::Eq, cases[0], cases[1])
                {
                    (Binop::Eq, a, b, c)
                } else if let Some((a, b, c)) =
                    Self::choose_fold_candidate(f, Binop::Ne, cases[0], cases[1])
                {
                    (Binop::Ne, a, b, c)
                } else {
                    return Err(
                        "EqSelDistributeTransform: sel cases did not match sel(p, {eq,ne}(a,c), {eq,ne}(b,c))"
                            .to_string(),
                    );
                };

                let next_text_id = Self::next_text_id(f);
                let sel_ty = Self::type_of(f, a);

                let sel_ab_ref = {
                    let new_index = f.nodes.len();
                    f.nodes.push(xlsynth_pir::ir::Node {
                        text_id: next_text_id,
                        name: None,
                        ty: sel_ty,
                        payload: NodePayload::Sel {
                            selector,
                            cases: vec![a, b],
                            default: None,
                        },
                        pos: None,
                    });
                    NodeRef { index: new_index }
                };

                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Eq, sel_ab_ref, c);
                if op == Binop::Ne {
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Binop(Binop::Ne, sel_ab_ref, c);
                }
                Ok(())
            }
            _ => Err(
                "EqSelDistributeTransform: expected eq or sel payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
