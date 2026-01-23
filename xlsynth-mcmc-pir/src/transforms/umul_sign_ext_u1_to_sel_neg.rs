// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `umul(x, sign_ext(b,w)) ↔ sel(b, cases=[0_w, neg(x)])`
///
/// Where `b: bits[1]` and `x: bits[w]`.
#[derive(Debug)]
pub struct UmulSignExtU1ToSelNegTransform;

impl UmulSignExtU1ToSelNegTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sign_ext_u1_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        let NodePayload::SignExt { arg, new_bit_count } = &f.get_node(r).payload else {
            return None;
        };
        if !Self::is_u1(f, *arg) {
            return None;
        }
        if Self::bits_width(f, r) != Some(*new_bit_count) {
            return None;
        }
        Some((*arg, *new_bit_count))
    }

    fn mk_sign_ext_u1_node(f: &mut IrFn, w: usize, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::SignExt {
                arg: b,
                new_bit_count: w,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_zero_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn neg_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Neg, arg) => Some(*arg),
            _ => None,
        }
    }

    fn mk_neg_node(f: &mut IrFn, w: usize, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Neg, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for UmulSignExtU1ToSelNegTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::UmulSignExtU1ToSelNeg
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: umul(x, sign_ext(b,w)) (either operand).
                NodePayload::Binop(Binop::Umul, a, b) => {
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    let mut ok = false;
                    for (x, sext) in [(*a, *b), (*b, *a)] {
                        if Self::bits_width(f, x) != Some(w) {
                            continue;
                        }
                        if Self::sign_ext_u1_parts(f, sext).is_some() {
                            ok = true;
                            break;
                        }
                    }
                    if ok {
                        out.push(TransformLocation::Node(nr));
                    }
                }

                // Fold: sel(b, cases=[0_w, neg(x)]).
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1(f, *selector) {
                        continue;
                    }
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    let zero = cases[0];
                    let neg_x = cases[1];
                    if !Self::is_zero_literal_node(f, zero, w) {
                        continue;
                    }
                    if Self::bits_width(f, neg_x) != Some(w) {
                        continue;
                    }
                    let Some(x) = Self::neg_arg(f, neg_x) else {
                        continue;
                    };
                    if Self::bits_width(f, x) != Some(w) {
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
                return Err(
                    "UmulSignExtU1ToSelNegTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "UmulSignExtU1ToSelNegTransform: output must be bits[w]".to_string())?;

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            // Expand: umul(x, sign_ext(b,w)) -> sel(b, [0, neg(x)])
            NodePayload::Binop(Binop::Umul, a, b) => {
                let mut matched: Option<(NodeRef, NodeRef)> = None;
                for (x, sext) in [(a, b), (b, a)] {
                    if Self::bits_width(f, x) != Some(w) {
                        continue;
                    }
                    let Some((b_u1, sext_w)) = Self::sign_ext_u1_parts(f, sext) else {
                        continue;
                    };
                    if sext_w != w {
                        continue;
                    }
                    matched = Some((x, b_u1));
                    break;
                }
                let Some((x, b_u1)) = matched else {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: expected umul(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                let zero = Self::mk_zero_literal_node(f, w);
                let neg_x = Self::mk_neg_node(f, w, x);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: b_u1,
                    cases: vec![zero, neg_x],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(b, [0, neg(x)]) -> umul(x, sign_ext(b,w))
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1(f, selector) {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: selector must be bits[1]".to_string()
                    );
                }
                let zero = cases[0];
                let neg_x = cases[1];
                if !Self::is_zero_literal_node(f, zero, w) {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: expected sel case0 to be 0_w literal"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, neg_x) != Some(w) {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: expected sel case1 to be bits[w]"
                            .to_string(),
                    );
                }
                let Some(x) = Self::neg_arg(f, neg_x) else {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: expected sel case1 to be neg(x)"
                            .to_string(),
                    );
                };
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "UmulSignExtU1ToSelNegTransform: expected x to be bits[w]".to_string()
                    );
                }

                let sext = Self::mk_sign_ext_u1_node(f, w, selector);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Umul, x, sext);
                Ok(())
            }

            _ => Err("UmulSignExtU1ToSelNegTransform: expected umul(..) or sel(..)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
