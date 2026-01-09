// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `xor(x, sign_ext(b,w)) ↔ sel(b, cases=[x, not(x)])`
#[derive(Debug)]
pub struct XorMaskSignExtToSelNotTransform;

impl XorMaskSignExtToSelNotTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sign_ext_mask_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::SignExt { arg, new_bit_count } => {
                if !Self::is_u1_selector(f, *arg) {
                    return None;
                }
                if Self::bits_width(f, r) != Some(*new_bit_count) {
                    return None;
                }
                Some((*arg, *new_bit_count))
            }
            _ => None,
        }
    }

    fn not_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Not, arg) => Some(*arg),
            _ => None,
        }
    }
}

impl PirTransform for XorMaskSignExtToSelNotTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::XorMaskSignExtToSelNot
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: xor(x, sign_ext(b,w))
                NodePayload::Nary(NaryOp::Xor, ops) if ops.len() == 2 => {
                    let (a, b) = (ops[0], ops[1]);
                    let mut ok = false;
                    for (x, mask) in [(a, b), (b, a)] {
                        if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                            if Self::bits_width(f, x) == Some(w)
                                && Self::bits_width(f, nr) == Some(w)
                            {
                                ok = true;
                                let _ = sel_b;
                            }
                        }
                    }
                    if ok {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // Fold: sel(b, cases=[x, not(x)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    let x = cases[0];
                    let not_x = cases[1];
                    if Self::bits_width(f, x) != Some(w) {
                        continue;
                    }
                    if Self::bits_width(f, not_x) != Some(w) {
                        continue;
                    }
                    if Self::not_arg(f, not_x) != Some(x) {
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
                    "XorMaskSignExtToSelNotTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Nary(NaryOp::Xor, ops) => {
                if ops.len() != 2 {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected 2-operand xor".to_string(),
                    );
                }
                let (a, b) = (ops[0], ops[1]);
                let mut matched: Option<(NodeRef, NodeRef, usize)> = None;
                for (x, mask) in [(a, b), (b, a)] {
                    if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                        if Self::bits_width(f, x) == Some(w) && Self::bits_width(f, target_ref) == Some(w) {
                            matched = Some((x, sel_b, w));
                            break;
                        }
                    }
                }
                let Some((x, sel_b, w)) = matched else {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected xor(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                // Create not(x) node.
                let text_id = Self::next_text_id(f);
                let not_index = f.nodes.len();
                f.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: Type::Bits(w),
                    payload: NodePayload::Unop(Unop::Not, x),
                    pos: None,
                });
                let not_x = NodeRef { index: not_index };

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: sel_b,
                    cases: vec![x, not_x],
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
                        "XorMaskSignExtToSelNotTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: selector must be bits[1]".to_string(),
                    );
                }
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "XorMaskSignExtToSelNotTransform: output must be bits[w]".to_string()
                })?;
                let x = cases[0];
                let not_x = cases[1];
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, not_x) != Some(w) {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: sel cases must be bits[w]".to_string(),
                    );
                }
                if Self::not_arg(f, not_x) != Some(x) {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected sel case1 to be not(case0)"
                            .to_string(),
                    );
                }

                // Create sign_ext(b,w) node.
                let text_id = Self::next_text_id(f);
                let se_index = f.nodes.len();
                f.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: Type::Bits(w),
                    payload: NodePayload::SignExt {
                        arg: selector,
                        new_bit_count: w,
                    },
                    pos: None,
                });
                let se_ref = NodeRef { index: se_index };

                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Xor, vec![x, se_ref]);
                Ok(())
            }
            _ => Err(
                "XorMaskSignExtToSelNotTransform: expected xor(...) or sel(...) payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
