// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `and(x, sign_ext(b,w)) ↔ sel(b, cases=[0_w, x])`
#[derive(Debug)]
pub struct AndMaskSignExtToSelTransform;

impl AndMaskSignExtToSelTransform {
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
}

impl PirTransform for AndMaskSignExtToSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AndMaskSignExtToSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: and(x, sign_ext(b,w)) (nary)
                NodePayload::Nary(NaryOp::And, ops) if ops.len() == 2 => {
                    let (a, b) = (ops[0], ops[1]);
                    let mut found = None;
                    for (x, mask) in [(a, b), (b, a)] {
                        if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                            if Self::bits_width(f, x) == Some(w)
                                && Self::bits_width(f, nr) == Some(w)
                            {
                                found = Some((sel_b, w));
                            }
                        }
                    }
                    if found.is_some() {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // Fold: sel(b, cases=[0_w, x])
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
                    if !Self::is_zero_literal_node(f, cases[0], w) {
                        continue;
                    }
                    if Self::bits_width(f, cases[1]) != Some(w) {
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
                    "AndMaskSignExtToSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Nary(NaryOp::And, ops) => {
                if ops.len() != 2 {
                    return Err("AndMaskSignExtToSelTransform: expected 2-operand and".to_string());
                }
                let (a, b) = (ops[0], ops[1]);
                // Identify x and sign_ext(b,w)
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
                        "AndMaskSignExtToSelTransform: expected and(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                let zero = Self::mk_zero_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: sel_b,
                    cases: vec![zero, x],
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
                        "AndMaskSignExtToSelTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("AndMaskSignExtToSelTransform: selector must be bits[1]".to_string());
                }
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "AndMaskSignExtToSelTransform: output must be bits[w]".to_string()
                })?;
                let zero_case = cases[0];
                let x = cases[1];
                if !Self::is_zero_literal_node(f, zero_case, w) {
                    return Err(
                        "AndMaskSignExtToSelTransform: expected sel case0 to be 0_w literal"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "AndMaskSignExtToSelTransform: expected sel case1 to be bits[w]".to_string(),
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

                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::And, vec![x, se_ref]);
                Ok(())
            }
            _ => Err(
                "AndMaskSignExtToSelTransform: expected and(...) or sel(...) payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
