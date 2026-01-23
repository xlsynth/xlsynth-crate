// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `or(x, sign_ext(b,w)) ↔ sel(b, cases=[x, all_ones_w])`
#[derive(Debug)]
pub struct OrMaskSignExtToSelTransform;

impl OrMaskSignExtToSelTransform {
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

    fn all_ones_value(w: usize) -> IrValue {
        if w == 0 {
            IrValue::from_bits(&IrBits::make_ubits(0, 0).expect("make_ubits"))
        } else if w <= 64 {
            let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
            IrValue::from_bits(&IrBits::make_ubits(w, mask).expect("make_ubits"))
        } else {
            // Avoid assuming a bounded integer representation.
            let ones: Vec<bool> = vec![true; w];
            IrValue::from_bits(&IrBits::from_lsb_is_0(&ones))
        }
    }

    fn mk_all_ones_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let value = Self::all_ones_value(w);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn is_all_ones_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        *v == Self::all_ones_value(w)
    }
}

impl PirTransform for OrMaskSignExtToSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::OrMaskSignExtToSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: or(x, sign_ext(b,w))
                NodePayload::Nary(NaryOp::Or, ops) if ops.len() == 2 => {
                    let (a, b) = (ops[0], ops[1]);
                    let mut ok = false;
                    for (x, mask) in [(a, b), (b, a)] {
                        if let Some((_sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                            if Self::bits_width(f, x) == Some(w)
                                && Self::bits_width(f, nr) == Some(w)
                            {
                                ok = true;
                            }
                        }
                    }
                    if ok {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // Fold: sel(b, cases=[x, all_ones_w])
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
                    let ones = cases[1];
                    if Self::bits_width(f, x) != Some(w) {
                        continue;
                    }
                    if !Self::is_all_ones_literal_node(f, ones, w) {
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
                    "OrMaskSignExtToSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Nary(NaryOp::Or, ops) => {
                if ops.len() != 2 {
                    return Err("OrMaskSignExtToSelTransform: expected 2-operand or".to_string());
                }
                let (a, b) = (ops[0], ops[1]);
                let mut matched: Option<(NodeRef, NodeRef, usize)> = None;
                for (x, mask) in [(a, b), (b, a)] {
                    if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                        if Self::bits_width(f, x) == Some(w) && Self::bits_width(f, target_ref) == Some(w)
                        {
                            matched = Some((x, sel_b, w));
                            break;
                        }
                    }
                }
                let Some((x, sel_b, w)) = matched else {
                    return Err(
                        "OrMaskSignExtToSelTransform: expected or(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                let ones = Self::mk_all_ones_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: sel_b,
                    cases: vec![x, ones],
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
                        "OrMaskSignExtToSelTransform: expected 2-case sel without default".to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("OrMaskSignExtToSelTransform: selector must be bits[1]".to_string());
                }
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "OrMaskSignExtToSelTransform: output must be bits[w]".to_string()
                })?;

                let x = cases[0];
                let ones = cases[1];
                if Self::bits_width(f, x) != Some(w) {
                    return Err("OrMaskSignExtToSelTransform: expected sel case0 to be bits[w]".to_string());
                }
                if !Self::is_all_ones_literal_node(f, ones, w) {
                    return Err(
                        "OrMaskSignExtToSelTransform: expected sel case1 to be all_ones_w literal"
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

                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Or, vec![x, se_ref]);
                Ok(())
            }
            _ => Err(
                "OrMaskSignExtToSelTransform: expected or(...) or sel(...) payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
