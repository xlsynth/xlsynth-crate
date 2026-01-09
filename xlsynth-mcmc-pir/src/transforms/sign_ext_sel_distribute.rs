// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `sign_ext(sel(p, cases=[a, b]), new_bit_count=n)
///    ↔ sel(p, cases=[sign_ext(a,n), sign_ext(b,n)])`
#[derive(Debug)]
pub struct SignExtSelDistributeTransform;

impl SignExtSelDistributeTransform {
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

    fn sign_ext_parts(payload: &NodePayload) -> Option<(NodeRef, usize)> {
        match payload {
            NodePayload::SignExt { arg, new_bit_count } => Some((*arg, *new_bit_count)),
            _ => None,
        }
    }

    fn mk_sign_ext_node(f: &mut IrFn, out_w: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::SignExt {
                arg,
                new_bit_count: out_w,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(
        f: &mut IrFn,
        out_w: usize,
        selector: NodeRef,
        a: NodeRef,
        b: NodeRef,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for SignExtSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SignExtSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: sign_ext(sel(...), n)
                NodePayload::SignExt { arg, new_bit_count } => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wout == Some(*new_bit_count) {
                            if *new_bit_count >= wa.unwrap() {
                                out.push(TransformLocation::Node(nr));
                            }
                        }
                    }
                }
                // Fold: sel(p, [sign_ext(a,n), sign_ext(b,n)])
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
                    let (a_arg, a_n) = match Self::sign_ext_parts(&f.get_node(cases[0]).payload) {
                        Some(v) => v,
                        None => continue,
                    };
                    let (b_arg, b_n) = match Self::sign_ext_parts(&f.get_node(cases[1]).payload) {
                        Some(v) => v,
                        None => continue,
                    };
                    if a_n != b_n {
                        continue;
                    }
                    let wa = Self::bits_width(f, a_arg);
                    let wb = Self::bits_width(f, b_arg);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wout == Some(a_n) {
                        if a_n >= wa.unwrap() {
                            out.push(TransformLocation::Node(nr));
                        }
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
                    "SignExtSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: sign_ext(sel(...), n) -> sel(...sign_ext...)
            NodePayload::SignExt { arg, new_bit_count } => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg).payload).ok_or_else(|| {
                        "SignExtSelDistributeTransform: expected sign_ext(sel(p, cases=[a,b]), ...)"
                            .to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err(
                        "SignExtSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .ok_or_else(|| {
                        "SignExtSelDistributeTransform: sel cases must be bits[w] with same width"
                            .to_string()
                    })?;
                if new_bit_count < in_w {
                    return Err(
                        "SignExtSelDistributeTransform: new_bit_count must be >= input width"
                            .to_string(),
                    );
                }
                let out_w = new_bit_count;
                if Self::bits_width(f, target_ref) != Some(out_w) {
                    return Err(
                        "SignExtSelDistributeTransform: output type must be bits[new_bit_count]"
                            .to_string(),
                    );
                }

                let se_a = Self::mk_sign_ext_node(f, out_w, a);
                let se_b = Self::mk_sign_ext_node(f, out_w, b);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![se_a, se_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p,[sign_ext(a,n),sign_ext(b,n)]) -> sign_ext(sel(p,[a,b]), n)
            NodePayload::Sel { selector, cases, default } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "SignExtSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "SignExtSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let (a_arg, a_n) =
                    Self::sign_ext_parts(&f.get_node(cases[0]).payload).ok_or_else(|| {
                        "SignExtSelDistributeTransform: expected sel case 0 to be sign_ext(a,...)"
                            .to_string()
                    })?;
                let (b_arg, b_n) =
                    Self::sign_ext_parts(&f.get_node(cases[1]).payload).ok_or_else(|| {
                        "SignExtSelDistributeTransform: expected sel case 1 to be sign_ext(b,...)"
                            .to_string()
                    })?;
                if a_n != b_n {
                    return Err(
                        "SignExtSelDistributeTransform: new_bit_count must match across cases"
                            .to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a_arg)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b_arg))
                    .ok_or_else(|| {
                        "SignExtSelDistributeTransform: sign_ext args must be bits[w] with same width"
                            .to_string()
                    })?;
                if a_n < in_w {
                    return Err(
                        "SignExtSelDistributeTransform: new_bit_count must be >= input width"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, target_ref) != Some(a_n) {
                    return Err(
                        "SignExtSelDistributeTransform: output type must be bits[new_bit_count]"
                            .to_string(),
                    );
                }

                let sel_ab = Self::mk_sel2_node(f, in_w, selector, a_arg, b_arg);
                f.get_node_mut(target_ref).payload = NodePayload::SignExt {
                    arg: sel_ab,
                    new_bit_count: a_n,
                };
                Ok(())
            }
            _ => Err(
                "SignExtSelDistributeTransform: expected sign_ext(sel(...)) or sel(sign_ext(...),sign_ext(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
