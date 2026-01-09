// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `neg(sel(p, cases=[a, b])) ↔ sel(p, cases=[neg(a), neg(b)])`
///
/// This is valid for `bits[w]` values, where `neg` is two's-complement
/// arithmetic negation modulo \(2^w\).
#[derive(Debug)]
pub struct NegSelDistributeTransform;

impl NegSelDistributeTransform {
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

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn neg_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Neg, arg) => Some(*arg),
            _ => None,
        }
    }

    fn mk_neg_node(f: &mut IrFn, w: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Neg, arg),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(f: &mut IrFn, w: usize, selector: NodeRef, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
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

impl PirTransform for NegSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NegSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: neg(sel(p, [a,b]))
                NodePayload::Unop(Unop::Neg, arg) => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wa == wout {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                // Fold: sel(p, [neg(a), neg(b)])
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
                    let a = match Self::neg_arg(f, cases[0]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let b = match Self::neg_arg(f, cases[1]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
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
                    "NegSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: neg(sel(p,[a,b])) -> sel(p,[neg(a),neg(b)])
            NodePayload::Unop(Unop::Neg, arg_sel_ref) => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg_sel_ref).payload).ok_or_else(|| {
                        "NegSelDistributeTransform: expected neg(sel(p, cases=[a,b]))".to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err("NegSelDistributeTransform: selector must be bits[1]".to_string());
                }
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NegSelDistributeTransform: sel cases and output must be bits[w] with same width"
                            .to_string()
                    })?;

                let neg_a = Self::mk_neg_node(f, w, a);
                let neg_b = Self::mk_neg_node(f, w, b);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![neg_a, neg_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p,[neg(a),neg(b)]) -> neg(sel(p,[a,b]))
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "NegSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("NegSelDistributeTransform: selector must be bits[1]".to_string());
                }
                let a = Self::neg_arg(f, cases[0]).ok_or_else(|| {
                    "NegSelDistributeTransform: expected sel case 0 to be neg(a)".to_string()
                })?;
                let b = Self::neg_arg(f, cases[1]).ok_or_else(|| {
                    "NegSelDistributeTransform: expected sel case 1 to be neg(b)".to_string()
                })?;

                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NegSelDistributeTransform: sel cases and output must be bits[w] with same width"
                            .to_string()
                    })?;

                let sel_ab = Self::mk_sel2_node(f, w, selector, a, b);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Neg, sel_ab);
                Ok(())
            }
            _ => Err(
                "NegSelDistributeTransform: expected neg(sel(...)) or sel(neg(...),neg(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
