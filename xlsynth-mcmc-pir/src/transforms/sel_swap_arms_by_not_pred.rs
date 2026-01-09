// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `sel(not(p), cases=[a, b]) ↔ sel(p, cases=[b, a])`
#[derive(Debug)]
pub struct SelSwapArmsByNotPredTransform;

impl SelSwapArmsByNotPredTransform {
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

    fn not_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Not, arg) => Some(*arg),
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

    fn mk_not_node(f: &mut IrFn, p: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Unop(Unop::Not, p),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for SelSwapArmsByNotPredTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelSwapArmsByNotPred
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let Some((sel, a, b)) = Self::sel2_parts(&f.get_node(nr).payload) else {
                continue;
            };
            let _ = (a, b);
            // We allow both directions:
            // - selector == not(p)  => remove not, swap arms
            // - selector == p       => add not, swap arms
            if Self::is_u1_selector(f, sel) {
                out.push(TransformLocation::Node(nr));
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SelSwapArmsByNotPredTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err("SelSwapArmsByNotPredTransform: expected sel payload".to_string());
        };
        if cases.len() != 2 || default.is_some() {
            return Err(
                "SelSwapArmsByNotPredTransform: expected 2-case sel without default".to_string(),
            );
        }

        // swap arms
        let a = cases[0];
        let b = cases[1];

        let new_selector = if let Some(p) = Self::not_arg(f, selector) {
            if !Self::is_u1_selector(f, p) {
                return Err("SelSwapArmsByNotPredTransform: not(p) arg must be bits[1]".to_string());
            }
            p
        } else {
            if !Self::is_u1_selector(f, selector) {
                return Err("SelSwapArmsByNotPredTransform: selector must be bits[1]".to_string());
            }
            Self::mk_not_node(f, selector)
        };

        f.get_node_mut(target_ref).payload = NodePayload::Sel {
            selector: new_selector,
            cases: vec![b, a],
            default: None,
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
