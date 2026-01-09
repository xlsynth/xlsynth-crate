// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `sub(x, y) ↔ add(x, neg(y))`
#[derive(Debug)]
pub struct SubToAddNegTransform;

impl SubToAddNegTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
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
}

impl PirTransform for SubToAddNegTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SubToAddNeg
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Sub, _, _) => out.push(TransformLocation::Node(nr)),
                NodePayload::Binop(Binop::Add, a, b) => {
                    if matches!(f.get_node(*a).payload, NodePayload::Unop(Unop::Neg, _))
                        || matches!(f.get_node(*b).payload, NodePayload::Unop(Unop::Neg, _))
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
                    "SubToAddNegTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "SubToAddNegTransform: output must be bits[w]".to_string())?;

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            // sub(x, y) -> add(x, neg(y))
            NodePayload::Binop(Binop::Sub, x, y) => {
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, y) != Some(w) {
                    return Err("SubToAddNegTransform: operands must be bits[w]".to_string());
                }
                let neg_y = Self::mk_neg_node(f, w, y);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, x, neg_y);
                Ok(())
            }

            // add(x, neg(y)) -> sub(x, y) (either arm)
            NodePayload::Binop(Binop::Add, a, b) => {
                let (x, y) = match (f.get_node(a).payload.clone(), f.get_node(b).payload.clone()) {
                    (NodePayload::Unop(Unop::Neg, y), _) => (b, y),
                    (_, NodePayload::Unop(Unop::Neg, y)) => (a, y),
                    _ => {
                        return Err("SubToAddNegTransform: expected add(x, neg(y))".to_string());
                    }
                };
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, y) != Some(w) {
                    return Err("SubToAddNegTransform: operands must be bits[w]".to_string());
                }
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Sub, x, y);
                Ok(())
            }

            _ => Err("SubToAddNegTransform: expected sub(..) or add(..)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
