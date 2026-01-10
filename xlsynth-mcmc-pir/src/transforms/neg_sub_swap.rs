// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `neg(sub(x, y)) ↔ sub(y, x)`
#[derive(Debug)]
pub struct NegSubSwapTransform;

impl NegSubSwapTransform {
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

    fn mk_sub_node(f: &mut IrFn, w: usize, x: NodeRef, y: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Binop(Binop::Sub, x, y),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NegSubSwapTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NegSubSwap
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Neg, arg) => {
                    if matches!(
                        f.get_node(*arg).payload,
                        NodePayload::Binop(Binop::Sub, _, _)
                    ) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Binop(Binop::Sub, _, _) => out.push(TransformLocation::Node(nr)),
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
                    "NegSubSwapTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "NegSubSwapTransform: output must be bits[w]".to_string())?;

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            // neg(sub(x, y)) -> sub(y, x)
            NodePayload::Unop(Unop::Neg, arg) => {
                let NodePayload::Binop(Binop::Sub, x, y) = f.get_node(arg).payload else {
                    return Err("NegSubSwapTransform: expected neg(sub(..))".to_string());
                };
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, y) != Some(w) {
                    return Err("NegSubSwapTransform: operands must be bits[w]".to_string());
                }
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Sub, y, x);
                Ok(())
            }

            // sub(y, x) -> neg(sub(x, y))
            NodePayload::Binop(Binop::Sub, y, x) => {
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, y) != Some(w) {
                    return Err("NegSubSwapTransform: operands must be bits[w]".to_string());
                }
                let inner_sub = Self::mk_sub_node(f, w, x, y);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Neg, inner_sub);
                Ok(())
            }

            _ => Err("NegSubSwapTransform: expected neg(sub(..)) or sub(..)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
