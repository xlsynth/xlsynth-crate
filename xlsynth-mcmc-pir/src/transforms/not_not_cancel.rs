// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `not(not(x)) ↔ x`
///
/// Since there is no "alias" node payload in PIR, we represent `x` as
/// `identity(x)` at the IR node level.
#[derive(Debug)]
pub struct NotNotCancelTransform;

impl NotNotCancelTransform {
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

    fn mk_not_node(f: &mut IrFn, w: usize, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Not, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NotNotCancelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NotNotCancel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Not, inner) => {
                    if matches!(f.get_node(*inner).payload, NodePayload::Unop(Unop::Not, _)) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Unop(Unop::Identity, _) => {
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
                    "NotNotCancelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // not(not(x)) -> identity(x)
            NodePayload::Unop(Unop::Not, inner) => {
                let NodePayload::Unop(Unop::Not, x) = f.get_node(inner).payload.clone() else {
                    return Err("NotNotCancelTransform: expected not(not(x))".to_string());
                };
                let w = Self::bits_width(f, x)
                    .filter(|wx| Some(*wx) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NotNotCancelTransform: expected bits[w] types for x and output".to_string()
                    })?;
                let _ = w;
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, x);
                Ok(())
            }

            // identity(x) -> not(not(x))
            NodePayload::Unop(Unop::Identity, x) => {
                let w = Self::bits_width(f, x)
                    .filter(|wx| Some(*wx) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NotNotCancelTransform: expected bits[w] types for x and output".to_string()
                    })?;
                let not1 = Self::mk_not_node(f, w, x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, not1);
                Ok(())
            }
            _ => Err("NotNotCancelTransform: expected not(not(x)) or identity(x)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
