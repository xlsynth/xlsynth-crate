// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `nor(xs...) ↔ not(or(xs...))`
#[derive(Debug)]
pub struct NorNotOrFoldTransform;

impl NorNotOrFoldTransform {
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

    fn mk_nary_bits_node(f: &mut IrFn, op: NaryOp, w: usize, ops: Vec<NodeRef>) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Nary(op, ops),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    // no dedicated `not` node builder needed; the transform rewrites in-place
}

impl PirTransform for NorNotOrFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NorNotOrFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Nary(NaryOp::Nor, ops) if !ops.is_empty() => {
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(f.get_node(*arg).payload, NodePayload::Nary(NaryOp::Or, _)) {
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
                    "NorNotOrFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // nor(xs...) -> not(or(xs...))
            NodePayload::Nary(NaryOp::Nor, ops) => {
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "NorNotOrFoldTransform: expected bits[w] output for nor".to_string()
                })?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NorNotOrFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                let or_ref = Self::mk_nary_bits_node(f, NaryOp::Or, w, ops);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, or_ref);
                Ok(())
            }

            // not(or(xs...)) -> nor(xs...)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Nary(NaryOp::Or, ops) = f.get_node(arg).payload.clone() else {
                    return Err("NorNotOrFoldTransform: expected not(or(...))".to_string());
                };
                let w = Self::bits_width(f, target_ref)
                    .ok_or_else(|| "NorNotOrFoldTransform: expected bits[w] output".to_string())?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NorNotOrFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Nor, ops);
                Ok(())
            }
            _ => Err("NorNotOrFoldTransform: expected nor(...) or not(or(...))".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
