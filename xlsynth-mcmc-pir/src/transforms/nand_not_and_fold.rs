// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `nand(xs...) ↔ not(and(xs...))`
#[derive(Debug)]
pub struct NandNotAndFoldTransform;

impl NandNotAndFoldTransform {
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

impl PirTransform for NandNotAndFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NandNotAndFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Nary(NaryOp::Nand, ops) if !ops.is_empty() => {
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(f.get_node(*arg).payload, NodePayload::Nary(NaryOp::And, _)) {
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
                    "NandNotAndFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // nand(xs...) -> not(and(xs...))
            NodePayload::Nary(NaryOp::Nand, ops) => {
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "NandNotAndFoldTransform: expected bits[w] output for nand".to_string()
                })?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NandNotAndFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                let and_ref = Self::mk_nary_bits_node(f, NaryOp::And, w, ops);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, and_ref);
                Ok(())
            }

            // not(and(xs...)) -> nand(xs...)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Nary(NaryOp::And, ops) = f.get_node(arg).payload.clone() else {
                    return Err("NandNotAndFoldTransform: expected not(and(...))".to_string());
                };
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "NandNotAndFoldTransform: expected bits[w] output".to_string()
                })?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NandNotAndFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Nand, ops);
                Ok(())
            }
            _ => Err("NandNotAndFoldTransform: expected nand(...) or not(and(...))".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
