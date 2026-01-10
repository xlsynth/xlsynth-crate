// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `not(eq(a,b)) ↔ ne(a,b)` and `not(ne(a,b)) ↔ eq(a,b)`
#[derive(Debug)]
pub struct NotEqNeFlipTransform;

impl NotEqNeFlipTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).ty, Type::Bits(1))
    }

    fn mk_cmp_node(f: &mut IrFn, op: Binop, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    // (no helper for building `not` nodes needed here)
}

impl PirTransform for NotEqNeFlipTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NotEqNeFlip
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(
                        f.get_node(*arg).payload,
                        NodePayload::Binop(Binop::Eq, _, _) | NodePayload::Binop(Binop::Ne, _, _)
                    ) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Binop(Binop::Eq, _, _) | NodePayload::Binop(Binop::Ne, _, _) => {
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
                    "NotEqNeFlipTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // not(eq(..)) -> ne(..) and not(ne(..)) -> eq(..)
            NodePayload::Unop(Unop::Not, arg) => {
                if !Self::is_u1(f, target_ref) {
                    return Err("NotEqNeFlipTransform: output must be bits[1]".to_string());
                }
                let arg_payload = f.get_node(arg).payload.clone();
                match arg_payload {
                    NodePayload::Binop(Binop::Eq, lhs, rhs) => {
                        f.get_node_mut(target_ref).payload =
                            NodePayload::Binop(Binop::Ne, lhs, rhs);
                        Ok(())
                    }
                    NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                        f.get_node_mut(target_ref).payload =
                            NodePayload::Binop(Binop::Eq, lhs, rhs);
                        Ok(())
                    }
                    _ => Err("NotEqNeFlipTransform: expected not(eq/ne(...))".to_string()),
                }
            }

            // eq/ne(a,b) -> not(ne/eq(a,b))
            NodePayload::Binop(op, lhs, rhs) if matches!(op, Binop::Eq | Binop::Ne) => {
                if !Self::is_u1(f, target_ref) {
                    return Err("NotEqNeFlipTransform: output must be bits[1]".to_string());
                }
                let flipped = if op == Binop::Eq {
                    Binop::Ne
                } else {
                    Binop::Eq
                };
                let cmp_ref = Self::mk_cmp_node(f, flipped, lhs, rhs);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, cmp_ref);
                Ok(())
            }
            _ => Err("NotEqNeFlipTransform: expected not(eq/ne(..)) or eq/ne(..)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
