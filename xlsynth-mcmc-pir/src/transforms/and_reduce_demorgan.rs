// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `and_reduce(x) ↔ not(or_reduce(not(x)))`
#[derive(Debug)]
pub struct AndReduceDeMorganTransform;

impl AndReduceDeMorganTransform {
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

    fn mk_unop_bits_node(f: &mut IrFn, op: Unop, w: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(op, arg),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for AndReduceDeMorganTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AndReduceDeMorgan
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::AndReduce, _) => {
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    let NodePayload::Unop(Unop::OrReduce, inner) = f.get_node(*arg).payload else {
                        continue;
                    };
                    let NodePayload::Unop(Unop::Not, _) = f.get_node(inner).payload else {
                        continue;
                    };
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
                    "AndReduceDeMorganTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        if Self::bits_width(f, target_ref) != Some(1) {
            return Err("AndReduceDeMorganTransform: output must be bits[1]".to_string());
        }

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // and_reduce(x) -> not(or_reduce(not(x)))
            NodePayload::Unop(Unop::AndReduce, x) => {
                let w = Self::bits_width(f, x).ok_or_else(|| {
                    "AndReduceDeMorganTransform: input must be bits[w]".to_string()
                })?;
                let not_x = Self::mk_unop_bits_node(f, Unop::Not, w, x);
                let or_reduce = Self::mk_unop_bits_node(f, Unop::OrReduce, 1, not_x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, or_reduce);
                Ok(())
            }

            // not(or_reduce(not(x))) -> and_reduce(x)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Unop(Unop::OrReduce, inner) = f.get_node(arg).payload else {
                    return Err(
                        "AndReduceDeMorganTransform: expected not(or_reduce(..))".to_string()
                    );
                };
                let NodePayload::Unop(Unop::Not, x) = f.get_node(inner).payload else {
                    return Err(
                        "AndReduceDeMorganTransform: expected not(or_reduce(not(..)))".to_string(),
                    );
                };
                if Self::bits_width(f, x).is_none() {
                    return Err("AndReduceDeMorganTransform: x must be bits[w]".to_string());
                }
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::AndReduce, x);
                Ok(())
            }
            _ => Err(
                "AndReduceDeMorganTransform: expected and_reduce(..) or not(or_reduce(not(..)))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
