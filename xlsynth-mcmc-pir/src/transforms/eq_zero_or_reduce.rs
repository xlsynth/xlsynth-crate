// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `eq(x, 0_w) ↔ not(or_reduce(x))`
#[derive(Debug)]
pub struct EqZeroOrReduceTransform;

impl EqZeroOrReduceTransform {
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

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn mk_zero_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_or_reduce_node(f: &mut IrFn, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Unop(Unop::OrReduce, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for EqZeroOrReduceTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::EqZeroOrReduce
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Eq, lhs, rhs) => {
                    let Some(w) = Self::bits_width(f, *lhs) else {
                        continue;
                    };
                    if Self::bits_width(f, *rhs) != Some(w) {
                        continue;
                    }
                    if Self::is_zero_literal_node(f, *lhs, w)
                        || Self::is_zero_literal_node(f, *rhs, w)
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(
                        f.get_node(*arg).payload,
                        NodePayload::Unop(Unop::OrReduce, _)
                    ) {
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
                    "EqZeroOrReduceTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        if !matches!(f.get_node(target_ref).ty, Type::Bits(1)) {
            return Err("EqZeroOrReduceTransform: output must be bits[1]".to_string());
        }

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // eq(x,0) -> not(or_reduce(x))
            NodePayload::Binop(Binop::Eq, lhs, rhs) => {
                let Some(w) = Self::bits_width(f, lhs) else {
                    return Err("EqZeroOrReduceTransform: x must be bits[w]".to_string());
                };
                if Self::bits_width(f, rhs) != Some(w) {
                    return Err(
                        "EqZeroOrReduceTransform: operands must have matching bits[w] types"
                            .to_string(),
                    );
                }
                let x = if Self::is_zero_literal_node(f, lhs, w) {
                    rhs
                } else if Self::is_zero_literal_node(f, rhs, w) {
                    lhs
                } else {
                    return Err("EqZeroOrReduceTransform: expected eq(x,0_w)".to_string());
                };
                let orr = Self::mk_or_reduce_node(f, x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, orr);
                Ok(())
            }

            // not(or_reduce(x)) -> eq(x,0)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Unop(Unop::OrReduce, x) = f.get_node(arg).payload.clone() else {
                    return Err("EqZeroOrReduceTransform: expected not(or_reduce(x))".to_string());
                };
                let w = Self::bits_width(f, x).ok_or_else(|| {
                    "EqZeroOrReduceTransform: or_reduce arg must be bits[w]".to_string()
                })?;
                let zero = Self::mk_zero_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Eq, x, zero);
                Ok(())
            }
            _ => Err("EqZeroOrReduceTransform: expected eq(x,0) or not(or_reduce(x))".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
