// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing small arithmetic reshapes:
///
/// - `add(add(a,b),c) ↔ add(a,add(b,c))`
/// - `sub(sub(a,b),c) ↔ sub(a,add(b,c))`
/// - `sub(a,sub(b,c)) ↔ add(sub(a,b),c)`
///
/// These are not “obvious simplifications”; they change the tree shape to give
/// MCMC more structural options, which is especially useful when optimizing
/// g8r-nodes×depth.
#[derive(Debug)]
pub struct ReassociateAddSubTransform;

impl ReassociateAddSubTransform {
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

    fn mk_binop_bits_node(f: &mut IrFn, op: Binop, w: usize, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Binop(op, a, b),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn is_bits_w(f: &IrFn, r: NodeRef, w: usize) -> bool {
        Self::bits_width(f, r) == Some(w)
    }
}

impl PirTransform for ReassociateAddSubTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ReassociateAddSub
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Add, a, b) => {
                    if matches!(f.get_node(*a).payload, NodePayload::Binop(Binop::Add, _, _))
                        || matches!(f.get_node(*b).payload, NodePayload::Binop(Binop::Add, _, _))
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Binop(Binop::Sub, a, b) => {
                    if matches!(f.get_node(*a).payload, NodePayload::Binop(Binop::Sub, _, _))
                        || matches!(f.get_node(*b).payload, NodePayload::Binop(Binop::Sub, _, _))
                        || matches!(f.get_node(*b).payload, NodePayload::Binop(Binop::Add, _, _))
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
                    "ReassociateAddSubTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "ReassociateAddSubTransform: output must be bits[w]".to_string())?;

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            NodePayload::Binop(Binop::Add, a0, b0) => {
                // add(add(a,b),c) -> add(a, add(b,c))
                if let NodePayload::Binop(Binop::Add, a, b) = f.get_node(a0).payload {
                    let c = b0;
                    if !Self::is_bits_w(f, a, w)
                        || !Self::is_bits_w(f, b, w)
                        || !Self::is_bits_w(f, c, w)
                    {
                        return Err(
                            "ReassociateAddSubTransform: operands must be bits[w]".to_string()
                        );
                    }
                    let bc = Self::mk_binop_bits_node(f, Binop::Add, w, b, c);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, a, bc);
                    return Ok(());
                }
                // add(a, add(b,c)) -> add(add(a,b), c)
                if let NodePayload::Binop(Binop::Add, b, c) = f.get_node(b0).payload {
                    let a = a0;
                    if !Self::is_bits_w(f, a, w)
                        || !Self::is_bits_w(f, b, w)
                        || !Self::is_bits_w(f, c, w)
                    {
                        return Err(
                            "ReassociateAddSubTransform: operands must be bits[w]".to_string()
                        );
                    }
                    let ab = Self::mk_binop_bits_node(f, Binop::Add, w, a, b);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, ab, c);
                    return Ok(());
                }
                // add(sub(a,b),c) -> sub(a, sub(b,c))
                if let NodePayload::Binop(Binop::Sub, a, b) = f.get_node(a0).payload {
                    let c = b0;
                    if !Self::is_bits_w(f, a, w)
                        || !Self::is_bits_w(f, b, w)
                        || !Self::is_bits_w(f, c, w)
                    {
                        return Err(
                            "ReassociateAddSubTransform: operands must be bits[w]".to_string()
                        );
                    }
                    let bc = Self::mk_binop_bits_node(f, Binop::Sub, w, b, c);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Sub, a, bc);
                    return Ok(());
                }
                // add(a, sub(b,c)) -> sub(add(a,c), b)  (a + (b - c) = (a - c)
                // + b etc.) Intentionally omitted for now; keep
                // the rewrite set small and symmetric.
            }
            NodePayload::Binop(Binop::Sub, a0, b0) => {
                // sub(sub(a,b),c) -> sub(a, add(b,c))
                if let NodePayload::Binop(Binop::Sub, a, b) = f.get_node(a0).payload {
                    let c = b0;
                    if !Self::is_bits_w(f, a, w)
                        || !Self::is_bits_w(f, b, w)
                        || !Self::is_bits_w(f, c, w)
                    {
                        return Err(
                            "ReassociateAddSubTransform: operands must be bits[w]".to_string()
                        );
                    }
                    let bc = Self::mk_binop_bits_node(f, Binop::Add, w, b, c);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Sub, a, bc);
                    return Ok(());
                }
                // sub(a, add(b,c)) -> sub(sub(a,b), c)
                if let NodePayload::Binop(Binop::Add, b, c) = f.get_node(b0).payload {
                    let a = a0;
                    if !Self::is_bits_w(f, a, w)
                        || !Self::is_bits_w(f, b, w)
                        || !Self::is_bits_w(f, c, w)
                    {
                        return Err(
                            "ReassociateAddSubTransform: operands must be bits[w]".to_string()
                        );
                    }
                    let ab = Self::mk_binop_bits_node(f, Binop::Sub, w, a, b);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Sub, ab, c);
                    return Ok(());
                }
                // sub(a, sub(b,c)) -> add(sub(a,b), c)
                if let NodePayload::Binop(Binop::Sub, b, c) = f.get_node(b0).payload {
                    let a = a0;
                    if !Self::is_bits_w(f, a, w)
                        || !Self::is_bits_w(f, b, w)
                        || !Self::is_bits_w(f, c, w)
                    {
                        return Err(
                            "ReassociateAddSubTransform: operands must be bits[w]".to_string()
                        );
                    }
                    let ab = Self::mk_binop_bits_node(f, Binop::Sub, w, a, b);
                    f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, ab, c);
                    return Ok(());
                }
            }
            _ => {}
        }

        Err("ReassociateAddSubTransform: no matching reshape pattern at target".to_string())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
