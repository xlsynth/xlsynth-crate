// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// - `eq(add(x, k_lit), c) ↔ eq(x, sub(c, k_lit))`
/// - `ne(add(x, k_lit), c) ↔ ne(x, sub(c, k_lit))`
///
/// This relies on the standard XLS bit-vector semantics where `add`/`sub` on
/// `bits[w]` are performed modulo \(2^w\).
#[derive(Debug)]
pub struct EqNeAddLiteralShiftTransform;

impl EqNeAddLiteralShiftTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_bits_type(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn is_cmp_op(op: Binop) -> bool {
        matches!(op, Binop::Eq | Binop::Ne)
    }

    fn is_literal(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).payload, NodePayload::Literal(_))
    }

    fn add_with_literal_parts(f: &IrFn, add_ref: NodeRef) -> Option<(NodeRef, NodeRef)> {
        match &f.get_node(add_ref).payload {
            NodePayload::Binop(Binop::Add, lhs, rhs) => {
                if Self::is_literal(f, *lhs) {
                    Some((*rhs, *lhs))
                } else if Self::is_literal(f, *rhs) {
                    Some((*lhs, *rhs))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn sub_with_literal_parts(f: &IrFn, sub_ref: NodeRef) -> Option<(NodeRef, NodeRef)> {
        match &f.get_node(sub_ref).payload {
            NodePayload::Binop(Binop::Sub, lhs, rhs) => {
                if Self::is_literal(f, *rhs) {
                    Some((*lhs, *rhs))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn mk_binop_node(f: &mut IrFn, op: Binop, ty: Type, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(xlsynth_pir::ir::Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Binop(op, a, b),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn apply_cmp_shift(
        f: &mut IrFn,
        target_ref: NodeRef,
        op: Binop,
        lhs: NodeRef,
        rhs: NodeRef,
    ) -> Result<(), String> {
        // Direction A: cmp(add(x,k), c) -> cmp(x, sub(c,k))
        if let Some((x, k_lit)) = Self::add_with_literal_parts(f, lhs) {
            let w = Self::is_bits_type(f, x).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, rhs) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            if Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }

            let sub_ref = Self::mk_binop_node(f, Binop::Sub, Type::Bits(w), rhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, x, sub_ref);
            return Ok(());
        }
        if let Some((x, k_lit)) = Self::add_with_literal_parts(f, rhs) {
            let w = Self::is_bits_type(f, x).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, lhs) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            if Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }

            let sub_ref = Self::mk_binop_node(f, Binop::Sub, Type::Bits(w), lhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, x, sub_ref);
            return Ok(());
        }

        // Direction B (fold): cmp(x, sub(c,k)) -> cmp(add(x,k), c)
        if let Some((c, k_lit)) = Self::sub_with_literal_parts(f, rhs) {
            let w = Self::is_bits_type(f, lhs).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, c) != Some(w) || Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c and k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            let add_ref = Self::mk_binop_node(f, Binop::Add, Type::Bits(w), lhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, add_ref, c);
            return Ok(());
        }
        if let Some((c, k_lit)) = Self::sub_with_literal_parts(f, lhs) {
            let w = Self::is_bits_type(f, rhs).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, c) != Some(w) || Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c and k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            let add_ref = Self::mk_binop_node(f, Binop::Add, Type::Bits(w), rhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, add_ref, c);
            return Ok(());
        }

        Err(
            "EqNeAddLiteralShiftTransform: target did not match expected cmp/add/sub-with-literal patterns"
                .to_string(),
        )
    }
}

impl PirTransform for EqNeAddLiteralShiftTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::EqNeAddLiteralShift
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match node.payload {
                NodePayload::Binop(op, lhs, rhs) if Self::is_cmp_op(op) => {
                    // Expand direction:
                    //   cmp(add(x,k_lit), c) or cmp(c, add(x,k_lit))
                    if Self::add_with_literal_parts(f, lhs).is_some()
                        || Self::add_with_literal_parts(f, rhs).is_some()
                    {
                        out.push(TransformLocation::Node(nr));
                        continue;
                    }
                    // Fold direction:
                    //   cmp(x, sub(c,k_lit)) or cmp(sub(c,k_lit), x)
                    if Self::sub_with_literal_parts(f, lhs).is_some()
                        || Self::sub_with_literal_parts(f, rhs).is_some()
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
                    "EqNeAddLiteralShiftTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Binop(op, lhs, rhs) if Self::is_cmp_op(op) => {
                Self::apply_cmp_shift(f, target_ref, op, lhs, rhs)
            }
            _ => Err(
                "EqNeAddLiteralShiftTransform: expected eq/ne binop payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
