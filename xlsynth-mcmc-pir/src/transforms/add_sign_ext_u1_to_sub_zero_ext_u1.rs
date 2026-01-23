// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `add(x, sign_ext(b,w)) ↔ sub(x, zero_ext(b,w))`
///
/// Where `b: bits[1]` and `x: bits[w]`.
#[derive(Debug)]
pub struct AddSignExtU1ToSubZeroExtU1Transform;

impl AddSignExtU1ToSubZeroExtU1Transform {
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

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sign_ext_u1_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        let NodePayload::SignExt { arg, new_bit_count } = &f.get_node(r).payload else {
            return None;
        };
        if !Self::is_u1(f, *arg) {
            return None;
        }
        if Self::bits_width(f, r) != Some(*new_bit_count) {
            return None;
        }
        Some((*arg, *new_bit_count))
    }

    fn zero_ext_u1_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::ZeroExt { arg, new_bit_count } => {
                if !Self::is_u1(f, *arg) {
                    return None;
                }
                if Self::bits_width(f, r) != Some(*new_bit_count) {
                    return None;
                }
                Some((*arg, *new_bit_count))
            }
            // Match `concat(0_{w-1}, b)` form as an alternative zero-extension encoding.
            NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                let hi = ops[0];
                let lo = ops[1];
                let w = Self::bits_width(f, r)?;
                if w == 0 {
                    return None;
                }
                if !Self::is_u1(f, lo) {
                    return None;
                }
                if Self::bits_width(f, hi) != Some(w.saturating_sub(1)) {
                    return None;
                }
                let NodePayload::Literal(v) = &f.get_node(hi).payload else {
                    return None;
                };
                let bits = IrBits::make_ubits(w.saturating_sub(1), 0).expect("make_ubits");
                let expected = IrValue::from_bits(&bits);
                if *v != expected {
                    return None;
                }
                Some((lo, w))
            }
            _ => None,
        }
    }

    fn mk_zero_ext_u1_node(f: &mut IrFn, w: usize, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::ZeroExt {
                arg: b,
                new_bit_count: w,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sign_ext_u1_node(f: &mut IrFn, w: usize, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::SignExt {
                arg: b,
                new_bit_count: w,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for AddSignExtU1ToSubZeroExtU1Transform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AddSignExtU1ToSubZeroExtU1
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: add(x, sign_ext(b,w)) (either operand).
                NodePayload::Binop(Binop::Add, a, b) => {
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    let mut ok = false;
                    for (x, sext) in [(*a, *b), (*b, *a)] {
                        if Self::bits_width(f, x) != Some(w) {
                            continue;
                        }
                        if Self::sign_ext_u1_parts(f, sext).is_some() {
                            ok = true;
                            break;
                        }
                    }
                    if ok {
                        out.push(TransformLocation::Node(nr));
                    }
                }

                // Fold: sub(x, zero_ext(b,w)) (zero_ext can also be concat(0...,b)).
                NodePayload::Binop(Binop::Sub, x, zext) => {
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    if Self::bits_width(f, *x) != Some(w) {
                        continue;
                    }
                    if Self::zero_ext_u1_parts(f, *zext).is_some() {
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
                    "AddSignExtU1ToSubZeroExtU1Transform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref).ok_or_else(|| {
            "AddSignExtU1ToSubZeroExtU1Transform: output must be bits[w]".to_string()
        })?;

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            // Expand: add(x, sign_ext(b,w)) -> sub(x, zero_ext(b,w))
            NodePayload::Binop(Binop::Add, a, b) => {
                let mut matched: Option<(NodeRef, NodeRef)> = None;
                for (x, sext) in [(a, b), (b, a)] {
                    if Self::bits_width(f, x) != Some(w) {
                        continue;
                    }
                    let Some((b_u1, sext_w)) = Self::sign_ext_u1_parts(f, sext) else {
                        continue;
                    };
                    if sext_w != w {
                        continue;
                    }
                    matched = Some((x, b_u1));
                    break;
                }
                let Some((x, b_u1)) = matched else {
                    return Err(
                        "AddSignExtU1ToSubZeroExtU1Transform: expected add(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                let zext = Self::mk_zero_ext_u1_node(f, w, b_u1);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Sub, x, zext);
                Ok(())
            }

            // Fold: sub(x, zero_ext(b,w)) -> add(x, sign_ext(b,w))
            NodePayload::Binop(Binop::Sub, x, zext) => {
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "AddSignExtU1ToSubZeroExtU1Transform: expected sub lhs to be bits[w]"
                            .to_string(),
                    );
                }
                let Some((b_u1, zext_w)) = Self::zero_ext_u1_parts(f, zext) else {
                    return Err(
                        "AddSignExtU1ToSubZeroExtU1Transform: expected sub(x, zero_ext(b,w)) pattern"
                            .to_string(),
                    );
                };
                if zext_w != w {
                    return Err(
                        "AddSignExtU1ToSubZeroExtU1Transform: expected zero_ext new_bit_count to match output width"
                            .to_string(),
                    );
                }
                let sext = Self::mk_sign_ext_u1_node(f, w, b_u1);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, x, sext);
                Ok(())
            }

            _ => {
                Err("AddSignExtU1ToSubZeroExtU1Transform: expected add(..) or sub(..)".to_string())
            }
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
