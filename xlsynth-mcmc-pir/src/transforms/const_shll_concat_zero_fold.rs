// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `concat(bit_slice(x, start=0, width=w-k), 0_k) ↔ shll(x, k)`
#[derive(Debug)]
pub struct ConstShllConcatZeroFoldTransform;

impl ConstShllConcatZeroFoldTransform {
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

    fn mk_ubits_literal_node(f: &mut IrFn, w: usize, value: u64) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, value).expect("make_ubits");
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

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn literal_u64_value(f: &IrFn, r: NodeRef) -> Option<u64> {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return None;
        };
        // This helper is only used for small shift constants in tests/transforms.
        v.to_u64().ok()
    }
}

impl PirTransform for ConstShllConcatZeroFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ConstShllConcatZeroFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // concat(bit_slice(x, 0, w-k), 0_k) -> shll(x, k)
                NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                    let (a, b) = (ops[0], ops[1]);
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    let (wa, wb) = match (Self::bits_width(f, a), Self::bits_width(f, b)) {
                        (Some(wa), Some(wb)) => (wa, wb),
                        _ => continue,
                    };
                    if wa + wb != w {
                        continue;
                    }
                    if wb == 0 || wb >= w {
                        continue;
                    }
                    if !Self::is_zero_literal_node(f, b, wb) {
                        continue;
                    }
                    let NodePayload::BitSlice { arg, start, width } = f.get_node(a).payload else {
                        continue;
                    };
                    if start != 0 || width != wa {
                        continue;
                    }
                    if Self::bits_width(f, arg) != Some(w) {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }

                // shll(x, k) -> concat(bit_slice(x, 0, w-k), 0_k) for constant k
                NodePayload::Binop(Binop::Shll, x, k) => {
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    if Self::bits_width(f, *x) != Some(w) {
                        continue;
                    }
                    let Some(k_u64) = Self::literal_u64_value(f, *k) else {
                        continue;
                    };
                    let Ok(k_usize) = usize::try_from(k_u64) else {
                        continue;
                    };
                    if k_usize == 0 || k_usize >= w {
                        continue;
                    }
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
                    "ConstShllConcatZeroFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref).ok_or_else(|| {
            "ConstShllConcatZeroFoldTransform: output must be bits[w]".to_string()
        })?;

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // concat(bit_slice(x, 0, w-k), 0_k) -> shll(x, k)
            NodePayload::Nary(NaryOp::Concat, ops) => {
                if ops.len() != 2 {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: only supports 2-operand concat"
                            .to_string(),
                    );
                }
                let a = ops[0];
                let b = ops[1];
                let (wa, wb) = match (Self::bits_width(f, a), Self::bits_width(f, b)) {
                    (Some(wa), Some(wb)) => (wa, wb),
                    _ => {
                        return Err(
                            "ConstShllConcatZeroFoldTransform: concat operands must be bits"
                                .to_string(),
                        );
                    }
                };
                if wa + wb != w {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: concat widths must sum to output width"
                            .to_string(),
                    );
                }
                if wb == 0 || wb >= w {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: shift amount must be in (0, w)"
                            .to_string(),
                    );
                }
                if !Self::is_zero_literal_node(f, b, wb) {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: expected RHS concat operand to be 0_k"
                            .to_string(),
                    );
                }
                let NodePayload::BitSlice {
                    arg: x,
                    start,
                    width,
                } = f.get_node(a).payload
                else {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: expected concat LHS to be bit_slice"
                            .to_string(),
                    );
                };
                if start != 0 || width != wa {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: expected bit_slice(x, start=0, width=w-k)"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: expected bit_slice arg bits[w]"
                            .to_string(),
                    );
                }
                let k_lit = Self::mk_ubits_literal_node(f, w, wb as u64);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Shll, x, k_lit);
                Ok(())
            }

            // shll(x, k) -> concat(bit_slice(x, 0, w-k), 0_k)
            NodePayload::Binop(Binop::Shll, x, k) => {
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: expected x to be bits[w]".to_string()
                    );
                }
                let Some(k_u64) = Self::literal_u64_value(f, k) else {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: expected constant shift amount"
                            .to_string(),
                    );
                };
                let k_usize = usize::try_from(k_u64).map_err(|_| {
                    "ConstShllConcatZeroFoldTransform: shift amount out of range".to_string()
                })?;
                if k_usize == 0 || k_usize >= w {
                    return Err(
                        "ConstShllConcatZeroFoldTransform: shift amount must be in (0, w)"
                            .to_string(),
                    );
                }
                let slice_w = w - k_usize;
                let text_id = Self::next_text_id(f);
                let slice_index = f.nodes.len();
                f.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: Type::Bits(slice_w),
                    payload: NodePayload::BitSlice {
                        arg: x,
                        start: 0,
                        width: slice_w,
                    },
                    pos: None,
                });
                let slice_ref = NodeRef { index: slice_index };

                let zero_ref = Self::mk_ubits_literal_node(f, k_usize, 0);

                f.get_node_mut(target_ref).payload =
                    NodePayload::Nary(NaryOp::Concat, vec![slice_ref, zero_ref]);
                Ok(())
            }

            _ => Err(
                "ConstShllConcatZeroFoldTransform: expected concat(...) or shll(...)".to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
