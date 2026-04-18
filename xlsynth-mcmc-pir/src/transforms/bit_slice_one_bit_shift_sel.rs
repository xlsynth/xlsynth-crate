// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `sel(p, [bit_slice(x,s,w), bit_slice(x,s+1,w)]) ↔ bit_slice(shrl(x,p), s,
/// w)`
#[derive(Debug)]
pub struct BitSliceOneBitShiftSelTransform;

impl BitSliceOneBitShiftSelTransform {
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

    fn is_u1(f: &IrFn, r: NodeRef) -> bool {
        Self::bits_width(f, r) == Some(1)
    }

    fn bit_slice_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize, usize)> {
        let NodePayload::BitSlice { arg, start, width } = f.get_node(r).payload else {
            return None;
        };
        Some((arg, start, width))
    }

    fn slices_are_in_bounds(x_width: usize, start: usize, width: usize) -> bool {
        start
            .checked_add(width)
            .is_some_and(|limit| limit <= x_width)
            && start
                .checked_add(1)
                .and_then(|next| next.checked_add(width))
                .is_some_and(|limit| limit <= x_width)
    }

    fn mk_bit_slice_node(f: &mut IrFn, arg: NodeRef, start: usize, width: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(width),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_shrl_node(f: &mut IrFn, x: NodeRef, p: NodeRef, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Binop(Binop::Shrl, x, p),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for BitSliceOneBitShiftSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceOneBitShiftSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::BitSlice { arg, start, width } => {
                    let NodePayload::Binop(Binop::Shrl, x, p) = f.get_node(*arg).payload else {
                        continue;
                    };
                    if !Self::is_u1(f, p) {
                        continue;
                    }
                    let Some(x_width) = Self::bits_width(f, x) else {
                        continue;
                    };
                    if Self::bits_width(f, *arg) != Some(x_width)
                        || Self::bits_width(f, nr) != Some(*width)
                    {
                        continue;
                    }
                    if Self::slices_are_in_bounds(x_width, *start, *width) {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
                    }
                }
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() || !Self::is_u1(f, *selector) {
                        continue;
                    }
                    let Some((x0, s0, w0)) = Self::bit_slice_parts(f, cases[0]) else {
                        continue;
                    };
                    let Some((x1, s1, w1)) = Self::bit_slice_parts(f, cases[1]) else {
                        continue;
                    };
                    if x0 != x1 || w0 != w1 || s1 != s0.saturating_add(1) {
                        continue;
                    }
                    let Some(x_width) = Self::bits_width(f, x0) else {
                        continue;
                    };
                    if Self::bits_width(f, nr) != Some(w0)
                        || Self::bits_width(f, cases[0]) != Some(w0)
                        || Self::bits_width(f, cases[1]) != Some(w0)
                    {
                        continue;
                    }
                    if Self::slices_are_in_bounds(x_width, s0, w0) {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
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
                    "BitSliceOneBitShiftSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            NodePayload::BitSlice { arg, start, width } => {
                let NodePayload::Binop(Binop::Shrl, x, p) = f.get_node(arg).payload else {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: expected bit_slice(shrl(x,p),...)"
                            .to_string(),
                    );
                };
                if !Self::is_u1(f, p) {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: shift amount must be bits[1]".to_string(),
                    );
                }
                let x_width = Self::bits_width(f, x).ok_or_else(|| {
                    "BitSliceOneBitShiftSelTransform: x must be bits[xw]".to_string()
                })?;
                if Self::bits_width(f, arg) != Some(x_width)
                    || Self::bits_width(f, target_ref) != Some(width)
                {
                    return Err("BitSliceOneBitShiftSelTransform: type width mismatch".to_string());
                }
                if !Self::slices_are_in_bounds(x_width, start, width) {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: adjacent slices out of bounds"
                            .to_string(),
                    );
                }
                let case0 = Self::mk_bit_slice_node(f, x, start, width);
                let case1 = Self::mk_bit_slice_node(f, x, start + 1, width);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![case0, case1],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1(f, selector) {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: selector must be bits[1]".to_string()
                    );
                }
                let (x0, s0, w0) = Self::bit_slice_parts(f, cases[0]).ok_or_else(|| {
                    "BitSliceOneBitShiftSelTransform: expected case0 bit_slice".to_string()
                })?;
                let (x1, s1, w1) = Self::bit_slice_parts(f, cases[1]).ok_or_else(|| {
                    "BitSliceOneBitShiftSelTransform: expected case1 bit_slice".to_string()
                })?;
                if x0 != x1 || w0 != w1 || s1 != s0.saturating_add(1) {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: expected adjacent bit_slices of same arg"
                            .to_string(),
                    );
                }
                let x_width = Self::bits_width(f, x0).ok_or_else(|| {
                    "BitSliceOneBitShiftSelTransform: bit_slice arg must be bits[xw]".to_string()
                })?;
                if Self::bits_width(f, target_ref) != Some(w0) {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: sel output must be bits[width]"
                            .to_string(),
                    );
                }
                if !Self::slices_are_in_bounds(x_width, s0, w0) {
                    return Err(
                        "BitSliceOneBitShiftSelTransform: adjacent slices out of bounds"
                            .to_string(),
                    );
                }
                let shrl = Self::mk_shrl_node(f, x0, selector, x_width);
                f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                    arg: shrl,
                    start: s0,
                    width: w0,
                };
                Ok(())
            }
            _ => Err(
                "BitSliceOneBitShiftSelTransform: expected bit_slice(...) or sel(...) payload"
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    fn find_payload(f: &IrFn, pred: impl Fn(&NodePayload) -> bool) -> NodeRef {
        f.node_refs()
            .into_iter()
            .find(|nr| pred(&f.get_node(*nr).payload))
            .expect("expected node")
    }

    #[test]
    fn bit_slice_one_bit_shift_sel_expands_slice_of_shrl() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[3] {
  shrl.10: bits[8] = shrl(x, p, id=10)
  ret bit_slice.20: bits[3] = bit_slice(shrl.10, start=2, width=3, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let bs_ref = find_payload(&f, |p| matches!(p, NodePayload::BitSlice { .. }));

        let mut t = BitSliceOneBitShiftSelTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
        t.apply(&mut f, &TransformLocation::Node(bs_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(bs_ref).payload,
            NodePayload::Sel { .. }
        ));
    }

    #[test]
    fn bit_slice_one_bit_shift_sel_folds_adjacent_slice_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[3] {
  bit_slice.10: bits[3] = bit_slice(x, start=2, width=3, id=10)
  bit_slice.11: bits[3] = bit_slice(x, start=3, width=3, id=11)
  ret sel.20: bits[3] = sel(p, cases=[bit_slice.10, bit_slice.11], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let sel_ref = find_payload(&f, |p| matches!(p, NodePayload::Sel { .. }));

        let t = BitSliceOneBitShiftSelTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        let NodePayload::BitSlice { arg, start, width } = f.get_node(sel_ref).payload else {
            panic!("expected bit_slice");
        };
        assert_eq!(start, 2);
        assert_eq!(width, 3);
        assert!(matches!(
            f.get_node(arg).payload,
            NodePayload::Binop(Binop::Shrl, _, _)
        ));
    }

    #[test]
    fn bit_slice_one_bit_shift_sel_rejects_out_of_bounds_adjacent_slice() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[3] {
  bit_slice.10: bits[3] = bit_slice(x, start=5, width=3, id=10)
  bit_slice.11: bits[3] = bit_slice(x, start=6, width=3, id=11)
  ret sel.20: bits[3] = sel(p, cases=[bit_slice.10, bit_slice.11], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = BitSliceOneBitShiftSelTransform;
        assert!(t.find_candidates(&f).is_empty());
    }

    #[test]
    fn bit_slice_one_bit_shift_sel_rejects_non_u1_shift_amount() {
        let ir_text = r#"fn t(s: bits[2] id=1, x: bits[8] id=2) -> bits[3] {
  shrl.10: bits[8] = shrl(x, s, id=10)
  ret bit_slice.20: bits[3] = bit_slice(shrl.10, start=2, width=3, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = BitSliceOneBitShiftSelTransform;
        assert!(t.find_candidates(&f).is_empty());
    }
}
