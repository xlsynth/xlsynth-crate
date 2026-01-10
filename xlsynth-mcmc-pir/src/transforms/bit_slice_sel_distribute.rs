// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `bit_slice(sel(p, cases=[a, b]), start=s, width=w)
///    ↔ sel(p, cases=[bit_slice(a,s,w), bit_slice(b,s,w)])`
#[derive(Debug)]
pub struct BitSliceSelDistributeTransform;

impl BitSliceSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn bit_slice_parts(payload: &NodePayload) -> Option<(NodeRef, usize, usize)> {
        match payload {
            NodePayload::BitSlice { arg, start, width } => Some((*arg, *start, *width)),
            _ => None,
        }
    }

    fn mk_bit_slice_node(
        f: &mut IrFn,
        out_w: usize,
        arg: NodeRef,
        start: usize,
        width: usize,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(
        f: &mut IrFn,
        out_w: usize,
        selector: NodeRef,
        a: NodeRef,
        b: NodeRef,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for BitSliceSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: bit_slice(sel(...), s, w)
                NodePayload::BitSlice { arg, start, width } => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wout == Some(*width) {
                            // Also require slice to be in-bounds to avoid constructing invalid IR.
                            if start.saturating_add(*width) <= wa.unwrap() {
                                out.push(TransformLocation::Node(nr));
                            }
                        }
                    }
                }
                // Fold: sel(p, [bit_slice(a,s,w), bit_slice(b,s,w)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let (a_arg, a_start, a_width) =
                        match Self::bit_slice_parts(&f.get_node(cases[0]).payload) {
                            Some(v) => v,
                            None => continue,
                        };
                    let (b_arg, b_start, b_width) =
                        match Self::bit_slice_parts(&f.get_node(cases[1]).payload) {
                            Some(v) => v,
                            None => continue,
                        };
                    if a_start != b_start || a_width != b_width {
                        continue;
                    }
                    let wa = Self::bits_width(f, a_arg);
                    let wb = Self::bits_width(f, b_arg);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wout == Some(a_width) {
                        if a_start.saturating_add(a_width) <= wa.unwrap() {
                            out.push(TransformLocation::Node(nr));
                        }
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
                    "BitSliceSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: bit_slice(sel(...), s, w) -> sel(...bit_slice...)
            NodePayload::BitSlice { arg, start, width } => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg).payload).ok_or_else(|| {
                        "BitSliceSelDistributeTransform: expected bit_slice(sel(p, cases=[a,b]), ...)"
                            .to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err(
                        "BitSliceSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .ok_or_else(|| {
                        "BitSliceSelDistributeTransform: sel cases must be bits[w] with same width"
                            .to_string()
                    })?;
                if start.saturating_add(width) > in_w {
                    return Err(
                        "BitSliceSelDistributeTransform: bit_slice start+width out of bounds"
                            .to_string(),
                    );
                }
                let out_w = width;
                if Self::bits_width(f, target_ref) != Some(out_w) {
                    return Err(
                        "BitSliceSelDistributeTransform: output type must be bits[width]".to_string(),
                    );
                }

                let bs_a = Self::mk_bit_slice_node(f, out_w, a, start, width);
                let bs_b = Self::mk_bit_slice_node(f, out_w, b, start, width);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![bs_a, bs_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p, [bit_slice(a,s,w), bit_slice(b,s,w)]) -> bit_slice(sel(p,[a,b]), s, w)
            NodePayload::Sel { selector, cases, default } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "BitSliceSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "BitSliceSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let (a_arg, a_start, a_width) =
                    Self::bit_slice_parts(&f.get_node(cases[0]).payload).ok_or_else(|| {
                        "BitSliceSelDistributeTransform: expected sel case 0 to be bit_slice(a,...)"
                            .to_string()
                    })?;
                let (b_arg, b_start, b_width) =
                    Self::bit_slice_parts(&f.get_node(cases[1]).payload).ok_or_else(|| {
                        "BitSliceSelDistributeTransform: expected sel case 1 to be bit_slice(b,...)"
                            .to_string()
                    })?;
                if a_start != b_start || a_width != b_width {
                    return Err(
                        "BitSliceSelDistributeTransform: bit_slice params must match across cases"
                            .to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a_arg)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b_arg))
                    .ok_or_else(|| {
                        "BitSliceSelDistributeTransform: bit_slice args must be bits[w] with same width"
                            .to_string()
                    })?;
                if a_start.saturating_add(a_width) > in_w {
                    return Err(
                        "BitSliceSelDistributeTransform: bit_slice start+width out of bounds"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, target_ref) != Some(a_width) {
                    return Err(
                        "BitSliceSelDistributeTransform: output type must be bits[width]".to_string(),
                    );
                }

                let sel_ab = Self::mk_sel2_node(f, in_w, selector, a_arg, b_arg);
                f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                    arg: sel_ab,
                    start: a_start,
                    width: a_width,
                };
                Ok(())
            }
            _ => Err(
                "BitSliceSelDistributeTransform: expected bit_slice(sel(...)) or sel(bit_slice(...),bit_slice(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
