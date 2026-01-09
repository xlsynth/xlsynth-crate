// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `bit_slice(bit_slice(x, s1, w1), s2, w2) ↔ bit_slice(x, s1+s2, w2)`
#[derive(Debug)]
pub struct BitSliceBitSliceFoldTransform;

impl BitSliceBitSliceFoldTransform {
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
}

impl PirTransform for BitSliceBitSliceFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceBitSliceFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::BitSlice { arg, .. } => {
                    if matches!(f.get_node(*arg).payload, NodePayload::BitSlice { .. }) {
                        out.push(TransformLocation::Node(nr));
                    } else {
                        // allow expansion too
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
                    "BitSliceBitSliceFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::BitSlice {
            arg,
            start: s2,
            width: w2,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err("BitSliceBitSliceFoldTransform: expected bit_slice payload".to_string());
        };
        if Self::bits_width(f, target_ref) != Some(w2) {
            return Err(
                "BitSliceBitSliceFoldTransform: output type must be bits[width]".to_string(),
            );
        }

        // Fold if arg is also a bit_slice.
        if let NodePayload::BitSlice {
            arg: x,
            start: s1,
            width: w1,
        } = f.get_node(arg).payload.clone()
        {
            let in_w = Self::bits_width(f, x).ok_or_else(|| {
                "BitSliceBitSliceFoldTransform: input must be bits[w]".to_string()
            })?;
            if s1.saturating_add(w1) > in_w {
                return Err("BitSliceBitSliceFoldTransform: inner slice out of bounds".to_string());
            }
            if s2.saturating_add(w2) > w1 {
                return Err("BitSliceBitSliceFoldTransform: outer slice out of bounds".to_string());
            }
            let s = s1.saturating_add(s2);
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: x,
                start: s,
                width: w2,
            };
            return Ok(());
        }

        // Expand: bit_slice(x,s,w) -> bit_slice(bit_slice(x,0,w1),s,w) with w1=s+w.
        let x = arg;
        let in_w = Self::bits_width(f, x)
            .ok_or_else(|| "BitSliceBitSliceFoldTransform: input must be bits[w]".to_string())?;
        let w1 = s2.saturating_add(w2);
        if w1 > in_w {
            return Err("BitSliceBitSliceFoldTransform: slice out of bounds".to_string());
        }
        let inner = Self::mk_bit_slice_node(f, w1, x, 0, w1);
        f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
            arg: inner,
            start: s2,
            width: w2,
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
