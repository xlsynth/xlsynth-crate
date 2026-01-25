// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform that recognizes the sel-chain produced by
/// `PrioritySelToSelChainTransform` and rebuilds a compact `priority_sel`.
///
/// This is effectively the reverse direction of
/// `PrioritySelToSelChainTransform`.
#[derive(Debug)]
pub struct SelChainToPrioritySelTransform;

impl SelChainToPrioritySelTransform {
    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn is_bits1(f: &IrFn, r: NodeRef) -> bool {
        Self::bits_width(f, r) == Some(1)
    }

    fn bit_slice_1_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::BitSlice { arg, start, width } => {
                if *width == 1 {
                    Some((*arg, *start))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Attempts to match the exact sel-chain encoding from
    /// `PrioritySelToSelChainTransform`.
    ///
    /// Returns `(selector_bits, cases, default)` where `cases.len() ==
    /// bits_width(selector_bits)`.
    fn match_sel_chain(f: &IrFn, start_sel: NodeRef) -> Option<(NodeRef, Vec<NodeRef>, NodeRef)> {
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = &f.get_node(start_sel).payload
        else {
            return None;
        };
        if default.is_some() || cases.len() != 2 {
            return None;
        }
        if !Self::is_bits1(f, *selector) {
            return None;
        }
        let (selector_bits, start_idx) = Self::bit_slice_1_parts(f, *selector)?;
        let m = Self::bits_width(f, selector_bits)?;
        if start_idx != 0 || m == 0 {
            return None;
        }

        let out_ty = f.get_node(start_sel).ty.clone();
        if !matches!(out_ty, Type::Bits(_)) {
            return None;
        }

        let mut collected_cases: Vec<NodeRef> = Vec::with_capacity(m);
        // Outer sel corresponds to bit 0 (see PrioritySelToSelChainTransform).
        let mut expected_bit: usize = 0;
        let mut cur = start_sel;
        loop {
            let NodePayload::Sel {
                selector,
                cases,
                default,
            } = &f.get_node(cur).payload
            else {
                return None;
            };
            if default.is_some() || cases.len() != 2 {
                return None;
            }
            if f.get_node(cur).ty != out_ty {
                return None;
            }
            if !Self::is_bits1(f, *selector) {
                return None;
            }
            let (cur_selector_bits, bit_idx) = Self::bit_slice_1_parts(f, *selector)?;
            if cur_selector_bits != selector_bits || bit_idx != expected_bit {
                return None;
            }

            // sel(bit_i, cases=[acc, case_i])
            collected_cases.push(cases[1]);
            if f.get_node(cases[1]).ty != out_ty {
                return None;
            }

            let next = cases[0];
            expected_bit = expected_bit.saturating_add(1);
            if expected_bit == m {
                // The next value should be the default `d` (not another sel).
                if f.get_node(next).ty != out_ty {
                    return None;
                }
                return Some((selector_bits, collected_cases, next));
            }
            // More bits expected; next must be another sel.
            if !matches!(f.get_node(next).payload, NodePayload::Sel { .. }) {
                return None;
            }
            cur = next;
        }
    }
}

impl PirTransform for SelChainToPrioritySelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelChainToPrioritySel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            // PrioritySelToSelChainTransform wraps the chain in an identity.
            let start_sel = match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Identity, arg)
                    if matches!(f.get_node(*arg).payload, NodePayload::Sel { .. }) =>
                {
                    *arg
                }
                NodePayload::Sel { .. } => nr,
                _ => continue,
            };
            if Self::match_sel_chain(f, start_sel).is_some() {
                out.push(TransformLocation::Node(nr));
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SelChainToPrioritySelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let start_sel = match f.get_node(target_ref).payload {
            NodePayload::Unop(Unop::Identity, arg) => arg,
            NodePayload::Sel { .. } => target_ref,
            _ => {
                return Err(
                    "SelChainToPrioritySelTransform: expected sel or identity(sel) payload"
                        .to_string(),
                );
            }
        };

        let Some((selector_bits, cases, default)) = Self::match_sel_chain(f, start_sel) else {
            return Err("SelChainToPrioritySelTransform: did not match sel-chain".to_string());
        };

        let m = Self::bits_width(f, selector_bits).ok_or_else(|| {
            "SelChainToPrioritySelTransform: selector must be bits[M]".to_string()
        })?;
        if cases.len() != m {
            return Err(
                "SelChainToPrioritySelTransform: internal cases length mismatch".to_string(),
            );
        }

        f.get_node_mut(target_ref).payload = NodePayload::PrioritySel {
            selector: selector_bits,
            cases,
            default: Some(default),
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::SelChainToPrioritySelTransform;
    use xlsynth_pir::ir::{NodePayload, NodeRef};
    use xlsynth_pir::ir_parser;

    use crate::transforms::{PirTransform, PrioritySelToSelChainTransform, TransformLocation};

    fn find_priority_sel_node(f: &xlsynth_pir::ir::Fn) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::PrioritySel { .. }) {
                return nr;
            }
        }
        panic!("expected priority_sel node");
    }

    #[test]
    fn sel_chain_to_priority_sel_roundtrips_priority_sel_to_sel_chain() {
        let ir_text = r#"fn t(p: bits[2] id=1, c0: bits[8] id=2, c1: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  ret priority_sel.10: bits[8] = priority_sel(p, cases=[c0, c1], default=d, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let ps_ref = find_priority_sel_node(&f);
        let to_chain = PrioritySelToSelChainTransform;
        to_chain
            .apply(&mut f, &TransformLocation::Node(ps_ref))
            .expect("to_chain apply");

        let to_ps = SelChainToPrioritySelTransform;
        to_ps
            .apply(&mut f, &TransformLocation::Node(ps_ref))
            .expect("to_ps apply");

        assert!(matches!(
            f.get_node(ps_ref).payload,
            NodePayload::PrioritySel { .. }
        ));
    }
}
