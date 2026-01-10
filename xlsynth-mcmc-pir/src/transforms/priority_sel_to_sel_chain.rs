// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform lowering a `priority_sel(bits[M])` into a
/// sel-chain.
///
/// Reverse direction is intentionally not implemented.
#[derive(Debug)]
pub struct PrioritySelToSelChainTransform;

impl PrioritySelToSelChainTransform {
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

    fn mk_bit_i(f: &mut IrFn, selector: NodeRef, i: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::BitSlice {
                arg: selector,
                start: i,
                width: 1,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2(f: &mut IrFn, ty: Type, p: NodeRef, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Sel {
                selector: p,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for PrioritySelToSelChainTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::PrioritySelToSelChain
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if let NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } = &f.get_node(nr).payload
            {
                if default.is_none() {
                    continue;
                }
                let Some(m) = Self::bits_width(f, *selector) else {
                    continue;
                };
                if cases.len() == m && m > 0 {
                    out.push(TransformLocation::Node(nr));
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "PrioritySelToSelChainTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err(
                "PrioritySelToSelChainTransform: expected priority_sel payload".to_string(),
            );
        };
        let Some(d) = default else {
            return Err("PrioritySelToSelChainTransform: expected default".to_string());
        };
        let Some(m) = Self::bits_width(f, selector) else {
            return Err("PrioritySelToSelChainTransform: selector must be bits[M]".to_string());
        };
        if cases.len() != m || m == 0 {
            return Err("PrioritySelToSelChainTransform: cases.len() must equal M".to_string());
        }

        // Output type must be bits[w] for this lowering (keeps it simple).
        let out_ty = f.get_node(target_ref).ty.clone();
        if !matches!(out_ty, Type::Bits(_)) {
            return Err("PrioritySelToSelChainTransform: only supports bits outputs".to_string());
        }

        let mut acc = d;
        for i in (0..m).rev() {
            let bit_i = Self::mk_bit_i(f, selector, i);
            // sel(bit_i, cases=[acc, cases[i]])
            acc = Self::mk_sel2(f, out_ty.clone(), bit_i, acc, cases[i]);
        }
        f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, acc);
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
