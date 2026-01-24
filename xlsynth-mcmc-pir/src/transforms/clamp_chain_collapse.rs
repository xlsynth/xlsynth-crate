// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Binop, Fn as IrFn, NaryOp, Node, NodePayload, NodeRef, Type, Unop};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A non-always-equivalent transform that attempts to collapse a deep
/// clamp-like selector tree into a single comparison-based clamp.
///
/// This targets patterns where a `sel` chooses between a value and a constant
/// bound, and the selector cone already contains some comparison of that value
/// to that same bound. We then rewrite to the canonical form:
///
/// `sel(ult(x, bound), cases=[bound, x])`
///
/// This is intentionally prover-filtered (not always equivalent).
#[derive(Debug)]
pub struct ClampChainCollapseTransform;

impl ClampChainCollapseTransform {
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

    fn is_literal(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).payload, NodePayload::Literal(_))
    }

    fn literal_u64_value(f: &IrFn, r: NodeRef) -> Option<u64> {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return None;
        };
        v.to_u64().ok()
    }

    fn mk_binop_node(f: &mut IrFn, op: Binop, ty: Type, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn unwrap_identity(f: &IrFn, r: NodeRef) -> NodeRef {
        match f.get_node(r).payload {
            NodePayload::Unop(Unop::Identity, arg) => arg,
            _ => r,
        }
    }

    fn selector_cone_contains_cmp(
        f: &IrFn,
        selector: NodeRef,
        x: NodeRef,
        bound: NodeRef,
        max_nodes: usize,
    ) -> bool {
        let mut todo: Vec<NodeRef> = vec![selector];
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        while let Some(r) = todo.pop() {
            if !seen.insert(r.index) {
                continue;
            }
            if seen.len() > max_nodes {
                return false;
            }
            let r = Self::unwrap_identity(f, r);
            match &f.get_node(r).payload {
                NodePayload::Binop(op, a, b)
                    if matches!(op, Binop::Ugt | Binop::Uge | Binop::Ult | Binop::Ule) =>
                {
                    if (*a == x && *b == bound) || (*a == bound && *b == x) {
                        return true;
                    }
                    todo.push(*a);
                    todo.push(*b);
                }
                NodePayload::Unop(_op, a) => {
                    todo.push(*a);
                }
                NodePayload::Nary(NaryOp::And, args) | NodePayload::Nary(NaryOp::Or, args) => {
                    for a in args {
                        todo.push(*a);
                    }
                }
                _ => {}
            }
        }
        false
    }
}

impl PirTransform for ClampChainCollapseTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ClampChainCollapse
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            let NodePayload::Sel {
                selector,
                cases,
                default,
            } = &f.get_node(nr).payload
            else {
                continue;
            };
            if default.is_some() || cases.len() != 2 {
                continue;
            }
            if Self::bits_width(f, *selector) != Some(1) {
                continue;
            }
            let Some(w) = Self::bits_width(f, nr) else {
                continue;
            };
            if Self::bits_width(f, cases[0]) != Some(w) || Self::bits_width(f, cases[1]) != Some(w)
            {
                continue;
            }
            // One case must be a literal; the other is the value we attempt to clamp.
            let (bound, x) = if Self::is_literal(f, cases[0]) && !Self::is_literal(f, cases[1]) {
                (cases[0], cases[1])
            } else if Self::is_literal(f, cases[1]) && !Self::is_literal(f, cases[0]) {
                (cases[1], cases[0])
            } else {
                continue;
            };
            // Prefer small bounds (fits in u64).
            if Self::literal_u64_value(f, bound).is_none() {
                continue;
            }
            // Only fire when the selector cone already references a compare between x and
            // bound.
            if !Self::selector_cone_contains_cmp(f, *selector, x, bound, 64) {
                continue;
            }
            out.push(TransformLocation::Node(nr));
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "ClampChainCollapseTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err("ClampChainCollapseTransform: expected sel payload".to_string());
        };
        if default.is_some() || cases.len() != 2 {
            return Err(
                "ClampChainCollapseTransform: expected 2-case sel without default".to_string(),
            );
        }
        if Self::bits_width(f, selector) != Some(1) {
            return Err("ClampChainCollapseTransform: selector must be bits[1]".to_string());
        }
        let Some(w) = Self::bits_width(f, target_ref) else {
            return Err("ClampChainCollapseTransform: output must be bits[w]".to_string());
        };

        let (bound, x) = if Self::is_literal(f, cases[0]) && !Self::is_literal(f, cases[1]) {
            (cases[0], cases[1])
        } else if Self::is_literal(f, cases[1]) && !Self::is_literal(f, cases[0]) {
            (cases[1], cases[0])
        } else {
            return Err(
                "ClampChainCollapseTransform: expected exactly one literal case".to_string(),
            );
        };
        if Self::bits_width(f, bound) != Some(w) || Self::bits_width(f, x) != Some(w) {
            return Err("ClampChainCollapseTransform: case widths must match output".to_string());
        }

        if !Self::selector_cone_contains_cmp(f, selector, x, bound, 64) {
            return Err("ClampChainCollapseTransform: selector cone did not reference x vs bound comparison".to_string());
        }

        // Rewrite to canonical clamp: sel(ult(x, bound), cases=[bound, x])
        let pred = Self::mk_binop_node(f, Binop::Ult, Type::Bits(1), x, bound);
        f.get_node_mut(target_ref).payload = NodePayload::Sel {
            selector: pred,
            cases: vec![bound, x],
            default: None,
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}
