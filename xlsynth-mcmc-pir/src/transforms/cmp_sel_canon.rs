// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform that canonicalizes 2-case `sel` expressions
/// that select between the two operands of a comparison.
///
/// For unsigned comparisons with `a: bits[w]`, `b: bits[w]`:
/// - `sel(ugt(a,b), cases=[a,b])` computes `min(a,b)` and becomes
///   `sel(ult(a,b), cases=[b,a])`
/// - `sel(ugt(a,b), cases=[b,a])` computes `max(a,b)` and becomes
///   `sel(ult(a,b), cases=[a,b])`
///
/// Similar normalization is applied for `ult/ule/uge` and signed variants.
#[derive(Debug)]
pub struct CmpSelCanonTransform;

impl CmpSelCanonTransform {
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

    fn mk_cmp_node(f: &mut IrFn, op: Binop, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn is_cmp_op(op: Binop) -> bool {
        matches!(
            op,
            Binop::Ugt
                | Binop::Ult
                | Binop::Uge
                | Binop::Ule
                | Binop::Sgt
                | Binop::Slt
                | Binop::Sge
                | Binop::Sle
        )
    }

    fn canonical_lt_op(op: Binop) -> Option<Binop> {
        match op {
            Binop::Ugt | Binop::Ult | Binop::Uge | Binop::Ule => Some(Binop::Ult),
            Binop::Sgt | Binop::Slt | Binop::Sge | Binop::Sle => Some(Binop::Slt),
            _ => None,
        }
    }

    fn sel_is_min(op: Binop, a: NodeRef, b: NodeRef, c0: NodeRef, c1: NodeRef) -> Option<bool> {
        // Returns Some(true) if this sel computes min(a,b), Some(false) if max(a,b).
        if c0 == b && c1 == a {
            match op {
                Binop::Ult | Binop::Ule | Binop::Slt | Binop::Sle => Some(true),
                Binop::Ugt | Binop::Uge | Binop::Sgt | Binop::Sge => Some(false),
                _ => None,
            }
        } else if c0 == a && c1 == b {
            match op {
                Binop::Ult | Binop::Ule | Binop::Slt | Binop::Sle => Some(false),
                Binop::Ugt | Binop::Uge | Binop::Sgt | Binop::Sge => Some(true),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl PirTransform for CmpSelCanonTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CmpSelCanon
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
            let NodePayload::Binop(op, a, b) = f.get_node(*selector).payload else {
                continue;
            };
            if !Self::is_cmp_op(op) {
                continue;
            }
            let Some(w) = Self::bits_width(f, a) else {
                continue;
            };
            if Self::bits_width(f, b) != Some(w) || Self::bits_width(f, nr) != Some(w) {
                continue;
            }
            if Self::bits_width(f, cases[0]) != Some(w) || Self::bits_width(f, cases[1]) != Some(w)
            {
                continue;
            }
            if Self::sel_is_min(op, a, b, cases[0], cases[1]).is_none() {
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
                    "CmpSelCanonTransform: expected TransformLocation::Node, got RewireOperand"
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
            return Err("CmpSelCanonTransform: expected sel payload".to_string());
        };
        if default.is_some() || cases.len() != 2 {
            return Err("CmpSelCanonTransform: expected 2-case sel without default".to_string());
        }
        if Self::bits_width(f, selector) != Some(1) {
            return Err("CmpSelCanonTransform: selector must be bits[1]".to_string());
        }

        let NodePayload::Binop(op, a, b) = f.get_node(selector).payload else {
            return Err("CmpSelCanonTransform: selector must be comparison binop".to_string());
        };
        if !Self::is_cmp_op(op) {
            return Err("CmpSelCanonTransform: selector must be a comparison".to_string());
        }
        let Some(canon_lt_op) = Self::canonical_lt_op(op) else {
            return Err("CmpSelCanonTransform: unsupported comparison op".to_string());
        };
        let is_min = Self::sel_is_min(op, a, b, cases[0], cases[1]).ok_or_else(|| {
            "CmpSelCanonTransform: sel cases did not match cmp operands".to_string()
        })?;

        // Canonicalize to `lt(a,b)` with cases [b,a] for min and [a,b] for max.
        let new_selector = Self::mk_cmp_node(f, canon_lt_op, a, b);
        let (new_c0, new_c1) = if is_min { (b, a) } else { (a, b) };

        f.get_node_mut(target_ref).payload = NodePayload::Sel {
            selector: new_selector,
            cases: vec![new_c0, new_c1],
            default: None,
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::CmpSelCanonTransform;
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef};
    use xlsynth_pir::ir_parser;

    use crate::transforms::{PirTransform, TransformLocation};

    fn find_sel_node(f: &xlsynth_pir::ir::Fn) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                return nr;
            }
        }
        panic!("expected sel node");
    }

    #[test]
    fn cmp_sel_canon_normalizes_ugt_min_to_ult_min() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ugt.3: bits[1] = ugt(x, y, id=3)
  ret sel.4: bits[8] = sel(ugt.3, cases=[x, y], id=4)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let sel_ref = find_sel_node(&f);
        let t = CmpSelCanonTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        let NodePayload::Sel {
            selector, cases, ..
        } = f.get_node(sel_ref).payload.clone()
        else {
            panic!("expected sel");
        };
        assert_eq!(cases.len(), 2);
        assert!(matches!(
            f.get_node(selector).payload,
            NodePayload::Binop(Binop::Ult, _, _)
        ));
    }
}
