// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;
use xlsynth_pir::ir_match as m;

/// Rewrites 1-bit selects into boolean sum-of-products form and back.
#[derive(Debug)]
pub struct BoolSelToSumOfProductsTransform;

impl BoolSelToSumOfProductsTransform {
    fn and_operands(ctx: &m::MatchCtx<'_>, node: NodeRef) -> Option<[NodeRef; 2]> {
        let ops = ctx.flattened_nary_operands(node, NaryOp::And)?;
        (ops.len() == 2).then_some([ops[0], ops[1]])
    }

    fn forward_parts(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef)> {
        let (selector, case0, case1) = mu::sel2_parts(f, nr)?;
        (mu::is_u1(f, nr) && mu::is_u1(f, selector) && mu::is_u1(f, case0) && mu::is_u1(f, case1))
            .then_some((selector, case0, case1))
    }

    fn reverse_parts(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef)> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        let ctx = m::MatchCtx::new(f);
        let or_ops = ctx.flattened_nary_operands(nr, NaryOp::Or)?;
        if or_ops.len() != 2 {
            return None;
        }
        for (true_term, false_term) in [(or_ops[0], or_ops[1]), (or_ops[1], or_ops[0])] {
            let Some(true_ops) = Self::and_operands(&ctx, true_term) else {
                continue;
            };
            for (selector, case1) in [(true_ops[0], true_ops[1]), (true_ops[1], true_ops[0])] {
                let Some(bindings) = ctx.matches(
                    false_term,
                    m::commutative(NaryOp::And, vec![m::not(m::exact(selector)), m::any("b")]),
                ) else {
                    continue;
                };
                let case0 = bindings.get_node("b")?;
                if mu::is_u1(f, selector) && mu::is_u1(f, case0) && mu::is_u1(f, case1) {
                    return Some((selector, case0, case1));
                }
            }
        }
        None
    }
}

impl PirTransform for BoolSelToSumOfProductsTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BoolSelToSumOfProducts
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::forward_parts(f, nr).is_some() || Self::reverse_parts(f, nr).is_some() {
                out.push(TransformCandidate {
                    location: TransformLocation::Node(nr),
                    always_equivalent: true,
                });
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err("BoolSelToSumOfProductsTransform: expected node location".to_string());
            }
        };
        if let Some((selector, case0, case1)) = Self::forward_parts(f, target) {
            let not_selector = mu::mk_unop(f, Unop::Not, Type::Bits(1), selector);
            let true_term = mu::mk_nary(f, NaryOp::And, Type::Bits(1), vec![selector, case1]);
            let false_term = mu::mk_nary(f, NaryOp::And, Type::Bits(1), vec![not_selector, case0]);
            f.get_node_mut(target).payload =
                NodePayload::Nary(NaryOp::Or, vec![true_term, false_term]);
            return Ok(());
        }
        if let Some((selector, case0, case1)) = Self::reverse_parts(f, target) {
            f.get_node_mut(target).payload = NodePayload::Sel {
                selector,
                cases: vec![case0, case1],
                default: None,
            };
            return Ok(());
        }
        Err("BoolSelToSumOfProductsTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_and_folds_with_commuted_boolean_operators() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[1] id=2, b: bits[1] id=3) -> bits[1] {
  ret out: bits[1] = sel(p, cases=[b, a], id=4)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = BoolSelToSumOfProductsTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));

        let sop_ir = r#"fn t(p: bits[1] id=1, a: bits[1] id=2, b: bits[1] id=3) -> bits[1] {
  not.4: bits[1] = not(p, id=4)
  and.5: bits[1] = and(a, p, id=5)
  and.6: bits[1] = and(b, not.4, id=6)
  ret out: bits[1] = or(and.6, and.5, id=7)
}"#;
        let mut f = ir_parser::Parser::new(sop_ir).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        let NodePayload::Sel {
            selector, cases, ..
        } = &f.get_node(target).payload
        else {
            panic!("expected sel");
        };
        assert!(matches!(
            f.get_node(*selector).payload,
            NodePayload::GetParam(_)
        ));
        assert_eq!(cases.len(), 2);
    }
}
