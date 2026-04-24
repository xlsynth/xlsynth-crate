// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Canonicalizes equivalent increment-carry predicates.
#[derive(Debug)]
pub struct IncrementCarryPredicateCanonTransform;

impl IncrementCarryPredicateCanonTransform {
    fn forward_bits_arg(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, usize)> {
        let NodePayload::Unop(Unop::AndReduce, bits) = f.get_node(nr).payload else {
            return None;
        };
        let width = mu::bits_width(f, bits)?;
        (width >= 1 && mu::is_u1(f, nr)).then_some((bits, width))
    }

    fn reverse_bits_arg(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, usize)> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        let NodePayload::Binop(Binop::Eq, lhs, rhs) = f.get_node(nr).payload else {
            return None;
        };
        for (add_ref, zero_ref) in [(lhs, rhs), (rhs, lhs)] {
            let width = mu::bits_width(f, add_ref)?;
            if width < 1 || !mu::is_zero_literal(f, zero_ref, width) {
                continue;
            }
            let NodePayload::Binop(Binop::Add, add_lhs, add_rhs) = f.get_node(add_ref).payload
            else {
                continue;
            };
            for (bits, one_ref) in [(add_lhs, add_rhs), (add_rhs, add_lhs)] {
                if mu::bits_width(f, bits) != Some(width) || !mu::is_literal_one(f, one_ref, width)
                {
                    continue;
                }
                return Some((bits, width));
            }
        }
        None
    }
}

impl PirTransform for IncrementCarryPredicateCanonTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::IncrementCarryPredicateCanon
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::forward_bits_arg(f, nr).is_some() || Self::reverse_bits_arg(f, nr).is_some() {
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
                return Err(
                    "IncrementCarryPredicateCanonTransform: expected node location".to_string(),
                );
            }
        };
        if let Some((bits, width)) = Self::forward_bits_arg(f, target) {
            let one = mu::mk_literal_ubits(f, width, 1);
            let zero = mu::mk_literal_ubits(f, width, 0);
            let add = mu::mk_binop(f, Binop::Add, Type::Bits(width), bits, one);
            f.get_node_mut(target).payload = NodePayload::Binop(Binop::Eq, add, zero);
            return Ok(());
        }
        if let Some((bits, _width)) = Self::reverse_bits_arg(f, target) {
            f.get_node_mut(target).payload = NodePayload::Unop(Unop::AndReduce, bits);
            return Ok(());
        }
        Err("IncrementCarryPredicateCanonTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn round_trips_across_small_widths() {
        let t = IncrementCarryPredicateCanonTransform;
        for width in 1..=8 {
            let ir_text = format!(
                "fn t(x: bits[{width}] id=1) -> bits[1] {{\n  ret out: bits[1] = and_reduce(x, id=2)\n}}"
            );
            let mut f = ir_parser::Parser::new(&ir_text).parse_fn().unwrap();
            let target = f.ret_node_ref.unwrap();
            t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
            assert!(matches!(
                f.get_node(target).payload,
                NodePayload::Binop(Binop::Eq, _, _)
            ));
            t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
            assert!(matches!(
                f.get_node(target).payload,
                NodePayload::Unop(Unop::AndReduce, _)
            ));
        }
    }
}
