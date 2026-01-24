// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform that swaps comparison direction by swapping
/// operands.
///
/// Unsigned:
/// - `ugt(x, y) ↔ ult(y, x)`
/// - `uge(x, y) ↔ ule(y, x)`
///
/// Signed:
/// - `sgt(x, y) ↔ slt(y, x)`
/// - `sge(x, y) ↔ sle(y, x)`
#[derive(Debug)]
pub struct CmpSwapTransform;

impl CmpSwapTransform {
    fn is_bits_type(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn swapable_cmp_op(op: Binop) -> bool {
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

    fn swapped_op(op: Binop) -> Option<Binop> {
        match op {
            Binop::Ugt => Some(Binop::Ult),
            Binop::Ult => Some(Binop::Ugt),
            Binop::Uge => Some(Binop::Ule),
            Binop::Ule => Some(Binop::Uge),
            Binop::Sgt => Some(Binop::Slt),
            Binop::Slt => Some(Binop::Sgt),
            Binop::Sge => Some(Binop::Sle),
            Binop::Sle => Some(Binop::Sge),
            _ => None,
        }
    }
}

impl PirTransform for CmpSwapTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CmpSwap
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            let NodePayload::Binop(op, lhs, rhs) = f.get_node(nr).payload else {
                continue;
            };
            if !Self::swapable_cmp_op(op) {
                continue;
            }
            if Self::is_bits_type(f, nr) != Some(1) {
                continue;
            }
            let Some(w) = Self::is_bits_type(f, lhs) else {
                continue;
            };
            if Self::is_bits_type(f, rhs) != Some(w) {
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
                    "CmpSwapTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::Binop(op, lhs, rhs) = f.get_node(target_ref).payload else {
            return Err("CmpSwapTransform: expected binop payload".to_string());
        };
        if !Self::swapable_cmp_op(op) {
            return Err("CmpSwapTransform: expected (u/s)(g/t|l/t|g/e|l/e) binop".to_string());
        }
        if Self::is_bits_type(f, target_ref) != Some(1) {
            return Err("CmpSwapTransform: target must be bits[1]".to_string());
        }
        let Some(w) = Self::is_bits_type(f, lhs) else {
            return Err("CmpSwapTransform: lhs must be bits[w]".to_string());
        };
        if Self::is_bits_type(f, rhs) != Some(w) {
            return Err("CmpSwapTransform: operands must have same bits width".to_string());
        }

        let new_op =
            Self::swapped_op(op).ok_or_else(|| "CmpSwapTransform: unsupported op".to_string())?;
        f.get_node_mut(target_ref).payload = NodePayload::Binop(new_op, rhs, lhs);
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::CmpSwapTransform;
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef};
    use xlsynth_pir::ir_parser;

    use crate::transforms::{PirTransform, TransformLocation};

    fn find_cmp_node(f: &xlsynth_pir::ir::Fn, op: Binop) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(o, _, _) if o == op) {
                return nr;
            }
        }
        panic!("expected cmp node {op:?}");
    }

    #[test]
    fn cmp_swap_flips_ugt_to_ult() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  ret ugt.3: bits[1] = ugt(x, y, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let r = find_cmp_node(&f, Binop::Ugt);
        let t = CmpSwapTransform;
        t.apply(&mut f, &TransformLocation::Node(r)).expect("apply");

        assert!(matches!(
            f.get_node(r).payload,
            NodePayload::Binop(Binop::Ult, _, _)
        ));
    }

    #[test]
    fn cmp_swap_roundtrips() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  ret ult.3: bits[1] = ult(x, y, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let r = find_cmp_node(&f, Binop::Ult);
        let t = CmpSwapTransform;
        t.apply(&mut f, &TransformLocation::Node(r))
            .expect("apply 1");
        t.apply(&mut f, &TransformLocation::Node(r))
            .expect("apply 2");

        assert!(matches!(
            f.get_node(r).payload,
            NodePayload::Binop(Binop::Ult, _, _)
        ));
    }
}
