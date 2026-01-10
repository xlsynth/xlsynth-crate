// SPDX-License-Identifier: Apache-2.0

#[allow(unused_imports)]
use std::collections::{HashSet, VecDeque};
use std::fmt;
#[allow(unused_imports)]
use std::mem;

#[allow(unused_imports)]
use xlsynth::{IrBits, IrValue};
#[allow(unused_imports)]
use xlsynth_pir::ir::{Binop, Fn as IrFn, NaryOp, Node, NodePayload, NodeRef, Type, Unop};
#[allow(unused_imports)]
use xlsynth_pir::ir_utils::{compute_users, remap_payload_with};

mod and_mask_sign_ext_to_sel;
mod and_reduce_demorgan;
mod bit_slice_bit_slice_fold;
mod bit_slice_concat_distribute;
mod bit_slice_sel_distribute;
mod carry_split_add;
mod clone_multi_user_node;
mod const_shll_concat_zero_fold;
mod eq_ne_add_literal_shift;
mod eq_sel_distribute;
mod eq_zero_or_reduce;
mod nand_not_and_fold;
mod ne_zero_or_reduce;
mod neg_neg_cancel;
mod neg_sel_distribute;
mod neg_sub_swap;
mod nor_not_or_fold;
mod not_eq_ne_flip;
mod not_not_cancel;
mod not_sel_distribute;
mod priority_sel_1_to_sel;
mod priority_sel_to_sel_chain;
mod reassociate_add_sub;
mod rewire_operand_to_same_type;
mod sel_same_arms_fold;
mod sel_swap_arms_by_not_pred;
mod sign_ext_sel_distribute;
mod sub_to_add_neg;
mod swap_commutative_binop_operands;
mod xor_mask_sign_ext_to_sel_not;

use and_mask_sign_ext_to_sel::AndMaskSignExtToSelTransform;
use and_reduce_demorgan::AndReduceDeMorganTransform;
use bit_slice_bit_slice_fold::BitSliceBitSliceFoldTransform;
use bit_slice_concat_distribute::BitSliceConcatDistributeTransform;
use bit_slice_sel_distribute::BitSliceSelDistributeTransform;
use carry_split_add::CarrySplitAddTransform;
use clone_multi_user_node::CloneMultiUserNodeTransform;
use const_shll_concat_zero_fold::ConstShllConcatZeroFoldTransform;
use eq_ne_add_literal_shift::EqNeAddLiteralShiftTransform;
use eq_sel_distribute::EqSelDistributeTransform;
use eq_zero_or_reduce::EqZeroOrReduceTransform;
use nand_not_and_fold::NandNotAndFoldTransform;
use ne_zero_or_reduce::NeZeroOrReduceTransform;
use neg_neg_cancel::NegNegCancelTransform;
use neg_sel_distribute::NegSelDistributeTransform;
use neg_sub_swap::NegSubSwapTransform;
use nor_not_or_fold::NorNotOrFoldTransform;
use not_eq_ne_flip::NotEqNeFlipTransform;
use not_not_cancel::NotNotCancelTransform;
use not_sel_distribute::NotSelDistributeTransform;
use priority_sel_1_to_sel::PrioritySel1ToSelTransform;
use priority_sel_to_sel_chain::PrioritySelToSelChainTransform;
use reassociate_add_sub::ReassociateAddSubTransform;
use rewire_operand_to_same_type::RewireOperandToSameTypeTransform;
use sel_same_arms_fold::SelSameArmsFoldTransform;
use sel_swap_arms_by_not_pred::SelSwapArmsByNotPredTransform;
use sign_ext_sel_distribute::SignExtSelDistributeTransform;
use sub_to_add_neg::SubToAddNegTransform;
use swap_commutative_binop_operands::SwapCommutativeBinopOperandsTransform;
use xor_mask_sign_ext_to_sel_not::XorMaskSignExtToSelNotTransform;

/// Kinds of PIR transforms used in PIR MCMC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum PirTransformKind {
    /// Swap the operands of a commutative binary operator (currently `add`).
    SwapCommutativeBinopOperands,
    /// Clone a node that has multiple users so that each user gets its own
    /// replica node.
    CloneMultiUserNode,
    /// Distributes equality/inequality over a select (and the reverse folding
    /// form):
    ///
    /// - `eq(sel(p,a,b),c) ↔ sel(p,eq(a,c),eq(b,c))`
    /// - `ne(sel(p,a,b),c) ↔ sel(p,ne(a,c),ne(b,c))`
    EqSelDistribute,
    /// Constant-shift equality through add-with-literal (mod 2^w):
    ///   `eq(add(x, k), c) ↔ eq(x, c - k)` (and same for `ne`)
    EqNeAddLiteralShift,
    /// Normalize subtraction via add+negation (two's complement) and reverse:
    /// `sub(x, y) ↔ add(x, neg(y))`
    SubToAddNeg,
    /// Normalize negated subtraction into swapped subtraction and reverse:
    /// `neg(sub(x, y)) ↔ sub(y, x)`
    NegSubSwap,
    /// Speculative arithmetic reshaping (always equivalent under modulo 2^w):
    /// reassociate/rotate small `add`/`sub` trees to explore different shapes.
    ReassociateAddSub,
    /// Expand/contract an `add(bits[w])` into a low+high half adder with
    /// explicit carry. This is a structure-changing move intended to affect
    /// depth/product.
    CarrySplitAdd,
    /// Distribute NOT over select (and reverse folding form):
    /// `not(sel(p, cases=[a, b])) ↔ sel(p, cases=[not(a), not(b)])`
    NotSelDistribute,
    /// Distribute NEG over select (and reverse folding form):
    /// `neg(sel(p, cases=[a, b])) ↔ sel(p, cases=[neg(a), neg(b)])`
    NegSelDistribute,
    /// Distribute bit_slice over select (and reverse folding form):
    /// `bit_slice(sel(p, cases=[a, b]), start=s, width=w)
    ///    ↔ sel(p, cases=[bit_slice(a,s,w), bit_slice(b,s,w)])`
    BitSliceSelDistribute,
    /// Distribute sign_ext over select (and reverse folding form):
    /// `sign_ext(sel(p, cases=[a, b]), new_bit_count=n)
    ///    ↔ sel(p, cases=[sign_ext(a,n), sign_ext(b,n)])`
    SignExtSelDistribute,
    /// Convert 1-bit priority_sel to sel (and reverse):
    /// `priority_sel(p:bits[1], cases=[a], default=b) ↔ sel(p, cases=[b, a])`
    PrioritySel1ToSel,
    /// Treat `sign_ext(b)` (b:bits[1]) as an all-ones/zeros mask and convert
    /// to/from sel: `and(x, sign_ext(b,w)) ↔ sel(b, cases=[0_w, x])`
    AndMaskSignExtToSel,
    /// `xor(x, sign_ext(b,w))` is a conditional invert; convert to/from sel:
    /// `xor(x, sign_ext(b,w)) ↔ sel(b, cases=[x, not(x)])`
    XorMaskSignExtToSelNot,
    /// Fold `sel` when both cases are identical:
    /// `sel(p, cases=[a, a]) ↔ a`
    SelSameArmsFold,
    /// Swap `sel` arms by negating predicate (2-case `sel` only):
    /// `sel(not(p), cases=[a, b]) ↔ sel(p, cases=[b, a])`
    SelSwapArmsByNotPred,
    /// Cancel double NOT:
    /// `not(not(x)) ↔ x`
    NotNotCancel,
    /// Cancel double NEG (two's complement):
    /// `neg(neg(x)) ↔ x`
    NegNegCancel,
    /// Flip eq/ne through NOT:
    /// `not(eq(a,b)) ↔ ne(a,b)`
    /// `not(ne(a,b)) ↔ eq(a,b)`
    NotEqNeFlip,
    /// Normalize NOR via NOT(OR) (variadic-friendly) and reverse:
    /// `nor(xs...) ↔ not(or(xs...))`
    NorNotOrFold,
    /// Normalize NAND via NOT(AND) (variadic-friendly) and reverse:
    /// `nand(xs...) ↔ not(and(xs...))`
    NandNotAndFold,
    /// Zero-test via OR-reduce and reverse:
    /// `eq(x, 0_w) ↔ not(or_reduce(x))`
    EqZeroOrReduce,
    /// Nonzero-test via OR-reduce and reverse:
    /// `ne(x, 0_w) ↔ or_reduce(x)`
    NeZeroOrReduce,
    /// Fold nested bit_slices:
    /// `bit_slice(bit_slice(x, s1, w1), s2, w2) ↔ bit_slice(x, s1+s2, w2)`
    BitSliceBitSliceFold,
    /// Normalize `and_reduce` via De Morgan and reverse:
    /// `and_reduce(x) ↔ not(or_reduce(not(x)))`
    AndReduceDeMorgan,
    /// Distribute bit_slice over concat (and reverse folding form):
    /// `bit_slice(concat(a,b), start=s, width=w) ↔ ...`
    /// (handle in-a / in-b / straddle cases)
    BitSliceConcatDistribute,
    /// Normalize a constant left shift encoded as concat+bit_slice and reverse:
    /// `concat(bit_slice(x, start=0, width=w-k), 0_k) ↔ shll(x, k)`
    ConstShllConcatZeroFold,
    /// Lower small priority_sel to a sel-chain (reverse optional):
    /// for selector bits[M], cases len=M:
    /// `priority_sel(sel, cases=[c0..cM-1], default=d)`
    ///   ↔ nested sel(bit_i, [acc, c_i]) with i from M-1 down to 0
    PrioritySelToSelChain,
    /// Rewire one operand of a node to any other node in the function with the
    /// same type.
    ///
    /// This is *not* guaranteed to preserve semantics and therefore requires an
    /// equivalence oracle check before accepting.
    RewireOperandToSameType,
}

impl fmt::Display for PirTransformKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PirTransformKind::SwapCommutativeBinopOperands => {
                write!(f, "SwapCommutativeBinopOperands")
            }
            PirTransformKind::CloneMultiUserNode => write!(f, "CloneMultiUserNode"),
            PirTransformKind::EqSelDistribute => write!(f, "EqSelDistribute"),
            PirTransformKind::EqNeAddLiteralShift => write!(f, "EqNeAddLiteralShift"),
            PirTransformKind::SubToAddNeg => write!(f, "SubToAddNeg"),
            PirTransformKind::NegSubSwap => write!(f, "NegSubSwap"),
            PirTransformKind::ReassociateAddSub => write!(f, "ReassociateAddSub"),
            PirTransformKind::CarrySplitAdd => write!(f, "CarrySplitAdd"),
            PirTransformKind::NotSelDistribute => write!(f, "NotSelDistribute"),
            PirTransformKind::NegSelDistribute => write!(f, "NegSelDistribute"),
            PirTransformKind::BitSliceSelDistribute => write!(f, "BitSliceSelDistribute"),
            PirTransformKind::SignExtSelDistribute => write!(f, "SignExtSelDistribute"),
            PirTransformKind::PrioritySel1ToSel => write!(f, "PrioritySel1ToSel"),
            PirTransformKind::AndMaskSignExtToSel => write!(f, "AndMaskSignExtToSel"),
            PirTransformKind::XorMaskSignExtToSelNot => write!(f, "XorMaskSignExtToSelNot"),
            PirTransformKind::SelSameArmsFold => write!(f, "SelSameArmsFold"),
            PirTransformKind::SelSwapArmsByNotPred => write!(f, "SelSwapArmsByNotPred"),
            PirTransformKind::NotNotCancel => write!(f, "NotNotCancel"),
            PirTransformKind::NegNegCancel => write!(f, "NegNegCancel"),
            PirTransformKind::NotEqNeFlip => write!(f, "NotEqNeFlip"),
            PirTransformKind::NorNotOrFold => write!(f, "NorNotOrFold"),
            PirTransformKind::NandNotAndFold => write!(f, "NandNotAndFold"),
            PirTransformKind::EqZeroOrReduce => write!(f, "EqZeroOrReduce"),
            PirTransformKind::NeZeroOrReduce => write!(f, "NeZeroOrReduce"),
            PirTransformKind::BitSliceBitSliceFold => write!(f, "BitSliceBitSliceFold"),
            PirTransformKind::AndReduceDeMorgan => write!(f, "AndReduceDeMorgan"),
            PirTransformKind::BitSliceConcatDistribute => write!(f, "BitSliceConcatDistribute"),
            PirTransformKind::ConstShllConcatZeroFold => write!(f, "ConstShllConcatZeroFold"),
            PirTransformKind::PrioritySelToSelChain => write!(f, "PrioritySelToSelChain"),
            PirTransformKind::RewireOperandToSameType => write!(f, "RewireOperandToSameType"),
        }
    }
}

/// A specific location in a PIR function where a transform can be applied.
#[derive(Debug, Clone)]
pub enum TransformLocation {
    Node(NodeRef),
    RewireOperand {
        node: NodeRef,
        operand_slot: usize,
        new_operand: NodeRef,
    },
}

/// Defines a reversible transformation that can be applied to a PIR function.
pub trait PirTransform: fmt::Debug + Send + Sync {
    /// Returns the specific `PirTransformKind` that this trait object
    /// represents.
    fn kind(&self) -> PirTransformKind;

    /// Finds all possible application sites for this transform in the given
    /// function.
    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation>;

    /// Applies the transform to the function at the given candidate site.
    ///
    /// The MCMC loop is expected to pass a clone of the function if rejection
    /// implies reverting to the prior state.
    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String>;

    /// Indicates whether this transform is always semantics preserving.
    ///
    /// When `true`, applying the transform cannot change the functional
    /// behaviour of the function, so equivalence checks can be skipped.
    fn always_equivalent(&self) -> bool {
        true
    }
}
pub fn get_all_pir_transforms() -> Vec<Box<dyn PirTransform>> {
    vec![
        Box::new(SwapCommutativeBinopOperandsTransform),
        Box::new(CloneMultiUserNodeTransform),
        Box::new(EqSelDistributeTransform),
        Box::new(EqNeAddLiteralShiftTransform),
        Box::new(SubToAddNegTransform),
        Box::new(NegSubSwapTransform),
        Box::new(ReassociateAddSubTransform),
        Box::new(CarrySplitAddTransform),
        Box::new(NotSelDistributeTransform),
        Box::new(NegSelDistributeTransform),
        Box::new(BitSliceSelDistributeTransform),
        Box::new(SignExtSelDistributeTransform),
        Box::new(PrioritySel1ToSelTransform),
        Box::new(AndMaskSignExtToSelTransform),
        Box::new(XorMaskSignExtToSelNotTransform),
        Box::new(SelSameArmsFoldTransform),
        Box::new(SelSwapArmsByNotPredTransform),
        Box::new(NotNotCancelTransform),
        Box::new(NegNegCancelTransform),
        Box::new(NotEqNeFlipTransform),
        Box::new(NorNotOrFoldTransform),
        Box::new(NandNotAndFoldTransform),
        Box::new(EqZeroOrReduceTransform),
        Box::new(NeZeroOrReduceTransform),
        Box::new(AndReduceDeMorganTransform),
        Box::new(BitSliceBitSliceFoldTransform),
        Box::new(BitSliceConcatDistributeTransform),
        Box::new(ConstShllConcatZeroFoldTransform),
        Box::new(PrioritySelToSelChainTransform),
        Box::new(RewireOperandToSameTypeTransform),
    ]
}

/// Returns a vector of weights for the given transforms.
///
/// For now all PIR transforms are given equal weight.
pub fn build_transform_weights<T: AsRef<[Box<dyn PirTransform>]>>(transforms: T) -> Vec<f64> {
    transforms.as_ref().iter().map(|_| 1.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn eq_sel_distribute_expands_eq_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret eq.20: bits[1] = eq(sel.10, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        match &f.get_node(eq_ref).payload {
            NodePayload::Sel {
                selector: _,
                cases,
                default,
            } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Binop(Binop::Eq, _, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_sel_distribute_folds_sel_of_eqs() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  eq.11: bits[1] = eq(a, c, id=11)
  eq.12: bits[1] = eq(b, c, id=12)
  ret sel.20: bits[1] = sel(p, cases=[eq.11, eq.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Binop(Binop::Eq, _, _)
        ));
    }

    #[test]
    fn and_reduce_demorgan_folds_not_or_reduce_not() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[1] {
  not.10: bits[8] = not(x, id=10)
  or_reduce.11: bits[1] = or_reduce(not.10, id=11)
  ret not.12: bits[1] = not(or_reduce.11, id=12)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ret_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if let NodePayload::Unop(Unop::Not, arg) = &f.get_node(nr).payload {
                if matches!(
                    f.get_node(*arg).payload,
                    NodePayload::Unop(Unop::OrReduce, _)
                ) {
                    ret_ref = Some(nr);
                }
            }
        }
        let ret_ref = ret_ref.expect("expected not(or_reduce(..)) node");

        let t = AndReduceDeMorganTransform;
        t.apply(&mut f, &TransformLocation::Node(ret_ref))
            .expect("apply");

        match &f.get_node(ret_ref).payload {
            NodePayload::Unop(Unop::AndReduce, arg) => {
                assert!(matches!(f.get_node(*arg).payload, NodePayload::GetParam(_)));
            }
            other => panic!("expected and_reduce after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn and_reduce_demorgan_expands_and_reduce() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[1] {
  ret and_reduce.10: bits[1] = and_reduce(x, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ret_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(
                f.get_node(nr).payload,
                NodePayload::Unop(Unop::AndReduce, _)
            ) {
                ret_ref = Some(nr);
            }
        }
        let ret_ref = ret_ref.expect("expected and_reduce node");

        let t = AndReduceDeMorganTransform;
        t.apply(&mut f, &TransformLocation::Node(ret_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Not, or_ref) = &f.get_node(ret_ref).payload else {
            panic!("expected not(...) after rewrite");
        };
        let NodePayload::Unop(Unop::OrReduce, not_ref) = &f.get_node(*or_ref).payload else {
            panic!("expected or_reduce(...) after rewrite");
        };
        let NodePayload::Unop(Unop::Not, x_ref) = &f.get_node(*not_ref).payload else {
            panic!("expected not(x) after rewrite");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
    }

    #[test]
    fn const_shll_concat_zero_folds_concat_slice_zero() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[8] {
  bit_slice.10: bits[5] = bit_slice(x, start=0, width=5, id=10)
  literal.11: bits[3] = literal(value=0, id=11)
  ret concat.12: bits[8] = concat(bit_slice.10, literal.11, id=12)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ret_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::Concat, _)) {
                ret_ref = Some(nr);
            }
        }
        let ret_ref = ret_ref.expect("expected ret node");

        let t = ConstShllConcatZeroFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(ret_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Shll, x_ref, k_ref) = &f.get_node(ret_ref).payload else {
            panic!("expected shll after rewrite");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
        let NodePayload::Literal(v) = &f.get_node(*k_ref).payload else {
            panic!("expected literal shift amount");
        };
        let bits = IrBits::make_ubits(8, 3).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        assert_eq!(*v, expected);
    }

    #[test]
    fn const_shll_concat_zero_expands_shll_const() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[8] {
  literal.10: bits[8] = literal(value=3, id=10)
  ret shll.11: bits[8] = shll(x, literal.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ret_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(
                f.get_node(nr).payload,
                NodePayload::Binop(Binop::Shll, _, _)
            ) {
                ret_ref = Some(nr);
            }
        }
        let ret_ref = ret_ref.expect("expected ret node");

        let t = ConstShllConcatZeroFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(ret_ref))
            .expect("apply");

        let NodePayload::Nary(NaryOp::Concat, ops) = &f.get_node(ret_ref).payload else {
            panic!("expected concat after rewrite");
        };
        assert_eq!(ops.len(), 2);
        let NodePayload::BitSlice { arg, start, width } = &f.get_node(ops[0]).payload else {
            panic!("expected bit_slice in concat");
        };
        assert!(matches!(f.get_node(*arg).payload, NodePayload::GetParam(_)));
        assert_eq!(*start, 0);
        assert_eq!(*width, 5);
        let NodePayload::Literal(v) = &f.get_node(ops[1]).payload else {
            panic!("expected literal zero in concat");
        };
        let bits = IrBits::make_ubits(3, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        assert_eq!(*v, expected);
    }

    #[test]
    fn sub_to_add_neg_expands_sub() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret sub.10: bits[8] = sub(x, y, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sub_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Sub, _, _)) {
                sub_ref = Some(nr);
            }
        }
        let sub_ref = sub_ref.expect("expected sub node");

        let t = SubToAddNegTransform;
        t.apply(&mut f, &TransformLocation::Node(sub_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Add, x_ref, neg_ref) = &f.get_node(sub_ref).payload else {
            panic!("expected add after rewrite");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
        let NodePayload::Unop(Unop::Neg, y_ref) = &f.get_node(*neg_ref).payload else {
            panic!("expected neg(y) after rewrite");
        };
        assert!(matches!(
            f.get_node(*y_ref).payload,
            NodePayload::GetParam(_)
        ));
    }

    #[test]
    fn sub_to_add_neg_folds_add_neg() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  neg.10: bits[8] = neg(y, id=10)
  ret add.11: bits[8] = add(x, neg.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut add_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _)) {
                add_ref = Some(nr);
            }
        }
        let add_ref = add_ref.expect("expected add node");

        let t = SubToAddNegTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Sub, x_ref, y_ref) = &f.get_node(add_ref).payload else {
            panic!("expected sub after rewrite");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
        assert!(matches!(
            f.get_node(*y_ref).payload,
            NodePayload::GetParam(_)
        ));
    }

    #[test]
    fn neg_sub_swap_folds_neg_sub_to_swapped_sub() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  sub.10: bits[8] = sub(x, y, id=10)
  ret neg.11: bits[8] = neg(sub.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut neg_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Neg, _)) {
                neg_ref = Some(nr);
            }
        }
        let neg_ref = neg_ref.expect("expected neg node");

        let t = NegSubSwapTransform;
        t.apply(&mut f, &TransformLocation::Node(neg_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Sub, y_ref, x_ref) = &f.get_node(neg_ref).payload else {
            panic!("expected swapped sub after rewrite");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
        assert!(matches!(
            f.get_node(*y_ref).payload,
            NodePayload::GetParam(_)
        ));
    }

    #[test]
    fn neg_sub_swap_expands_sub_to_neg_sub_swapped() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret sub.10: bits[8] = sub(x, y, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sub_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Sub, _, _)) {
                sub_ref = Some(nr);
            }
        }
        let sub_ref = sub_ref.expect("expected sub node");

        let t = NegSubSwapTransform;
        t.apply(&mut f, &TransformLocation::Node(sub_ref))
            .expect("apply");

        let NodePayload::Unop(Unop::Neg, inner_sub_ref) = &f.get_node(sub_ref).payload else {
            panic!("expected neg(sub(..)) after rewrite");
        };
        let NodePayload::Binop(Binop::Sub, y_ref, x_ref) = &f.get_node(*inner_sub_ref).payload
        else {
            panic!("expected inner sub after rewrite");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
        assert!(matches!(
            f.get_node(*y_ref).payload,
            NodePayload::GetParam(_)
        ));
    }

    #[test]
    fn reassociate_add_sub_reassociates_add_chain() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.10: bits[8] = add(a, b, id=10)
  ret add.11: bits[8] = add(add.10, c, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut add_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(nr).payload else {
                continue;
            };
            if matches!(
                f.get_node(lhs).payload,
                NodePayload::Binop(Binop::Add, _, _)
            ) && matches!(f.get_node(rhs).payload, NodePayload::GetParam(_))
            {
                add_ref = Some(nr);
            }
        }
        let add_ref = add_ref.expect("expected outer add node");

        let t = ReassociateAddSubTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Add, lhs, rhs) = &f.get_node(add_ref).payload else {
            panic!("expected add after rewrite");
        };
        assert!(matches!(f.get_node(*lhs).payload, NodePayload::GetParam(_)));
        let NodePayload::Binop(Binop::Add, rb, rc) = &f.get_node(*rhs).payload else {
            panic!("expected nested add on rhs after rewrite");
        };
        assert!(matches!(f.get_node(*rb).payload, NodePayload::GetParam(_)));
        assert!(matches!(f.get_node(*rc).payload, NodePayload::GetParam(_)));
    }

    #[test]
    fn carry_split_add_expands_and_folds_round_trip() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret add.10: bits[8] = add(x, y, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut add_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _)) {
                add_ref = Some(nr);
            }
        }
        let add_ref = add_ref.expect("expected add node");

        let t = CarrySplitAddTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("expand");

        let NodePayload::Nary(NaryOp::Concat, ops) = &f.get_node(add_ref).payload else {
            panic!("expected concat after expansion");
        };
        assert_eq!(ops.len(), 2);

        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("fold");

        let NodePayload::Binop(Binop::Add, x_ref, y_ref) = &f.get_node(add_ref).payload else {
            panic!("expected add after fold");
        };
        assert!(matches!(
            f.get_node(*x_ref).payload,
            NodePayload::GetParam(_)
        ));
        assert!(matches!(
            f.get_node(*y_ref).payload,
            NodePayload::GetParam(_)
        ));
    }

    #[test]
    fn eq_sel_distribute_expands_ne_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret ne.20: bits[1] = ne(sel.10, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ne_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Ne, _, _)) {
                ne_ref = Some(nr);
            }
        }
        let ne_ref = ne_ref.expect("expected ne node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(ne_ref))
            .expect("apply");

        match &f.get_node(ne_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Binop(Binop::Ne, _, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_sel_distribute_folds_sel_of_nes() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  ne.11: bits[1] = ne(a, c, id=11)
  ne.12: bits[1] = ne(b, c, id=12)
  ret sel.20: bits[1] = sel(p, cases=[ne.11, ne.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Binop(Binop::Ne, _, _)
        ));
    }

    #[test]
    fn eq_ne_add_literal_shift_expands_eq_add_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  add.11: bits[8] = add(x, literal.10, id=11)
  ret eq.20: bits[1] = eq(add.11, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        match &f.get_node(eq_ref).payload {
            NodePayload::Binop(Binop::Eq, _x, rhs) => {
                assert!(matches!(
                    f.get_node(*rhs).payload,
                    NodePayload::Binop(Binop::Sub, _, _)
                ));
            }
            other => panic!("expected eq(x, sub(...)) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_ne_add_literal_shift_expands_ne_add_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  add.11: bits[8] = add(literal.10, x, id=11)
  ret ne.20: bits[1] = ne(add.11, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ne_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Ne, _, _)) {
                ne_ref = Some(nr);
            }
        }
        let ne_ref = ne_ref.expect("expected ne node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(ne_ref))
            .expect("apply");

        match &f.get_node(ne_ref).payload {
            NodePayload::Binop(Binop::Ne, _x, rhs) => {
                assert!(matches!(
                    f.get_node(*rhs).payload,
                    NodePayload::Binop(Binop::Sub, _, _)
                ));
            }
            other => panic!("expected ne(x, sub(...)) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_ne_add_literal_shift_folds_eq_sub_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  sub.11: bits[8] = sub(c, literal.10, id=11)
  ret eq.20: bits[1] = eq(x, sub.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        match &f.get_node(eq_ref).payload {
            NodePayload::Binop(Binop::Eq, lhs, _c) => {
                assert!(matches!(
                    f.get_node(*lhs).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                ));
            }
            other => panic!("expected eq(add(...), c) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_ne_add_literal_shift_folds_ne_sub_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  sub.11: bits[8] = sub(c, literal.10, id=11)
  ret ne.20: bits[1] = ne(sub.11, x, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ne_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Ne, _, _)) {
                ne_ref = Some(nr);
            }
        }
        let ne_ref = ne_ref.expect("expected ne node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(ne_ref))
            .expect("apply");

        match &f.get_node(ne_ref).payload {
            NodePayload::Binop(Binop::Ne, lhs, _c) => {
                assert!(matches!(
                    f.get_node(*lhs).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                ));
            }
            other => panic!("expected ne(add(...), c) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn not_sel_distribute_expands_not_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret not.20: bits[8] = not(sel.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut not_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, _)) {
                not_ref = Some(nr);
            }
        }
        let not_ref = not_ref.expect("expected not node");

        let t = NotSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        match &f.get_node(not_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Unop(Unop::Not, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn not_sel_distribute_folds_sel_of_nots() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  not.11: bits[8] = not(a, id=11)
  not.12: bits[8] = not(b, id=12)
  ret sel.20: bits[8] = sel(p, cases=[not.11, not.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = NotSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn neg_sel_distribute_expands_neg_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret neg.20: bits[8] = neg(sel.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut neg_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Neg, _)) {
                neg_ref = Some(nr);
            }
        }
        let neg_ref = neg_ref.expect("expected neg node");

        let t = NegSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(neg_ref))
            .expect("apply");

        match &f.get_node(neg_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Unop(Unop::Neg, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn neg_sel_distribute_folds_sel_of_negs() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  neg.11: bits[8] = neg(a, id=11)
  neg.12: bits[8] = neg(b, id=12)
  ret sel.20: bits[8] = sel(p, cases=[neg.11, neg.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = NegSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Unop(Unop::Neg, _)
        ));
    }

    #[test]
    fn bit_slice_sel_distribute_expands_bit_slice_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[3] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret bit_slice.20: bits[3] = bit_slice(sel.10, start=1, width=3, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut bs_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::BitSlice { .. }) {
                bs_ref = Some(nr);
            }
        }
        let bs_ref = bs_ref.expect("expected bit_slice node");

        let t = BitSliceSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(bs_ref))
            .expect("apply");

        match &f.get_node(bs_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::BitSlice {
                            start: 1,
                            width: 3,
                            ..
                        }
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn bit_slice_sel_distribute_folds_sel_of_bit_slices() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[3] {
  bit_slice.11: bits[3] = bit_slice(a, start=1, width=3, id=11)
  bit_slice.12: bits[3] = bit_slice(b, start=1, width=3, id=12)
  ret sel.20: bits[3] = sel(p, cases=[bit_slice.11, bit_slice.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = BitSliceSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::BitSlice {
                start: 1,
                width: 3,
                ..
            }
        ));
    }

    #[test]
    fn sign_ext_sel_distribute_expands_sign_ext_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[16] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret sign_ext.20: bits[16] = sign_ext(sel.10, new_bit_count=16, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut se_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::SignExt { .. }) {
                se_ref = Some(nr);
            }
        }
        let se_ref = se_ref.expect("expected sign_ext node");

        let t = SignExtSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(se_ref))
            .expect("apply");

        match &f.get_node(se_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::SignExt {
                            new_bit_count: 16,
                            ..
                        }
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn sign_ext_sel_distribute_folds_sel_of_sign_exts() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[16] {
  sign_ext.11: bits[16] = sign_ext(a, new_bit_count=16, id=11)
  sign_ext.12: bits[16] = sign_ext(b, new_bit_count=16, id=12)
  ret sel.20: bits[16] = sel(p, cases=[sign_ext.11, sign_ext.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SignExtSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::SignExt {
                new_bit_count: 16,
                ..
            }
        ));
    }

    #[test]
    fn priority_sel1_to_sel_expands_priority_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret priority_sel.10: bits[8] = priority_sel(p, cases=[a], default=b, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ps_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::PrioritySel { .. }) {
                ps_ref = Some(nr);
            }
        }
        let ps_ref = ps_ref.expect("expected priority_sel node");

        let t = PrioritySel1ToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(ps_ref))
            .expect("apply");

        match &f.get_node(ps_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert!(default.is_none());
                assert_eq!(cases.len(), 2);
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn priority_sel1_to_sel_folds_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[b, a], id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = PrioritySel1ToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::PrioritySel { .. }
        ));
    }

    #[test]
    fn and_mask_sign_ext_to_sel_expands_and() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  sign_ext.10: bits[8] = sign_ext(b, new_bit_count=8, id=10)
  ret and.20: bits[8] = and(x, sign_ext.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut and_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::And, _)) {
                and_ref = Some(nr);
            }
        }
        let and_ref = and_ref.expect("expected and node");

        let t = AndMaskSignExtToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(and_ref))
            .expect("apply");

        match &f.get_node(and_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert!(default.is_none());
                assert_eq!(cases.len(), 2);
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn and_mask_sign_ext_to_sel_folds_sel() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  literal.10: bits[8] = literal(value=0, id=10)
  ret sel.20: bits[8] = sel(b, cases=[literal.10, x], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = AndMaskSignExtToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Nary(NaryOp::And, _)
        ));
    }

    #[test]
    fn xor_mask_sign_ext_to_sel_not_expands_xor() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  sign_ext.10: bits[8] = sign_ext(b, new_bit_count=8, id=10)
  ret xor.20: bits[8] = xor(x, sign_ext.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut xor_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::Xor, _)) {
                xor_ref = Some(nr);
            }
        }
        let xor_ref = xor_ref.expect("expected xor node");

        let t = XorMaskSignExtToSelNotTransform;
        t.apply(&mut f, &TransformLocation::Node(xor_ref))
            .expect("apply");

        match &f.get_node(xor_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert!(default.is_none());
                assert_eq!(cases.len(), 2);
                assert!(matches!(
                    f.get_node(cases[1]).payload,
                    NodePayload::Unop(Unop::Not, _)
                ));
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn xor_mask_sign_ext_to_sel_not_folds_sel() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  not.10: bits[8] = not(x, id=10)
  ret sel.20: bits[8] = sel(b, cases=[x, not.10], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = XorMaskSignExtToSelNotTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Nary(NaryOp::Xor, _)
        ));
    }

    #[test]
    fn sel_same_arms_fold_folds_sel_to_identity() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[a, a], id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SelSameArmsFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }

    #[test]
    fn sel_swap_arms_by_not_pred_removes_not_and_swaps() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  not.10: bits[1] = not(p, id=10)
  ret sel.20: bits[8] = sel(not.10, cases=[a, b], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SelSwapArmsByNotPredTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        match &f.get_node(sel_ref).payload {
            NodePayload::Sel {
                selector, cases, ..
            } => {
                assert_eq!(cases.len(), 2);
                // selector should now be `p`, not a `not(...)` node.
                assert!(matches!(
                    f.get_node(*selector).payload,
                    NodePayload::GetParam(_)
                ));
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn sel_swap_arms_by_not_pred_adds_not_and_swaps() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret sel.20: bits[8] = sel(p, cases=[b, a], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SelSwapArmsByNotPredTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        match &f.get_node(sel_ref).payload {
            NodePayload::Sel {
                selector, cases, ..
            } => {
                assert_eq!(cases.len(), 2);
                assert!(matches!(
                    f.get_node(*selector).payload,
                    NodePayload::Unop(Unop::Not, _)
                ));
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn not_not_cancel_folds_double_not() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[8] {
  not.10: bits[8] = not(x, id=10)
  ret not.11: bits[8] = not(not.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut out_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, _)) {
                out_ref = Some(nr);
            }
        }
        let out_ref = out_ref.expect("expected outer not node");

        let t = NotNotCancelTransform;
        t.apply(&mut f, &TransformLocation::Node(out_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(out_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }

    #[test]
    fn neg_neg_cancel_folds_double_neg() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[8] {
  neg.10: bits[8] = neg(x, id=10)
  ret neg.11: bits[8] = neg(neg.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut out_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Neg, _)) {
                out_ref = Some(nr);
            }
        }
        let out_ref = out_ref.expect("expected outer neg node");

        let t = NegNegCancelTransform;
        t.apply(&mut f, &TransformLocation::Node(out_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(out_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }

    #[test]
    fn not_eq_ne_flip_turns_not_eq_into_ne() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[1] {
  eq.10: bits[1] = eq(a, b, id=10)
  ret not.11: bits[1] = not(eq.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut not_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, _)) {
                not_ref = Some(nr);
            }
        }
        let not_ref = not_ref.expect("expected not node");

        let t = NotEqNeFlipTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(not_ref).payload,
            NodePayload::Binop(Binop::Ne, _, _)
        ));
    }

    #[test]
    fn nor_not_or_fold_expands_nor() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret nor.10: bits[8] = nor(a, b, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut nor_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::Nor, _)) {
                nor_ref = Some(nr);
            }
        }
        let nor_ref = nor_ref.expect("expected nor node");

        let t = NorNotOrFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(nor_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(nor_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn eq_zero_or_reduce_expands_eq_zero() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[1] {
  literal.10: bits[8] = literal(value=0, id=10)
  ret eq.20: bits[1] = eq(x, literal.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqZeroOrReduceTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(eq_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn bit_slice_bit_slice_fold_folds_nested_slices() {
        let ir_text = r#"fn t(x: bits[16] id=1) -> bits[4] {
  bit_slice.10: bits[8] = bit_slice(x, start=4, width=8, id=10)
  ret bit_slice.11: bits[4] = bit_slice(bit_slice.10, start=2, width=4, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut out_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::BitSlice { .. }) {
                out_ref = Some(nr);
            }
        }
        let out_ref = out_ref.expect("expected outer bit_slice node");

        let t = BitSliceBitSliceFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(out_ref))
            .expect("apply");

        match &f.get_node(out_ref).payload {
            NodePayload::BitSlice { start, width, .. } => {
                assert_eq!(*start, 6);
                assert_eq!(*width, 4);
            }
            other => panic!("expected bit_slice after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn priority_sel_to_sel_chain_lowers() {
        let ir_text = r#"fn t(p: bits[2] id=1, c0: bits[8] id=2, c1: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  ret priority_sel.10: bits[8] = priority_sel(p, cases=[c0, c1], default=d, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ps_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::PrioritySel { .. }) {
                ps_ref = Some(nr);
            }
        }
        let ps_ref = ps_ref.expect("expected priority_sel node");

        let t = PrioritySelToSelChainTransform;
        t.apply(&mut f, &TransformLocation::Node(ps_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(ps_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }
}
