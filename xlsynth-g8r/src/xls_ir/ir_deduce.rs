// SPDX-License-Identifier: Apache-2.0

//! Deduce result types for IR nodes from their payloads and operand types.
//!
//! This module provides a best-effort type deduction routine for a clear
//! subset of XLS IR operations where result types are unambiguous from the
//! operand types and payload attributes. Operations that require package-
//! level context (e.g. `invoke`) or richer semantics may return `Ok(None)`
//! to indicate deduction is not presently implemented.

use crate::xls_ir::ir::{ArrayTypeData, Binop, NaryOp, NodePayload, Type, Unop};

#[derive(Debug, PartialEq, Eq)]
pub enum DeduceError {
    MissingOperand(&'static str),
    ExpectedTuple(Type),
    TupleIndexOutOfBounds { index: usize, tuple_len: usize },
    ExpectedBits(&'static str),
    ExpectedArray(&'static str),
    ArrayEmpty,
    ArrayElementsNotSameType,
    ArrayConcatElementTypeMismatch,
    ArrayIndexNonArray,
    ConcatRequiresBits,
    SelectRequiresCaseOrDefault,
    SelectOperandListTooShort,
    SelectCasesNotSameType,
}

impl std::fmt::Display for DeduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeduceError::MissingOperand(ctx) => write!(f, "missing operand: {}", ctx),
            DeduceError::ExpectedTuple(got) => write!(f, "expected tuple operand, got {}", got),
            DeduceError::TupleIndexOutOfBounds { index, tuple_len } => write!(
                f,
                "tuple_index out of bounds: index={} tuple_len={}",
                index, tuple_len
            ),
            DeduceError::ExpectedBits(ctx) => write!(f, "expected bits operand for {}", ctx),
            DeduceError::ExpectedArray(ctx) => write!(f, "expected array operand for {}", ctx),
            DeduceError::ArrayEmpty => {
                write!(f, "cannot deduce array type from empty element list")
            }
            DeduceError::ArrayElementsNotSameType => {
                write!(f, "array elements must all have identical types")
            }
            DeduceError::ArrayConcatElementTypeMismatch => {
                write!(f, "array_concat element type mismatch")
            }
            DeduceError::ArrayIndexNonArray => {
                write!(f, "array_index indexing into non-array type")
            }
            DeduceError::ConcatRequiresBits => write!(f, "concat requires bits operands"),
            DeduceError::SelectRequiresCaseOrDefault => {
                write!(f, "select requires at least one case or default")
            }
            DeduceError::SelectOperandListTooShort => write!(f, "select operand list too short"),
            DeduceError::SelectCasesNotSameType => {
                write!(f, "select cases must have identical types")
            }
        }
    }
}

/// Attempts to deduce the result type for `payload` given the ordered
/// `operand_types` list (matching `ir_utils::operands(payload)`).
///
/// Returns Ok(Some(Type)) if deduction is supported for the payload, Ok(None)
/// if deduction is intentionally not implemented for the payload, and Err on
/// invalid/contradictory operand types for otherwise supported payloads.
pub fn deduce_result_type(
    payload: &NodePayload,
    operand_types: &[Type],
) -> Result<Option<Type>, DeduceError> {
    match payload {
        NodePayload::Nil => Ok(Some(Type::nil())),
        NodePayload::GetParam(_) => Ok(None),

        NodePayload::Tuple(_) => {
            let mut elems: Vec<Box<Type>> = Vec::with_capacity(operand_types.len());
            for t in operand_types.iter() {
                elems.push(Box::new(t.clone()));
            }
            Ok(Some(Type::Tuple(elems)))
        }
        NodePayload::Array(_) => {
            if operand_types.is_empty() {
                return Err(DeduceError::ArrayEmpty);
            }
            let first = operand_types.first().unwrap();
            for t in operand_types.iter() {
                if t != first {
                    return Err(DeduceError::ArrayElementsNotSameType);
                }
            }
            Ok(Some(Type::Array(ArrayTypeData {
                element_type: Box::new(first.clone()),
                element_count: operand_types.len(),
            })))
        }
        NodePayload::TupleIndex { tuple: _, index } => {
            let tuple_ty = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("tuple_index.tuple"))?;
            match tuple_ty {
                Type::Tuple(members) => {
                    if *index >= members.len() {
                        return Err(DeduceError::TupleIndexOutOfBounds {
                            index: *index,
                            tuple_len: members.len(),
                        });
                    }
                    Ok(Some(*members[*index].clone()))
                }
                _ => Err(DeduceError::ExpectedTuple(tuple_ty.clone())),
            }
        }

        NodePayload::Binop(binop, _, _) => match binop {
            // Arithmetic/logical ops that preserve lhs width.
            Binop::Add | Binop::Sub | Binop::Shll | Binop::Shrl | Binop::Shra => {
                let lhs_ty = operand_types
                    .get(0)
                    .ok_or(DeduceError::MissingOperand("binop.lhs"))?;
                match lhs_ty {
                    Type::Bits(w) => Ok(Some(Type::Bits(*w))),
                    _ => Err(DeduceError::ExpectedBits("binop.lhs")),
                }
            }
            // Array concatenation: element types must match; counts add.
            Binop::ArrayConcat => {
                let lhs_ty = operand_types
                    .get(0)
                    .ok_or(DeduceError::MissingOperand("array_concat.lhs"))?;
                let rhs_ty = operand_types
                    .get(1)
                    .ok_or(DeduceError::MissingOperand("array_concat.rhs"))?;
                match (lhs_ty, rhs_ty) {
                    (Type::Array(a), Type::Array(b)) => {
                        if a.element_type != b.element_type {
                            return Err(DeduceError::ArrayConcatElementTypeMismatch);
                        }
                        Ok(Some(Type::Array(ArrayTypeData {
                            element_type: a.element_type.clone(),
                            element_count: a.element_count + b.element_count,
                        })))
                    }
                    _ => Err(DeduceError::ExpectedArray("array_concat")),
                }
            }
            // Partial-product multiplies return (lo, hi) with width L+R each.
            Binop::Smulp | Binop::Umulp => {
                let lhs_ty = operand_types
                    .get(0)
                    .ok_or(DeduceError::MissingOperand("mulp.lhs"))?;
                let rhs_ty = operand_types
                    .get(1)
                    .ok_or(DeduceError::MissingOperand("mulp.rhs"))?;
                match (lhs_ty, rhs_ty) {
                    (Type::Bits(lw), Type::Bits(rw)) => {
                        let w = *lw + *rw;
                        Ok(Some(Type::Tuple(vec![
                            Box::new(Type::Bits(w)),
                            Box::new(Type::Bits(w)),
                        ])))
                    }
                    _ => Err(DeduceError::ExpectedBits("smulp/umulp")),
                }
            }
            // Comparisons (signed/unsigned) and equality produce bits[1].
            Binop::Eq
            | Binop::Ne
            | Binop::Uge
            | Binop::Ugt
            | Binop::Ult
            | Binop::Ule
            | Binop::Sgt
            | Binop::Sge
            | Binop::Slt
            | Binop::Sle => Ok(Some(Type::Bits(1))),
            // Division/modulus produce result with lhs width.
            Binop::Sdiv | Binop::Udiv | Binop::Umod | Binop::Smod => {
                let lhs_ty = operand_types
                    .get(0)
                    .ok_or(DeduceError::MissingOperand("binop.lhs"))?;
                match lhs_ty {
                    Type::Bits(w) => Ok(Some(Type::Bits(*w))),
                    _ => Err(DeduceError::ExpectedBits("binop.lhs")),
                }
            }

            // Leave other binops for future implementation.
            _ => Ok(None),
        },

        NodePayload::Unop(unop, _) => match unop {
            // Width-preserving unops: identity, not, reverse, neg
            Unop::Identity | Unop::Not | Unop::Reverse | Unop::Neg => {
                let arg_ty = operand_types
                    .get(0)
                    .ok_or(DeduceError::MissingOperand("unop.arg"))?;
                Ok(Some(arg_ty.clone()))
            }
            // Reductions yield a single bit.
            Unop::OrReduce | Unop::AndReduce | Unop::XorReduce => Ok(Some(Type::Bits(1))),
        },

        NodePayload::Literal(value) => {
            // Only support bits-typed literals for now.
            match value.bit_count() {
                Ok(w) => Ok(Some(Type::Bits(w))),
                Err(_) => Ok(None),
            }
        }

        NodePayload::SignExt {
            arg: _,
            new_bit_count,
        }
        | NodePayload::ZeroExt {
            arg: _,
            new_bit_count,
        } => Ok(Some(Type::Bits(*new_bit_count))),

        NodePayload::ArrayUpdate {
            array: _,
            value: _,
            indices: _,
            ..
        } => {
            let array_ty = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("array_update.array"))?;
            // We could validate value/index types in the future; for now, pass through.
            match array_ty {
                Type::Array(_) => Ok(Some(array_ty.clone())),
                _ => Err(DeduceError::ExpectedArray("array_update")),
            }
        }
        NodePayload::ArrayIndex {
            array: _, indices, ..
        } => {
            let mut cur = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("array_index.array"))?
                .clone();
            let index_depth = indices.len();
            for _ in 0..index_depth {
                match cur {
                    Type::Array(ArrayTypeData {
                        ref element_type, ..
                    }) => {
                        cur = *element_type.clone();
                    }
                    _ => {
                        return Err(DeduceError::ArrayIndexNonArray);
                    }
                }
            }
            Ok(Some(cur))
        }

        NodePayload::DynamicBitSlice { width, .. } | NodePayload::BitSlice { width, .. } => {
            Ok(Some(Type::Bits(*width)))
        }
        NodePayload::BitSliceUpdate {
            arg: _,
            start: _,
            update_value: _,
        } => {
            let arg_ty = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("bit_slice_update.arg"))?;
            Ok(Some(arg_ty.clone()))
        }

        NodePayload::Assert { .. } | NodePayload::Trace { .. } | NodePayload::AfterAll(_) => {
            Ok(Some(Type::Token))
        }

        NodePayload::Nary(op, _) => match op {
            // Bitwise n-ary ops preserve width.
            NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor => {
                let first = operand_types
                    .get(0)
                    .ok_or(DeduceError::MissingOperand("n-ary.first"))?;
                match first {
                    Type::Bits(w) => Ok(Some(Type::Bits(*w))),
                    _ => Err(DeduceError::ExpectedBits("n-ary")),
                }
            }
            // Concatenation sums bit widths.
            NaryOp::Concat => {
                let mut total: usize = 0;
                for t in operand_types.iter() {
                    match t {
                        Type::Bits(w) => total += *w,
                        _ => return Err(DeduceError::ConcatRequiresBits),
                    }
                }
                Ok(Some(Type::Bits(total)))
            }
        },
        // Selection family and enc/dec/one_hot.
        NodePayload::OneHot { .. } => {
            let arg_ty = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("one_hot.arg"))?;
            match arg_ty {
                Type::Bits(w) => Ok(Some(Type::Bits(*w + 1))),
                _ => Err(DeduceError::ExpectedBits("one_hot")),
            }
        }
        NodePayload::Sel { cases, default, .. }
        | NodePayload::PrioritySel { cases, default, .. } => {
            if cases.is_empty() && default.is_none() {
                return Err(DeduceError::SelectRequiresCaseOrDefault);
            }
            // operand_types: [selector, case0.., maybe default]
            let first_case_ty = operand_types
                .get(1)
                .or_else(|| operand_types.last())
                .ok_or(DeduceError::MissingOperand("select.case_or_default"))?;
            for (j, _) in cases.iter().enumerate() {
                let ty = operand_types
                    .get(1 + j)
                    .ok_or(DeduceError::SelectOperandListTooShort)?;
                if ty != first_case_ty {
                    return Err(DeduceError::SelectCasesNotSameType);
                }
            }
            Ok(Some(first_case_ty.clone()))
        }
        NodePayload::OneHotSel { cases, .. } => {
            if cases.is_empty() {
                return Err(DeduceError::SelectRequiresCaseOrDefault);
            }
            // operand_types: [selector, case0..]
            let first_case_ty = operand_types
                .get(1)
                .ok_or(DeduceError::MissingOperand("one_hot_sel.first_case"))?;
            for (j, _) in cases.iter().enumerate() {
                let ty = operand_types
                    .get(1 + j)
                    .ok_or(DeduceError::SelectOperandListTooShort)?;
                if ty != first_case_ty {
                    return Err(DeduceError::SelectCasesNotSameType);
                }
            }
            Ok(Some(first_case_ty.clone()))
        }
        NodePayload::Decode { width, .. } => Ok(Some(Type::Bits(*width))),
        NodePayload::Encode { .. } => {
            let arg_ty = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("encode.arg"))?;
            match arg_ty {
                Type::Bits(w) => Ok(Some(Type::Bits(ceil_log2_with_min_one(*w)))),
                _ => Err(DeduceError::ExpectedBits("encode")),
            }
        }
        NodePayload::CountedFor { .. } => {
            let init_ty = operand_types
                .get(0)
                .ok_or(DeduceError::MissingOperand("counted_for.init"))?;
            Ok(Some(init_ty.clone()))
        }
        NodePayload::Cover { .. } => Ok(Some(Type::Token)),
        // We need package context to look up the invoked function type.
        NodePayload::Invoke { .. } => Ok(None),
    }
}

fn ceil_log2_with_min_one(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        let mut v = n - 1;
        let mut k = 0usize;
        while v > 0 {
            k += 1;
            v >>= 1;
        }
        k
    }
}
