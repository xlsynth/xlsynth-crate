// SPDX-License-Identifier: Apache-2.0

//! Deduce result types for IR nodes from their payloads and operand types.
//!
//! This module provides a best-effort type deduction routine for a clear
//! subset of XLS IR operations where result types are unambiguous from the
//! operand types and payload attributes. Operations that require package-
//! level context (e.g. `invoke`) or richer semantics may return `Ok(None)`
//! to indicate deduction is not presently implemented.

use crate::xls_ir::ir::{ArrayTypeData, Binop, NaryOp, NodePayload, Type, Unop};

/// Attempts to deduce the result type for `payload` given the ordered
/// `operand_types` list (matching `ir_utils::operands(payload)`).
///
/// Returns Ok(Some(Type)) if deduction is supported for the payload, Ok(None)
/// if deduction is intentionally not implemented for the payload, and Err on
/// invalid/contradictory operand types for otherwise supported payloads.
pub fn deduce_result_type(
    payload: &NodePayload,
    operand_types: &[Type],
) -> Result<Option<Type>, String> {
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
                return Err("cannot deduce array type from empty element list".to_string());
            }
            let first = operand_types.first().unwrap();
            for t in operand_types.iter() {
                if t != first {
                    return Err("array elements must all have identical types".to_string());
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
                .ok_or_else(|| "missing tuple operand for tuple_index".to_string())?;
            match tuple_ty {
                Type::Tuple(members) => {
                    if *index >= members.len() {
                        return Err(format!(
                            "tuple_index out of bounds: index={} tuple_len={}",
                            index,
                            members.len()
                        ));
                    }
                    Ok(Some(*members[*index].clone()))
                }
                _ => Err(format!(
                    "tuple_index requires tuple operand, got {:?}",
                    tuple_ty
                )),
            }
        }

        NodePayload::Binop(binop, _, _) => match binop {
            // Arithmetic/logical ops that preserve lhs width.
            Binop::Add | Binop::Sub | Binop::Shll | Binop::Shrl | Binop::Shra => {
                let lhs_ty = operand_types
                    .get(0)
                    .ok_or_else(|| "missing lhs operand".to_string())?;
                match lhs_ty {
                    Type::Bits(w) => Ok(Some(Type::Bits(*w))),
                    _ => Err(format!(
                        "binop {:?} requires bits lhs, got {:?}",
                        binop, lhs_ty
                    )),
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

            // Leave other binops for future implementation.
            _ => Ok(None),
        },

        NodePayload::Unop(unop, _) => match unop {
            // Width-preserving unops: identity, not, reverse, neg
            Unop::Identity | Unop::Not | Unop::Reverse | Unop::Neg => {
                let arg_ty = operand_types
                    .get(0)
                    .ok_or_else(|| "missing unop operand".to_string())?;
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
                .ok_or_else(|| "missing array operand for array_update".to_string())?;
            // We could validate value/index types in the future; for now, pass through.
            match array_ty {
                Type::Array(_) => Ok(Some(array_ty.clone())),
                _ => Err(format!(
                    "array_update requires array type, got {:?}",
                    array_ty
                )),
            }
        }
        NodePayload::ArrayIndex {
            array: _, indices, ..
        } => {
            let mut cur = operand_types
                .get(0)
                .ok_or_else(|| "missing array operand for array_index".to_string())?
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
                        return Err("array_index indexing into non-array type".to_string());
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
                .ok_or_else(|| "missing arg operand for bit_slice_update".to_string())?;
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
                    .ok_or_else(|| "missing operand for n-ary op".to_string())?;
                match first {
                    Type::Bits(w) => Ok(Some(Type::Bits(*w))),
                    _ => Err(format!("n-ary op {:?} requires bits operands", op)),
                }
            }
            NaryOp::Concat => Ok(None),
        },

        NodePayload::Invoke { .. }
        | NodePayload::PrioritySel { .. }
        | NodePayload::OneHotSel { .. }
        | NodePayload::OneHot { .. }
        | NodePayload::Sel { .. }
        | NodePayload::Cover { .. }
        | NodePayload::Decode { .. }
        | NodePayload::Encode { .. }
        | NodePayload::CountedFor { .. } => Ok(None),
    }
}
