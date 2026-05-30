// SPDX-License-Identifier: Apache-2.0

//! Verification routines for PIR functions and packages.

use std::collections::HashSet;

use crate::ir::{self, Fn, NodePayload, Package, PackageMember, Type};
use crate::ir_deduce::{deduce_result_type, deduce_result_type_with};
use crate::ir_utils::operands;

mod package;

pub use package::{InstantiationPortDirection, ValidationError as VerifyError};

/// Verifies a standalone function and all constraints that do not require
/// package metadata or another function's signature.
pub fn verify_function(f: &Fn) -> Result<(), VerifyError> {
    package::validate_standalone_fn(f)
}

/// Verifies a function with access to the containing package for callee checks.
pub fn verify_function_in_package(f: &Fn, pkg: &Package) -> Result<(), VerifyError> {
    package::validate_fn(f, pkg)
}

/// Verifies a block with access to package members and block metadata.
pub fn verify_block_in_package(
    f: &Fn,
    metadata: &ir::BlockMetadata,
    pkg: &Package,
    member_index: usize,
) -> Result<(), VerifyError> {
    package::validate_block(f, metadata, pkg, member_index)
}

/// Verifies an entire PIR package, including function and block context.
pub fn verify_package(pkg: &Package) -> Result<(), VerifyError> {
    package::validate_package(pkg)
}

/// Verifies that all node text IDs within a function are unique.
pub fn verify_fn_unique_node_ids(f: &Fn) -> Result<(), String> {
    let mut seen: HashSet<usize> = HashSet::new();
    for (idx, n) in f.nodes.iter().enumerate() {
        if !seen.insert(n.text_id) {
            return Err(format!(
                "duplicate node id={} found at node index {} in function '{}'",
                n.text_id, idx, f.name
            ));
        }
    }
    Ok(())
}

/// Verifies that all NodeRef indices referenced by payloads are within bounds.
pub fn verify_fn_operand_indices_in_bounds(f: &Fn) -> Result<(), String> {
    let n = f.nodes.len();
    let check = |nr: ir::NodeRef, ctx: &str| -> Result<(), String> {
        if nr.index >= n {
            return Err(format!(
                "operand index {} out of bounds in {}; function '{}' has {} nodes",
                nr.index, ctx, f.name, n
            ));
        }
        Ok(())
    };
    for (i, node) in f.nodes.iter().enumerate() {
        match &node.payload {
            NodePayload::Nil | NodePayload::GetParam(_) | NodePayload::Literal(_) => {}
            NodePayload::Tuple(nodes)
            | NodePayload::Array(nodes)
            | NodePayload::ArrayConcat(nodes)
            | NodePayload::AfterAll(nodes)
            | NodePayload::Nary(_, nodes) => {
                for r in nodes.iter() {
                    check(*r, &format!("node {} payload list", i))?;
                }
            }
            NodePayload::TupleIndex { tuple, .. }
            | NodePayload::Unop(_, tuple)
            | NodePayload::Decode { arg: tuple, .. }
            | NodePayload::Encode { arg: tuple }
            | NodePayload::OneHot { arg: tuple, .. }
            | NodePayload::BitSlice { arg: tuple, .. } => {
                check(*tuple, &format!("node {} single", i))?;
            }
            NodePayload::ArraySlice { array, start, .. } => {
                check(*array, &format!("node {} array_slice.array", i))?;
                check(*start, &format!("node {} array_slice.start", i))?;
            }
            NodePayload::Binop(_, a, b) => {
                check(*a, &format!("node {} binop.lhs", i))?;
                check(*b, &format!("node {} binop.rhs", i))?;
            }
            NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
                check(*lhs, &format!("node {} ext_carry_out.lhs", i))?;
                check(*rhs, &format!("node {} ext_carry_out.rhs", i))?;
                check(*c_in, &format!("node {} ext_carry_out.c_in", i))?;
            }
            NodePayload::ExtPrioEncode { arg, lsb_prio: _ } => {
                check(*arg, &format!("node {} ext_prio_encode.arg", i))?;
            }
            NodePayload::ExtClz { arg, .. } => {
                check(*arg, &format!("node {} ext_clz.arg", i))?;
            }
            NodePayload::ExtNormalizeLeft { arg, .. } => {
                check(*arg, &format!("node {} ext_normalize_left.arg", i))?;
            }
            NodePayload::ExtMaskLow { count } => {
                check(*count, &format!("node {} ext_mask_low.count", i))?;
            }
            NodePayload::ExtNaryAdd { terms, arch: _ } => {
                for term in terms.iter() {
                    check(term.operand, &format!("node {} ext_nary_add.operand", i))?;
                }
            }
            NodePayload::SignExt { arg, .. } | NodePayload::ZeroExt { arg, .. } => {
                check(*arg, &format!("node {} ext.arg", i))?;
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } => {
                check(*array, &format!("node {} array_update.array", i))?;
                check(*value, &format!("node {} array_update.value", i))?;
                for r in indices.iter() {
                    check(*r, &format!("node {} array_update.index", i))?;
                }
            }
            NodePayload::ArrayIndex { array, indices, .. } => {
                check(*array, &format!("node {} array_index.array", i))?;
                for r in indices.iter() {
                    check(*r, &format!("node {} array_index.index", i))?;
                }
            }
            NodePayload::DynamicBitSlice { arg, start, .. } => {
                check(*arg, &format!("node {} dbs.arg", i))?;
                check(*start, &format!("node {} dbs.start", i))?;
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                check(*arg, &format!("node {} bsu.arg", i))?;
                check(*start, &format!("node {} bsu.start", i))?;
                check(*update_value, &format!("node {} bsu.update_value", i))?;
            }
            NodePayload::Assert {
                token, activate, ..
            } => {
                check(*token, &format!("node {} assert.token", i))?;
                check(*activate, &format!("node {} assert.activate", i))?;
            }
            NodePayload::Trace {
                token,
                activated,
                operands,
                ..
            } => {
                check(*token, &format!("node {} trace.token", i))?;
                check(*activated, &format!("node {} trace.activated", i))?;
                for r in operands.iter() {
                    check(*r, &format!("node {} trace.operand", i))?;
                }
            }
            NodePayload::InstantiationInput { arg, .. } => {
                check(*arg, &format!("node {} instantiation_input.arg", i))?;
            }
            NodePayload::InstantiationOutput { .. } => {}
            NodePayload::RegisterRead { .. } => {}
            NodePayload::RegisterWrite {
                arg,
                load_enable,
                reset,
                ..
            } => {
                check(*arg, &format!("node {} register_write.arg", i))?;
                if let Some(le) = load_enable {
                    check(*le, &format!("node {} register_write.load_enable", i))?;
                }
                if let Some(rst) = reset {
                    check(*rst, &format!("node {} register_write.reset", i))?;
                }
            }
            NodePayload::Invoke { operands, .. } => {
                for r in operands.iter() {
                    check(*r, &format!("node {} invoke.operand", i))?;
                }
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            }
            | NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                check(*selector, &format!("node {} sel.selector", i))?;
                for r in cases.iter() {
                    check(*r, &format!("node {} sel.case", i))?;
                }
                if let Some(d) = default {
                    check(*d, &format!("node {} sel.default", i))?;
                }
            }
            NodePayload::OneHotSel { selector, cases } => {
                check(*selector, &format!("node {} one_hot_sel.selector", i))?;
                for r in cases.iter() {
                    check(*r, &format!("node {} one_hot_sel.case", i))?;
                }
            }
            NodePayload::CountedFor {
                init,
                invariant_args,
                ..
            } => {
                check(*init, &format!("node {} counted_for.init", i))?;
                for r in invariant_args.iter() {
                    check(*r, &format!("node {} counted_for.invariant", i))?;
                }
            }
            NodePayload::Cover { predicate, .. } => {
                check(*predicate, &format!("node {} cover.predicate", i))?;
            }
        }
    }
    Ok(())
}

/// Verifies that for all nodes we can deduce, the node's `ty` matches the
/// type anticipated by the deduction routine.
pub fn verify_fn_types_agree_with_deduction(f: &Fn) -> Result<(), String> {
    for (i, node) in f.nodes.iter().enumerate() {
        // Gather operand types in operand order.
        let op_refs = operands(&node.payload);
        let mut op_types: Vec<Type> = Vec::with_capacity(op_refs.len());
        for nr in op_refs.iter() {
            op_types.push(f.get_node(*nr).ty.clone());
        }
        // Ask the deducer what the result type should be for this payload.
        match deduce_result_type(&node.payload, &op_types).map_err(|e| e.to_string())? {
            Some(deduced) => {
                if deduced != node.ty {
                    return Err(format!(
                        "type mismatch for node {} ({}): deduced {} vs actual {}",
                        i,
                        node.payload.get_operator(),
                        deduced,
                        node.ty
                    ));
                }
            }
            None => {
                // Deduction not implemented for this payload; skip.
            }
        }
    }
    Ok(())
}

fn type_contains_token(ty: &Type) -> bool {
    match ty {
        Type::Token => true,
        Type::Bits(_) => false,
        Type::Tuple(elements) => elements.iter().any(|element| type_contains_token(element)),
        Type::Array(array) => type_contains_token(&array.element_type),
    }
}

fn expect_bits_width(ty: &Type, context: &str) -> Result<usize, String> {
    match ty {
        Type::Bits(width) => Ok(*width),
        _ => Err(format!("{} must have bits type, got {}", context, ty)),
    }
}

fn expect_type(actual: &Type, expected: &Type, context: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{} must have type {}, got {}",
            context, expected, actual
        ))
    }
}

fn indexed_array_type<'a>(
    array_ty: &'a Type,
    index_count: usize,
    context: &str,
) -> Result<&'a Type, String> {
    if !matches!(array_ty, Type::Array(_)) {
        return Err(format!(
            "{} operand must have array type, got {}",
            context, array_ty
        ));
    }
    let mut current = array_ty;
    for _ in 0..index_count {
        let Type::Array(array) = current else {
            return Err(format!(
                "{} has more indices than dimensions in {}",
                context, array_ty
            ));
        };
        current = &array.element_type;
    }
    Ok(current)
}

fn minimum_unsigned_bit_count(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        (usize::BITS - value.leading_zeros()) as usize
    }
}

fn trace_operand_count(format: &str) -> Result<usize, String> {
    let mut offset = 0;
    let mut count = 0;
    let format_specs = [
        "{}", "{:u}", "{:d}", "{:x}", "{:0x}", "{:#x}", "{:b}", "{:0b}", "{:#b}",
    ];
    while offset < format.len() {
        let rest = &format[offset..];
        if rest.starts_with("{{") || rest.starts_with("}}") {
            offset += 2;
            continue;
        }
        if let Some(spec) = format_specs.iter().find(|spec| rest.starts_with(**spec)) {
            count += 1;
            offset += spec.len();
            continue;
        }
        let ch = rest.chars().next().expect("offset is within format");
        if ch == '{' || ch == '}' {
            return Err(format!("invalid trace format string {:?}", format));
        }
        offset += ch.len_utf8();
    }
    Ok(count)
}

/// Checks XLS `NodeChecker`-style local type and attribute constraints.
///
/// Unlike package validation, this routine does not require callees, register
/// declarations, or block instantiation metadata, so it is suitable for a
/// standalone function compiler.
pub fn verify_fn_xls_node_semantics(f: &Fn) -> Result<(), String> {
    verify_fn_operand_indices_in_bounds(f)?;
    for (node_index, node) in f.nodes.iter().enumerate() {
        verify_node_xls_semantics(f, node_index).map_err(|reason| {
            format!(
                "invalid node {} ({}) in function '{}': {}",
                node_index,
                node.payload.get_operator(),
                f.name,
                reason
            )
        })?;
    }
    Ok(())
}

pub(crate) fn verify_node_xls_semantics(f: &Fn, node_index: usize) -> Result<(), String> {
    let node = &f.nodes[node_index];
    let ty = |node_ref: ir::NodeRef| -> &Type { &f.get_node(node_ref).ty };

    match &node.payload {
        NodePayload::Nil
        | NodePayload::Tuple(_)
        | NodePayload::Array(_)
        | NodePayload::ArrayConcat(_)
        | NodePayload::TupleIndex { .. }
        | NodePayload::Literal(_)
        | NodePayload::InstantiationInput { .. }
        | NodePayload::InstantiationOutput { .. }
        | NodePayload::RegisterRead { .. }
        | NodePayload::RegisterWrite { .. }
        | NodePayload::Invoke { .. }
        | NodePayload::CountedFor { .. } => Ok(()),
        NodePayload::GetParam(param_id) => {
            if let Some(param) = f.params.iter().find(|param| param.id == *param_id) {
                expect_type(&node.ty, &param.ty, "parameter node result")
            } else {
                Ok(())
            }
        }
        NodePayload::ArraySlice {
            array,
            start,
            width,
        } => {
            let Type::Array(array_ty) = ty(*array) else {
                return Err(format!(
                    "array_slice operand must have array type, got {}",
                    ty(*array)
                ));
            };
            expect_bits_width(ty(*start), "array_slice start")?;
            if array_ty.element_count == 0 {
                return Err("array_slice cannot be applied to an empty array".to_string());
            }
            if *width == 0 {
                return Err("array_slice requires a positive width".to_string());
            }
            if type_contains_token(&node.ty) {
                return Err("array_slice result may not contain token values".to_string());
            }
            Ok(())
        }
        NodePayload::Binop(op, lhs, rhs) => {
            use ir::Binop;
            match op {
                Binop::Add | Binop::Sub | Binop::Sdiv | Binop::Udiv | Binop::Smod | Binop::Umod => {
                    expect_bits_width(&node.ty, node.payload.get_operator())?;
                    expect_type(ty(*lhs), &node.ty, "left operand")?;
                    expect_type(ty(*rhs), &node.ty, "right operand")
                }
                Binop::Shll | Binop::Shrl | Binop::Shra => {
                    expect_bits_width(&node.ty, node.payload.get_operator())?;
                    expect_type(ty(*lhs), &node.ty, "shifted operand")?;
                    expect_bits_width(ty(*rhs), "shift amount")?;
                    Ok(())
                }
                Binop::Smulp | Binop::Umulp => {
                    expect_bits_width(ty(*lhs), "partial-product left operand")?;
                    expect_bits_width(ty(*rhs), "partial-product right operand")?;
                    match &node.ty {
                        Type::Tuple(elements)
                            if elements.len() == 2
                                && elements[0] == elements[1]
                                && matches!(elements[0].as_ref(), Type::Bits(_)) =>
                        {
                            Ok(())
                        }
                        _ => Err(format!(
                            "partial-product result must have type (bits[N], bits[N]), got {}",
                            node.ty
                        )),
                    }
                }
                Binop::Eq | Binop::Ne => {
                    if ty(*lhs) == &Type::Token {
                        return Err("equality operands may not have token type".to_string());
                    }
                    expect_type(ty(*rhs), ty(*lhs), "right comparison operand")?;
                    expect_type(&node.ty, &Type::Bits(1), "comparison result")
                }
                Binop::Uge
                | Binop::Ugt
                | Binop::Ult
                | Binop::Ule
                | Binop::Sgt
                | Binop::Sge
                | Binop::Slt
                | Binop::Sle => {
                    let lhs_width = expect_bits_width(ty(*lhs), "left comparison operand")?;
                    let rhs_width = expect_bits_width(ty(*rhs), "right comparison operand")?;
                    if lhs_width != rhs_width {
                        return Err(format!(
                            "comparison operands must have equal bit widths, got {} and {}",
                            lhs_width, rhs_width
                        ));
                    }
                    expect_type(&node.ty, &Type::Bits(1), "comparison result")
                }
                Binop::Umul | Binop::Smul => {
                    expect_bits_width(ty(*lhs), "multiply left operand")?;
                    expect_bits_width(ty(*rhs), "multiply right operand")?;
                    expect_bits_width(&node.ty, "multiply result")?;
                    Ok(())
                }
                Binop::Gate => {
                    expect_type(ty(*lhs), &Type::Bits(1), "gate condition")?;
                    expect_type(ty(*rhs), &node.ty, "gate data operand")
                }
            }
        }
        NodePayload::Unop(op, arg) => {
            use ir::Unop;
            match op {
                Unop::Identity => expect_type(ty(*arg), &node.ty, "identity operand"),
                Unop::Neg | Unop::Not | Unop::Reverse => {
                    expect_bits_width(&node.ty, node.payload.get_operator())?;
                    expect_type(ty(*arg), &node.ty, "unary operand")
                }
                Unop::OrReduce | Unop::AndReduce | Unop::XorReduce => {
                    expect_bits_width(ty(*arg), "reduction operand")?;
                    expect_type(&node.ty, &Type::Bits(1), "reduction result")
                }
            }
        }
        NodePayload::SignExt { arg, new_bit_count } => {
            let old_width = expect_bits_width(ty(*arg), "sign_ext operand")?;
            if old_width == 0 {
                return Err("sign_ext requires a nonempty input".to_string());
            }
            if *new_bit_count < old_width {
                return Err(format!(
                    "sign_ext cannot truncate from bits[{}] to bits[{}]",
                    old_width, new_bit_count
                ));
            }
            expect_type(&node.ty, &Type::Bits(*new_bit_count), "sign_ext result")
        }
        NodePayload::ZeroExt { arg, new_bit_count } => {
            let old_width = expect_bits_width(ty(*arg), "zero_ext operand")?;
            if *new_bit_count < old_width {
                return Err(format!(
                    "zero_ext cannot truncate from bits[{}] to bits[{}]",
                    old_width, new_bit_count
                ));
            }
            expect_type(&node.ty, &Type::Bits(*new_bit_count), "zero_ext result")
        }
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            ..
        } => {
            expect_type(&node.ty, ty(*array), "array_update result")?;
            let indexed_ty = indexed_array_type(ty(*array), indices.len(), "array_update")?;
            for index in indices {
                expect_bits_width(ty(*index), "array_update index")?;
            }
            expect_type(ty(*value), indexed_ty, "array_update value")
        }
        NodePayload::ArrayIndex { array, indices, .. } => {
            let indexed_ty = indexed_array_type(ty(*array), indices.len(), "array_index")?;
            for index in indices {
                expect_bits_width(ty(*index), "array_index index")?;
            }
            if !indices.is_empty() {
                if let Type::Array(array_ty) = ty(*array) {
                    if array_ty.element_count == 0 {
                        return Err("array_index cannot be applied to an empty array".to_string());
                    }
                }
            }
            if type_contains_token(&node.ty) {
                return Err("array_index result may not contain token values".to_string());
            }
            expect_type(&node.ty, indexed_ty, "array_index result")
        }
        NodePayload::DynamicBitSlice { arg, start, width } => {
            let arg_width = expect_bits_width(ty(*arg), "dynamic_bit_slice operand")?;
            expect_bits_width(ty(*start), "dynamic_bit_slice start")?;
            if *width > arg_width {
                return Err(format!(
                    "dynamic_bit_slice width {} exceeds operand width {}",
                    width, arg_width
                ));
            }
            expect_type(&node.ty, &Type::Bits(*width), "dynamic_bit_slice result")
        }
        NodePayload::BitSlice { arg, start, width } => {
            let arg_width = expect_bits_width(ty(*arg), "bit_slice operand")?;
            if start
                .checked_add(*width)
                .map_or(true, |end| end > arg_width)
            {
                return Err(format!(
                    "bit_slice start {} plus width {} exceeds operand width {}",
                    start, width, arg_width
                ));
            }
            expect_type(&node.ty, &Type::Bits(*width), "bit_slice result")
        }
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => {
            expect_bits_width(ty(*arg), "bit_slice_update operand")?;
            expect_bits_width(ty(*start), "bit_slice_update start")?;
            expect_bits_width(ty(*update_value), "bit_slice_update value")?;
            expect_type(&node.ty, ty(*arg), "bit_slice_update result")
        }
        NodePayload::ExtCarryOut { .. }
        | NodePayload::ExtPrioEncode { .. }
        | NodePayload::ExtClz { .. }
        | NodePayload::ExtNormalizeLeft { .. } => Ok(()),
        NodePayload::ExtMaskLow { count } => {
            expect_bits_width(ty(*count), "ext_mask_low count")?;
            expect_bits_width(&node.ty, "ext_mask_low result")?;
            Ok(())
        }
        NodePayload::ExtNaryAdd { terms, .. } => {
            expect_bits_width(&node.ty, "ext_nary_add result")?;
            for term in terms {
                expect_bits_width(ty(term.operand), "ext_nary_add operand")?;
            }
            Ok(())
        }
        NodePayload::Assert {
            token, activate, ..
        } => {
            expect_type(ty(*token), &Type::Token, "assert token operand")?;
            expect_type(ty(*activate), &Type::Bits(1), "assert condition")?;
            expect_type(&node.ty, &Type::Token, "assert result")
        }
        NodePayload::Trace {
            token,
            activated,
            format,
            operands,
        } => {
            expect_type(ty(*token), &Type::Token, "trace token operand")?;
            expect_type(ty(*activated), &Type::Bits(1), "trace condition")?;
            expect_type(&node.ty, &Type::Token, "trace result")?;
            let expected_operands = trace_operand_count(format)?;
            if operands.len() != expected_operands {
                return Err(format!(
                    "trace format expects {} data operands, got {}",
                    expected_operands,
                    operands.len()
                ));
            }
            Ok(())
        }
        NodePayload::AfterAll(tokens) => {
            expect_type(&node.ty, &Type::Token, "after_all result")?;
            for token in tokens {
                expect_type(ty(*token), &Type::Token, "after_all operand")?;
            }
            Ok(())
        }
        NodePayload::Nary(op, operands) => {
            use ir::NaryOp;
            match op {
                NaryOp::And | NaryOp::Nor | NaryOp::Or | NaryOp::Xor | NaryOp::Nand => {
                    if operands.is_empty() {
                        return Err(format!(
                            "{} requires at least one operand",
                            node.payload.get_operator()
                        ));
                    }
                    expect_bits_width(&node.ty, node.payload.get_operator())?;
                    for operand in operands {
                        expect_type(ty(*operand), &node.ty, "bitwise operand")?;
                    }
                    Ok(())
                }
                NaryOp::Concat => {
                    let mut result_width = 0usize;
                    for operand in operands {
                        result_width += expect_bits_width(ty(*operand), "concat operand")?;
                    }
                    expect_type(&node.ty, &Type::Bits(result_width), "concat result")
                }
            }
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let selector_width = expect_bits_width(ty(*selector), "priority_sel selector")?;
            if cases.is_empty() {
                return Err("priority_sel requires at least one case".to_string());
            }
            if default.is_none() {
                return Err("priority_sel requires a default value".to_string());
            }
            if selector_width != cases.len() {
                return Err(format!(
                    "priority_sel selector has {} bits for {} cases",
                    selector_width,
                    cases.len()
                ));
            }
            for case in cases {
                expect_type(ty(*case), &node.ty, "priority_sel case")?;
            }
            expect_type(
                ty(default.expect("default checked above")),
                &node.ty,
                "priority_sel default",
            )
        }
        NodePayload::OneHotSel { selector, cases } => {
            let selector_width = expect_bits_width(ty(*selector), "one_hot_sel selector")?;
            if cases.is_empty() {
                return Err("one_hot_sel requires at least one case".to_string());
            }
            if selector_width != cases.len() {
                return Err(format!(
                    "one_hot_sel selector has {} bits for {} cases",
                    selector_width,
                    cases.len()
                ));
            }
            if type_contains_token(&node.ty) {
                return Err("one_hot_sel result may not contain token values".to_string());
            }
            for case in cases {
                expect_type(ty(*case), &node.ty, "one_hot_sel case")?;
            }
            Ok(())
        }
        NodePayload::OneHot { arg, .. } => {
            let arg_width = expect_bits_width(ty(*arg), "one_hot operand")?;
            expect_type(&node.ty, &Type::Bits(arg_width + 1), "one_hot result")
        }
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let selector_width = expect_bits_width(ty(*selector), "sel selector")?;
            if cases.is_empty() && default.is_none() {
                return Err("sel requires at least one case or a default".to_string());
            }
            let minimum_selector_width = if cases.is_empty() {
                0
            } else {
                minimum_unsigned_bit_count(cases.len() - 1)
            };
            if selector_width < minimum_selector_width {
                return Err(format!(
                    "sel selector needs at least {} bits for {} cases, got {}",
                    minimum_selector_width,
                    cases.len(),
                    selector_width
                ));
            }
            let complete_without_default =
                selector_width == minimum_selector_width && cases.len().is_power_of_two();
            if complete_without_default && default.is_some() {
                return Err("sel has a useless default value".to_string());
            }
            if !complete_without_default && default.is_none() {
                return Err(
                    "sel requires a default value for uncovered selector values".to_string()
                );
            }
            for case in cases {
                expect_type(ty(*case), &node.ty, "sel case")?;
            }
            if let Some(default) = default {
                expect_type(ty(*default), &node.ty, "sel default")?;
            }
            Ok(())
        }
        NodePayload::Cover { predicate, .. } => {
            expect_type(ty(*predicate), &Type::Bits(1), "cover predicate")?;
            expect_type(&node.ty, &Type::nil(), "cover result")
        }
        NodePayload::Decode { arg, width } => {
            let arg_width = expect_bits_width(ty(*arg), "decode operand")?;
            if let Some(max_width) = 1usize.checked_shl(arg_width as u32) {
                if *width > max_width {
                    return Err(format!(
                        "decode result width {} exceeds 2^{}",
                        width, arg_width
                    ));
                }
            }
            expect_type(&node.ty, &Type::Bits(*width), "decode result")
        }
        NodePayload::Encode { arg } => {
            expect_bits_width(ty(*arg), "encode operand")?;
            Ok(())
        }
    }
}

/// Verifies that all node text IDs are unique across the entire package
/// (functions and blocks), matching upstream XLS expectations.
pub fn verify_package_unique_node_ids(pkg: &Package) -> Result<(), String> {
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for m in pkg.members.iter() {
        let f: &Fn = match m {
            PackageMember::Function(f) => f,
            PackageMember::Block { func, .. } => func,
        };
        for (idx, n) in f.nodes.iter().enumerate() {
            if !seen.insert(n.text_id) {
                return Err(format!(
                    "duplicate node id={} found at node index {} in function '{}'",
                    n.text_id, idx, f.name
                ));
            }
        }
    }
    Ok(())
}

/// Like `verify_fn_types_agree_with_deduction`, but allows providing a
/// resolver for callee return types so that `invoke` result types can be
/// checked when package context is available.
pub fn verify_fn_types_agree_with_deduction_in_pkg(f: &Fn, pkg: &Package) -> Result<(), String> {
    for (i, node) in f.nodes.iter().enumerate() {
        let op_refs = operands(&node.payload);
        let mut op_types: Vec<Type> = Vec::with_capacity(op_refs.len());
        for nr in op_refs.iter() {
            op_types.push(f.get_node(*nr).ty.clone());
        }
        let resolver =
            |name: &str| -> Option<Type> { pkg.get_fn_type(name).map(|ft| ft.return_type) };
        match deduce_result_type_with(&node.payload, &op_types, resolver)
            .map_err(|e| e.to_string())?
        {
            Some(deduced) => {
                if deduced != node.ty {
                    return Err(format!(
                        "type mismatch for node {} ({}): deduced {} vs actual {}",
                        i,
                        node.payload.get_operator(),
                        deduced,
                        node.ty
                    ));
                }
            }
            None => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    #[test]
    fn type_mismatch_on_add_is_flagged() {
        let ir = r#"
fn foo(x: bits[8] id=1) -> bits[16] {
  add.2: bits[16] = add(x, x, id=2)
  ret identity.3: bits[16] = identity(add.2, id=3)
}
"#;
        let mut parser = Parser::new(ir);
        let f = parser.parse_fn().expect("parse fn");
        // add node is declared bits[16] but deduction expects bits[8]; should error.
        assert!(verify_fn_types_agree_with_deduction(&f).is_err());
    }

    #[test]
    fn type_mismatch_on_invoke_is_flagged_with_pkg_context() {
        let ir = r#"
package test

fn callee(x: bits[1] id=1) -> (bits[1], bits[1]) {
  ret tuple.3: (bits[1], bits[1]) = tuple(x, x, id=3)
}

fn foo(x: bits[1] id=1) -> bits[1] {
  invoke.2: bits[1] = invoke(x, to_apply=callee, id=2)
  ret identity.3: bits[1] = identity(invoke.2, id=3)
}
"#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().expect("parse package");
        let foo_fn = pkg
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) if f.name == "foo" => Some(f.clone()),
                _ => None,
            })
            .expect("find foo");
        assert!(verify_fn_types_agree_with_deduction_in_pkg(&foo_fn, &pkg).is_err());
    }

    #[test]
    fn standalone_verification_rejects_nodes_that_require_package_context() {
        let ir = r#"
package test

fn callee(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}

fn caller(x: bits[1] id=3) -> bits[1] {
  ret invoke.4: bits[1] = invoke(x, to_apply=callee, id=4)
}
"#;
        let pkg = Parser::new(ir).parse_package().expect("parse package");
        let caller = pkg.get_fn("caller").expect("find caller");

        assert!(matches!(
            verify_function(caller),
            Err(VerifyError::RequiresPackageContext { .. })
        ));
        verify_function_in_package(caller, &pkg).expect("package context resolves invoke");
        verify_package(&pkg).expect("valid package");
    }
}
