// SPDX-License-Identifier: Apache-2.0

//! Validation routines for XLS IR packages and functions.

use std::collections::HashSet;

use super::ir::{Fn, NaryOp, NodePayload, Package, PackageMember, Type};
use super::ir_deduce::deduce_result_type_with;
use super::ir_utils::operands;

/// Errors that can arise during validation of XLS IR structures.
#[derive(Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// Two package members share the same name.
    DuplicateMemberName(String),
    /// The `top` attribute references a missing function.
    MissingTopFunction(String),
    /// A node references an undefined operand (index out of bounds).
    OperandOutOfBounds {
        func: String,
        node_index: usize,
        operand: usize,
    },
    /// A node references an operand defined after the node.
    OperandUsesUndefined {
        func: String,
        node_index: usize,
        operand: usize,
    },
    /// A function's return node is missing.
    MissingReturnNode(String),
    /// A function's declared return type doesn't match the return node type.
    ReturnTypeMismatch {
        func: String,
        expected: Type,
        actual: Type,
    },
    /// A node's text id is not unique among non-parameter nodes.
    DuplicateTextId { func: String, text_id: usize },
    /// A parameter node's text id does not match its declared parameter id.
    ParamIdMismatch {
        func: String,
        param_name: String,
        expected: usize,
        actual: usize,
    },
    /// The function refers to another function that does not exist in the
    /// package.
    UnknownCallee { func: String, callee: String },
    /// Bitwise n-ary ops (and/or/xor/nand/nor) must have identical bits-typed
    /// operands.
    NaryBitwiseOperandTypeMismatch { func: String, node_index: usize },
    /// Two parameters share the same name within a function.
    DuplicateParamName { func: String, param_name: String },
    /// A parameter declared in the function signature has no corresponding
    /// GetParam node in the node list.
    MissingParamNode {
        func: String,
        param_name: String,
        expected_id: usize,
    },
    /// A GetParam node exists in the node list that does not correspond to any
    /// declared parameter in the function signature.
    ExtraParamNode { func: String, text_id: usize },
    /// A node name looks like a default textual id (e.g. op.id) but the
    /// operator prefix does not match the node's actual operator.
    NodeNameOpMismatch {
        func: String,
        node_index: usize,
        name: String,
        expected_op: String,
    },
    /// A node name looks like a default textual id (e.g. op.id) but the numeric
    /// suffix does not match the node's text id.
    NodeNameIdSuffixMismatch {
        func: String,
        node_index: usize,
        name: String,
        expected_id: usize,
    },
    /// A node's declared type does not match the type deduced from its
    /// operator and operand types.
    NodeTypeMismatch {
        func: String,
        node_index: usize,
        deduced: Type,
        actual: Type,
    },
    /// Type deduction failed for a node due to an internal error.
    TypeDeductionFailure {
        func: String,
        node_index: usize,
        reason: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::DuplicateMemberName(name) => {
                write!(f, "duplicate member name '{}'", name)
            }
            ValidationError::MissingTopFunction(name) => {
                write!(f, "top function '{}' not found", name)
            }
            ValidationError::OperandOutOfBounds {
                func,
                node_index,
                operand,
            } => {
                write!(
                    f,
                    "function '{}' node {} references operand {} out of bounds",
                    func, node_index, operand
                )
            }
            ValidationError::OperandUsesUndefined {
                func,
                node_index,
                operand,
            } => {
                write!(
                    f,
                    "function '{}' node {} uses operand {} before definition",
                    func, node_index, operand
                )
            }
            ValidationError::MissingReturnNode(func) => {
                write!(f, "function '{}' missing return node", func)
            }
            ValidationError::ReturnTypeMismatch {
                func,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' return type mismatch: expected {}, got {}",
                    func, expected, actual
                )
            }
            ValidationError::DuplicateTextId { func, text_id } => {
                write!(f, "function '{}' has duplicate text id {}", func, text_id)
            }
            ValidationError::ParamIdMismatch {
                func,
                param_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' param '{}' id mismatch: expected {}, got {}",
                    func, param_name, expected, actual
                )
            }
            ValidationError::UnknownCallee { func, callee } => {
                write!(
                    f,
                    "function '{}' references undefined callee '{}'",
                    func, callee
                )
            }
            ValidationError::NaryBitwiseOperandTypeMismatch { func, node_index } => {
                write!(
                    f,
                    "function '{}' node {} has mismatched operand types for bitwise n-ary op",
                    func, node_index
                )
            }
            ValidationError::DuplicateParamName { func, param_name } => {
                write!(
                    f,
                    "function '{}' has duplicate param name '{}'",
                    func, param_name
                )
            }
            ValidationError::MissingParamNode {
                func,
                param_name,
                expected_id,
            } => {
                write!(
                    f,
                    "function '{}' missing GetParam node for param '{}' (expected id={})",
                    func, param_name, expected_id
                )
            }
            ValidationError::ExtraParamNode { func, text_id } => {
                write!(
                    f,
                    "function '{}' has GetParam node with id {} not declared in signature",
                    func, text_id
                )
            }
            ValidationError::NodeNameOpMismatch {
                func,
                node_index,
                name,
                expected_op,
            } => {
                write!(
                    f,
                    "function '{}' node {} name '{}' operator prefix does not match op '{}'",
                    func, node_index, name, expected_op
                )
            }
            ValidationError::NodeNameIdSuffixMismatch {
                func,
                node_index,
                name,
                expected_id,
            } => {
                write!(
                    f,
                    "function '{}' node {} name '{}' id suffix does not match text id {}",
                    func, node_index, name, expected_id
                )
            }
            ValidationError::NodeTypeMismatch {
                func,
                node_index,
                deduced,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} type mismatch: deduced {} vs actual {}",
                    func, node_index, deduced, actual
                )
            }
            ValidationError::TypeDeductionFailure {
                func,
                node_index,
                reason,
            } => {
                write!(
                    f,
                    "function '{}' node {} type deduction failed: {}",
                    func, node_index, reason
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validates an entire package, ensuring all member names are unique, the top
/// function (if set) exists, and all contained functions are valid.
pub fn validate_package(p: &Package) -> Result<(), ValidationError> {
    let mut names = HashSet::new();
    for member in &p.members {
        let name = match member {
            PackageMember::Function(f) => &f.name,
            PackageMember::Block { func, .. } => &func.name,
        };
        if !names.insert(name.clone()) {
            return Err(ValidationError::DuplicateMemberName(name.clone()));
        }
    }

    if let Some(top) = &p.top_name {
        if !names.contains(top) {
            return Err(ValidationError::MissingTopFunction(top.clone()));
        }
    }

    for member in &p.members {
        match member {
            PackageMember::Function(f) => validate_fn(f, p)?,
            PackageMember::Block { func, .. } => validate_fn(func, p)?,
        }
    }

    // Enforce package-wide uniqueness of node text ids (including parameter nodes).
    let mut seen_ids: HashSet<usize> = HashSet::new();
    for member in &p.members {
        let f = match member {
            PackageMember::Function(f) => f,
            PackageMember::Block { func, .. } => func,
        };
        for node in f.nodes.iter() {
            // Skip synthetic Nil node at index 0 which is never emitted to IR text.
            if matches!(node.payload, NodePayload::Nil) {
                continue;
            }
            if !seen_ids.insert(node.text_id) {
                return Err(ValidationError::DuplicateTextId {
                    func: f.name.clone(),
                    text_id: node.text_id,
                });
            }
        }
    }

    Ok(())
}

/// Validates a function within the context of its parent package.
pub fn validate_fn(f: &Fn, parent: &Package) -> Result<(), ValidationError> {
    validate_fn_with(f, parent, |name: &str| {
        parent.get_fn_type(name).map(|ft| ft.return_type)
    })
}

/// Validates a function within the context of its parent package, using a
/// dependency-injected resolver for callee return types.
pub fn validate_fn_with<F>(
    f: &Fn,
    parent: &Package,
    callee_ret_type_resolver: F,
) -> Result<(), ValidationError>
where
    F: std::ops::Fn(&str) -> Option<Type>,
{
    // Track ids used by non-parameter nodes to ensure uniqueness.
    let mut seen_nonparam_ids: HashSet<usize> = HashSet::new();
    // Track GetParam node ids to verify 1:1 mapping with signature params.
    let mut seen_param_ids: HashSet<usize> = HashSet::new();
    // Map parameter names to their declared ids from the function signature, and
    // check name uniqueness.
    let mut param_name_to_id: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for p in &f.params {
        let name = p.name.as_str();
        if param_name_to_id.contains_key(name) {
            return Err(ValidationError::DuplicateParamName {
                func: f.name.clone(),
                param_name: p.name.clone(),
            });
        }
        param_name_to_id.insert(name, p.id.get_wrapped_id());
    }
    for (i, node) in f.nodes.iter().enumerate() {
        // Enforce: if a node has a name that looks like a default textual id
        // pattern '<prefix>.<digits>', then '<prefix>' must match the operator
        // and the numeric suffix must match the node's text id. This aligns with
        // external xlsynth verifier expectations and prevents misleading names.
        if let Some(ref name) = node.name {
            if let Some(dot_pos) = name.rfind('.') {
                let (prefix, suffix) = name.split_at(dot_pos);
                let suffix_digits = &suffix[1..]; // skip '.'
                if !suffix_digits.is_empty() && suffix_digits.chars().all(|c| c.is_ascii_digit()) {
                    let op_str = node.payload.get_operator();
                    if prefix != op_str {
                        return Err(ValidationError::NodeNameOpMismatch {
                            func: f.name.clone(),
                            node_index: i,
                            name: name.clone(),
                            expected_op: op_str.to_string(),
                        });
                    }
                    if let Ok(parsed_id) = suffix_digits.parse::<usize>() {
                        if parsed_id != node.text_id {
                            return Err(ValidationError::NodeNameIdSuffixMismatch {
                                func: f.name.clone(),
                                node_index: i,
                                name: name.clone(),
                                expected_id: node.text_id,
                            });
                        }
                    }
                }
            }
        }
        match &node.payload {
            NodePayload::GetParam(pid) => {
                let declared = node
                    .name
                    .as_ref()
                    .and_then(|n| param_name_to_id.get(n.as_str()))
                    .copied()
                    .unwrap_or(pid.get_wrapped_id());
                let actual_pid = pid.get_wrapped_id();
                // First: mismatch between declared and actual -> ParamIdMismatch.
                if actual_pid != declared || node.text_id != declared {
                    let param_name = node
                        .name
                        .clone()
                        .unwrap_or_else(|| "<unnamed-param>".to_string());
                    return Err(ValidationError::ParamIdMismatch {
                        func: f.name.clone(),
                        param_name,
                        expected: declared,
                        actual: node.text_id,
                    });
                }
                // Ensure this GetParam refers to a declared param id.
                if !param_name_to_id.values().any(|&v| v == actual_pid) {
                    return Err(ValidationError::ExtraParamNode {
                        func: f.name.clone(),
                        text_id: node.text_id,
                    });
                }
                // Ensure each GetParam id appears exactly once in the node list.
                if !seen_param_ids.insert(actual_pid) {
                    return Err(ValidationError::DuplicateTextId {
                        func: f.name.clone(),
                        text_id: actual_pid,
                    });
                }
            }
            _ => {
                if !seen_nonparam_ids.insert(node.text_id) {
                    return Err(ValidationError::DuplicateTextId {
                        func: f.name.clone(),
                        text_id: node.text_id,
                    });
                }
            }
        }
        // Ensure all operands refer to already defined nodes.
        for op in operands(&node.payload) {
            if op.index >= f.nodes.len() {
                return Err(ValidationError::OperandOutOfBounds {
                    func: f.name.clone(),
                    node_index: i,
                    operand: op.index,
                });
            }
            if op.index >= i {
                return Err(ValidationError::OperandUsesUndefined {
                    func: f.name.clone(),
                    node_index: i,
                    operand: op.index,
                });
            }
        }

        // Validate cross-package references.
        match &node.payload {
            NodePayload::Invoke { to_apply, .. } => {
                if !package_has_fn(parent, to_apply) {
                    return Err(ValidationError::UnknownCallee {
                        func: f.name.clone(),
                        callee: to_apply.clone(),
                    });
                }
            }
            NodePayload::CountedFor { body, .. } => {
                if !package_has_fn(parent, body) {
                    return Err(ValidationError::UnknownCallee {
                        func: f.name.clone(),
                        callee: body.clone(),
                    });
                }
            }
            _ => {}
        }

        // Enforce that bitwise n-ary ops have identically typed bit operands.
        if let NodePayload::Nary(op, elems) = &node.payload {
            match op {
                NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor => {
                    let first_ty = f.get_node(elems[0]).ty.clone();
                    log::trace!(
                        "validating nary op: {:?} first_ty: {:?}",
                        node.payload,
                        first_ty
                    );
                    // Require bits type and identical types across all operands.
                    for nr in elems.iter().skip(1) {
                        let operand_ty = &f.get_node(*nr).ty;
                        log::trace!(
                            "=> validating nary op: {:?} operand_ty: {:?}",
                            node.payload,
                            operand_ty
                        );
                        if operand_ty != &first_ty {
                            return Err(ValidationError::NaryBitwiseOperandTypeMismatch {
                                func: f.name.clone(),
                                node_index: i,
                            });
                        }
                    }
                }
                NaryOp::Concat => {
                    // Does not require identical types across all operands.
                }
            }
        }

        // After structural checks, ensure deduced node type matches declared.
        let op_refs = operands(&node.payload);
        let mut op_types: Vec<Type> = Vec::with_capacity(op_refs.len());
        for nr in op_refs.iter() {
            op_types.push(f.get_node(*nr).ty.clone());
        }
        match deduce_result_type_with(&node.payload, &op_types, |callee| {
            callee_ret_type_resolver(callee)
        }) {
            Ok(Some(deduced)) => {
                if deduced != node.ty {
                    return Err(ValidationError::NodeTypeMismatch {
                        func: f.name.clone(),
                        node_index: i,
                        deduced,
                        actual: node.ty.clone(),
                    });
                }
            }
            Ok(None) => {
                // No deduction available for this payload; skip.
            }
            Err(e) => {
                return Err(ValidationError::TypeDeductionFailure {
                    func: f.name.clone(),
                    node_index: i,
                    reason: e.to_string(),
                });
            }
        }
    }
    // Ensure every declared parameter has a corresponding GetParam node.
    for p in &f.params {
        let pid = p.id.get_wrapped_id();
        if !seen_param_ids.contains(&pid) {
            return Err(ValidationError::MissingParamNode {
                func: f.name.clone(),
                param_name: p.name.clone(),
                expected_id: pid,
            });
        }
    }

    let ret_node_ref = f
        .ret_node_ref
        .ok_or_else(|| ValidationError::MissingReturnNode(f.name.clone()))?;
    let ret_node = f.get_node(ret_node_ref);
    if ret_node.ty != f.ret_ty {
        return Err(ValidationError::ReturnTypeMismatch {
            func: f.name.clone(),
            expected: f.ret_ty.clone(),
            actual: ret_node.ty.clone(),
        });
    }

    Ok(())
}

fn package_has_fn(p: &Package, name: &str) -> bool {
    p.members.iter().any(|m| match m {
        PackageMember::Function(f) => f.name == name,
        PackageMember::Block { func, .. } => func.name == name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;
    use crate::ir_parser::Parser;

    #[test]
    fn validate_package_ok() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          ret add.2: bits[1] = add(x, x)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        validate_package(&pkg).unwrap();
    }

    #[test]
    fn undefined_operand_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          tmp: bits[1] = add(x, x, id=2)
          ret neg: bits[1] = neg(tmp, id=3)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg
                .members
                .iter_mut()
                .find_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    _ => None,
                })
                .unwrap();
            // Make node 1 reference a future node (index 2).
            if let NodePayload::Binop(_, ref mut a, _) = f.nodes[2].payload {
                *a = ir::NodeRef { index: 2 };
            }
        }
        let f = pkg
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) => Some(f),
                _ => None,
            })
            .unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::OperandUsesUndefined { .. })
        ));
    }

    #[test]
    fn return_type_mismatch_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          ret add.2: bits[1] = add(x, x)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg
                .members
                .iter_mut()
                .find_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    _ => None,
                })
                .unwrap();
            f.ret_ty = Type::Bits(2);
        }
        let f = pkg
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) => Some(f),
                _ => None,
            })
            .unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::ReturnTypeMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_text_id_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          a: bits[1] = add(x, x, id=2)
          b: bits[1] = add(a, x, id=2)
          ret b: bits[1] = identity(b, id=3)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::DuplicateTextId { .. })
        ));
    }

    #[test]
    fn name_operator_prefix_mismatch_fails() {
        let ir = r#"
        package test

        fn foo() -> bits[8] {
          ret one.2: bits[8] = literal(value=1, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let err = parser.parse_package().unwrap_err();
        assert_eq!(
            format!("{}", err),
            "ParseError: node name dotted prefix 'one' does not match operator 'literal'"
        );
    }

    #[test]
    fn manual_construct_one_dot_id_literal_fails() {
        // Build a function programmatically containing a node named "one.2"
        // with operator literal(id=2). This should fail with NodeNameOpMismatch.
        let mut pkg = ir::Package {
            name: "test".to_string(),
            file_table: ir::FileTable::new(),
            members: Vec::new(),
            top_name: Some("f".to_string()),
        };
        let lit_node = ir::Node {
            text_id: 2,
            name: Some("one.2".to_string()),
            ty: ir::Type::Bits(8),
            payload: ir::NodePayload::Literal(xlsynth::IrValue::make_ubits(8, 1).unwrap()),
            pos: None,
        };
        let f = ir::Fn {
            name: "f".to_string(),
            params: Vec::new(),
            ret_ty: ir::Type::Bits(8),
            nodes: vec![
                ir::Node {
                    text_id: 0,
                    name: Some("reserved_zero_node".to_string()),
                    ty: ir::Type::nil(),
                    payload: ir::NodePayload::Nil,
                    pos: None,
                },
                lit_node,
            ],
            ret_node_ref: Some(ir::NodeRef { index: 1 }),
        };
        pkg.members.push(ir::PackageMember::Function(f.clone()));
        let fref = pkg.get_top().unwrap();
        assert!(matches!(
            super::validate_fn(fref, &pkg),
            Err(ValidationError::NodeNameOpMismatch { .. })
        ));
    }

    #[test]
    fn param_id_mismatch_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1] id=7) -> bits[1] {
          x: bits[1] = param(name=x, id=1)
          ret x: bits[1] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::ParamIdMismatch { .. })
        ));
    }

    #[test]
    fn unknown_callee_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8]) -> bits[8] {
          ret r: bits[8] = invoke(x, to_apply=bar, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::UnknownCallee { .. })
        ));
    }

    #[test]
    fn duplicate_getparam_node_id_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8]) -> bits[8] {
          ret x: bits[8] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg.get_top_mut().unwrap();
            // Manually insert a duplicate GetParam node with the same id as 'x'.
            let pid = f.params[0].id;
            let dup = ir::Node {
                text_id: pid.get_wrapped_id(),
                name: Some(f.params[0].name.clone()),
                ty: f.params[0].ty.clone(),
                payload: ir::NodePayload::GetParam(pid),
                pos: None,
            };
            f.nodes.push(dup);
        }
        let f_ro = pkg.get_top().unwrap();
        assert!(matches!(
            validate_fn(f_ro, &pkg),
            Err(ValidationError::DuplicateTextId { .. })
        ));
    }

    #[test]
    fn missing_getparam_node_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8]) -> bits[8] {
          ret x: bits[8] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg.get_top_mut().unwrap();
            // Remove the GetParam node for 'x'. It should be at index 1.
            let idx = f
                .nodes
                .iter()
                .position(|n| matches!(n.payload, NodePayload::GetParam(_)))
                .unwrap();
            f.nodes.remove(idx);
        }
        let f_ro = pkg.get_top().unwrap();
        let err = validate_fn(f_ro, &pkg).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::MissingParamNode { .. } | ValidationError::OperandUsesUndefined { .. }
        ));
    }

    #[test]
    fn duplicate_param_name_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8], x: bits[8]) -> bits[8] {
          ret x: bits[8] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::DuplicateParamName { .. })
        ));
    }

    #[test]
    fn package_level_invoke_type_mismatch_fails() {
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
        let pkg = parser.parse_package().unwrap();
        // Public entry point should surface a node type mismatch error.
        let err = validate_package(&pkg).unwrap_err();
        match err {
            ValidationError::NodeTypeMismatch { .. } => {}
            other => panic!("expected NodeTypeMismatch, got {:?}", other),
        }
    }
}
