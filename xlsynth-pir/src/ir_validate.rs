// SPDX-License-Identifier: Apache-2.0

//! Validation routines for XLS IR packages and functions.

use std::collections::HashSet;

use super::ir::{Fn, NaryOp, NodePayload, Package, PackageMember, Type};
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
          tmp.2: bits[1] = add(x, x)
          ret neg.3: bits[1] = neg(tmp.2)
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
          a.2: bits[1] = add(x, x, id=2)
          b.2: bits[1] = add(a.2, x, id=2)
          ret b.2: bits[1] = identity(b.2, id=3)
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
}
