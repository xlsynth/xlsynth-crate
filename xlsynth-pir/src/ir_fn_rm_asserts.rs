// SPDX-License-Identifier: Apache-2.0

//! Helpers for removing assertion operations from PIR functions.

use crate::dce;
use crate::ir::{Fn, MemberType, NodePayload, NodeRef, Package, PackageMember};
use crate::ir_utils;
use std::fmt;

#[derive(Debug, Clone)]
pub struct RemoveAssertsResult {
    pub rewritten_fn: Fn,
    pub removed_assert_count: usize,
}

#[derive(Debug, Clone)]
pub struct RemoveAssertsPackageResult {
    pub rewritten_package: Package,
    pub selected_fn_name: String,
    pub removed_assert_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoveAssertsPackageError {
    TargetFunctionNotFound { name: String },
    TargetIsBlock { name: String },
    PackageTopIsBlock { name: String },
    PackageHasNoFunctions,
    PackageHasMultipleFunctionsWithoutTop,
}

impl fmt::Display for RemoveAssertsPackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RemoveAssertsPackageError::TargetFunctionNotFound { name } => {
                write!(f, "target function '{name}' not found")
            }
            RemoveAssertsPackageError::TargetIsBlock { name } => {
                write!(f, "target '{name}' is a block, not a function")
            }
            RemoveAssertsPackageError::PackageTopIsBlock { name } => {
                write!(
                    f,
                    "package top member '{name}' is a block; pass --top to select a function"
                )
            }
            RemoveAssertsPackageError::PackageHasNoFunctions => {
                write!(f, "package has no functions")
            }
            RemoveAssertsPackageError::PackageHasMultipleFunctionsWithoutTop => write!(
                f,
                "package has multiple functions and no explicit top function; pass --top"
            ),
        }
    }
}

impl std::error::Error for RemoveAssertsPackageError {}

/// Returns a clone of `f` with every `assert` bypassed and then full DCE
/// applied.
pub fn remove_asserts_from_fn(f: &Fn) -> RemoveAssertsResult {
    let mut rewritten = f.clone();
    let assert_refs: Vec<NodeRef> = rewritten
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            if matches!(node.payload, NodePayload::Assert { .. }) {
                Some(NodeRef { index })
            } else {
                None
            }
        })
        .collect();

    for assert_ref in assert_refs.iter().copied() {
        let NodePayload::Assert { token, .. } = rewritten.get_node(assert_ref).payload.clone()
        else {
            continue;
        };
        ir_utils::replace_node_with_ref(&mut rewritten, assert_ref, token)
            .expect("remove_asserts_from_fn: assert token bypass should be type-correct");
    }

    RemoveAssertsResult {
        rewritten_fn: dce::remove_dead_nodes(&rewritten),
        removed_assert_count: assert_refs.len(),
    }
}

/// Removes assertions from one selected function while leaving package top
/// metadata unchanged.
pub fn remove_asserts_from_package(
    pkg: &Package,
    target_fn: Option<&str>,
) -> Result<RemoveAssertsPackageResult, RemoveAssertsPackageError> {
    let selected_fn_name = resolve_target_fn_name(pkg, target_fn)?;
    let selected_fn = pkg
        .get_fn(&selected_fn_name)
        .expect("resolved target function should exist");
    let result = remove_asserts_from_fn(selected_fn);

    let mut rewritten_package = pkg.clone();
    let rewritten_target = rewritten_package
        .get_fn_mut(&selected_fn_name)
        .expect("resolved target function should exist in clone");
    *rewritten_target = result.rewritten_fn;

    Ok(RemoveAssertsPackageResult {
        rewritten_package,
        selected_fn_name,
        removed_assert_count: result.removed_assert_count,
    })
}

fn resolve_target_fn_name(
    pkg: &Package,
    target_fn: Option<&str>,
) -> Result<String, RemoveAssertsPackageError> {
    if let Some(target_fn) = target_fn {
        if pkg.get_fn(target_fn).is_some() {
            return Ok(target_fn.to_string());
        }
        if pkg.get_block(target_fn).is_some() {
            return Err(RemoveAssertsPackageError::TargetIsBlock {
                name: target_fn.to_string(),
            });
        }
        return Err(RemoveAssertsPackageError::TargetFunctionNotFound {
            name: target_fn.to_string(),
        });
    }

    match &pkg.top {
        Some((name, MemberType::Function)) => Ok(name.clone()),
        Some((name, MemberType::Block)) => {
            Err(RemoveAssertsPackageError::PackageTopIsBlock { name: name.clone() })
        }
        None => {
            let function_names: Vec<String> = pkg
                .members
                .iter()
                .filter_map(|member| match member {
                    PackageMember::Function(f) => Some(f.name.clone()),
                    PackageMember::Block { .. } => None,
                })
                .collect();
            match function_names.as_slice() {
                [name] => Ok(name.clone()),
                [] => Err(RemoveAssertsPackageError::PackageHasNoFunctions),
                _ => Err(RemoveAssertsPackageError::PackageHasMultipleFunctionsWithoutTop),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    fn parse_package(ir_text: &str) -> Package {
        let mut parser = Parser::new(ir_text);
        parser.parse_and_validate_package().unwrap()
    }

    fn op_count(f: &Fn, op: &str) -> usize {
        f.nodes
            .iter()
            .filter(|node| node.payload.get_operator() == op)
            .count()
    }

    #[test]
    fn remove_asserts_from_fn_bypasses_chained_asserts_and_dces_dead_logic() {
        let pkg = parse_package(
            r#"package p

top fn main(t: token id=1, x: bits[1] id=2) -> token {
  one: bits[1] = literal(value=1, id=3)
  dead_pred: bits[1] = and(x, one, id=4)
  assert.5: token = assert(t, dead_pred, message="m0", label="L0", id=5)
  ret assert.6: token = assert(assert.5, dead_pred, message="m1", label="L1", id=6)
  dead_unrelated: bits[1] = not(x, id=7)
}
"#,
        );
        let f = pkg.get_fn("main").unwrap();

        let result = remove_asserts_from_fn(f);
        let rewritten = result.rewritten_fn;

        assert_eq!(result.removed_assert_count, 2);
        assert_eq!(op_count(&rewritten, "assert"), 0);
        assert_eq!(op_count(&rewritten, "and"), 0);
        assert_eq!(op_count(&rewritten, "not"), 0);
        assert_eq!(rewritten.ret_node_ref, Some(NodeRef { index: 1 }));
    }

    #[test]
    fn remove_asserts_from_fn_keeps_non_assert_token_ops() {
        let pkg = parse_package(
            r#"package p

top fn main(t: token id=1, x: bits[1] id=2) -> token {
  assert.3: token = assert(t, x, message="m", label="L", id=3)
  ret trace.4: token = trace(assert.3, x, format="x={}", data_operands=[x], id=4)
}
"#,
        );
        let f = pkg.get_fn("main").unwrap();

        let result = remove_asserts_from_fn(f);
        let rewritten = result.rewritten_fn;

        assert_eq!(result.removed_assert_count, 1);
        assert_eq!(op_count(&rewritten, "assert"), 0);
        assert_eq!(op_count(&rewritten, "trace"), 1);
        let ret = rewritten.get_node(rewritten.ret_node_ref.unwrap());
        let NodePayload::Trace { token, .. } = &ret.payload else {
            panic!("expected trace return, got {:?}", ret.payload);
        };
        assert_eq!(*token, NodeRef { index: 1 });
    }

    #[test]
    fn remove_asserts_from_package_rewrites_selected_helper_and_preserves_top() {
        let pkg = parse_package(
            r#"package p

top fn main(t: token id=1, x: bits[1] id=2) -> token {
  ret assert.3: token = assert(t, x, message="main", label="M", id=3)
}

fn helper(t: token id=10, x: bits[1] id=11) -> token {
  ret assert.12: token = assert(t, x, message="helper", label="H", id=12)
}
"#,
        );

        let result = remove_asserts_from_package(&pkg, Some("helper")).unwrap();
        let rewritten = result.rewritten_package;

        assert_eq!(result.selected_fn_name, "helper");
        assert_eq!(result.removed_assert_count, 1);
        assert!(matches!(
            rewritten.top,
            Some((ref name, MemberType::Function)) if name == "main"
        ));
        assert_eq!(op_count(rewritten.get_fn("main").unwrap(), "assert"), 1);
        assert_eq!(op_count(rewritten.get_fn("helper").unwrap(), "assert"), 0);
    }

    #[test]
    fn remove_asserts_from_package_reports_selection_errors() {
        let block_top = parse_package(
            r#"package p

top block blk(x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  out: () = output_port(x, name=out, id=2)
}

fn helper(t: token id=10) -> token {
  ret t: token = param(name=t, id=10)
}
"#,
        );
        assert_eq!(
            remove_asserts_from_package(&block_top, None).unwrap_err(),
            RemoveAssertsPackageError::PackageTopIsBlock {
                name: "blk".to_string()
            }
        );
        assert_eq!(
            remove_asserts_from_package(&block_top, Some("blk")).unwrap_err(),
            RemoveAssertsPackageError::TargetIsBlock {
                name: "blk".to_string()
            }
        );
        assert_eq!(
            remove_asserts_from_package(&block_top, Some("missing")).unwrap_err(),
            RemoveAssertsPackageError::TargetFunctionNotFound {
                name: "missing".to_string()
            }
        );

        let no_functions = parse_package(
            r#"package p

block blk(x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  out: () = output_port(x, name=out, id=2)
}
"#,
        );
        assert_eq!(
            remove_asserts_from_package(&no_functions, None).unwrap_err(),
            RemoveAssertsPackageError::PackageHasNoFunctions
        );
    }

    #[test]
    fn remove_asserts_from_package_handles_no_top_packages_strictly() {
        let one_fn = parse_package(
            r#"package p

fn only(t: token id=1, x: bits[1] id=2) -> token {
  ret assert.3: token = assert(t, x, message="m", label="L", id=3)
}
"#,
        );
        let one_result = remove_asserts_from_package(&one_fn, None).unwrap();
        assert_eq!(one_result.selected_fn_name, "only");
        assert_eq!(
            op_count(
                one_result.rewritten_package.get_fn("only").unwrap(),
                "assert"
            ),
            0
        );
        assert!(one_result.rewritten_package.top.is_none());

        let multiple_fns = parse_package(
            r#"package p

fn a(t: token id=1) -> token {
  ret t: token = param(name=t, id=1)
}

fn b(t: token id=2) -> token {
  ret t: token = param(name=t, id=2)
}
"#,
        );
        assert_eq!(
            remove_asserts_from_package(&multiple_fns, None).unwrap_err(),
            RemoveAssertsPackageError::PackageHasMultipleFunctionsWithoutTop
        );
    }
}
