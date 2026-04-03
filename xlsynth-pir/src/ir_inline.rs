// SPDX-License-Identifier: Apache-2.0

//! Selective IR desugaring focused on call flattening at the chosen top
//! function.
//!
//! The initial implementation uses the external XLS toolchain's `opt_main`
//! with an explicit pass list. The API is kept backend-oriented so a future
//! runtime-backed implementation can preserve the same CLI contract and result
//! validation.

use crate::ir::{MemberType, NodePayload, Package, PackageMember};
use crate::ir_parser::Parser;
use crate::ir_validate;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrInlineBackend {
    Toolchain(PathBuf),
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrInlineOptions {
    pub unroll: bool,
}

impl Default for IrInlineOptions {
    fn default() -> Self {
        Self { unroll: true }
    }
}

/// Runs the `ir-inline` transform over IR text and returns the rewritten IR.
///
/// The result is validated and reduced to the members reachable from the
/// selected top function via `invoke` / `counted_for`.
pub fn run_ir_inline_over_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: IrInlineOptions,
    backend: IrInlineBackend,
) -> Result<String, String> {
    let resolved_top = resolve_top_name(ir_text, top)?;
    match backend {
        IrInlineBackend::Toolchain(tool_dir) => {
            run_ir_inline_over_ir_text_via_toolchain(ir_text, &resolved_top, options, &tool_dir)
        }
        IrInlineBackend::Runtime => {
            Err("ir_inline: runtime backend is not implemented yet".to_string())
        }
    }
}

fn resolve_top_name(ir_text: &str, top: Option<&str>) -> Result<String, String> {
    if let Some(name) = top {
        return Ok(name.to_string());
    }

    let mut parser = Parser::new(ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .map_err(|e| format!("ir_inline: PIR parse/validate failed: {e}"))?;
    match &pkg.top {
        Some((name, MemberType::Function)) => Ok(name.clone()),
        Some((name, MemberType::Block)) => Err(format!(
            "ir_inline: input package top member '{name}' is a block; pass --top to select a function"
        )),
        None => {
            let function_names = pkg
                .members
                .iter()
                .filter_map(|member| match member {
                    PackageMember::Function(f) => Some(f.name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            match function_names.as_slice() {
                [name] => Ok(name.clone()),
                [] => Err("ir_inline: input package has no functions".to_string()),
                _ => Err(
                    "ir_inline: input package has multiple functions and no explicit top function; pass --top"
                        .to_string(),
                ),
            }
        }
    }
}

fn run_ir_inline_over_ir_text_via_toolchain(
    ir_text: &str,
    top_name: &str,
    options: IrInlineOptions,
    tool_dir: &Path,
) -> Result<String, String> {
    let output_text = run_opt_main_with_passes(tool_dir, ir_text, top_name, options)?;

    let mut parser = Parser::new(&output_text);
    let output_pkg = parser.parse_and_validate_package().map_err(|e| {
        format!("ir_inline: PIR parse/validate failed after toolchain pass pipeline: {e}")
    })?;
    let output_top = output_pkg.get_fn(top_name).ok_or_else(|| {
        format!("ir_inline: toolchain output package is missing top function '{top_name}'")
    })?;

    let residual_invokes = residual_invoke_targets(output_top);
    if !residual_invokes.is_empty() {
        let targets = residual_invokes.into_iter().collect::<Vec<_>>().join(", ");
        return Err(format!(
            "ir_inline: toolchain backend did not eliminate invoke nodes from top '{top_name}': {targets}"
        ));
    }

    let residual_counted_for = residual_counted_for_targets(output_top);
    if options.unroll && !residual_counted_for.is_empty() {
        let targets = residual_counted_for
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "ir_inline: toolchain backend did not eliminate counted_for nodes from top '{top_name}': {targets}"
        ));
    }

    let reachable_pkg = retain_reachable_functions(&output_pkg, top_name)?;
    ir_validate::validate_package(&reachable_pkg)
        .map_err(|e| format!("ir_inline: retained package validation failed: {e}"))?;
    Ok(reachable_pkg.to_string())
}

fn build_opt_pass_pipeline(options: IrInlineOptions) -> String {
    let mut passes = Vec::new();
    if options.unroll {
        passes.push("loop_unroll");
    }
    passes.push("inlining");
    passes.push("dce");
    passes.join(" ")
}

fn run_opt_main_with_passes(
    tool_dir: &Path,
    ir_text: &str,
    top_name: &str,
    options: IrInlineOptions,
) -> Result<String, String> {
    let opt_main = tool_dir.join("opt_main");
    if !opt_main.exists() {
        return Err(format!(
            "ir_inline: opt_main not found in {}",
            tool_dir.display()
        ));
    }

    let tmp = tempfile::NamedTempFile::new()
        .map_err(|e| format!("ir_inline: failed to create temp file for opt_main: {e}"))?;
    std::fs::write(tmp.path(), ir_text)
        .map_err(|e| format!("ir_inline: failed to write temp IR file for opt_main: {e}"))?;

    let pass_pipeline = build_opt_pass_pipeline(options);
    let mut command = Command::new(&opt_main);
    command
        .arg(tmp.path())
        .arg("--top")
        .arg(top_name)
        .arg("--passes")
        .arg(&pass_pipeline);
    let output = command
        .output()
        .map_err(|e| format!("ir_inline: failed to spawn opt_main: {e}"))?;
    if !output.status.success() {
        let mut message = format!(
            "ir_inline: opt_main failed with status {} while running passes `{}`",
            output.status, pass_pipeline
        );
        if !output.stderr.is_empty() {
            message.push_str(": ");
            message.push_str(&String::from_utf8_lossy(&output.stderr));
        } else if !output.stdout.is_empty() {
            message.push_str(": ");
            message.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        return Err(message);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn residual_invoke_targets(f: &crate::ir::Fn) -> BTreeSet<String> {
    let mut targets = BTreeSet::new();
    for node in &f.nodes {
        if let NodePayload::Invoke { to_apply, .. } = &node.payload {
            targets.insert(to_apply.clone());
        }
    }
    targets
}

fn residual_counted_for_targets(f: &crate::ir::Fn) -> BTreeSet<String> {
    let mut targets = BTreeSet::new();
    for node in &f.nodes {
        if let NodePayload::CountedFor { body, .. } = &node.payload {
            targets.insert(body.clone());
        }
    }
    targets
}

fn direct_callee_names(f: &crate::ir::Fn) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for node in &f.nodes {
        match &node.payload {
            NodePayload::Invoke { to_apply, .. } => {
                names.insert(to_apply.clone());
            }
            NodePayload::CountedFor { body, .. } => {
                names.insert(body.clone());
            }
            _ => {}
        }
    }
    names
}

fn retain_reachable_functions(pkg: &Package, top_name: &str) -> Result<Package, String> {
    let mut reachable = BTreeSet::new();
    let mut worklist = vec![top_name.to_string()];
    while let Some(name) = worklist.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        let f = pkg.get_fn(&name).ok_or_else(|| {
            format!("ir_inline: reachable function '{name}' is missing from output package")
        })?;
        for callee in direct_callee_names(f) {
            if !reachable.contains(&callee) {
                worklist.push(callee);
            }
        }
    }

    let members = pkg
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Function(f) if reachable.contains(&f.name) => Some(member.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if members.is_empty() {
        return Err(format!(
            "ir_inline: no reachable function members found for top '{top_name}'"
        ));
    }

    Ok(Package {
        name: pkg.name.clone(),
        file_table: pkg.file_table.clone(),
        members,
        top: Some((top_name.to_string(), MemberType::Function)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_top_fn(ir_text: &str) -> crate::ir::Fn {
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        pkg.get_top_fn().unwrap().clone()
    }

    #[test]
    fn build_opt_pass_pipeline_matches_unroll_flag() {
        assert_eq!(
            build_opt_pass_pipeline(IrInlineOptions { unroll: true }),
            "loop_unroll inlining dce"
        );
        assert_eq!(
            build_opt_pass_pipeline(IrInlineOptions { unroll: false }),
            "inlining dce"
        );
    }

    #[test]
    fn residual_target_detection_distinguishes_invoke_and_counted_for() {
        let f = parse_top_fn(
            r#"package p

fn callee(x: bits[8] id=10) -> bits[8] {
  ret add.2: bits[8] = add(x, x, id=2)
}

fn body(i: bits[8] id=20, x: bits[8] id=21) -> bits[8] {
  ret add.4: bits[8] = add(i, x, id=4)
}

top fn main(x: bits[8] id=30) -> bits[8] {
  invoke.5: bits[8] = invoke(x, to_apply=callee, id=5)
  ret counted_for.6: bits[8] = counted_for(invoke.5, trip_count=3, stride=1, body=body, id=6)
}"#,
        );
        assert_eq!(
            residual_invoke_targets(&f),
            BTreeSet::from(["callee".to_string()])
        );
        assert_eq!(
            residual_counted_for_targets(&f),
            BTreeSet::from(["body".to_string()])
        );
    }

    #[test]
    fn resolve_top_name_uses_sole_function_when_package_has_no_top() {
        let ir_text = r#"package p

fn main(x: bits[8] id=1) -> bits[8] {
  ret not.2: bits[8] = not(x, id=2)
}"#;
        assert_eq!(resolve_top_name(ir_text, None).unwrap(), "main");
    }

    #[test]
    fn resolve_top_name_requires_explicit_top_when_package_has_multiple_functions_and_no_top() {
        let ir_text = r#"package p

fn helper(x: bits[8] id=1) -> bits[8] {
  ret not.2: bits[8] = not(x, id=2)
}

fn main(x: bits[8] id=10) -> bits[8] {
  ret add.11: bits[8] = add(x, x, id=11)
}"#;
        let err = resolve_top_name(ir_text, None).unwrap_err();
        assert!(
            err.contains("multiple functions and no explicit top function"),
            "err: {}",
            err
        );
    }
}
