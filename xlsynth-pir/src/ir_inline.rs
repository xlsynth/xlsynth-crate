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
use crate::ir_utils::direct_function_references;
use crate::ir_verify;
use std::collections::{BTreeSet, HashMap, HashSet};
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
/// selected top function via `invoke`, `map`, or `counted_for`.
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
            "ir_inline: toolchain backend did not eliminate invoke/map nodes from top '{top_name}': {targets}"
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

    let reachable_pkg = package_with_reachable_callees(&output_pkg, output_top)?;
    ir_verify::verify_package(&reachable_pkg)
        .map_err(|e| format!("ir_inline: retained package validation failed: {e}"))?;
    Ok(reachable_pkg.to_string())
}

fn build_opt_pass_pipeline(options: IrInlineOptions) -> String {
    let mut passes = Vec::new();
    if options.unroll {
        passes.push("loop_unroll");
    }
    passes.push("map_inlining");
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
        if let NodePayload::Invoke { to_apply, .. } | NodePayload::Map { to_apply, .. } =
            &node.payload
        {
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

/// Builds a package for this exact entry function, resolving only its callees
/// from `pkg`. The entry may be an edited copy of a function in the package.
pub fn package_with_reachable_callees(
    pkg: &Package,
    entry: &crate::ir::Fn,
) -> Result<Package, String> {
    let ordered = ordered_functions(pkg, &[&entry.name], Some(entry))?;
    let callees: HashSet<_> = ordered.iter().map(|f| f.name.as_str()).collect();
    // Preserve package order for this API and place the supplied entry last.
    let mut members: Vec<_> = pkg
        .members
        .iter()
        .filter_map(|m| match m {
            PackageMember::Function(f)
                if f.name != entry.name && callees.contains(f.name.as_str()) =>
            {
                Some(m.clone())
            }
            _ => None,
        })
        .collect();
    members.push(PackageMember::Function(entry.clone()));
    Ok(Package {
        name: pkg.name.clone(),
        file_table: pkg.file_table.clone(),
        members,
        top: Some((entry.name.clone(), MemberType::Function)),
    })
}

/// Collects each root and its transitive callees once, callees before callers.
/// Visits roots in the supplied order and sibling callees in name order.
/// Includes loop bodies and reports missing functions or recursive calls.
pub fn reachable_functions_in_dependency_order<'a>(
    pkg: &'a Package,
    roots: &[&str],
) -> Result<Vec<&'a crate::ir::Fn>, String> {
    ordered_functions(pkg, roots, None)
}

/// A standalone entry can override its package definition or be absent from it.
fn ordered_functions<'a>(
    pkg: &'a Package,
    roots: &[&str],
    entry: Option<&'a crate::ir::Fn>,
) -> Result<Vec<&'a crate::ir::Fn>, String> {
    let mut functions = HashMap::new();
    for member in &pkg.members {
        if let PackageMember::Function(f) = member
            && functions.insert(f.name.as_str(), f).is_some()
        {
            return Err(format!("ambiguous function name {:?} in package", f.name));
        }
    }
    if let Some(entry) = entry {
        functions.insert(&entry.name, entry);
    }
    let lookup = |name: &str| {
        functions
            .get(name)
            .copied()
            .ok_or_else(|| format!("reachable function '{name}' is missing from package"))
    };

    enum Visit<'a> {
        Enter(&'a crate::ir::Fn),
        Emit(&'a crate::ir::Fn),
    }
    let mut work = Vec::new();
    for root in roots.iter().rev() {
        work.push(Visit::Enter(lookup(root)?));
    }
    let mut active = HashSet::new();
    let mut finished = HashSet::new();
    let mut ordered = Vec::new();
    while let Some(visit) = work.pop() {
        match visit {
            Visit::Enter(f) => {
                if finished.contains(f.name.as_str()) {
                    continue;
                }
                if !active.insert(f.name.as_str()) {
                    return Err(format!("recursive function reference to '{}'", f.name));
                }
                work.push(Visit::Emit(f));
                for callee in direct_function_references(f).into_iter().rev() {
                    work.push(Visit::Enter(lookup(callee)?));
                }
            }
            Visit::Emit(f) => {
                active.remove(f.name.as_str());
                finished.insert(f.name.as_str());
                ordered.push(f);
            }
        }
    }
    Ok(ordered)
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
            "loop_unroll map_inlining inlining dce"
        );
        assert_eq!(
            build_opt_pass_pipeline(IrInlineOptions { unroll: false }),
            "map_inlining inlining dce"
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
    fn dependency_order_follows_loop_bodies_and_deduplicates_roots() {
        let pkg = Parser::new(
            r#"package p
top fn caller(x: bits[8] id=1) -> bits[8] {
  ret loop: bits[8] = counted_for(x, trip_count=3, stride=1, body=body, id=2)
}
fn body(i: bits[8] id=3, x: bits[8] id=4) -> bits[8] {
  ret sum: bits[8] = add(i, x, id=5)
}"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let ordered =
            reachable_functions_in_dependency_order(&pkg, &["caller", "body", "caller"]).unwrap();
        assert_eq!(
            ordered.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
            ["body", "caller"]
        );
    }

    #[test]
    fn dependency_order_rejects_missing_callees_and_cycles() {
        for (target, error) in [
            (
                "absent",
                "reachable function 'absent' is missing from package",
            ),
            ("child", "recursive function reference to 'child'"),
            ("entry", "recursive function reference to 'entry'"),
        ] {
            let ir = format!(
                r#"package p
top fn entry(x: bits[8] id=1) -> bits[8] {{
  ret y: bits[8] = invoke(x, to_apply=child, id=2)
}}
fn child(x: bits[8] id=3) -> bits[8] {{
  ret y: bits[8] = invoke(x, to_apply={target}, id=4)
}}"#
            );
            let pkg = Parser::new(&ir).parse_package().unwrap();
            assert_eq!(
                reachable_functions_in_dependency_order(&pkg, &["entry"]).unwrap_err(),
                error
            );
        }
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
