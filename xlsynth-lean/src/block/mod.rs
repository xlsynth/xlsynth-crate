// SPDX-License-Identifier: Apache-2.0

//! Emit Lean from XLS block IR, or use the typechecked DSLX function signature
//! to emit a named, typed interface for the same equations.

use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;
use xlsynth_pir::ir::{self, MemberType, PackageMember};

mod raw;
mod readable;
mod source_types;
mod stateful;

pub use stateful::{CompiledStatefulFunction, compile_stateful_function, stateful_ir_to_block};

pub use raw::{LeanBlockBundle, emit};
use source_types::emit_stateful_readable;

/// The block and its package-level context used during emission.
pub(crate) struct BlockContext<'a> {
    pub(crate) package_name: &'a str,
    pub(crate) file_table: &'a ir::FileTable,
    pub(crate) graph: &'a ir::Block,
    pub(crate) helpers: Vec<&'a ir::Fn>,
}

impl<'a> BlockContext<'a> {
    pub(crate) fn new(package: &'a ir::Package, graph: &'a ir::Block) -> Result<Self> {
        ensure!(
            package
                .get_block(&graph.name)
                .is_some_and(|member| std::ptr::eq(member, graph)),
            "block {} does not belong to package {}",
            graph.name,
            package.name
        );
        Ok(Self {
            package_name: &package.name,
            file_table: &package.file_table,
            graph,
            helpers: reachable_helpers(package, graph)?,
        })
    }
}

/// Lean requires each helper to be defined before its first caller.
fn reachable_helpers<'a>(package: &'a ir::Package, block: &ir::Block) -> Result<Vec<&'a ir::Fn>> {
    fn visit<'a>(
        package: &'a ir::Package,
        graph: &ir::NodeGraph,
        visiting: &mut BTreeSet<&'a str>,
        emitted: &mut BTreeSet<&'a str>,
        result: &mut Vec<&'a ir::Fn>,
    ) -> Result<()> {
        for node in &graph.nodes {
            let ir::NodePayload::Invoke { to_apply, .. } = &node.payload else {
                continue;
            };
            if !emitted.contains(to_apply.as_str()) {
                let callee = package.get_fn(to_apply).with_context(|| {
                    format!("unknown function {to_apply} invoked by {}", graph.name)
                })?;
                ensure!(
                    visiting.insert(&callee.name),
                    "recursive invoke of {to_apply}"
                );
                visit(package, callee, visiting, emitted, result)?;
                visiting.remove(callee.name.as_str());
                emitted.insert(&callee.name);
                result.push(callee);
            }
        }
        Ok(())
    }
    let mut result = Vec::new();
    visit(
        package,
        block,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
        &mut result,
    )?;
    Ok(result)
}

/// Select a block from a typed package.
///
/// `name` selects an explicit block. With no name, the declared top is
/// selected, or the sole block when there is no top. A function top, missing
/// block, or ambiguous selection returns an error.
pub fn select_block<'a>(package: &'a ir::Package, name: Option<&str>) -> Result<&'a ir::Block> {
    let selected_name = match (name, &package.top) {
        (Some(name), _) => Some(name),
        (None, Some((name, MemberType::Block))) => Some(name.as_str()),
        (None, Some((_, MemberType::Function))) => {
            bail!("package top is a function; supply a block name")
        }
        (None, None) => None,
    };
    let mut blocks = package.members.iter().filter_map(|member| match member {
        PackageMember::Block(block) if selected_name.is_none_or(|name| name == block.name) => {
            Some(block)
        }
        _ => None,
    });
    let graph = blocks.next().with_context(|| {
        format!(
            "no matching block for {}",
            selected_name.unwrap_or("<sole block>")
        )
    })?;
    ensure!(
        blocks.next().is_none(),
        "multiple blocks; supply a block name"
    );
    Ok(graph)
}
