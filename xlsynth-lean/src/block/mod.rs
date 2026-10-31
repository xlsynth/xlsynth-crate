// SPDX-License-Identifier: Apache-2.0

//! Emit Lean from XLS block IR or a checked BX block with a named interface.

use anyhow::{Context, Result, bail, ensure};
use std::borrow::Cow;
use std::collections::BTreeSet;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_graph;
use xlsynth_pir::ir::{self, MemberType, PackageMember};

mod bx;
mod helper_types;
mod library_notation;
mod raw;
mod readable;
mod source_types;
#[cfg(feature = "dslx-loop-bindings")]
pub use bx::compile_bx_with_dslx_loop_bindings;
pub use bx::{CompiledBx, compile_bx, compile_bx_with_dslx};

pub use raw::{LeanBlockBundle, emit, write_to};
pub(crate) use raw::{identifier, source_record_value};

/// The block and its package-level context used during emission.
pub(crate) struct BlockContext<'a> {
    pub(crate) package_name: &'a str,
    pub(crate) file_table: &'a ir::FileTable,
    pub(crate) graph: Cow<'a, ir::Block>,
    source_block: &'a ir::Block,
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
        let normalized = if graph
            .nodes
            .iter()
            .any(|node| node.payload.is_extension_op())
        {
            let mut lowered = graph.clone();
            desugar_extensions_in_graph(&mut lowered.graph)
                .with_context(|| format!("lowering extensions in block {}", graph.name))?;
            Cow::Owned(lowered)
        } else {
            Cow::Borrowed(graph)
        };
        Ok(Self {
            package_name: &package.name,
            file_table: &package.file_table,
            graph: normalized,
            source_block: graph,
            helpers: reachable_helpers(package, graph)?,
        })
    }

    /// Original nodes keep their indices during lowering. Appended helper
    /// nodes have fresh text IDs and belong only to the lowered graph.
    fn source_graph(&self, reference: ir::NodeRef) -> &ir::NodeGraph {
        if reference.index < self.source_block.nodes.len() {
            &self.source_block.graph
        } else {
            &self.graph.graph
        }
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
            let to_apply = match &node.payload {
                ir::NodePayload::Invoke { to_apply, .. } => to_apply,
                ir::NodePayload::CountedFor { body, .. } => body,
                _ => continue,
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
