// SPDX-License-Identifier: Apache-2.0

//! Generate Lean models from verified XLS block IR.
//!
//! [`emit`] preserves physical values and clock-step behavior.
//! Emission returns text; it does not invoke Lean or check the generated
//! proofs.

use anyhow::{Context, Result, bail, ensure};
use xlsynth_pir::ir::{self, MemberType, PackageMember};

mod raw;
pub use raw::emit;

/// The block and its package-level context used during emission.
pub(crate) struct BlockContext<'a> {
    pub(crate) package_name: &'a str,
    pub(crate) file_table: &'a ir::FileTable,
    pub(crate) graph: &'a ir::Block,
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
        })
    }
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
