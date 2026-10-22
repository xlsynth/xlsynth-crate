// SPDX-License-Identifier: Apache-2.0

//! Generate raw and source-named Lean models from verified XLS block IR.
//!
//! [`parse`] checks the package with native XLS and the typed Rust parser.
//! [`emit`] preserves physical values, while [`emit_readable`] adds checked
//! source names, total codecs, and correspondence proof source. Source-map
//! validation checks metadata against the supplied inputs; it does not prove
//! that a compiler produced the block from that source.
//! Emission returns text; it does not invoke Lean or check the generated
//! proofs.

use anyhow::{Context, Result, bail, ensure};
use xlsynth::IrPackage;
use xlsynth_pir::{
    ir::{self, MemberType, PackageMember},
    ir_parser::Parser,
};

mod raw;
mod readable;
mod source_map;

pub use raw::{LeanBlockBundle, emit, emit_readable};
pub use source_map::{
    Attachment, EnumMember, SourceMap, SourceType, StructField, TypeDecl, ValidatedSourceMap,
    collect_source_map, validate_source_map,
};

/// One native-verified block, its typed graph, and package source locations.
///
/// The graph is immutable outside this crate so callers cannot bypass the
/// validation performed by [`parse`].
#[derive(Debug)]
pub struct Block {
    pub(crate) package_name: String,
    pub(crate) file_table: ir::FileTable,
    pub(crate) graph: ir::Block,
}

/// Parse and verify a block independently with native XLS and typed Rust IR.
///
/// `name` selects an explicit block. With no name, the declared top is
/// selected, or the sole block when there is no top. A function top, missing
/// block, or ambiguous selection returns an error. Parsing does not guarantee
/// that every operation belongs to the smaller Lean emission subset.
/// Use an explicit name when selecting a non-top block; otherwise a valid
/// package can select a different block than the caller intended.
pub fn parse(text: &str, name: Option<&str>) -> Result<Block> {
    IrPackage::parse_ir(text, None)
        .context("native XLS package parse")?
        .verify()
        .context("native XLS package verification")?;
    let package = Parser::new(text)
        .parse_and_validate_package()
        .context("typed PIR package parse/validation")?;
    let selected_name = match (name, &package.top) {
        (Some(name), _) => Some(name),
        (None, Some((name, MemberType::Block))) => Some(name.as_str()),
        (None, Some((_, MemberType::Function))) => {
            bail!("package top is a function; supply a block name")
        }
        (None, None) => None,
    };
    let mut blocks = package
        .members
        .into_iter()
        .filter_map(|member| match member {
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
    Ok(Block {
        package_name: package.name,
        file_table: package.file_table,
        graph,
    })
}
