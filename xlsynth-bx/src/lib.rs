// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod ast;
pub mod check;
mod dslx_import;
mod import_link;
mod import_structs;
mod infer_literal;
mod layout;
pub mod lower;
pub mod parser;
mod typed;

pub use check::{CheckedModule, check_module, check_module_with_options};
pub use dslx_import::ImportedCall;
pub use lower::lower_module;
pub use parser::outline::{
    DslxOutline, OutlineDecl, OutlineKind, OutlineShape, OutlineType, parse_dslx_outline,
};
pub use parser::parse_module;
pub use xlsynth_pir::ir_builder::ResetBehavior;

use std::path::Path;
use xlsynth_pir::ir::{FileTable, Package};

/// Paths used by DSLX to resolve BX module imports and their dependencies.
#[derive(Clone, Debug, Default)]
pub struct ImportOptions<'a> {
    /// The bundled DSLX standard library is used by default.
    pub dslx_stdlib_path: Option<&'a Path>,
    /// Additional DSLX search directories, in lookup order.
    pub additional_search_paths: Vec<&'a Path>,
}

/// Parses and checks BX source, then links its blocks into the supplied IR
/// package.
pub fn compile(source: &str, package: Package, reset: ResetBehavior) -> Result<Package, String> {
    compile_with_options(source, package, reset, &ImportOptions::default())
}

/// Compiles BX source with the specified DSLX module search paths.
pub fn compile_with_options(
    source: &str,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<Package, String> {
    let module = parse_module(source)?;
    let checked = check_module_with_options(&module, options)?;
    lower_module(&checked, package, reset)
}

/// Creates a destination for BX modules whose externs require no linked IR.
pub fn empty_package(name: impl Into<String>) -> Package {
    Package {
        name: name.into(),
        file_table: FileTable::new(),
        members: Vec::new(),
        top: None,
    }
}
