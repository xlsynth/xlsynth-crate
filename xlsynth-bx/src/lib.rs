// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod ast;
mod bx_import;
pub mod check;
mod dslx_import;
mod effect_diagnostic;
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

/// Search paths for BX and DSLX imports and their dependencies.
#[derive(Clone, Debug, Default)]
pub struct ImportOptions<'a> {
    /// The bundled DSLX standard library is used by default.
    pub dslx_stdlib_path: Option<&'a Path>,
    /// Additional module search directories, in lookup order.
    pub additional_search_paths: Vec<&'a Path>,
}

/// Parses and checks BX source, then links its functions and blocks into the
/// supplied IR package.
pub fn compile(source: &str, package: Package, reset: ResetBehavior) -> Result<Package, String> {
    compile_with_options(source, package, reset, &ImportOptions::default())
}

/// Parse a BX source file with its imported template signatures available to
/// distinguish multi-callback calls from comma-separated comparisons.
pub fn parse_module_with_options(
    source: &str,
    options: &ImportOptions<'_>,
) -> Result<ast::Module, String> {
    bx_import::parse_source(source, options)
}

/// Compiles BX source with the specified BX and DSLX module search paths.
pub fn compile_with_options(
    source: &str,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<Package, String> {
    let (module, libraries) = bx_import::read(source, options)?;
    let checked = check::check_sources(&module, libraries.as_ref(), options)
        .map_err(bx_import::diagnostic)?;
    lower_module(&checked, package, reset).map_err(bx_import::diagnostic)
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
