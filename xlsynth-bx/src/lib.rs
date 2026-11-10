// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod ast;
mod builtin;
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
mod warnings;

pub use check::{CheckedModule, check_module, check_module_with_options};
pub use dslx_import::ImportedCall;
pub use lower::lower_module;
pub use parser::outline::{
    DslxOutline, OutlineDecl, OutlineFunctionParametric, OutlineKind, OutlineShape, OutlineType,
    parse_dslx_outline,
};
pub use parser::parse_module;
pub use typed::BoundParametricValue;
pub use warnings::{BxWarning, BxWarningKind};
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
/// supplied IR package. Warnings are treated as errors.
pub fn compile(source: &str, package: Package, reset: ResetBehavior) -> Result<Package, String> {
    compile_with_options(source, package, reset, &ImportOptions::default())
}

/// Parse a BX source file with its imported template signatures available to
/// distinguish calls with multiple parametrics from comma-separated
/// comparisons.
pub fn parse_module_with_options(
    source: &str,
    options: &ImportOptions<'_>,
) -> Result<ast::Module, String> {
    bx_import::parse_source(source, options)
}

/// Check BX source and its imports in one snapshot. The returned module
/// carries warnings alongside its checked declarations.
pub fn check_source_with_options(
    source: &str,
    options: &ImportOptions<'_>,
) -> Result<CheckedModule, String> {
    let (module, imports) = bx_import::read(source, options)?;
    check::check_sources(&module, imports, options).map_err(bx_import::diagnostic)
}

/// Compile BX source with the specified module search paths and treat
/// warnings as errors. Use [`check_source_with_options`] and [`lower_module`]
/// to inspect warnings and choose whether to continue.
pub fn compile_with_options(
    source: &str,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<Package, String> {
    let checked = check_source_with_options(source, options)?;
    // Keep link errors actionable even when the source also has warnings.
    let package = lower_module(&checked, package, reset)?;
    if !checked.warnings().is_empty() {
        let diagnostics = checked
            .warnings()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        return Err(format!(
            "{}\nBX warnings treated as errors",
            diagnostics.join("\n")
        ));
    }
    Ok(package)
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
