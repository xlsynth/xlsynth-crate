// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod ast;
mod builtin;
mod bx_import;
pub mod check;
mod cycle_diagnostic;
mod dslx_import;
mod effect_diagnostic;
mod import_link;
mod import_structs;
mod infer_literal;
pub mod invariant_prover;
mod layout;
pub mod lower;
pub mod parser;
mod test_runner;
mod typed;
mod warnings;

pub use check::{CheckedModule, check_module, check_module_with_options};
pub use dslx_import::ImportedCall;
pub use import_structs::ImportNames;
pub use lower::lower_module;
pub use parser::outline::{
    DslxOutline, OutlineDecl, OutlineFunctionParametric, OutlineKind, OutlineShape, OutlineType,
    parse_dslx_outline,
};
pub use parser::parse_module;
pub(crate) use test_runner::invariant::BxInvariantObligation;
pub use test_runner::invariant::{
    BxInvariantCase, BxInvariantProof, BxInvariantSuite, prepare_invariant_proofs,
};
pub use test_runner::{
    BxProofCase, BxProofSuite, BxQuickCheckProof, BxTestOptions, BxTestReport, BxTestResult,
    BxTestRun, FourValuedReport, QuickCheckOptions, QuickCheckReport, prepare_quickcheck_proofs,
    run_tests, run_tests_with_options, run_tests_with_quickcheck,
};
pub use typed::BoundParametricValue;
pub use warnings::{BxWarning, BxWarningKind};
pub use xlsynth_pir::ir_builder::ResetBehavior;

use std::{collections::BTreeMap, path::Path, rc::Rc};
use xlsynth::dslx;
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
/// carries warnings alongside its checked declarations. Lowering reports
/// combinational cycles and timing errors deferred by the wire checker.
pub fn check_source_with_options(
    source: &str,
    options: &ImportOptions<'_>,
) -> Result<CheckedModule, String> {
    let (module, imports) = bx_import::read(source, options)?;
    check::check_sources(&module, imports, options).map_err(bx_import::diagnostic)
}

/// Check BX source and its imports, retaining the root path for diagnostics.
pub fn check_source_file_with_options(
    source: &str,
    path: &Path,
    options: &ImportOptions<'_>,
) -> Result<CheckedModule, String> {
    let mut checked = check_source_with_options(source, options)?;
    for block in &mut checked.blocks {
        if let Some(source) = &mut block.source
            && source.path.is_none()
        {
            source.path = Some(path.to_owned());
        }
    }
    Ok(checked)
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

/// One DSLX compilation used for BX imports. The synthetic entry module calls
/// the selected DSLX specializations. Its type information owns their original
/// declarations, even if the source files change after BX compilation.
pub struct LinkedImport {
    pub module: dslx::TypecheckedModule,
    pub module_names: ImportNames,
    /// Each original IR function symbol and its actual destination symbol.
    pub symbols: BTreeMap<String, String>,
}

/// A written DSLX struct reference resolved by the same import graph as BX.
pub struct LinkedStruct {
    pub module: dslx::TypecheckedModule,
    pub module_names: ImportNames,
    pub annotation: dslx::TypeAnnotation,
}

/// The checked source and its linked hardware, with the import graphs that
/// produced its callable IR functions.
pub struct CompiledModule {
    /// Written annotations retain transparent aliases erased by BX checking.
    pub source: ast::Module,
    pub checked: CheckedModule,
    pub package: Package,
    pub imports: Vec<LinkedImport>,
    /// Indexed by the spelling used in BX, including unqualified use bindings.
    pub imported_structs: BTreeMap<String, Rc<LinkedStruct>>,
}

/// Compile BX while retaining the source graph for each linked DSLX import.
pub fn compile_with_source(
    source: &str,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledModule, String> {
    let mut modules = Vec::new();
    let mut sources = import_structs::CapturedSources::default();
    let (source, libraries) = bx_import::read(source, options)?;
    let checked = check::check_sources_with_tests(
        &source,
        libraries,
        options,
        None,
        |m| modules.push(m),
        Some(&mut sources),
    )
    .map_err(bx_import::diagnostic)?;
    let mut modules = modules.into_iter();
    let mut imports = Vec::new();
    let package = lower::lower_with_imports(&checked, package, reset, |linked| {
        let module = modules.next().expect("one checked module per linked batch");
        let symbols: BTreeMap<_, _> = linked
            .into_iter()
            .map(|(old, new)| (old, new.name))
            .collect();
        imports.push(LinkedImport {
            module,
            symbols,
            module_names: sources
                .names
                .clone()
                .expect("linked DSLX batch has a snapshot"),
        });
    })?;
    Ok(CompiledModule {
        source,
        checked,
        package,
        imports,
        imported_structs: sources.types,
    })
}
