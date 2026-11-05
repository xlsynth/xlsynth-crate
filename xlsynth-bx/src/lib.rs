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
pub use parser::parse_module;
pub use xlsynth_pir::ir_builder::ResetBehavior;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    rc::Rc,
};
use xlsynth::dslx;
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

/// One DSLX compilation used for BX imports. The synthetic entry module calls
/// the selected DSLX specializations. Its type information owns their original
/// declarations, even if the source files change after BX compilation.
pub struct LinkedImport {
    pub module: dslx::TypecheckedModule,
    /// Each original IR function symbol and its actual destination symbol.
    pub symbols: BTreeMap<String, String>,
    /// Bindings for retained functions, in the destination package's symbol
    /// and node ID space. Populated by the loop-binding compilation entry.
    #[cfg(feature = "dslx-loop-bindings")]
    pub loop_bindings: Vec<dslx::LoopBinding>,
}

/// A written DSLX struct reference resolved by the same import graph as BX.
pub struct LinkedStruct {
    pub module: dslx::TypecheckedModule,
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
    /// Full module subjects from BX `use` statements and their transitive DSLX
    /// `import` statements, including unused imports.
    pub imported_modules: BTreeSet<String>,
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
    compile_with_imports(source, package, reset, options, |module| {
        let text = module.to_ir_text().map_err(|e| e.to_string())?;
        Ok((
            text,
            LinkedImport {
                module,
                symbols: BTreeMap::new(),
                #[cfg(feature = "dslx-loop-bindings")]
                loop_bindings: Vec::new(),
            },
        ))
    })
}

/// Compile with source declarations and the checked DSLX loop bindings
/// produced by the very same conversion that supplied each imported body.
#[cfg(feature = "dslx-loop-bindings")]
pub fn compile_with_source_and_loop_bindings(
    source: &str,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledModule, String> {
    compile_with_imports(source, package, reset, options, |module| {
        let (text, loop_bindings) = module
            .convert_to_ir_text_with_loop_bindings()
            .map_err(|e| e.to_string())?;
        Ok((
            text,
            LinkedImport {
                module,
                symbols: BTreeMap::new(),
                loop_bindings,
            },
        ))
    })
}

fn compile_with_imports(
    source: &str,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    convert: impl Fn(dslx::TypecheckedModule) -> Result<(String, LinkedImport), String>,
) -> Result<CompiledModule, String> {
    let mut modules = Vec::new();
    let mut imported_structs = BTreeMap::new();
    let mut imported_modules = BTreeSet::new();
    let source = parse_module(source)?;
    let checked = check::check_with_imports(
        &source,
        options,
        |m| {
            let (ir, import) = convert(m)?;
            modules.push(import);
            Ok(ir)
        },
        |s, modules| {
            imported_structs = s.into_sources();
            imported_modules = modules;
        },
    )?;
    let mut modules = modules.into_iter();
    let mut imports = Vec::new();
    let package = lower::lower_with_imports(&checked, package, reset, |linked, _node_id_base| {
        let mut import = modules.next().expect("one checked module per linked batch");
        import.symbols = linked
            .into_iter()
            .map(|(old, new)| (old, new.name))
            .collect();
        #[cfg(feature = "dslx-loop-bindings")]
        {
            // The converter also visits unretained functions. Only linked
            // functions have a destination in this package.
            import
                .loop_bindings
                .retain(|b| import.symbols.contains_key(&b.ir_function));
            for b in &mut import.loop_bindings {
                b.ir_function = import.symbols[&b.ir_function].clone();
                b.ir_node_id = usize::try_from(b.ir_node_id)
                    .ok()
                    .and_then(|id| id.checked_add(_node_id_base))
                    .and_then(|id| i64::try_from(id).ok())
                    .ok_or("imported loop binding node ID exceeds the available ID space")?;
            }
        }
        imports.push(import);
        Ok(())
    })?;
    Ok(CompiledModule {
        source,
        checked,
        package,
        imports,
        imported_structs,
        imported_modules,
    })
}
