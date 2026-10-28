// SPDX-License-Identifier: Apache-2.0

//! Keep the source declarations and the block produced from them together.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use xlsynth::{IrPackage, dslx, dslx_path_to_module_name, mangle_dslx_name};
use xlsynth_pir::{ir_parser::Parser, stateful_fn_to_block::stateful_fn_to_block};

use super::{LeanBlockBundle, emit_stateful_readable, parse};

/// Lower a named IR function using its register suffix and outer return tuple.
/// Native verification covers the entire resulting package, including calls.
pub fn stateful_ir_to_block(ir: &str, top: &str) -> Result<String> {
    let package = Parser::new(ir).parse_and_validate_package()?;
    let converted = stateful_fn_to_block(package, top).map_err(anyhow::Error::msg)?;
    let text = converted.to_string();
    IrPackage::parse_ir(&text, None)?.verify()?;
    Ok(text)
}

/// A block and the exact typechecked DSLX module from which it was lowered.
pub struct CompiledStatefulFunction {
    block_ir: String,
    module: dslx::TypecheckedModule,
    source_top: String,
}

impl CompiledStatefulFunction {
    /// Complete package with the converted block marked top.
    pub fn block_ir(&self) -> &str {
        &self.block_ir
    }

    /// Emit both models, preserving the source types of the public interface.
    pub fn emit_readable(&self) -> Result<LeanBlockBundle> {
        let block = parse(&self.block_ir, None)?;
        emit_stateful_readable(&block, &self.module, &self.source_top)
    }
}

/// Typecheck once, lower the same module to function IR, and convert its top.
pub fn compile_stateful_function(
    source: &str,
    source_path: &Path,
    source_top: &str,
    stdlib: Option<&Path>,
    search_paths: &[&Path],
) -> Result<CompiledStatefulFunction> {
    let path = source_path.to_str().context("DSLX path is not UTF-8")?;
    ensure!(
        !source.contains('\0') && !path.contains('\0'),
        "DSLX source text and path cannot contain NUL bytes"
    );
    let module_name = dslx_path_to_module_name(source_path)?;
    let mut imports = dslx::ImportData::new(stdlib, search_paths);
    let module = dslx::parse_and_typecheck(source, path, module_name, &mut imports)?;
    let ir = module
        .convert_to_ir_text()
        .with_context(|| format!("lowering stateful function {source_top}"))?;
    let top = mangle_dslx_name(module_name, source_top)?;
    let block_ir = stateful_ir_to_block(&ir, &top)
        .with_context(|| format!("lowering stateful function {source_top}"))?;
    Ok(CompiledStatefulFunction {
        block_ir,
        module,
        source_top: source_top.to_owned(),
    })
}
