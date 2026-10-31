// SPDX-License-Identifier: Apache-2.0

//! Keep the source declarations and the block produced from them together.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use xlsynth::{IrPackage, dslx, dslx_path_to_module_name, mangle_dslx_name};
use xlsynth_pir::{
    dce::remove_dead_nodes,
    ir,
    ir_parser::Parser,
    ir_utils::{get_topological, remap_payload_with},
    stateful_fn_to_block::stateful_fn_to_block,
};

use super::{BlockContext, LeanBlockBundle, emit_stateful_readable, select_block};

/// Lower a named IR function using its register suffix and outer return tuple.
/// Native verification covers the entire resulting package, including calls.
pub fn stateful_ir_to_block(ir: &str, top: &str) -> Result<String> {
    verified_stateful_block(ir, top).map(|(_, text)| text)
}

fn verified_stateful_block(ir: &str, top: &str) -> Result<(ir::Package, String)> {
    let mut package = Parser::new(ir).parse_and_validate_package()?;
    for member in &mut package.members {
        if let ir::PackageMember::Function(function) = member {
            clean_model_function(function);
        }
    }
    let converted = stateful_fn_to_block(package, top).map_err(anyhow::Error::msg)?;
    let text = converted.to_string();
    IrPackage::parse_ir(&text, None)?.verify()?;
    Ok((converted, text))
}

/// The model pipeline only forwards identity nodes and removes dead nodes.
/// Keep live operations and function calls intact for readers of the model.
fn clean_model_function(function: &mut ir::Fn) {
    let mut forwarded: Vec<_> = (0..function.nodes.len())
        .map(|index| ir::NodeRef { index })
        .collect();
    for reference in get_topological(function) {
        let node = &mut function.nodes[reference.index];
        if let ir::NodePayload::Unop(ir::Unop::Identity, operand) = node.payload {
            forwarded[reference.index] = forwarded[operand.index];
        } else {
            node.payload =
                remap_payload_with(&node.payload, |(_, operand)| forwarded[operand.index]);
        }
    }
    function.ret_node_ref = function.ret_node_ref.map(|node| forwarded[node.index]);
    *function = remove_dead_nodes(function);
}

/// A block and the exact typechecked DSLX module from which it was lowered.
pub struct CompiledStatefulFunction {
    package: ir::Package,
    block_ir: String,
    module: dslx::TypecheckedModule,
    loop_bindings: Vec<dslx::LoopBinding>,
    source_top: String,
}

impl CompiledStatefulFunction {
    /// Complete package with the converted block marked top.
    pub fn block_ir(&self) -> &str {
        &self.block_ir
    }

    /// Emit both models, preserving the source types of the public interface.
    pub fn emit_readable(&self) -> Result<LeanBlockBundle> {
        let block = BlockContext::new(&self.package, select_block(&self.package, None)?)?;
        emit_stateful_readable(&block, &self.module, &self.source_top, &self.loop_bindings)
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
    let (ir, loop_bindings) = module
        .convert_to_ir_text_with_loop_bindings()
        .with_context(|| format!("lowering stateful function {source_top}"))?;
    let top = mangle_dslx_name(module_name, source_top)?;
    let (package, block_ir) = verified_stateful_block(&ir, &top)
        .with_context(|| format!("lowering stateful function {source_top}"))?;
    Ok(CompiledStatefulFunction {
        package,
        block_ir,
        module,
        loop_bindings,
        source_top: source_top.to_owned(),
    })
}
