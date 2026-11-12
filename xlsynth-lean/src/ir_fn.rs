// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::ir_expr::{check_supported, emit_pure_expr, lean_type, width};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir::{
    Fn as IrFunction, FunctionType, NodePayload as P, NodeRef, Package, PackageMember, Type,
};
use xlsynth_pir::ir_inline::package_with_reachable_functions;
use xlsynth_pir::ir_utils::{compact_and_toposort_in_place, external_function_references};
use xlsynth_pir::ir_verify::verify_package;

use crate::support::{IR_FUNCTION_SUPPORT, write_file};
use crate::{LeanError, LeanModuleName};

const MODULE_HEADER: &str = "-- SPDX-License-Identifier: Apache-2.0\nimport XlsynthLean.IR\n\n";

/// An executable word-level Lean module and the XLS signature of its public
/// `eval`.
#[derive(Debug)]
pub struct LeanFunction {
    module_name: LeanModuleName,
    signature: FunctionType,
    source: String,
}

impl LeanFunction {
    pub fn lean_source(&self) -> &str {
        &self.source
    }

    /// Declarations that can be nested in a batch driver after its imports.
    pub(crate) fn declarations(&self) -> &str {
        &self.source[MODULE_HEADER.len()..]
    }

    pub fn module_name(&self) -> &LeanModuleName {
        &self.module_name
    }

    pub fn signature(&self) -> &FunctionType {
        &self.signature
    }

    /// Writes the importable module, its Lean support source, and the toolchain
    /// pin.
    pub fn write_to(&self, root: &Path) -> Result<PathBuf, LeanError> {
        IR_FUNCTION_SUPPORT.write_to(root)?;
        self.write_module(root)
    }

    pub(crate) fn write_module(&self, root: &Path) -> Result<PathBuf, LeanError> {
        let path = self.module_name.source_path();
        write_file(root, &path, self.source.as_bytes())?;
        Ok(root.join(path))
    }
}

/// Converts a selected pure function and its callees to executable Lean
/// definitions. Every array dimension in these functions must be nonempty.
pub fn export_ir_function(
    package: &Package,
    top: Option<&str>,
    module_name: &LeanModuleName,
) -> Result<LeanFunction, LeanError> {
    export_with_bindings(package, top, module_name, Bindings::LocalLets)
}

/// Named nodes give proof authors a stable, independently unfoldable
/// declaration per SSA binding. Local lets share intermediate values during
/// execution.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bindings {
    LocalLets,
    NamedNodes,
}

pub(crate) fn export_with_bindings(
    package: &Package,
    top: Option<&str>,
    module_name: &LeanModuleName,
    bindings: Bindings,
) -> Result<LeanFunction, LeanError> {
    let entry = match top {
        Some(name) => package.get_named_or_top_or_sole_fn(name),
        None => package.get_top_or_sole_fn(),
    }
    .map_err(LeanError::InvalidIr)?;
    let ns = module_name.lean_identifier();
    let mut source =
        format!("{MODULE_HEADER}namespace {ns}\nset_option linter.unusedVariables false\n\n");
    source.push_str(&emit_function_defs(
        package,
        &[&entry.name],
        bindings,
        &|name| Ok(function_name(name)),
    )?);
    let signature = entry.get_type();
    let mut input_width = 0usize;
    let mut call = "eval".to_owned();
    for ty in &signature.param_types {
        write!(
            call,
            " (_root_.XlsynthLean.IR.Packed.unpack (inputs.toNat >>> {input_width}))"
        )
        .unwrap();
        input_width = input_width
            .checked_add(width(ty)?)
            .ok_or_else(|| LeanError::InvalidIr("combined input width overflow".into()))?;
    }
    let result_width = width(&signature.return_type)?;
    writeln!(source, "abbrev eval := {}\n", function_name(&entry.name)).unwrap();
    writeln!(source, "def evalPacked (inputs : BitVec {input_width}) : BitVec {result_width} :=\n  BitVec.ofNat _ (_root_.XlsynthLean.IR.Packed.pack ({call}))\n\nend {ns}").unwrap();
    Ok(LeanFunction {
        module_name: module_name.clone(),
        signature,
        source,
    })
}

/// Emits definitions for the union of the roots and their callees, in
/// dependency order. The caller supplies the enclosing namespace and the
/// XlsynthLean.IR support definitions. Use the same unambiguous Lean name for
/// each function's declaration and all references to it.
pub(crate) fn emit_function_defs(
    package: &Package,
    roots: &[&str],
    bindings: Bindings,
    callee_name: &dyn Fn(&str) -> Result<String, LeanError>,
) -> Result<String, LeanError> {
    let mut reachable =
        package_with_reachable_functions(package, roots).map_err(LeanError::InvalidIr)?;
    for member in &mut reachable.members {
        if let PackageMember::Function(f) = member {
            for node in &f.nodes {
                check_supported(f, node)?;
            }
            compact_and_toposort_in_place(f).map_err(LeanError::InvalidIr)?;
        }
    }
    verify_package(&reachable).map_err(|e| LeanError::InvalidIr(e.to_string()))?;
    for member in &mut reachable.members {
        if let PackageMember::Function(f) = member
            && f.nodes.iter().any(|n| n.payload.is_extension_op())
        {
            desugar_extensions_in_fn(f).map_err(|e| LeanError::InvalidIr(e.to_string()))?;
        }
    }

    // Emit callees in postorder. Package verification has ruled out recursion.
    let mut work: Vec<_> = roots
        .iter()
        .rev()
        .map(|&name| (name.to_owned(), false))
        .collect();
    let mut source = String::new();
    let mut emitted = HashSet::new();
    let functions: HashMap<_, _> = reachable
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Function(f) => Some((f.name.as_str(), f)),
            PackageMember::Block(_) => None,
        })
        .collect();
    // Typeclass search needs an instance per product constructor, field, and
    // terminal Unit.
    let instance_size = functions
        .values()
        .flat_map(|f| &f.nodes)
        .map(|n| packed_instance_size(&n.ty))
        .max()
        .unwrap_or(1);
    // Each let nests the remainder of the body. The default elaborator
    // recursion limit is 512, which even a modest word-level data path can
    // exceed.
    let depth = functions
        .values()
        .map(|f| f.nodes.len())
        .max()
        .unwrap_or(0)
        .max(instance_size);
    if depth >= 256 {
        writeln!(
            source,
            "set_option maxRecDepth {}\n",
            depth.saturating_mul(4).saturating_add(512)
        )
        .unwrap();
    }
    if instance_size >= 128 {
        writeln!(
            source,
            "set_option synthInstance.maxSize {}\n",
            instance_size.saturating_mul(2)
        )
        .unwrap();
    }
    while let Some((name, ready)) = work.pop() {
        if emitted.contains(&name) {
            continue;
        }
        let f = functions[name.as_str()];
        if ready {
            emit_function(&mut source, f, bindings, callee_name)?;
            emitted.insert(name);
        } else {
            work.push((name, true));
            work.extend(
                external_function_references(f)
                    .into_iter()
                    .rev()
                    .map(|n| (n, false)),
            );
        }
    }
    Ok(source)
}

fn packed_instance_size(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 1,
        Type::Tuple(fields) => {
            1 + fields
                .iter()
                .map(|t| 1 + packed_instance_size(t))
                .sum::<usize>()
        }
        Type::Array(a) => 1 + packed_instance_size(&a.element_type),
    }
}

/// Uses an injective ASCII spelling even for names that contain Lean syntax.
fn function_name(name: &str) -> String {
    let mut result = "ir_".to_owned();
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() {
            result.push(char::from(b));
        } else {
            write!(result, "_{b:02x}").unwrap();
        }
    }
    result
}

/// Emits one typed binding per IR node, retaining shared operands and calls.
fn emit_function(
    out: &mut String,
    f: &IrFunction,
    bindings: Bindings,
    callee_name: &dyn Fn(&str) -> Result<String, LeanError>,
) -> Result<(), LeanError> {
    let function = callee_name(&f.name)?;
    let parameters: Vec<_> = f.param_nodes().collect();
    let signature = parameters
        .iter()
        .map(|p| format!(" (n{} : {})", p.text_id, lean_type(&p.ty)))
        .collect::<String>();
    let arguments = parameters
        .iter()
        .map(|p| format!(" n{}", p.text_id))
        .collect::<String>();
    let result_type = lean_type(&f.ret_ty);
    let name = |r: NodeRef| {
        let node = f.get_node(r);
        let local = format!("n{}", node.text_id);
        if bindings == Bindings::NamedNodes
            && !matches!(node.payload, P::Param)
            && !arguments.is_empty()
        {
            format!("({local}{arguments})")
        } else {
            local
        }
    };
    match bindings {
        Bindings::LocalLets => {
            writeln!(out, "def {function}{signature} : {result_type} :=").unwrap()
        }
        Bindings::NamedNodes => writeln!(out, "namespace {function}").unwrap(),
    }
    for (index, n) in f.nodes.iter().enumerate() {
        if matches!(n.payload, P::Nil | P::Param) {
            continue;
        }
        let e = emit_pure_expr(f, NodeRef { index }, &|r| Ok(name(r)), callee_name)?;
        let ty = lean_type(&n.ty);
        match bindings {
            Bindings::LocalLets => writeln!(out, "  let n{} : {ty} := {e}", n.text_id).unwrap(),
            Bindings::NamedNodes => {
                writeln!(out, "def n{}{signature} : {ty} :=\n  {e}", n.text_id).unwrap();
            }
        }
    }
    let result = f.ret_node_ref.expect("verified return");
    match bindings {
        Bindings::LocalLets => writeln!(out, "  {}\n", name(result)).unwrap(),
        Bindings::NamedNodes => {
            let node = f.get_node(result);
            let value = if matches!(node.payload, P::Param) {
                format!("n{}", node.text_id)
            } else {
                format!("{function}.n{}{arguments}", node.text_id)
            };
            writeln!(
                out,
                "end {function}\n\ndef {function}{signature} : {result_type} :=\n  {value}\n"
            )
            .unwrap();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_utils::direct_function_references;

    #[test]
    fn block_roots_share_a_named_callee_and_use_block_operands() {
        let ir = r#"package helpers
fn widen(x: bits[4] id=1) -> bits[8] {
  ret y: bits[8] = sign_ext(x, new_bit_count=8, id=2)
}
fn scalar(x: bits[4] id=3) -> bits[8] {
  ret y: bits[8] = invoke(x, to_apply=widen, id=4)
}
fn lanes(a: bits[4][2] id=5) -> bits[8][2] {
  ret y: bits[8][2] = map(a, to_apply=widen, id=6)
}
top block consumer(x: bits[4], a: bits[4][2], y: bits[8], ys: bits[8][2]) {
  x: bits[4] = input_port(name=x, id=7)
  result: bits[8] = invoke(x, to_apply=scalar, id=8)
  y: () = output_port(result, name=y, id=9)
  a: bits[4][2] = input_port(name=a, id=10)
  many: bits[8][2] = invoke(a, to_apply=lanes, id=11)
  ys: () = output_port(many, name=ys, id=12)
}"#;
        let package = Parser::new(ir).parse_and_validate_package().unwrap();
        let helper_name = |name: &str| Ok(format!("Helpers.«{name}»"));
        let block = package.get_top_block().unwrap();
        let callees = direct_function_references(block);
        let roots: Vec<_> = callees.iter().map(String::as_str).collect();
        let defs = emit_function_defs(&package, &roots, Bindings::LocalLets, &helper_name).unwrap();

        let result = block.output_value(block.output_ports().next().unwrap());
        let expr = emit_pure_expr(
            block,
            result,
            &|r| {
                assert!(matches!(block.get_node(r).payload, P::InputPort { .. }));
                Ok("(x + 1)".to_string())
            },
            &helper_name,
        )
        .unwrap();
        let module = format!(
            r#"-- SPDX-License-Identifier: Apache-2.0
import Std

namespace Example

{defs}def observe (x : BitVec 4) : BitVec 8 := {expr}

#guard observe 7 == 248
#guard Helpers.«lanes» #[1, 15].toVector == #[1, 255].toVector

end Example
"#
        );
        assert_eq!(module, include_str!("../tests/goldens/shared_helpers.lean"));
    }
}
