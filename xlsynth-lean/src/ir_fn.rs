// SPDX-License-Identifier: Apache-2.0

//! Export a pure XLS call graph as Lean definitions shared by evaluators and
//! proofs.
//!
//! Each function has local lets for its computations and refers to callees by
//! name.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir::{
    Binop, Fn as IrFn, FunctionType, Node, NodePayload as P, NodeRef, Package, PackageMember, Type,
};
use xlsynth_pir::ir_inline::reachable_functions_in_dependency_order;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;
use xlsynth_pir::ir_verify::verify_package;

use crate::local_expr::{
    LeanOperand, LocalRenderer, RenderError, bit_width, write_elaboration_limits,
};
use crate::support::write_file;
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

    /// Writes the generated module under `root`. The Lean project supplies
    /// its `XlsynthLean.IR` import.
    pub fn write_to(&self, root: &Path) -> Result<PathBuf, LeanError> {
        let path = self.module_name.source_path();
        write_file(root, &path, self.source.as_bytes())?;
        Ok(root.join(path))
    }
}

/// Converts a selected pure function and its reachable callees to executable
/// Lean definitions.
///
/// Supply `top` to select a function by name. With `None`, the package must
/// have a function top or, if it has no top, a single function. The generated
/// `eval` names that function and preserves its positional argument and result
/// types.
///
/// Export fails on invalid IR or unsupported operations and types in the
/// selected call graph, including zero-length arrays.
pub fn export_ir_function(
    package: &Package,
    top: Option<&str>,
    module_name: &LeanModuleName,
) -> Result<LeanFunction, LeanError> {
    let entry = match top {
        Some(name) => package.get_named_or_top_or_sole_fn(name),
        None => package.get_top_or_sole_fn(),
    }
    .map_err(LeanError::InvalidIr)?;
    let functions = prepare_function_defs(package, &[&entry.name])?;
    let ns = module_name.lean_identifier();
    let mut source =
        format!("{MODULE_HEADER}namespace {ns}\nset_option linter.unusedVariables false\n\n");
    write_elaboration_limits(&mut source, functions.iter().map(|f| &f.graph));
    for f in &functions {
        emit_function(&mut source, f, &|name| Ok(function_name(name)))?;
    }
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

/// Checks and normalizes reachable helpers in Lean declaration order. The
/// enclosing exporter can inspect their signatures and size its module first.
pub(crate) fn prepare_function_defs(
    package: &Package,
    roots: &[&str],
) -> Result<Vec<IrFn>, LeanError> {
    let functions =
        reachable_functions_in_dependency_order(package, roots).map_err(LeanError::InvalidIr)?;
    let mut reachable = Package {
        name: package.name.clone(),
        file_table: package.file_table.clone(),
        members: Vec::with_capacity(functions.len()),
        top: None,
    };
    for f in functions {
        for node in &f.nodes {
            check_supported(f, node)?;
        }
        let mut f = f.clone();
        compact_and_toposort_in_place(&mut f).map_err(LeanError::InvalidIr)?;
        reachable.members.push(PackageMember::Function(f));
    }
    verify_package(&reachable).map_err(|e| LeanError::InvalidIr(e.to_string()))?;
    let mut prepared = Vec::with_capacity(reachable.members.len());
    for member in reachable.members {
        if let PackageMember::Function(mut f) = member {
            if f.nodes.iter().any(|n| n.payload.is_extension_op()) {
                desugar_extensions_in_fn(&mut f)
                    .map_err(|e| LeanError::InvalidIr(e.to_string()))?;
            }
            prepared.push(f);
        }
    }
    Ok(prepared)
}

/// Applies a Lean function to operands, grouping compound expressions.
pub(crate) fn invoke_expr(
    callee: String,
    operands: impl IntoIterator<Item = LeanOperand>,
) -> String {
    std::iter::once(callee)
        .chain(operands.into_iter().map(LeanOperand::into_argument))
        .collect::<Vec<_>>()
        .join(" ")
}

fn width(ty: &Type) -> Result<usize, LeanError> {
    bit_width(ty).map_err(LeanError::InvalidIr)
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

fn unsupported_type_reason(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Token => Some("token-valued type"),
        Type::Bits(_) => None,
        Type::Tuple(fields) => fields
            .iter()
            .find_map(|field| unsupported_type_reason(field)),
        Type::Array(a) if a.element_count == 0 => Some("zero-length array"),
        Type::Array(a) => unsupported_type_reason(&a.element_type),
    }
}

/// Check even unused nodes so an unsupported observation cannot silently
/// disappear.
fn check_supported(f: &IrFn, node: &Node) -> Result<(), LeanError> {
    let reason = match &node.payload {
        P::Binop(Binop::Smulp | Binop::Umulp, ..) => "partial-product multiply",
        P::CountedFor { .. } => "loop",
        P::AfterAll(_) | P::Assert { .. } | P::Trace { .. } | P::Cover { .. } => "token or event",
        P::InputPort { .. }
        | P::OutputPort { .. }
        | P::InstantiationInput { .. }
        | P::InstantiationOutput { .. }
        | P::RegisterRead { .. }
        | P::RegisterWrite { .. } => "block operation",
        _ => match unsupported_type_reason(&node.ty) {
            Some(reason) => reason,
            None => return Ok(()),
        },
    };
    Err(LeanError::UnsupportedIr {
        function: f.name.clone(),
        op: node.payload.get_operator().to_owned(),
        id: node.text_id,
        reason,
    })
}

/// Emits a verified function in dependency order with one typed let per
/// computation.
///
/// Each operand refers to its producer's binding by node ID. Two invokes of the
/// same callee with identical operands still bind distinct results.
pub(crate) fn emit_function(
    out: &mut String,
    f: &IrFn,
    callee_name: &dyn Fn(&str) -> Result<String, LeanError>,
) -> Result<(), LeanError> {
    let renderer = LocalRenderer::Function;
    let function = callee_name(&f.name)?;
    let signature = f
        .param_nodes()
        .map(|p| format!(" (n{} : {})", p.text_id, renderer.lean_type(&p.ty)))
        .collect::<String>();
    let result_type = renderer.lean_type(&f.ret_ty);
    let name = |r: NodeRef| LeanOperand::Atom(format!("n{}", f.get_node(r).text_id));
    writeln!(out, "def {function}{signature} : {result_type} :=").unwrap();
    for (index, n) in f.nodes.iter().enumerate() {
        let e = match &n.payload {
            P::Nil | P::Param => continue,
            P::Map { array, to_apply } => {
                format!(
                    "Vector.map {} {}",
                    callee_name(to_apply)?,
                    name(*array).into_argument()
                )
            }
            P::Invoke { to_apply, operands } => {
                invoke_expr(callee_name(to_apply)?, operands.iter().copied().map(name))
            }
            _ => renderer
                .local_rhs(f, NodeRef { index }, |r| {
                    Ok::<_, std::convert::Infallible>(name(r))
                })
                .map_err(|error| match error {
                    RenderError::Unsupported => LeanError::UnsupportedIr {
                        function: f.name.clone(),
                        op: n.payload.get_operator().to_owned(),
                        id: n.text_id,
                        reason: "no word-level translation",
                    },
                    RenderError::Invalid(reason) => LeanError::InvalidIr(reason),
                    RenderError::Operand(never) => match never {},
                })?,
        };
        let ty = renderer.lean_type(&n.ty);
        writeln!(out, "  let n{} : {ty} := {e}", n.text_id).unwrap();
    }
    let result = f.ret_node_ref.expect("verified return");
    writeln!(out, "  {}\n", name(result).into_argument()).unwrap();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_utils::direct_function_references;

    #[test]
    fn enclosing_exporter_owns_limits_for_helpers_and_packed_wrapper() {
        // Both signatures need elevated limits, and the caller needs more.
        let tuple = format!("({})", vec!["bits[1]"; 128].join(", "));
        let ir = format!(
            r#"package limits
fn helper(x: {tuple} id=1) -> {tuple} {{
  ret copied: {tuple} = identity(x, id=2)
}}
top fn caller(x: {tuple} id=3) -> ({tuple}, {tuple}) {{
  h: {tuple} = invoke(x, to_apply=helper, id=4)
  ret both: ({tuple}, {tuple}) = tuple(x, h, id=5)
}}"#
        );
        let pkg = Parser::new(&ir).parse_and_validate_package().unwrap();
        let name = LeanModuleName::new("Limits").unwrap();
        let module = export_ir_function(&pkg, None, &name).unwrap();
        let (header, defs) = module.lean_source().split_once("def ir_helper").unwrap();
        assert_eq!(
            header
                .lines()
                .filter(|line| line.starts_with("set_option") && !line.contains("linter"))
                .collect::<Vec<_>>(),
            [
                "set_option maxRecDepth 2580",
                "set_option synthInstance.maxSize 1034"
            ],
        );
        assert!(!defs.contains("set_option"));
        assert!(defs.contains("def evalPacked"));
    }

    #[test]
    fn block_roots_share_a_named_callee_and_use_block_operands() {
        let ir = r#"package helpers
fn scalar(x: bits[4] id=3) -> bits[8] {
  ret y: bits[8] = invoke(x, to_apply=widen, id=4)
}
fn lanes(a: bits[4][2] id=5) -> bits[8][2] {
  ret y: bits[8][2] = map(a, to_apply=widen, id=6)
}
fn widen(x: bits[4] id=1) -> bits[8] {
  ret y: bits[8] = sign_ext(x, new_bit_count=8, id=2)
}
fn update(rows: bits[8][2][2] id=13, outer: bits[1] id=14, v0: bits[1] id=15, value: bits[8] id=16) -> bits[8][2][2] {
  ret updated: bits[8][2][2] = array_update(rows, value, indices=[outer, v0], id=17)
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
        let roots: Vec<_> = direct_function_references(block).into_iter().collect();
        let functions = prepare_function_defs(&package, &roots).unwrap();
        let mut defs = String::new();
        write_elaboration_limits(
            &mut defs,
            std::iter::once(&block.graph).chain(functions.iter().map(|f| &f.graph)),
        );
        for f in &functions {
            emit_function(&mut defs, f, &helper_name).unwrap();
        }
        let result = block.output_value(block.output_ports().next().unwrap());
        let P::Invoke { to_apply, operands } = &block.get_node(result).payload else {
            panic!("expected an invoke");
        };
        let expr = invoke_expr(
            helper_name(to_apply).unwrap(),
            operands.iter().map(|&r| {
                assert!(matches!(block.get_node(r).payload, P::InputPort { .. }));
                LeanOperand::Expression("x + 1".to_owned())
            }),
        );
        // Use the caller's operand names in the shared renderer. An explicit
        // row binder named v0 would capture the caller's inner index.
        let update = package.get_fn("update").unwrap();
        let rhs = LocalRenderer::Block
            .local_rhs(update, update.ret_node_ref.unwrap(), |r| {
                Ok::<_, std::convert::Infallible>(LeanOperand::Atom(
                    update.get_node(r).name.clone().unwrap(),
                ))
            })
            .unwrap();
        let mut module = format!(
            r#"-- SPDX-License-Identifier: Apache-2.0
import XlsynthLean.IR

namespace Example

{defs}def observe (x : BitVec 4) : BitVec 8 := {expr}

#guard observe 7 == 248
#guard Helpers.«lanes» #[1, 15].toVector == #[1, 255].toVector

abbrev Rows := Vector (Vector (BitVec 8) 2) 2

def update (rows : Rows) (outer v0 : BitVec 1) (value : BitVec 8) : Rows :=
  {rhs}

def rows : Rows := #[#[1, 2].toVector, #[3, 4].toVector].toVector
#guard update rows 0 0 9 == #[#[9, 2].toVector, #[3, 4].toVector].toVector
#guard update rows 0 1 9 == #[#[1, 9].toVector, #[3, 4].toVector].toVector

end Example
"#
        );
        crate::block::library_notation::add_to_module(&mut module);
        assert_eq!(module, include_str!("../tests/goldens/shared_helpers.lean"));
    }
}
