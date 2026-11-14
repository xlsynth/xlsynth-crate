// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir::{
    Binop, Fn, FunctionType, Node, NodePayload as P, NodeRef, Package, PackageMember, Type,
};
use xlsynth_pir::ir_inline::package_with_reachable_callees;
use xlsynth_pir::ir_utils::{compact_and_toposort_in_place, external_function_references};
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

/// Converts a selected pure function and its callees to executable Lean
/// definitions. Every array dimension in these functions must be nonempty.
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
    let mut reachable =
        package_with_reachable_callees(package, entry).map_err(LeanError::InvalidIr)?;
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
    let mut work = vec![(entry.name.clone(), false)];
    let mut emitted = HashSet::new();
    let ns = module_name.lean_identifier();
    let mut source =
        format!("{MODULE_HEADER}namespace {ns}\nset_option linter.unusedVariables false\n\n");
    let functions: HashMap<_, _> = reachable
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Function(f) => Some((f.name.as_str(), f)),
            PackageMember::Block(_) => None,
        })
        .collect();
    write_elaboration_limits(&mut source, functions.values().map(|f| &f.graph));
    while let Some((name, ready)) = work.pop() {
        if emitted.contains(&name) {
            continue;
        }
        let f = functions[name.as_str()];
        if ready {
            emit_function(&mut source, f)?;
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
fn check_supported(f: &Fn, node: &Node) -> Result<(), LeanError> {
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

/// Emits one typed binding per IR node, retaining shared operands and calls.
fn emit_function(out: &mut String, f: &Fn) -> Result<(), LeanError> {
    let renderer = LocalRenderer::Function;
    let function = function_name(&f.name);
    let signature = f
        .param_nodes()
        .map(|p| format!(" (n{} : {})", p.text_id, renderer.lean_type(&p.ty)))
        .collect::<String>();
    let result_type = renderer.lean_type(&f.ret_ty);
    let name = |r: NodeRef| format!("n{}", f.get_node(r).text_id);
    writeln!(out, "def {function}{signature} : {result_type} :=").unwrap();
    for (index, n) in f.nodes.iter().enumerate() {
        let e = match &n.payload {
            P::Nil | P::Param => continue,
            P::Map { array, to_apply } => {
                format!("Vector.map {} {}", function_name(to_apply), name(*array))
            }
            P::Invoke { to_apply, operands } => std::iter::once(function_name(to_apply))
                .chain(operands.iter().copied().map(name))
                .collect::<Vec<_>>()
                .join(" "),
            _ => renderer
                .local_rhs(f, NodeRef { index }, |r| {
                    Ok::<_, std::convert::Infallible>(LeanOperand::Atom(name(r)))
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
    writeln!(out, "  {}\n", name(result)).unwrap();
    Ok(())
}
