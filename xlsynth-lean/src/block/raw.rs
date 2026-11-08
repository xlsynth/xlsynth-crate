// SPDX-License-Identifier: Apache-2.0

//! Direct Lean emission from verified, typed XLS block IR.
//!
//! The generated `XlsBlock.eval` evaluates every node from old registers and
//! current inputs. Its result contains old-state outputs, simultaneous next
//! registers, and every source assertion, including disconnected assertions.
//! No assertion narrows the input/state domain. Tuples remain typed products;
//! no enum decoding or payload projection is part of this translation.

use anyhow::{Context, Result, ensure};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use xlsynth_pir::{
    ir::{self, NodePayload, NodeRef, Register, Type},
    ir_utils,
};

use crate::LeanModuleName;
use crate::block::BlockContext;
use crate::local_expr::{LeanOperand, LocalRenderer, RenderError, write_elaboration_limits};
use crate::support::write_file;

/// Five Lean source files relating readable execution to the raw block model.
///
/// Emission does not run Lean; callers must elaborate the generated proofs.
pub struct LeanBlockBundle {
    /// File names and UTF-8 source in dependency order.
    ///
    /// The order is `Generated.lean`, `ReadableTypes.lean`,
    /// `ReadableGenerated.lean`, `ReadableRepresentation.lean`, then
    /// `ReadableCorrespondence.lean`.
    pub files: Vec<(String, String)>,
}

/// Emit a verified XLS block in the `XlsBlock` namespace.
/// The returned source imports `XlsynthLean.IR`. Use [`write_to`] to
/// write a named model into a Lean project that supplies this import.
/// Return an error if the block uses an unsupported operation.
///
/// Pass a verified block borrowed from `package`, whose file table supplies
/// source locations in diagnostics and generated comments. For saved IR, use
/// [`xlsynth_pir::ir_parser::Parser::parse_and_validate_package`].
/// Revalidate the package if you modify it before emission.
///
/// `State`, `Input`, and `Output` fields keep source names and exact widths.
/// When a field shares a name with a Lean-generated declaration (such as
/// `Input.rec`), the record exposes that field through a named accessor.
/// An XLS tuple `[a, b]` is represented by `(a, (b, ()))`; element `j` uses
/// `j` second projections followed by a first projection. Tokens are `Unit`.
/// `reset q x` forces only the reset port and evaluates an ordinary clock; it
/// does not invent an initial value for unreset storage or clear old outputs.
/// Bits of any width, tokens, tuples and nonempty arrays retain their physical
/// layout, including for aggregate one-hot and priority selects.
/// Records derive `Repr`, but whole-record `DecidableEq` is left to callers:
/// deriving it exceeds Lean's instance-search size limit on nested state.
/// Unsupported nodes, types, or reset/instance shapes return a source error.
/// Do not treat `reset` as a constructor for initialized state: registers
/// without a reset value still depend on the supplied old state.
pub fn emit(package: &ir::Package, block: &ir::Block) -> Result<String> {
    let mut module = emit_with_context(&BlockContext::new(package, block)?, "XlsBlock")?;
    super::library_notation::add_to_module(&mut module);
    Ok(module)
}

/// Write the generated block under `root`. The project supplies its
/// `XlsynthLean.IR` import. The module name determines both the file path and
/// the namespace, so blocks with different module names can be imported
/// together.
pub fn write_to(
    package: &ir::Package,
    block: &ir::Block,
    module_name: &LeanModuleName,
    root: &Path,
) -> Result<PathBuf> {
    let mut source = emit_with_context(
        &BlockContext::new(package, block)?,
        &module_name.lean_identifier(),
    )?;
    super::library_notation::add_to_module(&mut source);
    let path = module_name.source_path();
    write_file(root, &path, source.as_bytes())?;
    Ok(root.join(path))
}

pub(super) fn emit_with_context(block: &BlockContext<'_>, namespace: &str) -> Result<String> {
    emit_block(block, namespace).with_context(|| {
        format!(
            "Lean emission of block {} in package {}",
            block.graph.name, block.package_name
        )
    })
}

/// Assemble a clock-step model after checking the emitter's supported shapes.
///
/// Every register must have exactly one write; outputs use the data operand of
/// each output port, whose graph node itself has the empty tuple type.
fn emit_block(block: &BlockContext<'_>, namespace: &str) -> Result<String> {
    ensure!(
        block.graph.instantiations.is_empty(),
        "block instances are outside the supported Lean subset"
    );
    if let Some(reset) = &block.graph.reset {
        ensure!(
            !reset.asynchronous,
            "asynchronous reset on {} is outside the Lean clock-step contract",
            block.graph.port_name(reset.port)
        );
        ensure!(
            block.graph.port_type(reset.port) == &Type::Bits(1),
            "reset port must be bits[1]"
        );
    }

    let graph = &*block.graph;
    let writes = block
        .graph
        .registers
        .iter()
        .map(|register| {
            let mut writes = graph.nodes.iter().enumerate().filter_map(|(index, node)| {
                matches!(&node.payload, NodePayload::RegisterWrite { register: name, .. }
                    if name == &register.name)
                .then_some(NodeRef { index })
            });
            let write = writes
                .next()
                .with_context(|| format!("register {} has no write", register.name))?;
            ensure!(
                writes.next().is_none(),
                "register {} has multiple writes; exactly one is supported",
                register.name
            );
            Ok(write)
        })
        .collect::<Result<Vec<_>>>()?;
    let assertions: Vec<_> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            matches!(node.payload, NodePayload::Assert { .. }).then_some(NodeRef { index })
        })
        .collect();
    let names: Vec<_> = (0..graph.nodes.len())
        .map(|index| {
            let reference = NodeRef { index };
            let source_name = ir::node_textual_id(block.source_graph(reference), reference);
            let suffix: String = source_name
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            // The graph index makes normalization injective. The prefix keeps
            // source names such as q, x, and Output from shadowing the API.
            format!("xls_node_{index}_{suffix}")
        })
        .collect();

    let mut source = format!(
        "-- Generated from XLS block {} in package {}.\n\
         -- Exact BitVec values; tuple elements end in Unit.\n\
         -- Outputs read old registers; all writes commit together.\n\
         -- Source assertions are results, never assumptions.\n\n\
         import XlsynthLean.IR\n\nnamespace {namespace}\n\n",
        comment(&graph.name),
        comment(block.package_name)
    );
    write_elaboration_limits(
        &mut source,
        std::iter::once(&graph.graph).chain(block.helpers.iter().map(|f| &f.graph)),
    );
    if graph
        .nodes
        .iter()
        .chain(block.helpers.iter().flat_map(|helper| &helper.nodes))
        .any(|node| matches!(node.payload, NodePayload::CountedFor { .. }))
    {
        source.push_str(
            "def countedFor {width : L.Nat} {α : Type} (tripCount stride : L.Nat)
    (body : L.BitVec width → α → α) (init : α) : α :=
  L.Nat.fold tripCount (fun iteration _ carry => body (L.BitVec.ofNat width (iteration * stride)) carry) init

",
        );
    }
    if graph
        .nodes
        .iter()
        .chain(block.helpers.iter().flat_map(|f| &f.nodes))
        .any(|n| matches!(n.payload, NodePayload::BitSliceUpdate { .. }))
    {
        source.push_str(
            "abbrev bitSliceUpdate {n s w : L.Nat} (value : L.BitVec n) (start : L.BitVec s) (update : L.BitVec w) : L.BitVec n :=\n  _root_.XlsynthLean.IR.bitSliceUpdate value start update\n\n",
        );
    }
    for helper in &block.helpers {
        write_helper(&mut source, namespace, helper)?;
    }
    source.push_str(&record_type(
        namespace,
        "State",
        block
            .graph
            .registers
            .iter()
            .map(|r| (r.name.as_str(), &r.ty)),
    )?);
    source.push_str(&record_type(
        namespace,
        "Input",
        graph
            .input_ports()
            .map(|port| (graph.port_name(port), graph.port_type(port))),
    )?);
    source.push_str(&record_type(
        namespace,
        "Output",
        graph
            .output_ports()
            .map(|port| (graph.port_name(port), graph.port_type(port))),
    )?);
    let mut fields = Vec::new();
    let mut comments = Vec::new();
    for reference in &assertions {
        let node = graph.get_node(*reference);
        if let NodePayload::Assert { label, message, .. } = &node.payload {
            comments.push(format!(
                "id={}; label={}; message={}",
                node.text_id,
                comment(label),
                comment(message)
            ));
            fields.push((
                identifier(&ir::node_textual_id(graph, *reference))?,
                "L.Bool".to_owned(),
            ));
        }
    }
    source.push_str(&record_declaration(
        namespace,
        "Assertions",
        &fields,
        Some("L.Repr"),
        &comments,
    )?);
    source.push_str(&structure_header("Evaluation"));
    source.push_str(
        "  next : State\n\
         \x20 output : Output\n\
         \x20 assertions : Assertions\n\
         \x20 deriving L.Repr\n\n\
         def eval (q : State) (x : Input) : Evaluation :=\n",
    );

    for reference in ir_utils::get_topological_nodes(&graph.nodes) {
        let node = graph.get_node(reference);
        if !matches!(node.payload, NodePayload::Nil) {
            let context = node_context(block, reference);
            if let NodePayload::RegisterRead { register } = &node.payload {
                ensure!(
                    graph.registers.iter().any(|decl| decl.name == *register),
                    "{context}: no declaration for register {register}"
                );
            }
            let ty = lean_type(&node.ty).with_context(|| context.clone())?;
            let expression = node_expression(namespace, graph, reference, |operand| {
                Ok(LeanOperand::Atom(names[operand.index].clone()))
            })
            .with_context(|| context.clone())?;
            writeln!(source, "  -- {}", comment(&context))?;
            writeln!(
                source,
                "  let {} : {ty} := {expression}",
                names[reference.index]
            )?;
        }
    }

    let next_fields = block
        .graph
        .registers
        .iter()
        .zip(writes)
        .map(|(register, reference)| {
            let expression = register_update(block, register, reference, &names)
                .with_context(|| node_context(block, reference))?;
            Ok((identifier(&register.name)?, expression))
        })
        .collect::<Result<Vec<_>>>()?;
    let output_fields = block
        .graph
        .output_ports()
        .map(|port| {
            Ok((
                identifier(graph.port_name(port))?,
                names[graph.output_value(port).index].clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let assertion_fields = assertions
        .iter()
        .map(|reference| {
            let NodePayload::Assert { activate, .. } = graph.get_node(*reference).payload else {
                unreachable!("assertions were selected by payload")
            };
            Ok((
                identifier(&ir::node_textual_id(graph, *reference))?,
                predicate(&names[activate.index]),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    writeln!(
        source,
        "  {{ next := {},\n    output := {},\n    assertions := {} }}\n",
        source_record_value(&next_fields),
        source_record_value(&output_fields),
        source_record_value(&assertion_fields)
    )?;
    source.push_str(
        "def step (q : State) (x : Input) : State := (eval q x).next\n\n\
         def observe (q : State) (x : Input) : Output := (eval q x).output\n\n\
         def assertions (q : State) (x : Input) : Assertions := (eval q x).assertions\n\n",
    );
    let holds = assertion_fields
        .iter()
        .rev()
        .fold("L.True".to_string(), |rest, (name, _)| {
            format!("L.And (checks.{name} = true) ({rest})")
        });
    writeln!(
        source,
        "def assertionsHold (q : State) (x : Input) : Prop :=\n  let checks := assertions q x\n  {holds}\n"
    )?;
    source.push_str(
        "-- One clock with the declared reset port active. Other inputs and old\n\
         -- registers stay arbitrary. With no reset port this is an ordinary clock.\n\
         def reset (q : State) (x : Input) : Evaluation :=\n",
    );
    if let Some(reset) = &block.graph.reset {
        writeln!(
            source,
            "  eval q {{ x with {} := L.BitVec.ofNat 1 {} }}",
            storage_field(&identifier(graph.port_name(reset.port))?),
            u8::from(!reset.active_low)
        )?;
    } else {
        source.push_str("  eval q x\n");
    }
    writeln!(source, "\nend {namespace}")?;
    Ok(source)
}

pub(super) fn helper_path(namespace: &str, physical: &str) -> Result<String> {
    Ok(format!(
        "_root_.{namespace}.Helpers.{}",
        helper_identifier(physical)?
    ))
}

pub(super) fn helper_identifier(physical: &str) -> Result<String> {
    // Lean reserves this declaration name. A space cannot occur in a quoted
    // IR identifier accepted by identifier(), so the replacement is unique.
    if physical == "_root_" {
        Ok("«xls helper _root_»".into())
    } else {
        identifier(physical)
    }
}

fn write_helper(source: &mut String, namespace: &str, function: &ir::Fn) -> Result<()> {
    write!(source, "def Helpers.{}", helper_identifier(&function.name)?)?;
    for param in &function.params {
        write!(
            source,
            " (n{} : {})",
            param.index,
            lean_type(&function.get_node(*param).ty)?
        )?;
    }
    writeln!(source, " : {} :=", lean_type(&function.ret_ty)?)?;
    for reference in ir_utils::get_topological_nodes(&function.nodes) {
        let node = function.get_node(reference);
        if !matches!(node.payload, NodePayload::Nil | NodePayload::Param) {
            ensure!(
                !matches!(node.payload, NodePayload::Assert { .. }),
                "assertion in helper {} is unsupported",
                function.name
            );
            let expr = node_expression(namespace, function, reference, |operand| {
                Ok(LeanOperand::Atom(format!("n{}", operand.index)))
            })
            .with_context(|| format!("function {}, node {}", function.name, node.text_id))?;
            writeln!(
                source,
                "  let n{} : {} := {expr}",
                reference.index,
                lean_type(&node.ty)?
            )?;
        }
    }
    let result = function.ret_node_ref.context("helper has no return node")?;
    writeln!(source, "  n{}\n", result.index)?;
    Ok(())
}

/// Array reads clamp to the last element, so every dimension must be nonempty.
pub(crate) fn lean_type(ty: &Type) -> Result<String> {
    fn contains_empty_array(ty: &Type) -> bool {
        match ty {
            Type::Array(a) => a.element_count == 0 || contains_empty_array(&a.element_type),
            Type::Tuple(fields) => fields.iter().any(|field| contains_empty_array(field)),
            Type::Bits(_) | Type::Token => false,
        }
    }
    ensure!(!contains_empty_array(ty), "zero-length array in {ty}");
    Ok(LocalRenderer::Block.lean_type(ty))
}

/// Lean generates these declarations for every structure and inductive type.
/// Quoting a source name preserves its spelling but does not prevent a clash.
pub(super) fn generated_member(field: &str) -> bool {
    matches!(
        field,
        "rec"
            | "recOn"
            | "casesOn"
            | "noConfusion"
            | "noConfusionType"
            | "ctorIdx"
            | "_sizeOf_1"
            | "_sizeOf_inst"
    )
}

/// Keep the constructor in a separate namespace so every source field keeps its
/// name.
pub(super) fn structure_header(name: &str) -> String {
    format!("structure {name} where\n  Constructors.mk ::\n")
}

/// Emit source fields under their public names. Lean puts its own eliminators
/// next to the real structure, so only records with a competing field need a
/// separate underlying type. Quoted names containing ':' cannot occur in BX
/// or DSLX source.
pub(super) fn source_record(name: &str, fields: &[(String, String)]) -> Result<String> {
    // Source modules may declare their own type named Repr.
    record_declaration("XlsReadable", name, fields, Some("L.Repr"), &[])
}

pub(super) fn record_backing_name(name: &str, fields: &[(String, String)]) -> String {
    if has_generated_field(fields) {
        format!("«xls:implementation».{name}")
    } else {
        name.to_owned()
    }
}

pub(super) fn record_declaration(
    namespace: &str,
    name: &str,
    fields: &[(String, String)],
    derived_repr: Option<&str>,
    comments: &[String],
) -> Result<String> {
    let backing = record_backing_name(name, fields);
    let mut out = structure_header(&backing);
    let mut scope = RecordFields::default();
    for (index, (field, ty)) in fields.iter().enumerate() {
        if let Some(comment) = comments.get(index) {
            writeln!(out, "  -- {comment}")?;
        }
        scope.write(&mut out, &storage_field(field), ty)?;
    }
    if let Some(repr) = derived_repr {
        writeln!(out, "  deriving {repr}")?;
    }
    out.push('\n');
    if backing != name {
        writeln!(out, "abbrev {name} := {backing}\n")?;
        for (field, _) in fields {
            ensure!(
                field != "«_root_»",
                "cannot emit {namespace}.{name} field '_root_': Lean reserves this name for declarations, and this record needs a separate field accessor"
            );
            // This declaration opens the record namespace. An earlier reader
            // named like the record can shadow the type in later signatures.
            writeln!(
                out,
                "@[simp] abbrev {name}.{field} (p : _root_.{namespace}.{name}) := {backing}.{} p",
                storage_field(field)
            )?;
        }
        out.push('\n');
    }
    Ok(out)
}

pub(super) fn storage_field(field: &str) -> String {
    let spelling = field.trim_matches(['«', '»']);
    if generated_member(spelling) {
        format!("«field:{spelling}»")
    } else {
        field.to_owned()
    }
}

fn has_generated_field(fields: &[(String, String)]) -> bool {
    fields
        .iter()
        .any(|(field, _)| generated_member(field.trim_matches(['«', '»'])))
}

/// Brace literals require actual structure fields. Construct a record with
/// public field accessors positionally, in the declaration's field order.
pub(crate) fn source_record_value(fields: &[(String, String)]) -> String {
    if has_generated_field(fields) {
        format!(
            "⟨{}⟩",
            fields
                .iter()
                .map(|(_, v)| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        record_value(fields)
    }
}

/// Track the field scope of one Lean structure declaration.
#[derive(Default)]
pub(super) struct RecordFields {
    shadows_root: bool,
}

impl RecordFields {
    /// Lean elaborates a field type in the scope of preceding fields. Even the
    /// root qualifier can be captured if a field itself is named `_root_`.
    /// All hardware field types are independent of earlier field values.
    pub(super) fn write(
        &mut self,
        out: &mut String,
        field: &str,
        rooted_type: &str,
    ) -> std::fmt::Result {
        if self.shadows_root {
            writeln!(out, "  {field} : (by clear «_root_»; exact {rooted_type})")?;
        } else {
            writeln!(out, "  {field} : {rooted_type}")?;
        }
        self.shadows_root |= field == "«_root_»";
        Ok(())
    }
}

/// Declare fields in caller order, retaining names through Lean quotation.
fn record_type<'a>(
    namespace: &str,
    name: &str,
    fields: impl Iterator<Item = (&'a str, &'a Type)>,
) -> Result<String> {
    let fields = fields
        .map(|(field, ty)| {
            Ok((
                identifier(field).with_context(|| format!("{name} field {field}"))?,
                lean_type(ty).with_context(|| format!("{name} field {field}"))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    record_declaration(namespace, name, &fields, Some("L.Repr"), &[])
}

/// Assemble a record from already rendered field identifiers and expressions.
pub(crate) fn record_value(fields: &[(String, String)]) -> String {
    format!(
        "{{ {} }}",
        fields
            .iter()
            .map(|(name, expression)| format!("{name} := {expression}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

// Whitespace keeps this Lean local distinct from DSLX and IR identifiers.
const ARRAY_LOCAL: &str = "«indexed array»";

/// XLS clamps reads to the last element of each indexed dimension.
fn read_bound_array(index: &str) -> String {
    format!(
        "(L.Vector.getD {ARRAY_LOCAL} (L.BitVec.toNat ({index})) (L.Vector.back {ARRAY_LOCAL}))"
    )
}

/// Share the operand between the lookup and its fallback at each dimension.
pub(super) fn array_read(array: &str, index: &str) -> String {
    format!(
        "(let {ARRAY_LOCAL} := {array}; {})",
        read_bound_array(index)
    )
}

/// Updating an out-of-bounds index leaves the whole array unchanged.
pub(super) fn array_write(array: &str, indices: &[String], value: &str) -> String {
    match indices {
        [] => value.to_owned(),
        [index] => {
            format!("(L.Vector.setIfInBounds ({array}) (L.BitVec.toNat ({index})) ({value}))")
        }
        [index, remaining @ ..] => {
            let item = array_write(&read_bound_array(index), remaining, value);
            format!(
                "(let {ARRAY_LOCAL} := {array}; L.Vector.setIfInBounds {ARRAY_LOCAL} (L.BitVec.toNat ({index})) ({item}))"
            )
        }
    }
}

/// Interpret a rendered one-bit value as the Boolean condition that it is set.
fn predicate(value: &str) -> String {
    format!("({value} == (L.BitVec.ofNat 1 1))")
}

/// Render one supported graph operation using caller-supplied operand
/// expressions.
///
/// Resolve only the operands used by this operation. Register writes and
/// output ports return unit here; their effects are collected by `emit_block`.
pub(crate) fn node_expression(
    namespace: &str,
    graph: &ir::NodeGraph,
    reference: NodeRef,
    operand: impl Fn(NodeRef) -> Result<LeanOperand>,
) -> Result<String> {
    let node = graph.get_node(reference);
    match &node.payload {
        NodePayload::InputPort { name, .. } => Ok(format!("x.{}", identifier(name)?)),
        NodePayload::Invoke { to_apply, operands } => {
            let mut call = helper_path(namespace, to_apply)?;
            for arg in operands {
                write!(call, " {}", operand(*arg)?.into_argument())?;
            }
            Ok(call)
        }
        NodePayload::CountedFor {
            init,
            trip_count,
            stride,
            body,
            invariant_args,
        } => {
            let mut step = helper_path(namespace, body)?;
            if !invariant_args.is_empty() {
                step.push_str(" · ·");
                for arg in invariant_args {
                    write!(step, " {}", operand(*arg)?.into_argument())?;
                }
            }
            Ok(format!(
                "_root_.{namespace}.countedFor {trip_count} {stride} ({step}) {}",
                operand(*init)?.into_argument()
            ))
        }
        NodePayload::RegisterRead { register } => Ok(format!("q.{}", identifier(register)?)),
        // These effects are collected by emit_block, including disconnected
        // assertions. The output and write nodes themselves have unit type.
        NodePayload::OutputPort { .. }
        | NodePayload::RegisterWrite { .. }
        | NodePayload::AfterAll(_) => Ok("()".to_string()),
        NodePayload::Assert { token, .. } => Ok(operand(*token)?.into_argument()),
        other => LocalRenderer::Block
            .local_rhs(graph, reference, operand)
            .map_err(|error| match error {
                RenderError::Operand(error) => error,
                RenderError::Invalid(reason) => anyhow::anyhow!("{reason}"),
                RenderError::Unsupported => {
                    anyhow::anyhow!("unsupported Lean operation {}", other.get_operator())
                }
            }),
    }
}

/// Compute one next-state field with reset taking precedence over load enable.
///
/// The reference must be the selected write for this register, and every entry
/// in `names` must denote an expression evaluated from old state.
fn register_update(
    block: &BlockContext<'_>,
    register: &Register,
    reference: NodeRef,
    names: &[String],
) -> Result<String> {
    let NodePayload::RegisterWrite {
        arg,
        load_enable,
        reset,
        ..
    } = &block.graph.get_node(reference).payload
    else {
        unreachable!("register writes were selected by payload")
    };
    let enabled = if let Some(enable) = load_enable {
        format!(
            "(if {} then {} else q.{})",
            predicate(&names[enable.index]),
            names[arg.index],
            identifier(&register.name)?
        )
    } else {
        names[arg.index].clone()
    };
    if let Some(signal) = reset {
        let behavior = block
            .graph
            .reset
            .as_ref()
            .context("register reset has no block reset behavior")?;
        let value = register
            .reset_value
            .as_ref()
            .context("reset register has no reset value")?;
        Ok(format!(
            "(if {} == (L.BitVec.ofNat 1 {}) then {} else {enabled})",
            names[signal.index],
            u8::from(!behavior.active_low),
            LocalRenderer::Block.literal(value)
        ))
    } else {
        Ok(enabled)
    }
}

/// Quote a nonempty ASCII graphic name without merging distinct source names.
pub(crate) fn identifier(source: &str) -> Result<String> {
    ensure!(
        !source.is_empty() && source.chars().all(|c| c.is_ascii_graphic()),
        "unsupported Lean source identifier {source:?}"
    );
    // Quoting keeps source dots and keywords distinct instead of normalizing
    // fields. The ASCII check prevents embedded quote delimiters or newlines.
    Ok(format!("\u{00ab}{source}\u{00bb}"))
}

/// Describe a node and its available source locations for errors and Lean
/// comments.
pub(crate) fn node_context(block: &BlockContext<'_>, reference: NodeRef) -> String {
    let source_graph = block.source_graph(reference);
    let node = source_graph.get_node(reference);
    let locations = node
        .pos
        .iter()
        .flatten()
        .map(|position| {
            position
                .to_human_string(block.file_table)
                .unwrap_or_else(|| {
                    format!(
                        "file_number {}:{}:{}",
                        position.fileno,
                        position.lineno + 1,
                        position.colno + 1
                    )
                })
        })
        .collect::<Vec<_>>();
    format!(
        "node {} [id={}, op={}, type={}]{}",
        ir::node_textual_id(source_graph, reference),
        node.text_id,
        node.payload.get_operator(),
        node.ty,
        if locations.is_empty() {
            String::new()
        } else {
            format!(" at {}", locations.join(", "))
        }
    )
}

/// Escape source text so it cannot split or terminate a generated line comment.
pub(crate) fn comment(text: &str) -> String {
    text.chars().flat_map(char::escape_default).collect()
}
