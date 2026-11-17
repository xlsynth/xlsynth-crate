// SPDX-License-Identifier: Apache-2.0

//! Direct Lean emission from verified, typed XLS block IR.
//!
//! The generated `XlsBlock.eval` evaluates every node from old registers and
//! current inputs. Its result contains old-state outputs, simultaneous next
//! registers, and every source assertion, including disconnected assertions.
//! No assertion narrows the input/state domain. Tuples remain typed products;
//! no enum decoding or payload projection is part of this translation.
//! Each reachable IR function becomes a definition under `Helpers` in the
//! block's namespace. A caller's physical equations take its direct callees
//! as arguments so a checked model can supply source-typed calls at those
//! sites.

use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use xlsynth_pir::{
    ir::{self, NodePayload, NodeRef, Register, Type},
    ir_utils,
};

use crate::LeanModuleName;
use crate::block::BlockContext;
use crate::block::segments::{Segments, Stage, bundle_value, is_interface_projection, projection};
use crate::local_expr::{LeanOperand, LocalRenderer, RenderError, write_body_elaboration_limits};
use crate::support::write_file;

/// Emit a verified XLS block in the `XlsBlock` namespace.
///
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
    let block = BlockContext::new(package, block)?;
    let mut module = emit_with_context(&block, "XlsBlock")?;
    super::library_notation::add_to_module(&mut module);
    Ok(module)
}

/// Write the generated block under `root` and return the path of the Lean
/// source file.
///
/// The module name determines both the file path and the namespace, so blocks
/// with different module names can be imported together. The project must
/// supply the `XlsynthLean.IR` import. Errors from [`emit`]'s block checks and
/// from writing the destination file are propagated to the caller.
pub fn write_to(
    package: &ir::Package,
    block: &ir::Block,
    module_name: &LeanModuleName,
    root: &Path,
) -> Result<PathBuf> {
    let block = BlockContext::new(package, block)?;
    let mut source = emit_with_context(&block, &module_name.lean_identifier())?;
    super::library_notation::add_to_module(&mut source);
    let path = module_name.source_path();
    write_file(root, &path, source.as_bytes())?;
    Ok(root.join(path))
}

/// Select the physical equation layout for both raw exports and checked
/// bundles.
pub(super) fn emit_with_context(block: &BlockContext<'_>, namespace: &str) -> Result<String> {
    emit_block(block, namespace, Segments::new(block).as_ref()).with_context(|| {
        format!(
            "Lean emission of block {} in package {}",
            block.graph.name, block.package_name
        )
    })
}

/// Assemble a clock-step model after checking the emitter's supported shapes.
///
/// Every register must have exactly one write. Outputs use the data operand
/// of each output port, whose graph node itself has the empty tuple type.
/// In a segmented block the wrapper reads computed results from the final
/// cut and forwards direct input or old-state projections separately.
fn emit_block(
    block: &BlockContext<'_>,
    namespace: &str,
    segments: Option<&Segments>,
) -> Result<String> {
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
    for (index, node) in graph.nodes.iter().enumerate() {
        if let NodePayload::RegisterRead { register } = &node.payload {
            ensure!(
                graph.registers.iter().any(|decl| decl.name == *register),
                "{}: no declaration for register {register}",
                node_context(block, NodeRef { index })
            );
        }
    }
    let names: Vec<_> = (0..graph.nodes.len())
        .map(|index| {
            let reference = NodeRef { index };
            node_name(block.source_graph(reference), reference)
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
    let largest_body = match segments {
        None => graph.nodes.len(),
        Some(plan) => graph
            .registers
            .len()
            .max(graph.input_ports().count())
            .max(graph.output_ports().count())
            .max(assertions.len())
            .max(plan.leaves.iter().map(Vec::len).max().unwrap_or(0)),
    };
    write_body_elaboration_limits(
        &mut source,
        std::iter::once((&graph.graph, largest_body))
            .chain(block.helpers.iter().map(|f| (&f.graph, f.nodes.len()))),
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
         \x20 deriving L.Repr\n\n",
    );
    let callees = direct_callees(graph);
    let stage_args = match segments {
        Some(plan) => write_stages(&mut source, namespace, block, plan, &names)?,
        None => String::new(),
    };
    if callees.is_empty() {
        source.push_str("def eval");
    } else {
        source.push_str("abbrev evalWith");
        write_call_binders(&mut source, namespace, &callees)?;
    }
    source.push_str(" (q : State) (x : Input) : Evaluation :=\n");
    let entry_value = |reference| {
        // The first cut contains old registers, input ports, or the unit
        // value of a write/output used as a data operand.
        node_expression(namespace, graph, reference, |_| {
            unreachable!("entry values have no operands")
        })
        .with_context(|| node_context(block, reference))
    };
    let finished = if let Some(plan) = segments {
        let initial = bundle_value(&plan.cuts[0], entry_value)?;
        writeln!(
            source,
            "  let xls_finished := {} {initial}{stage_args}",
            plan.root().name()
        )?;
        Some(&plan.cuts[plan.root().end])
    } else {
        for reference in ir_utils::get_topological_nodes(&graph.nodes) {
            if !matches!(graph.get_node(reference).payload, NodePayload::Nil) {
                write_node_binding(
                    &mut source,
                    namespace,
                    block,
                    reference,
                    &names[reference.index],
                    |arg| names[arg.index].clone(),
                )?;
            }
        }
        None
    };

    let result_operand = |reference: NodeRef| -> Result<String> {
        match finished {
            None => Ok(names[reference.index].clone()),
            Some(_) if is_interface_projection(graph, reference) => {
                interface_value(namespace, graph, reference)
            }
            Some(wires) => {
                let index = wires
                    .binary_search_by_key(&reference.index, |wire| wire.index)
                    .ok()
                    .with_context(|| {
                        format!("missing result wire: {}", node_context(block, reference))
                    })?;
                Ok(projection(wires.len(), index, "xls_finished"))
            }
        }
    };
    let next_fields = block
        .graph
        .registers
        .iter()
        .zip(writes)
        .map(|(register, reference)| {
            let expression = register_update(block, register, reference, result_operand)
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
                result_operand(graph.output_value(port))?,
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
                predicate(&result_operand(activate)?),
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
    if !callees.is_empty() {
        writeln!(
            source,
            "def eval := evalWith{}\n",
            raw_call_arguments(namespace, &callees)?
        )?;
    }
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

/// Qualify a physical helper call with the block's absolute Lean namespace.
pub(super) fn helper_path(namespace: &str, physical: &str) -> Result<String> {
    Ok(format!(
        "_root_.{namespace}.Helpers.{}",
        helper_identifier(physical)?
    ))
}

/// Absolute name of the physical equations of a helper with direct calls.
pub(super) fn body_path(namespace: &str, physical: &str) -> Result<String> {
    Ok(format!(
        "_root_.{namespace}.Bodies.{}",
        helper_identifier(physical)?
    ))
}

/// Distinct direct calls in IR node order, also the order of a body's
/// parameters.
pub(super) fn direct_callees(graph: &ir::NodeGraph) -> Vec<&str> {
    let mut seen = BTreeSet::new();
    graph
        .nodes
        .iter()
        .filter_map(callee_name)
        .filter(|name| seen.insert(*name))
        .collect()
}

fn callee_name(node: &ir::Node) -> Option<&str> {
    match &node.payload {
        NodePayload::Invoke { to_apply, .. } => Some(to_apply),
        NodePayload::CountedFor { body, .. } => Some(body),
        _ => None,
    }
}

/// The space distinguishes callback parameters from every generated wire name.
fn call_parameter(physical: &str) -> String {
    format!("«call {physical}»")
}

/// The already emitted raw helper determines the exact physical Lean signature.
fn write_call_binders(out: &mut String, namespace: &str, callees: &[&str]) -> Result<()> {
    for &name in callees {
        write!(
            out,
            " ({} : type_of% {})",
            call_parameter(name),
            helper_path(namespace, name)?
        )?;
    }
    Ok(())
}

fn raw_call_arguments(namespace: &str, callees: &[&str]) -> Result<String> {
    let mut args = String::new();
    for &name in callees {
        write!(args, " {}", helper_path(namespace, name)?)?;
    }
    Ok(args)
}

/// Retain the IR name, with the graph index distinguishing normalized
/// spellings.
fn node_name(graph: &ir::NodeGraph, reference: NodeRef) -> String {
    let suffix: String = ir::node_textual_id(graph, reference)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("xls_node_{}_{suffix}", reference.index)
}

/// Choose the same Lean name for a physical helper's definition and uses.
///
/// Lean reserves `_root_`. Its replacement contains spaces, which
/// [`identifier`] rejects in IR names, so another helper cannot collide with
/// it.
pub(super) fn helper_identifier(physical: &str) -> Result<String> {
    if physical == "_root_" {
        Ok("«xls helper _root_»".into())
    } else {
        identifier(physical)
    }
}

/// Emit one copy of the equations and instantiate any calls with raw helpers.
fn write_helper(source: &mut String, namespace: &str, function: &ir::Fn) -> Result<()> {
    let name = helper_identifier(&function.name)?;
    let callees = direct_callees(function);
    if callees.is_empty() {
        write!(source, "def Helpers.{name}")?;
    } else {
        write!(source, "abbrev Bodies.{name}")?;
        write_call_binders(source, namespace, &callees)?;
    }
    let names = (0..function.nodes.len())
        .map(|index| node_name(function, NodeRef { index }))
        .collect::<Vec<_>>();
    for param in &function.params {
        write!(
            source,
            " ({} : {})",
            names[param.index],
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
                Ok(LeanOperand::Atom(names[operand.index].clone()))
            })
            .with_context(|| format!("function {}, node {}", function.name, node.text_id))?;
            writeln!(
                source,
                "  -- {} [id={}]\n  let {} : {} := {expr}",
                comment(&ir::node_textual_id(function, reference)),
                node.text_id,
                names[reference.index],
                lean_type(&node.ty)?
            )?;
        }
    }
    let result = function.ret_node_ref.context("helper has no return node")?;
    writeln!(source, "  {}\n", names[result.index])?;
    if !callees.is_empty() {
        writeln!(
            source,
            "def Helpers.{name} := {}{}\n",
            body_path(namespace, &function.name)?,
            raw_call_arguments(namespace, &callees)?
        )?;
    }
    Ok(())
}

/// Emit each computation once in a bounded leaf, passing live values directly
/// across its boundary. Internal stages compose two earlier definitions.
fn write_stages(
    source: &mut String,
    namespace: &str,
    block: &BlockContext<'_>,
    plan: &Segments,
    names: &[String],
) -> Result<String> {
    let leaf_callees = plan
        .leaves
        .iter()
        .map(|leaf| {
            leaf.iter()
                .filter_map(|&r| callee_name(block.graph.get_node(r)))
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    let stage_calls = |stage: Stage| -> Result<(String, String)> {
        let callees: BTreeSet<_> = leaf_callees[stage.start..stage.end]
            .iter()
            .flatten()
            .copied()
            .collect();
        let mut binders = String::new();
        let mut args = String::new();
        for name in callees {
            let param = call_parameter(name);
            write!(binders, " ({param} := {})", helper_path(namespace, name)?)?;
            write!(args, " {param}")?;
        }
        Ok((binders, args))
    };
    plan.write_definitions(source, block, stage_calls, |source, leaf| {
        let incoming = &plan.cuts[leaf];
        let value = |reference: NodeRef| match incoming
            .binary_search_by_key(&reference.index, |r| r.index)
        {
            Ok(index) => projection(incoming.len(), index, "w"),
            Err(_) => names[reference.index].clone(),
        };
        for &reference in &plan.leaves[leaf] {
            write_node_binding(
                source,
                namespace,
                block,
                reference,
                &names[reference.index],
                value,
            )?;
        }
        let result = plan.leaf_result(leaf, |reference| Ok(value(reference)))?;
        writeln!(source, "  {result}\n")?;
        Ok(())
    })?;
    Ok(stage_calls(plan.root())?.1)
}

/// Emit one typed raw let using the caller's names or cut projections for its
/// operands.
fn write_node_binding(
    source: &mut String,
    namespace: &str,
    block: &BlockContext<'_>,
    reference: NodeRef,
    name: &str,
    operand: impl Fn(NodeRef) -> String,
) -> Result<()> {
    let context = node_context(block, reference);
    let ty = lean_type(block.graph.get_node_ty(reference)).with_context(|| context.clone())?;
    let expression = node_expression(namespace, &block.graph, reference, |arg| {
        Ok(LeanOperand::Atom(operand(arg)))
    })
    .with_context(|| context.clone())?;
    writeln!(source, "  -- {}", comment(&context))?;
    writeln!(source, "  let {name} : {ty} := {expression}")?;
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

/// Place the constructor in a separate namespace so a source field can keep the
/// name `mk`.
pub(super) fn structure_header(name: &str) -> String {
    format!("structure {name} where\n  Constructors.mk ::\n")
}

/// Return the structure that owns the actual fields and constructor for this
/// public name.
pub(super) fn record_backing_name(name: &str, fields: &[(String, String)]) -> String {
    if has_generated_field(fields) {
        format!("«xls:implementation».{name}")
    } else {
        name.to_owned()
    }
}

/// Declare a Lean record and expose its fields in the supplied namespace.
///
/// If a field conflicts with a Lean-generated member, the actual structure
/// lives under a name containing a colon, which cannot occur in a BX or DSLX
/// identifier. The public type is then an abbreviation with
/// an explicit accessor for every field. Each accessor takes the fully
/// qualified public type so an earlier accessor such as `Packet.Packet` cannot
/// hide it. This form rejects `_root_` as a field because Lean reserves that
/// declaration name.
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

/// Return the underlying field identifier for declarations and Lean record
/// updates.
pub(super) fn storage_field(field: &str) -> String {
    let spelling = field.trim_matches(['«', '»']);
    if generated_member(spelling) {
        format!("«field:{spelling}»")
    } else {
        field.to_owned()
    }
}

/// Decide whether these rendered field names require positional construction.
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
    /// Write one quoted field and its root-qualified type in declaration order.
    ///
    /// Lean elaborates types in the scope of preceding fields. After a field
    /// named `_root_`, clear that local binding when resolving later types.
    /// Hardware field types are independent of earlier field values.
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

/// Emit a physical record in the block namespace, preserving IR field names and
/// order.
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

/// Interpret a rendered one-bit value as the Boolean condition that it is set.
fn predicate(value: &str) -> String {
    format!("({value} == (L.BitVec.ofNat 1 1))")
}

/// Render one supported graph operation using caller-supplied operand
/// expressions.
///
/// Resolve only the operands used by this operation. Register writes and
/// output ports return unit here; their effects are collected by `emit_block`.
/// For a counted loop, bind the invariant operands in IR order, leaving the
/// body's index and carry parameters for the fold to supply on each iteration.
fn node_expression(
    namespace: &str,
    graph: &ir::NodeGraph,
    reference: NodeRef,
    operand: impl Fn(NodeRef) -> Result<LeanOperand>,
) -> Result<String> {
    let node = graph.get_node(reference);
    match &node.payload {
        NodePayload::InputPort { name, .. } => Ok(format!("x.{}", identifier(name)?)),
        NodePayload::Invoke { to_apply, operands } => {
            let mut call = call_parameter(to_apply);
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
            let mut step = call_parameter(body);
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

/// Replay a verified field projection against the public inputs or old state.
fn interface_value(namespace: &str, graph: &ir::NodeGraph, reference: NodeRef) -> Result<String> {
    node_expression(namespace, graph, reference, |operand| {
        interface_value(namespace, graph, operand).map(LeanOperand::Expression)
    })
}

/// Compute one next-state field with reset taking precedence over load enable.
///
/// The reference must be the selected write for this register. Each operand
/// is evaluated from old state.
fn register_update(
    block: &BlockContext<'_>,
    register: &Register,
    reference: NodeRef,
    operand: impl Fn(NodeRef) -> Result<String>,
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
    let data = operand(*arg)?;
    let enabled = if let Some(enable) = load_enable {
        format!(
            "(if {} then {} else q.{})",
            predicate(&operand(*enable)?),
            data,
            identifier(&register.name)?
        )
    } else {
        data
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
            operand(*signal)?,
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
