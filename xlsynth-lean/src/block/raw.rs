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
use xlsynth_pir::{
    ir::{self, NodePayload, NodeRef, Register, Type},
    ir_utils,
};

use crate::block::BlockContext;
use crate::local_expr::{LeanOperand, LocalRenderer, RenderError, write_elaboration_limits};
use crate::support::IR_SOURCE;

/// Emit a self-contained Lean model of a verified XLS block.
/// Return an error if the block uses an unsupported operation.
///
/// Pass a verified block borrowed from `package`, whose file table supplies
/// source locations in diagnostics and generated comments. For saved IR, run
/// native XLS verification and typed PIR validation on the same input.
/// Revalidate the package if you modify it before emission.
///
/// `State`, `Input`, and `Output` fields keep source names and exact widths.
/// Emission rejects a register, data port, or assertion field whose spelling
/// collides with a Lean-generated declaration (such as `Input.rec`). The error
/// identifies the structure and field so proofs can rely on source spellings.
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
    emit_block(&block).with_context(|| {
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
fn emit_block(block: &BlockContext<'_>) -> Result<String> {
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
         -- Source assertions are results, never assumptions.\n\n",
        comment(&graph.name),
        comment(block.package_name)
    );
    // Standalone block modules can be imported alongside function modules.
    // Both get their helpers from the same Lean declarations.
    source.push_str(&IR_SOURCE.replace("XlsynthLean.IR", "XlsBlock.IR"));
    source.push_str("\nnamespace XlsBlock\n\n");
    write_elaboration_limits(&mut source, [&graph.graph]);
    source.push_str(&record_type(
        "State",
        block
            .graph
            .registers
            .iter()
            .map(|r| (r.name.as_str(), &r.ty)),
    )?);
    source.push_str(&record_type(
        "Input",
        graph
            .input_ports()
            .map(|port| (graph.port_name(port), graph.port_type(port))),
    )?);
    source.push_str(&record_type(
        "Output",
        graph
            .output_ports()
            .map(|port| (graph.port_name(port), graph.port_type(port))),
    )?);
    source.push_str(&structure_header("Assertions"));
    for reference in &assertions {
        let node = graph.get_node(*reference);
        if let NodePayload::Assert { label, message, .. } = &node.payload {
            let field = ir::node_textual_id(graph, *reference);
            validate_lean_member("Assertions", &field)?;
            writeln!(
                source,
                "  -- id={}; label={}; message={}\n  {} : _root_.Bool",
                node.text_id,
                comment(label),
                comment(message),
                identifier(&field)?
            )?;
        }
    }
    source.push_str("  deriving Repr\n\n");
    source.push_str(&structure_header("Evaluation"));
    source.push_str(
        "  next : State\n\
         \x20 output : Output\n\
         \x20 assertions : Assertions\n\
         \x20 deriving Repr\n\n\
         def eval (q : State) (x : Input) : Evaluation :=\n",
    );

    for reference in ir_utils::get_topological_nodes(&graph.nodes) {
        let node = graph.get_node(reference);
        if !matches!(node.payload, NodePayload::Nil) {
            let context = node_context(block, reference);
            let ty = lean_type(&node.ty).with_context(|| context.clone())?;
            let expression = node_expression(block, reference, |operand| {
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
        record_value(&next_fields),
        record_value(&output_fields),
        record_value(&assertion_fields)
    )?;
    source.push_str(
        "def step (q : State) (x : Input) : State := (eval q x).next\n\n\
         def observe (q : State) (x : Input) : Output := (eval q x).output\n\n\
         def assertions (q : State) (x : Input) : Assertions := (eval q x).assertions\n\n",
    );
    let holds = assertion_fields
        .iter()
        .rev()
        .fold("True".to_string(), |rest, (name, _)| {
            format!("And (checks.{name} = true) ({rest})")
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
            "  eval q {{ x with {} := BitVec.ofNat 1 {} }}",
            identifier(graph.port_name(reset.port))?,
            u8::from(!reset.active_low)
        )?;
    } else {
        source.push_str("  eval q x\n");
    }
    source.push_str("\nend XlsBlock\n");
    Ok(source)
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

/// Lean 4.34.1 generates these declarations in the same namespace as field
/// projections. Quoting a source name preserves its spelling and the collision.
pub(super) fn validate_lean_member(owner: &str, field: &str) -> Result<()> {
    ensure!(
        !matches!(
            field,
            "rec"
                | "recOn"
                | "casesOn"
                | "noConfusion"
                | "noConfusionType"
                | "ctorIdx"
                | "_sizeOf_1"
                | "_sizeOf_inst"
        ),
        "cannot emit {owner}.{field}: Lean generates a declaration with this name"
    );
    Ok(())
}

/// Keep the constructor in a separate namespace so a field can be named `mk`.
pub(super) fn structure_header(name: &str) -> String {
    format!("structure {name} where\n  Constructors.mk ::\n")
}

/// Declare fields in caller order, retaining names through Lean quotation.
fn record_type<'a>(
    name: &str,
    fields: impl Iterator<Item = (&'a str, &'a Type)>,
) -> Result<String> {
    let mut source = structure_header(name);
    for (field, ty) in fields {
        validate_lean_member(name, field)?;
        writeln!(
            source,
            "  {} : {}",
            identifier(field).with_context(|| format!("{name} field {field}"))?,
            lean_type(ty).with_context(|| format!("{name} field {field}"))?
        )?;
    }
    source.push_str("  deriving Repr\n\n");
    Ok(source)
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
    format!("({value} == (BitVec.ofNat 1 1))")
}

/// Render one supported graph operation using caller-supplied operand
/// expressions.
///
/// Resolve only the operands used by this operation. Register writes and
/// output ports return unit here; their effects are collected by `emit_block`.
pub(crate) fn node_expression(
    block: &BlockContext<'_>,
    reference: NodeRef,
    operand: impl Fn(NodeRef) -> Result<LeanOperand>,
) -> Result<String> {
    let graph = &*block.graph;
    let node = graph.get_node(reference);
    match &node.payload {
        NodePayload::InputPort { name, .. } => Ok(format!("x.{}", identifier(name)?)),
        NodePayload::RegisterRead { register } => {
            ensure!(
                graph.registers.iter().any(|r| r.name == *register),
                "no declaration for register {register}"
            );
            Ok(format!("q.{}", identifier(register)?))
        }
        // These effects are collected by emit_block, including disconnected
        // assertions. The output and write nodes themselves have unit type.
        NodePayload::OutputPort { .. }
        | NodePayload::RegisterWrite { .. }
        | NodePayload::AfterAll(_) => Ok("()".to_string()),
        NodePayload::Assert { token, .. } => Ok(operand(*token)?.into_argument()),
        other => LocalRenderer::Block
            .local_rhs(&graph.graph, reference, operand)
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
            "(if {} == (BitVec.ofNat 1 {}) then {} else {enabled})",
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
