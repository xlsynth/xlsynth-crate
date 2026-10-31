// SPDX-License-Identifier: Apache-2.0

//! Direct Lean emission from verified, typed XLS block IR.
//!
//! The generated `XlsBlock.eval` evaluates every node from old registers and
//! current inputs. Its result contains old-state outputs, simultaneous next
//! registers, and every source assertion, including disconnected assertions.
//! No assertion narrows the input/state domain. Tuples remain typed products;
//! no enum decoding or payload projection is part of this translation.

use anyhow::{Context, Result, bail, ensure};
use std::{collections::BTreeSet, fmt::Write};
use xlsynth_ir_value::{IrFormatPreference, IrValue};
use xlsynth_pir::{
    ir::{self, Binop, NaryOp, NodePayload, NodeRef, Register, Type, Unop},
    ir_utils,
};

use crate::block::BlockContext;

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

/// Emit a self-contained Lean model of a verified XLS block.
/// Return an error if the block uses an unsupported operation.
///
/// Pass a verified block borrowed from `package`, whose file table supplies
/// source locations in diagnostics and generated comments. For saved IR, run
/// native XLS verification and typed PIR validation on the same input.
/// Revalidate the package if you modify it before emission.
///
/// `State`, `Input`, and `Output` fields keep source names and exact widths.
/// An XLS tuple `[a, b]` is represented by `(a, (b, ()))`; element `j` uses
/// `j` second projections followed by a first projection. Tokens are `Unit`.
/// `reset q x` forces only the reset port and evaluates an ordinary clock; it
/// does not invent an initial value for unreset storage or clear old outputs.
/// One-hot selection supports bits only; ordinary and priority selects retain
/// tuple structure, matching the native reference evaluator's supported subset.
/// Records derive `Repr`, but whole-record `DecidableEq` is left to callers:
/// deriving it exceeds Lean's instance-search size limit on nested state.
/// Unsupported nodes, types, or reset/instance shapes return a source error.
/// Do not treat `reset` as a constructor for initialized state: registers
/// without a reset value still depend on the supplied old state.
pub fn emit(package: &ir::Package, block: &ir::Block) -> Result<String> {
    emit_with_context(&BlockContext::new(package, block)?)
}

pub(super) fn emit_with_context(block: &BlockContext<'_>) -> Result<String> {
    emit_block(block).with_context(|| {
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

    let graph = block.graph;
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
            let source_name = ir::node_textual_id(graph, NodeRef { index });
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
         -- Source assertions are results, never assumptions.\n\
         import Std\n\nnamespace XlsBlock\n\n",
        comment(&graph.name),
        comment(block.package_name)
    );
    if graph
        .nodes
        .iter()
        .chain(block.helpers.iter().flat_map(|helper| &helper.nodes))
        .any(|node| matches!(node.payload, NodePayload::CountedFor { .. }))
    {
        source.push_str(
            "def countedFor {width : Nat} {α : Type} (tripCount stride : Nat)
    (body : BitVec width → α → α) (init : α) : α :=
  Nat.fold tripCount (fun iteration _ carry => body (BitVec.ofNat width (iteration * stride)) carry) init

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
            "def bitSliceUpdate {n s w : Nat} (value : BitVec n) (start : BitVec s) (update : BitVec w) : BitVec n :=\n  let offset := min start.toNat n\n  let mask := ((BitVec.allOnes w).setWidth n) <<< offset\n  (value &&& ~~~mask) ||| ((update.setWidth n) <<< offset)\n\n",
        );
    }
    for helper in &block.helpers {
        write_helper(&mut source, helper)?;
    }
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
    let assertion_fields: Vec<_> = assertions
        .iter()
        .map(|reference| ir::node_textual_id(graph, *reference))
        .collect();
    source.push_str(&structure_header(
        "Assertions",
        assertion_fields.iter().map(String::as_str),
    )?);
    let mut fields = RecordFields::default();
    for reference in &assertions {
        let node = graph.get_node(*reference);
        if let NodePayload::Assert { label, message, .. } = &node.payload {
            writeln!(
                source,
                "  -- id={}; label={}; message={}",
                node.text_id,
                comment(label),
                comment(message)
            )?;
            fields.write(
                &mut source,
                &identifier(&ir::node_textual_id(graph, *reference))?,
                "_root_.Bool",
            )?;
        }
    }
    source.push_str(
        "  deriving Repr\n\n\
         structure Evaluation where\n\
         \x20 next : State\n\
         \x20 output : Output\n\
         \x20 assertions : Assertions\n\
         \x20 deriving Repr\n\n\
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
            let expression =
                node_expression(graph, reference, |operand| Ok(names[operand.index].clone()))
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

pub(super) fn helper_path(physical: &str) -> Result<String> {
    Ok(format!("_root_.XlsBlock.Helpers.{}", identifier(physical)?))
}

fn write_helper(source: &mut String, function: &ir::Fn) -> Result<()> {
    write!(source, "def Helpers.{}", identifier(&function.name)?)?;
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
            let expr = node_expression(function, reference, |operand| {
                Ok(format!("n{}", operand.index))
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

/// Render physical bit, token, tuple, and nonempty array types in Lean.
pub(crate) fn lean_type(ty: &Type) -> Result<String> {
    physical_type(ty, "")
}

pub(super) fn physical_field_type(ty: &Type) -> Result<String> {
    physical_type(ty, "_root_.")
}

fn physical_type(ty: &Type, prefix: &str) -> Result<String> {
    match ty {
        Type::Bits(width) => {
            ensure!(*width > 0, "bits[0] is outside the supported Lean subset");
            Ok(format!("{prefix}BitVec {width}"))
        }
        Type::Tuple(elements) => {
            elements
                .iter()
                .rev()
                .try_fold(format!("{prefix}Unit"), |tail, element| {
                    Ok(format!(
                        "{prefix}Prod ({}) ({tail})",
                        physical_type(element, prefix)?
                    ))
                })
        }
        Type::Token => Ok(format!("{prefix}Unit")),
        Type::Array(array) => {
            ensure!(
                array.element_count > 0,
                "zero-length arrays are unsupported"
            );
            Ok(format!(
                "{prefix}Vector ({}) {}",
                physical_type(&array.element_type, prefix)?,
                array.element_count
            ))
        }
    }
}

/// Lean generates these declarations for every structure and inductive type.
/// Quoting a source name preserves its spelling but does not prevent a clash.
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

/// Retain the default constructor unless it would occupy a source field's name.
pub(super) fn structure_header<'a>(
    name: &str,
    fields: impl Iterator<Item = &'a str>,
) -> Result<String> {
    let fields: BTreeSet<_> = fields.collect();
    for field in &fields {
        validate_lean_member(name, field)?;
    }
    let mut header = format!("structure {name} where\n");
    if fields.contains("mk") {
        let mut constructor = String::from("mk_");
        while fields.contains(constructor.as_str()) {
            constructor.push('_');
        }
        writeln!(header, "  {constructor} ::")?;
    }
    Ok(header)
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
    name: &str,
    fields: impl Iterator<Item = (&'a str, &'a Type)>,
) -> Result<String> {
    let fields: Vec<_> = fields.collect();
    let mut source = structure_header(name, fields.iter().map(|(field, _)| *field))?;
    let mut scope = RecordFields::default();
    for (field, ty) in fields {
        scope.write(
            &mut source,
            &identifier(field).with_context(|| format!("{name} field {field}"))?,
            &physical_field_type(ty).with_context(|| format!("{name} field {field}"))?,
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

/// Preserve tuple order in nested pairs terminated by the unit value.
pub(crate) fn tuple_value(elements: impl DoubleEndedIterator<Item = String>) -> String {
    elements.rev().fold("()".to_string(), |tail, element| {
        format!("({element}, {tail})")
    })
}

/// Project a zero-based tuple position using the nested-pair representation.
pub(crate) fn tuple_element(tuple: &str, index: usize) -> String {
    format!("({tuple}){}.1", ".2".repeat(index))
}

/// Render a physical literal whose type and shape were checked by the parser.
pub(crate) fn literal(ty: &Type, value: &IrValue) -> Result<String> {
    match ty {
        Type::Bits(width) => Ok(format!(
            "(BitVec.ofNat {width} {})",
            value
                .to_bits()
                .context("literal bits")?
                .to_string_fmt(IrFormatPreference::Hex, false)
                .replace('_', "")
        )),
        Type::Tuple(types) => {
            let values = value.get_elements().context("tuple literal elements")?;
            let elements = types
                .iter()
                .zip(&values)
                .map(|(ty, value)| literal(ty, value))
                .collect::<Result<Vec<_>>>()?;
            Ok(tuple_value(elements.into_iter()))
        }
        Type::Token => Ok("()".to_string()),
        Type::Array(array) => {
            let elements = value
                .get_elements()?
                .iter()
                .map(|item| literal(&array.element_type, item))
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("#v[{}]", elements.join(", ")))
        }
    }
}

/// XLS clamps reads to the last element of each indexed dimension.
pub(super) fn array_read(array: &str, index: &str) -> String {
    format!("(_root_.Vector.getD ({array}) ({index}).toNat (_root_.Vector.back ({array})))")
}

/// Updating an out-of-bounds index leaves the whole array unchanged.
pub(super) fn array_write(array: &str, indices: &[String], value: &str) -> String {
    if let Some((index, remaining)) = indices.split_first() {
        let item = if remaining.is_empty() {
            value.to_owned()
        } else {
            array_write(&array_read(array, index), remaining, value)
        };
        format!("(_root_.Vector.setIfInBounds ({array}) ({index}).toNat ({item}))")
    } else {
        value.to_owned()
    }
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
    graph: &ir::NodeGraph,
    reference: NodeRef,
    operand: impl Fn(NodeRef) -> Result<String>,
) -> Result<String> {
    let node = graph.get_node(reference);
    match &node.payload {
        NodePayload::InputPort { name, .. } => Ok(format!("x.{}", identifier(name)?)),
        // Output values are collected separately in declaration order. The
        // graph-resident output operation itself has the empty tuple type.
        NodePayload::OutputPort { .. } => Ok("()".to_string()),
        NodePayload::Literal(value) => literal(&node.ty, value),
        NodePayload::Invoke { to_apply, operands } => {
            let mut call = helper_path(to_apply)?;
            for arg in operands {
                write!(call, " ({})", operand(*arg)?)?;
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
            let mut step = helper_path(body)?;
            if !invariant_args.is_empty() {
                step.push_str(" · ·");
                for arg in invariant_args {
                    write!(step, " ({})", operand(*arg)?)?;
                }
            }
            Ok(format!(
                "_root_.XlsBlock.countedFor {trip_count} {stride} ({step}) ({})",
                operand(*init)?
            ))
        }
        NodePayload::RegisterRead { register } => Ok(format!("q.{}", identifier(register)?)),
        NodePayload::RegisterWrite { .. } => Ok("()".to_string()),
        NodePayload::Tuple(args) => Ok(tuple_value(
            args.iter()
                .map(|r| operand(*r))
                .collect::<Result<Vec<_>>>()?
                .into_iter(),
        )),
        NodePayload::TupleIndex { tuple, index } => Ok(tuple_element(&operand(*tuple)?, *index)),
        NodePayload::Array(args) => {
            let elements = args
                .iter()
                .map(|arg| operand(*arg))
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("#v[{}]", elements.join(", ")))
        }
        NodePayload::ArrayIndex { array, indices, .. } => {
            indices.iter().try_fold(operand(*array)?, |items, index| {
                Ok(array_read(&items, &operand(*index)?))
            })
        }
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            ..
        } => {
            let indices = indices
                .iter()
                .map(|index| operand(*index))
                .collect::<Result<Vec<_>>>()?;
            Ok(array_write(&operand(*array)?, &indices, &operand(*value)?))
        }
        NodePayload::Unop(Unop::Neg, arg) => Ok(format!("BitVec.neg {}", operand(*arg)?)),
        NodePayload::Unop(Unop::Not, arg) => Ok(format!("BitVec.not {}", operand(*arg)?)),
        NodePayload::Binop(Binop::Add, lhs, rhs) => {
            Ok(format!("BitVec.add {} {}", operand(*lhs)?, operand(*rhs)?))
        }
        NodePayload::Binop(Binop::Shll, lhs, rhs) => {
            let Type::Bits(width) = &node.ty else {
                bail!("expected bits result for shll");
            };
            // Larger shifts also yield zero. Bound the intermediate Nat shift.
            Ok(format!(
                "BitVec.shiftLeft {} (_root_.Nat.min (BitVec.toNat {}) {width})",
                operand(*lhs)?,
                operand(*rhs)?
            ))
        }
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) => {
            let operator = if *op == Binop::Eq { "==" } else { "!=" };
            Ok(format!(
                "BitVec.ofBool ({} {operator} {})",
                operand(*lhs)?,
                operand(*rhs)?
            ))
        }
        NodePayload::Nary(NaryOp::Concat, args) => {
            let (first, rest) = args.split_first().context("concat has no operands")?;
            rest.iter().try_fold(operand(*first)?, |left, right| {
                Ok(format!("(BitVec.append {left} {})", operand(*right)?))
            })
        }
        NodePayload::Nary(op @ (NaryOp::And | NaryOp::Or | NaryOp::Nor | NaryOp::Xor), args) => {
            let (first, rest) = args
                .split_first()
                .context("bitwise operation has no operands")?;
            let operator = match op {
                NaryOp::And => "and",
                NaryOp::Xor => "xor",
                _ => "or",
            };
            let reduced = rest.iter().try_fold(operand(*first)?, |left, right| {
                Ok::<_, anyhow::Error>(format!("(BitVec.{operator} {left} {})", operand(*right)?))
            })?;
            Ok(if *op == NaryOp::Nor {
                format!("BitVec.not {reduced}")
            } else {
                reduced
            })
        }
        NodePayload::Encode { arg } => {
            let Type::Bits(input_width) = graph.get_node(*arg).ty else {
                bail!("encode input must be bits");
            };
            let Type::Bits(width) = node.ty else {
                bail!("encode result must be bits");
            };
            // For multi-hot inputs XLS ORs the indices of all set bits.
            Ok(format!(
                "((List.range {input_width}).foldl (fun acc i => if ({}).getLsbD i then BitVec.or acc (BitVec.ofNat {width} i) else acc) (BitVec.ofNat {width} 0))",
                operand(*arg)?
            ))
        }
        NodePayload::BitSlice { arg, start, width } => Ok(format!(
            "BitVec.extractLsb' {start} {width} {}",
            operand(*arg)?
        )),
        NodePayload::DynamicBitSlice { arg, start, width } => {
            let Type::Bits(input_width) = graph.get_node(*arg).ty else {
                bail!("dynamic_bit_slice requires a bit-vector input");
            };
            Ok(format!(
                "BitVec.extractLsb' (_root_.Nat.min (BitVec.toNat {}) {input_width}) {width} {}",
                operand(*start)?,
                operand(*arg)?
            ))
        }
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => Ok(format!(
            "_root_.XlsBlock.bitSliceUpdate ({}) ({}) ({})",
            operand(*arg)?,
            operand(*start)?,
            operand(*update_value)?
        )),
        NodePayload::ZeroExt { arg, new_bit_count } => Ok(format!(
            "BitVec.setWidth {new_bit_count} {}",
            operand(*arg)?
        )),
        NodePayload::SignExt { arg, new_bit_count } => Ok(format!(
            "BitVec.signExtend {new_bit_count} {}",
            operand(*arg)?
        )),
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            // Native/PIR verification checks complete coverage when no default
            // exists; only that final covered case becomes the last branch.
            let (fallback, tested) = if let Some(default) = default {
                (*default, cases.as_slice())
            } else {
                let (last, rest) = cases.split_last().context("sel has no cases")?;
                (*last, rest)
            };
            let selector = operand(*selector)?;
            tested.iter().enumerate().rev().try_fold(
                operand(fallback)?,
                |otherwise, (index, case)| {
                    Ok(format!(
                        "(if {selector}.toNat == {index} then {} else {otherwise})",
                        operand(*case)?
                    ))
                },
            )
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let fallback = default.context("priority_sel requires an explicit default")?;
            let selector = operand(*selector)?;
            cases.iter().enumerate().rev().try_fold(
                operand(fallback)?,
                |otherwise, (index, case)| {
                    Ok(format!(
                        "(if {selector}.getLsbD {index} then {} else {otherwise})",
                        operand(*case)?
                    ))
                },
            )
        }
        NodePayload::OneHot { arg, lsb_prio } => {
            let input = operand(*arg)?;
            let candidate = if *lsb_prio {
                input.clone()
            } else {
                format!("(BitVec.reverse {input})")
            };
            let low = format!("(BitVec.and {candidate} (BitVec.neg {candidate}))");
            let selected = if *lsb_prio {
                low
            } else {
                format!("(BitVec.reverse {low})")
            };
            Ok(format!(
                "BitVec.append (BitVec.ofBool ({input} == 0)) {selected}"
            ))
        }
        NodePayload::OneHotSel { selector, cases } => {
            let Type::Bits(width) = &node.ty else {
                bail!("only bits-valued one_hot_sel is supported by Lean")
            };
            let selector = operand(*selector)?;
            cases.iter().enumerate().try_fold(
                format!("(BitVec.ofNat {width} 0)"),
                |merged, (index, case)| {
                    let case = operand(*case)?;
                    Ok(format!(
                        "(BitVec.or {merged} (if {selector}.getLsbD {index} then {case} else (BitVec.ofNat {width} 0)))"
                    ))
                },
            )
        }
        // Tokens have one value. All assertion predicates are returned
        // independently, so ignoring token data never drops an obligation.
        NodePayload::Assert { token, .. } => operand(*token),
        NodePayload::AfterAll(_) => Ok("()".to_string()),
        other => bail!("unsupported Lean operation {}", other.get_operator()),
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
            literal(&register.ty, value)?
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
    let node = block.graph.get_node(reference);
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
        ir::node_textual_id(block.graph, reference),
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
