// SPDX-License-Identifier: Apache-2.0

//! Source-shaped Lean values and executable expressions over a checked block.
//!
//! The selected DSLX signature supplies names and types. The block graph
//! supplies every equation and physical port. Codecs include undeclared
//! enum encodings so arbitrary physical state remains representable. This
//! module emits correspondence proof source but does not check it with Lean.
//!
//! Once checked by Lean, these proofs provide a sanity check on both the raw
//! and readable Lean models by establishing agreement on next state, outputs,
//! and assertion results. A translation error shared by both models can escape
//! this check; agreement between them does not establish agreement with XLS
//! execution.

use std::{collections::BTreeSet, fmt::Write};

mod loop_correspondence;

use anyhow::{Context, Result, bail, ensure};
use xlsynth_pir::{
    ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop},
    ir_utils,
};

use crate::block::{
    BlockContext,
    raw::{
        self as lean, LeanBlockBundle, identifier, lean_type, record_value, tuple_element,
        tuple_value,
    },
    source_types::{SourceType, StatefulInterface, TypeDecl, source_shape},
};

/// Source type annotations and collision-free names for one verified block.
///
/// All vectors use the checked graph's node or register order. Annotation
/// contexts memoize propagation through shared nodes; they are local to one
/// emission and cannot outlive the validated inputs.
struct Presentation<'a> {
    graph: &'a ir::NodeGraph,
    block: &'a BlockContext<'a>,
    declarations: &'a [TypeDecl],
    interface: &'a StatefulInterface,
    node_types: Vec<Option<SourceType>>,
    /// A shared graph node is traversed once for each distinct nominal context.
    annotation_contexts: Vec<Vec<SourceType>>,
    names: Vec<String>,
    register_fields: Vec<String>,
}

/// Emit five dependent modules after the raw emitter checks supported
/// operations.
pub(super) fn emit(
    block: &BlockContext<'_>,
    declarations: &[TypeDecl],
    interface: &StatefulInterface,
) -> Result<LeanBlockBundle> {
    let raw = lean::emit_with_context(block)?;
    let presentation = Presentation::new(block, declarations, interface, None)?;
    Ok(LeanBlockBundle {
        files: vec![
            ("Generated.lean".into(), raw),
            ("ReadableTypes.lean".into(), presentation.types_module()?),
            ("ReadableGenerated.lean".into(), presentation.executable()?),
            (
                "ReadableRepresentation.lean".into(),
                presentation.representation()?,
            ),
            (
                "ReadableCorrespondence.lean".into(),
                presentation.correspondence()?,
            ),
        ],
    })
}

impl<'a> Presentation<'a> {
    /// Allocate noncapturing names and infer types from checked register
    /// bindings.
    ///
    /// One-bit nodes use `Bool`; nominal types come from register reads,
    /// projections, and write destinations rather than matching widths alone.
    fn new(
        block: &'a BlockContext<'a>,
        declarations: &'a [TypeDecl],
        interface: &'a StatefulInterface,
        helper: Option<&'a ir::Fn>,
    ) -> Result<Self> {
        let graph: &ir::NodeGraph = match helper {
            Some(f) => f,
            None => block.graph,
        };
        let signature = helper.and_then(|f| interface.helpers.get(&f.name));
        let mut preferred: Vec<_> = graph.nodes.iter().map(|n| n.name.as_deref()).collect();
        if let Some((function, signature)) = helper.zip(signature) {
            for (param, (name, _)) in function.params.iter().zip(&signature.params) {
                preferred[param.index] = Some(name.as_str());
            }
            for (index, node) in graph.nodes.iter().enumerate() {
                if let Some((name, _)) = signature.node_bindings.get(&node.text_id) {
                    preferred[index] = Some(name);
                }
            }
        }
        let reserved: BTreeSet<_> = preferred.iter().flatten().copied().collect();
        let mut used: BTreeSet<String> = [
            "Bool",
            "BitVec",
            "Nat",
            "Unit",
            "Prod",
            "XlsSource",
            "XlsCodec",
            "XlsBlock",
            "XlsReadable",
            "_root_",
            "true",
            "false",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        if helper.is_none() {
            used.extend(["q".to_owned(), "x".to_owned()]);
        }
        let names = preferred
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                let name = if let Some(source) = source.filter(|s| !used.contains(*s)) {
                    source.to_owned()
                } else {
                    let mut fallback = format!("xls_node_{index}");
                    while reserved.contains(fallback.as_str()) || used.contains(&fallback) {
                        fallback.push('_');
                    }
                    fallback
                };
                used.insert(name.clone());
                identifier(&name)
            })
            .collect::<Result<Vec<_>>>()?;
        let register_fields = block
            .graph
            .registers
            .iter()
            .map(|r| identifier(&r.name))
            .collect::<Result<_>>()?;
        let mut result = Self {
            graph,
            block,
            declarations,
            interface,
            node_types: graph
                .nodes
                .iter()
                .map(|n| (n.ty == Type::Bits(1)).then_some(SourceType::Bool))
                .collect(),
            annotation_contexts: vec![Vec::new(); graph.nodes.len()],
            names,
            register_fields,
        };
        if let Some((function, signature)) = helper.zip(signature) {
            for (param, (_, ty)) in function.params.iter().zip(&signature.params) {
                result.node_types[param.index] = Some(ty.clone());
            }
            for (index, node) in graph.nodes.iter().enumerate() {
                if let Some((_, ty)) = signature.node_bindings.get(&node.text_id) {
                    result.node_types[index] = Some(ty.clone());
                }
            }
        }
        for reference in ir_utils::get_topological_nodes(&graph.nodes) {
            match &graph.get_node(reference).payload {
                NodePayload::CountedFor {
                    init,
                    body,
                    invariant_args,
                    ..
                } if let Some(signature) = interface.helpers.get(body) => {
                    result.node_types[reference.index] = Some(signature.result.clone());
                    result.annotate(*init, &signature.params[1].1)?;
                    for (arg, (_, ty)) in invariant_args.iter().zip(&signature.params[2..]) {
                        result.annotate(*arg, ty)?;
                    }
                }
                NodePayload::Invoke { to_apply, operands } => {
                    if let Some(signature) = interface.helpers.get(to_apply) {
                        result.node_types[reference.index] = Some(signature.result.clone());
                        for (arg, (_, ty)) in operands.iter().zip(&signature.params) {
                            result.annotate(*arg, ty)?;
                        }
                    }
                }
                NodePayload::InputPort { name, .. } => {
                    result.node_types[reference.index] = interface.inputs.get(name).cloned();
                }
                NodePayload::RegisterRead { register } => {
                    result.node_types[reference.index] = result.register_type(register)?;
                }
                NodePayload::ArrayIndex { array, indices, .. } => {
                    if let Some(ty) = &result.node_types[array.index] {
                        result.node_types[reference.index] =
                            Some(result.indexed_type(ty, indices.len())?.clone());
                    }
                }
                NodePayload::ArrayUpdate { array, .. } => {
                    if let Some(ty) = result.node_types[array.index].clone() {
                        result.annotate(reference, &ty)?;
                    }
                }
                NodePayload::TupleIndex { tuple, index } => {
                    if let Some(source_type) = &result.node_types[tuple.index]
                        && let SourceType::Struct { name } = result.unalias(source_type)?
                    {
                        let TypeDecl::Struct { fields, .. } = result.declaration(name)? else {
                            unreachable!("validated struct reference")
                        };
                        result.node_types[reference.index] =
                            Some(fields[*index].source_type.clone());
                    } else if let Some(source_type) = &result.node_types[tuple.index]
                        && let SourceType::Tuple { members } = result.unalias(source_type)?
                    {
                        result.node_types[reference.index] = Some(members[*index].clone());
                    }
                }
                _ => {
                    // Other nodes get their context from register destinations.
                }
            }
        }
        // Register destinations supply exact type context to tuple construction
        // and selections. A reused node can still be converted at another use.
        for node in &graph.nodes {
            if let NodePayload::RegisterWrite { arg, register, .. } = &node.payload
                && let Some(ty) = result.register_type(register)?
            {
                result.annotate(*arg, &ty)?;
            }
        }
        if let Some(function) = helper {
            if let Some(signature) = signature {
                result.annotate(
                    function.ret_node_ref.context("helper has no return node")?,
                    &signature.result,
                )?;
            }
        } else {
            for (port, ty) in block.graph.output_ports().zip(&interface.outputs) {
                result.annotate(block.graph.output_value(port), ty)?;
            }
        }
        Ok(result)
    }

    fn block(&self) -> &BlockContext<'a> {
        self.block
    }

    /// Resolve a nominal reference within this presentation's checked
    /// catalogue.
    fn declaration(&self, name: &str) -> Result<&TypeDecl> {
        self.declarations
            .iter()
            .find(|d| d.name() == name)
            .with_context(|| format!("validated source declaration {name}"))
    }

    fn register_type(&self, register: &str) -> Result<Option<SourceType>> {
        Ok(self.interface.registers.get(register).cloned())
    }

    /// Recover a declaration's nominal kind without erasing its name.
    fn named_type(&self, name: &str) -> Result<SourceType> {
        Ok(match self.declaration(name)? {
            TypeDecl::Enum { .. } => SourceType::Enum { name: name.into() },
            TypeDecl::Struct { .. } => SourceType::Struct { name: name.into() },
            TypeDecl::Alias { .. } => SourceType::Alias { name: name.into() },
        })
    }

    /// Follow the native-checked alias chain to its underlying source type.
    fn unalias<'b>(&'b self, ty: &'b SourceType) -> Result<&'b SourceType> {
        if let SourceType::Alias { name } = ty {
            let TypeDecl::Alias { target, .. } = self.declaration(name)? else {
                unreachable!("validated alias reference")
            };
            self.unalias(target)
        } else {
            Ok(ty)
        }
    }

    fn indexed_type<'b>(&'b self, mut ty: &'b SourceType, rank: usize) -> Result<&'b SourceType> {
        for _ in 0..rank {
            let SourceType::Array { element, .. } = self.unalias(ty)? else {
                anyhow::bail!("indexed source type is not an array: {ty:?}");
            };
            ty = element;
        }
        Ok(ty)
    }

    /// Propagate destination type context through tuples and selections once.
    ///
    /// A node retains its first inferred type, while each distinct context is
    /// propagated through its operands. Uses with another type convert through
    /// the physical representation instead of changing the shared binding.
    fn annotate(&mut self, reference: NodeRef, ty: &SourceType) -> Result<()> {
        if self.annotation_contexts[reference.index].contains(ty) {
            Ok(())
        } else {
            self.annotation_contexts[reference.index].push(ty.clone());
            if self.node_types[reference.index].is_none() {
                self.node_types[reference.index] = Some(ty.clone());
            }
            let payload = self.graph.get_node(reference).payload.clone();
            match payload {
                NodePayload::Array(args) => {
                    let element = self.indexed_type(ty, 1)?.clone();
                    for arg in args {
                        self.annotate(arg, &element)?;
                    }
                }
                NodePayload::ArrayUpdate {
                    array,
                    value,
                    indices,
                    ..
                } => {
                    let element = self.indexed_type(ty, indices.len())?.clone();
                    self.annotate(array, ty)?;
                    self.annotate(value, &element)?;
                }
                NodePayload::Tuple(args) => {
                    if let SourceType::Struct { name } = self.unalias(ty)? {
                        let TypeDecl::Struct { fields, .. } = self.declaration(name)? else {
                            unreachable!("validated struct reference")
                        };
                        let types: Vec<_> = fields.iter().map(|f| f.source_type.clone()).collect();
                        for (arg, ty) in args.into_iter().zip(types) {
                            self.annotate(arg, &ty)?;
                        }
                    } else if let SourceType::Tuple { members } = self.unalias(ty)? {
                        for (arg, ty) in args.into_iter().zip(members.clone()) {
                            self.annotate(arg, &ty)?;
                        }
                    }
                }
                NodePayload::Sel { cases, default, .. } => {
                    for arg in cases.into_iter().chain(default) {
                        self.annotate(arg, ty)?;
                    }
                }
                NodePayload::PrioritySel { cases, default, .. } => {
                    for arg in cases.into_iter().chain(default) {
                        self.annotate(arg, ty)?;
                    }
                }
                _ => {
                    // Non-structural expressions keep their inferred type.
                }
            }
            Ok(())
        }
    }

    /// Render a source type with qualified names that cannot capture built-ins.
    fn source_type(&self, ty: &SourceType) -> Result<String> {
        match ty {
            SourceType::Bool => Ok("_root_.Bool".into()),
            SourceType::Bits { width } => Ok(format!("_root_.BitVec {width}")),
            SourceType::Array { element, size } => Ok(format!(
                "_root_.Vector ({}) {size}",
                self.source_type(element)?
            )),
            SourceType::Tuple { members } => Ok(product_type(
                &members
                    .iter()
                    .map(|ty| self.source_type(ty))
                    .collect::<Result<Vec<_>>>()?,
                true,
            )),
            SourceType::Enum { name }
            | SourceType::Struct { name }
            | SourceType::Alias { name } => Ok(format!("_root_.XlsSource.{}", identifier(name)?)),
        }
    }

    /// Encode toward physical values or decode toward the requested source
    /// type.
    ///
    /// Absent annotations and plain bit vectors already use physical values.
    fn convert(&self, ty: Option<&SourceType>, value: &str, encode: bool) -> Result<String> {
        let operation = if encode { "encode" } else { "decode" };
        match ty {
            None | Some(SourceType::Bits { .. }) => Ok(value.into()),
            Some(SourceType::Bool) => Ok(format!("(XlsCodec.{operation}Bool ({value}))")),
            Some(SourceType::Array { element, .. }) => {
                let item = self.convert(Some(element), "item", encode)?;
                if item == "item" {
                    Ok(value.into())
                } else {
                    Ok(format!(
                        "(_root_.Vector.map (fun item => {item}) ({value}))"
                    ))
                }
            }
            Some(SourceType::Tuple { members }) => {
                let fields = members
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| {
                        let element = if encode {
                            product_element(value, index, members.len(), true)
                        } else {
                            tuple_element(value, index)
                        };
                        self.convert(Some(ty), &element, encode)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if encode {
                    Ok(tuple_value(fields.into_iter()))
                } else {
                    Ok(product_value(&fields, true))
                }
            }
            Some(
                SourceType::Enum { name }
                | SourceType::Struct { name }
                | SourceType::Alias { name },
            ) => Ok(format!(
                "(XlsCodec.{}.{operation} ({value}))",
                identifier(name)?
            )),
        }
    }

    /// Convert between annotations through their common physical
    /// representation.
    ///
    /// Callers supply contexts for the same graph value, so their shapes agree.
    fn coerce(
        &self,
        from: Option<&SourceType>,
        to: Option<&SourceType>,
        value: &str,
    ) -> Result<String> {
        if from == to {
            Ok(value.into())
        } else {
            self.convert(to, &self.convert(from, value, true)?, false)
        }
    }

    /// Add collision-free constructors for every unnamed physical enum
    /// encoding.
    ///
    /// Keeping unnamed encodings in the readable proof domain is provisional.
    /// This choice has not yet had a thorough design review. Revisit whether
    /// proofs should range only over declared enum values, with explicit
    /// obligations for legal initialization and transitions that preserve
    /// those values.
    fn enum_cases(&self, name: &str) -> Result<Vec<(String, u64)>> {
        let TypeDecl::Enum { width, members, .. } = self.declaration(name)? else {
            bail!("{name} is not an enum")
        };
        let mut names: BTreeSet<_> = members.iter().map(|m| m.name.clone()).collect();
        let mut cases = members
            .iter()
            .map(|m| (m.name.clone(), m.value))
            .collect::<Vec<_>>();
        for value in 0..(1u64 << width) {
            if !members.iter().any(|m| m.value == value) {
                let mut name = format!("encoding{value}");
                while !names.insert(name.clone()) {
                    name.push('_');
                }
                cases.push((name, value));
            }
        }
        Ok(cases)
    }

    /// Emit source declarations, total codecs, and codec round-trip proof
    /// source.
    fn types_module(&self) -> Result<String> {
        let mut out = String::from("import Std\n\nnamespace XlsSource\n\n");
        // Source members and generated codecs have disjoint namespaces. A
        // source field named `encode` must remain an ordinary record field.
        for declaration in self.declarations {
            let name = identifier(declaration.name())?;
            match declaration {
                TypeDecl::Enum { name: original, .. } => {
                    writeln!(out, "inductive {name} where")?;
                    for (member, _) in self.enum_cases(original)? {
                        lean::validate_lean_member(original, &member)?;
                        ensure!(
                            !matches!(
                                member.as_str(),
                                "ofNat" | "ofNat_ctorIdx" | "ctorElim" | "ctorElimType"
                            ),
                            "cannot emit {original}.{member}: Lean generates an enum declaration with this name"
                        );
                        writeln!(out, "  | {}", identifier(&member)?)?;
                    }
                    writeln!(out, "  deriving _root_.Repr, _root_.DecidableEq\n")?;
                }
                TypeDecl::Struct { fields, .. } => {
                    out.push_str(&lean::structure_header(
                        &name,
                        fields.iter().map(|field| field.name.as_str()),
                    )?);
                    let mut scope = lean::RecordFields::default();
                    for field in fields {
                        scope.write(
                            &mut out,
                            &identifier(&field.name)?,
                            &self.source_type(&field.source_type)?,
                        )?;
                    }
                    writeln!(out, "  deriving _root_.Repr\n")?;
                }
                TypeDecl::Alias { target, .. } => {
                    writeln!(out, "abbrev {name} := {}\n", self.source_type(target)?)?;
                }
            }
        }
        out.push_str(
            "end XlsSource\n\nnamespace XlsCodec\n\nattribute [local simp] Function.comp_def\n\n",
        );
        out.push_str(BOOL_REPRESENTATION);
        for declaration in self.declarations {
            let name = identifier(declaration.name())?;
            let nominal = self.named_type(declaration.name())?;
            let source_type = self.source_type(&nominal)?;
            let physical = source_shape(&nominal, self.declarations)?;
            writeln!(out, "namespace {name}\n")?;
            match declaration {
                TypeDecl::Enum {
                    name: original,
                    width,
                    ..
                } => {
                    let cases = self.enum_cases(original)?;
                    writeln!(out, "def encode : {source_type} → BitVec {width}")?;
                    for (member, value) in &cases {
                        writeln!(
                            out,
                            "  | .{} => BitVec.ofNat {width} {value}",
                            identifier(member)?
                        )?;
                    }
                    writeln!(out, "\ndef decode (x : BitVec {width}) : {source_type} :=")?;
                    let (last, rest) = cases.split_last().context("empty physical enum")?;
                    for (member, value) in rest {
                        writeln!(out, "  if x = {value} then .{} else", identifier(member)?)?;
                    }
                    writeln!(out, "  .{}\n", identifier(&last.0)?)?;
                    writeln!(
                        out,
                        "@[simp] theorem decode_encode (x : {source_type}) : decode (encode x) = x := by\n  cases x <;> rfl\n"
                    )?;
                    writeln!(
                        out,
                        "@[simp] theorem encode_decode (x : BitVec {width}) : encode (decode x) = x := by\n  rcases two_bit_cases x with rfl | rfl | rfl | rfl <;> rfl\n"
                    )?;
                }
                TypeDecl::Struct { fields, .. } => {
                    let encoded = fields
                        .iter()
                        .map(|f| {
                            self.convert(
                                Some(&f.source_type),
                                &format!("x.{}", identifier(&f.name)?),
                                true,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    writeln!(
                        out,
                        "def encode (x : {source_type}) : {} :=\n  {}\n",
                        lean_type(&physical)?,
                        tuple_value(encoded.into_iter())
                    )?;
                    let decoded = fields
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            Ok((
                                identifier(&f.name)?,
                                self.convert(Some(&f.source_type), &tuple_element("x", i), false)?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    writeln!(
                        out,
                        "def decode (x : {}) : {source_type} :=\n  {}\n",
                        lean_type(&physical)?,
                        record_value(&decoded)
                    )?;
                    writeln!(
                        out,
                        "@[simp] theorem decode_encode (x : {source_type}) : decode (encode x) = x := by\n  cases x\n  simp [decode, encode]\n"
                    )?;
                    let mut serial = 0;
                    let pattern = physical_pattern(&physical, &mut serial);
                    writeln!(
                        out,
                        "@[simp] theorem encode_decode (x : {}) : encode (decode x) = x := by\n  rcases x with {pattern}\n  simp [decode, encode]\n",
                        lean_type(&physical)?
                    )?;
                }
                TypeDecl::Alias { target, .. } => {
                    writeln!(
                        out,
                        "def encode (x : {source_type}) : {} := {}",
                        lean_type(&physical)?,
                        self.convert(Some(target), "x", true)?
                    )?;
                    writeln!(
                        out,
                        "def decode (x : {}) : {source_type} := {}\n",
                        lean_type(&physical)?,
                        self.convert(Some(target), "x", false)?
                    )?;
                    writeln!(
                        out,
                        "@[simp] theorem decode_encode (x : {source_type}) : decode (encode x) = x := by\n  simp [decode, encode]\n"
                    )?;
                    writeln!(
                        out,
                        "@[simp] theorem encode_decode (x : {}) : encode (decode x) = x := by\n  simp [decode, encode]\n",
                        lean_type(&physical)?
                    )?;
                }
            }
            writeln!(out, "end {name}\n")?;
        }
        out.push_str("end XlsCodec\n");
        Ok(out)
    }

    /// Resolve a physical register name to its ordered readable state field.
    fn register_index(&self, name: &str) -> Result<usize> {
        self.block()
            .graph
            .registers
            .iter()
            .position(|r| r.name == name)
            .with_context(|| format!("register {name} has no declaration"))
    }

    /// Inline anonymous structural expressions only within the given depth
    /// budget.
    ///
    /// Named nodes keep their bindings so readable equations retain graph
    /// sharing.
    fn operand(&self, reference: NodeRef, depth: usize) -> Result<String> {
        let node = self.graph.get_node(reference);
        let structural = matches!(
            node.payload,
            NodePayload::Literal(_) | NodePayload::Tuple(_) | NodePayload::Sel { .. }
        );
        if depth > 0 && structural && node.name.is_none() {
            self.expression(reference, depth - 1)
        } else {
            Ok(self.names[reference.index].clone())
        }
    }

    /// Render an operand in the caller's source or physical representation.
    fn typed_operand(
        &self,
        reference: NodeRef,
        target: Option<&SourceType>,
        depth: usize,
    ) -> Result<String> {
        self.coerce(
            self.node_types[reference.index].as_ref(),
            target,
            &self.operand(reference, depth)?,
        )
    }

    /// Recognize only the physical encodings used by supported two-bit enums.
    fn literal_number(&self, reference: NodeRef) -> Option<u64> {
        if let NodePayload::Literal(value) = &self.graph.get_node(reference).payload {
            (0..4).find(|encoding| value.bits_equals_u64_value(*encoding))
        } else {
            None
        }
    }

    /// Recognize concatenated equality tests against one nominal enum value.
    ///
    /// Returned encodings follow selector bit order, least-significant first,
    /// rather than enum declaration order; other selector shapes return `None`.
    fn enum_selector(&self, selector: NodeRef) -> Result<Option<(NodeRef, &str, Vec<u64>)>> {
        let comparisons = if let NodePayload::Nary(NaryOp::Concat, args) =
            &self.graph.get_node(selector).payload
        {
            args.iter()
                .rev()
                .map(|reference| {
                    if let NodePayload::Binop(Binop::Eq, left, right) =
                        self.graph.get_node(*reference).payload
                    {
                        self.literal_number(left)
                            .map(|value| (right, value))
                            .or_else(|| self.literal_number(right).map(|value| (left, value)))
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<_>>>()
        } else {
            None
        };
        if let Some(comparisons) = comparisons
            && let Some((subject, _)) = comparisons.first()
            && comparisons.iter().all(|(other, _)| other == subject)
            && let Some(ty) = &self.node_types[subject.index]
            && let SourceType::Enum { name } = self.unalias(ty)?
        {
            Ok(Some((
                *subject,
                name,
                comparisons.iter().map(|(_, value)| *value).collect(),
            )))
        } else {
            Ok(None)
        }
    }

    /// Render source-shaped equations where type context supports that
    /// spelling.
    ///
    /// Other operations reuse the raw operator translation with explicit
    /// codecs; the generated readable evaluator never delegates to the raw
    /// evaluator.
    fn expression(&self, reference: NodeRef, depth: usize) -> Result<String> {
        let node = self.graph.get_node(reference);
        let target = self.node_types[reference.index].as_ref();
        let operand = |r| self.typed_operand(r, target, depth);
        let boolean = |r| self.typed_operand(r, Some(&SourceType::Bool), depth);
        match &node.payload {
            NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
                invariant_args,
            } if let Some(signature) = self.interface.helpers.get(body) => {
                let mut step = format!(
                    "_root_.XlsReadable.Helpers.{}",
                    identifier(&signature.name)?
                );
                if !invariant_args.is_empty() {
                    step.push_str(" · ·");
                    for (arg, (_, ty)) in invariant_args.iter().zip(&signature.params[2..]) {
                        write!(step, " ({})", self.typed_operand(*arg, Some(ty), depth)?)?;
                    }
                }
                let init = self.typed_operand(*init, Some(&signature.result), depth)?;
                let call =
                    format!("(_root_.XlsBlock.countedFor {trip_count} {stride} ({step}) ({init}))");
                self.coerce(Some(&signature.result), target, &call)
            }
            NodePayload::Invoke { to_apply, operands }
                if self.interface.helpers.contains_key(to_apply) =>
            {
                let signature = &self.interface.helpers[to_apply];
                let mut call = format!(
                    "_root_.XlsReadable.Helpers.{}",
                    identifier(&signature.name)?
                );
                for (arg, (_, ty)) in operands.iter().zip(&signature.params) {
                    write!(call, " ({})", self.typed_operand(*arg, Some(ty), depth)?)?;
                }
                self.coerce(Some(&signature.result), target, &format!("({call})"))
            }
            NodePayload::InputPort { name, .. } => Ok(format!("x.{}", identifier(name)?)),
            NodePayload::Array(args) if let Some(ty) = target => {
                let element = self.indexed_type(ty, 1)?;
                let values = args
                    .iter()
                    .map(|arg| self.typed_operand(*arg, Some(element), depth))
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("#v[{}]", values.join(", ")))
            }
            NodePayload::ArrayIndex { array, indices, .. }
                if let Some(ty) = &self.node_types[array.index] =>
            {
                let expr =
                    indices
                        .iter()
                        .try_fold(self.operand(*array, depth)?, |items, index| {
                            Ok::<_, anyhow::Error>(lean::array_read(
                                &items,
                                &self.typed_operand(*index, None, depth)?,
                            ))
                        })?;
                self.coerce(Some(self.indexed_type(ty, indices.len())?), target, &expr)
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } if let Some(ty) = target => {
                let element = self.indexed_type(ty, indices.len())?;
                let indices = indices
                    .iter()
                    .map(|index| self.typed_operand(*index, None, depth))
                    .collect::<Result<Vec<_>>>()?;
                Ok(lean::array_write(
                    &self.typed_operand(*array, Some(ty), depth)?,
                    &indices,
                    &self.typed_operand(*value, Some(element), depth)?,
                ))
            }
            NodePayload::RegisterRead { register } => {
                let index = self.register_index(register)?;
                self.coerce(
                    self.register_type(register)?.as_ref(),
                    target,
                    &format!("q.{}", self.register_fields[index]),
                )
            }
            NodePayload::TupleIndex { tuple, index } => {
                if let Some(ty) = &self.node_types[tuple.index]
                    && let SourceType::Struct { name } = self.unalias(ty)?
                {
                    let TypeDecl::Struct { fields, .. } = self.declaration(name)? else {
                        unreachable!()
                    };
                    let field = &fields[*index];
                    self.coerce(
                        Some(&field.source_type),
                        target,
                        &format!(
                            "({}).{}",
                            self.operand(*tuple, depth)?,
                            identifier(&field.name)?
                        ),
                    )
                } else if let Some(ty) = &self.node_types[tuple.index]
                    && let SourceType::Tuple { members } = self.unalias(ty)?
                {
                    self.coerce(
                        Some(&members[*index]),
                        target,
                        &product_element(
                            &self.operand(*tuple, depth)?,
                            *index,
                            members.len(),
                            true,
                        ),
                    )
                } else {
                    let physical = self.typed_operand(*tuple, None, depth)?;
                    self.convert(target, &tuple_element(&physical, *index), false)
                }
            }
            NodePayload::Tuple(args) if target.is_some() => {
                if let SourceType::Struct { name } = self.unalias(target.unwrap())? {
                    let TypeDecl::Struct { fields, .. } = self.declaration(name)? else {
                        unreachable!()
                    };
                    let values = fields
                        .iter()
                        .zip(args)
                        .map(|(field, arg)| {
                            Ok((
                                identifier(&field.name)?,
                                self.typed_operand(*arg, Some(&field.source_type), depth)?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(record_value(&values))
                } else if let SourceType::Tuple { members } = self.unalias(target.unwrap())? {
                    let values = args
                        .iter()
                        .zip(members)
                        .map(|(arg, ty)| self.typed_operand(*arg, Some(ty), depth))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(product_value(&values, true))
                } else {
                    self.raw_expression(reference, depth)
                }
            }
            NodePayload::Literal(_) if target.is_some() => {
                if let SourceType::Enum { name } = self.unalias(target.unwrap())? {
                    let value = self
                        .literal_number(reference)
                        .context("two-bit enum literal")?;
                    let cases = self.enum_cases(name)?;
                    let (member, _) = cases
                        .iter()
                        .find(|(_, encoding)| *encoding == value)
                        .context("physical enum encoding")?;
                    Ok(format!(
                        "XlsSource.{}.{}",
                        identifier(name)?,
                        identifier(member)?
                    ))
                } else {
                    self.raw_expression(reference, depth)
                }
            }
            NodePayload::Unop(Unop::Not, arg) if node.ty == Type::Bits(1) => {
                Ok(format!("(!{})", boolean(*arg)?))
            }
            NodePayload::Nary(op @ (NaryOp::And | NaryOp::Or | NaryOp::Nor), args)
                if node.ty == Type::Bits(1) =>
            {
                let values = args
                    .iter()
                    .map(|r| boolean(*r))
                    .collect::<Result<Vec<_>>>()?;
                let symbol = if *op == NaryOp::And { " && " } else { " || " };
                let prefix = if *op == NaryOp::Nor { "!" } else { "" };
                Ok(format!("{prefix}({})", values.join(symbol)))
            }
            NodePayload::Sel {
                selector,
                cases,
                default: None,
            } if self.graph.get_node(*selector).ty == Type::Bits(1) && cases.len() == 2 => {
                Ok(format!(
                    "(if {} then {} else {})",
                    boolean(*selector)?,
                    operand(cases[1])?,
                    operand(cases[0])?
                ))
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default: Some(default),
            } => {
                if let Some((subject, name, values)) = self.enum_selector(*selector)? {
                    let branches = self
                        .enum_cases(name)?
                        .into_iter()
                        .map(|(member, value)| {
                            let selected = values
                                .iter()
                                .position(|encoding| *encoding == value)
                                .and_then(|index| cases.get(index))
                                .unwrap_or(default);
                            Ok(format!(
                                "| .{} => {}",
                                identifier(&member)?,
                                self.typed_operand(*selected, target, 12)?
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(format!(
                        "(match {} with\n    {})",
                        self.operand(subject, 0)?,
                        branches.join("\n    ")
                    ))
                } else {
                    self.raw_expression(reference, depth)
                }
            }
            _ => self.raw_expression(reference, depth),
        }
    }

    /// Reuse one raw graph operation with encoded operands and a decoded
    /// result.
    fn raw_expression(&self, reference: NodeRef, depth: usize) -> Result<String> {
        let expression = lean::node_expression(self.graph, reference, |operand| {
            self.typed_operand(operand, None, depth)
        })?;
        self.convert(
            self.node_types[reference.index].as_ref(),
            &expression,
            false,
        )
    }

    /// Emit readable clock-step equations with physical ports and assertion
    /// results.
    ///
    /// Raw emission has already established single writes and supported resets.
    /// All equations read old state, and reset takes precedence over load
    /// enable.
    fn executable(&self) -> Result<String> {
        let block = self.block;
        let mut out = format!(
            "-- Readable execution of XLS block {}.\n-- Equations come from the block. The DSLX signature supplies interface names and types.\nimport Generated\nimport ReadableTypes\n\nnamespace XlsReadable\n\n",
            lean::comment(&block.graph.name)
        );
        for function in &block.helpers {
            if let Some(signature) = self.interface.helpers.get(&function.name) {
                let model = Self::new(block, self.declarations, self.interface, Some(function))?;
                write!(out, "def Helpers.{}", identifier(&signature.name)?)?;
                for (param, (_, ty)) in function.params.iter().zip(&signature.params) {
                    write!(
                        out,
                        " ({} : {})",
                        model.names[param.index],
                        model.source_type(ty)?
                    )?;
                }
                writeln!(out, " : {} :=", model.source_type(&signature.result)?)?;
                for reference in ir_utils::get_topological_nodes(&function.nodes) {
                    let node = function.get_node(reference);
                    if !matches!(node.payload, NodePayload::Nil | NodePayload::Param) {
                        let ty = if let Some(ty) = &model.node_types[reference.index] {
                            model.source_type(ty)?
                        } else {
                            lean_type(&node.ty)?
                        };
                        writeln!(
                            out,
                            "  let {} : {ty} := {}",
                            model.names[reference.index],
                            model.expression(reference, 0)?
                        )?;
                    }
                }
                writeln!(
                    out,
                    "  {}\n",
                    model.typed_operand(
                        function.ret_node_ref.context("helper has no return node")?,
                        Some(&signature.result),
                        0
                    )?
                )?;
            }
        }
        // These field identifiers have already been quoted by identifier().
        out.push_str(&lean::structure_header(
            "State",
            self.register_fields
                .iter()
                .map(|field| field.trim_matches(['«', '»'])),
        )?);
        let mut scope = lean::RecordFields::default();
        for (index, register) in block.graph.registers.iter().enumerate() {
            let ty = if let Some(ty) = self.register_type(&register.name)? {
                self.source_type(&ty)?
            } else {
                lean::physical_field_type(&register.ty)?
            };
            scope.write(&mut out, &self.register_fields[index], &ty)?;
        }
        out.push_str("  deriving Repr\n\n");
        let interface = self.interface;
        out.push_str(&lean::structure_header(
            "Input",
            block
                .graph
                .input_ports()
                .map(|port| block.graph.port_name(port)),
        )?);
        let mut scope = lean::RecordFields::default();
        for port in block.graph.input_ports() {
            let name = block.graph.port_name(port);
            scope.write(
                &mut out,
                &identifier(name)?,
                &self.source_type(&interface.inputs[name])?,
            )?;
        }
        out.push_str("  deriving Repr\n\n");
        let types = interface
            .outputs
            .iter()
            .map(|ty| self.source_type(ty))
            .collect::<Result<Vec<_>>>()?;
        writeln!(out, "abbrev Output := {}", product_type(&types, false))?;
        out.push_str(
            r#"abbrev Assertions := XlsBlock.Assertions

structure Evaluation where
  next : State
  output : Output
  assertions : Assertions
  deriving Repr

def eval (q : State) (x : Input) : Evaluation :=
"#,
        );
        for reference in ir_utils::get_topological_nodes(&block.graph.nodes) {
            let node = block.graph.get_node(reference);
            if !matches!(node.payload, NodePayload::Nil) {
                let ty = if let Some(ty) = &self.node_types[reference.index] {
                    self.source_type(ty)?
                } else {
                    lean_type(&node.ty)?
                };
                writeln!(
                    out,
                    "  -- {}\n  let {} : {ty} := {}",
                    lean::comment(&lean::node_context(block, reference)),
                    self.names[reference.index],
                    self.expression(reference, 0)?
                )?;
            }
        }
        let next = block.graph.registers.iter().enumerate().map(|(index, register)| {
            let node = block.graph.nodes.iter().find(|n| matches!(&n.payload, NodePayload::RegisterWrite { register: name, .. } if name == &register.name)).context("checked single register write")?;
            let NodePayload::RegisterWrite { arg, load_enable, reset, .. } = &node.payload else { unreachable!() };
            let target = self.register_type(&register.name)?;
            let value = self.typed_operand(*arg, target.as_ref(), 0)?;
            let enabled = if let Some(enable) = load_enable {
                format!("(if {} then {value} else q.{})", self.typed_operand(*enable, Some(&SourceType::Bool), 0)?, self.register_fields[index])
            } else { value };
            let expression = if let Some(signal) = reset {
                let behavior = block.graph.reset.as_ref().context("checked reset behavior")?;
                let initial = lean::literal(&register.ty, register.reset_value.as_ref().context("checked reset value")?)?;
                let initial = self.convert(target.as_ref(), &initial, false)?;
                let condition = self.typed_operand(*signal, Some(&SourceType::Bool), 0)?;
                let condition = if behavior.active_low { format!("!{condition}") } else { condition };
                format!("(if {condition} then {initial} else {enabled})")
            } else { enabled };
            Ok((self.register_fields[index].clone(), expression))
        }).collect::<Result<Vec<_>>>()?;
        let output = block
            .graph
            .output_ports()
            .enumerate()
            .map(|(index, port)| {
                let target = Some(&self.interface.outputs[index]);
                Ok((
                    identifier(block.graph.port_name(port))?,
                    self.typed_operand(block.graph.output_value(port), target, 0)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let assertions = block
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                if let NodePayload::Assert { activate, .. } = &node.payload {
                    Some((NodeRef { index }, *activate))
                } else {
                    None
                }
            })
            .map(|(reference, activate)| {
                Ok((
                    identifier(&ir::node_textual_id(block.graph, reference))?,
                    self.typed_operand(activate, Some(&SourceType::Bool), 0)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        writeln!(
            out,
            "  {{ next := {},\n    output := {},\n    assertions := {} }}\n",
            record_value(&next),
            product_value(
                &output
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>(),
                false
            ),
            record_value(&assertions)
        )?;
        out.push_str("def step (q : State) (x : Input) : State := (eval q x).next\ndef observe (q : State) (x : Input) : Output := (eval q x).output\ndef assertions (q : State) (x : Input) : Assertions := (eval q x).assertions\n\n");
        let holds = assertions
            .iter()
            .rev()
            .fold("True".into(), |tail, (name, _)| {
                format!("And (checks.{name} = true) ({tail})")
            });
        writeln!(
            out,
            "def assertionsHold (q : State) (x : Input) : Prop :=\n  let checks := assertions q x\n  {holds}\n"
        )?;
        writeln!(out, "def reset (q : State) (x : Input) : Evaluation :=")?;
        if let Some(reset) = &block.graph.reset {
            writeln!(
                out,
                "  eval q {{ x with {} := {} }}",
                identifier(block.graph.port_name(reset.port))?,
                !reset.active_low
            )?;
        } else {
            out.push_str("  eval q x\n");
        }
        out.push_str("\nend XlsReadable\n");
        Ok(out)
    }

    /// Emit total state/result conversions and their round-trip proof source.
    ///
    /// Inputs and outputs use their source codecs and the block's port names.
    fn representation(&self) -> Result<String> {
        let mut out = String::from(
            "import ReadableGenerated\n\nnamespace XlsReadable\n\nattribute [local simp] Function.comp_def\n\n",
        );
        let encoded = self
            .block()
            .graph
            .registers
            .iter()
            .enumerate()
            .map(|(index, r)| {
                Ok((
                    identifier(&r.name)?,
                    self.convert(
                        self.register_type(&r.name)?.as_ref(),
                        &format!("q.{}", self.register_fields[index]),
                        true,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let decoded = self
            .block()
            .graph
            .registers
            .iter()
            .enumerate()
            .map(|(index, r)| {
                Ok((
                    self.register_fields[index].clone(),
                    self.convert(
                        self.register_type(&r.name)?.as_ref(),
                        &format!("q.{}", identifier(&r.name)?),
                        false,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        writeln!(
            out,
            "def encodeState (q : State) : XlsBlock.State :=\n  {}\n",
            record_value(&encoded)
        )?;
        writeln!(
            out,
            "def decodeState (q : XlsBlock.State) : State :=\n  {}\n",
            record_value(&decoded)
        )?;
        let interface = self.interface;
        for encode in [true, false] {
            let (operation, from, to) = if encode {
                ("encode", "", "XlsBlock.")
            } else {
                ("decode", "XlsBlock.", "")
            };
            let inputs = interface
                .inputs
                .iter()
                .map(|(name, ty)| {
                    let field = identifier(name)?;
                    Ok((
                        field.clone(),
                        self.convert(Some(ty), &format!("x.{field}"), encode)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            writeln!(
                out,
                "def {operation}Input (x : {from}Input) : {to}Input :=\n  {}\n",
                record_value(&inputs)
            )?;
            let outputs = interface
                .outputs
                .iter()
                .zip(self.block().graph.output_ports())
                .enumerate()
                .map(|(index, (ty, port))| {
                    let field = identifier(self.block().graph.port_name(port))?;
                    let source = if encode {
                        product_element("x", index, interface.outputs.len(), false)
                    } else {
                        format!("x.{field}")
                    };
                    Ok((field, self.convert(Some(ty), &source, encode)?))
                })
                .collect::<Result<Vec<_>>>()?;
            let expression = if encode {
                record_value(&outputs)
            } else {
                product_value(
                    &outputs.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
                    false,
                )
            };
            writeln!(
                out,
                "def {operation}Output (x : {from}Output) : {to}Output :=\n  {expression}\n"
            )?;
        }
        out.push_str(r#"def encodeAssertions (x : Assertions) : XlsBlock.Assertions := x
def decodeAssertions (x : XlsBlock.Assertions) : Assertions := x

def encodeEvaluation (e : Evaluation) : XlsBlock.Evaluation :=
  ⟨encodeState e.next, encodeOutput e.output, encodeAssertions e.assertions⟩
def decodeEvaluation (e : XlsBlock.Evaluation) : Evaluation :=
  ⟨decodeState e.next, decodeOutput e.output, decodeAssertions e.assertions⟩

@[simp] theorem decode_encode_state (q : State) : decodeState (encodeState q) = q := by
  cases q
  simp [decodeState, encodeState]
@[simp] theorem encode_decode_state (q : XlsBlock.State) : encodeState (decodeState q) = q := by
  cases q
  simp [decodeState, encodeState]
@[simp] theorem decode_encode_input (x : Input) : decodeInput (encodeInput x) = x := by
  cases x
  simp [decodeInput, encodeInput]
@[simp] theorem encode_decode_input (x : XlsBlock.Input) : encodeInput (decodeInput x) = x := by
  cases x
  simp [decodeInput, encodeInput]
@[simp] theorem decode_encode_output (x : Output) : decodeOutput (encodeOutput x) = x := by
  simp [decodeOutput, encodeOutput]
@[simp] theorem encode_decode_output (x : XlsBlock.Output) : encodeOutput (decodeOutput x) = x := by
  cases x
  simp [decodeOutput, encodeOutput]
@[simp] theorem decode_encode_evaluation (e : Evaluation) : decodeEvaluation (encodeEvaluation e) = e := by
  cases e
  simp [decodeEvaluation, encodeEvaluation, decodeAssertions, encodeAssertions]
@[simp] theorem encode_decode_evaluation (e : XlsBlock.Evaluation) : encodeEvaluation (decodeEvaluation e) = e := by
  cases e
  simp [decodeEvaluation, encodeEvaluation, decodeAssertions, encodeAssertions]

end XlsReadable
"#);
        Ok(out)
    }

    /// Build record destructuring patterns and collect enum binders for case
    /// splits.
    ///
    /// `serial` supplies distinct binders across the whole generated proof.
    fn source_pattern(
        &self,
        ty: &SourceType,
        serial: &mut usize,
        enums: &mut Vec<String>,
    ) -> Result<String> {
        match self.unalias(ty)? {
            SourceType::Tuple { members } => {
                let fields = members
                    .iter()
                    .map(|member| self.source_pattern(member, serial, enums))
                    .collect::<Result<Vec<_>>>()?;
                Ok(match fields.as_slice() {
                    [] => "⟨⟩".into(),
                    [field] => format!("⟨{field}, ⟨⟩⟩"),
                    _ => format!("⟨{}⟩", fields.join(", ")),
                })
            }
            SourceType::Struct { name } => {
                let TypeDecl::Struct { fields, .. } = self.declaration(name)? else {
                    unreachable!()
                };
                let parts = fields
                    .iter()
                    .map(|f| self.source_pattern(&f.source_type, serial, enums))
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("⟨{}⟩", parts.join(", ")))
            }
            other => {
                let name = format!("value{serial}");
                *serial += 1;
                if matches!(other, SourceType::Enum { .. }) {
                    enums.push(name.clone());
                }
                Ok(name)
            }
        }
    }

    /// Emit proofs relating readable and raw evaluation for arbitrary physical
    /// state.
    ///
    /// Structural rewrites preserve codec heads until round trips cancel; enum
    /// encoders unfold afterward. Successful text generation is not Lean proof
    /// checking and does not establish correspondence to XLS execution.
    fn correspondence(&self) -> Result<String> {
        if self
            .block
            .helpers
            .iter()
            .map(|f| &f.graph)
            .chain([self.graph])
            .any(|graph| {
                graph.nodes.iter().any(|node| {
                    matches!(&node.payload,
                NodePayload::CountedFor { body, .. } if self.interface.helpers.contains_key(body))
                })
            })
        {
            return self.loop_correspondence();
        }
        let mut out = String::from("import ReadableRepresentation\n\nnamespace XlsReadable\n\n");
        out.push_str(BOOL_CORRESPONDENCE);
        // Expose a record's fields only at a constructor. Unfolding its encoder
        // on an if or a decoded value would hide the structural rewrite rules.
        for declaration in self.declarations {
            if let TypeDecl::Struct { name, fields } = declaration {
                let binders = fields
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        Ok(format!(
                            "(field{index} : {})",
                            self.source_type(&field.source_type)?
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let values = (0..fields.len())
                    .map(|index| format!("field{index}"))
                    .collect::<Vec<_>>();
                let encoded = fields
                    .iter()
                    .zip(&values)
                    .map(|(field, value)| self.convert(Some(&field.source_type), value, true))
                    .collect::<Result<Vec<_>>>()?;
                writeln!(
                    out,
                    "private theorem {} {} :\n    XlsCodec.{}.encode ⟨{}⟩ =\n      {} := by\n  rfl\n",
                    identifier(&format!("encode_{name}_mk"))?,
                    binders.join(" "),
                    identifier(name)?,
                    values.join(", "),
                    tuple_value(encoded.into_iter())
                )?;
            }
        }
        out.push_str("-- Arbitrary physical states and inputs: no reachability, enum legality or readiness premise.\ntheorem encoded_eval (q : State) (x : Input) :\n    encodeEvaluation (eval q x) = XlsBlock.eval (encodeState q) (encodeInput x) := by\n");
        let mut serial = 0;
        let mut enums = Vec::new();
        let patterns = self
            .block()
            .graph
            .registers
            .iter()
            .map(|r| {
                if let Some(ty) = self.register_type(&r.name)? {
                    self.source_pattern(&ty, &mut serial, &mut enums)
                } else {
                    let name = format!("value{serial}");
                    serial += 1;
                    Ok(name)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        writeln!(out, "  rcases q with ⟨{}⟩", patterns.join(", "))?;
        let interface = self.interface;
        let inputs = self
            .block
            .graph
            .input_ports()
            .map(|port| {
                self.source_pattern(
                    &interface.inputs[self.block.graph.port_name(port)],
                    &mut serial,
                    &mut enums,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        writeln!(out, "  rcases x with ⟨{}⟩", inputs.join(", "))?;
        for name in enums {
            writeln!(out, "  all_goals cases {name}")?;
        }
        let structural_rules = {
            // Fix each function: bare apply_ite can distribute nested
            // conditionals through each other. Projections also cross selected
            // records and physical tuples, not just their enclosing codecs.
            let mut conditional_rules = BTreeSet::from([
                "apply_ite XlsCodec.encodeBool".to_owned(),
                "apply_ite XlsCodec.decodeBool".to_owned(),
            ]);
            for declaration in self.declarations {
                let name = identifier(declaration.name())?;
                conditional_rules.insert(format!("apply_ite XlsCodec.{name}.encode"));
                conditional_rules.insert(format!("apply_ite XlsCodec.{name}.decode"));
                if let TypeDecl::Struct { fields, .. } = declaration {
                    for field in fields {
                        conditional_rules.insert(format!(
                            "apply_ite XlsSource.{name}.{}",
                            identifier(&field.name)?
                        ));
                    }
                }
            }
            for graph in
                std::iter::once(self.graph).chain(self.block.helpers.iter().map(|f| &f.graph))
            {
                for node in &graph.nodes {
                    if let NodePayload::TupleIndex { tuple, index } = &node.payload {
                        conditional_rules.insert(format!(
                            "apply_ite (fun (value : {}) => {})",
                            lean_type(&graph.get_node(*tuple).ty)?,
                            tuple_element("value", *index)
                        ));
                    }
                }
            }
            let mut rules = conditional_rules.into_iter().collect::<Vec<_>>();
            rules.extend([
                "prod_fst_ite".into(),
                "prod_snd_ite".into(),
                "vector_map_ite".into(),
                "vector_getD_back_map".into(),
                "Vector.map_setIfInBounds".into(),
                "Vector.map_map".into(),
                "Function.comp_def".into(),
                "Vector.map_id'".into(),
            ]);
            for declaration in self.declarations.iter().rev() {
                let name = identifier(declaration.name())?;
                // Keep codec heads until roundtrips cancel.
                rules.push(format!("XlsCodec.{name}.encode_decode"));
                rules.push(format!("XlsCodec.{name}.decode_encode"));
                if let TypeDecl::Struct { name, .. } = declaration {
                    rules.push(identifier(&format!("encode_{name}_mk"))?);
                }
            }
            rules.push("XlsCodec.decode_encodeBool".into());
            rules.push("XlsCodec.encode_decodeBool".into());
            rules.join(",\n       ")
        };
        let mut helper_rules = String::new();
        for function in &self.block.helpers {
            if let Some(signature) = self.interface.helpers.get(&function.name) {
                write!(
                    helper_rules,
                    ", _root_.XlsReadable.Helpers.{}, {}",
                    identifier(&signature.name)?,
                    lean::helper_path(&function.name)?
                )?;
            }
        }
        writeln!(
            out,
            r#"  all_goals
    dsimp (config := {{ instances := true }}) only
      [eval, XlsBlock.eval, encodeEvaluation, encodeState, encodeInput,
       encodeOutput, encodeAssertions{helper_rules}]
  all_goals
    -- Distribute encoders before exposing constructors and cancelling roundtrips.
    simp (config := {{ instances := true, failIfUnchanged := false }}) only
      [{structural_rules}]
  all_goals
    -- Keep raw one-bit operations and selectors in the readable Boolean form.
    simp (config := {{ instances := true, failIfUnchanged := false }}) only
      [XlsCodec.raw_select, raw_flag, raw_not, raw_and, raw_or, decode_ofBool,
       {structural_rules}]"#
        )?;
        let enum_encoders = self
            .declarations
            .iter()
            .filter_map(|declaration| match declaration {
                TypeDecl::Enum { name, .. } => {
                    Some(identifier(name).map(|name| format!("XlsCodec.{name}.encode")))
                }
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        if !enum_encoders.is_empty() {
            writeln!(
                out,
                "  all_goals\n    -- Enum encodings now have constructor arguments; reduce them last.\n    dsimp (config := {{ instances := true }}) only [{}]",
                enum_encoders.join(", ")
            )?;
        }
        // A source tuple is encoded one field at a time, while the raw
        // register update selects the entire physical tuple at once.
        out.push_str("  all_goals try simp only [ite_prod_mk, ite_self]\n");
        out.push_str("  all_goals rfl\n\n");
        out.push_str(CORRESPONDENCE);
        out.push_str("\nend XlsReadable\n");
        Ok(out)
    }
}

/// Lean products have no trailing Unit. Only an explicit singleton DSLX
/// tuple needs one to distinguish it from the individual element.
fn product_type(types: &[String], singleton_tuple: bool) -> String {
    match types {
        [] => "_root_.Unit".into(),
        [ty] if singleton_tuple => format!("(_root_.Prod ({ty}) _root_.Unit)"),
        [ty] => ty.clone(),
        [head, rest @ ..] => format!("(_root_.Prod ({head}) ({}))", product_type(rest, false)),
    }
}

fn product_value(elements: &[String], singleton_tuple: bool) -> String {
    match elements {
        [] => "()".into(),
        [value] if singleton_tuple => format!("({value}, ())"),
        [value] => value.clone(),
        [head, rest @ ..] => format!("({head}, {})", product_value(rest, false)),
    }
}

fn product_element(value: &str, index: usize, len: usize, singleton_tuple: bool) -> String {
    if index + 1 < len || (len == 1 && singleton_tuple) {
        tuple_element(value, index)
    } else {
        format!("({value}){}", ".2".repeat(index))
    }
}

/// Destructure nested physical products, assigning distinct names to non-unit
/// leaves.
fn physical_pattern(ty: &Type, serial: &mut usize) -> String {
    match ty {
        Type::Tuple(fields) => {
            let parts = fields
                .iter()
                .map(|f| physical_pattern(f, serial))
                .collect::<Vec<_>>();
            parts
                .into_iter()
                .rev()
                .fold("⟨⟩".into(), |tail, head| format!("⟨{head}, {tail}⟩"))
        }
        Type::Token => "⟨⟩".into(),
        _ => {
            let name = format!("value{serial}");
            *serial += 1;
            name
        }
    }
}

/// Shared Boolean codecs and finite-bit lemmas used by generated source types.
const BOOL_REPRESENTATION: &str = r#"def encodeBool (b : Bool) : BitVec 1 := BitVec.ofBool b
def decodeBool (b : BitVec 1) : Bool := b == 1

theorem one_bit_cases (b : BitVec 1) : b = 0 ∨ b = 1 := by
  have bound := b.isLt
  have alternatives : b.toNat = 0 ∨ b.toNat = 1 := by omega
  rcases alternatives with zero | one
  · exact Or.inl (BitVec.eq_of_toNat_eq (by simpa using zero))
  · exact Or.inr (BitVec.eq_of_toNat_eq (by simpa using one))

theorem two_bit_cases (b : BitVec 2) : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by
  have bound := b.isLt
  have alternatives : b.toNat = 0 ∨ b.toNat = 1 ∨ b.toNat = 2 ∨ b.toNat = 3 := by omega
  rcases alternatives with zero | one | two | three
  · exact Or.inl (BitVec.eq_of_toNat_eq (by simpa using zero))
  · exact Or.inr (Or.inl (BitVec.eq_of_toNat_eq (by simpa using one)))
  · exact Or.inr (Or.inr (Or.inl (BitVec.eq_of_toNat_eq (by simpa using two))))
  · exact Or.inr (Or.inr (Or.inr (BitVec.eq_of_toNat_eq (by simpa using three))))

@[simp] theorem decode_encodeBool (b : Bool) : decodeBool (encodeBool b) = b := by
  cases b <;> rfl
@[simp] theorem encode_decodeBool (b : BitVec 1) : encodeBool (decodeBool b) = b := by
  rcases one_bit_cases b with rfl | rfl <;> rfl
@[simp] theorem encode_not (b : Bool) : encodeBool (!b) = BitVec.not (encodeBool b) := by
  cases b <;> rfl
@[simp] theorem encode_and (a b : Bool) : encodeBool (a && b) = BitVec.and (encodeBool a) (encodeBool b) := by
  cases a <;> cases b <;> rfl
@[simp] theorem encode_or (a b : Bool) : encodeBool (a || b) = BitVec.or (encodeBool a) (encodeBool b) := by
  cases a <;> cases b <;> rfl
@[simp] theorem raw_select {α : Type} (b : BitVec 1) (f t : α) :
    (if b.toNat == 0 then f else t) = (if decodeBool b then t else f) := by
  rcases one_bit_cases b with rfl | rfl <;> rfl

"#;

/// Rewrite rules aligning raw one-bit operations with readable Boolean
/// expressions.
const BOOL_CORRESPONDENCE: &str = r#"private theorem raw_not (b : BitVec 1) :
    BitVec.not b = XlsCodec.encodeBool (!XlsCodec.decodeBool b) := by
  simpa only [XlsCodec.encode_decodeBool] using
    (XlsCodec.encode_not (XlsCodec.decodeBool b)).symm

private theorem raw_and (a b : BitVec 1) :
    BitVec.and a b = XlsCodec.encodeBool (XlsCodec.decodeBool a && XlsCodec.decodeBool b) := by
  simpa only [XlsCodec.encode_decodeBool] using
    (XlsCodec.encode_and (XlsCodec.decodeBool a) (XlsCodec.decodeBool b)).symm

private theorem raw_or (a b : BitVec 1) :
    BitVec.or a b = XlsCodec.encodeBool (XlsCodec.decodeBool a || XlsCodec.decodeBool b) := by
  simpa only [XlsCodec.encode_decodeBool] using
    (XlsCodec.encode_or (XlsCodec.decodeBool a) (XlsCodec.decodeBool b)).symm

private theorem raw_flag (b : BitVec 1) :
    (b == BitVec.ofNat 1 1) = XlsCodec.decodeBool b := by
  rfl

private theorem decode_ofBool (b : Bool) :
    XlsCodec.decodeBool (BitVec.ofBool b) = b := by
  exact XlsCodec.decode_encodeBool b

private theorem prod_fst_ite {α β : Type} (p : Prop) [Decidable p] (x y : α × β) :
    (if p then x else y).1 = (if p then x.1 else y.1) := by split <;> rfl
private theorem prod_snd_ite {α β : Type} (p : Prop) [Decidable p] (x y : α × β) :
    (if p then x else y).2 = (if p then x.2 else y.2) := by split <;> rfl
private theorem ite_prod_mk {α β : Type} (p : Prop) [Decidable p] (a c : α) (b d : β) :
    (if p then (a, b) else (c, d)) = ((if p then a else c), (if p then b else d)) := by
  split <;> rfl

private theorem vector_map_ite {α β : Type} {n : Nat} (f : α → β)
    (p : Prop) [Decidable p] (x y : Vector α n) :
    (if p then x else y).map f = (if p then x.map f else y.map f) :=
  apply_ite (Vector.map f) p x y

private theorem vector_getD_back_map {α β : Type} {n : Nat} [NeZero n]
    (f : α → β) (xs : Vector α n) (i : Nat) :
    (xs.map f).getD i (xs.map f).back = f (xs.getD i xs.back) := by
  by_cases h : i < n <;> simp [Vector.getD, Array.getD, Vector.back_eq_getElem, h]

"#;

/// Step, reset, and trace proof source derived from the generated evaluation
/// theorem.
const CORRESPONDENCE: &str = r#"theorem decoded_eval (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeEvaluation (XlsBlock.eval q x) = eval (decodeState q) (decodeInput x) := by
  have h := congrArg decodeEvaluation (encoded_eval (decodeState q) (decodeInput x))
  simpa using h.symm

theorem encoded_step (q : State) (x : Input) :
    encodeState (step q x) = XlsBlock.step (encodeState q) (encodeInput x) := by
  exact congrArg XlsBlock.Evaluation.next (encoded_eval q x)
theorem decoded_step (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeState (XlsBlock.step q x) = step (decodeState q) (decodeInput x) := by
  exact congrArg Evaluation.next (decoded_eval q x)
theorem encoded_observe (q : State) (x : Input) :
    encodeOutput (observe q x) = XlsBlock.observe (encodeState q) (encodeInput x) := by
  exact congrArg XlsBlock.Evaluation.output (encoded_eval q x)
theorem decoded_observe (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeOutput (XlsBlock.observe q x) = observe (decodeState q) (decodeInput x) := by
  exact congrArg Evaluation.output (decoded_eval q x)
theorem encoded_assertions (q : State) (x : Input) :
    encodeAssertions (assertions q x) = XlsBlock.assertions (encodeState q) (encodeInput x) := by
  exact congrArg XlsBlock.Evaluation.assertions (encoded_eval q x)
theorem decoded_assertions (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeAssertions (XlsBlock.assertions q x) = assertions (decodeState q) (decodeInput x) := by
  exact congrArg Evaluation.assertions (decoded_eval q x)
theorem encoded_assertionsHold (q : State) (x : Input) :
    assertionsHold q x ↔ XlsBlock.assertionsHold (encodeState q) (encodeInput x) := by
  unfold assertionsHold XlsBlock.assertionsHold
  have h := encoded_assertions q x
  change assertions q x = XlsBlock.assertions (encodeState q) (encodeInput x) at h
  rw [h]

theorem encoded_reset (q : State) (x : Input) :
    encodeEvaluation (reset q x) = XlsBlock.reset (encodeState q) (encodeInput x) := by
  unfold reset XlsBlock.reset
  exact encoded_eval q _
theorem decoded_reset (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeEvaluation (XlsBlock.reset q x) = reset (decodeState q) (decodeInput x) := by
  have h := congrArg decodeEvaluation (encoded_reset (decodeState q) (decodeInput x))
  simpa using h.symm

def trace (initial : State) (input : Nat → Input) : Nat → State
  | 0 => initial
  | t + 1 => step (trace initial input t) (input t)
def rawTrace (initial : XlsBlock.State) (input : Nat → XlsBlock.Input) : Nat → XlsBlock.State
  | 0 => initial
  | t + 1 => XlsBlock.step (rawTrace initial input t) (input t)

/-- The initial physical state is arbitrary, including every undeclared enum encoding. -/
theorem encoded_trace (initial : State) (input : Nat → Input) (t : Nat) :
    encodeState (trace initial input t) =
      rawTrace (encodeState initial) (fun n => encodeInput (input n)) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    exact (encoded_step _ _).trans (congrArg (fun q => XlsBlock.step q (encodeInput (input t))) ih)
theorem decoded_trace (initial : XlsBlock.State) (input : Nat → XlsBlock.Input) (t : Nat) :
    decodeState (rawTrace initial input t) =
      trace (decodeState initial) (fun n => decodeInput (input n)) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    exact (decoded_step _ _).trans (congrArg (fun q => step q (decodeInput (input t))) ih)
theorem trace_output (initial : State) (input : Nat → Input) (t : Nat) :
    encodeOutput (observe (trace initial input t) (input t)) =
      XlsBlock.observe (rawTrace (encodeState initial) (fun n => encodeInput (input n)) t)
        (encodeInput (input t)) := by
  rw [encoded_observe, encoded_trace]
theorem trace_assertions (initial : State) (input : Nat → Input) (t : Nat) :
    encodeAssertions (assertions (trace initial input t) (input t)) =
      XlsBlock.assertions (rawTrace (encodeState initial) (fun n => encodeInput (input n)) t)
        (encodeInput (input t)) := by
  rw [encoded_assertions, encoded_trace]

theorem clock_relation_iff (q : Nat → State) (x : Nat → Input) :
    (∀ t, q (t + 1) = step (q t) (x t)) ↔
      (∀ t, encodeState (q (t + 1)) = XlsBlock.step (encodeState (q t)) (encodeInput (x t))) := by
  constructor
  · intro run t
    rw [run t, encoded_step]
  · intro run t
    have h := congrArg decodeState (run t)
    simpa [decoded_step] using h
"#;
