// SPDX-License-Identifier: Apache-2.0

//! Source-shaped Lean values and executable expressions over a checked block.
//!
//! Validated attachments and graph propagation supply nominal types; the graph
//! still supplies every equation and physical port. Codecs include undeclared
//! enum encodings so arbitrary physical state remains representable. This
//! module emits correspondence proof source but does not check it with Lean.

use std::{collections::BTreeSet, fmt::Write};

use anyhow::{Context, Result, bail};
use xlsynth_pir::{
    ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop},
    ir_utils,
};

use crate::block::{
    Block,
    raw::{
        self as lean, LeanBlockBundle, identifier, lean_type, record_value, tuple_element,
        tuple_value,
    },
    source_map::{SourceMap, SourceType, TypeDecl, ValidatedSourceMap, source_shape},
};

/// Source type annotations and collision-free names for one authenticated
/// graph.
///
/// All vectors use the checked graph's node or register order. Annotation
/// contexts memoize propagation through shared nodes; they are local to one
/// emission and cannot outlive the validated inputs.
struct Presentation<'a> {
    checked: &'a ValidatedSourceMap,
    /// Only checked attachments and graph propagation introduce nominal types.
    node_types: Vec<Option<SourceType>>,
    /// A shared graph node is traversed once for each distinct nominal context.
    annotation_contexts: Vec<Vec<SourceType>>,
    names: Vec<String>,
    register_fields: Vec<String>,
}

/// Emit five dependent modules after the raw emitter checks supported
/// operations.
pub(crate) fn emit(checked: &ValidatedSourceMap) -> Result<LeanBlockBundle> {
    let raw = lean::emit(checked.block())?;
    let presentation = Presentation::new(checked)?;
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
    /// One-bit nodes use `Bool`; nominal types come from attached reads,
    /// projections, and write destinations rather than matching widths alone.
    fn new(checked: &'a ValidatedSourceMap) -> Result<Self> {
        let block = checked.block();
        let reserved: BTreeSet<_> = block
            .graph
            .nodes
            .iter()
            .enumerate()
            .map(|(index, _)| ir::node_textual_id(&block.graph, NodeRef { index }))
            .collect();
        let mut used: BTreeSet<String> = [
            "q",
            "x",
            "Bool",
            "BitVec",
            "Nat",
            "Unit",
            "Prod",
            "XlsSource",
            "XlsCodec",
            "XlsBlock",
            "XlsReadable",
            "true",
            "false",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let names = block
            .graph
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let source = ir::node_textual_id(&block.graph, NodeRef { index });
                let name = if node.name.is_none() || used.contains(&source) {
                    let mut fallback = format!("xls_node_{index}");
                    while reserved.contains(&fallback) || used.contains(&fallback) {
                        fallback.push('_');
                    }
                    fallback
                } else {
                    source
                };
                used.insert(name.clone());
                identifier(&name)
            })
            .collect::<Result<Vec<_>>>()?;
        let reserved_fields: BTreeSet<_> = block
            .graph
            .registers
            .iter()
            .map(|register| register.name.trim_start_matches('_'))
            .collect();
        let mut fields = BTreeSet::new();
        let register_fields = block
            .graph
            .registers
            .iter()
            .enumerate()
            .map(|(index, r)| {
                let base = r.name.trim_start_matches('_');
                let name = if base.is_empty() || fields.contains(base) {
                    let mut fallback = format!("register_{index}");
                    while fields.contains(&fallback) || reserved_fields.contains(fallback.as_str())
                    {
                        fallback.push('_');
                    }
                    fallback
                } else {
                    base.to_owned()
                };
                fields.insert(name.clone());
                identifier(&name)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = Self {
            checked,
            node_types: block
                .graph
                .nodes
                .iter()
                .map(|n| (n.ty == Type::Bits(1)).then_some(SourceType::Bool))
                .collect(),
            annotation_contexts: vec![Vec::new(); block.graph.nodes.len()],
            names,
            register_fields,
        };
        for reference in ir_utils::get_topological_nodes(&block.graph.nodes) {
            match &block.graph.get_node(reference).payload {
                NodePayload::RegisterRead { register } => {
                    result.node_types[reference.index] = result.register_type(register)?;
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
                    }
                }
                _ => {
                    // Other nodes get their context from register destinations.
                }
            }
        }
        // Register destinations supply exact type context to tuple construction
        // and selections. A reused node can still be converted at another use.
        for node in &block.graph.nodes {
            if let NodePayload::RegisterWrite { arg, register, .. } = &node.payload
                && let Some(ty) = result.register_type(register)?
            {
                result.annotate(*arg, &ty)?;
            }
        }
        Ok(result)
    }

    fn block(&self) -> &Block {
        self.checked.block()
    }
    fn map(&self) -> &SourceMap {
        self.checked.map()
    }

    /// Resolve a nominal reference within this presentation's checked
    /// catalogue.
    fn declaration(&self, name: &str) -> Result<&TypeDecl> {
        self.map()
            .declarations
            .iter()
            .find(|d| d.name() == name)
            .with_context(|| format!("validated source declaration {name}"))
    }

    /// Return the explicit register attachment, leaving unattached storage
    /// physical.
    fn register_type(&self, register: &str) -> Result<Option<SourceType>> {
        self.map()
            .attachments
            .iter()
            .find(|a| a.register == register)
            .map(|a| self.named_type(&a.source_type))
            .transpose()
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
            let payload = self.block().graph.get_node(reference).payload.clone();
            match payload {
                NodePayload::Tuple(args) => {
                    if let SourceType::Struct { name } = self.unalias(ty)? {
                        let TypeDecl::Struct { fields, .. } = self.declaration(name)? else {
                            unreachable!("validated struct reference")
                        };
                        let types: Vec<_> = fields.iter().map(|f| f.source_type.clone()).collect();
                        for (arg, ty) in args.into_iter().zip(types) {
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
            SourceType::Enum { name }
            | SourceType::Struct { name }
            | SourceType::Alias { name } => Ok(format!("XlsSource.{}", identifier(name)?)),
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
        for declaration in &self.map().declarations {
            let name = identifier(declaration.name())?;
            match declaration {
                TypeDecl::Enum { name: original, .. } => {
                    writeln!(out, "inductive {name} where")?;
                    for (member, _) in self.enum_cases(original)? {
                        writeln!(out, "  | {}", identifier(&member)?)?;
                    }
                    writeln!(out, "  deriving _root_.Repr, _root_.DecidableEq\n")?;
                }
                TypeDecl::Struct { fields, .. } => {
                    writeln!(out, "structure {name} where")?;
                    for field in fields {
                        writeln!(
                            out,
                            "  {} : {}",
                            identifier(&field.name)?,
                            self.source_type(&field.source_type)?
                        )?;
                    }
                    writeln!(out, "  deriving _root_.Repr\n")?;
                }
                TypeDecl::Alias { target, .. } => {
                    writeln!(out, "abbrev {name} := {}\n", self.source_type(target)?)?;
                }
            }
        }
        out.push_str("end XlsSource\n\nnamespace XlsCodec\n\n");
        out.push_str(BOOL_REPRESENTATION);
        for declaration in &self.map().declarations {
            let name = identifier(declaration.name())?;
            let nominal = self.named_type(declaration.name())?;
            let source_type = self.source_type(&nominal)?;
            let physical = source_shape(&nominal, &self.map().declarations)?;
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
        let node = self.block().graph.get_node(reference);
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
        if let NodePayload::Literal(value) = &self.block().graph.get_node(reference).payload {
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
            &self.block().graph.get_node(selector).payload
        {
            args.iter()
                .rev()
                .map(|reference| {
                    if let NodePayload::Binop(Binop::Eq, left, right) =
                        self.block().graph.get_node(*reference).payload
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
        let block = self.block();
        let node = block.graph.get_node(reference);
        let target = self.node_types[reference.index].as_ref();
        let operand = |r| self.typed_operand(r, target, depth);
        let boolean = |r| self.typed_operand(r, Some(&SourceType::Bool), depth);
        match &node.payload {
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
            } if block.graph.get_node(*selector).ty == Type::Bits(1) && cases.len() == 2 => {
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
        let expression = lean::node_expression(self.block(), reference, |operand| {
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
        let block = self.block();
        let mut out = format!(
            "-- Readable execution of authenticated XLS block {}.\n-- Every equation comes from its typed graph; source types only name values.\nimport Generated\nimport ReadableTypes\n\nnamespace XlsReadable\n\nstructure State where\n",
            lean::comment(&block.graph.name)
        );
        for (index, register) in block.graph.registers.iter().enumerate() {
            let ty = if let Some(ty) = self.register_type(&register.name)? {
                self.source_type(&ty)?
            } else {
                lean_type(&register.ty)?
            };
            writeln!(out, "  {} : {ty}", self.register_fields[index])?;
        }
        out.push_str(
            r#"  deriving Repr

-- Physical ports retain their exact names, directions and widths.
abbrev Input := XlsBlock.Input
abbrev Output := XlsBlock.Output
abbrev Assertions := XlsBlock.Assertions

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
            .map(|port| {
                Ok((
                    identifier(block.graph.port_name(port))?,
                    self.typed_operand(block.graph.output_value(port), None, 0)?,
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
                    identifier(&ir::node_textual_id(&block.graph, reference))?,
                    self.typed_operand(activate, Some(&SourceType::Bool), 0)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        writeln!(
            out,
            "  {{ next := {},\n    output := {},\n    assertions := {} }}\n",
            record_value(&next),
            record_value(&output),
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
                "  eval q {{ x with {} := BitVec.ofNat 1 {} }}",
                identifier(block.graph.port_name(reset.port))?,
                u8::from(!reset.active_low)
            )?;
        } else {
            out.push_str("  eval q x\n");
        }
        out.push_str("\nend XlsReadable\n");
        Ok(out)
    }

    /// Emit total state/result conversions and their round-trip proof source.
    ///
    /// Inputs, outputs, and assertions retain the raw types and use identity
    /// conversions; attached register values use their source codecs.
    fn representation(&self) -> Result<String> {
        let mut out = String::from("import ReadableGenerated\n\nnamespace XlsReadable\n\n");
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
        out.push_str(r#"def encodeInput (x : Input) : XlsBlock.Input := x
def decodeInput (x : XlsBlock.Input) : Input := x
def encodeOutput (x : Output) : XlsBlock.Output := x
def decodeOutput (x : XlsBlock.Output) : Output := x
def encodeAssertions (x : Assertions) : XlsBlock.Assertions := x
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
@[simp] theorem decode_encode_evaluation (e : Evaluation) : decodeEvaluation (encodeEvaluation e) = e := by
  cases e
  simp [decodeEvaluation, encodeEvaluation, decodeOutput, encodeOutput, decodeAssertions, encodeAssertions]
@[simp] theorem encode_decode_evaluation (e : XlsBlock.Evaluation) : encodeEvaluation (decodeEvaluation e) = e := by
  cases e
  simp [decodeEvaluation, encodeEvaluation, decodeOutput, encodeOutput, decodeAssertions, encodeAssertions]

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
        let mut out = String::from("import ReadableRepresentation\n\nnamespace XlsReadable\n\n");
        out.push_str(BOOL_CORRESPONDENCE);
        // Expose a record's fields only at a constructor. Unfolding its encoder
        // on an if or a decoded value would hide the structural rewrite rules.
        for declaration in &self.map().declarations {
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
        for name in enums {
            writeln!(out, "  cases {name}")?;
        }
        let structural_rules = {
            // Fix each function: bare apply_ite can distribute nested
            // conditionals through each other. Projections also cross selected
            // records and physical tuples, not just their enclosing codecs.
            let mut conditional_rules = BTreeSet::from([
                "apply_ite XlsCodec.encodeBool".to_owned(),
                "apply_ite XlsCodec.decodeBool".to_owned(),
            ]);
            for declaration in &self.map().declarations {
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
            for node in &self.block().graph.nodes {
                if let NodePayload::TupleIndex { tuple, index } = &node.payload {
                    conditional_rules.insert(format!(
                        "apply_ite (fun (value : {}) => {})",
                        lean_type(&self.block().graph.get_node(*tuple).ty)?,
                        tuple_element("value", *index)
                    ));
                }
            }
            let mut rules = conditional_rules.into_iter().collect::<Vec<_>>();
            for declaration in self.map().declarations.iter().rev() {
                let name = identifier(declaration.name())?;
                // Keep codec heads until roundtrips cancel.
                rules.push(format!("XlsCodec.{name}.encode_decode"));
                if let TypeDecl::Struct { name, .. } = declaration {
                    rules.push(identifier(&format!("encode_{name}_mk"))?);
                }
            }
            rules.push("XlsCodec.decode_encodeBool".into());
            rules.push("XlsCodec.encode_decodeBool".into());
            rules.join(",\n       ")
        };
        writeln!(
            out,
            r#"  all_goals
    dsimp (config := {{ instances := true }}) only
      [eval, XlsBlock.eval, encodeEvaluation, encodeState, encodeInput,
       encodeOutput, encodeAssertions]
  all_goals
    -- Distribute encoders before exposing constructors and cancelling roundtrips.
    simp (config := {{ instances := true }}) only
      [{structural_rules}]
  all_goals
    -- Keep raw one-bit operations and selectors in the readable Boolean form.
    simp (config := {{ instances := true }}) only
      [XlsCodec.raw_select, raw_flag, raw_not, raw_and, raw_or, decode_ofBool,
       {structural_rules}]"#
        )?;
        let enum_encoders = self
            .map()
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
        out.push_str("  all_goals rfl\n\n");
        out.push_str(CORRESPONDENCE);
        out.push_str("\nend XlsReadable\n");
        Ok(out)
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

"#;

/// Step, reset, and trace proof source derived from the generated evaluation
/// theorem.
const CORRESPONDENCE: &str = r#"theorem decoded_eval (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeEvaluation (XlsBlock.eval q x) = eval (decodeState q) (decodeInput x) := by
  have h := congrArg decodeEvaluation (encoded_eval (decodeState q) x)
  simpa [encodeInput, decodeInput] using h.symm

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
  simpa [reset, XlsBlock.reset, encodeInput] using encoded_eval q _
theorem decoded_reset (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeEvaluation (XlsBlock.reset q x) = reset (decodeState q) (decodeInput x) := by
  have h := congrArg decodeEvaluation (encoded_reset (decodeState q) x)
  simpa [encodeInput, decodeInput] using h.symm

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
  | succ t ih => simp only [trace, rawTrace, encoded_step, ih]
theorem decoded_trace (initial : XlsBlock.State) (input : Nat → XlsBlock.Input) (t : Nat) :
    decodeState (rawTrace initial input t) =
      trace (decodeState initial) (fun n => decodeInput (input n)) t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [trace, rawTrace, decoded_step, ih]
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
    simpa [decoded_step, decodeInput, encodeInput] using h
"#;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use crate::block::source_map::{self as source_types, Attachment};

    use super::*;

    fn checked(source: &str, block: &str, register: &str, source_type: &str) -> ValidatedSourceMap {
        let source_path = Path::new("readable_fixture.x");
        let map = source_types::collect_source_map(
            block,
            source,
            source_path,
            &[Attachment {
                register: register.into(),
                source_type: source_type.into(),
            }],
        )
        .unwrap();
        source_types::validate_source_map(&map, block, source, source_path).unwrap()
    }

    // Verifies: The bundle retains total source types and separate readable
    // equations. Catches: Missing enum encodings, raw-evaluator delegation,
    // or fixed arbiter names.
    #[test]
    fn readable_machine_has_total_nominal_types_and_its_own_execution() {
        let checked = checked(
            include_str!("../../tests/fixtures/readable_machine/machine.x"),
            include_str!("../../tests/fixtures/readable_machine/machine.block.ir"),
            "machine",
            "Machine",
        );
        let bundle = emit(&checked).unwrap();
        assert_eq!(
            bundle
                .files
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            [
                "Generated.lean",
                "ReadableTypes.lean",
                "ReadableGenerated.lean",
                "ReadableRepresentation.lean",
                "ReadableCorrespondence.lean",
            ]
        );
        assert_eq!(bundle.files[0].1, lean::emit(checked.block()).unwrap());
        let types = &bundle.files[1].1;
        assert!(types.contains("inductive «Phase»"));
        assert!(types.contains("| .«Active» => BitVec.ofNat 2 2"));
        assert!(types.contains("| .«encoding1» => BitVec.ofNat 2 1"));
        assert!(types.contains("| .«encoding3» => BitVec.ofNat 2 3"));
        let executable = &bundle.files[2].1;
        assert!(executable.contains("(«old»).«flags»"));
        assert!(executable.contains("(«machine_flags»).«armed»"));
        assert!(executable.contains("if «phase_is_active» then «incoming» else «machine_sample»"));
        assert!(executable.contains("id=11"));
        assert!(!executable.contains("XlsBlock.eval"));
        assert!(!executable.contains("pendingA"));
        let correspondence = &bundle.files[4].1;
        assert!(correspondence.contains("theorem encoded_eval (q : State) (x : Input)"));
        assert!(correspondence.contains("theorem decoded_trace (initial : XlsBlock.State)"));
        // Constructor equations must preserve the nested nominal layout. The
        // fixture deliberately uses Phase 0/2 and Flags, not arbiter names.
        assert!(correspondence.contains(
            "private theorem «encode_Flags_mk» (field0 : _root_.Bool) (field1 : _root_.Bool) :\n    XlsCodec.«Flags».encode ⟨field0, field1⟩ =\n      ((XlsCodec.encodeBool (field0)), ((XlsCodec.encodeBool (field1)), ()))"
        ));
        assert!(correspondence.contains(
            "private theorem «encode_Machine_mk» (field0 : XlsSource.«Phase») (field1 : XlsSource.«Flags») (field2 : XlsSource.«Word») :\n    XlsCodec.«Machine».encode ⟨field0, field1, field2⟩ =\n      ((XlsCodec.«Phase».encode (field0)), ((XlsCodec.«Flags».encode (field1)), ((XlsCodec.«Word».encode (field2)), ())))"
        ));
        assert!(correspondence.contains("only [XlsCodec.«Phase».encode]"));
        assert!(!correspondence.contains("XlsCodec.«Priority»"));
        for forbidden in ["sorry", "native_decide", "bv_decide", "axiom "] {
            assert!(
                !bundle
                    .files
                    .iter()
                    .any(|(_, source)| source.contains(forbidden))
            );
        }
        assert_eq!(bundle.files, emit(&checked).unwrap().files);
    }

    // Verifies: Arbiter branches and correspondence rewrites follow the
    // physical graph. Catches: Lost fourth-encoding behavior or premature
    // codec unfolding in proofs.
    #[test]
    fn readable_arbiter_renders_graph_selected_order_and_fourth_encoding() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/arbiter/w2");
        let source = fs::read_to_string(directory.join("arbiter.x")).unwrap();
        let block = fs::read_to_string(directory.join("arbiter.block.ir")).unwrap();
        let checked = checked(&source, &block, "____state", "State");
        let bundle = emit(&checked).unwrap();
        let executable = &bundle.files[2].1;
        assert!(executable.contains("state» : XlsSource.«State»"));
        assert!(executable.contains("(«____state__1»).«priority»"));
        assert!(executable.contains("match «__state_priority» with"));
        for (case, first) in [("A", "A"), ("B", "B"), ("C", "C"), ("encoding3", "C")] {
            assert!(executable.contains(&format!("| .«{case}» => (if «available{first}»")));
        }
        assert!(executable.contains("else «__state_priority»"));
        assert!(executable.contains("id=310"));
        assert_eq!(bundle.files[0].1, lean::emit(checked.block()).unwrap());
        let proof = bundle.files[4]
            .1
            .split("theorem encoded_eval")
            .nth(1)
            .unwrap()
            .split("theorem decoded_eval")
            .next()
            .unwrap();
        // The failing broad simp unfolded codecs before their roundtrip rules
        // and lowered selectors to Nat. Pin this separation at the emitter
        // boundary; the native correspondence gate checks theorem closure.
        assert_eq!(
            proof
                .matches("dsimp (config := { instances := true }) only")
                .count(),
            2
        );
        assert_eq!(
            proof
                .matches("\n    simp (config := { instances := true }) only\n")
                .count(),
            2
        );
        assert!(proof.contains("apply_ite XlsCodec.«State».encode"));
        assert!(proof.contains("XlsCodec.«State».encode_decode"));
        assert!(proof.contains("«encode_State_mk»"));
        assert!(proof.contains("XlsCodec.raw_select, raw_flag, raw_not, raw_and, raw_or"));
        assert!(!proof.contains("\n       XlsCodec.decodeBool,"));
        assert!(!proof.contains("\n       XlsCodec.«State».decode,"));
        assert!(!proof.contains("instances := false"));
    }

    // Verifies: Selected records and fields receive codec and projection
    // rewrite rules. Catches: Proof source that cannot align selections
    // with their raw expressions.
    #[test]
    fn selected_fields_distribute_codecs_and_projections() {
        let source = include_str!("../../tests/fixtures/readable_machine/machine.x");
        let block = include_str!("../../tests/fixtures/readable_machine/selections.block.ir");
        let bundle = emit(&checked(source, block, "machine", "Machine")).unwrap();
        assert!(bundle.files[2].1.contains(
            "let «picked» : _root_.Bool := (if «choose» then «machine_flags_queued» else «machine_flags_armed»)"
        ));
        assert!(bundle.files[2].1.contains(
            "let «selected_flags» : XlsSource.«Flags» := (if «choose_flags» then «next_flags» else «machine_flags»)"
        ));
        assert!(bundle.files[2].1.contains("(«selected_flags»).«armed»"));
        // A constructor equation alone cannot align a selected Boolean field
        // or a projection from a selected record with its raw counterpart.
        // The native fixture check must also establish theorem closure.
        let proof = &bundle.files[4].1;
        assert!(proof.contains("apply_ite XlsCodec.encodeBool"));
        assert!(proof.contains("apply_ite XlsSource.«Flags».«armed»"));
        assert!(proof.contains(
            "apply_ite (fun (value : Prod (BitVec 1) (Prod (BitVec 1) (Unit))) => (value).1)"
        ));
    }

    // Verifies: Source members, local bindings, and register fields keep
    // distinct names. Catches: Source names capturing generated codecs,
    // built-ins, or other registers.
    #[test]
    fn source_and_register_names_cannot_capture_generated_names() {
        let source = include_str!("../../tests/fixtures/readable_machine/machine.x")
            .replace("armed: bool, queued: bool", "encode: bool, decode: bool")
            .replace("type Word = u2;", "type Bool = bool;\ntype Word = u2;");
        let block = include_str!("../../tests/fixtures/readable_machine/machine.block.ir")
            .replace("machine_flags_armed", "machine_flags_encode")
            .replace("machine_flags_queued", "machine_flags_decode")
            .replace("phase_is_active", "Bool")
            .replace("  reg machine", "  reg a(bits[1], reset_value=0)\n  reg _a(bits[1], reset_value=0)\n  reg register_1(bits[1], reset_value=0)\n  reg machine")
            .replace("  telemetry:", "  old_a: bits[1] = register_read(register=a, id=20)\n  old_b: bits[1] = register_read(register=_a, id=21)\n  old_c: bits[1] = register_read(register=register_1, id=22)\n  write_a: () = register_write(old_a, register=a, reset=rst, id=23)\n  write_b: () = register_write(old_b, register=_a, reset=rst, id=24)\n  write_c: () = register_write(old_c, register=register_1, reset=rst, id=25)\n  telemetry:");
        let bundle = emit(&checked(&source, &block, "machine", "Machine")).unwrap();
        let types = &bundle.files[1].1;
        let nominal = types.split("end XlsSource").next().unwrap();
        assert!(nominal.contains("«encode» : _root_.Bool"));
        assert!(!nominal.contains("def encode"));
        assert!(types.contains("namespace XlsCodec"));
        assert!(types.contains("def encode (x : XlsSource.«Flags»)"));
        let executable = &bundle.files[2].1;
        assert!(!executable.contains("let «Bool»"));
        for name in ["a", "register_1_", "register_1", "machine"] {
            assert!(
                executable.contains(&format!("  «{name}» :")),
                "missing independent register {name}"
            );
        }
    }

    // Verifies: A node named Prod cannot hide the physical tuple type
    // constructor. Catches: Retained local names making later tuple
    // annotations resolve incorrectly.
    #[test]
    fn retained_node_names_cannot_capture_physical_tuple_types() {
        let source = include_str!("../../tests/fixtures/readable_machine/machine.x");
        let block = include_str!("../../tests/fixtures/readable_machine/machine.block.ir")
            .replace("phase_is_active", "Prod")
            .replace("  telemetry:", "  raw_pair: (bits[1], bits[2]) = tuple(Prod, selected_sample, id=17)\n  unpacked: bits[2] = tuple_index(raw_pair, index=1, id=18)\n  telemetry:")
            .replace("output_port(selected_sample,", "output_port(unpacked,");
        let bundle = emit(&checked(source, &block, "machine", "Machine")).unwrap();
        let executable = &bundle.files[2].1;
        // The raw tuple depends on the named node, so its annotation occurs
        // after that binding and must still resolve Lean's type constructor.
        assert!(executable.contains("let «raw_pair» : Prod (BitVec 1)"));
        assert!(!executable.contains("let «Prod»"));
    }

    // Verifies: Enum selection follows physical values and selector bit order.
    // Catches: Reordered declarations changing branch selection or default
    // behavior.
    #[test]
    fn enum_selector_follows_physical_values_not_declaration_order() {
        let source = include_str!("../../tests/fixtures/readable_machine/machine.x")
            .replace("Idle = 0, Active = 2", "Active = 2, Idle = 0");
        let block = include_str!("../../tests/fixtures/readable_machine/machine.block.ir").replace(
            "  selected_sample: bits[2] = sel(phase_is_active, cases=[machine_sample, incoming], id=11)",
            "  idle_encoding: bits[2] = literal(value=0, id=17)\n  phase_is_idle: bits[1] = eq(machine_phase, idle_encoding, id=18)\n  phase_selector: bits[2] = concat(phase_is_active, phase_is_idle, id=19)\n  selected_sample: bits[2] = priority_sel(phase_selector, cases=[machine_sample, incoming], default=active_encoding, id=11)");
        let bundle = emit(&checked(&source, &block, "machine", "Machine")).unwrap();
        let executable = &bundle.files[2].1;
        for (member, result) in [
            ("Active", "incoming"),
            ("Idle", "machine_sample"),
            ("encoding1", "active_encoding"),
            ("encoding3", "active_encoding"),
        ] {
            assert!(executable.contains(&format!("| .«{member}» => «{result}»")));
        }
    }

    // Verifies: Shared selector nodes remain shared during annotation and
    // emission. Catches: Repeated traversal or expansion along
    // exponentially many graph paths.
    #[test]
    fn shared_selector_dag_does_not_expand_annotation_paths() {
        let source = include_str!("../../tests/fixtures/readable_machine/machine.x");
        let mut nodes = String::new();
        let mut previous = ["next_machine".to_owned(), "next_machine".to_owned()];
        for index in 0..50 {
            let name = format!("choice_{index}");
            writeln!(nodes, "  {name}: (bits[2], (bits[1], bits[1]), bits[2]) = sel(phase_is_active, cases=[{}, {}], id={})",
                previous[0], previous[1], 100 + index).unwrap();
            previous = [previous[1].clone(), name];
        }
        nodes.push_str(&format!(
            "  write: () = register_write({}, register=machine, reset=rst, id=13)",
            previous[1]
        ));
        let block = include_str!("../../tests/fixtures/readable_machine/machine.block.ir").replace(
            "  write: () = register_write(next_machine, register=machine, reset=rst, id=13)",
            &nodes,
        );
        // Fifty shared choices encode exponentially many paths but only fifty
        // additional nodes. Named choices also remain shared in the executable.
        let bundle = emit(&checked(source, &block, "machine", "Machine")).unwrap();
        assert_eq!(bundle.files[2].1.matches("let «choice_").count(), 50);
    }
}
