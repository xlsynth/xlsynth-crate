// SPDX-License-Identifier: Apache-2.0

//! Checked BX and DSLX declarations select representations at block and
//! function boundaries. Both interfaces instantiate the same physical IR
//! equations. Correspondence checks the codecs and call adapters; agreement
//! with XLS execution is checked separately.

use std::{borrow::Cow, collections::BTreeSet, fmt::Write};

mod callable_names;
mod calls;
mod forwarding;
mod source_names;

use anyhow::{Context, Result, bail, ensure};
use xlsynth_pir::ir::{self, NodePayload, NodeRef, Type};

use crate::block::{
    BlockContext,
    raw::{self as lean, LeanBlockBundle, identifier, lean_type},
    source_types::{SourceType, StatefulInterface, TypeDecl, resolve_aliases, source_shape},
};
use callable_names::CallableNames;
use calls::Calls;
use source_names::{declaring_name, module_identifier, source_type_in, write_struct};

/// Project an element from a physical XLS tuple, whose final pair ends in unit.
fn tuple_element(tuple: &str, index: usize) -> String {
    format!("({tuple}){}.1", ".2".repeat(index))
}

/// Build the nested pairs and final unit used by the raw model for every XLS
/// tuple.
fn tuple_value(elements: impl DoubleEndedIterator<Item = String>) -> String {
    elements.rev().fold("()".to_string(), |tail, element| {
        format!("({element}, {tail})")
    })
}

/// The checked declarations used by the public interfaces and their codecs.
struct Presentation<'a> {
    block: &'a BlockContext<'a>,
    declarations: &'a [TypeDecl],
    bx_namespace: &'a str,
    callables: &'a CallableNames<'a>,
    interface: &'a StatefulInterface,
    register_fields: Vec<String>,
}

/// Keep the linker's identity for nominal type comparisons and recover the
/// public BX spelling only when printing Lean.
fn type_name(key: &str) -> Cow<'_, str> {
    match xlsynth_bx::library_item(key) {
        Some((module, item)) => format!("{}::{item}", module.replace('.', "::")).into(),
        None => key.into(),
    }
}

/// Quote a displayed type name consistently across its Lean declaration, codecs
/// and proofs.
fn type_identifier(key: &str) -> Result<String> {
    identifier(&type_name(key))
}

/// The first module is the ordinary standalone IR export. Later modules add
/// checked call adapters, the block interface, and its representation laws.
pub(super) fn emit(
    block: &BlockContext<'_>,
    declarations: &[TypeDecl],
    interface: &StatefulInterface,
) -> Result<LeanBlockBundle> {
    source_names::check_type_namespaces(declarations, interface)?;
    let raw = lean::emit_with_context(block, "XlsBlock")?;
    let bx_namespace = if interface.bx_module.contains(':') {
        format!("file:{}", interface.bx_module)
    } else {
        interface.bx_module.clone()
    };
    let callables = CallableNames::new(block, declarations, interface, &bx_namespace)?;
    let presentation = Presentation {
        block,
        declarations,
        interface,
        bx_namespace: &bx_namespace,
        callables: &callables,
        register_fields: block
            .graph
            .registers
            .iter()
            .map(|r| identifier(&r.name))
            .collect::<Result<_>>()?,
    };
    let calls = Calls::new(&presentation)?;
    let mut bundle = LeanBlockBundle {
        files: vec![
            ("Generated.lean".into(), raw),
            ("ReadableTypes.lean".into(), presentation.types_module()?),
            (
                "ReadableGenerated.lean".into(),
                presentation.executable(&calls)?,
            ),
            ("ReadableRepresentation.lean".into(), representation_laws()),
            ("ReadableCorrespondence.lean".into(), calls.correspondence()),
        ],
    };
    for (_, module) in &mut bundle.files {
        super::library_notation::add_to_module(module);
    }
    Ok(bundle)
}

impl<'a> Presentation<'a> {
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

    fn source_type(&self, ty: &SourceType) -> Result<String> {
        source_type_in(self.bx_namespace, ty, None, &|_| false)
    }

    /// Encode toward physical values or decode toward the requested source
    /// type.
    ///
    /// Absent annotations and plain bit vectors already use physical values.
    /// A vector keeps its length in both representations. Only elements that
    /// need a codec require mapping a conversion over the vector.
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
                    Ok(format!("(L.Vector.map (fun item => {item}) ({value}))"))
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
                type_identifier(name)?
            )),
        }
    }
    /// Add collision-free constructors for every unnamed physical enum
    /// encoding.
    ///
    /// Giving every bit pattern a case lets both codec round trips hold for
    /// arbitrary initial register values. An unnamed case carries its physical
    /// encoding without attributing a source constructor to it.
    ///
    /// Keeping unnamed encodings in the readable proof domain is provisional.
    /// Restricting proofs to declared enum values would also require proving
    /// that initialization and transitions preserve those values.
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

    /// Emit the enum and return the inductive type name for deriving instances.
    ///
    /// One collision with a Lean-generated member moves all constructors under
    /// `Constructors` on a separate inductive type. Public abbreviations keep
    /// each source spelling usable in terms and patterns. Unlike Lean's
    /// `export` aliases, these abbreviations also support absolute references.
    fn write_enum(
        &self,
        out: &mut String,
        original: &str,
        namespace: &str,
        name: &str,
    ) -> Result<String> {
        let cases = self.enum_cases(original)?;
        let needs_backing = cases.iter().any(|(member, _)| {
            lean::generated_member(member)
                || matches!(
                    member.as_str(),
                    "ofNat" | "ofNat_ctorIdx" | "ctorElim" | "ctorElimType"
                )
        });
        let (backing, prefix) = if needs_backing {
            ensure!(
                !cases.iter().any(|(member, _)| member == "_root_"),
                "cannot emit {original}._root_: Lean reserves this name for the root namespace and this enum needs a public constructor abbreviation"
            );
            (format!("«xls:implementation».{name}"), "Constructors.")
        } else {
            (name.to_owned(), "")
        };
        let members = cases
            .iter()
            .map(|(member, _)| identifier(member))
            .collect::<Result<Vec<_>>>()?;
        writeln!(out, "inductive {backing} where")?;
        for member in &members {
            writeln!(out, "  | {prefix}{member}")?;
        }
        out.push('\n');
        if needs_backing {
            writeln!(out, "abbrev {name} := {backing}\n")?;
            for member in &members {
                // A match on this constructor infers its pattern namespace from
                // the result type.
                writeln!(
                    out,
                    "@[match_pattern] abbrev {name}.{member} : _root_.{namespace}.{name} := {backing}.{prefix}{member}"
                )?;
            }
            out.push('\n');
        }
        Ok(backing)
    }

    /// Emit source declarations, total codecs, and proofs of both codec round
    /// trips.
    ///
    /// Derive instances for the underlying inductive types and structures in
    /// the `XlsCodec` namespace to keep instance names out of DSLX namespaces.
    fn types_module(&self) -> Result<String> {
        let mut out = String::from("import Std\n\n");
        // Deriving beside Packet would occupy its module's source name
        // instReprPacket. Generate instances in XlsCodec in dependency order.
        let mut instances = String::new();
        // Source members and generated codecs have disjoint namespaces. A
        // source field named `encode` must remain an ordinary record field.
        for declaration in self.declarations {
            let (module, local) = declaring_name(self.bx_namespace, declaration.name());
            let namespace = module_identifier(module)?;
            let name = source_names::declaration_identifier(module, local, "type")?;
            writeln!(out, "namespace {namespace}\n")?;
            let derived = match declaration {
                TypeDecl::Enum { name: original, .. } => Some((
                    "L.Repr, L.DecidableEq",
                    self.write_enum(&mut out, original, &namespace, &name)?,
                )),
                TypeDecl::Struct { fields, .. } => {
                    let backing = write_struct(
                        &mut out,
                        self.bx_namespace,
                        module,
                        local,
                        fields,
                        self.declarations,
                    )?;
                    Some(("L.Repr", backing))
                }
                TypeDecl::Alias { target, .. } => {
                    writeln!(
                        out,
                        "abbrev {name} := {}\n",
                        source_type_in(
                            self.bx_namespace,
                            target,
                            Some((module, self.declarations)),
                            &|_| false
                        )?
                    )?;
                    None
                }
            };
            if let Some((classes, target)) = derived {
                writeln!(
                    instances,
                    "deriving instance {classes} for _root_.{namespace}.{target}"
                )?;
            }
            writeln!(out, "end {namespace}\n")?;
        }
        out.push_str("namespace XlsCodec\n\nattribute [local simp] L.Function.comp_def\n\n");
        out.push_str(&instances);
        out.push('\n');
        out.push_str(BOOL_REPRESENTATION);
        for declaration in self.declarations {
            let name = type_identifier(declaration.name())?;
            let nominal = self.named_type(declaration.name())?;
            // Codec namespaces and their arguments have their own scope,
            // independent of the readable evaluator's locals.
            let source_type = source_type_in(self.bx_namespace, &nominal, None, &|_| true)?;
            let physical = source_shape(&nominal, self.declarations)?;
            writeln!(out, "namespace {name}\n")?;
            match declaration {
                TypeDecl::Enum {
                    name: original,
                    width,
                    ..
                } => {
                    let cases = self.enum_cases(original)?;
                    writeln!(out, "def encode : {source_type} → L.BitVec {width}")?;
                    for (member, value) in &cases {
                        writeln!(
                            out,
                            "  | .{} => L.BitVec.ofNat {width} {value}",
                            identifier(member)?
                        )?;
                    }
                    writeln!(
                        out,
                        "\ndef decode (x : L.BitVec {width}) : {source_type} :="
                    )?;
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
                        "@[simp] theorem encode_decode (x : L.BitVec {width}) : encode (decode x) = x := by\n  rcases two_bit_cases x with rfl | rfl | rfl | rfl <;> rfl\n"
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
                        lean::source_record_value(&decoded)
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

    fn executable(&self, calls: &Calls) -> Result<String> {
        let block = self.block;
        let mut out =
            String::from("import Generated\nimport ReadableTypes\n\nnamespace XlsReadable\n\n");
        out.push_str(&calls.declarations);
        let mut fields = Vec::new();
        let mut preceding = BTreeSet::new();
        for (field, register) in self.register_fields.iter().zip(&block.graph.registers) {
            let ty = if let Some(ty) = self.register_type(&register.name)? {
                source_type_in(self.bx_namespace, &ty, None, &|name| {
                    preceding.contains(name)
                })?
            } else {
                lean_type(&register.ty)?
            };
            preceding.insert(field.trim_matches(['«', '»']));
            fields.push((field.clone(), ty));
        }
        out.push_str(&lean::source_record("State", &fields)?);
        let interface = self.interface;
        fields.clear();
        preceding.clear();
        for port in block.graph.input_ports() {
            let name = block.graph.port_name(port);
            let ty = source_type_in(self.bx_namespace, &interface.inputs[name], None, &|n| {
                preceding.contains(n)
            })?;
            fields.push((identifier(name)?, ty));
            preceding.insert(name);
        }
        out.push_str(&lean::source_record("Input", &fields)?);
        let types = interface
            .outputs
            .iter()
            .map(|ty| self.source_type(ty))
            .collect::<Result<Vec<_>>>()?;
        writeln!(out, "abbrev Output := {}", product_type(&types, false))?;
        out.push_str("abbrev Assertions := XlsBlock.Assertions\n\n");
        out.push_str(&lean::structure_header("Evaluation"));
        out.push_str(
            r#"  next : State
  output : Output
  assertions : Assertions
  deriving L.Repr

"#,
        );

        self.write_representation(&mut out)?;
        let encoded_args = "(encodeState q) (encodeInput x)";
        let decoded = format!("decodeEvaluation ({} {encoded_args})", calls.eval);
        let forwarded = self.forwarded_evaluation()?;
        if forwarded.is_some() {
            out.push_str(forwarding::LEAN_SUPPORT);
        }
        writeln!(out, "def eval (q : State) (x : Input) : Evaluation :=")?;
        if let Some(fields) = &forwarded {
            writeln!(out, "  let decoded := {decoded}\n  {fields}\n")?;
        } else {
            writeln!(out, "  {decoded}\n")?;
        }
        let evaluation_proof = if forwarded.is_some() {
            "by simp only [eval, preserveForward_eq]"
        } else {
            "rfl"
        };
        writeln!(
            out,
            "theorem eval_eq_decoded (q : State) (x : Input) :\n    eval q x = {decoded} := {evaluation_proof}\n"
        )?;
        out.push_str("def step (q : State) (x : Input) : State := (eval q x).next\ndef observe (q : State) (x : Input) : Output := (eval q x).output\ndef assertions (q : State) (x : Input) : Assertions := (eval q x).assertions\n\n");
        let assertions = block
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.payload, NodePayload::Assert { .. }))
            .map(|(index, _)| identifier(&ir::node_textual_id(&block.graph, NodeRef { index })))
            .collect::<Result<Vec<_>>>()?;
        let holds = assertions
            .iter()
            .rev()
            .fold("L.True".to_owned(), |tail, name| {
                format!("L.And (checks.{name} = true) ({tail})")
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
                lean::storage_field(&identifier(block.graph.port_name(reset.port))?),
                !reset.active_low
            )?;
        } else {
            out.push_str("  eval q x\n");
        }
        out.push_str("\nend XlsReadable\n");
        writeln!(
            out,
            "\nabbrev {} := _root_.XlsReadable.eval",
            self.callables.block.trim_start_matches("_root_.")
        )?;
        out.push_str(&self.callables.aliases);
        Ok(out)
    }

    /// These conversions run at the public evaluation boundary. Proofs of
    /// their inverse laws live in ReadableRepresentation.
    fn write_representation(&self, out: &mut String) -> Result<()> {
        let encoded = self
            .block
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
            .block
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
            lean::source_record_value(&encoded)
        )?;
        writeln!(
            out,
            "def decodeState (q : XlsBlock.State) : State :=\n  {}\n",
            lean::source_record_value(&decoded)
        )?;
        let interface = self.interface;
        for encode in [true, false] {
            let (operation, from, to) = if encode {
                ("encode", "", "XlsBlock.")
            } else {
                ("decode", "XlsBlock.", "")
            };
            let inputs = self
                .block
                .graph
                .input_ports()
                .map(|port| {
                    let name = self.block.graph.port_name(port);
                    let ty = &interface.inputs[name];
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
                lean::source_record_value(&inputs)
            )?;
            let outputs = interface
                .outputs
                .iter()
                .zip(self.block.graph.output_ports())
                .enumerate()
                .map(|(index, (ty, port))| {
                    let field = identifier(self.block.graph.port_name(port))?;
                    let source = if encode {
                        product_element("x", index, interface.outputs.len(), false)
                    } else {
                        format!("x.{field}")
                    };
                    Ok((field, self.convert(Some(ty), &source, encode)?))
                })
                .collect::<Result<Vec<_>>>()?;
            let expression = if encode {
                lean::source_record_value(&outputs)
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
        out.push_str(
            r#"def encodeAssertions (x : Assertions) : XlsBlock.Assertions := x
def decodeAssertions (x : XlsBlock.Assertions) : Assertions := x

def encodeEvaluation (e : Evaluation) : XlsBlock.Evaluation :=
  ⟨encodeState e.next, encodeOutput e.output, encodeAssertions e.assertions⟩
def decodeEvaluation (e : XlsBlock.Evaluation) : Evaluation :=
  ⟨decodeState e.next, decodeOutput e.output, decodeAssertions e.assertions⟩

"#,
        );
        Ok(())
    }
}

fn representation_laws() -> String {
    r#"import ReadableGenerated

namespace XlsReadable

attribute [local simp] L.Function.comp_def

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
"#.to_owned()
}

/// Render the readable product type of tuple members or output ports.
///
/// A single output has its own type. An explicit one-element source tuple
/// pairs that type with unit so that the tuple keeps its shape.
fn product_type(types: &[String], singleton_tuple: bool) -> String {
    match types {
        [] => "L.Unit".into(),
        [ty] if singleton_tuple => format!("(L.Prod ({ty}) L.Unit)"),
        [ty] => ty.clone(),
        [head, rest @ ..] => format!("(L.Prod ({head}) ({}))", product_type(rest, false)),
    }
}

/// Construct a readable value with the same singleton convention as
/// `product_type`.
fn product_value(elements: &[String], singleton_tuple: bool) -> String {
    match elements {
        [] => "()".into(),
        [value] if singleton_tuple => format!("({value}, ())"),
        [value] => value.clone(),
        [head, rest @ ..] => format!("({head}, {})", product_value(rest, false)),
    }
}

/// Project an in-range element, taking the final tail directly unless it ends
/// in unit.
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
const BOOL_REPRESENTATION: &str = r#"def encodeBool (b : L.Bool) : L.BitVec 1 := L.BitVec.ofBool b
def decodeBool (b : L.BitVec 1) : L.Bool := b == 1

theorem one_bit_cases (b : L.BitVec 1) : b = 0 ∨ b = 1 := by
  have bound := L.BitVec.isLt b
  have alternatives : L.BitVec.toNat b = 0 ∨ L.BitVec.toNat b = 1 := by omega
  rcases alternatives with zero | one
  · exact L.Or.inl (L.BitVec.eq_of_toNat_eq (by simpa using zero))
  · exact L.Or.inr (L.BitVec.eq_of_toNat_eq (by simpa using one))

theorem two_bit_cases (b : L.BitVec 2) : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by
  have bound := L.BitVec.isLt b
  have alternatives : L.BitVec.toNat b = 0 ∨ L.BitVec.toNat b = 1 ∨ L.BitVec.toNat b = 2 ∨ L.BitVec.toNat b = 3 := by omega
  rcases alternatives with zero | one | two | three
  · exact L.Or.inl (L.BitVec.eq_of_toNat_eq (by simpa using zero))
  · exact L.Or.inr (L.Or.inl (L.BitVec.eq_of_toNat_eq (by simpa using one)))
  · exact L.Or.inr (L.Or.inr (L.Or.inl (L.BitVec.eq_of_toNat_eq (by simpa using two))))
  · exact L.Or.inr (L.Or.inr (L.Or.inr (L.BitVec.eq_of_toNat_eq (by simpa using three))))

@[simp] theorem decode_encodeBool (b : L.Bool) : decodeBool (encodeBool b) = b := by
  cases b <;> rfl
@[simp] theorem encode_decodeBool (b : L.BitVec 1) : encodeBool (decodeBool b) = b := by
  rcases one_bit_cases b with rfl | rfl <;> rfl

"#;
/// Step, reset, and trace proof source derived from the generated evaluation
/// theorem.
const CORRESPONDENCE: &str = r#"theorem decoded_eval (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeEvaluation (XlsBlock.eval q x) = eval (decodeState q) (decodeInput x) := by
  have h := L.congrArg decodeEvaluation (encoded_eval (decodeState q) (decodeInput x))
  simpa using L.Eq.symm h

theorem encoded_step (q : State) (x : Input) :
    encodeState (step q x) = XlsBlock.step (encodeState q) (encodeInput x) := by
  have h := L.congrArg XlsBlock.Evaluation.next (encoded_eval q x)
  exact h
theorem decoded_step (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeState (XlsBlock.step q x) = step (decodeState q) (decodeInput x) := by
  have h := L.congrArg Evaluation.next (decoded_eval q x)
  exact h
theorem encoded_observe (q : State) (x : Input) :
    encodeOutput (observe q x) = XlsBlock.observe (encodeState q) (encodeInput x) := by
  simpa only [encodeEvaluation, observe, XlsBlock.observe] using
    L.congrArg XlsBlock.Evaluation.output (encoded_eval q x)
theorem decoded_observe (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeOutput (XlsBlock.observe q x) = observe (decodeState q) (decodeInput x) := by
  simpa only [decodeEvaluation, observe, XlsBlock.observe] using
    L.congrArg Evaluation.output (decoded_eval q x)
theorem encoded_assertions (q : State) (x : Input) :
    encodeAssertions (assertions q x) = XlsBlock.assertions (encodeState q) (encodeInput x) := by
  have h := L.congrArg XlsBlock.Evaluation.assertions (encoded_eval q x)
  exact h
theorem decoded_assertions (q : XlsBlock.State) (x : XlsBlock.Input) :
    decodeAssertions (XlsBlock.assertions q x) = assertions (decodeState q) (decodeInput x) := by
  have h := L.congrArg Evaluation.assertions (decoded_eval q x)
  exact h
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
  have h := L.congrArg decodeEvaluation (encoded_reset (decodeState q) (decodeInput x))
  simpa using L.Eq.symm h

def trace (initial : State) (input : L.Nat → Input) : L.Nat → State
  | 0 => initial
  | t + 1 => step (trace initial input t) (input t)
def rawTrace (initial : XlsBlock.State) (input : L.Nat → XlsBlock.Input) : L.Nat → XlsBlock.State
  | 0 => initial
  | t + 1 => XlsBlock.step (rawTrace initial input t) (input t)

/-- The initial physical state is arbitrary, including every undeclared enum encoding. -/
theorem encoded_trace (initial : State) (input : L.Nat → Input) (t : L.Nat) :
    encodeState (trace initial input t) =
      rawTrace (encodeState initial) (fun n => encodeInput (input n)) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    exact L.Eq.trans (encoded_step _ _) (L.congrArg (fun q => XlsBlock.step q (encodeInput (input t))) ih)
theorem decoded_trace (initial : XlsBlock.State) (input : L.Nat → XlsBlock.Input) (t : L.Nat) :
    decodeState (rawTrace initial input t) =
      trace (decodeState initial) (fun n => decodeInput (input n)) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    exact L.Eq.trans (decoded_step _ _) (L.congrArg (fun q => step q (decodeInput (input t))) ih)
theorem trace_output (initial : State) (input : L.Nat → Input) (t : L.Nat) :
    encodeOutput (observe (trace initial input t) (input t)) =
      XlsBlock.observe (rawTrace (encodeState initial) (fun n => encodeInput (input n)) t)
        (encodeInput (input t)) := by
  rw [encoded_observe, encoded_trace]
theorem trace_assertions (initial : State) (input : L.Nat → Input) (t : L.Nat) :
    encodeAssertions (assertions (trace initial input t) (input t)) =
      XlsBlock.assertions (rawTrace (encodeState initial) (fun n => encodeInput (input n)) t)
        (encodeInput (input t)) := by
  rw [encoded_assertions, encoded_trace]

theorem clock_relation_iff (q : L.Nat → State) (x : L.Nat → Input) :
    (∀ t, q (t + 1) = step (q t) (x t)) ↔
      (∀ t, encodeState (q (t + 1)) = XlsBlock.step (encodeState (q t)) (encodeInput (x t))) := by
  constructor
  · intro run t
    rw [run t, encoded_step]
  · intro run t
    have h := L.congrArg decodeState (run t)
    simpa [decoded_step] using h
"#;
