// SPDX-License-Identifier: Apache-2.0

//! Typechecked source names for an explicitly attached XLS block register.
//!
//! The catalogue describes representation only. Exact input hashes and retained
//! projection names can reject stale or conflicting metadata, but cannot prove
//! that an operator attached the right source declaration to a hardware
//! register.
//!
//! Collection typechecks the supplied leaf source and checks each attachment.
//! Validation repeats those checks before binding a persisted map to an owned
//! block; readable emission accepts only that immutable validated pair.

use std::{collections::BTreeSet, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xlsynth::{
    XlsIrValue, XlsynthError, dslx,
    dslx_bridge::{BridgeBuilder, StructMemberData, convert_leaf_module},
};
use xlsynth_pir::ir::{self, NodePayload, NodeRef};

use crate::block::{self, Block};

/// Serializable version-1 declarations and register attachments for exact
/// inputs.
///
/// Deserialization checks the schema, not the declarations or bindings. Pass a
/// map through [`validate_source_map`] before emission; matching hashes do not
/// establish which compiler run, if any, related the source and block.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceMap {
    /// Schema version; collection emits 1 and validation accepts only 1.
    pub version: u32,
    /// Lowercase SHA-256 hex digest of the supplied DSLX text bytes.
    pub source_sha256: String,
    /// Lowercase SHA-256 hex digest of the complete supplied IR text bytes.
    pub block_sha256: String,
    /// Name of the block selected by [`block::parse`] without an explicit name.
    pub block_name: String,
    /// Supported leaf declarations in source order.
    pub declarations: Vec<TypeDecl>,
    /// Caller-supplied bindings, with at most one per register after
    /// validation.
    pub attachments: Vec<Attachment>,
}

/// An operator assertion, checked for shape and required retained field names.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Attachment {
    /// Exact physical register name in the selected block.
    pub register: String,
    /// Name of a supported leaf declaration in the source catalogue.
    pub source_type: String,
}

/// A supported leaf declaration with nominal names retained from native DSLX.
///
/// Catalogue order and struct field order follow the source. Enum values come
/// from native constant evaluation, rather than declaration position.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypeDecl {
    /// An unsigned two-bit enum with distinct named physical encodings.
    Enum {
        name: String,
        width: usize,
        members: Vec<EnumMember>,
    },
    /// A nonempty struct whose ordered fields determine its physical tuple
    /// shape.
    Struct {
        name: String,
        fields: Vec<StructField>,
    },
    /// A named reference to a supported type, retaining alias-chain identity.
    Alias { name: String, target: SourceType },
}

impl TypeDecl {
    /// The declared DSLX type name, shared by every declaration kind.
    pub fn name(&self) -> &str {
        match self {
            Self::Enum { name, .. } | Self::Struct { name, .. } | Self::Alias { name, .. } => name,
        }
    }
}

/// One named, native-evaluated enum encoding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnumMember {
    /// DSLX constructor name.
    pub name: String,
    /// Physical encoding, validated to be distinct and less than four.
    pub value: u64,
}

/// A source field in declaration order and its nominal or primitive type.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StructField {
    /// DSLX field name, checked against retained tuple-projection names.
    pub name: String,
    /// Field type before erasing nominal names to its physical shape.
    pub source_type: SourceType,
}

/// A primitive type or nominal reference retained in the source catalogue.
///
/// Nominal references remain distinct even when their hardware widths coincide.
/// DSLX's unsigned one-bit type, including `bool`, uses the Boolean variant.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceType {
    /// An unsigned one-bit value rendered as Lean `Bool`.
    Bool,
    /// Unsigned physical bits; collection uses this for widths greater than
    /// one.
    Bits { width: usize },
    /// A reference to an enum declaration in the same catalogue.
    Enum { name: String },
    /// A reference to a struct declaration in the same catalogue.
    Struct { name: String },
    /// A reference to an alias declaration in the same catalogue.
    Alias { name: String },
}

/// An immutable source map paired with the block checked against its metadata.
///
/// Only [`validate_source_map`] constructs this input to readable emission.
/// Its ownership prevents later metadata edits or replacement of the graph
/// from invalidating the checks; it does not certify compiler provenance.
pub struct ValidatedSourceMap {
    map: SourceMap,
    block: Block,
}

impl std::fmt::Debug for ValidatedSourceMap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ValidatedSourceMap")
            .field("map", &self.map)
            .finish_non_exhaustive()
    }
}

impl ValidatedSourceMap {
    /// Inspect the checked catalogue and bindings without changing them.
    pub fn map(&self) -> &SourceMap {
        &self.map
    }

    /// Inspect the block parsed from the exact IR text checked against the map.
    ///
    /// Its input and output nodes determine physical port identity, direction,
    /// tuple paths, and widths. Matching a port's width to a declaration does
    /// not establish that port's nominal source type.
    pub fn block(&self) -> &Block {
        &self.block
    }
}

/// Typecheck a leaf module and attach its supported declarations to the block.
///
/// `source_path` supplies the native module name and relative-import location;
/// the source itself is always the supplied text, not reread from that path.
/// Block selection follows [`block::parse`] with no explicit name. Errors
/// reject invalid IR, unsupported source types, and attachments lacking the
/// required shape or retained field-name evidence. Supply the intended source
/// text explicitly: an unrelated file at `source_path` is not checked for you.
pub fn collect_source_map(
    block_text: &str,
    source_text: &str,
    source_path: &Path,
    attachments: &[Attachment],
) -> Result<SourceMap> {
    let block = block::parse(block_text, None).context("source-map block")?;
    let declarations = collect_declarations(source_text, source_path)?;
    validate_attachments(&block, &declarations, attachments)?;
    Ok(SourceMap {
        version: 1,
        source_sha256: digest(source_text),
        block_sha256: digest(block_text),
        block_name: block.graph.name.clone(),
        declarations,
        attachments: attachments.to_vec(),
    })
}

/// Check persisted metadata against exact input bytes and return its owned
/// block.
///
/// Validation checks the version, hashes, selected block name, re-typechecked
/// declarations, and register attachments. `source_path` has the same role as
/// in [`collect_source_map`]. Stale bytes or conflicting metadata return an
/// error. Updating hashes alone after editing a map cannot validate its
/// declarations; attachments remain the caller's claim about source provenance.
pub fn validate_source_map(
    map: &SourceMap,
    block_text: &str,
    source_text: &str,
    source_path: &Path,
) -> Result<ValidatedSourceMap> {
    ensure!(
        map.version == 1,
        "unsupported source-map version {}",
        map.version
    );
    ensure!(
        map.source_sha256 == digest(source_text),
        "source SHA256 mismatch: regenerate the source map"
    );
    ensure!(
        map.block_sha256 == digest(block_text),
        "block SHA256 mismatch: regenerate the source map"
    );
    let block = block::parse(block_text, None).context("source-map block")?;
    ensure!(
        map.block_name == block.graph.name,
        "source-map block name mismatch: expected {}",
        block.graph.name
    );
    let declarations = collect_declarations(source_text, source_path)?;
    ensure!(
        map.declarations == declarations,
        "source-map declarations differ from typechecked source: regenerate the source map"
    );
    validate_attachments(&block, &declarations, &map.attachments)?;
    Ok(ValidatedSourceMap {
        map: map.clone(),
        block,
    })
}

/// Hash exact UTF-8 input bytes so even whitespace edits invalidate persisted
/// maps.
fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// Typecheck supplied text with the path's module identity and import
/// directory.
fn collect_declarations(source_text: &str, source_path: &Path) -> Result<Vec<TypeDecl>> {
    let path = source_path
        .to_str()
        .context("DSLX source path must be UTF-8")?;
    let stem = source_path.file_stem().and_then(|stem| stem.to_str());
    ensure!(
        stem.is_some_and(|stem| !stem.is_empty()),
        "DSLX source path must have a module filename"
    );
    ensure!(
        !path.contains('\0') && !source_text.contains('\0'),
        "DSLX source text and path cannot contain NUL bytes"
    );
    let mut import_data =
        dslx::ImportData::new(None, &[source_path.parent().unwrap_or(Path::new("."))]);
    let mut catalogue = Catalogue {
        declarations: Vec::new(),
    };
    convert_leaf_module(&mut import_data, source_text, source_path, &mut catalogue).with_context(
        || {
            format!(
                "typechecking source catalogue from {}",
                source_path.display()
            )
        },
    )?;
    Ok(catalogue.declarations)
}

/// Declarations collected in source order from the native DSLX type checker.
struct Catalogue {
    declarations: Vec<TypeDecl>,
}

impl Catalogue {
    /// Append a declaration while preserving source order and unique type
    /// names.
    fn add(&mut self, declaration: TypeDecl) -> Result<()> {
        ensure!(
            self.declarations
                .iter()
                .all(|existing| existing.name() != declaration.name()),
            "duplicate source declaration {}",
            declaration.name()
        );
        self.declarations.push(declaration);
        Ok(())
    }

    /// Retain local nominal references and reject shapes outside version 1.
    ///
    /// Referenced declarations must already have been collected in source
    /// order.
    fn source_type(
        &self,
        annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<SourceType> {
        let definition = annotation
            .to_type_ref_type_annotation()
            .map(|reference| reference.get_type_ref().get_type_definition());
        if let Some(external) = definition
            .as_ref()
            .and_then(dslx::TypeDefinition::to_colon_ref)
        {
            bail!(
                "imported source type {} is unsupported by source-map version 1; use a concrete leaf declaration",
                external.get_attr()
            )
        } else if let Some(alias) = definition
            .as_ref()
            .and_then(dslx::TypeDefinition::to_type_alias)
        {
            let name = alias.get_identifier();
            ensure!(
                matches!(
                    declaration(&self.declarations, &name)?,
                    TypeDecl::Alias { .. }
                ),
                "source reference {name} is not a collected alias"
            );
            Ok(SourceType::Alias { name })
        } else if ty.is_enum() {
            let name = ty
                .get_enum_def()
                .context("resolving enum declaration")?
                .get_identifier();
            ensure!(
                matches!(
                    declaration(&self.declarations, &name)?,
                    TypeDecl::Enum { .. }
                ),
                "source reference {name} is not a collected enum"
            );
            Ok(SourceType::Enum { name })
        } else if ty.is_struct() {
            let definition = ty
                .get_struct_def()
                .context("resolving struct declaration")?;
            let name = definition.get_identifier();
            ensure!(
                !definition.is_parametric(),
                "parametric source struct {name} is unsupported; use a concrete non-parametric declaration"
            );
            ensure!(
                matches!(
                    declaration(&self.declarations, &name)?,
                    TypeDecl::Struct { .. }
                ),
                "source reference {name} is not a collected struct"
            );
            Ok(SourceType::Struct { name })
        } else if let Some((signed, width)) = ty.is_bits_like() {
            ensure!(
                !signed,
                "signed source bits are unsupported by source-map version 1"
            );
            ensure!(
                width > 0,
                "zero-width source bits are unsupported by source-map version 1"
            );
            if width == 1 {
                Ok(SourceType::Bool)
            } else {
                Ok(SourceType::Bits { width })
            }
        } else {
            bail!(
                "unsupported source shape {}; source-map version 1 supports unsigned bits, two-bit enums, aliases, and ordered structs",
                ty.to_string()
                    .context("rendering unsupported source type")?
            )
        }
    }
}

impl BridgeBuilder for Catalogue {
    /// Start a leaf-module catalogue without retaining earlier declarations.
    fn start_module(&mut self, _name: &str) -> Result<(), XlsynthError> {
        self.declarations.clear();
        Ok(())
    }

    fn end_module(&mut self, _name: &str) -> Result<(), XlsynthError> {
        Ok(())
    }

    /// Collect native-evaluated enum values after checking their two-bit
    /// encoding.
    fn add_enum_def(
        &mut self,
        name: &str,
        signed: bool,
        width: usize,
        members: &[(String, XlsIrValue)],
    ) -> Result<(), XlsynthError> {
        let result = (|| -> Result<()> {
            ensure!(
                !signed && width == 2,
                "enum {name} must have unsigned two-bit encoding for source-map version 1 (signed={signed}, width={width})"
            );
            let mut encodings = BTreeSet::new();
            let members = members
                .iter()
                .map(|(member, value)| {
                    let value = value
                        .to_u64()
                        .with_context(|| format!("evaluated encoding for {name}::{member}"))?;
                    ensure!(
                        value < 4 && encodings.insert(value),
                        "enum {name} has a duplicate or out-of-range encoding at {member}: {value}"
                    );
                    Ok(EnumMember {
                        name: member.clone(),
                        value,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(!members.is_empty(), "enum {name} has no members");
            self.add(TypeDecl::Enum {
                name: name.to_owned(),
                width,
                members,
            })
        })();
        result.map_err(|error| XlsynthError(format!("{error:#}")))
    }

    /// Preserve field order and nominal types in a supported nonempty struct.
    fn add_struct_def(
        &mut self,
        name: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        let result = (|| -> Result<()> {
            ensure!(
                !members.is_empty(),
                "empty source struct {name} is unsupported"
            );
            let fields = members
                .iter()
                .map(|member| {
                    Ok(StructField {
                        name: member.name.clone(),
                        source_type: self
                            .source_type(&member.type_annotation, &member.concrete_type)
                            .with_context(|| format!("source field {name}.{}", member.name))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            self.add(TypeDecl::Struct {
                name: name.to_owned(),
                fields,
            })
        })();
        result.map_err(|error| XlsynthError(format!("{error:#}")))
    }

    /// Retain an alias name and its supported target rather than flattening it.
    fn add_alias(
        &mut self,
        name: &str,
        annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        let result = self
            .source_type(annotation, ty)
            .with_context(|| format!("source alias {name}"))
            .and_then(|target| {
                self.add(TypeDecl::Alias {
                    name: name.to_owned(),
                    target,
                })
            });
        result.map_err(|error| XlsynthError(format!("{error:#}")))
    }

    /// Ignore constants as declarations after native type and value evaluation.
    ///
    /// Constants may influence types and enum encodings, but never supply
    /// hardware equations to the emitter.
    fn add_constant(
        &mut self,
        _name: &str,
        _definition: &dslx::ConstantDef,
        _ty: &dslx::Type,
        _value: &XlsIrValue,
    ) -> Result<(), XlsynthError> {
        Ok(())
    }
}

/// Resolve a name within the collected leaf declarations, rejecting unknowns.
fn declaration<'a>(declarations: &'a [TypeDecl], name: &str) -> Result<&'a TypeDecl> {
    declarations.iter().find(|declaration| declaration.name() == name)
        .with_context(|| format!("unknown supported source type {name}; only concrete leaf declarations can be attached"))
}

/// Erase nominal names to physical bits and ordered tuples for attachment
/// checks and readable codecs.
pub(super) fn source_shape(
    source_type: &SourceType,
    declarations: &[TypeDecl],
) -> Result<ir::Type> {
    match source_type {
        SourceType::Bool => Ok(ir::Type::Bits(1)),
        SourceType::Bits { width } => Ok(ir::Type::Bits(*width)),
        SourceType::Enum { name } => match declaration(declarations, name)? {
            TypeDecl::Enum { width, .. } => Ok(ir::Type::Bits(*width)),
            _ => bail!("source type {name} is not an enum"),
        },
        SourceType::Struct { name } => match declaration(declarations, name)? {
            TypeDecl::Struct { fields, .. } => Ok(ir::Type::Tuple(
                fields
                    .iter()
                    .map(|field| source_shape(&field.source_type, declarations).map(Box::new))
                    .collect::<Result<_>>()?,
            )),
            _ => bail!("source type {name} is not a struct"),
        },
        SourceType::Alias { name } => match declaration(declarations, name)? {
            TypeDecl::Alias { target, .. } => source_shape(target, declarations),
            _ => bail!("source type {name} is not an alias"),
        },
    }
}

/// Check unique register bindings, exact physical shapes, and field-name
/// evidence.
///
/// Native and PIR parsing own physical-interface validation; this check does
/// not infer nominal types for ports or for unattached registers.
fn validate_attachments(
    block: &Block,
    declarations: &[TypeDecl],
    attachments: &[Attachment],
) -> Result<()> {
    let mut registers = BTreeSet::new();
    for attachment in attachments {
        ensure!(
            registers.insert(&attachment.register),
            "duplicate or conflicting attachment for register {}",
            attachment.register
        );
        let register = block
            .graph
            .registers
            .iter()
            .find(|register| register.name == attachment.register)
            .with_context(|| {
                format!("attachment names unknown register {}", attachment.register)
            })?;
        let source_type = match declaration(declarations, &attachment.source_type)? {
            TypeDecl::Enum { name, .. } => SourceType::Enum { name: name.clone() },
            TypeDecl::Struct { name, .. } => SourceType::Struct { name: name.clone() },
            TypeDecl::Alias { name, .. } => SourceType::Alias { name: name.clone() },
        };
        let shape = source_shape(&source_type, declarations)?;
        ensure!(
            register.ty == shape,
            "attachment shape mismatch for {}={}: register has {}, source requires {}",
            register.name,
            attachment.source_type,
            register.ty,
            shape
        );
        let reads: Vec<_> = block.graph.nodes.iter().enumerate().filter_map(|(index, node)| {
            matches!(&node.payload, NodePayload::RegisterRead { register: name } if name == &register.name)
                .then_some(NodeRef { index })
        }).collect();
        validate_projection_names(
            block,
            declarations,
            &source_type,
            &reads,
            register.name.trim_start_matches('_'),
        )
        .with_context(|| {
            format!(
                "source attachment {}={}",
                register.name, attachment.source_type
            )
        })?;
    }
    Ok(())
}

/// Require retained names to corroborate each struct field's physical tuple
/// path.
///
/// Every field needs at least one matching retained projection name, and all
/// names with the expected path prefix must agree. Anonymous extra projections
/// are permitted; equal-width fields cannot substitute for this evidence.
fn validate_projection_names(
    block: &Block,
    declarations: &[TypeDecl],
    source_type: &SourceType,
    tuples: &[NodeRef],
    source_name: &str,
) -> Result<()> {
    match source_type {
        SourceType::Alias { name } => match declaration(declarations, name)? {
            TypeDecl::Alias { target, .. } => {
                validate_projection_names(block, declarations, target, tuples, source_name)
            }
            _ => bail!("source type {name} is not an alias"),
        },
        SourceType::Struct { name } => {
            let TypeDecl::Struct { fields, .. } = declaration(declarations, name)? else {
                bail!("source type {name} is not a struct")
            };
            for (field_index, field) in fields.iter().enumerate() {
                let projections: Vec<_> = block
                    .graph
                    .nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(node_index, node)| {
                        matches!(&node.payload, NodePayload::TupleIndex { tuple, index }
                        if tuples.contains(tuple) && *index == field_index)
                        .then_some((NodeRef { index: node_index }, node))
                    })
                    .collect();
                let mut corroborated = false;
                for (_, node) in &projections {
                    // A shape match cannot distinguish equally wide fields.
                    // One retained name is required and every retained name at
                    // this physical path must agree; anonymous extra uses are
                    // OK.
                    if let Some(retained) = node.name.as_deref().and_then(|node_name| {
                        node_name
                            .trim_start_matches('_')
                            .strip_prefix(&format!("{source_name}_"))
                    }) {
                        // A numeric suffix may be part of the source spelling.
                        // Only treat it as XLS disambiguation when that leaves
                        // the complete expected field name.
                        let matches = retained == field.name
                            || retained.rsplit_once("__").is_some_and(|(base, suffix)| {
                                base == field.name
                                    && !suffix.is_empty()
                                    && suffix.bytes().all(|byte| byte.is_ascii_digit())
                            });
                        ensure!(
                            matches,
                            "retained projection name conflict at {}: index {} of {} is {}, not {}",
                            node.name.as_deref().unwrap(),
                            field_index,
                            name,
                            field.name,
                            retained
                        );
                        corroborated = true;
                    }
                }
                ensure!(
                    corroborated,
                    "missing retained projection name for {source_name}.{}: index {field_index} of {name}; raw extraction remains available",
                    field.name
                );
                let projected_nodes: Vec<_> = projections.iter().map(|(node, _)| *node).collect();
                validate_projection_names(
                    block,
                    declarations,
                    &field.source_type,
                    &projected_nodes,
                    &format!("{source_name}_{}", field.name),
                )?;
            }
            Ok(())
        }
        SourceType::Bool | SourceType::Bits { .. } | SourceType::Enum { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    const SOURCE: &str = r#"
fn next_encoding(x: u2) -> u2 { x + u2:1 }
const BASE = u2:1;
type Payload = u2;
type Data = Payload;
enum Priority: u2 { A = u2:0, B = next_encoding(u2:0), C = BASE << u2:1 }
struct State { priority: Priority, pending: bool, payload: Data }
"#;
    const BLOCK: &str = r#"package source_types
top block source_types(clk: clock, payload: bits[2]) {
  reg state((bits[2], bits[1], bits[2]))
  old: (bits[2], bits[1], bits[2]) = register_read(register=state, id=1)
  state_priority: bits[2] = tuple_index(old, index=0, id=2)
  state_pending: bits[1] = tuple_index(old, index=1, id=3)
  state_payload: bits[2] = tuple_index(old, index=2, id=4)
  write: () = register_write(old, register=state, id=5)
  payload: () = output_port(state_payload, name=payload, id=6)
}
"#;

    const MACHINE_SOURCE: &str = include_str!("../../tests/fixtures/readable_machine/machine.x");
    const MACHINE_BLOCK: &str =
        include_str!("../../tests/fixtures/readable_machine/machine.block.ir");

    fn machine_attachments() -> Vec<Attachment> {
        vec![Attachment {
            register: "machine".into(),
            source_type: "Machine".into(),
        }]
    }

    fn attachments() -> Vec<Attachment> {
        vec![Attachment {
            register: "state".into(),
            source_type: "State".into(),
        }]
    }

    fn collect(source: &str, block: &str, bindings: &[Attachment]) -> Result<SourceMap> {
        collect_source_map(block, source, Path::new("source_types.x"), bindings)
    }

    // Verifies: Arbiter catalogues retain names, widths, and serialized
    // bindings. Catches: Lost nominal identity or metadata drift at
    // supported arbiter widths.
    #[test]
    fn current_arbiter_preserves_types_at_every_required_width() {
        for width in [2, 32, 1024, 4096] {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tests/fixtures/arbiter/w{width}"));
            let source_path = root.join("arbiter.x");
            let source = fs::read_to_string(&source_path).unwrap();
            let block = fs::read_to_string(root.join("arbiter.block.ir")).unwrap();
            let attachments = vec![Attachment {
                register: "____state".into(),
                source_type: "State".into(),
            }];
            let map = collect_source_map(&block, &source, &source_path, &attachments).unwrap();
            assert_eq!(
                map.declarations
                    .iter()
                    .map(TypeDecl::name)
                    .collect::<Vec<_>>(),
                ["Payload", "Priority", "State"]
            );
            assert_eq!(
                map.declarations[0],
                TypeDecl::Alias {
                    name: "Payload".into(),
                    target: SourceType::Bits { width }
                }
            );
            assert_eq!(
                map.declarations[1],
                TypeDecl::Enum {
                    name: "Priority".into(),
                    width: 2,
                    members: vec![
                        EnumMember {
                            name: "A".into(),
                            value: 0
                        },
                        EnumMember {
                            name: "B".into(),
                            value: 1
                        },
                        EnumMember {
                            name: "C".into(),
                            value: 2
                        }
                    ],
                }
            );
            let TypeDecl::Struct { fields, .. } = &map.declarations[2] else {
                panic!("expected State struct")
            };
            assert_eq!(
                fields
                    .iter()
                    .map(|field| field.name.as_str())
                    .collect::<Vec<_>>(),
                [
                    "priority", "pendingA", "pendingB", "pendingC", "payloadA", "payloadB",
                    "payloadC"
                ]
            );
            assert_eq!(
                fields[0].source_type,
                SourceType::Enum {
                    name: "Priority".into()
                }
            );
            assert!(
                fields[1..4]
                    .iter()
                    .all(|field| field.source_type == SourceType::Bool)
            );
            assert!(fields[4..].iter().all(|field| field.source_type
                == SourceType::Alias {
                    name: "Payload".into()
                }));
            let serialized = serde_json::to_string(&map).unwrap();
            let restored: SourceMap = serde_json::from_str(&serialized).unwrap();
            assert_eq!(
                validate_source_map(&restored, &block, &source, &source_path)
                    .unwrap()
                    .map(),
                &map
            );
        }
    }

    // Verifies: Native enum constants and alias chains retain their source
    // meaning. Catches: Ordinal enum encodings or aliases flattened to
    // coincident widths.
    #[test]
    fn native_constexprs_and_alias_chains_keep_nominal_identity() {
        let map = collect(SOURCE, BLOCK, &attachments()).unwrap();
        assert_eq!(
            map.declarations[1],
            TypeDecl::Alias {
                name: "Data".into(),
                target: SourceType::Alias {
                    name: "Payload".into()
                }
            }
        );
        let TypeDecl::Enum { members, .. } = &map.declarations[2] else {
            panic!("expected Priority enum")
        };
        assert_eq!(
            members
                .iter()
                .map(|member| member.value)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        let TypeDecl::Struct { fields, .. } = &map.declarations[3] else {
            panic!("expected State struct")
        };
        assert_eq!(
            fields[0].source_type,
            SourceType::Enum {
                name: "Priority".into()
            }
        );
        assert_eq!(
            fields[2].source_type,
            SourceType::Alias {
                name: "Data".into()
            }
        );
    }

    // Negative test: Stale bytes and edited catalogues return validation
    // errors.
    #[test]
    fn consumption_rejects_stale_bytes_and_edited_catalogues() {
        let map = collect(SOURCE, BLOCK, &attachments()).unwrap();
        let validate = |map: &SourceMap, block: &str, source: &str| {
            validate_source_map(map, block, source, Path::new("source_types.x"))
        };
        assert!(
            validate(&map, BLOCK, &format!("{SOURCE}// changed\n"))
                .unwrap_err()
                .to_string()
                .contains("source SHA256 mismatch")
        );
        assert!(
            validate(&map, &format!("{BLOCK}// changed\n"), SOURCE)
                .unwrap_err()
                .to_string()
                .contains("block SHA256 mismatch")
        );
        let mut edited = map.clone();
        let TypeDecl::Enum { members, .. } = &mut edited.declarations[2] else {
            panic!("expected enum")
        };
        members[1].value = 3;
        assert!(
            validate(&edited, BLOCK, SOURCE)
                .unwrap_err()
                .to_string()
                .contains("declarations differ")
        );
        let mut edited = map.clone();
        let TypeDecl::Struct { fields, .. } = &mut edited.declarations[3] else {
            panic!("expected struct")
        };
        fields.swap(0, 2);
        assert!(
            validate(&edited, BLOCK, SOURCE)
                .unwrap_err()
                .to_string()
                .contains("declarations differ")
        );
        let mut edited = map.clone();
        edited.block_name = "another_block".into();
        assert!(
            validate(&edited, BLOCK, SOURCE)
                .unwrap_err()
                .to_string()
                .contains("block name mismatch")
        );
        let edited = SourceMap { version: 2, ..map };
        assert!(
            validate(&edited, BLOCK, SOURCE)
                .unwrap_err()
                .to_string()
                .contains("unsupported source-map version")
        );
    }

    // Negative test: Duplicate, unknown, and mismatched attachments return
    // errors.
    #[test]
    fn attachments_reject_duplicates_conflicts_unknowns_and_wrong_shapes() {
        let map = collect(SOURCE, BLOCK, &attachments()).unwrap();
        let invalid = [
            (
                vec![attachments()[0].clone(), attachments()[0].clone()],
                "duplicate or conflicting",
            ),
            (
                vec![
                    attachments()[0].clone(),
                    Attachment {
                        register: "state".into(),
                        source_type: "Payload".into(),
                    },
                ],
                "duplicate or conflicting",
            ),
            (
                vec![Attachment {
                    register: "missing".into(),
                    source_type: "State".into(),
                }],
                "unknown register",
            ),
            (
                vec![Attachment {
                    register: "state".into(),
                    source_type: "Missing".into(),
                }],
                "unknown supported source type",
            ),
            (
                vec![Attachment {
                    register: "state".into(),
                    source_type: "Priority".into(),
                }],
                "shape mismatch",
            ),
        ];
        for (bindings, diagnostic) in invalid {
            let edited = SourceMap {
                attachments: bindings.clone(),
                ..map.clone()
            };
            let error = validate_source_map(&edited, BLOCK, SOURCE, Path::new("source_types.x"))
                .unwrap_err();
            assert!(format!("{error:#}").contains(diagnostic), "{error:#}");
            assert!(collect(SOURCE, BLOCK, &bindings).is_err());
        }
    }

    // Negative test: Swapped fields or missing retained names return collection
    // errors.
    #[test]
    fn retained_projection_names_reject_same_width_field_swaps() {
        let source = SOURCE.replace(
            "priority: Priority, pending: bool, payload: Data",
            "payload: Data, pending: bool, priority: Priority",
        );
        let error = collect(&source, BLOCK, &attachments()).unwrap_err();
        assert!(format!("{error:#}").contains("retained projection name conflict"));
        let block = BLOCK
            .replace("state_priority", "anonymous_first")
            .replace("state_pending", "anonymous_second")
            .replace("state_payload", "anonymous_third");
        let error = collect(&source, &block, &attachments()).unwrap_err();
        assert!(format!("{error:#}").contains("missing retained projection name"));
    }

    // Verifies: Nested attachments retain the exact block's physical port
    // interface. Catches: Width-only binding that accepts a renamed port
    // from different IR.
    #[test]
    fn nested_non_arbiter_names_and_physical_ports_stay_bound() {
        let map = collect(MACHINE_SOURCE, MACHINE_BLOCK, &machine_attachments()).unwrap();
        let checked =
            validate_source_map(&map, MACHINE_BLOCK, MACHINE_SOURCE, Path::new("machine.x"))
                .unwrap();
        let block = checked.block();
        let command = block.graph.get_input_port("command").unwrap();
        assert_eq!(
            block.graph.port_type(command),
            &ir::Type::Tuple(vec![
                Box::new(ir::Type::Bits(1)),
                Box::new(ir::Type::Bits(1)),
            ])
        );
        let outputs: Vec<_> = block.graph.output_ports().collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(block.graph.port_name(outputs[0]), "telemetry");
        assert_eq!(block.graph.port_type(outputs[0]), &ir::Type::Bits(2));
        assert_eq!(checked.map(), &map);

        // A same-name block with a renamed physical output is not the block
        // authenticated by this persisted map, even when its widths coincide.
        let changed = MACHINE_BLOCK.replace("telemetry", "another_output");
        let error = validate_source_map(&map, &changed, MACHINE_SOURCE, Path::new("machine.x"))
            .unwrap_err();
        assert!(format!("{error:#}").contains("block SHA256 mismatch"));
    }

    // Verifies: Source numeric suffixes survive optional XLS disambiguation
    // suffixes. Catches: Stripping part of a real field name or accepting a
    // conflicting one.
    #[test]
    fn numeric_suffix_in_source_field_is_not_lost_as_disambiguation() {
        let source = SOURCE.replace("pending: bool", "pending__1: bool");
        let exact = BLOCK.replace("state_pending", "state_pending__1");
        let disambiguated = BLOCK.replace("state_pending", "state_pending__1__2");
        for block in [&exact, &disambiguated] {
            let map = collect(&source, block, &attachments()).unwrap();
            validate_source_map(&map, block, &source, Path::new("source_types.x")).unwrap();
        }
        let conflicting = BLOCK.replace("state_pending", "state_pending__2");
        let error = collect(&source, &conflicting, &attachments()).unwrap_err();
        assert!(format!("{error:#}").contains("retained projection name conflict"));
    }

    // Negative test: Nested field swaps and absent or conflicting names return
    // errors.
    #[test]
    fn nested_same_width_swaps_and_missing_names_fail_independently() {
        let swapped =
            MACHINE_SOURCE.replace("armed: bool, queued: bool", "queued: bool, armed: bool");
        let error = collect(&swapped, MACHINE_BLOCK, &machine_attachments()).unwrap_err();
        assert!(format!("{error:#}").contains("retained projection name conflict"));

        let anonymous = MACHINE_BLOCK.replace("machine_flags_armed", "anonymous_flag");
        let error = collect(MACHINE_SOURCE, &anonymous, &machine_attachments()).unwrap_err();
        assert!(
            format!("{error:#}")
                .contains("missing retained projection name for machine_flags.armed")
        );

        // Matching evidence at index zero must not hide a conflicting second
        // use of that exact physical path.
        let conflicting = MACHINE_BLOCK.replace("  write:",
            "  machine_flags_queued__1: bits[1] = tuple_index(machine_flags, index=0, id=15)\n  write:");
        let error = collect(MACHINE_SOURCE, &conflicting, &machine_attachments()).unwrap_err();
        assert!(format!("{error:#}").contains("retained projection name conflict"));
    }

    // Verifies: A declared enum encoding of three survives collection and
    // validation. Catches: Treating the fourth physical encoding as
    // necessarily unnamed.
    #[test]
    fn metadata_preserves_explicit_fourth_enum_encoding() {
        let source = MACHINE_SOURCE.replace("Active = 2", "Active = 3");
        let map = collect(&source, MACHINE_BLOCK, &machine_attachments()).unwrap();
        let TypeDecl::Enum { members, .. } = &map.declarations[1] else {
            panic!("expected Phase enum")
        };
        assert_eq!(
            members
                .iter()
                .map(|member| member.value)
                .collect::<Vec<_>>(),
            [0, 3]
        );
        validate_source_map(&map, MACHINE_BLOCK, &source, Path::new("machine.x")).unwrap();
    }

    // Negative test: Unsupported source shapes return contextual collection
    // errors.
    #[test]
    fn unsupported_source_shapes_fail_at_collection() {
        for (source, diagnostic) in [
            ("type Unsupported = u8[2];", "unsupported source shape"),
            ("type Unsupported = (u2, bool);", "unsupported source shape"),
            ("type Unsupported = s8;", "signed source bits"),
            ("type Unsupported = uN[0];", "zero-width source bits"),
            (
                "enum Unsupported: u3 { A = 0, B = 1 }",
                "unsigned two-bit encoding",
            ),
            (
                "struct Generic<N: u32> { value: uN[N] } type Unsupported = Generic<u32:2>;",
                "parametric source struct",
            ),
        ] {
            let error = collect(source, BLOCK, &[]).unwrap_err();
            assert!(format!("{error:#}").contains(diagnostic), "{error:#}");
        }
    }
}
