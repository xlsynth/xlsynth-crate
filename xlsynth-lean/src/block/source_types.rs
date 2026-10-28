// SPDX-License-Identifier: Apache-2.0

//! Collect the selected function's typechecked DSLX interface for Lean.
use crate::block::{self, Block};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use xlsynth::{
    XlsIrValue, XlsynthError, dslx,
    dslx_bridge::{BridgeBuilder, FunctionParamData, StructMemberData, convert_imported_module},
};
use xlsynth_pir::ir;
/// A supported leaf declaration with nominal names retained from native DSLX.
///
/// Catalogue order and struct field order follow the source. Enum values come
/// from native constant evaluation, rather than declaration position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum TypeDecl {
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
    pub(super) fn name(&self) -> &str {
        match self {
            Self::Enum { name, .. } | Self::Struct { name, .. } | Self::Alias { name, .. } => name,
        }
    }
}

/// One named, native-evaluated enum encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EnumMember {
    /// DSLX constructor name.
    pub(super) name: String,
    /// Physical encoding, validated to be distinct and less than four.
    pub(super) value: u64,
}

/// A source field in declaration order and its nominal or primitive type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StructField {
    /// Field name from the DSLX struct declaration.
    pub(super) name: String,
    /// Field type before erasing nominal names to its physical shape.
    pub(super) source_type: SourceType,
}

/// A primitive type or nominal reference retained in the source catalogue.
///
/// Nominal references remain distinct even when their hardware widths coincide.
/// DSLX's unsigned one-bit type, including `bool`, uses the Boolean variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum SourceType {
    /// An unsigned one-bit value rendered as Lean `Bool`.
    Bool,
    /// Unsigned physical bits; collection uses this for widths greater than
    /// one.
    Bits { width: usize },
    /// Positional source tuple; its members retain their written types.
    Tuple { members: Vec<SourceType> },
    /// A reference to an enum declaration in the same catalogue.
    Enum { name: String },
    /// A reference to a struct declaration in the same catalogue.
    Struct { name: String },
    /// A reference to an alias declaration in the same catalogue.
    Alias { name: String },
}

impl SourceType {
    fn collect_names(&self, names: &mut BTreeSet<String>) {
        match self {
            Self::Bool | Self::Bits { .. } => {}
            Self::Tuple { members } => {
                for member in members {
                    member.collect_names(names);
                }
            }
            Self::Enum { name } | Self::Struct { name } | Self::Alias { name } => {
                names.insert(name.clone());
            }
        }
    }
}
/// Declarations collected in source order from the native DSLX type checker.
struct Catalogue<'a> {
    declarations: Vec<TypeDecl>,
    top: &'a str,
    type_info: dslx::TypeInfo,
    signature: Option<FunctionSignature>,
    unsupported: BTreeMap<String, String>,
}

struct FunctionSignature {
    params: Vec<(String, SourceType)>,
    returns: Vec<SourceType>,
}

impl Catalogue<'_> {
    // For a function, diagnose an unsupported declaration at the interface
    // element that actually needs it. An unrelated type is not an interface.
    fn finish_declaration(&mut self, name: &str, result: Result<()>) -> Result<(), XlsynthError> {
        if let Err(error) = result {
            self.unsupported
                .insert(name.to_owned(), format!("{error:#}"));
        }
        Ok(())
    }

    fn declaration(&self, name: &str) -> Result<&TypeDecl> {
        if let Some(reason) = self.unsupported.get(name) {
            bail!("unsupported source type {name}: {reason}")
        } else {
            declaration(&self.declarations, name)
        }
    }
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
                "imported source type {} is unsupported by stateful Lean; use a concrete leaf declaration",
                external.get_attr()
            )
        } else if let Some(alias) = definition
            .as_ref()
            .and_then(dslx::TypeDefinition::to_type_alias)
        {
            let name = alias.get_identifier();
            ensure!(
                matches!(self.declaration(&name)?, TypeDecl::Alias { .. }),
                "source reference {name} is not a collected alias"
            );
            Ok(SourceType::Alias { name })
        } else if let Some(elements) = annotation.tuple_members() {
            let members = elements
                .iter()
                .enumerate()
                .map(|(index, element)| {
                    self.annotation_type(element, &self.type_info)
                        .with_context(|| format!("tuple member {index}"))
                })
                .collect::<Result<_>>()?;
            Ok(SourceType::Tuple { members })
        } else if ty.is_enum() {
            let name = ty
                .get_enum_def()
                .context("resolving enum declaration")?
                .get_identifier();
            ensure!(
                matches!(self.declaration(&name)?, TypeDecl::Enum { .. }),
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
                matches!(self.declaration(&name)?, TypeDecl::Struct { .. }),
                "source reference {name} is not a collected struct"
            );
            Ok(SourceType::Struct { name })
        } else if let Some((signed, width)) = ty.is_bits_like() {
            ensure!(
                !signed,
                "signed source bits are unsupported by stateful Lean"
            );
            ensure!(
                width > 0,
                "zero-width source bits are unsupported by stateful Lean"
            );
            if width == 1 {
                Ok(SourceType::Bool)
            } else {
                Ok(SourceType::Bits { width })
            }
        } else {
            bail!(
                "unsupported source shape {}; stateful Lean supports unsigned bits, two-bit enums, aliases, and ordered structs",
                ty.to_string()
                    .context("rendering unsupported source type")?
            )
        }
    }

    fn annotation_type(
        &self,
        annotation: &dslx::TypeAnnotation,
        type_info: &dslx::TypeInfo,
    ) -> Result<SourceType> {
        let ty = type_info
            .get_type_for_type_annotation(annotation)
            .context("source annotation has no concrete type")?;
        self.source_type(annotation, &ty)
    }
}

impl BridgeBuilder for Catalogue<'_> {
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
                "enum {name} must have unsigned two-bit encoding for stateful Lean (signed={signed}, width={width})"
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
        self.finish_declaration(name, result)
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
        self.finish_declaration(name, result)
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
        self.finish_declaration(name, result)
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

    fn add_function_signature_typed(
        &mut self,
        name: &str,
        type_info: &dslx::TypeInfo,
        params: &[FunctionParamData],
        return_annotation: Option<&dslx::TypeAnnotation>,
        _return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        if self.top == name {
            let result = (|| -> Result<_> {
                let params = params
                    .iter()
                    .map(|param| {
                        let role = if param.name.starts_with("__reg_") {
                            "register"
                        } else {
                            "input"
                        };
                        let ty = self
                            .annotation_type(&param.type_annotation, type_info)
                            .with_context(|| format!("{role} {}", param.name))?;
                        Ok((param.name.clone(), ty))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let returns = if let Some(elements) =
                    return_annotation.and_then(dslx::TypeAnnotation::tuple_members)
                {
                    let registers: Vec<_> = params
                        .iter()
                        .filter_map(|(name, _)| name.strip_prefix("__reg_"))
                        .collect();
                    let output_count = elements
                        .len()
                        .checked_sub(registers.len())
                        .context("missing next-state return elements")?;
                    elements
                        .iter()
                        .enumerate()
                        .map(|(index, element)| {
                            self.annotation_type(element, type_info).with_context(|| {
                                if index < output_count {
                                    format!("output out{index}")
                                } else {
                                    format!(
                                        "next state of register {}",
                                        registers[index - output_count]
                                    )
                                }
                            })
                        })
                        .collect::<Result<_>>()?
                } else if let Some(annotation) = return_annotation {
                    let ty = self
                        .annotation_type(annotation, type_info)
                        .context("stateful function return type")?;
                    self.resolve_tuple(&ty)?.to_vec()
                } else {
                    // DSLX typechecks an omitted return annotation as unit.
                    Vec::new()
                };
                Ok(FunctionSignature { params, returns })
            })();
            self.signature = Some(result.map_err(|error| XlsynthError(format!("{error:#}")))?);
        }
        Ok(())
    }
}

impl Catalogue<'_> {
    fn resolve_tuple<'a>(&'a self, ty: &'a SourceType) -> Result<&'a [SourceType]> {
        match ty {
            SourceType::Tuple { members } => Ok(members),
            SourceType::Alias { name } => {
                let TypeDecl::Alias { target, .. } = declaration(&self.declarations, name)? else {
                    bail!("expected alias {name}")
                };
                self.resolve_tuple(target)
            }
            _ => bail!("stateful function requires a tuple return type, got {ty:?}"),
        }
    }
}

/// The selected function's proof-facing types, paired with validated block
/// names.
pub(super) struct StatefulInterface {
    pub(super) inputs: BTreeMap<String, SourceType>,
    pub(super) registers: BTreeMap<String, SourceType>,
    pub(super) outputs: Vec<SourceType>,
}

/// Emit one converted function's block behavior using its typechecked DSLX
/// signature.
///
/// Pass the original module from the lowering that produced `block`. Register
/// names and ordered output ports must follow the stateful function convention.
pub(super) fn emit_stateful_readable(
    block: &Block,
    module: &dslx::TypecheckedModule,
    top: &str,
) -> Result<block::LeanBlockBundle> {
    let mut catalogue = Catalogue {
        top,
        type_info: module.get_type_info(),
        declarations: Vec::new(),
        signature: None,
        unsupported: BTreeMap::new(),
    };
    convert_imported_module(module, &mut catalogue)
        .with_context(|| format!("source signature of {top}"))?;
    let FunctionSignature { params, returns } = catalogue
        .signature
        .context("selected stateful function not found")?;
    let graph = &block.graph;
    let mut inputs = BTreeMap::from([("rst".to_owned(), SourceType::Bool)]);
    let mut registers = BTreeMap::new();
    let output_count = returns
        .len()
        .checked_sub(graph.registers.len())
        .context("missing next-state return elements")?;
    let (outputs, next) = returns.split_at(output_count);
    let check = |name: &str, ty: &SourceType, physical: &ir::Type| -> Result<()> {
        let shape = source_shape(ty, &catalogue.declarations)?;
        ensure!(
            &shape == physical,
            "{name} has source type {ty:?} ({shape}) but block type {physical}"
        );
        Ok(())
    };
    for (name, ty) in params {
        if let Some(register) = name.strip_prefix("__reg_") {
            let index = registers.len();
            let storage = graph
                .registers
                .get(index)
                .with_context(|| format!("missing register {register}"))?;
            ensure!(
                storage.name == register,
                "expected register {register}, found {}",
                storage.name
            );
            check(&format!("register {register}"), &ty, &storage.ty)?;
            ensure!(
                resolve_aliases(&ty, &catalogue.declarations)?
                    == resolve_aliases(&next[index], &catalogue.declarations)?,
                "next state of register {register} has source type {:?}, expected {ty:?}",
                next[index]
            );
            registers.insert(register.to_owned(), ty);
        } else {
            ensure!(
                registers.is_empty() && !inputs.contains_key(&name),
                "invalid stateful input {name}"
            );
            inputs.insert(name, ty);
        }
    }
    ensure!(
        registers.len() == graph.registers.len(),
        "unexpected register in stateful block"
    );
    for port in graph.input_ports() {
        let name = graph.port_name(port);
        let ty = inputs
            .get(name)
            .with_context(|| format!("block input {name} missing from source"))?;
        check(&format!("input {name}"), ty, graph.port_type(port))?;
    }
    ensure!(
        inputs.len() == graph.input_ports().count(),
        "missing stateful block input"
    );
    ensure!(
        outputs.len() == graph.output_ports().count(),
        "stateful output count differs from signature"
    );
    for (index, (port, ty)) in graph.output_ports().zip(outputs).enumerate() {
        let name = graph.port_name(port);
        ensure!(
            name == format!("out{index}"),
            "unexpected stateful output port {name}"
        );
        check(&format!("output {name}"), ty, graph.port_type(port))?;
    }
    let interface = StatefulInterface {
        inputs,
        registers,
        outputs: outputs.to_vec(),
    };
    // A nominal reference can only target an earlier collected declaration.
    // Walk backwards to discover dependencies, then emit them in source order.
    let mut names = BTreeSet::new();
    for ty in interface
        .inputs
        .values()
        .chain(interface.registers.values())
        .chain(&interface.outputs)
    {
        ty.collect_names(&mut names);
    }
    for decl in catalogue.declarations.iter().rev() {
        if names.contains(decl.name()) {
            match decl {
                TypeDecl::Enum { .. } => {}
                TypeDecl::Struct { fields, .. } => {
                    for field in fields {
                        field.source_type.collect_names(&mut names);
                    }
                }
                TypeDecl::Alias { target, .. } => target.collect_names(&mut names),
            }
        }
    }
    catalogue
        .declarations
        .retain(|decl| names.contains(decl.name()));
    block::readable::emit(block, &catalogue.declarations, &interface)
}

/// Resolve a name within the collected leaf declarations, rejecting unknowns.
fn declaration<'a>(declarations: &'a [TypeDecl], name: &str) -> Result<&'a TypeDecl> {
    declarations
        .iter()
        .find(|declaration| declaration.name() == name)
        .with_context(|| format!("unknown supported source type {name}"))
}

/// DSLX aliases are transparent. Structs and enums retain their identity,
/// including when nested in positional tuples with equal physical shapes.
fn resolve_aliases(ty: &SourceType, declarations: &[TypeDecl]) -> Result<SourceType> {
    match ty {
        SourceType::Alias { name } => {
            let TypeDecl::Alias { target, .. } = declaration(declarations, name)? else {
                bail!("source type {name} is not an alias")
            };
            resolve_aliases(target, declarations)
        }
        SourceType::Tuple { members } => Ok(SourceType::Tuple {
            members: members
                .iter()
                .map(|ty| resolve_aliases(ty, declarations))
                .collect::<Result<_>>()?,
        }),
        nominal_or_primitive => Ok(nominal_or_primitive.clone()),
    }
}

/// Erase nominal types to the physical shapes used by block IR.
pub(super) fn source_shape(
    source_type: &SourceType,
    declarations: &[TypeDecl],
) -> Result<ir::Type> {
    match source_type {
        SourceType::Bool => Ok(ir::Type::Bits(1)),
        SourceType::Bits { width } => Ok(ir::Type::Bits(*width)),
        SourceType::Tuple { members } => Ok(ir::Type::Tuple(
            members
                .iter()
                .map(|ty| source_shape(ty, declarations).map(Box::new))
                .collect::<Result<_>>()?,
        )),
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
