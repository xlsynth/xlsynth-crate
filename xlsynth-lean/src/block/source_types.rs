// SPDX-License-Identifier: Apache-2.0

//! Collect the selected function's typechecked DSLX interface for Lean.
use crate::block::{self, BlockContext};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use xlsynth::{
    XlsIrValue, XlsynthError, dslx,
    dslx_bridge::{
        BridgeBuilder, FunctionParamData, StructMemberData, convert_function,
        convert_imported_module, convert_named_type_definition,
    },
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
    helpers: BTreeMap<String, HelperSignature>,
    context: ModuleContext,
    signature: Option<FunctionSignature>,
    unsupported: BTreeMap<String, String>,
}

#[derive(Clone)]
struct ModuleContext {
    module: dslx::Module,
    type_info: dslx::TypeInfo,
    qualifier: String,
}

struct FunctionSignature {
    params: Vec<(String, SourceType)>,
    returns: Vec<SourceType>,
}

impl Catalogue<'_> {
    fn qualified_name(&self, local: &str) -> String {
        if self.context.qualifier.is_empty() {
            local.to_owned()
        } else {
            format!("{}::{local}", self.context.qualifier)
        }
    }

    // Visit imported declarations on demand. The compiler already resolved each
    // import, and unrelated declarations need not be suitable for the
    // model's interface.
    fn nominal_reference(&mut self, local: &str) -> Result<SourceType> {
        let name = self.qualified_name(local);
        if !self.context.qualifier.is_empty()
            && !self.declarations.iter().any(|decl| decl.name() == name)
            && !self.unsupported.contains_key(&name)
        {
            convert_named_type_definition(
                &self.context.module.clone(),
                &self.context.type_info.clone(),
                local,
                self,
            )?;
        }
        match self.declaration(&name)? {
            TypeDecl::Enum { .. } => Ok(SourceType::Enum { name }),
            TypeDecl::Struct { .. } => Ok(SourceType::Struct { name }),
            TypeDecl::Alias { .. } => Ok(SourceType::Alias { name }),
        }
    }

    // For a function, diagnose an unsupported declaration at the interface
    // element that actually needs it. An unrelated type is not an interface.
    fn finish_declaration(&mut self, name: &str, result: Result<()>) -> Result<(), XlsynthError> {
        if let Err(error) = result {
            self.unsupported
                .insert(self.qualified_name(name), format!("{error:#}"));
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

    /// Resolve each reference in the declaring module's type environment.
    fn source_type(
        &mut self,
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
            let import = external
                .resolve_import_subject()
                .context("resolving source type import")?;
            let module = self
                .context
                .type_info
                .get_imported_module(&import)
                .context("missing imported module")?;
            let type_info = self
                .context
                .type_info
                .get_imported_type_info(&module)
                .context("missing imported type information")?;
            let qualifier = module.get_name();
            let previous = std::mem::replace(
                &mut self.context,
                ModuleContext {
                    module,
                    type_info,
                    qualifier,
                },
            );
            let result = self.nominal_reference(&external.get_attr());
            self.context = previous;
            result
        } else if let Some(alias) = definition
            .as_ref()
            .and_then(dslx::TypeDefinition::to_type_alias)
        {
            let name = alias.get_identifier();
            self.nominal_reference(&name)
        } else if let Some(elements) = annotation.tuple_members() {
            let members = elements
                .iter()
                .enumerate()
                .map(|(index, element)| {
                    self.annotation_type(element)
                        .with_context(|| format!("tuple member {index}"))
                })
                .collect::<Result<_>>()?;
            Ok(SourceType::Tuple { members })
        } else if ty.is_enum() {
            let name = ty
                .get_enum_def()
                .context("resolving enum declaration")?
                .get_identifier();
            self.nominal_reference(&name)
        } else if ty.is_struct() {
            let definition = ty
                .get_struct_def()
                .context("resolving struct declaration")?;
            let name = definition.get_identifier();
            ensure!(
                !definition.is_parametric(),
                "parametric source struct {name} is unsupported; use a concrete non-parametric declaration"
            );
            self.nominal_reference(&name)
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
                "unsupported source shape {} in stateful Lean",
                ty.to_string()
                    .context("rendering unsupported source type")?
            )
        }
    }

    fn annotation_type(&mut self, annotation: &dslx::TypeAnnotation) -> Result<SourceType> {
        let ty = self
            .context
            .type_info
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
                name: self.qualified_name(name),
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
                name: self.qualified_name(name),
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
                    name: self.qualified_name(name),
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
        _type_info: &dslx::TypeInfo,
        params: &[FunctionParamData],
        return_annotation: Option<&dslx::TypeAnnotation>,
        _return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        if self.context.qualifier.is_empty() && self.top == name {
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
                            .annotation_type(&param.type_annotation)
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
                            self.annotation_type(element).with_context(|| {
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
                        .annotation_type(annotation)
                        .context("stateful function return type")?;
                    self.resolve_tuple(&ty)?.to_vec()
                } else {
                    // DSLX typechecks an omitted return annotation as unit.
                    Vec::new()
                };
                Ok(FunctionSignature { params, returns })
            })();
            self.signature = Some(result.map_err(|error| XlsynthError(format!("{error:#}")))?);
        } else if let Ok(signature) = self.helper_signature(name, params, return_annotation) {
            // Key by the IR symbol so a function of the same name in an import
            // cannot acquire a local function's source types.
            let symbol = xlsynth::mangle_dslx_name(&self.context.module.get_name(), name)?;
            self.helpers.insert(symbol, signature);
        }
        Ok(())
    }
}

impl Catalogue<'_> {
    fn collect_imported_helpers(&mut self, block: &BlockContext<'_>) -> Result<()> {
        if block.helpers.is_empty() {
            return Ok(());
        }
        let root = self.context.clone();
        let root_name = root.module.get_name();
        let graph = root.type_info.build_function_call_graph(&root.module)?;
        let top = (0..graph.function_count())
            .filter_map(|i| graph.get_function(i))
            .find(|f| f.get_identifier() == self.top)
            .context("selected DSLX function missing from call graph")?;
        // The native graph has no callee order. Visit declarations in a stable
        // order so repeated translations produce the same Lean file.
        let mut pending = BTreeMap::from([(
            (root_name.clone(), top.get_identifier()),
            (root.clone(), top),
        )]);
        let mut graphs = BTreeMap::from([(root_name, graph)]);
        let mut visited = BTreeSet::new();
        while let Some(((module_name, function_name), (owner, caller))) = pending.pop_first() {
            if !visited.insert((module_name.clone(), function_name.clone())) {
                continue;
            }
            if !owner.qualifier.is_empty() && caller.get_parametric_binding_count() == 0 {
                let symbol = xlsynth::mangle_dslx_name(&module_name, &function_name)?;
                if block.helpers.iter().any(|f| f.name == symbol) {
                    self.context = owner.clone();
                    convert_function(&caller, &owner.type_info, self)?;
                }
            }
            let graph = match graphs.entry(module_name.clone()) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(owner.type_info.build_function_call_graph(&owner.module)?)
                }
            };
            for index in 0..graph.callee_count(&caller) {
                let callee = graph
                    .get_callee(&caller, index)
                    .context("missing DSLX callee")?;
                let module = callee.get_body().get_owner_module();
                let name = module.get_name();
                let key = (name.clone(), callee.get_identifier());
                let callee_owner = if name == module_name {
                    owner.clone()
                } else {
                    let type_info = owner
                        .type_info
                        .get_imported_type_info(&module)
                        .with_context(|| {
                            format!("missing type information for imported module {name}")
                        })?;
                    ModuleContext {
                        module,
                        type_info,
                        qualifier: name,
                    }
                };
                pending.insert(key, (callee_owner, callee));
            }
        }
        self.context = root;
        Ok(())
    }
    fn helper_signature(
        &mut self,
        name: &str,
        params: &[FunctionParamData],
        return_annotation: Option<&dslx::TypeAnnotation>,
    ) -> Result<HelperSignature> {
        let params = params
            .iter()
            .map(|p| Ok((p.name.clone(), self.annotation_type(&p.type_annotation)?)))
            .collect::<Result<_>>()?;
        let result = match return_annotation {
            Some(annotation) => self.annotation_type(annotation)?,
            None => SourceType::Tuple {
                members: Vec::new(),
            },
        };
        Ok(HelperSignature {
            name: self.qualified_name(name),
            params,
            result,
        })
    }

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
    pub(super) helpers: BTreeMap<String, HelperSignature>,
    pub(super) inputs: BTreeMap<String, SourceType>,
    pub(super) registers: BTreeMap<String, SourceType>,
    pub(super) outputs: Vec<SourceType>,
}

pub(super) struct HelperSignature {
    pub(super) name: String,
    pub(super) params: Vec<(String, SourceType)>,
    pub(super) result: SourceType,
}

/// Emit one converted function's block behavior using its typechecked DSLX
/// signature.
///
/// Pass the original module from the lowering that produced `block`. Register
/// names and ordered output ports must follow the stateful function convention.
pub(super) fn emit_stateful_readable(
    block: &BlockContext<'_>,
    module: &dslx::TypecheckedModule,
    top: &str,
) -> Result<block::LeanBlockBundle> {
    let mut catalogue = Catalogue {
        top,
        context: ModuleContext {
            module: module.get_module(),
            type_info: module.get_type_info(),
            qualifier: String::new(),
        },
        helpers: BTreeMap::new(),
        declarations: Vec::new(),
        signature: None,
        unsupported: BTreeMap::new(),
    };
    convert_imported_module(module, &mut catalogue)
        .with_context(|| format!("source signature of {top}"))?;
    catalogue.collect_imported_helpers(block)?;
    let FunctionSignature { params, returns } = catalogue
        .signature
        .context("selected stateful function not found")?;
    let graph = block.graph;
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
    catalogue
        .helpers
        .retain(|symbol, _| block.helpers.iter().any(|helper| &helper.name == symbol));
    let interface = StatefulInterface {
        helpers: catalogue.helpers,
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
        .chain(
            interface
                .helpers
                .values()
                .flat_map(|f| f.params.iter().map(|(_, ty)| ty).chain([&f.result])),
        )
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
