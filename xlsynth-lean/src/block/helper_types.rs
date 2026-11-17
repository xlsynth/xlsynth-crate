// SPDX-License-Identifier: Apache-2.0

//! Recover imported types used on the checked BX interface.

use super::source_types::{EnumMember, SourceType, StructField, TypeDecl};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use xlsynth::{
    XlsIrValue, XlsynthError, dslx,
    dslx_bridge::{BridgeBuilder, StructMemberData, convert_named_type_definition},
};
use xlsynth_bx::ImportNames;

/// Source types reachable through a checked BX interface annotation.
///
/// Looking up field and alias types before recording their owner puts
/// referenced declarations first. Unsupported declarations are remembered by
/// qualified name.
struct Catalogue<'a> {
    module_names: &'a ImportNames,
    /// Cross-module type lookups start from the module checked by the BX
    /// importer.
    entry: ModuleContext,
    declarations: Vec<TypeDecl>,
    context: ModuleContext,
    unsupported: BTreeMap<String, String>,
}

/// Keeps the declaring DSLX module, its type information, and the qualifier
/// for its names together while the collector traverses imported declarations.
#[derive(Clone)]
struct ModuleContext {
    module: dslx::Module,
    type_info: dslx::TypeInfo,
    qualifier: String,
}

impl ModuleContext {
    /// Find the type information for a declaration's module from the entry
    /// module.
    ///
    /// A type argument can come from the entry module or a transitive import.
    fn declaration_owner(&self, module: dslx::Module) -> Result<Self> {
        let qualifier = module.get_name();
        if self.module.get_name() == qualifier {
            Ok(self.clone())
        } else {
            let type_info = self
                .type_info
                .get_imported_type_info(&module)
                .with_context(|| format!("missing type information for module {qualifier}"))?;
            Ok(Self {
                module,
                type_info,
                qualifier,
            })
        }
    }
}

/// Collect an interface annotation before transparent aliases are erased.
pub(super) fn imported_struct(
    source: &xlsynth_bx::LinkedStruct,
    declarations: &mut Vec<TypeDecl>,
) -> Result<SourceType> {
    let mut catalogue = Catalogue::new(&source.module, &source.module_names);
    catalogue.declarations = std::mem::take(declarations);
    let ty = catalogue.annotation_type(&source.annotation)?;
    *declarations = catalogue.declarations;
    Ok(ty)
}

impl<'a> Catalogue<'a> {
    /// Start in the checked module that supplied the BX annotation.
    fn new(module: &dslx::TypecheckedModule, module_names: &'a ImportNames) -> Self {
        let root = ModuleContext {
            module: module.get_module(),
            qualifier: module.get_module().get_name(),
            type_info: module.get_type_info(),
        };
        Self {
            context: root.clone(),
            module_names,
            declarations: Vec::new(),
            unsupported: BTreeMap::new(),
            entry: root,
        }
    }

    /// Recover the declaring module's source name for use in Lean type names.
    fn source_qualifier(&self) -> &str {
        let native = self.context.qualifier.as_str();
        self.module_names.source_module_name(native)
    }

    /// Qualify a declaration in its own module, including transitive imports.
    fn qualified_name(&self, local: &str) -> String {
        format!("{}::{local}", self.source_qualifier())
    }

    /// Collect a referenced declaration on first use and retain its nominal
    /// identity.
    fn nominal_reference(&mut self, local: &str) -> Result<SourceType> {
        let name = self.qualified_name(local);
        if !self.declarations.iter().any(|decl| decl.name() == name)
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

    /// Report a conversion failure when a declaration is requested by name.
    fn finish_declaration(&mut self, name: &str, result: Result<()>) -> Result<(), XlsynthError> {
        if let Err(error) = result {
            self.unsupported
                .insert(self.qualified_name(name), format!("{error:#}"));
        }
        Ok(())
    }

    /// Look up a qualified declaration or report the reason its conversion
    /// failed.
    fn declaration(&self, name: &str) -> Result<&TypeDecl> {
        if let Some(reason) = self.unsupported.get(name) {
            bail!("unsupported source type {name}: {reason}")
        } else {
            self.declarations
                .iter()
                .find(|d| d.name() == name)
                .with_context(|| format!("unknown DSLX declaration {name}"))
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

    /// Recover a source type from its written annotation and checked type.
    ///
    /// Keep explicit aliases and qualified names when present. For a type
    /// parameter such as `T` instantiated with a struct, the checked type
    /// supplies the declaration that the annotation alone cannot identify.
    fn source_type(
        &mut self,
        annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<SourceType> {
        let reference = annotation.and_then(dslx::TypeAnnotation::to_type_ref_type_annotation);
        let definition = reference
            .as_ref()
            .map(|r| r.get_type_ref().get_type_definition());
        let record = ty
            .is_struct()
            .then(|| ty.get_struct_def())
            .transpose()?
            .filter(|definition| definition.is_parametric());
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
            let local = external.get_attr();
            let result = if let Some(record) = record.filter(|r| {
                r.get_identifier() == local
                    && r.get_owner_module().get_name() == self.context.qualifier
            }) {
                self.concrete_record(&record, ty)
            } else {
                self.nominal_reference(&local)
            };
            self.context = previous;
            result
        } else if let Some(alias) = definition
            .as_ref()
            .and_then(dslx::TypeDefinition::to_type_alias)
        {
            let name = alias.get_identifier();
            self.nominal_reference(&name)
        } else if let Some(elements) = annotation.and_then(dslx::TypeAnnotation::tuple_members) {
            let concrete = ty
                .get_tuple_members()
                .context("expected checked tuple type")?;
            ensure!(elements.len() == concrete.len(), "tuple arity mismatch");
            let members = elements
                .iter()
                .zip(&concrete)
                .enumerate()
                .map(|(index, (element, checked))| {
                    self.source_type(Some(element), checked)
                        .with_context(|| format!("tuple member {index}"))
                })
                .collect::<Result<_>>()?;
            Ok(SourceType::Tuple { members })
        } else if let Some(elements) = ty.get_tuple_members() {
            // A specialized type parameter has checked members but no written
            // tuple.
            let members = elements
                .iter()
                .map(|member| self.source_type(None, member))
                .collect::<Result<_>>()?;
            Ok(SourceType::Tuple { members })
        } else if ty.is_enum() || ty.is_struct() {
            self.checked_nominal_type(ty)
        } else if let Some((signed, width)) = ty.is_bits_like() {
            ensure!(
                !signed,
                "signed source bits are unsupported by readable Lean"
            );
            ensure!(
                width > 0,
                "zero-width source bits are unsupported by readable Lean"
            );
            if width == 1 {
                Ok(SourceType::Bool)
            } else {
                Ok(SourceType::Bits { width })
            }
        } else if ty.is_array() {
            let element_annotation = annotation
                .and_then(dslx::TypeAnnotation::to_array_type_annotation)
                .map(|array| array.get_element_type());
            let element =
                self.source_type(element_annotation.as_ref(), &ty.get_array_element_type())?;
            Ok(SourceType::Array {
                element: Box::new(element),
                size: ty.get_array_size(),
            })
        } else {
            bail!(
                "unsupported source shape {} in readable Lean",
                ty.to_string()
                    .context("rendering unsupported source type")?
            )
        }
    }

    /// Register a struct specialization under its evaluated arguments in
    /// declaration order.
    ///
    /// Omitted defaults and explicit equal values name the same Lean struct.
    /// Even an argument unused by the fields contributes to the name, so two
    /// DSLX types with the same layout can stay distinct in Lean.
    fn concrete_record(&mut self, record: &dslx::StructDef, ty: &dslx::Type) -> Result<SourceType> {
        let base = self.qualified_name(&record.get_identifier());
        let bindings: BTreeMap<_, _> = ty
            .get_struct_parametric_env()
            .with_context(|| format!("getting checked parametric bindings for {base}"))?
            .bindings()
            .into_iter()
            .collect();
        let arguments = record
            .parametric_bindings()
            .iter()
            .map(|binding| {
                let id = binding.get_identifier();
                let value = bindings
                    .get(&id)
                    .with_context(|| format!("missing checked binding {id} for {base}"))?;
                value
                    .convert_to_ir()?
                    .to_string_fmt_no_prefix(xlsynth::ir_value::IrFormatPreference::UnsignedDecimal)
                    .with_context(|| format!("rendering checked binding {id} for {base}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let name = format!("{base}<{}>", arguments.join(","));
        if self.declarations.iter().any(|decl| decl.name() == name) {
            return Ok(SourceType::Struct { name });
        }
        let fields = (0..record.get_member_count())
            .map(|i| {
                let member = record.get_member(i);
                Ok(StructField {
                    name: member.get_name(),
                    source_type: self
                        .source_type(Some(&member.get_type()), &ty.get_struct_member_type(i))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.add(TypeDecl::Struct {
            name: name.clone(),
            fields,
        })?;
        Ok(SourceType::Struct { name })
    }

    /// Resolve an annotation using its declaring module's type information.
    fn annotation_type(&mut self, annotation: &dslx::TypeAnnotation) -> Result<SourceType> {
        let ty = self
            .context
            .type_info
            .get_type_for_type_annotation(annotation)
            .context("source annotation has no concrete type")?;
        self.source_type(Some(annotation), &ty)
    }
}

impl BridgeBuilder for Catalogue<'_> {
    /// The annotation traversal selects its own declaration owner.
    fn start_module(&mut self, _name: &str) -> Result<(), XlsynthError> {
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
                "enum {name} must have unsigned two-bit encoding for readable Lean (signed={signed}, width={width})"
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
                            .source_type(Some(&member.type_annotation), &member.concrete_type)
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
            .source_type(Some(annotation), ty)
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
}

impl Catalogue<'_> {
    /// Resolve a declaration in its module and restore the caller's context on
    /// success or error.
    fn with_declaration_owner<T>(
        &mut self,
        module: dslx::Module,
        resolve: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if module.get_name() == self.context.module.get_name() {
            resolve(self)
        } else {
            let owner = self.entry.declaration_owner(module)?;
            let previous = std::mem::replace(&mut self.context, owner);
            let result = resolve(self);
            self.context = previous;
            result
        }
    }

    /// Recover a checked struct or enum using its declaring module's qualifier.
    ///
    /// Parametric structs also need their concrete bindings and field types.
    /// For an enum, the first member's value identifies its module, so an
    /// empty enum cannot be resolved here.
    fn checked_nominal_type(&mut self, ty: &dslx::Type) -> Result<SourceType> {
        if ty.is_struct() {
            let record = ty.get_struct_def()?;
            self.with_declaration_owner(record.get_owner_module(), |catalogue| {
                if record.is_parametric() {
                    catalogue.concrete_record(&record, ty)
                } else {
                    catalogue.nominal_reference(&record.get_identifier())
                }
            })
        } else {
            let enumeration = ty.get_enum_def()?;
            ensure!(enumeration.get_member_count() != 0, "empty enum");
            let module = enumeration.get_member(0).get_value().get_owner_module();
            self.with_declaration_owner(module, |catalogue| {
                catalogue.nominal_reference(&enumeration.get_identifier())
            })
        }
    }
}
