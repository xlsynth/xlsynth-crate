// SPDX-License-Identifier: Apache-2.0

//! Recover helper signatures from the checked DSLX compilation linked into BX.
//!
//! Lookup starts with the functions reachable from one block. Each import
//! maps their destination symbols back to its own DSLX functions. Signatures
//! then return to the destination names for checking against the block's IR.
use super::{
    BlockContext,
    source_types::{
        EnumMember, HelperSignature, SourceType, StatefulInterface, StructField, TypeDecl,
        resolve_aliases, source_shape,
    },
};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use xlsynth::{
    XlsIrValue, XlsynthError, dslx,
    dslx_bridge::{
        BridgeBuilder, FunctionParamData, StructMemberData, convert_function,
        convert_named_type_definition,
    },
};
use xlsynth_bx::ImportNames;
use xlsynth_pir::{ir, ir_parser::Parser};

/// Pairs a checked DSLX module with the destination names of its IR functions.
///
/// Externs compiled into the initial package keep their symbols. For a BX
/// import, the linker records any renames needed to avoid a collision.
pub(super) struct DslxLink {
    pub(super) module: dslx::TypecheckedModule,
    module_names: Option<ImportNames>,
    symbols: BTreeMap<String, String>,
    #[cfg(feature = "dslx-loop-bindings")]
    bindings: Vec<dslx::LoopBinding>,
}

impl DslxLink {
    /// Retain the import's checked declarations and the symbol map from the
    /// same link.
    pub(super) fn from_import(import: xlsynth_bx::LinkedImport) -> Self {
        Self {
            module: import.module,
            module_names: Some(import.module_names),
            symbols: import.symbols,
            #[cfg(feature = "dslx-loop-bindings")]
            bindings: import.loop_bindings,
        }
    }

    /// Build the initial IR package for BX externs using the checked module's
    /// own symbols.
    pub(super) fn compile(module: dslx::TypecheckedModule) -> Result<(ir::Package, Self)> {
        let text = module
            .to_ir_text()
            .context("converting checked DSLX for BX externs")?;
        Self::from_conversion(module, &text)
    }

    #[cfg(feature = "dslx-loop-bindings")]
    pub(super) fn compile_with_bindings(
        module: dslx::TypecheckedModule,
    ) -> Result<(ir::Package, Self)> {
        let (text, bindings) = module
            .convert_to_ir_text_with_loop_bindings()
            .context("collecting checked loop names for BX externs")?;
        let (package, mut link) = Self::from_conversion(module, &text)?;
        link.bindings = bindings;
        Ok((package, link))
    }

    fn from_conversion(module: dslx::TypecheckedModule, text: &str) -> Result<(ir::Package, Self)> {
        let package = Parser::new(text)
            .parse_and_validate_package()
            .context("parsing linked DSLX IR")?;
        let symbols = package
            .members
            .iter()
            .filter_map(|member| match member {
                ir::PackageMember::Function(function) => {
                    Some((function.name.clone(), function.name.clone()))
                }
                // Only IR functions can be called as BX helpers.
                _ => None,
            })
            .collect();
        Ok((
            package,
            Self {
                module,
                module_names: None,
                symbols,
                #[cfg(feature = "dslx-loop-bindings")]
                bindings: Vec::new(),
            },
        ))
    }

    /// Attach reachable checked signatures to their final linked IR symbols.
    ///
    /// `observe_owner` sees each source module contributing a type or helper so
    /// the caller can reject overlap with an independently checked graph.
    pub(super) fn add_signatures(
        &self,
        block: &BlockContext<'_>,
        declarations: &mut Vec<TypeDecl>,
        interface: &mut StatefulInterface,
        mut observe_owner: impl FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        let linked: BTreeSet<_> = block.helpers.iter().map(|f| f.name.as_str()).collect();
        let reachable = self
            .symbols
            .iter()
            .filter(|(_, destination)| linked.contains(destination.as_str()))
            .map(|(original, _)| original.as_str())
            .collect::<BTreeSet<_>>();
        if reachable.is_empty() {
            return Ok(());
        }
        let mut catalogue = Catalogue::new(&self.module, self.module_names.as_ref(), reachable);
        catalogue.collect_helpers()?;
        // These are the checked declarations behind reachable references,
        // including references brought into scope by DSLX use trees.
        for owner in catalogue
            .declarations
            .iter()
            .map(TypeDecl::name)
            .filter_map(|name| name.split_once("::").map(|(owner, _)| owner))
            .chain(
                catalogue
                    .helpers
                    .values()
                    .filter_map(|sig| sig.source_name.as_ref().map(|(owner, _)| owner.as_str())),
            )
        {
            observe_owner(owner)?;
        }
        catalogue.helpers = catalogue
            .helpers
            .into_iter()
            .map(|(original, sig)| (self.symbols[&original].clone(), sig))
            .collect();
        // From this point all call edges (including generated loop bodies)
        // come from the destination package.
        catalogue.reachable = self
            .symbols
            .values()
            .map(String::as_str)
            .filter(|destination| linked.contains(destination))
            .collect();
        catalogue.declarations = merge_declarations(
            std::mem::take(declarations),
            catalogue.declarations,
            interface,
        )?;
        catalogue.add_loop_bodies(block, interface);
        #[cfg(feature = "dslx-loop-bindings")]
        catalogue.collect_loop_helpers(
            block,
            &self.bindings,
            &interface.inputs,
            &interface.registers,
        )?;
        // Loop bindings can introduce local types absent from any signature.
        *declarations = catalogue.declarations;
        for function in &block.helpers {
            if let Some(sig) = catalogue.helpers.get(&function.name) {
                ensure!(
                    sig.params.len() == function.params.len(),
                    "arity of linked helper {}",
                    function.name
                );
                for ((name, ty), param) in sig.params.iter().zip(&function.params) {
                    ensure!(
                        source_shape(ty, declarations)? == function.get_node(*param).ty,
                        "type of linked helper {} parameter {name}",
                        function.name
                    );
                }
                ensure!(
                    source_shape(&sig.result, declarations)? == function.ret_ty,
                    "result of linked helper {}",
                    function.name
                );
            }
        }
        interface.helpers.extend(catalogue.helpers);
        Ok(())
    }
}

/// Merge the checked DSLX declarations with types already collected for the
/// block.
///
/// A reflected BX struct may match a DSLX alias to a struct. Their ordered
/// field names and types must agree after resolving aliases, even when the
/// raw IR could represent both with the same tuple. Update references inside
/// existing declarations and interfaces to use the checked alias.
fn merge_declarations(
    prior: Vec<TypeDecl>,
    mut merged: Vec<TypeDecl>,
    interface: &mut StatefulInterface,
) -> Result<Vec<TypeDecl>> {
    let mut replaced = Vec::new();
    let mut aliases = BTreeSet::new();
    for decl in prior {
        if let Some(index) = merged.iter().position(|d| d.name() == decl.name()) {
            if decl != merged[index] {
                if matches!(decl, TypeDecl::Struct { .. })
                    && matches!(merged[index], TypeDecl::Alias { .. })
                {
                    aliases.insert(decl.name().to_owned());
                }
                replaced.push((decl, index));
            }
        } else {
            merged.push(decl);
        }
    }
    // BX can refer to the imported spelling inside another record or aggregate.
    // Keep that spelling, but use its checked declaration kind throughout.
    fn restore_alias(ty: &mut SourceType, aliases: &BTreeSet<String>) {
        match ty {
            SourceType::Struct { name } if aliases.contains(name) => {
                *ty = SourceType::Alias {
                    name: std::mem::take(name),
                };
            }
            SourceType::Tuple { members } => {
                for ty in members {
                    restore_alias(ty, aliases);
                }
            }
            SourceType::Array { element, .. } => restore_alias(element, aliases),
            _ => {}
        }
    }
    if !aliases.is_empty() {
        for decl in merged
            .iter_mut()
            .chain(replaced.iter_mut().map(|(decl, _)| decl))
        {
            match decl {
                TypeDecl::Struct { fields, .. } => {
                    for field in fields {
                        restore_alias(&mut field.source_type, &aliases);
                    }
                }
                TypeDecl::Alias { target, .. } => restore_alias(target, &aliases),
                TypeDecl::Enum { .. } => {}
            }
        }
        for ty in interface
            .inputs
            .values_mut()
            .chain(interface.registers.values_mut())
            .chain(&mut interface.outputs)
        {
            restore_alias(ty, &aliases);
        }
        for sig in interface.helpers.values_mut() {
            for (_, ty) in &mut sig.params {
                restore_alias(ty, &aliases);
            }
            restore_alias(&mut sig.result, &aliases);
            for (_, ty) in sig.node_bindings.values_mut() {
                restore_alias(ty, &aliases);
            }
        }
    }
    for (old, index) in replaced {
        let new = &merged[index];
        let mut target = new;
        if let TypeDecl::Alias { target: ty, .. } = new
            && let SourceType::Struct { name } = resolve_aliases(ty, &merged)?
        {
            target = merged
                .iter()
                .find(|d| d.name() == name)
                .with_context(|| format!("unknown linked struct {name}"))?;
        }
        let (
            TypeDecl::Struct {
                fields: old_fields, ..
            },
            TypeDecl::Struct { fields, .. },
        ) = (&old, target)
        else {
            bail!("conflicting declarations for linked type {}", new.name());
        };
        let normalized = |fields: &[StructField]| -> Result<Vec<_>> {
            fields
                .iter()
                .map(|f| Ok((f.name.clone(), resolve_aliases(&f.source_type, &merged)?)))
                .collect()
        };
        ensure!(
            normalized(old_fields)? == normalized(fields)?,
            "conflicting fields in linked type {}",
            new.name()
        );
    }
    Ok(merged)
}

// Negative test: Reordered fields in an alias target must report a conflict.
#[cfg(test)]
#[test]
fn linked_struct_alias_rejects_reordered_equal_width_fields() {
    let record = |name: &str, field_names: [&str; 2]| TypeDecl::Struct {
        name: name.into(),
        fields: field_names
            .into_iter()
            .map(|name| StructField {
                name: name.into(),
                source_type: SourceType::Bits { width: 8 },
            })
            .collect(),
    };
    let reflected = vec![record("Alias", ["zebra", "apple"])];
    let checked = vec![
        record("Packet", ["apple", "zebra"]),
        TypeDecl::Alias {
            name: "Alias".into(),
            target: SourceType::Struct {
                name: "Packet".into(),
            },
        },
    ];
    let mut interface = StatefulInterface {
        bx_module: "machine".into(),
        inputs: Default::default(),
        outputs: Default::default(),
        registers: Default::default(),
        helpers: Default::default(),
    };
    let error = merge_declarations(reflected, checked, &mut interface).unwrap_err();
    assert_eq!(error.to_string(), "conflicting fields in linked type Alias");
}
/// Source types and helper signatures recovered while visiting the DSLX call
/// graph.
///
/// Looking up field and alias types before recording their owner puts
/// referenced declarations first. Unsupported declarations are remembered by
/// qualified name.
struct Catalogue<'a> {
    module_names: Option<&'a ImportNames>,
    /// The linked compilation's entry module supplies cross-module type lookups
    /// even when `context` holds an imported helper's specialization.
    entry: ModuleContext,
    declarations: Vec<TypeDecl>,
    reachable: BTreeSet<&'a str>,
    helpers: BTreeMap<String, HelperSignature>,
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
    /// A type argument can come from the entry module itself or a transitive
    /// import. The generic helper need not import the module that supplied it.
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
    let mut catalogue = Catalogue::new(&source.module, Some(&source.module_names), BTreeSet::new());
    catalogue.declarations = std::mem::take(declarations);
    let ty = catalogue.annotation_type(&source.annotation)?;
    *declarations = catalogue.declarations;
    Ok(ty)
}

impl<'a> Catalogue<'a> {
    /// Start in the checked entry module with the original IR symbols selected
    /// for this block.
    fn new(
        module: &dslx::TypecheckedModule,
        module_names: Option<&'a ImportNames>,
        reachable: BTreeSet<&'a str>,
    ) -> Self {
        let root = ModuleContext {
            module: module.get_module(),
            qualifier: module.get_module().get_name(),
            type_info: module.get_type_info(),
        };
        Self {
            context: root.clone(),
            module_names,
            reachable,
            declarations: Vec::new(),
            helpers: BTreeMap::new(),
            unsupported: BTreeMap::new(),
            entry: root,
        }
    }

    /// Recover the declaring module's source name for use in Lean type and
    /// helper names.
    fn source_qualifier(&self) -> &str {
        let native = self.context.qualifier.as_str();
        self.module_names
            .map_or(native, |names| names.source_module_name(native))
    }

    /// Recover the original IR symbol used as a key in the BX import's rename
    /// map.
    fn linked_symbol(&self, native: String) -> String {
        match self.module_names {
            Some(names) => names.restore_ir_name(&native),
            None => native,
        }
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

    /// Save a conversion failure for lookups of this name so other helpers can
    /// still use source types.
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

    /// Resolve an annotation using the type information for the current helper
    /// instance.
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
    /// Module entry needs no callback setup because the call graph walk selects
    /// the owner.
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

    /// Index a representable declaration by its original IR symbol for the
    /// linker's rename map.
    fn add_function_signature_typed(
        &mut self,
        name: &str,
        _type_info: &dslx::TypeInfo,
        params: &[FunctionParamData],
        return_annotation: Option<&dslx::TypeAnnotation>,
        _return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        if let Ok(signature) = self.helper_signature(
            name,
            params.iter().map(|p| (p.name.as_str(), &p.type_annotation)),
            return_annotation,
        ) {
            let symbol = xlsynth::mangle_dslx_name(&self.context.module.get_name(), name)?;
            let symbol = self.linked_symbol(symbol);
            self.helpers.insert(symbol, signature);
        }
        Ok(())
    }
}

impl Catalogue<'_> {
    /// Walk checked calls across modules and collect signatures for the
    /// selected original IR symbols.
    fn collect_helpers(&mut self) -> Result<()> {
        let root = self.context.clone();
        let root_name = root.module.get_name();
        let graph = root.type_info.build_function_call_graph(&root.module)?;
        // A BX extern may use any function in the linked compilation, including
        // an imported function reached by a pure entry-module wrapper.
        let mut pending: BTreeMap<_, _> = (0..root.module.get_member_count())
            .filter_map(|i| match root.module.get_member(i).to_matchable()? {
                dslx::MatchableModuleMember::Function(function) => Some((
                    (root_name.clone(), function.get_identifier()),
                    (root.clone(), function),
                )),
                // Only DSLX functions can start a walk of function calls.
                _ => None,
            })
            .collect();
        let mut graphs = BTreeMap::from([(root_name, graph)]);
        let mut visited = BTreeSet::new();
        while let Some(((module_name, function_name), (owner, caller))) = pending.pop_first() {
            if !visited.insert((module_name.clone(), function_name.clone())) {
                continue;
            }
            if caller.get_parametric_binding_count() == 0 {
                let symbol = xlsynth::mangle_dslx_name(&module_name, &function_name)?;
                let symbol = self.linked_symbol(symbol);
                if self.reachable.contains(symbol.as_str()) {
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
                // XLS includes its builtins in this graph, but their bodies
                // lower to IR operations and have no imported DSLX module.
                if name == "<builtin_stubs>" {
                    continue;
                }
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
                if callee.get_parametric_binding_count() != 0 {
                    self.collect_specializations(&owner.type_info, &callee_owner, &callee)?;
                }
                pending.insert(key, (callee_owner, callee));
            }
        }
        self.context = root;
        Ok(())
    }

    /// Recover a source signature for each reachable specialization recorded by
    /// the caller.
    ///
    /// The caller supplies the invocation's bindings and derived type
    /// information. The callee supplies the declarations those types describe.
    /// The IR symbol uses free parametric keys, while the readable name lists
    /// all bound arguments in the callee's declaration order.
    fn collect_specializations(
        &mut self,
        invoker: &dslx::TypeInfo,
        owner: &ModuleContext,
        function: &dslx::Function,
    ) -> Result<()> {
        let instances = invoker
            .get_unique_invocation_callee_data(function)
            .context("reading checked DSLX invocations")?;
        let bindings = function.parametric_bindings();
        let params: Vec<_> = (0..function.get_param_count())
            .map(|i| {
                let param = function.get_param(i);
                (param.get_name(), param.get_type_annotation())
            })
            .collect();
        let returns = function.get_return_type();
        let keys: Vec<_> = bindings.iter().map(|b| b.get_identifier()).collect();
        let free_keys: Vec<_> = bindings
            .iter()
            .zip(&keys)
            .filter(|(binding, _)| binding.get_expr().is_none())
            .map(|(_, key)| key.as_str())
            .collect();
        for index in 0..instances.len() {
            let instance = instances.get(index).context("missing checked invocation")?;
            let env = instance
                .callee_bindings()
                .context("missing callee bindings")?;
            let symbol = xlsynth::mangle_dslx_name_with_env(
                &owner.module.get_name(),
                &function.get_identifier(),
                xlsynth::DslxCallingConvention::Typical,
                &free_keys,
                Some(&env),
                None,
            )?;
            let symbol = self.linked_symbol(symbol);
            if self.helpers.contains_key(&symbol) || !self.reachable.contains(symbol.as_str()) {
                continue;
            }
            let type_info = instance
                .derived_type_info()
                .context("missing specialized callee type information")?;
            self.context = ModuleContext {
                type_info,
                ..owner.clone()
            };
            let values: BTreeMap<_, _> = env.bindings().into_iter().collect();
            let arguments = keys
                .iter()
                .map(|name| {
                    let value = values
                        .get(name)
                        .with_context(|| format!("missing binding for {name}"))?;
                    // Type arguments have DSLX spellings but no IR value.
                    Ok(match value.convert_to_ir() {
                        Ok(bits) => bits.to_string_fmt_no_prefix(
                            xlsynth::ir_value::IrFormatPreference::UnsignedDecimal,
                        )?,
                        Err(_) => value.to_text().replace(", ", ","),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let name = format!("{}<{}>", function.get_identifier(), arguments.join(","));
            // As with ordinary helpers, an unrepresentable source type can
            // still execute through the raw IR signature.
            if let Ok(signature) = self.helper_signature(
                &name,
                params.iter().map(|(name, ty)| (name.as_str(), ty)),
                returns.as_ref(),
            ) {
                self.helpers.insert(symbol, signature);
            }
        }
        Ok(())
    }
    /// Collect the declared type shared by all calls at each loop-body
    /// parameter.
    ///
    /// Only input ports, register reads and parameters of known helpers carry
    /// declarations here. A computed operand or a disagreement between callers
    /// leaves that parameter without a declared type.
    fn declared_loop_operands<'a>(
        &self,
        block: &'a BlockContext<'_>,
        inputs: &BTreeMap<String, SourceType>,
        registers: &BTreeMap<String, SourceType>,
    ) -> BTreeMap<(&'a str, usize), Option<SourceType>> {
        let mut operands = BTreeMap::new();
        let callers = std::iter::once((&block.graph.graph, None))
            .chain(block.helpers.iter().map(|&f| (&f.graph, Some(f))));
        for (graph, caller) in callers {
            for node in &graph.nodes {
                let ir::NodePayload::CountedFor {
                    body,
                    init,
                    invariant_args,
                    ..
                } = &node.payload
                else {
                    continue;
                };
                // Position zero is the iteration counter. The remaining
                // parameters receive the initial accumulator and the captures.
                for (index, arg) in std::iter::once(init).chain(invariant_args).enumerate() {
                    let ty = match &graph.get_node(*arg).payload {
                        ir::NodePayload::InputPort { name, .. } => inputs.get(name),
                        ir::NodePayload::RegisterRead { register } => registers.get(register),
                        ir::NodePayload::Param => caller.and_then(|f| {
                            let signature = self.helpers.get(&f.name)?;
                            Some(&signature.params.get(f.param_index(*arg)?)?.1)
                        }),
                        _ => None,
                    };
                    operands
                        .entry((body.as_str(), index + 1))
                        .and_modify(|prior: &mut Option<SourceType>| {
                            if prior.as_ref() != ty {
                                *prior = None;
                            }
                        })
                        .or_insert_with(|| ty.cloned());
                }
            }
        }
        operands
    }

    /// Build readable signatures for reachable loop bodies from their callers
    /// and IR.
    ///
    /// The iteration counter keeps its bit width. The accumulator sets the
    /// return type because one iteration's result feeds the next iteration.
    /// Parameters without an agreed declaration use their structural IR type.
    fn add_loop_bodies(&mut self, block: &BlockContext<'_>, interface: &StatefulInterface) {
        let operands = self.declared_loop_operands(block, &interface.inputs, &interface.registers);
        for function in &block.helpers {
            if !self.reachable.contains(function.name.as_str())
                || !operands.contains_key(&(function.name.as_str(), 1))
            {
                continue;
            }
            let params = function
                .param_nodes()
                .enumerate()
                .map(|(index, node)| {
                    let ty = if index == 0 {
                        let ir::Type::Bits(width) = node.ty else {
                            return None;
                        };
                        SourceType::Bits { width }
                    } else {
                        operands
                            .get(&(function.name.as_str(), index))
                            .and_then(Clone::clone)
                            .or_else(|| structural_type(&node.ty))?
                    };
                    Some((node.param_name().to_owned(), ty))
                })
                .collect::<Option<Vec<_>>>();
            if let Some(params) = params {
                self.helpers.insert(
                    function.name.clone(),
                    HelperSignature {
                        source_name: None,
                        result: params[1].1.clone(),
                        node_bindings: BTreeMap::new(),
                        params,
                    },
                );
            }
        }
    }

    /// Resolve a declaration in its module and restore the caller's context on
    /// success or error.
    ///
    /// Within the same module, keep the current type information because it
    /// can carry bindings for a particular helper invocation.
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

    #[cfg(feature = "dslx-loop-bindings")]
    fn loop_type(&mut self, ty: &dslx::Type) -> Result<SourceType> {
        if ty.is_enum() || ty.is_struct() {
            self.checked_nominal_type(ty)
        } else if let Some((false, width)) = ty.is_bits_like() {
            match width {
                0 => bail!("zero-width loop binding"),
                1 => Ok(SourceType::Bool),
                _ => Ok(SourceType::Bits { width }),
            }
        } else if ty.is_array() {
            Ok(SourceType::Array {
                element: Box::new(self.loop_type(&ty.get_array_element_type())?),
                size: ty.get_array_size(),
            })
        } else if let Some(tuple) = ty.get_tuple_members() {
            Ok(SourceType::Tuple {
                members: tuple
                    .iter()
                    .map(|item| self.loop_type(item))
                    .collect::<Result<_>>()?,
            })
        } else {
            bail!("unsupported checked loop type {ty}")
        }
    }

    #[cfg(feature = "dslx-loop-bindings")]
    fn collect_loop_helpers(
        &mut self,
        block: &BlockContext<'_>,
        bindings: &[dslx::LoopBinding],
        inputs: &BTreeMap<String, SourceType>,
        registers: &BTreeMap<String, SourceType>,
    ) -> Result<()> {
        if bindings.is_empty() {
            return Ok(());
        }
        let operands = self.declared_loop_operands(block, inputs, registers);
        let mut by_body = BTreeMap::<&str, Vec<_>>::new();
        for binding in bindings {
            by_body
                .entry(&binding.ir_function)
                .or_default()
                .push(binding);
        }
        for function in &block.helpers {
            let Some(bindings) = by_body.remove(function.name.as_str()) else {
                continue;
            };
            let by_id: BTreeMap<_, _> = bindings
                .iter()
                .map(|binding| Ok((usize::try_from(binding.ir_node_id)?, *binding)))
                .collect::<Result<_>>()?;
            // A checked binding may refer to the index after adding the range's
            // lower bound. Keep it on that node; the first physical
            // argument is a zero-based iteration count.
            let mut params = Vec::new();
            for (position, param) in function.params.iter().enumerate() {
                let node = function.get_node(*param);
                let source = by_id.get(&node.text_id);
                let (name, ty) = if let Some(binding) = source {
                    let Some(ty) = self
                        .loop_type(&binding.concrete_type)
                        .ok()
                        .or_else(|| operands.get(&(function.name.as_str(), position))?.clone())
                    else {
                        break;
                    };
                    (
                        binding.source_name.as_deref().unwrap_or(node.param_name()),
                        ty,
                    )
                } else if position == 0
                    && let ir::Type::Bits(width) = node.ty
                {
                    ("xls_iteration", SourceType::Bits { width })
                } else {
                    break;
                };
                if source_shape(&ty, &self.declarations)? != node.ty {
                    break;
                }
                params.push((name.to_owned(), ty));
            }
            // A compiler-created argument such as a token-wrapped carry can
            // still use the existing raw loop body when there is no
            // matching source signature.
            if params.len() == function.params.len() && params.len() >= 2 {
                let mut node_bindings = BTreeMap::new();
                for node in &function.nodes {
                    if !matches!(node.payload, ir::NodePayload::Param)
                        && let Some(binding) = by_id.get(&node.text_id)
                        && let Some(name) = &binding.source_name
                        && let Ok(ty) = self.loop_type(&binding.concrete_type)
                        && source_shape(&ty, &self.declarations)? == node.ty
                    {
                        node_bindings.insert(node.text_id, (name.clone(), ty));
                    }
                }
                self.helpers.insert(
                    function.name.clone(),
                    HelperSignature {
                        source_name: None,
                        result: params[1].1.clone(),
                        params,
                        node_bindings,
                    },
                );
            }
        }
        Ok(())
    }

    /// Resolve parameters and result in the declaring module's current type
    /// context.
    fn helper_signature<'a>(
        &mut self,
        name: &str,
        params: impl Iterator<Item = (&'a str, &'a dslx::TypeAnnotation)>,
        return_annotation: Option<&dslx::TypeAnnotation>,
    ) -> Result<HelperSignature> {
        let params = params
            .map(|(name, ty)| Ok((name.to_owned(), self.annotation_type(ty)?)))
            .collect::<Result<_>>()?;
        let result = match return_annotation {
            Some(annotation) => self.annotation_type(annotation)?,
            None => SourceType::Tuple {
                members: Vec::new(),
            },
        };
        Ok(HelperSignature {
            source_name: Some((self.source_qualifier().to_owned(), name.to_owned())),
            params,
            result,
            node_bindings: BTreeMap::new(),
        })
    }
}

/// Recover a readable type from the IR shape when no caller supplies a
/// declaration.
///
/// A single bit becomes a boolean, including inside tuples and arrays. A
/// token anywhere in the shape prevents constructing a readable value type.
fn structural_type(ty: &ir::Type) -> Option<SourceType> {
    match ty {
        ir::Type::Bits(1) => Some(SourceType::Bool),
        ir::Type::Bits(width) => Some(SourceType::Bits { width: *width }),
        ir::Type::Tuple(items) => Some(SourceType::Tuple {
            members: items
                .iter()
                .map(|item| structural_type(item))
                .collect::<Option<_>>()?,
        }),
        ir::Type::Array(array) => Some(SourceType::Array {
            element: Box::new(structural_type(&array.element_type)?),
            size: array.element_count,
        }),
        ir::Type::Token => None,
    }
}
