// SPDX-License-Identifier: Apache-2.0

//! Recover helper signatures from the checked DSLX compilation linked into BX.
use super::{
    BlockContext,
    source_types::{
        EnumMember, HelperSignature, SourceType, StatefulInterface, StructField, TypeDecl,
        source_shape,
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
use xlsynth_pir::{ir, ir_parser::Parser};

/// The owner of the checked declarations and the functions emitted from them.
/// BX adds its blocks and imports to the supplied package without renaming
/// these functions.
pub(super) struct DslxLink {
    module: dslx::TypecheckedModule,
    symbols: BTreeSet<String>,
}

impl DslxLink {
    pub(super) fn compile(module: dslx::TypecheckedModule) -> Result<(ir::Package, Self)> {
        let text = module
            .to_ir_text()
            .context("converting checked DSLX for BX externs")?;
        let package = Parser::new(&text)
            .parse_and_validate_package()
            .context("parsing linked DSLX IR")?;
        let symbols = package
            .members
            .iter()
            .filter_map(|member| match member {
                ir::PackageMember::Function(function) => Some(function.name.clone()),
                // Only IR functions can be called as BX helpers.
                _ => None,
            })
            .collect();
        Ok((package, Self { module, symbols }))
    }

    pub(super) fn add_signatures(
        &self,
        block: &BlockContext<'_>,
        declarations: &mut Vec<TypeDecl>,
        interface: &mut StatefulInterface,
    ) -> Result<()> {
        let reachable = block
            .helpers
            .iter()
            .filter(|f| self.symbols.contains(&f.name))
            .map(|f| f.name.as_str())
            .collect::<BTreeSet<_>>();
        if reachable.is_empty() {
            return Ok(());
        }
        let module = self.module.get_module();
        let root = ModuleContext {
            qualifier: module.get_name(),
            module,
            type_info: self.module.get_type_info(),
        };
        let mut catalogue = Catalogue {
            context: root.clone(),
            reachable,
            declarations: Vec::new(),
            helpers: BTreeMap::new(),
            unsupported: BTreeMap::new(),
        };
        catalogue.collect_helpers()?;
        for function in &block.helpers {
            if let Some(sig) = catalogue.helpers.get(&function.name) {
                ensure!(
                    sig.params.len() == function.params.len(),
                    "arity of linked helper {}",
                    function.name
                );
                for ((name, ty), param) in sig.params.iter().zip(&function.params) {
                    ensure!(
                        source_shape(ty, &catalogue.declarations)? == function.get_node(*param).ty,
                        "type of linked helper {} parameter {name}",
                        function.name
                    );
                }
                ensure!(
                    source_shape(&sig.result, &catalogue.declarations)? == function.ret_ty,
                    "result of linked helper {}",
                    function.name
                );
            }
        }
        declarations.extend(catalogue.declarations);
        interface.helpers.extend(catalogue.helpers);
        Ok(())
    }
}
/// Declarations collected in source order from the native DSLX type checker.
struct Catalogue<'a> {
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

impl Catalogue<'_> {
    fn qualified_name(&self, local: &str) -> String {
        format!("{}::{local}", self.context.qualifier)
    }

    // Visit imported declarations on demand. The compiler already resolved each
    // import, and unrelated declarations need not be suitable for the
    // model's interface.
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
        } else {
            bail!(
                "unsupported source shape {} in readable Lean",
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
        // Without a DSLX signature, calls to this helper use its IR types.
        if let Ok(signature) = self.helper_signature(name, params, return_annotation) {
            // Key by the IR symbol so a function of the same name in an import
            // cannot acquire a local function's source types.
            let symbol = xlsynth::mangle_dslx_name(&self.context.module.get_name(), name)?;
            self.helpers.insert(symbol, signature);
        }
        Ok(())
    }
}

impl Catalogue<'_> {
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
}
