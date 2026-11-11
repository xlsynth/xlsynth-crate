// SPDX-License-Identifier: Apache-2.0

//! Resolve DSLX imports and preserve the concrete IR functions selected by BX.

mod conversion_guard;
mod effect;
mod value_parametrics;

use crate::typed::BoundParametricValue;
pub(crate) use value_parametrics::NativeDataArg;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use xlsynth::dslx::{self, ImportData, MatchableModuleMember, TypecheckedModule};
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_parser;

use crate::ImportOptions;
use crate::ast::{StructDef, Type, UsePath};
use crate::check::{MAX_SPECIALIZATIONS, MAX_TYPE_SIZE, source_type_nodes};
use crate::import_structs::{DslxSourceLoader, ImportedStructs};
use crate::parser::{is_dslx_member_name, is_identifier_word};
use crate::typed::CheckedEnum;

/// An imported DSLX function specialized for concrete data types and value
/// bindings.
#[derive(Clone, Debug)]
pub struct ImportedCall {
    /// Full dotted DSLX module name, independent of the BX use spelling.
    pub module: String,
    /// Unqualified DSLX function name in the imported module.
    pub function: String,
    /// Concrete argument types in call order.
    pub args: Vec<Type>,
    /// Explicit compile-time arguments in declaration order.
    pub parametrics: Vec<BoundParametricValue>,
    /// Concrete result type inferred by DSLX.
    pub result: Type,
}

/// Concrete signatures and IR bodies captured from the same DSLX import graph.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResolvedImports {
    pub(crate) calls: Vec<ImportedCall>,
    pub(crate) batches: Vec<ImportBatch>,
    #[cfg(test)]
    type_witnesses: usize,
}

/// Reachable IR functions from one canonical DSLX module. Each root symbol
/// corresponds to a call ID in the same position.
#[derive(Clone, Debug)]
pub(crate) struct ImportBatch {
    pub(crate) package: Package,
    pub(crate) roots: Vec<String>,
    pub(crate) call_ids: Vec<usize>,
}

/// A use leaf names either a module or a member of its parent module.
enum UseBinding {
    Module(String),
    Member { subject: String, name: String },
}

/// Inputs selecting a native function invocation before DSLX supplies defaults.
#[derive(Eq, Hash, PartialEq)]
struct CallInputs {
    module: String,
    function: String,
    args: Vec<Type>,
    parametrics: Vec<BoundParametricValue>,
}

/// Per-check import graph and memoized concrete function signatures.
pub(crate) struct ImportResolver {
    bindings: BTreeMap<String, UseBinding>,
    subjects: BTreeSet<String>,
    stdlib_path: PathBuf,
    /// libxls v0.61 can leave ImportData invalid after a failed typecheck.
    /// Subsequent native operations report the original resolution error.
    imports: Result<Option<ImportData>, String>,
    structs: ImportedStructs,
    ids: HashMap<CallInputs, usize>,
    binding_types: BTreeMap<(String, String), Vec<Type>>,
    context_types: HashMap<value_parametrics::ContextInputs, Vec<Type>>,
    calls: Vec<ImportedCall>,
    explicit_count: usize,
    explicit_work: usize,
}

impl ImportResolver {
    /// Typecheck every import up front, including unused modules, so all calls
    /// in this BX module use a single snapshot of the DSLX sources.
    #[cfg(test)]
    pub(crate) fn new(
        declarations: &[UsePath],
        options: &ImportOptions<'_>,
    ) -> Result<Self, String> {
        let stdlib = options
            .dslx_stdlib_path
            .unwrap_or(xlsynth::default_dslx_stdlib_path());
        let native = DslxSourceLoader::new(stdlib, &options.additional_search_paths);
        Self::with_bindings(declarations, options, None, native)
    }

    /// Source paths remain intact for native typechecking. Each BX file can
    /// bind the same leaf to a different native module or member.
    pub(crate) fn with_bindings(
        declarations: &[UsePath],
        options: &ImportOptions<'_>,
        aliases: Option<&[String]>,
        native: DslxSourceLoader<'_>,
    ) -> Result<Self, String> {
        let stdlib_path = options
            .dslx_stdlib_path
            .unwrap_or(xlsynth::default_dslx_stdlib_path());
        for path in
            std::iter::once(stdlib_path).chain(options.additional_search_paths.iter().copied())
        {
            let text = path
                .to_str()
                .ok_or_else(|| format!("DSLX import path must be UTF-8: {path:?}"))?;
            if text.as_bytes().contains(&0) {
                return Err(format!("DSLX import path contains a NUL byte: {path:?}"));
            }
        }
        let mut names = BTreeSet::new();
        for (i, declaration) in declarations.iter().enumerate() {
            crate::bx_import::validate_use_path(declaration)?;
            let leaf = declaration.binding().expect("validated use");
            let binding = aliases.map_or(leaf, |a| a[i].as_str());
            if !names.insert(binding) {
                return Err(format!("duplicate use binding '{binding}'"));
            }
        }
        let (structs, imports) = ImportedStructs::from_sources(declarations, native)?;
        let mut bindings = BTreeMap::new();
        let mut subjects = BTreeSet::new();
        for (i, declaration) in declarations.iter().enumerate() {
            let (leaf, parents) = declaration.components.split_last().expect("validated use");
            let subject = declaration.components.join(".");
            let binding = if structs.has_module(&subject) {
                subjects.insert(subject.clone());
                UseBinding::Module(subject)
            } else {
                let parent = parents.join(".");
                subjects.insert(parent.clone());
                UseBinding::Member {
                    subject: parent,
                    name: leaf.clone(),
                }
            };
            bindings.insert(aliases.map_or(leaf, |a| &a[i]).clone(), binding);
        }
        Ok(Self {
            bindings,
            structs,
            subjects,
            stdlib_path: stdlib_path.to_owned(),
            imports: Ok(imports),
            ids: HashMap::new(),
            binding_types: BTreeMap::new(),
            context_types: HashMap::new(),
            calls: Vec::new(),
            explicit_count: 0,
            explicit_work: 0,
        })
    }

    pub(crate) fn has_imports(&self) -> bool {
        !self.subjects.is_empty()
    }

    /// Map a qualified module use or an unqualified member use to its DSLX
    /// module and member. Other names belong to BX declarations.
    pub(crate) fn lookup(&self, source_name: &str) -> Result<Option<(String, String)>, String> {
        if let Some((head, member)) = source_name.split_once("::") {
            return match self.bindings.get(head) {
                Some(UseBinding::Module(subject)) => Ok(Some((subject.clone(), member.to_owned()))),
                Some(UseBinding::Member { .. }) => Err(format!(
                    "use binding '{head}' is a DSLX member and cannot qualify '{source_name}'"
                )),
                None => Ok(None),
            };
        }
        match self.bindings.get(source_name) {
            Some(UseBinding::Member { subject, name }) => Ok(Some((subject.clone(), name.clone()))),
            Some(UseBinding::Module(_)) => Err(format!(
                "use binding '{source_name}' names a DSLX module and requires a member"
            )),
            None => Ok(None),
        }
    }

    /// A bare module prefix qualifies an ordinary member, not an enum variant.
    pub(crate) fn lookup_enum_prefix(
        &self,
        prefix: &str,
    ) -> Result<Option<(String, String)>, String> {
        if matches!(self.bindings.get(prefix), Some(UseBinding::Module(_))) {
            return Ok(None);
        }
        self.lookup(prefix)
    }

    /// Resolve a written enum variant during lexical checking of an
    /// unelaborated body.
    pub(crate) fn check_enum_member_name(
        &self,
        subject: &str,
        member: &str,
        variant: &str,
    ) -> Result<(), String> {
        if self.subjects.contains(subject) && self.structs.has_enum_member(subject, member, variant)
        {
            Ok(())
        } else {
            Err(format!(
                "unknown DSLX enum member '{subject}::{member}::{variant}'"
            ))
        }
    }

    /// Check visibility without typechecking an invocation in an empty fold.
    pub(crate) fn check_member_name(&self, subject: &str, member: &str) -> Result<(), String> {
        if self.structs.has_public_member(subject, member) {
            Ok(())
        } else {
            Err(format!(
                "unknown or private DSLX member '{subject}::{member}'"
            ))
        }
    }

    pub(crate) fn is_enum_type(&self, subject: &str, member: &str) -> bool {
        self.structs.is_enum_type(subject, member)
    }

    /// Enum and alias declarations hide a builtin once their use is in scope.
    pub(crate) fn is_enum_or_alias(&self, subject: &str, member: &str) -> bool {
        self.structs.is_enum_type(subject, member) || self.structs.is_public_alias(subject, member)
    }

    pub(crate) fn has_public_type(&self, subject: &str, member: &str) -> bool {
        self.structs.has_public_type(subject, member)
    }

    pub(crate) fn is_struct_type(&self, subject: &str, member: &str) -> bool {
        self.structs.is_struct_type(subject, member)
    }

    /// Check a static callback's declaration even when an empty structural
    /// loop skips invocation and signature resolution.
    pub(crate) fn check_function_name(&self, subject: &str, member: &str) -> Result<(), String> {
        if self.structs.has_public_function(subject, member) {
            Ok(())
        } else {
            Err(format!(
                "callback '{subject}::{member}' requires a public DSLX function declaration"
            ))
        }
    }

    /// Return the declared arity without instantiating an imported template.
    pub(crate) fn struct_template_arity(&self, subject: &str, member: &str) -> Option<usize> {
        if self.subjects.contains(subject) {
            self.structs.public_struct_template_arity(subject, member)
        } else {
            None
        }
    }

    /// Check the base declaration before generating a concrete DSLX type
    /// witness.
    pub(crate) fn check_struct_template_name(
        &self,
        subject: &str,
        member: &str,
    ) -> Result<(), String> {
        if self.struct_template_arity(subject, member).is_some() {
            Ok(())
        } else {
            Err(format!(
                "type application '{subject}::{member}' requires a public parametric DSLX struct declaration"
            ))
        }
    }

    /// Use the shared native context until the first failed resolution.
    fn with_imports<T>(
        &mut self,
        probe: impl FnOnce(&mut ImportedStructs, &mut ImportData) -> Result<T, String>,
    ) -> Result<T, String> {
        let imports = self
            .imports
            .as_mut()
            .map_err(|error| error.clone())?
            .as_mut()
            .expect("declared DSLX import graph");
        let result = probe(&mut self.structs, imports);
        if let Err(error) = &result {
            self.imports = Err(error.clone());
        }
        result
    }

    /// Resolve a source-level type reference using the declared import graph.
    pub(crate) fn resolve_named(&mut self, subject: &str, member: &str) -> Result<Type, String> {
        if !self.subjects.contains(subject) || !is_dslx_member_name(member) {
            return Err(format!(
                "invalid imported type reference '{subject}::{member}'"
            ));
        }
        self.with_imports(|structs, imports| {
            structs
                .resolve_exported_type(subject, member, imports)
                .map_err(|e| structs.remap_diagnostic(e))
        })
    }

    /// Resolve an explicitly applied public struct template in the same native
    /// context as concrete aliases and inferred call results.
    pub(crate) fn resolve_applied(
        &mut self,
        subject: &str,
        member: &str,
        args: &[u32],
    ) -> Result<Type, String> {
        self.check_struct_template_name(subject, member)?;
        self.with_imports(|structs, imports| {
            structs
                .resolve_applied(subject, member, args, imports)
                .map(Type::Named)
        })
    }

    /// Struct layouts discovered through named references and inferred call
    /// results.
    pub(crate) fn struct_defs(&self) -> &[StructDef] {
        self.structs.definitions()
    }

    /// Read the bindings of a concrete instance through its source use name.
    pub(crate) fn struct_parametric_values(
        &self,
        template: &str,
        canonical: &str,
    ) -> Result<Option<&[u32]>, String> {
        let Some((subject, member)) = self.lookup(template)? else {
            return Ok(None);
        };
        Ok(self
            .structs
            .struct_parametric_values(canonical, &subject, &member))
    }

    pub(crate) fn enum_definitions(&self) -> &[CheckedEnum] {
        self.structs.enum_definitions()
    }

    /// Ask the DSLX typechecker to specialize the function from concrete data
    /// arguments. The checked call supplies its inferred and default bindings.
    pub(crate) fn resolve(
        &mut self,
        module: &str,
        function: &str,
        args: Vec<Type>,
    ) -> Result<(usize, Type), String> {
        self.resolve_with_new_inputs(module, function, args, || Ok(()))
    }

    /// Reserve any caller-specific cost for a new concrete signature before
    /// asking DSLX to construct its probe. Canonical cache hits reuse that IR.
    pub(crate) fn resolve_with_new_inputs(
        &mut self,
        module: &str,
        function: &str,
        args: Vec<Type>,
        on_new_inputs: impl FnOnce() -> Result<(), String>,
    ) -> Result<(usize, Type), String> {
        self.resolve_inputs(module, function, args, Vec::new(), on_new_inputs)
    }

    /// Resolve an invocation with an explicit prefix of positional value
    /// bindings.
    pub(crate) fn resolve_explicit(
        &mut self,
        module: &str,
        function: &str,
        args: Vec<Type>,
        parametrics: Vec<BoundParametricValue>,
    ) -> Result<(usize, Type), String> {
        self.resolve_inputs(module, function, args, parametrics, || Ok(()))
    }

    fn resolve_inputs(
        &mut self,
        module: &str,
        function: &str,
        mut args: Vec<Type>,
        mut parametrics: Vec<BoundParametricValue>,
        on_new_inputs: impl FnOnce() -> Result<(), String>,
    ) -> Result<(usize, Type), String> {
        if !self.subjects.contains(module) {
            return Err(format!("undeclared DSLX import {module:?}"));
        }
        if !is_identifier_word(function) || !module.split('.').all(is_identifier_word) {
            return Err(format!(
                "cannot link '{module}::{function}': XLS IR function symbols require module and function names made of letters, digits, and underscores"
            ));
        }
        for arg in args
            .iter_mut()
            .chain(parametrics.iter_mut().map(|p| &mut p.ty))
        {
            normalize_import_type(arg, self.struct_defs(), self.enum_definitions())
                .map_err(|e| format!("{module}::{function}: {e}"))?;
        }
        let key = CallInputs {
            module: module.to_owned(),
            function: function.to_owned(),
            args,
            parametrics,
        };
        if let Some(&id) = self.ids.get(&key) {
            return Ok((id, self.calls[id].result.clone()));
        }
        on_new_inputs()?;
        if !key.parametrics.is_empty() {
            let work = key
                .args
                .iter()
                .fold(0usize, |sum, ty| sum.saturating_add(source_type_nodes(ty)));
            // The call table and cache key each retain the input types and
            // bindings.
            self.reserve_explicit_probe(self.explicit_count, &key.parametrics, work, 2)?;
            self.explicit_count += 1;
        }
        let id = self.calls.len();
        let mut probe_imports = ProbeImports::default();
        let (params, expr) = probe_call(
            &mut probe_imports,
            &self.structs,
            module,
            function,
            &key.args,
            &key.parametrics,
        )?;
        let witness = TypeWitness {
            imports: probe_imports.declarations(),
            params,
            expr,
        };
        let source = format!(
            r#"#![feature(generics)]
{}
fn capture<T: type>(_value: T) {{}}
fn probe_{id}({}) {{ capture({}) }}
"#,
            witness.imports, witness.params, witness.expr,
        );
        let result = self.with_imports(|structs, data| {
            structs
                .check_probe(&source, &format!("0bx_signature_{id}"), data)
                .and_then(|checked| {
                    let capture = ProbeCapture::new(&checked, |probe| {
                        effect::diagnose(&checked, probe, structs, module, function)
                    })?;
                    let ty = capture.param_type(0)?;
                    structs.reflect_result(module, function, &ty, data, &witness)
                })
                .map_err(|e| format!("{module}::{function}: {}", structs.remap_diagnostic(e)))
        })?;
        self.calls.push(ImportedCall {
            module: key.module.clone(),
            function: key.function.clone(),
            args: key.args.clone(),
            parametrics: key.parametrics.clone(),
            result: result.clone(),
        });
        self.ids.insert(key, id);
        Ok((id, result))
    }

    /// Reserve memory for a new explicit invocation or literal-context probe
    /// before generating native source or asking libxls to specialize the
    /// callee.
    fn reserve_explicit_probe(
        &mut self,
        count: usize,
        values: &[BoundParametricValue],
        data_work: usize,
        copies: usize,
    ) -> Result<(), String> {
        if count >= MAX_SPECIALIZATIONS {
            return Err(format!(
                "native DSLX function specialization limit of {MAX_SPECIALIZATIONS} exceeded"
            ));
        }
        let work = values
            .iter()
            .fold(data_work, |sum, v| {
                sum.saturating_add(source_type_nodes(&v.ty))
                    .saturating_add(crate::typed::constant_work(&v.value))
            })
            .saturating_mul(copies);
        let total = self.explicit_work.saturating_add(work);
        if total > MAX_TYPE_SIZE {
            return Err(format!(
                "native DSLX parametric bindings exceed the function specialization expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        self.explicit_work = total;
        Ok(())
    }

    /// Compile each imported module's requested specializations together.
    /// Separate packages keep DSLX names such as `a.b` and `a_b` distinct
    /// until BX links their generated IR symbols into the destination.
    pub(crate) fn finish(self) -> Result<ResolvedImports, String> {
        self.finish_with_calls(None)
    }

    /// Convert only the concrete DSLX calls used by retained BX definitions.
    pub(crate) fn finish_with_calls(
        self,
        used: Option<&BTreeSet<usize>>,
    ) -> Result<ResolvedImports, String> {
        let mut batches = Vec::new();
        if let Some(mut data) = self.imports? {
            let mut groups = BTreeMap::<(&str, bool), Vec<usize>>::new();
            for (id, call) in self.calls.iter().enumerate() {
                if used.is_some_and(|used| !used.contains(&id)) {
                    continue;
                }
                let concrete = call
                    .args
                    .iter()
                    .chain([&call.result])
                    .chain(call.parametrics.iter().map(|p| &p.ty))
                    .any(|ty| self.structs.contains_parametric_instance(ty));
                groups.entry((&call.module, concrete)).or_default().push(id);
            }
            for (index, ((subject, concrete), ids)) in groups.into_iter().enumerate() {
                // Keep invocation metadata for the supported concrete instances
                // independent of earlier type probes and ordinary imports.
                let mut isolated = concrete.then(|| self.structs.fresh_import_data());
                let batch = compile_batch(
                    BatchRoots {
                        subject,
                        call_ids: ids,
                        has_instances: concrete,
                    },
                    index,
                    &self.calls,
                    isolated.as_mut().unwrap_or(&mut data),
                    &self.stdlib_path,
                    &self.structs,
                )
                .map_err(|e| {
                    format!(
                        "compiling DSLX import '{subject}': {}",
                        self.structs.remap_diagnostic(e)
                    )
                })?;
                batches.push(batch);
            }
        }
        Ok(ResolvedImports {
            calls: self.calls,
            batches,
            #[cfg(test)]
            type_witnesses: self.structs.type_witnesses(),
        })
    }
}

fn normalize_import_type(
    ty: &mut Type,
    structs: &[StructDef],
    enums: &[CheckedEnum],
) -> Result<(), String> {
    match ty {
        Type::Bool => {
            *ty = Type::Bits {
                signed: false,
                width: 1,
            }
        }
        Type::Tuple(items) => items
            .iter_mut()
            .try_for_each(|item| normalize_import_type(item, structs, enums))?,
        Type::Array { element, .. } => normalize_import_type(element, structs, enums)?,
        Type::Enum {
            name,
            signed,
            width,
        } if !enums
            .iter()
            .any(|def| def.name == *name && def.signed == *signed && def.width == *width) =>
        {
            return Err(format!(
                "unsupported nominal type '{name}' in imported function signature"
            ));
        }
        Type::Named(name) if !structs.iter().any(|s| s.name == *name) => {
            return Err(format!(
                "unsupported nominal type '{name}' in imported function signature"
            ));
        }
        Type::Applied { name, .. }
        | Type::ParametricStruct { name, .. }
        | Type::TypeApplied { name, .. } => {
            return Err(format!("unresolved struct application '{name}'"));
        }
        Type::ParametricBits { .. } | Type::ParametricArray { .. } => {
            return Err(format!(
                "unresolved dimensions in imported function signature '{ty}'"
            ));
        }
        Type::Bits { .. } | Type::Named(_) | Type::Enum { .. } => {
            // Bits and resolved DSLX nominal types already have their canonical
            // type.
        }
    }
    Ok(())
}

/// Module aliases shared by every qualified type and call in a generated DSLX
/// probe.
#[derive(Default)]
struct ProbeImports {
    aliases: BTreeMap<String, usize>,
}

impl ProbeImports {
    fn qualify(&mut self, subject: &str, member: &str) -> String {
        let next = self.aliases.len();
        let index = self.aliases.entry(subject.to_owned()).or_insert(next);
        format!("bx_module_{index}::{member}")
    }

    fn declarations(&self) -> String {
        self.aliases
            .iter()
            .map(|(subject, index)| format!("import {subject} as bx_module_{index};\n"))
            .collect()
    }

    /// Render nominal arguments using aliases valid in the generated module.
    fn render_type(&mut self, ty: &Type, structs: &ImportedStructs) -> Result<String, String> {
        self.render_probe_type(ty, structs, false)
    }

    /// Private nominal spellings can reproduce an aggregate's constexpr
    /// contents or supply a typed leaf for native argument inference.
    fn render_probe_type(
        &mut self,
        ty: &Type,
        structs: &ImportedStructs,
        aggregate_value: bool,
    ) -> Result<String, String> {
        if aggregate_value
            && let Some((subject, member)) = structs
                .value_spelling(ty)
                .and_then(|name| name.split_once("::"))
        {
            return Ok(self.qualify(subject, member));
        }
        if matches!(ty, Type::Array { .. } | Type::Tuple(_))
            && let Some((subject, member)) = structs
                .aggregate_spelling(ty)
                .and_then(|p| p.split_once("::"))
        {
            return Ok(self.qualify(subject, member));
        }
        match ty {
            Type::Enum { name, .. } => {
                if !structs
                    .enum_definitions()
                    .iter()
                    .any(|def| def.name == *name && def.public)
                {
                    return Err(format!(
                        "unsupported nominal type '{name}' in imported function signature"
                    ));
                }
                let (subject, member) = name
                    .split_once("::")
                    .ok_or_else(|| format!("missing DSLX import for enum '{name}'"))?;
                Ok(self.qualify(subject, member))
            }
            Type::Named(name) => {
                let (subject, member) = structs
                    .public_spelling(name)?
                    .split_once("::")
                    .ok_or_else(|| format!("missing DSLX import for struct '{name}'"))?;
                Ok(self.qualify(subject, member))
            }
            Type::Tuple(items) => {
                let members = items
                    .iter()
                    .map(|t| self.render_probe_type(t, structs, aggregate_value))
                    .collect::<Result<Vec<_>, _>>()?;
                let trailing = if members.len() == 1 { "," } else { "" };
                Ok(format!("({}{trailing})", members.join(", ")))
            }
            Type::Array { element, size } => Ok(format!(
                "{}[{size}]",
                self.render_probe_type(element, structs, aggregate_value)?
            )),
            Type::Bits { .. } | Type::Bool => Ok(ty.to_string()),
            Type::Applied { name, .. }
            | Type::ParametricStruct { name, .. }
            | Type::TypeApplied { name, .. } => {
                Err(format!("unresolved struct application '{name}'"))
            }
            Type::ParametricBits { .. } | Type::ParametricArray { .. } => Err(format!(
                "unresolved dimensions in imported function signature '{ty}'"
            )),
        }
    }
}

/// Build parameter declarations and a call expression usable in a DSLX probe.
fn probe_call(
    imports: &mut ProbeImports,
    structs: &ImportedStructs,
    subject: &str,
    function: &str,
    args: &[Type],
    parametrics: &[BoundParametricValue],
) -> Result<(String, String), String> {
    let function = imports.specialized_name(subject, function, parametrics, structs)?;
    let mut params = Vec::with_capacity(args.len());
    let mut arguments = Vec::with_capacity(args.len());
    for (i, ty) in args.iter().enumerate() {
        let name = format!("arg_{i}");
        params.push(format!("{name}: {}", imports.render_type(ty, structs)?));
        arguments.push(name);
    }
    Ok((
        params.join(", "),
        format!("{function}({})", arguments.join(", ")),
    ))
}

/// A DSLX expression whose resolved type is being reflected into BX.
pub(crate) struct TypeWitness {
    pub(crate) imports: String,
    pub(crate) params: String,
    pub(crate) expr: String,
}

impl TypeWitness {
    pub(crate) fn project(&self, suffix: &str) -> Self {
        Self {
            imports: self.imports.clone(),
            params: self.params.clone(),
            expr: format!("({}){suffix}", self.expr),
        }
    }

    /// libxls v0.61 exposes concrete struct and array members but only
    /// *written* tuple annotations. Ask DSLX to destructure the actual
    /// tuple and capture the type of each binding, including its original
    /// nominal declaration.
    pub(crate) fn tuple_members(
        &self,
        ty: &dslx::Type,
        probe_index: usize,
        imports: &mut ImportData,
        structs: &ImportedStructs,
    ) -> Result<Vec<dslx::Type>, String> {
        // Only arity comes from the display, and DSLX checks it again when
        // destructuring. A struct's displayed name never identifies a type.
        let text = ty.to_string().map_err(|e| e.to_string())?;
        let body = text
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| format!("expected a DSLX tuple, found '{text}'"))?;
        let mut depth = 0usize;
        let mut arity = usize::from(!body.is_empty());
        for c in body.chars() {
            match c {
                '(' | '{' | '[' => depth += 1,
                ')' | '}' | ']' => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or("unbalanced DSLX tuple display")?
                }
                ',' if depth == 0 => arity += 1,
                _ => { /* Nested separators and other characters do not affect arity. */ }
            }
        }
        // Even zero-bit elements count toward BX's layout budget, as does
        // the enclosing tuple. Check before allocating the capture parameters.
        if arity >= MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX tuple exceeds maximum concrete type size {MAX_TYPE_SIZE}"
            ));
        }
        if arity == 0 || depth != 0 {
            return Err(format!("expected a nonempty DSLX tuple, found '{text}'"));
        }
        let bindings: Vec<_> = (0..arity).map(|i| format!("v{i}")).collect();
        let types: Vec<_> = (0..arity).map(|i| format!("T{i}: type")).collect();
        let params: Vec<_> = (0..arity).map(|i| format!("_v{i}: T{i}")).collect();
        let args = bindings.join(", ");
        // A runtime index permits type-only projections through empty arrays.
        let source = format!(
            r#"#![feature(generics)]
{}
fn capture<{}>({}) {{}}
fn probe(_bx_array_index: u32, {}) {{ let ({args},) = {}; capture({args}) }}
"#,
            self.imports,
            types.join(", "),
            params.join(", "),
            self.params,
            self.expr,
        );
        let checked = structs.check_probe(&source, &format!("0bx_tuple_{probe_index}"), imports)?;
        let capture = ProbeCapture::new(&checked, |_| None)?;
        (0..arity).map(|index| capture.param_type(index)).collect()
    }
}

/// The specialization of the generic capture call in a generated DSLX probe.
pub(crate) struct ProbeCapture {
    capture: dslx::Function,
    context: dslx::TypeInfo,
    invocation: dslx::InvocationCalleeData,
}

impl ProbeCapture {
    pub(crate) fn new(
        checked: &TypecheckedModule,
        effect_diagnostic: impl FnOnce(&dslx::Function) -> Option<String>,
    ) -> Result<Self, String> {
        let module = checked.get_module();
        let mut functions = (0..module.get_member_count()).filter_map(|i| {
            match module.get_member(i).to_matchable() {
                Some(MatchableModuleMember::Function(f)) => Some(f),
                _ => None,
            }
        });
        let capture = functions.next().ok_or("missing DSLX capture function")?;
        let probe = functions.next().ok_or("missing DSLX signature probe")?;
        let root = checked.get_type_info();
        if root
            .requires_implicit_token(&probe)
            .map_err(|e| e.to_string())?
        {
            return Err(effect_diagnostic(&probe).unwrap_or_else(|| {
                "function requires an implicit token and cannot be lifted into BX".into()
            }));
        }
        let invocation = root
            .get_unique_invocation_callee_data(&capture)
            .and_then(|entries| entries.get(0))
            .ok_or("missing DSLX capture invocation")?;
        let context = invocation
            .derived_type_info()
            .ok_or("missing DSLX type information for the capture invocation")?;
        Ok(Self {
            capture,
            context,
            invocation,
        })
    }

    pub(crate) fn param_type(&self, index: usize) -> Result<dslx::Type, String> {
        self.context
            .get_type_for_type_annotation(&self.capture.get_param(index).get_type_annotation())
            .ok_or_else(|| "could not resolve the captured DSLX type".to_string())
    }

    /// Concrete value bindings inferred by DSLX for the capture invocation.
    pub(crate) fn bindings(&self) -> Result<dslx::ParametricEnv, String> {
        self.invocation
            .callee_bindings()
            .ok_or_else(|| "could not resolve the captured DSLX parametric bindings".to_string())
    }
}

/// Calls checked together with one native conversion context.
struct BatchRoots<'a> {
    subject: &'a str,
    call_ids: Vec<usize>,
    has_instances: bool,
}

/// Recover each specialization's exact IR symbol from its generated probe.
fn compile_batch(
    roots: BatchRoots<'_>,
    index: usize,
    calls: &[ImportedCall],
    imports: &mut ImportData,
    stdlib_path: &Path,
    structs: &ImportedStructs,
) -> Result<ImportBatch, String> {
    let BatchRoots {
        subject,
        call_ids,
        has_instances,
    } = roots;
    let mut probe_imports = ProbeImports::default();
    let mut probes = String::new();
    for &id in &call_ids {
        let call = &calls[id];
        let (params, expr) = probe_call(
            &mut probe_imports,
            structs,
            subject,
            &call.function,
            &call.args,
            &call.parametrics,
        )?;
        probes.push_str(&format!("fn probe_{id}({params}) {{ let _ = {expr}; }}\n"));
    }
    let source = probe_imports.declarations() + &probes;
    let module = format!("0bx_batch_{index}");
    let checked = structs.check_probe(&source, &module, imports)?;
    let explicit_values = call_ids.iter().any(|&id| !calls[id].parametrics.is_empty());
    if has_instances {
        conversion_guard::check(
            &checked,
            &structs.effect_sources(),
            conversion_guard::GuardMode::ConcreteStructs,
            explicit_values,
        )?;
    }
    // XLS can embed filenames in generated symbols. Restore the logical
    // names before parsing: a temporary path can contain punctuation that
    // would otherwise turn a valid original identifier into invalid PIR.
    let type_parameters = structs.import_features(subject).type_parameters;
    // The probe imports the selected subject and the public spellings of
    // nominal arguments, including types nested in arrays and tuples.
    let enum_specializations = type_parameters
        && probe_imports
            .aliases
            .keys()
            .any(|owner| structs.import_features(owner).numeric_enums);
    let converted = structs.conversion_probe(&checked, &source, &module, enum_specializations)?;
    if !has_instances && (type_parameters || explicit_values) {
        conversion_guard::check(
            &converted.checked,
            &structs.effect_sources_for(converted.files),
            if enum_specializations {
                conversion_guard::GuardMode::NumericEnums
            } else {
                conversion_guard::GuardMode::TypeBindings
            },
            explicit_values,
        )?;
    }
    let ir = converted
        .checked
        .to_ir_text()
        .map_err(|e| converted.files.remap_diagnostic(e.to_string()))?;
    let mut ir = converted.files.restore_ir_names(&ir);
    // libxls accepts a numeric initial in a synthetic DSLX module name.
    // Give just the enclosing PIR package a legal initial; the generated
    // function symbols retain their collision-free prefix.
    let header = "package ";
    if !ir.starts_with(&format!("{header}{module}\n")) {
        return Err("unexpected DSLX probe package header".into());
    }
    ir.replace_range(header.len()..header.len() + 1, "_");
    let mut package = ir_parser::Parser::new(&ir)
        .parse_xls_package()
        .map_err(|e| format!("reading imported functions as PIR: {e}"))?;
    // Undo source edits in the conversion snapshot before restoring filenames.
    for member in &mut package.members {
        for node in &mut member.graph_mut().nodes {
            for pos in node.pos.iter_mut().flatten() {
                if let Some(path) = package.file_table.id_to_path.get(&pos.fileno) {
                    pos.colno =
                        converted
                            .files
                            .original_column(Path::new(path), pos.lineno, pos.colno);
                }
            }
        }
    }
    for path in package.file_table.id_to_path.values_mut() {
        let physical = Path::new(path);
        let logical = converted.files.logical_path(physical).unwrap_or(physical);
        if let Ok(relative) = logical.strip_prefix(stdlib_path) {
            let relative = relative
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            *path = format!("xls/dslx/stdlib/{relative}");
        } else if logical != Path::new(path) {
            *path = logical.to_string_lossy().into_owned();
        }
    }
    let functions: HashMap<_, _> = package
        .members
        .iter()
        .map(|m| (m.graph().name.as_str(), m.graph()))
        .collect();
    let mut roots = Vec::with_capacity(call_ids.len());
    for &id in &call_ids {
        let call = &calls[id];
        let name = xlsynth::mangle_dslx_name(&module, &format!("probe_{id}"))
            .map_err(|e| e.to_string())?;
        let probe = functions
            .get(name.as_str())
            .ok_or_else(|| format!("missing compiled probe for {subject}::{}", call.function))?;
        let mut invokes = probe.nodes.iter().filter_map(|n| match &n.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply),
            _ => None,
        });
        let symbol = invokes
            .next()
            .ok_or_else(|| format!("missing invoke for {subject}::{}", call.function))?;
        if invokes.next().is_some() {
            return Err(format!(
                "ambiguous compiled probe for {subject}::{}",
                call.function
            ));
        }
        roots.push(symbol.clone());
    }
    crate::import_link::retain_reachable_functions(&mut package, &roots)?;
    Ok(ImportBatch {
        package,
        roots,
        call_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::MAX_SYNTAX_DEPTH;
    use std::error::Error;
    use std::fs;
    use tempfile::tempdir;
    use xlsynth_pir::IrValue;
    use xlsynth_pir::ir_eval::eval_fn_in_package_value;

    #[test]
    fn applied_structs_share_native_layouts_with_aliases_and_inferred_results()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            r#"
pub struct Tag<N: u32 = {u32:7}, M: u32 = {N + N}> { value: u8 }
pub type Seven = Tag<7>;
pub fn make(x: u8) -> Seven { Seven { value: x } }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use wire;")?.uses;
        for inferred_first in [false, true] {
            let mut resolver = ImportResolver::new(&uses, &options)?;
            let byte = Type::Bits {
                signed: false,
                width: 8,
            };
            let expected = Type::Named("wire::Tag<u32:7, u32:14>".into());
            if inferred_first {
                assert_eq!(
                    resolver.resolve("wire", "make", vec![byte.clone()])?.1,
                    expected
                );
            }
            for args in [&[][..], &[7], &[7, 14]] {
                assert_eq!(resolver.resolve_applied("wire", "Tag", args)?, expected);
            }
            assert_eq!(resolver.resolve_named("wire", "Tag")?, expected);
            assert_eq!(resolver.resolve_named("wire", "Seven")?, expected);
            assert_eq!(resolver.resolve("wire", "make", vec![byte])?.1, expected);
            assert_eq!(resolver.struct_defs().len(), 1);
            let probes = resolver.structs.type_witnesses();
            assert_eq!(resolver.resolve_applied("wire", "Tag", &[7])?, expected);
            assert_eq!(resolver.structs.type_witnesses(), probes);
            assert_ne!(resolver.resolve_applied("wire", "Tag", &[7, 15])?, expected);
            assert_eq!(resolver.struct_defs().len(), 2);
        }
        Ok(())
    }

    #[test]
    fn invalid_applications_validate_the_template_before_probing_and_check_all_bindings()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            r#"#![feature(generics)]
pub struct Pair<N: u32, M: u32> { x: u8 }
pub type Alias = Pair<1, 2>;
pub struct Plain { x: u8 }
struct Hidden<N: u32> { x: u8 }
pub fn value() -> u8 { u8:0 }
pub struct Small<N: u8> { x: u8 }
pub struct Generic<N: u32, T: type = u8> { x: T }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use wire;")?.uses;
        let mut resolver = ImportResolver::new(&uses, &options)?;
        for member in ["Hidden", "Missing", "Plain", "value", "Alias", "Pair<1, 2>"] {
            assert_eq!(
                resolver.resolve_applied("wire", member, &[1]).unwrap_err(),
                format!(
                    "type application 'wire::{member}' requires a public parametric DSLX struct declaration"
                )
            );
        }
        assert_eq!(resolver.structs.type_witnesses(), 0);
        assert_eq!(
            resolver.resolve_applied("wire", "Pair", &[1, 2])?,
            Type::Named("wire::Pair<u32:1, u32:2>".into())
        );
        for (member, args) in [
            ("Pair", &[][..]),
            ("Pair", &[1]),
            ("Pair", &[1, 2, 3]),
            ("Small", &[1]),
            ("Generic", &[1]),
            ("Generic", &[1, 2]),
        ] {
            let mut resolver = ImportResolver::new(&uses, &options)?;
            let error = resolver.resolve_applied("wire", member, args).unwrap_err();
            assert!(error.starts_with(&format!("wire::{member}<")), "{error}");
            // Retrying against failed ImportData can abort inside libxls v0.61.
            // Every entrypoint that could issue another probe retains the
            // error.
            assert_eq!(
                resolver.resolve_applied("wire", "Pair", &[1]).unwrap_err(),
                error
            );
            assert_eq!(resolver.resolve_named("wire", "Alias").unwrap_err(), error);
            assert_eq!(
                resolver.resolve("wire", "value", vec![]).unwrap_err(),
                error
            );
            assert_eq!(resolver.finish().unwrap_err(), error);
        }
        Ok(())
    }

    #[test]
    fn import_argument_types_use_dslx_widths_and_reject_nested_nominals() {
        let module =
            crate::parse_module("extern fn f(x: (uN[0], s64, uN[65], (Packet,)[2])) -> ();")
                .unwrap();
        let mut ty = module.externs[0].params[0].ty.clone();
        assert_eq!(ty.to_string(), "(uN[0], s64, uN[65], (Packet,)[2])");
        assert_eq!(
            normalize_import_type(&mut ty, &[], &[]).unwrap_err(),
            "unsupported nominal type 'Packet' in imported function signature"
        );
    }

    #[test]
    fn invalid_use_components_from_a_programmatic_ast_are_rejected_before_loading_dslx() {
        for name in [
            "",
            ".m",
            "m..n",
            "foo::bar",
            "x\0y",
            "f(); } fn injected() {",
        ] {
            let declaration = UsePath {
                components: vec![name.to_owned()],
            };
            assert!(ImportResolver::new(&[declaration], &ImportOptions::default()).is_err());
        }
        let empty = UsePath {
            components: Vec::new(),
        };
        assert!(ImportResolver::new(&[empty], &ImportOptions::default()).is_err());
        let deep = UsePath {
            components: vec!["x".into(); MAX_SYNTAX_DEPTH + 1],
        };
        let Err(error) = ImportResolver::new(&[deep], &ImportOptions::default()) else {
            panic!("expected an excessive use path to fail");
        };
        assert_eq!(
            error,
            format!("DSLX use path exceeds {MAX_SYNTAX_DEPTH} components")
        );
    }

    #[test]
    fn a_module_shadows_a_same_named_member_even_when_the_module_is_invalid()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::create_dir(dir.path().join("fabric"))?;
        fs::write(
            dir.path().join("fabric.x"),
            "pub fn leaf() -> u8 { u8:1 } pub fn plain() -> u8 { u8:2 }",
        )?;
        let leaf = dir.path().join("fabric/leaf.x");
        fs::write(&leaf, "pub fn child() -> u8 { u8:3 }")?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let declarations = crate::parse_module("use fabric::{leaf, plain};")?.uses;
        let resolver = ImportResolver::new(&declarations, &options)?;
        assert_eq!(
            resolver.lookup("leaf::child")?,
            Some(("fabric.leaf".into(), "child".into())),
        );
        assert_eq!(
            resolver.lookup("plain")?,
            Some(("fabric".into(), "plain".into())),
        );
        assert!(resolver.lookup("leaf").is_err());
        assert!(resolver.lookup("plain::child").is_err());
        assert_eq!(resolver.lookup("local")?, None);

        fs::write(&leaf, "pub fn child() -> u8 { missing }")?;
        assert!(ImportResolver::new(&declarations, &options).is_err());
        for source in ["use missing;", "use fabric::missing;"] {
            let module = crate::parse_module(source)?;
            assert!(
                ImportResolver::new(&module.uses, &options).is_err(),
                "{source}"
            );
        }
        Ok(())
    }

    #[test]
    fn unused_member_uses_resolve_without_exposing_private_types_or_functions()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("device.x"),
            "struct Hidden { x: u8 } fn hidden(x: u8) -> u8 { x }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let declarations = "use device::{Hidden, hidden};";
        let unused =
            crate::parse_module(&format!("{declarations} block b(clk, x: u8) -> u8 {{ x }}"))?;
        crate::check_module_with_options(&unused, &options)?;
        for block in [
            "block b(clk, x: Hidden) -> u8 { x.x }",
            "block b(clk, x: u8) -> u8 { hidden(x) }",
        ] {
            let used = crate::parse_module(&format!("{declarations} {block}"))?;
            let error = crate::check_module_with_options(&used, &options).unwrap_err();
            assert!(error.contains("public"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn use_bound_calls_diagnose_unlinkable_ir_names() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        for (subject, function) in [("logic'", "identity"), ("logic", "identity'")] {
            fs::write(
                dir.path().join(format!("{subject}.x")),
                format!("pub fn {function}(x: u8) -> u8 {{ x }}"),
            )?;
            let block = crate::parse_module(&format!(
                "use {subject}::{function}; block b(clk, x: u8) -> u8 {{ {function}(x) }}"
            ))?;
            assert_eq!(
                crate::check_module_with_options(&block, &options).unwrap_err(),
                format!(
                    "block 'b': result: cannot link '{subject}::{function}': XLS IR function symbols require module and function names made of letters, digits, and underscores"
                )
            );
        }
        Ok(())
    }

    #[test]
    fn nested_struct_calls_reuse_dslx_provenance_in_either_declaration_order()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        for (module, source) in [
            ("b_function", "pub fn Header() -> u8 { u8:0 }"),
            ("c_other", "pub struct Header { tag: u8, channel: u8 }"),
            ("d_constant", "pub const Header = u8:0;"),
            ("z_protocol", "pub struct Header { tag: u8, channel: u8 }"),
            (
                "public_api",
                "import z_protocol; pub type Payload = z_protocol::Header;",
            ),
            (
                "a_factory",
                r#"
import b_function;
import c_other;
import d_constant;
import z_protocol;
pub struct Envelope { header: z_protocol::Header }
pub fn make(x: u8) -> Envelope {
    Envelope { header: z_protocol::Header { tag: x, channel: x + u8:1 } }
}
pub fn read(h: z_protocol::Header) -> u8 { h.channel }
"#,
            ),
        ] {
            fs::write(dir.path().join(format!("{module}.x")), source)?;
        }
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use public_api; use a_factory; use c_other;")?.uses;
        let byte = Type::Bits {
            signed: false,
            width: 8,
        };
        for alias_first in [true, false] {
            let mut resolver = ImportResolver::new(&uses, &options)?;
            if alias_first {
                resolver.resolve_named("public_api", "Payload")?;
            }
            let (_, envelope) = resolver.resolve("a_factory", "make", vec![byte.clone()])?;
            let Type::Named(name) = envelope else {
                panic!("expected Envelope")
            };
            let def = resolver
                .struct_defs()
                .iter()
                .find(|def| def.name == name)
                .ok_or("missing Envelope")?;
            let [field] = def.fields.as_slice() else {
                panic!("expected one Envelope field")
            };
            assert_eq!(field.name, "header");
            let header = field.ty.clone();
            // In the function-first order this call must have a public spelling
            // for the projected field before BX has seen the explicit alias.
            let (_, result) = resolver.resolve("a_factory", "read", vec![header.clone()])?;
            assert_eq!(result, byte);
            assert_eq!(header, resolver.resolve_named("public_api", "Payload")?);
            assert_ne!(header, resolver.resolve_named("c_other", "Header")?);
            let imports = resolver.finish()?;
            assert_eq!(imports.type_witnesses, 1, "alias_first={alias_first}");
            let [batch] = imports.batches.as_slice() else {
                panic!("expected one factory batch")
            };
            let make = batch
                .package
                .get_fn("__a_factory__make")
                .ok_or("missing make")?;
            let read = batch
                .package
                .get_fn("__a_factory__read")
                .ok_or("missing read")?;
            let made =
                eval_fn_in_package_value(&batch.package, make, &[IrValue::make_ubits(8, 41)?]);
            let read = eval_fn_in_package_value(&batch.package, read, &[made.get_element(0)?]);
            assert_eq!(read, IrValue::make_ubits(8, 42)?);
        }
        Ok(())
    }

    #[test]
    fn unrelated_struct_imports_work_with_a_public_declaration_named_self()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(dir.path().join("other.x"), "pub struct self { v: u8 }")?;
        fs::write(
            dir.path().join("wire.x"),
            r#"import other;
struct Hidden { v: u8 }
pub type self = Hidden;
pub type value_t = Hidden;
pub fn make(x: u8) -> Hidden { Hidden { v: x } }
pub fn read(value: Hidden) -> u8 { value.v }
"#,
        )?;
        let paths = [dir.path()];
        let mut native = ImportData::new(None, &paths);
        dslx::parse_and_typecheck(
            "import wire; fn probe(x: u8) -> u8 { let v: wire::value_t = wire::make(x); wire::read(v) }",
            "probe.x",
            "probe",
            &mut native,
        )?;
        let block = crate::parse_module(
            "use wire; block b(clk, x: u8) -> u8 { let v: wire::value_t = wire::make(x); wire::read(v) }",
        )?;
        let options = ImportOptions {
            additional_search_paths: paths.to_vec(),
            ..Default::default()
        };
        crate::check_module_with_options(&block, &options)?;
        Ok(())
    }

    #[test]
    fn public_alias_resolves_against_the_import_snapshot_after_its_private_struct_is_renamed()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let wire = dir.path().join("wire.x");
        fs::write(
            &wire,
            r#"
struct Hidden { tag: u8 }
pub type Item = Hidden;
pub struct Envelope { item: Hidden }
pub fn read(value: Hidden) -> u8 { value.tag }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use wire;")?.uses;
        let mut resolver = ImportResolver::new(&uses, &options)?;
        // Source discovery now sees Item as a spelling for a different
        // declaration. The original graph still resolves it to Hidden.
        fs::write(
            wire,
            "struct Renamed { payload: u32 } pub type Item = Renamed;",
        )?;

        let Type::Named(name) = resolver.resolve_named("wire", "Envelope")? else {
            panic!("expected Envelope");
        };
        let envelope = resolver
            .struct_defs()
            .iter()
            .find(|def| def.name == name)
            .ok_or("missing Envelope layout")?;
        let [item] = envelope.fields.as_slice() else {
            panic!("expected one Envelope field");
        };
        resolver.resolve("wire", "read", vec![item.ty.clone()])?;
        let imports = resolver.finish()?;
        let [batch] = imports.batches.as_slice() else {
            panic!("expected one wire batch");
        };
        let read = batch.package.get_fn("__wire__read").ok_or("missing read")?;
        let value = IrValue::parse_typed("(bits[8]:42)")?;
        assert_eq!(
            eval_fn_in_package_value(&batch.package, read, &[value]),
            IrValue::make_ubits(8, 42)?
        );
        Ok(())
    }

    #[test]
    fn projecting_a_private_field_does_not_probe_an_unrelated_function_use()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(dir.path().join("helper.x"), "pub fn read() -> u8 { u8:42 }")?;
        fs::write(
            dir.path().join("factory.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use helper::read;
struct Hidden { tag: u8 }
pub struct Frame { hidden: Hidden }
pub fn make() -> Frame { zero!<Frame>() }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let bx = crate::parse_module(
            "use factory; block sample(clk) -> u8 { factory::make().hidden.tag }",
        )?;
        let checked = crate::check_module_with_options(&bx, &options)?;
        assert_eq!(checked.imported_calls().len(), 1);
        Ok(())
    }

    #[test]
    fn use_leaf_keeps_its_alias_when_the_exported_type_changes() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let wire = dir.path().join("wire.x");
        fs::write(
            dir.path().join("factory.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use wire::Item;
pub struct Envelope { header: Item }
pub fn make() -> Envelope { zero!<Envelope>() }
pub fn read(header: Item) -> u8 { header.channel + u8:42 }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use factory;")?.uses;
        for replacement in [
            "pub struct Item { other: u32 }",
            "pub type Item = u32;",
            "pub enum Item : u8 { ZERO = 0 }",
            // The pinned member API cannot enumerate this sum's name. The
            // generated read parameter must recover it from the checked use.
            "pub enum Item { None, Some(u8), }",
        ] {
            fs::write(
                &wire,
                "struct Header { channel: u8 } pub type Item = Header;",
            )?;
            let mut resolver = ImportResolver::new(&uses, &options)?;
            fs::write(&wire, replacement)?;

            resolver.resolve("factory", "make", vec![])?;
            let [_, envelope] = resolver.struct_defs() else {
                panic!("expected Header and Envelope layouts");
            };
            resolver.resolve("factory", "read", vec![envelope.fields[0].ty.clone()])?;
            let imports = resolver.finish()?;
            let package = &imports.batches[0].package;
            let make = package.get_fn("__factory__make").ok_or("missing make")?;
            let read = package.get_fn("__factory__read").ok_or("missing read")?;
            let envelope = eval_fn_in_package_value(package, make, &[]);
            assert_eq!(
                eval_fn_in_package_value(package, read, &[envelope.get_element(0)?]),
                IrValue::make_ubits(8, 42)?,
                "{replacement}"
            );
        }
        Ok(())
    }

    #[test]
    fn register_alias_names_an_opaque_dslx_use_leaf_in_either_order() -> Result<(), Box<dyn Error>>
    {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            "pub struct Header { channel: u8 } pub type Item = Header;",
        )?;
        fs::write(
            dir.path().join("public_api.x"),
            "import wire; pub type Payload = wire::Header;",
        )?;
        fs::write(
            dir.path().join("factory.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use wire::Item;
pub struct Envelope { header: Item }
pub fn make() -> Envelope { zero!<Envelope>() }
pub fn read(value: Item) -> u8 { value.channel }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let reg = "reg packet: (Payload[1], u8);";
        let calls = "let frame = factory::make(); let channel = factory::read(frame.header);";
        for (first, second) in [(reg, calls), (calls, reg)] {
            let bx = format!(
                r#"use public_api::Payload;
use factory;
block sample(clk) -> u8 {{
    {first}
    {second}
    packet := ([frame.header], channel);
    packet.1
}}"#
            );
            let checked = crate::check_module_with_options(&crate::parse_module(&bx)?, &options)?;
            assert_eq!(checked.imported_calls().len(), 2);
            assert_eq!(checked.imports.type_witnesses, 1, "{bx}");
        }

        let consumer = format!("block consume(clk) -> u8 {{ {calls} channel }}");
        let storage = format!("block store(clk) -> u8 {{ {reg} packet := packet; packet.1 }}");
        for (first, second) in [(&consumer, &storage), (&storage, &consumer)] {
            let bx = format!("use public_api::Payload; use factory;\n{first}\n{second}");
            let checked = crate::check_module_with_options(&crate::parse_module(&bx)?, &options)?;
            assert_eq!(checked.imported_calls().len(), 2);
            assert_eq!(checked.imports.type_witnesses, 1, "{bx}");
        }

        for statements in [
            "let public_api = x; reg packet: public_api::Missing;",
            "reg packet: public_api::Missing; inst public_api: child;",
        ] {
            let bx = format!(
                r#"use public_api;
block child(clk) -> u8 {{ u8:0 }}
block sample(clk, x: u8) -> u8 {{ {statements} packet := x; x }}"#
            );
            let shadow = crate::parse_module(&bx)?;
            assert_eq!(
                crate::check_module_with_options(&shadow, &options).unwrap_err(),
                "block 'sample': 'public_api' names a local value here"
            );
        }

        let invalid = crate::parse_module(
            r#"use public_api;
block sample(clk) -> u8 {
    reg saved: public_api::Payload;
    missing := saved;
    reg bad: public_api::Missing;
    saved := saved;
    bad := bad;
    saved.channel
}"#,
        )?;
        assert_eq!(
            crate::check_module_with_options(&invalid, &options).unwrap_err(),
            "block 'sample': 'missing' is not a declared register"
        );
        Ok(())
    }

    #[test]
    fn checked_batch_retains_only_runtime_callees() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        // A user dependency can have the name of an ordinary generated module
        // and define a function with the name of the generated probe.
        fs::write(
            dir.path().join("bx_batch_0.x"),
            r#"#[extern_verilog("external_bound {fn} (.out({return}));")]
fn bound() -> u32 { u32:2 }
pub fn probe_0(x: u8) -> u8 {
    for (_i, acc): (u32, u8) in u32:0..bound() { acc + x }(x)
}"#,
        )?;
        fs::write(
            dir.path().join("wrapper.x"),
            "import bx_batch_0; pub fn apply(x: u8) -> u8 { bx_batch_0::probe_0(x) }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let module =
            crate::parse_module("use wrapper; block b(clk, x: u8) -> u8 { wrapper::apply(x) }")?;
        let checked = crate::check_module_with_options(&module, &options)?;
        let batch = &checked.imports.batches[0];
        assert_eq!(batch.roots, ["__wrapper__apply"]);
        // DSLX folds bound() into the trip count. The snapshot keeps apply,
        // the user's probe_0 and its loop body.
        assert!(batch.package.get_fn("__bx_batch_0__probe_0").is_some());
        assert!(batch.package.get_fn("__bx_batch_0__bound").is_none());
        assert_eq!(batch.package.members.len(), 3);
        xlsynth_pir::ir_verify::verify_package(&batch.package)?;
        Ok(())
    }
}
