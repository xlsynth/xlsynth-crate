// SPDX-License-Identifier: Apache-2.0
//! Typed DSLX metadata construction for the native PIR/Cranelift AOT compiler.
//!
//! This module typechecks DSLX modules and preserves their nominal types in
//! metadata consumed by `xlsynth-pir-compiler`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use xlsynth::dslx_bridge::{BridgeBuilder, convert_imported_module};
use xlsynth::{
    DslxCallingConvention, DslxConvertOptions, XlsynthError, dslx, dslx_path_to_module_name,
    mangle_dslx_name_with_calling_convention,
};

use super::dslx_type_metadata::{
    ConcreteDslxTypeShape, DslxTypeMetadata, RustTypeTarget, parse_concrete_dslx_type_shape,
    rust_module_path_from_dslx_module_name, rust_type_path_between_dslx_modules,
};
use super::{
    PIR_AOT_METADATA_FORMAT_VERSION, PirAotDecl, PirAotEntrypoint, PirAotEntrypointSource,
    PirAotEnumVariant, PirAotField, PirAotModule, PirAotPackageMetadata, PirAotParam,
    PirAotSignedness, PirAotType,
};

type MetadataResult<T> = Result<T, XlsynthError>;

fn pir_aot_signedness(is_signed: bool) -> PirAotSignedness {
    if is_signed {
        PirAotSignedness::Signed
    } else {
        PirAotSignedness::Unsigned
    }
}

/// Inputs required to compile one DSLX function into a typed DSLX AOT wrapper.
///
/// The generated module contains Rust type definitions for `type_module_paths`
/// and `dslx_path`, plus a `Runner` whose public signature uses canonical paths
/// to those generated Rust types.
pub struct TypedDslxAotBuildSpec<'a> {
    /// Base name used for emitted artifact filenames and generated symbol
    /// wrappers.
    pub name: &'a str,
    /// DSLX source file that contains the top function.
    pub dslx_path: &'a Path,
    /// Name of the DSLX function to compile as the AOT entrypoint.
    pub top: &'a str,
    /// DSLX conversion options used for typechecking, dependency discovery, and
    /// IR lowering.
    pub dslx_options: DslxConvertOptions<'a>,
    /// DSLX modules whose public types should be emitted beside the top module
    /// wrapper.
    pub type_module_paths: Vec<&'a Path>,
}

/// Returns the PIR wrapper top used for an implicit-token DSLX AOT entrypoint.
///
/// `base_name` is the sanitized AOT entrypoint name. The wrapper has the
/// user-visible DSLX signature; it creates the implicit token/activation values
/// and invokes the converted `__itok__...` function internally.
pub fn typed_dslx_implicit_token_entrypoint_wrapper_top(base_name: &str) -> String {
    format!("__xlsynth_pir_aot_{base_name}_dslx_entry")
}

/// Collects several typed DSLX AOT entrypoints into one generated Rust package.
fn sanitize_identifier(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (index, ch) in name.chars().enumerate() {
        let valid = ch == '_' || ch.is_ascii_alphanumeric();
        let ch = if valid { ch } else { '_' };
        if index == 0 && ch.is_ascii_digit() {
            out.push('_');
        }
        out.push(ch);
    }
    if out.is_empty() {
        "aot_entrypoint".to_string()
    } else {
        out
    }
}

/// A DSLX struct field paired with the lowered semantic type of that field.
///
/// Field order is the original DSLX declaration order. The AOT ABI for structs
/// is structural, so the order here is the order used when flattening fields to
/// leaf buffers and when reconstructing the generated Rust struct.
#[derive(Debug, Clone)]
struct TypedDslxField {
    /// Rust field identifier generated for the DSLX field.
    name: String,
    /// Lowered semantic type for this field.
    ty: TypedDslxType,
}

/// A DSLX semantic type that can be mapped to the current AOT ABI model.
///
/// Each variant carries the generated Rust type spelling used in public
/// signatures plus the structural facts needed to validate and traverse the
/// AOT layout. Types that cannot flatten to bits, enum-underlying bits,
/// structs, or fixed arrays are rejected during lowering.
#[derive(Debug, Clone)]
enum TypedDslxType {
    /// A DSLX bits-like type represented by `UBits<N>` or `SBits<N>`.
    Bits {
        /// Generated Rust type path used in signatures and helpers.
        rust_type: String,
        /// Whether generated conversion should use signed scalar semantics.
        is_signed: bool,
        /// Number of payload bits in the AOT ABI leaf.
        bit_count: usize,
    },
    /// A DSLX enum represented by a generated Rust enum and underlying bits.
    Enum {
        /// Generated Rust enum path used in metadata type references.
        rust_type: String,
    },
    /// A DSLX struct represented by a generated Rust struct.
    Struct {
        /// Generated Rust struct path used in signatures and literals.
        rust_type: String,
        /// Struct fields in DSLX declaration order.
        fields: Vec<TypedDslxField>,
    },
    /// An anonymous DSLX tuple represented structurally.
    Tuple {
        /// Tuple elements in positional order.
        elements: Vec<TypedDslxType>,
    },
    /// A fixed-size DSLX array represented by a Rust array.
    Array {
        /// Number of array elements.
        size: usize,
        /// Lowered semantic type for each element.
        element: Box<TypedDslxType>,
    },
}

/// A typed DSLX function parameter ready for metadata emission.
#[derive(Debug, Clone)]
struct TypedDslxParam {
    /// DSLX parameter name, sanitized later for generated Rust arguments.
    name: String,
    /// Lowered semantic type for this parameter.
    ty: TypedDslxType,
}

/// One concrete parametric struct definition that typed AOT must materialize.
///
/// Generated Rust type emission normally materializes concrete parametric
/// structs when the defining module itself references them. Typed AOT needs a
/// package-level view so direct imported instantiations can still be emitted in
/// the defining module even when only another module mentions them.
struct TypedConcreteParametricStruct {
    /// Exact DSLX struct declaration being specialized.
    ///
    /// Definition identity matters because sibling imports may both declare a
    /// parametric struct with the same DSLX identifier.
    struct_def: dslx::StructDef,
    /// Canonical DSLX module name that owns the original struct declaration.
    defining_module_name: String,
    /// Concrete Rust struct identifier to emit inside the defining module.
    ///
    /// The suffix is derived from the fully bound parametric values, so it is
    /// also the concrete-value portion of this specialization's key.
    rust_name: String,
    /// Concrete fields rendered relative to the defining module.
    fields: Vec<TypedDslxField>,
}

/// The typed DSLX view of one AOT entrypoint signature.
///
/// The recursive shapes are converted to the metadata model consumed by the
/// native compiler.
#[derive(Debug, Clone)]
struct TypedAotFunctionSignature {
    /// Parameters in DSLX function order.
    params: Vec<TypedDslxParam>,
    /// Lowered semantic type for the return value.
    return_type: TypedDslxType,
}

/// A DSLX struct definition handle kept for exact owner-module resolution.
///
/// Duplicate struct names can appear in sibling imported modules. Keeping the
/// definition object allows lookup by definition identity before falling back
/// to bare names.
struct TypedDslxStructDefContext {
    def: dslx::StructDef,
}

/// A DSLX type alias definition handle kept for recursive alias expansion.
struct TypedDslxAliasDefContext {
    name: String,
    def: dslx::TypeAlias,
}

/// Type and name information collected for one typechecked DSLX module.
///
/// The type context is the authoritative place to resolve struct members and
/// enum underlying types for definitions owned by this module. The name sets
/// are only fallback indexes for definitions where the DSLX bindings do not
/// expose enough direct owner information.
struct TypedDslxModuleContext {
    /// Canonical DSLX module name used to derive nested Rust module paths.
    dslx_name: String,
    /// Type information produced by DSLX typechecking for this module.
    type_info: dslx::TypeInfo,
    /// Struct names declared by this module.
    struct_names: BTreeSet<String>,
    /// Struct definitions declared by this module.
    struct_defs: Vec<TypedDslxStructDefContext>,
    /// Type aliases declared by this module.
    type_alias_defs: Vec<TypedDslxAliasDefContext>,
    /// Enum names declared by this module.
    enum_names: BTreeSet<String>,
}

/// Cross-module lookup state used while lowering typed DSLX AOT signatures.
///
/// The context contains the requested type modules and the top module in one
/// search space so nested fields and imported annotations can resolve to the
/// correct `TypeInfo` and generated Rust module path.
struct TypedDslxTypeContext {
    modules: Vec<TypedDslxModuleContext>,
}

struct TypedDslxPackageModule {
    canonical_path: PathBuf,
    typechecked: dslx::TypecheckedModule,
}

struct TypedDslxPackageTypecheckedModules {
    modules: Vec<TypedDslxPackageModule>,
}

/// A type annotation paired with the module context that owns it.
struct ResolvedDslxTypeAnnotation<'a> {
    type_info: &'a dslx::TypeInfo,
    annotation: dslx::TypeAnnotation,
}

/// A non-empty DSLX import subject such as `foo` or `foo.bar`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DslxImportSubject {
    segments: Vec<String>,
}

impl DslxImportSubject {
    /// Parses the subject token from a DSLX `import` statement.
    ///
    /// The token may be dotted, as in `foo.bar`, and empty path segments are
    /// ignored so malformed or blank subjects do not enter dependency
    /// traversal.
    fn from_token(token: &str) -> Option<Self> {
        let segments = token
            .split('.')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if segments.is_empty() {
            None
        } else {
            Some(Self { segments })
        }
    }

    /// Converts the import subject to the relative `.x` file path DSLX
    /// searches.
    ///
    /// For example, `foo.bar` becomes `foo/bar.x`. The caller is responsible
    /// for trying that relative path against the applicable import roots.
    fn relative_path(&self) -> PathBuf {
        let mut path = self
            .segments
            .iter()
            .fold(PathBuf::new(), |mut path, segment| {
                path.push(segment);
                path
            });
        path.set_extension("x");
        path
    }
}

/// Returns the DSLX files whose contents can affect a typed DSLX AOT build.
///
/// Cargo build scripts must report every source file that can change the
/// generated object, metadata, and wrapper artifacts. The roots are the top
/// DSLX module and every requested type module, then imports are followed
/// through the same search roots that DSLX conversion uses.
pub fn collect_typed_dslx_aot_dependencies(
    spec: &TypedDslxAotBuildSpec<'_>,
) -> MetadataResult<BTreeSet<PathBuf>> {
    let mut dependencies = BTreeSet::new();
    let mut pending_paths = std::iter::once(spec.dslx_path.to_path_buf())
        .chain(spec.type_module_paths.iter().map(|path| path.to_path_buf()))
        .collect::<Vec<_>>();

    while let Some(path) = pending_paths.pop() {
        let canonical_path = std::fs::canonicalize(&path).map_err(|e| {
            XlsynthError(format!(
                "AOT I/O failed while resolving DSLX dependency {}: {e}",
                path.display()
            ))
        })?;
        if dependencies.insert(canonical_path.clone()) {
            let dslx_text = std::fs::read_to_string(&canonical_path).map_err(|e| {
                XlsynthError(format!(
                    "AOT I/O failed for DSLX dependency {}: {e}",
                    canonical_path.display()
                ))
            })?;
            pending_paths.extend(
                dslx_import_subjects(&dslx_text)
                    .into_iter()
                    .filter_map(|subject| {
                        resolve_dslx_import_path(&canonical_path, &subject, spec)
                    }),
            );
        }
    }

    Ok(dependencies)
}

/// Extracts DSLX import subjects from source text for Cargo invalidation.
///
/// This is intentionally a lightweight source scan, not a full parser. It only
/// feeds `cargo:rerun-if-changed` discovery; actual typechecking still uses the
/// DSLX front end and reports authoritative syntax or import errors.
fn dslx_import_subjects(dslx_text: &str) -> Vec<DslxImportSubject> {
    dslx_text
        .lines()
        .flat_map(dslx_import_subjects_from_line)
        .collect()
}

/// Extracts import subjects from one source line after dropping trailing
/// comments.
///
/// The scan accepts multiple semicolon-delimited statements on a line because
/// build-script invalidation should be conservative for compact fixture files.
fn dslx_import_subjects_from_line(line: &str) -> Vec<DslxImportSubject> {
    let code = line.split("//").next().unwrap_or("").trim();
    code.split(';')
        .filter_map(|statement| {
            let mut tokens = statement.split_whitespace();
            if tokens.next() == Some("import") {
                tokens.next().and_then(DslxImportSubject::from_token)
            } else {
                None
            }
        })
        .collect()
}

/// Resolves one DSLX import subject against the same roots used by conversion.
///
/// Resolution tries the importing file's directory, explicit additional search
/// paths, and the configured or default DSLX standard library. Missing imports
/// are left unresolved here so typechecking can produce the canonical error.
fn resolve_dslx_import_path(
    importer_path: &Path,
    subject: &DslxImportSubject,
    spec: &TypedDslxAotBuildSpec<'_>,
) -> Option<PathBuf> {
    let importer_dir = importer_path.parent();
    let default_stdlib_path = xlsynth::default_dslx_stdlib_path();
    let stdlib_path = spec
        .dslx_options
        .dslx_stdlib_path
        .unwrap_or(default_stdlib_path);
    let relative_path = subject.relative_path();
    importer_dir
        .into_iter()
        .chain(spec.dslx_options.additional_search_paths.iter().copied())
        .chain(std::iter::once(stdlib_path))
        .map(|root| root.join(&relative_path))
        .find(|path| path.is_file())
}

/// Parses DSLX source text as a module with an explicit module name.
///
/// The explicit name is needed for type modules discovered through import
/// roots: their file name alone is not enough to preserve the dotted DSLX
/// module path that later maps to nested Rust modules.
fn parse_dslx_text_as_module(
    dslx_text: &str,
    path: &Path,
    module_name: &str,
    import_data: &mut dslx::ImportData,
) -> MetadataResult<dslx::TypecheckedModule> {
    let path_str = path.to_str().ok_or_else(|| {
        XlsynthError(format!(
            "AOT build environment error: DSLX path is not UTF-8: {}",
            path.display()
        ))
    })?;
    dslx::parse_and_typecheck(dslx_text, path_str, module_name, import_data)
}

/// Computes the canonical DSLX module name for an imported source path.
///
/// If the file is below an additional search root, the relative path becomes a
/// dotted module name such as `foo.widget`. Otherwise the file stem fallback
/// matches normal top-module handling.
fn dslx_module_name_from_import_path(
    path: &Path,
    search_paths: &[&Path],
) -> MetadataResult<String> {
    for search_path in search_paths {
        if let Ok(relative_path) = path.strip_prefix(search_path) {
            let without_extension = relative_path.with_extension("");
            let segments = without_extension
                .components()
                .filter_map(|component| match component {
                    std::path::Component::Normal(segment) => {
                        Some(segment.to_string_lossy().to_string())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if !segments.is_empty() {
                return Ok(segments.join("."));
            }
        }
    }
    Ok(dslx_path_to_module_name(path)?.to_string())
}

/// Validates package-wide options that must be shared by every entrypoint.
fn ensure_package_specs_compatible(specs: &[TypedDslxAotBuildSpec<'_>]) -> MetadataResult<()> {
    let Some(first) = specs.first() else {
        return Err(XlsynthError(
            "AOT invalid argument: typed DSLX package must contain at least one entrypoint"
                .to_string(),
        ));
    };
    for spec in specs.iter().skip(1) {
        if spec.dslx_options != first.dslx_options {
            return Err(XlsynthError(
                "AOT invalid argument: typed DSLX package entrypoints must use identical DSLX conversion options"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// Typechecks all unique modules participating in one shared typed DSLX
/// package.
fn typecheck_typed_dslx_package_modules(
    specs: &[TypedDslxAotBuildSpec<'_>],
) -> MetadataResult<TypedDslxPackageTypecheckedModules> {
    ensure_package_specs_compatible(specs)?;
    let first = &specs[0];
    let mut import_data = dslx::ImportData::new(
        first.dslx_options.dslx_stdlib_path,
        &first.dslx_options.additional_search_paths,
    );
    let mut seen_paths = BTreeSet::new();
    let mut modules = Vec::new();

    for path in specs.iter().flat_map(|spec| spec.type_module_paths.iter()) {
        let canonical_path = std::fs::canonicalize(path).map_err(|e| {
            XlsynthError(format!(
                "AOT I/O failed while resolving DSLX package module {}: {e}",
                path.display()
            ))
        })?;
        if seen_paths.insert(canonical_path.clone()) {
            let dslx_text = std::fs::read_to_string(&canonical_path).map_err(|e| {
                XlsynthError(format!(
                    "AOT I/O failed for DSLX package module {}: {e}",
                    canonical_path.display()
                ))
            })?;
            let module_name = dslx_module_name_from_import_path(
                path,
                &first.dslx_options.additional_search_paths,
            )?;
            modules.push(TypedDslxPackageModule {
                canonical_path,
                typechecked: parse_dslx_text_as_module(
                    &dslx_text,
                    path,
                    &module_name,
                    &mut import_data,
                )?,
            });
        }
    }

    for spec in specs {
        let canonical_path = std::fs::canonicalize(spec.dslx_path).map_err(|e| {
            XlsynthError(format!(
                "AOT I/O failed while resolving DSLX package top {}: {e}",
                spec.dslx_path.display()
            ))
        })?;
        if seen_paths.insert(canonical_path.clone()) {
            let dslx_text = std::fs::read_to_string(&canonical_path).map_err(|e| {
                XlsynthError(format!(
                    "AOT I/O failed for DSLX package top {}: {e}",
                    canonical_path.display()
                ))
            })?;
            let module_name = dslx_module_name_from_import_path(
                spec.dslx_path,
                &first.dslx_options.additional_search_paths,
            )?;
            modules.push(TypedDslxPackageModule {
                canonical_path,
                typechecked: parse_dslx_text_as_module(
                    &dslx_text,
                    spec.dslx_path,
                    &module_name,
                    &mut import_data,
                )?,
            });
        }
    }

    Ok(TypedDslxPackageTypecheckedModules { modules })
}

fn find_typed_dslx_package_top_module<'a>(
    typechecked: &'a TypedDslxPackageTypecheckedModules,
    spec: &TypedDslxAotBuildSpec<'_>,
) -> MetadataResult<&'a dslx::TypecheckedModule> {
    let canonical_top_path = std::fs::canonicalize(spec.dslx_path).map_err(|e| {
        XlsynthError(format!(
            "AOT I/O failed while resolving DSLX package top {}: {e}",
            spec.dslx_path.display()
        ))
    })?;
    typechecked
        .modules
        .iter()
        .find(|module| module.canonical_path == canonical_top_path)
        .map(|module| &module.typechecked)
        .ok_or_else(|| {
            XlsynthError(format!(
                "AOT typed DSLX package could not find top module for {}",
                spec.dslx_path.display()
            ))
        })
}

/// Collects the module-local type definitions needed for typed AOT lowering.
///
/// The result deliberately records struct definitions, not only names, because
/// imported modules may declare same-named structs and the lowerer must prefer
/// exact definition identity when the DSLX bindings expose it.
fn collect_module_context(typechecked_module: &dslx::TypecheckedModule) -> TypedDslxModuleContext {
    let module = typechecked_module.get_module();
    let mut struct_names = BTreeSet::new();
    let mut struct_defs = Vec::new();
    let mut type_alias_defs = Vec::new();
    let mut enum_names = BTreeSet::new();
    for index in 0..module.get_member_count() {
        if let Some(member) = module.get_member(index).to_matchable() {
            match member {
                dslx::MatchableModuleMember::StructDef(struct_def) => {
                    let name = struct_def.get_identifier();
                    struct_names.insert(name.clone());
                    struct_defs.push(TypedDslxStructDefContext { def: struct_def });
                }
                dslx::MatchableModuleMember::TypeAlias(type_alias) => {
                    type_alias_defs.push(TypedDslxAliasDefContext {
                        name: type_alias.get_identifier(),
                        def: type_alias,
                    });
                }
                dslx::MatchableModuleMember::EnumDef(enum_def) => {
                    enum_names.insert(enum_def.get_identifier());
                }
                _ => {
                    // Only named types matter when later qualifying generated
                    // Rust paths.
                }
            }
        }
    }
    TypedDslxModuleContext {
        dslx_name: module.get_name(),
        type_info: typechecked_module.get_type_info(),
        struct_names,
        struct_defs,
        type_alias_defs,
        enum_names,
    }
}

impl TypedDslxTypeContext {
    /// Builds lookup state from an arbitrary ordered set of typechecked
    /// modules.
    fn from_modules<'a>(modules: impl IntoIterator<Item = &'a dslx::TypecheckedModule>) -> Self {
        Self {
            modules: modules.into_iter().map(collect_module_context).collect(),
        }
    }

    /// Finds a type alias by module and DSLX alias identifier.
    fn type_alias_in_module<'a>(
        &'a self,
        module_name: &str,
        alias_name: &str,
    ) -> Option<(&'a TypedDslxModuleContext, &'a dslx::TypeAlias)> {
        self.modules
            .iter()
            .find(|module| module.dslx_name == module_name)
            .and_then(|module| {
                module
                    .type_alias_defs
                    .iter()
                    .find(|alias| alias.name == alias_name)
                    .map(|alias| (module, &alias.def))
            })
    }

    /// Finds the module context that produced a `TypeInfo` object.
    fn module_for_type_info<'a>(
        &'a self,
        type_info: &dslx::TypeInfo,
    ) -> Option<&'a TypedDslxModuleContext> {
        self.modules
            .iter()
            .find(|module| module.type_info.is_same_type_context(type_info))
    }

    /// Finds the module that owns a DSLX type alias definition.
    fn defining_module_for_type_alias(
        &self,
        current_type_info: Option<&dslx::TypeInfo>,
        type_alias: &dslx::TypeAlias,
    ) -> MetadataResult<Option<&TypedDslxModuleContext>> {
        let alias_name = type_alias.get_identifier();
        let exact_matches = self
            .modules
            .iter()
            .filter(|module| {
                module
                    .type_alias_defs
                    .iter()
                    .any(|known| known.def.is_same_definition(type_alias))
            })
            .collect::<Vec<_>>();
        match exact_matches.as_slice() {
            [module] => return Ok(Some(module)),
            modules if modules.len() > 1 => {
                return Err(XlsynthError(format!(
                    "AOT typed DSLX type lowering found multiple defining modules for type alias `{alias_name}`"
                )));
            }
            _ => {}
        }
        let name_matches = self
            .modules
            .iter()
            .filter(|module| {
                module
                    .type_alias_defs
                    .iter()
                    .any(|known| known.name == alias_name)
            })
            .collect::<Vec<_>>();
        match name_matches.as_slice() {
            [] => Ok(None),
            [module] => Ok(Some(module)),
            _ => {
                if let Some(current_type_info) = current_type_info {
                    let current_match = name_matches
                        .iter()
                        .copied()
                        .find(|module| module.type_info.is_same_type_context(current_type_info));
                    if current_match.is_some() {
                        return Ok(current_match);
                    }
                }
                Err(XlsynthError(format!(
                    "AOT typed DSLX type lowering found multiple DSLX type aliases named `{alias_name}`"
                )))
            }
        }
    }

    /// Resolves one type-reference annotation to the RHS annotation of a DSLX
    /// type alias, when the reference names an alias rather than a concrete
    /// type.
    fn type_alias_rhs_for_annotation<'a>(
        &'a self,
        current_type_info: &'a dslx::TypeInfo,
        type_annotation: &dslx::TypeAnnotation,
    ) -> MetadataResult<Option<ResolvedDslxTypeAnnotation<'a>>> {
        let Some(type_ref_annotation) = type_annotation.to_type_ref_type_annotation() else {
            return Ok(None);
        };
        let type_definition = type_ref_annotation.get_type_ref().get_type_definition();
        if let Some(colon_ref) = type_definition.to_colon_ref() {
            let alias_name = colon_ref.get_attr();
            let Some(import) = colon_ref.resolve_import_subject() else {
                return Ok(None);
            };
            let module_name = import.get_subject().join(".");
            let Some((module, type_alias)) = self.type_alias_in_module(&module_name, &alias_name)
            else {
                return Ok(None);
            };
            return Ok(Some(ResolvedDslxTypeAnnotation {
                type_info: &module.type_info,
                annotation: type_alias.get_type_annotation(),
            }));
        }
        let Some(type_alias) = type_definition.to_type_alias() else {
            return Ok(None);
        };
        let alias_name = type_alias.get_identifier();
        if let Some(module) = self.module_for_type_info(current_type_info) {
            if let Some(alias) = module
                .type_alias_defs
                .iter()
                .find(|alias| alias.name == alias_name)
            {
                return Ok(Some(ResolvedDslxTypeAnnotation {
                    type_info: &module.type_info,
                    annotation: alias.def.get_type_annotation(),
                }));
            }
        }
        Ok(Some(ResolvedDslxTypeAnnotation {
            type_info: current_type_info,
            annotation: type_alias.get_type_annotation(),
        }))
    }

    /// Expands type aliases until the annotation names a non-alias type.
    fn expand_type_alias_rhs_annotation<'a>(
        &'a self,
        current_type_info: &'a dslx::TypeInfo,
        type_annotation: &dslx::TypeAnnotation,
        depth: usize,
    ) -> MetadataResult<Option<ResolvedDslxTypeAnnotation<'a>>> {
        const MAX_ALIAS_EXPANSION_DEPTH: usize = 32;
        if depth >= MAX_ALIAS_EXPANSION_DEPTH {
            return Err(XlsynthError(format!(
                "AOT typed DSLX type lowering exceeded alias expansion depth of {MAX_ALIAS_EXPANSION_DEPTH}"
            )));
        }
        let Some(resolved) =
            self.type_alias_rhs_for_annotation(current_type_info, type_annotation)?
        else {
            return Ok(None);
        };
        let next = self.expand_type_alias_rhs_annotation(
            resolved.type_info,
            &resolved.annotation,
            depth + 1,
        )?;
        Ok(next.or(Some(resolved)))
    }

    /// Finds the module that owns a DSLX struct definition.
    ///
    /// Exact definition identity is preferred because same-named structs are
    /// legal in different imported modules. If exact identity is unavailable,
    /// the current type context disambiguates a bare-name match before the
    /// lowerer reports ambiguity.
    fn defining_module_for_struct(
        &self,
        current_type_info: Option<&dslx::TypeInfo>,
        struct_def: &dslx::StructDef,
    ) -> MetadataResult<Option<&TypedDslxModuleContext>> {
        let struct_name = struct_def.get_identifier();
        let exact_matches = self
            .modules
            .iter()
            .filter(|module| {
                module
                    .struct_defs
                    .iter()
                    .any(|known| known.def.is_same_definition(struct_def))
            })
            .collect::<Vec<_>>();
        match exact_matches.as_slice() {
            [module] => return Ok(Some(module)),
            modules if modules.len() > 1 => {
                return Err(XlsynthError(format!(
                    "AOT typed DSLX type lowering found multiple defining modules for struct `{struct_name}`"
                )));
            }
            _ => {}
        }
        let name_matches = self
            .modules
            .iter()
            .filter(|module| module.struct_names.contains(&struct_name))
            .collect::<Vec<_>>();
        match name_matches.as_slice() {
            [] => Ok(None),
            [module] => Ok(Some(module)),
            _ => {
                if let Some(current_type_info) = current_type_info {
                    let current_match = name_matches
                        .iter()
                        .copied()
                        .find(|module| module.type_info.is_same_type_context(current_type_info));
                    if current_match.is_some() {
                        return Ok(current_match);
                    }
                }
                Err(XlsynthError(format!(
                    "AOT typed DSLX type lowering found multiple DSLX structs named `{struct_name}`"
                )))
            }
        }
    }

    /// Finds the module that owns a DSLX enum definition.
    ///
    /// Non-empty enums expose owner information through their first member's
    /// value expression. Empty enums have no member to inspect, so the lookup
    /// falls back to the unique declared enum name across participating
    /// modules.
    fn defining_module_for_enum(
        &self,
        enum_def: &dslx::EnumDef,
    ) -> MetadataResult<Option<&TypedDslxModuleContext>> {
        if enum_def.get_member_count() == 0 {
            let enum_name = enum_def.get_identifier();
            return self
                .modules
                .iter()
                .find(|module| module.enum_names.contains(&enum_name))
                .map(Some)
                .ok_or_else(|| {
                    XlsynthError(format!(
                        "AOT typed DSLX type lowering could not find defining module for enum `{enum_name}`"
                    ))
                });
        }
        let defining_module_name = enum_def
            .get_member(0)
            .get_value()
            .get_owner_module()
            .get_name();
        self.modules
            .iter()
            .find(|module| module.dslx_name == defining_module_name)
            .map(Some)
            .ok_or_else(|| {
                XlsynthError(format!(
                    "AOT typed DSLX type lowering could not find defining module `{defining_module_name}` for enum `{}`",
                    enum_def.get_identifier()
                ))
            })
    }

    /// Resolves a module qualifier from an explicit DSLX type annotation.
    ///
    /// A colon reference such as `foo::Widget` is stronger evidence than the
    /// concrete type alone because aliases can hide the original source module.
    /// Preserving that qualifier keeps generated Rust paths canonical.
    fn defining_module_for_type_annotation(
        &self,
        type_annotation: &dslx::TypeAnnotation,
    ) -> MetadataResult<Option<&TypedDslxModuleContext>> {
        let defining_module_name = type_annotation
            .to_type_ref_type_annotation()
            .and_then(|annotation| {
                annotation
                    .get_type_ref()
                    .get_type_definition()
                    .to_colon_ref()
                    .and_then(|colon_ref| colon_ref.resolve_import_subject())
            })
            .map(|import| import.get_subject().join("."));
        match defining_module_name {
            Some(defining_module_name) => self
                .modules
                .iter()
                .find(|module| module.dslx_name == defining_module_name)
                .map(Some)
                .ok_or_else(|| {
                    XlsynthError(format!(
                        "AOT typed DSLX type lowering could not find defining module `{defining_module_name}`"
                    ))
                }),
            None => Ok(None),
        }
    }

    /// Selects the `TypeInfo` that should be used to inspect a resolved type.
    ///
    /// The current function's `TypeInfo` is correct for local bits and arrays,
    /// but imported structs and enums must use the defining module's `TypeInfo`
    /// when reading fields, underlying enum types, or constant values.
    fn type_context_for_resolved_type<'a>(
        &'a self,
        current: &'a dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> MetadataResult<&'a dslx::TypeInfo> {
        if let Some(module) = type_annotation
            .map(|annotation| self.defining_module_for_type_annotation(annotation))
            .transpose()?
            .flatten()
        {
            Ok(&module.type_info)
        } else if ty.is_enum() {
            self.type_info_for_enum(current, &ty.get_enum_def()?)
        } else if ty.is_struct() {
            self.type_info_for_struct(current, &ty.get_struct_def()?)
        } else {
            Ok(current)
        }
    }

    /// Returns the defining module's `TypeInfo` for a struct when it is known.
    ///
    /// Falling back to the current context keeps same-module definitions simple
    /// and lets the caller continue when a DSLX binding does not expose owner
    /// information for an otherwise local type.
    fn type_info_for_struct<'a>(
        &'a self,
        current: &'a dslx::TypeInfo,
        struct_def: &dslx::StructDef,
    ) -> MetadataResult<&'a dslx::TypeInfo> {
        let Some(module) = self.defining_module_for_struct(Some(current), struct_def)? else {
            return Ok(current);
        };
        Ok(&module.type_info)
    }

    /// Returns the defining module's `TypeInfo` for an enum when it is known.
    ///
    /// Enum lowering needs the owner context to evaluate discriminants and read
    /// the underlying type annotation with the same bindings DSLX typechecking
    /// used.
    fn type_info_for_enum<'a>(
        &'a self,
        current: &'a dslx::TypeInfo,
        enum_def: &dslx::EnumDef,
    ) -> MetadataResult<&'a dslx::TypeInfo> {
        let Some(module) = self.defining_module_for_enum(enum_def)? else {
            return Ok(current);
        };
        Ok(&module.type_info)
    }

    /// Renders the generated Rust type path from a typechecked DSLX type alone.
    ///
    /// A concrete `dslx::Type` knows the semantic shape, definitions, fields,
    /// widths, and enum underlying bits after typechecking. It does not always
    /// preserve the caller-facing source spelling, so aliases and imported
    /// annotation paths are handled by `rust_type_path_for_resolved_type`.
    /// Imported structs and enums are rendered relative to the generated module
    /// for the local DSLX file.
    fn rust_type_for_concrete_type(
        &self,
        local_module_name: &str,
        current_type_info: &dslx::TypeInfo,
        ty: &dslx::Type,
    ) -> MetadataResult<String> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let signed_str = if is_signed { "S" } else { "U" };
            Ok(format!("{signed_str}Bits<{bit_count}>"))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def()?;
            let enum_name = enum_def.get_identifier();
            let defining_module = self.defining_module_for_enum(&enum_def)?;
            Ok(match defining_module {
                Some(module) if module.dslx_name != local_module_name => {
                    rust_type_path_between_dslx_modules(
                        local_module_name,
                        &module.dslx_name,
                        &enum_name,
                    )
                }
                _ => enum_name,
            })
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def()?;
            let struct_name = struct_def.get_identifier();
            let defining_module =
                self.defining_module_for_struct(Some(current_type_info), &struct_def)?;
            Ok(match defining_module {
                Some(module) if module.dslx_name != local_module_name => {
                    rust_type_path_between_dslx_modules(
                        local_module_name,
                        &module.dslx_name,
                        &struct_name,
                    )
                }
                _ => struct_name,
            })
        } else if ty.is_array() {
            let element = ty.get_array_element_type();
            let element_rust_type =
                self.rust_type_for_concrete_type(local_module_name, current_type_info, &element)?;
            Ok(format!("[{element_rust_type}; {}]", ty.get_array_size()))
        } else {
            let ty_text = ty.to_string()?;
            if ty_text.trim_start().starts_with('(') {
                return Ok(
                    parse_concrete_dslx_type_shape(&ty_text)?.rust_type(RustTypeTarget::PirAot)
                );
            }
            Err(XlsynthError(format!(
                "AOT typed DSLX type lowering does not support DSLX type `{}`",
                ty_text
            )))
        }
    }

    /// Renders the generated Rust type path from source spelling plus concrete
    /// type.
    ///
    /// In this context, resolved means the caller has both the optional source
    /// `TypeAnnotation` and the typechecked `dslx::Type`. The annotation is
    /// checked first so public signatures preserve aliases and imported paths;
    /// the concrete type fallback is used when the annotation cannot name a
    /// generated Rust type directly, such as synthesized array element types.
    fn rust_type_path_for_resolved_type(
        &self,
        local_module_name: &str,
        current_type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> MetadataResult<String> {
        if let Some(type_annotation) = type_annotation {
            let rust_type = DslxTypeMetadata::rust_type_name_from_dslx_module(
                local_module_name,
                current_type_info,
                Some(type_annotation),
                ty,
            )?;
            if let Some(type_ref_annotation) = type_annotation.to_type_ref_type_annotation() {
                let type_definition = type_ref_annotation.get_type_ref().get_type_definition();
                if type_definition.to_colon_ref().is_none() {
                    let defining_module = if let Some(type_alias) = type_definition.to_type_alias()
                    {
                        self.defining_module_for_type_alias(Some(current_type_info), &type_alias)?
                    } else if ty.is_struct() {
                        self.defining_module_for_struct(
                            Some(current_type_info),
                            &ty.get_struct_def()?,
                        )?
                    } else if ty.is_enum() {
                        self.defining_module_for_enum(&ty.get_enum_def()?)?
                    } else {
                        None
                    };
                    if let Some(defining_module) = defining_module
                        && defining_module.dslx_name != local_module_name
                    {
                        return Ok(rust_type_path_between_dslx_modules(
                            local_module_name,
                            &defining_module.dslx_name,
                            &rust_type,
                        ));
                    }
                }
            }
            return Ok(rust_type);
        }
        self.rust_type_for_concrete_type(local_module_name, current_type_info, ty)
    }
}

/// Discovers concrete parametric structs that must be emitted outside their use
/// site.
///
/// Generated Rust type emission normally sees only the module currently being
/// rendered. This collector walks the full typed-AOT module set first so an
/// owner module can later emit the concrete Rust struct required by a direct
/// imported use in another module. Collected entries are deduplicated by
/// semantic definition identity plus concrete Rust name; same-named
/// declarations in sibling modules must remain separate.
struct TypedConcreteParametricStructCollector<'a> {
    context: &'a TypedDslxTypeContext,
    current_module_name: String,
    structs: Vec<TypedConcreteParametricStruct>,
}

impl<'a> TypedConcreteParametricStructCollector<'a> {
    /// Creates an empty collector for one typed-AOT wrapper build.
    fn new(context: &'a TypedDslxTypeContext) -> Self {
        Self {
            context,
            current_module_name: String::new(),
            structs: Vec::new(),
        }
    }

    /// Walks one resolved DSLX type and records reachable concrete struct
    /// specializations.
    ///
    /// `current_type_info` must describe the module that wrote the annotation
    /// so caller-local parametric expressions are evaluated in the caller's
    /// context. When recursion enters struct fields, the walk switches to the
    /// defining module context for field inspection while still emitting the
    /// eventual concrete item in the owner's Rust module.
    fn collect_type(
        &mut self,
        current_module_name: &str,
        current_type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> MetadataResult<()> {
        if let Some(type_annotation) = type_annotation {
            if let Some(array_annotation) = type_annotation.to_array_type_annotation() {
                if ty.is_array() {
                    let element_annotation = array_annotation.get_element_type();
                    let element_ty = ty.get_array_element_type();
                    self.collect_type(
                        current_module_name,
                        current_type_info,
                        Some(&element_annotation),
                        &element_ty,
                    )?;
                    return Ok(());
                }
            }
            if let Some(type_ref_annotation) = type_annotation.to_type_ref_type_annotation() {
                if type_ref_annotation.get_parametric_count() > 0 && ty.is_struct() {
                    let struct_def = ty.get_struct_def()?;
                    let Some(defining_module) = self
                        .context
                        .defining_module_for_struct(Some(current_type_info), &struct_def)?
                    else {
                        return Err(XlsynthError(format!(
                            "AOT typed DSLX specialization collection could not find defining module for struct `{}`",
                            struct_def.get_identifier()
                        )));
                    };
                    let rust_name = DslxTypeMetadata::rust_type_name_from_dslx_module(
                        &defining_module.dslx_name,
                        current_type_info,
                        Some(type_annotation),
                        ty,
                    )?;
                    let lowered = lower_typed_dslx_type(
                        self.context,
                        &defining_module.dslx_name,
                        &defining_module.type_info,
                        Some(type_annotation),
                        ty,
                        rust_name.clone(),
                    )?;
                    let TypedDslxType::Struct { fields, .. } = lowered else {
                        unreachable!("parametric structs lower to struct types");
                    };
                    let already_collected = self.structs.iter().any(|existing| {
                        existing.defining_module_name == defining_module.dslx_name
                            && existing.struct_def.is_same_definition(&struct_def)
                            && existing.rust_name == rust_name
                    });
                    if !already_collected {
                        self.structs.push(TypedConcreteParametricStruct {
                            struct_def,
                            defining_module_name: defining_module.dslx_name.clone(),
                            rust_name,
                            fields,
                        });
                    }
                }
            }
        }

        if ty.is_struct() {
            let struct_def = ty.get_struct_def()?;
            let struct_type_info = self
                .context
                .type_info_for_struct(current_type_info, &struct_def)?;
            let struct_module_name = self
                .context
                .defining_module_for_struct(Some(current_type_info), &struct_def)?
                .map(|module| module.dslx_name.as_str())
                .unwrap_or(current_module_name);
            let member_count = struct_def.get_member_count();
            for index in 0..member_count {
                let member = struct_def.get_member(index);
                let field_annotation = member.get_type();
                let field_ty = if struct_def.is_parametric() {
                    ty.get_struct_member_type(index)
                } else {
                    struct_type_info.get_type_for_struct_member(&member)
                };
                let field_type_info = self.context.type_context_for_resolved_type(
                    struct_type_info,
                    Some(&field_annotation),
                    &field_ty,
                )?;
                self.collect_type(
                    struct_module_name,
                    field_type_info,
                    Some(&field_annotation),
                    &field_ty,
                )?;
            }
        } else if ty.is_array() {
            let element_ty = ty.get_array_element_type();
            self.collect_type(current_module_name, current_type_info, None, &element_ty)?;
        }
        Ok(())
    }

    /// Returns collected specializations in deterministic discovery order.
    fn into_structs(self) -> Vec<TypedConcreteParametricStruct> {
        self.structs
    }
}

/// Feeds the specialization collector from the typed DSLX traversal callbacks.
///
/// Untyped traversal callbacks and non-type-bearing members are intentionally
/// ignored: only typed structs, aliases, and function signatures can expose the
/// concrete imported instantiations this prepass must materialize.
impl BridgeBuilder for TypedConcreteParametricStructCollector<'_> {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.current_module_name = module_name.to_string();
        Ok(())
    }

    fn end_module(&mut self, _module_name: &str) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_enum_def(
        &mut self,
        _dslx_name: &str,
        _is_signed: bool,
        _underlying_bit_count: usize,
        _members: &[(String, xlsynth::IrValue)],
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_struct_def(
        &mut self,
        _dslx_name: &str,
        _members: &[xlsynth::dslx_bridge::StructMemberData],
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_struct_def_typed(
        &mut self,
        _dslx_name: &str,
        type_info: &dslx::TypeInfo,
        members: &[xlsynth::dslx_bridge::StructMemberData],
    ) -> Result<(), XlsynthError> {
        let module_name = self.current_module_name.clone();
        for member in members {
            self.collect_type(
                &module_name,
                type_info,
                Some(&member.type_annotation),
                &member.concrete_type,
            )?;
        }
        Ok(())
    }

    fn add_alias(
        &mut self,
        _dslx_name: &str,
        _type_annotation: &dslx::TypeAnnotation,
        _ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_alias_typed(
        &mut self,
        _dslx_name: &str,
        type_info: &dslx::TypeInfo,
        type_annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        let module_name = self.current_module_name.clone();
        self.collect_type(&module_name, type_info, Some(type_annotation), ty)
    }

    fn add_constant(
        &mut self,
        _name: &str,
        _constant_def: &dslx::ConstantDef,
        _ty: &dslx::Type,
        _ir_value: &xlsynth::IrValue,
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_function_signature_typed(
        &mut self,
        _dslx_name: &str,
        type_info: &dslx::TypeInfo,
        params: &[xlsynth::dslx_bridge::FunctionParamData],
        return_type_annotation: Option<&dslx::TypeAnnotation>,
        return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        let module_name = self.current_module_name.clone();
        for param in params {
            if let Some(concrete_type) = &param.concrete_type {
                self.collect_type(
                    &module_name,
                    type_info,
                    Some(&param.type_annotation),
                    concrete_type,
                )?;
            }
        }
        if let (Some(type_annotation), Some(ty)) = (return_type_annotation, return_type) {
            self.collect_type(&module_name, type_info, Some(type_annotation), ty)?;
        }
        Ok(())
    }
}

fn collect_typed_concrete_parametric_structs_from_modules<'a>(
    context: &TypedDslxTypeContext,
    modules: impl IntoIterator<Item = &'a dslx::TypecheckedModule>,
) -> MetadataResult<Vec<TypedConcreteParametricStruct>> {
    let mut collector = TypedConcreteParametricStructCollector::new(context);
    for module in modules {
        convert_imported_module(module, &mut collector)?;
    }
    Ok(collector.into_structs())
}

/// Builds serializable typed AOT metadata from typed DSLX package specs.
///
/// The result preserves the same public module paths, type names, aliases, and
/// entrypoint ownership that native typed DSLX AOT wrapper generation uses.
pub fn build_pir_aot_package_metadata(
    specs: &[TypedDslxAotBuildSpec<'_>],
) -> MetadataResult<PirAotPackageMetadata> {
    ensure_package_specs_compatible(specs)?;
    let typechecked = typecheck_typed_dslx_package_modules(specs)?;
    let context = TypedDslxTypeContext::from_modules(
        typechecked.modules.iter().map(|module| &module.typechecked),
    );
    let concrete_parametric_structs = collect_typed_concrete_parametric_structs_from_modules(
        &context,
        typechecked.modules.iter().map(|module| &module.typechecked),
    )?;
    let mut concrete_structs_by_module =
        concrete_parametric_structs
            .into_iter()
            .fold(BTreeMap::new(), |mut by_module, item| {
                by_module
                    .entry(item.defining_module_name.clone())
                    .or_insert_with(Vec::new)
                    .push(item);
                by_module
            });

    let mut modules = Vec::with_capacity(typechecked.modules.len());
    for module in &typechecked.modules {
        let module_name = module.typechecked.get_module().get_name();
        modules.push(PirAotModule {
            path: rust_module_path_from_dslx_module_name(&module_name),
            declarations: typed_aot_declarations_from_dslx_module(
                &context,
                &module.typechecked,
                concrete_structs_by_module
                    .remove(&module_name)
                    .unwrap_or_default(),
            )?,
        });
    }
    if let Some((module_name, _)) = concrete_structs_by_module.into_iter().next() {
        return Err(XlsynthError(format!(
            "AOT typed DSLX metadata collection requires package module `{module_name}` to be emitted"
        )));
    }

    let mut entrypoints = Vec::with_capacity(specs.len());
    for spec in specs {
        entrypoints.push(typed_aot_entrypoint_from_dslx_spec(
            &context,
            &typechecked,
            spec,
        )?);
    }

    Ok(PirAotPackageMetadata {
        format_version: PIR_AOT_METADATA_FORMAT_VERSION,
        modules,
        entrypoints,
    })
}

fn typed_aot_declarations_from_dslx_module(
    context: &TypedDslxTypeContext,
    typechecked_module: &dslx::TypecheckedModule,
    concrete_structs: Vec<TypedConcreteParametricStruct>,
) -> MetadataResult<Vec<PirAotDecl>> {
    let module = typechecked_module.get_module();
    let type_info = typechecked_module.get_type_info();
    let module_name = module.get_name();
    let module_path = rust_module_path_from_dslx_module_name(&module_name);
    let mut declarations = Vec::new();

    for concrete_struct in concrete_structs {
        declarations.push(PirAotDecl::Struct {
            name: concrete_struct.rust_name,
            fields: concrete_struct
                .fields
                .iter()
                .map(|field| {
                    Ok(PirAotField {
                        name: field.name.clone(),
                        ty: typed_aot_type_from_lowered_dslx_type(&module_path, &field.ty)?,
                    })
                })
                .collect::<MetadataResult<Vec<_>>>()?,
        });
    }

    for index in 0..module.get_member_count() {
        let Some(member) = module.get_member(index).to_matchable() else {
            continue;
        };
        match member {
            dslx::MatchableModuleMember::EnumDef(enum_def) => {
                declarations.push(typed_aot_enum_decl_from_dslx(&enum_def, &type_info)?);
            }
            dslx::MatchableModuleMember::StructDef(struct_def) => {
                if struct_def.is_parametric() {
                    continue;
                }
                let fields = (0..struct_def.get_member_count())
                    .map(|field_index| {
                        let member = struct_def.get_member(field_index);
                        let annotation = member.get_type();
                        let concrete_type = type_info.get_type_for_struct_member(&member);
                        Ok(PirAotField {
                            name: member.get_name(),
                            ty: typed_aot_type_from_dslx_type(
                                context,
                                &module_name,
                                &module_path,
                                &type_info,
                                Some(&annotation),
                                &concrete_type,
                            )?,
                        })
                    })
                    .collect::<MetadataResult<Vec<_>>>()?;
                declarations.push(PirAotDecl::Struct {
                    name: struct_def.get_identifier(),
                    fields,
                });
            }
            dslx::MatchableModuleMember::TypeAlias(type_alias) => {
                let annotation = type_alias.get_type_annotation();
                let concrete_type = type_info
                    .get_type_for_type_annotation(&annotation)
                    .ok_or_else(|| {
                        XlsynthError(format!(
                            "AOT typed DSLX metadata could not resolve alias `{}`",
                            type_alias.get_identifier()
                        ))
                    })?;
                declarations.push(PirAotDecl::Alias {
                    name: type_alias.get_identifier(),
                    target: typed_aot_type_from_dslx_type(
                        context,
                        &module_name,
                        &module_path,
                        &type_info,
                        Some(&annotation),
                        &concrete_type,
                    )?,
                });
            }
            dslx::MatchableModuleMember::ConstantDef(_)
            | dslx::MatchableModuleMember::Function(_)
            | dslx::MatchableModuleMember::Quickcheck(_) => {}
        }
    }

    Ok(declarations)
}

fn typed_aot_enum_decl_from_dslx(
    enum_def: &dslx::EnumDef,
    type_info: &dslx::TypeInfo,
) -> MetadataResult<PirAotDecl> {
    let underlying = type_info
        .get_type_for_type_annotation(&enum_def.get_underlying())
        .ok_or_else(|| {
            XlsynthError(format!(
                "AOT typed DSLX metadata could not resolve enum `{}` underlying type",
                enum_def.get_identifier()
            ))
        })?;
    let (is_signed, bit_count) = underlying.is_bits_like().ok_or_else(|| {
        XlsynthError(format!(
            "AOT typed DSLX metadata expected enum `{}` to have bits-like underlying type",
            enum_def.get_identifier()
        ))
    })?;
    let variants = (0..enum_def.get_member_count())
        .map(|index| {
            let member = enum_def.get_member(index);
            let value = type_info
                .get_const_expr(&member.get_value())?
                .convert_to_ir()?
                .to_bits()?
                .to_u64()?;
            Ok(PirAotEnumVariant {
                name: member.get_name(),
                value,
            })
        })
        .collect::<MetadataResult<Vec<_>>>()?;
    Ok(PirAotDecl::Enum {
        name: enum_def.get_identifier(),
        signedness: pir_aot_signedness(is_signed),
        bit_count,
        variants,
    })
}

fn typed_aot_entrypoint_from_dslx_spec(
    context: &TypedDslxTypeContext,
    typechecked: &TypedDslxPackageTypecheckedModules,
    spec: &TypedDslxAotBuildSpec<'_>,
) -> MetadataResult<PirAotEntrypoint> {
    let top_module = find_typed_dslx_package_top_module(typechecked, spec)?;
    let top_module_name = top_module.get_module().get_name();
    let base_name = sanitize_identifier(spec.name);
    let owning_module_path = rust_module_path_from_dslx_module_name(&top_module_name);
    let typed_signature =
        build_typed_dslx_function_signature(context, top_module, spec.top, &top_module_name)?;
    let ir_top = if spec.dslx_options.force_implicit_token_calling_convention {
        typed_dslx_implicit_token_entrypoint_wrapper_top(&base_name)
    } else {
        mangle_dslx_name_with_calling_convention(
            dslx_path_to_module_name(spec.dslx_path)?,
            spec.top,
            DslxCallingConvention::Typical,
        )?
    };
    Ok(PirAotEntrypoint {
        name: base_name,
        source: PirAotEntrypointSource::GeneratedIr { ir_top },
        owning_module: rust_module_path_from_dslx_module_name(&top_module_name),
        params: typed_signature
            .params
            .iter()
            .map(|param| {
                Ok(PirAotParam {
                    name: param.name.clone(),
                    ty: typed_aot_type_from_lowered_dslx_type(&owning_module_path, &param.ty)?,
                })
            })
            .collect::<MetadataResult<Vec<_>>>()?,
        return_type: typed_aot_type_from_lowered_dslx_type(
            &owning_module_path,
            &typed_signature.return_type,
        )?,
    })
}

fn typed_aot_type_from_dslx_type(
    context: &TypedDslxTypeContext,
    module_name: &str,
    module_path: &[String],
    type_info: &dslx::TypeInfo,
    type_annotation: Option<&dslx::TypeAnnotation>,
    ty: &dslx::Type,
) -> MetadataResult<PirAotType> {
    let type_info = context.type_context_for_resolved_type(type_info, type_annotation, ty)?;
    let rust_type =
        context.rust_type_path_for_resolved_type(module_name, type_info, type_annotation, ty)?;
    let lowered = lower_typed_dslx_type(
        context,
        module_name,
        type_info,
        type_annotation,
        ty,
        rust_type,
    )?;
    typed_aot_type_from_lowered_dslx_type(module_path, &lowered)
}

fn typed_aot_type_from_lowered_dslx_type(
    current_module_path: &[String],
    ty: &TypedDslxType,
) -> MetadataResult<PirAotType> {
    match ty {
        TypedDslxType::Bits {
            rust_type,
            is_signed,
            bit_count,
            ..
        } => Ok(typed_aot_type_ref_from_rust_type_path(current_module_path, rust_type)
            .unwrap_or(PirAotType::Bits {
                signedness: pir_aot_signedness(*is_signed),
                bit_count: *bit_count,
            })),
        TypedDslxType::Enum { rust_type, .. } | TypedDslxType::Struct { rust_type, .. } => {
            typed_aot_type_ref_from_rust_type_path(current_module_path, rust_type).ok_or_else(|| {
                XlsynthError(format!(
                    "AOT typed DSLX metadata could not resolve generated type path `{rust_type}`"
                ))
            })
        }
        TypedDslxType::Tuple { elements } => Ok(PirAotType::Tuple {
            elements: elements
                .iter()
                .map(|element| typed_aot_type_from_lowered_dslx_type(current_module_path, element))
                .collect::<MetadataResult<Vec<_>>>()?,
        }),
        TypedDslxType::Array { size, element } => Ok(PirAotType::Array {
            size: *size,
            element: Box::new(typed_aot_type_from_lowered_dslx_type(
                current_module_path,
                element,
            )?),
        }),
    }
}

fn typed_aot_type_ref_from_rust_type_path(
    current_module_path: &[String],
    rust_type: &str,
) -> Option<PirAotType> {
    let rust_type = rust_type.trim();
    if rust_type.starts_with('[') || rust_type.starts_with('(') || rust_type.contains('<') {
        return None;
    }
    let mut module = current_module_path.to_vec();
    let mut tail = Vec::new();
    for segment in rust_type.split("::") {
        match segment {
            "" | "crate" => return None,
            "self" => {}
            "super" => {
                module.pop()?;
            }
            segment => tail.push(segment.to_string()),
        }
    }
    let name = tail.pop()?;
    module.extend(tail);
    Some(PirAotType::TypeRef { module, name })
}

fn lower_concrete_dslx_type_shape(
    shape: &ConcreteDslxTypeShape,
    rust_type: String,
) -> TypedDslxType {
    match shape {
        ConcreteDslxTypeShape::Bits {
            is_signed,
            bit_count,
        } => TypedDslxType::Bits {
            rust_type,
            is_signed: *is_signed,
            bit_count: *bit_count,
        },
        ConcreteDslxTypeShape::Tuple { elements } => TypedDslxType::Tuple {
            elements: elements
                .iter()
                .map(|element| {
                    lower_concrete_dslx_type_shape(
                        element,
                        element.rust_type(RustTypeTarget::PirAot),
                    )
                })
                .collect(),
        },
        ConcreteDslxTypeShape::Array { size, element } => TypedDslxType::Array {
            size: *size,
            element: Box::new(lower_concrete_dslx_type_shape(
                element,
                element.rust_type(RustTypeTarget::PirAot),
            )),
        },
    }
}

/// Lowers one DSLX type into the semantic model used by typed AOT generation.
///
/// The caller supplies the generated Rust type spelling for the outer type so
/// aliases and imported paths are preserved. Nested fields and array elements
/// resolve their own contexts recursively before code generation flattens them
/// to AOT leaves.
fn lower_typed_dslx_type(
    context: &TypedDslxTypeContext,
    local_module_name: &str,
    current_type_info: &dslx::TypeInfo,
    type_annotation: Option<&dslx::TypeAnnotation>,
    ty: &dslx::Type,
    rust_type: String,
) -> MetadataResult<TypedDslxType> {
    if let Some((is_signed, bit_count)) = ty.is_bits_like() {
        Ok(TypedDslxType::Bits {
            rust_type,
            is_signed,
            bit_count,
        })
    } else if ty.is_enum() {
        Ok(TypedDslxType::Enum { rust_type })
    } else if ty.is_struct() {
        let struct_def = ty.get_struct_def()?;
        let struct_type_info = context.type_info_for_struct(current_type_info, &struct_def)?;
        let definition_member_count = struct_def.get_member_count();
        let concrete_member_count = if struct_def.is_parametric() {
            ty.get_struct_member_count()
        } else {
            definition_member_count
        };
        if concrete_member_count != definition_member_count {
            return Err(XlsynthError(format!(
                "AOT typed DSLX type lowering found {definition_member_count} definition member(s) but {concrete_member_count} concrete member type(s) for struct `{}`",
                struct_def.get_identifier()
            )));
        }
        let fields = (0..concrete_member_count)
            .map(|index| {
                let member = struct_def.get_member(index);
                let field_annotation = member.get_type();
                let field_type = if struct_def.is_parametric() {
                    ty.get_struct_member_type(index)
                } else {
                    struct_type_info.get_type_for_struct_member(&member)
                };
                let field_type_info = context.type_context_for_resolved_type(
                    struct_type_info,
                    Some(&field_annotation),
                    &field_type,
                )?;
                let rust_type = context.rust_type_path_for_resolved_type(
                    local_module_name,
                    field_type_info,
                    Some(&field_annotation),
                    &field_type,
                )?;
                Ok(TypedDslxField {
                    name: member.get_name(),
                    ty: lower_typed_dslx_type(
                        context,
                        local_module_name,
                        field_type_info,
                        Some(&field_annotation),
                        &field_type,
                        rust_type,
                    )?,
                })
            })
            .collect::<MetadataResult<Vec<_>>>()?;
        Ok(TypedDslxType::Struct { rust_type, fields })
    } else if ty.is_array() {
        let element = ty.get_array_element_type();
        let expanded_annotation = type_annotation
            .map(|annotation| {
                context.expand_type_alias_rhs_annotation(current_type_info, annotation, 0)
            })
            .transpose()?
            .flatten();
        let effective_type_info = expanded_annotation
            .as_ref()
            .map(|annotation| annotation.type_info)
            .unwrap_or(current_type_info);
        let effective_annotation = expanded_annotation
            .as_ref()
            .map(|annotation| &annotation.annotation)
            .or(type_annotation);
        let element_annotation = effective_annotation
            .and_then(|annotation| annotation.to_array_type_annotation())
            .map(|annotation| annotation.get_element_type());
        let element_type_info = context.type_context_for_resolved_type(
            effective_type_info,
            element_annotation.as_ref(),
            &element,
        )?;
        let element_rust_type = context.rust_type_path_for_resolved_type(
            local_module_name,
            element_type_info,
            element_annotation.as_ref(),
            &element,
        )?;
        Ok(TypedDslxType::Array {
            size: ty.get_array_size(),
            element: Box::new(lower_typed_dslx_type(
                context,
                local_module_name,
                element_type_info,
                element_annotation.as_ref(),
                &element,
                element_rust_type,
            )?),
        })
    } else {
        let ty_text = ty.to_string()?;
        if ty_text.trim_start().starts_with('(') {
            return Ok(lower_concrete_dslx_type_shape(
                &parse_concrete_dslx_type_shape(&ty_text)?,
                rust_type,
            ));
        }
        Err(XlsynthError(format!(
            "AOT typed DSLX type lowering does not support DSLX type `{}`",
            ty_text
        )))
    }
}

/// Finds the top DSLX function selected for AOT emission.
///
/// This searches only the already-typechecked top module. Missing functions are
/// reported before AOT metadata validation so build scripts fail with the DSLX
/// function name the caller supplied.
fn find_dslx_function(
    typechecked_module: &dslx::TypecheckedModule,
    function_name: &str,
) -> MetadataResult<dslx::Function> {
    let module = typechecked_module.get_module();
    for index in 0..module.get_member_count() {
        if let Some(dslx::MatchableModuleMember::Function(function)) =
            module.get_member(index).to_matchable()
        {
            if function.get_identifier() == function_name {
                return Ok(function);
            }
        }
    }
    Err(XlsynthError(format!(
        "AOT typed DSLX type lowering could not find DSLX function `{function_name}`"
    )))
}

/// Builds the typed DSLX signature for the selected top function.
///
/// The result preserves the public names of generated Rust types and the
/// recursive semantic shape for every parameter and the return value. A return
/// annotation is required because the generated runner needs an explicit Rust
/// return type.
fn build_typed_dslx_function_signature(
    context: &TypedDslxTypeContext,
    top_module: &dslx::TypecheckedModule,
    top: &str,
    rust_module_name: &str,
) -> MetadataResult<TypedAotFunctionSignature> {
    let type_info = top_module.get_type_info();
    let function = find_dslx_function(top_module, top)?;
    let params = (0..function.get_param_count())
        .map(|index| {
            let param = function.get_param(index);
            let annotation = param.get_type_annotation();
            let concrete_type = type_info
                .get_type_for_type_annotation(&annotation)
                .ok_or_else(|| {
                    XlsynthError(format!(
                        "AOT typed DSLX type lowering could not resolve type for parameter `{}`",
                        param.get_name()
                    ))
                })?;
            let param_type_info = context.type_context_for_resolved_type(
                &type_info,
                Some(&annotation),
                &concrete_type,
            )?;
            let rust_type = context.rust_type_path_for_resolved_type(
                rust_module_name,
                param_type_info,
                Some(&annotation),
                &concrete_type,
            )?;
            Ok(TypedDslxParam {
                name: param.get_name(),
                ty: lower_typed_dslx_type(
                    context,
                    rust_module_name,
                    param_type_info,
                    Some(&annotation),
                    &concrete_type,
                    rust_type,
                )?,
            })
        })
        .collect::<MetadataResult<Vec<_>>>()?;

    let return_annotation = function.get_return_type().ok_or_else(|| {
        XlsynthError(format!(
            "AOT typed DSLX type lowering requires function `{top}` to have an explicit return type"
        ))
    })?;
    let return_type = type_info
        .get_type_for_type_annotation(&return_annotation)
        .ok_or_else(|| {
            XlsynthError(format!(
                "AOT typed DSLX type lowering could not resolve return type for function `{top}`"
            ))
        })?;
    let return_type_info = context.type_context_for_resolved_type(
        &type_info,
        Some(&return_annotation),
        &return_type,
    )?;
    let return_rust_type = context.rust_type_path_for_resolved_type(
        rust_module_name,
        return_type_info,
        Some(&return_annotation),
        &return_type,
    )?;
    let typed_dslx_return_type = lower_typed_dslx_type(
        context,
        rust_module_name,
        return_type_info,
        Some(&return_annotation),
        &return_type,
        return_rust_type.clone(),
    )?;
    Ok(TypedAotFunctionSignature {
        params,
        return_type: typed_dslx_return_type,
    })
}
