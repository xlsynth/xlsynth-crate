// SPDX-License-Identifier: Apache-2.0

//! Build-script helpers for emitting standalone native PIR AOT wrappers.

pub mod metadata;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub use metadata::{
    PIR_AOT_METADATA_FORMAT_VERSION, PirAotDecl, PirAotEntrypoint, PirAotEntrypointSource,
    PirAotEnumVariant, PirAotField, PirAotModule, PirAotPackageMetadata, PirAotParam,
    PirAotSignedness, PirAotType,
};
pub use xlsynth::DslxConvertOptions;
use xlsynth::aot_builder as xlsynth_aot_builder;
pub use xlsynth::aot_builder::TypedDslxAotBuildSpec;
use xlsynth::aot_builder::{
    build_native_typed_dslx_aot_package_metadata, collect_typed_dslx_aot_dependencies,
    typed_dslx_implicit_token_entrypoint_wrapper_top,
};
use xlsynth::{
    DslxCallingConvention, IrPackage, convert_dslx_to_ir_text, dslx_path_to_module_name,
    mangle_dslx_name_with_calling_convention, optimize_ir,
};
use xlsynth_pir::ir::{NodePayload, PackageMember, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler_runtime::{
    AssumptionFailureKind, EventKind, EventSiteMetadata, TraceTupleFieldLayout, TraceValueLayout,
};

use crate::{
    AotArtifact, AotEntrypointArtifact, AotPackageEntrypoint, CompilerError,
    NativeTupleFieldLayout, NativeValueLayout, ScalarLayout, WideBitsLayout, compile_aot_package,
    compile_package_aot,
};

/// Inputs required to emit one generated native AOT wrapper from PIR text.
pub struct AotBuildSpec<'a> {
    pub name: &'a str,
    pub pir_text: &'a str,
    pub top: &'a str,
}

/// Output files and metadata produced for one generated native AOT module.
pub struct GeneratedAotModule {
    pub rust_file: PathBuf,
    pub object_file: PathBuf,
    pub artifact: AotArtifact,
}

/// Cargo build-script environment used for DSLX-to-PIR AOT conversion.
pub struct CargoDslxEnv {
    dslx_stdlib_path: PathBuf,
    additional_search_paths: Vec<PathBuf>,
}

impl CargoDslxEnv {
    /// Creates a DSLX build environment from the standard Cargo/Bazel vars.
    pub fn new() -> Result<Self, CompilerError> {
        println!("cargo:rerun-if-env-changed=DSLX_STDLIB_PATH");
        println!("cargo:rerun-if-env-changed=XLSYNTH_ARTIFACT_CONFIG");
        let dslx_stdlib_path = if let Some(path) = std::env::var_os("DSLX_STDLIB_PATH") {
            PathBuf::from(path)
        } else if let Some(config_path) = std::env::var_os("XLSYNTH_ARTIFACT_CONFIG") {
            let config_path = PathBuf::from(config_path);
            config_path
                .parent()
                .ok_or_else(|| {
                    CompilerError::InvalidArgument(format!(
                        "XLSYNTH_ARTIFACT_CONFIG has no parent directory: {}",
                        config_path.display()
                    ))
                })?
                .join("dslx_stdlib")
        } else {
            xlsynth::default_dslx_stdlib_path().to_path_buf()
        };
        Ok(Self {
            dslx_stdlib_path,
            additional_search_paths: Vec::new(),
        })
    }

    /// Adds a DSLX import search root.
    pub fn with_search_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.additional_search_paths.push(path.into());
        self
    }

    /// Returns the DSLX standard library path selected for this build.
    pub fn dslx_stdlib_path(&self) -> &Path {
        &self.dslx_stdlib_path
    }

    /// Returns conversion options borrowing the selected stdlib and search
    /// paths.
    pub fn dslx_options(&self) -> DslxConvertOptions<'_> {
        DslxConvertOptions {
            dslx_stdlib_path: Some(&self.dslx_stdlib_path),
            additional_search_paths: self
                .additional_search_paths
                .iter()
                .map(PathBuf::as_path)
                .collect(),
            ..Default::default()
        }
    }
}

/// One DSLX function exposed by a typed AOT package.
#[derive(Clone, Copy)]
pub struct DslxAotEntrypoint<'a> {
    pub name: &'a str,
    pub top: &'a str,
}

impl<'a> DslxAotEntrypoint<'a> {
    /// Creates an entrypoint whose generated module name and DSLX top match.
    pub const fn new(name: &'a str, top: &'a str) -> Self {
        Self { name, top }
    }
}

fn export_generated_rust_file(env_var: &str, rust_file: &Path) -> Result<(), CompilerError> {
    if env_var.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "AOT generated Rust env var must not be empty".into(),
        ));
    }
    println!("cargo:rustc-env={env_var}={}", rust_file.display());
    Ok(())
}

/// Builds PIR AOT package metadata from typed DSLX entrypoint specs.
pub fn build_pir_aot_package_metadata_from_dslx_specs(
    specs: &[TypedDslxAotBuildSpec<'_>],
) -> Result<PirAotPackageMetadata, CompilerError> {
    build_native_typed_dslx_aot_package_metadata(specs)
        .map_err(|error| CompilerError::Backend(error.to_string()))
        .and_then(pir_aot_package_metadata_from_dslx_bridge_metadata)
}

fn pir_aot_package_metadata_from_dslx_bridge_metadata(
    metadata: xlsynth_aot_builder::TypedAotPackageMetadata,
) -> Result<PirAotPackageMetadata, CompilerError> {
    Ok(PirAotPackageMetadata {
        format_version: PIR_AOT_METADATA_FORMAT_VERSION,
        modules: metadata
            .modules
            .into_iter()
            .map(pir_aot_module_from_dslx_bridge_metadata)
            .collect::<Result<Vec<_>, _>>()?,
        entrypoints: metadata
            .entrypoints
            .into_iter()
            .map(pir_aot_entrypoint_from_dslx_bridge_metadata)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn pir_aot_module_from_dslx_bridge_metadata(
    module: xlsynth_aot_builder::TypedAotModule,
) -> Result<PirAotModule, CompilerError> {
    Ok(PirAotModule {
        path: module.path,
        declarations: module
            .declarations
            .into_iter()
            .map(pir_aot_decl_from_dslx_bridge_metadata)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn pir_aot_decl_from_dslx_bridge_metadata(
    decl: xlsynth_aot_builder::TypedAotDecl,
) -> Result<PirAotDecl, CompilerError> {
    Ok(match decl {
        xlsynth_aot_builder::TypedAotDecl::Struct { name, fields } => PirAotDecl::Struct {
            name,
            fields: fields
                .into_iter()
                .map(pir_aot_field_from_dslx_bridge_metadata)
                .collect(),
        },
        xlsynth_aot_builder::TypedAotDecl::Enum {
            name,
            signedness,
            bit_count,
            variants,
        } => PirAotDecl::Enum {
            name,
            signedness: pir_aot_signedness_from_dslx_bridge_metadata(signedness),
            bit_count,
            variants: variants
                .into_iter()
                .map(pir_aot_enum_variant_from_dslx_bridge_metadata)
                .collect::<Result<Vec<_>, _>>()?,
        },
        xlsynth_aot_builder::TypedAotDecl::Alias { name, target } => PirAotDecl::Alias {
            name,
            target: pir_aot_type_from_dslx_bridge_metadata(target),
        },
    })
}

fn pir_aot_field_from_dslx_bridge_metadata(
    field: xlsynth_aot_builder::TypedAotField,
) -> PirAotField {
    PirAotField {
        name: field.name,
        ty: pir_aot_type_from_dslx_bridge_metadata(field.ty),
    }
}

fn pir_aot_enum_variant_from_dslx_bridge_metadata(
    variant: xlsynth_aot_builder::TypedAotEnumVariant,
) -> Result<PirAotEnumVariant, CompilerError> {
    let value = u64::try_from(variant.value).map_err(|_| {
        CompilerError::InvalidArgument(format!(
            "PIR AOT enum variant `{}` value {} exceeds the current 64-bit metadata limit",
            variant.name, variant.value
        ))
    })?;
    Ok(PirAotEnumVariant {
        name: variant.name,
        value,
    })
}

fn pir_aot_signedness_from_dslx_bridge_metadata(
    signedness: xlsynth_aot_builder::TypedAotSignedness,
) -> PirAotSignedness {
    match signedness {
        xlsynth_aot_builder::TypedAotSignedness::Unsigned => PirAotSignedness::Unsigned,
        xlsynth_aot_builder::TypedAotSignedness::Signed => PirAotSignedness::Signed,
    }
}

fn pir_aot_type_from_dslx_bridge_metadata(ty: xlsynth_aot_builder::TypedAotType) -> PirAotType {
    match ty {
        xlsynth_aot_builder::TypedAotType::Bits {
            signedness,
            bit_count,
        } => PirAotType::Bits {
            signedness: pir_aot_signedness_from_dslx_bridge_metadata(signedness),
            bit_count,
        },
        xlsynth_aot_builder::TypedAotType::Token => PirAotType::Token,
        xlsynth_aot_builder::TypedAotType::Array { size, element } => PirAotType::Array {
            size,
            element: Box::new(pir_aot_type_from_dslx_bridge_metadata(*element)),
        },
        xlsynth_aot_builder::TypedAotType::Tuple { elements } => PirAotType::Tuple {
            elements: elements
                .into_iter()
                .map(pir_aot_type_from_dslx_bridge_metadata)
                .collect(),
        },
        xlsynth_aot_builder::TypedAotType::TypeRef { module, name } => {
            PirAotType::TypeRef { module, name }
        }
    }
}

fn pir_aot_entrypoint_from_dslx_bridge_metadata(
    entrypoint: xlsynth_aot_builder::TypedAotEntrypoint,
) -> Result<PirAotEntrypoint, CompilerError> {
    let source = match entrypoint.ir_file {
        Some(ir_file) => PirAotEntrypointSource::IrFile {
            ir_file,
            ir_top: entrypoint.ir_top,
        },
        None => PirAotEntrypointSource::GeneratedIr {
            ir_top: entrypoint.ir_top,
        },
    };
    Ok(PirAotEntrypoint {
        name: entrypoint.name,
        source,
        owning_module: entrypoint.owning_module,
        params: entrypoint
            .params
            .into_iter()
            .map(pir_aot_param_from_dslx_bridge_metadata)
            .collect(),
        return_type: pir_aot_type_from_dslx_bridge_metadata(entrypoint.return_type),
    })
}

fn pir_aot_param_from_dslx_bridge_metadata(
    param: xlsynth_aot_builder::TypedAotParam,
) -> PirAotParam {
    PirAotParam {
        name: param.name,
        ty: pir_aot_type_from_dslx_bridge_metadata(param.ty),
    }
}

/// Builds one metadata-backed shared wrapper from checked-in IR files.
pub struct TypedIrAotPackageBuilder<'a> {
    name: &'a str,
    metadata_path: PathBuf,
}

/// Output files and entrypoint metadata produced for a typed IR AOT package.
pub struct GeneratedTypedIrAotPackage {
    pub rust_file: PathBuf,
    pub object_file: PathBuf,
    pub entrypoints: Vec<AotEntrypointArtifact>,
}

struct PreparedPackageEntrypoint {
    base_name: String,
    entrypoint_symbol: String,
    package: xlsynth_pir::ir::Package,
    top: String,
}

impl<'a> TypedIrAotPackageBuilder<'a> {
    /// Creates a typed IR AOT package backed by a metadata JSON file.
    ///
    /// IR file paths in the metadata are resolved relative to the metadata
    /// file's parent directory.
    pub fn new(name: &'a str, metadata_path: impl Into<PathBuf>) -> Self {
        Self {
            name,
            metadata_path: metadata_path.into(),
        }
    }

    /// Emits a shared wrapper and linked package object into Cargo's `OUT_DIR`.
    pub fn build(&self) -> Result<GeneratedTypedIrAotPackage, CompilerError> {
        let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
            CompilerError::Backend("OUT_DIR is required while emitting an AOT package".into())
        })?;
        self.build_with_out_dir(Path::new(&out_dir))
    }

    /// Emits a shared wrapper and exports its Rust file path in a rustc env
    /// var.
    pub fn build_and_export_env(
        &self,
        env_var: &str,
    ) -> Result<GeneratedTypedIrAotPackage, CompilerError> {
        let output = self.build()?;
        export_generated_rust_file(env_var, &output.rust_file)?;
        Ok(output)
    }

    /// Emits a shared wrapper and linked package object into `out_dir`.
    pub fn build_with_out_dir(
        &self,
        out_dir: &Path,
    ) -> Result<GeneratedTypedIrAotPackage, CompilerError> {
        if self.name.is_empty() {
            return Err(CompilerError::InvalidArgument(
                "AOT package name must not be empty".into(),
            ));
        }
        let metadata = PirAotPackageMetadata::from_json_file(&self.metadata_path)?;
        let metadata_dir = self.metadata_path.parent().ok_or_else(|| {
            CompilerError::InvalidArgument(format!(
                "PIR AOT metadata path `{}` has no parent directory",
                self.metadata_path.display()
            ))
        })?;
        println!("cargo:rerun-if-changed={}", self.metadata_path.display());
        if metadata.entrypoints.is_empty() {
            return Err(CompilerError::InvalidArgument(
                "AOT package must contain at least one entrypoint".into(),
            ));
        }
        let typed_package = ValidatedPirAotPackage::new(&metadata)?;
        let mut prepared_entrypoints = Vec::with_capacity(metadata.entrypoints.len());
        let mut entrypoint_names = BTreeSet::new();
        for entrypoint in &metadata.entrypoints {
            let base_name = sanitize_identifier(&entrypoint.name);
            if base_name.is_empty() {
                return Err(CompilerError::InvalidArgument(
                    "AOT entrypoint name must contain an identifier character".into(),
                ));
            }
            if !entrypoint_names.insert(base_name.clone()) {
                return Err(CompilerError::InvalidArgument(format!(
                    "AOT package contains duplicate entrypoint name '{base_name}'"
                )));
            }
            let PirAotEntrypointSource::IrFile { ir_file, ir_top } = &entrypoint.source else {
                return Err(CompilerError::InvalidArgument(format!(
                    "PIR AOT metadata entrypoint `{}` must use source.kind `ir_file` for an IR-backed package",
                    entrypoint.name
                )));
            };
            let ir_path = metadata_dir.join(ir_file);
            println!("cargo:rerun-if-changed={}", ir_path.display());
            let pir_text = fs::read_to_string(&ir_path).map_err(|error| {
                CompilerError::Backend(format!(
                    "failed to read IR file {} for PIR AOT entrypoint `{}`: {error}",
                    ir_path.display(),
                    entrypoint.name
                ))
            })?;
            let package = Parser::new(&pir_text)
                .parse_and_validate_package()
                .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
            let entrypoint_symbol = format!("__xlsynth_pir_aot_{base_name}");
            prepared_entrypoints.push(PreparedPackageEntrypoint {
                base_name,
                entrypoint_symbol,
                package,
                top: ir_top.clone(),
            });
        }
        let package_name = sanitize_identifier(self.name);
        let package_object_name = format!("xlsynth_pir_aot_package_{package_name}");
        let package_entrypoints = prepared_entrypoints
            .iter()
            .map(|entrypoint| AotPackageEntrypoint {
                package: &entrypoint.package,
                function_name: &entrypoint.top,
                entrypoint_symbol: &entrypoint.entrypoint_symbol,
            })
            .collect::<Vec<_>>();
        let package_artifact = compile_aot_package(&package_object_name, &package_entrypoints)?;
        let ordered_entrypoints = prepared_entrypoints
            .iter()
            .map(|entrypoint| entrypoint.base_name.clone())
            .zip(package_artifact.entrypoints)
            .collect::<Vec<_>>();
        for (entrypoint, (_, entrypoint_artifact)) in
            metadata.entrypoints.iter().zip(&ordered_entrypoints)
        {
            typed_package.validate_entrypoint_layout(entrypoint, entrypoint_artifact)?;
        }

        fs::create_dir_all(out_dir).map_err(|error| CompilerError::Backend(error.to_string()))?;
        let rust_file = out_dir.join(format!("{package_name}_typed_ir_pir_aot_package.rs"));
        fs::write(
            &rust_file,
            render_pir_aot_package(
                self.name,
                &typed_package,
                &ordered_entrypoints,
                "typed IR AOT package",
            )?,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

        let object_file = out_dir.join(format!("{package_name}.pir_aot.o"));
        fs::write(&object_file, &package_artifact.object_code)
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        cc::Build::new()
            .cargo_metadata(true)
            .object(&object_file)
            .compile(&package_object_name);
        let entrypoints = ordered_entrypoints
            .into_iter()
            .map(|(_, entrypoint_artifact)| entrypoint_artifact)
            .collect();

        Ok(GeneratedTypedIrAotPackage {
            rust_file,
            object_file,
            entrypoints,
        })
    }
}

/// Collects DSLX entrypoints into one shared wrapper and package object.
///
/// Each entrypoint is lowered to XLS IR and optimized by XLS before Cranelift
/// compiles the shared native object.
pub struct TypedDslxAotPackageBuilder<'a> {
    name: &'a str,
    specs: Vec<TypedDslxAotBuildSpec<'a>>,
}

/// Output files and entrypoint metadata produced for a typed DSLX AOT package.
pub struct GeneratedTypedDslxAotPackage {
    pub rust_file: PathBuf,
    pub object_file: PathBuf,
    pub entrypoints: Vec<AotEntrypointArtifact>,
}

struct PreparedDslxAotEntrypoint {
    package: xlsynth_pir::ir::Package,
    top: String,
}

impl<'a> TypedDslxAotPackageBuilder<'a> {
    /// Creates an empty typed DSLX AOT package.
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            specs: Vec::new(),
        }
    }

    /// Adds one top-level DSLX function to the generated AOT package.
    pub fn add_entrypoint(mut self, spec: TypedDslxAotBuildSpec<'a>) -> Self {
        self.specs.push(spec);
        self
    }

    /// Adds multiple entrypoints from the same DSLX file and type bridge set.
    pub fn add_dslx_file(
        mut self,
        dslx_path: &'a Path,
        dslx_options: DslxConvertOptions<'a>,
        type_module_paths: impl IntoIterator<Item = &'a Path>,
        entrypoints: impl IntoIterator<Item = DslxAotEntrypoint<'a>>,
    ) -> Self {
        let type_module_paths = type_module_paths.into_iter().collect::<Vec<_>>();
        for entrypoint in entrypoints {
            self.specs.push(TypedDslxAotBuildSpec {
                name: entrypoint.name,
                dslx_path,
                top: entrypoint.top,
                dslx_options: dslx_options.clone(),
                type_module_paths: type_module_paths.clone(),
            });
        }
        self
    }

    /// Emits a shared wrapper and linked package object into Cargo's `OUT_DIR`.
    pub fn build(&self) -> Result<GeneratedTypedDslxAotPackage, CompilerError> {
        let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
            CompilerError::Backend("OUT_DIR is required while emitting an AOT package".into())
        })?;
        self.build_with_out_dir(Path::new(&out_dir))
    }

    /// Emits a shared wrapper and exports its Rust file path in a rustc env
    /// var.
    pub fn build_and_export_env(
        &self,
        env_var: &str,
    ) -> Result<GeneratedTypedDslxAotPackage, CompilerError> {
        let output = self.build()?;
        export_generated_rust_file(env_var, &output.rust_file)?;
        Ok(output)
    }

    /// Emits a shared wrapper and linked package object into `out_dir`.
    pub fn build_with_out_dir(
        &self,
        out_dir: &Path,
    ) -> Result<GeneratedTypedDslxAotPackage, CompilerError> {
        if self.name.is_empty() {
            return Err(CompilerError::InvalidArgument(
                "AOT package name must not be empty".into(),
            ));
        }
        if self.specs.is_empty() {
            return Err(CompilerError::InvalidArgument(
                "AOT package must contain at least one entrypoint".into(),
            ));
        }
        let mut prepared_entrypoints = Vec::with_capacity(self.specs.len());
        let mut entrypoint_names = BTreeSet::new();
        for spec in &self.specs {
            let base_name = sanitize_identifier(spec.name);
            if base_name.is_empty() {
                return Err(CompilerError::InvalidArgument(
                    "AOT build spec name must contain an identifier character".into(),
                ));
            }
            if !entrypoint_names.insert(base_name.clone()) {
                return Err(CompilerError::InvalidArgument(format!(
                    "AOT package contains duplicate entrypoint name '{base_name}'"
                )));
            }
            let prepared = prepare_dslx_aot_entrypoint(spec, &base_name)?;
            let entrypoint_symbol = format!("__xlsynth_pir_aot_{base_name}");
            prepared_entrypoints.push(PreparedPackageEntrypoint {
                base_name,
                entrypoint_symbol,
                package: prepared.package,
                top: prepared.top,
            });
        }
        let metadata = build_pir_aot_package_metadata_from_dslx_specs(&self.specs)?;
        let typed_package = ValidatedPirAotPackage::new(&metadata)?;
        let package_name = sanitize_identifier(self.name);
        let package_object_name = format!("xlsynth_pir_aot_package_{package_name}");
        let package_entrypoints = prepared_entrypoints
            .iter()
            .map(|entrypoint| AotPackageEntrypoint {
                package: &entrypoint.package,
                function_name: &entrypoint.top,
                entrypoint_symbol: &entrypoint.entrypoint_symbol,
            })
            .collect::<Vec<_>>();
        let package_artifact = compile_aot_package(&package_object_name, &package_entrypoints)?;
        let ordered_entrypoints = prepared_entrypoints
            .iter()
            .map(|entrypoint| entrypoint.base_name.clone())
            .zip(package_artifact.entrypoints)
            .collect::<Vec<_>>();
        for (entrypoint, (_, entrypoint_artifact)) in
            metadata.entrypoints.iter().zip(&ordered_entrypoints)
        {
            typed_package.validate_entrypoint_layout(entrypoint, entrypoint_artifact)?;
        }
        let generated_source = render_pir_aot_package(
            self.name,
            &typed_package,
            &ordered_entrypoints,
            "typed DSLX AOT package",
        )?;

        fs::create_dir_all(out_dir).map_err(|error| CompilerError::Backend(error.to_string()))?;
        let rust_file = out_dir.join(format!("{package_name}_typed_dslx_pir_aot_package.rs"));
        fs::write(&rust_file, generated_source)
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        let object_file = out_dir.join(format!("{package_name}.pir_aot.o"));
        fs::write(&object_file, &package_artifact.object_code)
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        cc::Build::new()
            .cargo_metadata(true)
            .object(&object_file)
            .compile(&package_object_name);
        Ok(GeneratedTypedDslxAotPackage {
            rust_file,
            object_file,
            entrypoints: ordered_entrypoints
                .into_iter()
                .map(|(_, entrypoint_artifact)| entrypoint_artifact)
                .collect(),
        })
    }
}

/// Emits a generated AOT object and Rust wrapper into Cargo's `OUT_DIR`.
pub fn emit_aot_module_from_pir_text(
    spec: &AotBuildSpec<'_>,
) -> Result<GeneratedAotModule, CompilerError> {
    let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
        CompilerError::Backend("OUT_DIR is required while emitting an AOT module".into())
    })?;
    emit_aot_module_from_pir_text_with_out_dir(spec, Path::new(&out_dir))
}

/// Emits a generated AOT object and Rust wrapper into `out_dir`.
pub fn emit_aot_module_from_pir_text_with_out_dir(
    spec: &AotBuildSpec<'_>,
    out_dir: &Path,
) -> Result<GeneratedAotModule, CompilerError> {
    let base_name = sanitize_identifier(spec.name);
    if base_name.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "AOT build spec name must contain an identifier character".into(),
        ));
    }
    let package = Parser::new(spec.pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    let function = package.get_fn(spec.top).ok_or_else(|| {
        CompilerError::InvalidFunction(format!("PIR package has no function '{}'", spec.top))
    })?;
    let entrypoint_symbol = format!("__xlsynth_pir_aot_{base_name}");
    let artifact = compile_package_aot(&package, spec.top, &entrypoint_symbol)?;

    fs::create_dir_all(out_dir).map_err(|error| CompilerError::Backend(error.to_string()))?;
    let object_file = out_dir.join(format!("{base_name}.pir_aot.o"));
    let rust_file = out_dir.join(format!("{base_name}_pir_aot_wrapper.rs"));
    fs::write(&object_file, &artifact.object_code)
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    fs::write(&rust_file, render_generated_module(function, &artifact)?)
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

    cc::Build::new()
        .cargo_metadata(true)
        .object(&object_file)
        .compile(&format!("xlsynth_pir_aot_{base_name}"));

    Ok(GeneratedAotModule {
        rust_file,
        object_file,
        artifact,
    })
}

fn prepare_dslx_aot_entrypoint(
    spec: &TypedDslxAotBuildSpec<'_>,
    base_name: &str,
) -> Result<PreparedDslxAotEntrypoint, CompilerError> {
    for dependency in collect_typed_dslx_aot_dependencies(spec)
        .map_err(|error| CompilerError::Backend(error.to_string()))?
    {
        println!("cargo:rerun-if-changed={}", dependency.display());
    }
    let dslx_text = fs::read_to_string(spec.dslx_path)
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let pir_text = convert_dslx_to_ir_text(&dslx_text, spec.dslx_path, &spec.dslx_options)
        .map_err(|error| CompilerError::Backend(error.to_string()))?
        .ir;
    let calling_convention = if spec.dslx_options.force_implicit_token_calling_convention {
        DslxCallingConvention::ImplicitToken
    } else {
        DslxCallingConvention::Typical
    };
    let dslx_top = mangle_dslx_name_with_calling_convention(
        dslx_path_to_module_name(spec.dslx_path)
            .map_err(|error| CompilerError::Backend(error.to_string()))?,
        spec.top,
        calling_convention,
    )
    .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let optimized_pir_text = optimize_dslx_aot_pir_text(
        &pir_text,
        &dslx_top,
        spec.dslx_path.file_name().and_then(|name| name.to_str()),
    )?;
    let (pir_text, top) = if spec.dslx_options.force_implicit_token_calling_convention {
        let wrapper_top = typed_dslx_implicit_token_entrypoint_wrapper_top(base_name);
        (
            append_implicit_token_entrypoint_wrapper(&optimized_pir_text, &dslx_top, &wrapper_top)?,
            wrapper_top,
        )
    } else {
        (optimized_pir_text, dslx_top)
    };
    let package = Parser::new(&pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    Ok(PreparedDslxAotEntrypoint { package, top })
}

/// Runs the XLS IR optimizer before a DSLX-derived package is lowered by
/// Cranelift.
fn optimize_dslx_aot_pir_text(
    pir_text: &str,
    top: &str,
    filename: Option<&str>,
) -> Result<String, CompilerError> {
    let original_package = Parser::new(pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    let original_event_labels = original_package
        .members
        .iter()
        .flat_map(|member| match member {
            PackageMember::Function(function) => function.nodes.iter(),
            PackageMember::Block { func, .. } => func.nodes.iter(),
        })
        .filter_map(|node| match &node.payload {
            NodePayload::Assert { label, .. } | NodePayload::Cover { label, .. } => {
                Some(label.clone())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let ir_package = IrPackage::parse_ir(pir_text, filename)
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let optimized_pir_text = optimize_ir(&ir_package, top)
        .map(|package| package.to_string())
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let mut optimized_package = Parser::new(&optimized_pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    for member in &mut optimized_package.members {
        let function = match member {
            PackageMember::Function(function) => function,
            PackageMember::Block { func, .. } => func,
        };
        for node in &mut function.nodes {
            let label = match &mut node.payload {
                NodePayload::Assert { label, .. } | NodePayload::Cover { label, .. } => label,
                _ => continue,
            };
            if let Some(original) = original_event_labels
                .iter()
                .filter(|original| {
                    label.as_str() == original.as_str()
                        || label
                            .strip_suffix(original.as_str())
                            .is_some_and(|prefix| prefix.ends_with('_'))
                })
                .max_by_key(|original| original.len())
            {
                label.clone_from(original);
            }
        }
    }
    Ok(optimized_package.to_string())
}

/// Appends a user-signature wrapper around a DSLX implicit-token entrypoint.
fn append_implicit_token_entrypoint_wrapper(
    pir_text: &str,
    implicit_top: &str,
    wrapper_top: &str,
) -> Result<String, CompilerError> {
    let package = Parser::new(pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    let callee = package
        .members
        .iter()
        .find_map(|member| match member {
            PackageMember::Function(function) if function.name == implicit_top => Some(function),
            _ => None,
        })
        .ok_or_else(|| {
            CompilerError::InvalidFunction(format!(
                "implicit-token DSLX top `{implicit_top}` not found in converted PIR package"
            ))
        })?;
    if callee.params.len() < 2 {
        return Err(CompilerError::InvalidFunction(format!(
            "implicit-token DSLX top `{implicit_top}` has {} parameters, expected token, activation, and user parameters",
            callee.params.len()
        )));
    }
    if callee.params[0].ty != Type::Token {
        return Err(CompilerError::InvalidFunction(format!(
            "implicit-token DSLX top `{implicit_top}` first parameter must be token, got {}",
            callee.params[0].ty
        )));
    }
    if callee.params[1].ty != Type::Bits(1) {
        return Err(CompilerError::InvalidFunction(format!(
            "implicit-token DSLX top `{implicit_top}` second parameter must be bits[1] activation, got {}",
            callee.params[1].ty
        )));
    }
    let Type::Tuple(return_elements) = &callee.ret_ty else {
        return Err(CompilerError::InvalidFunction(format!(
            "implicit-token DSLX top `{implicit_top}` must return (token, value), got {}",
            callee.ret_ty
        )));
    };
    if return_elements.len() != 2 || return_elements[0].as_ref() != &Type::Token {
        return Err(CompilerError::InvalidFunction(format!(
            "implicit-token DSLX top `{implicit_top}` must return (token, value), got {}",
            callee.ret_ty
        )));
    }

    let user_params = &callee.params[2..];
    let user_return_type = return_elements[1].as_ref();
    let first_wrapper_id = package_max_text_id(&package) + 1;
    let params = user_params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            format!(
                "{}: {} id={}",
                param.name,
                param.ty,
                first_wrapper_id + index
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let user_args = user_params
        .iter()
        .map(|param| param.name.as_str())
        .collect::<Vec<_>>();
    let invoke_args = std::iter::once("__xlsynth_token")
        .chain(std::iter::once("__xlsynth_activated"))
        .chain(user_args.iter().copied())
        .collect::<Vec<_>>()
        .join(", ");
    let token_id = first_wrapper_id + user_params.len();
    let activated_id = first_wrapper_id + user_params.len() + 1;
    let invoke_id = first_wrapper_id + user_params.len() + 2;
    let return_id = first_wrapper_id + user_params.len() + 3;
    let wrapper = format!(
        r#"fn {wrapper_top}({params}) -> {user_return_type} {{
  __xlsynth_token: token = after_all(id={token_id})
  __xlsynth_activated: bits[1] = literal(value=1, id={activated_id})
  __xlsynth_result: {callee_return_type} = invoke({invoke_args}, to_apply={implicit_top}, id={invoke_id})
  ret __xlsynth_output: {user_return_type} = tuple_index(__xlsynth_result, index=1, id={return_id})
}}"#,
        callee_return_type = callee.ret_ty,
    );

    Ok(format!("{pir_text}\n\n{wrapper}\n"))
}

/// Returns the maximum package-wide node text ID used by emitted PIR members.
fn package_max_text_id(package: &xlsynth_pir::ir::Package) -> usize {
    package
        .members
        .iter()
        .flat_map(|member| match member {
            PackageMember::Function(function) => function.nodes.iter(),
            PackageMember::Block { func, .. } => func.nodes.iter(),
        })
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
}

struct ValidatedPirAotPackage<'a> {
    metadata: &'a PirAotPackageMetadata,
    decls: BTreeMap<PirAotDeclKey, &'a PirAotDecl>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PirAotDeclKey {
    module: Vec<String>,
    name: String,
}

impl<'a> ValidatedPirAotPackage<'a> {
    fn new(metadata: &'a PirAotPackageMetadata) -> Result<Self, CompilerError> {
        if metadata.format_version != PIR_AOT_METADATA_FORMAT_VERSION {
            return Err(CompilerError::InvalidArgument(format!(
                "unsupported PIR AOT metadata format_version {}; expected {}",
                metadata.format_version, PIR_AOT_METADATA_FORMAT_VERSION
            )));
        }
        let mut module_paths = BTreeSet::new();
        let mut decls = BTreeMap::new();
        for module in &metadata.modules {
            validate_module_path(&module.path)?;
            if !module_paths.insert(module.path.clone()) {
                return Err(CompilerError::InvalidArgument(format!(
                    "PIR AOT metadata contains duplicate module path `{}`",
                    render_module_path_for_error(&module.path)
                )));
            }
            let mut module_decl_names = BTreeSet::new();
            for decl in &module.declarations {
                let name = pir_aot_decl_name(decl);
                validate_type_name(name)?;
                if !module_decl_names.insert(name.to_string()) {
                    return Err(CompilerError::InvalidArgument(format!(
                        "PIR AOT metadata module `{}` contains duplicate declaration `{name}`",
                        render_module_path_for_error(&module.path)
                    )));
                }
                decls.insert(
                    PirAotDeclKey {
                        module: module.path.clone(),
                        name: name.to_string(),
                    },
                    decl,
                );
            }
        }
        let mut entrypoint_names = BTreeSet::new();
        for entrypoint in &metadata.entrypoints {
            validate_type_name(&entrypoint.name)?;
            match &entrypoint.source {
                PirAotEntrypointSource::IrFile { ir_file, ir_top } => {
                    validate_ir_file_path(ir_file)?;
                    validate_ir_top(ir_top)?;
                }
                PirAotEntrypointSource::GeneratedIr { ir_top } => {
                    validate_ir_top(ir_top)?;
                }
            }
            validate_module_path(&entrypoint.owning_module)?;
            if !module_paths.contains(&entrypoint.owning_module) {
                return Err(CompilerError::InvalidArgument(format!(
                    "PIR AOT metadata entrypoint `{}` uses unknown owning module `{}`",
                    entrypoint.name,
                    render_module_path_for_error(&entrypoint.owning_module)
                )));
            }
            if !entrypoint_names.insert(entrypoint.name.clone()) {
                return Err(CompilerError::InvalidArgument(format!(
                    "PIR AOT metadata contains duplicate entrypoint `{}`",
                    entrypoint.name
                )));
            }
        }
        let package = Self { metadata, decls };
        for module in &metadata.modules {
            for decl in &module.declarations {
                package.validate_decl(&module.path, decl)?;
            }
        }
        for entrypoint in &metadata.entrypoints {
            for param in &entrypoint.params {
                validate_value_name(&param.name)?;
                package.lower_type_to_layout(&param.ty, &mut Vec::new())?;
            }
            package.lower_type_to_layout(&entrypoint.return_type, &mut Vec::new())?;
        }
        Ok(package)
    }

    fn validate_entrypoint_layout(
        &self,
        entrypoint: &PirAotEntrypoint,
        artifact: &AotEntrypointArtifact,
    ) -> Result<(), CompilerError> {
        if entrypoint.params.len() != artifact.param_layouts.len() {
            return Err(CompilerError::InvalidFunction(format!(
                "PIR AOT metadata entrypoint `{}` parameter count {} does not match compiled PIR parameter count {}",
                entrypoint.name,
                entrypoint.params.len(),
                artifact.param_layouts.len()
            )));
        }
        for (index, (param, actual)) in entrypoint
            .params
            .iter()
            .zip(artifact.param_layouts.iter())
            .enumerate()
        {
            let expected = self.lower_type_to_layout(&param.ty, &mut Vec::new())?;
            if &expected != actual {
                return Err(CompilerError::InvalidFunction(format!(
                    "PIR AOT metadata layout mismatch for entrypoint `{}` parameter {index} `{}`: metadata has {expected:?}, compiled PIR has {actual:?}",
                    entrypoint.name, param.name
                )));
            }
        }
        let expected = self.lower_type_to_layout(&entrypoint.return_type, &mut Vec::new())?;
        if expected != artifact.result_layout {
            return Err(CompilerError::InvalidFunction(format!(
                "PIR AOT metadata layout mismatch for entrypoint `{}` return value: metadata has {expected:?}, compiled PIR has {:?}",
                entrypoint.name, artifact.result_layout
            )));
        }
        Ok(())
    }

    fn validate_decl(
        &self,
        module_path: &[String],
        decl: &PirAotDecl,
    ) -> Result<(), CompilerError> {
        match decl {
            PirAotDecl::Struct { fields, .. } => {
                let mut field_names = BTreeSet::new();
                for field in fields {
                    validate_value_name(&field.name)?;
                    if !field_names.insert(field.name.clone()) {
                        return Err(CompilerError::InvalidArgument(format!(
                            "PIR AOT metadata struct `{}` has duplicate field `{}`",
                            render_type_path_for_error(module_path, pir_aot_decl_name(decl)),
                            field.name
                        )));
                    }
                    self.lower_type_to_layout(&field.ty, &mut Vec::new())?;
                }
            }
            PirAotDecl::Enum {
                signedness,
                bit_count,
                variants,
                ..
            } => {
                if *bit_count == 0 {
                    return Err(CompilerError::InvalidArgument(format!(
                        "PIR AOT metadata enum `{}` has zero bit width",
                        render_type_path_for_error(module_path, pir_aot_decl_name(decl))
                    )));
                }
                if *bit_count > 64 {
                    return Err(CompilerError::UnsupportedType(format!(
                        "PIR AOT enum `{}` uses bits[{bit_count}], but enum values are currently limited to 64 bits",
                        render_type_path_for_error(module_path, pir_aot_decl_name(decl))
                    )));
                }
                validate_scalar_bit_count(*signedness, *bit_count)?;
                let mut variant_names = BTreeSet::new();
                for variant in variants {
                    validate_type_name(&variant.name)?;
                    if !variant_names.insert(variant.name.clone()) {
                        return Err(CompilerError::InvalidArgument(format!(
                            "PIR AOT metadata enum `{}` has duplicate variant `{}`",
                            render_type_path_for_error(module_path, pir_aot_decl_name(decl)),
                            variant.name
                        )));
                    }
                    if *bit_count < 64 && variant.value >= (1u64 << *bit_count) {
                        return Err(CompilerError::InvalidArgument(format!(
                            "PIR AOT metadata enum variant `{}::{}` value {} does not fit bits[{bit_count}]",
                            render_type_path_for_error(module_path, pir_aot_decl_name(decl)),
                            variant.name,
                            variant.value
                        )));
                    }
                }
            }
            PirAotDecl::Alias { target, .. } => {
                self.lower_type_to_layout(target, &mut Vec::new())?;
            }
        }
        Ok(())
    }

    fn lower_type_to_layout(
        &self,
        ty: &PirAotType,
        active_refs: &mut Vec<PirAotDeclKey>,
    ) -> Result<NativeValueLayout, CompilerError> {
        match ty {
            PirAotType::Bits { bit_count, .. } => Ok(native_bits_layout(*bit_count)),
            PirAotType::Token => Ok(NativeValueLayout::Token),
            PirAotType::Array { size, element } => {
                if *size == 0 {
                    return Err(CompilerError::InvalidArgument(
                        "PIR AOT metadata arrays must have nonzero size".into(),
                    ));
                }
                Ok(NativeValueLayout::Array {
                    element: Box::new(self.lower_type_to_layout(element, active_refs)?),
                    element_count: *size,
                })
            }
            PirAotType::Tuple { elements } => self.lower_fields_to_tuple_layout(
                elements
                    .iter()
                    .map(|element| self.lower_type_to_layout(element, active_refs))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            PirAotType::TypeRef { module, name } => {
                let key = PirAotDeclKey {
                    module: module.clone(),
                    name: name.clone(),
                };
                if active_refs.contains(&key) {
                    return Err(CompilerError::InvalidArgument(format!(
                        "PIR AOT metadata contains cyclic type reference through `{}`",
                        render_type_path_for_error(module, name)
                    )));
                }
                let decl = self.decls.get(&key).ok_or_else(|| {
                    CompilerError::InvalidArgument(format!(
                        "PIR AOT metadata references unknown type `{}`",
                        render_type_path_for_error(module, name)
                    ))
                })?;
                active_refs.push(key);
                let result = match decl {
                    PirAotDecl::Struct { fields, .. } => self.lower_fields_to_tuple_layout(
                        fields
                            .iter()
                            .map(|field| self.lower_type_to_layout(&field.ty, active_refs))
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    PirAotDecl::Enum { bit_count, .. } => Ok(native_bits_layout(*bit_count)),
                    PirAotDecl::Alias { target, .. } => {
                        self.lower_type_to_layout(target, active_refs)
                    }
                };
                active_refs.pop();
                result
            }
        }
    }

    fn lower_fields_to_tuple_layout(
        &self,
        field_layouts: Vec<NativeValueLayout>,
    ) -> Result<NativeValueLayout, CompilerError> {
        let mut fields = Vec::with_capacity(field_layouts.len());
        let mut byte_count = 0usize;
        let mut alignment = 1usize;
        for layout in field_layouts {
            byte_count = align_up_local(byte_count, layout.alignment())?;
            fields.push(NativeTupleFieldLayout {
                layout: Box::new(layout.clone()),
                offset: byte_count,
            });
            byte_count = byte_count.checked_add(layout.byte_count()).ok_or_else(|| {
                CompilerError::UnsupportedType("PIR AOT tuple layout size overflow".into())
            })?;
            alignment = alignment.max(layout.alignment());
        }
        byte_count = align_up_local(byte_count, alignment)?;
        Ok(NativeValueLayout::Tuple {
            fields,
            byte_count,
            alignment,
        })
    }
}

#[derive(Default)]
struct RenderModuleNode {
    items: Vec<String>,
    children: BTreeMap<String, RenderModuleNode>,
}

#[derive(Debug)]
struct GeneratedTupleType {
    name: String,
    elements: Vec<PirAotType>,
}

#[derive(Debug, Default)]
struct GeneratedTupleTypes {
    types: Vec<GeneratedTupleType>,
    reserved_root_names: BTreeSet<String>,
}

impl GeneratedTupleTypes {
    fn collect(package: &ValidatedPirAotPackage<'_>) -> Self {
        let mut registry = Self::with_reserved_root_names(package);
        for module in &package.metadata.modules {
            for decl in &module.declarations {
                registry.collect_from_decl(decl);
            }
        }
        for entrypoint in &package.metadata.entrypoints {
            for param in &entrypoint.params {
                registry.collect_from_type(&param.ty);
            }
            registry.collect_from_type(&entrypoint.return_type);
        }
        registry
    }

    fn with_reserved_root_names(package: &ValidatedPirAotPackage<'_>) -> Self {
        let mut reserved_root_names = BTreeSet::from(["Token".to_string()]);
        for module in &package.metadata.modules {
            if let Some(root_name) = module.path.first() {
                reserved_root_names.insert(root_name.clone());
            }
        }
        for (signedness, bit_count) in collect_scalar_aliases(package) {
            reserved_root_names.insert(scalar_alias_name(signedness, bit_count));
        }
        Self {
            types: Vec::new(),
            reserved_root_names,
        }
    }

    fn collect_from_decl(&mut self, decl: &PirAotDecl) {
        match decl {
            PirAotDecl::Struct { fields, .. } => {
                for field in fields {
                    self.collect_from_type(&field.ty);
                }
            }
            PirAotDecl::Enum { .. } => {}
            PirAotDecl::Alias { target, .. } => self.collect_from_type(target),
        }
    }

    fn collect_from_type(&mut self, ty: &PirAotType) {
        match ty {
            PirAotType::Bits { .. } | PirAotType::Token | PirAotType::TypeRef { .. } => {}
            PirAotType::Array { element, .. } => self.collect_from_type(element),
            PirAotType::Tuple { elements } => {
                for element in elements {
                    self.collect_from_type(element);
                }
                self.intern(elements);
            }
        }
    }

    fn intern(&mut self, elements: &[PirAotType]) {
        if self.type_name(elements).is_some() {
            return;
        }
        let name = self.allocate_name();
        self.types.push(GeneratedTupleType {
            name,
            elements: elements.to_vec(),
        });
    }

    fn allocate_name(&mut self) -> String {
        let mut index = self.types.len();
        loop {
            let candidate = format!("XlsynthPirAotTuple{index}");
            if self.reserved_root_names.insert(candidate.clone()) {
                return candidate;
            }
            index += 1;
        }
    }

    fn type_name(&self, elements: &[PirAotType]) -> Option<&str> {
        self.types
            .iter()
            .find(|tuple_type| tuple_type.elements == elements)
            .map(|tuple_type| tuple_type.name.as_str())
    }

    fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

fn render_pir_aot_package(
    package_name: &str,
    package: &ValidatedPirAotPackage<'_>,
    entrypoint_artifacts: &[(String, AotEntrypointArtifact)],
    source_kind: &str,
) -> Result<String, CompilerError> {
    if package.metadata.entrypoints.len() != entrypoint_artifacts.len() {
        return Err(CompilerError::InvalidArgument(format!(
            "PIR AOT metadata has {} entrypoints, but build has {} entrypoint artifacts",
            package.metadata.entrypoints.len(),
            entrypoint_artifacts.len()
        )));
    }
    let tuple_types = GeneratedTupleTypes::collect(package);
    let mut root = RenderModuleNode::default();
    for module in &package.metadata.modules {
        let mut items = vec![
            "#![allow(dead_code)]".to_string(),
            "#![allow(unused_imports)]".to_string(),
        ];
        for decl in &module.declarations {
            items.push(render_pir_aot_decl(
                package,
                &tuple_types,
                &module.path,
                decl,
            )?);
        }
        insert_render_module_items(&mut root, &module.path, items)?;
    }
    for (entrypoint, (_, entrypoint_artifact)) in package
        .metadata
        .entrypoints
        .iter()
        .zip(entrypoint_artifacts.iter())
    {
        let runner_module_name = format!("aot_{}", sanitize_identifier(&entrypoint.name));
        let mut runner_path = entrypoint.owning_module.clone();
        runner_path.push(runner_module_name);
        let param_names = entrypoint
            .params
            .iter()
            .map(|param| param.name.clone())
            .collect::<Vec<_>>();
        let param_types = entrypoint
            .params
            .iter()
            .map(|param| render_pir_aot_type(package, &tuple_types, &param.ty, &runner_path))
            .collect::<Result<Vec<_>, _>>()?;
        let output_type =
            render_pir_aot_type(package, &tuple_types, &entrypoint.return_type, &runner_path)?;
        insert_render_module_items(
            &mut root,
            &runner_path,
            vec![
                "#![allow(dead_code)]".to_string(),
                "#![allow(unused_imports)]".to_string(),
                "use super::*;".to_string(),
                render_runner_items(
                    entrypoint_artifact,
                    &param_names,
                    &param_types,
                    &output_type,
                    "",
                    false,
                ),
            ],
        )?;
    }

    Ok(trim_trailing_line_whitespace(format!(
        "// SPDX-License-Identifier: Apache-2.0\n// Generated by xlsynth_pir_compiler::aot from {source_kind} {package_name:?}.\n\n{}{}",
        render_pir_aot_runtime_items(package, &tuple_types)?,
        render_module_node_children(&root, 0),
    )))
}

fn render_pir_aot_runtime_items(
    package: &ValidatedPirAotPackage<'_>,
    tuple_types: &GeneratedTupleTypes,
) -> Result<String, CompilerError> {
    let mut output =
        "pub use xlsynth_pir_compiler_runtime::{AllZeros, ExecutionOptions, Token};\n".to_string();
    for (signedness, bit_count) in collect_scalar_aliases(package) {
        output.push_str(&format!(
            "pub type {} = {};\n",
            scalar_alias_name(signedness, bit_count),
            scalar_runtime_type(signedness, bit_count)
        ));
    }
    if !tuple_types.is_empty() {
        output.push('\n');
        for tuple_type in &tuple_types.types {
            output.push_str(&render_generated_tuple_type(
                package,
                tuple_types,
                tuple_type,
            )?);
            output.push('\n');
        }
    }
    output.push('\n');
    Ok(output)
}

fn render_generated_tuple_type(
    package: &ValidatedPirAotPackage<'_>,
    tuple_types: &GeneratedTupleTypes,
    tuple_type: &GeneratedTupleType,
) -> Result<String, CompilerError> {
    let fields = tuple_type
        .elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            Ok(format!(
                "    pub field{index}: {},\n",
                render_pir_aot_type(package, tuple_types, element, &[])?
            ))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?
        .concat();
    let all_zeros = tuple_type
        .elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            Ok(format!(
                "            field{index}: {},\n",
                render_pir_aot_all_zeros_expr(package, tuple_types, element, &[])?
            ))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?
        .concat();
    Ok(format!(
        "#[repr(C)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{\n{fields}}}\nimpl {name} {{\n    /// Constructs a value whose DSLX/PIR data bits are all zero.\n    pub fn all_zeros() -> Self {{\n        Self {{\n{all_zeros}        }}\n    }}\n}}\nimpl AllZeros for {name} {{\n    fn all_zeros() -> Self {{\n        {name}::all_zeros()\n    }}\n}}\n",
        name = tuple_type.name,
    ))
}

fn insert_render_module_items(
    root: &mut RenderModuleNode,
    path: &[String],
    items: Vec<String>,
) -> Result<(), CompilerError> {
    let mut node = root;
    for segment in path {
        node = node.children.entry(segment.clone()).or_default();
    }
    if !node.items.is_empty() {
        return Err(CompilerError::InvalidArgument(format!(
            "duplicate generated Rust module `{}`",
            render_module_path_for_error(path)
        )));
    }
    node.items = items;
    Ok(())
}

fn render_module_node_children(node: &RenderModuleNode, indent: usize) -> String {
    let mut output = String::new();
    for (name, child) in &node.children {
        output.push_str(&render_module_node(name, child, indent));
    }
    output
}

fn render_module_node(name: &str, node: &RenderModuleNode, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let child_pad = " ".repeat(indent + 4);
    let mut output = format!("{pad}pub mod {name} {{\n");
    for item in &node.items {
        for line in item.trim_end().lines() {
            output.push_str(&child_pad);
            output.push_str(line);
            output.push('\n');
        }
        output.push('\n');
    }
    output.push_str(&render_module_node_children(node, indent + 4));
    output.push_str(&format!("{pad}}}\n"));
    output
}

fn trim_trailing_line_whitespace(text: String) -> String {
    let mut output = String::with_capacity(text.len());
    for line in text.lines() {
        output.push_str(line.trim_end());
        output.push('\n');
    }
    output
}

fn collect_scalar_aliases(
    package: &ValidatedPirAotPackage<'_>,
) -> BTreeSet<(PirAotSignedness, usize)> {
    let mut aliases = BTreeSet::new();
    for module in &package.metadata.modules {
        for decl in &module.declarations {
            collect_scalar_aliases_from_decl(decl, &mut aliases);
        }
    }
    for entrypoint in &package.metadata.entrypoints {
        for param in &entrypoint.params {
            collect_scalar_aliases_from_type(&param.ty, &mut aliases);
        }
        collect_scalar_aliases_from_type(&entrypoint.return_type, &mut aliases);
    }
    aliases
}

fn collect_scalar_aliases_from_decl(
    decl: &PirAotDecl,
    aliases: &mut BTreeSet<(PirAotSignedness, usize)>,
) {
    match decl {
        PirAotDecl::Struct { fields, .. } => {
            for field in fields {
                collect_scalar_aliases_from_type(&field.ty, aliases);
            }
        }
        PirAotDecl::Enum {
            signedness,
            bit_count,
            ..
        } => {
            aliases.insert((*signedness, *bit_count));
        }
        PirAotDecl::Alias { target, .. } => {
            collect_scalar_aliases_from_type(target, aliases);
        }
    }
}

fn collect_scalar_aliases_from_type(
    ty: &PirAotType,
    aliases: &mut BTreeSet<(PirAotSignedness, usize)>,
) {
    match ty {
        PirAotType::Bits {
            signedness,
            bit_count,
        } => {
            aliases.insert((*signedness, *bit_count));
        }
        PirAotType::Token | PirAotType::TypeRef { .. } => {}
        PirAotType::Array { element, .. } => collect_scalar_aliases_from_type(element, aliases),
        PirAotType::Tuple { elements } => {
            for element in elements {
                collect_scalar_aliases_from_type(element, aliases);
            }
        }
    }
}

fn scalar_alias_name(signedness: PirAotSignedness, bit_count: usize) -> String {
    match signedness {
        PirAotSignedness::Unsigned => format!("U{bit_count}"),
        PirAotSignedness::Signed => format!("S{bit_count}"),
    }
}

fn scalar_runtime_type(signedness: PirAotSignedness, bit_count: usize) -> String {
    let (unsigned_prefix, signed_prefix) = match bit_count {
        0 => {
            return match signedness {
                PirAotSignedness::Unsigned => {
                    "xlsynth_pir_compiler_runtime::UnsignedBits0".to_string()
                }
                PirAotSignedness::Signed => "xlsynth_pir_compiler_runtime::SignedBits0".to_string(),
            };
        }
        1..=8 => ("UnsignedBitsInU8", "SignedBitsInU8"),
        9..=16 => ("UnsignedBitsInU16", "SignedBitsInU16"),
        17..=32 => ("UnsignedBitsInU32", "SignedBitsInU32"),
        33..=64 => ("UnsignedBitsInU64", "SignedBitsInU64"),
        _ => {
            let limb_count = bit_count.div_ceil(64);
            return match signedness {
                PirAotSignedness::Unsigned => {
                    format!(
                        "xlsynth_pir_compiler_runtime::UnsignedWideBits<{bit_count}, {limb_count}>"
                    )
                }
                PirAotSignedness::Signed => {
                    format!(
                        "xlsynth_pir_compiler_runtime::SignedWideBits<{bit_count}, {limb_count}>"
                    )
                }
            };
        }
    };
    let type_name = match signedness {
        PirAotSignedness::Unsigned => unsigned_prefix,
        PirAotSignedness::Signed => signed_prefix,
    };
    format!("xlsynth_pir_compiler_runtime::{type_name}<{bit_count}>")
}

fn render_scalar_alias_type(
    current_module_path: &[String],
    signedness: PirAotSignedness,
    bit_count: usize,
) -> String {
    render_root_item_path(
        current_module_path,
        &scalar_alias_name(signedness, bit_count),
    )
}

fn render_root_item_path(current_module_path: &[String], name: &str) -> String {
    if current_module_path.is_empty() {
        return name.to_string();
    }
    let mut segments = vec!["super".to_string(); current_module_path.len()];
    segments.push(name.to_string());
    segments.join("::")
}

fn render_pir_aot_decl(
    package: &ValidatedPirAotPackage<'_>,
    tuple_types: &GeneratedTupleTypes,
    module_path: &[String],
    decl: &PirAotDecl,
) -> Result<String, CompilerError> {
    match decl {
        PirAotDecl::Struct { name, fields } => {
            let all_zeros_trait = render_root_item_path(module_path, "AllZeros");
            let rendered_fields = fields
                .iter()
                .map(|field| {
                    Ok(format!(
                        "    pub {}: {},\n",
                        sanitize_identifier(&field.name),
                        render_pir_aot_type(package, tuple_types, &field.ty, module_path)?
                    ))
                })
                .collect::<Result<Vec<_>, CompilerError>>()?
                .concat();
            let all_zeros = fields
                .iter()
                .map(|field| {
                    Ok(format!(
                        "            {}: {},\n",
                        sanitize_identifier(&field.name),
                        render_pir_aot_all_zeros_expr(
                            package,
                            tuple_types,
                            &field.ty,
                            module_path,
                        )?
                    ))
                })
                .collect::<Result<Vec<_>, CompilerError>>()?
                .concat();
            Ok(format!(
                "#[repr(C)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{\n{rendered_fields}}}\nimpl {name} {{\n    /// Constructs a value whose DSLX/PIR data bits are all zero.\n    pub fn all_zeros() -> Self {{\n        Self {{\n{all_zeros}        }}\n    }}\n}}\nimpl {all_zeros_trait} for {name} {{\n    fn all_zeros() -> Self {{\n        {name}::all_zeros()\n    }}\n}}\n"
            ))
        }
        PirAotDecl::Enum {
            name,
            signedness,
            bit_count,
            variants,
        } => {
            let bits_type = render_scalar_alias_type(module_path, *signedness, *bit_count);
            let all_zeros_trait = render_root_item_path(module_path, "AllZeros");
            let constants = variants
                .iter()
                .map(|variant| {
                    format!(
                        "    #[allow(non_upper_case_globals)]\n    pub const {}: Self = Self({bits_type}::from_raw_bits({}));\n",
                        sanitize_identifier(&variant.name),
                        variant.value
                    )
                })
                .collect::<Vec<_>>()
                .concat();
            Ok(format!(
                "#[repr(transparent)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name}(pub {bits_type});\nimpl {name} {{\n    /// Constructs a value whose DSLX/PIR data bits are all zero.\n    pub fn all_zeros() -> Self {{\n        Self(<{bits_type} as {all_zeros_trait}>::all_zeros())\n    }}\n{constants}}}\nimpl {all_zeros_trait} for {name} {{\n    fn all_zeros() -> Self {{\n        {name}::all_zeros()\n    }}\n}}\n"
            ))
        }
        PirAotDecl::Alias { name, target } => Ok(format!(
            "pub type {name} = {};\n",
            render_pir_aot_type(package, tuple_types, target, module_path)?
        )),
    }
}

fn render_pir_aot_type(
    package: &ValidatedPirAotPackage<'_>,
    tuple_types: &GeneratedTupleTypes,
    ty: &PirAotType,
    current_module_path: &[String],
) -> Result<String, CompilerError> {
    match ty {
        PirAotType::Bits {
            signedness,
            bit_count,
        } => Ok(render_scalar_alias_type(
            current_module_path,
            *signedness,
            *bit_count,
        )),
        PirAotType::Token => Ok(render_root_item_path(current_module_path, "Token")),
        PirAotType::Array { size, element } => Ok(format!(
            "[{}; {size}]",
            render_pir_aot_type(package, tuple_types, element, current_module_path)?
        )),
        PirAotType::Tuple { elements } => tuple_types
            .type_name(elements)
            .map(|name| render_root_item_path(current_module_path, name))
            .ok_or_else(|| {
                CompilerError::InvalidArgument(
                    "PIR AOT tuple type was not registered for rendering".into(),
                )
            }),
        PirAotType::TypeRef { module, name } => {
            let key = PirAotDeclKey {
                module: module.clone(),
                name: name.clone(),
            };
            package.decls.get(&key).ok_or_else(|| {
                CompilerError::InvalidArgument(format!(
                    "PIR AOT metadata references unknown type `{}`",
                    render_type_path_for_error(module, name)
                ))
            })?;
            Ok(render_relative_type_path(current_module_path, module, name))
        }
    }
}

fn render_pir_aot_all_zeros_expr(
    package: &ValidatedPirAotPackage<'_>,
    tuple_types: &GeneratedTupleTypes,
    ty: &PirAotType,
    current_module_path: &[String],
) -> Result<String, CompilerError> {
    Ok(format!(
        "<{} as {}>::all_zeros()",
        render_pir_aot_type(package, tuple_types, ty, current_module_path)?,
        render_root_item_path(current_module_path, "AllZeros"),
    ))
}

fn render_relative_type_path(current: &[String], target_module: &[String], name: &str) -> String {
    let common_len = current
        .iter()
        .zip(target_module.iter())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count();
    if common_len == current.len() && common_len == target_module.len() {
        return name.to_string();
    }
    let mut segments = Vec::new();
    for _ in common_len..current.len() {
        segments.push("super".to_string());
    }
    segments.extend(target_module.iter().skip(common_len).cloned());
    segments.push(name.to_string());
    segments.join("::")
}

fn native_bits_layout(bit_count: usize) -> NativeValueLayout {
    if bit_count <= 64 {
        let byte_count = match bit_count {
            0 => 0,
            1..=8 => 1,
            9..=16 => 2,
            17..=32 => 4,
            33..=64 => 8,
            _ => unreachable!(),
        };
        NativeValueLayout::Scalar(ScalarLayout {
            bit_count,
            byte_count,
        })
    } else {
        NativeValueLayout::WideBits(WideBitsLayout {
            bit_count,
            limb_count: bit_count.div_ceil(64),
        })
    }
}

fn validate_scalar_bit_count(
    signedness: PirAotSignedness,
    bit_count: usize,
) -> Result<(), CompilerError> {
    if bit_count == 0 {
        let prefix = match signedness {
            PirAotSignedness::Unsigned => "u",
            PirAotSignedness::Signed => "s",
        };
        return Err(CompilerError::InvalidArgument(format!(
            "PIR AOT metadata {prefix}{bit_count} type must have nonzero width"
        )));
    }
    Ok(())
}

fn align_up_local(value: usize, alignment: usize) -> Result<usize, CompilerError> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| CompilerError::UnsupportedType("PIR AOT layout size overflow".into()))
}

fn pir_aot_decl_name(decl: &PirAotDecl) -> &str {
    match decl {
        PirAotDecl::Struct { name, .. }
        | PirAotDecl::Enum { name, .. }
        | PirAotDecl::Alias { name, .. } => name,
    }
}

fn validate_module_path(path: &[String]) -> Result<(), CompilerError> {
    if path.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "PIR AOT metadata module path must not be empty".into(),
        ));
    }
    for segment in path {
        validate_type_name(segment)?;
    }
    Ok(())
}

fn validate_type_name(name: &str) -> Result<(), CompilerError> {
    validate_identifier(name, "type or module name")
}

fn validate_value_name(name: &str) -> Result<(), CompilerError> {
    validate_identifier(name, "value name")
}

fn validate_ir_file_path(path: &str) -> Result<(), CompilerError> {
    if path.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "PIR AOT metadata ir_file must not be empty".into(),
        ));
    }
    let path = Path::new(path);
    if path.is_absolute() {
        return Err(CompilerError::InvalidArgument(
            "PIR AOT metadata ir_file must be relative".into(),
        ));
    }
    for component in path.components() {
        match component {
            std::path::Component::Normal(_) => {}
            _ => {
                return Err(CompilerError::InvalidArgument(
                    "PIR AOT metadata ir_file must be a relative path without `..`".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_ir_top(name: &str) -> Result<(), CompilerError> {
    if name.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "PIR AOT metadata ir_top must not be empty".into(),
        ));
    }
    Ok(())
}

fn validate_identifier(name: &str, label: &str) -> Result<(), CompilerError> {
    if name.is_empty() {
        return Err(CompilerError::InvalidArgument(format!(
            "PIR AOT metadata {label} must not be empty"
        )));
    }
    if sanitize_identifier(name) != name {
        return Err(CompilerError::InvalidArgument(format!(
            "PIR AOT metadata {label} `{name}` is not a supported Rust identifier"
        )));
    }
    Ok(())
}

fn render_module_path_for_error(path: &[String]) -> String {
    path.join("::")
}

fn render_type_path_for_error(module: &[String], name: &str) -> String {
    if module.is_empty() {
        name.to_string()
    } else {
        format!("{}::{name}", module.join("::"))
    }
}

fn sanitize_identifier(name: &str) -> String {
    let mut output = String::new();
    for (index, ch) in name.chars().enumerate() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if index == 0 && ch.is_ascii_digit() {
                output.push('_');
            }
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    output
}

fn render_generated_module(
    function: &xlsynth_pir::ir::Fn,
    artifact: &AotArtifact,
) -> Result<String, CompilerError> {
    let entrypoint_artifact = AotEntrypointArtifact::from(artifact);
    let mut declarations = Vec::new();
    let mut input_type_names = Vec::new();
    for (index, param) in function.params.iter().enumerate() {
        let name = format!("Input{index}");
        let value_type = render_value_type(&param.ty, &name, &mut declarations)?;
        input_type_names.push(value_type);
    }
    let output_type = render_value_type(&function.ret_ty, "Output", &mut declarations)?;
    let type_declarations = if declarations.is_empty() {
        String::new()
    } else {
        format!("{}\n", declarations.concat())
    };
    let param_names = function
        .params
        .iter()
        .map(|param| param.name.clone())
        .collect::<Vec<_>>();
    Ok(format!(
        "// SPDX-License-Identifier: Apache-2.0\n{}",
        render_runner_items(
            &entrypoint_artifact,
            &param_names,
            &input_type_names,
            &output_type,
            &type_declarations,
            true,
        )
    ))
}

fn render_runner_items(
    artifact: &AotEntrypointArtifact,
    param_names: &[String],
    input_type_names: &[String],
    output_type_name: &str,
    type_declarations: &str,
    emit_native_value_types: bool,
) -> String {
    let metadata = render_function_metadata(&artifact.metadata.event_sites);
    let params = param_names
        .iter()
        .zip(input_type_names.iter())
        .map(|(param, ty)| format!("{}: &{ty}", sanitize_identifier(param)))
        .collect::<Vec<_>>()
        .join(", ");
    let args = param_names
        .iter()
        .map(|param| sanitize_identifier(param))
        .collect::<Vec<_>>();
    let pointer_entries = args
        .iter()
        .map(|arg| format!("std::ptr::from_ref({arg}).cast::<u8>()"))
        .collect::<Vec<_>>()
        .join(", ");
    let signature_inputs = if params.is_empty() {
        String::new()
    } else {
        format!(", {params}")
    };
    let public_runtime_imports = if emit_native_value_types {
        r#"pub use xlsynth_pir_compiler_runtime::{
    AllZeros, Bits0, BitsInU8, BitsInU16, BitsInU32, BitsInU64, ExecutionOptions,
    ExecutionResult, RunError, Token, WideBits,
};
"#
    } else {
        "pub use xlsynth_pir_compiler_runtime::{ExecutionOptions, ExecutionResult, RunError};\n"
    };

    format!(
        r#"// Generated by xlsynth_pir_compiler::aot.

{public_runtime_imports}use std::sync::LazyLock;
#[allow(unused_imports)]
use xlsynth_pir_compiler_runtime::{{
    AssumptionFailureKind, CompiledFunctionMetadata, EventKind, EventSiteMetadata,
    ExecutionContext, RawExecutionContext, TraceTupleFieldLayout, TraceValueLayout,
    xlsynth_pir_record_assert, xlsynth_pir_record_assumption_failure,
    xlsynth_pir_record_cover, xlsynth_pir_record_trace,
    xlsynth_pir_runtime_wide_binop, xlsynth_pir_runtime_wide_bit_slice_update,
    xlsynth_pir_runtime_wide_dynamic_bit_slice, xlsynth_pir_runtime_wide_mulp,
    xlsynth_pir_runtime_wide_unary_op,
}};

unsafe extern "C" {{
    #[link_name = "{symbol}"]
    fn linked_entrypoint(
        inputs: *const *const u8,
        output: *mut u8,
        scratch: *mut u8,
        context: *mut RawExecutionContext,
    ) -> i32;
}}

const SCRATCH_BYTE_COUNT: usize = {scratch_bytes};
const SCRATCH_ALIGNMENT: usize = {scratch_alignment};
static FUNCTION_METADATA: LazyLock<CompiledFunctionMetadata> =
    LazyLock::new(|| {metadata});

{type_declarations}fn ensure_runtime_symbols_linked() {{
    std::hint::black_box([
        xlsynth_pir_record_assert as *const () as usize,
        xlsynth_pir_record_assumption_failure as *const () as usize,
        xlsynth_pir_record_cover as *const () as usize,
        xlsynth_pir_record_trace as *const () as usize,
        xlsynth_pir_runtime_wide_binop as *const () as usize,
        xlsynth_pir_runtime_wide_dynamic_bit_slice as *const () as usize,
        xlsynth_pir_runtime_wide_bit_slice_update as *const () as usize,
        xlsynth_pir_runtime_wide_unary_op as *const () as usize,
        xlsynth_pir_runtime_wide_mulp as *const () as usize,
    ]);
}}

/// Reusable runner for the generated native AOT entrypoint.
pub struct Runner {{
    scratch: Vec<u64>,
    context: ExecutionContext<'static>,
}}

impl Runner {{
    /// Creates a runner with reusable scratch storage and an event collector.
    pub fn new() -> Result<Self, RunError> {{
        ensure_runtime_symbols_linked();
        if SCRATCH_ALIGNMENT > std::mem::align_of::<u64>() {{
            return Err(RunError(format!(
                "unsupported AOT scratch alignment: {{SCRATCH_ALIGNMENT}}"
            )));
        }}
        Ok(Self {{
            scratch: vec![0u64; SCRATCH_BYTE_COUNT.div_ceil(std::mem::size_of::<u64>())],
            context: ExecutionContext::new(&*FUNCTION_METADATA),
        }})
    }}

    fn invoke(
        &mut self,
        inputs: &[*const u8],
        output: *mut u8,
        options: ExecutionOptions,
    ) -> Result<(), RunError> {{
        self.context.clear_with_options(options);
        let scratch = if self.scratch.is_empty() {{
            std::ptr::null_mut()
        }} else {{
            self.scratch.as_mut_ptr().cast::<u8>()
        }};
        let mut raw_context = self.context.raw_context();
        let status = unsafe {{
            linked_entrypoint(inputs.as_ptr(), output, scratch, &mut raw_context)
        }};
        if status == 0 {{
            Ok(())
        }} else {{
            Err(RunError(format!("compiled AOT entrypoint returned status {{status}}")))
        }}
    }}

    fn reject_failures(context: &ExecutionContext<'_>) -> Result<(), RunError> {{
        if let Some(failure) = context.assertion_failures().first() {{
            return Err(RunError(format!(
                "compiled assertion failed at node {{}}: {{}}",
                failure.node_text_id, failure.message
            )));
        }}
        if let Some(failure) = context.assumption_failures().first() {{
            return Err(RunError(format!(
                "compiled assumed-in-bounds condition failed at node {{}}: {{:?}}",
                failure.node_text_id, failure.kind
            )));
        }}
        Ok(())
    }}

    /// Runs into caller-owned output storage and returns observable events.
    pub fn run_with_events(&mut self{signature_inputs}, output: &mut {output_type_name}, options: ExecutionOptions) -> Result<ExecutionResult, RunError> {{
        let input_pointers = [{pointer_entries}];
        self.invoke(&input_pointers, std::ptr::from_mut(output).cast::<u8>(), options)?;
        Ok(self.context.result())
    }}

    /// Runs into caller-owned output storage, rejecting assertion/assumption failures.
    pub fn run(&mut self{signature_inputs}, output: &mut {output_type_name}) -> Result<(), RunError> {{
        let input_pointers = [{pointer_entries}];
        self.invoke(
            &input_pointers,
            std::ptr::from_mut(output).cast::<u8>(),
            ExecutionOptions::NO_EVENTS,
        )?;
        Self::reject_failures(&self.context)
    }}
}}

/// Creates a reusable runner for this native AOT entrypoint.
pub fn new_runner() -> Result<Runner, RunError> {{
    Runner::new()
}}

std::thread_local! {{
    static THREAD_LOCAL_RUNNER: std::cell::RefCell<Option<Runner>> =
        const {{ std::cell::RefCell::new(None) }};
}}

/// Runs a closure with this thread's cached runner.
pub fn with_thread_local_runner<T>(
    f: impl FnOnce(&mut Runner) -> T,
) -> Result<T, RunError> {{
    THREAD_LOCAL_RUNNER.with(|runner_cell| {{
        let mut runner_slot = runner_cell.borrow_mut();
        if runner_slot.is_none() {{
            runner_slot.replace(new_runner()?);
        }}
        Ok(f(runner_slot.as_mut().expect("runner was initialized")))
    }})
}}

/// Runs a fallible closure with this thread's cached runner.
pub fn try_with_thread_local_runner<T>(
    f: impl FnOnce(&mut Runner) -> Result<T, RunError>,
) -> Result<T, RunError> {{
    with_thread_local_runner(f)?
}}
"#,
        symbol = artifact.entrypoint_symbol,
        scratch_bytes = artifact.scratch_byte_count,
        scratch_alignment = artifact.scratch_alignment,
        metadata = metadata,
        type_declarations = type_declarations,
        public_runtime_imports = public_runtime_imports,
        output_type_name = output_type_name,
        signature_inputs = signature_inputs,
        pointer_entries = pointer_entries,
    )
}

fn render_value_type(
    ty: &Type,
    name: &str,
    declarations: &mut Vec<String>,
) -> Result<String, CompilerError> {
    match ty {
        Type::Token => Ok("Token".into()),
        Type::Bits(width) => Ok(render_native_bits_type(*width)),
        Type::Array(array) => {
            let element_name = format!("{name}Element");
            let element =
                render_value_type(array.element_type.as_ref(), &element_name, declarations)?;
            Ok(format!("[{element}; {}]", array.element_count))
        }
        Type::Tuple(fields) => {
            let mut rendered_fields = Vec::new();
            let mut all_zeros = Vec::new();
            for (index, field) in fields.iter().enumerate() {
                let field_name = format!("{name}Field{index}");
                let rendered = render_value_type(field, &field_name, declarations)?;
                rendered_fields.push(format!("    pub field{index}: {rendered},\n"));
                all_zeros.push(format!(
                    "            field{index}: <{rendered} as AllZeros>::all_zeros(),\n"
                ));
            }
            declarations.push(format!(
                "#[repr(C)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{\n{fields}}}\nimpl {name} {{\n    /// Constructs a value whose DSLX/PIR data bits are all zero.\n    pub fn all_zeros() -> Self {{\n        Self {{\n{all_zeros}        }}\n    }}\n}}\nimpl AllZeros for {name} {{\n    fn all_zeros() -> Self {{\n        {name}::all_zeros()\n    }}\n}}\n",
                fields = rendered_fields.concat(),
                all_zeros = all_zeros.concat()
            ));
            Ok(name.to_string())
        }
    }
}

fn render_native_bits_type(width: usize) -> String {
    match width {
        0 => "Bits0".to_string(),
        1..=8 => format!("BitsInU8<{width}>"),
        9..=16 => format!("BitsInU16<{width}>"),
        17..=32 => format!("BitsInU32<{width}>"),
        33..=64 => format!("BitsInU64<{width}>"),
        _ => format!("WideBits<{width}, {}>", width.div_ceil(64)),
    }
}

fn render_function_metadata(sites: &[EventSiteMetadata]) -> String {
    let sites = sites
        .iter()
        .map(render_event_site)
        .collect::<Vec<_>>()
        .join(", ");
    format!("CompiledFunctionMetadata {{ event_sites: vec![{sites}] }}")
}

fn render_event_site(site: &EventSiteMetadata) -> String {
    format!(
        "EventSiteMetadata {{ node_text_id: {}, kind: {}, label: {}, message: {}, format: {}, verbosity: {}, operand_layouts: vec![{}] }}",
        site.node_text_id,
        render_event_kind(site.kind),
        render_optional_string(&site.label),
        render_optional_string(&site.message),
        render_optional_string(&site.format),
        site.verbosity,
        site.operand_layouts
            .iter()
            .map(render_trace_layout)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_event_kind(kind: EventKind) -> String {
    match kind {
        EventKind::Assert => "EventKind::Assert".into(),
        EventKind::Cover => "EventKind::Cover".into(),
        EventKind::Trace => "EventKind::Trace".into(),
        EventKind::Assumption(kind) => format!(
            "EventKind::Assumption({})",
            match kind {
                AssumptionFailureKind::ArrayIndexOutOfBounds => {
                    "AssumptionFailureKind::ArrayIndexOutOfBounds"
                }
                AssumptionFailureKind::ArrayUpdateOutOfBounds => {
                    "AssumptionFailureKind::ArrayUpdateOutOfBounds"
                }
            }
        ),
    }
}

fn render_optional_string(value: &Option<String>) -> String {
    value
        .as_ref()
        .map(|value| format!("Some({value:?}.to_string())"))
        .unwrap_or_else(|| "None".into())
}

fn render_trace_layout(layout: &TraceValueLayout) -> String {
    match layout {
        TraceValueLayout::Bits {
            bit_count,
            byte_count,
        } => {
            format!("TraceValueLayout::Bits {{ bit_count: {bit_count}, byte_count: {byte_count} }}")
        }
        TraceValueLayout::WideBits {
            bit_count,
            limb_count,
        } => format!(
            "TraceValueLayout::WideBits {{ bit_count: {bit_count}, limb_count: {limb_count} }}"
        ),
        TraceValueLayout::Array {
            element,
            element_count,
        } => format!(
            "TraceValueLayout::Array {{ element: Box::new({}), element_count: {element_count} }}",
            render_trace_layout(element)
        ),
        TraceValueLayout::Tuple { fields, byte_count } => format!(
            "TraceValueLayout::Tuple {{ fields: vec![{}], byte_count: {byte_count} }}",
            fields
                .iter()
                .map(render_trace_tuple_field)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TraceValueLayout::Token => "TraceValueLayout::Token".into(),
    }
}

fn render_trace_tuple_field(field: &TraceTupleFieldLayout) -> String {
    format!(
        "TraceTupleFieldLayout {{ layout: Box::new({}), offset: {} }}",
        render_trace_layout(field.layout.as_ref()),
        field.offset
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pir_aot_metadata_rejects_wide_enum_values() {
        let metadata = PirAotPackageMetadata {
            format_version: PIR_AOT_METADATA_FORMAT_VERSION,
            modules: vec![PirAotModule {
                path: vec!["m".to_string()],
                declarations: vec![PirAotDecl::Enum {
                    name: "WideEnum".to_string(),
                    signedness: PirAotSignedness::Unsigned,
                    bit_count: 65,
                    variants: vec![PirAotEnumVariant {
                        name: "Zero".to_string(),
                        value: 0,
                    }],
                }],
            }],
            entrypoints: vec![],
        };
        let error = match ValidatedPirAotPackage::new(&metadata) {
            Ok(_) => panic!("wide enum metadata should be rejected"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("enum values are currently limited to 64 bits"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn pir_aot_tuple_metadata_renders_repr_c_tuple_wrappers() {
        let u8_ty = PirAotType::Bits {
            signedness: PirAotSignedness::Unsigned,
            bit_count: 8,
        };
        let u16_ty = PirAotType::Bits {
            signedness: PirAotSignedness::Unsigned,
            bit_count: 16,
        };
        let pair_ty = PirAotType::Tuple {
            elements: vec![u8_ty.clone(), u16_ty.clone()],
        };
        let nested_ty = PirAotType::Tuple {
            elements: vec![pair_ty.clone(), u8_ty.clone()],
        };
        let metadata = PirAotPackageMetadata {
            format_version: PIR_AOT_METADATA_FORMAT_VERSION,
            modules: vec![PirAotModule {
                path: vec!["m".to_string()],
                declarations: vec![
                    PirAotDecl::Struct {
                        name: "Holder".to_string(),
                        fields: vec![PirAotField {
                            name: "pair".to_string(),
                            ty: pair_ty.clone(),
                        }],
                    },
                    PirAotDecl::Alias {
                        name: "PairAlias".to_string(),
                        target: pair_ty.clone(),
                    },
                    PirAotDecl::Alias {
                        name: "NestedAlias".to_string(),
                        target: nested_ty.clone(),
                    },
                ],
            }],
            entrypoints: vec![PirAotEntrypoint {
                name: "entry".to_string(),
                source: PirAotEntrypointSource::GeneratedIr {
                    ir_top: "entry".to_string(),
                },
                owning_module: vec!["m".to_string()],
                params: vec![PirAotParam {
                    name: "pair".to_string(),
                    ty: pair_ty,
                }],
                return_type: PirAotType::Array {
                    size: 2,
                    element: Box::new(nested_ty),
                },
            }],
        };
        let package = ValidatedPirAotPackage::new(&metadata).expect("metadata should validate");
        let artifact = AotEntrypointArtifact {
            entrypoint_symbol: "__xlsynth_pir_aot_entry".to_string(),
            param_layouts: Vec::new(),
            result_layout: NativeValueLayout::Token,
            metadata: Default::default(),
            scratch_byte_count: 0,
            scratch_alignment: 1,
        };
        let rendered = render_pir_aot_package(
            "tuple_test",
            &package,
            &[("entry".to_string(), artifact)],
            "test package",
        )
        .expect("package should render");

        assert!(rendered.contains(
            "#[repr(C)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct XlsynthPirAotTuple0"
        ));
        assert!(rendered.contains("pub field0: U8,"));
        assert!(rendered.contains("pub field1: U16,"));
        assert!(rendered.contains("impl XlsynthPirAotTuple0"));
        assert!(rendered.contains("impl AllZeros for XlsynthPirAotTuple0"));
        assert!(rendered.contains("field0: <U8 as AllZeros>::all_zeros(),"));
        assert!(rendered.contains("field1: <U16 as AllZeros>::all_zeros(),"));
        assert!(rendered.contains(
            "#[repr(C)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct XlsynthPirAotTuple1"
        ));
        assert!(rendered.contains("pub field0: XlsynthPirAotTuple0,"));
        assert!(rendered.contains("impl AllZeros for XlsynthPirAotTuple1"));
        assert!(!rendered.contains("impl Default for XlsynthPirAotTuple"));
        assert!(rendered.contains("pub type PairAlias = super::XlsynthPirAotTuple0;"));
        assert!(rendered.contains("pub pair: super::XlsynthPirAotTuple0,"));
        assert!(rendered.contains("pair: &super::super::XlsynthPirAotTuple0"));
        assert!(rendered.contains("output: &mut [super::super::XlsynthPirAotTuple1; 2]"));
        assert!(!rendered.contains("(U8, U16)"));
    }

    #[test]
    fn zero_bit_aot_types_use_zero_sized_runtime_wrappers() {
        assert_eq!(
            scalar_runtime_type(PirAotSignedness::Unsigned, 0),
            "xlsynth_pir_compiler_runtime::UnsignedBits0"
        );
        assert_eq!(
            scalar_runtime_type(PirAotSignedness::Signed, 0),
            "xlsynth_pir_compiler_runtime::SignedBits0"
        );

        let mut declarations = Vec::new();
        assert_eq!(
            render_value_type(&Type::Bits(0), "ZeroBits", &mut declarations)
                .expect("bits[0] should render"),
            "Bits0"
        );
        assert!(declarations.is_empty());
    }

    #[test]
    fn dslx_aot_pir_is_optimized_before_cranelift_lowering() {
        let pir_text = r#"package optimizer_test

top fn main(x: bits[32]) -> bits[32] {
  zero: bits[32] = literal(value=0, id=2)
  ret sum: bits[32] = add(x, zero, id=3)
}
"#;
        assert!(pir_text.contains("add(x, zero,"));

        let optimized = optimize_dslx_aot_pir_text(pir_text, "main", Some("optimizer_test.ir"))
            .expect("IR optimization should succeed");

        assert!(!optimized.contains("add(x, zero,"));
    }
}
