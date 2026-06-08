// SPDX-License-Identifier: Apache-2.0

//! Build-script helpers for emitting standalone native PIR AOT wrappers.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub use xlsynth::DslxConvertOptions;
use xlsynth::aot_builder::{
    NativeTypedDslxFunctionSignature, NativeTypedDslxType, collect_typed_dslx_aot_dependencies,
    render_native_typed_dslx_generated_module, render_native_typed_dslx_package_generated_module,
};
pub use xlsynth::aot_builder::{
    TypedAotDecl, TypedAotEntrypoint, TypedAotEnumVariant, TypedAotField, TypedAotModule,
    TypedAotPackageMetadata, TypedAotParam, TypedAotType, TypedDslxAotBuildSpec,
    build_native_typed_dslx_aot_package_metadata,
};
use xlsynth::{
    DslxCallingConvention, convert_dslx_to_ir_text, dslx_path_to_module_name,
    mangle_dslx_name_with_calling_convention,
};
use xlsynth_pir::ir::{PackageMember, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler_runtime::{
    AssumptionFailureKind, EventKind, EventSiteMetadata, TraceTupleFieldLayout, TraceValueLayout,
};

use crate::{
    AotArtifact, CompilerError, NativeTupleFieldLayout, NativeValueLayout, ScalarLayout,
    WideBitsLayout, compile_package_aot,
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

/// Builds one metadata-backed shared wrapper from checked-in IR files.
pub struct TypedIrAotPackageBuilder<'a> {
    name: &'a str,
    metadata_path: PathBuf,
}

/// Output files and artifacts produced for a typed IR AOT package.
pub struct GeneratedTypedIrAotPackage {
    pub rust_file: PathBuf,
    pub object_files: Vec<PathBuf>,
    pub artifacts: Vec<AotArtifact>,
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

    /// Emits a shared wrapper and linked objects into Cargo's `OUT_DIR`.
    pub fn build(&self) -> Result<GeneratedTypedIrAotPackage, CompilerError> {
        let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
            CompilerError::Backend("OUT_DIR is required while emitting an AOT package".into())
        })?;
        self.build_with_out_dir(Path::new(&out_dir))
    }

    /// Emits a shared wrapper and linked objects into `out_dir`.
    pub fn build_with_out_dir(
        &self,
        out_dir: &Path,
    ) -> Result<GeneratedTypedIrAotPackage, CompilerError> {
        if self.name.is_empty() {
            return Err(CompilerError::InvalidArgument(
                "AOT package name must not be empty".into(),
            ));
        }
        let metadata = TypedAotPackageMetadata::from_json_file(&self.metadata_path)
            .map_err(|error| CompilerError::InvalidArgument(error.to_string()))?;
        let metadata_dir = self.metadata_path.parent().ok_or_else(|| {
            CompilerError::InvalidArgument(format!(
                "typed AOT metadata path `{}` has no parent directory",
                self.metadata_path.display()
            ))
        })?;
        println!("cargo:rerun-if-changed={}", self.metadata_path.display());
        if metadata.entrypoints.is_empty() {
            return Err(CompilerError::InvalidArgument(
                "AOT package must contain at least one entrypoint".into(),
            ));
        }
        let typed_package = ValidatedTypedAotPackage::new(&metadata)?;
        let mut ordered_artifacts = Vec::with_capacity(metadata.entrypoints.len());
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
            let ir_file = entrypoint.ir_file.as_ref().ok_or_else(|| {
                CompilerError::InvalidArgument(format!(
                    "typed AOT metadata entrypoint `{}` must specify ir_file",
                    entrypoint.name
                ))
            })?;
            let ir_path = metadata_dir.join(ir_file);
            println!("cargo:rerun-if-changed={}", ir_path.display());
            let pir_text = fs::read_to_string(&ir_path).map_err(|error| {
                CompilerError::Backend(format!(
                    "failed to read IR file {} for typed AOT entrypoint `{}`: {error}",
                    ir_path.display(),
                    entrypoint.name
                ))
            })?;
            let package = Parser::new(&pir_text)
                .parse_and_validate_package()
                .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
            let entrypoint_symbol = format!("__xlsynth_pir_aot_{base_name}");
            let artifact = compile_package_aot(&package, &entrypoint.ir_top, &entrypoint_symbol)?;
            typed_package.validate_entrypoint_layout(entrypoint, &artifact)?;
            ordered_artifacts.push((base_name, artifact));
        }

        fs::create_dir_all(out_dir).map_err(|error| CompilerError::Backend(error.to_string()))?;
        let package_name = sanitize_identifier(self.name);
        let rust_file = out_dir.join(format!("{package_name}_typed_ir_pir_aot_package.rs"));
        fs::write(
            &rust_file,
            render_typed_ir_aot_package(self.name, &typed_package, &ordered_artifacts)?,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

        let mut object_files = Vec::with_capacity(ordered_artifacts.len());
        let mut artifacts = Vec::with_capacity(ordered_artifacts.len());
        for (base_name, artifact) in ordered_artifacts {
            let object_file = out_dir.join(format!("{base_name}.pir_aot.o"));
            fs::write(&object_file, &artifact.object_code)
                .map_err(|error| CompilerError::Backend(error.to_string()))?;
            cc::Build::new()
                .cargo_metadata(true)
                .object(&object_file)
                .compile(&format!("xlsynth_pir_aot_{base_name}"));
            object_files.push(object_file);
            artifacts.push(artifact);
        }

        Ok(GeneratedTypedIrAotPackage {
            rust_file,
            object_files,
            artifacts,
        })
    }
}

/// Collects independently compiled DSLX entrypoints into one shared wrapper.
pub struct TypedDslxAotPackageBuilder<'a> {
    name: &'a str,
    specs: Vec<TypedDslxAotBuildSpec<'a>>,
}

/// Output files and artifacts produced for a typed DSLX AOT package.
pub struct GeneratedTypedDslxAotPackage {
    pub rust_file: PathBuf,
    pub object_files: Vec<PathBuf>,
    pub artifacts: Vec<AotArtifact>,
}

impl<'a> TypedDslxAotPackageBuilder<'a> {
    /// Creates an empty typed DSLX AOT package.
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            specs: Vec::new(),
        }
    }

    /// Adds one independently compiled top-level DSLX function.
    pub fn add_entrypoint(mut self, spec: TypedDslxAotBuildSpec<'a>) -> Self {
        self.specs.push(spec);
        self
    }

    /// Emits a shared wrapper and linked objects into Cargo's `OUT_DIR`.
    pub fn build(&self) -> Result<GeneratedTypedDslxAotPackage, CompilerError> {
        let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
            CompilerError::Backend("OUT_DIR is required while emitting an AOT package".into())
        })?;
        self.build_with_out_dir(Path::new(&out_dir))
    }

    /// Emits a shared wrapper and linked objects into `out_dir`.
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
        let mut artifacts = Vec::with_capacity(self.specs.len());
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
            artifacts.push(compile_dslx_artifact(spec, &base_name)?.1);
        }
        let generated_source = render_native_typed_dslx_package_generated_module(
            self.name,
            &self.specs,
            |index, signature| {
                validate_native_typed_dslx_signature(signature, &artifacts[index])
                    .map_err(|error| xlsynth::XlsynthError(error.to_string()))?;
                Ok(render_typed_dslx_runner_items(signature, &artifacts[index]))
            },
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

        fs::create_dir_all(out_dir).map_err(|error| CompilerError::Backend(error.to_string()))?;
        let package_name = sanitize_identifier(self.name);
        let rust_file = out_dir.join(format!("{package_name}_typed_dslx_pir_aot_package.rs"));
        fs::write(&rust_file, generated_source)
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        let mut object_files = Vec::with_capacity(self.specs.len());
        for (spec, artifact) in self.specs.iter().zip(artifacts.iter()) {
            let base_name = sanitize_identifier(spec.name);
            let object_file = out_dir.join(format!("{base_name}.pir_aot.o"));
            fs::write(&object_file, &artifact.object_code)
                .map_err(|error| CompilerError::Backend(error.to_string()))?;
            cc::Build::new()
                .cargo_metadata(true)
                .object(&object_file)
                .compile(&format!("xlsynth_pir_aot_{base_name}"));
            object_files.push(object_file);
        }
        Ok(GeneratedTypedDslxAotPackage {
            rust_file,
            object_files,
            artifacts,
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

/// Emits a typed native AOT wrapper for one DSLX top-level function.
pub fn emit_aot_module_from_dslx_file(
    spec: &TypedDslxAotBuildSpec<'_>,
) -> Result<GeneratedAotModule, CompilerError> {
    let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
        CompilerError::Backend("OUT_DIR is required while emitting an AOT module".into())
    })?;
    emit_aot_module_from_dslx_file_with_out_dir(spec, Path::new(&out_dir))
}

/// Emits a typed native AOT wrapper for one DSLX top-level function into
/// `out_dir`.
pub fn emit_aot_module_from_dslx_file_with_out_dir(
    spec: &TypedDslxAotBuildSpec<'_>,
    out_dir: &Path,
) -> Result<GeneratedAotModule, CompilerError> {
    let base_name = sanitize_identifier(spec.name);
    if base_name.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "AOT build spec name must contain an identifier character".into(),
        ));
    }
    let (dslx_text, artifact) = compile_dslx_artifact(spec, &base_name)?;
    let generated_source =
        render_native_typed_dslx_generated_module(spec, &dslx_text, |signature| {
            validate_native_typed_dslx_signature(signature, &artifact)
                .map_err(|error| xlsynth::XlsynthError(error.to_string()))?;
            Ok(render_typed_dslx_runner_items(signature, &artifact))
        })
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

    fs::create_dir_all(out_dir).map_err(|error| CompilerError::Backend(error.to_string()))?;
    let object_file = out_dir.join(format!("{base_name}.pir_aot.o"));
    let rust_file = out_dir.join(format!("{base_name}_typed_dslx_pir_aot_wrapper.rs"));
    fs::write(&object_file, &artifact.object_code)
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    fs::write(&rust_file, generated_source)
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

fn compile_dslx_artifact(
    spec: &TypedDslxAotBuildSpec<'_>,
    base_name: &str,
) -> Result<(String, AotArtifact), CompilerError> {
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
    let (pir_text, top) = if spec.dslx_options.force_implicit_token_calling_convention {
        let wrapper_top = format!("__xlsynth_pir_aot_{base_name}_dslx_entry");
        (
            append_implicit_token_entrypoint_wrapper(&pir_text, &dslx_top, &wrapper_top)?,
            wrapper_top,
        )
    } else {
        (pir_text, dslx_top)
    };
    let package = Parser::new(&pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    let entrypoint_symbol = format!("__xlsynth_pir_aot_{base_name}");
    let artifact = compile_package_aot(&package, &top, &entrypoint_symbol)?;
    Ok((dslx_text, artifact))
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

fn render_typed_dslx_runner_items(
    signature: &NativeTypedDslxFunctionSignature,
    artifact: &AotArtifact,
) -> String {
    let param_names = signature
        .params
        .iter()
        .map(|param| param.name.clone())
        .collect::<Vec<_>>();
    let param_types = signature
        .params
        .iter()
        .map(|param| param.rust_type.clone())
        .collect::<Vec<_>>();
    render_runner_items(
        artifact,
        &param_names,
        &param_types,
        &signature.return_rust_type,
        "",
        false,
    )
}

fn validate_native_typed_dslx_signature(
    signature: &NativeTypedDslxFunctionSignature,
    artifact: &AotArtifact,
) -> Result<(), CompilerError> {
    if signature.params.len() != artifact.param_layouts.len() {
        return Err(CompilerError::InvalidFunction(format!(
            "typed DSLX parameter count {} does not match compiled PIR parameter count {}",
            signature.params.len(),
            artifact.param_layouts.len()
        )));
    }
    for (index, (param, layout)) in signature
        .params
        .iter()
        .zip(artifact.param_layouts.iter())
        .enumerate()
    {
        validate_native_typed_dslx_type(
            &format!("parameter {index} `{}`", param.name),
            &param.ty,
            layout,
        )?;
    }
    validate_native_typed_dslx_type(
        "return value",
        &signature.return_type,
        &artifact.result_layout,
    )
}

fn validate_native_typed_dslx_type(
    label: &str,
    dslx_type: &NativeTypedDslxType,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    match (dslx_type, layout) {
        (
            NativeTypedDslxType::Bits { bit_count } | NativeTypedDslxType::Enum { bit_count },
            NativeValueLayout::Scalar(scalar),
        ) if *bit_count == scalar.bit_count => Ok(()),
        (
            NativeTypedDslxType::Bits { bit_count } | NativeTypedDslxType::Enum { bit_count },
            NativeValueLayout::WideBits(wide),
        ) if *bit_count == wide.bit_count => Ok(()),
        (
            NativeTypedDslxType::Array { size, element },
            NativeValueLayout::Array {
                element: native_element,
                element_count,
            },
        ) if size == element_count => validate_native_typed_dslx_type(
            &format!("{label} array element"),
            element,
            native_element,
        ),
        (
            NativeTypedDslxType::Struct { fields },
            NativeValueLayout::Tuple {
                fields: native_fields,
                ..
            },
        ) if fields.len() == native_fields.len() => {
            for (field, native_field) in fields.iter().zip(native_fields.iter()) {
                validate_native_typed_dslx_type(
                    &format!("{label}.{}", field.name),
                    &field.ty,
                    &native_field.layout,
                )?;
            }
            Ok(())
        }
        _ => Err(CompilerError::InvalidFunction(format!(
            "typed DSLX native layout mismatch for {label}: DSLX has {dslx_type:?}, compiled PIR has {layout:?}"
        ))),
    }
}

struct ValidatedTypedAotPackage<'a> {
    metadata: &'a TypedAotPackageMetadata,
    decls: BTreeMap<TypedAotDeclKey, &'a TypedAotDecl>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TypedAotDeclKey {
    module: Vec<String>,
    name: String,
}

impl<'a> ValidatedTypedAotPackage<'a> {
    fn new(metadata: &'a TypedAotPackageMetadata) -> Result<Self, CompilerError> {
        if metadata.format_version != 1 {
            return Err(CompilerError::InvalidArgument(format!(
                "unsupported typed AOT metadata format_version {}; expected 1",
                metadata.format_version
            )));
        }
        let mut module_paths = BTreeSet::new();
        let mut decls = BTreeMap::new();
        for module in &metadata.modules {
            validate_module_path(&module.path)?;
            if !module_paths.insert(module.path.clone()) {
                return Err(CompilerError::InvalidArgument(format!(
                    "typed AOT metadata contains duplicate module path `{}`",
                    render_module_path_for_error(&module.path)
                )));
            }
            let mut module_decl_names = BTreeSet::new();
            for decl in &module.declarations {
                let name = typed_aot_decl_name(decl);
                validate_type_name(name)?;
                if !module_decl_names.insert(name.to_string()) {
                    return Err(CompilerError::InvalidArgument(format!(
                        "typed AOT metadata module `{}` contains duplicate declaration `{name}`",
                        render_module_path_for_error(&module.path)
                    )));
                }
                decls.insert(
                    TypedAotDeclKey {
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
            if let Some(ir_file) = &entrypoint.ir_file {
                validate_ir_file_path(ir_file)?;
            }
            validate_module_path(&entrypoint.owning_module)?;
            if !module_paths.contains(&entrypoint.owning_module) {
                return Err(CompilerError::InvalidArgument(format!(
                    "typed AOT metadata entrypoint `{}` uses unknown owning module `{}`",
                    entrypoint.name,
                    render_module_path_for_error(&entrypoint.owning_module)
                )));
            }
            if !entrypoint_names.insert(entrypoint.name.clone()) {
                return Err(CompilerError::InvalidArgument(format!(
                    "typed AOT metadata contains duplicate entrypoint `{}`",
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
        entrypoint: &TypedAotEntrypoint,
        artifact: &AotArtifact,
    ) -> Result<(), CompilerError> {
        if entrypoint.params.len() != artifact.param_layouts.len() {
            return Err(CompilerError::InvalidFunction(format!(
                "typed AOT metadata entrypoint `{}` parameter count {} does not match compiled PIR parameter count {}",
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
                    "typed AOT metadata layout mismatch for entrypoint `{}` parameter {index} `{}`: metadata has {expected:?}, compiled PIR has {actual:?}",
                    entrypoint.name, param.name
                )));
            }
        }
        let expected = self.lower_type_to_layout(&entrypoint.return_type, &mut Vec::new())?;
        if expected != artifact.result_layout {
            return Err(CompilerError::InvalidFunction(format!(
                "typed AOT metadata layout mismatch for entrypoint `{}` return value: metadata has {expected:?}, compiled PIR has {:?}",
                entrypoint.name, artifact.result_layout
            )));
        }
        Ok(())
    }

    fn validate_decl(
        &self,
        module_path: &[String],
        decl: &TypedAotDecl,
    ) -> Result<(), CompilerError> {
        match decl {
            TypedAotDecl::Struct { fields, .. } => {
                let mut field_names = BTreeSet::new();
                for field in fields {
                    validate_value_name(&field.name)?;
                    if !field_names.insert(field.name.clone()) {
                        return Err(CompilerError::InvalidArgument(format!(
                            "typed AOT metadata struct `{}` has duplicate field `{}`",
                            render_type_path_for_error(module_path, typed_aot_decl_name(decl)),
                            field.name
                        )));
                    }
                    self.lower_type_to_layout(&field.ty, &mut Vec::new())?;
                }
            }
            TypedAotDecl::Enum {
                bit_count,
                variants,
                ..
            } => {
                if *bit_count == 0 {
                    return Err(CompilerError::InvalidArgument(format!(
                        "typed AOT metadata enum `{}` has zero bit width",
                        render_type_path_for_error(module_path, typed_aot_decl_name(decl))
                    )));
                }
                let mut variant_names = BTreeSet::new();
                for variant in variants {
                    validate_type_name(&variant.name)?;
                    if !variant_names.insert(variant.name.clone()) {
                        return Err(CompilerError::InvalidArgument(format!(
                            "typed AOT metadata enum `{}` has duplicate variant `{}`",
                            render_type_path_for_error(module_path, typed_aot_decl_name(decl)),
                            variant.name
                        )));
                    }
                    if *bit_count < 128 && variant.value >= (1u128 << *bit_count) {
                        return Err(CompilerError::InvalidArgument(format!(
                            "typed AOT metadata enum variant `{}::{}` value {} does not fit bits[{bit_count}]",
                            render_type_path_for_error(module_path, typed_aot_decl_name(decl)),
                            variant.name,
                            variant.value
                        )));
                    }
                }
            }
            TypedAotDecl::Alias { target, .. } => {
                self.lower_type_to_layout(target, &mut Vec::new())?;
            }
        }
        Ok(())
    }

    fn lower_type_to_layout(
        &self,
        ty: &TypedAotType,
        active_refs: &mut Vec<TypedAotDeclKey>,
    ) -> Result<NativeValueLayout, CompilerError> {
        match ty {
            TypedAotType::Bits { bit_count } => {
                if *bit_count == 0 {
                    return Err(CompilerError::InvalidArgument(
                        "typed AOT metadata bits type must have nonzero width".into(),
                    ));
                }
                Ok(native_bits_layout(*bit_count))
            }
            TypedAotType::Token => Ok(NativeValueLayout::Token),
            TypedAotType::Array { size, element } => {
                if *size == 0 {
                    return Err(CompilerError::InvalidArgument(
                        "typed AOT metadata arrays must have nonzero size".into(),
                    ));
                }
                Ok(NativeValueLayout::Array {
                    element: Box::new(self.lower_type_to_layout(element, active_refs)?),
                    element_count: *size,
                })
            }
            TypedAotType::Tuple { elements } => self.lower_fields_to_tuple_layout(
                elements
                    .iter()
                    .map(|element| self.lower_type_to_layout(element, active_refs))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TypedAotType::TypeRef { module, name } => {
                let key = TypedAotDeclKey {
                    module: module.clone(),
                    name: name.clone(),
                };
                if active_refs.contains(&key) {
                    return Err(CompilerError::InvalidArgument(format!(
                        "typed AOT metadata contains cyclic type reference through `{}`",
                        render_type_path_for_error(module, name)
                    )));
                }
                let decl = self.decls.get(&key).ok_or_else(|| {
                    CompilerError::InvalidArgument(format!(
                        "typed AOT metadata references unknown type `{}`",
                        render_type_path_for_error(module, name)
                    ))
                })?;
                active_refs.push(key);
                let result = match decl {
                    TypedAotDecl::Struct { fields, .. } => self.lower_fields_to_tuple_layout(
                        fields
                            .iter()
                            .map(|field| self.lower_type_to_layout(&field.ty, active_refs))
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    TypedAotDecl::Enum { bit_count, .. } => Ok(native_bits_layout(*bit_count)),
                    TypedAotDecl::Alias { target, .. } => {
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
                CompilerError::UnsupportedType("typed AOT tuple layout size overflow".into())
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

fn render_typed_ir_aot_package(
    package_name: &str,
    package: &ValidatedTypedAotPackage<'_>,
    artifacts: &[(String, AotArtifact)],
) -> Result<String, CompilerError> {
    if package.metadata.entrypoints.len() != artifacts.len() {
        return Err(CompilerError::InvalidArgument(format!(
            "typed AOT metadata has {} entrypoints, but build has {} artifacts",
            package.metadata.entrypoints.len(),
            artifacts.len()
        )));
    }
    let mut root = RenderModuleNode::default();
    for module in &package.metadata.modules {
        let mut items = vec![
            "#![allow(dead_code)]".to_string(),
            "#![allow(unused_imports)]".to_string(),
            "use super::{BitsInU8, BitsInU16, BitsInU32, BitsInU64, Token, WideBits};".to_string(),
        ];
        for decl in &module.declarations {
            items.push(render_typed_aot_decl(package, &module.path, decl)?);
        }
        insert_render_module_items(&mut root, &module.path, items)?;
    }
    for (entrypoint, (_, artifact)) in package.metadata.entrypoints.iter().zip(artifacts.iter()) {
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
            .map(|param| render_typed_aot_type(package, &param.ty, &runner_path))
            .collect::<Result<Vec<_>, _>>()?;
        let output_type = render_typed_aot_type(package, &entrypoint.return_type, &runner_path)?;
        insert_render_module_items(
            &mut root,
            &runner_path,
            vec![
                "#![allow(dead_code)]".to_string(),
                "#![allow(unused_imports)]".to_string(),
                "use super::*;".to_string(),
                render_runner_items(
                    artifact,
                    &param_names,
                    &param_types,
                    &output_type,
                    "",
                    false,
                ),
            ],
        )?;
    }

    Ok(format!(
        "// SPDX-License-Identifier: Apache-2.0\n// Generated by xlsynth_pir_compiler::aot from typed IR AOT package {package_name:?}.\n\n{}{}",
        render_typed_ir_runtime_imports(),
        render_module_node_children(&root, 0),
    ))
}

fn render_typed_ir_runtime_imports() -> &'static str {
    r#"pub use xlsynth_pir_compiler_runtime::{
    BitsInU8, BitsInU16, BitsInU32, BitsInU64, Token, WideBits,
};
"#
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

fn render_typed_aot_decl(
    package: &ValidatedTypedAotPackage<'_>,
    module_path: &[String],
    decl: &TypedAotDecl,
) -> Result<String, CompilerError> {
    match decl {
        TypedAotDecl::Struct { name, fields } => {
            let rendered_fields = fields
                .iter()
                .map(|field| {
                    Ok(format!(
                        "    pub {}: {},\n",
                        sanitize_identifier(&field.name),
                        render_typed_aot_type(package, &field.ty, module_path)?
                    ))
                })
                .collect::<Result<Vec<_>, CompilerError>>()?
                .concat();
            Ok(format!(
                "#[repr(C)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{\n{rendered_fields}}}\n"
            ))
        }
        TypedAotDecl::Enum {
            name,
            bit_count,
            variants,
        } => {
            if *bit_count > 64 {
                return Err(CompilerError::UnsupportedType(format!(
                    "typed AOT enum `{}` uses bits[{bit_count}], but generated enum constants currently support widths up to 64",
                    render_type_path_for_error(module_path, name)
                )));
            }
            let bits_type = render_native_bits_type(*bit_count);
            let constants = variants
                .iter()
                .map(|variant| {
                    format!(
                        "    #[allow(non_upper_case_globals)]\n    pub const {}: Self = Self({bits_type}::wrapping({}));\n",
                        sanitize_identifier(&variant.name),
                        variant.value
                    )
                })
                .collect::<Vec<_>>()
                .concat();
            Ok(format!(
                "#[repr(transparent)]\n#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]\npub struct {name}(pub {bits_type});\nimpl {name} {{\n{constants}}}\n"
            ))
        }
        TypedAotDecl::Alias { name, target } => Ok(format!(
            "pub type {name} = {};\n",
            render_typed_aot_type(package, target, module_path)?
        )),
    }
}

fn render_typed_aot_type(
    package: &ValidatedTypedAotPackage<'_>,
    ty: &TypedAotType,
    current_module_path: &[String],
) -> Result<String, CompilerError> {
    match ty {
        TypedAotType::Bits { bit_count } => Ok(render_native_bits_type(*bit_count)),
        TypedAotType::Token => Ok("Token".to_string()),
        TypedAotType::Array { size, element } => Ok(format!(
            "[{}; {size}]",
            render_typed_aot_type(package, element, current_module_path)?
        )),
        TypedAotType::Tuple { elements } => {
            let rendered = elements
                .iter()
                .map(|element| render_typed_aot_type(package, element, current_module_path))
                .collect::<Result<Vec<_>, _>>()?;
            if rendered.len() == 1 {
                Ok(format!("({},)", rendered[0]))
            } else {
                Ok(format!("({})", rendered.join(", ")))
            }
        }
        TypedAotType::TypeRef { module, name } => {
            let key = TypedAotDeclKey {
                module: module.clone(),
                name: name.clone(),
            };
            package.decls.get(&key).ok_or_else(|| {
                CompilerError::InvalidArgument(format!(
                    "typed AOT metadata references unknown type `{}`",
                    render_type_path_for_error(module, name)
                ))
            })?;
            Ok(render_relative_type_path(current_module_path, module, name))
        }
    }
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

fn align_up_local(value: usize, alignment: usize) -> Result<usize, CompilerError> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| CompilerError::UnsupportedType("typed AOT layout size overflow".into()))
}

fn typed_aot_decl_name(decl: &TypedAotDecl) -> &str {
    match decl {
        TypedAotDecl::Struct { name, .. }
        | TypedAotDecl::Enum { name, .. }
        | TypedAotDecl::Alias { name, .. } => name,
    }
}

fn validate_module_path(path: &[String]) -> Result<(), CompilerError> {
    if path.is_empty() {
        return Err(CompilerError::InvalidArgument(
            "typed AOT metadata module path must not be empty".into(),
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
            "typed AOT metadata ir_file must not be empty".into(),
        ));
    }
    let path = Path::new(path);
    if path.is_absolute() {
        return Err(CompilerError::InvalidArgument(
            "typed AOT metadata ir_file must be relative".into(),
        ));
    }
    for component in path.components() {
        match component {
            std::path::Component::Normal(_) => {}
            _ => {
                return Err(CompilerError::InvalidArgument(
                    "typed AOT metadata ir_file must be a relative path without `..`".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_identifier(name: &str, label: &str) -> Result<(), CompilerError> {
    if name.is_empty() {
        return Err(CompilerError::InvalidArgument(format!(
            "typed AOT metadata {label} must not be empty"
        )));
    }
    if sanitize_identifier(name) != name {
        return Err(CompilerError::InvalidArgument(format!(
            "typed AOT metadata {label} `{name}` is not a supported Rust identifier"
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
            artifact,
            &param_names,
            &input_type_names,
            &output_type,
            &type_declarations,
            true,
        )
    ))
}

fn render_runner_items(
    artifact: &AotArtifact,
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
    BitsInU8, BitsInU16, BitsInU32, BitsInU64, ExecutionResult, RunError, RunResult, Token,
    WideBits,
};
"#
    } else {
        "pub use xlsynth_pir_compiler_runtime::{ExecutionResult, RunError, RunResult};\n"
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

    fn invoke(&mut self, inputs: &[*const u8], output: *mut u8) -> Result<(), RunError> {{
        self.context.clear();
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

    fn reject_failures(events: &ExecutionResult) -> Result<(), RunError> {{
        if let Some(failure) = events.assertion_failures.first() {{
            return Err(RunError(format!(
                "compiled assertion failed at node {{}}: {{}}",
                failure.node_text_id, failure.message
            )));
        }}
        if let Some(failure) = events.assumption_failures.first() {{
            return Err(RunError(format!(
                "compiled assumed-in-bounds condition failed at node {{}}: {{:?}}",
                failure.node_text_id, failure.kind
            )));
        }}
        Ok(())
    }}

    /// Runs into caller-owned output storage and returns observable events.
    pub fn run_into_with_events(&mut self{signature_inputs}, output: &mut {output_type_name}) -> Result<ExecutionResult, RunError> {{
        let input_pointers = [{pointer_entries}];
        self.invoke(&input_pointers, std::ptr::from_mut(output).cast::<u8>())?;
        Ok(self.context.result())
    }}

    /// Runs into caller-owned output storage, rejecting assertion/assumption failures.
    pub fn run_into(&mut self{signature_inputs}, output: &mut {output_type_name}) -> Result<(), RunError> {{
        let events = self.run_into_with_events({arg_calls}output)?;
        Self::reject_failures(&events)
    }}

    /// Runs and returns output together with observable event records.
    pub fn run_with_events(&mut self{signature_inputs}) -> Result<RunResult<{output_type_name}>, RunError> {{
        let input_pointers = [{pointer_entries}];
        let mut output = std::mem::MaybeUninit::<{output_type_name}>::uninit();
        self.invoke(&input_pointers, output.as_mut_ptr().cast::<u8>())?;
        Ok(RunResult {{
            output: unsafe {{ output.assume_init() }},
            events: self.context.result(),
        }})
    }}

    /// Runs and returns only output, rejecting assertion/assumption failures.
    pub fn run(&mut self{signature_inputs}) -> Result<{output_type_name}, RunError> {{
        let result = self.run_with_events({arg_calls_no_output})?;
        Self::reject_failures(&result.events)?;
        Ok(result.output)
    }}
}}

/// Creates a reusable runner for this native AOT entrypoint.
pub fn new_runner() -> Result<Runner, RunError> {{
    Runner::new()
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
        arg_calls = render_call_args(&args, true),
        arg_calls_no_output = render_call_args(&args, false),
    )
}

fn render_call_args(args: &[String], with_trailing_comma: bool) -> String {
    if args.is_empty() {
        return String::new();
    }
    let suffix = if with_trailing_comma { ", " } else { "" };
    format!("{}{suffix}", args.join(", "))
}

fn render_value_type(
    ty: &Type,
    name: &str,
    declarations: &mut Vec<String>,
) -> Result<String, CompilerError> {
    match ty {
        Type::Token => Ok("Token".into()),
        Type::Bits(width) => {
            if *width == 0 {
                return Err(CompilerError::UnsupportedType(
                    "bits[0] wrapper storage is unsupported".into(),
                ));
            }
            Ok(render_native_bits_type(*width))
        }
        Type::Array(array) => {
            let element_name = format!("{name}Element");
            let element =
                render_value_type(array.element_type.as_ref(), &element_name, declarations)?;
            Ok(format!("[{element}; {}]", array.element_count))
        }
        Type::Tuple(fields) => {
            let mut rendered_fields = Vec::new();
            for (index, field) in fields.iter().enumerate() {
                let field_name = format!("{name}Field{index}");
                let rendered = render_value_type(field, &field_name, declarations)?;
                rendered_fields.push(format!("    pub field{index}: {rendered},\n"));
            }
            declarations.push(format!(
                "#[repr(C)]\n#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]\npub struct {name} {{\n{}}}\n",
                rendered_fields.concat()
            ));
            Ok(name.to_string())
        }
    }
}

fn render_native_bits_type(width: usize) -> String {
    match width {
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
        "EventSiteMetadata {{ node_text_id: {}, kind: {}, label: {}, message: {}, format: {}, operand_layouts: vec![{}] }}",
        site.node_text_id,
        render_event_kind(site.kind),
        render_optional_string(&site.label),
        render_optional_string(&site.message),
        render_optional_string(&site.format),
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
