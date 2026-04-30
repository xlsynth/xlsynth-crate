// SPDX-License-Identifier: Apache-2.0
//! Build-script helpers that compile XLS IR or DSLX sources to AOT artifacts
//! and generated Rust wrappers.
//!
//! The IR path emits structural wrapper types derived from AOT metadata alone.
//! The DSLX path additionally typechecks the source modules, keeps the DSLX
//! bridge type names, validates that those semantic types flatten to the AOT
//! ABI metadata, and emits a `Runner` that packs and unpacks bridge values
//! directly.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::aot_entrypoint_metadata::{
    get_entrypoint_function_signature, AotEntrypointMetadata, AotFunctionSignature, AotType,
    AotTypeLayout,
};
use crate::aot_lib::{AotCompiled, AotResult};
use crate::dslx_bridge::{convert_imported_module, BridgeBuilder};
use crate::rust_bridge_builder::{
    render_rust_module_fragments, rust_type_path_between_dslx_modules, RustBridgeBuilder,
};
use crate::xlsynth_error::XlsynthError;
use crate::{
    convert_dslx_to_ir_text, dslx, dslx_path_to_module_name,
    mangle_dslx_name_with_calling_convention, DslxCallingConvention, DslxConvertOptions,
};

#[derive(Debug, Clone)]
/// Inputs required to compile one XLS IR function into generated AOT wrapper
/// artifacts.
pub struct AotBuildSpec<'a> {
    /// Base name used for emitted artifact filenames and generated symbol
    /// wrappers.
    pub name: &'a str,
    /// XLS IR package text that contains `top`.
    pub ir_text: &'a str,
    /// Name of the IR function to compile as the AOT entrypoint.
    pub top: &'a str,
}

/// Inputs required to compile one DSLX function into a typed DSLX AOT wrapper.
///
/// The generated module contains Rust bridge definitions for
/// `type_module_paths`, Rust bridge definitions for `dslx_path`, and a `Runner`
/// whose public signature uses canonical paths to those generated Rust bridge
/// types.
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

/// Collects several typed DSLX AOT entrypoints into one generated Rust package.
///
/// Package builds emit the participating DSLX modules once and place one
/// runner module per entrypoint beneath the DSLX module that owns the selected
/// top function. That keeps the public Rust types nominally shared across
/// runners instead of minting one copy per generated wrapper file.
pub struct TypedDslxAotPackageBuilder<'a> {
    /// Base name used for the generated shared Rust source file.
    name: &'a str,
    /// Entrypoints to compile into the package.
    specs: Vec<TypedDslxAotBuildSpec<'a>>,
}

/// Paths and metadata for one AOT entrypoint inside a shared typed DSLX
/// package.
#[derive(Debug, Clone)]
pub struct GeneratedTypedDslxAotEntrypoint {
    /// Build spec name after validation and filename sanitization.
    pub name: String,
    /// Object file containing the compiled AOT entrypoint.
    pub object_file: PathBuf,
    /// Serialized entrypoint metadata consumed by the generated descriptor.
    pub entrypoints_proto_file: PathBuf,
    /// Parsed AOT metadata for the selected entrypoint.
    pub metadata: AotEntrypointMetadata,
}

/// Paths and metadata for a generated shared typed DSLX AOT package.
#[derive(Debug, Clone)]
pub struct GeneratedTypedDslxAotPackage {
    /// Package name after validation and filename sanitization.
    pub name: String,
    /// Generated Rust source file that callers include once from build output.
    pub rust_file: PathBuf,
    /// Compiled AOT artifacts for each requested package entrypoint.
    pub entrypoints: Vec<GeneratedTypedDslxAotEntrypoint>,
}

struct TypedDslxCompiledEntrypoint<'a> {
    spec: &'a TypedDslxAotBuildSpec<'a>,
    base_name: String,
    dslx_text: String,
    proto_file_name: String,
    object_file: PathBuf,
    proto_file: PathBuf,
    metadata: AotEntrypointMetadata,
    signature: AotFunctionSignature,
}

/// Paths and metadata for emitted AOT wrapper artifacts in a build output
/// directory.
///
/// The generated Rust wrapper includes typed encode/decode helpers and a thin
/// `Runner` wrapper over `xlsynth::AotRunner`.
#[derive(Debug, Clone)]
pub struct GeneratedAotModule {
    /// Build spec name after validation and filename sanitization.
    pub name: String,
    /// Generated Rust source file that callers include from their build output.
    pub rust_file: PathBuf,
    /// Object file containing the compiled AOT entrypoint.
    pub object_file: PathBuf,
    /// Serialized entrypoint metadata consumed by the generated descriptor.
    pub entrypoints_proto_file: PathBuf,
    /// Parsed AOT metadata for the selected entrypoint.
    pub metadata: AotEntrypointMetadata,
}

/// Compiles IR text into raw AOT object code and parsed entrypoint metadata.
pub fn compile_ir_to_aot(ir_text: &str, top: &str) -> AotResult<AotCompiled> {
    AotCompiled::compile_ir(ir_text, top)
}

/// Emits AOT artifacts into Cargo's `OUT_DIR`.
///
/// This is the build-script friendly entry point and requires `OUT_DIR` to be
/// set in the environment.
pub fn emit_aot_module_from_ir_text(spec: &AotBuildSpec<'_>) -> AotResult<GeneratedAotModule> {
    let out_dir = std::env::var("OUT_DIR").map_err(|e| {
        XlsynthError(format!(
            "AOT build environment error: OUT_DIR was not set while emitting AOT module: {e}"
        ))
    })?;
    emit_aot_module_from_ir_text_with_out_dir(spec, Path::new(&out_dir))
}

/// Emits typed DSLX AOT artifacts into Cargo's `OUT_DIR`.
///
/// This DSLX-aware entry point is additive to the IR-only API and preserves
/// the old structural wrapper behavior for existing callers.
pub fn emit_typed_dslx_aot_module_from_file(
    spec: &TypedDslxAotBuildSpec<'_>,
) -> AotResult<GeneratedAotModule> {
    let out_dir = std::env::var("OUT_DIR").map_err(|e| {
        XlsynthError(format!(
            "AOT build environment error: OUT_DIR was not set while emitting typed DSLX AOT module: {e}"
        ))
    })?;
    emit_typed_dslx_aot_module_from_file_with_out_dir(spec, Path::new(&out_dir))
}

impl<'a> TypedDslxAotPackageBuilder<'a> {
    /// Creates an empty package builder.
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            specs: Vec::new(),
        }
    }

    /// Appends one typed DSLX AOT entrypoint to the package.
    pub fn add_entrypoint(mut self, spec: TypedDslxAotBuildSpec<'a>) -> Self {
        self.specs.push(spec);
        self
    }

    /// Emits the package into Cargo's `OUT_DIR`.
    pub fn build(&self) -> AotResult<GeneratedTypedDslxAotPackage> {
        let out_dir = std::env::var("OUT_DIR").map_err(|e| {
            XlsynthError(format!(
                "AOT build environment error: OUT_DIR was not set while emitting typed DSLX AOT package: {e}"
            ))
        })?;
        self.build_with_out_dir(Path::new(&out_dir))
    }

    /// Emits the package into an explicit output directory.
    pub fn build_with_out_dir(&self, out_dir: &Path) -> AotResult<GeneratedTypedDslxAotPackage> {
        emit_typed_dslx_aot_package_with_out_dir(self, out_dir)
    }
}

fn compile_typed_dslx_entrypoint_artifacts<'a>(
    spec: &'a TypedDslxAotBuildSpec<'a>,
    out_dir: &Path,
) -> AotResult<TypedDslxCompiledEntrypoint<'a>> {
    if spec.name.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: typed DSLX build spec name must not be empty".to_string(),
        ));
    }
    if spec.top.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: typed DSLX build spec top function must not be empty"
                .to_string(),
        ));
    }

    for dslx_source_path in collect_typed_dslx_aot_dependencies(spec)? {
        println!("cargo:rerun-if-changed={}", dslx_source_path.display());
    }

    let dslx_text = std::fs::read_to_string(spec.dslx_path).map_err(|e| {
        XlsynthError(format!(
            "AOT I/O failed for {}: {e}",
            spec.dslx_path.display()
        ))
    })?;
    let ir_text = convert_dslx_to_ir_text(&dslx_text, spec.dslx_path, &spec.dslx_options)?.ir;
    let dslx_module_name = dslx_path_to_module_name(spec.dslx_path)?;
    let calling_convention = if spec.dslx_options.force_implicit_token_calling_convention {
        DslxCallingConvention::ImplicitToken
    } else {
        DslxCallingConvention::Typical
    };
    let aot_top =
        mangle_dslx_name_with_calling_convention(dslx_module_name, spec.top, calling_convention)?;
    let compile = compile_ir_to_aot(&ir_text, &aot_top)?;
    let base_name = sanitize_identifier(spec.name);
    let AotCompiled {
        object_code,
        entrypoints_proto,
        metadata,
    } = compile;
    let signature = get_entrypoint_function_signature(&entrypoints_proto)
        .map_err(|e| XlsynthError(format!("AOT metadata parse failed: {}", e.0)))?;

    let object_file = out_dir.join(format!("{base_name}.aot.o"));
    let proto_file = out_dir.join(format!("{base_name}.entrypoints.pb"));
    write_file(&object_file, &object_code)?;
    write_file(&proto_file, &entrypoints_proto)?;
    let proto_file_name = proto_file
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| {
            XlsynthError(format!(
                "AOT build environment error: failed to derive UTF-8 file name from proto path {}",
                proto_file.display()
            ))
        })?
        .to_string();
    emit_link_archive(&base_name, &object_file)?;

    Ok(TypedDslxCompiledEntrypoint {
        spec,
        base_name,
        dslx_text,
        proto_file_name,
        object_file,
        proto_file,
        metadata,
        signature,
    })
}

/// Emits typed DSLX AOT artifacts into an explicit output directory.
///
/// This compiles the DSLX top function, emits bridge definitions for selected
/// DSLX modules, validates those semantic types against AOT metadata, and
/// writes a generated Rust module that encodes/decodes typed DSLX values
/// directly.
pub fn emit_typed_dslx_aot_module_from_file_with_out_dir(
    spec: &TypedDslxAotBuildSpec<'_>,
    out_dir: &Path,
) -> AotResult<GeneratedAotModule> {
    let compiled = compile_typed_dslx_entrypoint_artifacts(spec, out_dir)?;
    let rust_file = out_dir.join(format!("{}_typed_dslx_aot_wrapper.rs", compiled.base_name));
    let generated = render_typed_dslx_generated_module(
        spec,
        &compiled.dslx_text,
        &compiled.base_name,
        &compiled.proto_file_name,
        &compiled.metadata,
        &compiled.signature,
    )?;
    write_file(&rust_file, generated.as_bytes())?;
    run_rustfmt_best_effort(&rust_file);

    Ok(GeneratedAotModule {
        name: compiled.base_name,
        rust_file,
        object_file: compiled.object_file,
        entrypoints_proto_file: compiled.proto_file,
        metadata: compiled.metadata,
    })
}

/// Emits AOT object/proto/wrapper artifacts into an explicit output directory.
///
/// This compiles the target function, writes the object and entrypoint proto,
/// generates a typed Rust wrapper module, and emits a static archive suitable
/// for linking into the final crate.
pub fn emit_aot_module_from_ir_text_with_out_dir(
    spec: &AotBuildSpec<'_>,
    out_dir: &Path,
) -> AotResult<GeneratedAotModule> {
    if spec.name.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: build spec name must not be empty".to_string(),
        ));
    }
    if spec.top.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: build spec top function must not be empty".to_string(),
        ));
    }

    let compile = compile_ir_to_aot(spec.ir_text, spec.top)?;
    let base_name = sanitize_identifier(spec.name);
    let AotCompiled {
        object_code,
        entrypoints_proto,
        metadata: selected_metadata,
    } = compile;
    let signature = get_entrypoint_function_signature(&entrypoints_proto)
        .map_err(|e| XlsynthError(format!("AOT metadata parse failed: {}", e.0)))?;

    let object_file = out_dir.join(format!("{base_name}.aot.o"));
    let proto_file = out_dir.join(format!("{base_name}.entrypoints.pb"));
    let rust_file = out_dir.join(format!("{base_name}_aot_wrapper.rs"));

    write_file(&object_file, &object_code)?;
    write_file(&proto_file, &entrypoints_proto)?;

    let proto_file_name = proto_file
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| {
            XlsynthError(format!(
                "AOT build environment error: failed to derive UTF-8 file name from proto path {}",
                proto_file.display()
            ))
        })?;

    let generated = render_generated_module(
        &base_name,
        proto_file_name,
        &selected_metadata,
        &signature,
        spec.name,
        spec.top,
    )?;
    write_file(&rust_file, generated.as_bytes())?;
    run_rustfmt_best_effort(&rust_file);

    emit_link_archive(&base_name, &object_file)?;

    Ok(GeneratedAotModule {
        name: base_name,
        rust_file,
        object_file,
        entrypoints_proto_file: proto_file,
        metadata: selected_metadata,
    })
}

/// Reads IR text from disk and emits AOT artifacts into `OUT_DIR`.
///
/// This helper also emits `cargo:rerun-if-changed` for the IR file so Cargo
/// rebuilds generated artifacts when the input IR changes.
pub fn emit_aot_module_from_ir_file(
    name: &str,
    ir_path: &Path,
    top: &str,
) -> AotResult<GeneratedAotModule> {
    println!("cargo:rerun-if-changed={}", ir_path.display());
    let ir_text = std::fs::read_to_string(ir_path)
        .map_err(|e| XlsynthError(format!("AOT I/O failed for {}: {e}", ir_path.display())))?;
    let spec = AotBuildSpec {
        name,
        ir_text: &ir_text,
        top,
    };
    emit_aot_module_from_ir_text(&spec)
}

fn write_file(path: &Path, contents: &[u8]) -> AotResult<()> {
    std::fs::write(path, contents)
        .map_err(|e| XlsynthError(format!("AOT I/O failed for {}: {e}", path.display())))
}

fn run_rustfmt_best_effort(path: &Path) {
    let _ = Command::new("rustfmt").arg(path).status();
}

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

fn sanitize_value_identifier(name: &str, fallback: &str) -> String {
    let mut out = String::with_capacity(name.len().max(fallback.len()));
    for (index, ch) in name.chars().enumerate() {
        let ch = if ch == '_' || ch.is_ascii_alphanumeric() {
            ch
        } else {
            '_'
        };
        if index == 0 && ch.is_ascii_digit() {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    if out.is_empty() {
        out = fallback.to_string();
    }
    if is_rust_keyword(&out) {
        out.push('_');
    }
    out
}

fn sanitize_type_identifier(name: &str, fallback: &str) -> String {
    let mut out = String::new();
    let mut start_word = true;
    for ch in name.chars() {
        if ch == '_' || ch.is_ascii_alphanumeric() {
            if out.is_empty() && ch.is_ascii_digit() {
                out.push('_');
            }
            if start_word {
                out.push(ch.to_ascii_uppercase());
                start_word = false;
            } else {
                out.push(ch);
            }
        } else {
            start_word = true;
        }
    }
    if out.is_empty() {
        fallback.to_string()
    } else {
        out
    }
}

fn is_rust_keyword(name: &str) -> bool {
    matches!(
        name,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
            | "abstract"
            | "become"
            | "box"
            | "do"
            | "final"
            | "macro"
            | "override"
            | "priv"
            | "try"
            | "typeof"
            | "unsized"
            | "virtual"
            | "yield"
    )
}

fn scalar_bits_rust_type(bit_count: usize) -> &'static str {
    match bit_count {
        1 => "bool",
        0..=8 => "u8",
        9..=16 => "u16",
        17..=32 => "u32",
        33..=64 => "u64",
        _ => panic!(
            "scalar_bits_rust_type only supports widths <= 64, got {}",
            bit_count
        ),
    }
}

fn scalar_bits_native_width(bit_count: usize) -> usize {
    match bit_count {
        1 => 1,
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        _ => panic!(
            "scalar_bits_native_width only supports widths <= 64, got {}",
            bit_count
        ),
    }
}

fn scalar_bits_storage_bytes(bit_count: usize) -> usize {
    match bit_count {
        1 => 1,
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        33..=64 => 8,
        _ => panic!(
            "scalar_bits_storage_bytes only supports widths <= 64, got {}",
            bit_count
        ),
    }
}

fn format_usize_array(values: &[usize]) -> String {
    if values.is_empty() {
        "&[]".to_string()
    } else {
        format!(
            "&[{}]",
            values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[derive(Debug, Clone)]
enum ResolvedType {
    Bits {
        bit_count: usize,
    },
    Tuple {
        name: String,
        fields: Vec<ResolvedType>,
    },
    Array {
        size: usize,
        element: Box<ResolvedType>,
    },
    Token,
}

#[derive(Debug, Clone)]
struct TupleDef {
    name: String,
    field_types: Vec<ResolvedType>,
}

#[derive(Debug, Default)]
struct TypeResolver {
    bit_widths: BTreeSet<usize>,
    tuple_defs: Vec<TupleDef>,
    used_type_names: HashSet<String>,
}

impl TypeResolver {
    fn lower_type(&mut self, ty: &AotType, hint: &str) -> ResolvedType {
        match ty {
            AotType::Bits { bit_count } => {
                self.bit_widths.insert(*bit_count);
                ResolvedType::Bits {
                    bit_count: *bit_count,
                }
            }
            AotType::Token => ResolvedType::Token,
            AotType::Array { size, element } => ResolvedType::Array {
                size: *size,
                element: Box::new(self.lower_type(element, &format!("{hint}Element"))),
            },
            AotType::Tuple { elements } => {
                let tuple_name = self.allocate_type_name(hint);
                let mut field_types = Vec::with_capacity(elements.len());
                for (index, element) in elements.iter().enumerate() {
                    field_types
                        .push(self.lower_type(element, &format!("{tuple_name}Field{index}")));
                }
                self.tuple_defs.push(TupleDef {
                    name: tuple_name.clone(),
                    field_types: field_types.clone(),
                });
                ResolvedType::Tuple {
                    name: tuple_name,
                    fields: field_types,
                }
            }
        }
    }

    fn allocate_type_name(&mut self, hint: &str) -> String {
        let base = sanitize_type_identifier(hint, "GeneratedType");
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while !self.used_type_names.insert(candidate.clone()) {
            suffix += 1;
            candidate = format!("{base}{suffix}");
        }
        candidate
    }
}

fn rust_type_name(ty: &ResolvedType) -> String {
    match ty {
        ResolvedType::Bits { bit_count } => format!("Bits{bit_count}"),
        ResolvedType::Tuple { name, .. } => name.clone(),
        ResolvedType::Array { size, element } => format!("[{}; {size}]", rust_type_name(element)),
        ResolvedType::Token => "Token".to_string(),
    }
}

fn is_named_tuple(ty: &ResolvedType, name: &str) -> bool {
    matches!(ty, ResolvedType::Tuple { name: ty_name, .. } if ty_name == name)
}

fn render_type_declarations(
    resolver: &TypeResolver,
    input_types: &[ResolvedType],
    output_type: &ResolvedType,
) -> String {
    let mut out = String::new();
    out.push_str("#[derive(Debug, Clone, PartialEq, Eq)]\n");
    out.push_str("pub struct Token {}\n\n");

    for bit_count in &resolver.bit_widths {
        if *bit_count <= 64 {
            out.push_str(&format!(
                "pub type Bits{bit_count} = {};\n",
                scalar_bits_rust_type(*bit_count)
            ));
        } else {
            let byte_count = bit_count.div_ceil(8);
            out.push_str(&format!("pub type Bits{bit_count} = [u8; {byte_count}];\n"));
        }
    }
    if !resolver.bit_widths.is_empty() {
        out.push('\n');
    }

    for tuple in &resolver.tuple_defs {
        out.push_str("#[derive(Debug, Clone, PartialEq, Eq)]\n");
        if tuple.field_types.is_empty() {
            out.push_str(&format!("pub struct {} {{}}\n\n", tuple.name));
            continue;
        }
        out.push_str(&format!("pub struct {} {{\n", tuple.name));
        for (index, field_ty) in tuple.field_types.iter().enumerate() {
            out.push_str(&format!(
                "    pub field{index}: {},\n",
                rust_type_name(field_ty)
            ));
        }
        out.push_str("}\n\n");
    }

    for (index, input_ty) in input_types.iter().enumerate() {
        let input_name = format!("Input{index}");
        if !is_named_tuple(input_ty, &input_name) {
            out.push_str(&format!(
                "pub type {input_name} = {};\n",
                rust_type_name(input_ty)
            ));
        }
    }
    if !input_types.is_empty() {
        out.push('\n');
    }

    if !is_named_tuple(output_type, "Output") {
        out.push_str(&format!(
            "pub type Output = {};\n\n",
            rust_type_name(output_type)
        ));
    }

    out
}

fn render_layout_constants(prefix: &str, layouts: &[AotTypeLayout]) -> String {
    let mut out = String::new();
    for (index, layout) in layouts.iter().enumerate() {
        out.push_str(&format!(
            "const {prefix}{index}_LAYOUT: &[xlsynth::aot_entrypoint_metadata::AotElementLayout] = &[\n"
        ));
        for element in &layout.elements {
            out.push_str(&format!(
                "    xlsynth::aot_entrypoint_metadata::AotElementLayout {{ offset: {}, data_size: {}, padded_size: {} }},\n",
                element.offset, element.data_size, element.padded_size
            ));
        }
        out.push_str("];\n");
    }
    out
}

fn push_line(lines: &mut Vec<String>, text: impl AsRef<str>) {
    lines.push(text.as_ref().to_string());
}

fn emit_pack_statements(
    ty: &ResolvedType,
    value_expr: &str,
    layout_name: &str,
    dst_name: &str,
    leaf_index_expr: &str,
    lines: &mut Vec<String>,
    next_loop_index: &mut usize,
) {
    match ty {
        ResolvedType::Bits { bit_count } => {
            if *bit_count <= 64 {
                if *bit_count == 1 {
                    push_line(
                        lines,
                        format!("let encoded_bit: u8 = u8::from(*&({value_expr}));"),
                    );
                    push_line(
                        lines,
                        format!(
                            "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &[encoded_bit]);"
                        ),
                    );
                } else {
                    let native_width = scalar_bits_native_width(*bit_count);
                    if *bit_count == 0 {
                        push_line(
                            lines,
                            format!(
                                "assert!(({value_expr}) == 0, \"AOT encode overflow: value does not fit in 0 bits\");"
                            ),
                        );
                    } else if *bit_count < native_width {
                        push_line(
                            lines,
                            format!(
                                "assert!((({value_expr}) >> {bit_count}) == 0, \"AOT encode overflow: value does not fit in {bit_count} bits\");"
                            ),
                        );
                    }
                    push_line(
                        lines,
                        format!(
                            "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &({value_expr}).to_ne_bytes());"
                        ),
                    );
                }
            } else {
                let bit_remainder = bit_count % 8;
                if bit_remainder != 0 {
                    let last_byte_index = bit_count.div_ceil(8) - 1;
                    push_line(
                        lines,
                        format!(
                            "assert!((({value_expr})[{last_byte_index}] >> {bit_remainder}) == 0, \"AOT encode overflow: value does not fit in {bit_count} bits\");"
                        ),
                    );
                }
                push_line(
                    lines,
                    format!(
                        "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &({value_expr}));"
                    ),
                );
            }
        }
        ResolvedType::Token => {
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &[]);"
                ),
            );
        }
        ResolvedType::Tuple { fields, .. } => {
            let mut offset = 0usize;
            for (index, field) in fields.iter().enumerate() {
                let field_leaf_base = if offset == 0 {
                    leaf_index_expr.to_string()
                } else {
                    format!("{leaf_index_expr} + {offset}")
                };
                emit_pack_statements(
                    field,
                    &format!("({value_expr}).field{index}"),
                    layout_name,
                    dst_name,
                    &field_leaf_base,
                    lines,
                    next_loop_index,
                );
                offset = offset.saturating_add(leaf_count(field));
            }
        }
        ResolvedType::Array { size, element } => {
            let element_leaves = leaf_count(element);
            if *size == 0 || element_leaves == 0 {
                return;
            }
            let loop_name = format!("index_{}", *next_loop_index);
            *next_loop_index += 1;
            push_line(lines, format!("for {loop_name} in 0..{size} {{"));
            let element_leaf_base = if element_leaves == 1 {
                format!("{leaf_index_expr} + {loop_name}")
            } else {
                format!("{leaf_index_expr} + {loop_name} * {element_leaves}")
            };
            emit_pack_statements(
                element,
                &format!("({value_expr})[{loop_name}]"),
                layout_name,
                dst_name,
                &element_leaf_base,
                lines,
                next_loop_index,
            );
            push_line(lines, "}");
        }
    }
}

fn render_decode_expr(
    ty: &ResolvedType,
    layout_name: &str,
    src_name: &str,
    leaf_index_expr: &str,
    next_loop_index: &mut usize,
) -> String {
    match ty {
        ResolvedType::Bits { bit_count } => {
            if *bit_count <= 64 {
                if *bit_count == 1 {
                    format!(
                        "{{ let mut dst_bytes = [0u8; 1]; \
                        xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut dst_bytes); \
                        assert!(dst_bytes[0] <= 1, \"AOT decode overflow: value does not fit in 1 bit\"); \
                        dst_bytes[0] != 0 }}"
                    )
                } else {
                    let native_type = scalar_bits_rust_type(*bit_count);
                    let storage_bytes = scalar_bits_storage_bytes(*bit_count);
                    let native_width = scalar_bits_native_width(*bit_count);
                    let mut expr = format!(
                        "{{ let mut dst_bytes = [0u8; {storage_bytes}]; \
                        xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut dst_bytes); \
                        let decoded = {native_type}::from_ne_bytes(dst_bytes); "
                    );
                    if *bit_count == 0 {
                        expr.push_str(
                            "assert!(decoded == 0, \"AOT decode overflow: value does not fit in 0 bits\"); ",
                        );
                    } else if *bit_count < native_width {
                        expr.push_str(&format!(
                            "assert!((decoded >> {bit_count}) == 0, \"AOT decode overflow: value does not fit in {bit_count} bits\"); "
                        ));
                    }
                    expr.push_str("decoded }");
                    expr
                }
            } else {
                let byte_count = bit_count.div_ceil(8);
                let mut expr = format!(
                    "{{ let mut dst_bytes = [0u8; {byte_count}]; \
                    xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut dst_bytes); "
                );
                let bit_remainder = bit_count % 8;
                if bit_remainder != 0 {
                    let last_byte_index = bit_count.div_ceil(8) - 1;
                    expr.push_str(&format!(
                        "assert!((dst_bytes[{last_byte_index}] >> {bit_remainder}) == 0, \"AOT decode overflow: value does not fit in {bit_count} bits\"); "
                    ));
                }
                expr.push_str("dst_bytes }");
                expr
            }
        }
        ResolvedType::Token => {
            format!(
                "{{ let mut dst_bytes = [0u8; 0]; \
                xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut dst_bytes); \
                Token {{}} }}"
            )
        }
        ResolvedType::Tuple { name, fields } => {
            if fields.is_empty() {
                return format!("{name} {{}}");
            }
            let mut field_entries = Vec::with_capacity(fields.len());
            let mut offset = 0usize;
            for (index, field) in fields.iter().enumerate() {
                let field_leaf_base = if offset == 0 {
                    leaf_index_expr.to_string()
                } else {
                    format!("{leaf_index_expr} + {offset}")
                };
                field_entries.push(format!(
                    "field{index}: {}",
                    render_decode_expr(
                        field,
                        layout_name,
                        src_name,
                        &field_leaf_base,
                        next_loop_index
                    )
                ));
                offset = offset.saturating_add(leaf_count(field));
            }
            format!("{name} {{ {} }}", field_entries.join(", "))
        }
        ResolvedType::Array { size: _, element } => {
            let element_leaves = leaf_count(element);
            let loop_name = format!("index_{}", *next_loop_index);
            *next_loop_index += 1;
            let element_leaf_base = if element_leaves == 0 {
                leaf_index_expr.to_string()
            } else if element_leaves == 1 {
                format!("{leaf_index_expr} + {loop_name}")
            } else {
                format!("{leaf_index_expr} + {loop_name} * {element_leaves}")
            };
            let element_expr = render_decode_expr(
                element,
                layout_name,
                src_name,
                &element_leaf_base,
                next_loop_index,
            );
            format!("std::array::from_fn(|{loop_name}| {{ {element_expr} }})")
        }
    }
}

fn leaf_count(ty: &ResolvedType) -> usize {
    match ty {
        ResolvedType::Bits { .. } => 1,
        ResolvedType::Token => 1,
        ResolvedType::Tuple { fields, .. } => fields.iter().map(leaf_count).sum(),
        ResolvedType::Array { size, element } => {
            if *size == 0 {
                0
            } else {
                size.saturating_mul(leaf_count(element))
            }
        }
    }
}

fn render_encode_function(index: usize, ty: &ResolvedType, expected_size: usize) -> String {
    let layout_name = format!("INPUT{index}_LAYOUT");
    let mut lines = Vec::new();
    push_line(
        &mut lines,
        "#[allow(clippy::deref_addrof, clippy::explicit_auto_deref, clippy::identity_op)]",
    );
    push_line(
        &mut lines,
        format!("fn encode_input_{index}(_value: &Input{index}, dst: &mut [u8]) {{"),
    );
    push_line(
        &mut lines,
        format!("debug_assert_eq!(dst.len(), {expected_size});"),
    );
    push_line(&mut lines, "dst.fill(0);");
    let mut loop_index = 0usize;
    emit_pack_statements(
        ty,
        "_value",
        &layout_name,
        "dst",
        "0usize",
        &mut lines,
        &mut loop_index,
    );
    let expected_leaves = leaf_count(ty);
    push_line(
        &mut lines,
        format!("debug_assert_eq!({layout_name}.len(), {expected_leaves});"),
    );
    push_line(&mut lines, "}");
    lines.join("\n")
}

fn render_decode_function(ty: &ResolvedType, expected_size: usize) -> String {
    let layout_name = "OUTPUT0_LAYOUT";
    let mut lines = Vec::new();
    push_line(
        &mut lines,
        "#[allow(clippy::identity_op, clippy::let_and_return)]",
    );
    push_line(&mut lines, "fn decode_output_0(src: &[u8]) -> Output {");
    push_line(
        &mut lines,
        format!("debug_assert_eq!(src.len(), {expected_size});"),
    );
    let mut loop_index = 0usize;
    let decode_expr = render_decode_expr(ty, layout_name, "src", "0usize", &mut loop_index);
    push_line(&mut lines, format!("let output: Output = {decode_expr};"));
    let expected_leaves = leaf_count(ty);
    push_line(
        &mut lines,
        format!("debug_assert_eq!({layout_name}.len(), {expected_leaves});"),
    );
    push_line(&mut lines, "output");
    push_line(&mut lines, "}");
    lines.join("\n")
}

fn validate_signature_and_layouts(
    metadata: &AotEntrypointMetadata,
    signature: &AotFunctionSignature,
) -> AotResult<()> {
    if signature.params.len() != metadata.input_buffer_sizes.len() {
        return Err(XlsynthError(format!(
            "AOT metadata mismatch: parameter count={} but input buffer count={}",
            signature.params.len(),
            metadata.input_buffer_sizes.len()
        )));
    }
    if signature.input_layouts.len() != metadata.input_buffer_sizes.len() {
        return Err(XlsynthError(format!(
            "AOT metadata mismatch: input layout count={} but input buffer count={}",
            signature.input_layouts.len(),
            metadata.input_buffer_sizes.len()
        )));
    }
    if signature.output_layouts.len() != metadata.output_buffer_sizes.len() {
        return Err(XlsynthError(format!(
            "AOT metadata mismatch: output layout count={} but output buffer count={}",
            signature.output_layouts.len(),
            metadata.output_buffer_sizes.len()
        )));
    }
    if signature.output_layouts.len() != 1 {
        return Err(XlsynthError(format!(
            "AOT generated typed wrapper currently expects exactly 1 output; got {}",
            signature.output_layouts.len()
        )));
    }
    for (index, (layout, size)) in signature
        .input_layouts
        .iter()
        .zip(metadata.input_buffer_sizes.iter())
        .enumerate()
    {
        if layout.size != *size {
            return Err(XlsynthError(format!(
                "AOT metadata mismatch for input {index}: layout size={} buffer size={size}",
                layout.size
            )));
        }
    }
    for (index, (layout, size)) in signature
        .output_layouts
        .iter()
        .zip(metadata.output_buffer_sizes.iter())
        .enumerate()
    {
        if layout.size != *size {
            return Err(XlsynthError(format!(
                "AOT metadata mismatch for output {index}: layout size={} buffer size={size}",
                layout.size
            )));
        }
    }
    Ok(())
}

fn make_unique_argument_names(signature: &AotFunctionSignature) -> Vec<String> {
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(signature.params.len());
    for (index, param) in signature.params.iter().enumerate() {
        let base = sanitize_value_identifier(&param.name, &format!("arg{index}"));
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while !used.insert(candidate.clone()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        out.push(candidate);
    }
    out
}

/// A DSLX enum member as generated Rust code needs to name and encode it.
///
/// `value` is already rendered as a signed or unsigned scalar literal according
/// to the enum underlying type. The encode and decode generators use this
/// value when translating between the Rust bridge enum and the AOT ABI bits.
#[derive(Debug, Clone)]
struct TypedDslxEnumVariant {
    /// Rust enum variant identifier emitted by the DSLX bridge.
    name: String,
    /// Scalar discriminant literal used in generated match arms.
    value: String,
}

/// A DSLX struct field paired with the lowered semantic type of that field.
///
/// Field order is the original DSLX declaration order. The AOT ABI for structs
/// is structural, so the order here is the order used when flattening fields to
/// leaf buffers and when reconstructing the generated Rust struct.
#[derive(Debug, Clone)]
struct TypedDslxField {
    /// Rust field identifier emitted by the DSLX bridge.
    name: String,
    /// Lowered semantic type for this field.
    ty: TypedDslxType,
}

/// A DSLX semantic type that can be mapped to the current AOT ABI model.
///
/// Each variant carries the Rust bridge type spelling used in generated public
/// signatures plus the structural facts needed to validate and traverse the
/// AOT layout. Types that cannot flatten to bits, enum-underlying bits,
/// structs, or fixed arrays are rejected during lowering.
#[derive(Debug, Clone)]
enum TypedDslxType {
    /// A DSLX bits-like type represented by `IrUBits<N>` or `IrSBits<N>`.
    Bits {
        /// Rust bridge type path used in generated signatures and helpers.
        rust_type: String,
        /// Whether generated conversion should use signed scalar semantics.
        is_signed: bool,
        /// Number of payload bits in the AOT ABI leaf.
        bit_count: usize,
    },
    /// A DSLX enum represented by a Rust bridge enum and underlying bits.
    Enum {
        /// Rust bridge enum path used in generated signatures and match arms.
        rust_type: String,
        /// Whether the enum's underlying bits are signed.
        is_signed: bool,
        /// Width of the enum's underlying bits.
        bit_count: usize,
        /// Enum variants and their scalar discriminants.
        variants: Vec<TypedDslxEnumVariant>,
    },
    /// A DSLX struct represented by a Rust bridge struct.
    Struct {
        /// Rust bridge struct path used in generated signatures and literals.
        rust_type: String,
        /// Struct fields in DSLX declaration order.
        fields: Vec<TypedDslxField>,
    },
    /// A fixed-size DSLX array represented by a Rust array.
    Array {
        /// Rust bridge array type spelling used in generated signatures.
        rust_type: String,
        /// Number of array elements.
        size: usize,
        /// Lowered semantic type for each element.
        element: Box<TypedDslxType>,
    },
}

/// A typed DSLX function parameter ready for generated runner emission.
///
/// `rust_type` is kept beside `ty` because the public runner signature needs
/// the caller-facing alias or imported path, while `ty` is the recursive shape
/// used for ABI validation and buffer traversal.
#[derive(Debug, Clone)]
struct TypedDslxParam {
    /// DSLX parameter name, sanitized later for generated Rust arguments.
    name: String,
    /// Rust bridge type spelling used in the generated public signature.
    rust_type: String,
    /// Lowered semantic type for this parameter.
    ty: TypedDslxType,
}

/// One concrete parametric struct definition that typed AOT must materialize.
///
/// Rust bridge generation normally emits concrete parametric structs when the
/// defining module itself references them. Typed AOT needs a package-level view
/// so direct imported instantiations can still be emitted in the defining
/// module even when only another module mentions them.
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
/// This is the semantic counterpart to `AotFunctionSignature`: it preserves
/// Rust bridge names while still carrying enough structure to prove that the
/// DSLX type boundary matches the compiled AOT metadata.
#[derive(Debug, Clone)]
struct TypedAotFunctionSignature {
    /// Parameters in DSLX function order.
    params: Vec<TypedDslxParam>,
    /// Rust bridge return type spelling used in the generated public signature.
    return_rust_type: String,
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
/// The context contains the requested bridge modules and the top module in one
/// search space so nested fields and imported annotations can resolve to the
/// correct `TypeInfo` and generated Rust module path.
struct TypedDslxTypeContext {
    modules: Vec<TypedDslxModuleContext>,
}

/// Typechecking result for all DSLX modules participating in one AOT wrapper.
///
/// Bridge modules are emitted before the top module so their public type
/// definitions are available to the generated runner. The top module owns the
/// function selected as the AOT entrypoint.
struct TypedDslxTypecheckedModules {
    bridge_modules: Vec<dslx::TypecheckedModule>,
    top_module: dslx::TypecheckedModule,
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

impl TypedDslxType {
    fn rust_type(&self) -> &str {
        match self {
            TypedDslxType::Bits { rust_type, .. }
            | TypedDslxType::Enum { rust_type, .. }
            | TypedDslxType::Struct { rust_type, .. }
            | TypedDslxType::Array { rust_type, .. } => rust_type,
        }
    }
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
/// generated object/proto/wrapper artifacts. The roots are the top DSLX module
/// and every generated bridge module, then imports are followed through the
/// same search roots that DSLX conversion uses.
fn collect_typed_dslx_aot_dependencies(
    spec: &TypedDslxAotBuildSpec<'_>,
) -> AotResult<BTreeSet<PathBuf>> {
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
    let default_stdlib_path = Path::new(xlsynth_sys::DSLX_STDLIB_PATH);
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
/// The explicit name is needed for bridge modules discovered through import
/// roots: their file name alone is not enough to preserve the dotted DSLX
/// module path that later maps to nested Rust modules.
fn parse_dslx_text_as_module(
    dslx_text: &str,
    path: &Path,
    module_name: &str,
    import_data: &mut dslx::ImportData,
) -> AotResult<dslx::TypecheckedModule> {
    let path_str = path.to_str().ok_or_else(|| {
        XlsynthError(format!(
            "AOT build environment error: DSLX path is not UTF-8: {}",
            path.display()
        ))
    })?;
    dslx::parse_and_typecheck(dslx_text, path_str, module_name, import_data)
}

/// Parses a DSLX source file using the module name implied by its path.
///
/// This is used for the top module because its file path is the public build
/// spec input, so `dslx_path_to_module_name` is the same convention used by
/// the DSLX-to-IR lowering path.
fn parse_dslx_file(
    dslx_text: &str,
    path: &Path,
    import_data: &mut dslx::ImportData,
) -> AotResult<dslx::TypecheckedModule> {
    let module_name = dslx_path_to_module_name(path)?;
    parse_dslx_text_as_module(dslx_text, path, module_name, import_data)
}

/// Computes the canonical DSLX module name for an imported source path.
///
/// If the file is below an additional search root, the relative path becomes a
/// dotted module name such as `foo.widget`. Otherwise the file stem fallback
/// matches normal top-module handling.
fn dslx_module_name_from_import_path(path: &Path, search_paths: &[&Path]) -> AotResult<String> {
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

/// Typechecks the bridge modules and top module in one import-data context.
///
/// Sharing `ImportData` keeps definitions and `TypeInfo` objects comparable
/// across modules. Typechecking bridge modules first also gives the generated
/// Rust module renderer a stable ordering for public type definitions.
fn typecheck_typed_dslx_modules(
    spec: &TypedDslxAotBuildSpec<'_>,
    top_dslx_text: &str,
) -> AotResult<TypedDslxTypecheckedModules> {
    let mut import_data = dslx::ImportData::new(
        spec.dslx_options.dslx_stdlib_path,
        &spec.dslx_options.additional_search_paths,
    );
    let mut bridge_modules = Vec::with_capacity(spec.type_module_paths.len());
    for bridge_path in &spec.type_module_paths {
        let bridge_text = std::fs::read_to_string(bridge_path).map_err(|e| {
            XlsynthError(format!("AOT I/O failed for {}: {e}", bridge_path.display()))
        })?;
        let module_name = dslx_module_name_from_import_path(
            bridge_path,
            &spec.dslx_options.additional_search_paths,
        )?;
        bridge_modules.push(parse_dslx_text_as_module(
            &bridge_text,
            bridge_path,
            &module_name,
            &mut import_data,
        )?);
    }
    let top_module = parse_dslx_file(top_dslx_text, spec.dslx_path, &mut import_data)?;
    Ok(TypedDslxTypecheckedModules {
        bridge_modules,
        top_module,
    })
}

fn ensure_package_specs_compatible(specs: &[TypedDslxAotBuildSpec<'_>]) -> AotResult<()> {
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

fn typecheck_typed_dslx_package_modules(
    specs: &[TypedDslxAotBuildSpec<'_>],
) -> AotResult<TypedDslxPackageTypecheckedModules> {
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
                &canonical_path,
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
            modules.push(TypedDslxPackageModule {
                canonical_path,
                typechecked: parse_dslx_file(&dslx_text, spec.dslx_path, &mut import_data)?,
            });
        }
    }

    Ok(TypedDslxPackageTypecheckedModules { modules })
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
    /// Builds lookup state from all typechecked modules in wrapper emission
    /// order.
    ///
    /// The top module is included after bridge modules so callers can omit it
    /// from `type_module_paths` while still allowing return and parameter types
    /// declared in the top file.
    fn new(typechecked: &TypedDslxTypecheckedModules) -> Self {
        Self::from_modules(
            typechecked
                .bridge_modules
                .iter()
                .chain(std::iter::once(&typechecked.top_module)),
        )
    }

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

    /// Resolves one type-reference annotation to the RHS annotation of a DSLX
    /// type alias, when the reference names an alias rather than a concrete
    /// type.
    fn type_alias_rhs_for_annotation<'a>(
        &'a self,
        current_type_info: &'a dslx::TypeInfo,
        type_annotation: &dslx::TypeAnnotation,
    ) -> AotResult<Option<ResolvedDslxTypeAnnotation<'a>>> {
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
    ) -> AotResult<Option<ResolvedDslxTypeAnnotation<'a>>> {
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
    ) -> AotResult<Option<&TypedDslxModuleContext>> {
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
    ) -> AotResult<Option<&TypedDslxModuleContext>> {
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
    ) -> AotResult<Option<&TypedDslxModuleContext>> {
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
    ) -> AotResult<&'a dslx::TypeInfo> {
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
    ) -> AotResult<&'a dslx::TypeInfo> {
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
    ) -> AotResult<&'a dslx::TypeInfo> {
        let Some(module) = self.defining_module_for_enum(enum_def)? else {
            return Ok(current);
        };
        Ok(&module.type_info)
    }

    /// Renders the Rust bridge type path from a typechecked DSLX type alone.
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
    ) -> AotResult<String> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let signed_str = if is_signed { "S" } else { "U" };
            Ok(format!("Ir{signed_str}Bits<{bit_count}>"))
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
            Err(XlsynthError(format!(
                "AOT typed DSLX type lowering does not support DSLX type `{}`",
                ty.to_string()?
            )))
        }
    }

    /// Renders the Rust bridge type path from source spelling plus concrete
    /// type.
    ///
    /// In this context, resolved means the caller has both the optional source
    /// `TypeAnnotation` and the typechecked `dslx::Type`. The annotation is
    /// checked first so public signatures preserve aliases and imported paths;
    /// the concrete type fallback is used when the annotation cannot name a
    /// bridge type directly, such as synthesized array element types.
    fn rust_type_path_for_resolved_type(
        &self,
        local_module_name: &str,
        current_type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> AotResult<String> {
        if type_annotation.is_some() {
            return RustBridgeBuilder::rust_type_name_from_dslx_module(
                local_module_name,
                current_type_info,
                type_annotation,
                ty,
            );
        }
        self.rust_type_for_concrete_type(local_module_name, current_type_info, ty)
    }
}

struct TypedConcreteParametricStructCollector<'a> {
    context: &'a TypedDslxTypeContext,
    current_module_name: String,
    structs: Vec<TypedConcreteParametricStruct>,
}

impl<'a> TypedConcreteParametricStructCollector<'a> {
    fn new(context: &'a TypedDslxTypeContext) -> Self {
        Self {
            context,
            current_module_name: String::new(),
            structs: Vec::new(),
        }
    }

    fn collect_type(
        &mut self,
        current_module_name: &str,
        current_type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> AotResult<()> {
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
                    let rust_name = RustBridgeBuilder::rust_type_name_from_dslx_module(
                        &defining_module.dslx_name,
                        &defining_module.type_info,
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

    fn into_structs(self) -> Vec<TypedConcreteParametricStruct> {
        self.structs
    }
}

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
        _members: &[(String, crate::IrValue)],
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_struct_def(
        &mut self,
        _dslx_name: &str,
        _members: &[crate::dslx_bridge::StructMemberData],
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_struct_def_typed(
        &mut self,
        _dslx_name: &str,
        type_info: &dslx::TypeInfo,
        members: &[crate::dslx_bridge::StructMemberData],
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
        _ir_value: &crate::IrValue,
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_function_signature_typed(
        &mut self,
        _dslx_name: &str,
        type_info: &dslx::TypeInfo,
        params: &[crate::dslx_bridge::FunctionParamData],
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

fn collect_typed_concrete_parametric_structs(
    context: &TypedDslxTypeContext,
    typechecked: &TypedDslxTypecheckedModules,
) -> AotResult<Vec<TypedConcreteParametricStruct>> {
    collect_typed_concrete_parametric_structs_from_modules(
        context,
        typechecked
            .bridge_modules
            .iter()
            .chain(std::iter::once(&typechecked.top_module)),
    )
}

fn collect_typed_concrete_parametric_structs_from_modules<'a>(
    context: &TypedDslxTypeContext,
    modules: impl IntoIterator<Item = &'a dslx::TypecheckedModule>,
) -> AotResult<Vec<TypedConcreteParametricStruct>> {
    let mut collector = TypedConcreteParametricStructCollector::new(context);
    for module in modules {
        convert_imported_module(module, &mut collector)?;
    }
    Ok(collector.into_structs())
}

fn render_typed_concrete_parametric_struct(
    concrete_struct: &TypedConcreteParametricStruct,
) -> String {
    let mut lines = vec![
        "#[allow(non_camel_case_types)]".to_string(),
        "#[derive(Debug, Clone, PartialEq, Eq)]".to_string(),
        format!("pub struct {} {{", concrete_struct.rust_name),
    ];
    lines.extend(
        concrete_struct
            .fields
            .iter()
            .map(|field| format!("    pub {}: {},", field.name, field.ty.rust_type())),
    );
    lines.push("}\n".to_string());
    lines.join("\n")
}

/// Lowers one DSLX type into the semantic model used by typed AOT generation.
///
/// The caller supplies the Rust bridge type spelling for the outer type so
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
) -> AotResult<TypedDslxType> {
    if let Some((is_signed, bit_count)) = ty.is_bits_like() {
        Ok(TypedDslxType::Bits {
            rust_type,
            is_signed,
            bit_count,
        })
    } else if ty.is_enum() {
        let enum_def = ty.get_enum_def()?;
        let enum_type_info = context.type_info_for_enum(current_type_info, &enum_def)?;
        let underlying = enum_type_info
            .get_type_for_type_annotation(&enum_def.get_underlying())
            .ok_or_else(|| {
                XlsynthError(format!(
                    "AOT typed DSLX type lowering could not resolve underlying type for enum `{}`",
                    enum_def.get_identifier()
                ))
            })?;
        let (is_signed, bit_count) = underlying.is_bits_like().ok_or_else(|| {
            XlsynthError(format!(
                "AOT typed DSLX type lowering expected enum `{}` to have bits-like underlying type",
                enum_def.get_identifier()
            ))
        })?;
        let variants = (0..enum_def.get_member_count())
            .map(|index| {
                let member = enum_def.get_member(index);
                let value = enum_type_info
                    .get_const_expr(&member.get_value())?
                    .convert_to_ir()?;
                let value = if is_signed {
                    value.to_i64()?.to_string()
                } else {
                    value.to_u64()?.to_string()
                };
                Ok(TypedDslxEnumVariant {
                    name: member.get_name(),
                    value,
                })
            })
            .collect::<AotResult<Vec<_>>>()?;
        Ok(TypedDslxType::Enum {
            rust_type,
            is_signed,
            bit_count,
            variants,
        })
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
            .collect::<AotResult<Vec<_>>>()?;
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
            rust_type,
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
        Err(XlsynthError(format!(
            "AOT typed DSLX type lowering does not support DSLX type `{}`",
            ty.to_string()?
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
) -> AotResult<dslx::Function> {
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
/// The result preserves public Rust bridge names and the recursive semantic
/// shape for every parameter and the return value. A return annotation is
/// required because the generated runner needs an explicit Rust return type.
fn build_typed_dslx_function_signature(
    context: &TypedDslxTypeContext,
    top_module: &dslx::TypecheckedModule,
    top: &str,
    rust_module_name: &str,
) -> AotResult<TypedAotFunctionSignature> {
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
                rust_type: rust_type.clone(),
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
        .collect::<AotResult<Vec<_>>>()?;

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
        return_rust_type,
        return_type: typed_dslx_return_type,
    })
}

/// Returns the Rust type spelling stored on a lowered DSLX type.
///
/// Generated helper functions use this spelling in their signatures while the
/// recursive variant data drives the actual leaf packing or decoding work.
fn typed_dslx_rust_type_name(ty: &TypedDslxType) -> &str {
    match ty {
        TypedDslxType::Bits { rust_type, .. }
        | TypedDslxType::Enum { rust_type, .. }
        | TypedDslxType::Struct { rust_type, .. }
        | TypedDslxType::Array { rust_type, .. } => rust_type,
    }
}

/// Counts the number of AOT ABI leaves produced by one lowered DSLX type.
///
/// Bits and enums each occupy one leaf. Structs concatenate field leaves in
/// declaration order, and arrays repeat the element leaf layout for every
/// element.
fn typed_dslx_leaf_count(ty: &TypedDslxType) -> usize {
    match ty {
        TypedDslxType::Bits { .. } | TypedDslxType::Enum { .. } => 1,
        TypedDslxType::Struct { fields, .. } => fields
            .iter()
            .map(|field| typed_dslx_leaf_count(&field.ty))
            .sum(),
        TypedDslxType::Array { size, element, .. } => {
            size.saturating_mul(typed_dslx_leaf_count(element))
        }
    }
}

/// Converts a lowered typed DSLX type into the structural AOT metadata shape.
///
/// This is the validation bridge between the DSLX semantic model and the
/// compiled entrypoint metadata. It intentionally drops Rust names because AOT
/// metadata only knows bits, tuples, and arrays.
fn flatten_typed_dslx_type_to_aot_type(ty: &TypedDslxType) -> AotType {
    match ty {
        TypedDslxType::Bits { bit_count, .. } | TypedDslxType::Enum { bit_count, .. } => {
            AotType::Bits {
                bit_count: *bit_count,
            }
        }
        TypedDslxType::Struct { fields, .. } => AotType::Tuple {
            elements: fields
                .iter()
                .map(|field| flatten_typed_dslx_type_to_aot_type(&field.ty))
                .collect(),
        },
        TypedDslxType::Array { size, element, .. } => AotType::Array {
            size: *size,
            element: Box::new(flatten_typed_dslx_type_to_aot_type(element)),
        },
    }
}

/// Verifies that one typed DSLX boundary type matches the compiled AOT ABI.
///
/// The label is included in diagnostics so callers can distinguish parameter,
/// field, and return mismatches without inspecting generated source.
fn validate_typed_dslx_type_matches_aot(
    label: &str,
    typed_dslx_type: &TypedDslxType,
    aot: &AotType,
) -> AotResult<()> {
    let flattened = flatten_typed_dslx_type_to_aot_type(typed_dslx_type);
    if flattened == *aot {
        Ok(())
    } else {
        Err(XlsynthError(format!(
            "AOT typed DSLX type mismatch for {label}: DSLX semantic type flattens to {flattened:?}, but AOT metadata has {aot:?}"
        )))
    }
}

/// Verifies that a typed DSLX function signature matches AOT metadata.
///
/// This check is the last line of defense before generating a public `Runner`.
/// If the DSLX lowerer and AOT compiler disagree on flattening, the build fails
/// instead of emitting a wrapper that would mispack buffers at runtime.
fn validate_typed_dslx_function_matches_aot(
    typed_signature: &TypedAotFunctionSignature,
    signature: &AotFunctionSignature,
) -> AotResult<()> {
    if typed_signature.params.len() != signature.params.len() {
        return Err(XlsynthError(format!(
            "AOT typed DSLX type mismatch: DSLX parameter count={} but AOT metadata parameter count={}",
            typed_signature.params.len(),
            signature.params.len()
        )));
    }
    for (index, (param, aot_param)) in typed_signature
        .params
        .iter()
        .zip(signature.params.iter())
        .enumerate()
    {
        validate_typed_dslx_type_matches_aot(
            &format!("input {index} `{}`", param.name),
            &param.ty,
            &aot_param.ty,
        )?;
    }
    validate_typed_dslx_type_matches_aot(
        "return",
        &typed_signature.return_type,
        &signature.return_type,
    )
}

/// Appends generated statements that pack a typed DSLX value into AOT leaves.
///
/// The generated statements write little-endian leaf bytes at offsets described
/// by entrypoint metadata layouts. Struct and array traversal must stay in lock
/// step with `typed_dslx_leaf_count` and `flatten_typed_dslx_type_to_aot_type`.
fn emit_typed_dslx_pack_statements(
    ty: &TypedDslxType,
    value_expr: &str,
    layout_name: &str,
    dst_name: &str,
    leaf_index_expr: &str,
    lines: &mut Vec<String>,
    next_loop_index: &mut usize,
) {
    match ty {
        TypedDslxType::Bits { .. } => {
            push_line(
                lines,
                format!("let encoded_bytes = ({value_expr}).to_le_bytes()?;"),
            );
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &encoded_bytes);"
                ),
            );
        }
        TypedDslxType::Enum {
            rust_type,
            is_signed,
            bit_count,
            variants,
        } => {
            let scalar_type = if *is_signed { "i64" } else { "u64" };
            push_line(
                lines,
                format!("let encoded_value: {scalar_type} = match {value_expr} {{"),
            );
            for variant in variants {
                push_line(
                    lines,
                    format!("    {rust_type}::{} => {},", variant.name, variant.value),
                );
            }
            push_line(lines, "};");
            let ir_bits_wrapper_name = if *is_signed { "IrSBits" } else { "IrUBits" };
            let constructor = if *is_signed { "from_i64" } else { "from_u64" };
            push_line(
                lines,
                format!(
                    "let encoded_bits = xlsynth::{ir_bits_wrapper_name}::<{bit_count}>::{constructor}(encoded_value)?;"
                ),
            );
            push_line(lines, "let encoded_bytes = encoded_bits.to_le_bytes()?;");
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &encoded_bytes);"
                ),
            );
        }
        TypedDslxType::Struct { fields, .. } => {
            let mut offset = 0usize;
            for field in fields {
                let field_leaf_base = if offset == 0 {
                    leaf_index_expr.to_string()
                } else {
                    format!("{leaf_index_expr} + {offset}")
                };
                emit_typed_dslx_pack_statements(
                    &field.ty,
                    &format!("({value_expr}).{}", field.name),
                    layout_name,
                    dst_name,
                    &field_leaf_base,
                    lines,
                    next_loop_index,
                );
                offset = offset.saturating_add(typed_dslx_leaf_count(&field.ty));
            }
        }
        TypedDslxType::Array { size, element, .. } => {
            let element_leaves = typed_dslx_leaf_count(element);
            if *size == 0 || element_leaves == 0 {
                return;
            }
            let loop_name = format!("index_{}", *next_loop_index);
            *next_loop_index += 1;
            push_line(lines, format!("for {loop_name} in 0..{size} {{"));
            let element_leaf_base = if element_leaves == 1 {
                format!("{leaf_index_expr} + {loop_name}")
            } else {
                format!("{leaf_index_expr} + {loop_name} * {element_leaves}")
            };
            emit_typed_dslx_pack_statements(
                element,
                &format!("({value_expr})[{loop_name}]"),
                layout_name,
                dst_name,
                &element_leaf_base,
                lines,
                next_loop_index,
            );
            push_line(lines, "}");
        }
    }
}

/// Renders one generated input encoder function for a typed DSLX parameter.
///
/// The encoder zeroes the destination buffer before writing leaves so padding
/// bytes in the ABI buffer do not retain stale contents between runner calls.
fn render_typed_dslx_encode_function(
    index: usize,
    ty: &TypedDslxType,
    expected_size: usize,
) -> String {
    let layout_name = format!("INPUT{index}_LAYOUT");
    let mut lines = Vec::new();
    push_line(
        &mut lines,
        "#[allow(clippy::deref_addrof, clippy::explicit_auto_deref, clippy::identity_op)]",
    );
    push_line(
        &mut lines,
        format!(
            "fn encode_input_{index}(_value: &{}, dst: &mut [u8]) -> Result<(), xlsynth::XlsynthError> {{",
            typed_dslx_rust_type_name(ty)
        ),
    );
    push_line(
        &mut lines,
        format!("debug_assert_eq!(dst.len(), {expected_size});"),
    );
    push_line(&mut lines, "dst.fill(0);");
    let mut loop_index = 0usize;
    emit_typed_dslx_pack_statements(
        ty,
        "*_value",
        &layout_name,
        "dst",
        "0usize",
        &mut lines,
        &mut loop_index,
    );
    let expected_leaves = typed_dslx_leaf_count(ty);
    push_line(
        &mut lines,
        format!("debug_assert_eq!({layout_name}.len(), {expected_leaves});"),
    );
    push_line(&mut lines, "Ok(())");
    push_line(&mut lines, "}");
    lines.join("\n")
}

/// Allocates a unique temporary identifier for generated decode code.
///
/// The prefix encodes the temporary's role for readability, while the counter
/// avoids accidental shadowing across recursive struct and array decoding.
fn next_temp(prefix: &str, next_temp_index: &mut usize) -> String {
    let name = format!("{prefix}_{}", *next_temp_index);
    *next_temp_index += 1;
    name
}

/// Appends generated statements that decode AOT leaves into a typed DSLX value.
///
/// The returned string is the generated Rust expression or temporary name that
/// holds the decoded value. Structs and arrays recursively decode their leaves
/// before constructing the Rust bridge value expected by the public runner API.
fn emit_typed_dslx_decode_statements(
    ty: &TypedDslxType,
    layout_name: &str,
    src_name: &str,
    leaf_index_expr: &str,
    lines: &mut Vec<String>,
    next_loop_index: &mut usize,
    next_temp_index: &mut usize,
) -> String {
    match ty {
        TypedDslxType::Bits {
            is_signed,
            bit_count,
            ..
        } => {
            let bytes_name = next_temp("decoded_bytes", next_temp_index);
            let value_name = next_temp("decoded_value", next_temp_index);
            let byte_count = bit_count.div_ceil(8);
            push_line(
                lines,
                format!("let mut {bytes_name} = vec![0u8; {byte_count}];"),
            );
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut {bytes_name});"
                ),
            );
            let ir_bits_wrapper_name = if *is_signed { "IrSBits" } else { "IrUBits" };
            push_line(
                lines,
                format!(
                    "let {value_name} = xlsynth::{ir_bits_wrapper_name}::<{bit_count}>::from_le_bytes(&{bytes_name})?;"
                ),
            );
            value_name
        }
        TypedDslxType::Enum {
            rust_type,
            is_signed,
            bit_count,
            variants,
        } => {
            let bytes_name = next_temp("decoded_bytes", next_temp_index);
            let bits_name = next_temp("decoded_bits", next_temp_index);
            let scalar_name = next_temp("decoded_scalar", next_temp_index);
            let value_name = next_temp("decoded_value", next_temp_index);
            let byte_count = bit_count.div_ceil(8);
            push_line(
                lines,
                format!("let mut {bytes_name} = vec![0u8; {byte_count}];"),
            );
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut {bytes_name});"
                ),
            );
            let ir_bits_wrapper_name = if *is_signed { "IrSBits" } else { "IrUBits" };
            let scalar_method = if *is_signed { "to_i64" } else { "to_u64" };
            push_line(
                lines,
                format!(
                    "let {bits_name} = xlsynth::{ir_bits_wrapper_name}::<{bit_count}>::from_le_bytes(&{bytes_name})?;"
                ),
            );
            push_line(
                lines,
                format!("let {scalar_name} = {bits_name}.{scalar_method}()?;"),
            );
            push_line(lines, format!("let {value_name} = match {scalar_name} {{"));
            for variant in variants {
                push_line(
                    lines,
                    format!("    {} => {rust_type}::{},", variant.value, variant.name),
                );
            }
            push_line(
                lines,
                format!(
                    "    value => return Err(xlsynth::XlsynthError(format!(\"AOT decode invalid enum value {{value}} for {rust_type}\"))),"
                ),
            );
            push_line(lines, "};");
            value_name
        }
        TypedDslxType::Struct { rust_type, fields } => {
            let field_values = fields
                .iter()
                .scan(0usize, |offset, field| {
                    let field_leaf_base = if *offset == 0 {
                        leaf_index_expr.to_string()
                    } else {
                        format!("{leaf_index_expr} + {}", *offset)
                    };
                    *offset = offset.saturating_add(typed_dslx_leaf_count(&field.ty));
                    Some((
                        field.name.clone(),
                        emit_typed_dslx_decode_statements(
                            &field.ty,
                            layout_name,
                            src_name,
                            &field_leaf_base,
                            lines,
                            next_loop_index,
                            next_temp_index,
                        ),
                    ))
                })
                .collect::<Vec<_>>();
            let value_name = next_temp("decoded_value", next_temp_index);
            push_line(lines, format!("let {value_name} = {rust_type} {{"));
            for (field_name, field_value) in field_values {
                push_line(lines, format!("    {field_name}: {field_value},"));
            }
            push_line(lines, "};");
            value_name
        }
        TypedDslxType::Array {
            rust_type,
            size,
            element,
        } => {
            let value_name = next_temp("decoded_value", next_temp_index);
            if *size == 0 {
                push_line(lines, format!("let {value_name}: {rust_type} = [];"));
                return value_name;
            }
            let items_name = next_temp("decoded_items", next_temp_index);
            let loop_name = format!("index_{}", *next_loop_index);
            *next_loop_index += 1;
            let element_leaves = typed_dslx_leaf_count(element);
            push_line(
                lines,
                format!("let mut {items_name} = Vec::with_capacity({size});"),
            );
            push_line(lines, format!("for {loop_name} in 0..{size} {{"));
            let element_leaf_base = if element_leaves == 1 {
                format!("{leaf_index_expr} + {loop_name}")
            } else {
                format!("{leaf_index_expr} + {loop_name} * {element_leaves}")
            };
            let element_value = emit_typed_dslx_decode_statements(
                element,
                layout_name,
                src_name,
                &element_leaf_base,
                lines,
                next_loop_index,
                next_temp_index,
            );
            push_line(lines, format!("{items_name}.push({element_value});"));
            push_line(lines, "}");
            push_line(
                lines,
                format!(
                    "let {value_name}: {rust_type} = match std::convert::TryInto::try_into({items_name}) {{"
                ),
            );
            push_line(lines, "    Ok(value) => value,");
            push_line(
                lines,
                format!(
                    "    Err(values) => return Err(xlsynth::XlsynthError(format!(\"AOT decode internal error: expected {size} array elements, got {{}}\", values.len()))),"
                ),
            );
            push_line(lines, "};");
            value_name
        }
    }
}

/// Renders the generated output decoder for the typed DSLX return value.
///
/// The decoder mirrors the input encoders: it reads bytes according to the AOT
/// output layout and reconstructs the Rust bridge type promised by the public
/// `Runner` signature.
fn render_typed_dslx_decode_function(ty: &TypedDslxType, expected_size: usize) -> String {
    let layout_name = "OUTPUT0_LAYOUT";
    let mut lines = Vec::new();
    push_line(
        &mut lines,
        "#[allow(clippy::deref_addrof, clippy::explicit_auto_deref, clippy::identity_op)]",
    );
    push_line(
        &mut lines,
        format!(
            "fn decode_output_0(src: &[u8]) -> Result<{}, xlsynth::XlsynthError> {{",
            typed_dslx_rust_type_name(ty)
        ),
    );
    push_line(
        &mut lines,
        format!("debug_assert_eq!(src.len(), {expected_size});"),
    );
    let mut loop_index = 0usize;
    let mut temp_index = 0usize;
    let value_name = emit_typed_dslx_decode_statements(
        ty,
        layout_name,
        "src",
        "0usize",
        &mut lines,
        &mut loop_index,
        &mut temp_index,
    );
    let expected_leaves = typed_dslx_leaf_count(ty);
    push_line(
        &mut lines,
        format!("debug_assert_eq!({layout_name}.len(), {expected_leaves});"),
    );
    push_line(&mut lines, format!("Ok({value_name})"));
    push_line(&mut lines, "}");
    lines.join("\n")
}

/// Produces unique generated argument names for the typed runner methods.
///
/// DSLX parameter names can collide with each other after Rust identifier
/// sanitization, or with Rust keywords. The generated names preserve readable
/// stems while guaranteeing the `run` and `run_with_events` signatures compile.
fn make_unique_typed_dslx_argument_names(params: &[TypedDslxParam]) -> Vec<String> {
    let mut used = HashSet::new();
    params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let base = sanitize_value_identifier(&param.name, &format!("arg{index}"));
            let mut candidate = base.clone();
            let mut suffix = 1usize;
            while !used.insert(candidate.clone()) {
                suffix += 1;
                candidate = format!("{base}_{suffix}");
            }
            candidate
        })
        .collect()
}

/// Renders the generated runner items inserted into the top DSLX bridge module.
///
/// The epilogue owns the linked symbol declaration, descriptor, typed
/// encode/decode helpers, and `Runner` API. Keeping it inside the top module
/// lets public signatures name bridge types without an adapter layer or extra
/// reexports.
fn render_typed_dslx_runner_epilogue(
    base_name: &str,
    proto_file_name: &str,
    metadata: &AotEntrypointMetadata,
    signature: &AotFunctionSignature,
    typed_signature: &TypedAotFunctionSignature,
) -> AotResult<String> {
    validate_signature_and_layouts(metadata, signature)?;
    validate_typed_dslx_function_matches_aot(typed_signature, signature)?;

    let link_symbol_literal = format!("{:?}", metadata.symbol);
    let symbol_ident = format!("__xlsynth_aot_linked_symbol_{base_name}");
    let input_sizes = format_usize_array(&metadata.input_buffer_sizes);
    let input_alignments = format_usize_array(&metadata.input_buffer_alignments);
    let output_sizes = format_usize_array(&metadata.output_buffer_sizes);
    let output_alignments = format_usize_array(&metadata.output_buffer_alignments);
    let input_layout_constants = render_layout_constants("INPUT", &signature.input_layouts);
    let output_layout_constants = render_layout_constants("OUTPUT", &signature.output_layouts);

    let mut helper_blocks = Vec::new();
    for (index, param) in typed_signature.params.iter().enumerate() {
        helper_blocks.push(render_typed_dslx_encode_function(
            index,
            &param.ty,
            metadata.input_buffer_sizes[index],
        ));
    }
    helper_blocks.push(render_typed_dslx_decode_function(
        &typed_signature.return_type,
        metadata.output_buffer_sizes[0],
    ));
    let helper_functions = helper_blocks.join("\n\n");

    let arg_names = make_unique_typed_dslx_argument_names(&typed_signature.params);
    let run_params = typed_signature
        .params
        .iter()
        .zip(arg_names.iter())
        .map(|(param, name)| format!("{name}: &{}", param.rust_type))
        .collect::<Vec<_>>()
        .join(", ");
    let run_signature = if run_params.is_empty() {
        "&mut self".to_string()
    } else {
        format!("&mut self, {run_params}")
    };

    let mut encode_body = String::new();
    for (index, name) in arg_names.iter().enumerate() {
        encode_body.push_str(&format!(
            "        encode_input_{index}({name}, self.inner.input_mut({index}))?;\n"
        ));
    }

    Ok(format!(
        "unsafe extern \"C\" {{\n\
    #[link_name = {link_symbol_literal}]\n\
    fn {symbol_ident}();\n\
}}\n\
\n\
const ENTRYPOINTS_PROTO: &[u8] = include_bytes!(\"{proto_file_name}\");\n\
const INPUT_BUFFER_SIZES: &[usize] = {input_sizes};\n\
const INPUT_BUFFER_ALIGNMENTS: &[usize] = {input_alignments};\n\
const OUTPUT_BUFFER_SIZES: &[usize] = {output_sizes};\n\
const OUTPUT_BUFFER_ALIGNMENTS: &[usize] = {output_alignments};\n\
\n\
{input_layout_constants}\
{output_layout_constants}\
\n\
{helper_functions}\n\
\n\
pub fn descriptor() -> xlsynth::AotEntrypointDescriptor<'static> {{\n\
    unsafe {{\n\
        xlsynth::AotEntrypointDescriptor::from_raw_parts_unchecked(\n\
            ENTRYPOINTS_PROTO,\n\
            {symbol_ident} as *const () as usize,\n\
            xlsynth::AotEntrypointMetadata {{\n\
                symbol: {link_symbol_literal}.to_string(),\n\
                input_buffer_sizes: INPUT_BUFFER_SIZES.to_vec(),\n\
                input_buffer_alignments: INPUT_BUFFER_ALIGNMENTS.to_vec(),\n\
                output_buffer_sizes: OUTPUT_BUFFER_SIZES.to_vec(),\n\
                output_buffer_alignments: OUTPUT_BUFFER_ALIGNMENTS.to_vec(),\n\
                temp_buffer_size: {temp_size},\n\
                temp_buffer_alignment: {temp_align},\n\
            }},\n\
        )\n\
    }}\n\
}}\n\
\n\
/// Reusable runner for the generated typed DSLX AOT entrypoint.\n\
///\n\
/// A runner caches the ABI buffers owned by `xlsynth::AotRunner`; create one\n\
/// runner per concurrent caller instead of sharing it across threads.\n\
pub struct Runner {{\n\
    inner: xlsynth::AotRunner<'static>,\n\
}}\n\
\n\
impl Runner {{\n\
    /// # Errors\n\
    ///\n\
    /// Returns an error if the descriptor metadata cannot initialize an AOT\n\
    /// runner.\n\
    pub fn new() -> Result<Self, xlsynth::XlsynthError> {{\n\
        Ok(Self {{\n\
            inner: xlsynth::AotRunner::new(descriptor())?,\n\
        }})\n\
    }}\n\
\n\
    /// Runs the entrypoint and returns the output together with trace/assert events.\n\
    ///\n\
    /// # Errors\n\
    ///\n\
    /// Returns an error if input packing, AOT execution, or output decoding\n\
    /// fails.\n\
    pub fn run_with_events({run_signature}) -> Result<xlsynth::AotRunResult<{return_type}>, xlsynth::XlsynthError> {{\n\
{encode_body}\
        let result = self.inner.run_with_events(|inner| decode_output_0(inner.output(0)))?;\n\
        Ok(xlsynth::AotRunResult {{\n\
            output: result.output?,\n\
            trace_messages: result.trace_messages,\n\
            assert_messages: result.assert_messages,\n\
        }})\n\
    }}\n\
\n\
    /// Runs the entrypoint and returns only the decoded output value.\n\
    ///\n\
    /// # Errors\n\
    ///\n\
    /// Returns an error if input packing, AOT execution, or output decoding\n\
    /// fails.\n\
    pub fn run({run_signature}) -> Result<{return_type}, xlsynth::XlsynthError> {{\n\
{encode_body}\
        self.inner.run()?;\n\
        decode_output_0(self.inner.output(0))\n\
    }}\n\
}}\n\
\n\
pub fn new_runner() -> Result<Runner, xlsynth::XlsynthError> {{\n\
    Runner::new()\n\
}}\n",
        return_type = typed_signature.return_rust_type.as_str(),
        temp_size = metadata.temp_buffer_size,
        temp_align = metadata.temp_buffer_alignment,
    ))
}

/// Renders the complete generated Rust module for a typed DSLX AOT wrapper.
///
/// Bridge modules are rendered first, then the top module is rendered with the
/// AOT runner epilogue appended. The generated tree is one include-able Rust
/// source file for build scripts to place under Cargo's output directory.
fn render_typed_dslx_generated_module(
    spec: &TypedDslxAotBuildSpec<'_>,
    top_dslx_text: &str,
    base_name: &str,
    proto_file_name: &str,
    metadata: &AotEntrypointMetadata,
    signature: &AotFunctionSignature,
) -> AotResult<String> {
    let typechecked = typecheck_typed_dslx_modules(spec, top_dslx_text)?;
    let context = TypedDslxTypeContext::new(&typechecked);
    let top_module_name = typechecked.top_module.get_module().get_name();
    let typed_signature = build_typed_dslx_function_signature(
        &context,
        &typechecked.top_module,
        spec.top,
        &top_module_name,
    )?;
    let runner_epilogue = render_typed_dslx_runner_epilogue(
        base_name,
        proto_file_name,
        metadata,
        signature,
        &typed_signature,
    )?;
    let concrete_parametric_structs =
        collect_typed_concrete_parametric_structs(&context, &typechecked)?;
    let mut leading_items_by_module =
        concrete_parametric_structs
            .into_iter()
            .fold(BTreeMap::new(), |mut items, item| {
                items
                    .entry(item.defining_module_name.clone())
                    .or_insert_with(Vec::new)
                    .push(render_typed_concrete_parametric_struct(&item));
                items
            });

    let mut modules = Vec::with_capacity(spec.type_module_paths.len() + 1);
    for bridge_module in &typechecked.bridge_modules {
        let module_name = bridge_module.get_module().get_name();
        let mut builder = RustBridgeBuilder::new()
            .with_leading_items(
                leading_items_by_module
                    .remove(&module_name)
                    .unwrap_or_default(),
            )
            .with_deferred_parametric_struct_emission();
        convert_imported_module(bridge_module, &mut builder)?;
        modules.push(builder.module_fragment());
    }

    let mut top_builder = RustBridgeBuilder::new()
        .with_leading_items(
            leading_items_by_module
                .remove(&top_module_name)
                .unwrap_or_default(),
        )
        .with_deferred_parametric_struct_emission()
        .with_runner_items(runner_epilogue);
    convert_imported_module(&typechecked.top_module, &mut top_builder)?;
    modules.push(top_builder.module_fragment());
    if let Some((module_name, _)) = leading_items_by_module.into_iter().next() {
        return Err(XlsynthError(format!(
            "AOT typed DSLX specialization collection requires bridge module `{module_name}` to be emitted"
        )));
    }

    Ok(format!(
        "// SPDX-License-Identifier: Apache-2.0\n// Generated by xlsynth::aot_builder from DSLX build spec {:?}.\n\n{}\n",
        spec.name,
        render_rust_module_fragments(modules)
    ))
}

fn emit_typed_dslx_aot_package_with_out_dir(
    builder: &TypedDslxAotPackageBuilder<'_>,
    out_dir: &Path,
) -> AotResult<GeneratedTypedDslxAotPackage> {
    if builder.name.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: typed DSLX package name must not be empty".to_string(),
        ));
    }
    ensure_package_specs_compatible(&builder.specs)?;

    let package_name = sanitize_identifier(builder.name);
    let mut seen_entrypoint_names = BTreeSet::new();
    let mut compiled = Vec::with_capacity(builder.specs.len());
    for spec in &builder.specs {
        let entrypoint = compile_typed_dslx_entrypoint_artifacts(spec, out_dir)?;
        if !seen_entrypoint_names.insert(entrypoint.base_name.clone()) {
            return Err(XlsynthError(format!(
                "AOT invalid argument: typed DSLX package contains duplicate entrypoint name `{}`",
                entrypoint.base_name
            )));
        }
        compiled.push(entrypoint);
    }

    let typechecked = typecheck_typed_dslx_package_modules(&builder.specs)?;
    let context = TypedDslxTypeContext::from_modules(
        typechecked.modules.iter().map(|module| &module.typechecked),
    );
    let concrete_parametric_structs = collect_typed_concrete_parametric_structs_from_modules(
        &context,
        typechecked.modules.iter().map(|module| &module.typechecked),
    )?;
    let mut leading_items_by_module =
        concrete_parametric_structs
            .into_iter()
            .fold(BTreeMap::new(), |mut items, item| {
                items
                    .entry(item.defining_module_name.clone())
                    .or_insert_with(Vec::new)
                    .push(render_typed_concrete_parametric_struct(&item));
                items
            });

    let mut modules = Vec::with_capacity(typechecked.modules.len() + compiled.len());
    for module in &typechecked.modules {
        let module_name = module.typechecked.get_module().get_name();
        let mut builder = RustBridgeBuilder::new()
            .with_leading_items(
                leading_items_by_module
                    .remove(&module_name)
                    .unwrap_or_default(),
            )
            .with_deferred_parametric_struct_emission();
        convert_imported_module(&module.typechecked, &mut builder)?;
        modules.push(builder.module_fragment());
    }
    if let Some((module_name, _)) = leading_items_by_module.into_iter().next() {
        return Err(XlsynthError(format!(
            "AOT typed DSLX specialization collection requires package module `{module_name}` to be emitted"
        )));
    }

    for entrypoint in &compiled {
        let canonical_top_path = std::fs::canonicalize(entrypoint.spec.dslx_path).map_err(|e| {
            XlsynthError(format!(
                "AOT I/O failed while resolving DSLX package top {}: {e}",
                entrypoint.spec.dslx_path.display()
            ))
        })?;
        let top_module = typechecked
            .modules
            .iter()
            .find(|module| module.canonical_path == canonical_top_path)
            .map(|module| &module.typechecked)
            .ok_or_else(|| {
                XlsynthError(format!(
                    "AOT typed DSLX package could not find top module for {}",
                    entrypoint.spec.dslx_path.display()
                ))
            })?;
        let top_module_name = top_module.get_module().get_name();
        let runner_module_name = format!("{top_module_name}.aot_{}", entrypoint.base_name);
        let typed_signature = build_typed_dslx_function_signature(
            &context,
            top_module,
            entrypoint.spec.top,
            &runner_module_name,
        )?;
        let runner_epilogue = render_typed_dslx_runner_epilogue(
            &entrypoint.base_name,
            &entrypoint.proto_file_name,
            &entrypoint.metadata,
            &entrypoint.signature,
            &typed_signature,
        )?;
        let mut runner_builder = RustBridgeBuilder::new()
            .with_leading_items(["use super::*;".to_string()])
            .with_runner_items(runner_epilogue);
        runner_builder.start_module(&runner_module_name)?;
        runner_builder.end_module(&runner_module_name)?;
        modules.push(runner_builder.module_fragment());
    }

    let rust_file = out_dir.join(format!("{package_name}_typed_dslx_aot_package.rs"));
    let generated = format!(
        "// SPDX-License-Identifier: Apache-2.0\n// Generated by xlsynth::aot_builder from typed DSLX AOT package {:?}.\n\n{}\n",
        builder.name,
        render_rust_module_fragments(modules)
    );
    write_file(&rust_file, generated.as_bytes())?;
    run_rustfmt_best_effort(&rust_file);

    Ok(GeneratedTypedDslxAotPackage {
        name: package_name,
        rust_file,
        entrypoints: compiled
            .into_iter()
            .map(|entrypoint| GeneratedTypedDslxAotEntrypoint {
                name: entrypoint.base_name,
                object_file: entrypoint.object_file,
                entrypoints_proto_file: entrypoint.proto_file,
                metadata: entrypoint.metadata,
            })
            .collect(),
    })
}

/// Renders the generated Rust wrapper source for one compiled AOT entrypoint.
fn render_generated_module(
    base_name: &str,
    proto_file_name: &str,
    metadata: &AotEntrypointMetadata,
    signature: &AotFunctionSignature,
    source_name: &str,
    top: &str,
) -> AotResult<String> {
    validate_signature_and_layouts(metadata, signature)?;

    let link_symbol_literal = format!("{:?}", metadata.symbol);
    let symbol_ident = format!("__xlsynth_aot_linked_symbol_{base_name}");
    let input_sizes = format_usize_array(&metadata.input_buffer_sizes);
    let input_alignments = format_usize_array(&metadata.input_buffer_alignments);
    let output_sizes = format_usize_array(&metadata.output_buffer_sizes);
    let output_alignments = format_usize_array(&metadata.output_buffer_alignments);

    let mut resolver = TypeResolver::default();
    let input_types = signature
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| resolver.lower_type(&param.ty, &format!("Input{index}")))
        .collect::<Vec<_>>();
    let output_type = resolver.lower_type(&signature.return_type, "Output");
    let type_declarations = render_type_declarations(&resolver, &input_types, &output_type);

    let input_layout_constants = render_layout_constants("INPUT", &signature.input_layouts);
    let output_layout_constants = render_layout_constants("OUTPUT", &signature.output_layouts);

    let mut helper_blocks = Vec::new();
    for (index, input_type) in input_types.iter().enumerate() {
        helper_blocks.push(render_encode_function(
            index,
            input_type,
            metadata.input_buffer_sizes[index],
        ));
    }
    helper_blocks.push(render_decode_function(
        &output_type,
        metadata.output_buffer_sizes[0],
    ));
    let helper_functions = helper_blocks.join("\n\n");

    let arg_names = make_unique_argument_names(signature);
    let run_params = arg_names
        .iter()
        .enumerate()
        .map(|(index, name)| format!("{name}: &Input{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let run_signature = if run_params.is_empty() {
        "&mut self".to_string()
    } else {
        format!("&mut self, {run_params}")
    };

    let mut run_body = String::new();
    let mut run_with_events_body = String::new();
    for (index, name) in arg_names.iter().enumerate() {
        run_body.push_str(&format!(
            "        encode_input_{index}({name}, self.inner.input_mut({index}));\n"
        ));
        run_with_events_body.push_str(&format!(
            "        encode_input_{index}({name}, self.inner.input_mut({index}));\n"
        ));
    }
    run_body.push_str("        self.inner.run()?;\n");
    run_body.push_str("        let output = decode_output_0(self.inner.output(0));\n");
    run_body.push_str("        Ok(output)\n");

    run_with_events_body.push_str("        self.inner.run_with_events(|inner| {\n");
    run_with_events_body.push_str("            let output = decode_output_0(inner.output(0));\n");
    run_with_events_body.push_str("            output\n");
    run_with_events_body.push_str("        })\n");

    Ok(format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
// Generated by xlsynth::aot_builder from build spec {source_name:?} (top={top:?}, function={function_name:?}).\n\
\n\
unsafe extern \"C\" {{\n\
    #[link_name = {link_symbol_literal}]\n\
    fn {symbol_ident}();\n\
}}\n\
\n\
const ENTRYPOINTS_PROTO: &[u8] = include_bytes!(\"{proto_file_name}\");\n\
const INPUT_BUFFER_SIZES: &[usize] = {input_sizes};\n\
const INPUT_BUFFER_ALIGNMENTS: &[usize] = {input_alignments};\n\
const OUTPUT_BUFFER_SIZES: &[usize] = {output_sizes};\n\
const OUTPUT_BUFFER_ALIGNMENTS: &[usize] = {output_alignments};\n\
\n\
{type_declarations}\
{input_layout_constants}\
{output_layout_constants}\
\n\
{helper_functions}\n\
\n\
pub fn descriptor() -> xlsynth::AotEntrypointDescriptor<'static> {{\n\
    unsafe {{\n\
        xlsynth::AotEntrypointDescriptor::from_raw_parts_unchecked(\n\
            ENTRYPOINTS_PROTO,\n\
            {symbol_ident} as *const () as usize,\n\
            xlsynth::AotEntrypointMetadata {{\n\
                symbol: {link_symbol_literal}.to_string(),\n\
                input_buffer_sizes: INPUT_BUFFER_SIZES.to_vec(),\n\
                input_buffer_alignments: INPUT_BUFFER_ALIGNMENTS.to_vec(),\n\
                output_buffer_sizes: OUTPUT_BUFFER_SIZES.to_vec(),\n\
                output_buffer_alignments: OUTPUT_BUFFER_ALIGNMENTS.to_vec(),\n\
                temp_buffer_size: {temp_size},\n\
                temp_buffer_alignment: {temp_align},\n\
            }},\n\
        )\n\
    }}\n\
}}\n\
\n\
pub struct Runner {{\n\
    inner: xlsynth::AotRunner<'static>,\n\
}}\n\
\n\
impl Runner {{\n\
    pub fn new() -> Result<Self, xlsynth::XlsynthError> {{\n\
        Ok(Self {{\n\
            inner: xlsynth::AotRunner::new(descriptor())?,\n\
        }})\n\
    }}\n\
\n\
    pub fn run_with_events({run_signature}) -> Result<xlsynth::AotRunResult<Output>, xlsynth::XlsynthError> {{\n\
{run_with_events_body}\
    }}\n\
\n\
    pub fn run({run_signature}) -> Result<Output, xlsynth::XlsynthError> {{\n\
{run_body}\
    }}\n\
}}\n\
\n\
pub fn new_runner() -> Result<Runner, xlsynth::XlsynthError> {{\n\
    Runner::new()\n\
}}\n",
        function_name = signature.function_name,
        temp_size = metadata.temp_buffer_size,
        temp_align = metadata.temp_buffer_alignment,
    ))
}

fn emit_link_archive(base_name: &str, object_file: &Path) -> AotResult<()> {
    cc::Build::new()
        .warnings(false)
        .object(object_file)
        .compile(&format!("xlsynth_aot_{base_name}"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aot_entrypoint_metadata::AotType;

    #[test]
    fn sanitize_identifier_rewrites_non_ident_chars() {
        assert_eq!(sanitize_identifier("foo-bar"), "foo_bar");
        assert_eq!(sanitize_identifier("3abc"), "_3abc");
        assert_eq!(sanitize_identifier(""), "aot_entrypoint");
    }

    #[test]
    fn sanitize_value_identifier_handles_keywords() {
        assert_eq!(sanitize_value_identifier("type", "arg"), "type_");
    }

    #[test]
    fn render_type_declarations_do_not_emit_default_impls() {
        let mut resolver = TypeResolver::default();
        let input_ty = resolver.lower_type(
            &AotType::Tuple {
                elements: vec![
                    AotType::Bits { bit_count: 8 },
                    AotType::Array {
                        size: 128,
                        element: Box::new(AotType::Bits { bit_count: 16 }),
                    },
                    AotType::Bits { bit_count: 257 },
                ],
            },
            "Input0",
        );
        let output_ty = resolver.lower_type(&AotType::Tuple { elements: vec![] }, "Output");

        let declarations = render_type_declarations(&resolver, &[input_ty], &output_ty);

        assert!(
            declarations.contains("#[derive(Debug, Clone, PartialEq, Eq)]\npub struct Input0 {")
        );
        assert!(
            declarations.contains("pub field1: [Bits16; 128],"),
            "declarations should preserve large array field types: {}",
            declarations
        );
        assert!(
            declarations.contains("pub type Bits257 = [u8; 33];"),
            "declarations should preserve wide bits byte-array aliases: {}",
            declarations
        );
        assert!(
            !declarations.contains("Default"),
            "generated declarations should not reference Default: {}",
            declarations
        );
    }

    // Negative test: rejects typed DSLX/AOT metadata mismatches.
    #[test]
    fn typed_dslx_type_validation_rejects_aot_metadata_mismatch() {
        let typed_dslx_type = TypedDslxType::Struct {
            rust_type: "ReturnType".to_string(),
            fields: vec![TypedDslxField {
                name: "value".to_string(),
                ty: TypedDslxType::Bits {
                    rust_type: "xlsynth::IrUBits<8>".to_string(),
                    is_signed: false,
                    bit_count: 8,
                },
            }],
        };
        let aot = AotType::Tuple {
            elements: vec![AotType::Bits { bit_count: 16 }],
        };

        let error =
            validate_typed_dslx_type_matches_aot("return", &typed_dslx_type, &aot).unwrap_err();
        assert!(error
            .to_string()
            .contains("AOT typed DSLX type mismatch for return"));
    }

    // Verifies: typed DSLX AOT dependency tracking follows transitive DSLX imports.
    // Catches: build scripts missing rerun-if-changed for imported modules.
    #[test]
    fn typed_dslx_aot_dependencies_follow_transitive_imports() {
        let tmpdir = xlsynth_test_helpers::make_test_tmpdir("xlsynth_aot_builder_dependencies");
        let top_path = tmpdir.path().join("top.x");
        let helper_path = tmpdir.path().join("helper.x");
        let constants_path = tmpdir.path().join("constants.x");
        let bridge_path = tmpdir.path().join("bridge.x");
        std::fs::write(
            &top_path,
            "import helper as h; pub fn frob(x: u8) -> u8 { h::inc(x) }",
        )
        .unwrap();
        std::fs::write(
            &helper_path,
            "import constants; pub fn inc(x: u8) -> u8 { x + constants::ONE }",
        )
        .unwrap();
        std::fs::write(&constants_path, "pub const ONE = u8:1;").unwrap();
        std::fs::write(&bridge_path, "pub struct Widget { value: u8 }").unwrap();

        let dslx_options = DslxConvertOptions {
            additional_search_paths: vec![tmpdir.path()],
            ..Default::default()
        };
        let spec = TypedDslxAotBuildSpec {
            name: "dependencies",
            dslx_path: &top_path,
            top: "frob",
            dslx_options,
            type_module_paths: vec![&bridge_path],
        };

        let dependencies = collect_typed_dslx_aot_dependencies(&spec).unwrap();

        assert_eq!(
            dependencies,
            BTreeSet::from([
                std::fs::canonicalize(&bridge_path).unwrap(),
                std::fs::canonicalize(&constants_path).unwrap(),
                std::fs::canonicalize(&helper_path).unwrap(),
                std::fs::canonicalize(&top_path).unwrap(),
            ])
        );
    }

    // Verifies: duplicate struct names resolve by the exact defining module.
    // Catches: bare-name owner lookup that selects the wrong imported struct.
    #[test]
    fn typed_dslx_type_lowering_uses_struct_definition_owner_when_names_collide() {
        let tmpdir =
            xlsynth_test_helpers::make_test_tmpdir("xlsynth_aot_builder_duplicate_struct_names");
        let a_path = tmpdir.path().join("a.x");
        let b_path = tmpdir.path().join("b.x");
        let top_path = tmpdir.path().join("top.x");
        std::fs::write(&a_path, "pub struct Widget { value: u8 }").unwrap();
        std::fs::write(&b_path, "pub struct Widget { value: u16 }").unwrap();
        std::fs::write(
            &top_path,
            "import a; import b; pub fn frob(widget: a::Widget) -> a::Widget { widget }",
        )
        .unwrap();

        let dslx_options = DslxConvertOptions {
            additional_search_paths: vec![tmpdir.path()],
            ..Default::default()
        };
        let spec = TypedDslxAotBuildSpec {
            name: "duplicate_struct_names",
            dslx_path: &top_path,
            top: "frob",
            dslx_options,
            type_module_paths: vec![&a_path, &b_path],
        };
        let top_dslx_text = std::fs::read_to_string(&top_path).unwrap();
        let typechecked = typecheck_typed_dslx_modules(&spec, &top_dslx_text).unwrap();
        let context = TypedDslxTypeContext::new(&typechecked);

        let typed_signature =
            build_typed_dslx_function_signature(&context, &typechecked.top_module, "frob", "top")
                .expect("duplicate struct names should resolve by defining module");

        assert_eq!(typed_signature.params.len(), 1);
        assert_eq!(typed_dslx_leaf_count(&typed_signature.params[0].ty), 1);
        assert_eq!(typed_dslx_leaf_count(&typed_signature.return_type), 1);
    }

    // Verifies: concrete specializations remain distinct when sibling imports
    // declare same-named parametric structs with the same bound values.
    // Catches: dedupe keyed only by generated Rust name.
    #[test]
    fn concrete_parametric_struct_collection_uses_struct_definition_identity_when_names_collide() {
        let tmpdir = xlsynth_test_helpers::make_test_tmpdir(
            "xlsynth_aot_builder_duplicate_parametric_struct_names",
        );
        let a_path = tmpdir.path().join("a.x");
        let b_path = tmpdir.path().join("b.x");
        let top_path = tmpdir.path().join("top.x");
        std::fs::write(&a_path, "pub struct Widget<N: u32> { value: bits[N] }").unwrap();
        std::fs::write(&b_path, "pub struct Widget<N: u32> { value: bits[N] }").unwrap();
        std::fs::write(
            &top_path,
            "import a; import b; pub fn frob(lhs: a::Widget<u32:8>, rhs: b::Widget<u32:8>) -> (a::Widget<u32:8>, b::Widget<u32:8>) { (lhs, rhs) }",
        )
        .unwrap();

        let dslx_options = DslxConvertOptions {
            additional_search_paths: vec![tmpdir.path()],
            ..Default::default()
        };
        let spec = TypedDslxAotBuildSpec {
            name: "duplicate_parametric_struct_names",
            dslx_path: &top_path,
            top: "frob",
            dslx_options,
            type_module_paths: vec![&a_path, &b_path],
        };
        let top_dslx_text = std::fs::read_to_string(&top_path).unwrap();
        let typechecked = typecheck_typed_dslx_modules(&spec, &top_dslx_text).unwrap();
        let context = TypedDslxTypeContext::new(&typechecked);

        let structs = collect_typed_concrete_parametric_structs(&context, &typechecked)
            .expect("same-named imported parametric structs should remain distinct");

        assert_eq!(structs.len(), 2);
        assert_eq!(
            structs
                .iter()
                .map(|item| (item.defining_module_name.as_str(), item.rust_name.as_str()))
                .collect::<Vec<_>>(),
            vec![("a", "Widget__N_8"), ("b", "Widget__N_8")]
        );
    }
}
