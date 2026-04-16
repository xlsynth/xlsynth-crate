// SPDX-License-Identifier: Apache-2.0
//! Build-script helpers that compile XLS IR to AOT artifacts and generate Rust
//! wrappers.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::aot_entrypoint_metadata::{
    get_entrypoint_function_signature, AotEntrypointMetadata, AotFunctionSignature, AotType,
    AotTypeLayout,
};
use crate::aot_lib::{AotCompiled, AotResult};
use crate::dslx_bridge::convert_imported_module;
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
    pub name: &'a str,
    pub ir_text: &'a str,
    pub top: &'a str,
}

/// Inputs required to compile one DSLX function into a pretty-type AOT wrapper.
///
/// The generated module contains Rust bridge definitions for `bridge_paths`,
/// Rust bridge definitions for `dslx_path`, and a `Runner` whose public
/// signature uses canonical paths to those generated Rust bridge types.
pub struct PrettyAotBuildSpec<'a> {
    pub name: &'a str,
    pub dslx_path: &'a Path,
    pub top: &'a str,
    pub dslx_options: DslxConvertOptions<'a>,
    pub bridge_paths: Vec<&'a Path>,
}

#[derive(Debug, Clone)]
/// Paths and metadata for emitted AOT wrapper artifacts in a build output
/// directory.
///
/// The generated Rust wrapper includes typed encode/decode helpers and a thin
/// `Runner` wrapper over `xlsynth::AotRunner`.
pub struct GeneratedAotModule {
    pub name: String,
    pub rust_file: PathBuf,
    pub object_file: PathBuf,
    pub entrypoints_proto_file: PathBuf,
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

/// Emits pretty-type AOT artifacts into Cargo's `OUT_DIR`.
///
/// This DSLX-aware entry point is additive to the IR-only API and preserves
/// the old structural wrapper behavior for existing callers.
pub fn emit_pretty_aot_module_from_dslx_file(
    spec: &PrettyAotBuildSpec<'_>,
) -> AotResult<GeneratedAotModule> {
    let out_dir = std::env::var("OUT_DIR").map_err(|e| {
        XlsynthError(format!(
            "AOT build environment error: OUT_DIR was not set while emitting pretty AOT module: {e}"
        ))
    })?;
    emit_pretty_aot_module_from_dslx_file_with_out_dir(spec, Path::new(&out_dir))
}

/// Emits pretty-type AOT artifacts into an explicit output directory.
///
/// This compiles the DSLX top function, emits bridge definitions for selected
/// DSLX modules, validates those semantic types against AOT metadata, and
/// writes a generated Rust module that encodes/decodes pretty values directly.
pub fn emit_pretty_aot_module_from_dslx_file_with_out_dir(
    spec: &PrettyAotBuildSpec<'_>,
    out_dir: &Path,
) -> AotResult<GeneratedAotModule> {
    if spec.name.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: pretty build spec name must not be empty".to_string(),
        ));
    }
    if spec.top.is_empty() {
        return Err(XlsynthError(
            "AOT invalid argument: pretty build spec top function must not be empty".to_string(),
        ));
    }

    for dslx_source_path in collect_pretty_aot_dslx_dependencies(spec)? {
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
        metadata: selected_metadata,
    } = compile;
    let signature = get_entrypoint_function_signature(&entrypoints_proto)
        .map_err(|e| XlsynthError(format!("AOT metadata parse failed: {}", e.0)))?;

    let object_file = out_dir.join(format!("{base_name}.aot.o"));
    let proto_file = out_dir.join(format!("{base_name}.entrypoints.pb"));
    let rust_file = out_dir.join(format!("{base_name}_pretty_aot_wrapper.rs"));

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

    let generated = render_pretty_generated_module(
        spec,
        &dslx_text,
        &base_name,
        proto_file_name,
        &selected_metadata,
        &signature,
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

#[derive(Debug, Clone)]
struct PrettyEnumVariant {
    name: String,
    value: String,
}

#[derive(Debug, Clone)]
struct PrettyField {
    name: String,
    ty: PrettyType,
}

#[derive(Debug, Clone)]
enum PrettyType {
    Bits {
        rust_type: String,
        is_signed: bool,
        bit_count: usize,
    },
    Enum {
        rust_type: String,
        is_signed: bool,
        bit_count: usize,
        variants: Vec<PrettyEnumVariant>,
    },
    Struct {
        rust_type: String,
        fields: Vec<PrettyField>,
    },
    Array {
        rust_type: String,
        size: usize,
        element: Box<PrettyType>,
    },
}

#[derive(Debug, Clone)]
struct PrettyParam {
    name: String,
    rust_type: String,
    ty: PrettyType,
}

#[derive(Debug, Clone)]
struct PrettyFunctionTypes {
    params: Vec<PrettyParam>,
    return_rust_type: String,
    return_type: PrettyType,
}

struct PrettyStructDefContext {
    def: dslx::StructDef,
}

struct PrettyModuleContext {
    dslx_name: String,
    type_info: dslx::TypeInfo,
    struct_names: BTreeSet<String>,
    struct_defs: Vec<PrettyStructDefContext>,
    enum_names: BTreeSet<String>,
}

struct PrettyTypeContext {
    modules: Vec<PrettyModuleContext>,
}

struct PrettyTypecheckedModules {
    bridge_modules: Vec<dslx::TypecheckedModule>,
    top_module: dslx::TypecheckedModule,
}

/// A non-empty DSLX import subject such as `foo` or `foo.bar`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DslxImportSubject {
    segments: Vec<String>,
}

impl DslxImportSubject {
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

/// Returns the DSLX files whose contents can affect a pretty AOT build.
///
/// Cargo build scripts must report every source file that can change the
/// generated object/proto/wrapper artifacts. The roots are the top DSLX module
/// and every generated bridge module, then imports are followed through the
/// same search roots that DSLX conversion uses.
fn collect_pretty_aot_dslx_dependencies(
    spec: &PrettyAotBuildSpec<'_>,
) -> AotResult<BTreeSet<PathBuf>> {
    let mut dependencies = BTreeSet::new();
    let mut pending_paths = std::iter::once(spec.dslx_path.to_path_buf())
        .chain(spec.bridge_paths.iter().map(|path| path.to_path_buf()))
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

fn dslx_import_subjects(dslx_text: &str) -> Vec<DslxImportSubject> {
    dslx_text
        .lines()
        .flat_map(dslx_import_subjects_from_line)
        .collect()
}

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

fn resolve_dslx_import_path(
    importer_path: &Path,
    subject: &DslxImportSubject,
    spec: &PrettyAotBuildSpec<'_>,
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

fn parse_dslx_file(
    dslx_text: &str,
    path: &Path,
    import_data: &mut dslx::ImportData,
) -> AotResult<dslx::TypecheckedModule> {
    let module_name = dslx_path_to_module_name(path)?;
    parse_dslx_text_as_module(dslx_text, path, module_name, import_data)
}

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

fn typecheck_pretty_modules(
    spec: &PrettyAotBuildSpec<'_>,
    top_dslx_text: &str,
) -> AotResult<PrettyTypecheckedModules> {
    let mut import_data = dslx::ImportData::new(
        spec.dslx_options.dslx_stdlib_path,
        &spec.dslx_options.additional_search_paths,
    );
    let mut bridge_modules = Vec::with_capacity(spec.bridge_paths.len());
    for bridge_path in &spec.bridge_paths {
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
    Ok(PrettyTypecheckedModules {
        bridge_modules,
        top_module,
    })
}

fn collect_module_context(typechecked_module: &dslx::TypecheckedModule) -> PrettyModuleContext {
    let module = typechecked_module.get_module();
    let mut struct_names = BTreeSet::new();
    let mut struct_defs = Vec::new();
    let mut enum_names = BTreeSet::new();
    for index in 0..module.get_member_count() {
        if let Some(member) = module.get_member(index).to_matchable() {
            match member {
                dslx::MatchableModuleMember::StructDef(struct_def) => {
                    let name = struct_def.get_identifier();
                    struct_names.insert(name.clone());
                    struct_defs.push(PrettyStructDefContext { def: struct_def });
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
    PrettyModuleContext {
        dslx_name: module.get_name(),
        type_info: typechecked_module.get_type_info(),
        struct_names,
        struct_defs,
        enum_names,
    }
}

impl PrettyTypeContext {
    fn new(typechecked: &PrettyTypecheckedModules) -> Self {
        let modules = typechecked
            .bridge_modules
            .iter()
            .chain(std::iter::once(&typechecked.top_module))
            .map(collect_module_context)
            .collect();
        Self { modules }
    }

    fn owner_context_for_struct(
        &self,
        current_type_info: Option<&dslx::TypeInfo>,
        struct_def: &dslx::StructDef,
    ) -> AotResult<Option<&PrettyModuleContext>> {
        let struct_name = struct_def.get_identifier();
        let exact_matches = self
            .modules
            .iter()
            .filter(|module| {
                module
                    .struct_defs
                    .iter()
                    .any(|known| known.def.ptr_eq(struct_def))
            })
            .collect::<Vec<_>>();
        match exact_matches.as_slice() {
            [module] => return Ok(Some(module)),
            modules if modules.len() > 1 => {
                return Err(XlsynthError(format!(
                    "AOT pretty type lowering found multiple owner modules for struct `{struct_name}`"
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
                        .find(|module| module.type_info.ptr_eq(current_type_info));
                    if current_match.is_some() {
                        return Ok(current_match);
                    }
                }
                Err(XlsynthError(format!(
                    "AOT pretty type lowering found multiple DSLX structs named `{struct_name}`"
                )))
            }
        }
    }

    fn owner_context_for_enum(
        &self,
        enum_def: &dslx::EnumDef,
    ) -> AotResult<Option<&PrettyModuleContext>> {
        if enum_def.get_member_count() == 0 {
            let enum_name = enum_def.get_identifier();
            return self
                .modules
                .iter()
                .find(|module| module.enum_names.contains(&enum_name))
                .map(Some)
                .ok_or_else(|| {
                    XlsynthError(format!(
                        "AOT pretty type lowering could not find owner module for enum `{enum_name}`"
                    ))
                });
        }
        let owner_name = enum_def
            .get_member(0)
            .get_value()
            .get_owner_module()
            .get_name();
        self.modules
            .iter()
            .find(|module| module.dslx_name == owner_name)
            .map(Some)
            .ok_or_else(|| {
                XlsynthError(format!(
                    "AOT pretty type lowering could not find owner module `{owner_name}` for enum `{}`",
                    enum_def.get_identifier()
                ))
            })
    }

    fn owner_context_for_type_annotation(
        &self,
        type_annotation: &dslx::TypeAnnotation,
    ) -> AotResult<Option<&PrettyModuleContext>> {
        let owner_name = type_annotation
            .to_type_ref_type_annotation()
            .and_then(|annotation| {
                annotation
                    .get_type_ref()
                    .get_type_definition()
                    .to_colon_ref()
                    .and_then(|colon_ref| colon_ref.resolve_import_subject())
            })
            .map(|import| import.get_subject().join("."));
        match owner_name {
            Some(owner_name) => self
                .modules
                .iter()
                .find(|module| module.dslx_name == owner_name)
                .map(Some)
                .ok_or_else(|| {
                    XlsynthError(format!(
                        "AOT pretty type lowering could not find owner module `{owner_name}`"
                    ))
                }),
            None => Ok(None),
        }
    }

    fn type_info_for_type_annotation_or_type<'a>(
        &'a self,
        current: &'a dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> AotResult<&'a dslx::TypeInfo> {
        if let Some(module) = type_annotation
            .map(|annotation| self.owner_context_for_type_annotation(annotation))
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

    fn type_info_for_struct<'a>(
        &'a self,
        current: &'a dslx::TypeInfo,
        struct_def: &dslx::StructDef,
    ) -> AotResult<&'a dslx::TypeInfo> {
        let Some(module) = self.owner_context_for_struct(Some(current), struct_def)? else {
            return Ok(current);
        };
        Ok(&module.type_info)
    }

    fn type_info_for_enum<'a>(
        &'a self,
        current: &'a dslx::TypeInfo,
        enum_def: &dslx::EnumDef,
    ) -> AotResult<&'a dslx::TypeInfo> {
        let Some(module) = self.owner_context_for_enum(enum_def)? else {
            return Ok(current);
        };
        Ok(&module.type_info)
    }

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
            let owner = self.owner_context_for_enum(&enum_def)?;
            Ok(match owner {
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
            let owner = self.owner_context_for_struct(Some(current_type_info), &struct_def)?;
            Ok(match owner {
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
                "AOT pretty type lowering does not support DSLX type `{}`",
                ty.to_string()?
            )))
        }
    }

    fn rust_type_for_type_annotation_or_type(
        &self,
        local_module_name: &str,
        current_type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> AotResult<String> {
        if let Some(rust_type) = type_annotation.and_then(|annotation| {
            RustBridgeBuilder::rust_type_ref_name_from_dslx_module(local_module_name, annotation)
        }) {
            Ok(rust_type)
        } else {
            self.rust_type_for_concrete_type(local_module_name, current_type_info, ty)
        }
    }
}

fn lower_pretty_type(
    context: &PrettyTypeContext,
    local_module_name: &str,
    current_type_info: &dslx::TypeInfo,
    ty: &dslx::Type,
    rust_type: String,
) -> AotResult<PrettyType> {
    if let Some((is_signed, bit_count)) = ty.is_bits_like() {
        Ok(PrettyType::Bits {
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
                    "AOT pretty type lowering could not resolve underlying type for enum `{}`",
                    enum_def.get_identifier()
                ))
            })?;
        let (is_signed, bit_count) = underlying.is_bits_like().ok_or_else(|| {
            XlsynthError(format!(
                "AOT pretty type lowering expected enum `{}` to have bits-like underlying type",
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
                Ok(PrettyEnumVariant {
                    name: member.get_name(),
                    value,
                })
            })
            .collect::<AotResult<Vec<_>>>()?;
        Ok(PrettyType::Enum {
            rust_type,
            is_signed,
            bit_count,
            variants,
        })
    } else if ty.is_struct() {
        let struct_def = ty.get_struct_def()?;
        let struct_type_info = context.type_info_for_struct(current_type_info, &struct_def)?;
        let fields = (0..struct_def.get_member_count())
            .map(|index| {
                let member = struct_def.get_member(index);
                let field_annotation = member.get_type();
                let field_type = struct_type_info.get_type_for_struct_member(&member);
                let field_type_info = context.type_info_for_type_annotation_or_type(
                    struct_type_info,
                    Some(&field_annotation),
                    &field_type,
                )?;
                let rust_type = context.rust_type_for_type_annotation_or_type(
                    local_module_name,
                    field_type_info,
                    Some(&field_annotation),
                    &field_type,
                )?;
                Ok(PrettyField {
                    name: member.get_name(),
                    ty: lower_pretty_type(
                        context,
                        local_module_name,
                        field_type_info,
                        &field_type,
                        rust_type,
                    )?,
                })
            })
            .collect::<AotResult<Vec<_>>>()?;
        Ok(PrettyType::Struct { rust_type, fields })
    } else if ty.is_array() {
        let element = ty.get_array_element_type();
        let element_rust_type =
            context.rust_type_for_concrete_type(local_module_name, current_type_info, &element)?;
        Ok(PrettyType::Array {
            rust_type,
            size: ty.get_array_size(),
            element: Box::new(lower_pretty_type(
                context,
                local_module_name,
                current_type_info,
                &element,
                element_rust_type,
            )?),
        })
    } else {
        Err(XlsynthError(format!(
            "AOT pretty type lowering does not support DSLX type `{}`",
            ty.to_string()?
        )))
    }
}

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
        "AOT pretty type lowering could not find DSLX function `{function_name}`"
    )))
}

fn build_pretty_function_types(
    context: &PrettyTypeContext,
    top_module: &dslx::TypecheckedModule,
    top: &str,
) -> AotResult<PrettyFunctionTypes> {
    let module_name = top_module.get_module().get_name();
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
                        "AOT pretty type lowering could not resolve type for parameter `{}`",
                        param.get_name()
                    ))
                })?;
            let param_type_info = context.type_info_for_type_annotation_or_type(
                &type_info,
                Some(&annotation),
                &concrete_type,
            )?;
            let rust_type = context.rust_type_for_type_annotation_or_type(
                &module_name,
                param_type_info,
                Some(&annotation),
                &concrete_type,
            )?;
            Ok(PrettyParam {
                name: param.get_name(),
                rust_type: rust_type.clone(),
                ty: lower_pretty_type(
                    context,
                    &module_name,
                    param_type_info,
                    &concrete_type,
                    rust_type,
                )?,
            })
        })
        .collect::<AotResult<Vec<_>>>()?;

    let return_annotation = function.get_return_type().ok_or_else(|| {
        XlsynthError(format!(
            "AOT pretty type lowering requires function `{top}` to have an explicit return type"
        ))
    })?;
    let return_type = type_info
        .get_type_for_type_annotation(&return_annotation)
        .ok_or_else(|| {
            XlsynthError(format!(
                "AOT pretty type lowering could not resolve return type for function `{top}`"
            ))
        })?;
    let return_type_info = context.type_info_for_type_annotation_or_type(
        &type_info,
        Some(&return_annotation),
        &return_type,
    )?;
    let return_rust_type = context.rust_type_for_type_annotation_or_type(
        &module_name,
        return_type_info,
        Some(&return_annotation),
        &return_type,
    )?;
    let pretty_return_type = lower_pretty_type(
        context,
        &module_name,
        return_type_info,
        &return_type,
        return_rust_type.clone(),
    )?;
    Ok(PrettyFunctionTypes {
        params,
        return_rust_type,
        return_type: pretty_return_type,
    })
}

fn pretty_rust_type_name(ty: &PrettyType) -> &str {
    match ty {
        PrettyType::Bits { rust_type, .. }
        | PrettyType::Enum { rust_type, .. }
        | PrettyType::Struct { rust_type, .. }
        | PrettyType::Array { rust_type, .. } => rust_type,
    }
}

fn pretty_leaf_count(ty: &PrettyType) -> usize {
    match ty {
        PrettyType::Bits { .. } | PrettyType::Enum { .. } => 1,
        PrettyType::Struct { fields, .. } => fields
            .iter()
            .map(|field| pretty_leaf_count(&field.ty))
            .sum(),
        PrettyType::Array { size, element, .. } => size.saturating_mul(pretty_leaf_count(element)),
    }
}

fn flatten_pretty_type_to_aot_type(ty: &PrettyType) -> AotType {
    match ty {
        PrettyType::Bits { bit_count, .. } | PrettyType::Enum { bit_count, .. } => AotType::Bits {
            bit_count: *bit_count,
        },
        PrettyType::Struct { fields, .. } => AotType::Tuple {
            elements: fields
                .iter()
                .map(|field| flatten_pretty_type_to_aot_type(&field.ty))
                .collect(),
        },
        PrettyType::Array { size, element, .. } => AotType::Array {
            size: *size,
            element: Box::new(flatten_pretty_type_to_aot_type(element)),
        },
    }
}

fn validate_pretty_type_matches_aot(
    label: &str,
    pretty: &PrettyType,
    aot: &AotType,
) -> AotResult<()> {
    let flattened = flatten_pretty_type_to_aot_type(pretty);
    if flattened == *aot {
        Ok(())
    } else {
        Err(XlsynthError(format!(
            "AOT pretty type mismatch for {label}: DSLX semantic type flattens to {flattened:?}, but AOT metadata has {aot:?}"
        )))
    }
}

fn validate_pretty_function_matches_aot(
    pretty: &PrettyFunctionTypes,
    signature: &AotFunctionSignature,
) -> AotResult<()> {
    if pretty.params.len() != signature.params.len() {
        return Err(XlsynthError(format!(
            "AOT pretty type mismatch: DSLX parameter count={} but AOT metadata parameter count={}",
            pretty.params.len(),
            signature.params.len()
        )));
    }
    for (index, (param, aot_param)) in pretty
        .params
        .iter()
        .zip(signature.params.iter())
        .enumerate()
    {
        validate_pretty_type_matches_aot(
            &format!("input {index} `{}`", param.name),
            &param.ty,
            &aot_param.ty,
        )?;
    }
    validate_pretty_type_matches_aot("return", &pretty.return_type, &signature.return_type)
}

fn emit_pretty_pack_statements(
    ty: &PrettyType,
    value_expr: &str,
    layout_name: &str,
    dst_name: &str,
    leaf_index_expr: &str,
    lines: &mut Vec<String>,
    next_loop_index: &mut usize,
) {
    match ty {
        PrettyType::Bits { .. } => {
            push_line(
                lines,
                format!("let encoded_bytes = ({value_expr}).to_bytes()?;"),
            );
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &encoded_bytes);"
                ),
            );
        }
        PrettyType::Enum {
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
            let bits_type = if *is_signed { "IrSBits" } else { "IrUBits" };
            let constructor = if *is_signed { "from_i64" } else { "from_u64" };
            push_line(
                lines,
                format!(
                    "let encoded_bits = xlsynth::{bits_type}::<{bit_count}>::{constructor}(encoded_value)?;"
                ),
            );
            push_line(lines, "let encoded_bytes = encoded_bits.to_bytes()?;");
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &encoded_bytes);"
                ),
            );
        }
        PrettyType::Struct { fields, .. } => {
            let mut offset = 0usize;
            for field in fields {
                let field_leaf_base = if offset == 0 {
                    leaf_index_expr.to_string()
                } else {
                    format!("{leaf_index_expr} + {offset}")
                };
                emit_pretty_pack_statements(
                    &field.ty,
                    &format!("({value_expr}).{}", field.name),
                    layout_name,
                    dst_name,
                    &field_leaf_base,
                    lines,
                    next_loop_index,
                );
                offset = offset.saturating_add(pretty_leaf_count(&field.ty));
            }
        }
        PrettyType::Array { size, element, .. } => {
            let element_leaves = pretty_leaf_count(element);
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
            emit_pretty_pack_statements(
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

fn render_pretty_encode_function(index: usize, ty: &PrettyType, expected_size: usize) -> String {
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
            pretty_rust_type_name(ty)
        ),
    );
    push_line(
        &mut lines,
        format!("debug_assert_eq!(dst.len(), {expected_size});"),
    );
    push_line(&mut lines, "dst.fill(0);");
    let mut loop_index = 0usize;
    emit_pretty_pack_statements(
        ty,
        "*_value",
        &layout_name,
        "dst",
        "0usize",
        &mut lines,
        &mut loop_index,
    );
    let expected_leaves = pretty_leaf_count(ty);
    push_line(
        &mut lines,
        format!("debug_assert_eq!({layout_name}.len(), {expected_leaves});"),
    );
    push_line(&mut lines, "Ok(())");
    push_line(&mut lines, "}");
    lines.join("\n")
}

fn next_temp(prefix: &str, next_temp_index: &mut usize) -> String {
    let name = format!("{prefix}_{}", *next_temp_index);
    *next_temp_index += 1;
    name
}

fn emit_pretty_decode_statements(
    ty: &PrettyType,
    layout_name: &str,
    src_name: &str,
    leaf_index_expr: &str,
    lines: &mut Vec<String>,
    next_loop_index: &mut usize,
    next_temp_index: &mut usize,
) -> String {
    match ty {
        PrettyType::Bits {
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
            let bits_type = if *is_signed { "IrSBits" } else { "IrUBits" };
            push_line(
                lines,
                format!(
                    "let {value_name} = xlsynth::{bits_type}::<{bit_count}>::from_le_bytes(&{bytes_name})?;"
                ),
            );
            value_name
        }
        PrettyType::Enum {
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
            let bits_type = if *is_signed { "IrSBits" } else { "IrUBits" };
            let scalar_method = if *is_signed { "to_i64" } else { "to_u64" };
            push_line(
                lines,
                format!(
                    "let {bits_name} = xlsynth::{bits_type}::<{bit_count}>::from_le_bytes(&{bytes_name})?;"
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
        PrettyType::Struct { rust_type, fields } => {
            let field_values = fields
                .iter()
                .scan(0usize, |offset, field| {
                    let field_leaf_base = if *offset == 0 {
                        leaf_index_expr.to_string()
                    } else {
                        format!("{leaf_index_expr} + {}", *offset)
                    };
                    *offset = offset.saturating_add(pretty_leaf_count(&field.ty));
                    Some((
                        field.name.clone(),
                        emit_pretty_decode_statements(
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
        PrettyType::Array {
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
            let element_leaves = pretty_leaf_count(element);
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
            let element_value = emit_pretty_decode_statements(
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

fn render_pretty_decode_function(ty: &PrettyType, expected_size: usize) -> String {
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
            pretty_rust_type_name(ty)
        ),
    );
    push_line(
        &mut lines,
        format!("debug_assert_eq!(src.len(), {expected_size});"),
    );
    let mut loop_index = 0usize;
    let mut temp_index = 0usize;
    let value_name = emit_pretty_decode_statements(
        ty,
        layout_name,
        "src",
        "0usize",
        &mut lines,
        &mut loop_index,
        &mut temp_index,
    );
    let expected_leaves = pretty_leaf_count(ty);
    push_line(
        &mut lines,
        format!("debug_assert_eq!({layout_name}.len(), {expected_leaves});"),
    );
    push_line(&mut lines, format!("Ok({value_name})"));
    push_line(&mut lines, "}");
    lines.join("\n")
}

fn make_unique_pretty_argument_names(params: &[PrettyParam]) -> Vec<String> {
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

fn render_pretty_runner_epilogue(
    base_name: &str,
    proto_file_name: &str,
    metadata: &AotEntrypointMetadata,
    signature: &AotFunctionSignature,
    pretty: &PrettyFunctionTypes,
    source_name: &str,
    top: &str,
) -> AotResult<String> {
    validate_signature_and_layouts(metadata, signature)?;
    validate_pretty_function_matches_aot(pretty, signature)?;

    let link_symbol_literal = format!("{:?}", metadata.symbol);
    let symbol_ident = format!("__xlsynth_aot_linked_symbol_{base_name}");
    let input_sizes = format_usize_array(&metadata.input_buffer_sizes);
    let input_alignments = format_usize_array(&metadata.input_buffer_alignments);
    let output_sizes = format_usize_array(&metadata.output_buffer_sizes);
    let output_alignments = format_usize_array(&metadata.output_buffer_alignments);
    let input_layout_constants = render_layout_constants("INPUT", &signature.input_layouts);
    let output_layout_constants = render_layout_constants("OUTPUT", &signature.output_layouts);

    let mut helper_blocks = Vec::new();
    for (index, param) in pretty.params.iter().enumerate() {
        helper_blocks.push(render_pretty_encode_function(
            index,
            &param.ty,
            metadata.input_buffer_sizes[index],
        ));
    }
    helper_blocks.push(render_pretty_decode_function(
        &pretty.return_type,
        metadata.output_buffer_sizes[0],
    ));
    let helper_functions = helper_blocks.join("\n\n");

    let arg_names = make_unique_pretty_argument_names(&pretty.params);
    let run_params = pretty
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
        "// Generated pretty AOT runner for build spec {source_name:?} (top={top:?}, function={function_name:?}).\n\
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
        function_name = signature.function_name,
        return_type = pretty.return_rust_type.as_str(),
        temp_size = metadata.temp_buffer_size,
        temp_align = metadata.temp_buffer_alignment,
    ))
}

fn render_pretty_generated_module(
    spec: &PrettyAotBuildSpec<'_>,
    top_dslx_text: &str,
    base_name: &str,
    proto_file_name: &str,
    metadata: &AotEntrypointMetadata,
    signature: &AotFunctionSignature,
) -> AotResult<String> {
    let typechecked = typecheck_pretty_modules(spec, top_dslx_text)?;
    let context = PrettyTypeContext::new(&typechecked);
    let pretty = build_pretty_function_types(&context, &typechecked.top_module, spec.top)?;
    let runner_epilogue = render_pretty_runner_epilogue(
        base_name,
        proto_file_name,
        metadata,
        signature,
        &pretty,
        spec.name,
        spec.top,
    )?;

    let mut modules = Vec::with_capacity(spec.bridge_paths.len() + 1);
    for bridge_module in &typechecked.bridge_modules {
        let mut builder = RustBridgeBuilder::new();
        convert_imported_module(bridge_module, &mut builder)?;
        modules.push(builder.module_fragment());
    }

    let mut top_builder = RustBridgeBuilder::new().with_module_epilogue(runner_epilogue);
    convert_imported_module(&typechecked.top_module, &mut top_builder)?;
    modules.push(top_builder.module_fragment());

    Ok(format!(
        "// SPDX-License-Identifier: Apache-2.0\n// Generated by xlsynth::aot_builder from DSLX build spec {:?}.\n\n{}\n",
        spec.name,
        render_rust_module_fragments(modules)
    ))
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

    #[test]
    fn pretty_type_validation_rejects_aot_metadata_mismatch() {
        let pretty = PrettyType::Struct {
            rust_type: "ReturnType".to_string(),
            fields: vec![PrettyField {
                name: "value".to_string(),
                ty: PrettyType::Bits {
                    rust_type: "xlsynth::IrUBits<8>".to_string(),
                    is_signed: false,
                    bit_count: 8,
                },
            }],
        };
        let aot = AotType::Tuple {
            elements: vec![AotType::Bits { bit_count: 16 }],
        };

        let error = validate_pretty_type_matches_aot("return", &pretty, &aot).unwrap_err();
        assert!(error
            .to_string()
            .contains("AOT pretty type mismatch for return"));
    }

    #[test]
    fn pretty_aot_dependencies_follow_transitive_imports() {
        let tmpdir = xlsynth_test_helpers::make_test_tmpdir("xlsynth_aot_builder_dependencies");
        let top_path = tmpdir.path().join("top.x");
        let helper_path = tmpdir.path().join("helper.x");
        let constants_path = tmpdir.path().join("constants.x");
        let bridge_path = tmpdir.path().join("bridge.x");
        std::fs::write(
            &top_path,
            "import helper as h; pub fn route(x: u8) -> u8 { h::inc(x) }",
        )
        .unwrap();
        std::fs::write(
            &helper_path,
            "import constants; pub fn inc(x: u8) -> u8 { x + constants::ONE }",
        )
        .unwrap();
        std::fs::write(&constants_path, "pub const ONE = u8:1;").unwrap();
        std::fs::write(&bridge_path, "pub struct Packet { value: u8 }").unwrap();

        let dslx_options = DslxConvertOptions {
            additional_search_paths: vec![tmpdir.path()],
            ..Default::default()
        };
        let spec = PrettyAotBuildSpec {
            name: "dependencies",
            dslx_path: &top_path,
            top: "route",
            dslx_options,
            bridge_paths: vec![&bridge_path],
        };

        let dependencies = collect_pretty_aot_dslx_dependencies(&spec).unwrap();

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

    #[test]
    fn pretty_type_lowering_uses_struct_definition_owner_when_names_collide() {
        let tmpdir =
            xlsynth_test_helpers::make_test_tmpdir("xlsynth_aot_builder_duplicate_struct_names");
        let a_path = tmpdir.path().join("a.x");
        let b_path = tmpdir.path().join("b.x");
        let top_path = tmpdir.path().join("top.x");
        std::fs::write(&a_path, "pub struct Packet { value: u8 }").unwrap();
        std::fs::write(&b_path, "pub struct Packet { value: u16 }").unwrap();
        std::fs::write(
            &top_path,
            "import a; import b; pub fn route(packet: a::Packet) -> a::Packet { packet }",
        )
        .unwrap();

        let dslx_options = DslxConvertOptions {
            additional_search_paths: vec![tmpdir.path()],
            ..Default::default()
        };
        let spec = PrettyAotBuildSpec {
            name: "duplicate_struct_names",
            dslx_path: &top_path,
            top: "route",
            dslx_options,
            bridge_paths: vec![&a_path, &b_path],
        };
        let top_dslx_text = std::fs::read_to_string(&top_path).unwrap();
        let typechecked = typecheck_pretty_modules(&spec, &top_dslx_text).unwrap();
        let context = PrettyTypeContext::new(&typechecked);

        let pretty = build_pretty_function_types(&context, &typechecked.top_module, "route")
            .expect("duplicate struct names should resolve by defining module");

        assert_eq!(pretty.params.len(), 1);
        assert_eq!(pretty_leaf_count(&pretty.params[0].ty), 1);
        assert_eq!(pretty_leaf_count(&pretty.return_type), 1);
    }
}
