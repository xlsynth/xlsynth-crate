// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::aot_entrypoint_metadata::{
    get_entrypoint_function_signature, AotEntrypointMetadata, AotFunctionSignature, AotType,
    AotTypeLayout,
};
use crate::aot_lib::{AotCompiled, AotResult};
use crate::xlsynth_error::XlsynthError;

#[derive(Debug, Clone)]
pub struct AotBuildSpec<'a> {
    pub name: &'a str,
    pub ir_text: &'a str,
    pub top: &'a str,
}

#[derive(Debug, Clone)]
pub struct GeneratedAotModule {
    pub name: String,
    pub rust_file: PathBuf,
    pub object_file: PathBuf,
    pub entrypoints_proto_file: PathBuf,
    pub metadata: AotEntrypointMetadata,
}

pub fn compile_ir_to_aot(ir_text: &str, top: &str) -> AotResult<AotCompiled> {
    AotCompiled::compile_ir(ir_text, top)
}

pub fn emit_aot_module_from_ir_text(spec: &AotBuildSpec<'_>) -> AotResult<GeneratedAotModule> {
    let out_dir = std::env::var("OUT_DIR").map_err(|e| {
        XlsynthError(format!(
            "AOT build environment error: OUT_DIR was not set while emitting AOT module: {e}"
        ))
    })?;
    emit_aot_module_from_ir_text_with_out_dir(spec, Path::new(&out_dir))
}

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
    out.push_str("#[derive(Debug, Clone, PartialEq, Eq, Default)]\n");
    out.push_str("pub struct Token {}\n\n");

    for bit_count in &resolver.bit_widths {
        if *bit_count <= 64 {
            out.push_str(&format!("pub type Bits{bit_count} = u64;\n"));
        } else {
            let byte_count = bit_count.div_ceil(8);
            out.push_str(&format!("pub type Bits{bit_count} = [u8; {byte_count}];\n"));
        }
    }
    if !resolver.bit_widths.is_empty() {
        out.push('\n');
    }

    for tuple in &resolver.tuple_defs {
        out.push_str("#[derive(Debug, Clone, PartialEq, Eq, Default)]\n");
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
                push_line(
                    lines,
                    format!(
                        "xlsynth::aot_runner::write_leaf_element({dst_name}, &{layout_name}[{leaf_index_expr}], &({value_expr}).to_ne_bytes());"
                    ),
                );
            } else {
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

fn emit_unpack_statements(
    ty: &ResolvedType,
    value_expr: &str,
    layout_name: &str,
    src_name: &str,
    leaf_index_expr: &str,
    lines: &mut Vec<String>,
    next_loop_index: &mut usize,
) {
    match ty {
        ResolvedType::Bits { bit_count } => {
            if *bit_count <= 64 {
                push_line(lines, "let mut dst_bytes = [0u8; 8];");
                push_line(
                    lines,
                    format!(
                        "xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut dst_bytes);"
                    ),
                );
                push_line(
                    lines,
                    format!("{value_expr} = u64::from_ne_bytes(dst_bytes);"),
                );
            } else {
                push_line(
                    lines,
                    format!("let dst_bytes: &mut [u8] = &mut ({value_expr});"),
                );
                push_line(
                    lines,
                    format!(
                        "xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], dst_bytes);"
                    ),
                );
            }
        }
        ResolvedType::Token => {
            push_line(lines, "let mut dst_bytes = [0u8; 0];");
            push_line(
                lines,
                format!(
                    "xlsynth::aot_runner::read_leaf_element({src_name}, &{layout_name}[{leaf_index_expr}], &mut dst_bytes);"
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
                emit_unpack_statements(
                    field,
                    &format!("({value_expr}).field{index}"),
                    layout_name,
                    src_name,
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
            emit_unpack_statements(
                element,
                &format!("({value_expr})[{loop_name}]"),
                layout_name,
                src_name,
                &element_leaf_base,
                lines,
                next_loop_index,
            );
            push_line(lines, "}");
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
        "fn decode_output_0(src: &[u8], _value: &mut Output) {",
    );
    push_line(
        &mut lines,
        format!("debug_assert_eq!(src.len(), {expected_size});"),
    );
    let mut loop_index = 0usize;
    emit_unpack_statements(
        ty,
        "(*_value)",
        layout_name,
        "src",
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
    run_body.push_str("        let mut output: Output = Default::default();\n");
    run_body.push_str("        decode_output_0(self.inner.output(0), &mut output);\n");
    run_body.push_str("        Ok(output)\n");

    run_with_events_body.push_str("        self.inner.run_with_events(|inner| {\n");
    run_with_events_body.push_str("            let mut output: Output = Default::default();\n");
    run_with_events_body.push_str("            decode_output_0(inner.output(0), &mut output);\n");
    run_with_events_body.push_str("            output\n");
    run_with_events_body.push_str("        })\n");

    Ok(format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
// Generated by xlsynth::aot_builder from build spec {source_name:?} (top={top:?}, function={function_name:?}).\n\
\n\
extern \"C\" {{\n\
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
    xlsynth::AotEntrypointDescriptor {{\n\
        entrypoints_proto: ENTRYPOINTS_PROTO,\n\
        function_ptr: {symbol_ident} as *const () as usize,\n\
        metadata: xlsynth::AotEntrypointMetadata {{\n\
            symbol: {link_symbol_literal}.to_string(),\n\
            input_buffer_sizes: INPUT_BUFFER_SIZES.to_vec(),\n\
            input_buffer_alignments: INPUT_BUFFER_ALIGNMENTS.to_vec(),\n\
            output_buffer_sizes: OUTPUT_BUFFER_SIZES.to_vec(),\n\
            output_buffer_alignments: OUTPUT_BUFFER_ALIGNMENTS.to_vec(),\n\
            temp_buffer_size: {temp_size},\n\
            temp_buffer_alignment: {temp_align},\n\
        }},\n\
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
}
