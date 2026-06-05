// SPDX-License-Identifier: Apache-2.0

//! Build-script helpers for emitting standalone native PIR AOT wrappers.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub use xlsynth::DslxConvertOptions;
pub use xlsynth::aot_builder::TypedDslxAotBuildSpec;
use xlsynth::aot_builder::{
    NativeTypedDslxFunctionSignature, NativeTypedDslxType, collect_typed_dslx_aot_dependencies,
    render_native_typed_dslx_generated_module, render_native_typed_dslx_package_generated_module,
};
use xlsynth::{
    DslxCallingConvention, convert_dslx_to_ir_text, dslx_path_to_module_name,
    mangle_dslx_name_with_calling_convention,
};
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler_runtime::{
    AssumptionFailureKind, EventKind, EventSiteMetadata, TraceTupleFieldLayout, TraceValueLayout,
};

use crate::{AotArtifact, CompilerError, NativeValueLayout, compile_aot};

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
    let artifact = compile_aot(function, &entrypoint_symbol)?;

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
    let top = mangle_dslx_name_with_calling_convention(
        dslx_path_to_module_name(spec.dslx_path)
            .map_err(|error| CompilerError::Backend(error.to_string()))?,
        spec.top,
        calling_convention,
    )
    .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let package = Parser::new(&pir_text)
        .parse_and_validate_package()
        .map_err(|error| CompilerError::InvalidFunction(error.to_string()))?;
    let function = package.get_fn(&top).ok_or_else(|| {
        CompilerError::InvalidFunction(format!("PIR package has no function '{top}'"))
    })?;
    let entrypoint_symbol = format!("__xlsynth_pir_aot_{base_name}");
    let artifact = compile_aot(function, &entrypoint_symbol)?;
    Ok((dslx_text, artifact))
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
