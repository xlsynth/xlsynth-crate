// SPDX-License-Identifier: Apache-2.0

pub mod dslx;
pub mod dslx_bridge;
pub mod ir_analysis;
pub mod ir_builder;
pub mod ir_package;
pub mod ir_value;
mod lib_support;
pub mod rust_bridge_builder;
pub mod sv_bridge_builder;
pub mod vast;
pub mod vast_helpers;
pub mod vast_helpers_options;
pub mod xlsynth_error;

use std::ffi::CString;
use std::os::raw::c_char;

use ir_package::ScheduleAndCodegenResult;
pub use ir_value::{IrBits, IrSBits, IrUBits};
use lib_support::xls_schedule_and_codegen_package;
use lib_support::{
    c_str_to_rust, c_str_to_rust_no_dealloc, xls_mangle_dslx_name, xls_mangle_dslx_name_full,
    xls_optimize_ir,
};

pub use ir_analysis::{Interval, IntervalSet, IrAnalysis, KnownBits};
pub use ir_builder::BValue;
pub use ir_builder::FnBuilder;
pub use ir_package::IrFunction;
pub use ir_package::IrFunctionJit;
pub use ir_package::IrPackage;
pub use ir_package::IrType;
pub use ir_package::RunResult;
pub use ir_value::IrValue;
pub use xlsynth_error::XlsynthError;
use xlsynth_sys::CIrValue;

pub fn dslx_path_to_module_name(path: &std::path::Path) -> Result<&str, XlsynthError> {
    let stem = path.file_stem();
    match stem {
        None => Err(XlsynthError(
            "Failed to extract module name from path".to_string(),
        )),
        Some(stem) => Ok(stem.to_str().unwrap()),
    }
}

#[derive(Default)]
pub struct DslxConvertOptions<'a> {
    pub dslx_stdlib_path: Option<&'a std::path::Path>,
    pub additional_search_paths: Vec<&'a std::path::Path>,
    pub enable_warnings: Option<&'a [String]>,
    pub disable_warnings: Option<&'a [String]>,
    pub force_implicit_token_calling_convention: bool,
}

pub struct DslxToIrTextResult {
    pub ir: String,
    pub warnings: Vec<String>,
}

struct CStringPtrArray {
    _storage: Vec<CString>,
    pointers: Vec<*const c_char>,
}

impl CStringPtrArray {
    fn from_strings(strings: &[String]) -> Self {
        let c_strings = strings
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect::<Vec<_>>();
        let pointers = c_strings.iter().map(|cstr| cstr.as_ptr()).collect();
        Self {
            _storage: c_strings,
            pointers,
        }
    }

    fn from_optional(strings: Option<&[String]>) -> Option<Self> {
        strings.map(Self::from_strings)
    }

    fn parts(&self) -> (*const *const c_char, usize) {
        if self.pointers.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (self.pointers.as_ptr(), self.pointers.len())
        }
    }
}

fn optional_ptr_parts(array: Option<&CStringPtrArray>) -> (*const *const c_char, usize) {
    array
        .map(|list| list.parts())
        .unwrap_or((std::ptr::null(), 0))
}

/// Converts a DSLX module's source text into an IR package. Returns the IR
/// text.
///
/// `options` allows specification of standard library path and additional
/// search paths where we look for DSLX modules.
pub fn convert_dslx_to_ir_text(
    dslx: &str,
    path: &std::path::Path,
    options: &DslxConvertOptions,
) -> Result<DslxToIrTextResult, XlsynthError> {
    // Extract the module name from the path; e.g. "foo/bar/baz.x" -> "baz"
    let module_name = dslx_path_to_module_name(path)?;
    let path_str = path.to_str().unwrap();
    let stdlib_path = options
        .dslx_stdlib_path
        .unwrap_or_else(|| std::path::Path::new(xlsynth_sys::DSLX_STDLIB_PATH));
    let stdlib_path = stdlib_path.to_str().unwrap();
    let search_paths = options
        .additional_search_paths
        .iter()
        .map(|p| p.to_str().unwrap())
        .collect::<Vec<&str>>();

    let mut search_paths_cstrs = vec![];
    for p in search_paths {
        search_paths_cstrs.push(CString::new(p).unwrap());
    }

    let dslx = CString::new(dslx).unwrap();
    let c_path = CString::new(path_str).unwrap();
    let c_module_name = CString::new(module_name).unwrap();
    let dslx_stdlib_path = CString::new(stdlib_path).unwrap();

    let enable_warnings = CStringPtrArray::from_optional(options.enable_warnings);
    let disable_warnings = CStringPtrArray::from_optional(options.disable_warnings);

    unsafe {
        let additional_search_paths_ptrs: Vec<*const std::os::raw::c_char> = search_paths_cstrs
            .iter()
            .map(|cstr| cstr.as_ptr())
            .collect();

        let (enable_warnings_ptr, enable_warnings_len) =
            optional_ptr_parts(enable_warnings.as_ref());
        let (disable_warnings_ptr, disable_warnings_len) =
            optional_ptr_parts(disable_warnings.as_ref());

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();

        let mut warnings_out: *mut *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut warnings_out_count: usize = 0;

        let force_itok = options.force_implicit_token_calling_convention;

        // Call the function
        let success = xlsynth_sys::xls_convert_dslx_to_ir_with_warnings(
            dslx.as_ptr(),
            c_path.as_ptr(),
            c_module_name.as_ptr(),
            dslx_stdlib_path.as_ptr(),
            additional_search_paths_ptrs.as_ptr(),
            additional_search_paths_ptrs.len(),
            enable_warnings_ptr,
            enable_warnings_len,
            disable_warnings_ptr,
            disable_warnings_len,
            false,
            force_itok,
            &mut warnings_out,
            &mut warnings_out_count,
            &mut error_out,
            &mut ir_out,
        );

        // Some integrity checks for the API contract.
        assert!((warnings_out_count == 0) == warnings_out.is_null());
        assert!(error_out.is_null() == success);
        assert!(success != ir_out.is_null());

        let mut warnings = Vec::new();
        if warnings_out_count > 0 {
            for i in 0..warnings_out_count {
                let warning = *warnings_out.wrapping_add(i);
                warnings.push(c_str_to_rust_no_dealloc(warning));
            }
        }

        xlsynth_sys::xls_c_strs_free(warnings_out, warnings_out_count);

        if success {
            let ir_text = c_str_to_rust(ir_out);
            if ir_text.trim().is_empty() {
                return Err(XlsynthError(format!(
                    "INTERNAL: DSLX to IR conversion returned empty IR text for module '{module_name}' ({path_str})."
                )));
            }
            Ok(DslxToIrTextResult {
                ir: ir_text,
                warnings,
            })
        } else {
            let error_out_str = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub fn xls_parse_typed_value(s: &str) -> Result<IrValue, XlsynthError> {
    unsafe {
        let c_str = CString::new(s).unwrap();
        let mut ir_value_out: *mut CIrValue = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success =
            xlsynth_sys::xls_parse_typed_value(c_str.as_ptr(), &mut error_out, &mut ir_value_out);
        if success {
            Ok(IrValue { ptr: ir_value_out })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub struct DslxToIrPackageResult {
    pub ir: IrPackage,
    pub warnings: Vec<String>,
}

/// Converts DSLX source text into an IR package.
pub fn convert_dslx_to_ir(
    dslx: &str,
    path: &std::path::Path,
    options: &DslxConvertOptions,
) -> Result<DslxToIrPackageResult, XlsynthError> {
    let convert_result = convert_dslx_to_ir_text(dslx, path, options)?;
    // Get the filename as an Option<&str>
    let filename = path.file_name().and_then(|s| s.to_str());
    let ir = IrPackage::parse_ir(&convert_result.ir, filename)?;
    Ok(DslxToIrPackageResult {
        ir,
        warnings: convert_result.warnings,
    })
}

/// Optimizes an IR package -- this produces a new IR package with the optimized
/// IR contents.
pub fn optimize_ir(ir: &IrPackage, top: &str) -> Result<IrPackage, XlsynthError> {
    let ir_text = xls_optimize_ir(&ir.to_string(), top)?;
    IrPackage::parse_ir(&ir_text, ir.filename())
}

pub fn schedule_and_codegen(
    ir: &IrPackage,
    scheduling_options_flags_proto: &str,
    codegen_flags_proto: &str,
) -> Result<ScheduleAndCodegenResult, XlsynthError> {
    let guard = ir.ptr.write().unwrap();
    xls_schedule_and_codegen_package(
        &ir.ptr,
        guard,
        scheduling_options_flags_proto,
        codegen_flags_proto,
        false,
    )
}

pub fn mangle_dslx_name(module: &str, name: &str) -> Result<String, XlsynthError> {
    xls_mangle_dslx_name(module, name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Indicates the calling convention used by a DSLX function for mangling.
pub enum DslxCallingConvention {
    /// Typical DSLX function (no implicit token parameter in the mangled name).
    Typical,
    /// DSLX function that has an implicit token parameter; mangled name is
    /// prefixed with `__itok`.
    ImplicitToken,
    /// DSLX proc-next convention (e.g. used for state machine next functions).
    ProcNext,
}

impl From<DslxCallingConvention> for xlsynth_sys::XlsCallingConvention {
    fn from(value: DslxCallingConvention) -> Self {
        match value {
            DslxCallingConvention::Typical => xlsynth_sys::XLS_CALLING_CONVENTION_TYPICAL,
            DslxCallingConvention::ImplicitToken => {
                xlsynth_sys::XLS_CALLING_CONVENTION_IMPLICIT_TOKEN
            }
            DslxCallingConvention::ProcNext => xlsynth_sys::XLS_CALLING_CONVENTION_PROC_NEXT,
        }
    }
}

/// Mangles a DSLX function name according to the given calling convention.
/// This wraps `mangle_dslx_name` and applies the implicit-token prefix when
/// `DslxCallingConvention::ImplicitToken` is specified.
pub fn mangle_dslx_name_with_calling_convention(
    module: &str,
    name: &str,
    cc: DslxCallingConvention,
) -> Result<String, XlsynthError> {
    // Delegate to the full-featured mangler-with-env with no
    // free-keys/parametrics/scope.
    mangle_dslx_name_with_env(module, name, cc, &[], None, None)
}

/// Full-featured DSLX name mangling API variant that accepts a prebuilt
/// ParametricEnv.
///
/// - free_keys: Parametric identifier names whose bound values should be
///   encoded into the mangled name. Every key must be present in `env`; pass an
///   empty slice when the target is non-parametric.
/// - scope: Optional scope qualifier inserted between the module and function
///   names (e.g. an impl/struct name like "Point"). Use `None` for no scope.
pub fn mangle_dslx_name_with_env(
    module: &str,
    name: &str,
    convention: DslxCallingConvention,
    free_keys: &[&str],
    env: Option<&dslx::ParametricEnv>,
    scope: Option<&str>,
) -> Result<String, XlsynthError> {
    xls_mangle_dslx_name_full(module, name, convention, free_keys, env, scope)
}

fn x_path_to_rs_filename(path: &std::path::Path) -> String {
    let mut out = path.file_stem().unwrap().to_str().unwrap().to_string();
    out.push_str(".rs");
    out
}

/// Converts a DSLX module (i.e. `.x` file) into its corresponding Rust bridge
/// code, and emits that Rust code to a corresponding filename in the `out_dir`.
pub fn x_path_to_rs_bridge(
    relpath: &str,
    out_dir: &std::path::Path,
    root_dir: &std::path::Path,
) -> std::path::PathBuf {
    let mut import_data = dslx::ImportData::new(None, &[root_dir]);
    let path = std::path::PathBuf::from(relpath);
    let dslx = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("DSLX file should be readable: {:?}", path));

    // Generate the bridge code.
    let mut builder = rust_bridge_builder::RustBridgeBuilder::new();
    dslx_bridge::convert_leaf_module(&mut import_data, &dslx, &path, &mut builder)
        .expect("expect bridge building success");
    let rs = builder.build();

    // Write this out to the corresponding Rust filename in the `out_dir`.
    let out_path = out_dir.join(x_path_to_rs_filename(&path));
    std::fs::write(&out_path, rs).unwrap();
    out_path
}

// Wrapper around `x_path_to_rs_bridge` where:
//
// - the `out_dir` comes from the environment variable `OUT_DIR` which is
//   populated e.g. by `cargo` in `build.rs` execution.
// - the working directory comes from the repository root
pub fn x_path_to_rs_bridge_via_env(relpath: &str) -> std::path::PathBuf {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR should be set");
    let metadata = cargo_metadata::MetadataCommand::new()
        .exec()
        .expect("cargo metadata should be available");
    let root_dir = metadata.workspace_root.as_path().as_std_path();
    x_path_to_rs_bridge(relpath, std::path::Path::new(&out_dir), root_dir)
}

#[cfg(test)]
mod tests {
    use lib_support::xls_format_preference_from_string;
    use xlsynth_sys::XlsFormatPreference;

    use super::*;

    #[test]
    fn test_mangle_dslx_name_full_basic() {
        let mangled = mangle_dslx_name_with_env(
            "my_mod",
            "f",
            DslxCallingConvention::Typical,
            &[],
            None,
            None,
        )
        .expect("mangle success");
        assert_eq!(mangled, "__my_mod__f");
    }

    #[test]
    fn test_mangle_dslx_name_full_scope() {
        let mangled = mangle_dslx_name_with_env(
            "my_mod",
            "f",
            DslxCallingConvention::Typical,
            &[],
            None,
            Some("Point"),
        )
        .expect("mangle success (scoped)");
        assert_eq!(mangled, "__my_mod__Point__f");
    }

    #[test]
    fn test_mangle_dslx_name_full_parametrics() {
        // Prepare parametric bindings X=42, Y=64.
        let x_val = dslx::InterpValue::make_ubits(32, 42);
        let y_val = dslx::InterpValue::make_ubits(32, 64);
        let free_keys = ["X", "Y"]; // order should not matter for result here
        let env = dslx::ParametricEnv::new(&[("X", &x_val), ("Y", &y_val)]).expect("env");

        let mangled = mangle_dslx_name_with_env(
            "my_mod",
            "p",
            DslxCallingConvention::Typical,
            &free_keys,
            Some(&env),
            None,
        )
        .expect("mangle with params");
        assert_eq!(mangled, "__my_mod__p__42_64");
    }

    #[test]
    fn test_mangle_dslx_name_full_implicit_token() {
        let mangled_itok = mangle_dslx_name_with_env(
            "my_mod",
            "f",
            DslxCallingConvention::ImplicitToken,
            &[],
            None,
            None,
        )
        .expect("mangle implicit-token");
        assert_eq!(mangled_itok, "__itok__my_mod__f");
    }

    #[test]
    fn test_mangle_dslx_name_full_proc_next() {
        let mangled_next = mangle_dslx_name_with_env(
            "my_mod",
            "f",
            DslxCallingConvention::ProcNext,
            &[],
            None,
            None,
        )
        .expect("mangle proc-next");
        assert_eq!(mangled_next, "__my_mod__f_next");
    }

    #[test]
    fn test_mangle_dslx_name_full_composed() {
        // Combine implicit-token, scope, and parametric env (no special chars).
        let x_val = dslx::InterpValue::make_ubits(32, 7);
        let y_val = dslx::InterpValue::make_ubits(32, 9);
        let env = dslx::ParametricEnv::new(&[("X", &x_val), ("Y", &y_val)]).expect("env");
        let mangled = mangle_dslx_name_with_env(
            "my_mod",
            "g_step1",
            DslxCallingConvention::ImplicitToken,
            &["X", "Y"],
            Some(&env),
            Some("Impl"),
        )
        .expect("mangle composed");
        assert_eq!(mangled, "__itok__my_mod__Impl__g_step1__7_9");
    }

    #[test]
    fn test_convert_dslx_to_ir() {
        let ir = convert_dslx_to_ir_text(
            "fn f(x: u32) -> u32 { x }",
            std::path::Path::new("/memfile/test_mod.x"),
            &DslxConvertOptions::default(),
        )
        .expect("ir conversion should succeed");
        assert_eq!(
            ir.ir,
            "package test_mod

file_number 0 \"/memfile/test_mod.x\"

fn __test_mod__f(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}
"
        );
        assert!(ir.warnings.is_empty());
    }

    #[test]
    fn test_parse_typed_value_garbage() {
        let e: XlsynthError = xls_parse_typed_value("asdf").expect_err("should not parse");
        assert_eq!(
            e.0,
            "INVALID_ARGUMENT: Expected token of type \"(\" @ 1:1, but found: Token(\"ident\", value=\"asdf\") @ 1:1"
        );
    }

    #[test]
    fn test_parse_typed_value_bits32_42() {
        let v: IrValue = xls_parse_typed_value("bits[32]:42").expect("should parse ok");
        assert_eq!(v.to_string(), "bits[32]:42");
    }

    #[test]
    fn test_xls_format_preference_from_string() {
        let fmt: XlsFormatPreference = xls_format_preference_from_string("default")
            .expect("should convert to format preference");
        assert_eq!(fmt, 0);

        let fmt: XlsFormatPreference =
            xls_format_preference_from_string("hex").expect("should convert to format preference");
        assert_eq!(fmt, 4);

        xls_format_preference_from_string("blah")
            .expect_err("should not convert to format preference");
    }
}
