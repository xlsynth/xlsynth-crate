// SPDX-License-Identifier: Apache-2.0

pub mod dslx;
pub mod ir_package;
pub mod ir_value;
pub mod vast;
pub mod xlsynth_error;

use std::ffi::{CStr, CString};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use ir_package::{IrFunctionType, IrPackagePtr, IrType};
use ir_value::IrBits;
use xlsynth_sys::{CIrBits, CIrPackage, CIrValue};
use xlsynth_sys::{CIrFunction, CIrFunctionType, CIrType, XlsFormatPreference};

pub use ir_package::IrFunction;
pub use ir_package::IrPackage;
pub use ir_value::IrValue;
pub use xlsynth_error::XlsynthError;

/// Converts a C string that was given from the XLS library into a Rust string
/// and deallocates the original C string.
unsafe fn c_str_to_rust(xls_c_str: *mut std::os::raw::c_char) -> String {
    if xls_c_str.is_null() {
        return String::new();
    }

    let c_str: &CStr = CStr::from_ptr(xls_c_str);
    let result: String = String::from_utf8_lossy(c_str.to_bytes()).to_string();

    // We release the C string via a call to the XLS library so that it can use the
    // same allocator it used to allocate the string for deallocation and we don't
    // need to assume the Rust code and dynmic library are using the same underlying
    // allocator.
    xlsynth_sys::xls_c_str_free(xls_c_str);
    result
}

pub fn xls_convert_dslx_to_ir(dslx: &str, path: &std::path::Path) -> Result<String, XlsynthError> {
    // Extract the module name from the path; e.g. "foo/bar/baz.x" -> "baz"
    let module_name = path.file_stem().unwrap().to_str().unwrap();
    let path_str = path.to_str().unwrap();

    unsafe {
        let dslx = CString::new(dslx).unwrap();
        let c_path = CString::new(path_str).unwrap();
        let c_module_name = CString::new(module_name).unwrap();
        let stdlib_path = xlsynth_sys::DSLX_STDLIB_PATH;
        let dslx_stdlib_path = CString::new(stdlib_path).unwrap();

        let additional_search_paths_ptrs: Vec<*const std::os::raw::c_char> = vec![];

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();

        // Call the function
        let success = xlsynth_sys::xls_convert_dslx_to_ir(
            dslx.as_ptr(),
            c_path.as_ptr(),
            c_module_name.as_ptr(),
            dslx_stdlib_path.as_ptr(),
            additional_search_paths_ptrs.as_ptr(),
            additional_search_paths_ptrs.len(),
            &mut error_out,
            &mut ir_out,
        );

        if success {
            return Ok(c_str_to_rust(ir_out));
        } else {
            let error_out_str = c_str_to_rust(error_out);
            return Err(XlsynthError(error_out_str));
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
            return Ok(IrValue { ptr: ir_value_out });
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            return Err(XlsynthError(error_out_str));
        }
    }
}

pub(crate) fn xls_value_free(p: *mut CIrValue) -> Result<(), XlsynthError> {
    unsafe {
        xlsynth_sys::xls_value_free(p);
        return Ok(());
    }
}

pub(crate) fn xls_package_free(p: *mut CIrPackage) -> Result<(), XlsynthError> {
    unsafe {
        xlsynth_sys::xls_package_free(p);
        return Ok(());
    }
}

pub(crate) fn xls_value_to_string(p: *mut CIrValue) -> Result<String, XlsynthError> {
    unsafe {
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_value_to_string(p, &mut c_str_out);
        if success {
            return Ok(c_str_to_rust(c_str_out));
        }
        return Err(XlsynthError(
            "Failed to convert XLS value to string via C API".to_string(),
        ));
    }
}

pub(crate) fn xls_value_get_bits(p: *const CIrValue) -> Result<IrBits, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut bits_out: *mut CIrBits = std::ptr::null_mut();
        let success = xlsynth_sys::xls_value_get_bits(p, &mut error_out, &mut bits_out);
        if success {
            return Ok(IrBits { ptr: bits_out });
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

pub(crate) fn xls_format_preference_from_string(
    s: &str,
) -> Result<XlsFormatPreference, XlsynthError> {
    unsafe {
        let c_str = CString::new(s).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: XlsFormatPreference = -1;
        let success = xlsynth_sys::xls_format_preference_from_string(
            c_str.as_ptr(),
            &mut error_out,
            &mut result_out,
        );
        if success {
            return Ok(result_out);
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

pub(crate) fn xls_value_to_string_format_preference(
    p: *mut CIrValue,
    fmt: XlsFormatPreference,
) -> Result<String, XlsynthError> {
    unsafe {
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_value_to_string_format_preference(
            p,
            fmt,
            &mut error_out,
            &mut c_str_out,
        );
        if success {
            return Ok(c_str_to_rust(c_str_out));
        }
        return Err(XlsynthError(
            "Failed to convert XLS value to string via C API".to_string(),
        ));
    }
}

pub(crate) fn xls_value_eq(
    lhs: *const CIrValue,
    rhs: *const CIrValue,
) -> Result<bool, XlsynthError> {
    unsafe {
        return Ok(xlsynth_sys::xls_value_eq(lhs, rhs));
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_parse_ir_package(
///     const char* ir, const char* filename,
///     char** error_out,
///     struct xls_package** xls_package_out);
/// ```
pub(crate) fn xls_parse_ir_package(
    ir: &str,
    filename: Option<&str>,
) -> Result<crate::ir_package::IrPackage, XlsynthError> {
    unsafe {
        let ir = CString::new(ir).unwrap();
        // The filename is allowed to be a null pointer if there is no filename.
        let filename_ptr = filename
            .map(|s| CString::new(s).unwrap())
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null());
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut xls_package_out: *mut CIrPackage = std::ptr::null_mut();
        let success = xlsynth_sys::xls_parse_ir_package(
            ir.as_ptr(),
            filename_ptr,
            &mut error_out,
            &mut xls_package_out,
        );
        if success {
            let package = crate::ir_package::IrPackage {
                ptr: Arc::new(RwLock::new(IrPackagePtr(xls_package_out))),
                filename: filename.map(|s| s.to_string()),
            };
            return Ok(package);
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_type_to_string(struct xls_type* type, char** error_out,
/// char** result_out);
/// ```
pub(crate) fn xls_type_to_string(t: *const CIrType) -> Result<String, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_type_to_string(t, &mut error_out, &mut c_str_out);
        if success {
            return Ok(c_str_to_rust(c_str_out));
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_package_get_type_for_value(struct xls_package* package,
// struct xls_value* value, char** error_out,
// struct xls_type** result_out);
// ```
pub(crate) fn xls_package_get_type_for_value(
    package: *const CIrPackage,
    value: *const CIrValue,
) -> Result<IrType, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CIrType = std::ptr::null_mut();
        let success = xlsynth_sys::xls_package_get_type_for_value(
            package,
            value,
            &mut error_out,
            &mut result_out,
        );
        if success {
            let ir_type = IrType { ptr: result_out };
            return Ok(ir_type);
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}
/// Bindings for the C API function:
/// ```c
/// bool xls_package_get_function(s
///    struct xls_package* package,
///    const char* function_name, char** error_out,
///    struct xls_function** result_out);
/// ```
pub(crate) fn xls_package_get_function(
    package: &Arc<RwLock<IrPackagePtr>>,
    guard: RwLockReadGuard<IrPackagePtr>,
    function_name: &str,
) -> Result<crate::ir_package::IrFunction, XlsynthError> {
    unsafe {
        let function_name = CString::new(function_name).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CIrFunction = std::ptr::null_mut();
        let success = xlsynth_sys::xls_package_get_function(
            guard.const_c_ptr(),
            function_name.as_ptr(),
            &mut error_out,
            &mut result_out,
        );
        if success {
            let function = crate::ir_package::IrFunction {
                parent: package.clone(),
                ptr: result_out,
            };
            return Ok(function);
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_function_get_type(struct xls_function* function, char** error_out,
/// struct xls_function_type** xls_fn_type_out);
/// ```
pub(crate) fn xls_function_get_type(
    _package_write_guard: &RwLockWriteGuard<IrPackagePtr>,
    function: *const CIrFunction,
) -> Result<IrFunctionType, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut xls_fn_type_out: *mut CIrFunctionType = std::ptr::null_mut();
        let success =
            xlsynth_sys::xls_function_get_type(function, &mut error_out, &mut xls_fn_type_out);
        if success {
            let ir_type = IrFunctionType {
                ptr: xls_fn_type_out,
            };
            return Ok(ir_type);
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_function_type_to_string(struct xls_function_type* xls_function_type,
/// char** error_out, char** string_out);
/// ```
pub(crate) fn xls_function_type_to_string(
    t: *const CIrFunctionType,
) -> Result<String, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_function_type_to_string(t, &mut error_out, &mut c_str_out);
        if success {
            return Ok(c_str_to_rust(c_str_out));
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

pub(crate) fn xls_function_get_name(function: *const CIrFunction) -> Result<String, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_function_get_name(function, &mut error_out, &mut c_str_out);
        if success {
            return Ok(c_str_to_rust(c_str_out));
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_interpret_function(
///     struct xls_function* function, size_t argc,
///     const struct xls_value** args, char** error_out,
///     struct xls_value** result_out);
/// ```
pub(crate) fn xls_interpret_function(
    _package_guard: &RwLockReadGuard<IrPackagePtr>,
    function: *const CIrFunction,
    args: &[IrValue],
) -> Result<IrValue, XlsynthError> {
    unsafe {
        let args_ptrs: Vec<*const CIrValue> =
            args.iter().map(|v| -> *const CIrValue { v.ptr }).collect();
        let argc = args_ptrs.len();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CIrValue = std::ptr::null_mut();
        let success = xlsynth_sys::xls_interpret_function(
            function,
            argc,
            args_ptrs.as_ptr(),
            &mut error_out,
            &mut result_out,
        );
        if success {
            let result = IrValue { ptr: result_out };
            return Ok(result);
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Binding for the C API function:
/// ```c
/// bool xls_optimize_ir(const char* ir, const char* top, char** error_out,
/// char** ir_out);
/// ```
pub(crate) fn xls_optimize_ir(ir: &str, top: &str) -> Result<String, XlsynthError> {
    unsafe {
        let ir = CString::new(ir).unwrap();
        let top = CString::new(top).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success =
            xlsynth_sys::xls_optimize_ir(ir.as_ptr(), top.as_ptr(), &mut error_out, &mut ir_out);
        if success {
            return Ok(c_str_to_rust(ir_out));
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Binding for the C API function:
/// ```c
/// bool xls_mangle_dslx_name(const char* module_name, const char* function_name,
/// char** error_out, char** mangled_out);
/// ```
pub(crate) fn xls_mangle_dslx_name(
    module_name: &str,
    function_name: &str,
) -> Result<String, XlsynthError> {
    unsafe {
        let module_name = CString::new(module_name).unwrap();
        let function_name = CString::new(function_name).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut mangled_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_mangle_dslx_name(
            module_name.as_ptr(),
            function_name.as_ptr(),
            &mut error_out,
            &mut mangled_out,
        );
        if success {
            return Ok(c_str_to_rust(mangled_out));
        }
        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
    }
}

/// Binding for the C API function:
/// ```c
/// bool xls_package_to_string(const struct xls_package* p, char** string_out) {
/// ```
pub(crate) fn xls_package_to_string(p: *const CIrPackage) -> Result<String, XlsynthError> {
    unsafe {
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_package_to_string(p, &mut c_str_out);
        if success {
            return Ok(c_str_to_rust(c_str_out));
        }
        return Err(XlsynthError(
            "Failed to convert XLS package to string via C API".to_string(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_dslx_to_ir() {
        let ir = xls_convert_dslx_to_ir(
            "fn f(x: u32) -> u32 { x }",
            std::path::Path::new("/memfile/test_mod.x"),
        )
        .expect("ir conversion should succeed");
        assert_eq!(
            ir,
            "package test_mod

file_number 0 \"/memfile/test_mod.x\"

fn __test_mod__f(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}
"
        );
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
pub fn convert_dslx_to_ir(dslx: &str, path: &std::path::Path) -> Result<IrPackage, XlsynthError> {
    let ir_text = xls_convert_dslx_to_ir(dslx, path)?;
    // Get the filename as an Option<&str>
    let filename = path.file_name().and_then(|s| s.to_str());
    IrPackage::parse_ir(&ir_text, filename)
}

pub fn optimize_ir(ir: &IrPackage, top: &str) -> Result<IrPackage, XlsynthError> {
    let ir_text = xls_optimize_ir(&ir.to_string(), top)?;
    IrPackage::parse_ir(&ir_text, ir.filename())
}

pub fn mangle_dslx_name(module: &str, name: &str) -> Result<String, XlsynthError> {
    xls_mangle_dslx_name(module, name)
}
