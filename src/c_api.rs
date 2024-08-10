// SPDX-License-Identifier: Apache-2.0

//! Wrappers around the C API for the XLS dynamic shared object.

use std::ffi::CString;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::ir_package::{IrFunctionType, IrPackagePtr, IrType};
use crate::ir_value::IrValue;
use crate::xlsynth_error::XlsynthError;

extern crate libc;

mod ffi {
    #[repr(C)]
    pub struct CIrValue {
        _private: [u8; 0], // Ensures the struct cannot be instantiated
    }

    #[repr(C)]
    pub struct CIrPackage {
        _private: [u8; 0], // Ensures the struct cannot be instantiated
    }

    #[repr(C)]
    pub struct CIrFunction {
        _private: [u8; 0], // Ensures the struct cannot be instantiated
    }

    #[repr(C)]
    pub(crate) struct CIrType {
        _private: [u8; 0], // Ensures the struct cannot be instantiated
    }

    #[repr(C)]
    pub(crate) struct CIrFunctionType {
        _private: [u8; 0], // Ensures the struct cannot be instantiated
    }

    pub type XlsFormatPreference = i32;

    extern "C" {
        pub fn xls_convert_dslx_to_ir(
            dslx: *const std::os::raw::c_char,
            path: *const std::os::raw::c_char,
            module_name: *const std::os::raw::c_char,
            dslx_stdlib_path: *const std::os::raw::c_char,
            additional_search_paths: *const *const std::os::raw::c_char,
            additional_search_paths_count: libc::size_t,
            error_out: *mut *mut std::os::raw::c_char,
            ir_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_parse_typed_value(
            text: *const std::os::raw::c_char,
            error_out: *mut *mut std::os::raw::c_char,
            value_out: *mut *mut CIrValue,
        ) -> bool;
        pub fn xls_value_free(value: *mut CIrValue);
        pub fn xls_package_free(package: *mut CIrPackage);
        pub fn xls_value_to_string(
            value: *const CIrValue,
            str_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_format_preference_from_string(
            s: *const std::os::raw::c_char,
            error_out: *mut *mut std::os::raw::c_char,
            result_out: *mut XlsFormatPreference,
        ) -> bool;
        pub fn xls_value_to_string_format_preference(
            value: *const CIrValue,
            fmt: XlsFormatPreference,
            error_out: *mut *mut std::os::raw::c_char,
            str_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_value_eq(value: *const CIrValue, value: *const CIrValue) -> bool;
        pub fn xls_parse_ir_package(
            ir: *const std::os::raw::c_char,
            filename: *const std::os::raw::c_char,
            error_out: *mut *mut std::os::raw::c_char,
            xls_package_out: *mut *mut CIrPackage,
        ) -> bool;
        pub fn xls_type_to_string(
            t: *const CIrType,
            error_out: *mut *mut std::os::raw::c_char,
            result_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_package_get_type_for_value(
            package: *const CIrPackage,
            value: *const CIrValue,
            error_out: *mut *mut std::os::raw::c_char,
            result_out: *mut *mut CIrType,
        ) -> bool;
        pub fn xls_package_get_function(
            package: *const CIrPackage,
            function_name: *const std::os::raw::c_char,
            error_out: *mut *mut std::os::raw::c_char,
            result_out: *mut *mut CIrFunction,
        ) -> bool;
        pub fn xls_function_get_type(
            function: *const CIrFunction,
            error_out: *mut *mut std::os::raw::c_char,
            xls_fn_type_out: *mut *mut CIrFunctionType,
        ) -> bool;
        pub fn xls_function_type_to_string(
            t: *const CIrFunctionType,
            error_out: *mut *mut std::os::raw::c_char,
            string_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_function_get_name(
            function: *const CIrFunction,
            error_out: *mut *mut std::os::raw::c_char,
            name_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_interpret_function(
            function: *const CIrFunction,
            argc: libc::size_t,
            args: *const *const CIrValue,
            error_out: *mut *mut std::os::raw::c_char,
            result_out: *mut *mut CIrValue,
        ) -> bool;
        pub fn xls_optimize_ir(
            ir: *const std::os::raw::c_char,
            top: *const std::os::raw::c_char,
            error_out: *mut *mut std::os::raw::c_char,
            ir_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_mangle_dslx_name(
            module_name: *const std::os::raw::c_char,
            function_name: *const std::os::raw::c_char,
            error_out: *mut *mut std::os::raw::c_char,
            mangled_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
        pub fn xls_package_to_string(
            p: *const CIrPackage,
            string_out: *mut *mut std::os::raw::c_char,
        ) -> bool;
    }
}

pub(crate) type CIrValue = ffi::CIrValue;
pub(crate) type CIrPackage = ffi::CIrPackage;
pub(crate) type CIrFunction = ffi::CIrFunction;
pub(crate) type CIrFunctionType = ffi::CIrFunctionType;
pub(crate) type CIrType = ffi::CIrType;
pub(crate) type XlsFormatPreference = ffi::XlsFormatPreference;

pub fn xls_convert_dslx_to_ir(dslx: &str, path: &std::path::Path) -> Result<String, XlsynthError> {
    // Extract the module name from the path; e.g. "foo/bar/baz.x" -> "baz"
    let module_name = path.file_stem().unwrap().to_str().unwrap();
    let path_str = path.to_str().unwrap();

    unsafe {
        let dslx = CString::new(dslx).unwrap();
        let c_path = CString::new(path_str).unwrap();
        let c_module_name = CString::new(module_name).unwrap();
        let stdlib_path = env!("DSLX_STDLIB_PATH");
        let dslx_stdlib_path = CString::new(stdlib_path).unwrap();

        let additional_search_paths_ptrs: Vec<*const std::os::raw::c_char> = vec![];

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();

        // Call the function
        let success = ffi::xls_convert_dslx_to_ir(
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
            let ir_out_str = if !ir_out.is_null() {
                CString::from_raw(ir_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(ir_out_str);
        } else {
            let error_out_str = if !error_out.is_null() {
                CString::from_raw(error_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Err(XlsynthError(error_out_str));
        }
    }
}

pub fn xls_parse_typed_value(s: &str) -> Result<IrValue, XlsynthError> {
    unsafe {
        let c_str = CString::new(s).unwrap();
        let mut ir_value_out: *mut CIrValue = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = ffi::xls_parse_typed_value(c_str.as_ptr(), &mut error_out, &mut ir_value_out);
        if success {
            return Ok(IrValue { ptr: ir_value_out });
        } else {
            let error_out_str: String = if !error_out.is_null() {
                CString::from_raw(error_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Err(XlsynthError(error_out_str));
        }
    }
}

pub(crate) fn xls_value_free(p: *mut CIrValue) -> Result<(), XlsynthError> {
    unsafe {
        ffi::xls_value_free(p);
        return Ok(());
    }
}

pub(crate) fn xls_package_free(p: *mut CIrPackage) -> Result<(), XlsynthError> {
    unsafe {
        ffi::xls_package_free(p);
        return Ok(());
    }
}

pub(crate) fn xls_value_to_string(p: *mut CIrValue) -> Result<String, XlsynthError> {
    unsafe {
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = ffi::xls_value_to_string(p, &mut c_str_out);
        if success {
            let s: String = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(s);
        }
        return Err(XlsynthError("Failed to convert XLS value to string via C API".to_string()));
    }
}

pub(crate) fn xls_format_preference_from_string(
    s: &str,
) -> Result<XlsFormatPreference, XlsynthError> {
    unsafe {
        let c_str = CString::new(s).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: XlsFormatPreference = -1;
        let success =
            ffi::xls_format_preference_from_string(c_str.as_ptr(), &mut error_out, &mut result_out);
        if success {
            return Ok(result_out);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success =
            ffi::xls_value_to_string_format_preference(p, fmt, &mut error_out, &mut c_str_out);
        if success {
            let s: String = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(s);
        }
        return Err(XlsynthError("Failed to convert XLS value to string via C API".to_string()));
    }
}

pub(crate) fn xls_value_eq(
    lhs: *const CIrValue,
    rhs: *const CIrValue,
) -> Result<bool, XlsynthError> {
    unsafe {
        return Ok(ffi::xls_value_eq(lhs, rhs));
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
        let success = ffi::xls_parse_ir_package(
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
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_type_to_string(t, &mut error_out, &mut c_str_out);
        if success {
            let out_str = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(out_str);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success =
            ffi::xls_package_get_type_for_value(package, value, &mut error_out, &mut result_out);
        if success {
            let ir_type = IrType { ptr: result_out };
            return Ok(ir_type);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_package_get_function(
            guard.const_c_ptr(),
            function_name.as_ptr(),
            &mut error_out,
            &mut result_out,
        );
        if success {
            let function =
                crate::ir_package::IrFunction { parent: package.clone(), ptr: result_out };
            return Ok(function);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_function_get_type(function, &mut error_out, &mut xls_fn_type_out);
        if success {
            let ir_type = IrFunctionType { ptr: xls_fn_type_out };
            return Ok(ir_type);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_function_type_to_string(t, &mut error_out, &mut c_str_out);
        if success {
            let out_str = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(out_str);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
        return Err(XlsynthError(error_out_str));
    }
}

pub(crate) fn xls_function_get_name(function: *const CIrFunction) -> Result<String, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = ffi::xls_function_get_name(function, &mut error_out, &mut c_str_out);
        if success {
            let out_str = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(out_str);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_interpret_function(
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
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_optimize_ir(ir.as_ptr(), top.as_ptr(), &mut error_out, &mut ir_out);
        if success {
            let ir_out_str = if !ir_out.is_null() {
                CString::from_raw(ir_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(ir_out_str);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_mangle_dslx_name(
            module_name.as_ptr(),
            function_name.as_ptr(),
            &mut error_out,
            &mut mangled_out,
        );
        if success {
            let mangled_out_str = if !mangled_out.is_null() {
                CString::from_raw(mangled_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(mangled_out_str);
        }
        let error_out_str: String = if !error_out.is_null() {
            CString::from_raw(error_out).into_string().unwrap()
        } else {
            String::new()
        };
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
        let success = ffi::xls_package_to_string(p, &mut c_str_out);
        if success {
            let s: String = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(s);
        }
        return Err(XlsynthError("Failed to convert XLS package to string via C API".to_string()));
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

fn __test_mod__f(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
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
