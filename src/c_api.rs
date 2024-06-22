// SPDX-License-Identifier: Apache-2.0

//! Wrappers around the C API for the XLS dynamic shared object.

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;
use std::ffi::CString;
use std::sync::Mutex;

use crate::ir_package::{IrFunctionType, IrType};
use crate::ir_value::IrValue;
use crate::xlsynth_error::XlsynthError;

extern crate libc;
extern crate libloading;

const DSO_VERSION_TAG: &str = env!("XLS_DSO_VERSION_TAG");

static LIBRARY: OnceCell<Mutex<Library>> = OnceCell::new();

fn get_library() -> &'static Mutex<Library> {
    LIBRARY.get_or_init(|| {
        let dso_extension = if cfg!(target_os = "macos") {
            "dylib"
        } else if cfg!(target_os = "linux") {
            "so"
        } else {
            panic!("Running on an unknown OS");
        };
        let so_filename = format!("libxls-{}.{}", DSO_VERSION_TAG, dso_extension);
        let library = unsafe {
            Library::new(so_filename.clone()).expect("dynamic library should be present")
        };
        Mutex::new(library)
    })
}

#[repr(C)]
pub(crate) struct CIrValue {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub(crate) struct CIrPackage {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub(crate) struct CIrFunction {
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

pub(crate) type XlsFormatPreference = i32;

type XlsValueToString =
    unsafe extern "C" fn(value: *const CIrValue, str_out: *mut *mut std::os::raw::c_char) -> bool;

pub fn xls_convert_dslx_to_ir(dslx: &str) -> Result<String, XlsynthError> {
    type XlsConvertDslxToIr = unsafe extern "C" fn(
        dslx: *const std::os::raw::c_char,
        path: *const std::os::raw::c_char,
        module_name: *const std::os::raw::c_char,
        dslx_stdlib_path: *const std::os::raw::c_char,
        additional_search_paths: *const *const std::os::raw::c_char,
        additional_search_paths_count: libc::size_t,
        error_out: *mut *mut std::os::raw::c_char,
        ir_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym_convert_dslx_to_ir: Symbol<XlsConvertDslxToIr> =
            match lib.get(b"xls_convert_dslx_to_ir") {
                Ok(f) => f,
                Err(e) => {
                    return Err(XlsynthError(format!(
                        "Failed to load symbol `xls_convert_dslx_to_ir`: {}",
                        e
                    )))
                }
            };
        let dslx = CString::new(dslx).unwrap();
        let path = CString::new("test_mod.x").unwrap();
        let module_name = CString::new("test_mod").unwrap();
        let stdlib_path = env!("DSLX_STDLIB_PATH");
        let dslx_stdlib_path = CString::new(stdlib_path).unwrap();

        let additional_search_paths_ptrs: Vec<*const std::os::raw::c_char> = vec![];

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();

        // Call the function
        let success = dlsym_convert_dslx_to_ir(
            dslx.as_ptr(),
            path.as_ptr(),
            module_name.as_ptr(),
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
    type XlsParseTypedValue = unsafe extern "C" fn(
        text: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        value_out: *mut *mut CIrValue,
    ) -> bool;
    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsParseTypedValue> = match lib.get(b"xls_parse_typed_value") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_parse_typed_value`: {}",
                    e
                )))
            }
        };

        let c_str = CString::new(s).unwrap();
        let mut ir_value_out: *mut CIrValue = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(c_str.as_ptr(), &mut error_out, &mut ir_value_out);
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
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<unsafe extern "C" fn(*mut CIrValue)> = match lib.get(b"xls_value_free") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_value_free`: {}",
                    e
                )))
            }
        };
        dlsym(p);
        return Ok(());
    }
}

pub(crate) fn xls_package_free(p: *mut CIrPackage) -> Result<(), XlsynthError> {
    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<unsafe extern "C" fn(*mut CIrPackage)> =
            match lib.get(b"xls_package_free") {
                Ok(f) => f,
                Err(e) => {
                    return Err(XlsynthError(format!(
                        "Failed to load symbol `xls_package_free`: {}",
                        e
                    )))
                }
            };
        dlsym(p);
        return Ok(());
    }
}

pub(crate) fn xls_value_to_string(p: *mut CIrValue) -> Result<String, XlsynthError> {
    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsValueToString> = match lib.get(b"xls_value_to_string") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_value_to_string`: {}",
                    e
                )))
            }
        };

        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(p, &mut c_str_out);
        if success {
            let s: String = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(s);
        }
        return Err(XlsynthError(
            "Failed to convert XLS value to string via C API".to_string(),
        ));
    }
}

pub(crate) fn xls_format_preference_from_string(
    s: &str,
) -> Result<XlsFormatPreference, XlsynthError> {
    // Invokes the function with signature:
    // ```c
    // bool xls_format_preference_from_string(const char* s, char** error_out,
    //   xls_format_preference* result_out);
    // ```
    //
    // Note that the format preference enum is `int32_t``.
    type XlsFormatPreferenceFromString = unsafe extern "C" fn(
        s: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut XlsFormatPreference,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsFormatPreferenceFromString> =
            match lib.get(b"xls_format_preference_from_string") {
                Ok(f) => f,
                Err(e) => {
                    return Err(XlsynthError(format!(
                        "Failed to load symbol `xls_format_preference_from_string`: {}",
                        e
                    )))
                }
            };

        let c_str = CString::new(s).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: XlsFormatPreference = -1;
        let success = dlsym(c_str.as_ptr(), &mut error_out, &mut result_out);
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
    type XlsValueToStringFormatPreference = unsafe extern "C" fn(
        value: *const CIrValue,
        fmt: XlsFormatPreference,
        error_out: *mut *mut std::os::raw::c_char,
        str_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsValueToStringFormatPreference> =
            match lib.get(b"xls_value_to_string_format_preference") {
                Ok(f) => f,
                Err(e) => {
                    return Err(XlsynthError(format!(
                        "Failed to load symbol `xls_value_to_string_format_preference`: {}",
                        e
                    )))
                }
            };

        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(p, fmt, &mut error_out, &mut c_str_out);
        if success {
            let s: String = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(s);
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
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<unsafe extern "C" fn(*const CIrValue, *const CIrValue) -> bool> =
            match lib.get(b"xls_value_eq") {
                Ok(f) => f,
                Err(e) => {
                    return Err(XlsynthError(format!(
                        "Failed to load symbol `xls_value_eq`: {}",
                        e
                    )))
                }
            };
        return Ok(dlsym(lhs, rhs));
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
    type XlsParseIrPackage = unsafe extern "C" fn(
        ir: *const std::os::raw::c_char,
        filename: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        xls_package_out: *mut *mut CIrPackage,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsParseIrPackage> = match lib.get(b"xls_parse_ir_package") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_parse_ir_package`: {}",
                    e
                )))
            }
        };

        let ir = CString::new(ir).unwrap();
        // The filename is allowed to be a null pointer if there is no filename.
        let filename_ptr = filename
            .map(|s| CString::new(s).unwrap())
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null());
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut xls_package_out: *mut CIrPackage = std::ptr::null_mut();
        let success = dlsym(
            ir.as_ptr(),
            filename_ptr,
            &mut error_out,
            &mut xls_package_out,
        );
        if success {
            let package = crate::ir_package::IrPackage {
                ptr: xls_package_out,
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
    type XlsTypeToString = unsafe extern "C" fn(
        t: *const CIrType,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsTypeToString> = match lib.get(b"xls_type_to_string") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_type_to_string`: {}",
                    e
                )))
            }
        };

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(t, &mut error_out, &mut c_str_out);
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
    type XlsPackageGetTypeForValue = unsafe extern "C" fn(
        package: *const CIrPackage,
        value: *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrType,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsPackageGetTypeForValue> =
            match lib.get(b"xls_package_get_type_for_value") {
                Ok(f) => f,
                Err(e) => {
                    return Err(XlsynthError(format!(
                        "Failed to load symbol `xls_package_get_type_for_value`: {}",
                        e
                    )))
                }
            };

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CIrType = std::ptr::null_mut();
        let success = dlsym(package, value, &mut error_out, &mut result_out);
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
    package: *const CIrPackage,
    function_name: &str,
) -> Result<crate::ir_package::IrFunction, XlsynthError> {
    type XlsPackageGetFunction = unsafe extern "C" fn(
        package: *const CIrPackage,
        function_name: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrFunction,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsPackageGetFunction> = match lib.get(b"xls_package_get_function") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_package_get_function`: {}",
                    e
                )))
            }
        };

        let function_name = CString::new(function_name).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CIrFunction = std::ptr::null_mut();
        let success = dlsym(
            package,
            function_name.as_ptr(),
            &mut error_out,
            &mut result_out,
        );
        if success {
            let function = crate::ir_package::IrFunction { ptr: result_out };
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
    function: *const CIrFunction,
) -> Result<IrFunctionType, XlsynthError> {
    type XlsFunctionGetType = unsafe extern "C" fn(
        function: *const CIrFunction,
        error_out: *mut *mut std::os::raw::c_char,
        xls_fn_type_out: *mut *mut CIrFunctionType,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsFunctionGetType> = match lib.get(b"xls_function_get_type") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_function_get_type`: {}",
                    e
                )))
            }
        };

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut xls_fn_type_out: *mut CIrFunctionType = std::ptr::null_mut();
        let success = dlsym(function, &mut error_out, &mut xls_fn_type_out);
        if success {
            let ir_type = IrFunctionType {
                ptr: xls_fn_type_out,
            };
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
    type XlsFunctionTypeToString = unsafe extern "C" fn(
        t: *const CIrFunctionType,
        error_out: *mut *mut std::os::raw::c_char,
        string_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsFunctionTypeToString> = match lib.get(b"xls_function_type_to_string") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_function_type_to_string`: {}",
                    e
                )))
            }
        };

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(t, &mut error_out, &mut c_str_out);
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
    type XlsFunctionGetName = unsafe extern "C" fn(
        function: *const CIrFunction,
        error_out: *mut *mut std::os::raw::c_char,
        name_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsFunctionGetName> = match lib.get(b"xls_function_get_name") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_function_get_name`: {}",
                    e
                )))
            }
        };

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(function, &mut error_out, &mut c_str_out);
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
    function: *const CIrFunction,
    args: &[IrValue],
) -> Result<IrValue, XlsynthError> {
    type XlsInterpretFunction = unsafe extern "C" fn(
        function: *const CIrFunction,
        argc: libc::size_t,
        args: *const *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrValue,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsInterpretFunction> = match lib.get(b"xls_interpret_function") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_interpret_function`: {}",
                    e
                )))
            }
        };

        let args_ptrs: Vec<*const CIrValue> =
            args.iter().map(|v| -> *const CIrValue { v.ptr }).collect();
        let argc = args_ptrs.len();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CIrValue = std::ptr::null_mut();
        let success = dlsym(
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
    type XlsOptimizeIr = unsafe extern "C" fn(
        ir: *const std::os::raw::c_char,
        top: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        ir_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsOptimizeIr> = match lib.get(b"xls_optimize_ir") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_optimize_ir`: {}",
                    e
                )))
            }
        };

        let ir = CString::new(ir).unwrap();
        let top = CString::new(top).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(ir.as_ptr(), top.as_ptr(), &mut error_out, &mut ir_out);
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
    type XlsMangleDslxName = unsafe extern "C" fn(
        module_name: *const std::os::raw::c_char,
        function_name: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        mangled_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsMangleDslxName> = match lib.get(b"xls_mangle_dslx_name") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_mangle_dslx_name`: {}",
                    e
                )))
            }
        };

        let module_name = CString::new(module_name).unwrap();
        let function_name = CString::new(function_name).unwrap();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut mangled_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(
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
    type XlsPackageToString = unsafe extern "C" fn(
        p: *const CIrPackage,
        string_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    unsafe {
        let lib = get_library().lock().unwrap();
        let dlsym: Symbol<XlsPackageToString> = match lib.get(b"xls_package_to_string") {
            Ok(f) => f,
            Err(e) => {
                return Err(XlsynthError(format!(
                    "Failed to load symbol `xls_package_to_string`: {}",
                    e
                )))
            }
        };

        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = dlsym(p, &mut c_str_out);
        if success {
            let s: String = if !c_str_out.is_null() {
                CString::from_raw(c_str_out).into_string().unwrap()
            } else {
                String::new()
            };
            return Ok(s);
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
        let ir = xls_convert_dslx_to_ir("fn f(x: u32) -> u32 { x }")
            .expect("ir conversion should succeed");
        assert_eq!(
            ir,
            "package test_mod

file_number 0 \"test_mod.x\"

fn __test_mod__f(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
"
        );
    }

    #[test]
    fn test_parse_typed_value_garbage() {
        let e: XlsynthError = xls_parse_typed_value("asdf").expect_err("should not parse");
        assert_eq!(e.0, "INVALID_ARGUMENT: Expected token of type \"(\" @ 1:1, but found: Token(\"ident\", value=\"asdf\") @ 1:1");
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
