// SPDX-License-Identifier: Apache-2.0

//! Support functions that are not exposed as primary APIs from `lib.rs` but
//! support the implementation of functions exposed via `lib.rs`.`
//!
//! (Things in this module are generally crate-visible vs public to provide that
//! support.)

use std::{
    ffi::{CStr, CString},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use xlsynth_sys::{
    CIrBits, CIrFunction, CIrFunctionType, CIrPackage, CIrType, CIrValue,
    CScheduleAndCodegenResult, XlsFormatPreference,
};

use crate::{
    ir_package::{IrFunctionType, IrPackagePtr, IrType, ScheduleAndCodegenResult},
    IrBits, IrValue, XlsynthError,
};

/// Binding for the C API function:
/// ```c
/// bool xls_package_to_string(const struct xls_package* p, char** string_out) {
/// ```
pub(crate) fn xls_package_to_string(p: *const CIrPackage) -> Result<String, XlsynthError> {
    unsafe {
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_package_to_string(p, &mut c_str_out);
        if success {
            Ok(c_str_to_rust(c_str_out))
        } else {
            Err(XlsynthError(
                "Failed to convert XLS package to string via C API".to_string(),
            ))
        }
    }
}

pub(crate) unsafe fn c_str_to_rust_no_dealloc(xls_c_str: *mut std::os::raw::c_char) -> String {
    if xls_c_str.is_null() {
        String::new()
    } else {
        let c_str: &CStr = CStr::from_ptr(xls_c_str);
        String::from_utf8_lossy(c_str.to_bytes()).to_string()
    }
}

/// Converts a C string that was given from the XLS library into a Rust string
/// and deallocates the original C string.
pub(crate) unsafe fn c_str_to_rust(xls_c_str: *mut std::os::raw::c_char) -> String {
    let result = c_str_to_rust_no_dealloc(xls_c_str);

    // We release the C string via a call to the XLS library so that it can use the
    // same allocator it used to allocate the string for deallocation and we don't
    // need to assume the Rust code and dynmic library are using the same underlying
    // allocator.
    xlsynth_sys::xls_c_str_free(xls_c_str);
    result
}

pub(crate) fn xls_value_free(p: *mut CIrValue) {
    unsafe { xlsynth_sys::xls_value_free(p) }
}

pub(crate) fn xls_package_free(p: *mut CIrPackage) {
    unsafe { xlsynth_sys::xls_package_free(p) }
}

pub(crate) fn xls_value_to_string(p: *mut CIrValue) -> Result<String, XlsynthError> {
    unsafe {
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_value_to_string(p, &mut c_str_out);
        if success {
            Ok(c_str_to_rust(c_str_out))
        } else {
            Err(XlsynthError(
                "Failed to convert XLS value to string via C API".to_string(),
            ))
        }
    }
}

pub(crate) fn xls_value_get_element(
    p: *mut CIrValue,
    index: usize,
) -> Result<IrValue, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut element_out: *mut CIrValue = std::ptr::null_mut();
        let success =
            xlsynth_sys::xls_value_get_element(p, index, &mut error_out, &mut element_out);
        if success {
            Ok(IrValue { ptr: element_out })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_value_get_element_count(p: *const CIrValue) -> Result<usize, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut count: i64 = 0;
        let success = xlsynth_sys::xls_value_get_element_count(p, &mut error_out, &mut count);
        if success {
            Ok(count as usize)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_value_make_ubits(value: u64, bit_count: usize) -> Result<IrValue, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result: *mut CIrValue = std::ptr::null_mut();
        let success =
            xlsynth_sys::xls_value_make_ubits(bit_count as i64, value, &mut error_out, &mut result);
        if success {
            Ok(IrValue { ptr: result })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_value_make_sbits(value: i64, bit_count: usize) -> Result<IrValue, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result: *mut CIrValue = std::ptr::null_mut();
        let success =
            xlsynth_sys::xls_value_make_sbits(bit_count as i64, value, &mut error_out, &mut result);
        if success {
            Ok(IrValue { ptr: result })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_value_make_tuple(elements: &[IrValue]) -> IrValue {
    unsafe {
        // The C API call takes ownership of the elements that are in the array.
        let elements_ptrs: Vec<*mut CIrValue> = elements
            .iter()
            .map(|v| xlsynth_sys::xls_value_clone(v.ptr))
            .collect();
        let result = xlsynth_sys::xls_value_make_tuple(elements_ptrs.len(), elements_ptrs.as_ptr());
        assert!(!result.is_null());
        IrValue { ptr: result }
    }
}

pub(crate) fn xls_bits_to_debug_str(p: *const CIrBits) -> String {
    unsafe {
        let c_str_out = xlsynth_sys::xls_bits_to_debug_string(p);
        c_str_to_rust(c_str_out)
    }
}

pub(crate) fn xls_bits_make_ubits(bit_count: usize, value: u64) -> Result<IrBits, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut bits_out: *mut CIrBits = std::ptr::null_mut();
        let success = xlsynth_sys::xls_bits_make_ubits(
            bit_count as i64,
            value,
            &mut error_out,
            &mut bits_out,
        );
        if success {
            Ok(IrBits { ptr: bits_out })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_bits_make_sbits(bit_count: usize, value: i64) -> Result<IrBits, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut bits_out: *mut CIrBits = std::ptr::null_mut();
        let success = xlsynth_sys::xls_bits_make_sbits(
            bit_count as i64,
            value,
            &mut error_out,
            &mut bits_out,
        );
        if success {
            Ok(IrBits { ptr: bits_out })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_value_get_bits(p: *const CIrValue) -> Result<IrBits, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut bits_out: *mut CIrBits = std::ptr::null_mut();
        let success = xlsynth_sys::xls_value_get_bits(p, &mut error_out, &mut bits_out);
        if success {
            Ok(IrBits { ptr: bits_out })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(result_out)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(c_str_to_rust(c_str_out))
        } else {
            Err(XlsynthError(
                "Failed to convert XLS value to string via C API".to_string(),
            ))
        }
    }
}

pub(crate) fn xls_bits_to_string(
    p: *const CIrBits,
    fmt: XlsFormatPreference,
    include_bit_count: bool,
) -> Result<String, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_bits_to_string(
            p,
            fmt,
            include_bit_count,
            &mut error_out,
            &mut c_str_out,
        );
        if success {
            Ok(c_str_to_rust(c_str_out))
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_value_eq(
    lhs: *const CIrValue,
    rhs: *const CIrValue,
) -> Result<bool, XlsynthError> {
    unsafe { Ok(xlsynth_sys::xls_value_eq(lhs, rhs)) }
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
            Ok(package)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(c_str_to_rust(c_str_out))
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

/// Bindings for the C API function:
/// ```c
/// bool xls_package_get_type_for_value(struct xls_package* package,
// struct xls_value* value, char** error_out,
/// struct xls_type** result_out);
/// ```
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
            Ok(ir_type)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}
/// Bindings for the C API function:
/// ```c
/// bool xls_package_get_function(
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
            Ok(function)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(ir_type)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(c_str_to_rust(c_str_out))
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_function_get_name(function: *const CIrFunction) -> Result<String, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = xlsynth_sys::xls_function_get_name(function, &mut error_out, &mut c_str_out);
        if success {
            Ok(c_str_to_rust(c_str_out))
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(IrValue { ptr: result_out })
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(c_str_to_rust(ir_out))
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
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
            Ok(c_str_to_rust(mangled_out))
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

pub(crate) fn xls_schedule_and_codegen_package(
    _package: &Arc<RwLock<IrPackagePtr>>,
    guard: RwLockWriteGuard<IrPackagePtr>,
    scheduling_options_flags_proto_str: &str,
    codegen_flags_proto_str: &str,
    with_delay_model: bool,
) -> Result<ScheduleAndCodegenResult, XlsynthError> {
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut CScheduleAndCodegenResult = std::ptr::null_mut();
        let scheduling_options_flags_proto =
            CString::new(scheduling_options_flags_proto_str).unwrap();
        let codegen_flags_proto = CString::new(codegen_flags_proto_str).unwrap();
        let success = xlsynth_sys::xls_schedule_and_codegen_package(
            guard.mut_c_ptr(),
            scheduling_options_flags_proto.as_ptr(),
            codegen_flags_proto.as_ptr(),
            with_delay_model,
            &mut error_out,
            &mut result_out,
        );
        if success {
            assert!(!result_out.is_null());
            let result = ScheduleAndCodegenResult { ptr: result_out };
            Ok(result)
        } else {
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}
