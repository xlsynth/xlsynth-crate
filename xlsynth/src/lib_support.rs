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
    CIrBValue, CIrBits, CIrFunction, CIrFunctionJit, CIrFunctionType, CIrPackage, CIrType,
    CIrValue, CScheduleAndCodegenResult, CTraceMessage, XlsFormatPreference,
};

use crate::{
    ir_package::{IrFunctionType, IrPackagePtr, IrType, ScheduleAndCodegenResult},
    IrBits, IrFunction, IrValue, XlsynthError,
};

// Wrapper around the raw pointer that frees (i.e. when the wrapping refcount
// drops to zero).
pub(crate) struct IrFnBuilderPtr {
    pub(crate) ptr: *mut xlsynth_sys::CIrFunctionBuilder,
}

impl Drop for IrFnBuilderPtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                xlsynth_sys::xls_function_builder_free(self.ptr);
            }
        }
    }
}

// Wrapper around the raw pointer that frees (i.e. when the wrapping refcount
// drops to zero).
pub(crate) struct BValuePtr {
    pub(crate) ptr: *mut CIrBValue,
}

impl Drop for BValuePtr {
    fn drop(&mut self) {
        unsafe {
            xlsynth_sys::xls_bvalue_free(self.ptr);
        }
    }
}

/// Macro for performing FFI calls to fallible functions with a trailing output
/// parameter.
///
/// You call it by passing in the FFI function, its regular arguments,
/// then a semicolon and the identifier to populate with the output.
/// It returns a Result<(), XlsynthError>.
///
/// For instance, calling:
/// ```rust-snippet
/// let mut ir_out: *mut c_char = std::ptr::null_mut();
/// xls_call!(xlsynth_sys::xls_optimize_ir, ir.as_ptr(), top.as_ptr(); ir_out)?;
/// ```
/// will call the FFI function with the arguments and automatically convert the
/// error if it occurs.
macro_rules! xls_ffi_call {
    ($func:path $(, $arg:expr )* ; $out:ident) => {{
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let success = $func($($arg,)* &mut error_out, &mut $out);
            if success {
                Ok(())
            } else {
                Err(XlsynthError(c_str_to_rust(error_out)))
            }
        }
    }};
}

// Variant of the above macro for when we have no return value.
macro_rules! xls_ffi_call_noreturn {
    ($func:path $(, $arg:expr )*) => {{
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let success = $func($($arg,)* &mut error_out);
            if success {
                Ok(())
            } else {
                Err(XlsynthError(c_str_to_rust(error_out)))
            }
        }
    }};
}

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

/// Returns a CString built from the given string and a pointer to its contents.
/// The CString must be kept alive as long as the pointer is used.
fn cstring_and_ptr(name: &str) -> (CString, *const std::os::raw::c_char) {
    let cstr = CString::new(name).unwrap();
    let ptr = cstr.as_ptr();
    (cstr, ptr)
}

/// Like [`cstring_and_ptr`] but for optional strings. The returned pointer is
/// null if the option is `None`.
fn optional_cstring_and_ptr(name: Option<&str>) -> (Option<CString>, *const std::os::raw::c_char) {
    if let Some(s) = name {
        let cstr = CString::new(s).unwrap();
        let ptr = cstr.as_ptr();
        (Some(cstr), ptr)
    } else {
        (None, std::ptr::null())
    }
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
    let mut element_out: *mut CIrValue = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_value_get_element, p, index; element_out)?;
    Ok(IrValue { ptr: element_out })
}

pub(crate) fn xls_value_get_element_count(p: *const CIrValue) -> Result<usize, XlsynthError> {
    let mut count: i64 = 0;
    xls_ffi_call!(xlsynth_sys::xls_value_get_element_count, p; count)?;
    Ok(count as usize)
}

pub(crate) fn xls_value_make_ubits(value: u64, bit_count: usize) -> Result<IrValue, XlsynthError> {
    let mut result: *mut CIrValue = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_value_make_ubits, bit_count as i64, value; result)?;
    Ok(IrValue { ptr: result })
}

pub(crate) fn xls_value_make_sbits(value: i64, bit_count: usize) -> Result<IrValue, XlsynthError> {
    let mut result: *mut CIrValue = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_value_make_sbits, bit_count as i64, value; result)?;
    Ok(IrValue { ptr: result })
}

pub(crate) fn xls_value_make_tuple(elements: &[IrValue]) -> IrValue {
    unsafe {
        let elements_ptrs: Vec<*const CIrValue> =
            elements.iter().map(|v| v.ptr as *const CIrValue).collect();
        let result = xlsynth_sys::xls_value_make_tuple(elements_ptrs.len(), elements_ptrs.as_ptr());
        assert!(!result.is_null());
        IrValue { ptr: result }
    }
}

pub(crate) fn xls_value_make_array(elements: &[IrValue]) -> Result<IrValue, XlsynthError> {
    let elements_ptrs: Vec<*const CIrValue> =
        elements.iter().map(|v| v.ptr as *const CIrValue).collect();
    let mut result: *mut CIrValue = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_value_make_array, elements_ptrs.len(), elements_ptrs.as_ptr(); result)?;
    Ok(IrValue { ptr: result })
}

pub(crate) fn xls_bits_to_debug_str(p: *const CIrBits) -> String {
    unsafe {
        let c_str_out = xlsynth_sys::xls_bits_to_debug_string(p);
        c_str_to_rust(c_str_out)
    }
}

pub(crate) fn xls_bits_make_ubits(bit_count: usize, value: u64) -> Result<IrBits, XlsynthError> {
    let mut result: *mut CIrBits = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_bits_make_ubits, bit_count as i64, value; result)?;
    Ok(IrBits { ptr: result })
}

pub(crate) fn xls_bits_make_sbits(bit_count: usize, value: i64) -> Result<IrBits, XlsynthError> {
    let mut result: *mut CIrBits = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_bits_make_sbits, bit_count as i64, value; result)?;
    Ok(IrBits { ptr: result })
}

pub(crate) fn xls_value_get_bits(p: *const CIrValue) -> Result<IrBits, XlsynthError> {
    let mut result: *mut CIrBits = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_value_get_bits, p; result)?;
    Ok(IrBits { ptr: result })
}

pub(crate) fn xls_format_preference_from_string(
    s: &str,
) -> Result<XlsFormatPreference, XlsynthError> {
    let c_str = CString::new(s).unwrap();
    let mut result_out: XlsFormatPreference = -1;
    xls_ffi_call!(xlsynth_sys::xls_format_preference_from_string, c_str.as_ptr(); result_out)?;
    Ok(result_out)
}

pub(crate) fn xls_value_to_string_format_preference(
    p: *mut CIrValue,
    fmt: XlsFormatPreference,
) -> Result<String, XlsynthError> {
    let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_value_to_string_format_preference, p, fmt; c_str_out)?;
    Ok(unsafe { c_str_to_rust(c_str_out) })
}

pub(crate) fn xls_bits_to_string(
    p: *const CIrBits,
    fmt: XlsFormatPreference,
    include_bit_count: bool,
) -> Result<String, XlsynthError> {
    let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_bits_to_string, p, fmt, include_bit_count; c_str_out)?;
    Ok(unsafe { c_str_to_rust(c_str_out) })
}

pub(crate) fn xls_value_eq(
    lhs: *const CIrValue,
    rhs: *const CIrValue,
) -> Result<bool, XlsynthError> {
    unsafe { Ok(xlsynth_sys::xls_value_eq(lhs, rhs)) }
}

pub(crate) fn xls_parse_ir_package(
    ir: &str,
    filename: Option<&str>,
) -> Result<crate::ir_package::IrPackage, XlsynthError> {
    let ir_cstring = CString::new(ir).unwrap();
    let (_filename_cstr, filename_ptr) = optional_cstring_and_ptr(filename);
    let mut xls_package_out: *mut CIrPackage = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_parse_ir_package, ir_cstring.as_ptr(), filename_ptr; xls_package_out)?;
    let package = crate::ir_package::IrPackage {
        ptr: Arc::new(RwLock::new(IrPackagePtr(xls_package_out))),
        filename: filename.map(|s| s.to_string()),
    };
    Ok(package)
}

pub(crate) fn xls_package_new(name: &str) -> Result<crate::ir_package::IrPackage, XlsynthError> {
    let (_name_cstr, name_ptr) = cstring_and_ptr(name);
    let xls_package_out: *mut CIrPackage = unsafe { xlsynth_sys::xls_package_create(name_ptr) };
    Ok(crate::ir_package::IrPackage {
        ptr: Arc::new(RwLock::new(IrPackagePtr(xls_package_out))),
        filename: None,
    })
}

pub(crate) fn xls_function_builder_new(
    package: *mut CIrPackage,
    name: &str,
    should_verify: bool,
) -> Arc<RwLock<IrFnBuilderPtr>> {
    let (_name_cstr, name_ptr) = cstring_and_ptr(name);
    let fn_builder =
        unsafe { xlsynth_sys::xls_function_builder_create(name_ptr, package, should_verify) };
    assert!(!fn_builder.is_null());
    Arc::new(RwLock::new(IrFnBuilderPtr { ptr: fn_builder }))
}

pub(crate) fn xls_function_builder_add_parameter(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    name: &str,
    type_: &IrType,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = cstring_and_ptr(name);
    let type_raw = type_.ptr;
    let bvalue_raw =
        unsafe { xlsynth_sys::xls_function_builder_add_parameter(builder.ptr, name_ptr, type_raw) };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_dynamic_bit_slice(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
    start: RwLockReadGuard<BValuePtr>,
    width: u64,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let width_i64 = width as i64;
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_dynamic_bit_slice(
            builder_base,
            value.ptr,
            start.ptr,
            width_i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_bit_slice(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
    start: u64,
    width: u64,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_bit_slice(
            builder_base,
            value.ptr,
            start as i64,
            width as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_concat(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    values: &[RwLockReadGuard<BValuePtr>],
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let values_ptrs: Vec<*mut CIrBValue> = values.iter().map(|v| v.ptr).collect();
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_concat(
            builder_base,
            values_ptrs.as_ptr(),
            values.len() as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_literal(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: &IrValue,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw =
        unsafe { xlsynth_sys::xls_builder_base_add_literal(builder_base, value.ptr, name_ptr) };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_tuple(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    elements: &[RwLockReadGuard<BValuePtr>],
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let mut elements_ptrs: Vec<*mut CIrBValue> = elements.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_tuple(
            builder_base,
            elements_ptrs.as_mut_ptr(),
            elements.len() as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_tuple_index(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    tuple: RwLockReadGuard<BValuePtr>,
    index: u64,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_tuple_index(
            builder_base,
            tuple.ptr,
            index as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_build_with_return_value(
    package: &Arc<RwLock<IrPackagePtr>>,
    _package_guard: RwLockWriteGuard<IrPackagePtr>,
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    return_value: RwLockReadGuard<BValuePtr>,
) -> Result<IrFunction, XlsynthError> {
    let return_value_ptr = return_value.ptr;
    let mut result_out: *mut CIrFunction = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_function_builder_build_with_return_value, builder.ptr, return_value_ptr; result_out)?;
    Ok(IrFunction {
        parent: package.clone(),
        ptr: result_out,
    })
}

pub(crate) fn xls_type_to_string(t: *const CIrType) -> Result<String, XlsynthError> {
    let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_type_to_string, t; c_str_out)?;
    Ok(unsafe { c_str_to_rust(c_str_out) })
}

pub(crate) fn xls_type_get_flat_bit_count(t: *const CIrType) -> u64 {
    let bit_count = unsafe { xlsynth_sys::xls_type_get_flat_bit_count(t) };
    assert!(bit_count >= 0, "bit count must be non-negative");
    bit_count as u64
}

pub(crate) fn xls_package_set_top_by_name(
    package: *mut CIrPackage,
    name: &str,
) -> Result<(), XlsynthError> {
    let (_name_cstr, name_ptr) = cstring_and_ptr(name);
    xls_ffi_call_noreturn!(xlsynth_sys::xls_package_set_top_by_name, package, name_ptr)?;
    Ok(())
}

pub(crate) fn xls_package_get_bits_type(package: *mut CIrPackage, bit_count: u64) -> IrType {
    let type_raw = unsafe { xlsynth_sys::xls_package_get_bits_type(package, bit_count as i64) };
    IrType { ptr: type_raw }
}

pub(crate) fn xls_package_get_tuple_type(package: *mut CIrPackage, members: &[IrType]) -> IrType {
    let mut members_ptrs: Vec<*mut CIrType> = members.iter().map(|v| v.ptr).collect();
    let members_ptr = members_ptrs.as_mut_ptr();
    let member_count = members_ptrs.len() as i64;
    let type_raw =
        unsafe { xlsynth_sys::xls_package_get_tuple_type(package, members_ptr, member_count) };
    IrType { ptr: type_raw }
}

pub(crate) fn xls_package_get_array_type(
    package: *mut CIrPackage,
    element_type: *mut CIrType,
    size: i64,
) -> IrType {
    let type_raw = unsafe { xlsynth_sys::xls_package_get_array_type(package, element_type, size) };
    IrType { ptr: type_raw }
}

pub(crate) fn xls_package_get_token_type(package: *mut CIrPackage) -> IrType {
    let type_raw = unsafe { xlsynth_sys::xls_package_get_token_type(package) };
    IrType { ptr: type_raw }
}

pub(crate) fn xls_package_get_type_for_value(
    package: *const CIrPackage,
    value: *const CIrValue,
) -> Result<IrType, XlsynthError> {
    let mut result_out: *mut CIrType = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_package_get_type_for_value, package, value; result_out)?;
    Ok(IrType { ptr: result_out })
}

pub(crate) fn xls_package_get_function(
    package: &Arc<RwLock<IrPackagePtr>>,
    guard: RwLockReadGuard<IrPackagePtr>,
    function_name: &str,
) -> Result<crate::ir_package::IrFunction, XlsynthError> {
    let function_name = CString::new(function_name).unwrap();
    let mut result_out: *mut CIrFunction = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_package_get_function, guard.const_c_ptr(), function_name.as_ptr(); result_out)?;
    Ok(crate::ir_package::IrFunction {
        parent: package.clone(),
        ptr: result_out,
    })
}

pub(crate) fn xls_function_get_type(
    _package_write_guard: &RwLockWriteGuard<IrPackagePtr>,
    function: *const CIrFunction,
) -> Result<IrFunctionType, XlsynthError> {
    let mut xls_fn_type_out: *mut CIrFunctionType = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_function_get_type, function; xls_fn_type_out)?;
    Ok(IrFunctionType {
        ptr: xls_fn_type_out,
    })
}

pub(crate) fn xls_function_type_to_string(
    t: *const CIrFunctionType,
) -> Result<String, XlsynthError> {
    let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_function_type_to_string, t; c_str_out)?;
    Ok(unsafe { c_str_to_rust(c_str_out) })
}

pub(crate) fn xls_function_get_name(function: *const CIrFunction) -> Result<String, XlsynthError> {
    let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_function_get_name, function; c_str_out)?;
    Ok(unsafe { c_str_to_rust(c_str_out) })
}

pub(crate) fn xls_interpret_function(
    _package_guard: &RwLockReadGuard<IrPackagePtr>,
    function: *const CIrFunction,
    args: &[IrValue],
) -> Result<IrValue, XlsynthError> {
    let args_ptrs: Vec<*const CIrValue> =
        args.iter().map(|v| -> *const CIrValue { v.ptr }).collect();
    let argc = args_ptrs.len();
    let mut result_out: *mut CIrValue = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_interpret_function, function, argc, args_ptrs.as_ptr(); result_out)?;
    Ok(IrValue { ptr: result_out })
}

pub(crate) fn xls_optimize_ir(ir: &str, top: &str) -> Result<String, XlsynthError> {
    let ir = CString::new(ir).unwrap();
    let top = CString::new(top).unwrap();
    let mut ir_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_optimize_ir, ir.as_ptr(), top.as_ptr(); ir_out)?;
    let ir_str = unsafe { c_str_to_rust(ir_out) };
    Ok(ir_str)
}

pub struct TraceMessage {
    pub message: String,
    pub verbosity: i64,
}

pub struct RunResult {
    pub value: IrValue,
    pub trace_messages: Vec<TraceMessage>,
    pub assert_messages: Vec<String>,
}

// Helper that converts the trace messages to Rust objects and deallocates the C
// value.
unsafe fn trace_messages_to_rust(
    c_trace_messages: *mut CTraceMessage,
    count: usize,
) -> Vec<TraceMessage> {
    if c_trace_messages.is_null() {
        return Vec::new();
    }
    let mut trace_messages: Vec<TraceMessage> = Vec::new();
    for i in 0..count {
        let trace_message: &CTraceMessage = unsafe { &*c_trace_messages.wrapping_add(i) };
        trace_messages.push(TraceMessage {
            message: unsafe { c_str_to_rust_no_dealloc(trace_message.message) },
            verbosity: trace_message.verbosity,
        });
    }
    xlsynth_sys::xls_trace_messages_free(c_trace_messages, count);
    trace_messages
}

unsafe fn c_strs_to_rust(c_strs: *mut *mut std::os::raw::c_char, count: usize) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    for i in 0..count {
        let xls_c_str: *mut std::os::raw::c_char = unsafe { *c_strs.wrapping_add(i) };
        result.push(c_str_to_rust_no_dealloc(xls_c_str));
    }
    result
}

pub(crate) fn xls_make_function_jit(
    _package_guard: &RwLockReadGuard<IrPackagePtr>,
    function: *const CIrFunction,
) -> Result<*mut CIrFunctionJit, XlsynthError> {
    let mut ptr: *mut CIrFunctionJit = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_make_function_jit, function; ptr)?;
    Ok(ptr)
}

pub(crate) fn xls_function_jit_run(
    _package_guard: &RwLockReadGuard<IrPackagePtr>,
    jit: *const CIrFunctionJit,
    args: &[IrValue],
) -> Result<RunResult, XlsynthError> {
    let mut result_out: *mut CIrValue = std::ptr::null_mut();
    let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    let argc = args.len();
    let args_ptrs: Vec<*const CIrValue> =
        args.iter().map(|v| -> *const CIrValue { v.ptr }).collect();
    let mut trace_messages_out: *mut CTraceMessage = std::ptr::null_mut();
    let mut trace_messages_count: usize = 0;
    let mut assert_messages_out: *mut *mut std::os::raw::c_char = std::ptr::null_mut();
    let mut assert_messages_count: usize = 0;
    let success = unsafe {
        xlsynth_sys::xls_function_jit_run(
            jit,
            argc,
            args_ptrs.as_ptr(),
            &mut error_out,
            &mut trace_messages_out,
            &mut trace_messages_count,
            &mut assert_messages_out,
            &mut assert_messages_count,
            &mut result_out,
        )
    };
    if !success {
        let error_message = unsafe { c_str_to_rust(error_out) };
        return Err(XlsynthError(format!(
            "Failed to run JIT function: {}",
            error_message
        )));
    }
    let trace_messages =
        unsafe { trace_messages_to_rust(trace_messages_out, trace_messages_count) };
    let assert_messages = unsafe { c_strs_to_rust(assert_messages_out, assert_messages_count) };
    Ok(RunResult {
        value: IrValue { ptr: result_out },
        trace_messages,
        assert_messages,
    })
}

pub(crate) fn xls_mangle_dslx_name(
    module_name: &str,
    function_name: &str,
) -> Result<String, XlsynthError> {
    let module_name = CString::new(module_name).unwrap();
    let function_name = CString::new(function_name).unwrap();
    let mut mangled_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_mangle_dslx_name, module_name.as_ptr(), function_name.as_ptr(); mangled_out)?;
    Ok(unsafe { c_str_to_rust(mangled_out) })
}

pub(crate) fn xls_schedule_and_codegen_package(
    _package: &Arc<RwLock<IrPackagePtr>>,
    guard: RwLockWriteGuard<IrPackagePtr>,
    scheduling_options_flags_proto_str: &str,
    codegen_flags_proto_str: &str,
    with_delay_model: bool,
) -> Result<ScheduleAndCodegenResult, XlsynthError> {
    let scheduling_options_flags_proto = CString::new(scheduling_options_flags_proto_str).unwrap();
    let codegen_flags_proto = CString::new(codegen_flags_proto_str).unwrap();
    let mut result_out: *mut CScheduleAndCodegenResult = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_schedule_and_codegen_package,
        guard.mut_c_ptr(),
        scheduling_options_flags_proto.as_ptr(),
        codegen_flags_proto.as_ptr(),
        with_delay_model;
        result_out
    )?;
    assert!(!result_out.is_null());
    Ok(ScheduleAndCodegenResult { ptr: result_out })
}

pub(crate) fn xls_function_builder_add_array(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    element_type: &IrType,
    elements: &[RwLockReadGuard<BValuePtr>],
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let elements_ptrs: Vec<*mut CIrBValue> = elements.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_array(
            builder_base,
            element_type.ptr,
            elements_ptrs.as_ptr(),
            elements_ptrs.len() as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_array_index_multi(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    array: RwLockReadGuard<BValuePtr>,
    index: &[RwLockReadGuard<BValuePtr>],
    assumed_in_bounds: bool,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let index_ptrs: Vec<*mut CIrBValue> = index.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_array_index(
            builder_base,
            array.ptr,
            index_ptrs.as_ptr(),
            index_ptrs.len() as i64,
            assumed_in_bounds,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_bit_slice_update(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
    start: RwLockReadGuard<BValuePtr>,
    update: RwLockReadGuard<BValuePtr>,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_bit_slice_update(
            builder_base,
            value.ptr,
            start.ptr,
            update.ptr,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_select(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    selector: RwLockReadGuard<BValuePtr>,
    cases: &[RwLockReadGuard<BValuePtr>],
    default_value: RwLockReadGuard<BValuePtr>,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let cases_ptrs: Vec<*mut CIrBValue> = cases.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_select(
            builder_base,
            selector.ptr,
            cases_ptrs.as_ptr(),
            cases_ptrs.len() as i64,
            default_value.ptr,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_array_concat(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    arrays: &[RwLockReadGuard<BValuePtr>],
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let arrays_ptrs: Vec<*mut CIrBValue> = arrays.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_array_concat(
            builder_base,
            arrays_ptrs.as_ptr(),
            arrays_ptrs.len() as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_array_slice(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    array: RwLockReadGuard<BValuePtr>,
    start: RwLockReadGuard<BValuePtr>,
    width: i64,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_array_slice(
            builder_base,
            array.ptr,
            start.ptr,
            width,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_array_update(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    array: RwLockReadGuard<BValuePtr>,
    update_value: RwLockReadGuard<BValuePtr>,
    indices: &[RwLockReadGuard<BValuePtr>],
    assumed_in_bounds: bool,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let (_name_cstr, name_ptr) = optional_cstring_and_ptr(name);
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let indices_ptrs: Vec<*mut CIrBValue> = indices.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_array_update(
            builder_base,
            array.ptr,
            update_value.ptr,
            indices_ptrs.as_ptr(),
            indices_ptrs.len() as i64,
            assumed_in_bounds,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_sign_extend(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
    new_bit_count: i64,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let name_cstr = name.map(|s| CString::new(s).unwrap());
    let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_sign_extend(
            builder_base,
            value.ptr,
            new_bit_count,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_zero_extend(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
    new_bit_count: i64,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let name_cstr = name.map(|s| CString::new(s).unwrap());
    let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_zero_extend(
            builder_base,
            value.ptr,
            new_bit_count,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_one_hot(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    input: RwLockReadGuard<BValuePtr>,
    lsb_is_priority: bool,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let name_cstr = name.map(|s| CString::new(s).unwrap());
    let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_one_hot(
            builder_base,
            input.ptr,
            lsb_is_priority,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_priority_select(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    selector: RwLockReadGuard<BValuePtr>,
    cases: &[RwLockReadGuard<BValuePtr>],
    default_value: RwLockReadGuard<BValuePtr>,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let name_cstr = name.map(|s| CString::new(s).unwrap());
    let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let cases_ptrs: Vec<*mut CIrBValue> = cases.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_priority_select(
            builder_base,
            selector.ptr,
            cases_ptrs.as_ptr(),
            cases_ptrs.len() as i64,
            default_value.ptr,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_one_hot_select(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    selector: RwLockReadGuard<BValuePtr>,
    cases: &[RwLockReadGuard<BValuePtr>],
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let name_cstr = name.map(|s| CString::new(s).unwrap());
    let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let cases_ptrs: Vec<*mut CIrBValue> = cases.iter().map(|v| v.ptr).collect();
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_one_hot_select(
            builder_base,
            selector.ptr,
            cases_ptrs.as_ptr(),
            cases_ptrs.len() as i64,
            name_ptr,
        )
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_add_decode(
    builder: RwLockWriteGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
    width: Option<u64>,
    name: Option<&str>,
) -> Arc<RwLock<BValuePtr>> {
    let name_cstr = name.map(|s| CString::new(s).unwrap());
    let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };

    let mut width_value = width.unwrap_or(0);
    let width_ptr: *mut i64 = if width.is_some() {
        &mut width_value as *mut u64 as *mut i64
    } else {
        std::ptr::null_mut()
    };
    let bvalue_raw = unsafe {
        xlsynth_sys::xls_builder_base_add_decode(builder_base, value.ptr, width_ptr, name_ptr)
    };
    Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw }))
}

pub(crate) fn xls_function_builder_last_value(
    builder: RwLockReadGuard<IrFnBuilderPtr>,
) -> Result<Arc<RwLock<BValuePtr>>, XlsynthError> {
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let mut bvalue_raw: *mut CIrBValue = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_builder_base_get_last_value, builder_base; bvalue_raw)?;
    assert!(!bvalue_raw.is_null());
    Ok(Arc::new(RwLock::new(BValuePtr { ptr: bvalue_raw })))
}

pub(crate) fn xls_function_builder_get_type(
    builder: RwLockReadGuard<IrFnBuilderPtr>,
    value: RwLockReadGuard<BValuePtr>,
) -> Option<IrType> {
    let builder_base = unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
    let type_raw = unsafe { xlsynth_sys::xls_builder_base_get_type(builder_base, value.ptr) };
    if type_raw.is_null() {
        None
    } else {
        Some(IrType { ptr: type_raw })
    }
}

macro_rules! impl_binary_ir_builder {
    ($fn_name:ident, $ffi_func:ident) => {
        pub(crate) fn $fn_name(
            builder: std::sync::RwLockWriteGuard<IrFnBuilderPtr>,
            a: std::sync::RwLockReadGuard<BValuePtr>,
            b: std::sync::RwLockReadGuard<BValuePtr>,
            name: Option<&str>,
        ) -> std::sync::Arc<std::sync::RwLock<BValuePtr>> {
            let builder_base =
                unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
            let name_cstr = name.map(|s| std::ffi::CString::new(s).unwrap());
            let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
            let builder_base =
                unsafe { xlsynth_sys::$ffi_func(builder_base, a.ptr, b.ptr, name_ptr) };
            std::sync::Arc::new(std::sync::RwLock::new(BValuePtr { ptr: builder_base }))
        }
    };
}

impl_binary_ir_builder!(xls_function_builder_add_add, xls_builder_base_add_add);
impl_binary_ir_builder!(xls_function_builder_add_sub, xls_builder_base_add_sub);
impl_binary_ir_builder!(xls_function_builder_add_shra, xls_builder_base_add_shra);
impl_binary_ir_builder!(xls_function_builder_add_shrl, xls_builder_base_add_shrl);
impl_binary_ir_builder!(xls_function_builder_add_shll, xls_builder_base_add_shll);
impl_binary_ir_builder!(xls_function_builder_add_nor, xls_builder_base_add_nor);
impl_binary_ir_builder!(xls_function_builder_add_and, xls_builder_base_add_and);
impl_binary_ir_builder!(xls_function_builder_add_nand, xls_builder_base_add_nand);
impl_binary_ir_builder!(xls_function_builder_add_or, xls_builder_base_add_or);
impl_binary_ir_builder!(xls_function_builder_add_xor, xls_builder_base_add_xor);
impl_binary_ir_builder!(xls_function_builder_add_eq, xls_builder_base_add_eq);
impl_binary_ir_builder!(xls_function_builder_add_ne, xls_builder_base_add_ne);
impl_binary_ir_builder!(xls_function_builder_add_ule, xls_builder_base_add_ule);
impl_binary_ir_builder!(xls_function_builder_add_ult, xls_builder_base_add_ult);
impl_binary_ir_builder!(xls_function_builder_add_uge, xls_builder_base_add_uge);
impl_binary_ir_builder!(xls_function_builder_add_ugt, xls_builder_base_add_ugt);
impl_binary_ir_builder!(xls_function_builder_add_sle, xls_builder_base_add_sle);
impl_binary_ir_builder!(xls_function_builder_add_slt, xls_builder_base_add_slt);
impl_binary_ir_builder!(xls_function_builder_add_sgt, xls_builder_base_add_sgt);
impl_binary_ir_builder!(xls_function_builder_add_sge, xls_builder_base_add_sge);
impl_binary_ir_builder!(xls_function_builder_add_umul, xls_builder_base_add_umul);
impl_binary_ir_builder!(xls_function_builder_add_smul, xls_builder_base_add_smul);

macro_rules! impl_unary_ir_builder {
    ($fn_name:ident, $ffi_func:ident) => {
        pub(crate) fn $fn_name(
            builder: std::sync::RwLockWriteGuard<IrFnBuilderPtr>,
            a: std::sync::RwLockReadGuard<BValuePtr>,
            name: Option<&str>,
        ) -> std::sync::Arc<std::sync::RwLock<BValuePtr>> {
            let builder_base =
                unsafe { xlsynth_sys::xls_function_builder_as_builder_base(builder.ptr) };
            let name_cstr = name.map(|s| std::ffi::CString::new(s).unwrap());
            let name_ptr = name_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
            let builder_base = unsafe { xlsynth_sys::$ffi_func(builder_base, a.ptr, name_ptr) };
            std::sync::Arc::new(std::sync::RwLock::new(BValuePtr { ptr: builder_base }))
        }
    };
}

impl_unary_ir_builder!(xls_function_builder_add_not, xls_builder_base_add_not);
impl_unary_ir_builder!(xls_function_builder_add_negate, xls_builder_base_add_negate);
impl_unary_ir_builder!(
    xls_function_builder_add_reverse,
    xls_builder_base_add_reverse
);
impl_unary_ir_builder!(
    xls_function_builder_add_or_reduce,
    xls_builder_base_add_or_reduce
);
impl_unary_ir_builder!(
    xls_function_builder_add_and_reduce,
    xls_builder_base_add_and_reduce
);
impl_unary_ir_builder!(
    xls_function_builder_add_xor_reduce,
    xls_builder_base_add_xor_reduce
);
impl_unary_ir_builder!(xls_function_builder_add_ctz, xls_builder_base_add_ctz);
impl_unary_ir_builder!(xls_function_builder_add_clz, xls_builder_base_add_clz);
impl_unary_ir_builder!(xls_function_builder_add_encode, xls_builder_base_add_encode);
impl_unary_ir_builder!(
    xls_function_builder_add_identity,
    xls_builder_base_add_identity
);
