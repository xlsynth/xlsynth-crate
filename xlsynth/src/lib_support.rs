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
    CIrBits, CIrFunction, CIrFunctionJit, CIrFunctionType, CIrPackage, CIrType, CIrValue,
    CScheduleAndCodegenResult, CTraceMessage, XlsFormatPreference,
};

use crate::{
    ir_package::{IrFunctionType, IrPackagePtr, IrType, ScheduleAndCodegenResult},
    IrBits, IrValue, XlsynthError,
};

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
    let filename_cstr = filename.map(|s| CString::new(s).unwrap());
    let filename_ptr = filename_cstr
        .map(|s| s.as_ptr())
        .unwrap_or(std::ptr::null());
    let mut xls_package_out: *mut CIrPackage = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_parse_ir_package, ir_cstring.as_ptr(), filename_ptr; xls_package_out)?;
    let package = crate::ir_package::IrPackage {
        ptr: Arc::new(RwLock::new(IrPackagePtr(xls_package_out))),
        filename: filename.map(|s| s.to_string()),
    };
    Ok(package)
}

pub(crate) fn xls_type_to_string(t: *const CIrType) -> Result<String, XlsynthError> {
    let mut c_str_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    xls_ffi_call!(xlsynth_sys::xls_type_to_string, t; c_str_out)?;
    Ok(unsafe { c_str_to_rust(c_str_out) })
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
