// SPDX-License-Identifier: Apache-2.0

use crate::aot_entrypoint_metadata::get_entrypoint_metadata;
pub use crate::aot_entrypoint_metadata::AotEntrypointMetadata;
use crate::ir_package::{IrPackage, TraceMessage};
use crate::xlsynth_error::XlsynthError;

pub type AotResult<T> = Result<T, XlsynthError>;

#[derive(Debug, Clone)]
pub struct AotCompiled {
    pub object_code: Vec<u8>,
    pub entrypoints_proto: Vec<u8>,
    pub metadata: AotEntrypointMetadata,
}

impl AotCompiled {
    pub fn compile_ir(ir_text: &str, top: &str) -> AotResult<Self> {
        if top.is_empty() {
            return Err(XlsynthError(
                "AOT invalid argument: top function name must not be empty".to_string(),
            ));
        }

        let package = IrPackage::parse_ir(ir_text, Some("aot_lib.ir"))
            .map_err(|e| XlsynthError(format!("AOT FFI call failed: {}", e.0)))?;
        let function = package
            .get_function(top)
            .map_err(|e| XlsynthError(format!("AOT FFI call failed: {}", e.0)))?;
        let (object_code, entrypoints_proto) = aot_compile_function(function.ptr)?;

        let metadata = get_entrypoint_metadata(&entrypoints_proto)
            .map_err(|e| XlsynthError(format!("AOT metadata parse failed: {}", e.0)))?;

        Ok(Self {
            object_code,
            entrypoints_proto,
            metadata,
        })
    }
}

fn aot_compile_function(function: *mut xlsynth_sys::CIrFunction) -> AotResult<(Vec<u8>, Vec<u8>)> {
    let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
    let mut object_code_out: *mut u8 = std::ptr::null_mut();
    let mut object_code_count_out = 0usize;
    let mut proto_out: *mut u8 = std::ptr::null_mut();
    let mut proto_count_out = 0usize;

    let success = unsafe {
        xlsynth_sys::xls_aot_compile_function(
            function,
            &mut error_out,
            &mut object_code_out,
            &mut object_code_count_out,
            &mut proto_out,
            &mut proto_count_out,
        )
    };
    if !success {
        return Err(XlsynthError(format!("AOT FFI call failed: {}", unsafe {
            crate::lib_support::c_str_to_rust(error_out)
        })));
    }
    if object_code_out.is_null() {
        return Err(XlsynthError(
            "AOT FFI call failed: xls_aot_compile_function returned null object_code buffer"
                .to_string(),
        ));
    }
    if proto_out.is_null() {
        unsafe {
            xlsynth_sys::xls_aot_object_code_free(object_code_out);
        }
        return Err(XlsynthError(
            "AOT FFI call failed: xls_aot_compile_function returned null entrypoints proto buffer"
                .to_string(),
        ));
    }

    let object_code = unsafe {
        std::slice::from_raw_parts(object_code_out as *const u8, object_code_count_out).to_vec()
    };
    let entrypoints_proto =
        unsafe { std::slice::from_raw_parts(proto_out as *const u8, proto_count_out).to_vec() };

    unsafe {
        xlsynth_sys::xls_aot_object_code_free(object_code_out);
        xlsynth_sys::xls_aot_entrypoints_proto_free(proto_out);
    }

    Ok((object_code, entrypoints_proto))
}

pub(crate) struct AotExecContext {
    ptr: *mut xlsynth_sys::CXlsAotExecContext,
}

impl AotExecContext {
    pub(crate) fn create(entrypoints_proto: &[u8]) -> Result<Self, XlsynthError> {
        if entrypoints_proto.is_empty() {
            return Err(XlsynthError(
                "AOT invalid argument: entrypoints proto bytes must not be empty".to_string(),
            ));
        }

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut out: *mut xlsynth_sys::CXlsAotExecContext = std::ptr::null_mut();
        let success = unsafe {
            xlsynth_sys::xls_aot_exec_context_create(
                entrypoints_proto.as_ptr(),
                entrypoints_proto.len(),
                &mut error_out,
                &mut out,
            )
        };

        if !success {
            return Err(XlsynthError(format!("AOT FFI call failed: {}", unsafe {
                crate::lib_support::c_str_to_rust(error_out)
            })));
        }

        if out.is_null() {
            return Err(XlsynthError(
                "AOT FFI call failed: xls_aot_exec_context_create reported success but returned nullptr"
                    .to_string(),
            ));
        }

        Ok(Self { ptr: out })
    }

    pub(crate) fn as_ptr(&self) -> *mut xlsynth_sys::CXlsAotExecContext {
        self.ptr
    }

    pub(crate) fn clear_events(&mut self) {
        unsafe {
            xlsynth_sys::xls_aot_exec_context_clear_events(self.ptr);
        }
    }

    pub(crate) fn trace_message(&self, index: usize) -> Result<TraceMessage, XlsynthError> {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut trace = xlsynth_sys::CTraceMessage {
            message: std::ptr::null_mut(),
            verbosity: 0,
        };

        let success = unsafe {
            xlsynth_sys::xls_aot_exec_context_get_trace_message(
                self.ptr,
                index,
                &mut error_out,
                &mut trace,
            )
        };
        if !success {
            return Err(XlsynthError(format!("AOT FFI call failed: {}", unsafe {
                crate::lib_support::c_str_to_rust(error_out)
            })));
        }

        Ok(TraceMessage {
            message: unsafe { crate::lib_support::c_str_to_rust(trace.message) },
            verbosity: trace.verbosity,
        })
    }

    pub(crate) fn assert_message(&self, index: usize) -> Result<String, XlsynthError> {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut assert_out: *mut std::os::raw::c_char = std::ptr::null_mut();

        let success = unsafe {
            xlsynth_sys::xls_aot_exec_context_get_assert_message(
                self.ptr,
                index,
                &mut error_out,
                &mut assert_out,
            )
        };
        if !success {
            return Err(XlsynthError(format!("AOT FFI call failed: {}", unsafe {
                crate::lib_support::c_str_to_rust(error_out)
            })));
        }

        Ok(unsafe { crate::lib_support::c_str_to_rust(assert_out) })
    }
}

impl Drop for AotExecContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                xlsynth_sys::xls_aot_exec_context_free(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}
