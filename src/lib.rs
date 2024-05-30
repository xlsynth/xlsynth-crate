extern crate libloading;
extern crate libc;

use libloading::{Library, Symbol};
use std::ffi::CString;

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

#[derive(Debug)]
pub struct XlsynthError(String);

pub struct Xlsynth {
    lib: Library
}

impl Xlsynth {
    pub fn new() -> Result<Self, libloading::Error> {
        unsafe {
            let lib = Library::new("libxls.dylib")?;
            Ok(Xlsynth {lib})
        }
    }

    pub fn convert_dslx_to_ir(&mut self, dslx: &str) -> Result<String, XlsynthError> {
        unsafe {
        let dlsym_convert_dslx_to_ir: Symbol<XlsConvertDslxToIr> = match self.lib.get(b"xls_convert_dslx_to_ir") {
            Ok(f) => f,
            Err(e) => return Err(XlsynthError(format!("Failed to load symbol `xls_convert_dslx_to_ir`: {}", e)))
        };
        let dslx = CString::new(dslx).unwrap();
        let path = CString::new("test_mod.x").unwrap();
        let module_name = CString::new("test_mod").unwrap();
        let dslx_stdlib_path = CString::new("/does/not/exist/").unwrap();
        
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_dslx_to_ir() {
        let mut xlsynth = Xlsynth::new().unwrap();
        let ir = xlsynth.convert_dslx_to_ir("fn f(x: u32) -> u32 { x }").unwrap();
        assert_eq!(ir, "package test_mod

file_number 0 \"test_mod.x\"

fn __test_mod__f(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
");
    }
}
