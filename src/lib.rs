extern crate libc;
extern crate libloading;

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;
use std::ffi::CString;
use std::sync::Mutex;

static LIBRARY: OnceCell<Mutex<Library>> = OnceCell::new();

fn get_library() -> &'static Mutex<Library> {
    LIBRARY.get_or_init(|| {
        let library =
            unsafe { Library::new("libxls.dylib").expect("dynamic library should be present") };
        Mutex::new(library)
    })
}

#[repr(C)]
struct CIrValue {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

pub struct IrValue {
    ptr: *mut CIrValue,
}

impl std::fmt::Display for IrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            xls_value_to_string(self.ptr).expect("stringify success")
        )
    }
}

impl std::fmt::Debug for IrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            xls_value_to_string(self.ptr).expect("stringify success")
        )
    }
}

impl Drop for IrValue {
    fn drop(&mut self) {
        xls_value_free(self.ptr).expect("dealloc success");
    }
}

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

type XlsParseTypedValue = unsafe extern "C" fn(
    text: *const std::os::raw::c_char,
    error_out: *mut *mut std::os::raw::c_char,
    value_out: *mut *mut CIrValue,
) -> bool;

type XlsValueToString =
    unsafe extern "C" fn(value: *const CIrValue, str_out: *mut *mut std::os::raw::c_char) -> bool;

#[derive(Debug)]
pub struct XlsynthError(pub String);

pub fn convert_dslx_to_ir(dslx: &str) -> Result<String, XlsynthError> {
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

pub fn parse_typed_value(s: &str) -> Result<IrValue, XlsynthError> {
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

fn xls_value_free(p: *mut CIrValue) -> Result<(), XlsynthError> {
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

fn xls_value_to_string(p: *mut CIrValue) -> Result<String, XlsynthError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_dslx_to_ir() {
        let ir =
            convert_dslx_to_ir("fn f(x: u32) -> u32 { x }").expect("ir conversion should succeed");
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
        let e: XlsynthError = parse_typed_value("asdf").expect_err("should not parse");
        assert_eq!(e.0, "INVALID_ARGUMENT: Expected token of type \"(\" @ 1:1, but found: Token(\"ident\", value=\"asdf\") @ 1:1");
    }

    #[test]
    fn test_parse_typed_value_bits32_42() {
        let v: IrValue = parse_typed_value("bits[32]:42").expect("should parse ok");
        assert_eq!(v.to_string(), "bits[32]:42");
    }
}
