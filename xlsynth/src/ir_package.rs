// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub use crate::lib_support::RunResult;
use crate::lib_support::{self, c_str_to_rust, xls_function_jit_run, xls_make_function_jit};
use crate::xlsynth_error::XlsynthError;
use crate::{
    lib_support::{
        xls_function_get_name, xls_function_get_type, xls_function_type_to_string,
        xls_interpret_function, xls_package_free, xls_package_get_function,
        xls_package_get_type_for_value, xls_package_to_string, xls_parse_ir_package,
        xls_type_to_string,
    },
    IrValue,
};
use xlsynth_sys::{CIrFunction, CIrPackage, CScheduleAndCodegenResult};

// -- ScheduleAndCodegenResult

pub struct ScheduleAndCodegenResult {
    pub(crate) ptr: *mut CScheduleAndCodegenResult,
}

impl Drop for ScheduleAndCodegenResult {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                xlsynth_sys::xls_schedule_and_codegen_result_free(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl ScheduleAndCodegenResult {
    pub fn get_verilog_text(&self) -> Result<String, XlsynthError> {
        unsafe {
            let verilog = xlsynth_sys::xls_schedule_and_codegen_result_get_verilog_text(self.ptr);
            let verilog_str = c_str_to_rust(verilog);
            Ok(verilog_str)
        }
    }
}

// -- IrPackage & IrPackagePtr

/// We wrap up the raw C pointer in this type that implements `Drop` trait --
/// this allows us to wrap it up in an Arc so that types with derived lifetimes
/// (e.g. C function pointers) can grab a hold of the Arc to prevent
/// deallocation of its backing type.
pub(crate) struct IrPackagePtr(pub *mut CIrPackage);

impl IrPackagePtr {
    pub fn mut_c_ptr(&self) -> *mut CIrPackage {
        self.0
    }
    pub fn const_c_ptr(&self) -> *const CIrPackage {
        self.0
    }
}

impl Drop for IrPackagePtr {
    fn drop(&mut self) {
        if !self.0.is_null() {
            xls_package_free(self.0);
            self.0 = std::ptr::null_mut();
        }
    }
}

pub struct IrPackage {
    pub(crate) ptr: Arc<RwLock<IrPackagePtr>>,
    // TODO(cdleary): 2024-06-23 This is the filename that is passed to the "parse package IR"
    // C API functionality. We should be able to remove this field and recover the value
    // from the built package if we had the appropriate C API.
    pub(crate) filename: Option<String>,
}

unsafe impl Send for IrPackage {}
unsafe impl Sync for IrPackage {}

impl IrPackage {
    pub fn new(name: &str) -> Result<Self, XlsynthError> {
        lib_support::xls_package_new(name)
    }

    pub fn parse_ir(ir: &str, filename: Option<&str>) -> Result<Self, XlsynthError> {
        xls_parse_ir_package(ir, filename)
    }

    pub fn parse_ir_from_path(path: &std::path::Path) -> Result<Self, XlsynthError> {
        let ir = match std::fs::read_to_string(path) {
            Ok(ir) => ir,
            Err(e) => {
                return Err(XlsynthError(format!("Failed to read IR from path: {}", e)));
            }
        };
        let filename = path.file_name().and_then(|s| s.to_str());
        let ir_package = Self::parse_ir(&ir, filename)?;
        Ok(ir_package)
    }

    pub fn set_top_by_name(&mut self, name: &str) -> Result<(), XlsynthError> {
        let write_guard = self.ptr.write().unwrap();
        lib_support::xls_package_set_top_by_name(write_guard.mut_c_ptr(), name)
    }

    pub fn get_function(&self, name: &str) -> Result<IrFunction, XlsynthError> {
        let read_guard = self.ptr.read().unwrap();
        xls_package_get_function(&self.ptr, read_guard, name)
    }

    pub fn to_string(&self) -> String {
        let read_guard = self.ptr.read().unwrap();
        xls_package_to_string(read_guard.const_c_ptr()).unwrap()
    }

    pub fn get_type_for_value(&self, value: &IrValue) -> Result<IrType, XlsynthError> {
        let write_guard = self.ptr.write().unwrap();
        xls_package_get_type_for_value(write_guard.mut_c_ptr(), value.ptr)
    }

    pub fn get_bits_type(&self, bit_count: u64) -> IrType {
        let write_guard = self.ptr.write().unwrap();
        lib_support::xls_package_get_bits_type(write_guard.mut_c_ptr(), bit_count)
    }

    pub fn get_tuple_type(&self, members: &[IrType]) -> IrType {
        let write_guard = self.ptr.write().unwrap();
        lib_support::xls_package_get_tuple_type(write_guard.mut_c_ptr(), members)
    }

    pub fn types_eq(&self, a: &IrType, b: &IrType) -> Result<bool, XlsynthError> {
        let _read_guard = self.ptr.read().unwrap();
        Ok(a.ptr == b.ptr)
    }

    pub fn get_token_type(&self) -> IrType {
        let write_guard = self.ptr.write().unwrap();
        lib_support::xls_package_get_token_type(write_guard.mut_c_ptr())
    }

    pub fn get_array_type(&self, element_type: &IrType, size: i64) -> IrType {
        let write_guard = self.ptr.write().unwrap();
        lib_support::xls_package_get_array_type(write_guard.mut_c_ptr(), element_type.ptr, size)
    }

    pub fn filename(&self) -> Option<&str> {
        match self.filename {
            Some(ref s) => Some(s),
            None => None,
        }
    }
}

pub struct IrType {
    pub(crate) ptr: *mut xlsynth_sys::CIrType,
}

impl std::fmt::Display for IrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", xls_type_to_string(self.ptr).unwrap())
    }
}

impl IrType {
    pub fn get_flat_bit_count(&self) -> u64 {
        lib_support::xls_type_get_flat_bit_count(self.ptr)
    }
}

pub struct IrFunctionType {
    pub(crate) ptr: *mut xlsynth_sys::CIrFunctionType,
}

impl std::fmt::Display for IrFunctionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", xls_function_type_to_string(self.ptr).unwrap())
    }
}

impl IrFunctionType {
    /// Returns the number of parameters in this function type.
    pub fn param_count(&self) -> usize {
        lib_support::xls_function_type_param_count(self.ptr) as usize
    }

    /// Returns the type of the `i`th parameter.
    pub fn param_type(&self, index: usize) -> Result<IrType, XlsynthError> {
        lib_support::xls_function_type_get_param_type(self.ptr, index)
    }

    /// Returns the return type of the function.
    pub fn return_type(&self) -> IrType {
        lib_support::xls_function_type_get_return_type(self.ptr)
    }
}

pub struct IrFunction {
    pub(crate) parent: Arc<RwLock<IrPackagePtr>>,
    pub(crate) ptr: *mut CIrFunction,
}

unsafe impl Send for IrFunction {}
unsafe impl Sync for IrFunction {}

impl IrFunction {
    pub fn interpret(&self, args: &[IrValue]) -> Result<IrValue, XlsynthError> {
        let package_read_guard: RwLockReadGuard<IrPackagePtr> = self.parent.read().unwrap();
        xls_interpret_function(&package_read_guard, self.ptr, args)
    }

    pub fn get_name(&self) -> String {
        xls_function_get_name(self.ptr).unwrap()
    }

    pub fn get_type(&self) -> Result<IrFunctionType, XlsynthError> {
        let package_write_guard: RwLockWriteGuard<IrPackagePtr> = self.parent.write().unwrap();
        xls_function_get_type(&package_write_guard, self.ptr)
    }

    /// Returns the name of the `i`th parameter to this function.
    pub fn param_name(&self, index: usize) -> Result<String, XlsynthError> {
        lib_support::xls_function_get_param_name(self.ptr, index)
    }
}

pub struct IrFunctionJit {
    pub(crate) parent: Arc<RwLock<IrPackagePtr>>,
    pub(crate) ptr: *mut xlsynth_sys::CIrFunctionJit,
}

impl Drop for IrFunctionJit {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                xlsynth_sys::xls_function_jit_free(self.ptr);
            }
        }
    }
}

impl IrFunctionJit {
    pub fn new(function: &IrFunction) -> Result<Self, XlsynthError> {
        let package_read_guard: RwLockReadGuard<IrPackagePtr> = function.parent.read().unwrap();
        let ptr = xls_make_function_jit(&package_read_guard, function.ptr)?;
        Ok(IrFunctionJit {
            parent: function.parent.clone(),
            ptr,
        })
    }

    pub fn run(&self, args: &[IrValue]) -> Result<RunResult, XlsynthError> {
        let package_read_guard: RwLockReadGuard<IrPackagePtr> = self.parent.read().unwrap();
        xls_function_jit_run(&package_read_guard, self.ptr, args)
    }
}

#[cfg(test)]
mod tests {
    use crate::FnBuilder;

    use super::*;

    #[test]
    fn test_ir_package_parse() {
        let ir =
            "package test\nfn f() -> bits[32] { ret literal.1: bits[32] = literal(value=42) }\n";
        let package = IrPackage::parse_ir(ir, None).expect("parse success");
        let f = package.get_function("f").expect("should find function");
        assert_eq!(f.get_name(), "f");
        let result = f.interpret(&[]).expect("interpret success");
        assert_eq!(result, IrValue::parse_typed("bits[32]:42").unwrap());
    }

    #[test]
    fn test_plus_one_fn_interp() {
        let ir = "package test\nfn plus_one(x: bits[32]) -> bits[32] {
    literal.2: bits[32] = literal(value=1)
    ret add.1: bits[32] = add(x, literal.2)
}";
        let package = IrPackage::parse_ir(ir, None).expect("parse success");
        let f = package
            .get_function("plus_one")
            .expect("should find function");
        assert_eq!(f.get_name(), "plus_one");

        // Inspect the function type.
        let f_type = f.get_type().expect("get type success");
        assert_eq!(f_type.to_string(), "(bits[32]) -> bits[32]".to_string());

        let ft = IrValue::parse_typed("bits[32]:42").unwrap();
        let result = f.interpret(&[ft]).expect("interpret success");
        let want = IrValue::parse_typed("bits[32]:43").unwrap();
        assert_eq!(result, want);

        assert_eq!(
            package.get_type_for_value(&want).unwrap().to_string(),
            "bits[32]".to_string()
        );
    }

    #[test]
    fn test_ir_package_set_top_by_name() {
        let mut package = IrPackage::new("test_package").unwrap();
        // Build the identity function inside of the package that is not marked as top.
        let u32 = package.get_bits_type(32);
        let mut builder = FnBuilder::new(&mut package, "f", true);
        let x = builder.param("x", &u32);
        let _f = builder.build_with_return_value(&x);

        assert_eq!(
            package.to_string(),
            "package test_package\n\nfn f(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}
"
        );
        package.set_top_by_name("f").unwrap();
        assert_eq!(
            package.to_string(),
            "package test_package\n\ntop fn f(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}
"
        );
    }

    #[test]
    fn test_ir_type_tuple() {
        let package = IrPackage::new("test_package").unwrap();
        let u32 = package.get_bits_type(32);
        let u64 = package.get_bits_type(64);
        let tuple = package.get_tuple_type(&[u32, u64]);
        assert_eq!(tuple.to_string(), "(bits[32], bits[64])");
    }
}
