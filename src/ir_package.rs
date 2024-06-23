// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::c_api;
use crate::c_api::{CIrFunction, CIrPackage};
use crate::xlsynth_error::XlsynthError;
use crate::IrValue;

pub struct IrPackage {
    pub(crate) ptr: Arc<RwLock<*mut CIrPackage>>,
    // TODO(cdleary): 2024-06-23 This is the filename that is passed to the "parse package IR"
    // C API functionality. We should be able to remove this field and recover the value
    // from the built package if we had the appropriate C API.
    pub(crate) filename: Option<String>,
}

unsafe impl Send for IrPackage {}
unsafe impl Sync for IrPackage {}

impl IrPackage {
    pub fn parse_ir(ir: &str, filename: Option<&str>) -> Result<Self, XlsynthError> {
        c_api::xls_parse_ir_package(ir, filename)
    }

    pub fn get_function(&self, name: &str) -> Result<IrFunction, XlsynthError> {
        let read_guard = self.ptr.read().unwrap();
        c_api::xls_package_get_function(&self.ptr, read_guard, name)
    }

    pub fn to_string(&self) -> String {
        let read_guard = self.ptr.read().unwrap();
        c_api::xls_package_to_string(*read_guard).unwrap()
    }

    pub fn get_type_for_value(&self, value: &IrValue) -> Result<IrType, XlsynthError> {
        let write_guard = self.ptr.write().unwrap();
        c_api::xls_package_get_type_for_value(*write_guard, value.ptr)
    }

    pub fn filename(&self) -> Option<&str> {
        match self.filename {
            Some(ref s) => Some(s),
            None => None,
        }
    }
}

impl Drop for IrPackage {
    fn drop(&mut self) {
        let mut write_guard = self.ptr.write().unwrap();
        if !write_guard.is_null() {
            c_api::xls_package_free(*write_guard).expect("dealloc success");
            *write_guard = std::ptr::null_mut();
        }
    }
}

pub struct IrType {
    pub(crate) ptr: *mut c_api::CIrType,
}

impl std::fmt::Display for IrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", c_api::xls_type_to_string(self.ptr).unwrap())
    }
}

pub struct IrFunctionType {
    pub(crate) ptr: *mut c_api::CIrFunctionType,
}

impl std::fmt::Display for IrFunctionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            c_api::xls_function_type_to_string(self.ptr).unwrap()
        )
    }
}

pub struct IrFunction {
    pub(crate) parent: Arc<RwLock<*mut CIrPackage>>,
    pub(crate) ptr: *mut CIrFunction,
}

unsafe impl Send for IrFunction {}
unsafe impl Sync for IrFunction {}

impl IrFunction {
    pub fn interpret(&self, args: &[IrValue]) -> Result<IrValue, XlsynthError> {
        let package_read_guard: RwLockReadGuard<*mut CIrPackage> = self.parent.read().unwrap();
        c_api::xls_interpret_function(&package_read_guard, self.ptr, args)
    }

    pub fn get_name(&self) -> String {
        c_api::xls_function_get_name(self.ptr).unwrap()
    }

    pub fn get_type(&self) -> Result<IrFunctionType, XlsynthError> {
        let package_write_guard: RwLockWriteGuard<*mut CIrPackage> = self.parent.write().unwrap();
        c_api::xls_function_get_type(&package_write_guard, self.ptr)
    }
}

#[cfg(test)]
mod tests {
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
}
