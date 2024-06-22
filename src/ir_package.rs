// SPDX-License-Identifier: Apache-2.0

use crate::c_api;
use crate::c_api::{CIrFunction, CIrPackage};
use crate::xlsynth_error::XlsynthError;
use crate::IrValue;

pub struct IrPackage {
    pub(crate) ptr: *mut CIrPackage,
}

impl IrPackage {
    #[allow(dead_code)]
    pub fn parse_ir(ir: &str, filename: Option<&str>) -> Result<Self, XlsynthError> {
        c_api::xls_parse_ir_package(ir, filename)
    }

    pub fn get_function(&self, name: &str) -> Result<IrFunction, XlsynthError> {
        c_api::xls_package_get_function(self.ptr, name)
    }

    pub fn to_string(&self) -> String {
        c_api::xls_package_to_string(self.ptr).unwrap()
    }

    pub fn get_type_for_value(&self, value: &IrValue) -> Result<IrType, XlsynthError> {
        c_api::xls_package_get_type_for_value(self.ptr, value.ptr)
    }
}

impl Drop for IrPackage {
    fn drop(&mut self) {
        c_api::xls_package_free(self.ptr).expect("dealloc success");
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
        write!(f, "{}", c_api::xls_function_type_to_string(self.ptr).unwrap())
    }
}

pub struct IrFunction {
    pub(crate) ptr: *mut CIrFunction,
}

impl IrFunction {
    pub fn interpret(&self, args: &[IrValue]) -> Result<IrValue, XlsynthError> {
        c_api::xls_interpret_function(self.ptr, args)
    }

    pub fn get_name(&self) -> String {
        c_api::xls_function_get_name(self.ptr).unwrap()
    }

    pub fn get_type(&self) -> Result<IrFunctionType, XlsynthError> {
        c_api::xls_function_get_type(self.ptr)
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
        assert_eq!(
            f_type.to_string(),
            "(bits[32]) -> bits[32]".to_string()
        );

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
