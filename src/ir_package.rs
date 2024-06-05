use crate::c_api::{
    xls_interpret_function, xls_package_free, xls_package_get_function, xls_parse_ir_package,
    CIrFunction, CIrPackage,
};
use crate::xlsynth_error::XlsynthError;
use crate::IrValue;

pub struct IrPackage {
    pub(crate) ptr: *mut CIrPackage,
}

impl IrPackage {
    #[allow(dead_code)]
    pub fn parse_ir(ir: &str, filename: Option<&str>) -> Result<Self, XlsynthError> {
        xls_parse_ir_package(ir, filename)
    }

    pub fn get_function(&self, name: &str) -> Result<IrFunction, XlsynthError> {
        xls_package_get_function(self.ptr, name)
    }
}

impl Drop for IrPackage {
    fn drop(&mut self) {
        xls_package_free(self.ptr).expect("dealloc success");
    }
}

pub struct IrFunction {
    #[allow(dead_code)]
    pub(crate) ptr: *mut CIrFunction,
}

impl IrFunction {
    pub fn interpret(&self, args: &[IrValue]) -> Result<IrValue, XlsynthError> {
        xls_interpret_function(self.ptr, args)
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
        let result = f.interpret(&[]).expect("interpret success");
        assert_eq!(result, IrValue::parse_typed("bits[32]:42").unwrap());
    }

    #[test]
    fn test_plus_one_fn_interp() {
        let ir = "package test\nfn f(x: bits[32]) -> bits[32] {
    literal.2: bits[32] = literal(value=1)
    ret add.1: bits[32] = add(x, literal.2)
}";
        let package = IrPackage::parse_ir(ir, None).expect("parse success");
        let f = package.get_function("f").expect("should find function");
        let ft = IrValue::parse_typed("bits[32]:42").unwrap();
        let result = f.interpret(&[ft]).expect("interpret success");
        let want = IrValue::parse_typed("bits[32]:43").unwrap();
        assert_eq!(result, want);
    }
}
