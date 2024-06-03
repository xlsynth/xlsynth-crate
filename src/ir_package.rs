use crate::c_api::{
    xls_package_free, xls_package_get_function, xls_parse_ir_package, CIrFunction, CIrPackage,
};
use crate::xlsynth_error::XlsynthError;

pub(crate) struct IrPackage {
    pub(crate) ptr: *mut CIrPackage,
}

impl IrPackage {
    #[allow(dead_code)]
    pub fn parse_ir(ir: &str, filename: Option<&str>) -> Result<Self, XlsynthError> {
        xls_parse_ir_package(ir, filename)
    }

    #[allow(dead_code)]
    fn get_function(&self, name: &str) -> Result<IrFunction, XlsynthError> {
        xls_package_get_function(self.ptr, name)
    }
}

impl Drop for IrPackage {
    fn drop(&mut self) {
        xls_package_free(self.ptr).expect("dealloc success");
    }
}

pub(crate) struct IrFunction {
    #[allow(dead_code)]
    pub(crate) ptr: *mut CIrFunction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_package_parse() {
        let ir =
            "package test\nfn f() -> bits[32] { ret literal.1: bits[32] = literal(value=42) }\n";
        let package = IrPackage::parse_ir(ir, None).expect("parse success");
        package.get_function("f").expect("should find function");
    }
}
