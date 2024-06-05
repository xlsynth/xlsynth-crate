pub mod c_api;
pub mod ir_package;
pub mod ir_value;
pub mod xlsynth_error;

pub use ir_package::IrPackage;
pub use ir_value::IrValue;
pub use xlsynth_error::XlsynthError;

pub fn convert_dslx_to_ir(dslx: &str) -> Result<IrPackage, XlsynthError> {
    let ir_text = c_api::xls_convert_dslx_to_ir(dslx)?;
    IrPackage::parse_ir(&ir_text, None)
}

pub fn optimize_ir(ir: &IrPackage, top: &str) -> Result<IrPackage, XlsynthError> {
    let ir_text = c_api::xls_optimize_ir(&ir.to_string(), top)?;
    IrPackage::parse_ir(&ir_text, None)
}

pub fn mangle_dslx_name(module: &str, name: &str) -> Result<String, XlsynthError> {
    c_api::xls_mangle_dslx_name(module, name)
}
