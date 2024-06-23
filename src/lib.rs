// SPDX-License-Identifier: Apache-2.0

pub mod c_api;
pub mod ir_package;
pub mod ir_value;
pub mod xlsynth_error;

pub use ir_package::IrFunction;
pub use ir_package::IrPackage;
pub use ir_value::IrValue;
pub use xlsynth_error::XlsynthError;

pub fn convert_dslx_to_ir(dslx: &str, path: &std::path::Path) -> Result<IrPackage, XlsynthError> {
    let ir_text = c_api::xls_convert_dslx_to_ir(dslx, path)?;
    // Get the filename as an Option<&str>
    let filename = path.file_name().and_then(|s| s.to_str());
    IrPackage::parse_ir(&ir_text, filename)
}

pub fn optimize_ir(ir: &IrPackage, top: &str) -> Result<IrPackage, XlsynthError> {
    let ir_text = c_api::xls_optimize_ir(&ir.to_string(), top)?;
    IrPackage::parse_ir(&ir_text, ir.filename())
}

pub fn mangle_dslx_name(module: &str, name: &str) -> Result<String, XlsynthError> {
    c_api::xls_mangle_dslx_name(module, name)
}