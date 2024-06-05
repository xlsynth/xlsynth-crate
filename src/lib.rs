pub mod c_api;
pub mod ir_package;
pub mod ir_value;
pub mod xlsynth_error;

pub use ir_value::IrValue;
pub use ir_package::IrPackage;
pub use c_api::xls_convert_dslx_to_ir as convert_dslx_to_ir;
use xlsynth_error::XlsynthError;

// TODO(cdleary): 2024-06-04: This is a temporary implementation until it's plumbed out of the DSO.
pub fn mangle_dslx(module_name: &str, function_name: &str) -> Result<String, XlsynthError> {
    let mangled = format!("__{}__{}", module_name, function_name);
    Ok(mangled)
}