use crate::c_api::{xls_parse_typed_value, xls_value_free, xls_value_to_string, CIrValue};
use crate::xlsynth_error::XlsynthError;

pub struct IrValue {
    pub(crate) ptr: *mut CIrValue,
}

impl IrValue {
    pub fn parse_typed(s: &str) -> Result<Self, XlsynthError> {
        xls_parse_typed_value(s)
    }
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
