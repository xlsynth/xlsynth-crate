// SPDX-License-Identifier: Apache-2.0

//! Enumeration of HDL versions that the code generators can emit.
//! Using a dedicated type avoids the ambiguity of passing around a bool.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerilogVersion {
    Verilog,
    SystemVerilog,
}

impl VerilogVersion {
    /// Convenience helper.
    pub fn is_system_verilog(self) -> bool {
        matches!(self, VerilogVersion::SystemVerilog)
    }
}

impl Default for VerilogVersion {
    fn default() -> Self {
        VerilogVersion::Verilog
    }
}
