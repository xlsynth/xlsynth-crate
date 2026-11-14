// SPDX-License-Identifier: Apache-2.0

//! Deterministic, Rust-native SystemVerilog generation for XLS block IR and
//! DSLX types.

mod arithmetic;
mod block;
mod error;
mod hierarchy;
mod ops;
mod options;
mod priority;
mod slicing;
mod stages;
pub mod sv_bridge_builder;

pub use block::inline_invokes_for_codegen;
pub use error::BlockCodegenError;
pub use options::{BlockCodegenOptions, BlockCodegenOutput, BlockCodegenWithRegisterNames, Layout};

use xlsynth_pir::ir::Package;

/// Emits the selected block and its transitive block dependencies.
/// Call [`inline_invokes_for_codegen`] first if those blocks contain function
/// calls.
pub fn emit_system_verilog(
    package: &Package,
    options: &BlockCodegenOptions,
) -> Result<BlockCodegenOutput, BlockCodegenError> {
    block::emit_package(package, options, None)
}

/// Emits the RTL and records the allocated signal name for each representable
/// register in the selected top block.
/// The same input requirements apply as for [`emit_system_verilog`].
pub fn emit_system_verilog_with_register_names(
    package: &Package,
    options: &BlockCodegenOptions,
) -> Result<BlockCodegenWithRegisterNames, BlockCodegenError> {
    let mut top_register_names = std::collections::BTreeMap::new();
    let rtl = block::emit_package(package, options, Some(&mut top_register_names))?;
    Ok(BlockCodegenWithRegisterNames {
        system_verilog: rtl.system_verilog,
        top_register_names,
    })
}
