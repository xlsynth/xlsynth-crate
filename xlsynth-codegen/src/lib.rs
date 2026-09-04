// SPDX-License-Identifier: Apache-2.0

//! Deterministic, Rust-native SystemVerilog generation for XLS block IR.

mod arithmetic;
mod block;
mod error;
mod hierarchy;
mod ops;
mod options;
mod priority;
mod slicing;
mod stages;

pub use error::BlockCodegenError;
pub use options::{BlockCodegenOptions, BlockCodegenOutput, Layout};

use xlsynth_pir::ir::Package;

/// Emits the selected block and its transitive block dependencies.
pub fn emit_system_verilog(
    package: &Package,
    options: &BlockCodegenOptions,
) -> Result<BlockCodegenOutput, BlockCodegenError> {
    block::emit_package(package, options)
}
