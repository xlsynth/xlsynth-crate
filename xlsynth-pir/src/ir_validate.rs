// SPDX-License-Identifier: Apache-2.0

//! Compatibility names for the PIR verification API.
//!
//! New code should use [`crate::ir_verify`]. This module remains public so
//! callers using the previous validation terminology do not break.

use crate::ir::{BlockMetadata, Fn, Package};

pub use crate::ir_verify::{InstantiationPortDirection, VerifyError as ValidationError};

/// Compatibility alias for [`crate::ir_verify::verify_package`].
pub fn validate_package(package: &Package) -> Result<(), ValidationError> {
    crate::ir_verify::verify_package(package)
}

/// Compatibility alias for [`crate::ir_verify::verify_function_in_package`].
pub fn validate_fn(function: &Fn, parent: &Package) -> Result<(), ValidationError> {
    crate::ir_verify::verify_function_in_package(function, parent)
}

/// Compatibility alias for [`crate::ir_verify::verify_block_in_package`].
pub fn validate_block(
    function: &Fn,
    metadata: &BlockMetadata,
    parent: &Package,
    member_index: usize,
) -> Result<(), ValidationError> {
    crate::ir_verify::verify_block_in_package(function, metadata, parent, member_index)
}
