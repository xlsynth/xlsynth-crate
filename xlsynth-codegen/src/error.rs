// SPDX-License-Identifier: Apache-2.0

use std::fmt;

/// Explains why a block cannot be represented as supported SystemVerilog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockCodegenError {
    /// The package has no uniquely identifiable block top.
    TopSelection(String),
    /// The input block violates a structural code-generation requirement.
    InvalidBlock(String),
    /// A valid IR construct has no supported SystemVerilog representation.
    Unsupported(String),
    /// Register dependencies cannot form a strictly layered pipeline.
    NotPipeline(String),
    /// Building the SystemVerilog syntax tree failed.
    Emission(String),
}

impl fmt::Display for BlockCodegenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopSelection(message)
            | Self::InvalidBlock(message)
            | Self::Unsupported(message)
            | Self::NotPipeline(message)
            | Self::Emission(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for BlockCodegenError {}

impl From<xlsynth_vast::VastError> for BlockCodegenError {
    fn from(error: xlsynth_vast::VastError) -> Self {
        Self::Emission(error.to_string())
    }
}

impl From<xlsynth::XlsynthError> for BlockCodegenError {
    fn from(error: xlsynth::XlsynthError) -> Self {
        Self::Emission(error.to_string())
    }
}

impl From<xlsynth_pir::ValueError> for BlockCodegenError {
    fn from(error: xlsynth_pir::ValueError) -> Self {
        Self::Emission(error.to_string())
    }
}
