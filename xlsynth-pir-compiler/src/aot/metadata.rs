// SPDX-License-Identifier: Apache-2.0

//! PIR AOT package metadata serialized beside checked-in IR.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::CompilerError;

pub const PIR_AOT_METADATA_FORMAT_VERSION: u32 = 3;

/// Serializable public API and source manifest for one PIR AOT package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PirAotPackageMetadata {
    pub format_version: u32,
    pub modules: Vec<PirAotModule>,
    pub entrypoints: Vec<PirAotEntrypoint>,
}

/// One generated public module containing type declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PirAotModule {
    pub path: Vec<String>,
    pub declarations: Vec<PirAotDecl>,
}

/// One public type declaration in PIR AOT metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PirAotDecl {
    Struct {
        name: String,
        fields: Vec<PirAotField>,
    },
    Enum {
        name: String,
        signedness: PirAotSignedness,
        bit_count: usize,
        variants: Vec<PirAotEnumVariant>,
    },
    Alias {
        name: String,
        target: PirAotType,
    },
}

/// One public field in a metadata-backed generated struct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PirAotField {
    pub name: String,
    pub ty: PirAotType,
}

/// One public enum-like constant in a metadata-backed generated enum wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PirAotEnumVariant {
    pub name: String,
    pub value: u64,
}

/// Signedness of a public DSLX-style bits value in PIR AOT metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PirAotSignedness {
    Unsigned,
    Signed,
}

/// Serializable public type expression for PIR AOT metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PirAotType {
    Bits {
        signedness: PirAotSignedness,
        bit_count: usize,
    },
    Token,
    Array {
        size: usize,
        element: Box<PirAotType>,
    },
    Tuple {
        elements: Vec<PirAotType>,
    },
    TypeRef {
        module: Vec<String>,
        name: String,
    },
}

/// Source IR backing one public PIR AOT entrypoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PirAotEntrypointSource {
    /// Entrypoint compiled from a checked-in IR file.
    IrFile { ir_file: String, ir_top: String },
    /// Entrypoint compiled from IR generated during this build.
    GeneratedIr { ir_top: String },
}

impl PirAotEntrypointSource {
    pub fn ir_top(&self) -> &str {
        match self {
            Self::IrFile { ir_top, .. } | Self::GeneratedIr { ir_top } => ir_top,
        }
    }

    pub fn ir_file(&self) -> Option<&str> {
        match self {
            Self::IrFile { ir_file, .. } => Some(ir_file),
            Self::GeneratedIr { .. } => None,
        }
    }
}

/// One PIR AOT entrypoint exposed as a generated runner module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PirAotEntrypoint {
    pub name: String,
    pub source: PirAotEntrypointSource,
    pub owning_module: Vec<String>,
    pub params: Vec<PirAotParam>,
    pub return_type: PirAotType,
}

/// One typed public parameter in a generated PIR AOT runner signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PirAotParam {
    pub name: String,
    pub ty: PirAotType,
}

impl PirAotPackageMetadata {
    /// Reads PIR AOT metadata from a JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self, CompilerError> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            CompilerError::InvalidArgument(format!("failed to read {}: {error}", path.display()))
        })?;
        serde_json::from_str(&text).map_err(|error| {
            CompilerError::InvalidArgument(format!(
                "failed to parse PIR AOT metadata {}: {error}",
                path.display()
            ))
        })
    }

    /// Serializes PIR AOT metadata as deterministic pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, CompilerError> {
        serde_json::to_string_pretty(self)
            .map(|mut json| {
                json.push('\n');
                json
            })
            .map_err(|error| {
                CompilerError::Backend(format!("failed to serialize PIR AOT metadata: {error}"))
            })
    }
}
