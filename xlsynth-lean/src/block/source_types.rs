// SPDX-License-Identifier: Apache-2.0

//! Nominal source types and their physical block shapes.
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use xlsynth_pir::ir;

/// A source type declaration used by the executable equations and codecs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum TypeDecl {
    /// An unsigned two-bit enum with distinct named physical encodings.
    Enum {
        name: String,
        width: usize,
        members: Vec<EnumMember>,
    },
    /// A nonempty struct whose ordered fields determine its physical tuple
    /// shape.
    Struct {
        name: String,
        fields: Vec<StructField>,
    },
    /// A named reference to a supported type, retaining alias-chain identity.
    Alias { name: String, target: SourceType },
}

impl TypeDecl {
    /// The declared source type name.
    pub(super) fn name(&self) -> &str {
        match self {
            Self::Enum { name, .. } | Self::Struct { name, .. } | Self::Alias { name, .. } => name,
        }
    }
}

/// One named, native-evaluated enum encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EnumMember {
    /// DSLX constructor name.
    pub(super) name: String,
    /// Physical encoding, validated to be distinct and less than four.
    pub(super) value: u64,
}

/// A source field in declaration order and its nominal or primitive type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StructField {
    /// Field name from the struct declaration.
    pub(super) name: String,
    /// Field type before erasing nominal names to its physical shape.
    pub(super) source_type: SourceType,
}

/// A primitive type or nominal reference retained in the source catalogue.
///
/// Nominal references remain distinct even when their hardware widths coincide.
/// DSLX's unsigned one-bit type, including `bool`, uses the Boolean variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum SourceType {
    /// An unsigned one-bit value rendered as Lean `Bool`.
    Bool,
    /// Unsigned physical bits; collection uses this for widths greater than
    /// one.
    Bits { width: usize },
    /// Positional source tuple; its members retain their written types.
    Tuple { members: Vec<SourceType> },
    /// Fixed-length array whose elements retain their checked source type.
    Array {
        element: Box<SourceType>,
        size: usize,
    },
    /// A reference to an enum declaration in the same catalogue.
    Enum { name: String },
    /// A reference to a struct declaration in the same catalogue.
    Struct { name: String },
    /// A reference to an alias declaration in the same catalogue.
    Alias { name: String },
}

/// Carries the checked BX port and register types into the readable Lean model.
///
/// Interface validation matches BX declarations to IR names and physical types.
/// The readable evaluator and its codecs look up inputs and registers by name.
/// They pair each entry in `outputs` with the IR output port at the same index
/// when constructing the result or converting it between readable and raw Lean.
pub(super) struct StatefulInterface {
    pub(super) helpers: BTreeMap<String, HelperSignature>,
    pub(super) inputs: BTreeMap<String, SourceType>,
    pub(super) registers: BTreeMap<String, SourceType>,
    pub(super) outputs: Vec<SourceType>,
}

/// Source name and declared parameter and return types of a linked IR function.
pub(super) struct HelperSignature {
    pub(super) name: String,
    pub(super) params: Vec<(String, SourceType)>,
    pub(super) result: SourceType,
}

/// Resolve a name within the collected leaf declarations, rejecting unknowns.
fn declaration<'a>(declarations: &'a [TypeDecl], name: &str) -> Result<&'a TypeDecl> {
    declarations
        .iter()
        .find(|declaration| declaration.name() == name)
        .with_context(|| format!("unknown supported source type {name}"))
}

/// Expand transparent aliases while preserving named struct and enum
/// identities.
pub(super) fn resolve_aliases(ty: &SourceType, declarations: &[TypeDecl]) -> Result<SourceType> {
    match ty {
        SourceType::Alias { name } => {
            let TypeDecl::Alias { target, .. } = declaration(declarations, name)? else {
                bail!("source type {name} is not an alias")
            };
            resolve_aliases(target, declarations)
        }
        SourceType::Tuple { members } => Ok(SourceType::Tuple {
            members: members
                .iter()
                .map(|ty| resolve_aliases(ty, declarations))
                .collect::<Result<_>>()?,
        }),
        SourceType::Array { element, size } => Ok(SourceType::Array {
            element: Box::new(resolve_aliases(element, declarations)?),
            size: *size,
        }),
        nominal_or_primitive => Ok(nominal_or_primitive.clone()),
    }
}

/// Erase nominal types to the physical shapes used by block IR.
pub(super) fn source_shape(
    source_type: &SourceType,
    declarations: &[TypeDecl],
) -> Result<ir::Type> {
    match source_type {
        SourceType::Bool => Ok(ir::Type::Bits(1)),
        SourceType::Bits { width } => Ok(ir::Type::Bits(*width)),
        SourceType::Array { element, size } => Ok(ir::Type::new_array(
            source_shape(element, declarations)?,
            *size,
        )),
        SourceType::Tuple { members } => Ok(ir::Type::Tuple(
            members
                .iter()
                .map(|ty| source_shape(ty, declarations).map(Box::new))
                .collect::<Result<_>>()?,
        )),
        SourceType::Enum { name } => match declaration(declarations, name)? {
            TypeDecl::Enum { width, .. } => Ok(ir::Type::Bits(*width)),
            _ => bail!("source type {name} is not an enum"),
        },
        SourceType::Struct { name } => match declaration(declarations, name)? {
            TypeDecl::Struct { fields, .. } => Ok(ir::Type::Tuple(
                fields
                    .iter()
                    .map(|field| source_shape(&field.source_type, declarations).map(Box::new))
                    .collect::<Result<_>>()?,
            )),
            _ => bail!("source type {name} is not a struct"),
        },
        SourceType::Alias { name } => match declaration(declarations, name)? {
            TypeDecl::Alias { target, .. } => source_shape(target, declarations),
            _ => bail!("source type {name} is not an alias"),
        },
    }
}
