// SPDX-License-Identifier: Apache-2.0

//! Nominal source types and their physical block shapes.
//!
//! BX interface validation uses [`source_shape`] to check widths and tuple
//! positions against block IR. The readable emitter keeps the declarations
//! themselves so it can turn the same positions back into named fields.
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use xlsynth_pir::ir;

/// A source type declaration used by the executable equations and codecs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TypeDecl {
    pub(super) name: String,
    pub(super) fields: Vec<StructField>,
}

impl TypeDecl {
    pub(super) fn name(&self) -> &str {
        &self.name
    }
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
    /// A reference to a struct declaration in the same catalogue.
    Struct { name: String },
}

/// Carries the checked BX port and register types into the readable Lean model.
///
/// Interface validation matches BX declarations to IR names and physical types.
/// The readable evaluator and its codecs look up inputs and registers by name.
/// They pair each entry in `outputs` with the IR output port at the same index
/// when constructing the result or converting it between readable and raw Lean.
pub(super) struct StatefulInterface {
    /// Root BX module for unqualified source declarations.
    pub(super) bx_module: String,
    /// Written BX block name, qualified by the linker for a library
    /// declaration.
    pub(super) block_name: String,
    pub(super) inputs: BTreeMap<String, SourceType>,
    pub(super) registers: BTreeMap<String, SourceType>,
    pub(super) outputs: Vec<SourceType>,
}

/// Resolve a name within the collected leaf declarations, rejecting unknowns.
fn declaration<'a>(declarations: &'a [TypeDecl], name: &str) -> Result<&'a TypeDecl> {
    declarations
        .iter()
        .find(|declaration| declaration.name() == name)
        .with_context(|| format!("unknown supported source type {name}"))
}

/// Erase nominal types to the physical shapes used by block IR.
///
/// Array lengths survive erasure even when a named element becomes a tuple or
/// bits.
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
        SourceType::Struct { name } => Ok(ir::Type::Tuple(
            declaration(declarations, name)?
                .fields
                .iter()
                .map(|field| source_shape(&field.source_type, declarations).map(Box::new))
                .collect::<Result<_>>()?,
        )),
    }
}
