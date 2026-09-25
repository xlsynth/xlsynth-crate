// SPDX-License-Identifier: Apache-2.0

//! Native Rust representations and parsers for XLS IR values.
//!
//! This crate provides bitvectors of arbitrary width, recursive values and
//! their structural types, and typed value and `.irvals` parsing without native
//! library dependencies.

mod ir_type;
pub mod ir_values;
pub mod value;

pub use ir_type::{ArrayTypeData, StartAndLimit, Type};
pub use ir_values::{
    IrValuesFile, IrValuesFileKind, NamedIrValue, NamedIrValueSet, parse_ir_values,
    parse_ir_values_file,
};
pub use value::{IrArray, IrBits, IrFormatPreference, IrValue, ValueError};
