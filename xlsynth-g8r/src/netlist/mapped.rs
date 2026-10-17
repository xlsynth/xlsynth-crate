// SPDX-License-Identifier: Apache-2.0

//! Owned netlist representation shared by technology mappers.

use crate::netlist::parse::{Net, NetlistModule};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// One mapped module with the nets and interned symbols referenced by its
/// cells.
#[derive(Debug)]
pub struct MappedNetlist {
    pub module: NetlistModule,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
}
