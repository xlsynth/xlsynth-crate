// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod ast;
pub mod check;
mod layout;
pub mod lower;
pub mod parser;
mod typed;

pub use check::{CheckedModule, check_module};
pub use lower::lower_module;
pub use parser::parse_module;
pub use xlsynth_pir::ir_builder::ResetBehavior;

use xlsynth_pir::ir::{FileTable, Package};

/// Parses and checks BX source, then links its blocks into the supplied IR
/// package.
pub fn compile(source: &str, package: Package, reset: ResetBehavior) -> Result<Package, String> {
    let module = parse_module(source)?;
    let checked = check_module(&module)?;
    lower_module(&checked, package, reset)
}

/// Creates a destination for BX modules whose externs require no linked IR.
pub fn empty_package(name: impl Into<String>) -> Package {
    Package {
        name: name.into(),
        file_table: FileTable::new(),
        members: Vec::new(),
        top: None,
    }
}
