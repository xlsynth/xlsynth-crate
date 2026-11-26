// SPDX-License-Identifier: Apache-2.0

//! Validate mapped cells before the alternate library is used for sizing.

use crate::liberty::Library;
use crate::liberty_model::Cell;
use crate::netlist::io::{ParsedNetlist, select_module};
use anyhow::{Context, Result, ensure};
use std::collections::{HashMap, HashSet};

/// Checks each mapped master and removes conflicting shared masters from
/// the alternate sizing view before the search chooses replacements.
pub fn validate_and_filter_sizing_library(
    mapped: &ParsedNetlist,
    mapping: &Library,
    sizing: &mut Library,
) -> Result<()> {
    let module = select_module(mapped, None)?;
    let name = mapped
        .interner
        .resolve(module.name)
        .context("missing module name")?;
    crate::netlist::mapped::validate_module_name(name, sizing).map_err(anyhow::Error::msg)?;
    let mut cells = HashMap::with_capacity(sizing.cells.len());
    for cell in &sizing.cells {
        ensure!(
            cells.insert(cell.name.as_str(), cell).is_none(),
            "duplicate Liberty cell name '{}' in sizing library",
            cell.name
        );
    }
    let mut incompatible = HashSet::new();
    for cell in &mapping.cells {
        if cells
            .get(cell.name.as_str())
            .is_some_and(|other| !compatible_cell(cell, mapping, other, sizing))
        {
            incompatible.insert(cell.name.as_str());
        }
    }
    for instance in &module.instances {
        let name = mapped
            .interner
            .resolve(instance.type_name)
            .context("missing cell name")?;
        ensure!(
            cells.contains_key(name),
            "sizing library is missing mapped cell '{name}'"
        );
        ensure!(
            !incompatible.contains(name),
            "sizing library has incompatible behavior or pins for cell '{name}'"
        );
    }
    sizing
        .cells
        .retain(|cell| !incompatible.contains(cell.name.as_str()));
    Ok(())
}

/// Exact pin equations cover clock gates and large cells outside ordinary
/// sizing families. For different equations use the resizer's Boolean test.
fn compatible_cell(cell: &Cell, mapping: &Library, other: &Cell, sizing: &Library) -> bool {
    if cell.sequential != other.sequential || cell.clock_gate != other.clock_gate {
        return false;
    }
    if cell.unsupported_semantics == other.unsupported_semantics
        && pin_equations(cell, mapping) == pin_equations(other, sizing)
    {
        return true;
    }
    super::cell_signature(cell, mapping)
        .is_some_and(|signature| super::cell_signature(other, sizing) == Some(signature))
}

/// A named Liberty pin's exact interface and Boolean equation.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct PinEquation<'a> {
    name: &'a str,
    direction: i32,
    is_clocking_pin: bool,
    function: &'a str,
}

fn pin_equations<'a>(cell: &'a Cell, library: &'a Library) -> Vec<PinEquation<'a>> {
    let mut pins: Vec<_> = cell
        .pins
        .iter()
        .map(|pin| PinEquation {
            name: library.resolve_string(&pin.name),
            direction: pin.direction,
            is_clocking_pin: pin.is_clocking_pin,
            function: library.resolve_string(&pin.function),
        })
        .collect();
    pins.sort_unstable();
    pins
}
