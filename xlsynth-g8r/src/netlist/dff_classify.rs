// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for identifying DFF-like cells from a Liberty library.

use crate::liberty_proto::Library;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct DffCellSets {
    /// Cells treated as identity (D->Q): `Q = D`.
    pub identity: HashSet<String>,
    /// Cells treated as inverted (D->QN): `QN = NOT(D)`.
    pub inverted: HashSet<String>,
}

/// Computes DFF-like cell sets from Liberty pin functions.
///
/// Behavior matches `netlist::gv2ir`:
/// - `explicit_identity` seeds the identity set (typically from `--dff_cells`).
/// - `dff_cell_formula`, when set, adds any cell whose *output* pin has a
///   `function` string exactly matching the formula.
/// - `dff_cell_invert_formula`, when set, adds any cell whose *output* pin has
///   a `function` string exactly matching the invert formula.
pub fn classify_dff_cells_from_liberty(
    liberty_lib: &Library,
    explicit_identity: &HashSet<String>,
    dff_cell_formula: Option<&str>,
    dff_cell_invert_formula: Option<&str>,
) -> DffCellSets {
    let mut identity: HashSet<String> = explicit_identity.clone();
    if let Some(target_formula) = dff_cell_formula {
        for cell in &liberty_lib.cells {
            if cell
                .pins
                .iter()
                .any(|p| p.direction == 1 && p.function == target_formula)
            {
                identity.insert(cell.name.clone());
            }
        }
    }

    let mut inverted: HashSet<String> = HashSet::new();
    if let Some(invert_formula) = dff_cell_invert_formula {
        for cell in &liberty_lib.cells {
            if cell
                .pins
                .iter()
                .any(|p| p.direction == 1 && p.function == invert_formula)
            {
                inverted.insert(cell.name.clone());
            }
        }
    }

    DffCellSets { identity, inverted }
}
