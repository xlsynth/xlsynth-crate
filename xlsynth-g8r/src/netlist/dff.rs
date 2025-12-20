// SPDX-License-Identifier: Apache-2.0

//! DFF cell identification helpers for gate-level netlists.

use crate::liberty_proto::{Library, PinDirection};
use anyhow::{Result, anyhow};
use std::collections::HashSet;

/// DFF cell types classified into identity and inverted output categories.
///
/// These are primarily used by consumers that need to treat DFFs as boundary
/// elements (e.g. combinational path analysis or netlist-to-AIG projection).
#[derive(Debug, Clone)]
pub struct DffCellClassification {
    /// Cell types treated as DFFs with identity output behavior (Q = D).
    pub identity: HashSet<String>,
    /// Cell types treated as DFFs with inverted output behavior (QN = NOT(D)).
    pub inverted: HashSet<String>,
}

impl DffCellClassification {
    pub fn all_cell_types(&self) -> HashSet<String> {
        let mut out = self.identity.clone();
        out.extend(self.inverted.iter().cloned());
        out
    }
}

/// Classifies DFF cell types from explicit names plus optional Liberty formula
/// matches.
///
/// - `explicit_identity` are treated as identity-output DFFs.
/// - If `identity_output_formula` is set, any cell with an **output** pin
///   function exactly matching it is also treated as an identity-output DFF.
/// - If `inverted_output_formula` is set, any cell with an **output** pin
///   function exactly matching it is treated as an inverted-output DFF.
pub fn classify_dff_cells(
    lib: &Library,
    explicit_identity: &HashSet<String>,
    identity_output_formula: Option<&str>,
    inverted_output_formula: Option<&str>,
) -> Result<DffCellClassification> {
    let mut identity: HashSet<String> = explicit_identity.clone();
    let mut inverted: HashSet<String> = HashSet::new();

    if let Some(target_formula) = identity_output_formula {
        for cell in &lib.cells {
            if cell
                .pins
                .iter()
                .any(|p| p.direction == PinDirection::Output as i32 && p.function == target_formula)
            {
                identity.insert(cell.name.clone());
            }
        }
    }

    if let Some(invert_formula) = inverted_output_formula {
        for cell in &lib.cells {
            if cell
                .pins
                .iter()
                .any(|p| p.direction == PinDirection::Output as i32 && p.function == invert_formula)
            {
                inverted.insert(cell.name.clone());
            }
        }
    }

    let overlap: Vec<String> = identity
        .intersection(&inverted)
        .cloned()
        .collect::<Vec<String>>();
    if !overlap.is_empty() {
        return Err(anyhow!(format!(
            "DFF cell classification has overlap between identity and inverted sets: {}",
            overlap.join(", ")
        )));
    }

    Ok(DffCellClassification { identity, inverted })
}
