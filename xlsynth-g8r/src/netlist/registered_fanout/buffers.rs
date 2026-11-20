// SPDX-License-Identifier: Apache-2.0

//! Characterized buffer choices for registered output edits.

use crate::liberty_model::{Library, PinDirection};
use crate::netlist::electrical::{InputLoad, OutputCharacterization, PinLimits};

/// Buffering stays within both the explicit pin rule and the characterized
/// table range. Keeping this choice here leaves the electrical analysis free
/// to report those two different constraints independently.
pub(super) fn capacitance_ceiling(rule: Option<f64>, characterized: Option<f64>) -> Option<f64> {
    match (rule, characterized) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// A characterized non-inverting buffer that can drive a proposed tree.
#[derive(Clone, Debug)]
pub(super) struct Buffer {
    pub cell: usize,
    pub input: String,
    pub output: String,
    pub input_cap: [f64; 2],
    pub input_fanout_load: f64,
    pub max_fanout: Option<f64>,
    pub max_capacitance: Option<f64>,
    pub max_transition: Option<f64>,
}

/// Only offer newly instantiable non-inverting cells whose input-to-output
/// timing the same STA implementation can independently evaluate.
pub(super) fn buffers(
    lib: &Library,
    prepared: &crate::netlist::sta::PreparedStaLibrary<'_>,
) -> Vec<Buffer> {
    let mut options = (0..lib.cells.len())
        .filter_map(|idx| buffer_model(lib, idx))
        .filter(|buffer| {
            prepared
                .validate_independent_combinational_cell(buffer.cell)
                .is_ok()
        })
        .collect::<Vec<_>>();
    options.sort_by(|a, b| {
        let (a, b) = (&lib.cells[a.cell], &lib.cells[b.cell]);
        a.area.total_cmp(&b.area).then_with(|| a.name.cmp(&b.name))
    });
    options
}

pub(super) fn buffer_model(lib: &Library, cell: usize) -> Option<Buffer> {
    let candidate = lib.cells.get(cell)?;
    if candidate.pins.len() != 2
        || candidate.use_for_size_only == Some(true)
        || !candidate.area.is_finite()
        || candidate.area <= 0.0
        || candidate.validate_mapping_policy_coverage().is_err()
    {
        return None;
    }
    // Resolves a unique cell and checks the exact truth table, functional
    // coverage, dont_use, clock pins and both port directions.
    let input = crate::liberty::io_defaults::buffer_input_pin(lib, &candidate.name).ok()?;
    let output = candidate
        .pins
        .iter()
        .find(|p| p.direction == PinDirection::Output as i32)?;
    let load = InputLoad::for_pin(input)?;
    let limits = PinLimits::for_pin(output);
    if !limits.fanout.is_none_or(|v| v.is_finite() && v >= 0.) {
        return None;
    }
    Some(Buffer {
        cell,
        input: lib.resolve_string(&input.name).to_owned(),
        output: lib.resolve_string(&output.name).to_owned(),
        input_cap: load.capacitance,
        input_fanout_load: load.fanout,
        max_fanout: limits.fanout,
        max_capacitance: capacitance_ceiling(
            limits.capacitance,
            OutputCharacterization::for_pin(lib, output).max_load,
        ),
        max_transition: limits.transition,
    })
}
