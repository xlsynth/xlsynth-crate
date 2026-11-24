// SPDX-License-Identifier: Apache-2.0

//! Characterized buffer choices for registered output edits.

use crate::liberty::io_defaults::buffer_input_pin;
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::electrical::{InputLoad, OutputCharacterization, PinLimits, RiseFall};
use crate::netlist::sta::PreparedStaLibrary;

/// Returns the smaller of the output pin's maximum capacitance and the
/// largest load covered by its timing tables, or the only known bound.
///
/// The former is an electrical design rule; the latter bounds the
/// characterized timing data.
pub(super) fn capacitance_ceiling(rule: Option<f64>, characterized: Option<f64>) -> Option<f64> {
    match (rule, characterized) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// A non-inverting library buffer with the pin names and load information
/// needed to insert and time it in a buffer tree.
#[derive(Clone, Debug)]
pub(super) struct InsertableBufferCell {
    pub cell_index: usize,
    pub input_pin_name: String,
    pub output_pin_name: String,
    pub input_capacitance: RiseFall<f64>,
    pub input_fanout_load: f64,
    pub max_fanout: Option<f64>,
    pub output_load_ceiling: Option<f64>,
    pub max_transition: Option<f64>,
}

/// Selects the buffer with the highest ratio of supported output
/// capacitance to input capacitance. This favors cells that can
/// drive substantial loads while presenting a small load upstream.
///
/// Breaks ties by smaller area, then cell name.
pub(super) fn highest_gain_buffer(
    library: &Library,
    prepared: &PreparedStaLibrary<'_>,
) -> Option<InsertableBufferCell> {
    let candidates = (0..library.cells.len())
        .filter_map(|cell_index| buffer_for_insertion(library, cell_index))
        .filter(|b| {
            b.output_load_ceiling
                .is_some_and(|c| c.is_finite() && c > 0.)
                && [b.input_capacitance.rise, b.input_capacitance.fall]
                    .into_iter()
                    .all(|c| c.is_finite() && c > 0.)
                && prepared
                    .validate_independent_combinational_cell(b.cell_index)
                    .is_ok()
        });
    let gain = |b: &InsertableBufferCell| {
        b.output_load_ceiling.unwrap() / b.input_capacitance.rise.max(b.input_capacitance.fall)
    };
    candidates.min_by(|a, b| {
        let (left, right) = (&library.cells[a.cell_index], &library.cells[b.cell_index]);
        gain(b)
            .total_cmp(&gain(a))
            .then(left.area.total_cmp(&right.area))
            .then(left.name.cmp(&right.name))
    })
}

/// Returns a model for a non-inverting buffer the mapper can insert into the
/// netlist. `cell_index` selects an entry in `library.cells`. Returns `None`
/// if the cell is missing or unsuitable for insertion.
pub(super) fn buffer_for_insertion(
    library: &Library,
    cell_index: usize,
) -> Option<InsertableBufferCell> {
    let candidate = library.cells.get(cell_index)?;
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
    let input = buffer_input_pin(library, &candidate.name).ok()?;
    let output = candidate
        .pins
        .iter()
        .find(|p| p.direction == PinDirection::Output as i32)?;
    let load = InputLoad::for_pin(input)?;
    let limits = PinLimits::for_pin(output);
    if !limits.fanout.is_none_or(|v| v.is_finite() && v >= 0.) {
        return None;
    }
    Some(InsertableBufferCell {
        cell_index,
        input_pin_name: library.resolve_string(&input.name).to_owned(),
        output_pin_name: library.resolve_string(&output.name).to_owned(),
        input_capacitance: load.capacitance,
        input_fanout_load: load.fanout,
        max_fanout: limits.fanout,
        output_load_ceiling: capacitance_ceiling(
            limits.capacitance,
            OutputCharacterization::for_pin(library, output).max_load,
        ),
        max_transition: limits.transition,
    })
}
