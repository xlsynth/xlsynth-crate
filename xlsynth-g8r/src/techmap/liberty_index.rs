// SPDX-License-Identifier: Apache-2.0

//! Clean-sheet Liberty function index for combinational cut matching.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, Pin, PinDirection};
use crate::liberty_proto::TimingTableKind;
use crate::techmap::truth::{transform_truth, variable_truth};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One concrete cell/output/pin-permutation match.
#[derive(Clone, Debug)]
pub(super) struct CellBinding {
    pub cell_name: String,
    pub output_pin_name: String,
    pub input_pin_names: Vec<String>,
    /// For each cell input pin, the cut-leaf variable connected to it.
    pub input_to_leaf: Vec<usize>,
    /// For each cell input pin, whether the selected cut leaf is complemented.
    pub input_negated: Vec<bool>,
    /// Conservative scalar delay estimate for each cell input pin.
    pub input_delays: Vec<Option<f64>>,
    pub area: f64,
}

impl CellBinding {
    /// Returns whether every connected input has a usable timing estimate.
    pub fn has_complete_timing(&self) -> bool {
        self.input_delays.iter().all(Option::is_some)
    }

    /// Returns a deterministic identity used for tie-breaking.
    pub fn stable_key(&self) -> (&str, &str, &[String], &[usize], &[bool]) {
        (
            self.cell_name.as_str(),
            self.output_pin_name.as_str(),
            self.input_pin_names.as_slice(),
            self.input_to_leaf.as_slice(),
            self.input_negated.as_slice(),
        )
    }
}

/// Counts what the clean-sheet Liberty index accepted and skipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LibertyIndexStats {
    pub indexed_cell_outputs: usize,
    pub indexed_bindings: usize,
    pub skipped_cells: usize,
}

/// Exact truth-table index over eligible single-output combinational cells.
pub(super) struct LibertyCellIndex {
    by_truth: BTreeMap<(usize, u64), Vec<CellBinding>>,
    pub stats: LibertyIndexStats,
}

impl LibertyCellIndex {
    /// Builds a function index without relying on standard-cell family names.
    pub fn build(library: &Library, max_arity: usize) -> Result<Self> {
        let mut by_truth: BTreeMap<(usize, u64), Vec<CellBinding>> = BTreeMap::new();
        let mut stats = LibertyIndexStats::default();
        for cell in &library.cells {
            let Some(indexed) = index_cell(library, cell, max_arity)? else {
                stats.skipped_cells += 1;
                continue;
            };
            stats.indexed_cell_outputs += 1;
            for (truth, binding) in indexed {
                by_truth
                    .entry((binding.input_pin_names.len(), truth))
                    .or_default()
                    .push(binding);
                stats.indexed_bindings += 1;
            }
        }
        for bindings in by_truth.values_mut() {
            bindings.sort_by(binding_order);
        }
        if by_truth.is_empty() {
            return Err(anyhow!(
                "Liberty library has no eligible single-output combinational cells with parseable functions"
            ));
        }
        Ok(Self { by_truth, stats })
    }

    /// Returns every deterministic cell binding for one cut truth table.
    pub fn matches(&self, arity: usize, truth: u64) -> &[CellBinding] {
        self.by_truth
            .get(&(arity, truth))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the cheapest unary identity cell, if the library has one.
    pub fn best_buffer(&self) -> Option<&CellBinding> {
        self.matches(1, variable_truth(1, 0))
            .iter()
            .find(|binding| !binding.input_negated[0])
    }

    /// Returns the cheapest unary inverter cell, if the library has one.
    pub fn best_inverter(&self) -> Option<&CellBinding> {
        self.matches(1, 0b01)
            .iter()
            .find(|binding| !binding.input_negated[0])
    }

    /// Returns the cheapest zero-input constant driver, if available.
    pub fn best_constant(&self, value: bool) -> Option<&CellBinding> {
        self.matches(0, u64::from(value)).first()
    }
}

fn index_cell(
    library: &Library,
    cell: &Cell,
    max_arity: usize,
) -> Result<Option<Vec<(u64, CellBinding)>>> {
    if cell.dont_use == Some(true)
        || !cell.sequential.is_empty()
        || cell.clock_gate.is_some()
        || !cell.area.is_finite()
        || cell.area < 0.0
    {
        return Ok(None);
    }

    let input_pins: Vec<&Pin> = cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32)
        .collect();
    if input_pins.iter().any(|pin| pin.is_clocking_pin) || input_pins.len() > max_arity {
        return Ok(None);
    }
    let output_pins: Vec<&Pin> = cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Output as i32)
        .collect();
    if output_pins.len() != 1 {
        return Ok(None);
    }
    let output_pin = output_pins[0];
    let formula_text = library.resolve_string(&output_pin.function);
    if formula_text.is_empty() {
        return Ok(None);
    }
    let term = match parse_formula(formula_text) {
        Ok(term) => term,
        Err(_) => return Ok(None),
    };

    let input_pin_names: Vec<String> = input_pins
        .iter()
        .map(|pin| library.resolve_string(&pin.name).to_string())
        .collect();
    let formula_inputs: BTreeSet<String> = term.inputs().into_iter().collect();
    let declared_inputs: BTreeSet<String> = input_pin_names.iter().cloned().collect();
    if declared_inputs.len() != input_pin_names.len() || formula_inputs != declared_inputs {
        return Ok(None);
    }

    let truth = formula_truth(&term, input_pin_names.as_slice())?;
    let output_pin_name = library.resolve_string(&output_pin.name).to_string();
    let input_delays: Vec<Option<f64>> = input_pin_names
        .iter()
        .map(|input_name| estimated_input_delay(library, output_pin, input_name.as_str()))
        .collect();
    let mut indexed = Vec::new();
    for input_to_leaf in permutations(input_pin_names.len()) {
        for input_negated in polarity_vectors(input_pin_names.len()) {
            indexed.push((
                transform_truth(truth, input_to_leaf.as_slice(), input_negated.as_slice()),
                CellBinding {
                    cell_name: cell.name.clone(),
                    output_pin_name: output_pin_name.clone(),
                    input_pin_names: input_pin_names.clone(),
                    input_to_leaf: input_to_leaf.clone(),
                    input_negated,
                    input_delays: input_delays.clone(),
                    area: cell.area,
                },
            ));
        }
    }
    Ok(Some(indexed))
}

fn formula_truth(term: &Term, input_pin_names: &[String]) -> Result<u64> {
    let mut truth = 0u64;
    for assignment in 0..(1usize << input_pin_names.len()) {
        let values: HashMap<String, bool> = input_pin_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), ((assignment >> index) & 1) != 0))
            .collect();
        let value = term.evaluate_partial(&values).ok_or_else(|| {
            anyhow!("cell formula did not evaluate after binding every declared input")
        })?;
        if value {
            truth |= 1u64 << assignment;
        }
    }
    Ok(truth)
}

fn estimated_input_delay(library: &Library, output_pin: &Pin, input_pin_name: &str) -> Option<f64> {
    let mut max_delay: Option<f64> = None;
    for arc in &output_pin.timing_arcs {
        if library.resolve_string(&arc.related_pin) != input_pin_name {
            continue;
        }
        for table in &arc.tables {
            if !matches!(
                table.kind,
                TimingTableKind::CellRise | TimingTableKind::CellFall
            ) {
                continue;
            }
            for value in library.timing_table_values(table) {
                let value = f64::from(*value);
                if !value.is_finite() {
                    continue;
                }
                max_delay = Some(max_delay.map_or(value, |current| current.max(value)));
            }
        }
    }
    max_delay
}

fn permutations(size: usize) -> Vec<Vec<usize>> {
    let mut values: Vec<usize> = (0..size).collect();
    let mut result = Vec::new();
    collect_permutations(0, values.as_mut_slice(), &mut result);
    result
}

fn polarity_vectors(size: usize) -> Vec<Vec<bool>> {
    (0..(1usize << size))
        .map(|mask| (0..size).map(|index| ((mask >> index) & 1) != 0).collect())
        .collect()
}

fn collect_permutations(start: usize, values: &mut [usize], result: &mut Vec<Vec<usize>>) {
    if start == values.len() {
        result.push(values.to_vec());
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        collect_permutations(start + 1, values, result);
        values.swap(start, index);
    }
}

fn binding_order(lhs: &CellBinding, rhs: &CellBinding) -> std::cmp::Ordering {
    lhs.area
        .total_cmp(&rhs.area)
        .then_with(|| lhs.stable_key().cmp(&rhs.stable_key()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_model::{Cell, LibraryBuilder, Pin};

    fn pin(
        builder: &mut LibraryBuilder,
        direction: PinDirection,
        name: &str,
        function: &str,
    ) -> Pin {
        Pin {
            direction: direction as i32,
            name: builder.intern_string(name).unwrap(),
            function: builder.intern_string(function).unwrap(),
            ..Default::default()
        }
    }

    #[test]
    fn indexes_formula_without_cell_name_conventions() {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![Cell {
            name: "mystery_gate".to_string(),
            pins: vec![
                pin(&mut builder, PinDirection::Input, "A", ""),
                pin(&mut builder, PinDirection::Input, "B", ""),
                pin(&mut builder, PinDirection::Output, "Y", "A * B"),
            ],
            area: 2.5,
            ..Default::default()
        }];
        let library = builder.finish();

        let index = LibertyCellIndex::build(&library, 6).unwrap();

        assert_eq!(index.matches(2, 0b1000).len(), 2);
        assert_eq!(index.matches(2, 0b1000)[0].cell_name, "mystery_gate");
    }

    #[test]
    fn skips_sequential_and_unused_input_cells() {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![
            Cell {
                name: "unused_input".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "A"),
                ],
                area: 1.0,
                ..Default::default()
            },
            Cell {
                name: "good".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "!A"),
                ],
                area: 1.0,
                ..Default::default()
            },
        ];
        let library = builder.finish();

        let index = LibertyCellIndex::build(&library, 6).unwrap();

        assert_eq!(index.stats.indexed_cell_outputs, 1);
        assert_eq!(index.best_inverter().unwrap().cell_name, "good");
    }
}
