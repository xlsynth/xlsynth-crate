// SPDX-License-Identifier: Apache-2.0

//! Clean-sheet Liberty function index for combinational cut matching.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, Pin, PinDirection};
use crate::liberty_proto::TimingTableKind;
use crate::netlist::sta::{
    CombinationalOutputLoad, effective_input_capacitance_for_mapping,
    validate_output_pin_for_basic_sta,
};
use crate::techmap::truth::{transform_truth, variable_truth};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet, HashMap};

const NF_ROOT_VARIANTS_PER_FUNCTION: usize = 1;

/// One concrete cell/output/pin-permutation match.
#[derive(Clone, Debug)]
pub(super) struct CellBinding {
    pub cell_name: String,
    pub cell_index: usize,
    pub output_pin_name: String,
    pub output_pin_index: usize,
    pub input_pin_names: Vec<String>,
    /// For each cell input pin, the cut-leaf variable connected to it.
    pub input_to_leaf: Vec<usize>,
    /// For each cell input pin, whether the selected cut leaf is complemented.
    pub input_negated: Vec<bool>,
    /// Conservative scalar delay estimate for each cell input pin.
    pub input_delays: Vec<Option<f64>>,
    /// Rise/fall sink capacitance for each cell input pin.
    pub input_capacitances: Vec<CombinationalOutputLoad>,
    /// Whether gv-stats-style rise/fall timing can evaluate this binding.
    pub timing_complete: bool,
    pub area: f64,
}

impl CellBinding {
    /// Returns whether gv-stats can evaluate this output's timing arcs.
    pub fn has_complete_timing(&self) -> bool {
        self.timing_complete
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

    /// Returns the indexed Liberty output pin for timing evaluation.
    pub fn output_pin<'a>(&self, library: &'a Library) -> &'a Pin {
        &library.cells[self.cell_index].pins[self.output_pin_index]
    }

    /// Returns the largest scalar fallback delay across this cell's inputs.
    pub fn worst_nominal_delay(&self) -> f64 {
        self.input_delays
            .iter()
            .copied()
            .flatten()
            .reduce(f64::max)
            .unwrap_or(0.0)
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
    #[cfg(test)]
    pub fn build(library: &Library, max_arity: usize) -> Result<Self> {
        Self::build_with_root_limit(library, max_arity, None)
    }

    /// Builds the compact root-cell library used by NF-style mapping. ABC NF
    /// keeps one area-root per native function. Drive-strength selection is a
    /// separate sizing problem rather than part of structural cut mapping.
    pub fn build_nf(library: &Library, max_arity: usize) -> Result<Self> {
        Self::build_with_root_limit(library, max_arity, Some(NF_ROOT_VARIANTS_PER_FUNCTION))
    }

    fn build_with_root_limit(
        library: &Library,
        max_arity: usize,
        root_limit: Option<usize>,
    ) -> Result<Self> {
        let mut by_truth: BTreeMap<(usize, u64), Vec<CellBinding>> = BTreeMap::new();
        let mut stats = LibertyIndexStats::default();
        let mut indexed_cells = Vec::new();
        let mut nf_roots: BTreeMap<(usize, u64), Vec<Vec<(u64, CellBinding)>>> = BTreeMap::new();
        for (cell_index, cell) in library.cells.iter().enumerate() {
            let Some(indexed) = index_cell(library, cell_index, cell, max_arity)? else {
                stats.skipped_cells += 1;
                continue;
            };
            if let Some(root_limit) = root_limit {
                // index_cell emits identity/no-input-negation first, so its
                // first truth is the cell's native declared-pin function.
                let native_key = (indexed[0].1.input_pin_names.len(), indexed[0].0);
                let roots = nf_roots.entry(native_key).or_default();
                if roots
                    .iter()
                    .any(|existing| root_binding_dominates(&existing[0].1, &indexed[0].1))
                {
                    continue;
                }
                roots.retain(|existing| !root_binding_dominates(&indexed[0].1, &existing[0].1));
                roots.push(indexed);
                roots.sort_by(|lhs, rhs| root_binding_order(&lhs[0].1, &rhs[0].1));
                roots.truncate(root_limit);
            } else {
                indexed_cells.push(indexed);
            }
        }
        if root_limit.is_some() {
            indexed_cells.extend(
                nf_roots
                    .into_values()
                    .flatten()
                    .map(deduplicate_nf_configurations),
            );
        }
        for indexed in indexed_cells {
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
    cell_index: usize,
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

    let input_pin_indices: Vec<usize> = cell
        .pins
        .iter()
        .enumerate()
        .filter(|(_, pin)| pin.direction == PinDirection::Input as i32)
        .map(|(pin_index, _)| pin_index)
        .collect();
    let input_pins: Vec<&Pin> = input_pin_indices
        .iter()
        .map(|pin_index| &cell.pins[*pin_index])
        .collect();
    if input_pins.iter().any(|pin| pin.is_clocking_pin) || input_pins.len() > max_arity {
        return Ok(None);
    }
    let output_pin_indices: Vec<usize> = cell
        .pins
        .iter()
        .enumerate()
        .filter(|(_, pin)| pin.direction == PinDirection::Output as i32)
        .map(|(pin_index, _)| pin_index)
        .collect();
    if output_pin_indices.len() != 1 {
        return Ok(None);
    }
    let output_pin_index = output_pin_indices[0];
    let output_pin = &cell.pins[output_pin_index];
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
    let input_capacitances: Vec<CombinationalOutputLoad> = input_pin_indices
        .iter()
        .map(|pin_index| {
            effective_input_capacitance_for_mapping(
                &cell.pins[*pin_index],
                format!(
                    "technology-map load pin '{}.{}'",
                    cell.name,
                    library.resolve_string(&cell.pins[*pin_index].name)
                )
                .as_str(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let timing_complete = validate_output_pin_for_basic_sta(
        library,
        cell.name.as_str(),
        output_pin,
        input_pin_names.as_slice(),
    )
    .is_ok();
    let mut indexed = Vec::new();
    for input_to_leaf in permutations(input_pin_names.len()) {
        for input_negated in polarity_vectors(input_pin_names.len()) {
            indexed.push((
                transform_truth(truth, input_to_leaf.as_slice(), input_negated.as_slice()),
                CellBinding {
                    cell_name: cell.name.clone(),
                    cell_index,
                    output_pin_name: output_pin_name.clone(),
                    output_pin_index,
                    input_pin_names: input_pin_names.clone(),
                    input_to_leaf: input_to_leaf.clone(),
                    input_negated,
                    input_delays: input_delays.clone(),
                    input_capacitances: input_capacitances.clone(),
                    timing_complete,
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
    if size == 0 {
        return vec![Vec::new()];
    }
    let mut values: Vec<usize> = (0..size).collect();
    let mut result = Vec::new();
    let schedule = abc_permutation_schedule(size);
    for swap_index in schedule {
        result.push(values.clone());
        if size > 1 {
            values.swap(swap_index, swap_index + 1);
        }
    }
    result
}

fn polarity_vectors(size: usize) -> Vec<Vec<bool>> {
    if size == 0 {
        return vec![Vec::new()];
    }
    let mut values = vec![false; size];
    let mut result = Vec::new();
    for flip_index in abc_gray_code_schedule(size) {
        result.push(values.clone());
        values[flip_index] = !values[flip_index];
    }
    result
}

/// Returns ABC's adjacent-swap schedule for visiting every pin permutation.
fn abc_permutation_schedule(size: usize) -> Vec<usize> {
    if size == 1 {
        return vec![0];
    }
    if size == 2 {
        return vec![0, 0];
    }
    let prior = abc_permutation_schedule(size - 1);
    let group_count = factorial(size) / size / 2;
    let mut schedule = Vec::with_capacity(factorial(size));
    for group in 0..group_count {
        for index in (1..size).rev() {
            schedule.push(index - 1);
        }
        schedule.push(prior[2 * group] + 1);
        for index in 0..(size - 1) {
            schedule.push(index);
        }
        schedule.push(prior[2 * group + 1]);
    }
    schedule
}

/// Returns ABC's bit-flip schedule for visiting every polarity vector.
fn abc_gray_code_schedule(size: usize) -> Vec<usize> {
    let mut schedule = Vec::with_capacity(1usize << size);
    for bit in 0..size {
        schedule.push(bit);
        for index in 1..(1usize << bit) {
            schedule.push(schedule[index - 1]);
        }
    }
    schedule.push(size - 1);
    schedule
}

fn factorial(value: usize) -> usize {
    (1..=value).product()
}

/// Matches ABC NF's default `fPinPerm=0` behavior: for one root cell and one
/// transformed truth, keep the first configuration for each leaf-polarity
/// mask instead of retaining equivalent pin permutations.
fn deduplicate_nf_configurations(indexed: Vec<(u64, CellBinding)>) -> Vec<(u64, CellBinding)> {
    let mut seen = BTreeSet::new();
    let mut deduplicated = Vec::new();
    for (truth, binding) in indexed {
        let mut leaf_negated = vec![false; binding.input_to_leaf.len()];
        for (input_index, leaf_index) in binding.input_to_leaf.iter().copied().enumerate() {
            leaf_negated[leaf_index] = binding.input_negated[input_index];
        }
        if seen.insert((truth, leaf_negated)) {
            deduplicated.push((truth, binding));
        }
    }
    deduplicated
}

fn binding_order(lhs: &CellBinding, rhs: &CellBinding) -> std::cmp::Ordering {
    lhs.area
        .total_cmp(&rhs.area)
        .then_with(|| lhs.stable_key().cmp(&rhs.stable_key()))
}

fn root_binding_dominates(lhs: &CellBinding, rhs: &CellBinding) -> bool {
    lhs.area <= rhs.area && (lhs.area < rhs.area || lhs.stable_key() <= rhs.stable_key())
}

fn root_binding_order(lhs: &CellBinding, rhs: &CellBinding) -> std::cmp::Ordering {
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

    #[test]
    fn permutation_generator_is_complete() {
        let generated = permutations(4);
        let unique = generated.iter().cloned().collect::<BTreeSet<_>>();

        assert_eq!(generated.len(), 24);
        assert_eq!(unique.len(), 24);
    }

    #[test]
    fn polarity_generator_is_complete() {
        let generated = polarity_vectors(4);
        let unique = generated.iter().cloned().collect::<BTreeSet<_>>();

        assert_eq!(generated.len(), 16);
        assert_eq!(unique.len(), 16);
    }
}
