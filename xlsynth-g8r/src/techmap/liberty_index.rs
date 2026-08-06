// SPDX-License-Identifier: Apache-2.0

//! Clean-sheet Liberty function index for combinational cut matching.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, Pin, PinDirection};
use crate::liberty_proto::TimingTableKind;
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, TimingQueryDiagnosticCounts,
    effective_input_capacitance_for_mapping, evaluate_combinational_cell_output_timing,
    validate_output_pin_for_basic_sta,
};
use crate::techmap::truth::{MAX_TRUTH_TABLE_INPUTS, transform_truth, variable_truth};
use anyhow::{Result, anyhow};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const NF_ROOT_VARIANTS_PER_FUNCTION: usize = 2;
// Sparse libraries rarely expose enough drive diversity to repay extra roots.
const MINIMUM_CELLS_FOR_NF_PARETO_ROOTS: usize = 512;
const REPRESENTATIVE_OUTPUT_FANOUT: f64 = 2.0;
const PARETO_REPRESENTATIVE_OUTPUT_FANOUT: f64 = 2.5;

/// Selects how equally priced Liberty roots are ordered for one NF index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NfRootTiePolicy {
    Fastest,
    StableIdentity,
}

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
    /// Boolean-interchangeable input-pin masks in native Liberty pin order.
    pub symmetric_input_masks: [u8; MAX_TRUTH_TABLE_INPUTS],
    /// Whether gv-stats-style rise/fall timing can evaluate this binding.
    pub timing_complete: bool,
    pub area: f64,
}

/// Stable arena handle for one concrete Liberty binding.
///
/// The index owns binding payloads once and mapping stores this compact handle
/// while exploring candidates. This avoids repeatedly cloning pin-name and
/// timing vectors in the mapper's hot path.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct CellBindingId(usize);

/// Fixed, gv-stats-interpolated input-pin delays for the NF root library.
pub(super) struct RepresentativePinDelayTable {
    output_load: f64,
    pin_delays: Vec<SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>>,
}

impl RepresentativePinDelayTable {
    /// Returns the representative fanout load shared by the indexed cells.
    pub(super) fn output_load(&self) -> f64 {
        self.output_load
    }

    /// Returns the characterized delay for one concrete cell input pin.
    pub(super) fn pin_delay(&self, binding: CellBindingId, input_index: usize) -> f64 {
        self.pin_delays[binding.0][input_index]
    }
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
    bindings: Vec<CellBinding>,
    by_truth: BTreeMap<(usize, u64), Vec<CellBindingId>>,
    /// Tracks whether this index retained bounded area/delay root variants.
    has_pareto_roots: bool,
    /// Dense rank of each stable key, used by mapper tie-breaks without
    /// repeatedly comparing cell/pin-name strings in the inner loop.
    stable_key_ranks: Vec<usize>,
    pub stats: LibertyIndexStats,
}

impl LibertyCellIndex {
    /// Builds a function index without relying on standard-cell family names.
    #[cfg(test)]
    pub fn build(library: &Library, max_arity: usize) -> Result<Self> {
        Self::build_with_root_limit(library, max_arity, None, NfRootTiePolicy::Fastest)
    }

    /// Builds a bounded area/delay Pareto root library for NF-style mapping.
    pub fn build_nf(library: &Library, max_arity: usize) -> Result<Self> {
        let eligible_combinational_cells = library
            .cells
            .iter()
            .filter(|cell| {
                cell.dont_use != Some(true)
                    && cell.sequential.is_empty()
                    && cell.clock_gate.is_none()
                    && cell
                        .pins
                        .iter()
                        .filter(|pin| pin.direction == PinDirection::Output as i32)
                        .count()
                        == 1
                    && cell
                        .pins
                        .iter()
                        .filter(|pin| pin.direction == PinDirection::Input as i32)
                        .count()
                        <= max_arity
            })
            .count();
        let root_limit = if eligible_combinational_cells >= MINIMUM_CELLS_FOR_NF_PARETO_ROOTS {
            NF_ROOT_VARIANTS_PER_FUNCTION
        } else {
            1
        };
        Self::build_with_root_limit(
            library,
            max_arity,
            Some(root_limit),
            NfRootTiePolicy::Fastest,
        )
    }

    /// Builds an alternate deterministic root library for oversized NF covers.
    pub(super) fn build_nf_stable_roots(library: &Library, max_arity: usize) -> Result<Self> {
        Self::build_with_root_limit(
            library,
            max_arity,
            Some(NF_ROOT_VARIANTS_PER_FUNCTION),
            NfRootTiePolicy::StableIdentity,
        )
    }

    fn build_with_root_limit(
        library: &Library,
        max_arity: usize,
        root_limit: Option<usize>,
        root_tie_policy: NfRootTiePolicy,
    ) -> Result<Self> {
        let mut bindings_by_truth: BTreeMap<(usize, u64), Vec<CellBinding>> = BTreeMap::new();
        let mut stats = LibertyIndexStats::default();
        let mut indexed_cells = Vec::new();
        let mut nf_roots: BTreeMap<(usize, u64), Vec<(u64, CellBinding)>> = BTreeMap::new();
        for (cell_index, cell) in library.cells.iter().enumerate() {
            let Some((native_truth, native_binding)) =
                index_native_cell(library, cell_index, cell, max_arity)?
            else {
                stats.skipped_cells += 1;
                continue;
            };
            if let Some(root_limit) = root_limit {
                let native_key = (native_binding.input_pin_names.len(), native_truth);
                let roots = nf_roots.entry(native_key).or_default();
                if roots.iter().any(|(_, existing)| {
                    root_binding_dominates(existing, &native_binding, root_tie_policy)
                }) {
                    continue;
                }
                roots.retain(|(_, existing)| {
                    !root_binding_dominates(&native_binding, existing, root_tie_policy)
                });
                roots.push((native_truth, native_binding));
                roots.sort_by(|lhs, rhs| root_binding_order(&lhs.1, &rhs.1, root_tie_policy));
                roots.truncate(root_limit);
            } else {
                indexed_cells.push(expand_native_cell(native_truth, &native_binding));
            }
        }
        if root_limit.is_some() {
            indexed_cells.extend(
                nf_roots
                    .into_values()
                    .flatten()
                    .map(|(truth, binding)| expand_native_cell(truth, &binding))
                    .map(deduplicate_nf_configurations),
            );
        }
        for indexed in indexed_cells {
            stats.indexed_cell_outputs += 1;
            for (truth, binding) in indexed {
                bindings_by_truth
                    .entry((binding.input_pin_names.len(), truth))
                    .or_default()
                    .push(binding);
                stats.indexed_bindings += 1;
            }
        }
        for bindings in bindings_by_truth.values_mut() {
            bindings.sort_by(binding_order);
        }
        if bindings_by_truth.is_empty() {
            return Err(anyhow!(
                "Liberty library has no eligible single-output combinational cells with parseable functions"
            ));
        }
        let mut bindings = Vec::with_capacity(stats.indexed_bindings);
        let mut by_truth = BTreeMap::new();
        for (key, key_bindings) in bindings_by_truth {
            let mut ids = Vec::with_capacity(key_bindings.len());
            for binding in key_bindings {
                let id = CellBindingId(bindings.len());
                bindings.push(binding);
                ids.push(id);
            }
            by_truth.insert(key, ids);
        }
        let stable_key_ranks = build_stable_key_ranks(bindings.as_slice());
        Ok(Self {
            bindings,
            by_truth,
            has_pareto_roots: root_limit.is_some_and(|limit| limit > 1)
                && root_tie_policy == NfRootTiePolicy::Fastest,
            stable_key_ranks,
            stats,
        })
    }

    /// Returns every deterministic binding handle for one cut truth table.
    pub fn matches(&self, arity: usize, truth: u64) -> &[CellBindingId] {
        self.by_truth
            .get(&(arity, truth))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Resolves one compact binding handle to its immutable payload.
    pub fn binding(&self, id: CellBindingId) -> &CellBinding {
        &self.bindings[id.0]
    }

    /// Samples each native cell once at a representative slew and fanout load.
    pub(super) fn representative_pin_delays(
        &self,
        library: &Library,
        input_transition: f64,
    ) -> Result<RepresentativePinDelayTable> {
        if !input_transition.is_finite() || input_transition < 0.0 {
            return Err(anyhow!(
                "representative input transition must be non-negative and finite; got {}",
                input_transition
            ));
        }

        let mut visited_cells = vec![false; library.cells.len()];
        let mut input_loads = Vec::new();
        for binding in &self.bindings {
            if std::mem::replace(&mut visited_cells[binding.cell_index], true) {
                continue;
            }
            for capacitance in &binding.input_capacitances {
                let load = capacitance.rise.max(capacitance.fall);
                if load.is_finite() && load > 0.0 {
                    input_loads.push(load);
                }
            }
        }
        input_loads.sort_by(f64::total_cmp);
        let median_input_load = match input_loads.len() {
            0 => 0.0,
            count if count % 2 == 1 => input_loads[count / 2],
            count => (input_loads[count / 2 - 1] + input_loads[count / 2]) / 2.0,
        };
        let output_fanout = if self.has_pareto_roots {
            PARETO_REPRESENTATIVE_OUTPUT_FANOUT
        } else {
            REPRESENTATIVE_OUTPUT_FANOUT
        };
        let output_load = output_fanout * median_input_load;
        let representative_load = CombinationalOutputLoad {
            rise: output_load,
            fall: output_load,
        };
        let input_timing = SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: input_transition,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: input_transition,
            },
        };
        let known_pin_values = HashMap::new();
        let mut diagnostics = TimingQueryDiagnosticCounts::default();
        let mut native_pin_delays: Vec<Option<SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>>> =
            vec![None; library.cells.len()];
        let mut pin_delays = Vec::with_capacity(self.bindings.len());

        for binding in &self.bindings {
            if native_pin_delays[binding.cell_index].is_none() {
                let mut delays = SmallVec::with_capacity(binding.input_pin_names.len());
                for (input_index, input_name) in binding.input_pin_names.iter().enumerate() {
                    let delay = if binding.has_complete_timing() {
                        let output_timing = evaluate_combinational_cell_output_timing(
                            library,
                            binding.cell_name.as_str(),
                            binding.output_pin(library),
                            &[(input_name.as_str(), input_timing)],
                            representative_load,
                            &known_pin_values,
                            &mut diagnostics,
                        )?;
                        output_timing.rise.arrival.max(output_timing.fall.arrival)
                    } else {
                        binding.input_delays[input_index]
                            .filter(|delay| delay.is_finite() && *delay >= 0.0)
                            .unwrap_or(1.0)
                    };
                    if !delay.is_finite() || delay < 0.0 {
                        return Err(anyhow!(
                            "cell '{}', input '{}', has invalid representative pin delay {}",
                            binding.cell_name,
                            input_name,
                            delay
                        ));
                    }
                    delays.push(delay);
                }
                native_pin_delays[binding.cell_index] = Some(delays);
            }
            pin_delays.push(
                native_pin_delays[binding.cell_index]
                    .as_ref()
                    .expect("representative pin delays were just initialized")
                    .clone(),
            );
        }

        Ok(RepresentativePinDelayTable {
            output_load,
            pin_delays,
        })
    }

    /// Compares deterministic binding identities using precomputed ranks.
    ///
    /// Equal stable keys intentionally share one rank so this is exactly the
    /// same ordering relation as comparing CellBinding::stable_key directly.
    pub fn stable_key_order(&self, lhs: CellBindingId, rhs: CellBindingId) -> std::cmp::Ordering {
        self.stable_key_ranks[lhs.0].cmp(&self.stable_key_ranks[rhs.0])
    }

    /// Returns whether every eligible binding supports full gv-stats timing.
    ///
    /// When true, any cover emitted from this index can rely on the final
    /// parsed-netlist STA pass instead of first retiming the selected cover.
    pub fn all_bindings_have_complete_timing(&self) -> bool {
        self.bindings.iter().all(CellBinding::has_complete_timing)
    }

    /// Returns the cheapest unary identity cell, if the library has one.
    pub fn best_buffer(&self) -> Option<&CellBinding> {
        self.matches(1, variable_truth(1, 0))
            .iter()
            .copied()
            .find(|id| !self.binding(*id).input_negated[0])
            .map(|id| self.binding(id))
    }

    /// Picks the fastest legal output buffer at the actual external load.
    pub(super) fn best_output_buffer(
        &self,
        library: &Library,
        input_transition: f64,
        output_load: f64,
    ) -> Result<Option<CellBinding>> {
        let fallback = self.best_buffer().cloned();
        if output_load <= 0.0 || fallback.is_none() {
            return Ok(fallback);
        }

        let input_timing = SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: input_transition,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: input_transition,
            },
        };
        let load = CombinationalOutputLoad {
            rise: output_load,
            fall: output_load,
        };
        let known_pin_values = HashMap::new();
        let mut diagnostics = TimingQueryDiagnosticCounts::default();
        let mut winner: Option<(f64, CellBinding)> = None;

        for (cell_index, cell) in library.cells.iter().enumerate() {
            let Some((truth, binding)) = index_native_cell(library, cell_index, cell, 1)? else {
                continue;
            };
            if truth != variable_truth(1, 0) || !binding.has_complete_timing() {
                continue;
            }
            if binding
                .output_pin(library)
                .max_capacitance
                .is_some_and(|maximum| maximum.is_finite() && output_load > maximum)
            {
                continue;
            }

            let timing = evaluate_combinational_cell_output_timing(
                library,
                binding.cell_name.as_str(),
                binding.output_pin(library),
                &[(binding.input_pin_names[0].as_str(), input_timing)],
                load,
                &known_pin_values,
                &mut diagnostics,
            )?;
            let delay = timing.rise.arrival.max(timing.fall.arrival);
            if winner
                .as_ref()
                .is_none_or(|(winner_delay, winner_binding)| {
                    delay
                        .total_cmp(winner_delay)
                        .then_with(|| binding.area.total_cmp(&winner_binding.area))
                        .then_with(|| binding.stable_key().cmp(&winner_binding.stable_key()))
                        .is_lt()
                })
            {
                winner = Some((delay, binding));
            }
        }

        Ok(winner.map(|(_, binding)| binding).or(fallback))
    }

    /// Returns the cheapest unary inverter cell, if the library has one.
    pub fn best_inverter(&self) -> Option<&CellBinding> {
        self.matches(1, 0b01)
            .iter()
            .copied()
            .find(|id| !self.binding(*id).input_negated[0])
            .map(|id| self.binding(id))
    }
}

/// Builds one dense rank per distinct stable binding identity.
fn build_stable_key_ranks(bindings: &[CellBinding]) -> Vec<usize> {
    let mut ordered_ids: Vec<usize> = (0..bindings.len()).collect();
    ordered_ids.sort_by(|lhs, rhs| {
        bindings[*lhs]
            .stable_key()
            .cmp(&bindings[*rhs].stable_key())
    });
    let mut ranks = vec![0; bindings.len()];
    let mut rank = 0usize;
    for (position, binding_id) in ordered_ids.iter().copied().enumerate() {
        if position > 0
            && bindings[ordered_ids[position - 1]].stable_key() != bindings[binding_id].stable_key()
        {
            rank += 1;
        }
        ranks[binding_id] = rank;
    }
    ranks
}

/// Extracts a cell once before deciding whether its root is worth expanding.
fn index_native_cell(
    library: &Library,
    cell_index: usize,
    cell: &Cell,
    max_arity: usize,
) -> Result<Option<(u64, CellBinding)>> {
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
    let input_count = input_pin_names.len();
    Ok(Some((
        truth,
        CellBinding {
            cell_name: cell.name.clone(),
            cell_index,
            output_pin_name,
            output_pin_index,
            input_pin_names,
            input_to_leaf: (0..input_count).collect(),
            input_negated: vec![false; input_count],
            input_delays,
            input_capacitances,
            symmetric_input_masks: symmetric_input_masks(input_count, truth),
            timing_complete,
            area: cell.area,
        },
    )))
}

/// Groups exactly those Liberty inputs that can exchange signals safely.
fn symmetric_input_masks(input_count: usize, truth: u64) -> [u8; MAX_TRUTH_TABLE_INPUTS] {
    let mut masks = [0_u8; MAX_TRUTH_TABLE_INPUTS];
    for input in 0..input_count {
        masks[input] = 1_u8 << input;
    }

    for first_input in 0..input_count {
        for second_input in (first_input + 1)..input_count {
            let swap_mask = (1_usize << first_input) | (1_usize << second_input);
            let interchangeable = (0..(1_usize << input_count)).all(|assignment| {
                let first = (assignment >> first_input) & 1;
                let second = (assignment >> second_input) & 1;
                let swapped = if first == second {
                    assignment
                } else {
                    assignment ^ swap_mask
                };
                ((truth >> assignment) & 1) == ((truth >> swapped) & 1)
            });
            if interchangeable {
                let merged = masks[first_input] | masks[second_input];
                for mask in masks.iter_mut().take(input_count) {
                    if *mask & merged != 0 {
                        *mask = merged;
                    }
                }
            }
        }
    }
    masks
}

/// Expands only a retained native root into deterministic Boolean bindings.
fn expand_native_cell(truth: u64, native: &CellBinding) -> Vec<(u64, CellBinding)> {
    let mut indexed = Vec::new();
    for input_to_leaf in permutations(native.input_pin_names.len()) {
        for input_negated in polarity_vectors(native.input_pin_names.len()) {
            indexed.push((
                transform_truth(truth, input_to_leaf.as_slice(), input_negated.as_slice()),
                CellBinding {
                    input_to_leaf: input_to_leaf.clone(),
                    input_negated,
                    ..native.clone()
                },
            ));
        }
    }
    indexed
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

fn root_binding_dominates(lhs: &CellBinding, rhs: &CellBinding, policy: NfRootTiePolicy) -> bool {
    if policy == NfRootTiePolicy::StableIdentity {
        return root_binding_order(lhs, rhs, policy).is_le();
    }

    let area_order = lhs.area.total_cmp(&rhs.area);
    let delay_order = lhs
        .worst_nominal_delay()
        .total_cmp(&rhs.worst_nominal_delay());
    if area_order.is_gt() || delay_order.is_gt() {
        return false;
    }
    area_order.is_lt() || delay_order.is_lt() || lhs.stable_key().cmp(&rhs.stable_key()).is_le()
}

fn root_binding_order(
    lhs: &CellBinding,
    rhs: &CellBinding,
    policy: NfRootTiePolicy,
) -> std::cmp::Ordering {
    lhs.area
        .total_cmp(&rhs.area)
        .then_with(|| {
            if policy == NfRootTiePolicy::Fastest {
                lhs.worst_nominal_delay()
                    .total_cmp(&rhs.worst_nominal_delay())
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .then_with(|| lhs.stable_key().cmp(&rhs.stable_key()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_model::{Cell, LibraryBuilder, LuTableTemplate, Pin, TimingArc};
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::cell_catalog::test_utils::sizing_library;

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

    /// Builds a complete two-dimensional Liberty arc for pin-delay sampling.
    fn representative_timing_arc(
        builder: &mut LibraryBuilder,
        related_pin: &str,
        rise_delays: [f64; 4],
        fall_delays: [f64; 4],
    ) -> TimingArc {
        let tables = [
            (TimingTableKind::CellRise, rise_delays),
            (TimingTableKind::CellFall, fall_delays),
            (TimingTableKind::RiseTransition, [0.1; 4]),
            (TimingTableKind::FallTransition, [0.1; 4]),
        ]
        .into_iter()
        .map(|(kind, values)| {
            builder
                .add_timing_table_f64(
                    kind,
                    1,
                    vec![],
                    vec![],
                    vec![],
                    values.to_vec(),
                    vec![2, 2],
                    "",
                )
                .expect("representative timing table should be valid")
        })
        .collect();
        builder
            .add_timing_arc(related_pin, "positive_unate", "combinational", "", tables)
            .expect("representative timing arc should be valid")
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
        assert_eq!(
            index.binding(index.matches(2, 0b1000)[0]).cell_name,
            "mystery_gate"
        );
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
    fn loaded_output_buffers_use_exact_timing_across_all_drive_variants() {
        let library = sizing_library();
        let index = LibertyCellIndex::build_nf(&library, 6).unwrap();

        assert_eq!(index.best_buffer().unwrap().cell_name, "BUF");
        assert_eq!(
            index
                .best_output_buffer(&library, 0.01, 0.5)
                .unwrap()
                .unwrap()
                .cell_name,
            "BUF_FAST"
        );
        assert_eq!(
            index
                .best_output_buffer(&library, 0.01, 0.0)
                .unwrap()
                .unwrap()
                .cell_name,
            "BUF"
        );
    }

    #[test]
    fn symmetric_input_masks_preserve_asymmetric_complex_cell_inputs() {
        assert_eq!(symmetric_input_masks(2, 0b1000)[..2], [0b11, 0b11]);

        let mut truth = 0_u64;
        for assignment in 0..8 {
            let first = (assignment & 1) != 0;
            let second = (assignment & 2) != 0;
            let independent = (assignment & 4) != 0;
            if (first && second) || independent {
                truth |= 1_u64 << assignment;
            }
        }
        assert_eq!(symmetric_input_masks(3, truth)[..3], [0b011, 0b011, 0b100]);
    }

    #[test]
    fn nf_root_area_ties_prefer_the_faster_characterized_liberty_cell() {
        let mut builder = LibraryBuilder::new();
        let mut cells = Vec::new();
        for (name, delay) in [("A_SLOW", 9.0), ("Z_FAST", 1.0)] {
            let mut output = pin(&mut builder, PinDirection::Output, "Y", "A * B");
            for input in ["A", "B"] {
                let table = builder
                    .add_timing_table_f64(
                        TimingTableKind::CellRise,
                        0,
                        vec![],
                        vec![],
                        vec![],
                        vec![delay],
                        vec![],
                        "",
                    )
                    .unwrap();
                output.timing_arcs.push(
                    builder
                        .add_timing_arc(input, "", "combinational", "", vec![table])
                        .unwrap(),
                );
            }
            cells.push(Cell {
                name: name.to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    output,
                ],
                area: 1.0,
                ..Default::default()
            });
        }
        builder.cells = cells;
        let library = builder.finish();

        let index = LibertyCellIndex::build_nf(&library, 6).unwrap();

        assert_eq!(index.stats.indexed_cell_outputs, 1);
        assert_eq!(
            index.binding(index.matches(2, 0b1000)[0]).cell_name,
            "Z_FAST"
        );

        let stable = LibertyCellIndex::build_nf_stable_roots(&library, 6).unwrap();

        assert_eq!(stable.stats.indexed_cell_outputs, 1);
        assert_eq!(
            stable.binding(stable.matches(2, 0b1000)[0]).cell_name,
            "A_SLOW"
        );
    }

    #[test]
    fn nf_roots_preserve_bounded_nondominated_area_delay_alternatives() {
        let mut builder = LibraryBuilder::new();
        let mut cells = Vec::new();
        for (name, area, delay) in [
            ("SMALL_SLOW", 1.0, 9.0),
            ("DOMINATED", 2.0, 10.0),
            ("LARGE_FAST", 3.0, 1.0),
        ] {
            let mut output = pin(&mut builder, PinDirection::Output, "Y", "A * B");
            for input in ["A", "B"] {
                let table = builder
                    .add_timing_table_f64(
                        TimingTableKind::CellRise,
                        0,
                        vec![],
                        vec![],
                        vec![],
                        vec![delay],
                        vec![],
                        "",
                    )
                    .unwrap();
                output.timing_arcs.push(
                    builder
                        .add_timing_arc(input, "", "combinational", "", vec![table])
                        .unwrap(),
                );
            }
            cells.push(Cell {
                name: name.to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    output,
                ],
                area,
                ..Default::default()
            });
        }
        builder.cells = cells;
        let library = builder.finish();

        let index =
            LibertyCellIndex::build_with_root_limit(&library, 6, Some(2), NfRootTiePolicy::Fastest)
                .expect("bounded Pareto roots should be indexed");
        let root_names = index
            .matches(2, 0b1000)
            .iter()
            .map(|binding| index.binding(*binding).cell_name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(index.stats.indexed_cell_outputs, 2);
        assert_eq!(root_names, vec!["SMALL_SLOW", "LARGE_FAST"]);
    }

    #[test]
    fn nf_pareto_roots_are_reserved_for_large_combinational_libraries() {
        let mut library = sizing_library();
        let compact = LibertyCellIndex::build_nf(&library, 6)
            .expect("compact libraries should preserve a single area root");

        assert_eq!(compact.matches(2, 0b1000).len(), 1);
        assert_eq!(
            compact.binding(compact.matches(2, 0b1000)[0]).cell_name,
            "AND2"
        );

        let filler = library.cells[0].clone();
        for index in library.cells.len()..MINIMUM_CELLS_FOR_NF_PARETO_ROOTS {
            let mut cell = filler.clone();
            cell.name = format!("ADDITIONAL_BUFFER_{index}");
            library.cells.push(cell);
        }
        let rich = LibertyCellIndex::build_nf(&library, 6)
            .expect("large libraries should preserve fast Pareto alternatives");
        let root_names = rich
            .matches(2, 0b1000)
            .iter()
            .map(|binding| rich.binding(*binding).cell_name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(root_names, vec!["AND2", "AND2_FAST"]);
    }

    #[test]
    fn representative_pin_delays_interpolate_each_input_at_median_fanout() {
        let mut builder = LibraryBuilder::new();
        builder.lu_table_templates = vec![LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "representative_2d".to_string(),
            variable_1: "input_net_transition".to_string().into(),
            variable_2: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.1, 0.3],
            index_2: vec![1.0, 3.0],
            ..Default::default()
        }];
        let mut input_a = pin(&mut builder, PinDirection::Input, "A", "");
        input_a.capacitance = Some(0.5);
        let mut input_b = pin(&mut builder, PinDirection::Input, "B", "");
        input_b.capacitance = Some(1.5);
        let mut output = pin(&mut builder, PinDirection::Output, "Y", "A * B");
        output.timing_arcs = vec![
            representative_timing_arc(
                &mut builder,
                "A",
                [10.0, 20.0, 30.0, 40.0],
                [5.0, 7.0, 9.0, 11.0],
            ),
            representative_timing_arc(
                &mut builder,
                "B",
                [2.0, 4.0, 6.0, 8.0],
                [12.0, 14.0, 16.0, 18.0],
            ),
        ];
        builder.cells = vec![Cell {
            name: "AND2".to_string(),
            pins: vec![input_a, input_b, output],
            area: 1.0,
            ..Default::default()
        }];
        let library = builder.finish();
        let index = LibertyCellIndex::build_nf(&library, 6)
            .expect("the characterized AND2 should be indexed");

        let delays = index
            .representative_pin_delays(&library, 0.2)
            .expect("representative Liberty table interpolation should succeed");
        let binding = index.matches(2, 0b1000)[0];

        assert!((delays.output_load() - 2.0).abs() < 1e-12);
        assert!((delays.pin_delay(binding, 0) - 25.0).abs() < 1e-12);
        assert!((delays.pin_delay(binding, 1) - 15.0).abs() < 1e-12);
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
