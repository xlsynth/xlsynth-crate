// SPDX-License-Identifier: Apache-2.0

//! Incremental Liberty-aware sizing for scalar combinational mapped netlists.

use crate::liberty_model::{Library, PinDirection};
use crate::netlist::cell_catalog::{CatalogCell, CellCatalog};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistModule, PortDirection};
use crate::netlist::report::build_area_report;
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, StaOptions, TimingQueryDiagnosticCounts,
    effective_input_capacitance_for_mapping, evaluate_combinational_cell_output_timing,
};
use crate::netlist::utils::scalar_constant_output_assignments;
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Bounded critical-path upsizing and timing-protected area-recovery options.
#[derive(Clone, Debug, PartialEq)]
pub struct ResizeOptions {
    pub sta_options: StaOptions,
    pub max_iterations: usize,
    pub max_area_iterations: usize,
    pub max_candidate_paths: usize,
    pub max_evaluations_per_iteration: usize,
    pub max_cell_candidates_per_instance: usize,
    pub improvement_epsilon: f64,
    pub area_epsilon: f64,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            sta_options: StaOptions::default(),
            max_iterations: 16,
            max_area_iterations: 32,
            max_candidate_paths: 8,
            max_evaluations_per_iteration: 64,
            max_cell_candidates_per_instance: 4,
            improvement_epsilon: 1e-9,
            area_epsilon: 1e-12,
        }
    }
}

/// One accepted exact-function Liberty cell substitution.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResizeStep {
    pub instance: String,
    pub old_cell: String,
    pub new_cell: String,
    pub delay_before: f64,
    pub delay_after: f64,
    pub area_before: f64,
    pub area_after: f64,
}

/// Machine-readable results of buffered-netlist upsizing and downsizing.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResizeStats {
    pub initial_delay: f64,
    pub final_delay: f64,
    pub initial_area: f64,
    pub final_area: f64,
    pub evaluations: usize,
    pub failed_evaluations: usize,
    pub recomputed_instances: usize,
    pub upsizes: usize,
    pub downsizes: usize,
    pub replacements: Vec<ResizeStep>,
}

#[derive(Clone, Debug)]
struct InstanceInput {
    name: String,
    net: Option<NetIndex>,
}

#[derive(Clone, Debug)]
struct InstanceTiming {
    cell_index: usize,
    output_pin_index: usize,
    output_net: NetIndex,
    inputs: Vec<InstanceInput>,
    known_pin_values: HashMap<String, bool>,
}

#[derive(Clone, Debug)]
struct TimingScore {
    endpoint_delays: Vec<f64>,
}

impl TimingScore {
    /// Uses all ordered outputs to avoid moving delay between tied endpoints.
    fn improvement_over(&self, previous: &Self, epsilon: f64) -> Option<f64> {
        for (candidate, current) in self.endpoint_delays.iter().zip(&previous.endpoint_delays) {
            let delta = current - candidate;
            if delta.abs() > epsilon {
                return (delta > 0.0).then_some(delta);
            }
        }
        None
    }

    /// Returns the largest actual combinational output arrival.
    fn worst_delay(&self) -> f64 {
        self.endpoint_delays.first().copied().unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
struct TrialResult {
    score: TimingScore,
    recomputed_instances: usize,
}

#[derive(Clone, Debug)]
struct ResizeMove {
    instance_index: usize,
    cell_index: usize,
    score: TimingScore,
    ranking: f64,
    area: f64,
}

/// Sizes a buffered scalar netlist for delay, then recovers noncritical area.
pub fn resize_netlist(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &ResizeOptions,
) -> Result<ResizeStats> {
    validate_options(options)?;
    let catalog = CellCatalog::new(library)?;
    let initial_area = build_area_report(module, interner, library)?.area;
    let mut timing =
        IncrementalCombinationalSta::new(module, nets, interner, library, options.sta_options)?;
    let mut score = timing.score();
    let mut stats = ResizeStats {
        initial_delay: score.worst_delay(),
        final_delay: score.worst_delay(),
        initial_area,
        final_area: initial_area,
        ..ResizeStats::default()
    };
    let mut area = initial_area;

    for _ in 0..options.max_iterations {
        let mut best: Option<ResizeMove> = None;
        let mut round_evaluations = 0;
        let critical = timing.critical_instances(options.max_candidate_paths);
        'instances: for instance_index in critical {
            let Some(current) = catalog.by_name(timing.cell_name(instance_index)) else {
                continue;
            };
            let mut alternatives = catalog
                .family(current)
                .filter(|candidate| candidate.name != current.name)
                .filter(|candidate| candidate.area + options.area_epsilon >= current.area)
                .take(options.max_cell_candidates_per_instance);
            for candidate in &mut alternatives {
                if round_evaluations == options.max_evaluations_per_iteration {
                    break 'instances;
                }
                round_evaluations += 1;
                stats.evaluations += 1;
                let evaluation = match timing.evaluate_cell_substitution(
                    instance_index,
                    candidate.cell_index,
                    false,
                ) {
                    Ok(evaluation) => evaluation,
                    Err(error) => {
                        stats.failed_evaluations += 1;
                        log::debug!("rejecting resize trial '{}': {error:#}", candidate.name);
                        continue;
                    }
                };
                stats.recomputed_instances += evaluation.recomputed_instances;
                let Some(improvement) = evaluation
                    .score
                    .improvement_over(&score, options.improvement_epsilon)
                else {
                    continue;
                };
                let area_cost = (candidate.area - current.area).max(options.area_epsilon);
                let ranking = improvement / area_cost;
                let trial_area = area - current.area + candidate.area;
                let is_better = best.as_ref().is_none_or(|existing| {
                    ranking > existing.ranking + options.improvement_epsilon
                        || ((ranking - existing.ranking).abs() <= options.improvement_epsilon
                            && (
                                evaluation.score.worst_delay(),
                                instance_index,
                                &candidate.name,
                            ) < (
                                existing.score.worst_delay(),
                                existing.instance_index,
                                &library.cells[existing.cell_index].name,
                            ))
                });
                if is_better {
                    best = Some(ResizeMove {
                        instance_index,
                        cell_index: candidate.cell_index,
                        score: evaluation.score,
                        ranking,
                        area: trial_area,
                    });
                }
            }
        }
        let Some(best) = best else {
            break;
        };
        commit_move(
            module,
            interner,
            library,
            &mut timing,
            &mut stats,
            &mut score,
            &mut area,
            best,
            true,
        )?;
    }

    let delay_limit = score.worst_delay();
    for _ in 0..options.max_area_iterations {
        let candidates = area_recovery_instances(&timing, &catalog, options);
        let mut best: Option<ResizeMove> = None;
        let mut round_evaluations = 0;
        'instances: for instance_index in candidates {
            let Some(current) = catalog.by_name(timing.cell_name(instance_index)) else {
                continue;
            };
            let mut alternatives: Vec<&CatalogCell> = catalog
                .family(current)
                .filter(|candidate| candidate.area + options.area_epsilon < current.area)
                .collect();
            alternatives.sort_by(|lhs, rhs| {
                lhs.area
                    .total_cmp(&rhs.area)
                    .then_with(|| lhs.name.cmp(&rhs.name))
            });
            for candidate in alternatives
                .into_iter()
                .take(options.max_cell_candidates_per_instance)
            {
                if round_evaluations == options.max_evaluations_per_iteration {
                    break 'instances;
                }
                round_evaluations += 1;
                stats.evaluations += 1;
                let evaluation = match timing.evaluate_cell_substitution(
                    instance_index,
                    candidate.cell_index,
                    false,
                ) {
                    Ok(evaluation) => evaluation,
                    Err(error) => {
                        stats.failed_evaluations += 1;
                        log::debug!(
                            "rejecting area-recovery trial '{}': {error:#}",
                            candidate.name
                        );
                        continue;
                    }
                };
                stats.recomputed_instances += evaluation.recomputed_instances;
                if evaluation.score.worst_delay() > delay_limit + options.improvement_epsilon {
                    continue;
                }
                let trial_area = area - current.area + candidate.area;
                let is_better = best.as_ref().is_none_or(|existing| {
                    trial_area + options.area_epsilon < existing.area
                        || ((trial_area - existing.area).abs() <= options.area_epsilon
                            && (
                                evaluation.score.worst_delay(),
                                instance_index,
                                &candidate.name,
                            ) < (
                                existing.score.worst_delay(),
                                existing.instance_index,
                                &library.cells[existing.cell_index].name,
                            ))
                });
                if is_better {
                    best = Some(ResizeMove {
                        instance_index,
                        cell_index: candidate.cell_index,
                        score: evaluation.score,
                        ranking: current.area - candidate.area,
                        area: trial_area,
                    });
                }
            }
        }
        let Some(best) = best else {
            break;
        };
        commit_move(
            module,
            interner,
            library,
            &mut timing,
            &mut stats,
            &mut score,
            &mut area,
            best,
            false,
        )?;
    }

    stats.final_delay = score.worst_delay();
    stats.final_area = build_area_report(module, interner, library)?.area;
    Ok(stats)
}

/// Commits a validated move to both the netlist and incremental timing state.
#[allow(clippy::too_many_arguments)]
fn commit_move(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    timing: &mut IncrementalCombinationalSta<'_>,
    stats: &mut ResizeStats,
    score: &mut TimingScore,
    area: &mut f64,
    selected: ResizeMove,
    upsize: bool,
) -> Result<()> {
    let instance = &mut module.instances[selected.instance_index];
    let instance_name = interner
        .resolve(instance.instance_name)
        .ok_or_else(|| anyhow!("cannot resolve resized instance name"))?
        .to_string();
    let old_cell = timing.cell_name(selected.instance_index).to_string();
    let new_cell = library.cells[selected.cell_index].name.clone();
    let before_delay = score.worst_delay();
    let before_area = *area;
    let committed =
        timing.evaluate_cell_substitution(selected.instance_index, selected.cell_index, true)?;
    stats.recomputed_instances += committed.recomputed_instances;
    instance.type_name = interner.get_or_intern(new_cell.as_str());
    *score = committed.score;
    *area = selected.area;
    stats.replacements.push(ResizeStep {
        instance: instance_name,
        old_cell,
        new_cell,
        delay_before: before_delay,
        delay_after: score.worst_delay(),
        area_before: before_area,
        area_after: *area,
    });
    if upsize {
        stats.upsizes += 1;
    } else {
        stats.downsizes += 1;
    }
    Ok(())
}

/// Visits the largest recoverable cells before less useful area trials.
fn area_recovery_instances(
    timing: &IncrementalCombinationalSta<'_>,
    catalog: &CellCatalog,
    options: &ResizeOptions,
) -> Vec<usize> {
    let mut instances = Vec::new();
    for index in 0..timing.instances.len() {
        let Some(current) = catalog.by_name(timing.cell_name(index)) else {
            continue;
        };
        let saving = catalog
            .family(current)
            .filter(|candidate| candidate.area + options.area_epsilon < current.area)
            .map(|candidate| current.area - candidate.area)
            .fold(0.0_f64, f64::max);
        if saving > options.area_epsilon {
            instances.push((index, saving));
        }
    }
    instances.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
    instances.into_iter().map(|(index, _)| index).collect()
}

/// Rejects budgets and timing assumptions that could produce invalid trials.
fn validate_options(options: &ResizeOptions) -> Result<()> {
    if options.max_candidate_paths == 0
        || options.max_evaluations_per_iteration == 0
        || options.max_cell_candidates_per_instance == 0
    {
        return Err(anyhow!("resizer search bounds must be greater than zero"));
    }
    if !options.improvement_epsilon.is_finite() || options.improvement_epsilon < 0.0 {
        return Err(anyhow!(
            "resizer improvement_epsilon must be finite and nonnegative"
        ));
    }
    if !options.area_epsilon.is_finite() || options.area_epsilon < 0.0 {
        return Err(anyhow!(
            "resizer area_epsilon must be finite and nonnegative"
        ));
    }
    if !options.sta_options.primary_input_transition.is_finite()
        || options.sta_options.primary_input_transition < 0.0
        || !options.sta_options.module_output_load.is_finite()
        || options.sta_options.module_output_load < 0.0
    {
        return Err(anyhow!(
            "resizer STA options must be finite and nonnegative"
        ));
    }
    Ok(())
}

/// Reusable exact-NLDM timing graph for same-pin combinational substitutions.
struct IncrementalCombinationalSta<'a> {
    library: &'a Library,
    instances: Vec<InstanceTiming>,
    drivers: Vec<Option<usize>>,
    loads: Vec<CombinationalOutputLoad>,
    net_timing: Vec<Option<SignalTiming>>,
    successors: Vec<Vec<usize>>,
    topo_positions: Vec<usize>,
    outputs: Vec<NetIndex>,
    diagnostic_counts: TimingQueryDiagnosticCounts,
}

impl<'a> IncrementalCombinationalSta<'a> {
    /// Builds connectivity once and performs the initial full timing pass.
    fn new(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &'a Library,
        options: StaOptions,
    ) -> Result<Self> {
        let constant_outputs = scalar_constant_output_assignments(module, nets)?;
        let by_name: HashMap<&str, usize> = library
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| (cell.name.as_str(), index))
            .collect();
        let mut drivers = vec![None; nets.len()];
        let mut loads = vec![CombinationalOutputLoad::default(); nets.len()];
        let mut net_timing = vec![None; nets.len()];
        let mut outputs = Vec::new();
        let mut instances = Vec::with_capacity(module.instances.len());

        for port in &module.ports {
            let net = module.find_net_index(port.name, nets).ok_or_else(|| {
                anyhow!(
                    "sizing port '{}' has no scalar net",
                    interner.resolve(port.name).unwrap_or("<unknown>")
                )
            })?;
            match port.direction {
                PortDirection::Input => {
                    let timing = EdgeTiming {
                        arrival: 0.0,
                        transition: options.primary_input_transition,
                    };
                    net_timing[net.0] = Some(SignalTiming {
                        rise: timing,
                        fall: timing,
                    });
                }
                PortDirection::Output => {
                    outputs.push(net);
                    loads[net.0].rise += options.module_output_load;
                    loads[net.0].fall += options.module_output_load;
                    if constant_outputs.contains_key(&net.0) {
                        let timing = EdgeTiming {
                            arrival: 0.0,
                            transition: 0.0,
                        };
                        net_timing[net.0] = Some(SignalTiming {
                            rise: timing,
                            fall: timing,
                        });
                    }
                }
                PortDirection::Inout => {
                    return Err(anyhow!(
                        "incremental combinational sizing does not support inout ports"
                    ));
                }
            }
        }

        for (instance_index, instance) in module.instances.iter().enumerate() {
            let name = interner
                .resolve(instance.type_name)
                .ok_or_else(|| anyhow!("cannot resolve sized cell name"))?;
            let cell_index = *by_name
                .get(name)
                .ok_or_else(|| anyhow!("sized netlist references unknown cell '{name}'"))?;
            let cell = &library.cells[cell_index];
            if !cell.sequential.is_empty() || cell.clock_gate.is_some() {
                return Err(anyhow!(
                    "incremental combinational sizing does not yet support sequential cell '{name}'"
                ));
            }

            let mut input_pins = Vec::new();
            let mut known_pin_values = HashMap::new();
            let mut output = None;
            for (pin_id, net_ref) in &instance.connections {
                let pin_name = interner
                    .resolve(*pin_id)
                    .ok_or_else(|| anyhow!("cannot resolve sized cell pin"))?;
                let pin = cell
                    .pins
                    .iter()
                    .find(|pin| library.resolve_string(&pin.name) == pin_name)
                    .ok_or_else(|| anyhow!("cell '{name}' has no pin '{pin_name}'"))?;
                if pin.direction == PinDirection::Input as i32 {
                    let net = match net_ref {
                        NetRef::Simple(net) => {
                            let capacitance = effective_input_capacitance_for_mapping(
                                pin,
                                format!("sizing pin '{name}.{pin_name}'").as_str(),
                            )?;
                            loads[net.0].rise += capacitance.rise;
                            loads[net.0].fall += capacitance.fall;
                            Some(*net)
                        }
                        NetRef::Literal(bits) => {
                            if bits.get_bit_count() != 1 {
                                return Err(anyhow!(
                                    "incremental sizing requires one-bit literal cell inputs"
                                ));
                            }
                            known_pin_values.insert(
                                pin_name.to_string(),
                                bits.get_bit(0).map_err(|error| anyhow!("{error}"))?,
                            );
                            None
                        }
                        NetRef::Unconnected | NetRef::UnknownLiteral(_) => None,
                        _ => {
                            return Err(anyhow!(
                                "incremental sizing requires scalar simple-net connections"
                            ));
                        }
                    };
                    input_pins.push(InstanceInput {
                        name: pin_name.to_string(),
                        net,
                    });
                } else if pin.direction == PinDirection::Output as i32 {
                    let NetRef::Simple(net) = net_ref else {
                        return Err(anyhow!("sized cell outputs must drive a scalar net"));
                    };
                    if output
                        .replace((
                            cell.pins
                                .iter()
                                .position(|candidate| {
                                    library.resolve_string(&candidate.name) == pin_name
                                })
                                .expect("connected output was found above"),
                            *net,
                        ))
                        .is_some()
                    {
                        return Err(anyhow!(
                            "incremental sizing requires single-output combinational cells"
                        ));
                    }
                    if drivers[net.0].replace(instance_index).is_some() {
                        return Err(anyhow!("sized net has multiple cell drivers"));
                    }
                }
            }
            let (output_pin_index, output_net) =
                output.ok_or_else(|| anyhow!("sized cell '{name}' has no output connection"))?;
            input_pins.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
            instances.push(InstanceTiming {
                cell_index,
                output_pin_index,
                output_net,
                inputs: input_pins,
                known_pin_values,
            });
        }

        for net in constant_outputs.keys() {
            if drivers[*net].is_some() {
                return Err(anyhow!(
                    "incremental sizing found a cell driving a constant-tied output"
                ));
            }
        }

        let mut successors = vec![Vec::new(); instances.len()];
        let mut indegrees = vec![0_usize; instances.len()];
        for (instance_index, instance) in instances.iter().enumerate() {
            let mut predecessors = BTreeSet::new();
            for input in &instance.inputs {
                if let Some(net) = input.net
                    && let Some(driver) = drivers[net.0]
                {
                    predecessors.insert(driver);
                }
            }
            indegrees[instance_index] = predecessors.len();
            for predecessor in predecessors {
                successors[predecessor].push(instance_index);
            }
        }
        for next in &mut successors {
            next.sort_unstable();
        }
        let mut ready: BTreeSet<usize> = indegrees
            .iter()
            .enumerate()
            .filter_map(|(index, count)| (*count == 0).then_some(index))
            .collect();
        let mut topo_order = Vec::with_capacity(instances.len());
        while let Some(index) = ready.pop_first() {
            topo_order.push(index);
            for successor in &successors[index] {
                indegrees[*successor] -= 1;
                if indegrees[*successor] == 0 {
                    ready.insert(*successor);
                }
            }
        }
        if topo_order.len() != instances.len() {
            return Err(anyhow!("incremental sizing detected a combinational cycle"));
        }
        let mut topo_positions = vec![0; instances.len()];
        for (position, instance_index) in topo_order.iter().copied().enumerate() {
            topo_positions[instance_index] = position;
        }

        let mut state = Self {
            library,
            instances,
            drivers,
            loads,
            net_timing,
            successors,
            topo_positions,
            outputs,
            diagnostic_counts: TimingQueryDiagnosticCounts::default(),
        };
        for index in topo_order {
            state.recompute_instance(index)?;
        }
        Ok(state)
    }

    /// Resolves the currently committed Liberty cell at one mapped instance.
    fn cell_name(&self, instance_index: usize) -> &str {
        self.library.cells[self.instances[instance_index].cell_index]
            .name
            .as_str()
    }

    /// Returns endpoint delays in stable worst-first order.
    fn score(&self) -> TimingScore {
        let mut endpoint_delays: Vec<f64> = self
            .outputs
            .iter()
            .filter_map(|net| self.net_timing[net.0])
            .map(|timing| timing.rise.arrival.max(timing.fall.arrival))
            .collect();
        endpoint_delays.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
        TimingScore { endpoint_delays }
    }

    /// Traces the highest-arrival output cones in deterministic priority order.
    fn critical_instances(&self, max_paths: usize) -> Vec<usize> {
        let mut endpoints: Vec<(NetIndex, f64)> = self
            .outputs
            .iter()
            .copied()
            .filter_map(|net| {
                self.net_timing[net.0]
                    .map(|timing| (net, timing.rise.arrival.max(timing.fall.arrival)))
            })
            .collect();
        endpoints.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.0.cmp(&rhs.0.0)));
        let mut ranks: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        for (path_index, (mut net, _)) in endpoints.into_iter().take(max_paths).enumerate() {
            let mut depth = 0;
            while let Some(instance_index) = self.drivers[net.0] {
                ranks
                    .entry(instance_index)
                    .and_modify(|rank| rank.0 += 1)
                    .or_insert((1, path_index));
                let next = self.instances[instance_index]
                    .inputs
                    .iter()
                    .filter_map(|input| {
                        let net = input.net?;
                        let timing = self.net_timing[net.0]?;
                        Some((net, timing.rise.arrival.max(timing.fall.arrival)))
                    })
                    .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| rhs.0.0.cmp(&lhs.0.0)));
                let Some((next_net, _)) = next else {
                    break;
                };
                net = next_net;
                depth += 1;
                if depth > self.instances.len() {
                    break;
                }
            }
        }
        let mut result: Vec<(usize, (usize, usize))> = ranks.into_iter().collect();
        result.sort_by(|lhs, rhs| {
            rhs.1
                .0
                .cmp(&lhs.1.0)
                .then_with(|| lhs.1.1.cmp(&rhs.1.1))
                .then_with(|| lhs.0.cmp(&rhs.0))
        });
        result.into_iter().map(|(instance, _)| instance).collect()
    }

    /// Evaluates or commits one replacement while updating only its dirty cone.
    fn evaluate_cell_substitution(
        &mut self,
        instance_index: usize,
        new_cell_index: usize,
        commit: bool,
    ) -> Result<TrialResult> {
        let original = self
            .instances
            .get(instance_index)
            .ok_or_else(|| anyhow!("resizer instance index is out of range"))?
            .clone();
        let new_cell = self
            .library
            .cells
            .get(new_cell_index)
            .ok_or_else(|| anyhow!("resizer replacement cell is out of range"))?;
        let old_cell = &self.library.cells[original.cell_index];
        let mut changed_loads = Vec::new();
        for input in &original.inputs {
            let Some(net) = input.net else {
                continue;
            };
            let old_pin = old_cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == input.name)
                .ok_or_else(|| anyhow!("resizer old cell is missing input '{}'", input.name))?;
            let new_pin = new_cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == input.name)
                .ok_or_else(|| anyhow!("resizer new cell is missing input '{}'", input.name))?;
            if new_pin.direction != old_pin.direction {
                return Err(anyhow!("resizer replacement changes input pin direction"));
            }
            let old_cap = effective_input_capacitance_for_mapping(
                old_pin,
                format!("old resize input '{}.{}'", old_cell.name, input.name).as_str(),
            )?;
            let new_cap = effective_input_capacitance_for_mapping(
                new_pin,
                format!("new resize input '{}.{}'", new_cell.name, input.name).as_str(),
            )?;
            changed_loads.push((net, self.loads[net.0], old_cap, new_cap));
        }

        let old_output = &old_cell.pins[original.output_pin_index];
        let output_name = self.library.resolve_string(&old_output.name);
        let new_output_pin_index = new_cell
            .pins
            .iter()
            .position(|pin| {
                pin.direction == PinDirection::Output as i32
                    && self.library.resolve_string(&pin.name) == output_name
            })
            .ok_or_else(|| anyhow!("resizer replacement changes the output pin interface"))?;

        let mut dirty = BTreeSet::new();
        dirty.insert((self.topo_positions[instance_index], instance_index));
        for (net, _, _, _) in &changed_loads {
            if let Some(driver) = self.drivers[net.0] {
                dirty.insert((self.topo_positions[driver], driver));
            }
        }
        let mut saved_timings = Vec::new();

        self.instances[instance_index].cell_index = new_cell_index;
        self.instances[instance_index].output_pin_index = new_output_pin_index;
        for (net, _, old_cap, new_cap) in &changed_loads {
            self.loads[net.0].rise += new_cap.rise - old_cap.rise;
            self.loads[net.0].fall += new_cap.fall - old_cap.fall;
            if self.loads[net.0].rise < -1e-12 || self.loads[net.0].fall < -1e-12 {
                self.restore_trial(&original, &changed_loads, &saved_timings);
                return Err(anyhow!("resizer substitution produced a negative net load"));
            }
            self.loads[net.0].rise = self.loads[net.0].rise.max(0.0);
            self.loads[net.0].fall = self.loads[net.0].fall.max(0.0);
        }

        let result = (|| {
            while let Some((_, index)) = dirty.pop_first() {
                let output_net = self.instances[index].output_net;
                let previous_timing = self.net_timing[output_net.0];
                saved_timings.push((output_net, previous_timing));
                self.recompute_instance(index)?;
                if self.net_timing[output_net.0] != previous_timing {
                    for successor in &self.successors[index] {
                        dirty.insert((self.topo_positions[*successor], *successor));
                    }
                }
            }
            Ok(TrialResult {
                score: self.score(),
                recomputed_instances: saved_timings.len(),
            })
        })();

        if result.is_err() || !commit {
            self.restore_trial(&original, &changed_loads, &saved_timings);
        }
        result
    }

    /// Restores changed cell choice, input loads, and dirty-cone net timings.
    fn restore_trial(
        &mut self,
        original: &InstanceTiming,
        changed_loads: &[(
            NetIndex,
            CombinationalOutputLoad,
            CombinationalOutputLoad,
            CombinationalOutputLoad,
        )],
        saved_timings: &[(NetIndex, Option<SignalTiming>)],
    ) {
        let index = self.drivers[original.output_net.0]
            .expect("a sized instance must remain its original net's driver");
        self.instances[index] = original.clone();
        for (net, old_load, _, _) in changed_loads {
            self.loads[net.0] = *old_load;
        }
        for (net, timing) in saved_timings {
            self.net_timing[net.0] = *timing;
        }
    }

    /// Recomputes one mapped cell using the exact gv-stats NLDM evaluator.
    fn recompute_instance(&mut self, instance_index: usize) -> Result<()> {
        let instance = &self.instances[instance_index];
        let cell = &self.library.cells[instance.cell_index];
        let mut input_timings = Vec::with_capacity(instance.inputs.len());
        for input in &instance.inputs {
            if let Some(net) = input.net {
                let timing = self.net_timing[net.0].ok_or_else(|| {
                    anyhow!(
                        "cell '{}.{}' depends on a net with no timing source",
                        cell.name,
                        input.name
                    )
                })?;
                input_timings.push((input.name.as_str(), timing));
            }
        }
        let output = evaluate_combinational_cell_output_timing(
            self.library,
            cell.name.as_str(),
            &cell.pins[instance.output_pin_index],
            input_timings.as_slice(),
            self.loads[instance.output_net.0],
            &instance.known_pin_values,
            &mut self.diagnostic_counts,
        )?;
        self.net_timing[instance.output_net.0] = Some(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IncrementalCombinationalSta, ResizeOptions, resize_netlist};
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};
    use crate::netlist::report::build_sta_report;
    use crate::netlist::sta::StaOptions;

    #[test]
    fn upsizes_a_critical_combinational_gate() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
        )
        .unwrap();

        assert_eq!(stats.upsizes, 1);
        assert_eq!(stats.downsizes, 0);
        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AND2_FAST")
        );
        let full =
            build_sta_report(&module, &nets, &interner, &library, StaOptions::default()).unwrap();
        assert!((stats.final_delay - full.delay).abs() < 1e-9);
    }

    #[test]
    fn recovers_noncritical_area_without_increasing_delay() {
        let source = r#"
module top(a, b, critical, noncritical);
  input a, b;
  output critical, noncritical;
  AND2 slow_path (.A(a), .B(b), .Y(critical));
  BUF_FAST easy_path (.A(a), .Y(noncritical));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_iterations: 0,
                ..ResizeOptions::default()
            },
        )
        .unwrap();

        assert_eq!(stats.upsizes, 0);
        assert_eq!(stats.downsizes, 1);
        assert!(stats.final_area < stats.initial_area);
        assert!(stats.final_delay <= stats.initial_delay + 1e-9);
        assert_eq!(interner.resolve(module.instances[1].type_name), Some("BUF"));
    }

    #[test]
    fn rejected_incremental_trial_restores_exact_timing() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (module, nets, interner) = parse_module(source);
        let mut state = IncrementalCombinationalSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
        )
        .unwrap();
        let before = state.score().worst_delay();
        let fast_index = library
            .cells
            .iter()
            .position(|cell| cell.name == "AND2_FAST")
            .unwrap();
        let trial = state
            .evaluate_cell_substitution(0, fast_index, false)
            .unwrap();

        assert!(trial.score.worst_delay() < before);
        assert_eq!(state.score().worst_delay(), before);
        assert_eq!(state.cell_name(0), "AND2");
    }

    #[test]
    fn incremental_trial_stops_at_unchanged_upstream_timing() {
        let source = r#"
module top(a, b, y, other);
  input a, b;
  output y, other;
  wire root;
  BUF driver (.A(a), .Y(root));
  AND2 logic (.A(root), .B(b), .Y(y));
  BUF unrelated (.A(root), .Y(other));
endmodule
"#;
        let library = sizing_library();
        let (module, nets, interner) = parse_module(source);
        let mut state = IncrementalCombinationalSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
        )
        .unwrap();
        let before = state.score().worst_delay();
        let fast_index = library
            .cells
            .iter()
            .position(|cell| cell.name == "AND2_FAST")
            .unwrap();
        let trial = state
            .evaluate_cell_substitution(1, fast_index, false)
            .unwrap();

        assert_eq!(trial.recomputed_instances, 2);
        assert!(trial.score.worst_delay() < before);
        assert_eq!(state.score().worst_delay(), before);
    }

    #[test]
    fn sizes_logic_without_materializing_constant_output_cells() {
        let source = r#"
module top(a, b, y, zero, one);
  input a, b;
  output y, zero, one;
  assign zero = 1'b0;
  assign one = 1'b1;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
        )
        .expect("constant output assignments should not prevent sizing");

        assert_eq!(stats.upsizes, 1);
        assert_eq!(module.instances.len(), 1);
        assert_eq!(module.assigns.len(), 2);
        let full = build_sta_report(&module, &nets, &interner, &library, StaOptions::default())
            .expect("sized constant-output module should remain timing-complete");
        assert!((stats.final_delay - full.delay).abs() < 1e-9);
    }
}
