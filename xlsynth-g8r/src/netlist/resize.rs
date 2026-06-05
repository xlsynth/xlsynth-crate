// SPDX-License-Identifier: Apache-2.0

//! Standard-cell resizing for mapped gate-level netlists.

use crate::liberty::LibraryWithTimingData;
use crate::liberty_proto::Cell;
use crate::netlist::io::resolve_symbol;
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::report::build_area_report;
use crate::netlist::sta::{
    StaOptions, StaReport, analyze_register_boundary_max_arrival, is_sequential_boundary_cell,
};
use crate::netlist::stages::analyze_register_stages;
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Search limits and filtering for delay-oriented cell resizing.
#[derive(Clone, Debug, PartialEq)]
pub struct DelayResizeOptions {
    pub sta_options: StaOptions,
    pub max_iterations: usize,
    pub max_candidate_paths: usize,
    pub max_evaluations_per_iteration: usize,
    pub max_cell_candidates_per_instance: usize,
    pub improvement_epsilon: f64,
    pub resize_sequential_cells: bool,
    pub dont_use_patterns: Vec<String>,
}

impl Default for DelayResizeOptions {
    fn default() -> Self {
        Self {
            sta_options: StaOptions::default(),
            max_iterations: 25,
            max_candidate_paths: 8,
            max_evaluations_per_iteration: 128,
            max_cell_candidates_per_instance: 4,
            improvement_epsilon: 1e-9,
            resize_sequential_cells: true,
            dont_use_patterns: Vec::new(),
        }
    }
}

/// One accepted standard-cell substitution.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DelayResizeStep {
    pub iteration: usize,
    pub instance_index: usize,
    pub instance_name: String,
    pub old_cell: String,
    pub new_cell: String,
    pub delay_before: f64,
    pub delay_after: f64,
}

/// Machine-readable summary of one resizing run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DelayResizeSummary {
    pub initial_delay: f64,
    pub final_delay: f64,
    pub initial_area: f64,
    pub final_area: f64,
    pub evaluations: usize,
    pub failed_evaluations: usize,
    pub iterations: usize,
    pub replacements: Vec<DelayResizeStep>,
}

/// Resized module plus its optimization summary.
#[derive(Clone, Debug, PartialEq)]
pub struct DelayResizeResult {
    pub module: NetlistModule,
    pub summary: DelayResizeSummary,
}

/// Search limits and filtering for area minimization under a fixed delay cap.
#[derive(Clone, Debug, PartialEq)]
pub struct AreaResizeOptions {
    pub sta_options: StaOptions,
    pub max_iterations: usize,
    pub max_evaluations_per_iteration: usize,
    pub max_cell_candidates_per_instance: usize,
    pub delay_epsilon: f64,
    pub area_epsilon: f64,
    pub resize_sequential_cells: bool,
    pub dont_use_patterns: Vec<String>,
}

impl Default for AreaResizeOptions {
    fn default() -> Self {
        Self {
            sta_options: StaOptions::default(),
            max_iterations: 25,
            max_evaluations_per_iteration: 128,
            max_cell_candidates_per_instance: 4,
            delay_epsilon: 1e-9,
            area_epsilon: 1e-12,
            resize_sequential_cells: true,
            dont_use_patterns: Vec::new(),
        }
    }
}

/// One accepted area-reducing standard-cell substitution.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AreaResizeStep {
    pub iteration: usize,
    pub instance_index: usize,
    pub instance_name: String,
    pub old_cell: String,
    pub new_cell: String,
    pub delay_before: f64,
    pub delay_after: f64,
    pub area_before: f64,
    pub area_after: f64,
}

/// Machine-readable summary of one constrained area-resizing run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AreaResizeSummary {
    pub delay_limit: f64,
    pub initial_delay: f64,
    pub final_delay: f64,
    pub initial_area: f64,
    pub final_area: f64,
    pub evaluations: usize,
    pub failed_evaluations: usize,
    pub iterations: usize,
    pub replacements: Vec<AreaResizeStep>,
}

/// Area-resized module plus its optimization summary.
#[derive(Clone, Debug, PartialEq)]
pub struct AreaResizeResult {
    pub module: NetlistModule,
    pub summary: AreaResizeSummary,
}

#[derive(Clone, Debug)]
struct TimingScore {
    endpoint_delays: Vec<f64>,
}

impl TimingScore {
    fn from_report(report: &StaReport) -> Self {
        let mut endpoint_delays: Vec<f64> = report
            .register_input_arrivals
            .iter()
            .copied()
            .flatten()
            .collect();
        endpoint_delays.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
        Self { endpoint_delays }
    }

    fn worst_delay(&self) -> f64 {
        self.endpoint_delays.first().copied().unwrap_or(0.0)
    }

    fn is_better_than(&self, other: &Self, epsilon: f64) -> bool {
        for (candidate, current) in self.endpoint_delays.iter().zip(&other.endpoint_delays) {
            let difference = candidate - current;
            if difference.abs() <= epsilon {
                continue;
            }
            return difference < 0.0;
        }
        self.endpoint_delays.len() < other.endpoint_delays.len()
    }
}

#[derive(Clone, Debug)]
struct TrialMove {
    instance_index: usize,
    new_cell: String,
    report: StaReport,
    score: TimingScore,
}

#[derive(Clone, Debug)]
struct AreaTrialMove {
    instance_index: usize,
    new_cell: String,
    report: StaReport,
    delay: f64,
    area: f64,
}

/// Minimizes register-to-register critical-path delay by replacing instances
/// with Liberty-compatible standard cells.
pub fn resize_for_minimum_register_delay(
    module: &NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: &DelayResizeOptions,
) -> Result<DelayResizeResult> {
    validate_delay_options(options)?;
    let stages = analyze_register_stages(module, nets, interner, library.as_proto())?;
    if stages.register_indices.is_empty() {
        return Err(anyhow!(
            "delay resizing requires sequential boundaries; selected module contains no registers"
        ));
    }

    let equivalent_cells = build_equivalent_cell_candidates(
        module,
        interner,
        library.as_proto(),
        options.dont_use_patterns.as_slice(),
    )?;
    let initial_area = build_area_report(module, interner, library.as_proto())?.area;
    let mut resized_module = module.clone();
    let mut current_report = analyze_register_boundary_max_arrival(
        &resized_module,
        nets,
        interner,
        library,
        options.sta_options,
        false,
        stages.register_indices.as_slice(),
    )?;
    let mut current_score = TimingScore::from_report(&current_report);
    if current_score.endpoint_delays.is_empty() {
        return Err(anyhow!(
            "delay resizing found no register-to-register timing endpoints"
        ));
    }
    let initial_delay = current_score.worst_delay();
    let mut replacements = Vec::new();
    let mut evaluations = 0usize;
    let mut failed_evaluations = 0usize;

    for iteration in 0..options.max_iterations {
        let candidate_instances =
            prioritized_critical_instances(&current_report, options.max_candidate_paths);
        let mut best_move: Option<TrialMove> = None;
        let mut iteration_evaluations = 0usize;

        'instances: for instance_index in candidate_instances {
            let instance = &resized_module.instances[instance_index];
            let current_cell = resolve_symbol(interner, instance.type_name, "instance cell type")?;
            let current_cell_proto = library
                .as_proto()
                .cells
                .iter()
                .find(|cell| cell.name == current_cell)
                .ok_or_else(|| anyhow!("library does not contain cell '{}'", current_cell))?;
            if !options.resize_sequential_cells && is_sequential_boundary_cell(current_cell_proto) {
                continue;
            }
            let Some(candidates) = equivalent_cells.get(current_cell.as_str()) else {
                continue;
            };
            for new_cell in candidates
                .iter()
                .take(options.max_cell_candidates_per_instance)
            {
                if iteration_evaluations >= options.max_evaluations_per_iteration {
                    break 'instances;
                }
                iteration_evaluations += 1;
                evaluations += 1;

                let old_type = resized_module.instances[instance_index].type_name;
                let new_type = interner.get_or_intern(new_cell.as_str());
                resized_module.instances[instance_index].type_name = new_type;
                let trial_report = analyze_register_boundary_max_arrival(
                    &resized_module,
                    nets,
                    interner,
                    library,
                    options.sta_options,
                    false,
                    stages.register_indices.as_slice(),
                );
                resized_module.instances[instance_index].type_name = old_type;

                let trial_report = match trial_report {
                    Ok(report) => report,
                    Err(error) => {
                        failed_evaluations += 1;
                        log::debug!(
                            "delay resize rejected trial instance={} cell={} -> {}: {:#}",
                            instance_index,
                            current_cell,
                            new_cell,
                            error
                        );
                        continue;
                    }
                };
                let trial_score = TimingScore::from_report(&trial_report);
                let reference_score = best_move
                    .as_ref()
                    .map(|trial| &trial.score)
                    .unwrap_or(&current_score);
                if trial_score.is_better_than(reference_score, options.improvement_epsilon) {
                    best_move = Some(TrialMove {
                        instance_index,
                        new_cell: new_cell.clone(),
                        report: trial_report,
                        score: trial_score,
                    });
                }
            }
        }

        let Some(best_move) = best_move else {
            break;
        };
        let instance = &mut resized_module.instances[best_move.instance_index];
        let old_cell = resolve_symbol(interner, instance.type_name, "instance cell type")?;
        let instance_name = resolve_symbol(interner, instance.instance_name, "instance name")?;
        let delay_before = current_score.worst_delay();
        instance.type_name = interner.get_or_intern(best_move.new_cell.as_str());
        let delay_after = best_move.score.worst_delay();
        replacements.push(DelayResizeStep {
            iteration,
            instance_index: best_move.instance_index,
            instance_name,
            old_cell,
            new_cell: best_move.new_cell,
            delay_before,
            delay_after,
        });
        let replacement = replacements
            .last()
            .expect("replacement was appended immediately above");
        log::info!(
            "gv-resize iteration={} instance={} old_cell={} new_cell={} delay_before={:.9} delay_after={:.9}",
            replacement.iteration,
            replacement.instance_name,
            replacement.old_cell,
            replacement.new_cell,
            replacement.delay_before,
            replacement.delay_after
        );
        current_report = best_move.report;
        current_score = best_move.score;
    }

    let final_area = build_area_report(&resized_module, interner, library.as_proto())?.area;
    Ok(DelayResizeResult {
        module: resized_module,
        summary: DelayResizeSummary {
            initial_delay,
            final_delay: current_score.worst_delay(),
            initial_area,
            final_area,
            evaluations,
            failed_evaluations,
            iterations: replacements.len(),
            replacements,
        },
    })
}

/// Minimizes cell area while keeping the worst register-to-register delay at
/// or below the input netlist's initial delay.
pub fn resize_for_minimum_area_under_register_delay(
    module: &NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: &AreaResizeOptions,
) -> Result<AreaResizeResult> {
    validate_area_options(options)?;
    let stages = analyze_register_stages(module, nets, interner, library.as_proto())?;
    if stages.register_indices.is_empty() {
        return Err(anyhow!(
            "area resizing requires sequential boundaries; selected module contains no registers"
        ));
    }

    let equivalent_cells = build_equivalent_cell_candidates(
        module,
        interner,
        library.as_proto(),
        options.dont_use_patterns.as_slice(),
    )?;
    let cell_by_name: HashMap<&str, &Cell> = library
        .as_proto()
        .cells
        .iter()
        .map(|cell| (cell.name.as_str(), cell))
        .collect();
    let mut resized_module = module.clone();
    let mut current_area = build_area_report(&resized_module, interner, library.as_proto())?.area;
    let initial_area = current_area;
    let mut current_report = analyze_register_boundary_max_arrival(
        &resized_module,
        nets,
        interner,
        library,
        options.sta_options,
        false,
        stages.register_indices.as_slice(),
    )?;
    let current_score = TimingScore::from_report(&current_report);
    if current_score.endpoint_delays.is_empty() {
        return Err(anyhow!(
            "area resizing found no register-to-register timing endpoints"
        ));
    }
    let initial_delay = current_score.worst_delay();
    let delay_limit = initial_delay;
    let mut current_delay = initial_delay;
    let mut replacements = Vec::new();
    let mut evaluations = 0usize;
    let mut failed_evaluations = 0usize;

    for iteration in 0..options.max_iterations {
        let candidate_instances = prioritized_area_instances(
            &resized_module,
            interner,
            &cell_by_name,
            &equivalent_cells,
            &current_report,
            options.resize_sequential_cells,
            options.area_epsilon,
        )?;
        let mut best_move: Option<AreaTrialMove> = None;
        let mut iteration_evaluations = 0usize;

        'instances: for instance_index in candidate_instances {
            let instance = &resized_module.instances[instance_index];
            let current_cell = resolve_symbol(interner, instance.type_name, "instance cell type")?;
            let current_cell_proto = cell_by_name
                .get(current_cell.as_str())
                .ok_or_else(|| anyhow!("library does not contain cell '{}'", current_cell))?;
            if !options.resize_sequential_cells && is_sequential_boundary_cell(current_cell_proto) {
                continue;
            }
            let Some(candidates) = equivalent_cells.get(current_cell.as_str()) else {
                continue;
            };
            for new_cell in candidates
                .iter()
                .filter(|new_cell| {
                    cell_by_name[new_cell.as_str()].area
                        < current_cell_proto.area - options.area_epsilon
                })
                .take(options.max_cell_candidates_per_instance)
            {
                if iteration_evaluations >= options.max_evaluations_per_iteration {
                    break 'instances;
                }
                iteration_evaluations += 1;
                evaluations += 1;

                let new_cell_proto = cell_by_name[new_cell.as_str()];
                let trial_area = current_area - current_cell_proto.area + new_cell_proto.area;
                let old_type = resized_module.instances[instance_index].type_name;
                let new_type = interner.get_or_intern(new_cell.as_str());
                resized_module.instances[instance_index].type_name = new_type;
                let trial_report = analyze_register_boundary_max_arrival(
                    &resized_module,
                    nets,
                    interner,
                    library,
                    options.sta_options,
                    false,
                    stages.register_indices.as_slice(),
                );
                resized_module.instances[instance_index].type_name = old_type;

                let trial_report = match trial_report {
                    Ok(report) => report,
                    Err(error) => {
                        failed_evaluations += 1;
                        log::debug!(
                            "area resize rejected trial instance={} cell={} -> {}: {:#}",
                            instance_index,
                            current_cell,
                            new_cell,
                            error
                        );
                        continue;
                    }
                };
                let trial_delay = TimingScore::from_report(&trial_report).worst_delay();
                if trial_delay > delay_limit + options.delay_epsilon {
                    log::debug!(
                        "area resize rejected timing instance={} cell={} -> {} trial_delay={:.9} delay_limit={:.9}",
                        instance_index,
                        current_cell,
                        new_cell,
                        trial_delay,
                        delay_limit
                    );
                    continue;
                }
                let is_better = best_move.as_ref().map_or(true, |best| {
                    trial_area < best.area - options.area_epsilon
                        || ((trial_area - best.area).abs() <= options.area_epsilon
                            && (trial_delay < best.delay - options.delay_epsilon
                                || ((trial_delay - best.delay).abs() <= options.delay_epsilon
                                    && (instance_index, new_cell.as_str())
                                        < (best.instance_index, best.new_cell.as_str()))))
                });
                if is_better {
                    best_move = Some(AreaTrialMove {
                        instance_index,
                        new_cell: new_cell.clone(),
                        report: trial_report,
                        delay: trial_delay,
                        area: trial_area,
                    });
                }
            }
        }

        let Some(best_move) = best_move else {
            break;
        };
        let instance = &mut resized_module.instances[best_move.instance_index];
        let old_cell = resolve_symbol(interner, instance.type_name, "instance cell type")?;
        let instance_name = resolve_symbol(interner, instance.instance_name, "instance name")?;
        let area_before = current_area;
        let delay_before = current_delay;
        instance.type_name = interner.get_or_intern(best_move.new_cell.as_str());
        replacements.push(AreaResizeStep {
            iteration,
            instance_index: best_move.instance_index,
            instance_name,
            old_cell,
            new_cell: best_move.new_cell,
            delay_before,
            delay_after: best_move.delay,
            area_before,
            area_after: best_move.area,
        });
        let replacement = replacements
            .last()
            .expect("replacement was appended immediately above");
        log::info!(
            "gv-resize iteration={} instance={} old_cell={} new_cell={} area_before={:.9} area_after={:.9} delay_before={:.9} delay_after={:.9}",
            replacement.iteration,
            replacement.instance_name,
            replacement.old_cell,
            replacement.new_cell,
            replacement.area_before,
            replacement.area_after,
            replacement.delay_before,
            replacement.delay_after
        );
        current_report = best_move.report;
        current_delay = best_move.delay;
        current_area = best_move.area;
    }

    let final_area = build_area_report(&resized_module, interner, library.as_proto())?.area;
    let final_delay = TimingScore::from_report(&current_report).worst_delay();
    Ok(AreaResizeResult {
        module: resized_module,
        summary: AreaResizeSummary {
            delay_limit,
            initial_delay,
            final_delay,
            initial_area,
            final_area,
            evaluations,
            failed_evaluations,
            iterations: replacements.len(),
            replacements,
        },
    })
}

fn validate_delay_options(options: &DelayResizeOptions) -> Result<()> {
    if options.max_candidate_paths == 0 {
        return Err(anyhow!("max_candidate_paths must be greater than zero"));
    }
    if options.max_evaluations_per_iteration == 0 {
        return Err(anyhow!(
            "max_evaluations_per_iteration must be greater than zero"
        ));
    }
    if options.max_cell_candidates_per_instance == 0 {
        return Err(anyhow!(
            "max_cell_candidates_per_instance must be greater than zero"
        ));
    }
    if !options.improvement_epsilon.is_finite() || options.improvement_epsilon < 0.0 {
        return Err(anyhow!(
            "improvement_epsilon must be a non-negative finite value"
        ));
    }
    Ok(())
}

fn validate_area_options(options: &AreaResizeOptions) -> Result<()> {
    if options.max_evaluations_per_iteration == 0 {
        return Err(anyhow!(
            "max_evaluations_per_iteration must be greater than zero"
        ));
    }
    if options.max_cell_candidates_per_instance == 0 {
        return Err(anyhow!(
            "max_cell_candidates_per_instance must be greater than zero"
        ));
    }
    if !options.delay_epsilon.is_finite() || options.delay_epsilon < 0.0 {
        return Err(anyhow!("delay_epsilon must be a non-negative finite value"));
    }
    if !options.area_epsilon.is_finite() || options.area_epsilon < 0.0 {
        return Err(anyhow!("area_epsilon must be a non-negative finite value"));
    }
    Ok(())
}

fn build_equivalent_cell_candidates(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &crate::liberty_proto::Library,
    dont_use_patterns: &[String],
) -> Result<HashMap<String, Vec<String>>> {
    let cell_by_name: HashMap<&str, &Cell> = library
        .cells
        .iter()
        .map(|cell| (cell.name.as_str(), cell))
        .collect();
    let used_cells: BTreeSet<String> = module
        .instances
        .iter()
        .map(|instance| resolve_symbol(interner, instance.type_name, "instance cell type"))
        .collect::<Result<_>>()?;
    let mut used_signatures = BTreeMap::<String, Vec<String>>::new();
    for cell_name in &used_cells {
        let cell = cell_by_name
            .get(cell_name.as_str())
            .ok_or_else(|| anyhow!("library does not contain cell '{}'", cell_name))?;
        used_signatures
            .entry(cell_compatibility_signature(cell))
            .or_default()
            .push(cell_name.clone());
    }

    let mut candidates_by_signature = BTreeMap::<String, Vec<&Cell>>::new();
    for cell in &library.cells {
        let signature = cell_compatibility_signature(cell);
        if used_signatures.contains_key(signature.as_str())
            && !dont_use_patterns
                .iter()
                .any(|pattern| wildcard_matches(pattern, cell.name.as_str()))
        {
            candidates_by_signature
                .entry(signature)
                .or_default()
                .push(cell);
        }
    }

    let mut result = HashMap::new();
    for (signature, used_cell_names) in used_signatures {
        let candidates = candidates_by_signature
            .get(signature.as_str())
            .cloned()
            .unwrap_or_default();
        for current_cell_name in used_cell_names {
            let mut alternatives: Vec<&Cell> = candidates
                .iter()
                .copied()
                .filter(|candidate| candidate.name != current_cell_name)
                .collect();
            alternatives.sort_by(|lhs, rhs| {
                rhs.area
                    .partial_cmp(&lhs.area)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| lhs.name.cmp(&rhs.name))
            });
            result.insert(
                current_cell_name,
                alternatives
                    .into_iter()
                    .map(|cell| cell.name.clone())
                    .collect(),
            );
        }
    }
    Ok(result)
}

fn prioritized_area_instances(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    cell_by_name: &HashMap<&str, &Cell>,
    equivalent_cells: &HashMap<String, Vec<String>>,
    timing_report: &StaReport,
    resize_sequential_cells: bool,
    area_epsilon: f64,
) -> Result<Vec<usize>> {
    let mut path_delay_by_instance = HashMap::<usize, f64>::new();
    for (endpoint_delay, path) in timing_report
        .register_input_arrivals
        .iter()
        .zip(&timing_report.register_input_critical_paths)
    {
        let Some(endpoint_delay) = endpoint_delay else {
            continue;
        };
        for instance_index in path {
            path_delay_by_instance
                .entry(*instance_index)
                .and_modify(|delay| *delay = delay.max(*endpoint_delay))
                .or_insert(*endpoint_delay);
        }
    }
    let mut instances = Vec::new();
    for (instance_index, instance) in module.instances.iter().enumerate() {
        let current_cell = resolve_symbol(interner, instance.type_name, "instance cell type")?;
        let current_cell_proto = cell_by_name
            .get(current_cell.as_str())
            .ok_or_else(|| anyhow!("library does not contain cell '{}'", current_cell))?;
        if !resize_sequential_cells && is_sequential_boundary_cell(current_cell_proto) {
            continue;
        }
        let Some(candidates) = equivalent_cells.get(current_cell.as_str()) else {
            continue;
        };
        let best_area = candidates
            .iter()
            .filter_map(|candidate| cell_by_name.get(candidate.as_str()))
            .map(|candidate| candidate.area)
            .fold(current_cell_proto.area, f64::min);
        let area_saving = current_cell_proto.area - best_area;
        if area_saving > area_epsilon {
            let path_delay = path_delay_by_instance
                .get(&instance_index)
                .copied()
                .unwrap_or(0.0);
            instances.push((instance_index, path_delay, area_saving));
        }
    }
    instances.sort_by(|lhs, rhs| {
        lhs.1
            .total_cmp(&rhs.1)
            .then_with(|| rhs.2.total_cmp(&lhs.2))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    Ok(instances
        .into_iter()
        .map(|(instance_index, _, _)| instance_index)
        .collect())
}

fn cell_compatibility_signature(cell: &Cell) -> String {
    let mut pins: Vec<String> = cell
        .pins
        .iter()
        .map(|pin| {
            format!(
                "{}:{}:{}:{}",
                pin.name,
                pin.direction,
                pin.is_clocking_pin,
                compact_formula(pin.function.as_str())
            )
        })
        .collect();
    pins.sort();
    let mut sequential: Vec<String> = cell
        .sequential
        .iter()
        .map(|seq| {
            format!(
                "{}:{}:{}:{}:{}:{}",
                seq.kind,
                seq.state_var,
                compact_formula(seq.next_state.as_str()),
                compact_formula(seq.clock_expr.as_str()),
                compact_formula(seq.clear_expr.as_str()),
                compact_formula(seq.preset_expr.as_str())
            )
        })
        .collect();
    sequential.sort();
    let clock_gate = cell
        .clock_gate
        .as_ref()
        .map(|clock_gate| {
            let mut enable_pins = clock_gate.enable_pins.clone();
            let mut test_pins = clock_gate.test_pins.clone();
            enable_pins.sort();
            test_pins.sort();
            format!(
                "{}:{}:{}:{}",
                clock_gate.clock_pin,
                clock_gate.output_pin,
                enable_pins.join(","),
                test_pins.join(",")
            )
        })
        .unwrap_or_default();
    format!(
        "pins=[{}];seq=[{}];clock_gate={}",
        pins.join("|"),
        sequential.join("|"),
        clock_gate
    )
}

fn compact_formula(formula: &str) -> String {
    formula
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn prioritized_critical_instances(report: &StaReport, max_paths: usize) -> Vec<usize> {
    let mut endpoints: Vec<(usize, f64, &[usize])> = report
        .register_input_arrivals
        .iter()
        .enumerate()
        .filter_map(|(instance_index, delay)| {
            let delay = (*delay)?;
            let path = report
                .register_input_critical_paths
                .get(instance_index)?
                .as_slice();
            (!path.is_empty()).then_some((instance_index, delay, path))
        })
        .collect();
    endpoints.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

    let mut priority = BTreeMap::<usize, (usize, usize, usize)>::new();
    for (path_rank, (_, _, path)) in endpoints.into_iter().take(max_paths).enumerate() {
        for (path_position, instance_index) in path.iter().copied().enumerate() {
            let entry = priority
                .entry(instance_index)
                .or_insert((0, path_rank, path_position));
            entry.0 += 1;
            entry.1 = entry.1.min(path_rank);
            entry.2 = entry.2.min(path_position);
        }
    }
    let mut instances: Vec<(usize, (usize, usize, usize))> = priority.into_iter().collect();
    instances.sort_by(|lhs, rhs| {
        rhs.1
            .0
            .cmp(&lhs.1.0)
            .then_with(|| lhs.1.1.cmp(&rhs.1.1))
            .then_with(|| rhs.1.2.cmp(&lhs.1.2))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    instances
        .into_iter()
        .map(|(instance_index, _)| instance_index)
        .collect()
}

fn wildcard_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let mut matches = vec![vec![false; value.len() + 1]; pattern.len() + 1];
    matches[0][0] = true;
    for pattern_index in 0..pattern.len() {
        match pattern[pattern_index] {
            b'*' => {
                for value_index in 0..=value.len() {
                    matches[pattern_index + 1][value_index] |= matches[pattern_index][value_index];
                    if value_index > 0 {
                        matches[pattern_index + 1][value_index] |=
                            matches[pattern_index + 1][value_index - 1];
                    }
                }
            }
            b'?' => {
                for value_index in 0..value.len() {
                    matches[pattern_index + 1][value_index + 1] |=
                        matches[pattern_index][value_index];
                }
            }
            expected => {
                for value_index in 0..value.len() {
                    if value[value_index] == expected {
                        matches[pattern_index + 1][value_index + 1] |=
                            matches[pattern_index][value_index];
                    }
                }
            }
        }
    }
    matches[pattern.len()][value.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_proto::{
        Library, Pin, PinDirection, Sequential, SequentialKind, TimingArc, TimingTable,
    };
    use crate::netlist::parse::{Parser as NetlistParser, TokenScanner};

    fn scalar_table(kind: &str, value: f64) -> TimingTable {
        TimingTable {
            kind: kind.to_string(),
            values: vec![value],
            dimensions: vec![],
            ..Default::default()
        }
    }

    fn combinational_arc(related_pin: &str, delay: f64) -> TimingArc {
        TimingArc {
            related_pin: related_pin.to_string(),
            timing_sense: "positive_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", delay),
                scalar_table("cell_fall", delay),
                scalar_table("rise_transition", 1.0),
                scalar_table("fall_transition", 1.0),
            ],
            ..Default::default()
        }
    }

    fn dff_cell(name: &str, area: f64, c2q: f64, setup: f64) -> Cell {
        Cell {
            name: name.to_string(),
            pins: vec![
                Pin {
                    direction: PinDirection::Input as i32,
                    name: "D".to_string(),
                    capacitance: Some(1.0),
                    timing_arcs: vec![TimingArc {
                        related_pin: "CLK".to_string(),
                        timing_type: "setup_rising".to_string(),
                        tables: vec![
                            scalar_table("rise_constraint", setup),
                            scalar_table("fall_constraint", setup),
                        ],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Pin {
                    direction: PinDirection::Input as i32,
                    name: "CLK".to_string(),
                    is_clocking_pin: true,
                    capacitance: Some(1.0),
                    ..Default::default()
                },
                Pin {
                    direction: PinDirection::Output as i32,
                    function: "IQ".to_string(),
                    name: "Q".to_string(),
                    timing_arcs: vec![TimingArc {
                        related_pin: "CLK".to_string(),
                        timing_type: "rising_edge".to_string(),
                        tables: vec![
                            scalar_table("cell_rise", c2q),
                            scalar_table("cell_fall", c2q),
                            scalar_table("rise_transition", 1.0),
                            scalar_table("fall_transition", 1.0),
                        ],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            area,
            sequential: vec![Sequential {
                state_var: "IQ".to_string(),
                next_state: "D".to_string(),
                clock_expr: "CLK".to_string(),
                kind: SequentialKind::Ff as i32,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn buffer_cell(name: &str, area: f64, delay: f64) -> Cell {
        Cell {
            name: name.to_string(),
            pins: vec![
                Pin {
                    direction: PinDirection::Input as i32,
                    name: "A".to_string(),
                    capacitance: Some(1.0),
                    ..Default::default()
                },
                Pin {
                    direction: PinDirection::Output as i32,
                    function: "A".to_string(),
                    name: "Y".to_string(),
                    timing_arcs: vec![combinational_arc("A", delay)],
                    ..Default::default()
                },
            ],
            area,
            ..Default::default()
        }
    }

    #[test]
    fn delay_resize_improves_combinational_and_sequential_cells() {
        let src = r#"
module top (clk, d, y);
  input clk;
  input d;
  output y;
  wire q;
  wire n;
  DFF_X1 launch (.D(d), .CLK(clk), .Q(q));
  BUF_X1 logic (.A(q), .Y(n));
  DFF_X1 capture (.D(n), .CLK(clk), .Q(y));
endmodule
"#;
        let scanner = TokenScanner::from_str(src);
        let mut parser = NetlistParser::new(scanner);
        let modules = parser.parse_file().expect("netlist should parse");
        let module = modules[0].clone();
        let library = LibraryWithTimingData::from_proto(Library {
            cells: vec![
                dff_cell("DFF_X1", 1.0, 5.0, 5.0),
                dff_cell("DFF_X2", 2.0, 2.0, 2.0),
                buffer_cell("BUF_X1", 1.0, 10.0),
                buffer_cell("BUF_X2", 2.0, 4.0),
            ],
            ..Default::default()
        });

        let result = resize_for_minimum_register_delay(
            &module,
            parser.nets.as_slice(),
            &mut parser.interner,
            &library,
            &DelayResizeOptions {
                max_iterations: 8,
                max_candidate_paths: 1,
                max_evaluations_per_iteration: 16,
                max_cell_candidates_per_instance: 2,
                ..Default::default()
            },
        )
        .expect("delay resize should succeed");

        assert_eq!(result.summary.initial_delay, 20.0);
        assert_eq!(result.summary.final_delay, 8.0);
        assert_eq!(result.summary.replacements.len(), 3);
        let final_cells: Vec<&str> = result
            .module
            .instances
            .iter()
            .map(|instance| parser.interner.resolve(instance.type_name).unwrap())
            .collect();
        assert_eq!(final_cells, vec!["DFF_X2", "BUF_X2", "DFF_X2"]);
    }

    #[test]
    fn area_resize_uses_noncritical_slack_without_increasing_worst_delay() {
        let src = r#"
module top (clk, d0, d1, y0, y1);
  input clk;
  input d0;
  input d1;
  output y0;
  output y1;
  wire q0;
  wire q1;
  wire n0;
  wire n1;
  DFF_X1 launch0 (.D(d0), .CLK(clk), .Q(q0));
  BUF_X1 critical_logic (.A(q0), .Y(n0));
  DFF_X1 capture0 (.D(n0), .CLK(clk), .Q(y0));
  DFF_X1 launch1 (.D(d1), .CLK(clk), .Q(q1));
  BUF_X2 noncritical_logic (.A(q1), .Y(n1));
  DFF_X1 capture1 (.D(n1), .CLK(clk), .Q(y1));
endmodule
"#;
        let scanner = TokenScanner::from_str(src);
        let mut parser = NetlistParser::new(scanner);
        let modules = parser.parse_file().expect("netlist should parse");
        let module = modules[0].clone();
        let library = LibraryWithTimingData::from_proto(Library {
            cells: vec![
                dff_cell("DFF_X1", 1.0, 1.0, 1.0),
                buffer_cell("BUF_X1", 1.0, 10.0),
                buffer_cell("BUF_X2", 2.0, 4.0),
            ],
            ..Default::default()
        });

        let result = resize_for_minimum_area_under_register_delay(
            &module,
            parser.nets.as_slice(),
            &mut parser.interner,
            &library,
            &AreaResizeOptions {
                max_iterations: 4,
                max_evaluations_per_iteration: 8,
                max_cell_candidates_per_instance: 2,
                ..Default::default()
            },
        )
        .expect("area resize should succeed");

        assert_eq!(result.summary.delay_limit, 12.0);
        assert_eq!(result.summary.final_delay, 12.0);
        assert_eq!(result.summary.initial_area, 7.0);
        assert_eq!(result.summary.final_area, 6.0);
        assert_eq!(result.summary.replacements.len(), 1);
        let final_cells: Vec<&str> = result
            .module
            .instances
            .iter()
            .map(|instance| parser.interner.resolve(instance.type_name).unwrap())
            .collect();
        assert_eq!(
            final_cells,
            vec!["DFF_X1", "BUF_X1", "DFF_X1", "DFF_X1", "BUF_X1", "DFF_X1"]
        );
    }

    #[test]
    fn wildcard_matching_supports_orfs_style_patterns() {
        assert!(wildcard_matches("*_FAX*", "P2_FAX1"));
        assert!(wildcard_matches("P*EN*_*", "P1ENRA_DFFQ_X2"));
        assert!(!wildcard_matches("*_FAX*", "P2_INVX4"));
        assert!(wildcard_matches("INV?", "INV4"));
    }

    #[test]
    fn timing_score_uses_lexicographic_endpoint_order() {
        let current = TimingScore {
            endpoint_delays: vec![10.0, 10.0, 8.0],
        };
        let plateau_improvement = TimingScore {
            endpoint_delays: vec![10.0, 9.0, 8.0],
        };
        let regression = TimingScore {
            endpoint_delays: vec![10.1, 1.0, 1.0],
        };
        assert!(plateau_improvement.is_better_than(&current, 1e-9));
        assert!(!regression.is_better_than(&current, 1e-9));
    }
}
