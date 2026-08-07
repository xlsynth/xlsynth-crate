// SPDX-License-Identifier: Apache-2.0

//! Register-aware final technology mapping for synchronous transition AIGs.

use super::{
    MappedNetlist, TechMapOptions, TechMapTimingConstraints, TechMapTimingModel,
    map_choice_aig_to_netlist, map_choice_aig_to_netlist_with_nf_constraints, scalar_bit_name,
};
use crate::aig::{ChoiceAig, GateFn, SequentialGateFn};
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, Pin, PinDirection};
use crate::netlist::buffer::BufferStats;
use crate::netlist::optimize::{merge_buffer_stats, merge_resize_stats};
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistAssignKind, NetlistInstance,
    NetlistModule, NetlistPort, PortDirection, Pos, Span,
};
use crate::netlist::report::build_netlist_report_with_primary_input_arrivals;
use crate::netlist::resize::ResizeStats;
use crate::netlist::sequential_liberty::{
    GvEvalSequentialCellSpec, get_gv_eval_sequential_cell_spec,
};
use crate::netlist::sta::{
    CombinationalOutputLoad, PreparedStaLibrary, ScopedBoundaryTimingDefaultsSuppression,
    StaOptions, TimingQueryDiagnosticCounts, analyze_register_boundary_max_arrival,
    analyze_register_boundary_max_arrival_with_prepared_library,
    analyze_register_boundary_max_arrival_with_primary_input_arrivals,
    effective_input_capacitance_for_mapping, effective_representative_driver,
    evaluate_combinational_cell_output_timing, evaluate_primary_input_driver_timing,
    evaluate_sequential_cell_capture_timing_with_predecessor,
    evaluate_sequential_cell_output_timing, resolved_module_output_load,
};
use crate::netlist::timing_buffer::{
    BufferTimingConstraints, consolidate_timing_aware_buffers,
    insert_speculative_timing_aware_buffers, insert_timing_aware_buffers,
};
use crate::netlist::timing_resize::{
    recover_final_timing_protected_area, resize_timing_aware_netlist,
};
use crate::netlist::utils::scalar_constant_output_assignments;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use string_interner::StringInterner;
use string_interner::backend::StringBackend;
use string_interner::symbol::SymbolU32;
use xlsynth::IrBits;

/// Numerical tolerance used when checking an exact register setup violation.
const REGISTER_SETUP_SLACK_EPSILON: f64 = 1e-9;
/// Bound full sequential STA while consolidating buffered physical designs.
const MAX_SEQUENTIAL_BUFFER_CONSOLIDATION_INSTANCES: usize = 8192;
/// Avoid repeated complete timing of exceptionally large buffer forests.
const MAX_SEQUENTIAL_BUFFER_CONSOLIDATION_BUFFERS: usize = 512;
/// Revisit moderately broad data roots after their sink cells were resized.
const COORDINATED_SEQUENTIAL_MAX_FANOUT: usize = 6;
/// Retune affected paths without repeating the original complete sizing effort.
const MAX_POST_BUFFER_RESIZE_ITERATIONS: usize = 12;
/// Bound speculative downsizing; a final complete recovery pass still follows.
const MAX_POST_BUFFER_AREA_ITERATIONS: usize = 16;

/// Controller-owned timing constraints for one synchronous mapped design.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SequentialTechMapConstraints {
    /// Arrival times of external, flattened primary-input bits.
    pub primary_input_arrivals: BTreeMap<String, f64>,
    /// Required times of external, flattened primary-output bits.
    pub primary_output_required: BTreeMap<String, f64>,
    /// Common positive-edge register clock period, in Liberty time units.
    pub clock_period: Option<f64>,
}

/// One metadata-verified, positively clocked standard-cell flip-flop.
#[derive(Clone, Debug)]
struct FlipFlopBinding {
    cell_name: String,
    clock_pin: String,
    data_pin: String,
    output_pin: String,
    tied_inputs: BTreeMap<String, bool>,
    invert_data: bool,
    area: f64,
    clock_to_output: f64,
    setup: f64,
}

/// Maps a one-clock transition function and reinstates physical flip-flops.
pub fn map_sequential_choice_aig_to_netlist(
    design: &SequentialGateFn,
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    design
        .validate()
        .map_err(|error| anyhow!("invalid sequential design '{}': {error}", design.name))?;
    validate_constraints(design, constraints)?;
    validate_transition_interface(&design.transition, choice_aig.graph())?;

    let mut effective_options = options.clone();
    if effective_options.module_name.is_none() {
        effective_options.module_name = Some(design.name.clone());
    }
    // State Q/D ports are only a flattened interchange while mapping the
    // transition. Buffer the restored physical FF netlist, not those exposed
    // combinational pseudo-inputs and pseudo-outputs.
    effective_options.buffer_options = None;
    effective_options.resize_options = None;
    let sta_options = StaOptions {
        primary_input_transition: options.primary_input_transition,
        module_output_load: options.module_output_load,
    };

    if design.registers.is_empty() {
        let timing = TechMapTimingConstraints {
            primary_input_arrivals: constraints.primary_input_arrivals.clone(),
            primary_output_required: constraints.primary_output_required.clone(),
        };
        let mut mapped = map_transition(choice_aig, library, &timing, &effective_options)?;
        reinstate_sequential_boundary(&mut mapped, design, None)?;
        finalize_sequential_mapping(&mut mapped, library, constraints, sta_options)?;
        apply_requested_sequential_optimization(
            &mut mapped,
            library,
            constraints,
            options,
            sta_options,
            None,
        )?;
        return Ok(mapped);
    }

    for register in &design.registers {
        if register.initial_value.is_some() {
            bail!(
                "register '{}' has an explicit initial value; sequential technology mapping does not support power-up initialization",
                register.name
            );
        }
    }

    let flip_flops = index_flip_flops(library, sta_options)?;
    let mut failures = Vec::new();
    for flip_flop in flip_flops {
        match map_with_flip_flop(
            design,
            choice_aig,
            library,
            constraints,
            &effective_options,
            sta_options,
            &flip_flop,
            options.resize_options.is_some() && library.boundary_timing_defaults.is_some(),
        ) {
            Ok(mut mapped) => {
                apply_requested_sequential_optimization(
                    &mut mapped,
                    library,
                    constraints,
                    options,
                    sta_options,
                    Some(&flip_flop),
                )?;
                return Ok(mapped);
            }
            Err(error) => {
                if failures.len() < 4 {
                    failures.push(format!("{}: {error:#}", flip_flop.cell_name));
                }
            }
        }
    }
    bail!(
        "no usable positive-edge synchronous flip-flop produced a valid sequential mapping: {}",
        failures.join("; ")
    )
}

/// Buffers and resizes restored physical registers under full Liberty STA.
fn apply_requested_sequential_optimization(
    mapped: &mut MappedNetlist,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: &TechMapOptions,
    sta_options: StaOptions,
    flip_flop: Option<&FlipFlopBinding>,
) -> Result<()> {
    if options.buffer_options.is_none() && options.resize_options.is_none() {
        return Ok(());
    }
    let timing_constraints = BufferTimingConstraints {
        primary_input_arrivals: constraints.primary_input_arrivals.clone(),
        primary_output_required: constraints.primary_output_required.clone(),
        clock_period: constraints.clock_period,
    };
    let mut buffer_stats = options
        .buffer_options
        .as_ref()
        .map(|buffer_options| {
            insert_timing_aware_buffers(
                &mut mapped.module,
                &mut mapped.nets,
                &mut mapped.interner,
                library,
                buffer_options,
                sta_options,
                &timing_constraints,
            )
            .context("inserting register-aware, timing-driven buffers")
        })
        .transpose()?;
    let mut resize_stats = options
        .resize_options
        .as_ref()
        .map(|resize_options| {
            let mut resize_options = resize_options.clone();
            resize_options.sta_options = sta_options;
            resize_timing_aware_netlist(
                &mut mapped.module,
                &mapped.nets,
                &mut mapped.interner,
                library,
                &resize_options,
                &timing_constraints,
            )
            .context("resizing register-aware mapped gates and physical flip-flops")
        })
        .transpose()?;
    if let Some(flip_flop) = flip_flop
        && resize_stats.is_some()
        && library.boundary_timing_defaults.is_some()
    {
        // Keep input isolation while sizing launch registers, then remove only
        // electrically safe artifacts from their final characterized variants.
        bypass_register_boundary_identity_buffers(
            mapped,
            library,
            flip_flop,
            sta_options,
            RegisterBoundaryCleanupPhase::InputsOnly,
        )
        .context("removing register-input identity buffers after physical sizing")?;
    }
    finalize_sequential_mapping(mapped, library, constraints, sta_options)
        .context("verifying buffered and resized sequential mapping")?;
    if let (Some(buffering), Some(sizing)) = (buffer_stats.as_mut(), resize_stats.as_mut()) {
        revisit_sequential_buffering_after_sizing(
            mapped,
            library,
            constraints,
            options,
            &timing_constraints,
            buffering,
            sizing,
        )
        .context("revisiting register-aware buffering after physical gate and register sizing")?;
    }
    if let (Some(configured), Some(sizing), Some(buffering)) = (
        options.buffer_options.as_ref(),
        resize_stats.as_mut(),
        buffer_stats.as_mut(),
    ) && (1..=MAX_SEQUENTIAL_BUFFER_CONSOLIDATION_BUFFERS).contains(&buffering.buffers_inserted)
        && mapped.module.instances.len() <= MAX_SEQUENTIAL_BUFFER_CONSOLIDATION_INSTANCES
    {
        let mut recovery_options = configured.clone();
        recovery_options.module_output_load = sta_options.module_output_load;
        let recovered = consolidate_timing_aware_buffers(
            &mut mapped.module,
            mapped.nets.as_slice(),
            &mut mapped.interner,
            library,
            &recovery_options,
            sta_options,
            &timing_constraints,
            mapped.stats.worst_estimated_output_arrival,
        )
        .context("consolidating register-aware, timing-protected buffer trees")?;
        if recovered.buffers_removed > 0 {
            buffering.buffers_inserted = buffering
                .buffers_inserted
                .saturating_sub(recovered.buffers_removed);
            buffering.area_added = (buffering.area_added - recovered.area_recovered).max(0.0);
            buffering.max_fanout_after = recovered.max_fanout_after;
            buffering.max_load_after = recovered.max_load_after;
            buffering.unresolved_overloaded_nets = recovered.unresolved_overloaded_nets;
            buffering.final_worst_delay = Some(recovered.final_delay);
            buffering.timing_evaluations += recovered.timing_evaluations;
            sizing.final_delay = recovered.final_delay;
            finalize_sequential_mapping(mapped, library, constraints, sta_options)
                .context("verifying consolidated register-aware buffer trees")?;
        }
    }
    if let (Some(configured), Some(sizing)) =
        (options.resize_options.as_ref(), resize_stats.as_mut())
        && configured.max_area_iterations > 0
    {
        let mut resize_options = configured.clone();
        resize_options.sta_options = sta_options;
        let recovered = recover_final_timing_protected_area(
            &mut mapped.module,
            mapped.nets.as_slice(),
            &mut mapped.interner,
            library,
            &resize_options,
            &timing_constraints,
        )
        .context("recovering final register-aware area without worsening exact timing")?;
        if recovered.downsizes > 0 {
            finalize_sequential_mapping(mapped, library, constraints, sta_options)
                .context("verifying final register-aware area recovery")?;
        }
        merge_resize_stats(sizing, recovered);
    }
    if let Some(stats) = resize_stats.as_mut() {
        stats.final_area = mapped.stats.selected_area;
        stats.final_delay = mapped.stats.worst_estimated_output_arrival;
    }
    if let Some(stats) = buffer_stats.as_mut() {
        stats.final_worst_delay = Some(mapped.stats.worst_estimated_output_arrival);
    }
    mapped.stats.buffer_stats = buffer_stats;
    mapped.stats.resize_stats = resize_stats;
    Ok(())
}

/// Keeps post-sizing buffer and resizing changes only when capture timing
/// improves.
fn revisit_sequential_buffering_after_sizing(
    mapped: &mut MappedNetlist,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: &TechMapOptions,
    timing_constraints: &BufferTimingConstraints,
    buffer_stats: &mut BufferStats,
    resize_stats: &mut ResizeStats,
) -> Result<()> {
    let (Some(buffer_options), Some(resize_options)) = (
        options.buffer_options.as_ref(),
        options.resize_options.as_ref(),
    ) else {
        return Ok(());
    };
    if (resize_stats.upsizes == 0 && resize_stats.downsizes == 0 && resize_stats.pin_swaps == 0)
        || mapped.module.instances.len() > MAX_SEQUENTIAL_BUFFER_CONSOLIDATION_INSTANCES
        || buffer_stats.buffers_inserted > MAX_SEQUENTIAL_BUFFER_CONSOLIDATION_BUFFERS
        || (buffer_stats.max_fanout_after <= COORDINATED_SEQUENTIAL_MAX_FANOUT
            && buffer_stats.unresolved_overloaded_nets == 0)
    {
        return Ok(());
    }

    let sta_options = StaOptions {
        primary_input_transition: options.primary_input_transition,
        module_output_load: options.module_output_load,
    };
    let mut trial = MappedNetlist {
        module: mapped.module.clone(),
        nets: mapped.nets.clone(),
        interner: mapped.interner.clone(),
        stats: mapped.stats.clone(),
    };
    let mut exploratory_buffer_options = buffer_options.clone();
    exploratory_buffer_options.module_output_load = options.module_output_load;
    let trial_buffer_stats = insert_speculative_timing_aware_buffers(
        &mut trial.module,
        &mut trial.nets,
        &mut trial.interner,
        library,
        &exploratory_buffer_options,
        sta_options,
        timing_constraints,
    )?;
    if trial_buffer_stats.buffers_inserted == 0 {
        return Ok(());
    }

    let mut exploratory_resize_options = resize_options.clone();
    exploratory_resize_options.sta_options = sta_options;
    exploratory_resize_options.max_outer_iterations =
        exploratory_resize_options.max_outer_iterations.min(2);
    exploratory_resize_options.max_iterations = exploratory_resize_options
        .max_iterations
        .min(MAX_POST_BUFFER_RESIZE_ITERATIONS);
    exploratory_resize_options.max_area_iterations = exploratory_resize_options
        .max_area_iterations
        .min(MAX_POST_BUFFER_AREA_ITERATIONS);
    let trial_resize_stats = resize_timing_aware_netlist(
        &mut trial.module,
        &trial.nets,
        &mut trial.interner,
        library,
        &exploratory_resize_options,
        timing_constraints,
    )?;
    if let Err(error) = finalize_sequential_mapping(&mut trial, library, constraints, sta_options) {
        log::debug!("rejecting post-sizing register-aware buffering: {error:#}");
        return Ok(());
    }
    let previous_delay = mapped
        .stats
        .worst_register_to_register_arrival
        .unwrap_or(mapped.stats.worst_estimated_output_arrival);
    let trial_delay = trial
        .stats
        .worst_register_to_register_arrival
        .unwrap_or(trial.stats.worst_estimated_output_arrival);
    if trial_delay + exploratory_resize_options.improvement_epsilon >= previous_delay {
        log::debug!(
            "rejecting post-sizing register-aware buffering: capture delay {} does not improve {}",
            trial_delay,
            previous_delay,
        );
        return Ok(());
    }

    log::debug!(
        "accepted post-sizing register-aware buffering: capture delay {} -> {}, buffers={}",
        previous_delay,
        trial_delay,
        trial_buffer_stats.buffers_inserted,
    );
    *mapped = trial;
    merge_buffer_stats(buffer_stats, trial_buffer_stats);
    merge_resize_stats(resize_stats, trial_resize_stats);
    Ok(())
}

/// Validates non-negative, finite timing against actual external endpoints.
fn validate_constraints(
    design: &SequentialGateFn,
    constraints: &SequentialTechMapConstraints,
) -> Result<()> {
    if let Some(period) = constraints.clock_period {
        if !period.is_finite() || period <= 0.0 {
            bail!("clock period must be finite and strictly positive; got {period}");
        }
    }

    let external_inputs = design
        .inputs
        .iter()
        .flat_map(|id| {
            let port = &design.transition.inputs[id.index()];
            (0..port.get_bit_count())
                .map(move |bit| scalar_bit_name(&port.name, bit, port.get_bit_count()))
        })
        .collect::<BTreeSet<_>>();
    let external_outputs = design
        .outputs
        .iter()
        .flat_map(|id| {
            let port = &design.transition.outputs[id.index()];
            (0..port.get_bit_count())
                .map(move |bit| scalar_bit_name(&port.name, bit, port.get_bit_count()))
        })
        .collect::<BTreeSet<_>>();

    for (name, arrival) in &constraints.primary_input_arrivals {
        if !external_inputs.contains(name) {
            bail!("timing constraint names unknown external primary input '{name}'");
        }
        if !arrival.is_finite() || *arrival < 0.0 {
            bail!(
                "primary input arrival for '{name}' must be non-negative and finite; got {arrival}"
            );
        }
    }
    for (name, required) in &constraints.primary_output_required {
        if !external_outputs.contains(name) {
            bail!("timing constraint names unknown external primary output '{name}'");
        }
        if !required.is_finite() || *required < 0.0 {
            bail!(
                "primary output required time for '{name}' must be non-negative and finite; got {required}"
            );
        }
    }

    if let Some(clock) = &design.clock {
        if external_inputs.contains(&clock.name) || external_outputs.contains(&clock.name) {
            bail!(
                "clock name '{}' collides with an external data port",
                clock.name
            );
        }
    }

    let external_output_ids = design
        .outputs
        .iter()
        .map(|id| id.index())
        .collect::<BTreeSet<_>>();
    for register in &design.registers {
        if external_output_ids.contains(&register.d.index()) {
            bail!(
                "register '{}' D transition output is also an external output; sequential technology mapping requires distinct register and external endpoints",
                register.name
            );
        }
    }
    Ok(())
}

/// Requires the optimized AIG to preserve every transition bit and its order.
fn validate_transition_interface(original: &GateFn, optimized: &GateFn) -> Result<()> {
    let original_inputs = flatten_input_names(original)?;
    let optimized_inputs = flatten_input_names(optimized)?;
    check_flattened_interface("input", &original_inputs, &optimized_inputs)?;
    let original_outputs = flatten_output_names(original)?;
    let optimized_outputs = flatten_output_names(optimized)?;
    check_flattened_interface("output", &original_outputs, &optimized_outputs)
}

/// Collects transition inputs in AIGER-compatible, low-to-high bit order.
fn flatten_input_names(graph: &GateFn) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    for input in &graph.inputs {
        if input.get_bit_count() == 0 {
            bail!("transition input '{}' has zero bits", input.name);
        }
        for (bit_index, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            if bit.negated {
                bail!(
                    "transition input '{}' has a negated input binding",
                    input.name
                );
            }
            let name = scalar_bit_name(&input.name, bit_index, input.get_bit_count());
            if !seen.insert(name.clone()) {
                bail!("transition has duplicate flattened input '{name}'");
            }
            names.push(name);
        }
    }
    Ok(names)
}

/// Collects transition outputs in AIGER-compatible, low-to-high bit order.
fn flatten_output_names(graph: &GateFn) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    for output in &graph.outputs {
        if output.get_bit_count() == 0 {
            bail!("transition output '{}' has zero bits", output.name);
        }
        for bit_index in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit_index, output.get_bit_count());
            if !seen.insert(name.clone()) {
                bail!("transition has duplicate flattened output '{name}'");
            }
            names.push(name);
        }
    }
    Ok(names)
}

/// Reports the first changed bit instead of silently rebinding register state.
fn check_flattened_interface(kind: &str, expected: &[String], actual: &[String]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "optimized transition {kind} interface has {} bits; original transition has {}",
            actual.len(),
            expected.len()
        );
    }
    for (index, (expected_name, actual_name)) in expected.iter().zip(actual).enumerate() {
        if expected_name != actual_name {
            bail!(
                "optimized transition {kind} bit {index} is '{actual_name}'; expected '{expected_name}'"
            );
        }
    }
    Ok(())
}

/// Selects the compact constrained NF implementation only for NF modes.
fn map_transition(
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    if matches!(
        options.timing_model,
        TechMapTimingModel::NfUnit | TechMapTimingModel::NfLiberty
    ) {
        map_choice_aig_to_netlist_with_nf_constraints(choice_aig, library, constraints, options)
    } else {
        map_choice_aig_to_netlist(choice_aig, library, constraints, options)
    }
}

/// Maps with one characterized flip-flop and then restores the clock boundary.
#[allow(clippy::too_many_arguments)]
fn map_with_flip_flop(
    design: &SequentialGateFn,
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: &TechMapOptions,
    sta_options: StaOptions,
    flip_flop: &FlipFlopBinding,
    defer_input_cleanup: bool,
) -> Result<MappedNetlist> {
    if let Some(period) = constraints.clock_period {
        if period < flip_flop.setup {
            bail!(
                "clock period {period} is shorter than flip-flop '{}' setup requirement {}; a non-negative register required time cannot be constructed",
                flip_flop.cell_name,
                flip_flop.setup
            );
        }
    }

    let adjusted = adjust_register_data_phase(design, choice_aig, flip_flop.invert_data)?;
    let mut timing = TechMapTimingConstraints {
        primary_input_arrivals: constraints.primary_input_arrivals.clone(),
        primary_output_required: constraints.primary_output_required.clone(),
    };
    for register in &design.registers {
        let q = &design.transition.inputs[register.q.index()];
        let d = &design.transition.outputs[register.d.index()];
        for bit in 0..q.get_bit_count() {
            timing.primary_input_arrivals.insert(
                scalar_bit_name(&q.name, bit, q.get_bit_count()),
                flip_flop.clock_to_output,
            );
            if let Some(period) = constraints.clock_period {
                timing.primary_output_required.insert(
                    scalar_bit_name(&d.name, bit, d.get_bit_count()),
                    period - flip_flop.setup,
                );
            }
        }
    }

    let physical_inputs = design
        .inputs
        .iter()
        .flat_map(|input| {
            let port = &design.transition.inputs[input.index()];
            (0..port.get_bit_count())
                .map(move |bit| scalar_bit_name(&port.name, bit, port.get_bit_count()))
        })
        .collect::<BTreeSet<_>>();
    let physical_outputs = design
        .outputs
        .iter()
        .flat_map(|output| {
            let port = &design.transition.outputs[output.index()];
            (0..port.get_bit_count())
                .map(move |bit| scalar_bit_name(&port.name, bit, port.get_bit_count()))
        })
        .collect::<BTreeSet<_>>();
    // Preserve actual external electrical conditions while excluding synthetic
    // register Q/D interchange ports from virtual driving and output loading.
    let mut mapped = {
        let _physical_boundary = ScopedBoundaryTimingDefaultsSuppression::for_physical_ports(
            physical_inputs,
            physical_outputs,
        );
        map_transition(&adjusted, library, &timing, options)
            .with_context(|| format!("mapping transition for flip-flop '{}'", flip_flop.cell_name))
    }?;
    reinstate_sequential_boundary(&mut mapped, design, Some(flip_flop))?;
    bypass_register_boundary_identity_buffers(
        &mut mapped,
        library,
        flip_flop,
        sta_options,
        if defer_input_cleanup {
            RegisterBoundaryCleanupPhase::OutputsOnly
        } else {
            RegisterBoundaryCleanupPhase::All
        },
    )
    .context("removing redundant transition-interface identity buffers")?;
    finalize_sequential_mapping(&mut mapped, library, constraints, sta_options)?;
    Ok(mapped)
}

/// Identifies one scalar bit without conflating packed-port bit positions.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RegisterBoundaryBit {
    net: usize,
    bit: u32,
}

/// Records the exact physical Liberty pin connected to one scalar bit.
#[derive(Clone, Copy, Debug)]
struct RegisterBoundaryPinUse {
    instance: usize,
    connection: usize,
    cell: usize,
    pin: usize,
}

/// Stores the real physical pin indices of a Boolean unary identity cell.
#[derive(Clone, Copy, Debug)]
struct RegisterBoundaryIdentityCell {
    input_pin: usize,
    output_pin: usize,
}

/// Records the complete physical load already attached to one external input.
#[derive(Clone, Copy, Debug, Default)]
struct RegisterBoundaryInputLoad {
    capacitance: CombinationalOutputLoad,
    fanout: f64,
    has_other_consumers: bool,
}

/// Selects boundary artifacts appropriate to one physical optimization phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegisterBoundaryCleanupPhase {
    All,
    InputsOnly,
    OutputsOnly,
}

impl RegisterBoundaryCleanupPhase {
    /// Returns whether isolated package-input-to-register buffers may be
    /// removed.
    fn removes_inputs(self) -> bool {
        matches!(self, Self::All | Self::InputsOnly)
    }

    /// Returns whether isolated register-to-package-output buffers may be
    /// removed.
    fn removes_outputs(self) -> bool {
        matches!(self, Self::All | Self::OutputsOnly)
    }
}

/// Recognizes unary identity by Liberty truth, independently of cell spelling.
fn register_boundary_identity_cell(
    library: &Library,
    cell: &Cell,
) -> Option<RegisterBoundaryIdentityCell> {
    if !cell.sequential.is_empty() || cell.clock_gate.is_some() {
        return None;
    }
    let mut inputs = cell
        .pins
        .iter()
        .enumerate()
        .filter(|(_, pin)| pin.direction == PinDirection::Input as i32);
    let (input_pin, input) = inputs.next()?;
    if inputs.next().is_some() || input.is_clocking_pin {
        return None;
    }
    let mut outputs = cell
        .pins
        .iter()
        .enumerate()
        .filter(|(_, pin)| pin.direction == PinDirection::Output as i32);
    let (output_pin, output) = outputs.next()?;
    if outputs.next().is_some() {
        return None;
    }
    let function = parse_formula(library.resolve_string(&output.function)).ok()?;
    let input_name = library.resolve_string(&input.name).to_string();
    for value in [false, true] {
        if function.evaluate_partial(&HashMap::from([(input_name.clone(), value)])) != Some(value) {
            return None;
        }
    }
    Some(RegisterBoundaryIdentityCell {
        input_pin,
        output_pin,
    })
}

/// Returns the exact scalar bit for a one-bit Liberty pin connection.
fn register_boundary_scalar_bit(reference: &NetRef, nets: &[Net]) -> Option<RegisterBoundaryBit> {
    match reference {
        NetRef::Simple(index) if nets.get(index.0)?.width_bits() == 1 => {
            Some(RegisterBoundaryBit {
                net: index.0,
                bit: nets[index.0].bit_number(0)?,
            })
        }
        NetRef::BitSelect(index, bit) if nets.get(index.0)?.bit_offset(*bit).is_some() => {
            Some(RegisterBoundaryBit {
                net: index.0,
                bit: *bit,
            })
        }
        _ => None,
    }
}

/// Marks every bit touched by a structural assignment or non-scalar pin.
fn protect_register_boundary_reference(
    reference: &NetRef,
    nets: &[Net],
    protected: &mut BTreeSet<RegisterBoundaryBit>,
) {
    match reference {
        NetRef::Simple(index) | NetRef::PartSelect(index, _, _) => {
            if let Some(net) = nets.get(index.0) {
                for offset in 0..net.width_bits() {
                    if let Some(bit) = net.bit_number(offset) {
                        protected.insert(RegisterBoundaryBit { net: index.0, bit });
                    }
                }
            }
        }
        NetRef::BitSelect(index, bit) => {
            protected.insert(RegisterBoundaryBit {
                net: index.0,
                bit: *bit,
            });
        }
        NetRef::Concat(parts) => {
            for part in parts {
                protect_register_boundary_reference(part, nets, protected);
            }
        }
        NetRef::Literal(_) | NetRef::UnknownLiteral(_) | NetRef::Unconnected => {
            // Literal and absent connections have no physical net bit to alias.
        }
    }
}

/// Protects all physical bits mentioned by a preserved assignment expression.
fn protect_register_boundary_expression(
    expression: &AssignExpr,
    nets: &[Net],
    protected: &mut BTreeSet<RegisterBoundaryBit>,
) {
    match expression {
        AssignExpr::Leaf(reference) => {
            protect_register_boundary_reference(reference, nets, protected);
        }
        AssignExpr::Not(inner) => protect_register_boundary_expression(inner, nets, protected),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            protect_register_boundary_expression(lhs, nets, protected);
            protect_register_boundary_expression(rhs, nets, protected);
        }
    }
}

/// Checks that directly loading a register Q is legal and no slower.
fn can_bypass_register_output_buffer(
    library: &Library,
    flip_flop: &FlipFlopBinding,
    flip_flop_output: &Pin,
    buffer: &Cell,
    identity: RegisterBoundaryIdentityCell,
    options: StaOptions,
) -> Result<bool> {
    let direct_load = resolved_module_output_load(library, options)?;
    if flip_flop_output.max_capacitance.is_some_and(|maximum| {
        direct_load.rise.max(direct_load.fall) > maximum + REGISTER_SETUP_SLACK_EPSILON
    }) {
        return Ok(false);
    }

    let buffer_input = &buffer.pins[identity.input_pin];
    let buffered_q_load = effective_input_capacitance_for_mapping(
        buffer_input,
        &format!(
            "register output buffer '{}.{}'",
            buffer.name,
            library.resolve_string(&buffer_input.name)
        ),
    )?;
    let known_flip_flop_inputs = flip_flop
        .tied_inputs
        .iter()
        .map(|(name, value)| (name.clone(), *value))
        .collect::<HashMap<_, _>>();
    let mut diagnostics = TimingQueryDiagnosticCounts::default();
    let direct = evaluate_sequential_cell_output_timing(
        library,
        &flip_flop.cell_name,
        flip_flop_output,
        direct_load,
        &known_flip_flop_inputs,
        &mut diagnostics,
    )?;
    if flip_flop_output
        .max_transition
        .or(library.default_max_transition)
        .is_some_and(|maximum| {
            direct.rise.transition.max(direct.fall.transition)
                > maximum + REGISTER_SETUP_SLACK_EPSILON
        })
    {
        return Ok(false);
    }

    let buffered_launch = evaluate_sequential_cell_output_timing(
        library,
        &flip_flop.cell_name,
        flip_flop_output,
        buffered_q_load,
        &known_flip_flop_inputs,
        &mut diagnostics,
    )?;
    let buffered = evaluate_combinational_cell_output_timing(
        library,
        &buffer.name,
        &buffer.pins[identity.output_pin],
        &[(library.resolve_string(&buffer_input.name), buffered_launch)],
        direct_load,
        &HashMap::new(),
        &mut diagnostics,
    )?;
    Ok(
        direct.rise.arrival <= buffered.rise.arrival + REGISTER_SETUP_SLACK_EPSILON
            && direct.fall.arrival <= buffered.fall.arrival + REGISTER_SETUP_SLACK_EPSILON,
    )
}

/// Keeps a register-input buffer whenever direct wiring worsens actual capture.
fn can_bypass_register_input_buffer(
    library: &Library,
    flip_flop: &FlipFlopBinding,
    flip_flop_input: &Pin,
    buffer: &Cell,
    identity: RegisterBoundaryIdentityCell,
    buffered_source_load: RegisterBoundaryInputLoad,
    options: StaOptions,
) -> Result<bool> {
    let buffer_input = &buffer.pins[identity.input_pin];
    let buffered_input_load = effective_input_capacitance_for_mapping(
        buffer_input,
        "buffered register-boundary data input",
    )?;
    let direct_input_load = effective_input_capacitance_for_mapping(
        flip_flop_input,
        "direct register-boundary data input",
    )?;
    let representative_driver = effective_representative_driver(library)?;
    if (representative_driver.is_none() || buffered_source_load.has_other_consumers)
        && (direct_input_load.rise > buffered_input_load.rise + REGISTER_SETUP_SLACK_EPSILON
            || direct_input_load.fall > buffered_input_load.fall + REGISTER_SETUP_SLACK_EPSILON
            || flip_flop_input.fanout_load.unwrap_or(0.0)
                > buffer_input.fanout_load.unwrap_or(0.0) + REGISTER_SETUP_SLACK_EPSILON)
    {
        return Ok(false);
    }

    let direct_source_load = RegisterBoundaryInputLoad {
        capacitance: CombinationalOutputLoad {
            rise: buffered_source_load.capacitance.rise - buffered_input_load.rise
                + direct_input_load.rise,
            fall: buffered_source_load.capacitance.fall - buffered_input_load.fall
                + direct_input_load.fall,
        },
        fanout: buffered_source_load.fanout - buffer_input.fanout_load.unwrap_or(0.0)
            + flip_flop_input.fanout_load.unwrap_or(0.0),
        has_other_consumers: buffered_source_load.has_other_consumers,
    };
    if let Some(driver) = representative_driver.as_ref() {
        if driver.output_pin.max_capacitance.is_some_and(|maximum| {
            direct_source_load
                .capacitance
                .rise
                .max(direct_source_load.capacitance.fall)
                > maximum + REGISTER_SETUP_SLACK_EPSILON
        }) || driver
            .output_pin
            .max_fanout
            .or(library.default_max_fanout)
            .is_some_and(|maximum| {
                direct_source_load.fanout > maximum + REGISTER_SETUP_SLACK_EPSILON
            })
        {
            return Ok(false);
        }
    }

    let mut diagnostics = TimingQueryDiagnosticCounts::default();
    let direct = evaluate_primary_input_driver_timing(
        library,
        representative_driver.as_ref(),
        options,
        0.0,
        direct_source_load.capacitance,
        &mut diagnostics,
    )?;
    let direct_transition = direct.rise.transition.max(direct.fall.transition);
    if flip_flop_input
        .max_transition
        .or(library.default_max_transition)
        .is_some_and(|maximum| direct_transition > maximum + REGISTER_SETUP_SLACK_EPSILON)
        || representative_driver
            .as_ref()
            .and_then(|driver| driver.output_pin.max_transition)
            .is_some_and(|maximum| direct_transition > maximum + REGISTER_SETUP_SLACK_EPSILON)
    {
        return Ok(false);
    }

    let buffered_source = evaluate_primary_input_driver_timing(
        library,
        representative_driver.as_ref(),
        options,
        0.0,
        buffered_source_load.capacitance,
        &mut diagnostics,
    )?;
    let buffered = evaluate_combinational_cell_output_timing(
        library,
        &buffer.name,
        &buffer.pins[identity.output_pin],
        &[(library.resolve_string(&buffer_input.name), buffered_source)],
        direct_input_load,
        &HashMap::new(),
        &mut diagnostics,
    )?;
    let known_flip_flop_inputs = flip_flop
        .tied_inputs
        .iter()
        .map(|(name, value)| (name.clone(), *value))
        .collect::<HashMap<_, _>>();
    let Some(direct_capture) = evaluate_sequential_cell_capture_timing_with_predecessor(
        library,
        &flip_flop.cell_name,
        flip_flop_input,
        direct,
        &known_flip_flop_inputs,
        &mut diagnostics,
    )?
    else {
        return Ok(false);
    };
    let Some(buffered_capture) = evaluate_sequential_cell_capture_timing_with_predecessor(
        library,
        &flip_flop.cell_name,
        flip_flop_input,
        buffered,
        &known_flip_flop_inputs,
        &mut diagnostics,
    )?
    else {
        return Ok(false);
    };

    Ok(direct_capture.arrival <= buffered_capture.arrival + REGISTER_SETUP_SLACK_EPSILON)
}

/// Rebinds only state-, clock-, control-, and polarity-equivalent resized FFs.
fn compatible_register_boundary_binding(
    library: &Library,
    original: &FlipFlopBinding,
    cell: &Cell,
) -> Option<FlipFlopBinding> {
    if cell.name == original.cell_name {
        return Some(original.clone());
    }
    let spec = get_gv_eval_sequential_cell_spec(cell, library)
        .ok()
        .flatten()?;
    if spec.clock.is_negated || spec.clock.pin_name != original.clock_pin {
        return None;
    }

    let mut found_data = false;
    let mut controls = BTreeSet::new();
    for pin in cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32)
    {
        let name = library.resolve_string(&pin.name);
        if name == spec.clock.pin_name {
            if !pin.is_clocking_pin {
                return None;
            }
            continue;
        }
        if pin.is_clocking_pin {
            return None;
        }
        if name == original.data_pin {
            if found_data {
                return None;
            }
            found_data = true;
        } else if !controls.insert(name.to_string()) {
            return None;
        }
    }
    let original_controls = original
        .tied_inputs
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if !found_data || controls != original_controls {
        return None;
    }

    let data_inverted = data_formula_polarity(
        &spec.next_state,
        &spec,
        &original.data_pin,
        &original.tied_inputs,
    )?;
    let output = cell.pins.iter().find(|pin| {
        pin.direction == PinDirection::Output as i32
            && library.resolve_string(&pin.name) == original.output_pin
    })?;
    let output_formula = parse_formula(library.resolve_string(&output.function)).ok()?;
    let output_inverted = state_formula_polarity(&output_formula, &spec)?;
    if data_inverted ^ output_inverted != original.invert_data {
        return None;
    }

    Some(FlipFlopBinding {
        cell_name: cell.name.clone(),
        area: cell.area,
        ..original.clone()
    })
}

/// Removes identity cells used solely to expose temporary register Q/D ports.
fn bypass_register_boundary_identity_buffers(
    mapped: &mut MappedNetlist,
    library: &Library,
    flip_flop: &FlipFlopBinding,
    options: StaOptions,
    phase: RegisterBoundaryCleanupPhase,
) -> Result<()> {
    let library_indices = library
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut drivers = BTreeMap::<RegisterBoundaryBit, Vec<RegisterBoundaryPinUse>>::new();
    let mut sinks = BTreeMap::<RegisterBoundaryBit, Vec<RegisterBoundaryPinUse>>::new();
    let mut protected = BTreeSet::new();
    let mut instance_cells = Vec::with_capacity(mapped.module.instances.len());

    for (instance_index, instance) in mapped.module.instances.iter().enumerate() {
        let cell_name = mapped
            .interner
            .resolve(instance.type_name)
            .ok_or_else(|| anyhow!("cannot resolve register-boundary instance cell"))?;
        let cell_index = *library_indices.get(cell_name).ok_or_else(|| {
            anyhow!("register-boundary instance references unknown cell '{cell_name}'")
        })?;
        instance_cells.push(cell_index);
        let cell = &library.cells[cell_index];
        for (connection_index, (name, reference)) in instance.connections.iter().enumerate() {
            let pin_name = mapped
                .interner
                .resolve(*name)
                .ok_or_else(|| anyhow!("cannot resolve register-boundary pin on '{cell_name}'"))?;
            let (pin_index, pin) = cell
                .pins
                .iter()
                .enumerate()
                .find(|(_, pin)| library.resolve_string(&pin.name) == pin_name)
                .ok_or_else(|| anyhow!("cell '{cell_name}' has no physical pin '{pin_name}'"))?;
            let Some(bit) = register_boundary_scalar_bit(reference, &mapped.nets) else {
                protect_register_boundary_reference(reference, &mapped.nets, &mut protected);
                continue;
            };
            let use_record = RegisterBoundaryPinUse {
                instance: instance_index,
                connection: connection_index,
                cell: cell_index,
                pin: pin_index,
            };
            if pin.direction == PinDirection::Output as i32 {
                drivers.entry(bit).or_default().push(use_record);
            } else if pin.direction == PinDirection::Input as i32 {
                sinks.entry(bit).or_default().push(use_record);
            } else {
                protected.insert(bit);
            }
        }
    }

    for assignment in &mapped.module.assigns {
        protect_register_boundary_reference(&assignment.lhs, &mapped.nets, &mut protected);
        protect_register_boundary_expression(&assignment.rhs, &mapped.nets, &mut protected);
    }

    let mut module_inputs = BTreeSet::new();
    let mut module_outputs = BTreeSet::new();
    for port in &mapped.module.ports {
        let net = mapped
            .module
            .find_net_index(port.name, &mapped.nets)
            .ok_or_else(|| anyhow!("register-boundary port has no physical net"))?;
        let destination = match port.direction {
            PortDirection::Input => &mut module_inputs,
            PortDirection::Output => &mut module_outputs,
            PortDirection::Inout => continue,
        };
        for offset in 0..mapped.nets[net.0].width_bits() {
            let bit = mapped.nets[net.0]
                .bit_number(offset)
                .ok_or_else(|| anyhow!("register-boundary port has an invalid bit index"))?;
            destination.insert(RegisterBoundaryBit { net: net.0, bit });
        }
    }

    let mut identities = HashMap::<usize, Option<RegisterBoundaryIdentityCell>>::new();
    let mut actual_flip_flops = HashMap::<usize, Option<FlipFlopBinding>>::new();
    let mut removed = BTreeSet::new();
    let mut possibly_dead_nets = BTreeSet::new();

    for (instance_index, cell_index) in instance_cells.iter().copied().enumerate() {
        let Some(identity) = *identities.entry(cell_index).or_insert_with(|| {
            register_boundary_identity_cell(library, &library.cells[cell_index])
        }) else {
            continue;
        };
        let instance = &mapped.module.instances[instance_index];
        let Some((input_connection, input_reference)) =
            instance
                .connections
                .iter()
                .enumerate()
                .find_map(|(index, (name, reference))| {
                    (mapped.interner.resolve(*name)
                        == Some(library.resolve_string(
                            &library.cells[cell_index].pins[identity.input_pin].name,
                        )))
                    .then_some((index, reference.clone()))
                })
        else {
            continue;
        };
        let Some((output_connection, output_reference)) =
            instance
                .connections
                .iter()
                .enumerate()
                .find_map(|(index, (name, reference))| {
                    (mapped.interner.resolve(*name)
                        == Some(library.resolve_string(
                            &library.cells[cell_index].pins[identity.output_pin].name,
                        )))
                    .then_some((index, reference.clone()))
                })
        else {
            continue;
        };
        let Some(input_bit) = register_boundary_scalar_bit(&input_reference, &mapped.nets) else {
            continue;
        };
        let Some(output_bit) = register_boundary_scalar_bit(&output_reference, &mapped.nets) else {
            continue;
        };
        if protected.contains(&input_bit)
            || protected.contains(&output_bit)
            || drivers.get(&output_bit).is_none_or(|uses| {
                uses.len() != 1
                    || uses[0].instance != instance_index
                    || uses[0].connection != output_connection
            })
        {
            continue;
        }

        if module_inputs.contains(&input_bit)
            && !module_outputs.contains(&output_bit)
            && drivers.get(&input_bit).is_none_or(Vec::is_empty)
            && let Some(output_sinks) = sinks.get(&output_bit)
            && output_sinks.len() == 1
        {
            if !phase.removes_inputs() {
                continue;
            }
            let sink = output_sinks[0];
            let sink_cell = &library.cells[sink.cell];
            let sink_pin = &sink_cell.pins[sink.pin];
            if library.resolve_string(&sink_pin.name) == flip_flop.data_pin
                && let Some(actual_flip_flop) = actual_flip_flops
                    .entry(sink.cell)
                    .or_insert_with(|| {
                        compatible_register_boundary_binding(library, flip_flop, sink_cell)
                    })
                    .as_ref()
            {
                let mut source_load = RegisterBoundaryInputLoad::default();
                if let Some(source_sinks) = sinks.get(&input_bit) {
                    source_load.has_other_consumers = source_sinks.len() != 1
                        || source_sinks[0].instance != instance_index
                        || source_sinks[0].connection != input_connection;
                    for source_sink in source_sinks {
                        let pin = &library.cells[source_sink.cell].pins[source_sink.pin];
                        let capacitance = effective_input_capacitance_for_mapping(
                            pin,
                            "register-boundary primary-input sink",
                        )?;
                        source_load.capacitance.rise += capacitance.rise;
                        source_load.capacitance.fall += capacitance.fall;
                        source_load.fanout += pin.fanout_load.unwrap_or(0.0);
                    }
                }
                if module_outputs.contains(&input_bit) {
                    source_load.has_other_consumers = true;
                    let output_load = resolved_module_output_load(library, options)?;
                    source_load.capacitance.rise += output_load.rise;
                    source_load.capacitance.fall += output_load.fall;
                }
                if can_bypass_register_input_buffer(
                    library,
                    actual_flip_flop,
                    sink_pin,
                    &library.cells[cell_index],
                    identity,
                    source_load,
                    options,
                )? {
                    mapped.module.instances[sink.instance].connections[sink.connection].1 =
                        input_reference;
                    removed.insert(instance_index);
                    possibly_dead_nets.insert(output_bit.net);
                }
            }
            continue;
        }

        if !phase.removes_outputs()
            || !module_outputs.contains(&output_bit)
            || module_inputs.contains(&input_bit)
            || module_outputs.contains(&input_bit)
            || sinks.get(&output_bit).is_some_and(|uses| !uses.is_empty())
            || sinks.get(&input_bit).is_none_or(|uses| {
                uses.len() != 1
                    || uses[0].instance != instance_index
                    || uses[0].connection != input_connection
            })
        {
            continue;
        }
        let Some(q_drivers) = drivers.get(&input_bit) else {
            continue;
        };
        if q_drivers.len() != 1 {
            continue;
        }
        let q_driver = q_drivers[0];
        let q_cell = &library.cells[q_driver.cell];
        let q_pin = &q_cell.pins[q_driver.pin];
        if q_cell.name != flip_flop.cell_name
            || library.resolve_string(&q_pin.name) != flip_flop.output_pin
            || !can_bypass_register_output_buffer(
                library,
                flip_flop,
                q_pin,
                &library.cells[cell_index],
                identity,
                options,
            )?
        {
            continue;
        }
        mapped.module.instances[q_driver.instance].connections[q_driver.connection].1 =
            output_reference;
        removed.insert(instance_index);
        possibly_dead_nets.insert(input_bit.net);
    }

    if removed.is_empty() {
        return Ok(());
    }
    let mut index = 0usize;
    mapped.module.instances.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
    let mut remaining_nets = BTreeSet::new();
    for instance in &mapped.module.instances {
        for (_, reference) in &instance.connections {
            let mut indices = Vec::new();
            reference.collect_net_indices(&mut indices);
            remaining_nets.extend(indices.into_iter().map(|net| net.0));
        }
    }
    for assignment in &mapped.module.assigns {
        let mut indices = Vec::new();
        assignment.lhs.collect_net_indices(&mut indices);
        assignment.rhs.collect_net_indices(&mut indices);
        remaining_nets.extend(indices.into_iter().map(|net| net.0));
    }
    mapped
        .module
        .wires
        .retain(|wire| !possibly_dead_nets.contains(&wire.0) || remaining_nets.contains(&wire.0));
    Ok(())
}

/// Inverts physical D roots without discarding ABC structural-choice links.
fn adjust_register_data_phase(
    design: &SequentialGateFn,
    choice_aig: &ChoiceAig,
    invert_data: bool,
) -> Result<ChoiceAig> {
    if !invert_data {
        return Ok(choice_aig.clone());
    }
    let mut graph = choice_aig.graph().clone();
    let mut positions = BTreeMap::new();
    for (output_index, output) in graph.outputs.iter().enumerate() {
        for bit_index in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit_index, output.get_bit_count());
            positions.insert(name, (output_index, bit_index));
        }
    }
    let mut inverted = BTreeSet::new();
    for register in &design.registers {
        let output = &design.transition.outputs[register.d.index()];
        for bit_index in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit_index, output.get_bit_count());
            if !inverted.insert(name.clone()) {
                continue;
            }
            let (port_index, optimized_bit) = positions.get(&name).copied().ok_or_else(|| {
                anyhow!("optimized transition is missing register D bit '{name}'")
            })?;
            let operand = *graph.outputs[port_index].bit_vector.get_lsb(optimized_bit);
            graph.outputs[port_index]
                .bit_vector
                .set_lsb(optimized_bit, operand.negate());
        }
    }
    ChoiceAig::new(graph, choice_aig.sibling_links().to_vec()).map_err(|error| {
        anyhow!("could not preserve choice links while phasing register D: {error}")
    })
}

/// Builds and timing-characterizes eligible Liberty FFs without cell-name
/// rules.
fn index_flip_flops(library: &Library, options: StaOptions) -> Result<Vec<FlipFlopBinding>> {
    let prepared_library = PreparedStaLibrary::new(library)
        .context("indexing Liberty cells for sequential timing characterization")?;
    let mut candidates = Vec::new();
    let mut rejected = BTreeSet::new();
    for cell in &library.cells {
        if cell.sequential.is_empty() || cell.dont_use == Some(true) {
            continue;
        }
        if !cell.area.is_finite() || cell.area < 0.0 {
            rejected.insert(format!("{} has invalid cell area", cell.name));
            continue;
        }
        let spec = match get_gv_eval_sequential_cell_spec(cell, library) {
            Ok(Some(spec)) => spec,
            Ok(None) => continue,
            Err(error) => {
                rejected.insert(error);
                continue;
            }
        };
        if spec.clock.is_negated {
            rejected.insert(format!("{} requires a negative clock edge", cell.name));
            continue;
        }
        match enumerate_flip_flop_bindings(cell, &spec, library, &prepared_library, options) {
            Ok(bindings) if !bindings.is_empty() => candidates.extend(bindings),
            Ok(_) => {
                rejected.insert(format!(
                    "{} has no supported state output and data-input phase",
                    cell.name
                ));
            }
            Err(error) => {
                rejected.insert(format!("{}: {error:#}", cell.name));
            }
        }
    }
    candidates.sort_by(|lhs, rhs| {
        lhs.area
            .total_cmp(&rhs.area)
            .then_with(|| {
                (lhs.clock_to_output + lhs.setup).total_cmp(&(rhs.clock_to_output + rhs.setup))
            })
            .then_with(|| lhs.invert_data.cmp(&rhs.invert_data))
            .then_with(|| lhs.cell_name.cmp(&rhs.cell_name))
            .then_with(|| lhs.output_pin.cmp(&rhs.output_pin))
            .then_with(|| lhs.data_pin.cmp(&rhs.data_pin))
            .then_with(|| lhs.tied_inputs.cmp(&rhs.tied_inputs))
    });
    if candidates.is_empty() {
        let reasons = rejected.into_iter().take(6).collect::<Vec<_>>().join("; ");
        bail!(
            "Liberty has no usable positive-edge synchronous flip-flop with characterized clock-to-Q and setup timing{}",
            if reasons.is_empty() {
                String::new()
            } else {
                format!(": {reasons}")
            }
        );
    }
    Ok(candidates)
}

/// Enumerates state-output polarities and constant-safe scan/control settings.
fn enumerate_flip_flop_bindings(
    cell: &Cell,
    spec: &GvEvalSequentialCellSpec,
    library: &Library,
    prepared_library: &PreparedStaLibrary<'_>,
    options: StaOptions,
) -> Result<Vec<FlipFlopBinding>> {
    let mut input_names = cell
        .pins
        .iter()
        .filter(|pin| {
            pin.direction == PinDirection::Input as i32
                && !pin.is_clocking_pin
                && library.resolve_string(&pin.name) != spec.clock.pin_name
        })
        .map(|pin| library.resolve_string(&pin.name).to_string())
        .collect::<Vec<_>>();
    input_names.sort();
    if input_names.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("flip-flop contains duplicate data/control input pins");
    }
    if input_names.is_empty() {
        bail!("flip-flop has no data input");
    }
    if input_names.len() > 9 {
        bail!("flip-flop has more than eight constant-select control inputs");
    }

    let allowed_names = input_names
        .iter()
        .cloned()
        .chain(std::iter::once(spec.state_var.clone()))
        .chain(spec.complementary_state_var.iter().cloned())
        .collect::<BTreeSet<_>>();
    for name in spec.next_state.inputs() {
        if !allowed_names.contains(&name) {
            bail!("next_state references unsupported input or clock '{name}'");
        }
    }

    let mut output_choices = Vec::new();
    for pin in cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Output as i32)
    {
        let pin_name = library.resolve_string(&pin.name);
        let formula_text = library.resolve_string(&pin.function);
        if formula_text.is_empty() {
            continue;
        }
        let Ok(term) = parse_formula(formula_text) else {
            continue;
        };
        if term.inputs().iter().any(|name| {
            name != &spec.state_var
                && spec
                    .complementary_state_var
                    .as_ref()
                    .is_none_or(|complement| name != complement)
        }) {
            continue;
        }
        let Some(output_inverted) = state_formula_polarity(&term, spec) else {
            continue;
        };
        output_choices.push((pin_name.to_string(), output_inverted));
    }

    let mut candidates = Vec::new();
    for data_pin in &input_names {
        let control_pins = input_names
            .iter()
            .filter(|name| *name != data_pin)
            .cloned()
            .collect::<Vec<_>>();
        for assignment in 0..(1usize << control_pins.len()) {
            let tied_inputs = control_pins
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), ((assignment >> index) & 1) != 0))
                .collect::<BTreeMap<_, _>>();
            let Some(data_inverted) =
                data_formula_polarity(&spec.next_state, spec, data_pin, &tied_inputs)
            else {
                continue;
            };
            for (output_pin, output_inverted) in &output_choices {
                let mut candidate = FlipFlopBinding {
                    cell_name: cell.name.clone(),
                    clock_pin: spec.clock.pin_name.clone(),
                    data_pin: data_pin.clone(),
                    output_pin: output_pin.clone(),
                    tied_inputs: tied_inputs.clone(),
                    invert_data: data_inverted ^ output_inverted,
                    area: cell.area,
                    clock_to_output: 0.0,
                    setup: 0.0,
                };
                if let Ok((clock_to_output, setup)) =
                    characterize_flip_flop(&candidate, prepared_library, options)
                {
                    candidate.clock_to_output = clock_to_output;
                    candidate.setup = setup;
                    candidates.push(candidate);
                }
            }
        }
    }
    Ok(candidates)
}

/// Recognizes a real state output independently of the output pin's spelling.
fn state_formula_polarity(term: &Term, spec: &GvEvalSequentialCellSpec) -> Option<bool> {
    let mut values = HashMap::new();
    values.insert(spec.state_var.clone(), false);
    if let Some(complement) = &spec.complementary_state_var {
        values.insert(complement.clone(), true);
    }
    let at_zero = term.evaluate_partial(&values)?;
    values.insert(spec.state_var.clone(), true);
    if let Some(complement) = &spec.complementary_state_var {
        values.insert(complement.clone(), false);
    }
    let at_one = term.evaluate_partial(&values)?;
    (at_zero != at_one).then_some(at_zero)
}

/// Verifies that tied controls reduce next-state logic to `D` or `!D`.
fn data_formula_polarity(
    term: &Term,
    spec: &GvEvalSequentialCellSpec,
    data_pin: &str,
    tied_inputs: &BTreeMap<String, bool>,
) -> Option<bool> {
    let mut values = tied_inputs
        .iter()
        .map(|(name, value)| (name.clone(), *value))
        .collect::<HashMap<_, _>>();
    let mut polarity = None;
    for old_state in [false, true] {
        values.insert(spec.state_var.clone(), old_state);
        if let Some(complement) = &spec.complementary_state_var {
            values.insert(complement.clone(), !old_state);
        }
        values.insert(data_pin.to_string(), false);
        let at_zero = term.evaluate_partial(&values)?;
        values.insert(data_pin.to_string(), true);
        let at_one = term.evaluate_partial(&values)?;
        if at_zero == at_one || polarity.is_some_and(|previous| previous != at_zero) {
            return None;
        }
        polarity = Some(at_zero);
    }
    polarity
}

/// Measures Liberty setup and clock-to-Q with the production STA engine.
fn characterize_flip_flop(
    binding: &FlipFlopBinding,
    library: &PreparedStaLibrary<'_>,
    options: StaOptions,
) -> Result<(f64, f64)> {
    let probe_options = StaOptions {
        module_output_load: 0.0,
        ..options
    };
    let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
    let data = interner.get_or_intern("__seq_probe_data");
    let clock = interner.get_or_intern("__seq_probe_clock");
    let output = interner.get_or_intern("__seq_probe_output");
    let nets = vec![
        Net {
            name: data,
            width: None,
        },
        Net {
            name: clock,
            width: None,
        },
        Net {
            name: output,
            width: None,
        },
    ];
    let mut connections = vec![
        (
            interner.get_or_intern(&binding.data_pin),
            NetRef::Simple(NetIndex(0)),
        ),
        (
            interner.get_or_intern(&binding.clock_pin),
            NetRef::Simple(NetIndex(1)),
        ),
        (
            interner.get_or_intern(&binding.output_pin),
            NetRef::Simple(NetIndex(2)),
        ),
    ];
    for (name, value) in &binding.tied_inputs {
        connections.push((interner.get_or_intern(name), literal_net_ref(*value)?));
    }
    connections.sort_by(|(lhs, _), (rhs, _)| interner.resolve(*lhs).cmp(&interner.resolve(*rhs)));
    let module = NetlistModule {
        name: interner.get_or_intern("__seq_probe"),
        net_index_range: 0..nets.len(),
        ports: vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: data,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: clock,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: output,
            },
        ],
        wires: vec![],
        assigns: vec![],
        instances: vec![NetlistInstance {
            type_name: interner.get_or_intern(&binding.cell_name),
            instance_name: interner.get_or_intern("__seq_probe_register"),
            connections,
            inst_lineno: 1,
            inst_colno: 1,
        }],
    };
    // The characterization probe's Q output and D launch are synthetic ports,
    // so optional package-level external boundary models must not alter them.
    let _internal_boundary = ScopedBoundaryTimingDefaultsSuppression::new();
    let launch = analyze_register_boundary_max_arrival_with_prepared_library(
        &module,
        &nets,
        &interner,
        library,
        probe_options,
        false,
        &[0],
    )?;
    let q_timing = launch
        .timing_for_net(NetIndex(2))
        .ok_or_else(|| anyhow!("flip-flop output has no characterized clock-to-Q timing"))?;
    let clock_to_output = q_timing.rise.arrival.max(q_timing.fall.arrival);
    let capture = analyze_register_boundary_max_arrival_with_prepared_library(
        &module,
        &nets,
        &interner,
        library,
        probe_options,
        true,
        &[],
    )?;
    let setup = capture
        .register_input_arrivals
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| anyhow!("flip-flop data pin has no characterized setup timing"))?;
    if !clock_to_output.is_finite() || !setup.is_finite() {
        bail!("flip-flop setup and clock-to-Q timing must be finite");
    }
    Ok((clock_to_output, setup))
}

/// Produces an ordinary one-bit sequential constant connection.
fn literal_net_ref(value: bool) -> Result<NetRef> {
    IrBits::make_ubits(1, u64::from(value))
        .map(NetRef::Literal)
        .map_err(|error| anyhow!("could not construct one-bit sequential tie-off: {error}"))
}

/// Replaces exposed transition-state ports with internal FF-connected wires.
fn reinstate_sequential_boundary(
    mapped: &mut MappedNetlist,
    design: &SequentialGateFn,
    binding: Option<&FlipFlopBinding>,
) -> Result<()> {
    let mut port_nets = BTreeMap::new();
    for port in &mapped.module.ports {
        let name = mapped
            .interner
            .resolve(port.name)
            .ok_or_else(|| anyhow!("could not resolve mapped transition port"))?
            .to_string();
        let net = mapped
            .module
            .find_net_index(port.name, &mapped.nets)
            .ok_or_else(|| anyhow!("mapped transition port '{name}' has no net"))?;
        if port_nets.insert(name.clone(), net).is_some() {
            bail!("mapped transition contains duplicate port '{name}'");
        }
    }
    let constant_outputs = scalar_constant_output_assignments(&mapped.module, &mapped.nets)
        .context("validating mapped transition constant outputs")?;

    let mut external_names = BTreeSet::new();
    let mut hidden_scalar_nets = BTreeSet::new();
    let mut external_bit_refs = BTreeMap::new();
    let mut ports = Vec::new();
    for id in &design.inputs {
        let input = &design.transition.inputs[id.index()];
        if !external_names.insert(input.name.clone()) {
            bail!(
                "sequential module has duplicate external port '{}'",
                input.name
            );
        }
        ports.push(restore_external_port(
            mapped,
            &port_nets,
            &input.name,
            input.get_bit_count(),
            PortDirection::Input,
            &mut hidden_scalar_nets,
            &mut external_bit_refs,
        )?);
    }

    let clock_net = if let Some(clock) = &design.clock {
        if mapped
            .nets
            .iter()
            .any(|net| mapped.interner.resolve(net.name) == Some(clock.name.as_str()))
        {
            bail!("clock '{}' collides with a transition net", clock.name);
        }
        if !external_names.insert(clock.name.clone()) {
            bail!("clock '{}' collides with an external data port", clock.name);
        }
        let symbol = mapped.interner.get_or_intern(&clock.name);
        let index = NetIndex(mapped.nets.len());
        mapped.nets.push(Net {
            name: symbol,
            width: None,
        });
        ports.push(NetlistPort {
            direction: PortDirection::Input,
            width: None,
            name: symbol,
        });
        Some(index)
    } else {
        None
    };

    for id in &design.outputs {
        let output = &design.transition.outputs[id.index()];
        if !external_names.insert(output.name.clone()) {
            bail!(
                "sequential module has duplicate external port '{}'",
                output.name
            );
        }
        ports.push(restore_external_port(
            mapped,
            &port_nets,
            &output.name,
            output.get_bit_count(),
            PortDirection::Output,
            &mut hidden_scalar_nets,
            &mut external_bit_refs,
        )?);
    }

    for instance in &mut mapped.module.instances {
        for (_, reference) in &mut instance.connections {
            remap_external_net_ref(reference, &external_bit_refs)?;
        }
    }

    let register_d_nets = design
        .registers
        .iter()
        .flat_map(|register| {
            let output = &design.transition.outputs[register.d.index()];
            (0..output.get_bit_count())
                .map(move |bit| scalar_bit_name(&output.name, bit, output.get_bit_count()))
        })
        .filter_map(|name| port_nets.get(&name).copied())
        .map(|net| net.0)
        .collect::<BTreeSet<_>>();
    let mut packed_output_constants = Vec::new();
    for id in &design.outputs {
        let output = &design.transition.outputs[id.index()];
        if output.get_bit_count() == 1 {
            continue;
        }
        for bit in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit, output.get_bit_count());
            let net = *port_nets
                .get(&name)
                .ok_or_else(|| anyhow!("mapped transition is missing external output '{name}'"))?;
            if let Some(value) = constant_outputs.get(&net.0).copied() {
                let packed_ref = external_bit_refs
                    .get(&net.0)
                    .cloned()
                    .ok_or_else(|| anyhow!("packed constant output '{name}' has no bit binding"))?;
                packed_output_constants.push((packed_ref, value, net.0));
            }
        }
    }
    let packed_constant_nets = packed_output_constants
        .iter()
        .map(|(_, _, index)| *index)
        .collect::<BTreeSet<_>>();
    mapped.module.assigns.retain(|assign| {
        !matches!(
            assign.lhs,
            NetRef::Simple(net)
                if register_d_nets.contains(&net.0) || packed_constant_nets.contains(&net.0)
        )
    });
    for assign in &mut mapped.module.assigns {
        remap_external_net_ref(&mut assign.lhs, &external_bit_refs)?;
        remap_external_assign_expr(&mut assign.rhs, &external_bit_refs)?;
    }

    for (output, value, _) in packed_output_constants {
        let position = Pos {
            lineno: 1,
            colno: 1,
        };
        mapped.module.assigns.push(NetlistAssign {
            kind: NetlistAssignKind::Continuous,
            lhs: output,
            rhs: AssignExpr::Leaf(literal_net_ref(value)?),
            span: Span {
                start: position,
                limit: position,
            },
        });
    }

    if let Some(flip_flop) = binding {
        let mut used_instance_names = mapped
            .module
            .instances
            .iter()
            .map(|instance| {
                mapped
                    .interner
                    .resolve(instance.instance_name)
                    .ok_or_else(|| anyhow!("could not resolve mapped instance name"))
                    .map(str::to_string)
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let clock_net = clock_net.ok_or_else(|| {
            anyhow!("a registered sequential design must expose its positive-edge clock")
        })?;
        for register in &design.registers {
            let q = &design.transition.inputs[register.q.index()];
            let d = &design.transition.outputs[register.d.index()];
            for bit in 0..q.get_bit_count() {
                let q_name = scalar_bit_name(&q.name, bit, q.get_bit_count());
                let d_name = scalar_bit_name(&d.name, bit, d.get_bit_count());
                let q_net = *port_nets
                    .get(&q_name)
                    .ok_or_else(|| anyhow!("missing register Q net '{q_name}'"))?;
                let d_net = *port_nets
                    .get(&d_name)
                    .ok_or_else(|| anyhow!("missing register D net '{d_name}'"))?;
                let data = match constant_outputs.get(&d_net.0).copied() {
                    Some(value) => literal_net_ref(value)?,
                    None => NetRef::Simple(d_net),
                };
                let preferred = format!(
                    "u_reg_{}",
                    scalar_bit_name(&register.name, bit, q.get_bit_count())
                );
                let instance_name = unique_instance_name(&preferred, &mut used_instance_names);
                let mut connections = vec![
                    (
                        mapped.interner.get_or_intern(&flip_flop.clock_pin),
                        NetRef::Simple(clock_net),
                    ),
                    (mapped.interner.get_or_intern(&flip_flop.data_pin), data),
                    (
                        mapped.interner.get_or_intern(&flip_flop.output_pin),
                        NetRef::Simple(q_net),
                    ),
                ];
                for (pin, value) in &flip_flop.tied_inputs {
                    connections
                        .push((mapped.interner.get_or_intern(pin), literal_net_ref(*value)?));
                }
                connections.sort_by(|(lhs, _), (rhs, _)| {
                    mapped
                        .interner
                        .resolve(*lhs)
                        .cmp(&mapped.interner.resolve(*rhs))
                });
                mapped.module.instances.push(NetlistInstance {
                    type_name: mapped.interner.get_or_intern(&flip_flop.cell_name),
                    instance_name: mapped.interner.get_or_intern(&instance_name),
                    connections,
                    inst_lineno: 1,
                    inst_colno: 1,
                });
            }
        }
    }

    let public_nets = ports.iter().map(|port| port.name).collect::<BTreeSet<_>>();
    mapped.module.ports = ports;
    mapped.module.wires = mapped
        .nets
        .iter()
        .enumerate()
        .filter_map(|(index, net)| {
            (!public_nets.contains(&net.name)
                && !hidden_scalar_nets.contains(&index)
                && !(register_d_nets.contains(&index) && constant_outputs.contains_key(&index)))
            .then_some(NetIndex(index))
        })
        .collect();
    mapped.module.net_index_range = 0..mapped.nets.len();
    Ok(())
}

/// Restores one original scalar or packed external transition port.
fn restore_external_port(
    mapped: &mut MappedNetlist,
    port_nets: &BTreeMap<String, NetIndex>,
    name: &str,
    bit_count: usize,
    direction: PortDirection,
    hidden_scalar_nets: &mut BTreeSet<usize>,
    external_bit_refs: &mut BTreeMap<usize, NetRef>,
) -> Result<NetlistPort> {
    if bit_count == 0 {
        bail!("external port '{name}' has zero bits");
    }
    if bit_count == 1 {
        let net = *port_nets
            .get(name)
            .ok_or_else(|| anyhow!("mapped transition is missing external port '{name}'"))?;
        return Ok(NetlistPort {
            direction,
            width: None,
            name: mapped.nets[net.0].name,
        });
    }

    if mapped
        .nets
        .iter()
        .any(|net| mapped.interner.resolve(net.name) == Some(name))
    {
        bail!("packed external port '{name}' collides with an existing transition net");
    }
    let msb = u32::try_from(bit_count - 1)
        .map_err(|_| anyhow!("external port '{name}' is too wide for a Verilog packed range"))?;
    let symbol = mapped.interner.get_or_intern(name);
    let packed_net = NetIndex(mapped.nets.len());
    mapped.nets.push(Net {
        name: symbol,
        width: Some((msb, 0)),
    });

    for bit in 0..bit_count {
        let scalar = scalar_bit_name(name, bit, bit_count);
        let scalar_net = *port_nets
            .get(&scalar)
            .ok_or_else(|| anyhow!("mapped transition is missing external port bit '{scalar}'"))?;
        let bit = u32::try_from(bit)
            .map_err(|_| anyhow!("external port bit '{scalar}' exceeds a Verilog bit index"))?;
        if external_bit_refs
            .insert(scalar_net.0, NetRef::BitSelect(packed_net, bit))
            .is_some()
        {
            bail!("external port bit '{scalar}' is bound more than once");
        }
        hidden_scalar_nets.insert(scalar_net.0);
    }

    Ok(NetlistPort {
        direction,
        width: Some((msb, 0)),
        name: symbol,
    })
}

/// Rebinds old flattened external nets directly to packed Verilog bits.
fn remap_external_net_ref(
    reference: &mut NetRef,
    external_bit_refs: &BTreeMap<usize, NetRef>,
) -> Result<()> {
    match reference {
        NetRef::Simple(index) => {
            if let Some(replacement) = external_bit_refs.get(&index.0) {
                *reference = replacement.clone();
            }
        }
        NetRef::BitSelect(index, bit) => {
            if let Some(replacement) = external_bit_refs.get(&index.0) {
                if *bit != 0 {
                    bail!("flattened scalar external net uses invalid bit selection {bit}");
                }
                *reference = replacement.clone();
            }
        }
        NetRef::PartSelect(index, msb, lsb) => {
            if let Some(replacement) = external_bit_refs.get(&index.0) {
                if *msb != 0 || *lsb != 0 {
                    bail!("flattened scalar external net uses an invalid part selection");
                }
                *reference = replacement.clone();
            }
        }
        NetRef::Concat(parts) => {
            for part in parts {
                remap_external_net_ref(part, external_bit_refs)?;
            }
        }
        NetRef::Literal(_) | NetRef::UnknownLiteral(_) | NetRef::Unconnected => {
            // Constants and unconnected pins do not refer to a flattened port.
        }
    }
    Ok(())
}

/// Rebinds packed external references throughout a structural assignment.
fn remap_external_assign_expr(
    expression: &mut AssignExpr,
    external_bit_refs: &BTreeMap<usize, NetRef>,
) -> Result<()> {
    match expression {
        AssignExpr::Leaf(reference) => remap_external_net_ref(reference, external_bit_refs),
        AssignExpr::Not(inner) => remap_external_assign_expr(inner, external_bit_refs),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            remap_external_assign_expr(lhs, external_bit_refs)?;
            remap_external_assign_expr(rhs, external_bit_refs)
        }
    }
}

/// Selects a collision-free, deterministic physical register instance name.
fn unique_instance_name(preferred: &str, used: &mut BTreeSet<String>) -> String {
    if used.insert(preferred.to_string()) {
        return preferred.to_string();
    }
    for suffix in 1usize.. {
        let name = format!("{preferred}__{suffix}");
        if used.insert(name.clone()) {
            return name;
        }
    }
    unreachable!("an unbounded suffix sequence must yield a unique instance name")
}

/// Checks each output deadline against both exact physical launch classes.
fn validate_exact_output_requirements(
    mapped: &MappedNetlist,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: StaOptions,
) -> Result<()> {
    if constraints.primary_output_required.is_empty() {
        return Ok(());
    }

    let primary_launch = analyze_register_boundary_max_arrival_with_primary_input_arrivals(
        &mapped.module,
        mapped.nets.as_slice(),
        &mapped.interner,
        library,
        options,
        true,
        &[],
        &constraints.primary_input_arrivals,
    )
    .context("checking exact primary-input-to-output required times")?;

    let register_indices = mapped
        .module
        .instances
        .iter()
        .enumerate()
        .filter_map(|(index, instance)| {
            mapped
                .interner
                .resolve(instance.type_name)
                .and_then(|name| library.cells.iter().find(|cell| cell.name == name))
                .is_some_and(|cell| !cell.sequential.is_empty())
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let register_launch = if register_indices.is_empty() {
        None
    } else {
        Some(
            analyze_register_boundary_max_arrival(
                &mapped.module,
                mapped.nets.as_slice(),
                &mapped.interner,
                library,
                options,
                false,
                &register_indices,
            )
            .context("checking exact register-to-output required times")?,
        )
    };

    for (name, required) in &constraints.primary_output_required {
        let launch_timings = [
            ("primary-input", primary_launch.timing_for_output_bit(name)),
            (
                "register",
                register_launch
                    .as_ref()
                    .and_then(|report| report.timing_for_output_bit(name)),
            ),
        ];
        for (launch, timing) in launch_timings {
            let Some(timing) = timing else {
                // Constant outputs and unreachable launch classes have no
                // physical propagation path that can violate this deadline.
                continue;
            };
            let arrival = timing.rise.arrival.max(timing.fall.arrival);
            if arrival > required + REGISTER_SETUP_SLACK_EPSILON {
                bail!(
                    "primary output '{name}' violates required time {required}: exact {launch} launch arrival is {arrival}"
                );
            }
        }
    }

    Ok(())
}

/// Replaces transition-only estimates with exact full sequential STA and area.
fn finalize_sequential_mapping(
    mapped: &mut MappedNetlist,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: StaOptions,
) -> Result<()> {
    let report = build_netlist_report_with_primary_input_arrivals(
        &mapped.module,
        mapped.nets.as_slice(),
        &mapped.interner,
        library,
        options,
        &constraints.primary_input_arrivals,
    )
    .context("building exact register-aware mapped-netlist report")?;
    validate_exact_output_requirements(mapped, library, constraints, options)?;
    mapped.stats.selected_instance_count = report.cell_count;
    mapped.stats.selected_area = report.cell_area;
    mapped.stats.sequential_instance_count = mapped
        .module
        .instances
        .iter()
        .filter(|instance| {
            mapped
                .interner
                .resolve(instance.type_name)
                .and_then(|name| library.cells.iter().find(|cell| cell.name == name))
                .is_some_and(|cell| !cell.sequential.is_empty())
        })
        .count();
    mapped.stats.sequential_area = report.sequential_cell_area;
    mapped.stats.worst_input_to_register_arrival = report.max_input_to_register_delay;
    mapped.stats.worst_register_to_register_arrival = report.max_register_to_register_delay;
    mapped.stats.worst_register_to_output_arrival = report.max_register_to_output_delay;
    mapped.stats.clock_period = constraints.clock_period;
    let worst_capture = report
        .max_input_to_register_delay
        .into_iter()
        .chain(report.max_register_to_register_delay)
        .reduce(f64::max);
    let worst_register_slack = constraints
        .clock_period
        .zip(worst_capture)
        .map(|(period, arrival)| period - arrival);
    if let Some(slack) = worst_register_slack {
        if slack < -REGISTER_SETUP_SLACK_EPSILON {
            let period = constraints
                .clock_period
                .expect("register slack requires a clock period");
            let arrival = worst_capture.expect("register slack requires a capture arrival");
            bail!(
                "clock period {period} violates exact register setup timing: capture arrival {arrival} gives setup slack {slack}, below the -{REGISTER_SETUP_SLACK_EPSILON} tolerance"
            );
        }
    }
    mapped.stats.worst_register_slack = worst_register_slack;
    mapped.stats.worst_estimated_output_arrival = report
        .max_delay
        .into_iter()
        .chain(report.max_input_to_register_delay)
        .chain(report.max_register_to_register_delay)
        .chain(report.max_register_to_output_delay)
        .reduce(f64::max)
        .unwrap_or(0.0);
    mapped.stats.buffer_stats = None;
    mapped.stats.resize_stats = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        AigBitVector, AigOperand, ClockPort, RegisterBinding, TransitionInputId, TransitionOutputId,
    };
    use crate::aig_sim::sequential::{SequentialState, simulate};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_model::{
        LibraryBuilder, LuTableTemplate, Pin, Sequential, SequentialKind, TimingArc, TimingTable,
    };
    use crate::liberty_proto::{BoundaryTimingDefaults, TimingTableKind};
    use crate::netlist::buffer::BufferOptions;
    use crate::netlist::cell_catalog::test_utils::parse_module;
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
    use crate::netlist::resize::ResizeOptions;
    use crate::netlist::sequential_liberty::SequentialClockSpec;
    use crate::netlist::timing_buffer::tests::{
        high_fanout_register_source, registered_timing_library,
    };
    use crate::techmap::TechMapStats;

    /// Builds one constant, fully characterized Liberty timing table.
    fn test_timing_table(
        builder: &mut LibraryBuilder,
        kind: TimingTableKind,
        value: f64,
    ) -> TimingTable {
        builder
            .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![value], vec![], "")
            .expect("construct scalar sequential-mapper timing table")
    }

    /// Builds complete rise, fall, and slew characterization for one pin.
    fn test_output_arc(
        builder: &mut LibraryBuilder,
        related_pin: &str,
        sense: &str,
        timing_type: &str,
        delay: f64,
    ) -> TimingArc {
        let tables = vec![
            test_timing_table(builder, TimingTableKind::CellRise, delay),
            test_timing_table(builder, TimingTableKind::CellFall, delay),
            test_timing_table(builder, TimingTableKind::RiseTransition, 0.1),
            test_timing_table(builder, TimingTableKind::FallTransition, 0.1),
        ];
        builder
            .add_timing_arc(related_pin, sense, timing_type, "", tables)
            .expect("construct complete sequential-mapper output arc")
    }

    /// Builds a small, timing-complete positive-edge standard-cell library.
    fn test_library() -> Library {
        let mut builder = LibraryBuilder::new();
        let a = builder.intern_string("A").expect("intern A");
        let y = builder.intern_string("Y").expect("intern Y");
        let d = builder.intern_string("D").expect("intern D");
        let clock = builder.intern_string("CLK").expect("intern CLK");
        let q = builder.intern_string("Q").expect("intern Q");
        let identity = builder.intern_string("A").expect("intern identity");
        let inversion = builder.intern_string("!A").expect("intern inversion");
        let state = builder.intern_string("IQ").expect("intern state");
        let buffer_arc = test_output_arc(&mut builder, "A", "positive_unate", "combinational", 1.0);
        let inverter_arc =
            test_output_arc(&mut builder, "A", "negative_unate", "combinational", 1.0);
        let clock_arc = test_output_arc(&mut builder, "CLK", "non_unate", "rising_edge", 0.5);
        let setup_tables = vec![
            test_timing_table(&mut builder, TimingTableKind::RiseConstraint, 0.25),
            test_timing_table(&mut builder, TimingTableKind::FallConstraint, 0.25),
        ];
        let setup = builder
            .add_timing_arc("CLK", "", "setup_rising", "", setup_tables)
            .expect("construct setup arc");

        builder.cells = vec![
            Cell {
                name: "BUF".to_string(),
                pins: vec![
                    Pin {
                        name: a,
                        direction: PinDirection::Input as i32,
                        capacitance: Some(0.02),
                        ..Pin::default()
                    },
                    Pin {
                        name: y,
                        direction: PinDirection::Output as i32,
                        function: identity,
                        timing_arcs: vec![buffer_arc],
                        ..Pin::default()
                    },
                ],
                area: 1.0,
                ..Cell::default()
            },
            Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        name: a,
                        direction: PinDirection::Input as i32,
                        capacitance: Some(0.02),
                        ..Pin::default()
                    },
                    Pin {
                        name: y,
                        direction: PinDirection::Output as i32,
                        function: inversion,
                        timing_arcs: vec![inverter_arc],
                        ..Pin::default()
                    },
                ],
                area: 1.0,
                ..Cell::default()
            },
            Cell {
                name: "DFF".to_string(),
                pins: vec![
                    Pin {
                        name: d,
                        direction: PinDirection::Input as i32,
                        capacitance: Some(0.02),
                        timing_arcs: vec![setup],
                        ..Pin::default()
                    },
                    Pin {
                        name: clock,
                        direction: PinDirection::Input as i32,
                        is_clocking_pin: true,
                        capacitance: Some(0.01),
                        ..Pin::default()
                    },
                    Pin {
                        name: q,
                        direction: PinDirection::Output as i32,
                        function: state,
                        timing_arcs: vec![clock_arc],
                        ..Pin::default()
                    },
                ],
                area: 4.0,
                sequential: vec![Sequential {
                    state_var: "IQ".to_string(),
                    complementary_state_var: Some("IQN".to_string()),
                    next_state: "D".to_string(),
                    clock_expr: "CLK".to_string(),
                    kind: SequentialKind::Ff as i32,
                    ..Sequential::default()
                }],
                ..Cell::default()
            },
        ];
        builder.finish()
    }

    /// Makes flip-flop clock-to-Q observably depend on external output load.
    fn load_dependent_flip_flop_library() -> Library {
        let mut builder = LibraryBuilder::from_library(test_library());
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "seq_probe_output_load".to_string(),
            variable_1: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.0, 1.0],
            ..LuTableTemplate::default()
        });
        let mut tables = Vec::new();
        for (kind, values) in [
            (TimingTableKind::CellRise, vec![0.5, 10.5]),
            (TimingTableKind::CellFall, vec![0.5, 10.5]),
            (TimingTableKind::RiseTransition, vec![0.1, 0.2]),
            (TimingTableKind::FallTransition, vec![0.1, 0.2]),
        ] {
            tables.push(
                builder
                    .add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2], "")
                    .expect("construct load-sensitive flip-flop timing table"),
            );
        }
        let clock_arc = builder
            .add_timing_arc("CLK", "non_unate", "rising_edge", "", tables)
            .expect("construct load-sensitive clock-to-Q arc");
        let q_name = builder.intern_string("Q").expect("intern Q");
        let cell = builder
            .cells
            .iter_mut()
            .find(|cell| cell.name == "DFF")
            .expect("find synthetic flip-flop");
        let output = cell
            .pins
            .iter_mut()
            .find(|pin| pin.name == q_name)
            .expect("find synthetic flip-flop state output");
        output.timing_arcs = vec![clock_arc];
        builder.finish()
    }

    /// Adds a fast, oversized identity cell without relying on a buffer name.
    fn fast_boundary_buffer_library(library: Library, input_capacitance: f64) -> Library {
        let mut builder = LibraryBuilder::from_library(library);
        let input = builder
            .intern_string("A")
            .expect("intern fast identity input");
        let output = builder
            .intern_string("Y")
            .expect("intern fast identity output");
        let identity = builder
            .intern_string("!!A")
            .expect("intern semantically equivalent identity");
        let arc = test_output_arc(&mut builder, "A", "positive_unate", "combinational", 0.01);
        builder.cells.push(Cell {
            name: "DRIVE24".to_string(),
            pins: vec![
                Pin {
                    name: input,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(input_capacitance),
                    ..Pin::default()
                },
                Pin {
                    name: output,
                    direction: PinDirection::Output as i32,
                    function: identity,
                    max_capacitance: Some(2.0),
                    timing_arcs: vec![arc],
                    ..Pin::default()
                },
            ],
            area: 24.0,
            ..Cell::default()
        });
        builder.finish()
    }

    /// Adds a slow-slew external driver and optionally slew-sensitive setup.
    fn slow_representative_driver_library(
        register_max_transition: Option<f64>,
        slew_sensitive_setup: bool,
    ) -> Library {
        let mut builder = LibraryBuilder::from_library(test_library());
        let input = builder.intern_string("A").expect("intern source input");
        let output = builder.intern_string("Y").expect("intern source output");
        let identity = builder
            .intern_string("A")
            .expect("intern source identity function");
        let tables = vec![
            test_timing_table(&mut builder, TimingTableKind::CellRise, 0.0),
            test_timing_table(&mut builder, TimingTableKind::CellFall, 0.0),
            test_timing_table(&mut builder, TimingTableKind::RiseTransition, 0.5),
            test_timing_table(&mut builder, TimingTableKind::FallTransition, 0.5),
        ];
        let source_arc = builder
            .add_timing_arc("A", "positive_unate", "combinational", "", tables)
            .expect("construct slow-transition representative-driver arc");
        builder.cells.push(Cell {
            name: "SOURCE".to_string(),
            pins: vec![
                Pin {
                    name: input,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.02),
                    ..Pin::default()
                },
                Pin {
                    name: output,
                    direction: PinDirection::Output as i32,
                    function: identity,
                    timing_arcs: vec![source_arc],
                    ..Pin::default()
                },
            ],
            area: 32.0,
            ..Cell::default()
        });

        let setup = if slew_sensitive_setup {
            builder.lu_table_templates.push(LuTableTemplate {
                kind: "lu_table_template".to_string().into(),
                name: "register_data_transition".to_string(),
                variable_1: "constrained_pin_transition".to_string().into(),
                index_1: vec![0.1, 0.5],
                ..LuTableTemplate::default()
            });
            let template = builder.lu_table_templates.len() as u32;
            let mut tables = Vec::new();
            for kind in [
                TimingTableKind::RiseConstraint,
                TimingTableKind::FallConstraint,
            ] {
                tables.push(
                    builder
                        .add_timing_table_f64(
                            kind,
                            template,
                            vec![],
                            vec![],
                            vec![],
                            vec![0.1, 4.1],
                            vec![2],
                            "",
                        )
                        .expect("construct transition-sensitive register setup"),
                );
            }
            Some(
                builder
                    .add_timing_arc("CLK", "", "setup_rising", "", tables)
                    .expect("construct transition-sensitive register setup arc"),
            )
        } else {
            None
        };
        let data_name = builder.intern_string("D").expect("intern register data");
        let data_pin = builder
            .cells
            .iter_mut()
            .find(|cell| cell.name == "DFF")
            .and_then(|cell| cell.pins.iter_mut().find(|pin| pin.name == data_name))
            .expect("find synthetic register data pin");
        data_pin.max_transition = register_max_transition;
        if let Some(setup) = setup {
            data_pin.timing_arcs = vec![setup];
        }

        let mut library = builder.finish();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "SOURCE".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 1,
        });
        library
    }

    /// Creates a two-bit registered transition with canonical external ports.
    fn test_design() -> SequentialGateFn {
        let mut builder = GateBuilder::new(
            "unit_pipeline__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 2);
        let q = builder.add_input("state__q".to_string(), 2);
        builder.add_output("out".to_string(), q);
        builder.add_output("state__d".to_string(), data);
        SequentialGateFn::new(
            "unit_pipeline".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .expect("construct registered mapper test design")
    }

    /// Identifies actual identity cells between register outputs and package
    /// outputs.
    fn buffered_register_output_cells(mapped: &MappedNetlist, library: &Library) -> Vec<String> {
        let output = mapped
            .module
            .find_net_index(
                mapped
                    .interner
                    .get("out")
                    .expect("intern registered package output"),
                &mapped.nets,
            )
            .expect("find registered package output");
        let mut buffers = Vec::new();
        for instance in &mapped.module.instances {
            let cell_name = mapped
                .interner
                .resolve(instance.type_name)
                .expect("resolve registered-mapping cell type");
            let cell = library
                .cells
                .iter()
                .find(|cell| cell.name == cell_name)
                .expect("resolve registered-mapping Liberty cell");
            let Some(identity) = register_boundary_identity_cell(library, cell) else {
                continue;
            };
            let output_reference = instance
                .connections
                .iter()
                .find(|(name, _)| {
                    mapped.interner.resolve(*name)
                        == Some(library.resolve_string(&cell.pins[identity.output_pin].name))
                })
                .map(|(_, reference)| reference)
                .expect("find identity-buffer output connection");
            let Some(output_bit) = register_boundary_scalar_bit(output_reference, &mapped.nets)
            else {
                continue;
            };
            if output_bit.net != output.0 {
                continue;
            }
            let input_reference = instance
                .connections
                .iter()
                .find(|(name, _)| {
                    mapped.interner.resolve(*name)
                        == Some(library.resolve_string(&cell.pins[identity.input_pin].name))
                })
                .map(|(_, reference)| reference)
                .expect("find identity-buffer input connection");
            let input_bit = register_boundary_scalar_bit(input_reference, &mapped.nets)
                .expect("resolve identity-buffer register-output source");
            let driven_by_register_output = mapped.module.instances.iter().any(|source| {
                let source_name = mapped
                    .interner
                    .resolve(source.type_name)
                    .expect("resolve potential register-output driver type");
                let source_cell = library
                    .cells
                    .iter()
                    .find(|candidate| candidate.name == source_name)
                    .expect("resolve potential register-output Liberty cell");
                !source_cell.sequential.is_empty()
                    && source.connections.iter().any(|(name, reference)| {
                        let Some(pin) = source_cell.pins.iter().find(|pin| {
                            mapped.interner.resolve(*name)
                                == Some(library.resolve_string(&pin.name))
                        }) else {
                            return false;
                        };
                        pin.direction == PinDirection::Output as i32
                            && register_boundary_scalar_bit(reference, &mapped.nets)
                                .is_some_and(|bit| bit == input_bit)
                    })
            });
            assert!(
                driven_by_register_output,
                "physical output identity buffer '{cell_name}' must be driven by a register"
            );
            buffers.push(cell_name.to_string());
        }
        buffers.sort();
        buffers
    }

    /// Creates parsed Liberty state metadata without relying on pin spelling.
    fn state_spec(next_state: &str) -> GvEvalSequentialCellSpec {
        GvEvalSequentialCellSpec {
            state_var: "IQ".to_string(),
            complementary_state_var: Some("IQN".to_string()),
            next_state: parse_formula(next_state).expect("parse test next-state formula"),
            next_state_text: next_state.to_string(),
            clock: SequentialClockSpec {
                pin_name: "CLK".to_string(),
                is_negated: false,
            },
        }
    }

    #[test]
    fn recognizes_state_and_complementary_state_output_polarities() {
        let spec = state_spec("D");
        let positive = parse_formula("IQ").expect("parse state formula");
        let complementary = parse_formula("IQN").expect("parse complementary state formula");

        assert_eq!(state_formula_polarity(&positive, &spec), Some(false));
        assert_eq!(state_formula_polarity(&complementary, &spec), Some(true));
    }

    #[test]
    fn reduces_scan_and_enable_controls_to_a_deterministic_data_phase() {
        let scan = state_spec("(!D * !SE) + (SE * SI)");
        let scan_controls = BTreeMap::from([("SE".to_string(), false), ("SI".to_string(), false)]);
        assert_eq!(
            data_formula_polarity(&scan.next_state, &scan, "D", &scan_controls),
            Some(true)
        );

        let enabled = state_spec("(!EN * IQ) + (EN * D)");
        assert_eq!(
            data_formula_polarity(
                &enabled.next_state,
                &enabled,
                "D",
                &BTreeMap::from([("EN".to_string(), true)]),
            ),
            Some(false)
        );
        assert_eq!(
            data_formula_polarity(
                &enabled.next_state,
                &enabled,
                "D",
                &BTreeMap::from([("EN".to_string(), false)]),
            ),
            None
        );
    }

    #[test]
    fn register_data_phase_adjustment_preserves_structural_choice_links() {
        let mut builder = GateBuilder::new(
            "choice_transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data: AigOperand = builder
            .add_input("data".to_string(), 1)
            .try_into()
            .expect("extract data bit");
        let q: AigOperand = builder
            .add_input("state__q".to_string(), 1)
            .try_into()
            .expect("extract state bit");
        let sibling = builder.add_and_binary(data, q);
        let live = builder.add_and_binary(data, q);
        builder.add_output("out".to_string(), q.into());
        builder.add_output("state__d".to_string(), live.into());
        let transition = builder.build();
        let design = SequentialGateFn::new(
            "choice_pipeline".to_string(),
            transition.clone(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .expect("construct choice-rich registered design");
        let mut siblings = vec![None; transition.gates.len()];
        siblings[live.node.id] = Some(sibling.node);
        let choices = ChoiceAig::new(transition, siblings).expect("construct structural choices");

        let adjusted = adjust_register_data_phase(&design, &choices, true)
            .expect("adjust physical flip-flop data phase");

        assert_eq!(adjusted.sibling_links(), choices.sibling_links());
        assert_eq!(adjusted.sibling_link_count(), 1);
        assert_eq!(
            *adjusted.graph().outputs[1].bit_vector.get_lsb(0),
            choices.graph().outputs[1].bit_vector.get_lsb(0).negate()
        );
        assert_eq!(
            adjusted.graph().outputs[0].bit_vector.get_lsb(0),
            choices.graph().outputs[0].bit_vector.get_lsb(0)
        );
    }

    #[test]
    fn rejects_unknown_external_endpoint_constraints() {
        let design = test_design();
        let bad_input = SequentialTechMapConstraints {
            primary_input_arrivals: BTreeMap::from([("state__q_0".to_string(), 1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let input_error = validate_constraints(&design, &bad_input)
            .expect_err("internal register state must not be an external arrival");
        assert_eq!(
            input_error.to_string(),
            "timing constraint names unknown external primary input 'state__q_0'"
        );

        let bad_output = SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([("state__d_0".to_string(), 1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let output_error = validate_constraints(&design, &bad_output)
            .expect_err("internal register data must not be an external required endpoint");
        assert_eq!(
            output_error.to_string(),
            "timing constraint names unknown external primary output 'state__d_0'"
        );
    }

    #[test]
    fn rejects_nonpositive_and_nonfinite_clock_periods() {
        let design = test_design();
        for period in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let constraints = SequentialTechMapConstraints {
                clock_period: Some(period),
                ..SequentialTechMapConstraints::default()
            };
            let error = validate_constraints(&design, &constraints)
                .expect_err("sequential clock period must be finite and positive");
            assert!(error.to_string().contains("strictly positive"));
        }
    }

    #[test]
    fn rejects_nonfinite_external_endpoint_times() {
        let design = test_design();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let input = SequentialTechMapConstraints {
                primary_input_arrivals: BTreeMap::from([("data_0".to_string(), value)]),
                ..SequentialTechMapConstraints::default()
            };
            let error =
                validate_constraints(&design, &input).expect_err("external arrival must be finite");
            assert!(error.to_string().contains("non-negative and finite"));

            let output = SequentialTechMapConstraints {
                primary_output_required: BTreeMap::from([("out_0".to_string(), value)]),
                ..SequentialTechMapConstraints::default()
            };
            let error = validate_constraints(&design, &output)
                .expect_err("external required time must be finite");
            assert!(error.to_string().contains("non-negative and finite"));
        }
    }

    #[test]
    fn rejects_negative_external_arrivals_and_required_times() {
        let design = test_design();
        let negative_arrival = SequentialTechMapConstraints {
            primary_input_arrivals: BTreeMap::from([("data_0".to_string(), -1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let arrival_error = validate_constraints(&design, &negative_arrival)
            .expect_err("negative primary-input arrival must be rejected before FF indexing");
        assert_eq!(
            arrival_error.to_string(),
            "primary input arrival for 'data_0' must be non-negative and finite; got -1"
        );

        let negative_required = SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([("out_0".to_string(), -1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let required_error = validate_constraints(&design, &negative_required)
            .expect_err("negative primary-output requirement must be rejected before FF indexing");
        assert_eq!(
            required_error.to_string(),
            "primary output required time for 'out_0' must be non-negative and finite; got -1"
        );
    }

    #[test]
    fn rejects_clock_period_shorter_than_characterized_flip_flop_setup() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let constraints = SequentialTechMapConstraints {
            clock_period: Some(0.2),
            ..SequentialTechMapConstraints::default()
        };

        let error = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &test_library(),
            &constraints,
            &TechMapOptions::default(),
        )
        .expect_err("clock period below flip-flop setup cannot yield a valid required time");
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("setup requirement")
                && diagnostic.contains("non-negative register required time"),
            "unexpected infeasible setup diagnostic: {diagnostic}"
        );
    }

    #[test]
    fn characterizes_register_q_without_external_module_output_load() {
        let library = load_dependent_flip_flop_library();
        let candidates = index_flip_flops(
            &library,
            StaOptions {
                primary_input_transition: 0.01,
                module_output_load: 1.0,
            },
        )
        .expect("characterize an internally loaded flip-flop");
        let binding = candidates
            .iter()
            .find(|binding| binding.cell_name == "DFF")
            .expect("find load-sensitive synthetic flip-flop");

        assert!(
            (binding.clock_to_output - 0.5).abs() < 1e-12,
            "internal register Q incorrectly inherited the external output load: {}",
            binding.clock_to_output
        );
        assert!((binding.setup - 0.25).abs() < 1e-12);
    }

    #[test]
    fn characterizes_register_without_external_boundary_driver_or_receiver() {
        let mut library = load_dependent_flip_flop_library();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 3,
        });
        let candidates = index_flip_flops(&library, StaOptions::default())
            .expect("suppress external boundary defaults while characterizing synthetic FF ports");
        let binding = candidates
            .iter()
            .find(|binding| binding.cell_name == "DFF")
            .expect("find load-sensitive synthetic flip-flop");

        assert!((binding.clock_to_output - 0.5).abs() < 1e-12);
        assert!((binding.setup - 0.25).abs() < 1e-12);
    }

    #[test]
    fn restores_external_boundary_models_after_registered_transition_mapping() {
        let design = test_design();
        let mut library = load_dependent_flip_flop_library();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 1,
        });
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("restore package boundary models after mapping internal Q/D pseudo-ports");

        assert_eq!(mapped.stats.selected_instance_count, 2);
        assert!(
            (mapped.stats.worst_input_to_register_arrival.unwrap() - 1.25).abs() < 1e-12,
            "physical external data must still include the representative-driver delay"
        );
        assert!(
            (mapped.stats.worst_register_to_output_arrival.unwrap() - 0.7).abs() < 1e-12,
            "physical register output must still include the receiver-pin default load"
        );
    }

    #[test]
    fn preserves_real_transition_boundaries_while_hiding_register_interchange_ports() {
        let mut builder = GateBuilder::new(
            "mixed_boundary__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 2);
        let state = builder.add_input("state__q".to_string(), 2);
        builder.add_output("out".to_string(), state);
        builder.add_output("bypass".to_string(), data.clone());
        builder.add_output("state__d".to_string(), data);
        let design = SequentialGateFn::new(
            "mixed_boundary".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0), TransitionOutputId::new(1)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(2),
                initial_value: None,
            }],
        )
        .expect("construct mixed real and synthetic transition interfaces");
        let mut library = fast_boundary_buffer_library(test_library(), 0.2);
        let cheap_buffer = library
            .cells
            .iter_mut()
            .find(|cell| cell.name == "BUF")
            .expect("find minimum-area synthetic-interchange buffer");
        cheap_buffer
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find minimum-area buffer output")
            .max_capacitance = Some(0.3);
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 1,
        });
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                module_output_load: 0.5,
                ..TechMapOptions::default()
            },
        )
        .expect("distinguish physical external boundaries from register Q/D interchange");

        let bypass = mapped
            .module
            .find_net_index(mapped.interner.get("bypass").unwrap(), &mapped.nets)
            .expect("find packed physical bypass output");
        let physical_output_buffers = mapped
            .module
            .instances
            .iter()
            .filter(|instance| mapped.interner.resolve(instance.type_name) == Some("DRIVE24"))
            .collect::<Vec<_>>();
        assert_eq!(physical_output_buffers.len(), 2);
        for (bit, instance) in physical_output_buffers.into_iter().enumerate() {
            let output = instance
                .connections
                .iter()
                .find(|(name, _)| mapped.interner.resolve(*name) == Some("Y"))
                .expect("find physically selected output-buffer pin");
            assert_eq!(output.1, NetRef::BitSelect(bypass, bit as u32));
        }
        assert_eq!(mapped.stats.selected_instance_count, 4);
        assert!((mapped.stats.worst_input_to_register_arrival.unwrap() - 1.25).abs() < 1e-12);
        assert!((mapped.stats.worst_register_to_output_arrival.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bypasses_oversized_identity_cells_on_packed_register_boundaries() {
        let design = test_design();
        let library = fast_boundary_buffer_library(test_library(), 0.2);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                module_output_load: 0.05,
                ..TechMapOptions::default()
            },
        )
        .expect("bypass high-drive identity connections on both register boundaries");

        assert_eq!(mapped.stats.sequential_instance_count, 2);
        assert_eq!(mapped.stats.selected_instance_count, 2);
        assert_eq!(mapped.stats.selected_area, 8.0);
        let data = mapped
            .module
            .find_net_index(mapped.interner.get("data").unwrap(), &mapped.nets)
            .expect("resolve packed external input");
        let output = mapped
            .module
            .find_net_index(mapped.interner.get("out").unwrap(), &mapped.nets)
            .expect("resolve packed external output");
        for (bit, instance) in mapped.module.instances.iter().enumerate() {
            assert_eq!(mapped.interner.resolve(instance.type_name), Some("DFF"));
            let data_connection = instance
                .connections
                .iter()
                .find(|(name, _)| mapped.interner.resolve(*name) == Some("D"))
                .expect("find directly connected physical register data pin");
            assert_eq!(data_connection.1, NetRef::BitSelect(data, bit as u32));
            let output_connection = instance
                .connections
                .iter()
                .find(|(name, _)| mapped.interner.resolve(*name) == Some("Q"))
                .expect("find directly connected physical register output pin");
            assert_eq!(output_connection.1, NetRef::BitSelect(output, bit as u32));
        }

        let projected = project_labeled_sequential_netlist_aig(
            &mapped.module,
            &mapped.nets,
            &mapped.interner,
            &library,
            Some("clk"),
        )
        .expect("project directly wired physical register boundaries");
        let inputs = [0, 1, 3, 2, 0]
            .into_iter()
            .map(|value| vec![IrBits::make_ubits(2, value).unwrap()])
            .collect::<Vec<_>>();
        let expected = simulate(&design, &inputs, SequentialState::all_zeros(&design)).unwrap();
        let actual = simulate(
            &projected.sequential_gate_fn,
            &inputs,
            SequentialState::all_zeros(&projected.sequential_gate_fn),
        )
        .unwrap();
        assert_eq!(actual.external_outputs(), expected.external_outputs());
    }

    #[test]
    fn retains_input_buffer_when_representative_driver_violates_register_slew() {
        let design = test_design();
        let library = slow_representative_driver_library(Some(0.2), false);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("preserve register-input buffers that repair the actual source slew");

        assert_eq!(
            mapped
                .module
                .instances
                .iter()
                .filter(|instance| mapped.interner.resolve(instance.type_name) == Some("BUF"))
                .count(),
            2
        );
        assert_eq!(mapped.stats.selected_instance_count, 4);
        assert!((mapped.stats.worst_input_to_register_arrival.unwrap() - 1.25).abs() < 1e-12);
    }

    #[test]
    fn retains_input_buffer_when_direct_slew_worsens_register_setup() {
        let design = test_design();
        let library = slow_representative_driver_library(None, true);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("preserve register-input buffers that improve setup-inclusive arrival");

        assert_eq!(
            mapped
                .module
                .instances
                .iter()
                .filter(|instance| mapped.interner.resolve(instance.type_name) == Some("BUF"))
                .count(),
            2
        );
        assert_eq!(mapped.stats.selected_instance_count, 4);
        assert!((mapped.stats.worst_input_to_register_arrival.unwrap() - 1.1).abs() < 1e-6);
    }

    #[test]
    fn bypasses_private_input_buffer_when_driver_can_charge_larger_register_load() {
        let design = test_design();
        let mut library = slow_representative_driver_library(None, false);
        let flip_flop = library
            .cells
            .iter_mut()
            .find(|cell| cell.name == "DFF")
            .expect("find synthetic physical register");
        flip_flop
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
            .expect("find synthetic register data input")
            .capacitance = Some(0.04);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("allow a characterized private driver to charge a larger register input");

        assert_eq!(mapped.stats.selected_instance_count, 2);
        assert!(
            mapped
                .module
                .instances
                .iter()
                .all(|instance| mapped.interner.resolve(instance.type_name) == Some("DFF"))
        );
        assert!((mapped.stats.worst_input_to_register_arrival.unwrap() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn defers_private_input_cleanup_until_after_register_drive_sizing() {
        let design = test_design();
        let mut builder = LibraryBuilder::from_library(slow_representative_driver_library(
            /* register_max_transition= */ None, /* slew_sensitive_setup= */ false,
        ));
        let fast_clock = test_output_arc(&mut builder, "CLK", "non_unate", "rising_edge", 0.05);
        let mut fast_register = builder
            .cells
            .iter()
            .find(|cell| cell.name == "DFF")
            .cloned()
            .expect("clone the original physical register family");
        fast_register.name = "DFF_FAST".to_string();
        fast_register.area = 5.0;
        fast_register
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
            .expect("find faster register data input")
            .capacitance = Some(0.04);
        fast_register
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find faster register state output")
            .timing_arcs = vec![fast_clock];
        builder.cells.push(fast_register);
        let library = builder.finish();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                resize_options: Some(ResizeOptions::default()),
                ..TechMapOptions::default()
            },
        )
        .expect("retain input isolation while sizing and remove it from the final FF variant");

        assert_eq!(mapped.stats.selected_instance_count, 2);
        assert!(
            mapped
                .module
                .instances
                .iter()
                .all(|instance| mapped.interner.resolve(instance.type_name) == Some("DFF_FAST"))
        );
        assert!((mapped.stats.worst_input_to_register_arrival.unwrap() - 0.25).abs() < 1e-12);
        let register_output_delay = mapped.stats.worst_register_to_output_arrival.unwrap();
        assert!(
            (register_output_delay - 0.05).abs() < 1e-6,
            "float32 Liberty tables must preserve the upsized register delay: {register_output_delay}"
        );
        let sizing = mapped
            .stats
            .resize_stats
            .as_ref()
            .expect("preserve sizing diagnostics after deferred boundary cleanup");
        assert_eq!(sizing.register_upsizes, 2);
        assert_eq!(sizing.final_area, mapped.stats.selected_area);
        assert_eq!(
            sizing.final_delay,
            mapped.stats.worst_estimated_output_arrival
        );
    }

    #[test]
    fn retains_output_buffer_when_register_cannot_drive_external_load() {
        let design = test_design();
        let mut library = fast_boundary_buffer_library(test_library(), 0.025);
        let flip_flop = library
            .cells
            .iter_mut()
            .find(|cell| cell.name == "DFF")
            .expect("find synthetic physical register");
        flip_flop
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find synthetic register output")
            .max_capacitance = Some(0.03);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                module_output_load: 0.05,
                ..TechMapOptions::default()
            },
        )
        .expect("retain an electrically necessary register-output buffer");

        let retained = buffered_register_output_cells(&mapped, &library);
        assert_eq!(retained.len(), 2);
        assert!(retained.iter().all(|cell| cell == "DRIVE24"));
        assert_eq!(mapped.stats.selected_instance_count, 4);
    }

    #[test]
    fn retains_output_buffer_when_direct_clock_to_q_is_slower() {
        let design = test_design();
        let library = fast_boundary_buffer_library(load_dependent_flip_flop_library(), 0.025);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                module_output_load: 0.5,
                ..TechMapOptions::default()
            },
        )
        .expect("retain a register-output buffer that improves actual output arrival");

        let retained = buffered_register_output_cells(&mapped, &library);
        assert_eq!(retained.len(), 2);
        assert!(retained.iter().all(|cell| cell == "DRIVE24"));
    }

    #[test]
    fn preserves_shared_register_output_buffers_on_register_capture_paths() {
        let mut design = test_design();
        design.transition.outputs[1].bit_vector = design.transition.inputs[1].bit_vector.clone();
        let library = fast_boundary_buffer_library(test_library(), 0.2);
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                module_output_load: 0.05,
                ..TechMapOptions::default()
            },
        )
        .expect("preserve identity cells sharing register-output capture fanout");

        let output_buffers = buffered_register_output_cells(&mapped, &library);
        assert_eq!(output_buffers.len(), 2);
        assert!(output_buffers.iter().all(|cell| cell == "DRIVE24"));
        let identity_count = mapped
            .module
            .instances
            .iter()
            .filter(|instance| {
                let name = mapped
                    .interner
                    .resolve(instance.type_name)
                    .expect("resolve shared register-boundary cell type");
                let cell = library
                    .cells
                    .iter()
                    .find(|cell| cell.name == name)
                    .expect("resolve shared register-boundary Liberty cell");
                register_boundary_identity_cell(&library, cell).is_some()
            })
            .count();
        assert_eq!(identity_count, 4);
        assert!(mapped.stats.worst_register_to_register_arrival.is_some());
    }

    #[test]
    fn rejects_exact_clock_setup_slack_below_numerical_tolerance() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let library = test_library();
        let mut mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map an unconstrained sequential test design");
        let arrival = mapped
            .stats
            .worst_input_to_register_arrival
            .expect("synthetic design must have a register capture path");

        let near_constraints = SequentialTechMapConstraints {
            clock_period: Some(arrival - REGISTER_SETUP_SLACK_EPSILON / 2.0),
            ..SequentialTechMapConstraints::default()
        };
        finalize_sequential_mapping(
            &mut mapped,
            &library,
            &near_constraints,
            StaOptions::default(),
        )
        .expect("rounding-sized negative setup slack should remain within tolerance");
        assert!(
            mapped
                .stats
                .worst_register_slack
                .is_some_and(|slack| { slack < 0.0 && slack >= -REGISTER_SETUP_SLACK_EPSILON })
        );

        let violating_constraints = SequentialTechMapConstraints {
            clock_period: Some(arrival - 2.0 * REGISTER_SETUP_SLACK_EPSILON),
            ..SequentialTechMapConstraints::default()
        };
        let error = finalize_sequential_mapping(
            &mut mapped,
            &library,
            &violating_constraints,
            StaOptions::default(),
        )
        .expect_err("exact negative setup slack must reject the current flip-flop candidate");
        let diagnostic = error.to_string();
        assert!(
            diagnostic.contains("clock period")
                && diagnostic.contains("exact register setup")
                && diagnostic.contains("setup slack"),
            "unexpected exact setup violation diagnostic: {diagnostic}"
        );
    }

    #[test]
    fn checks_exact_output_required_times_for_register_and_input_launches() {
        let library = test_library();
        let register_design = test_design();
        let register_choices = ChoiceAig::without_choices(register_design.transition.clone());
        let register_mapped = map_sequential_choice_aig_to_netlist(
            &register_design,
            &register_choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map register-launched visible output");
        let register_arrival = register_mapped
            .stats
            .worst_register_to_output_arrival
            .expect("register output must have clock-to-Q timing");
        let register_constraints = SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([(
                "out_0".to_string(),
                register_arrival - 0.01,
            )]),
            ..SequentialTechMapConstraints::default()
        };
        let register_error = validate_exact_output_requirements(
            &register_mapped,
            &library,
            &register_constraints,
            StaOptions::default(),
        )
        .expect_err("exact register-to-output timing must satisfy its bit deadline");
        assert!(
            register_error.to_string().contains("register launch"),
            "unexpected register output deadline diagnostic: {register_error:#}"
        );

        let mut input_design = test_design();
        input_design.transition.outputs[0].bit_vector =
            input_design.transition.inputs[0].bit_vector.clone();
        let input_choices = ChoiceAig::without_choices(input_design.transition.clone());
        let input_mapped = map_sequential_choice_aig_to_netlist(
            &input_design,
            &input_choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map primary-input-launched visible output");
        let primary_arrivals =
            BTreeMap::from([("data_0".to_string(), 5.0), ("data_1".to_string(), 5.0)]);
        let primary_launch = analyze_register_boundary_max_arrival_with_primary_input_arrivals(
            &input_mapped.module,
            &input_mapped.nets,
            &input_mapped.interner,
            &library,
            StaOptions::default(),
            true,
            &[],
            &primary_arrivals,
        )
        .expect("analyze externally launched packed output");
        let output_timing = primary_launch
            .timing_for_output_bit("out_0")
            .expect("resolve packed output bit timing");
        let input_arrival = output_timing.rise.arrival.max(output_timing.fall.arrival);
        let input_constraints = SequentialTechMapConstraints {
            primary_input_arrivals: primary_arrivals,
            primary_output_required: BTreeMap::from([("out_0".to_string(), input_arrival - 0.01)]),
            ..SequentialTechMapConstraints::default()
        };
        let input_error = validate_exact_output_requirements(
            &input_mapped,
            &library,
            &input_constraints,
            StaOptions::default(),
        )
        .expect_err("exact primary-input-to-output timing must include source arrival");
        assert!(
            input_error.to_string().contains("primary-input launch"),
            "unexpected primary-input output deadline diagnostic: {input_error:#}"
        );
    }

    #[test]
    fn reports_renamed_or_missing_flattened_transition_bits() {
        let design = test_design();
        let mut renamed = design.transition.clone();
        renamed.inputs[0].name = "wrong".to_string();
        let error = validate_transition_interface(&design.transition, &renamed)
            .expect_err("optimized transition must preserve input names");
        assert_eq!(
            error.to_string(),
            "optimized transition input bit 0 is 'wrong_0'; expected 'data_0'"
        );

        let mut missing = design.transition.clone();
        missing.outputs.pop();
        let error = validate_transition_interface(&design.transition, &missing)
            .expect_err("optimized transition must preserve output bit count");
        assert_eq!(
            error.to_string(),
            "optimized transition output interface has 2 bits; original transition has 4"
        );
    }

    #[test]
    fn accepts_register_aware_post_mapping_buffering() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &test_library(),
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                buffer_options: Some(BufferOptions::default()),
                ..TechMapOptions::default()
            },
        )
        .expect("register-aware buffering must run after physical FF restoration");

        assert_eq!(mapped.stats.sequential_instance_count, 2);
        assert!(mapped.stats.buffer_stats.is_some());
        assert!(mapped.stats.resize_stats.is_none());
    }

    #[test]
    fn accepts_register_aware_post_mapping_resizing() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &test_library(),
            &SequentialTechMapConstraints::default(),
            &TechMapOptions {
                resize_options: Some(ResizeOptions::default()),
                ..TechMapOptions::default()
            },
        )
        .expect("register-aware sizing must run after physical FF restoration");

        assert_eq!(mapped.stats.sequential_instance_count, 2);
        assert!(mapped.stats.buffer_stats.is_none());
        assert!(mapped.stats.resize_stats.is_some());
    }

    #[test]
    fn revisits_resized_register_fanout_when_exact_capture_timing_improves() {
        let library = registered_timing_library();
        let (module, nets, interner) = parse_module(high_fanout_register_source());
        let mut mapped = MappedNetlist {
            module,
            nets,
            interner,
            stats: TechMapStats::default(),
        };
        let constraints = SequentialTechMapConstraints::default();
        finalize_sequential_mapping(&mut mapped, &library, &constraints, StaOptions::default())
            .expect("characterize the overloaded physical launch register");
        let initial_delay = mapped
            .stats
            .worst_register_to_register_arrival
            .expect("the synthetic pipeline has a capture path");
        let mut buffer_stats = BufferStats {
            max_fanout_after: 8,
            ..BufferStats::default()
        };
        let mut resize_stats = ResizeStats {
            upsizes: 1,
            ..ResizeStats::default()
        };
        let options = TechMapOptions {
            buffer_options: Some(BufferOptions {
                max_fanout: 12,
                ..BufferOptions::default()
            }),
            resize_options: Some(ResizeOptions::default()),
            ..TechMapOptions::default()
        };

        revisit_sequential_buffering_after_sizing(
            &mut mapped,
            &library,
            &constraints,
            &options,
            &BufferTimingConstraints::default(),
            &mut buffer_stats,
            &mut resize_stats,
        )
        .expect("rebuffer and resize a newly overloaded physical register output");

        assert!(buffer_stats.buffers_inserted > 0);
        assert!(mapped.stats.worst_register_to_register_arrival.unwrap() < initial_delay);
    }

    #[test]
    fn rejects_post_sizing_register_buffers_that_do_not_improve_capture_timing() {
        let library = test_library();
        let (module, nets, interner) = parse_module(high_fanout_register_source());
        let mut mapped = MappedNetlist {
            module,
            nets,
            interner,
            stats: TechMapStats::default(),
        };
        let constraints = SequentialTechMapConstraints::default();
        finalize_sequential_mapping(&mut mapped, &library, &constraints, StaOptions::default())
            .expect("characterize a load-independent physical launch register");
        let initial_delay = mapped.stats.worst_register_to_register_arrival;
        let initial_instances = mapped.module.instances.clone();
        let mut buffer_stats = BufferStats {
            max_fanout_after: 8,
            ..BufferStats::default()
        };
        let mut resize_stats = ResizeStats {
            upsizes: 1,
            ..ResizeStats::default()
        };
        let options = TechMapOptions {
            buffer_options: Some(BufferOptions {
                max_fanout: 12,
                ..BufferOptions::default()
            }),
            resize_options: Some(ResizeOptions::default()),
            ..TechMapOptions::default()
        };

        revisit_sequential_buffering_after_sizing(
            &mut mapped,
            &library,
            &constraints,
            &options,
            &BufferTimingConstraints::default(),
            &mut buffer_stats,
            &mut resize_stats,
        )
        .expect("discard speculative buffers that only add register-path delay");

        assert_eq!(buffer_stats.buffers_inserted, 0);
        assert_eq!(mapped.module.instances, initial_instances);
        assert_eq!(
            mapped.stats.worst_register_to_register_arrival,
            initial_delay
        );
    }

    #[test]
    fn rejects_dont_use_negative_edge_asynchronous_and_latch_flip_flops() {
        for case in ["dont_use", "negative_edge", "asynchronous", "latch"] {
            let mut library = test_library();
            let cell = library
                .cells
                .iter_mut()
                .find(|cell| cell.name == "DFF")
                .expect("find synthetic flip-flop");
            match case {
                "dont_use" => cell.dont_use = Some(true),
                "negative_edge" => cell.sequential[0].clock_expr = "!CLK".to_string(),
                "asynchronous" => cell.sequential[0].clear_expr = "!RST".to_string(),
                "latch" => cell.sequential[0].kind = SequentialKind::Latch as i32,
                _ => unreachable!("all synthetic rejection cases are enumerated"),
            }
            let error = index_flip_flops(&library, StaOptions::default())
                .expect_err("unsupported sequential cells must not enter the FF index");
            assert!(
                error.to_string().contains("no usable positive-edge"),
                "unexpected {case} diagnostic: {error:#}"
            );
        }
    }

    #[test]
    fn maps_packed_constant_outputs_and_constant_register_inputs() {
        let mut builder = GateBuilder::new(
            "constant_pipeline__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let _data = builder.add_input("data".to_string(), 2);
        let state = builder.add_input("state__q".to_string(), 2);
        let visible = AigBitVector::from_lsb_is_index_0(&[*state.get_lsb(0), builder.get_false()]);
        let next_state = builder
            .add_literal(&IrBits::make_ubits(2, 0b10).expect("construct constant register input"));
        builder.add_output("out".to_string(), visible);
        builder.add_output("state__d".to_string(), next_state);
        let design = SequentialGateFn::new(
            "constant_pipeline".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .expect("construct constant-output registered design");
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let library = test_library();

        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map packed constant outputs and directly tied register data");
        emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)
            .expect("packed constant output must be serializable as a standard-cell netlist");
        assert_eq!(mapped.stats.sequential_instance_count, 2);
        assert_eq!(
            mapped.stats.selected_instance_count,
            mapped.module.instances.len()
        );
        assert_eq!(mapped.module.assigns.len(), 1);
        let assignment = &mapped.module.assigns[0];
        assert_eq!(assignment.kind, NetlistAssignKind::Continuous);
        assert!(matches!(assignment.lhs, NetRef::BitSelect(_, 1)));
        assert!(matches!(
            &assignment.rhs,
            AssignExpr::Leaf(NetRef::Literal(bits)) if bits.is_zero()
        ));

        let projected = project_labeled_sequential_netlist_aig(
            &mapped.module,
            &mapped.nets,
            &mapped.interner,
            &library,
            Some("clk"),
        )
        .expect("project packed constant-output netlist");
        let inputs = [0, 1, 2]
            .into_iter()
            .map(|value| {
                vec![IrBits::make_ubits(2, value).expect("construct packed test stimulus")]
            })
            .collect::<Vec<_>>();
        let original = simulate(&design, &inputs, SequentialState::all_zeros(&design))
            .expect("simulate native constant-output transition");
        let actual = simulate(
            &projected.sequential_gate_fn,
            &inputs,
            SequentialState::all_zeros(&projected.sequential_gate_fn),
        )
        .expect("simulate physical constant-output netlist");
        assert_eq!(actual.external_outputs(), original.external_outputs());
    }

    #[test]
    fn maps_state_free_packed_constants_as_zero_area_assignments() {
        let mut builder = GateBuilder::new(
            "constant_no_state__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let _data = builder.add_input("data".to_string(), 1);
        let zero = builder.get_false();
        let one = builder.get_true();
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[zero, one, zero, one]),
        );
        let design = SequentialGateFn::new(
            "constant_no_state".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![],
        )
        .expect("construct state-free packed constant design");
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &test_library(),
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map cleaned packed constants without synthetic cells");

        assert_eq!(mapped.stats.selected_instance_count, 0);
        assert_eq!(mapped.stats.sequential_instance_count, 0);
        assert_eq!(mapped.stats.selected_area, 0.0);
        assert_eq!(mapped.module.assigns.len(), 4);
        let emitted = emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)
            .expect("emit zero-area packed constant outputs");
        assert_eq!(
            emitted,
            "module constant_no_state(data, clk, out);\n  input data;\n  input clk;\n  output [3:0] out;\n  assign out[0] = 1'b0;\n  assign out[1] = 1'b1;\n  assign out[2] = 1'b0;\n  assign out[3] = 1'b1;\nendmodule\n"
        );
    }

    #[test]
    fn avoids_collisions_when_naming_physical_register_instances() {
        let mut used = BTreeSet::from(["u_reg_state".to_string(), "u_reg_state__1".to_string()]);
        assert_eq!(
            unique_instance_name("u_reg_state", &mut used),
            "u_reg_state__2"
        );
        assert_eq!(
            unique_instance_name("u_reg_state", &mut used),
            "u_reg_state__3"
        );
    }
}
