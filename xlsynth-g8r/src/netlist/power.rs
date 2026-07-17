// SPDX-License-Identifier: Apache-2.0

//! Sample-driven dynamic-power estimation for labeled gate-level netlists.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;
use xlsynth::IrBits;

use crate::aig::{AigNode, AigOperand, GateFn};
use crate::aig_sim::gate_simd;
use crate::aig_sim::sequential::{
    SequentialSettledPhase, SequentialTrace, build_clocked_phase_inputs,
};
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty::lut::{
    RawLutQuery, RawLutQueryDiagnostics, evaluate_power_lut, evaluate_timing_lut_raw,
    power_lut_input_transition_range, timing_lut_input_transition_range,
};
use crate::liberty_model::{
    Cell, Library, Pin, PinDirection, PowerTransition, StringId, TimingTable,
};
use crate::liberty_proto::TimingTableKind;
use crate::netlist::gatefn_from_netlist::{
    LabeledNetlistAig, LabeledSequentialNetlistAig, PinConnection, SequentialAigSignal,
    SequentialClockEdge,
};
use crate::netlist::parse::PortDirection;

/// The fixed number of logarithmic slew buckets used by dynamic-power analysis.
pub const GV_POWER_SLEW_BUCKET_COUNT: usize = 32;

/// Numeric assumptions for sample-driven dynamic-power analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GvDynamicPowerOptions {
    pub primary_input_transition: f64,
    pub module_output_load: f64,
    pub cycle_time: Option<f64>,
}

impl Default for GvDynamicPowerOptions {
    fn default() -> Self {
        Self {
            primary_input_transition: 0.01,
            module_output_load: 0.0,
            cycle_time: None,
        }
    }
}

/// Numeric assumptions for sequential sample-driven dynamic-power analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GvSequentialDynamicPowerOptions {
    pub primary_input_transition: f64,
    pub clock_transition: f64,
    pub module_output_load: f64,
    pub cycle_time: Option<f64>,
}

impl Default for GvSequentialDynamicPowerOptions {
    fn default() -> Self {
        Self {
            primary_input_transition: 0.01,
            clock_transition: 0.01,
            module_output_load: 0.0,
            cycle_time: None,
        }
    }
}

/// One logarithmically placed slew bucket, evaluated at its arithmetic
/// midpoint.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvSlewBucket {
    pub lower: f64,
    pub upper: f64,
    pub midpoint: f64,
}

/// Weighted edge counts assigned to the global slew grid.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvSlewHistogram {
    pub rise: Vec<f64>,
    pub fall: Vec<f64>,
}

/// Dynamic-power totals for one output-driving signal.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvOutputPowerReport {
    pub pin_name: String,
    pub rise_count: usize,
    pub fall_count: usize,
    pub rise_load: f64,
    pub fall_load: f64,
    pub switching_energy: f64,
    pub slew_histogram: GvSlewHistogram,
}

/// Dynamic-power totals owned by one cell pin.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvPinInternalPowerReport {
    pub pin_name: String,
    pub internal_energy: f64,
}

/// Dynamic-power details for one standard-cell instance.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvInstancePowerReport {
    pub instance_name: String,
    pub cell_type: String,
    pub internal_energy: f64,
    pub switching_energy: f64,
    pub pin_internal_energy: Vec<GvPinInternalPowerReport>,
    pub outputs: Vec<GvOutputPowerReport>,
}

/// Counts approximations and boundary behavior encountered by power analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GvDynamicPowerDiagnostics {
    pub lut_below_min_clamp_count: usize,
    pub lut_above_max_clamp_count: usize,
    pub when_evaluation_count: usize,
    pub when_changed_during_transition_count: usize,
    pub multiply_attributed_output_transition_count: usize,
    pub unattributed_output_transition_count: usize,
    pub multiply_attributed_internal_event_count: usize,
}

/// Sample-driven dynamic energy, with optional average power when a cycle time
/// is known.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvDynamicPowerReport {
    pub module_name: String,
    pub sample_count: usize,
    pub transition_count: usize,
    pub nominal_voltage: f64,
    pub primary_input_transition: f64,
    pub module_output_load: f64,
    pub cycle_time: Option<f64>,
    pub time_unit: String,
    pub capacitance_unit: String,
    pub voltage_unit: String,
    pub energy_unit: String,
    pub power_unit: Option<String>,
    pub slew_buckets: Vec<GvSlewBucket>,
    pub primary_input_switching_energy: f64,
    pub cell_internal_energy: f64,
    pub cell_output_switching_energy: f64,
    pub total_dynamic_energy: f64,
    pub average_energy_per_transition: f64,
    pub average_dynamic_power: Option<f64>,
    pub instances: Vec<GvInstancePowerReport>,
    pub diagnostics: GvDynamicPowerDiagnostics,
}

/// Sequential dynamic energy over settled input, clock, and state phases.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GvSequentialDynamicPowerReport {
    pub module_name: String,
    pub clock_port_name: String,
    pub active_edge: Option<SequentialClockEdge>,
    pub cycle_count: usize,
    pub phase_transition_count: usize,
    pub input_settle_transition_count: usize,
    pub active_edge_transition_count: usize,
    pub inactive_edge_transition_count: usize,
    pub clock_transition_count: usize,
    pub nominal_voltage: f64,
    pub primary_input_transition: f64,
    pub clock_transition: f64,
    pub module_output_load: f64,
    pub cycle_time: Option<f64>,
    pub time_unit: String,
    pub capacitance_unit: String,
    pub voltage_unit: String,
    pub energy_unit: String,
    pub power_unit: Option<String>,
    pub slew_buckets: Vec<GvSlewBucket>,
    pub primary_input_switching_energy: f64,
    pub clock_switching_energy: f64,
    pub cell_internal_energy: f64,
    pub cell_output_switching_energy: f64,
    pub total_dynamic_energy: f64,
    pub average_energy_per_cycle: f64,
    pub average_dynamic_power: Option<f64>,
    pub instances: Vec<GvInstancePowerReport>,
    pub diagnostics: GvDynamicPowerDiagnostics,
}

#[derive(Clone, Copy, Debug, Default)]
struct EdgeValue<T> {
    rise: T,
    fall: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SignalId(usize);

#[derive(Clone, Copy, Debug)]
enum SignalOwner {
    ModuleInput,
    Clock,
    CellOutput { instance: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PowerSignal {
    Operand(AigOperand),
    Clock,
}

#[derive(Clone, Debug)]
struct Signal {
    source: PowerSignal,
    owner: SignalOwner,
    load: EdgeValue<f64>,
    edges: EdgeValue<usize>,
    histogram: EdgeValue<Vec<f64>>,
}

#[derive(Clone, Copy, Debug)]
enum ResolvedSource {
    Signal(SignalId),
    Literal,
}

#[derive(Clone, Debug)]
struct InstanceTopology {
    cell: usize,
    liberty_pins: Vec<usize>,
    pin_index_by_name: HashMap<String, usize>,
    pin_sources: Vec<ResolvedSource>,
    output_signals: Vec<Option<SignalId>>,
    is_sequential: bool,
    has_when_conditions: bool,
    state_aliases: Vec<StateAlias>,
}

#[derive(Debug)]
struct PowerTopology {
    signals: Vec<Signal>,
    instances: Vec<InstanceTopology>,
    instance_order: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TimingCauseKey {
    output: SignalId,
    source: SignalId,
    source_rise: bool,
    output_rise: bool,
    instance: usize,
    output_pin: usize,
    arc: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct InternalEventKey {
    instance: usize,
    owner_pin: usize,
    group: usize,
    source: SignalId,
    source_rise: bool,
    owner_output_rise: Option<bool>,
}

#[derive(Debug, Default)]
struct ObservedActivity {
    timing_causes: BTreeMap<TimingCauseKey, f64>,
    internal_events: BTreeMap<InternalEventKey, f64>,
    unattributed: BTreeMap<(SignalId, bool), usize>,
    diagnostics: GvDynamicPowerDiagnostics,
}

#[derive(Clone, Debug)]
struct SampleState {
    signal_values: Vec<bool>,
    pin_values: Vec<Vec<bool>>,
    state_alias_values: Vec<Vec<bool>>,
}

#[derive(Clone, Debug)]
struct PowerPortBit {
    bit_number: u32,
    signal: PowerSignal,
}

#[derive(Clone, Debug)]
struct PowerPort {
    name: String,
    direction: PortDirection,
    bits_lsb_to_msb: Vec<PowerPortBit>,
}

#[derive(Clone, Debug)]
struct PowerPin {
    pin_name: String,
    direction: PinDirection,
    signal: PowerSignal,
    connection: PinConnection,
}

#[derive(Clone, Debug)]
struct PowerInstance {
    instance_name: String,
    cell_type: String,
    pins: Vec<PowerPin>,
    is_sequential: bool,
    state_signal: Option<PowerSignal>,
}

#[derive(Clone, Debug)]
struct StateAlias {
    name: String,
    signal: PowerSignal,
    complemented: bool,
}

#[derive(Debug)]
struct PowerModelView<'a> {
    module_name: &'a str,
    gate_fn: &'a GateFn,
    module_ports: Vec<PowerPort>,
    instances: Vec<PowerInstance>,
}

#[derive(Clone, Copy, Debug)]
enum PowerTransitionKind {
    Combinational,
    InputSettle,
    ActiveClockEdge { clock_rise: bool },
    InactiveClockEdge,
}

fn combinational_power_view(model: &LabeledNetlistAig) -> PowerModelView<'_> {
    PowerModelView {
        module_name: &model.module_name,
        gate_fn: &model.gate_fn,
        module_ports: model
            .module_ports
            .iter()
            .map(|port| PowerPort {
                name: port.name.clone(),
                direction: port.direction.clone(),
                bits_lsb_to_msb: port
                    .bits_lsb_to_msb
                    .iter()
                    .map(|bit| PowerPortBit {
                        bit_number: bit.bit_number,
                        signal: PowerSignal::Operand(bit.operand),
                    })
                    .collect(),
            })
            .collect(),
        instances: model
            .instances
            .iter()
            .map(|instance| PowerInstance {
                instance_name: instance.instance_name.clone(),
                cell_type: instance.cell_type.clone(),
                pins: instance
                    .pins
                    .iter()
                    .map(|pin| PowerPin {
                        pin_name: pin.pin_name.clone(),
                        direction: pin.direction,
                        signal: PowerSignal::Operand(pin.operand),
                        connection: pin.connection.clone(),
                    })
                    .collect(),
                is_sequential: false,
                state_signal: None,
            })
            .collect(),
    }
}

fn sequential_power_signal(signal: SequentialAigSignal) -> PowerSignal {
    match signal {
        SequentialAigSignal::Operand(operand) => PowerSignal::Operand(operand),
        SequentialAigSignal::Clock => PowerSignal::Clock,
    }
}

fn sequential_power_view(
    model: &LabeledSequentialNetlistAig,
) -> Result<PowerModelView<'_>, String> {
    let instances = model
        .instances
        .iter()
        .map(|instance| {
            Ok(PowerInstance {
                instance_name: instance.instance_name.clone(),
                cell_type: instance.cell_type.clone(),
                pins: instance
                    .pins
                    .iter()
                    .map(|pin| PowerPin {
                        pin_name: pin.pin_name.clone(),
                        direction: pin.direction,
                        signal: sequential_power_signal(pin.signal),
                        connection: pin.connection.clone(),
                    })
                    .collect(),
                is_sequential: instance.state_register_index.is_some(),
                state_signal: sequential_state_signal(model, instance)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(PowerModelView {
        module_name: &model.module_name,
        gate_fn: &model.sequential_gate_fn.transition,
        module_ports: model
            .module_ports
            .iter()
            .map(|port| PowerPort {
                name: port.name.clone(),
                direction: port.direction.clone(),
                bits_lsb_to_msb: port
                    .bits_lsb_to_msb
                    .iter()
                    .map(|bit| PowerPortBit {
                        bit_number: bit.bit_number,
                        signal: sequential_power_signal(bit.signal),
                    })
                    .collect(),
            })
            .collect(),
        instances,
    })
}

/// Returns the mapped logical Q signal for one projected FF instance.
fn sequential_state_signal(
    model: &LabeledSequentialNetlistAig,
    instance: &crate::netlist::gatefn_from_netlist::SequentialInstanceAigBinding,
) -> Result<Option<PowerSignal>, String> {
    let Some(register_index) = instance.state_register_index else {
        return Ok(None);
    };
    let register = model
        .sequential_gate_fn
        .registers
        .get(register_index)
        .ok_or_else(|| {
            format!(
                "instance '{}' references missing state register {}",
                instance.instance_name, register_index
            )
        })?;
    let q_input = model
        .sequential_gate_fn
        .transition
        .inputs
        .get(register.q.index())
        .ok_or_else(|| {
            format!(
                "instance '{}' register Q input {} is out of range",
                instance.instance_name,
                register.q.index()
            )
        })?;
    if q_input.get_bit_count() != 1 {
        return Err(format!(
            "instance '{}' state register has width {}; expected scalar mapped FF state",
            instance.instance_name,
            q_input.get_bit_count()
        ));
    }
    Ok(Some(PowerSignal::Operand(*q_input.bit_vector.get_lsb(0))))
}

/// Estimates dynamic energy from consecutive ordered input samples.
pub fn analyze_dynamic_power_bits(
    model: &LabeledNetlistAig,
    library: &Library,
    batch_inputs: &[Vec<IrBits>],
    options: GvDynamicPowerOptions,
) -> Result<GvDynamicPowerReport, String> {
    validate_options(options)?;
    if batch_inputs.len() < 2 {
        return Err("dynamic-power analysis requires at least two ordered samples".to_string());
    }
    let nominal_voltage = nominal_voltage(library)?;

    let power_model = combinational_power_view(model);
    let mut topology = build_topology(&power_model, library, options.module_output_load)?;
    let mut activity = observe_activity(&power_model, library, batch_inputs, &mut topology)?;
    let buckets = build_slew_buckets(library, &[options.primary_input_transition])?;
    seed_root_histograms(
        &mut topology,
        &buckets,
        options.primary_input_transition,
        options.primary_input_transition,
    );
    let mut lut_diagnostics = RawLutQueryDiagnostics::default();
    propagate_slew_histograms(
        &power_model,
        library,
        &mut topology,
        &activity,
        &buckets,
        options.primary_input_transition,
        options.primary_input_transition,
        &mut lut_diagnostics,
    )?;
    let (instance_internal, pin_internal) = calculate_internal_energy(
        &power_model,
        library,
        &topology,
        &activity,
        &buckets,
        options.primary_input_transition,
        options.primary_input_transition,
        &mut lut_diagnostics,
    )?;
    activity.diagnostics.lut_below_min_clamp_count = lut_diagnostics.below_min_clamp_count;
    activity.diagnostics.lut_above_max_clamp_count = lut_diagnostics.above_max_clamp_count;

    let voltage_squared = nominal_voltage * nominal_voltage;
    let mut primary_input_switching_energy = 0.0;
    let mut cell_output_switching_energy = 0.0;
    let mut instance_switching = vec![0.0; power_model.instances.len()];
    for signal in &topology.signals {
        let energy = switching_energy(signal, voltage_squared);
        match signal.owner {
            SignalOwner::ModuleInput => primary_input_switching_energy += energy,
            SignalOwner::Clock => unreachable!("combinational power view has no clock signal"),
            SignalOwner::CellOutput { instance } => {
                cell_output_switching_energy += energy;
                instance_switching[instance] += energy;
            }
        }
    }
    let cell_internal_energy: f64 = instance_internal.iter().sum();
    let total_dynamic_energy =
        primary_input_switching_energy + cell_output_switching_energy + cell_internal_energy;
    let transition_count = batch_inputs.len() - 1;
    let average_energy_per_transition = total_dynamic_energy / transition_count as f64;
    let average_dynamic_power = options
        .cycle_time
        .map(|cycle_time| total_dynamic_energy / (transition_count as f64 * cycle_time));

    let instances = build_instance_reports(
        &power_model,
        library,
        &topology,
        &instance_internal,
        &pin_internal,
        &instance_switching,
        voltage_squared,
    );
    let (time_unit, capacitance_unit, voltage_unit, energy_unit, power_unit) =
        report_units(library, options.cycle_time);
    Ok(GvDynamicPowerReport {
        module_name: power_model.module_name.to_string(),
        sample_count: batch_inputs.len(),
        transition_count,
        nominal_voltage,
        primary_input_transition: options.primary_input_transition,
        module_output_load: options.module_output_load,
        cycle_time: options.cycle_time,
        time_unit,
        capacitance_unit,
        voltage_unit,
        energy_unit,
        power_unit,
        slew_buckets: buckets,
        primary_input_switching_energy,
        cell_internal_energy,
        cell_output_switching_energy,
        total_dynamic_energy,
        average_energy_per_transition,
        average_dynamic_power,
        instances,
        diagnostics: activity.diagnostics,
    })
}

/// Estimates sequential dynamic energy from one settled trace.
///
/// The trace is expanded into inactive-input, active-edge, and inactive-edge
/// phases. Functional output sampling remains the responsibility of the
/// sequential evaluator; this routine only uses the extra phases for no-glitch
/// power accounting.
pub fn analyze_sequential_dynamic_power(
    model: &LabeledSequentialNetlistAig,
    library: &Library,
    trace: &SequentialTrace,
    options: GvSequentialDynamicPowerOptions,
) -> Result<GvSequentialDynamicPowerReport, String> {
    validate_sequential_options(options)?;
    let nominal_voltage = nominal_voltage(library)?;
    let clock = model
        .clock
        .as_ref()
        .ok_or_else(|| "sequential dynamic-power analysis requires a selected clock".to_string())?;
    let active_clock_level = match clock.active_edge {
        Some(SequentialClockEdge::Rising) => true,
        Some(SequentialClockEdge::Falling) => false,
        // A clock hint can preserve a clock after every FF was optimized
        // away. No sequential arc remains whose polarity could depend on it.
        None => true,
    };
    let phase_inputs = build_clocked_phase_inputs(&model.sequential_gate_fn, trace)?;
    let batch_inputs = phase_inputs
        .iter()
        .map(|phase| phase.transition_inputs.clone())
        .collect::<Vec<_>>();
    let clock_values = phase_inputs
        .iter()
        .map(|phase| {
            Some(if phase.phase == SequentialSettledPhase::PostActiveEdge {
                active_clock_level
            } else {
                !active_clock_level
            })
        })
        .collect::<Vec<_>>();
    let transition_kinds = phase_inputs
        .windows(2)
        .map(|pair| match pair[1].phase {
            SequentialSettledPhase::PreEdge => PowerTransitionKind::InputSettle,
            SequentialSettledPhase::PostActiveEdge => PowerTransitionKind::ActiveClockEdge {
                clock_rise: active_clock_level,
            },
            SequentialSettledPhase::PostInactiveEdge => PowerTransitionKind::InactiveClockEdge,
        })
        .collect::<Vec<_>>();

    let power_model = sequential_power_view(model)?;
    let mut topology = build_topology(&power_model, library, options.module_output_load)?;
    let mut activity = observe_ordered_activity(
        &power_model,
        library,
        &batch_inputs,
        &clock_values,
        &transition_kinds,
        &mut topology,
    )?;
    let buckets = build_slew_buckets(
        library,
        &[options.primary_input_transition, options.clock_transition],
    )?;
    seed_root_histograms(
        &mut topology,
        &buckets,
        options.primary_input_transition,
        options.clock_transition,
    );
    let mut lut_diagnostics = RawLutQueryDiagnostics::default();
    propagate_slew_histograms(
        &power_model,
        library,
        &mut topology,
        &activity,
        &buckets,
        options.primary_input_transition,
        options.clock_transition,
        &mut lut_diagnostics,
    )?;
    let (instance_internal, pin_internal) = calculate_internal_energy(
        &power_model,
        library,
        &topology,
        &activity,
        &buckets,
        options.primary_input_transition,
        options.clock_transition,
        &mut lut_diagnostics,
    )?;
    activity.diagnostics.lut_below_min_clamp_count = lut_diagnostics.below_min_clamp_count;
    activity.diagnostics.lut_above_max_clamp_count = lut_diagnostics.above_max_clamp_count;

    let voltage_squared = nominal_voltage * nominal_voltage;
    let mut primary_input_switching_energy = 0.0;
    let mut clock_switching_energy = 0.0;
    let mut cell_output_switching_energy = 0.0;
    let mut instance_switching = vec![0.0; power_model.instances.len()];
    for signal in &topology.signals {
        let energy = switching_energy(signal, voltage_squared);
        match signal.owner {
            SignalOwner::ModuleInput => primary_input_switching_energy += energy,
            SignalOwner::Clock => clock_switching_energy += energy,
            SignalOwner::CellOutput { instance } => {
                cell_output_switching_energy += energy;
                instance_switching[instance] += energy;
            }
        }
    }
    let cell_internal_energy: f64 = instance_internal.iter().sum();
    let total_dynamic_energy = primary_input_switching_energy
        + clock_switching_energy
        + cell_output_switching_energy
        + cell_internal_energy;
    let cycle_count = phase_inputs.len() / 3;
    let phase_transition_count = phase_inputs.len() - 1;
    let instances = build_instance_reports(
        &power_model,
        library,
        &topology,
        &instance_internal,
        &pin_internal,
        &instance_switching,
        voltage_squared,
    );
    let (time_unit, capacitance_unit, voltage_unit, energy_unit, power_unit) =
        report_units(library, options.cycle_time);
    Ok(GvSequentialDynamicPowerReport {
        module_name: power_model.module_name.to_string(),
        clock_port_name: clock.port_name.clone(),
        active_edge: clock.active_edge,
        cycle_count,
        phase_transition_count,
        input_settle_transition_count: cycle_count - 1,
        active_edge_transition_count: cycle_count,
        inactive_edge_transition_count: cycle_count,
        clock_transition_count: cycle_count.saturating_mul(2),
        nominal_voltage,
        primary_input_transition: options.primary_input_transition,
        clock_transition: options.clock_transition,
        module_output_load: options.module_output_load,
        cycle_time: options.cycle_time,
        time_unit,
        capacitance_unit,
        voltage_unit,
        energy_unit,
        power_unit,
        slew_buckets: buckets,
        primary_input_switching_energy,
        clock_switching_energy,
        cell_internal_energy,
        cell_output_switching_energy,
        total_dynamic_energy,
        average_energy_per_cycle: total_dynamic_energy / cycle_count as f64,
        average_dynamic_power: options
            .cycle_time
            .map(|cycle_time| total_dynamic_energy / (cycle_count as f64 * cycle_time)),
        instances,
        diagnostics: activity.diagnostics,
    })
}

fn validate_options(options: GvDynamicPowerOptions) -> Result<(), String> {
    validate_common_options(
        options.primary_input_transition,
        options.module_output_load,
        options.cycle_time,
    )
}

fn validate_sequential_options(options: GvSequentialDynamicPowerOptions) -> Result<(), String> {
    validate_common_options(
        options.primary_input_transition,
        options.module_output_load,
        options.cycle_time,
    )?;
    if !options.clock_transition.is_finite() || options.clock_transition <= 0.0 {
        return Err(format!(
            "clock_transition must be finite and positive; got {}",
            options.clock_transition
        ));
    }
    Ok(())
}

fn validate_common_options(
    primary_input_transition: f64,
    module_output_load: f64,
    cycle_time: Option<f64>,
) -> Result<(), String> {
    if !primary_input_transition.is_finite() || primary_input_transition <= 0.0 {
        return Err(format!(
            "primary_input_transition must be finite and positive; got {}",
            primary_input_transition
        ));
    }
    if !module_output_load.is_finite() || module_output_load < 0.0 {
        return Err(format!(
            "module_output_load must be finite and non-negative; got {}",
            module_output_load
        ));
    }
    if let Some(cycle_time) = cycle_time
        && (!cycle_time.is_finite() || cycle_time <= 0.0)
    {
        return Err(format!(
            "cycle_time must be finite and positive; got {cycle_time}"
        ));
    }
    Ok(())
}

fn nominal_voltage(library: &Library) -> Result<f64, String> {
    let nominal_voltage = library.nominal_voltage.ok_or_else(|| {
        "Liberty proto has no nominal_voltage; switching energy requires a voltage".to_string()
    })?;
    if !nominal_voltage.is_finite() || nominal_voltage <= 0.0 {
        return Err(format!(
            "Liberty nominal_voltage must be finite and positive; got {nominal_voltage}"
        ));
    }
    Ok(nominal_voltage)
}

fn report_units(
    library: &Library,
    cycle_time: Option<f64>,
) -> (String, String, String, String, Option<String>) {
    let units = library.units.as_ref();
    let time_unit = units
        .map(|value| value.time_unit.clone())
        .unwrap_or_default();
    let capacitance_unit = units
        .map(|value| value.capacitance_unit.clone())
        .unwrap_or_default();
    let voltage_unit = units
        .map(|value| value.voltage_unit.clone())
        .unwrap_or_default();
    let energy_unit = format!("{}^2*{}", voltage_unit, capacitance_unit);
    let power_unit = cycle_time.map(|_| format!("{energy_unit}/{time_unit}"));
    (
        time_unit,
        capacitance_unit,
        voltage_unit,
        energy_unit,
        power_unit,
    )
}

fn build_instance_reports(
    model: &PowerModelView<'_>,
    library: &Library,
    topology: &PowerTopology,
    instance_internal: &[f64],
    pin_internal: &[Vec<f64>],
    instance_switching: &[f64],
    voltage_squared: f64,
) -> Vec<GvInstancePowerReport> {
    model
        .instances
        .iter()
        .enumerate()
        .map(|(instance_index, binding)| {
            let topology_instance = &topology.instances[instance_index];
            let cell = &library.cells[topology_instance.cell];
            let pin_internal_energy = binding
                .pins
                .iter()
                .enumerate()
                .filter_map(|(pin_index, pin)| {
                    let energy = pin_internal[instance_index][pin_index];
                    (energy != 0.0).then(|| GvPinInternalPowerReport {
                        pin_name: pin.pin_name.clone(),
                        internal_energy: energy,
                    })
                })
                .collect();
            let outputs = binding
                .pins
                .iter()
                .enumerate()
                .filter_map(|(pin_index, pin)| {
                    let signal_id = topology_instance.output_signals[pin_index]?;
                    let signal = &topology.signals[signal_id.0];
                    Some(GvOutputPowerReport {
                        pin_name: pin.pin_name.clone(),
                        rise_count: signal.edges.rise,
                        fall_count: signal.edges.fall,
                        rise_load: signal.load.rise,
                        fall_load: signal.load.fall,
                        switching_energy: switching_energy(signal, voltage_squared),
                        slew_histogram: GvSlewHistogram {
                            rise: signal.histogram.rise.clone(),
                            fall: signal.histogram.fall.clone(),
                        },
                    })
                })
                .collect();
            debug_assert_eq!(cell.name, binding.cell_type);
            GvInstancePowerReport {
                instance_name: binding.instance_name.clone(),
                cell_type: binding.cell_type.clone(),
                internal_energy: instance_internal[instance_index],
                switching_energy: instance_switching[instance_index],
                pin_internal_energy,
                outputs,
            }
        })
        .collect()
}

fn build_topology(
    model: &PowerModelView<'_>,
    library: &Library,
    module_output_load: f64,
) -> Result<PowerTopology, String> {
    let cell_by_name: HashMap<_, _> = library
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.name.as_str(), index))
        .collect();
    let mut signals = Vec::new();
    let mut net_drivers = HashMap::new();
    let mut operand_sources: HashMap<AigOperand, Vec<SignalId>> = HashMap::new();
    let mut clock_signal = None;
    for port in &model.module_ports {
        if port.direction != PortDirection::Input {
            continue;
        }
        for bit in &port.bits_lsb_to_msb {
            let signal_id = add_signal(
                &mut signals,
                &mut operand_sources,
                &mut clock_signal,
                bit.signal,
                if matches!(bit.signal, PowerSignal::Clock) {
                    SignalOwner::Clock
                } else {
                    SignalOwner::ModuleInput
                },
            );
            insert_net_driver(
                &mut net_drivers,
                (port.name.clone(), bit.bit_number),
                signal_id,
            )?;
        }
    }

    let mut instances = Vec::with_capacity(model.instances.len());
    for (instance_index, instance) in model.instances.iter().enumerate() {
        let cell = *cell_by_name
            .get(instance.cell_type.as_str())
            .ok_or_else(|| {
                format!(
                    "instance '{}' references missing Liberty cell '{}'",
                    instance.instance_name, instance.cell_type
                )
            })?;
        let pin_by_name: HashMap<_, _> = library.cells[cell]
            .pins
            .iter()
            .enumerate()
            .map(|(index, pin)| (library.resolve_string(&pin.name), index))
            .collect();
        let mut output_signals = vec![None; instance.pins.len()];
        let mut liberty_pins = Vec::with_capacity(instance.pins.len());
        for (binding_pin_index, pin) in instance.pins.iter().enumerate() {
            let liberty_pin_index = *pin_by_name.get(pin.pin_name.as_str()).ok_or_else(|| {
                format!(
                    "instance '{}' pin '{}' is absent from Liberty cell '{}'",
                    instance.instance_name, pin.pin_name, instance.cell_type
                )
            })?;
            liberty_pins.push(liberty_pin_index);
            if pin.direction == PinDirection::Output {
                let signal_id = add_signal(
                    &mut signals,
                    &mut operand_sources,
                    &mut clock_signal,
                    pin.signal,
                    SignalOwner::CellOutput {
                        instance: instance_index,
                    },
                );
                output_signals[binding_pin_index] = Some(signal_id);
                if let PinConnection::Net {
                    net_name,
                    bit_number,
                } = &pin.connection
                {
                    insert_net_driver(
                        &mut net_drivers,
                        (net_name.clone(), *bit_number),
                        signal_id,
                    )?;
                }
            }
        }
        let pin_index_by_name = instance
            .pins
            .iter()
            .enumerate()
            .map(|(index, pin)| (pin.pin_name.clone(), index))
            .collect();
        let has_when_conditions = liberty_pins.iter().any(|liberty_pin_index| {
            let pin = &library.cells[cell].pins[*liberty_pin_index];
            pin.timing_arcs
                .iter()
                .any(|arc| !library.resolve_string(&arc.when).is_empty())
                || pin
                    .internal_power
                    .iter()
                    .any(|group| !library.resolve_string(&group.when).is_empty())
        });
        let state_aliases = state_aliases_for_instance(instance, &library.cells[cell])?;
        instances.push(InstanceTopology {
            cell,
            liberty_pins,
            pin_index_by_name,
            pin_sources: vec![ResolvedSource::Literal; instance.pins.len()],
            output_signals,
            is_sequential: instance.is_sequential,
            has_when_conditions,
            state_aliases,
        });
    }

    for (instance_index, instance) in model.instances.iter().enumerate() {
        for (pin_index, pin) in instance.pins.iter().enumerate() {
            if let Some(signal_id) = instances[instance_index].output_signals[pin_index] {
                instances[instance_index].pin_sources[pin_index] =
                    ResolvedSource::Signal(signal_id);
                continue;
            }
            let source = resolve_source(
                model,
                &pin.connection,
                pin.signal,
                &net_drivers,
                &operand_sources,
                clock_signal,
                &format!(
                    "instance '{}' pin '{}'",
                    instance.instance_name, pin.pin_name
                ),
            )?;
            instances[instance_index].pin_sources[pin_index] = source;
            if pin.direction == PinDirection::Input
                && let ResolvedSource::Signal(signal_id) = source
            {
                let liberty_pin = &library.cells[instances[instance_index].cell].pins
                    [instances[instance_index].liberty_pins[pin_index]];
                let capacitance = effective_input_capacitance(liberty_pin, &pin.pin_name)?;
                signals[signal_id.0].load.rise += capacitance.rise;
                signals[signal_id.0].load.fall += capacitance.fall;
            }
        }
    }
    for port in &model.module_ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        for bit in &port.bits_lsb_to_msb {
            let connection = PinConnection::Net {
                net_name: port.name.clone(),
                bit_number: bit.bit_number,
            };
            if let ResolvedSource::Signal(signal_id) = resolve_source(
                model,
                &connection,
                bit.signal,
                &net_drivers,
                &operand_sources,
                clock_signal,
                &format!("module output '{}[{}]'", port.name, bit.bit_number),
            )? {
                signals[signal_id.0].load.rise += module_output_load;
                signals[signal_id.0].load.fall += module_output_load;
            }
        }
    }
    let instance_order = topological_instance_order(&instances, &signals)?;
    Ok(PowerTopology {
        signals,
        instances,
        instance_order,
    })
}

/// Builds Liberty state-variable aliases backed by the mapped register Q.
fn state_aliases_for_instance(
    instance: &PowerInstance,
    cell: &Cell,
) -> Result<Vec<StateAlias>, String> {
    let Some(signal) = instance.state_signal else {
        return Ok(Vec::new());
    };
    let [sequential] = cell.sequential.as_slice() else {
        return Err(format!(
            "instance '{}' has mapped state but Liberty cell '{}' does not have exactly one sequential entry",
            instance.instance_name, instance.cell_type
        ));
    };
    if sequential.state_var.is_empty() {
        return Err(format!(
            "instance '{}' has mapped state but Liberty cell '{}' has no state_var",
            instance.instance_name, instance.cell_type
        ));
    }
    let mut aliases = vec![StateAlias {
        name: sequential.state_var.clone(),
        signal,
        complemented: false,
    }];
    if let Some(name) = sequential
        .complementary_state_var
        .as_ref()
        .filter(|name| !name.is_empty())
    {
        aliases.push(StateAlias {
            name: name.clone(),
            signal,
            complemented: true,
        });
    }
    Ok(aliases)
}

fn add_signal(
    signals: &mut Vec<Signal>,
    operand_sources: &mut HashMap<AigOperand, Vec<SignalId>>,
    clock_signal: &mut Option<SignalId>,
    source: PowerSignal,
    owner: SignalOwner,
) -> SignalId {
    if matches!(source, PowerSignal::Clock)
        && let Some(signal_id) = clock_signal
    {
        return *signal_id;
    }
    let signal_id = SignalId(signals.len());
    signals.push(Signal {
        source,
        owner,
        load: EdgeValue::default(),
        edges: EdgeValue::default(),
        histogram: EdgeValue::default(),
    });
    match source {
        PowerSignal::Operand(operand) => {
            operand_sources.entry(operand).or_default().push(signal_id);
        }
        PowerSignal::Clock => *clock_signal = Some(signal_id),
    }
    signal_id
}

fn insert_net_driver(
    net_drivers: &mut HashMap<(String, u32), SignalId>,
    key: (String, u32),
    signal_id: SignalId,
) -> Result<(), String> {
    if let Some(previous) = net_drivers.insert(key.clone(), signal_id) {
        return Err(format!(
            "net '{}[{}]' has multiple dynamic-power drivers ({previous:?} and {signal_id:?})",
            key.0, key.1
        ));
    }
    Ok(())
}

fn resolve_source(
    model: &PowerModelView<'_>,
    connection: &PinConnection,
    signal: PowerSignal,
    net_drivers: &HashMap<(String, u32), SignalId>,
    operand_sources: &HashMap<AigOperand, Vec<SignalId>>,
    clock_signal: Option<SignalId>,
    context: &str,
) -> Result<ResolvedSource, String> {
    if matches!(
        connection,
        PinConnection::Literal(_) | PinConnection::Unconnected
    ) {
        return Ok(ResolvedSource::Literal);
    }
    if let PinConnection::Net {
        net_name,
        bit_number,
    } = connection
        && let Some(signal_id) = net_drivers.get(&(net_name.clone(), *bit_number))
    {
        return Ok(ResolvedSource::Signal(*signal_id));
    }
    match signal {
        PowerSignal::Clock => clock_signal
            .map(ResolvedSource::Signal)
            .ok_or_else(|| format!("{context}: selected clock has no physical driver")),
        PowerSignal::Operand(operand) => {
            match operand_sources
                .get(&operand)
                .map(Vec::as_slice)
                .unwrap_or(&[])
            {
                [signal_id] => Ok(ResolvedSource::Signal(*signal_id)),
                [] => match model.gate_fn.get(operand.node) {
                    AigNode::Literal { .. } => Ok(ResolvedSource::Literal),
                    _ => Err(format!(
                        "{context}: no physical or AIG-equivalent signal driver was found"
                    )),
                },
                candidates => Err(format!(
                    "{context}: assign-alias fallback is ambiguous among {} equivalent drivers",
                    candidates.len()
                )),
            }
        }
    }
}

fn effective_input_capacitance(pin: &Pin, context: &str) -> Result<EdgeValue<f64>, String> {
    let result = EdgeValue {
        rise: pin
            .rise_capacitance
            .or(pin.capacitance)
            .or(pin.fall_capacitance)
            .unwrap_or(0.0),
        fall: pin
            .fall_capacitance
            .or(pin.capacitance)
            .or(pin.rise_capacitance)
            .unwrap_or(0.0),
    };
    for (name, value) in [("rise", result.rise), ("fall", result.fall)] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "Liberty pin '{context}' has invalid {name} capacitance {value}"
            ));
        }
    }
    Ok(result)
}

fn topological_instance_order(
    instances: &[InstanceTopology],
    signals: &[Signal],
) -> Result<Vec<usize>, String> {
    let mut outgoing = vec![BTreeSet::new(); instances.len()];
    let mut indegree = vec![0usize; instances.len()];
    for (consumer, instance) in instances.iter().enumerate() {
        // FF outputs are timing roots driven by the selected clock, so their
        // D-side fanin must not create a combinational dependency cycle.
        let dependencies: BTreeSet<_> = if instance.is_sequential {
            BTreeSet::new()
        } else {
            instance
                .pin_sources
                .iter()
                .filter_map(|source| match source {
                    ResolvedSource::Signal(signal_id) => match signals[signal_id.0].owner {
                        SignalOwner::CellOutput { instance } if instance != consumer => {
                            Some(instance)
                        }
                        _ => None,
                    },
                    ResolvedSource::Literal => None,
                })
                .collect()
        };
        indegree[consumer] = dependencies.len();
        for dependency in dependencies {
            outgoing[dependency].insert(consumer);
        }
    }
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut result = Vec::with_capacity(instances.len());
    while let Some(instance) = ready.pop_first() {
        result.push(instance);
        for successor in &outgoing[instance] {
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                ready.insert(*successor);
            }
        }
    }
    if result.len() != instances.len() {
        return Err("cell connectivity contains a combinational cycle".to_string());
    }
    Ok(result)
}

fn observe_activity(
    model: &PowerModelView<'_>,
    library: &Library,
    batch_inputs: &[Vec<IrBits>],
    topology: &mut PowerTopology,
) -> Result<ObservedActivity, String> {
    let clock_values = vec![None; batch_inputs.len()];
    let transition_kinds = vec![PowerTransitionKind::Combinational; batch_inputs.len() - 1];
    observe_ordered_activity(
        model,
        library,
        batch_inputs,
        &clock_values,
        &transition_kinds,
        topology,
    )
}

fn observe_ordered_activity(
    model: &PowerModelView<'_>,
    library: &Library,
    batch_inputs: &[Vec<IrBits>],
    clock_values: &[Option<bool>],
    transition_kinds: &[PowerTransitionKind],
    topology: &mut PowerTopology,
) -> Result<ObservedActivity, String> {
    if clock_values.len() != batch_inputs.len() {
        return Err(format!(
            "power sample clock-value count {} does not match sample count {}",
            clock_values.len(),
            batch_inputs.len()
        ));
    }
    if transition_kinds.len() != batch_inputs.len().saturating_sub(1) {
        return Err(format!(
            "power transition-kind count {} does not match sample count {}",
            transition_kinds.len(),
            batch_inputs.len()
        ));
    }
    let mut result = ObservedActivity::default();
    let mut condition_cache = HashMap::new();
    let mut previous = None;
    let live_nodes = vec![true; model.gate_fn.gates.len()];
    let mut nodes = Vec::new();
    for chunk_start in (0..batch_inputs.len()).step_by(256) {
        let chunk_len = (batch_inputs.len() - chunk_start).min(256);
        let packed = gate_simd::pack_ordered_input_chunk(
            &model.gate_fn,
            batch_inputs,
            chunk_start,
            chunk_len,
        );
        gate_simd::eval_live_node_values_dense_into(
            &model.gate_fn,
            &packed,
            &live_nodes,
            &mut nodes,
        );
        for lane in 0..chunk_len {
            let sample_index = chunk_start + lane;
            let current = sample_state(model, topology, &nodes, lane, clock_values[sample_index])?;
            if let Some(previous) = previous.as_ref() {
                observe_transition(
                    model,
                    library,
                    topology,
                    previous,
                    &current,
                    transition_kinds[sample_index - 1],
                    &mut condition_cache,
                    &mut result,
                )?;
            }
            previous = Some(current);
        }
    }
    Ok(result)
}

fn sample_state(
    model: &PowerModelView<'_>,
    topology: &PowerTopology,
    nodes: &[gate_simd::Vec256],
    lane: usize,
    clock_value: Option<bool>,
) -> Result<SampleState, String> {
    let operand_value = |operand: AigOperand| {
        nodes
            .get(operand.node.id)
            .copied()
            .map(|value| value.apply_neg(operand.negated).get_lane(lane))
            .ok_or_else(|| format!("AIG node {} is out of range", operand.node.id))
    };
    let signal_value = |signal: PowerSignal| match signal {
        PowerSignal::Operand(operand) => operand_value(operand),
        PowerSignal::Clock => {
            clock_value.ok_or_else(|| "power sample omitted selected clock value".to_string())
        }
    };
    let signal_values = topology
        .signals
        .iter()
        .map(|signal| signal_value(signal.source))
        .collect::<Result<Vec<_>, _>>()?;
    let pin_values = model
        .instances
        .iter()
        .map(|instance| {
            instance
                .pins
                .iter()
                .map(|pin| signal_value(pin.signal))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state_alias_values = topology
        .instances
        .iter()
        .map(|instance| {
            instance
                .state_aliases
                .iter()
                .map(|alias| {
                    signal_value(alias.signal)
                        .map(|value| if alias.complemented { !value } else { value })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SampleState {
        signal_values,
        pin_values,
        state_alias_values,
    })
}

#[allow(clippy::too_many_arguments)]
fn observe_transition(
    model: &PowerModelView<'_>,
    library: &Library,
    topology: &mut PowerTopology,
    previous: &SampleState,
    current: &SampleState,
    transition_kind: PowerTransitionKind,
    condition_cache: &mut HashMap<String, Term>,
    result: &mut ObservedActivity,
) -> Result<(), String> {
    for (signal_index, (before, after)) in previous
        .signal_values
        .iter()
        .zip(&current.signal_values)
        .enumerate()
    {
        if before != after {
            let edge = &mut topology.signals[signal_index].edges;
            if *after {
                edge.rise += 1;
            } else {
                edge.fall += 1;
            }
        }
    }
    for (instance_index, instance) in model.instances.iter().enumerate() {
        let topology_instance = &topology.instances[instance_index];
        let previous_values = &previous.pin_values[instance_index];
        let current_values = &current.pin_values[instance_index];
        if previous_values == current_values {
            continue;
        }
        let cell = &library.cells[topology_instance.cell];
        let previous_state_alias_values = &previous.state_alias_values[instance_index];
        let current_state_alias_values = &current.state_alias_values[instance_index];
        let previous_map = topology_instance.has_when_conditions.then(|| {
            pin_value_map(
                instance,
                topology_instance,
                previous_values,
                previous_state_alias_values,
            )
        });
        let current_map = topology_instance.has_when_conditions.then(|| {
            pin_value_map(
                instance,
                topology_instance,
                current_values,
                current_state_alias_values,
            )
        });
        for (output_pin_index, output_signal) in
            topology_instance.output_signals.iter().copied().enumerate()
        {
            let Some(output_signal) = output_signal else {
                continue;
            };
            let output_rise = current_values[output_pin_index];
            if previous_values[output_pin_index] == output_rise {
                continue;
            }
            let output_pin = &cell.pins[topology_instance.liberty_pins[output_pin_index]];
            let mut candidates = Vec::new();
            for (arc_index, arc) in output_pin.timing_arcs.iter().enumerate() {
                let timing_type = arc.timing_type_str(library);
                let is_sequential_edge_arc =
                    instance.is_sequential && is_sequential_edge_timing_type(timing_type);
                if !timing_type_allows_output(
                    timing_type,
                    output_rise,
                    transition_kind,
                    instance.is_sequential,
                ) || transition_table(arc.tables.as_slice(), output_rise).is_none()
                {
                    continue;
                }
                let related_name = library.resolve_string(&arc.related_pin);
                let Some(&related_pin_index) =
                    topology_instance.pin_index_by_name.get(related_name)
                else {
                    return Err(format!(
                        "instance '{}' output '{}' timing arc names unknown related pin '{}'",
                        instance.instance_name,
                        instance.pins[output_pin_index].pin_name,
                        related_name
                    ));
                };
                let source_rise = current_values[related_pin_index];
                if previous_values[related_pin_index] == source_rise
                    || (!is_sequential_edge_arc
                        && !timing_sense_matches(
                            arc.timing_sense_str(library),
                            source_rise,
                            output_rise,
                        ))
                    || !when_is_true(
                        library,
                        arc.when,
                        previous_map.as_ref(),
                        current_map.as_ref(),
                        condition_cache,
                        &mut result.diagnostics,
                    )?
                {
                    continue;
                }
                let ResolvedSource::Signal(source) =
                    topology_instance.pin_sources[related_pin_index]
                else {
                    continue;
                };
                if is_sequential_edge_arc
                    && !matches!(topology.signals[source.0].owner, SignalOwner::Clock)
                {
                    continue;
                }
                candidates.push(TimingCauseKey {
                    output: output_signal,
                    source,
                    source_rise,
                    output_rise,
                    instance: instance_index,
                    output_pin: output_pin_index,
                    arc: arc_index,
                });
            }
            if candidates.is_empty() {
                *result
                    .unattributed
                    .entry((output_signal, output_rise))
                    .or_default() += 1;
                result.diagnostics.unattributed_output_transition_count += 1;
            } else {
                if candidates.len() > 1 {
                    result
                        .diagnostics
                        .multiply_attributed_output_transition_count += 1;
                }
                let weight = 1.0 / candidates.len() as f64;
                for candidate in candidates {
                    *result.timing_causes.entry(candidate).or_default() += weight;
                }
            }
        }
        for owner_pin_index in 0..instance.pins.len() {
            let pin = &cell.pins[topology_instance.liberty_pins[owner_pin_index]];
            let owner_output_rise =
                topology_instance.output_signals[owner_pin_index].and_then(|_| {
                    (previous_values[owner_pin_index] != current_values[owner_pin_index])
                        .then_some(current_values[owner_pin_index])
                });
            // Output-owned internal-power groups describe the owner's output
            // transition. A related input or clock transition alone must not
            // charge them while that output is stable.
            if pin.direction == PinDirection::Output as i32 && owner_output_rise.is_none() {
                continue;
            }
            let mut candidates = Vec::new();
            for (group_index, group) in pin.internal_power.iter().enumerate() {
                if !when_is_true(
                    library,
                    group.when,
                    previous_map.as_ref(),
                    current_map.as_ref(),
                    condition_cache,
                    &mut result.diagnostics,
                )? {
                    continue;
                }
                let source_pin_indices = if group.related_pins.is_empty() {
                    vec![owner_pin_index]
                } else {
                    group
                        .related_pins
                        .iter()
                        .map(|related| {
                            let name = library.resolve_string(related);
                            topology_instance
                                .pin_index_by_name
                                .get(name)
                                .copied()
                                .ok_or_else(|| {
                                format!(
                                    "instance '{}' internal_power on '{}' names unknown related pin '{}'",
                                    instance.instance_name,
                                    instance.pins[owner_pin_index].pin_name,
                                    name
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                candidates.extend(
                    source_pin_indices
                        .into_iter()
                        .filter_map(|source_pin_index| {
                            let source_rise = current_values[source_pin_index];
                            if previous_values[source_pin_index] == source_rise {
                                return None;
                            }
                            match topology_instance.pin_sources[source_pin_index] {
                                ResolvedSource::Signal(source) => Some(InternalEventKey {
                                    instance: instance_index,
                                    owner_pin: owner_pin_index,
                                    group: group_index,
                                    source,
                                    source_rise,
                                    owner_output_rise,
                                }),
                                ResolvedSource::Literal => None,
                            }
                        }),
                );
            }
            if candidates.len() > 1 {
                result.diagnostics.multiply_attributed_internal_event_count += 1;
            }
            if !candidates.is_empty() {
                let weight = 1.0 / candidates.len() as f64;
                for candidate in candidates {
                    *result.internal_events.entry(candidate).or_default() += weight;
                }
            }
        }
    }
    Ok(())
}

fn pin_value_map(
    instance: &PowerInstance,
    topology_instance: &InstanceTopology,
    values: &[bool],
    state_alias_values: &[bool],
) -> HashMap<String, bool> {
    let mut result: HashMap<_, _> = instance
        .pins
        .iter()
        .zip(values)
        .map(|(pin, value)| (pin.pin_name.clone(), *value))
        .collect();
    result.extend(
        topology_instance
            .state_aliases
            .iter()
            .zip(state_alias_values)
            .map(|(alias, value)| (alias.name.clone(), *value)),
    );
    result
}

fn when_is_true(
    library: &Library,
    when: StringId,
    previous_values: Option<&HashMap<String, bool>>,
    current_values: Option<&HashMap<String, bool>>,
    cache: &mut HashMap<String, Term>,
    diagnostics: &mut GvDynamicPowerDiagnostics,
) -> Result<bool, String> {
    let expression = library.resolve_string(&when);
    if expression.is_empty() {
        return Ok(true);
    }
    let previous_values = previous_values.ok_or_else(|| {
        format!("internal error: no previous pin values for Liberty when '{expression}'")
    })?;
    let current_values = current_values.ok_or_else(|| {
        format!("internal error: no current pin values for Liberty when '{expression}'")
    })?;
    if !cache.contains_key(expression) {
        let parsed = parse_formula(expression)
            .map_err(|error| format!("failed to parse Liberty when '{expression}': {error}"))?;
        cache.insert(expression.to_string(), parsed);
    }
    let term = cache
        .get(expression)
        .ok_or_else(|| format!("internal error: no parsed Liberty when for '{expression}'"))?;
    let previous = term.evaluate_partial(previous_values).ok_or_else(|| {
        format!("Liberty when '{expression}' could not be evaluated from cell pin values")
    })?;
    let current = term.evaluate_partial(current_values).ok_or_else(|| {
        format!("Liberty when '{expression}' could not be evaluated from cell pin values")
    })?;
    diagnostics.when_evaluation_count += 1;
    if previous != current {
        diagnostics.when_changed_during_transition_count += 1;
    }
    Ok(previous)
}

fn is_sequential_edge_timing_type(timing_type: &str) -> bool {
    matches!(timing_type, "rising_edge" | "falling_edge")
}

fn timing_type_allows_output(
    timing_type: &str,
    output_rise: bool,
    transition_kind: PowerTransitionKind,
    is_sequential_instance: bool,
) -> bool {
    if is_sequential_instance {
        return match transition_kind {
            PowerTransitionKind::ActiveClockEdge { clock_rise } => match timing_type {
                "rising_edge" => clock_rise,
                "falling_edge" => !clock_rise,
                _ => false,
            },
            PowerTransitionKind::Combinational
            | PowerTransitionKind::InputSettle
            | PowerTransitionKind::InactiveClockEdge => false,
        };
    }
    match timing_type {
        "" | "combinational" => true,
        "combinational_rise" => output_rise,
        "combinational_fall" => !output_rise,
        _ => false,
    }
}

fn timing_sense_matches(timing_sense: &str, source_rise: bool, output_rise: bool) -> bool {
    match timing_sense {
        "positive_unate" => source_rise == output_rise,
        "negative_unate" => source_rise != output_rise,
        "" | "non_unate" => true,
        _ => true,
    }
}

fn transition_table(tables: &[TimingTable], rise: bool) -> Option<&TimingTable> {
    let kind = if rise {
        TimingTableKind::RiseTransition
    } else {
        TimingTableKind::FallTransition
    };
    tables.iter().find(|table| table.kind == kind)
}

fn build_slew_buckets(
    library: &Library,
    seed_transitions: &[f64],
) -> Result<Vec<GvSlewBucket>, String> {
    let Some((&first_transition, remaining_transitions)) = seed_transitions.split_first() else {
        return Err("dynamic-power slew buckets require at least one seed transition".to_string());
    };
    let mut minimum = first_transition;
    let mut maximum = first_transition;
    for transition in remaining_transitions {
        include_slew_range(&mut minimum, &mut maximum, *transition, *transition);
    }
    for cell in &library.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                for table in &arc.tables {
                    if !matches!(
                        table.kind,
                        TimingTableKind::RiseTransition | TimingTableKind::FallTransition
                    ) {
                        continue;
                    }
                    if let Some((lower, upper)) =
                        timing_lut_input_transition_range(library, table, "timing transition table")
                            .map_err(|error| error.to_string())?
                    {
                        include_slew_range(&mut minimum, &mut maximum, lower, upper);
                    }
                }
            }
            for group in &pin.internal_power {
                for table in &group.tables {
                    if let Some((lower, upper)) =
                        power_lut_input_transition_range(library, table, "internal power table")
                            .map_err(|error| error.to_string())?
                    {
                        include_slew_range(&mut minimum, &mut maximum, lower, upper);
                    }
                }
            }
        }
    }
    if maximum <= minimum {
        minimum /= 2.0;
        maximum *= 2.0;
    }
    if minimum <= 0.0 {
        minimum = (first_transition.min(maximum) / 1024.0).max(f64::MIN_POSITIVE);
    }
    let ratio = (maximum / minimum).powf(1.0 / GV_POWER_SLEW_BUCKET_COUNT as f64);
    let mut lower = minimum;
    Ok((0..GV_POWER_SLEW_BUCKET_COUNT)
        .map(|index| {
            let upper = if index + 1 == GV_POWER_SLEW_BUCKET_COUNT {
                maximum
            } else {
                minimum * ratio.powf((index + 1) as f64)
            };
            let bucket = GvSlewBucket {
                lower,
                upper,
                midpoint: (lower + upper) / 2.0,
            };
            lower = upper;
            bucket
        })
        .collect())
}

fn include_slew_range(minimum: &mut f64, maximum: &mut f64, lower: f64, upper: f64) {
    if lower.is_finite() && lower > 0.0 {
        *minimum = minimum.min(lower);
    }
    if upper.is_finite() && upper > 0.0 {
        *maximum = maximum.max(upper);
    }
}

fn seed_root_histograms(
    topology: &mut PowerTopology,
    buckets: &[GvSlewBucket],
    primary_input_transition: f64,
    clock_transition: f64,
) {
    let primary_input_bucket = slew_bucket_index(buckets, primary_input_transition);
    let clock_bucket = slew_bucket_index(buckets, clock_transition);
    for signal in &mut topology.signals {
        signal.histogram.rise = vec![0.0; buckets.len()];
        signal.histogram.fall = vec![0.0; buckets.len()];
        let bucket = match signal.owner {
            SignalOwner::ModuleInput => Some(primary_input_bucket),
            SignalOwner::Clock => Some(clock_bucket),
            SignalOwner::CellOutput { .. } => None,
        };
        if let Some(bucket) = bucket {
            signal.histogram.rise[bucket] = signal.edges.rise as f64;
            signal.histogram.fall[bucket] = signal.edges.fall as f64;
        }
    }
}

fn signal_source_transition(
    owner: SignalOwner,
    primary_input_transition: f64,
    clock_transition: f64,
) -> f64 {
    match owner {
        SignalOwner::Clock => clock_transition,
        SignalOwner::ModuleInput | SignalOwner::CellOutput { .. } => primary_input_transition,
    }
}

fn output_fallback_transition(
    model: &PowerModelView<'_>,
    topology: &PowerTopology,
    output: SignalId,
    primary_input_transition: f64,
    clock_transition: f64,
) -> f64 {
    match topology.signals[output.0].owner {
        SignalOwner::CellOutput { instance } if model.instances[instance].is_sequential => {
            clock_transition
        }
        SignalOwner::ModuleInput | SignalOwner::Clock | SignalOwner::CellOutput { .. } => {
            primary_input_transition
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn propagate_slew_histograms(
    model: &PowerModelView<'_>,
    library: &Library,
    topology: &mut PowerTopology,
    activity: &ObservedActivity,
    buckets: &[GvSlewBucket],
    primary_input_transition: f64,
    clock_transition: f64,
    diagnostics: &mut RawLutQueryDiagnostics,
) -> Result<(), String> {
    let causes_by_output: BTreeMap<SignalId, Vec<_>> =
        activity
            .timing_causes
            .iter()
            .fold(BTreeMap::new(), |mut map, (key, count)| {
                map.entry(key.output).or_default().push((*key, *count));
                map
            });
    for instance_index in &topology.instance_order {
        let output_signals: Vec<_> = topology.instances[*instance_index]
            .output_signals
            .iter()
            .flatten()
            .copied()
            .collect();
        for output_id in output_signals {
            if let Some(causes) = causes_by_output.get(&output_id) {
                for (cause, cause_count) in causes {
                    let source_histogram = if cause.source_rise {
                        topology.signals[cause.source.0].histogram.rise.clone()
                    } else {
                        topology.signals[cause.source.0].histogram.fall.clone()
                    };
                    let source_total: f64 = source_histogram.iter().sum();
                    let output_load = if cause.output_rise {
                        topology.signals[output_id.0].load.rise
                    } else {
                        topology.signals[output_id.0].load.fall
                    };
                    let cell = &library.cells[topology.instances[cause.instance].cell];
                    let liberty_pin =
                        topology.instances[cause.instance].liberty_pins[cause.output_pin];
                    let arc = &cell.pins[liberty_pin].timing_arcs[cause.arc];
                    let table =
                        transition_table(&arc.tables, cause.output_rise).ok_or_else(|| {
                            format!(
                                "instance '{}' pin '{}' lost its selected transition table",
                                model.instances[cause.instance].instance_name,
                                model.instances[cause.instance].pins[cause.output_pin].pin_name
                            )
                        })?;
                    if source_total == 0.0 {
                        let fallback_bucket = slew_bucket_index(
                            buckets,
                            signal_source_transition(
                                topology.signals[cause.source.0].owner,
                                primary_input_transition,
                                clock_transition,
                            ),
                        );
                        add_histogram_mass(
                            &mut topology.signals[output_id.0],
                            cause.output_rise,
                            fallback_bucket,
                            *cause_count,
                        );
                        continue;
                    }
                    for (bucket_index, source_mass) in source_histogram.iter().enumerate() {
                        if *source_mass == 0.0 {
                            continue;
                        }
                        let output_slew = evaluate_timing_lut_raw(
                            library,
                            table,
                            RawLutQuery {
                                input_transition: buckets[bucket_index].midpoint,
                                output_load,
                            },
                            diagnostics,
                            &format!(
                                "instance '{}' pin '{}' {}",
                                model.instances[cause.instance].instance_name,
                                model.instances[cause.instance].pins[cause.output_pin].pin_name,
                                table.kind_str()
                            ),
                        )
                        .map_err(|error| error.to_string())?;
                        if output_slew < 0.0 {
                            return Err(format!(
                                "Liberty transition lookup produced negative slew {output_slew}"
                            ));
                        }
                        let destination = slew_bucket_index(buckets, output_slew);
                        add_histogram_mass(
                            &mut topology.signals[output_id.0],
                            cause.output_rise,
                            destination,
                            cause_count * source_mass / source_total,
                        );
                    }
                }
            }
            for output_rise in [false, true] {
                let missing = activity
                    .unattributed
                    .get(&(output_id, output_rise))
                    .copied()
                    .unwrap_or(0);
                if missing != 0 {
                    let fallback_bucket = slew_bucket_index(
                        buckets,
                        output_fallback_transition(
                            model,
                            topology,
                            output_id,
                            primary_input_transition,
                            clock_transition,
                        ),
                    );
                    add_histogram_mass(
                        &mut topology.signals[output_id.0],
                        output_rise,
                        fallback_bucket,
                        missing as f64,
                    );
                }
                let target = if output_rise {
                    topology.signals[output_id.0].edges.rise
                } else {
                    topology.signals[output_id.0].edges.fall
                } as f64;
                let fallback_bucket = slew_bucket_index(
                    buckets,
                    output_fallback_transition(
                        model,
                        topology,
                        output_id,
                        primary_input_transition,
                        clock_transition,
                    ),
                );
                let histogram = if output_rise {
                    &mut topology.signals[output_id.0].histogram.rise
                } else {
                    &mut topology.signals[output_id.0].histogram.fall
                };
                let actual: f64 = histogram.iter().sum();
                if (target - actual).abs() > 1e-8 * target.max(1.0) {
                    histogram[fallback_bucket] += target - actual;
                }
            }
        }
    }
    Ok(())
}

fn add_histogram_mass(signal: &mut Signal, rise: bool, bucket: usize, mass: f64) {
    if rise {
        signal.histogram.rise[bucket] += mass;
    } else {
        signal.histogram.fall[bucket] += mass;
    }
}

#[allow(clippy::too_many_arguments)]
fn calculate_internal_energy(
    model: &PowerModelView<'_>,
    library: &Library,
    topology: &PowerTopology,
    activity: &ObservedActivity,
    buckets: &[GvSlewBucket],
    primary_input_transition: f64,
    clock_transition: f64,
    diagnostics: &mut RawLutQueryDiagnostics,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let mut instance_energy = vec![0.0; model.instances.len()];
    let mut pin_energy: Vec<_> = model
        .instances
        .iter()
        .map(|instance| vec![0.0; instance.pins.len()])
        .collect();
    for (event, count) in &activity.internal_events {
        let topology_instance = &topology.instances[event.instance];
        let cell = &library.cells[topology_instance.cell];
        let liberty_pin = topology_instance.liberty_pins[event.owner_pin];
        let group = &cell.pins[liberty_pin].internal_power[event.group];
        // Directional power tables describe the owning pin's transition. The
        // related/source pin transition instead selects the slew histogram.
        let power_rise = event.owner_output_rise.unwrap_or(event.source_rise);
        let exact: Vec<_> = group
            .tables
            .iter()
            .filter(|table| {
                table.transition
                    == if power_rise {
                        PowerTransition::Rise
                    } else {
                        PowerTransition::Fall
                    }
            })
            .collect();
        let tables: Vec<_> = if exact.is_empty() {
            group
                .tables
                .iter()
                .filter(|table| table.transition == PowerTransition::Both)
                .collect()
        } else {
            exact
        };
        if tables.is_empty() {
            continue;
        }
        if tables.len() > 1 {
            return Err(format!(
                "instance '{}' pin '{}' internal_power has {} tables for one transition direction",
                model.instances[event.instance].instance_name,
                model.instances[event.instance].pins[event.owner_pin].pin_name,
                tables.len()
            ));
        }
        let histogram = if event.source_rise {
            &topology.signals[event.source.0].histogram.rise
        } else {
            &topology.signals[event.source.0].histogram.fall
        };
        let histogram_total: f64 = histogram.iter().sum();
        let owner_load = topology_instance.output_signals[event.owner_pin]
            .map(|signal_id| {
                let load = topology.signals[signal_id.0].load;
                match event.owner_output_rise {
                    Some(true) => load.rise,
                    Some(false) => load.fall,
                    None => (load.rise + load.fall) / 2.0,
                }
            })
            .unwrap_or(0.0);
        let table = tables[0];
        let mut average_energy = 0.0;
        if histogram_total == 0.0 {
            let fallback_transition = signal_source_transition(
                topology.signals[event.source.0].owner,
                primary_input_transition,
                clock_transition,
            );
            average_energy = evaluate_power_lut(
                library,
                table,
                RawLutQuery {
                    input_transition: fallback_transition,
                    output_load: owner_load,
                },
                diagnostics,
                "internal power",
            )
            .map_err(|error| error.to_string())?;
        } else {
            for (bucket_index, mass) in histogram.iter().enumerate() {
                if *mass == 0.0 {
                    continue;
                }
                let energy = evaluate_power_lut(
                    library,
                    table,
                    RawLutQuery {
                        input_transition: buckets[bucket_index].midpoint,
                        output_load: owner_load,
                    },
                    diagnostics,
                    &format!(
                        "instance '{}' pin '{}' internal_power",
                        model.instances[event.instance].instance_name,
                        model.instances[event.instance].pins[event.owner_pin].pin_name
                    ),
                )
                .map_err(|error| error.to_string())?;
                average_energy += mass * energy / histogram_total;
            }
        }
        let energy = count * average_energy;
        instance_energy[event.instance] += energy;
        pin_energy[event.instance][event.owner_pin] += energy;
    }
    Ok((instance_energy, pin_energy))
}

fn slew_bucket_index(buckets: &[GvSlewBucket], slew: f64) -> usize {
    buckets
        .partition_point(|bucket| bucket.upper < slew)
        .min(buckets.len() - 1)
}

fn switching_energy(signal: &Signal, voltage_squared: f64) -> f64 {
    0.5 * voltage_squared
        * (signal.edges.rise as f64 * signal.load.rise
            + signal.edges.fall as f64 * signal.load.fall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slew_grid_has_fixed_log_buckets_with_arithmetic_midpoints() {
        let buckets = build_slew_buckets(&Library::default(), &[4.0]).unwrap();
        assert_eq!(buckets.len(), GV_POWER_SLEW_BUCKET_COUNT);
        assert_eq!(buckets.first().unwrap().lower, 2.0);
        assert_eq!(buckets.last().unwrap().upper, 8.0);
        for pair in buckets.windows(2) {
            assert_eq!(pair[0].upper, pair[1].lower);
            let first_ratio = pair[0].upper / pair[0].lower;
            let second_ratio = pair[1].upper / pair[1].lower;
            assert!((first_ratio - second_ratio).abs() < 1e-12);
        }
        for bucket in buckets {
            assert_eq!(bucket.midpoint, (bucket.lower + bucket.upper) / 2.0);
        }
    }
}
