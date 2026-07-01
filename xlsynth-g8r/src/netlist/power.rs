// SPDX-License-Identifier: Apache-2.0

//! Sample-driven dynamic-power estimation for labeled combinational netlists.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;
use xlsynth::IrBits;

use crate::aig::{AigNode, AigOperand};
use crate::aig_sim::gate_simd;
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty::lut::{
    RawLutQuery, RawLutQueryDiagnostics, evaluate_power_lut, evaluate_timing_lut_raw,
    power_lut_input_transition_range, timing_lut_input_transition_range,
};
use crate::liberty_model::{Library, Pin, PinDirection, PowerTransition, StringId, TimingTable};
use crate::liberty_proto::TimingTableKind;
use crate::netlist::gatefn_from_netlist::{LabeledNetlistAig, PinConnection};
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
    CellOutput { instance: usize },
}

#[derive(Clone, Debug)]
struct Signal {
    operand: AigOperand,
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
    pin_sources: Vec<ResolvedSource>,
    output_signals: Vec<Option<SignalId>>,
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
    let nominal_voltage = library.nominal_voltage.ok_or_else(|| {
        "Liberty proto has no nominal_voltage; switching energy requires a voltage".to_string()
    })?;
    if !nominal_voltage.is_finite() || nominal_voltage <= 0.0 {
        return Err(format!(
            "Liberty nominal_voltage must be finite and positive; got {nominal_voltage}"
        ));
    }

    let mut topology = build_topology(model, library, options.module_output_load)?;
    let mut activity = observe_activity(model, library, batch_inputs, &mut topology)?;
    let buckets = build_slew_buckets(library, options.primary_input_transition)?;
    seed_primary_input_histograms(&mut topology, &buckets, options.primary_input_transition);
    let mut lut_diagnostics = RawLutQueryDiagnostics::default();
    propagate_slew_histograms(
        model,
        library,
        &mut topology,
        &activity,
        &buckets,
        options.primary_input_transition,
        &mut lut_diagnostics,
    )?;
    let (instance_internal, pin_internal) = calculate_internal_energy(
        model,
        library,
        &topology,
        &activity,
        &buckets,
        options.primary_input_transition,
        &mut lut_diagnostics,
    )?;
    activity.diagnostics.lut_below_min_clamp_count = lut_diagnostics.below_min_clamp_count;
    activity.diagnostics.lut_above_max_clamp_count = lut_diagnostics.above_max_clamp_count;

    let voltage_squared = nominal_voltage * nominal_voltage;
    let mut primary_input_switching_energy = 0.0;
    let mut cell_output_switching_energy = 0.0;
    let mut instance_switching = vec![0.0; model.instances.len()];
    for signal in &topology.signals {
        let energy = switching_energy(signal, voltage_squared);
        match signal.owner {
            SignalOwner::ModuleInput => primary_input_switching_energy += energy,
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

    let instances = model
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
        .collect();

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
    let power_unit = options
        .cycle_time
        .map(|_| format!("{energy_unit}/{time_unit}"));
    Ok(GvDynamicPowerReport {
        module_name: model.module_name.clone(),
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

fn validate_options(options: GvDynamicPowerOptions) -> Result<(), String> {
    if !options.primary_input_transition.is_finite() || options.primary_input_transition <= 0.0 {
        return Err(format!(
            "primary_input_transition must be finite and positive; got {}",
            options.primary_input_transition
        ));
    }
    if !options.module_output_load.is_finite() || options.module_output_load < 0.0 {
        return Err(format!(
            "module_output_load must be finite and non-negative; got {}",
            options.module_output_load
        ));
    }
    if let Some(cycle_time) = options.cycle_time
        && (!cycle_time.is_finite() || cycle_time <= 0.0)
    {
        return Err(format!(
            "cycle_time must be finite and positive; got {cycle_time}"
        ));
    }
    Ok(())
}

fn build_topology(
    model: &LabeledNetlistAig,
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
    for port in &model.module_ports {
        if port.direction != PortDirection::Input {
            continue;
        }
        for bit in &port.bits_lsb_to_msb {
            let signal_id = add_signal(
                &mut signals,
                &mut operand_sources,
                bit.operand,
                SignalOwner::ModuleInput,
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
                    pin.operand,
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
        instances.push(InstanceTopology {
            cell,
            liberty_pins,
            pin_sources: vec![ResolvedSource::Literal; instance.pins.len()],
            output_signals,
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
                pin.operand,
                &net_drivers,
                &operand_sources,
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
                bit.operand,
                &net_drivers,
                &operand_sources,
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

fn add_signal(
    signals: &mut Vec<Signal>,
    operand_sources: &mut HashMap<AigOperand, Vec<SignalId>>,
    operand: AigOperand,
    owner: SignalOwner,
) -> SignalId {
    let signal_id = SignalId(signals.len());
    signals.push(Signal {
        operand,
        owner,
        load: EdgeValue::default(),
        edges: EdgeValue::default(),
        histogram: EdgeValue::default(),
    });
    operand_sources.entry(operand).or_default().push(signal_id);
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
    model: &LabeledNetlistAig,
    connection: &PinConnection,
    operand: AigOperand,
    net_drivers: &HashMap<(String, u32), SignalId>,
    operand_sources: &HashMap<AigOperand, Vec<SignalId>>,
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
        let dependencies: BTreeSet<_> = instance
            .pin_sources
            .iter()
            .filter_map(|source| match source {
                ResolvedSource::Signal(signal_id) => match signals[signal_id.0].owner {
                    SignalOwner::CellOutput { instance } if instance != consumer => Some(instance),
                    _ => None,
                },
                ResolvedSource::Literal => None,
            })
            .collect();
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
    model: &LabeledNetlistAig,
    library: &Library,
    batch_inputs: &[Vec<IrBits>],
    topology: &mut PowerTopology,
) -> Result<ObservedActivity, String> {
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
            let current = sample_state(model, topology, &nodes, lane)?;
            if let Some(previous) = previous.as_ref() {
                observe_transition(
                    model,
                    library,
                    topology,
                    previous,
                    &current,
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
    model: &LabeledNetlistAig,
    topology: &PowerTopology,
    nodes: &[gate_simd::Vec256],
    lane: usize,
) -> Result<SampleState, String> {
    let operand_value = |operand: AigOperand| {
        nodes
            .get(operand.node.id)
            .copied()
            .map(|value| value.apply_neg(operand.negated).get_lane(lane))
            .ok_or_else(|| format!("AIG node {} is out of range", operand.node.id))
    };
    let signal_values = topology
        .signals
        .iter()
        .map(|signal| operand_value(signal.operand))
        .collect::<Result<Vec<_>, _>>()?;
    let pin_values = model
        .instances
        .iter()
        .map(|instance| {
            instance
                .pins
                .iter()
                .map(|pin| operand_value(pin.operand))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SampleState {
        signal_values,
        pin_values,
    })
}

#[allow(clippy::too_many_arguments)]
fn observe_transition(
    model: &LabeledNetlistAig,
    library: &Library,
    topology: &mut PowerTopology,
    previous: &SampleState,
    current: &SampleState,
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
        let cell = &library.cells[topology_instance.cell];
        let pin_index_by_name: HashMap<_, _> = instance
            .pins
            .iter()
            .enumerate()
            .map(|(index, pin)| (pin.pin_name.as_str(), index))
            .collect();
        let previous_values = &previous.pin_values[instance_index];
        let current_values = &current.pin_values[instance_index];
        let previous_map = pin_value_map(instance, previous_values);
        let current_map = pin_value_map(instance, current_values);
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
                if !timing_type_allows_output(arc.timing_type_str(library), output_rise)
                    || transition_table(arc.tables.as_slice(), output_rise).is_none()
                {
                    continue;
                }
                let related_name = library.resolve_string(&arc.related_pin);
                let Some(&related_pin_index) = pin_index_by_name.get(related_name) else {
                    return Err(format!(
                        "instance '{}' output '{}' timing arc names unknown related pin '{}'",
                        instance.instance_name,
                        instance.pins[output_pin_index].pin_name,
                        related_name
                    ));
                };
                let source_rise = current_values[related_pin_index];
                if previous_values[related_pin_index] == source_rise
                    || !timing_sense_matches(
                        arc.timing_sense_str(library),
                        source_rise,
                        output_rise,
                    )
                    || !when_is_true(
                        library,
                        arc.when,
                        &previous_map,
                        &current_map,
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
            let mut candidates = Vec::new();
            for (group_index, group) in pin.internal_power.iter().enumerate() {
                if !when_is_true(
                    library,
                    group.when,
                    &previous_map,
                    &current_map,
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
                            pin_index_by_name.get(name).copied().ok_or_else(|| {
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
    instance: &crate::netlist::gatefn_from_netlist::InstanceAigBinding,
    values: &[bool],
) -> HashMap<String, bool> {
    instance
        .pins
        .iter()
        .zip(values)
        .map(|(pin, value)| (pin.pin_name.clone(), *value))
        .collect()
}

fn when_is_true(
    library: &Library,
    when: StringId,
    previous_values: &HashMap<String, bool>,
    current_values: &HashMap<String, bool>,
    cache: &mut HashMap<String, Term>,
    diagnostics: &mut GvDynamicPowerDiagnostics,
) -> Result<bool, String> {
    let expression = library.resolve_string(&when);
    if expression.is_empty() {
        return Ok(true);
    }
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

fn timing_type_allows_output(timing_type: &str, output_rise: bool) -> bool {
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
    primary_input_transition: f64,
) -> Result<Vec<GvSlewBucket>, String> {
    let mut minimum = primary_input_transition;
    let mut maximum = primary_input_transition;
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
        minimum = (primary_input_transition.min(maximum) / 1024.0).max(f64::MIN_POSITIVE);
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

fn seed_primary_input_histograms(
    topology: &mut PowerTopology,
    buckets: &[GvSlewBucket],
    primary_input_transition: f64,
) {
    let bucket = slew_bucket_index(buckets, primary_input_transition);
    for signal in &mut topology.signals {
        signal.histogram.rise = vec![0.0; buckets.len()];
        signal.histogram.fall = vec![0.0; buckets.len()];
        if matches!(signal.owner, SignalOwner::ModuleInput) {
            signal.histogram.rise[bucket] = signal.edges.rise as f64;
            signal.histogram.fall[bucket] = signal.edges.fall as f64;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn propagate_slew_histograms(
    model: &LabeledNetlistAig,
    library: &Library,
    topology: &mut PowerTopology,
    activity: &ObservedActivity,
    buckets: &[GvSlewBucket],
    primary_input_transition: f64,
    diagnostics: &mut RawLutQueryDiagnostics,
) -> Result<(), String> {
    let fallback_bucket = slew_bucket_index(buckets, primary_input_transition);
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
    model: &LabeledNetlistAig,
    library: &Library,
    topology: &PowerTopology,
    activity: &ObservedActivity,
    buckets: &[GvSlewBucket],
    primary_input_transition: f64,
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
            average_energy = evaluate_power_lut(
                library,
                table,
                RawLutQuery {
                    input_transition: primary_input_transition,
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
        let buckets = build_slew_buckets(&Library::default(), 4.0).unwrap();
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
