// SPDX-License-Identifier: Apache-2.0

//! Functional evaluation of Liberty-backed gate-level netlists.

use std::path::Path;

use anyhow::{Result, anyhow};
use serde::Serialize;
use xlsynth::{IrBits, IrValue};

use crate::aig::{AigOperand, SequentialGateFn};
use crate::aig_sim::count_toggles;
use crate::aig_sim::gate_sim::{self, Collect};
use crate::aig_sim::gate_simd;
use crate::aig_sim::sequential::{
    self, SequentialState, SequentialToggleActivity, SequentialTrace,
    count_sequential_toggle_activity_with_all_node_counts,
};
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::gatefn_from_netlist::{
    project_labeled_netlist_aig_with_boundaries,
    project_labeled_sequential_netlist_aig_with_boundaries,
};
use crate::netlist::hierarchy::elaborate_hierarchy;
use crate::netlist::io::{load_liberty_from_path, parse_netlist_from_path, select_module};
use crate::netlist::parse::PortDirection;
use crate::netlist::power::{GvDynamicPowerOptions, GvDynamicPowerReport};

pub use crate::netlist::gatefn_from_netlist::{
    InstanceAigBinding, InstancePinAigBinding, LabeledAigBit, LabeledNetlistAig,
    LabeledSequentialAigBit, LabeledSequentialClock, LabeledSequentialNetlistAig,
    ModuleBoundaryAigBinding, ModulePortAigBinding, PinConnection, SequentialAigSignal,
    SequentialClockEdge, SequentialInstanceAigBinding, SequentialInstancePinAigBinding,
    SequentialModuleBoundaryAigBinding, SequentialModulePortAigBinding,
};

#[derive(Debug, Clone, Default)]
pub struct GvEvalOptions {
    pub module_name: Option<String>,
    /// Optional top-level clock name for sequential loading. This is needed
    /// when synthesis has optimized away every state element and no FF clock
    /// pin remains from which to infer the clock.
    pub clock_port_name: Option<String>,
}

/// Input or output direction used in the source-labeled toggle report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToggleDirection {
    Input,
    Output,
}

/// Aggregate transition counts, with each physical bit or pin use counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GvToggleAggregate {
    pub module_input_toggles: usize,
    pub module_output_toggles: usize,
    pub cell_input_pin_toggles: usize,
    pub cell_output_pin_toggles: usize,
}

/// Transition activity for one module-port bit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModulePortBitToggleActivity {
    pub bit_number: u32,
    pub toggle_count: usize,
    pub toggle_rate: f64,
}

/// Transition activity for one module port in source declaration order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModulePortToggleActivity {
    pub port_name: String,
    pub direction: ToggleDirection,
    pub bits_lsb_to_msb: Vec<ModulePortBitToggleActivity>,
}

/// Transition activity at one flattened child-module boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModuleBoundaryToggleActivity {
    pub instance_path: String,
    pub module_name: String,
    pub ports: Vec<ModulePortToggleActivity>,
}

/// Source-level connection serialized for a standard-cell pin report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TogglePinConnection {
    Net { net_name: String, bit_number: u32 },
    Literal { value: bool },
    Unconnected,
}

/// Transition activity for one explicitly connected external standard-cell pin.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InstancePinToggleActivity {
    pub pin_name: String,
    pub direction: ToggleDirection,
    pub connection: TogglePinConnection,
    pub toggle_count: usize,
    pub toggle_rate: f64,
}

/// Transition activity for one standard-cell instance in netlist source order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InstanceToggleActivity {
    pub instance_name: String,
    pub cell_type: String,
    pub pins: Vec<InstancePinToggleActivity>,
}

/// Source-labeled toggle activity across consecutive ordered input samples.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GvToggleActivity {
    pub module_name: String,
    pub sample_count: usize,
    pub transition_count: usize,
    pub aggregate: GvToggleAggregate,
    pub module_ports: Vec<ModulePortToggleActivity>,
    /// Boundary activity is informative and excluded from aggregate totals.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub module_boundaries: Vec<ModuleBoundaryToggleActivity>,
    pub instances: Vec<InstanceToggleActivity>,
}

/// Activity kind for one labeled signal in a sequential transition graph.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GvSequentialSignalToggleActivity {
    /// Settled phase-to-phase transitions observed in the transition AIG.
    Sampled {
        toggle_count: usize,
        toggle_rate: f64,
    },
    /// A selected clock is metadata rather than a transition-AIG input.
    Clock { toggle_count: usize },
}

/// Transition activity for one sequential module-port bit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequentialModulePortBitToggleActivity {
    pub bit_number: u32,
    pub activity: GvSequentialSignalToggleActivity,
}

/// Transition activity for one sequential module port.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequentialModulePortToggleActivity {
    pub port_name: String,
    pub direction: ToggleDirection,
    pub bits_lsb_to_msb: Vec<SequentialModulePortBitToggleActivity>,
}

/// Transition activity at one flattened child-module boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequentialModuleBoundaryToggleActivity {
    pub instance_path: String,
    pub module_name: String,
    pub ports: Vec<SequentialModulePortToggleActivity>,
}

/// Transition activity for one sequential standard-cell pin.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequentialInstancePinToggleActivity {
    pub pin_name: String,
    pub direction: ToggleDirection,
    pub connection: TogglePinConnection,
    pub activity: GvSequentialSignalToggleActivity,
}

/// Transition activity for one sequential standard-cell instance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequentialInstanceToggleActivity {
    pub instance_name: String,
    pub cell_type: String,
    pub pins: Vec<SequentialInstancePinToggleActivity>,
}

/// Active-edge metadata for the clock omitted from transition-AIG inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GvSequentialClockActivity {
    pub port_name: String,
    pub active_edge: Option<SequentialClockEdge>,
    /// One active edge per simulated input record, not a full waveform count.
    pub active_edge_count: usize,
    /// One inactive-to-active and one active-to-inactive transition per cycle.
    pub toggle_count: usize,
}

/// Source-labeled toggle activity for one sequential gate-level trace.
///
/// The flattened sequential fields preserve the native g8r-eval categories,
/// while module_ports and instances preserve mapped netlist labels.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GvSequentialToggleActivity {
    pub module_name: String,
    #[serde(flatten)]
    pub sequential: SequentialToggleActivity,
    pub clock: Option<GvSequentialClockActivity>,
    /// Settled labeled totals, including synthetic selected-clock transitions.
    pub labeled_aggregate: GvToggleAggregate,
    pub module_ports: Vec<SequentialModulePortToggleActivity>,
    /// Boundary activity is informative and excluded from labeled aggregate
    /// totals.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub module_boundaries: Vec<SequentialModuleBoundaryToggleActivity>,
    pub instances: Vec<SequentialInstanceToggleActivity>,
}

/// Loads, validates, and projects one combinational netlist module.
pub fn load_labeled_netlist_aig(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    options: &GvEvalOptions,
) -> Result<LabeledNetlistAig> {
    let liberty = load_liberty_from_path(liberty_proto_path)?;
    load_labeled_netlist_aig_with_liberty(netlist_path, &liberty, options)
}

/// Loads, validates, and projects one combinational netlist module using an
/// already parsed Liberty model.
pub fn load_labeled_netlist_aig_with_liberty(
    netlist_path: &Path,
    liberty: &Library,
    options: &GvEvalOptions,
) -> Result<LabeledNetlistAig> {
    let parsed = parse_netlist_from_path(netlist_path)?;
    let module = select_module(&parsed, options.module_name.as_deref())?;
    let elaborated = elaborate_hierarchy(&parsed, module)?;
    project_labeled_netlist_aig_with_boundaries(
        &elaborated.module,
        &elaborated.nets,
        &elaborated.interner,
        liberty,
        &elaborated.module_boundaries,
    )
    .map_err(|e| anyhow!(e))
}

/// Loads one FF-only gate-level module as a labeled sequential transition AIG.
///
/// One simulation cycle represents one active edge of the selected clock.
/// Positive-edge and negative-edge FFs are each supported, but a module may
/// not mix them. Latches, multiple clocks, derived clocks, and asynchronous
/// clear/preset cells are rejected.
pub fn load_labeled_sequential_netlist_aig(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    options: &GvEvalOptions,
) -> Result<LabeledSequentialNetlistAig> {
    let liberty = load_liberty_from_path(liberty_proto_path)?;
    load_labeled_sequential_netlist_aig_with_liberty(netlist_path, &liberty, options)
}

/// Loads one FF-only gate-level module using an already parsed Liberty model.
///
/// The returned model preserves mapped instance and pin labels while its
/// sequential gate function can be evaluated with the sequential simulator.
/// Register names are internal to
/// the mapped netlist; callers comparing pre- and post-map behavior should
/// compare visible outputs rather than trying to correlate state elements.
pub fn load_labeled_sequential_netlist_aig_with_liberty(
    netlist_path: &Path,
    liberty: &Library,
    options: &GvEvalOptions,
) -> Result<LabeledSequentialNetlistAig> {
    let parsed = parse_netlist_from_path(netlist_path)?;
    let module = select_module(&parsed, options.module_name.as_deref())?;
    let elaborated = elaborate_hierarchy(&parsed, module)?;
    project_labeled_sequential_netlist_aig_with_boundaries(
        &elaborated.module,
        &elaborated.nets,
        &elaborated.interner,
        liberty,
        options.clock_port_name.as_deref(),
        &elaborated.module_boundaries,
    )
    .map_err(|error| anyhow!(error))
}

/// Loads one FF-only gate-level module as a sequential transition function.
pub fn load_sequential_netlist_gate_fn(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    options: &GvEvalOptions,
) -> Result<SequentialGateFn> {
    Ok(
        load_labeled_sequential_netlist_aig(netlist_path, liberty_proto_path, options)?
            .sequential_gate_fn,
    )
}

impl LabeledNetlistAig {
    /// Evaluates one sample expressed as bits values in module-input order.
    pub fn evaluate_bits(&self, inputs: &[IrBits]) -> Result<Vec<IrBits>, String> {
        self.validate_input_bits(inputs)?;
        Ok(gate_sim::eval(&self.gate_fn, inputs, Collect::None).outputs)
    }

    /// Evaluates one typed IR tuple and returns a typed output value.
    pub fn evaluate_ir_value(&self, args: &IrValue) -> Result<IrValue, String> {
        let inputs = self.lower_arg_tuple(args)?;
        let outputs = self.evaluate_bits(&inputs)?;
        Ok(outputs_to_ir_value(&outputs))
    }

    /// Evaluates ordered typed IR tuples, using SIMD evaluation for batches.
    pub fn evaluate_ir_values(&self, args: &[IrValue]) -> Result<Vec<IrValue>, String> {
        let batch_inputs = self.lower_ir_values(args)?;
        if batch_inputs.is_empty() {
            return Ok(Vec::new());
        }
        let batch_outputs = if batch_inputs.len() == 1 {
            vec![gate_sim::eval(&self.gate_fn, &batch_inputs[0], Collect::None).outputs]
        } else {
            gate_simd::eval_ordered_batch(&self.gate_fn, &batch_inputs)?
        };
        Ok(batch_outputs
            .iter()
            .map(|outputs| outputs_to_ir_value(outputs))
            .collect())
    }

    /// Counts source-labeled toggles across consecutive typed IR tuples.
    pub fn count_toggle_activity(&self, args: &[IrValue]) -> Result<GvToggleActivity, String> {
        let batch_inputs = self.lower_ir_values(args)?;
        self.count_toggle_activity_bits(&batch_inputs)
    }

    /// Estimates dynamic energy from ordered samples and Liberty power data.
    pub fn analyze_dynamic_power(
        &self,
        library: &crate::liberty_model::Library,
        args: &[IrValue],
        options: GvDynamicPowerOptions,
    ) -> Result<GvDynamicPowerReport, String> {
        let batch_inputs = self.lower_ir_values(args)?;
        crate::netlist::power::analyze_dynamic_power_bits(self, library, &batch_inputs, options)
    }

    /// Counts source-labeled toggles across consecutive lowered input vectors.
    pub fn count_toggle_activity_bits(
        &self,
        batch_inputs: &[Vec<IrBits>],
    ) -> Result<GvToggleActivity, String> {
        let per_node_toggles =
            count_toggles::count_all_node_toggles_simd(&self.gate_fn, batch_inputs)?;
        let transition_count = batch_inputs.len() - 1;

        let module_ports = self
            .module_ports
            .iter()
            .map(|port| module_port_toggle_activity(port, &per_node_toggles, transition_count))
            .collect::<Result<Vec<ModulePortToggleActivity>, String>>()?;
        let module_boundaries = self
            .module_boundaries
            .iter()
            .map(|boundary| {
                let ports = boundary
                    .ports
                    .iter()
                    .map(|port| {
                        module_port_toggle_activity(port, &per_node_toggles, transition_count)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(ModuleBoundaryToggleActivity {
                    instance_path: boundary.instance_path.clone(),
                    module_name: boundary.module_name.clone(),
                    ports,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let instances = self
            .instances
            .iter()
            .map(|instance| {
                let pins = instance
                    .pins
                    .iter()
                    .map(|pin| {
                        let direction = cell_pin_direction(pin.direction)?;
                        let (toggle_count, toggle_rate) =
                            operand_toggle_stats(pin.operand, &per_node_toggles, transition_count)?;
                        Ok(InstancePinToggleActivity {
                            pin_name: pin.pin_name.clone(),
                            direction,
                            connection: toggle_pin_connection(&pin.connection),
                            toggle_count,
                            toggle_rate,
                        })
                    })
                    .collect::<Result<Vec<InstancePinToggleActivity>, String>>()?;
                Ok(InstanceToggleActivity {
                    instance_name: instance.instance_name.clone(),
                    cell_type: instance.cell_type.clone(),
                    pins,
                })
            })
            .collect::<Result<Vec<InstanceToggleActivity>, String>>()?;

        let aggregate = GvToggleAggregate {
            module_input_toggles: sum_module_port_toggles(&module_ports, ToggleDirection::Input),
            module_output_toggles: sum_module_port_toggles(&module_ports, ToggleDirection::Output),
            cell_input_pin_toggles: sum_instance_pin_toggles(&instances, ToggleDirection::Input),
            cell_output_pin_toggles: sum_instance_pin_toggles(&instances, ToggleDirection::Output),
        };
        Ok(GvToggleActivity {
            module_name: self.module_name.clone(),
            sample_count: batch_inputs.len(),
            transition_count,
            aggregate,
            module_ports,
            module_boundaries,
            instances,
        })
    }

    fn lower_ir_values(&self, args: &[IrValue]) -> Result<Vec<Vec<IrBits>>, String> {
        args.iter()
            .enumerate()
            .map(|(sample_index, sample)| {
                self.lower_arg_tuple(sample)
                    .map_err(|e| format!("input sample {}: {}", sample_index + 1, e))
            })
            .collect()
    }

    fn lower_arg_tuple(&self, args: &IrValue) -> Result<Vec<IrBits>, String> {
        let elements = args
            .get_elements()
            .map_err(|e| format!("argument value is not a tuple: {e}"))?;
        let inputs = elements
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value.to_bits().map_err(|e| {
                    let input_name = self
                        .gate_fn
                        .inputs
                        .get(index)
                        .map(|input| input.name.as_str())
                        .unwrap_or("<extra>");
                    format!(
                        "argument {} for input '{}' is not bits-typed: {}",
                        index, input_name, e
                    )
                })
            })
            .collect::<Result<Vec<IrBits>, String>>()?;
        self.validate_input_bits(&inputs)?;
        Ok(inputs)
    }

    fn validate_input_bits(&self, inputs: &[IrBits]) -> Result<(), String> {
        if inputs.len() != self.gate_fn.inputs.len() {
            let input_names = self
                .gate_fn
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<&str>>()
                .join(", ");
            return Err(format!(
                "module '{}' expects {} input values [{}], got {}",
                self.module_name,
                self.gate_fn.inputs.len(),
                input_names,
                inputs.len()
            ));
        }
        for (index, (input, expected)) in inputs.iter().zip(self.gate_fn.inputs.iter()).enumerate()
        {
            let actual_width = input.get_bit_count();
            let expected_width = expected.get_bit_count();
            if actual_width != expected_width {
                return Err(format!(
                    "argument {} for input '{}' has width {}, expected {}",
                    index, expected.name, actual_width, expected_width
                ));
            }
        }
        Ok(())
    }
}

impl LabeledSequentialNetlistAig {
    /// Simulates ordered typed external-input tuples over active clock edges.
    pub fn simulate_ir_values(
        &self,
        args: &[IrValue],
        initial_state: SequentialState,
    ) -> Result<SequentialTrace, String> {
        let external_inputs = self.lower_ir_values(args)?;
        self.simulate_bits(&external_inputs, initial_state)
    }

    /// Simulates ordered lowered external inputs over active clock edges.
    pub fn simulate_bits(
        &self,
        external_inputs: &[Vec<IrBits>],
        initial_state: SequentialState,
    ) -> Result<SequentialTrace, String> {
        sequential::simulate(&self.sequential_gate_fn, external_inputs, initial_state)
    }

    /// Counts native sequential and source-labeled mapped-netlist activity.
    pub fn count_toggle_activity(
        &self,
        trace: &SequentialTrace,
    ) -> Result<GvSequentialToggleActivity, String> {
        let counted =
            count_sequential_toggle_activity_with_all_node_counts(&self.sequential_gate_fn, trace)?;
        let sequential = counted.activity;
        let per_node_toggles = counted.per_node_toggles;
        let transition_count = sequential.logic_transition_count;
        let cycle_count = sequential.cycle_count;
        let clock_toggle_count = sequential.clock_transition_count;

        let module_ports = self
            .module_ports
            .iter()
            .map(|port| {
                sequential_module_port_toggle_activity(
                    port,
                    &per_node_toggles,
                    transition_count,
                    clock_toggle_count,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        let module_boundaries = self
            .module_boundaries
            .iter()
            .map(|boundary| {
                let ports = boundary
                    .ports
                    .iter()
                    .map(|port| {
                        sequential_module_port_toggle_activity(
                            port,
                            &per_node_toggles,
                            transition_count,
                            clock_toggle_count,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(SequentialModuleBoundaryToggleActivity {
                    instance_path: boundary.instance_path.clone(),
                    module_name: boundary.module_name.clone(),
                    ports,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let instances = self
            .instances
            .iter()
            .map(|instance| {
                let pins = instance
                    .pins
                    .iter()
                    .map(|pin| {
                        Ok(SequentialInstancePinToggleActivity {
                            pin_name: pin.pin_name.clone(),
                            direction: cell_pin_direction(pin.direction)?,
                            connection: toggle_pin_connection(&pin.connection),
                            activity: sequential_signal_toggle_activity(
                                pin.signal,
                                &per_node_toggles,
                                transition_count,
                                clock_toggle_count,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(SequentialInstanceToggleActivity {
                    instance_name: instance.instance_name.clone(),
                    cell_type: instance.cell_type.clone(),
                    pins,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let labeled_aggregate = GvToggleAggregate {
            module_input_toggles: sum_sequential_module_port_toggles(
                &module_ports,
                ToggleDirection::Input,
            ),
            module_output_toggles: sum_sequential_module_port_toggles(
                &module_ports,
                ToggleDirection::Output,
            ),
            cell_input_pin_toggles: sum_sequential_instance_pin_toggles(
                &instances,
                ToggleDirection::Input,
            ),
            cell_output_pin_toggles: sum_sequential_instance_pin_toggles(
                &instances,
                ToggleDirection::Output,
            ),
        };
        let clock = self.clock.as_ref().map(|clock| GvSequentialClockActivity {
            port_name: clock.port_name.clone(),
            active_edge: clock.active_edge,
            active_edge_count: cycle_count,
            toggle_count: clock_toggle_count,
        });

        Ok(GvSequentialToggleActivity {
            module_name: self.module_name.clone(),
            sequential,
            clock,
            labeled_aggregate,
            module_ports,
            module_boundaries,
            instances,
        })
    }

    fn lower_ir_values(&self, args: &[IrValue]) -> Result<Vec<Vec<IrBits>>, String> {
        args.iter()
            .enumerate()
            .map(|(cycle_index, sample)| {
                self.lower_arg_tuple(sample)
                    .map_err(|error| format!("input cycle {}: {}", cycle_index + 1, error))
            })
            .collect()
    }

    fn lower_arg_tuple(&self, args: &IrValue) -> Result<Vec<IrBits>, String> {
        let elements = args
            .get_elements()
            .map_err(|error| format!("argument value is not a tuple: {error}"))?;
        let inputs = elements
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value.to_bits().map_err(|error| {
                    let input_name = self
                        .sequential_gate_fn
                        .inputs
                        .get(index)
                        .map(|input_id| {
                            self.sequential_gate_fn.transition.inputs[input_id.index()]
                                .name
                                .as_str()
                        })
                        .unwrap_or("<extra>");
                    format!(
                        "argument {} for input '{}' is not bits-typed: {}",
                        index, input_name, error
                    )
                })
            })
            .collect::<Result<Vec<IrBits>, String>>()?;
        self.validate_input_bits(&inputs)?;
        Ok(inputs)
    }

    fn validate_input_bits(&self, inputs: &[IrBits]) -> Result<(), String> {
        let design = &self.sequential_gate_fn;
        if inputs.len() != design.inputs.len() {
            let input_names = design
                .inputs
                .iter()
                .map(|input_id| design.transition.inputs[input_id.index()].name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "module '{}' expects {} external input values [{}], got {}",
                self.module_name,
                design.inputs.len(),
                input_names,
                inputs.len()
            ));
        }
        for (index, (input, input_id)) in inputs.iter().zip(&design.inputs).enumerate() {
            let expected = &design.transition.inputs[input_id.index()];
            let actual_width = input.get_bit_count();
            let expected_width = expected.get_bit_count();
            if actual_width != expected_width {
                return Err(format!(
                    "argument {} for input '{}' has width {}, expected {}",
                    index, expected.name, actual_width, expected_width
                ));
            }
        }
        Ok(())
    }
}

fn module_port_direction(direction: &PortDirection) -> Result<ToggleDirection, String> {
    match direction {
        PortDirection::Input => Ok(ToggleDirection::Input),
        PortDirection::Output => Ok(ToggleDirection::Output),
        PortDirection::Inout => Err("inout module ports cannot be counted by gv-eval".to_string()),
    }
}

fn module_port_toggle_activity(
    port: &ModulePortAigBinding,
    per_node_toggles: &[usize],
    transition_count: usize,
) -> Result<ModulePortToggleActivity, String> {
    let direction = module_port_direction(&port.direction)?;
    let bits_lsb_to_msb = port
        .bits_lsb_to_msb
        .iter()
        .map(|bit| {
            let (toggle_count, toggle_rate) =
                operand_toggle_stats(bit.operand, per_node_toggles, transition_count)?;
            Ok(ModulePortBitToggleActivity {
                bit_number: bit.bit_number,
                toggle_count,
                toggle_rate,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ModulePortToggleActivity {
        port_name: port.name.clone(),
        direction,
        bits_lsb_to_msb,
    })
}

fn cell_pin_direction(direction: PinDirection) -> Result<ToggleDirection, String> {
    match direction {
        PinDirection::Input => Ok(ToggleDirection::Input),
        PinDirection::Output => Ok(ToggleDirection::Output),
        PinDirection::Invalid => {
            Err("invalid standard-cell pin direction cannot be counted by gv-eval".to_string())
        }
    }
}

fn toggle_pin_connection(connection: &PinConnection) -> TogglePinConnection {
    match connection {
        PinConnection::Net {
            net_name,
            bit_number,
        } => TogglePinConnection::Net {
            net_name: net_name.clone(),
            bit_number: *bit_number,
        },
        PinConnection::Literal(value) => TogglePinConnection::Literal { value: *value },
        PinConnection::Unconnected => TogglePinConnection::Unconnected,
    }
}

fn operand_toggle_stats(
    operand: AigOperand,
    per_node_toggles: &[usize],
    transition_count: usize,
) -> Result<(usize, f64), String> {
    let toggle_count = per_node_toggles
        .get(operand.node.id)
        .copied()
        .ok_or_else(|| format!("AIG operand node {} is out of range", operand.node.id))?;
    Ok((toggle_count, toggle_count as f64 / transition_count as f64))
}

fn sequential_signal_toggle_activity(
    signal: SequentialAigSignal,
    per_node_toggles: &[usize],
    transition_count: usize,
    clock_toggle_count: usize,
) -> Result<GvSequentialSignalToggleActivity, String> {
    match signal {
        SequentialAigSignal::Operand(operand) => {
            let (toggle_count, toggle_rate) =
                operand_toggle_stats(operand, per_node_toggles, transition_count)?;
            Ok(GvSequentialSignalToggleActivity::Sampled {
                toggle_count,
                toggle_rate,
            })
        }
        SequentialAigSignal::Clock => Ok(GvSequentialSignalToggleActivity::Clock {
            toggle_count: clock_toggle_count,
        }),
    }
}

/// Builds one source-labeled sequential module-port activity report.
fn sequential_module_port_toggle_activity(
    port: &SequentialModulePortAigBinding,
    per_node_toggles: &[usize],
    transition_count: usize,
    clock_toggle_count: usize,
) -> Result<SequentialModulePortToggleActivity, String> {
    let direction = module_port_direction(&port.direction)?;
    let bits_lsb_to_msb = port
        .bits_lsb_to_msb
        .iter()
        .map(|bit| {
            Ok(SequentialModulePortBitToggleActivity {
                bit_number: bit.bit_number,
                activity: sequential_signal_toggle_activity(
                    bit.signal,
                    per_node_toggles,
                    transition_count,
                    clock_toggle_count,
                )?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SequentialModulePortToggleActivity {
        port_name: port.name.clone(),
        direction,
        bits_lsb_to_msb,
    })
}

fn sequential_signal_toggle_count(activity: &GvSequentialSignalToggleActivity) -> usize {
    match activity {
        GvSequentialSignalToggleActivity::Sampled { toggle_count, .. } => *toggle_count,
        GvSequentialSignalToggleActivity::Clock { toggle_count } => *toggle_count,
    }
}

fn sum_sequential_module_port_toggles(
    ports: &[SequentialModulePortToggleActivity],
    direction: ToggleDirection,
) -> usize {
    ports
        .iter()
        .filter(|port| port.direction == direction)
        .flat_map(|port| port.bits_lsb_to_msb.iter())
        .map(|bit| sequential_signal_toggle_count(&bit.activity))
        .sum()
}

fn sum_sequential_instance_pin_toggles(
    instances: &[SequentialInstanceToggleActivity],
    direction: ToggleDirection,
) -> usize {
    instances
        .iter()
        .flat_map(|instance| instance.pins.iter())
        .filter(|pin| pin.direction == direction)
        .map(|pin| sequential_signal_toggle_count(&pin.activity))
        .sum()
}

fn sum_module_port_toggles(
    ports: &[ModulePortToggleActivity],
    direction: ToggleDirection,
) -> usize {
    ports
        .iter()
        .filter(|port| port.direction == direction)
        .flat_map(|port| port.bits_lsb_to_msb.iter())
        .map(|bit| bit.toggle_count)
        .sum()
}

fn sum_instance_pin_toggles(
    instances: &[InstanceToggleActivity],
    direction: ToggleDirection,
) -> usize {
    instances
        .iter()
        .flat_map(|instance| instance.pins.iter())
        .filter(|pin| pin.direction == direction)
        .map(|pin| pin.toggle_count)
        .sum()
}

fn outputs_to_ir_value(outputs: &[IrBits]) -> IrValue {
    if outputs.len() == 1 {
        IrValue::from_bits(&outputs[0])
    } else {
        IrValue::make_tuple(
            &outputs
                .iter()
                .map(IrValue::from_bits)
                .collect::<Vec<IrValue>>(),
        )
    }
}
