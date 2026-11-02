// SPDX-License-Identifier: Apache-2.0

//! Construction of a sequential pipeline from combinational gate functions.

use std::collections::BTreeSet;
use std::ops::Range;

use xlsynth_pir::IrBits;

use crate::aig::gate::{AigBitVector, AigOperand, GateFn};
use crate::aig::import::import_gate_fn;
use crate::aig::sequential_gate::{
    ClockPort, RegisterBinding, RegisterControl, SequentialGateFn, SynchronousReset,
    TransitionInputId, TransitionOutputId, canonical_register_d_name, canonical_register_q_name,
    canonical_transition_name, uniquify_transition_port_name,
};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

/// Controls register insertion while combining combinational stage functions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SequentialPipelineOptions {
    pub name: String,
    pub clock: ClockPort,
    pub flop_inputs: bool,
    pub flop_outputs: bool,
    pub input_valid_signal: Option<String>,
    pub output_valid_signal: Option<String>,
    pub reset_signal: Option<String>,
    pub reset_active_low: bool,
}

/// A stitched transition graph and the stage responsible for each register's D.
pub(crate) struct StitchedGatePipeline {
    pub design: SequentialGateFn,
    pub register_stages: Vec<usize>,
}

/// One contiguous packed register or an omitted enable bit at a stage boundary.
pub(crate) struct BankSegment {
    pub bits: Range<usize>,
    pub enable: Option<usize>,
    pub store: bool,
}

/// Stitches combinational gate functions into one sequential pipeline design.
///
/// Every adjacent pair of stages receives one register layer. Stage output bits
/// are flattened LSB-first and partitioned across the next stage inputs in
/// their declared order.
pub fn stitch_gate_fns_into_pipeline(
    stages: &[GateFn],
    options: &SequentialPipelineOptions,
) -> Result<SequentialGateFn, String> {
    Ok(stitch_gate_fns_with_layout(stages, options)?.design)
}

/// Retains the stage boundaries while lowering controls to effective D.
pub(crate) fn stitch_gate_fns_with_layout(
    stages: &[GateFn],
    options: &SequentialPipelineOptions,
) -> Result<StitchedGatePipeline, String> {
    stitch_with_banks(stages, options, &[])
}

/// Stitches the proved interstage bank segments and lowers their controls to D.
pub(crate) fn stitch_with_banks(
    stages: &[GateFn],
    options: &SequentialPipelineOptions,
    banks: &[Vec<BankSegment>],
) -> Result<StitchedGatePipeline, String> {
    let mut stitched = assemble_pipeline(stages, options, banks)?;
    stitched.design = stitched.design.legalize_register_controls()?;
    Ok(stitched)
}

/// Stitches stages while keeping valid enables and synchronous resets explicit.
///
/// Data registers hold when valid is deasserted. Valid registers load every
/// edge and reset to zero; reset does not initialize the data registers.
pub fn stitch_gate_fns_into_pipeline_with_controls(
    stages: &[GateFn],
    options: &SequentialPipelineOptions,
) -> Result<SequentialGateFn, String> {
    Ok(assemble_pipeline(stages, options, &[])?.design)
}

/// Builds register bindings in stage order, including optional wrapper flops.
fn assemble_pipeline(
    stages: &[GateFn],
    options: &SequentialPipelineOptions,
    banks: &[Vec<BankSegment>],
) -> Result<StitchedGatePipeline, String> {
    if stages.is_empty() {
        return Err("cannot stitch an empty stage list into a pipeline".to_string());
    }
    if options.output_valid_signal.is_some() && options.input_valid_signal.is_none() {
        return Err("output_valid_signal requires input_valid_signal".to_string());
    }
    if options.reset_signal.is_some() && options.input_valid_signal.is_none() {
        return Err("reset_signal requires input_valid_signal".to_string());
    }
    verify_adjacent_flat_widths(stages)?;

    let mut builder = GateBuilder::new(
        canonical_transition_name(&options.name),
        GateBuilderOptions::no_opt(),
    );
    let mut external_inputs = Vec::new();
    let mut external_outputs = Vec::new();
    let mut registers = Vec::new();
    let mut register_stages = Vec::new();
    let mut register_names = BTreeSet::new();
    let mut transition_input_names = BTreeSet::new();
    let mut transition_output_names = stages
        .last()
        .expect("non-empty stage list checked above")
        .outputs
        .iter()
        .map(|output| output.name.clone())
        .collect::<BTreeSet<String>>();
    if transition_output_names.contains(&options.clock.name) {
        return Err(format!(
            "clock name '{}' collides with a stage output",
            options.clock.name
        ));
    }
    if let Some(output_valid_signal) = &options.output_valid_signal {
        if !transition_output_names.insert(output_valid_signal.clone()) {
            return Err(format!(
                "output valid signal '{}' collides with a stage output",
                output_valid_signal
            ));
        }
    }

    let mut input_bindings = Vec::new();
    for input in &stages[0].inputs {
        if input.name == options.clock.name {
            return Err(format!(
                "clock name '{}' collides with a stage input",
                options.clock.name
            ));
        }
        if !transition_input_names.insert(input.name.clone()) {
            return Err(format!("duplicate pipeline input name '{}'", input.name));
        }
        let binding = builder.add_input(input.name.clone(), input.get_bit_count());
        external_inputs.push(TransitionInputId::new(builder.inputs.len() - 1));
        input_bindings.push(binding);
    }
    let mut current_valid = if let Some(input_valid_signal) = &options.input_valid_signal {
        if input_valid_signal == &options.clock.name {
            return Err(format!(
                "input valid signal '{}' collides with the clock",
                input_valid_signal
            ));
        }
        if !transition_input_names.insert(input_valid_signal.clone()) {
            return Err(format!(
                "input valid signal '{}' collides with a stage input",
                input_valid_signal
            ));
        }
        let valid = builder.add_input(input_valid_signal.clone(), 1);
        external_inputs.push(TransitionInputId::new(builder.inputs.len() - 1));
        Some(*valid.get_lsb(0))
    } else {
        None
    };
    let reset = if let Some(reset_signal) = &options.reset_signal {
        if reset_signal == &options.clock.name {
            return Err(format!(
                "reset signal '{}' collides with the clock",
                reset_signal
            ));
        }
        if !transition_input_names.insert(reset_signal.clone()) {
            return Err(format!(
                "reset signal '{}' collides with another pipeline input",
                reset_signal
            ));
        }
        let reset = builder.add_input(reset_signal.clone(), 1);
        external_inputs.push(TransitionInputId::new(builder.inputs.len() - 1));
        Some(*reset.get_lsb(0))
    } else {
        None
    };
    if options.flop_inputs {
        let input_names = stages[0]
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<&str>>();
        let layer = add_register_layer(
            0,
            &input_names,
            &input_bindings,
            current_valid,
            reset,
            options,
            &mut builder,
            &mut register_names,
            &mut transition_input_names,
            &mut transition_output_names,
            &mut registers,
        );
        input_bindings = layer.data_values;
        current_valid = layer.valid;
    }

    for (stage_index, stage) in stages.iter().enumerate() {
        let imported_outputs = import_gate_fn(stage, &input_bindings, &mut builder)?;
        if stage_index + 1 == stages.len() {
            let mut output_values = imported_outputs;
            if options.flop_outputs {
                let output_names = stage
                    .outputs
                    .iter()
                    .map(|output| output.name.as_str())
                    .collect::<Vec<&str>>();
                let layer = add_register_layer(
                    stages.len() - 1 + usize::from(options.flop_inputs),
                    &output_names,
                    &output_values,
                    current_valid,
                    reset,
                    options,
                    &mut builder,
                    &mut register_names,
                    &mut transition_input_names,
                    &mut transition_output_names,
                    &mut registers,
                );
                output_values = layer.data_values;
                current_valid = layer.valid;
            }
            for (output, bit_vector) in stage.outputs.iter().zip(output_values) {
                builder.add_output(output.name.clone(), bit_vector);
                external_outputs.push(TransitionOutputId::new(builder.outputs.len() - 1));
            }
            if let (Some(output_valid_signal), Some(valid)) =
                (&options.output_valid_signal, current_valid)
            {
                builder.add_output(output_valid_signal.clone(), AigBitVector::from_bit(valid));
                external_outputs.push(TransitionOutputId::new(builder.outputs.len() - 1));
            }
            register_stages.resize(registers.len(), stage_index);
            continue;
        }

        let flat_output = flatten_bit_vectors(&imported_outputs);
        let next_stage = &stages[stage_index + 1];
        let register_layer = stage_index + usize::from(options.flop_inputs);
        if let Some(segments) = banks.get(stage_index).filter(|s| !s.is_empty()) {
            let mut bits = Vec::with_capacity(flat_output.get_bit_count());
            for segment in segments {
                if !segment.store {
                    bits.push(builder.get_false());
                    continue;
                }
                let source = flat_output.get_lsb_slice(segment.bits.start, segment.bits.len());
                let load = segment
                    .enable
                    .map(|i| {
                        let en = *flat_output.get_lsb(i);
                        current_valid.map_or(en, |v| builder.add_and_binary(v, en))
                    })
                    .or(current_valid);
                let enable = load.map(|en| {
                    add_control_endpoint(
                        &mut builder,
                        &mut transition_output_names,
                        &format!("p{register_layer}_bank_at_{}__enable", segment.bits.start),
                        en,
                        false,
                    )
                });
                let q = add_data_register(
                    register_layer,
                    &format!("bank_at_{}", segment.bits.start),
                    &source,
                    enable,
                    &mut builder,
                    &mut register_names,
                    &mut transition_input_names,
                    &mut transition_output_names,
                    &mut registers,
                );
                bits.extend(q.iter_lsb_to_msb().copied());
            }
            let registered = AigBitVector::from_lsb_is_index_0(&bits);
            let mut start = 0;
            input_bindings = next_stage
                .inputs
                .iter()
                .map(|port| {
                    let slice = registered.get_lsb_slice(start, port.get_bit_count());
                    start += port.get_bit_count();
                    slice
                })
                .collect();
            current_valid = add_register_layer(
                register_layer,
                &[],
                &[],
                current_valid,
                reset,
                options,
                &mut builder,
                &mut register_names,
                &mut transition_input_names,
                &mut transition_output_names,
                &mut registers,
            )
            .valid;
            register_stages.resize(registers.len(), stage_index);
            continue;
        }
        let mut offset = 0usize;
        let mut boundary_values = Vec::with_capacity(next_stage.inputs.len());
        for next_input in &next_stage.inputs {
            boundary_values.push(flat_output.get_lsb_slice(offset, next_input.get_bit_count()));
            offset += next_input.get_bit_count();
        }
        let input_names = next_stage
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<&str>>();
        let layer = add_register_layer(
            register_layer,
            &input_names,
            &boundary_values,
            current_valid,
            reset,
            options,
            &mut builder,
            &mut register_names,
            &mut transition_input_names,
            &mut transition_output_names,
            &mut registers,
        );
        input_bindings = layer.data_values;
        current_valid = layer.valid;
        // The first stage also checks the optional wrapper input flops.
        register_stages.resize(registers.len(), stage_index);
    }

    let clock = if registers.is_empty() {
        None
    } else {
        Some(options.clock.clone())
    };
    let result = SequentialGateFn::new(
        options.name.clone(),
        builder.build(),
        external_inputs,
        external_outputs,
        clock,
        registers,
    )?;
    result.validate()?;
    Ok(StitchedGatePipeline {
        design: result,
        register_stages,
    })
}

struct RegisterLayer {
    data_values: Vec<AigBitVector>,
    valid: Option<AigOperand>,
}

#[allow(clippy::too_many_arguments)]
fn add_register_layer(
    layer_number: usize,
    data_names: &[&str],
    source_values: &[AigBitVector],
    current_valid: Option<AigOperand>,
    reset: Option<AigOperand>,
    options: &SequentialPipelineOptions,
    builder: &mut GateBuilder,
    register_names: &mut BTreeSet<String>,
    transition_input_names: &mut BTreeSet<String>,
    transition_output_names: &mut BTreeSet<String>,
    registers: &mut Vec<RegisterBinding>,
) -> RegisterLayer {
    let enable = current_valid
        .filter(|_| !source_values.is_empty())
        .map(|valid| {
            add_control_endpoint(
                builder,
                transition_output_names,
                &format!("p{layer_number}__enable"),
                valid,
                false,
            )
        });
    let data_values = data_names
        .iter()
        .zip(source_values)
        .map(|(name, source)| {
            add_data_register(
                layer_number,
                name,
                source,
                enable,
                builder,
                register_names,
                transition_input_names,
                transition_output_names,
                registers,
            )
        })
        .collect();

    let valid = current_valid.map(|valid| {
        let input_valid_signal = options
            .input_valid_signal
            .as_deref()
            .expect("valid register requires input valid signal");
        let register_name = uniquify_register_name(
            &format!("p{layer_number}_{input_valid_signal}"),
            register_names,
        );
        let q_name = uniquify_transition_port_name(
            &canonical_register_q_name(&register_name),
            transition_input_names,
        );
        let q_value = builder.add_input(q_name, 1);
        let q = TransitionInputId::new(builder.inputs.len() - 1);
        let reset = reset.map(|reset| SynchronousReset {
            control: add_control_endpoint(
                builder,
                transition_output_names,
                &format!("p{layer_number}__reset"),
                reset,
                options.reset_active_low,
            ),
            value: IrBits::from_lsb_is_0(&[false]),
        });
        let d_value = AigBitVector::from_bit(valid);
        let d_name = uniquify_transition_port_name(
            &canonical_register_d_name(&register_name),
            transition_output_names,
        );
        builder.add_output(d_name, d_value);
        let d = TransitionOutputId::new(builder.outputs.len() - 1);
        registers.push(RegisterBinding {
            name: register_name,
            q,
            d,
            enable: None,
            reset,
        });
        *q_value.get_lsb(0)
    });

    RegisterLayer { data_values, valid }
}

/// Names a shared scalar control as part of the combinational interface.
/// Binds one data register to its source and optional load-enable endpoint.
#[allow(clippy::too_many_arguments)]
fn add_data_register(
    layer: usize,
    name: &str,
    source: &AigBitVector,
    enable: Option<RegisterControl>,
    builder: &mut GateBuilder,
    register_names: &mut BTreeSet<String>,
    input_names: &mut BTreeSet<String>,
    output_names: &mut BTreeSet<String>,
    registers: &mut Vec<RegisterBinding>,
) -> AigBitVector {
    let name = uniquify_register_name(&format!("p{layer}_{name}"), register_names);
    let q_name = uniquify_transition_port_name(&canonical_register_q_name(&name), input_names);
    let q_value = builder.add_input(q_name, source.get_bit_count());
    let q = TransitionInputId::new(builder.inputs.len() - 1);
    let d_name = uniquify_transition_port_name(&canonical_register_d_name(&name), output_names);
    let d = TransitionOutputId::new(builder.outputs.len());
    builder.add_output(d_name, source.clone());
    registers.push(RegisterBinding {
        name,
        q,
        d,
        enable,
        reset: None,
    });
    q_value
}

fn add_control_endpoint(
    builder: &mut GateBuilder,
    names: &mut BTreeSet<String>,
    preferred_name: &str,
    bit: AigOperand,
    active_low: bool,
) -> RegisterControl {
    let name = uniquify_transition_port_name(preferred_name, names);
    let signal = TransitionOutputId::new(builder.outputs.len());
    builder.add_output(name, AigBitVector::from_bit(bit));
    RegisterControl { signal, active_low }
}

fn verify_adjacent_flat_widths(stages: &[GateFn]) -> Result<(), String> {
    for (index, pair) in stages.windows(2).enumerate() {
        let output_width = pair[0]
            .outputs
            .iter()
            .map(|output| output.get_bit_count())
            .sum::<usize>();
        let input_width = pair[1]
            .inputs
            .iter()
            .map(|input| input.get_bit_count())
            .sum::<usize>();
        if output_width != input_width {
            return Err(format!(
                "cannot stitch stages {} and {}: output width {} does not match input width {}",
                index,
                index + 1,
                output_width,
                input_width
            ));
        }
    }
    Ok(())
}

fn flatten_bit_vectors(bit_vectors: &[AigBitVector]) -> AigBitVector {
    let bits = bit_vectors
        .iter()
        .flat_map(|bit_vector| bit_vector.iter_lsb_to_msb().copied())
        .collect::<Vec<AigOperand>>();
    AigBitVector::from_lsb_is_index_0(&bits)
}

fn uniquify_register_name(preferred_name: &str, used_names: &mut BTreeSet<String>) -> String {
    if used_names.insert(preferred_name.to_string()) {
        return preferred_name.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{preferred_name}__{suffix}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix sequence must provide a unique register name")
}
