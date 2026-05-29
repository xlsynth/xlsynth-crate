// SPDX-License-Identifier: Apache-2.0

//! Construction of a sequential pipeline from combinational gate functions.

use std::collections::BTreeSet;

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn};
use crate::aig::sequential_gate::{
    ClockPort, RegisterBinding, SequentialGateFn, TransitionInputId, TransitionOutputId,
    canonical_register_d_name, canonical_register_q_name, canonical_transition_name,
    uniquify_transition_port_name,
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

/// Stitches combinational gate functions into one sequential pipeline design.
///
/// Every adjacent pair of stages receives one register layer. Stage output bits
/// are flattened LSB-first and partitioned across the next stage inputs in
/// their declared order.
pub fn stitch_gate_fns_into_pipeline(
    stages: &[GateFn],
    options: &SequentialPipelineOptions,
) -> Result<SequentialGateFn, String> {
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
            continue;
        }

        let flat_output = flatten_bit_vectors(&imported_outputs);
        let next_stage = &stages[stage_index + 1];
        let register_layer = stage_index + usize::from(options.flop_inputs);
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
    Ok(result)
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
    let mut data_values = Vec::with_capacity(source_values.len());
    for (data_name, source_value) in data_names.iter().zip(source_values) {
        let register_name =
            uniquify_register_name(&format!("p{layer_number}_{data_name}"), register_names);
        let q_name = uniquify_transition_port_name(
            &canonical_register_q_name(&register_name),
            transition_input_names,
        );
        let q_value = builder.add_input(q_name, source_value.get_bit_count());
        let q = TransitionInputId::new(builder.inputs.len() - 1);
        let d_value = match current_valid {
            Some(valid) => builder.add_mux2_vec(&valid, source_value, &q_value),
            None => source_value.clone(),
        };
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
            initial_value: None,
        });
        data_values.push(q_value);
    }

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
        let d_bit = match reset {
            Some(reset) if options.reset_active_low => builder.add_and_binary(valid, reset),
            Some(reset) => builder.add_and_binary(valid, reset.negate()),
            None => valid,
        };
        let d_value = AigBitVector::from_bit(d_bit);
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
            initial_value: None,
        });
        *q_value.get_lsb(0)
    });

    RegisterLayer { data_values, valid }
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

fn import_gate_fn(
    source: &GateFn,
    input_bindings: &[AigBitVector],
    builder: &mut GateBuilder,
) -> Result<Vec<AigBitVector>, String> {
    if source.inputs.len() != input_bindings.len() {
        return Err(format!(
            "stage '{}' expects {} input port(s), but {} bindings were supplied",
            source.name,
            source.inputs.len(),
            input_bindings.len()
        ));
    }
    let mut imported_nodes = vec![None; source.gates.len()];
    for (input, binding) in source.inputs.iter().zip(input_bindings) {
        if input.get_bit_count() != binding.get_bit_count() {
            return Err(format!(
                "stage '{}' input '{}' expects {} bit(s), but its binding has {}",
                source.name,
                input.name,
                input.get_bit_count(),
                binding.get_bit_count()
            ));
        }
        for (source_bit, binding_bit) in input
            .bit_vector
            .iter_lsb_to_msb()
            .zip(binding.iter_lsb_to_msb())
        {
            if source_bit.negated {
                return Err(format!(
                    "stage '{}' input '{}' contains a negated input operand",
                    source.name, input.name
                ));
            }
            builder.add_pir_node_ids(
                binding_bit.node,
                source.gates[source_bit.node.id].get_pir_node_ids(),
            );
            imported_nodes[source_bit.node.id] = Some(*binding_bit);
        }
    }
    for operand in source.post_order_operands(/* discard_inputs= */ false) {
        if imported_nodes[operand.node.id].is_some() {
            continue;
        }
        let source_node = &source.gates[operand.node.id];
        let imported = match source_node {
            AigNode::Input { name, .. } => {
                return Err(format!(
                    "stage '{}' references unbound input node '{}'",
                    source.name, name
                ));
            }
            AigNode::Literal { value, .. } => {
                if *value {
                    builder.get_true()
                } else {
                    builder.get_false()
                }
            }
            AigNode::And2 { a, b, tags, .. } => {
                let imported_a = mapped_operand(source, *a, &imported_nodes)?;
                let imported_b = mapped_operand(source, *b, &imported_nodes)?;
                let imported = builder.add_and_binary(imported_a, imported_b);
                if let Some(tags) = tags {
                    for tag in tags {
                        builder.add_tag(imported.node, tag.clone());
                    }
                }
                imported
            }
        };
        builder.add_pir_node_ids(imported.node, source_node.get_pir_node_ids());
        imported_nodes[operand.node.id] = Some(imported);
    }
    source
        .outputs
        .iter()
        .map(|output| {
            let imported_bits = output
                .bit_vector
                .iter_lsb_to_msb()
                .map(|bit| mapped_operand(source, *bit, &imported_nodes))
                .collect::<Result<Vec<AigOperand>, String>>()?;
            Ok(AigBitVector::from_lsb_is_index_0(&imported_bits))
        })
        .collect()
}

fn mapped_operand(
    source: &GateFn,
    operand: AigOperand,
    imported_nodes: &[Option<AigOperand>],
) -> Result<AigOperand, String> {
    let base = imported_nodes[operand.node.id].ok_or_else(|| {
        format!(
            "stage '{}' does not have an imported value for AIG node {}",
            source.name, operand.node.id
        )
    })?;
    if operand.negated {
        Ok(base.negate())
    } else {
        Ok(base)
    }
}
