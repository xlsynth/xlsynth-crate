// SPDX-License-Identifier: Apache-2.0

//! Bit-granular, initialization-aware cleanup of ordinary sequential AIGs.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use xlsynth::IrBits;

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn};
use crate::aig::sequential_gate::{
    RegisterBinding, SequentialGateFn, TransitionInputId, TransitionOutputId,
};
use crate::aig_serdes::gate2ir::{GateFnInterfaceSchema, repack_gate_fn_interface_with_schema};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

/// Determines whether unspecified register initialization must be preserved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegisterInitializationPolicy {
    /// Preserve explicit initial values and the simulator's all-zero startup.
    #[default]
    PreserveCycleZero,
    /// Treat absent initial values as unspecified, as in hardware synthesis.
    UninitializedDontCare,
}

/// Controls deterministic sequential state cleanup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SequentialCleanupOptions {
    /// Specifies whether absent register initialization may be optimized.
    pub initialization_policy: RegisterInitializationPolicy,
}

/// Counts state and AIG reductions across all fixed-point cleanup rounds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SequentialCleanupStats {
    /// Number of individual register bits before cleanup.
    pub initial_register_bits: usize,
    /// Number of individual register bits after cleanup.
    pub final_register_bits: usize,
    /// Register bits absent from every observable sequential state cone.
    pub dead_register_bits: usize,
    /// Register bits replaced by their proven constant values.
    pub constant_register_bits: usize,
    /// Register bits merged with equal or complementary next-state bits.
    pub merged_register_bits: usize,
    /// Number of AIG AND nodes before cleanup.
    pub initial_and_nodes: usize,
    /// Number of AIG AND nodes after cleanup.
    pub final_and_nodes: usize,
    /// Number of state-classification rounds, including the final fixed point.
    pub iterations: usize,
}

/// Returns the cleaned sequential transition and reproducible reduction data.
#[derive(Debug, Clone)]
pub struct SequentialCleanupResult {
    /// Sequential design with matching cleaned transition and state metadata.
    pub design: SequentialGateFn,
    /// Bit-granular register and combinational cleanup accounting.
    pub stats: SequentialCleanupStats,
}

#[derive(Debug, Clone, Copy)]
struct StateBit {
    q: AigOperand,
    d: AigOperand,
    explicit_initial: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateAction {
    Keep,
    Dead,
    Constant(bool),
    Duplicate {
        representative: usize,
        negated: bool,
    },
}

/// Rebinds an optimized, scalar AIGER transition to native packed state ports.
pub fn rebind_sequential_transition(
    design: &SequentialGateFn,
    optimized_transition: &GateFn,
) -> Result<SequentialGateFn, String> {
    design.validate()?;
    validate_flattened_interface(&design.transition, optimized_transition)?;
    let schema = GateFnInterfaceSchema::from_gate_fn(&design.transition)?;
    let mut transition =
        repack_gate_fn_interface_with_schema(optimized_transition.clone(), &schema)?;
    transition.name.clone_from(&design.transition.name);
    SequentialGateFn::new(
        design.name.clone(),
        transition,
        design.inputs.clone(),
        design.outputs.clone(),
        design.clock.clone(),
        design.registers.clone(),
    )
}

/// Rebinds an optimized ordinary AIG and cleans its sequential state.
pub fn cleanup_sequential_transition(
    design: &SequentialGateFn,
    optimized_transition: &GateFn,
    options: &SequentialCleanupOptions,
) -> Result<SequentialCleanupResult, String> {
    let rebound = rebind_sequential_transition(design, optimized_transition)?;
    cleanup_sequential_gate_fn(&rebound, options)
}

/// Removes dead, constant, and redundant register bits to a fixed point.
pub fn cleanup_sequential_gate_fn(
    design: &SequentialGateFn,
    options: &SequentialCleanupOptions,
) -> Result<SequentialCleanupResult, String> {
    design.validate()?;
    let mut current = design.clone();
    let mut stats = SequentialCleanupStats {
        initial_register_bits: register_bit_count(design),
        initial_and_nodes: count_and_nodes(&design.transition),
        ..SequentialCleanupStats::default()
    };

    while !current.registers.is_empty() {
        stats.iterations += 1;
        let (state_bits, bits_by_register) = collect_state_bits(&current)?;
        let live_bits = mark_live_state_bits(&current, &state_bits);
        let mut actions = vec![StateAction::Keep; state_bits.len()];

        for (index, bit) in state_bits.iter().enumerate() {
            if !live_bits[index] {
                actions[index] = StateAction::Dead;
                continue;
            }

            if let Some(value) = removable_constant(&current.transition, bit, options) {
                actions[index] = StateAction::Constant(value);
            }
        }

        mark_duplicate_state_bits(&state_bits, &mut actions, options);

        let dead_count = actions
            .iter()
            .filter(|action| matches!(action, StateAction::Dead))
            .count();
        let constant_count = actions
            .iter()
            .filter(|action| matches!(action, StateAction::Constant(_)))
            .count();
        let duplicate_count = actions
            .iter()
            .filter(|action| matches!(action, StateAction::Duplicate { .. }))
            .count();

        if dead_count + constant_count + duplicate_count == 0 {
            break;
        }

        stats.dead_register_bits += dead_count;
        stats.constant_register_bits += constant_count;
        stats.merged_register_bits += duplicate_count;
        current = rebuild_cleaned_design(&current, &state_bits, &bits_by_register, &actions)?;
    }

    stats.final_register_bits = register_bit_count(&current);
    stats.final_and_nodes = count_and_nodes(&current.transition);
    current.validate()?;
    Ok(SequentialCleanupResult {
        design: current,
        stats,
    })
}

/// Counts physical state bits rather than packed register declarations.
fn register_bit_count(design: &SequentialGateFn) -> usize {
    design
        .registers
        .iter()
        .map(|register| design.transition.inputs[register.q.index()].get_bit_count())
        .sum()
}

/// Counts actual AND nodes without counting AIG inputs or constants.
fn count_and_nodes(graph: &GateFn) -> usize {
    graph
        .gates
        .iter()
        .filter(|node| matches!(node, AigNode::And2 { .. }))
        .count()
}

/// Flattens packed register bindings while retaining their original bit order.
fn collect_state_bits(
    design: &SequentialGateFn,
) -> Result<(Vec<StateBit>, Vec<Vec<usize>>), String> {
    let mut result = Vec::with_capacity(register_bit_count(design));
    let mut bits_by_register = Vec::with_capacity(design.registers.len());
    for register in &design.registers {
        let q = &design.transition.inputs[register.q.index()];
        let d = &design.transition.outputs[register.d.index()];
        let mut register_bits = Vec::with_capacity(q.get_bit_count());
        for bit_index in 0..q.get_bit_count() {
            let explicit_initial = register
                .initial_value
                .as_ref()
                .map(|initial| {
                    initial.get_bit(bit_index).map_err(|error| {
                        format!(
                            "failed to read register '{}' initial bit {}: {}",
                            register.name, bit_index, error
                        )
                    })
                })
                .transpose()?;
            register_bits.push(result.len());
            result.push(StateBit {
                q: *q.bit_vector.get_lsb(bit_index),
                d: *d.bit_vector.get_lsb(bit_index),
                explicit_initial,
            });
        }
        bits_by_register.push(register_bits);
    }
    Ok((result, bits_by_register))
}

/// Marks state reachable from visible outputs through any number of cycles.
fn mark_live_state_bits(design: &SequentialGateFn, state_bits: &[StateBit]) -> Vec<bool> {
    let graph = &design.transition;
    let mut state_by_node = vec![None; graph.gates.len()];
    for (index, bit) in state_bits.iter().enumerate() {
        state_by_node[bit.q.node.id] = Some(index);
    }

    let mut stack = Vec::new();
    for output_id in &design.outputs {
        stack.extend(
            graph.outputs[output_id.index()]
                .bit_vector
                .iter_lsb_to_msb()
                .map(|operand| operand.node.id),
        );
    }

    let mut visited = vec![false; graph.gates.len()];
    let mut live = vec![false; state_bits.len()];
    while let Some(node_id) = stack.pop() {
        if visited[node_id] {
            continue;
        }
        visited[node_id] = true;
        if let Some(state_index) = state_by_node[node_id] {
            live[state_index] = true;
            stack.push(state_bits[state_index].d.node.id);
        }
        stack.extend(
            graph.gates[node_id]
                .operands()
                .map(|operand| operand.node.id),
        );
    }
    live
}

/// Returns a constant state value only when startup semantics permit removal.
fn removable_constant(
    graph: &GateFn,
    bit: &StateBit,
    options: &SequentialCleanupOptions,
) -> Option<bool> {
    let value = match &graph.gates[bit.d.node.id] {
        AigNode::Literal { value, .. } => Some(*value ^ bit.d.negated),
        _ if bit.d == bit.q => match options.initialization_policy {
            RegisterInitializationPolicy::PreserveCycleZero => {
                Some(bit.explicit_initial.unwrap_or(false))
            }
            RegisterInitializationPolicy::UninitializedDontCare => bit.explicit_initial,
        },
        _ => None,
    }?;

    match options.initialization_policy {
        RegisterInitializationPolicy::PreserveCycleZero
            if bit.explicit_initial.unwrap_or(false) != value =>
        {
            None
        }
        RegisterInitializationPolicy::UninitializedDontCare
            if bit.explicit_initial.is_some_and(|initial| initial != value) =>
        {
            None
        }
        RegisterInitializationPolicy::PreserveCycleZero
        | RegisterInitializationPolicy::UninitializedDontCare => Some(value),
    }
}

/// Merges equal and complementary state using initialization-safe classes.
fn mark_duplicate_state_bits(
    state_bits: &[StateBit],
    actions: &mut [StateAction],
    options: &SequentialCleanupOptions,
) {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, bit) in state_bits.iter().enumerate() {
        if actions[index] == StateAction::Keep {
            groups.entry(bit.d.node.id).or_default().push(index);
        }
    }

    for indices in groups.into_values() {
        if indices.len() < 2 {
            continue;
        }
        match options.initialization_policy {
            RegisterInitializationPolicy::PreserveCycleZero => {
                let mut representatives: BTreeMap<bool, usize> = BTreeMap::new();
                for index in indices {
                    let bit = state_bits[index];
                    let normalized_initial = bit.explicit_initial.unwrap_or(false) ^ bit.d.negated;
                    if let Some(&representative) = representatives.get(&normalized_initial) {
                        actions[index] = StateAction::Duplicate {
                            representative,
                            negated: bit.d.negated ^ state_bits[representative].d.negated,
                        };
                    } else {
                        representatives.insert(normalized_initial, index);
                    }
                }
            }
            RegisterInitializationPolicy::UninitializedDontCare => {
                let mut explicit_representatives: [Option<usize>; 2] = [None, None];
                let mut first_uninitialized = None;
                for &index in &indices {
                    let bit = state_bits[index];
                    if let Some(initial) = bit.explicit_initial {
                        let normalized = usize::from(initial ^ bit.d.negated);
                        if explicit_representatives[normalized].is_none() {
                            explicit_representatives[normalized] = Some(index);
                        }
                    } else if first_uninitialized.is_none() {
                        first_uninitialized = Some(index);
                    }
                }
                let unspecified_representative = explicit_representatives
                    .into_iter()
                    .flatten()
                    .min()
                    .or(first_uninitialized);

                for index in indices {
                    let bit = state_bits[index];
                    let representative = if let Some(initial) = bit.explicit_initial {
                        explicit_representatives[usize::from(initial ^ bit.d.negated)]
                    } else {
                        unspecified_representative
                    };
                    if let Some(representative) = representative
                        && representative != index
                    {
                        actions[index] = StateAction::Duplicate {
                            representative,
                            negated: bit.d.negated ^ state_bits[representative].d.negated,
                        };
                    }
                }
            }
        }
    }
}

/// Rebuilds a state-reduced transition while retaining every external port.
fn rebuild_cleaned_design(
    design: &SequentialGateFn,
    state_bits: &[StateBit],
    bits_by_register: &[Vec<usize>],
    actions: &[StateAction],
) -> Result<SequentialGateFn, String> {
    let graph = &design.transition;
    let mut builder = GateBuilder::new(graph.name.clone(), GateBuilderOptions::opt());
    let mut operands = vec![None; graph.gates.len()];
    let mut external_input_positions = vec![None; graph.inputs.len()];
    let mut register_input_positions = vec![None; graph.inputs.len()];
    for (position, input_id) in design.inputs.iter().enumerate() {
        external_input_positions[input_id.index()] = Some(position);
    }
    for (register_index, register) in design.registers.iter().enumerate() {
        register_input_positions[register.q.index()] = Some(register_index);
    }

    let mut new_external_inputs = vec![None; design.inputs.len()];
    let mut new_register_inputs = vec![None; design.registers.len()];
    for (old_input_index, input) in graph.inputs.iter().enumerate() {
        if let Some(external_position) = external_input_positions[old_input_index] {
            let input_id = TransitionInputId::new(builder.inputs.len());
            let new_bits = builder.add_input(input.name.clone(), input.get_bit_count());
            for (old_bit, new_bit) in input
                .bit_vector
                .iter_lsb_to_msb()
                .zip(new_bits.iter_lsb_to_msb())
            {
                bind_input_operand(&mut builder, &mut operands, graph, *old_bit, *new_bit)?;
            }
            new_external_inputs[external_position] = Some(input_id);
            continue;
        }

        let register_index = register_input_positions[old_input_index].ok_or_else(|| {
            format!(
                "transition input '{}' is not bound to an external port or register",
                input.name
            )
        })?;
        let survivors = bits_by_register[register_index]
            .iter()
            .copied()
            .filter(|&index| actions[index] == StateAction::Keep)
            .collect::<Vec<_>>();
        if survivors.is_empty() {
            continue;
        }

        let input_id = TransitionInputId::new(builder.inputs.len());
        let new_bits = builder.add_input(input.name.clone(), survivors.len());
        for (new_index, &state_index) in survivors.iter().enumerate() {
            bind_input_operand(
                &mut builder,
                &mut operands,
                graph,
                state_bits[state_index].q,
                *new_bits.get_lsb(new_index),
            )?;
        }
        new_register_inputs[register_index] = Some(input_id);
    }

    for (index, bit) in state_bits.iter().enumerate() {
        match actions[index] {
            StateAction::Constant(value) => {
                let replacement = if value {
                    builder.get_true()
                } else {
                    builder.get_false()
                };
                operands[bit.q.node.id] = Some(replacement);
                builder.add_pir_node_ids(
                    replacement.node,
                    graph.gates[bit.q.node.id].get_pir_node_ids(),
                );
            }
            StateAction::Duplicate {
                representative,
                negated,
            } => {
                let replacement = operands[state_bits[representative].q.node.id]
                    .ok_or_else(|| "duplicate state has no surviving representative".to_string())?;
                let replacement = if negated {
                    replacement.negate()
                } else {
                    replacement
                };
                operands[bit.q.node.id] = Some(replacement);
                builder.add_pir_node_ids(
                    replacement.node,
                    graph.gates[bit.q.node.id].get_pir_node_ids(),
                );
            }
            StateAction::Keep | StateAction::Dead => {
                // Kept Q inputs were already rebuilt; dead Q cones are
                // unreachable.
            }
        }
    }

    let reachable = mark_retained_combinational_nodes(design, state_bits, actions);
    for (node_index, node) in graph.gates.iter().enumerate() {
        if !reachable[node_index] {
            continue;
        }
        match node {
            AigNode::Input { .. } => {
                if operands[node_index].is_none() {
                    return Err(format!(
                        "reachable transition input node %{} has no retained state binding",
                        node_index
                    ));
                }
            }
            AigNode::Literal {
                value,
                pir_node_ids,
            } => {
                let operand = if *value {
                    builder.get_true()
                } else {
                    builder.get_false()
                };
                builder.add_pir_node_ids(operand.node, pir_node_ids);
                operands[node_index] = Some(operand);
            }
            AigNode::And2 {
                a,
                b,
                tags,
                pir_node_ids,
            } => {
                let lhs = map_operand(&operands, *a)?;
                let rhs = map_operand(&operands, *b)?;
                let operand = builder.add_and_binary(lhs, rhs);
                builder.add_pir_node_ids(operand.node, pir_node_ids);
                if let Some(tags) = tags {
                    for tag in tags {
                        builder.add_tag(operand.node, tag.clone());
                    }
                }
                operands[node_index] = Some(operand);
            }
        }
    }

    let mut external_output_positions = vec![None; graph.outputs.len()];
    let mut register_output_positions = vec![None; graph.outputs.len()];
    for (position, output_id) in design.outputs.iter().enumerate() {
        external_output_positions[output_id.index()] = Some(position);
    }
    for (register_index, register) in design.registers.iter().enumerate() {
        register_output_positions[register.d.index()] = Some(register_index);
    }

    let mut new_external_outputs = vec![None; design.outputs.len()];
    let mut new_register_outputs = vec![None; design.registers.len()];
    for (old_output_index, output) in graph.outputs.iter().enumerate() {
        if let Some(external_position) = external_output_positions[old_output_index] {
            let mapped_bits = output
                .bit_vector
                .iter_lsb_to_msb()
                .map(|operand| map_operand(&operands, *operand))
                .collect::<Result<Vec<_>, _>>()?;
            let output_id = TransitionOutputId::new(builder.outputs.len());
            builder.add_output(
                output.name.clone(),
                AigBitVector::from_lsb_is_index_0(&mapped_bits),
            );
            new_external_outputs[external_position] = Some(output_id);
            continue;
        }

        let register_index = register_output_positions[old_output_index].ok_or_else(|| {
            format!(
                "transition output '{}' is not bound to an external port or register",
                output.name
            )
        })?;
        let survivors = bits_by_register[register_index]
            .iter()
            .copied()
            .filter(|&index| actions[index] == StateAction::Keep)
            .collect::<Vec<_>>();
        if survivors.is_empty() {
            continue;
        }
        let mapped_bits = survivors
            .iter()
            .map(|&index| map_operand(&operands, state_bits[index].d))
            .collect::<Result<Vec<_>, _>>()?;
        let output_id = TransitionOutputId::new(builder.outputs.len());
        builder.add_output(
            output.name.clone(),
            AigBitVector::from_lsb_is_index_0(&mapped_bits),
        );
        new_register_outputs[register_index] = Some(output_id);
    }

    let mut registers = Vec::with_capacity(design.registers.len());
    for (register_index, register) in design.registers.iter().enumerate() {
        let (Some(q), Some(d)) = (
            new_register_inputs[register_index],
            new_register_outputs[register_index],
        ) else {
            continue;
        };
        let initial_value = register.initial_value.as_ref().map(|_| {
            let initial_bits = bits_by_register[register_index]
                .iter()
                .copied()
                .filter(|&index| actions[index] == StateAction::Keep)
                .map(|index| {
                    state_bits[index]
                        .explicit_initial
                        .expect("explicit packed state contains every initial bit")
                })
                .collect::<Vec<_>>();
            IrBits::from_lsb_is_0(&initial_bits)
        });
        registers.push(RegisterBinding {
            name: register.name.clone(),
            q,
            d,
            initial_value,
        });
    }

    let transition = if builder.outputs.is_empty() {
        GateFn {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            gates: builder.gates,
        }
    } else {
        builder.build()
    };
    let inputs = new_external_inputs
        .into_iter()
        .map(|input| input.ok_or_else(|| "external input was not rebuilt".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = new_external_outputs
        .into_iter()
        .map(|output| output.ok_or_else(|| "external output was not rebuilt".to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    SequentialGateFn::new(
        design.name.clone(),
        transition,
        inputs,
        outputs,
        design.clock.clone(),
        registers,
    )
}

/// Binds an original AIG input node and carries its existing provenance.
fn bind_input_operand(
    builder: &mut GateBuilder,
    operands: &mut [Option<AigOperand>],
    graph: &GateFn,
    old: AigOperand,
    new: AigOperand,
) -> Result<(), String> {
    if old.negated {
        return Err(format!(
            "transition input node %{} uses an unsupported inverted input binding",
            old.node.id
        ));
    }
    builder.add_pir_node_ids(new.node, graph.gates[old.node.id].get_pir_node_ids());
    operands[old.node.id] = Some(new);
    Ok(())
}

/// Finds ordinary AIG cones retained after state substitution.
fn mark_retained_combinational_nodes(
    design: &SequentialGateFn,
    state_bits: &[StateBit],
    actions: &[StateAction],
) -> Vec<bool> {
    let graph = &design.transition;
    let mut reachable = vec![false; graph.gates.len()];
    let mut stack = Vec::new();
    for output_id in &design.outputs {
        stack.extend(
            graph.outputs[output_id.index()]
                .bit_vector
                .iter_lsb_to_msb()
                .map(|operand| operand.node.id),
        );
    }
    for (index, bit) in state_bits.iter().enumerate() {
        if actions[index] == StateAction::Keep {
            stack.push(bit.d.node.id);
        }
    }
    while let Some(node_id) = stack.pop() {
        if reachable[node_id] {
            continue;
        }
        reachable[node_id] = true;
        stack.extend(
            graph.gates[node_id]
                .operands()
                .map(|operand| operand.node.id),
        );
    }
    reachable
}

/// Maps an old signed AIG operand through the rebuilt node table.
fn map_operand(operands: &[Option<AigOperand>], operand: AigOperand) -> Result<AigOperand, String> {
    let mapped = operands
        .get(operand.node.id)
        .and_then(|mapped| *mapped)
        .ok_or_else(|| {
            format!(
                "reachable transition operand %{} was not rebuilt in topological order",
                operand.node.id
            )
        })?;
    Ok(if operand.negated {
        mapped.negate()
    } else {
        mapped
    })
}

/// Rejects reordered, renamed, or missing ABC transition boundary bits.
fn validate_flattened_interface(original: &GateFn, optimized: &GateFn) -> Result<(), String> {
    let original_inputs = flattened_input_names(original)?;
    let optimized_inputs = flattened_input_names(optimized)?;
    check_flattened_port_names("input", &original_inputs, &optimized_inputs)?;
    let original_outputs = flattened_output_names(original)?;
    let optimized_outputs = flattened_output_names(optimized)?;
    check_flattened_port_names("output", &original_outputs, &optimized_outputs)
}

/// Returns scalar input names in stable AIGER order.
fn flattened_input_names(graph: &GateFn) -> Result<Vec<String>, String> {
    flattened_port_names(
        "input",
        graph
            .inputs
            .iter()
            .map(|input| (input.name.as_str(), input.get_bit_count())),
    )
}

/// Returns scalar output names in stable AIGER order.
fn flattened_output_names(graph: &GateFn) -> Result<Vec<String>, String> {
    flattened_port_names(
        "output",
        graph
            .outputs
            .iter()
            .map(|output| (output.name.as_str(), output.get_bit_count())),
    )
}

/// Expands packed port names using the canonical binary AIGER convention.
fn flattened_port_names<'a>(
    kind: &str,
    ports: impl Iterator<Item = (&'a str, usize)>,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut seen = BTreeSet::new();
    for (name, bit_count) in ports {
        if bit_count == 0 {
            return Err(format!("transition {kind} '{name}' has zero bits"));
        }
        for bit_index in 0..bit_count {
            let scalar = if bit_count == 1 {
                name.to_string()
            } else {
                format!("{name}_{bit_index}")
            };
            if !seen.insert(scalar.clone()) {
                return Err(format!(
                    "transition has duplicate flattened {kind} '{scalar}'"
                ));
            }
            result.push(scalar);
        }
    }
    Ok(result)
}

/// Reports the exact scalar transition boundary that ABC failed to preserve.
fn check_flattened_port_names(
    kind: &str,
    expected: &[String],
    actual: &[String],
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "optimized transition {kind} interface has {} bits; original transition has {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if expected != actual {
            return Err(format!(
                "optimized transition {kind} bit {index} is '{actual}'; expected '{expected}'"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::sequential_gate::ClockPort;
    use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    use crate::aig_serdes::load_aiger_auto::load_aiger_auto;
    use crate::aig_sim::sequential::{SequentialState, simulate};

    /// Builds a visible two-bit register with next state `bits[2]:2`.
    fn constant_state_design(initial_value: Option<IrBits>) -> SequentialGateFn {
        let mut builder = GateBuilder::new(
            "constant_state__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let _data = builder.add_input("data".to_string(), 1);
        let q = builder.add_input("state__q".to_string(), 2);
        let zero = builder.get_false();
        let one = builder.get_true();
        builder.add_output("out".to_string(), q);
        builder.add_output(
            "state__d".to_string(),
            AigBitVector::from_lsb_is_index_0(&[zero, one]),
        );
        SequentialGateFn::new(
            "constant_state".to_string(),
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
                initial_value,
            }],
        )
        .expect("construct constant-state sequential fixture")
    }

    /// Builds deterministic single-bit stimuli for sequential trace checks.
    fn samples() -> Vec<Vec<IrBits>> {
        [false, true, true, false, true, false, false, true]
            .into_iter()
            .map(|bit| vec![IrBits::from_lsb_is_0(&[bit])])
            .collect()
    }

    /// Compares external outputs under the simulator's all-zero state policy.
    fn assert_zero_initialized_traces_equal(
        original: &SequentialGateFn,
        cleaned: &SequentialGateFn,
    ) {
        let inputs = samples();
        let expected = simulate(original, &inputs, SequentialState::all_zeros(original))
            .expect("simulate original zero-initialized design");
        let actual = simulate(cleaned, &inputs, SequentialState::all_zeros(cleaned))
            .expect("simulate cleaned zero-initialized design");
        assert_eq!(actual.external_outputs(), expected.external_outputs());
    }

    #[test]
    fn preserves_cycle_zero_when_constant_next_state_changes_initial_value() {
        let design = constant_state_design(None);
        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("clean while preserving all-zero startup");

        assert_eq!(result.stats.initial_register_bits, 2);
        assert_eq!(result.stats.final_register_bits, 1);
        assert_eq!(result.stats.constant_register_bits, 1);
        assert_eq!(result.design.registers.len(), 1);
        let remaining_q = &result.design.transition.inputs[result.design.registers[0].q.index()];
        assert_eq!(remaining_q.name, "state__q");
        assert_eq!(remaining_q.get_bit_count(), 1);
        assert_eq!(result.design.transition.outputs[0].get_bit_count(), 2);
        assert_zero_initialized_traces_equal(&design, &result.design);
    }

    #[test]
    fn removes_unspecified_constant_state_after_hardware_startup() {
        let design = constant_state_design(None);
        let result = cleanup_sequential_gate_fn(
            &design,
            &SequentialCleanupOptions {
                initialization_policy: RegisterInitializationPolicy::UninitializedDontCare,
            },
        )
        .expect("clean unspecified constant register bits");

        assert_eq!(result.stats.initial_register_bits, 2);
        assert_eq!(result.stats.final_register_bits, 0);
        assert_eq!(result.stats.constant_register_bits, 2);
        assert!(result.design.registers.is_empty());
        assert_eq!(
            result.design.clock,
            Some(ClockPort {
                name: "clk".to_string()
            })
        );

        let inputs = samples();
        let expected = simulate(&design, &inputs, SequentialState::all_zeros(&design))
            .expect("simulate unspecified original");
        let actual = simulate(
            &result.design,
            &inputs,
            SequentialState::all_zeros(&result.design),
        )
        .expect("simulate state-free result");
        assert_eq!(
            &actual.external_outputs()[1..],
            &expected.external_outputs()[1..],
            "unspecified constant state must agree after its first active edge"
        );
    }

    #[test]
    fn never_removes_a_constant_conflicting_with_explicit_initialization() {
        let initial = IrBits::from_lsb_is_0(&[false, false]);
        let design = constant_state_design(Some(initial));
        for initialization_policy in [
            RegisterInitializationPolicy::PreserveCycleZero,
            RegisterInitializationPolicy::UninitializedDontCare,
        ] {
            let result = cleanup_sequential_gate_fn(
                &design,
                &SequentialCleanupOptions {
                    initialization_policy,
                },
            )
            .expect("respect explicitly initialized transition");
            assert_eq!(result.stats.final_register_bits, 1);
            assert_eq!(result.stats.constant_register_bits, 1);
            let remaining_initial = result.design.registers[0]
                .initial_value
                .as_ref()
                .expect("retain surviving explicit initialization");
            assert_eq!(remaining_initial, &IrBits::from_lsb_is_0(&[false]));
        }
    }

    #[test]
    fn removes_explicitly_initialized_stable_constants_at_cycle_zero() {
        let design = constant_state_design(Some(IrBits::from_lsb_is_0(&[false, true])));
        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("remove explicitly stable state");
        assert!(result.design.registers.is_empty());
        assert_eq!(result.stats.constant_register_bits, 2);

        let inputs = samples();
        let original_state = SequentialState::from_g8r_initial_values(&design)
            .expect("read explicit original initial values");
        let cleaned_state = SequentialState::all_zeros(&result.design);
        let expected = simulate(&design, &inputs, original_state)
            .expect("simulate explicitly initialized original");
        let actual = simulate(&result.design, &inputs, cleaned_state)
            .expect("simulate state-free constant output");
        assert_eq!(actual.external_outputs(), expected.external_outputs());
    }

    #[test]
    fn removes_dead_state_through_transitive_next_state_cones() {
        let mut builder = GateBuilder::new(
            "dead_state__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 1);
        let live = builder.add_input("live__q".to_string(), 1);
        let dead_a = builder.add_input("dead_a__q".to_string(), 1);
        let dead_b = builder.add_input("dead_b__q".to_string(), 1);
        builder.add_output("out".to_string(), live.clone());
        builder.add_output("live__d".to_string(), data.clone());
        builder.add_output("dead_a__d".to_string(), dead_b);
        builder.add_output("dead_b__d".to_string(), dead_a);
        let design = SequentialGateFn::new(
            "dead_state".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![
                RegisterBinding {
                    name: "live".to_string(),
                    q: TransitionInputId::new(1),
                    d: TransitionOutputId::new(1),
                    initial_value: None,
                },
                RegisterBinding {
                    name: "dead_a".to_string(),
                    q: TransitionInputId::new(2),
                    d: TransitionOutputId::new(2),
                    initial_value: None,
                },
                RegisterBinding {
                    name: "dead_b".to_string(),
                    q: TransitionInputId::new(3),
                    d: TransitionOutputId::new(3),
                    initial_value: None,
                },
            ],
        )
        .expect("construct dead-state fixture");

        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("remove mutually dependent unobservable state");
        assert_eq!(result.stats.dead_register_bits, 2);
        assert_eq!(result.stats.final_register_bits, 1);
        assert_eq!(result.design.registers[0].name, "live");
        assert_zero_initialized_traces_equal(&design, &result.design);
    }

    #[test]
    fn merges_equal_next_state_bits_without_changing_cycle_zero() {
        let mut builder = GateBuilder::new(
            "equal_state__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 1);
        let state = builder.add_input("state__q".to_string(), 3);
        let d = *data.get_lsb(0);
        builder.add_output("out".to_string(), state);
        builder.add_output(
            "state__d".to_string(),
            AigBitVector::from_lsb_is_index_0(&[d, d, d]),
        );
        let design = SequentialGateFn::new(
            "equal_state".to_string(),
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
        .expect("construct duplicate-state fixture");

        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("merge identical next-state bits");
        assert_eq!(result.stats.merged_register_bits, 2);
        assert_eq!(result.stats.final_register_bits, 1);
        assert_eq!(result.design.transition.outputs[0].get_bit_count(), 3);
        assert_zero_initialized_traces_equal(&design, &result.design);
    }

    #[test]
    fn merges_complementary_state_only_when_initial_polarity_agrees() {
        let mut builder = GateBuilder::new(
            "complementary_state__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 1);
        let state = builder.add_input("state__q".to_string(), 2);
        let d = *data.get_lsb(0);
        builder.add_output("out".to_string(), state);
        builder.add_output(
            "state__d".to_string(),
            AigBitVector::from_lsb_is_index_0(&[d, d.negate()]),
        );
        let design = SequentialGateFn::new(
            "complementary_state".to_string(),
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
                initial_value: Some(IrBits::from_lsb_is_0(&[false, true])),
            }],
        )
        .expect("construct complementary-state fixture");

        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("merge equivalent complementary state");
        assert_eq!(result.stats.merged_register_bits, 1);
        assert_eq!(result.stats.final_register_bits, 1);

        let inputs = samples();
        let expected = simulate(
            &design,
            &inputs,
            SequentialState::from_g8r_initial_values(&design)
                .expect("read complementary initial state"),
        )
        .expect("simulate original complementary state");
        let actual = simulate(
            &result.design,
            &inputs,
            SequentialState::from_g8r_initial_values(&result.design)
                .expect("read compacted complementary initial state"),
        )
        .expect("simulate compacted complementary state");
        assert_eq!(actual.external_outputs(), expected.external_outputs());
    }

    #[test]
    fn reaches_a_fixed_point_after_constant_state_substitution() {
        let mut builder = GateBuilder::new(
            "constant_chain__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let _data = builder.add_input("data".to_string(), 1);
        let state = builder.add_input("state__q".to_string(), 3);
        let zero = builder.get_false();
        let first = *state.get_lsb(0);
        let second = *state.get_lsb(1);
        builder.add_output("out".to_string(), state);
        builder.add_output(
            "state__d".to_string(),
            AigBitVector::from_lsb_is_index_0(&[zero, first, second]),
        );
        let design = SequentialGateFn::new(
            "constant_chain".to_string(),
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
        .expect("construct cascading-constant fixture");

        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("propagate constant state to a fixed point");
        assert_eq!(result.stats.constant_register_bits, 3);
        assert_eq!(result.stats.final_register_bits, 0);
        assert!(result.stats.iterations >= 3);
        assert_zero_initialized_traces_equal(&design, &result.design);
    }

    #[test]
    fn retains_observable_register_pipeline_latency() {
        let mut builder = GateBuilder::new(
            "pipeline__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 1);
        let state = builder.add_input("state__q".to_string(), 1);
        builder.add_output("out".to_string(), state);
        builder.add_output("state__d".to_string(), data);
        let design = SequentialGateFn::new(
            "pipeline".to_string(),
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
        .expect("construct one-cycle pipeline");

        for initialization_policy in [
            RegisterInitializationPolicy::PreserveCycleZero,
            RegisterInitializationPolicy::UninitializedDontCare,
        ] {
            let result = cleanup_sequential_gate_fn(
                &design,
                &SequentialCleanupOptions {
                    initialization_policy,
                },
            )
            .expect("preserve observable pipeline latency");
            assert_eq!(result.stats.initial_register_bits, 1);
            assert_eq!(result.stats.final_register_bits, 1);
            assert_eq!(result.stats.dead_register_bits, 0);
            assert_eq!(result.stats.constant_register_bits, 0);
            assert_eq!(result.stats.merged_register_bits, 0);
            assert_zero_initialized_traces_equal(&design, &result.design);
        }
    }

    #[test]
    fn rebinds_flat_aiger_bits_to_original_packed_register_ports() {
        let design = constant_state_design(None);
        let bytes = emit_aiger_binary(&design.transition, true)
            .expect("encode packed transition with scalar symbols");
        let flattened = load_aiger_auto(&bytes, GateBuilderOptions::no_opt())
            .expect("load scalar ABC-style binary AIG");
        assert_eq!(flattened.gate_fn.inputs.len(), 3);

        let rebound = rebind_sequential_transition(&design, &flattened.gate_fn)
            .expect("restore packed transition interface");
        assert_eq!(rebound.transition.inputs.len(), 2);
        assert_eq!(rebound.transition.inputs[1].name, "state__q");
        assert_eq!(rebound.transition.inputs[1].get_bit_count(), 2);
        assert_eq!(rebound.transition.outputs[1].name, "state__d");
        assert_eq!(rebound.transition.outputs[1].get_bit_count(), 2);
        assert_zero_initialized_traces_equal(&design, &rebound);
    }

    #[test]
    fn rejects_reordered_scalar_aiger_transition_ports() {
        let design = constant_state_design(None);
        let bytes = emit_aiger_binary(&design.transition, true)
            .expect("encode packed transition with scalar symbols");
        let mut flattened = load_aiger_auto(&bytes, GateBuilderOptions::no_opt())
            .expect("load scalar binary transition")
            .gate_fn;
        flattened.inputs.swap(0, 1);

        let error = rebind_sequential_transition(&design, &flattened)
            .expect_err("state rebinding must reject reordered ABC ports");
        assert_eq!(
            error,
            "optimized transition input bit 0 is 'state__q_0'; expected 'data'"
        );
    }

    #[test]
    fn handles_designs_with_no_observable_external_outputs() {
        let mut builder = GateBuilder::new(
            "unobservable__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 1);
        let _state = builder.add_input("state__q".to_string(), 1);
        builder.add_output("state__d".to_string(), data);
        let design = SequentialGateFn::new(
            "unobservable".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(0),
                initial_value: None,
            }],
        )
        .expect("construct zero-observable-output fixture");

        let result = cleanup_sequential_gate_fn(&design, &SequentialCleanupOptions::default())
            .expect("remove entirely unobservable state");
        assert_eq!(result.stats.dead_register_bits, 1);
        assert!(result.design.registers.is_empty());
        assert!(result.design.transition.outputs.is_empty());
        result
            .design
            .validate()
            .expect("state-free zero-output transition remains valid");
    }
}
