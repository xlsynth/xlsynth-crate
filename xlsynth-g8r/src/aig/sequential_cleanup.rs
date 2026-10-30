// SPDX-License-Identifier: Apache-2.0

//! Bit-granular synthesis cleanup of ordinary sequential AIGs.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::aig::dce::dce_safe;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn};
use crate::aig::sequential_gate::uniquify_transition_port_name;
use crate::aig::sequential_gate::{
    RegisterBinding, SequentialGateFn, TransitionInputId, TransitionOutputId,
};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

mod substitutions;
use substitutions::mark_duplicate_state_bits;

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
    /// Number of state-classification rounds (zero for register-free inputs).
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

/// One bit of packed state and its effective next-state expression.
#[derive(Debug, Clone, Copy)]
struct StateBit {
    q: AigOperand,
    d: AigOperand,
}

/// Flattened state and the original packed register-to-bit correspondence.
struct StateLayout {
    bits: Vec<StateBit>,
    by_register: Vec<Vec<usize>>,
}

/// Replacement of an original state bit in a cleanup round.
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

/// Removes dead, constant, and redundant register bits to a fixed point.
///
/// Unspecified startup is a synthesis don't-care: constant and duplicate state
/// substitutions may restrict power-up behavior. Reset and enable behavior is
/// preserved through the effective D expressions. This is not equivalence from
/// every independent cycle-zero assignment; consumers must not depend on such
/// assignments. Explicit controls must be legalized before cleanup.
/// The pass is optional and leaves its input unchanged.
///
/// External ports and the clock are retained even if unused. Packed state is
/// compacted in original LSB order. Equality is structural, not SAT-based.
/// Cascading constants propagate across combinational users and D/Q boundaries
/// before reconstruction, visiting each dependency at most once per round.
/// Duplicate substitutions likewise propagate through affected fanouts before
/// reconstruction, without requiring one whole-graph pass per pipeline stage.
pub fn cleanup_sequential_gate_fn(
    design: &SequentialGateFn,
) -> Result<SequentialCleanupResult, String> {
    design.validate_legalized()?;
    let mut current = design.clone();
    // Normalize arbitrary DAG storage order and remove dead logic even when no
    // state is removed (or there are no registers at all).
    current.transition = dce_safe(&current.transition);
    let mut stats = SequentialCleanupStats {
        initial_register_bits: design.register_bit_count(),
        initial_and_nodes: count_and_nodes(&design.transition),
        ..SequentialCleanupStats::default()
    };

    while !current.registers.is_empty() {
        stats.iterations += 1;
        let StateLayout {
            bits: state_bits,
            by_register: bits_by_register,
        } = collect_state_bits(&current);
        let live_bits = mark_live_state_bits(&current, &state_bits);
        let mut actions = vec![StateAction::Keep; state_bits.len()];

        for (index, is_live) in live_bits.iter().enumerate() {
            if !is_live {
                actions[index] = StateAction::Dead;
            }
        }
        mark_constant_state_bits(&current.transition, &state_bits, &mut actions);
        mark_duplicate_state_bits(&current.transition, &state_bits, &mut actions);

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
        current.transition = dce_safe(&current.transition);
    }

    stats.final_register_bits = current.register_bit_count();
    stats.final_and_nodes = count_and_nodes(&current.transition);
    current.validate_legalized()?;
    Ok(SequentialCleanupResult {
        design: current,
        stats,
    })
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
fn collect_state_bits(design: &SequentialGateFn) -> StateLayout {
    let mut result = Vec::with_capacity(design.register_bit_count());
    let mut bits_by_register = Vec::with_capacity(design.registers.len());
    for register in &design.registers {
        let q = &design.transition.inputs[register.q.index()];
        let d = &design.transition.outputs[register.d.index()];
        let mut register_bits = Vec::with_capacity(q.get_bit_count());
        for bit_index in 0..q.get_bit_count() {
            register_bits.push(result.len());
            result.push(StateBit {
                q: *q.bit_vector.get_lsb(bit_index),
                d: *d.bit_vector.get_lsb(bit_index),
            });
        }
        bits_by_register.push(register_bits);
    }
    StateLayout {
        bits: result,
        by_register: bits_by_register,
    }
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

/// A dependency to revisit when an AIG node becomes provably constant.
#[derive(Clone, Copy)]
enum ConstantUser {
    And(usize),
    Register(usize),
}

/// Records monotonic constant information and schedules its users exactly once.
fn record_constant(
    values: &mut [Option<bool>],
    pending: &mut Vec<usize>,
    node: usize,
    value: bool,
) {
    if let Some(previous) = values[node] {
        debug_assert_eq!(previous, value, "constant propagation must be consistent");
    } else {
        values[node] = Some(value);
        pending.push(node);
    }
}

/// Batches constant-D substitutions to a fixed point without
/// rebuilding the graph. Returns the dependency-visit count for scaling tests.
fn mark_constant_state_bits(
    graph: &GateFn,
    state_bits: &[StateBit],
    actions: &mut [StateAction],
) -> usize {
    let mut users = vec![Vec::new(); graph.gates.len()];
    let mut values = vec![None; graph.gates.len()];
    let mut pending = Vec::new();
    for (node_id, node) in graph.gates.iter().enumerate() {
        match node {
            AigNode::And2 { a, b, .. } => {
                users[a.node.id].push(ConstantUser::And(node_id));
                users[b.node.id].push(ConstantUser::And(node_id));
                if *a == b.negate() {
                    record_constant(&mut values, &mut pending, node_id, false);
                }
            }
            AigNode::Literal { value, .. } => {
                record_constant(&mut values, &mut pending, node_id, *value);
            }
            AigNode::Input { .. } => {
                // External inputs and unproven Q values remain unknown.
            }
        }
    }
    for (index, bit) in state_bits.iter().enumerate() {
        if actions[index] != StateAction::Keep {
            continue;
        }
        users[bit.d.node.id].push(ConstantUser::Register(index));
    }

    let mut visits = 0;
    while let Some(node_id) = pending.pop() {
        for user in &users[node_id] {
            visits += 1;
            match *user {
                ConstantUser::And(id) => {
                    let AigNode::And2 { a, b, .. } = &graph.gates[id] else {
                        unreachable!("AND dependency refers to an AND node");
                    };
                    let a = values[a.node.id].map(|value| value ^ a.negated);
                    let b = values[b.node.id].map(|value| value ^ b.negated);
                    let value = match (a, b) {
                        (Some(false), _) | (_, Some(false)) => Some(false),
                        (Some(true), Some(true)) => Some(true),
                        _ => None,
                    };
                    if let Some(value) = value {
                        record_constant(&mut values, &mut pending, id, value);
                    }
                }
                ConstantUser::Register(index) => {
                    let bit = &state_bits[index];
                    if actions[index] == StateAction::Keep {
                        let value = values[bit.d.node.id]
                            .expect("only proven constants schedule their users")
                            ^ bit.d.negated;
                        actions[index] = StateAction::Constant(value);
                        record_constant(&mut values, &mut pending, bit.q.node.id, value);
                    }
                }
            }
        }
    }
    visits
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

    // Output roles are not exclusive: an external port and several registers
    // may share one transition output, and packed registers may retain
    // different subsets of its bits. Cache by original output and retained
    // bit positions.
    let mut rebuilt_outputs = BTreeMap::new();
    let mut used_output_names = BTreeSet::new();
    let mut new_external_outputs = Vec::with_capacity(design.outputs.len());
    for id in &design.outputs {
        let bits = (0..graph.outputs[id.index()].get_bit_count()).collect::<Vec<_>>();
        let new_id = rebuild_output(
            &mut builder,
            graph,
            &operands,
            *id,
            &bits,
            &mut rebuilt_outputs,
            &mut used_output_names,
        )?;
        new_external_outputs.push(new_id);
    }
    let mut new_register_outputs = vec![None; design.registers.len()];
    for (register_index, register) in design.registers.iter().enumerate() {
        if new_register_inputs[register_index].is_none() {
            continue;
        }
        let bits = bits_by_register[register_index]
            .iter()
            .enumerate()
            .filter_map(|(bit, &index)| (actions[index] == StateAction::Keep).then_some(bit))
            .collect::<Vec<_>>();
        let new_id = rebuild_output(
            &mut builder,
            graph,
            &operands,
            register.d,
            &bits,
            &mut rebuilt_outputs,
            &mut used_output_names,
        )?;
        new_register_outputs[register_index] = Some(new_id);
    }

    let mut registers = Vec::with_capacity(design.registers.len());
    for (register_index, register) in design.registers.iter().enumerate() {
        let (Some(q), Some(d)) = (
            new_register_inputs[register_index],
            new_register_outputs[register_index],
        ) else {
            continue;
        };
        registers.push(RegisterBinding {
            enable: None,
            reset: None,
            name: register.name.clone(),
            q,
            d,
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
    SequentialGateFn::new(
        design.name.clone(),
        transition,
        inputs,
        new_external_outputs,
        design.clock.clone(),
        registers,
    )
}

/// Rebuilds one packed output slice, preserving aliases between output roles.
fn rebuild_output(
    builder: &mut GateBuilder,
    graph: &GateFn,
    operands: &[Option<AigOperand>],
    old_id: TransitionOutputId,
    bits: &[usize],
    rebuilt: &mut BTreeMap<(TransitionOutputId, Vec<usize>), TransitionOutputId>,
    used_names: &mut BTreeSet<String>,
) -> Result<TransitionOutputId, String> {
    let key = (old_id, bits.to_vec());
    if let Some(id) = rebuilt.get(&key) {
        return Ok(*id);
    }
    let output = &graph.outputs[old_id.index()];
    let mapped_bits = bits
        .iter()
        .map(|&bit| map_operand(operands, *output.bit_vector.get_lsb(bit)))
        .collect::<Result<Vec<_>, _>>()?;
    let id = TransitionOutputId::new(builder.outputs.len());
    let name = uniquify_transition_port_name(&output.name, used_names);
    builder.add_output(name, AigBitVector::from_lsb_is_index_0(&mapped_bits));
    rebuilt.insert(key, id);
    Ok(id)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::sequential_gate::ClockPort;
    use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    use crate::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
    use crate::aig_serdes::load_aiger_auto::load_aiger_auto;
    use crate::aig_serdes::sequential_transition::rebind_sequential_transition;
    use crate::aig_sim::sequential::{SequentialState, simulate};
    use crate::block2sequential::block_ir_to_sequential_gate_fn;
    use crate::gatify::ir2gate::GatifyOptions;
    use xlsynth_pir::IrBits;
    use xlsynth_test_helpers::compare_golden_text;

    #[test]
    fn shared_output_roles_survive_partial_packed_cleanup() {
        let mut b = GateBuilder::new("shared".into(), GateBuilderOptions::no_opt());
        let data = b.add_input("data".into(), 2);
        let q0 = b.add_input("q0".into(), 2);
        let q1 = b.add_input("q1".into(), 2);
        let dead = b.add_input("dead".into(), 1);
        b.add_output("shared_d".into(), data);
        b.add_output(
            "visible".into(),
            AigBitVector::from_lsb_is_index_0(&[*q0.get_lsb(0), *q1.get_lsb(1)]),
        );
        b.add_output("dead_d".into(), dead);
        let design = SequentialGateFn::new(
            "shared".into(),
            b.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0), TransitionOutputId::new(1)],
            Some(ClockPort { name: "clk".into() }),
            vec![
                RegisterBinding {
                    enable: None,
                    reset: None,
                    name: "a".into(),
                    q: TransitionInputId::new(1),
                    d: TransitionOutputId::new(0),
                },
                RegisterBinding {
                    enable: None,
                    reset: None,
                    name: "b".into(),
                    q: TransitionInputId::new(2),
                    d: TransitionOutputId::new(0),
                },
                RegisterBinding {
                    enable: None,
                    reset: None,
                    name: "dead".into(),
                    q: TransitionInputId::new(3),
                    d: TransitionOutputId::new(2),
                },
            ],
        )
        .unwrap();
        let cleaned = cleanup_sequential_gate_fn(&design).unwrap();
        assert_eq!(cleaned.stats.final_register_bits, 2);
        assert_eq!(cleaned.stats.dead_register_bits, 3);
        let inputs = (0..12)
            .map(|i| vec![IrBits::from_lsb_is_0(&[i % 2 == 0, i % 3 == 0])])
            .collect::<Vec<_>>();
        let expected = simulate(&design, &inputs, SequentialState::all_zeros(&design)).unwrap();
        let actual = simulate(
            &cleaned.design,
            &inputs,
            SequentialState::all_zeros(&cleaned.design),
        )
        .unwrap();
        assert_eq!(expected.external_outputs(), actual.external_outputs());
        assert_eq!(
            cleaned.design.transition.outputs[cleaned.design.outputs[0].index()].name,
            "shared_d"
        );
        let again = cleanup_sequential_gate_fn(&cleaned.design).unwrap();
        compare_golden_text(
            &format!("{}\n", emit_g8r(&cleaned.design).unwrap()),
            "tests/goldens/sequential_cleanup_shared.g8r",
        );
        assert_eq!(
            emit_g8r(&again.design).unwrap(),
            emit_g8r(&cleaned.design).unwrap()
        );
        assert_eq!(
            encode_g8r_binary(&again.design).unwrap(),
            encode_g8r_binary(&cleaned.design).unwrap()
        );
    }

    #[test]
    fn cleans_dead_logic_without_state_and_accepts_non_topological_storage() {
        let mut b = GateBuilder::new("unordered".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 2);
        let a = b.add_and_binary(*x.get_lsb(0), *x.get_lsb(1));
        let z = b.add_and_binary(a, x.get_lsb(0).negate());
        b.add_and_binary(*x.get_lsb(0), x.get_lsb(1).negate());
        b.add_output("out".into(), AigBitVector::from_lsb_is_index_0(&[z]));
        let mut graph = b.build();
        let i = a.node.id;
        let j = z.node.id;
        graph.gates.swap(i, j);
        let remap = |id| {
            if id == i {
                j
            } else if id == j {
                i
            } else {
                id
            }
        };
        for node in &mut graph.gates {
            if let AigNode::And2 { a, b, .. } = node {
                a.node.id = remap(a.node.id);
                b.node.id = remap(b.node.id);
            }
        }
        graph.outputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(&[AigOperand {
            node: crate::aig::gate::AigRef { id: i },
            negated: false,
        }]);
        let design = SequentialGateFn::from_gate_fn(graph);
        let result = cleanup_sequential_gate_fn(&design).unwrap();
        assert_eq!(result.stats.initial_and_nodes, 3);
        assert_eq!(result.stats.final_and_nodes, 2);
        assert_eq!(result.design.inputs, design.inputs);
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
                    enable: None,
                    reset: None,
                    name: "live".to_string(),
                    q: TransitionInputId::new(1),
                    d: TransitionOutputId::new(1),
                },
                RegisterBinding {
                    enable: None,
                    reset: None,
                    name: "dead_a".to_string(),
                    q: TransitionInputId::new(2),
                    d: TransitionOutputId::new(2),
                },
                RegisterBinding {
                    enable: None,
                    reset: None,
                    name: "dead_b".to_string(),
                    q: TransitionInputId::new(3),
                    d: TransitionOutputId::new(3),
                },
            ],
        )
        .expect("construct dead-state fixture");

        let result = cleanup_sequential_gate_fn(&design)
            .expect("remove mutually dependent unobservable state");
        assert_eq!(result.stats.dead_register_bits, 2);
        assert_eq!(result.stats.final_register_bits, 1);
        assert_eq!(result.design.registers[0].name, "live");
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
                enable: None,
                reset: None,
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
            }],
        )
        .expect("construct one-cycle pipeline");

        let result =
            cleanup_sequential_gate_fn(&design).expect("preserve observable pipeline latency");
        assert_eq!(result.stats.initial_register_bits, 1);
        assert_eq!(result.stats.final_register_bits, 1);
        assert_eq!(result.stats.dead_register_bits, 0);
        assert_zero_initialized_traces_equal(&design, &result.design);
    }

    #[test]
    fn rebinds_flat_aiger_bits_to_original_packed_register_ports() {
        let design = constant_state_design();
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
        let design = constant_state_design();
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
                enable: None,
                reset: None,
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(0),
            }],
        )
        .expect("construct zero-observable-output fixture");

        let result =
            cleanup_sequential_gate_fn(&design).expect("remove entirely unobservable state");
        assert_eq!(result.stats.dead_register_bits, 1);
        assert!(result.design.registers.is_empty());
        assert!(result.design.transition.outputs.is_empty());
        result
            .design
            .validate()
            .expect("state-free zero-output transition remains valid");
    }

    #[test]
    fn constant_d_removal_refines_unspecified_startup() {
        let design = constant_state_design();
        let result = cleanup_sequential_gate_fn(&design).unwrap();
        assert_eq!(result.stats.constant_register_bits, 2);
        assert_eq!(result.stats.final_register_bits, 0);
        let before = simulate(
            &design,
            &samples(),
            SequentialState::from_register_values(
                &design,
                vec![IrBits::from_lsb_is_0(&[false, true])],
            )
            .unwrap(),
        )
        .unwrap();
        let after = simulate(
            &result.design,
            &samples(),
            SequentialState::all_zeros(&result.design),
        )
        .unwrap();
        assert_eq!(before.external_outputs(), after.external_outputs());
    }

    #[test]
    fn merges_wide_equal_and_complementary_d_with_corresponding_startup() {
        let mut b = GateBuilder::new("duplicates".into(), GateBuilderOptions::no_opt());
        let data = b.add_input("data".into(), 1);
        let q = b.add_input("q".into(), 129);
        let d = *data.get_lsb(0);
        b.add_output("out".into(), q);
        b.add_output(
            "d".into(),
            AigBitVector::from_lsb_is_index_0(
                &(0..129)
                    .map(|i| if i % 3 == 1 { d.negate() } else { d })
                    .collect::<Vec<_>>(),
            ),
        );
        let design = SequentialGateFn::new(
            "duplicates".into(),
            b.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort { name: "clk".into() }),
            vec![RegisterBinding {
                enable: None,
                reset: None,
                name: "state".into(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
            }],
        )
        .unwrap();
        let result = cleanup_sequential_gate_fn(&design).unwrap();
        assert_eq!(result.stats.merged_register_bits, 128);
        assert_eq!(result.stats.final_register_bits, 1);
        for value in [false, true] {
            let before = simulate(
                &design,
                &samples(),
                SequentialState::from_register_values(
                    &design,
                    vec![IrBits::from_lsb_is_0(
                        &(0..129).map(|i| value ^ (i % 3 == 1)).collect::<Vec<_>>(),
                    )],
                )
                .unwrap(),
            )
            .unwrap();
            let after = simulate(
                &result.design,
                &samples(),
                SequentialState::from_register_values(
                    &result.design,
                    vec![IrBits::from_lsb_is_0(&[value])],
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(before.external_outputs(), after.external_outputs());
        }
    }

    #[test]
    fn batches_large_constant_chains_with_linear_dependency_visits() {
        for width in [128, 16_384] {
            for inverted in [false, true] {
                let mut b = GateBuilder::new("chain".into(), GateBuilderOptions::no_opt());
                let q = b.add_input("q".into(), width);
                let mut ds = vec![b.get_false()];
                for i in 1..width {
                    let previous = *q.get_lsb(i - 1);
                    ds.push(if inverted {
                        previous.negate()
                    } else {
                        previous
                    });
                }
                b.add_output("out".into(), q);
                b.add_output("d".into(), AigBitVector::from_lsb_is_index_0(&ds));
                let design = SequentialGateFn::new(
                    "chain".into(),
                    b.build(),
                    vec![],
                    vec![TransitionOutputId::new(0)],
                    Some(ClockPort { name: "clk".into() }),
                    vec![RegisterBinding {
                        enable: None,
                        reset: None,
                        name: "state".into(),
                        q: TransitionInputId::new(0),
                        d: TransitionOutputId::new(1),
                    }],
                )
                .unwrap();
                let StateLayout { bits, .. } = collect_state_bits(&design);
                let mut actions = vec![StateAction::Keep; width];
                let visits = mark_constant_state_bits(&design.transition, &bits, &mut actions);
                assert!(visits <= width);
                for (i, action) in actions.iter().enumerate() {
                    assert_eq!(*action, StateAction::Constant(inverted && i % 2 == 1));
                }
                let result = cleanup_sequential_gate_fn(&design).unwrap();
                assert_eq!(result.stats.iterations, 1);
                assert_eq!(result.stats.constant_register_bits, width);
                assert_eq!(result.stats.final_register_bits, 0);
            }
        }
    }

    /// Builds two equivalent pipelines with already-hashed combinational logic.
    fn duplicate_pipeline(
        stages: usize,
        complemented: bool,
        gated: bool,
        reversed: bool,
    ) -> SequentialGateFn {
        let mut b = GateBuilder::new("pipeline".into(), GateBuilderOptions::opt());
        let inputs = b.add_input("inputs".into(), 2);
        let q = b.add_input("q".into(), 2 * stages);
        let position = |stage| 2 * if reversed { stages - 1 - stage } else { stage };
        let mut ds = vec![b.get_false(); 2 * stages];
        for stage in 0..stages {
            let (a, c) = if stage == 0 {
                (*inputs.get_lsb(0), *inputs.get_lsb(0))
            } else {
                let a = *q.get_lsb(position(stage - 1));
                let c = *q.get_lsb(position(stage - 1) + 1);
                (a, if complemented { c.negate() } else { c })
            };
            let a = if gated {
                b.add_and_binary(a, *inputs.get_lsb(1))
            } else {
                a
            };
            let c = if gated {
                b.add_and_binary(c, *inputs.get_lsb(1))
            } else {
                c
            };
            ds[position(stage)] = a;
            ds[position(stage) + 1] = if complemented { c.negate() } else { c };
        }
        b.add_output("out".into(), q.get_lsb_slice(position(stages - 1), 2));
        b.add_output("d".into(), AigBitVector::from_lsb_is_index_0(&ds));
        SequentialGateFn::new(
            "pipeline".into(),
            b.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort { name: "clk".into() }),
            vec![RegisterBinding {
                enable: None,
                reset: None,
                name: "state".into(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
            }],
        )
        .unwrap()
    }

    #[test]
    fn batches_duplicate_pipelines_through_combinational_fanouts() {
        for stages in [4, 128, 1_024] {
            for complemented in [false, true] {
                for gated in [false, true] {
                    for reversed in [false, true] {
                        let design = duplicate_pipeline(stages, complemented, gated, reversed);
                        let StateLayout { bits, .. } = collect_state_bits(&design);
                        let mut actions = vec![StateAction::Keep; bits.len()];
                        let visits =
                            mark_duplicate_state_bits(&design.transition, &bits, &mut actions);
                        assert!(
                            visits <= 16 * design.transition.gates.len(),
                            "visits={visits}"
                        );
                        assert_eq!(
                            actions
                                .iter()
                                .filter(|a| matches!(a, StateAction::Duplicate { .. }))
                                .count(),
                            stages
                        );
                        let result = cleanup_sequential_gate_fn(&design).unwrap();
                        assert_eq!(result.stats.iterations, 2);
                        assert_eq!(result.stats.merged_register_bits, stages);
                        assert_eq!(result.stats.final_register_bits, stages);
                        if stages == 4 {
                            let roots = (0..stages).map(|i| i % 3 == 0).collect::<Vec<_>>();
                            let initial = roots
                                .iter()
                                .flat_map(|&v| [v, v ^ complemented])
                                .collect::<Vec<_>>();
                            let inputs = (0..stages + 4)
                                .map(|cycle| {
                                    vec![IrBits::from_lsb_is_0(&[cycle % 2 == 0, cycle % 3 != 0])]
                                })
                                .collect::<Vec<_>>();
                            let before = simulate(
                                &design,
                                &inputs,
                                SequentialState::from_register_values(
                                    &design,
                                    vec![IrBits::from_lsb_is_0(&initial)],
                                )
                                .unwrap(),
                            )
                            .unwrap();
                            let after = simulate(
                                &result.design,
                                &inputs,
                                SequentialState::from_register_values(
                                    &result.design,
                                    vec![IrBits::from_lsb_is_0(&roots)],
                                )
                                .unwrap(),
                            )
                            .unwrap();
                            assert_eq!(before.external_outputs(), after.external_outputs());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn duplicate_substitution_exposes_constant_and_complementary_state() {
        let mut b = GateBuilder::new("constants".into(), GateBuilderOptions::opt());
        let x = b.add_input("x".into(), 1);
        let q = b.add_input("q".into(), 8);
        let zero_after_merge = b.add_and_binary(*q.get_lsb(0), q.get_lsb(1).negate());
        let ds = [
            *x.get_lsb(0),
            *x.get_lsb(0),
            zero_after_merge,
            zero_after_merge,
            zero_after_merge,
            zero_after_merge,
            q.get_lsb(2).negate(),
            q.get_lsb(6).negate(),
        ];
        b.add_output("out".into(), q);
        b.add_output("d".into(), AigBitVector::from_lsb_is_index_0(&ds));
        let design = SequentialGateFn::new(
            "constants".into(),
            b.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort { name: "clk".into() }),
            vec![RegisterBinding {
                enable: None,
                reset: None,
                name: "state".into(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
            }],
        )
        .unwrap();
        let result = cleanup_sequential_gate_fn(&design).unwrap();
        assert_eq!(result.stats.iterations, 2);
        assert_eq!(result.stats.merged_register_bits, 1);
        assert_eq!(result.stats.constant_register_bits, 6);
        assert_eq!(result.stats.final_register_bits, 1);
        for root in [false, true] {
            let before = simulate(
                &design,
                &samples(),
                SequentialState::from_register_values(
                    &design,
                    vec![IrBits::from_lsb_is_0(&[
                        root, root, false, false, false, false, true, false,
                    ])],
                )
                .unwrap(),
            )
            .unwrap();
            let after = simulate(
                &result.design,
                &samples(),
                SequentialState::from_register_values(
                    &result.design,
                    vec![IrBits::from_lsb_is_0(&[root])],
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(before.external_outputs(), after.external_outputs());
        }
    }

    /// Uses unrelated startup assignments; comparison begins after a reset
    /// edge.
    fn assert_post_reset_traces_equal(
        design: &SequentialGateFn,
        inputs: &[Vec<IrBits>],
    ) -> SequentialCleanupResult {
        let result = cleanup_sequential_gate_fn(design).unwrap();
        for original_one in [false, true] {
            for cleaned_one in [false, true] {
                let state = |d: &SequentialGateFn, one| {
                    SequentialState::from_register_values(
                        d,
                        d.registers
                            .iter()
                            .map(|r| {
                                IrBits::from_lsb_is_0(&vec![
                                    one;
                                    d.transition.inputs[r.q.index()]
                                        .get_bit_count()
                                ])
                            })
                            .collect(),
                    )
                    .unwrap()
                };
                let before = simulate(design, inputs, state(design, original_one)).unwrap();
                let after =
                    simulate(&result.design, inputs, state(&result.design, cleaned_one)).unwrap();
                assert_eq!(
                    &before.external_outputs()[1..],
                    &after.external_outputs()[1..]
                );
            }
        }
        result
    }

    #[test]
    fn preserves_distinct_reset_values_and_reset_priority_over_enable() {
        let ir = r#"package reset_controls
top block top(clk: clock, rst: bits[1], data: bits[1], enable: bits[1], a: bits[1], b: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg a(bits[1], reset_value=0)
  reg b(bits[1], reset_value=1)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[1] = input_port(name=data, id=2)
  enable: bits[1] = input_port(name=enable, id=3)
  qa: bits[1] = register_read(register=a, id=4)
  qb: bits[1] = register_read(register=b, id=5)
  wa: () = register_write(data, register=a, load_enable=enable, reset=rst, id=6)
  wb: () = register_write(data, register=b, load_enable=enable, reset=rst, id=7)
  a: () = output_port(qa, name=a, id=8)
  b: () = output_port(qb, name=b, id=9)
}
"#;
        let design = block_ir_to_sequential_gate_fn(ir, GatifyOptions::all_opts_enabled()).unwrap();
        let inputs = [
            (true, false, false),
            (false, true, false),
            (false, true, true),
            (false, false, false),
            (true, false, false),
            (false, false, true),
        ]
        .into_iter()
        .map(|(r, d, e)| {
            [r, d, e]
                .into_iter()
                .map(|v| IrBits::from_lsb_is_0(&[v]))
                .collect()
        })
        .collect::<Vec<_>>();
        let result = assert_post_reset_traces_equal(&design, &inputs);
        assert_eq!(result.stats.final_register_bits, 2);
        assert_eq!(result.stats.merged_register_bits, 0);
    }

    #[test]
    fn merges_unreset_data_behind_resettable_valid() {
        let ir = r#"package masked_data
top block top(clk: clock, rst: bits[1], data: bits[8], valid_in: bits[1], a: bits[8], b: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg valid(bits[1], reset_value=0)
  reg a(bits[8])
  reg b(bits[8])
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[8] = input_port(name=data, id=2)
  valid_in: bits[1] = input_port(name=valid_in, id=3)
  qv: bits[1] = register_read(register=valid, id=4)
  qa: bits[8] = register_read(register=a, id=5)
  qb: bits[8] = register_read(register=b, id=6)
  wv: () = register_write(valid_in, register=valid, reset=rst, id=7)
  wa: () = register_write(data, register=a, id=8)
  wb: () = register_write(data, register=b, id=9)
  zero: bits[8] = literal(value=0, id=10)
  ma: bits[8] = sel(qv, cases=[zero, qa], id=11)
  mb: bits[8] = sel(qv, cases=[zero, qb], id=12)
  a: () = output_port(ma, name=a, id=13)
  b: () = output_port(mb, name=b, id=14)
}
"#;
        for enabled in [false, true] {
            let ir = if enabled {
                ir.replace(
                    "register_write(data, register=a, id=8)",
                    "register_write(data, register=a, load_enable=valid_in, id=8)",
                )
                .replace(
                    "register_write(data, register=b, id=9)",
                    "register_write(data, register=b, load_enable=valid_in, id=9)",
                )
            } else {
                ir.to_string()
            };
            let design =
                block_ir_to_sequential_gate_fn(&ir, GatifyOptions::all_opts_enabled()).unwrap();
            let inputs = (0..16)
                .map(|cycle| {
                    vec![
                        IrBits::from_lsb_is_0(&[cycle == 0 || cycle == 8]),
                        IrBits::from_lsb_is_0(
                            &(0..8).map(|bit| (cycle + bit) % 3 == 0).collect::<Vec<_>>(),
                        ),
                        IrBits::from_lsb_is_0(&[cycle % 3 == 2]),
                    ]
                })
                .collect::<Vec<_>>();
            let result = assert_post_reset_traces_equal(&design, &inputs);
            assert_eq!(
                result.stats.merged_register_bits,
                if enabled { 0 } else { 8 }
            );
            assert_eq!(
                result.stats.final_register_bits,
                if enabled { 17 } else { 9 }
            );
        }
    }

    /// Builds a visible two-bit register with next state `bits[2]:2`.
    fn constant_state_design() -> SequentialGateFn {
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
                enable: None,
                reset: None,
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
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
}
