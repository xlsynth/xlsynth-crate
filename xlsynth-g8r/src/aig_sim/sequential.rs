// SPDX-License-Identifier: Apache-2.0

//! Cycle-based simulation and toggle accounting for native sequential G8R
//! designs.

use serde::Serialize;
use xlsynth::IrBits;

use crate::aig::SequentialGateFn;
use crate::aig_sim::count_toggles::{NodeToggleStats, count_toggle_activity};
use crate::aig_sim::gate_sim::{self, Collect};

/// Register values in [`SequentialGateFn::registers`] order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialState {
    values: Vec<IrBits>,
}

impl SequentialState {
    /// Constructs state from explicit register values and validates their
    /// shape.
    pub fn from_register_values(
        design: &SequentialGateFn,
        values: Vec<IrBits>,
    ) -> Result<Self, String> {
        if values.len() != design.registers.len() {
            return Err(format!(
                "initial state has {} register values, expected {}",
                values.len(),
                design.registers.len()
            ));
        }
        for (index, (value, register)) in values.iter().zip(&design.registers).enumerate() {
            let expected_width = design.transition.inputs[register.q.index()].get_bit_count();
            if value.get_bit_count() != expected_width {
                return Err(format!(
                    "initial state register {} ('{}') has width {}, expected {}",
                    index,
                    register.name,
                    value.get_bit_count(),
                    expected_width
                ));
            }
        }
        Ok(Self { values })
    }

    /// Uses initial values declared on every G8R register.
    pub fn from_g8r_initial_values(design: &SequentialGateFn) -> Result<Self, String> {
        let missing = design
            .registers
            .iter()
            .filter(|register| register.initial_value.is_none())
            .map(|register| register.name.clone())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "G8R registers without initial values: {:?}",
                missing
            ));
        }
        let values = design
            .registers
            .iter()
            .map(|register| {
                register
                    .initial_value
                    .clone()
                    .expect("missing initial values were rejected")
            })
            .collect();
        Self::from_register_values(design, values)
    }

    /// Constructs an all-zero value for every register.
    pub fn all_zeros(design: &SequentialGateFn) -> Self {
        let values = design
            .registers
            .iter()
            .map(|register| {
                let width = design.transition.inputs[register.q.index()].get_bit_count();
                IrBits::from_lsb_is_0(&vec![false; width])
            })
            .collect();
        Self { values }
    }

    /// Returns register values in design register order.
    pub fn values(&self) -> &[IrBits] {
        &self.values
    }

    /// Consumes this state and returns values in design register order.
    pub fn into_values(self) -> Vec<IrBits> {
        self.values
    }
}

/// A complete cycle trace, including the transition-function boundary values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialTrace {
    external_inputs: Vec<Vec<IrBits>>,
    external_outputs: Vec<Vec<IrBits>>,
    register_outputs: Vec<Vec<IrBits>>,
    register_inputs: Vec<Vec<IrBits>>,
    transition_inputs: Vec<Vec<IrBits>>,
    final_state: SequentialState,
}

impl SequentialTrace {
    /// Returns externally visible outputs for each simulated cycle.
    pub fn external_outputs(&self) -> &[Vec<IrBits>] {
        &self.external_outputs
    }

    /// Returns the state after the final simulated clock edge.
    pub fn final_state(&self) -> &SequentialState {
        &self.final_state
    }
}

/// Aggregate combinational-logic activity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogicToggleActivity {
    pub gate_input_toggles: usize,
    pub gate_output_toggles: usize,
}

/// Activity at externally visible input and output ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterfaceToggleActivity {
    pub external_input_toggles: usize,
    pub external_output_toggles: usize,
}

/// Activity at one register's D and Q boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegisterToggleActivity {
    pub name: String,
    pub width: usize,
    /// D-pin changes between consecutive cycle evaluations.
    pub input_toggles: usize,
    /// Actual Q changes across simulated clock edges.
    pub output_toggles: usize,
}

/// Aggregate and per-register boundary activity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegisterToggleActivities {
    pub aggregate_input_toggles: usize,
    pub aggregate_output_toggles: usize,
    pub entries: Vec<RegisterToggleActivity>,
}

/// Categorized toggle activity for one sequential G8R trace.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequentialToggleActivity {
    pub cycle_count: usize,
    pub logic_transition_count: usize,
    pub logic: LogicToggleActivity,
    pub interface: InterfaceToggleActivity,
    pub registers: RegisterToggleActivities,
    pub nodes: Vec<NodeToggleStats>,
}

/// Simulates one input record per cycle and commits register D after each
/// cycle.
pub fn simulate(
    design: &SequentialGateFn,
    external_inputs: &[Vec<IrBits>],
    initial_state: SequentialState,
) -> Result<SequentialTrace, String> {
    design.validate()?;
    validate_external_inputs(design, external_inputs)?;
    let mut state = SequentialState::from_register_values(design, initial_state.into_values())?;
    let mut external_outputs = Vec::with_capacity(external_inputs.len());
    let mut register_outputs = Vec::with_capacity(external_inputs.len());
    let mut register_inputs = Vec::with_capacity(external_inputs.len());
    let mut transition_inputs_trace = Vec::with_capacity(external_inputs.len());

    for cycle_inputs in external_inputs {
        let mut transition_inputs = vec![None; design.transition.inputs.len()];
        for (value, input_id) in cycle_inputs.iter().zip(&design.inputs) {
            transition_inputs[input_id.index()] = Some(value.clone());
        }
        for (value, register) in state.values.iter().zip(&design.registers) {
            transition_inputs[register.q.index()] = Some(value.clone());
        }
        let transition_inputs = transition_inputs
            .into_iter()
            .map(|value| value.expect("validated G8R bindings cover every transition input"))
            .collect::<Vec<_>>();
        let result = gate_sim::eval(&design.transition, &transition_inputs, Collect::None);
        let visible_outputs = design
            .outputs
            .iter()
            .map(|id| result.outputs[id.index()].clone())
            .collect::<Vec<_>>();
        let next_register_values = design
            .registers
            .iter()
            .map(|register| result.outputs[register.d.index()].clone())
            .collect::<Vec<_>>();

        transition_inputs_trace.push(transition_inputs);
        external_outputs.push(visible_outputs);
        register_outputs.push(state.values.clone());
        register_inputs.push(next_register_values.clone());
        state = SequentialState::from_register_values(design, next_register_values)?;
    }

    Ok(SequentialTrace {
        external_inputs: external_inputs.to_vec(),
        external_outputs,
        register_outputs,
        register_inputs,
        transition_inputs: transition_inputs_trace,
        final_state: state,
    })
}

/// Counts combinational, interface, and register-boundary activity separately.
pub fn count_sequential_toggle_activity(
    design: &SequentialGateFn,
    trace: &SequentialTrace,
) -> Result<SequentialToggleActivity, String> {
    if trace.transition_inputs.len() < 2 {
        return Err(format!(
            "toggle stimulus must contain at least two cycles; got {}",
            trace.transition_inputs.len()
        ));
    }
    validate_trace_shape(design, trace)?;
    let logic_activity = count_toggle_activity(&design.transition, &trace.transition_inputs);
    let entries = design
        .registers
        .iter()
        .enumerate()
        .map(|(register_index, register)| {
            let register_inputs = trace
                .register_inputs
                .iter()
                .map(|cycle| cycle[register_index].clone())
                .collect::<Vec<_>>();
            let input_toggles = count_adjacent_port_toggles(&register_inputs);
            let output_toggles = trace
                .register_outputs
                .iter()
                .zip(&trace.register_inputs)
                .map(|(before, after)| {
                    count_bit_differences(&before[register_index], &after[register_index])
                })
                .sum();
            RegisterToggleActivity {
                name: register.name.clone(),
                width: design.transition.inputs[register.q.index()].get_bit_count(),
                input_toggles,
                output_toggles,
            }
        })
        .collect::<Vec<_>>();
    let aggregate_input_toggles = entries.iter().map(|entry| entry.input_toggles).sum();
    let aggregate_output_toggles = entries.iter().map(|entry| entry.output_toggles).sum();

    Ok(SequentialToggleActivity {
        cycle_count: trace.transition_inputs.len(),
        logic_transition_count: trace.transition_inputs.len() - 1,
        logic: LogicToggleActivity {
            gate_input_toggles: logic_activity.aggregate.gate_input_toggles,
            gate_output_toggles: logic_activity.aggregate.gate_output_toggles,
        },
        interface: InterfaceToggleActivity {
            external_input_toggles: count_adjacent_samples(&trace.external_inputs),
            external_output_toggles: count_adjacent_samples(&trace.external_outputs),
        },
        registers: RegisterToggleActivities {
            aggregate_input_toggles,
            aggregate_output_toggles,
            entries,
        },
        nodes: logic_activity.nodes,
    })
}

fn validate_external_inputs(
    design: &SequentialGateFn,
    external_inputs: &[Vec<IrBits>],
) -> Result<(), String> {
    for (cycle_index, cycle) in external_inputs.iter().enumerate() {
        if cycle.len() != design.inputs.len() {
            return Err(format!(
                "cycle {} has {} external inputs, expected {}",
                cycle_index + 1,
                cycle.len(),
                design.inputs.len()
            ));
        }
        for (input_index, (value, input_id)) in cycle.iter().zip(&design.inputs).enumerate() {
            let expected_width = design.transition.inputs[input_id.index()].get_bit_count();
            if value.get_bit_count() != expected_width {
                return Err(format!(
                    "cycle {} external input {} ('{}') has width {}, expected {}",
                    cycle_index + 1,
                    input_index,
                    design.transition.inputs[input_id.index()].name,
                    value.get_bit_count(),
                    expected_width
                ));
            }
        }
    }
    Ok(())
}

fn validate_trace_shape(design: &SequentialGateFn, trace: &SequentialTrace) -> Result<(), String> {
    let cycle_count = trace.transition_inputs.len();
    for (name, observed) in [
        ("external inputs", trace.external_inputs.len()),
        ("external outputs", trace.external_outputs.len()),
        ("register outputs", trace.register_outputs.len()),
        ("register inputs", trace.register_inputs.len()),
    ] {
        if observed != cycle_count {
            return Err(format!(
                "sequential trace has {} {} cycles, expected {}",
                observed, name, cycle_count
            ));
        }
    }
    for (cycle_index, values) in trace.register_outputs.iter().enumerate() {
        if values.len() != design.registers.len() {
            return Err(format!(
                "sequential trace register outputs at cycle {} have {} values, expected {}",
                cycle_index + 1,
                values.len(),
                design.registers.len()
            ));
        }
    }
    for (cycle_index, values) in trace.register_inputs.iter().enumerate() {
        if values.len() != design.registers.len() {
            return Err(format!(
                "sequential trace register inputs at cycle {} have {} values, expected {}",
                cycle_index + 1,
                values.len(),
                design.registers.len()
            ));
        }
    }
    Ok(())
}

fn count_adjacent_samples(samples: &[Vec<IrBits>]) -> usize {
    samples
        .windows(2)
        .map(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(before, after)| count_bit_differences(before, after))
                .sum::<usize>()
        })
        .sum()
}

fn count_adjacent_port_toggles(samples: &[IrBits]) -> usize {
    samples
        .windows(2)
        .map(|pair| count_bit_differences(&pair[0], &pair[1]))
        .sum()
}

fn count_bit_differences(before: &IrBits, after: &IrBits) -> usize {
    assert_eq!(before.get_bit_count(), after.get_bit_count());
    (0..before.get_bit_count())
        .filter(|&bit_index| {
            before.get_bit(bit_index).expect("bit index is in range")
                != after.get_bit(bit_index).expect("bit index is in range")
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        ClockPort, RegisterBinding, SequentialGateFn, TransitionInputId, TransitionOutputId,
    };
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn bits(width: usize, value: u64) -> IrBits {
        IrBits::make_ubits(width, value).unwrap()
    }

    fn accumulator_design(initial_value: Option<IrBits>) -> SequentialGateFn {
        let mut builder = GateBuilder::new(
            "accumulator__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let x = builder.add_input("x".to_string(), 1);
        let q = builder.add_input("acc__q".to_string(), 1);
        let d = builder.add_xor_vec(&x, &q);
        builder.add_output("y".to_string(), q);
        builder.add_output("acc__d".to_string(), d);
        SequentialGateFn::new(
            "accumulator".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "acc".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value,
            }],
        )
        .unwrap()
    }

    fn wide_register_design(width: usize) -> SequentialGateFn {
        let mut builder = GateBuilder::new(
            "wide_register__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let q = builder.add_input("wide__q".to_string(), width);
        builder.add_output("y".to_string(), q.clone());
        builder.add_output("wide__d".to_string(), q);
        SequentialGateFn::new(
            "wide_register".to_string(),
            builder.build(),
            vec![],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "wide".to_string(),
                q: TransitionInputId::new(0),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .unwrap()
    }

    #[test]
    fn declared_and_explicit_initial_state_are_validated() {
        let design = accumulator_design(Some(bits(1, 1)));
        assert_eq!(
            SequentialState::from_g8r_initial_values(&design)
                .unwrap()
                .values(),
            &[bits(1, 1)]
        );
        assert!(
            SequentialState::from_register_values(&design, vec![bits(2, 0)])
                .unwrap_err()
                .contains("width 2, expected 1")
        );

        let missing = accumulator_design(None);
        let error = SequentialState::from_g8r_initial_values(&missing).unwrap_err();
        assert!(error.contains("acc"), "{error}");
    }

    #[test]
    fn simulation_emits_current_outputs_then_commits_next_state() {
        let design = accumulator_design(Some(bits(1, 1)));
        let trace = simulate(
            &design,
            &[vec![bits(1, 1)], vec![bits(1, 0)]],
            SequentialState::from_g8r_initial_values(&design).unwrap(),
        )
        .unwrap();

        assert_eq!(
            trace.external_outputs,
            vec![vec![bits(1, 1)], vec![bits(1, 0)]]
        );
        assert_eq!(
            trace.register_inputs,
            vec![vec![bits(1, 0)], vec![bits(1, 0)]]
        );
        assert_eq!(trace.final_state.values(), &[bits(1, 0)]);
    }

    #[test]
    fn zero_initialization_supports_registers_wider_than_u64() {
        let design = wide_register_design(65);
        let state = SequentialState::all_zeros(&design);
        assert_eq!(state.values()[0].get_bit_count(), 65);
        assert!((0..65).all(|index| !state.values()[0].get_bit(index).unwrap()));
    }

    #[test]
    fn toggle_activity_separates_logic_interface_and_register_boundaries() {
        let design = accumulator_design(Some(bits(1, 1)));
        let trace = simulate(
            &design,
            &[vec![bits(1, 1)], vec![bits(1, 0)]],
            SequentialState::from_g8r_initial_values(&design).unwrap(),
        )
        .unwrap();
        let activity = count_sequential_toggle_activity(&design, &trace).unwrap();

        assert_eq!(activity.cycle_count, 2);
        assert_eq!(activity.logic_transition_count, 1);
        assert_eq!(activity.interface.external_input_toggles, 1);
        assert_eq!(activity.interface.external_output_toggles, 1);
        assert_eq!(activity.registers.aggregate_input_toggles, 0);
        assert_eq!(activity.registers.aggregate_output_toggles, 1);
        assert_eq!(activity.registers.entries[0].name, "acc");
        assert_eq!(activity.registers.entries[0].input_toggles, 0);
        assert_eq!(activity.registers.entries[0].output_toggles, 1);
    }
}
