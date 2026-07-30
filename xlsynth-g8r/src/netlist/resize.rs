// SPDX-License-Identifier: Apache-2.0

//! Shared Liberty-aware resizing options and the combinational sizing API.

use crate::liberty_model::Library;
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::sta::StaOptions;
use crate::netlist::timing_buffer::BufferTimingConstraints;
use crate::netlist::timing_resize::resize_timing_aware_netlist;
use anyhow::{Result, anyhow};
use serde::Serialize;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

#[cfg(test)]
use crate::liberty_model::PinDirection;
#[cfg(test)]
use crate::netlist::parse::{NetIndex, NetRef, PortDirection};
#[cfg(test)]
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, TimingQueryDiagnosticCounts,
    effective_input_capacitance_for_mapping, evaluate_combinational_cell_output_timing,
};
#[cfg(test)]
use crate::netlist::utils::scalar_constant_output_assignments;
#[cfg(test)]
use std::collections::{BTreeSet, HashMap};

/// Bounded critical-path upsizing and timing-protected area-recovery options.
#[derive(Clone, Debug, PartialEq)]
pub struct ResizeOptions {
    pub sta_options: StaOptions,
    /// Maximum alternating timing-optimization and area-recovery rounds.
    pub max_outer_iterations: usize,
    pub max_iterations: usize,
    pub max_area_iterations: usize,
    pub max_candidate_paths: usize,
    pub max_evaluations_per_iteration: usize,
    pub max_cell_candidates_per_instance: usize,
    pub improvement_epsilon: f64,
    pub area_epsilon: f64,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            sta_options: StaOptions::default(),
            max_outer_iterations: 3,
            max_iterations: 16,
            max_area_iterations: 32,
            max_candidate_paths: 32,
            max_evaluations_per_iteration: 64,
            max_cell_candidates_per_instance: 8,
            improvement_epsilon: 1e-9,
            area_epsilon: 1e-12,
        }
    }
}

/// One accepted exact-function Liberty cell substitution.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResizeStep {
    pub instance: String,
    pub old_cell: String,
    pub new_cell: String,
    pub delay_before: f64,
    pub delay_after: f64,
    pub area_before: f64,
    pub area_after: f64,
}

/// One accepted Boolean-safe exchange of characterized cell input pins.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PinSwapStep {
    pub instance: String,
    pub cell: String,
    pub first_pin: String,
    pub second_pin: String,
    pub delay_before: f64,
    pub delay_after: f64,
}

/// Machine-readable results of buffered-netlist upsizing and downsizing.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResizeStats {
    pub initial_delay: f64,
    pub final_delay: f64,
    pub initial_area: f64,
    pub final_area: f64,
    /// Completed alternating timing and timing-protected recovery rounds.
    pub outer_iterations: usize,
    pub evaluations: usize,
    /// Exact incremental trials spent evaluating Boolean-safe pin exchanges.
    pub pin_swap_evaluations: usize,
    pub failed_evaluations: usize,
    pub recomputed_instances: usize,
    pub upsizes: usize,
    pub downsizes: usize,
    /// Accepted drive-strength increases of physical flip-flops.
    pub register_upsizes: usize,
    /// Accepted timing-preserving drive-strength decreases of flip-flops.
    pub register_downsizes: usize,
    /// Accepted zero-area exchanges of combinational Liberty input pins.
    pub pin_swaps: usize,
    /// Aggregate clock-pin capacitance before register-aware sizing.
    pub initial_clock_load: Option<f64>,
    /// Aggregate clock-pin capacitance after register-aware sizing.
    pub final_clock_load: Option<f64>,
    pub replacements: Vec<ResizeStep>,
    pub pin_swap_steps: Vec<PinSwapStep>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct InstanceInput {
    name: String,
    net: Option<NetIndex>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct InstanceTiming {
    cell_index: usize,
    output_pin_index: usize,
    output_net: NetIndex,
    inputs: Vec<InstanceInput>,
    known_pin_values: HashMap<String, bool>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct TimingScore {
    endpoint_delays: Vec<f64>,
}

#[cfg(test)]
impl TimingScore {
    /// Returns the largest actual combinational output arrival.
    fn worst_delay(&self) -> f64 {
        self.endpoint_delays.first().copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct TrialResult {
    score: TimingScore,
    recomputed_instances: usize,
}

/// Sizes a combinational netlist with the shared exact-Liberty timing engine.
pub fn resize_netlist(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &ResizeOptions,
) -> Result<ResizeStats> {
    resize_timing_aware_netlist(
        module,
        nets,
        interner,
        library,
        options,
        &BufferTimingConstraints::default(),
    )
}

/// Rejects budgets and timing assumptions that could produce invalid trials.
pub(crate) fn validate_options(options: &ResizeOptions) -> Result<()> {
    if options.max_outer_iterations == 0
        || options.max_candidate_paths == 0
        || options.max_evaluations_per_iteration == 0
        || options.max_cell_candidates_per_instance == 0
    {
        return Err(anyhow!("resizer search bounds must be greater than zero"));
    }
    if !options.improvement_epsilon.is_finite() || options.improvement_epsilon < 0.0 {
        return Err(anyhow!(
            "resizer improvement_epsilon must be finite and nonnegative"
        ));
    }
    if !options.area_epsilon.is_finite() || options.area_epsilon < 0.0 {
        return Err(anyhow!(
            "resizer area_epsilon must be finite and nonnegative"
        ));
    }
    if !options.sta_options.primary_input_transition.is_finite()
        || options.sta_options.primary_input_transition < 0.0
        || !options.sta_options.module_output_load.is_finite()
        || options.sta_options.module_output_load < 0.0
    {
        return Err(anyhow!(
            "resizer STA options must be finite and nonnegative"
        ));
    }
    Ok(())
}

/// Reusable exact-NLDM timing graph for same-pin combinational substitutions.
#[cfg(test)]
struct IncrementalCombinationalSta<'a> {
    library: &'a Library,
    instances: Vec<InstanceTiming>,
    drivers: Vec<Option<usize>>,
    loads: Vec<CombinationalOutputLoad>,
    net_timing: Vec<Option<SignalTiming>>,
    successors: Vec<Vec<usize>>,
    topo_positions: Vec<usize>,
    outputs: Vec<NetIndex>,
    diagnostic_counts: TimingQueryDiagnosticCounts,
}

#[cfg(test)]
impl<'a> IncrementalCombinationalSta<'a> {
    /// Builds connectivity once and performs the initial full timing pass.
    fn new(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &'a Library,
        options: StaOptions,
    ) -> Result<Self> {
        let constant_outputs = scalar_constant_output_assignments(module, nets)?;
        let by_name: HashMap<&str, usize> = library
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| (cell.name.as_str(), index))
            .collect();
        let mut drivers = vec![None; nets.len()];
        let mut loads = vec![CombinationalOutputLoad::default(); nets.len()];
        let mut net_timing = vec![None; nets.len()];
        let mut outputs = Vec::new();
        let mut instances = Vec::with_capacity(module.instances.len());

        for port in &module.ports {
            let net = module.find_net_index(port.name, nets).ok_or_else(|| {
                anyhow!(
                    "sizing port '{}' has no scalar net",
                    interner.resolve(port.name).unwrap_or("<unknown>")
                )
            })?;
            match port.direction {
                PortDirection::Input => {
                    let timing = EdgeTiming {
                        arrival: 0.0,
                        transition: options.primary_input_transition,
                    };
                    net_timing[net.0] = Some(SignalTiming {
                        rise: timing,
                        fall: timing,
                    });
                }
                PortDirection::Output => {
                    outputs.push(net);
                    loads[net.0].rise += options.module_output_load;
                    loads[net.0].fall += options.module_output_load;
                    if constant_outputs.contains_key(&net.0) {
                        let timing = EdgeTiming {
                            arrival: 0.0,
                            transition: 0.0,
                        };
                        net_timing[net.0] = Some(SignalTiming {
                            rise: timing,
                            fall: timing,
                        });
                    }
                }
                PortDirection::Inout => {
                    return Err(anyhow!(
                        "incremental combinational sizing does not support inout ports"
                    ));
                }
            }
        }

        for (instance_index, instance) in module.instances.iter().enumerate() {
            let name = interner
                .resolve(instance.type_name)
                .ok_or_else(|| anyhow!("cannot resolve sized cell name"))?;
            let cell_index = *by_name
                .get(name)
                .ok_or_else(|| anyhow!("sized netlist references unknown cell '{name}'"))?;
            let cell = &library.cells[cell_index];
            if !cell.sequential.is_empty() || cell.clock_gate.is_some() {
                return Err(anyhow!(
                    "incremental combinational sizing does not yet support sequential cell '{name}'"
                ));
            }

            let mut input_pins = Vec::new();
            let mut known_pin_values = HashMap::new();
            let mut output = None;
            for (pin_id, net_ref) in &instance.connections {
                let pin_name = interner
                    .resolve(*pin_id)
                    .ok_or_else(|| anyhow!("cannot resolve sized cell pin"))?;
                let pin = cell
                    .pins
                    .iter()
                    .find(|pin| library.resolve_string(&pin.name) == pin_name)
                    .ok_or_else(|| anyhow!("cell '{name}' has no pin '{pin_name}'"))?;
                if pin.direction == PinDirection::Input as i32 {
                    let net = match net_ref {
                        NetRef::Simple(net) => {
                            let capacitance = effective_input_capacitance_for_mapping(
                                pin,
                                format!("sizing pin '{name}.{pin_name}'").as_str(),
                            )?;
                            loads[net.0].rise += capacitance.rise;
                            loads[net.0].fall += capacitance.fall;
                            Some(*net)
                        }
                        NetRef::Literal(bits) => {
                            if bits.get_bit_count() != 1 {
                                return Err(anyhow!(
                                    "incremental sizing requires one-bit literal cell inputs"
                                ));
                            }
                            known_pin_values.insert(
                                pin_name.to_string(),
                                bits.get_bit(0).map_err(|error| anyhow!("{error}"))?,
                            );
                            None
                        }
                        NetRef::Unconnected | NetRef::UnknownLiteral(_) => None,
                        _ => {
                            return Err(anyhow!(
                                "incremental sizing requires scalar simple-net connections"
                            ));
                        }
                    };
                    input_pins.push(InstanceInput {
                        name: pin_name.to_string(),
                        net,
                    });
                } else if pin.direction == PinDirection::Output as i32 {
                    let NetRef::Simple(net) = net_ref else {
                        return Err(anyhow!("sized cell outputs must drive a scalar net"));
                    };
                    if output
                        .replace((
                            cell.pins
                                .iter()
                                .position(|candidate| {
                                    library.resolve_string(&candidate.name) == pin_name
                                })
                                .expect("connected output was found above"),
                            *net,
                        ))
                        .is_some()
                    {
                        return Err(anyhow!(
                            "incremental sizing requires single-output combinational cells"
                        ));
                    }
                    if drivers[net.0].replace(instance_index).is_some() {
                        return Err(anyhow!("sized net has multiple cell drivers"));
                    }
                }
            }
            let (output_pin_index, output_net) =
                output.ok_or_else(|| anyhow!("sized cell '{name}' has no output connection"))?;
            input_pins.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
            instances.push(InstanceTiming {
                cell_index,
                output_pin_index,
                output_net,
                inputs: input_pins,
                known_pin_values,
            });
        }

        for net in constant_outputs.keys() {
            if drivers[*net].is_some() {
                return Err(anyhow!(
                    "incremental sizing found a cell driving a constant-tied output"
                ));
            }
        }

        let mut successors = vec![Vec::new(); instances.len()];
        let mut indegrees = vec![0_usize; instances.len()];
        for (instance_index, instance) in instances.iter().enumerate() {
            let mut predecessors = BTreeSet::new();
            for input in &instance.inputs {
                if let Some(net) = input.net
                    && let Some(driver) = drivers[net.0]
                {
                    predecessors.insert(driver);
                }
            }
            indegrees[instance_index] = predecessors.len();
            for predecessor in predecessors {
                successors[predecessor].push(instance_index);
            }
        }
        for next in &mut successors {
            next.sort_unstable();
        }
        let mut ready: BTreeSet<usize> = indegrees
            .iter()
            .enumerate()
            .filter_map(|(index, count)| (*count == 0).then_some(index))
            .collect();
        let mut topo_order = Vec::with_capacity(instances.len());
        while let Some(index) = ready.pop_first() {
            topo_order.push(index);
            for successor in &successors[index] {
                indegrees[*successor] -= 1;
                if indegrees[*successor] == 0 {
                    ready.insert(*successor);
                }
            }
        }
        if topo_order.len() != instances.len() {
            return Err(anyhow!("incremental sizing detected a combinational cycle"));
        }
        let mut topo_positions = vec![0; instances.len()];
        for (position, instance_index) in topo_order.iter().copied().enumerate() {
            topo_positions[instance_index] = position;
        }

        let mut state = Self {
            library,
            instances,
            drivers,
            loads,
            net_timing,
            successors,
            topo_positions,
            outputs,
            diagnostic_counts: TimingQueryDiagnosticCounts::default(),
        };
        for index in topo_order {
            state.recompute_instance(index)?;
        }
        Ok(state)
    }

    /// Resolves the currently committed Liberty cell at one mapped instance.
    fn cell_name(&self, instance_index: usize) -> &str {
        self.library.cells[self.instances[instance_index].cell_index]
            .name
            .as_str()
    }

    /// Returns endpoint delays in stable worst-first order.
    fn score(&self) -> TimingScore {
        let mut endpoint_delays: Vec<f64> = self
            .outputs
            .iter()
            .filter_map(|net| self.net_timing[net.0])
            .map(|timing| timing.rise.arrival.max(timing.fall.arrival))
            .collect();
        endpoint_delays.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
        TimingScore { endpoint_delays }
    }

    /// Evaluates or commits one replacement while updating only its dirty cone.
    fn evaluate_cell_substitution(
        &mut self,
        instance_index: usize,
        new_cell_index: usize,
        commit: bool,
    ) -> Result<TrialResult> {
        let original = self
            .instances
            .get(instance_index)
            .ok_or_else(|| anyhow!("resizer instance index is out of range"))?
            .clone();
        let new_cell = self
            .library
            .cells
            .get(new_cell_index)
            .ok_or_else(|| anyhow!("resizer replacement cell is out of range"))?;
        let old_cell = &self.library.cells[original.cell_index];
        let mut changed_loads = Vec::new();
        for input in &original.inputs {
            let Some(net) = input.net else {
                continue;
            };
            let old_pin = old_cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == input.name)
                .ok_or_else(|| anyhow!("resizer old cell is missing input '{}'", input.name))?;
            let new_pin = new_cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == input.name)
                .ok_or_else(|| anyhow!("resizer new cell is missing input '{}'", input.name))?;
            if new_pin.direction != old_pin.direction {
                return Err(anyhow!("resizer replacement changes input pin direction"));
            }
            let old_cap = effective_input_capacitance_for_mapping(
                old_pin,
                format!("old resize input '{}.{}'", old_cell.name, input.name).as_str(),
            )?;
            let new_cap = effective_input_capacitance_for_mapping(
                new_pin,
                format!("new resize input '{}.{}'", new_cell.name, input.name).as_str(),
            )?;
            changed_loads.push((net, self.loads[net.0], old_cap, new_cap));
        }

        let old_output = &old_cell.pins[original.output_pin_index];
        let output_name = self.library.resolve_string(&old_output.name);
        let new_output_pin_index = new_cell
            .pins
            .iter()
            .position(|pin| {
                pin.direction == PinDirection::Output as i32
                    && self.library.resolve_string(&pin.name) == output_name
            })
            .ok_or_else(|| anyhow!("resizer replacement changes the output pin interface"))?;

        let mut dirty = BTreeSet::new();
        dirty.insert((self.topo_positions[instance_index], instance_index));
        for (net, _, _, _) in &changed_loads {
            if let Some(driver) = self.drivers[net.0] {
                dirty.insert((self.topo_positions[driver], driver));
            }
        }
        let mut saved_timings = Vec::new();

        self.instances[instance_index].cell_index = new_cell_index;
        self.instances[instance_index].output_pin_index = new_output_pin_index;
        for (net, _, old_cap, new_cap) in &changed_loads {
            self.loads[net.0].rise += new_cap.rise - old_cap.rise;
            self.loads[net.0].fall += new_cap.fall - old_cap.fall;
            if self.loads[net.0].rise < -1e-12 || self.loads[net.0].fall < -1e-12 {
                self.restore_trial(&original, &changed_loads, &saved_timings);
                return Err(anyhow!("resizer substitution produced a negative net load"));
            }
            self.loads[net.0].rise = self.loads[net.0].rise.max(0.0);
            self.loads[net.0].fall = self.loads[net.0].fall.max(0.0);
        }

        let result = (|| {
            while let Some((_, index)) = dirty.pop_first() {
                let output_net = self.instances[index].output_net;
                let previous_timing = self.net_timing[output_net.0];
                saved_timings.push((output_net, previous_timing));
                self.recompute_instance(index)?;
                if self.net_timing[output_net.0] != previous_timing {
                    for successor in &self.successors[index] {
                        dirty.insert((self.topo_positions[*successor], *successor));
                    }
                }
            }
            Ok(TrialResult {
                score: self.score(),
                recomputed_instances: saved_timings.len(),
            })
        })();

        if result.is_err() || !commit {
            self.restore_trial(&original, &changed_loads, &saved_timings);
        }
        result
    }

    /// Restores changed cell choice, input loads, and dirty-cone net timings.
    fn restore_trial(
        &mut self,
        original: &InstanceTiming,
        changed_loads: &[(
            NetIndex,
            CombinationalOutputLoad,
            CombinationalOutputLoad,
            CombinationalOutputLoad,
        )],
        saved_timings: &[(NetIndex, Option<SignalTiming>)],
    ) {
        let index = self.drivers[original.output_net.0]
            .expect("a sized instance must remain its original net's driver");
        self.instances[index] = original.clone();
        for (net, old_load, _, _) in changed_loads {
            self.loads[net.0] = *old_load;
        }
        for (net, timing) in saved_timings {
            self.net_timing[net.0] = *timing;
        }
    }

    /// Recomputes one mapped cell using the exact gv-stats NLDM evaluator.
    fn recompute_instance(&mut self, instance_index: usize) -> Result<()> {
        let instance = &self.instances[instance_index];
        let cell = &self.library.cells[instance.cell_index];
        let mut input_timings = Vec::with_capacity(instance.inputs.len());
        for input in &instance.inputs {
            if let Some(net) = input.net {
                let timing = self.net_timing[net.0].ok_or_else(|| {
                    anyhow!(
                        "cell '{}.{}' depends on a net with no timing source",
                        cell.name,
                        input.name
                    )
                })?;
                input_timings.push((input.name.as_str(), timing));
            }
        }
        let output = evaluate_combinational_cell_output_timing(
            self.library,
            cell.name.as_str(),
            &cell.pins[instance.output_pin_index],
            input_timings.as_slice(),
            self.loads[instance.output_net.0],
            &instance.known_pin_values,
            &mut self.diagnostic_counts,
        )?;
        self.net_timing[instance.output_net.0] = Some(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IncrementalCombinationalSta, ResizeOptions, resize_netlist};
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};
    use crate::netlist::report::build_sta_report;
    use crate::netlist::sta::StaOptions;

    #[test]
    fn upsizes_a_critical_combinational_gate() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
        )
        .unwrap();

        assert_eq!(stats.upsizes, 1);
        assert_eq!(stats.downsizes, 0);
        assert_eq!(stats.initial_clock_load, None);
        assert_eq!(stats.final_clock_load, None);
        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AND2_FAST")
        );
        let full =
            build_sta_report(&module, &nets, &interner, &library, StaOptions::default()).unwrap();
        assert!((stats.final_delay - full.delay).abs() < 1e-9);
    }

    #[test]
    fn recovers_noncritical_area_without_increasing_delay() {
        let source = r#"
module top(a, b, critical, noncritical);
  input a, b;
  output critical, noncritical;
  AND2 slow_path (.A(a), .B(b), .Y(critical));
  BUF_FAST easy_path (.A(a), .Y(noncritical));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_iterations: 0,
                ..ResizeOptions::default()
            },
        )
        .unwrap();

        assert_eq!(stats.upsizes, 0);
        assert_eq!(stats.downsizes, 1);
        assert!(stats.final_area < stats.initial_area);
        assert!(stats.final_delay <= stats.initial_delay + 1e-9);
        assert_eq!(interner.resolve(module.instances[1].type_name), Some("BUF"));
    }

    #[test]
    fn rejected_incremental_trial_restores_exact_timing() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (module, nets, interner) = parse_module(source);
        let mut state = IncrementalCombinationalSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
        )
        .unwrap();
        let before = state.score().worst_delay();
        let fast_index = library
            .cells
            .iter()
            .position(|cell| cell.name == "AND2_FAST")
            .unwrap();
        let trial = state
            .evaluate_cell_substitution(0, fast_index, false)
            .unwrap();

        assert!(trial.score.worst_delay() < before);
        assert_eq!(state.score().worst_delay(), before);
        assert_eq!(state.cell_name(0), "AND2");
    }

    #[test]
    fn incremental_trial_stops_at_unchanged_upstream_timing() {
        let source = r#"
module top(a, b, y, other);
  input a, b;
  output y, other;
  wire root;
  BUF driver (.A(a), .Y(root));
  AND2 logic (.A(root), .B(b), .Y(y));
  BUF unrelated (.A(root), .Y(other));
endmodule
"#;
        let library = sizing_library();
        let (module, nets, interner) = parse_module(source);
        let mut state = IncrementalCombinationalSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
        )
        .unwrap();
        let before = state.score().worst_delay();
        let fast_index = library
            .cells
            .iter()
            .position(|cell| cell.name == "AND2_FAST")
            .unwrap();
        let trial = state
            .evaluate_cell_substitution(1, fast_index, false)
            .unwrap();

        assert_eq!(trial.recomputed_instances, 2);
        assert!(trial.score.worst_delay() < before);
        assert_eq!(state.score().worst_delay(), before);
    }

    #[test]
    fn sizes_logic_without_materializing_constant_output_cells() {
        let source = r#"
module top(a, b, y, zero, one);
  input a, b;
  output y, zero, one;
  assign zero = 1'b0;
  assign one = 1'b1;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
        )
        .expect("constant output assignments should not prevent sizing");

        assert_eq!(stats.upsizes, 1);
        assert_eq!(module.instances.len(), 1);
        assert_eq!(module.assigns.len(), 2);
        let full = build_sta_report(&module, &nets, &interner, &library, StaOptions::default())
            .expect("sized constant-output module should remain timing-complete");
        assert!((stats.final_delay - full.delay).abs() < 1e-9);
    }
}
