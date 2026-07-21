// SPDX-License-Identifier: Apache-2.0

//! Deterministic reconstruction of a selected cover into parsed netlist data.

use crate::aig::ChoiceAig;
use crate::netlist::parse::{
    Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
};
use crate::techmap::cover::{CoverPlan, SolutionChoice, SolutionId, SourceKind};
use crate::techmap::liberty_index::{CellBinding, LibertyCellIndex};
use crate::techmap::{TechMapOptions, scalar_bit_name};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, HashMap, HashSet};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use xlsynth::IrBits;

/// Netlist data and actual emitted cell-area accounting.
pub(super) struct EmittedNetlist {
    pub module: NetlistModule,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
    pub area: f64,
}

#[derive(Clone, Copy, Debug)]
enum Signal {
    Net(NetIndex),
    Literal(bool),
}

#[derive(Clone, Debug)]
struct OutputPlan {
    name: String,
    net: NetIndex,
    solution: SolutionId,
}

/// Emits the selected final cover without any ABC feedback representation.
pub(super) fn emit_cover(
    choice_aig: &ChoiceAig,
    plan: &CoverPlan,
    cell_index: &LibertyCellIndex,
    options: &TechMapOptions,
) -> Result<EmittedNetlist> {
    let graph = choice_aig.graph();
    let module_name = options
        .module_name
        .clone()
        .unwrap_or_else(|| graph.name.clone());
    let mut builder = NetlistBuilder::new(module_name.as_str());
    let mut input_net_by_node = HashMap::new();

    for input in &graph.inputs {
        let bit_count = input.get_bit_count();
        for (bit_index, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            if bit.negated {
                return Err(anyhow!(
                    "technology mapping does not support negated input-port bindings"
                ));
            }
            let name = scalar_bit_name(input.name.as_str(), bit_index, bit_count);
            let net = builder.add_port(name.as_str(), PortDirection::Input)?;
            if let Some(previous) = input_net_by_node.insert(bit.node.id, net) {
                if previous != net {
                    return Err(anyhow!(
                        "AIG input node {} is bound to multiple primary-input ports",
                        bit.node.id
                    ));
                }
            }
        }
    }

    let expected_output_count: usize = graph
        .outputs
        .iter()
        .map(|output| output.get_bit_count())
        .sum();
    if plan.output_solutions.len() != expected_output_count {
        return Err(anyhow!(
            "cover has {} output solutions but graph has {} output bits",
            plan.output_solutions.len(),
            expected_output_count
        ));
    }
    let mut output_plans = Vec::with_capacity(expected_output_count);
    let mut output_index = 0usize;
    for output in &graph.outputs {
        let bit_count = output.get_bit_count();
        for (bit_index, _) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            let name = scalar_bit_name(output.name.as_str(), bit_index, bit_count);
            let net = builder.add_port(name.as_str(), PortDirection::Output)?;
            output_plans.push(OutputPlan {
                name,
                net,
                solution: plan.output_solutions[output_index],
            });
            output_index += 1;
        }
    }

    let mut owner_net_by_solution = BTreeMap::new();
    for output in &output_plans {
        if matches!(
            plan.solutions[output.solution.0].choice,
            SolutionChoice::Cell { .. }
        ) {
            owner_net_by_solution
                .entry(output.solution)
                .or_insert(output.net);
        }
    }

    let mut emitter = CoverEmitter {
        plan,
        cell_index,
        builder: &mut builder,
        input_net_by_node,
        owner_net_by_solution,
        signal_by_solution: HashMap::new(),
    };
    for output in &output_plans {
        let signal = emitter.emit_solution(output.solution)?;
        emitter.connect_output(signal, output.net, output.name.as_str())?;
    }

    Ok(builder.finish())
}

struct CoverEmitter<'a> {
    plan: &'a CoverPlan,
    cell_index: &'a LibertyCellIndex,
    builder: &'a mut NetlistBuilder,
    input_net_by_node: HashMap<usize, NetIndex>,
    owner_net_by_solution: BTreeMap<SolutionId, NetIndex>,
    signal_by_solution: HashMap<SolutionId, Signal>,
}

impl CoverEmitter<'_> {
    fn emit_solution(&mut self, solution_id: SolutionId) -> Result<Signal> {
        if let Some(signal) = self.signal_by_solution.get(&solution_id).copied() {
            return Ok(signal);
        }
        let choice = self
            .plan
            .solutions
            .get(solution_id.0)
            .ok_or_else(|| anyhow!("cover references missing solution {}", solution_id.0))?
            .choice
            .clone();
        let signal = match choice {
            SolutionChoice::Source(SourceKind::Input(node)) => {
                let net = self
                    .input_net_by_node
                    .get(&node.id)
                    .copied()
                    .ok_or_else(|| {
                        anyhow!(
                            "cover input source {} is not bound to a primary-input port",
                            node.id
                        )
                    })?;
                Signal::Net(net)
            }
            SolutionChoice::Source(SourceKind::Literal(value)) => Signal::Literal(value),
            SolutionChoice::Cell { binding, inputs } => {
                let input_signals = inputs
                    .iter()
                    .map(|input| self.emit_solution(*input))
                    .collect::<Result<Vec<_>>>()?;
                let output_net = self
                    .owner_net_by_solution
                    .get(&solution_id)
                    .copied()
                    .unwrap_or_else(|| self.builder.fresh_internal_net());
                self.builder
                    .add_cell(binding, input_signals.as_slice(), output_net)?;
                Signal::Net(output_net)
            }
        };
        self.signal_by_solution.insert(solution_id, signal);
        Ok(signal)
    }

    fn connect_output(
        &mut self,
        signal: Signal,
        output_net: NetIndex,
        output_name: &str,
    ) -> Result<()> {
        match signal {
            Signal::Net(net) if net == output_net => Ok(()),
            Signal::Literal(value) => {
                let binding = self.cell_index.best_constant(value).ok_or_else(|| {
                    anyhow!(
                        "output '{}' is constant {} but Liberty has no matching zero-input constant driver",
                        output_name,
                        u8::from(value)
                    )
                })?;
                self.builder.add_cell(binding.clone(), &[], output_net)
            }
            Signal::Net(net) => {
                if let Some(buffer) = self.cell_index.best_buffer() {
                    return self
                        .builder
                        .add_cell(buffer.clone(), &[Signal::Net(net)], output_net);
                }
                let inverter = self.cell_index.best_inverter().ok_or_else(|| {
                    anyhow!(
                        "output '{}' needs an identity connection but Liberty has neither a buffer nor an inverter",
                        output_name
                    )
                })?;
                let intermediate = self.builder.fresh_internal_net();
                self.builder
                    .add_cell(inverter.clone(), &[Signal::Net(net)], intermediate)?;
                self.builder
                    .add_cell(inverter.clone(), &[Signal::Net(intermediate)], output_net)
            }
        }
    }
}

struct NetlistBuilder {
    module_name: SymbolU32,
    interner: StringInterner<StringBackend<SymbolU32>>,
    nets: Vec<Net>,
    net_index_by_sym: HashMap<SymbolU32, NetIndex>,
    port_net_indices: HashSet<NetIndex>,
    wire_net_indices: Vec<NetIndex>,
    wire_net_set: HashSet<NetIndex>,
    ports: Vec<NetlistPort>,
    instances: Vec<NetlistInstance>,
    used_net_names: HashSet<String>,
    instance_counter: usize,
    internal_net_counter: usize,
    area: f64,
}

impl NetlistBuilder {
    fn new(module_name: &str) -> Self {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let module_name = interner.get_or_intern(module_name);
        Self {
            module_name,
            interner,
            nets: Vec::new(),
            net_index_by_sym: HashMap::new(),
            port_net_indices: HashSet::new(),
            wire_net_indices: Vec::new(),
            wire_net_set: HashSet::new(),
            ports: Vec::new(),
            instances: Vec::new(),
            used_net_names: HashSet::new(),
            instance_counter: 0,
            internal_net_counter: 0,
            area: 0.0,
        }
    }

    fn add_port(&mut self, name: &str, direction: PortDirection) -> Result<NetIndex> {
        if !self.used_net_names.insert(name.to_string()) {
            return Err(anyhow!("duplicate mapped port name '{}'", name));
        }
        let name_sym = self.interner.get_or_intern(name);
        let net = self.ensure_net(name, true);
        self.ports.push(NetlistPort {
            direction,
            width: None,
            name: name_sym,
        });
        Ok(net)
    }

    fn fresh_internal_net(&mut self) -> NetIndex {
        loop {
            let name = format!("n_tm_{}", self.internal_net_counter);
            self.internal_net_counter += 1;
            if self.used_net_names.insert(name.clone()) {
                return self.ensure_net(name.as_str(), false);
            }
        }
    }

    fn ensure_net(&mut self, name: &str, is_port: bool) -> NetIndex {
        let sym = self.interner.get_or_intern(name);
        let net = if let Some(existing) = self.net_index_by_sym.get(&sym).copied() {
            existing
        } else {
            let created = NetIndex(self.nets.len());
            self.nets.push(Net {
                name: sym,
                width: None,
            });
            self.net_index_by_sym.insert(sym, created);
            created
        };
        if is_port {
            if self.port_net_indices.insert(net) && self.wire_net_set.remove(&net) {
                self.wire_net_indices.retain(|candidate| *candidate != net);
            }
        } else if !self.port_net_indices.contains(&net) && self.wire_net_set.insert(net) {
            self.wire_net_indices.push(net);
        }
        net
    }

    fn add_cell(
        &mut self,
        binding: CellBinding,
        input_signals: &[Signal],
        output_net: NetIndex,
    ) -> Result<()> {
        if input_signals.len() != binding.input_pin_names.len() {
            return Err(anyhow!(
                "cell '{}' expects {} inputs but emitter received {}",
                binding.cell_name,
                binding.input_pin_names.len(),
                input_signals.len()
            ));
        }
        let instance_name = format!("u_tm_{}", self.instance_counter);
        self.instance_counter += 1;
        let mut connections: Vec<(SymbolU32, NetRef)> = binding
            .input_pin_names
            .iter()
            .zip(input_signals.iter())
            .map(|(pin_name, signal)| {
                (
                    self.interner.get_or_intern(pin_name.as_str()),
                    signal_to_netref(*signal),
                )
            })
            .collect();
        connections.push((
            self.interner
                .get_or_intern(binding.output_pin_name.as_str()),
            NetRef::Simple(output_net),
        ));
        connections.sort_by(|(lhs, _), (rhs, _)| {
            self.interner
                .resolve(*lhs)
                .unwrap_or("")
                .cmp(self.interner.resolve(*rhs).unwrap_or(""))
        });
        self.instances.push(NetlistInstance {
            type_name: self.interner.get_or_intern(binding.cell_name.as_str()),
            instance_name: self.interner.get_or_intern(instance_name.as_str()),
            connections,
            inst_lineno: 1,
            inst_colno: 1,
        });
        self.area += binding.area;
        Ok(())
    }

    fn finish(self) -> EmittedNetlist {
        let module = NetlistModule {
            name: self.module_name,
            net_index_range: 0..self.nets.len(),
            ports: self.ports,
            wires: self.wire_net_indices,
            assigns: Vec::new(),
            instances: self.instances,
        };
        EmittedNetlist {
            module,
            nets: self.nets,
            interner: self.interner,
            area: self.area,
        }
    }
}

fn signal_to_netref(signal: Signal) -> NetRef {
    match signal {
        Signal::Net(net) => NetRef::Simple(net),
        Signal::Literal(value) => NetRef::Literal(
            IrBits::make_ubits(1, u64::from(value)).expect("one-bit literal should be valid"),
        ),
    }
}
