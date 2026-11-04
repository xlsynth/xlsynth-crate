// SPDX-License-Identifier: Apache-2.0

//! Prepare scalar combinational logic and restore physical sequential cells.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::IrBits;

use super::SequentialMappingOptions;
use crate::aig::gate::{Input, Output};
use crate::aig::sequential_cleanup::{SequentialCleanupStats, cleanup_sequential_gate_fn};
use crate::aig::{AigBitVector, AigNode, AigOperand, GateFn, SequentialGateFn, TransitionInputId};
use crate::liberty_model::{ClockGate, Library};
use crate::netlist::io::ParsedNetlist;
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistAssignKind, NetlistInstance,
    NetlistPort, PortDirection, Pos, Span,
};
use crate::netlist::register_mapping::{
    RegisterBoundaryPorts, RegisterMappingLibrary, RegisterMappingPlan,
    validate_sequential_mapping_policy,
};

/// Physical register and clock-gate choices in one normalized library.
pub(crate) struct SequentialMappingCatalog {
    registers: RegisterMappingLibrary,
    /// Index of the smallest usable, certified ICG in the normalized library.
    clock_gate_cell: Option<usize>,
}

/// Legalized design, selected cells, and scalar endpoints for a single mapping.
pub(crate) struct PreparedSequentialMapping<'lib> {
    pub(crate) design: SequentialGateFn,
    pub(crate) registers: RegisterMappingPlan,
    pub(crate) cleanup_stats: Option<SequentialCleanupStats>,
    boundary: Boundary,
    gates: Option<ClockGatePlan<'lib>>,
}

impl SequentialMappingCatalog {
    pub(crate) fn new(library: &Library) -> Result<Self, String> {
        let registers = RegisterMappingLibrary::new(library)?;
        let clock_gate_cell = library
            .cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| {
                cell.clock_gate.is_some()
                    && validate_sequential_mapping_policy(cell).is_ok()
                    && cell.positive_edge_clock_gate(library).is_ok()
            })
            .min_by(|(_, a), (_, b)| a.area.total_cmp(&b.area).then_with(|| a.name.cmp(&b.name)))
            .map(|(index, _)| index);
        Ok(Self {
            registers,
            clock_gate_cell,
        })
    }

    /// Legalizes state controls, selects physical cells, and prepares their
    /// logic boundary.
    pub(crate) fn prepare<'lib>(
        &self,
        library: &'lib Library,
        source: &SequentialGateFn,
        sequential: &SequentialMappingOptions,
    ) -> Result<PreparedSequentialMapping<'lib>, String> {
        let mut design = source.legalize_register_controls()?;
        let cleanup_stats = if sequential.cleanup_registers {
            let result = cleanup_sequential_gate_fn(&design)?;
            design = result.design;
            Some(result.stats)
        } else {
            None
        };
        validate_names(&design)?;
        if library.cells.iter().any(|c| c.name == design.name) {
            return Err("top module name conflicts with a Liberty cell name".into());
        }
        let ports = RegisterBoundaryPorts::new(&design);
        let gates = self.plan_clock_gates(library, &design, sequential, &ports)?;
        let registers = self.registers.map_registers_with_data(
            &design,
            &sequential.register_selection,
            &ports,
            |reg, bit| gates.as_ref()?.bits.get(&(reg.q, bit)).map(|g| g.data),
        )?;
        let boundary = Boundary::new(&design, &registers, gates.as_ref());
        Ok(PreparedSequentialMapping {
            design,
            registers,
            cleanup_stats,
            boundary,
            gates,
        })
    }

    /// Keeps only clock-gate groups that preserve the direct FF boundary wires.
    fn plan_clock_gates<'lib>(
        &self,
        library: &'lib Library,
        design: &SequentialGateFn,
        options: &SequentialMappingOptions,
        ports: &RegisterBoundaryPorts,
    ) -> Result<Option<ClockGatePlan<'lib>>, String> {
        let Some(policy) = &options.clock_gating else {
            return Ok(None);
        };
        policy.validate()?;
        if design.registers.is_empty() {
            return Ok(None);
        }
        let Some(index) = self.clock_gate_cell else {
            return Ok(None);
        };
        let cell = &library.cells[index];
        // Bypassing a mux can expose a primary-input -> FF -> primary-output
        // path. Such a bit needs a non-inverting FF to preserve both wires.
        let direct_io_ok = self
            .registers
            .supports_direct_input_to_output(&options.register_selection);
        let groups = super::plan_clock_gates(design, policy, |q, data| {
            direct_io_ok || !(ports.is_direct_input(data) && ports.is_direct_output(q))
        });
        if groups.is_empty() {
            return Ok(None);
        }
        let mut enables = Vec::with_capacity(groups.len());
        let mut bits = BTreeMap::new();
        for (group_index, group) in groups.into_iter().enumerate() {
            enables.push(group.enable);
            for member in group.members {
                let q = design.registers[member.register_index].q;
                bits.insert(
                    (q, member.bit_index),
                    GatedBit {
                        data: member.data,
                        group_index,
                    },
                );
            }
        }
        Ok(Some(ClockGatePlan {
            cell_name: &cell.name,
            pins: cell.clock_gate.as_ref().expect("certified clock gate"),
            enables,
            bits,
        }))
    }
}

/// A boundary endpoint either aliases an input net or consumes mapped logic.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BoundarySignal {
    Direct(String),
    Mapped(String),
}

impl BoundarySignal {
    fn port_name(&self) -> &str {
        match self {
            Self::Direct(name) | Self::Mapped(name) => name,
        }
    }
}

/// Load operand and shared clock for an individual physical FF bit.
struct GatedBit {
    data: AigOperand,
    group_index: usize,
}

/// Native clock-gate wiring and the corresponding combinational mapper
/// boundary.
struct ClockGatePlan<'a> {
    cell_name: &'a str,
    pins: &'a ClockGate,
    enables: Vec<AigOperand>,
    bits: BTreeMap<(TransitionInputId, usize), GatedBit>,
}

/// Scalar mapping interface that preserves direct wires across the boundary.
struct Boundary {
    graph: GateFn,
    inputs: BTreeMap<(usize, usize), String>,
    external_outputs: Vec<Vec<BoundarySignal>>,
    register_data: Vec<BoundarySignal>,
    gate_enables: Vec<BoundarySignal>,
}

impl Boundary {
    fn new(
        design: &SequentialGateFn,
        plan: &RegisterMappingPlan,
        gates: Option<&ClockGatePlan<'_>>,
    ) -> Self {
        let mut graph = design.transition.clone();
        graph.name = "mapped".into();
        graph.inputs.clear();
        graph.outputs.clear();
        let mut phases = vec![false; graph.gates.len()];
        for bit in &plan.bits {
            let q = design.transition.inputs[bit.q.index()]
                .bit_vector
                .get_lsb(bit.bit_index);
            phases[q.node.id] = bit.output_inverted;
        }
        let rephase = |mut operand: AigOperand| {
            operand.negated ^= phases[operand.node.id];
            operand
        };
        // Rewrite every use, including feedback, while the input nodes now
        // denote physical FF outputs. No explicit inverter nodes are needed.
        for gate in &mut graph.gates {
            if let AigNode::And2 { a, b, .. } = gate {
                *a = rephase(*a);
                *b = rephase(*b);
            }
        }
        let mut inputs = BTreeMap::new();
        let mut input_nodes = BTreeMap::new();
        for (id, input) in design.transition.inputs.iter().enumerate() {
            for (bit, operand) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                let name = format!("i{}", graph.inputs.len());
                graph.inputs.push(Input {
                    name: name.clone(),
                    bit_vector: AigBitVector::from_bit(*operand),
                });
                input_nodes.insert(operand.node.id, name.clone());
                inputs.insert((id, bit), name);
            }
        }
        let mut external_outputs = Vec::new();
        for id in &design.outputs {
            let mut names = Vec::new();
            for operand in design.transition.outputs[id.index()]
                .bit_vector
                .iter_lsb_to_msb()
            {
                names.push(add_boundary_signal(
                    &mut graph,
                    rephase(*operand),
                    &input_nodes,
                ));
            }
            external_outputs.push(names);
        }
        let mut register_data = Vec::new();
        for bit in &plan.bits {
            let gated_bit = gates.and_then(|g| g.bits.get(&(bit.q, bit.bit_index)));
            let effective_d = design.transition.outputs[bit.d.index()]
                .bit_vector
                .get_lsb(bit.bit_index);
            let data = gated_bit.map_or(*effective_d, |g| g.data);
            let mut operand = rephase(data);
            // Physical D = load data XOR cell polarity XOR state phase.
            // `rephase` expresses the load operand using physical state inputs.
            operand.negated ^=
                plan.cells[bit.cell_index].inverts_data_to_output ^ bit.output_inverted;
            register_data.push(add_boundary_signal(&mut graph, operand, &input_nodes));
        }
        let gate_enables = gates
            .into_iter()
            .flat_map(|g| &g.enables)
            .map(|enable| add_boundary_signal(&mut graph, rephase(*enable), &input_nodes))
            .collect();
        Self {
            graph,
            inputs,
            external_outputs,
            register_data,
            gate_enables,
        }
    }
}

fn add_boundary_signal(
    graph: &mut GateFn,
    operand: AigOperand,
    input_nodes: &BTreeMap<usize, String>,
) -> BoundarySignal {
    if !operand.negated {
        if let Some(input) = input_nodes.get(&operand.node.id) {
            // A direct connection bypasses combinational mapping and any
            // buffering the backend might apply to an output.
            return BoundarySignal::Direct(input.clone());
        }
    }
    let name = format!("o{}", graph.outputs.len());
    graph.outputs.push(Output {
        name: name.clone(),
        bit_vector: AigBitVector::from_bit(operand),
    });
    BoundarySignal::Mapped(name)
}

/// Rejects unrepresentable names and collisions in the emitted port namespace.
fn validate_names(design: &SequentialGateFn) -> Result<(), String> {
    // Module names have their own namespace, and zero-width logical ports are
    // not emitted. Validate their spelling without reserving a physical port.
    let names = std::iter::once((design.name.as_str(), false))
        .chain(design.inputs.iter().map(|id| {
            let port = &design.transition.inputs[id.index()];
            (port.name.as_str(), port.get_bit_count() != 0)
        }))
        .chain(design.outputs.iter().map(|id| {
            let port = &design.transition.outputs[id.index()];
            (port.name.as_str(), port.get_bit_count() != 0)
        }))
        .chain(design.clock.iter().map(|c| (c.name.as_str(), true)));
    let mut physical_ports = BTreeSet::new();
    for (name, physical_port) in names {
        if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(format!(
                "name cannot be represented as a Verilog identifier: {name:?}"
            ));
        }
        if physical_port && !physical_ports.insert(name) {
            return Err(format!("duplicate external port or clock name: {name:?}"));
        }
    }
    Ok(())
}

impl PreparedSequentialMapping<'_> {
    /// Scalar combinational graph passed to the selected technology mapper.
    pub(crate) fn graph(&self) -> &GateFn {
        &self.boundary.graph
    }

    /// Replaces the mapped scalar interface with external ports, FFs, and ICGs.
    pub(crate) fn assemble(&self, parsed: &mut ParsedNetlist, backend: &str) -> Result<(), String> {
        let design = &self.design;
        let plan = &self.registers;
        let boundary = &self.boundary;
        let gates = self.gates.as_ref();
        let prefix = backend.to_ascii_lowercase();
        let [module] = parsed.modules.as_slice() else {
            return Err(format!(
                "{backend} must emit exactly one combinational module"
            ));
        };
        let mut ports = BTreeMap::new();
        for port in &module.ports {
            let name = parsed.interner.resolve(port.name).unwrap().to_string();
            if port.width.is_some() {
                return Err(format!("{backend} boundary ports must be scalar"));
            }
            let net = module
                .find_net_index(port.name, &parsed.nets)
                .ok_or_else(|| format!("{backend} port net missing"))?;
            ports.insert(name, (port.direction.clone(), net));
        }
        let expected = boundary
            .graph
            .inputs
            .iter()
            .map(|p| (p.name.clone(), PortDirection::Input))
            .chain(
                boundary
                    .graph
                    .outputs
                    .iter()
                    .map(|p| (p.name.clone(), PortDirection::Output)),
            )
            .collect::<BTreeMap<_, _>>();
        if ports
            .iter()
            .map(|(n, (d, _))| (n.clone(), d.clone()))
            .collect::<BTreeMap<_, _>>()
            != expected
        {
            return Err(format!(
                "{backend} changed the scalar boundary names, directions, or counts"
            ));
        }
        let mut used = design
            .inputs
            .iter()
            .map(|id| design.transition.inputs[id.index()].name.clone())
            .chain(
                design
                    .outputs
                    .iter()
                    .map(|id| design.transition.outputs[id.index()].name.clone()),
            )
            .chain(design.clock.iter().map(|c| c.name.clone()))
            .collect::<BTreeSet<_>>();
        for (i, net) in parsed.nets.iter_mut().enumerate() {
            net.name = parsed
                .interner
                .get_or_intern(fresh_name(&mut used, format!("__{prefix}_n{i}")));
        }
        let module = &mut parsed.modules[0];
        module.name = parsed.interner.get_or_intern(&design.name);
        module.ports.clear();
        module.wires = (0..parsed.nets.len()).map(NetIndex).collect();
        for (i, instance) in module.instances.iter_mut().enumerate() {
            instance.instance_name = parsed
                .interner
                .get_or_intern(fresh_name(&mut used, format!("__{prefix}_c{i}")));
        }
        for id in &design.inputs {
            let input = &design.transition.inputs[id.index()];
            if let Some(net) = add_port(
                parsed,
                &input.name,
                input.get_bit_count(),
                PortDirection::Input,
            )? {
                for bit in 0..input.get_bit_count() {
                    add_alias(
                        parsed,
                        NetRef::Simple(ports[&boundary.inputs[&(id.index(), bit)]].1),
                        bit_ref(net, bit, input.get_bit_count()),
                    );
                }
            }
        }
        for (index, id) in design.outputs.iter().enumerate() {
            let output = &design.transition.outputs[id.index()];
            if let Some(net) = add_port(
                parsed,
                &output.name,
                output.get_bit_count(),
                PortDirection::Output,
            )? {
                for bit in 0..output.get_bit_count() {
                    add_alias(
                        parsed,
                        bit_ref(net, bit, output.get_bit_count()),
                        NetRef::Simple(ports[boundary.external_outputs[index][bit].port_name()].1),
                    );
                }
            }
        }
        let clock = design
            .clock
            .as_ref()
            .map(|clock| add_port(parsed, &clock.name, 1, PortDirection::Input))
            .transpose()?
            .flatten();
        let mut gated_clocks = Vec::with_capacity(boundary.gate_enables.len());
        if let Some(gates) = gates {
            let clock = clock.ok_or("mapped ICG has no clock")?;
            for (index, enable) in boundary.gate_enables.iter().enumerate() {
                let gated_clock = NetIndex(parsed.nets.len());
                parsed.nets.push(Net {
                    name: parsed
                        .interner
                        .get_or_intern(fresh_name(&mut used, format!("__gclk_{index}"))),
                    width: None,
                });
                parsed.modules[0].wires.push(gated_clock);
                let pins = gates.pins;
                let mut connections = vec![
                    (
                        parsed.interner.get_or_intern(&pins.clock_pin),
                        NetRef::Simple(clock),
                    ),
                    (
                        parsed.interner.get_or_intern(&pins.enable_pins[0]),
                        NetRef::Simple(ports[enable.port_name()].1),
                    ),
                    (
                        parsed.interner.get_or_intern(&pins.output_pin),
                        NetRef::Simple(gated_clock),
                    ),
                ];
                for pin in &pins.test_pins {
                    connections.push((
                        parsed.interner.get_or_intern(pin),
                        NetRef::Literal(IrBits::bool(false)),
                    ));
                }
                parsed.modules[0].instances.push(NetlistInstance {
                    type_name: parsed.interner.get_or_intern(gates.cell_name),
                    instance_name: parsed
                        .interner
                        .get_or_intern(fresh_name(&mut used, format!("__icg_{index}"))),
                    connections,
                    inst_lineno: 0,
                    inst_colno: 0,
                });
                gated_clocks.push(gated_clock);
            }
        }
        for (index, bit) in plan.bits.iter().enumerate() {
            let cell = &plan.cells[bit.cell_index];
            let clock = gates
                .and_then(|g| g.bits.get(&(bit.q, bit.bit_index)))
                .map(|g| gated_clocks[g.group_index])
                .or(clock)
                .ok_or("mapped FF has no clock")?;
            let mut connections = vec![
                (
                    parsed.interner.get_or_intern(&cell.clock_pin),
                    NetRef::Simple(clock),
                ),
                (
                    parsed.interner.get_or_intern(&cell.data_pin),
                    NetRef::Simple(ports[boundary.register_data[index].port_name()].1),
                ),
                (
                    parsed.interner.get_or_intern(&cell.output_pin),
                    NetRef::Simple(ports[&boundary.inputs[&(bit.q.index(), bit.bit_index)]].1),
                ),
            ];
            for (pin, value) in &cell.tied_inputs {
                connections.push((
                    parsed.interner.get_or_intern(pin),
                    NetRef::Literal(IrBits::from_lsb_is_0(&[*value])),
                ));
            }
            for pin in &cell.unused_outputs {
                connections.push((parsed.interner.get_or_intern(pin), NetRef::Unconnected));
            }
            parsed.modules[0].instances.push(NetlistInstance {
                type_name: parsed.interner.get_or_intern(&cell.cell_name),
                instance_name: parsed
                    .interner
                    .get_or_intern(fresh_name(&mut used, format!("__reg_{index}"))),
                connections,
                inst_lineno: 0,
                inst_colno: 0,
            });
        }
        parsed.modules[0].net_index_range = 0..parsed.nets.len();
        Ok(())
    }
}

fn fresh_name(used: &mut BTreeSet<String>, base: String) -> String {
    let mut name = base.clone();
    let mut suffix = 0;
    while !used.insert(name.clone()) {
        suffix += 1;
        name = format!("{base}_{suffix}");
    }
    name
}

fn add_port(
    parsed: &mut ParsedNetlist,
    name: &str,
    width: usize,
    direction: PortDirection,
) -> Result<Option<NetIndex>, String> {
    if width == 0 {
        return Ok(None);
    }
    let width = if width == 1 {
        None
    } else {
        Some((
            u32::try_from(width - 1).map_err(|_| "port width exceeds Verilog index range")?,
            0,
        ))
    };
    let symbol = parsed.interner.get_or_intern(name);
    let index = NetIndex(parsed.nets.len());
    parsed.nets.push(Net {
        name: symbol,
        width,
    });
    parsed.modules[0].ports.push(NetlistPort {
        direction,
        width,
        name: symbol,
    });
    Ok(Some(index))
}

fn bit_ref(net: NetIndex, bit: usize, width: usize) -> NetRef {
    if width == 1 {
        NetRef::Simple(net)
    } else {
        NetRef::BitSelect(net, bit as u32)
    }
}

fn add_alias(parsed: &mut ParsedNetlist, lhs: NetRef, rhs: NetRef) {
    parsed.modules[0].assigns.push(NetlistAssign {
        kind: NetlistAssignKind::Continuous,
        lhs,
        rhs: AssignExpr::Leaf(rhs),
        span: Span {
            start: Pos {
                lineno: 0,
                colno: 0,
            },
            limit: Pos {
                lineno: 0,
                colno: 0,
            },
        },
    });
}

#[cfg(test)]
pub(in crate::netlist) mod tests;
