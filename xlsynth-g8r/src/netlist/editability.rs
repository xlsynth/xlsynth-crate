// SPDX-License-Identifier: Apache-2.0

//! Determine which registered data connections a netlist transformation may
//! edit without disturbing control cones, fixed connections or unpriced
//! boundaries. This analysis is valid only for its unchanged netlist snapshot.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, Pin, PinDirection};
use crate::netlist::connectivity::{PinConnection, exact::ExactConnectivity};
use crate::netlist::electrical::{
    InputLoad, Load, OutputCharacterization, PinLimits, SignalElectrical,
};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::normalized::{BitSource, NormalizedBitIndex, NormalizedNetlistModule};
use crate::netlist::parse::{InstIndex, NetRef, PortId};
use crate::netlist::register_mapping::AssembledRegisters;
use crate::netlist::sequential_liberty::classify_tied_ff_data;
use anyhow::{Result, anyhow, ensure};
use std::collections::{HashMap, HashSet};

/// A scalar cell input pin whose connection an optimizer may change.
#[derive(Clone, Debug)]
pub struct EditablePin {
    pub binding: PinConnection,
    pub original: NetRef,
    pub capacitance: [f64; 2],
    pub fanout_load: f64,
}

/// A selected physical driver with its editable and fixed load portions.
#[derive(Debug)]
pub struct EditableOutput {
    pub binding: PinConnection,
    pub cell: usize,
    pub source: NetRef,
    pub source_bit: crate::netlist::bit_ref::NetBit,
    pub instance_name: String,
    pub pin_name: String,
    pub sinks: Vec<EditablePin>,
    /// Rise/fall capacitance that remains physically connected after editing.
    pub fixed_load: [f64; 2],
    pub fixed_sink_count: usize,
    pub fixed_fanout_load: f64,
    /// Replacing or deleting this source can affect clock or control logic.
    /// Redirecting a different data sink need not modify that logic.
    pub in_control_fanin: bool,
    pub electrical: SignalElectrical,
}

/// Discovery keeps all FF launches, even those whose Q is not editable: every
/// candidate must be timed against the same complete register population.
pub struct EditableOutputs {
    pub roots: Vec<EditableOutput>,
    pub launches: Vec<usize>,
}

/// Finds register outputs and identifies which connected cell input pins an
/// optimizer may disconnect from their current net and connect to another net.
/// For example, an optimizer might insert a buffer and change an input pin's
/// connection from the register output to the buffer output.
///
/// For each register output, determines:
/// - Which cell input pins may have their connections changed.
/// - Which connections must remain unchanged and their combined load.
/// - Whether the output also feeds clock or control logic.
///
/// Handles scalar nets, individual bits of vector nets,
/// continuous assignments, and bidirectional `tran` connections.
///
/// Changing a pin's connection requires known cell, connection, and required
/// timing information. Callers must also respect the reported control-cone
/// information when choosing edits.
///
/// The result is valid for the analyzed netlist snapshot. Recompute it
/// after changing connectivity. Any supplied register assembly metadata
/// must describe the same netlist.
pub fn analyze_editable_outputs(
    netlist: &MappedNetlist,
    lib: &Library,
    module_output_load: f64,
    assembled: Option<AssembledRegisters<'_>>,
) -> Result<EditableOutputs> {
    ensure!(
        module_output_load.is_finite() && module_output_load >= 0.,
        "invalid external load"
    );
    let n = NormalizedNetlistModule::new(&netlist.module, &netlist.nets, &netlist.interner)?;
    let count = n.bit_count();
    let wiring = ExactConnectivity::new(&n);
    let group = |bit: NormalizedBitIndex| wiring.group(bit);
    let mut written = wiring.assigned.clone();
    let mut blocked = vec![false; count];
    let mut control: Vec<_> = wiring
        .issues
        .iter()
        .map(|issue| {
            log::debug!(
                "uneditable electrical group {}: {:?}",
                issue.bit.index(),
                issue.kind
            );
            issue.bit
        })
        .collect();
    for &g in control.iter().chain(&wiring.input_groups) {
        blocked[g.index()] = true;
    }
    let outputs = &wiring.output_ports;
    let mut constants = HashMap::new();
    for &(bit, value) in &wiring.literals {
        if constants.insert(bit, value).is_some() {
            blocked[bit.index()] = true;
        }
    }
    if let Some(bindings) = assembled {
        ensure!(
            bindings.instances().end <= n.instances.len(),
            "stale assembled register indices"
        );
        for i in bindings.instances() {
            let mapping = bindings
                .initial_mapping(i)
                .ok_or_else(|| anyhow!("missing assembled register mapping"))?;
            let inst = &n.instances[i];
            ensure!(
                netlist.interner.resolve(inst.type_name) == Some(mapping.cell_name.as_str()),
                "assembled register master changed"
            );
            for (position, pin) in [&mapping.clock_pin, &mapping.data_pin, &mapping.output_pin]
                .into_iter()
                .enumerate()
            {
                ensure!(
                    inst.connections
                        .get(position)
                        .is_some_and(
                            |conn| netlist.interner.resolve(conn.port) == Some(pin.as_str())
                        ),
                    "assembled register binding changed"
                );
            }
            for (pin, value) in &mapping.tied_inputs {
                ensure!(
                    inst.connections
                        .iter()
                        .any(|c| netlist.interner.resolve(c.port) == Some(pin.as_str())
                            && c.bits.as_slice() == [BitSource::Literal(*value)]),
                    "assembled register tie changed"
                );
            }
        }
    }
    let catalog = pin_roles(netlist, lib, assembled)?;
    let mut drivers = HashMap::new();
    let mut combinational_driver = HashMap::new();
    let mut launches: Vec<usize> = assembled
        .map(|a| a.instances().collect())
        .unwrap_or_default();
    for (i, inst) in n.instances.iter().enumerate() {
        let shape = &catalog[&inst.type_name];
        if lib.cells[shape.cell].has_ff_group() {
            if let Some(bindings) = assembled {
                // Assembly records the complete original launch population.
                // Appending a register requires fresh discovery; omitting it
                // could make an apparently faster trial lose an untimed path.
                ensure!(
                    bindings.instances().contains(&i),
                    "register outside assembled boundary"
                );
            } else {
                launches.push(i);
            }
        }
        let duplicate_pin = wiring.has_duplicate_pin(inst.raw_index);
        for (j, conn) in inst.connections.iter().enumerate() {
            let role = shape.pins.get(&conn.port);
            let [BitSource::Bit(bit)] = conn.bits.as_slice() else {
                // Liberty pins are scalar. A wider reference cannot be edited
                // or assigned capacitance to just one of its constituent bits.
                if conn.bits.len() > 1 {
                    for source in &conn.bits {
                        if let BitSource::Bit(b) = source {
                            blocked[group(*b).index()] = true;
                            control.push(group(*b));
                        }
                    }
                }
                continue;
            };
            let g = group(*bit);
            if duplicate_pin {
                blocked[g.index()] = true;
                control.push(g);
            }
            match role {
                Some(Role::Input { control: true, .. }) => control.push(g),
                Some(Role::Input { .. }) => {}
                Some(Role::Output {
                    state,
                    constant,
                    limits,
                    characterization,
                }) => {
                    if std::mem::replace(&mut written[bit.index()], true) {
                        blocked[g.index()] = true;
                    }
                    if let Some(level) = constant {
                        if constants.insert(g, *level).is_some() {
                            blocked[g.index()] = true;
                        }
                    }
                    if shape.combinational {
                        ensure!(
                            combinational_driver.insert(g, i).is_none(),
                            "multiple combinational drivers for one electrical net"
                        );
                    }
                    if *state {
                        let root = EditableOutput {
                            binding: PinConnection {
                                instance: InstIndex(i),
                                connection: j,
                            },
                            cell: shape.cell,
                            source: netlist.module.instances[i].connections[j].1.clone(),
                            source_bit: n.bit(*bit),
                            instance_name: netlist
                                .interner
                                .resolve(inst.instance_name)
                                .ok_or_else(|| anyhow!("unresolved instance name"))?
                                .to_owned(),
                            pin_name: netlist
                                .interner
                                .resolve(conn.port)
                                .ok_or_else(|| anyhow!("unresolved output pin"))?
                                .to_owned(),
                            sinks: vec![],
                            fixed_load: [0.; 2],
                            fixed_sink_count: 0,
                            fixed_fanout_load: 0.,
                            in_control_fanin: false,
                            electrical: SignalElectrical {
                                load: Load::with_outputs(
                                    outputs.get(&g).copied().unwrap_or(0),
                                    module_output_load,
                                ),
                                transition: None,
                                limits: *limits,
                                characterization: *characterization,
                            },
                        };
                        if drivers.insert(g, root).is_some() {
                            blocked[g.index()] = true;
                        }
                    }
                }
                None | Some(Role::Unknown) => {
                    blocked[g.index()] = true;
                    control.push(g);
                }
            }
        }
    }
    constants.retain(|bit, _| !blocked[bit.index()]);
    // A conditional timing arc may be active at one gate and inactive at
    // another instance of the same master. Memoize by the actually bound
    // input bits, including tie-cell outputs and simple assigns.
    let mut specializations = HashMap::new();
    let mut fixed_capture_pins = HashSet::new();
    for inst in &n.instances {
        let shape = &catalog[&inst.type_name];
        if shape.combinational {
            continue;
        }
        let coverage = shape.conditional.as_ref().map(|next| {
            let mut low = 0u64;
            let mut high = 0u64;
            for conn in &inst.connections {
                let Some(&position) = next.positions.get(&conn.port) else {
                    continue;
                };
                let value = match conn.bits.as_slice() {
                    [BitSource::Literal(bit)] => Some(*bit),
                    [BitSource::Bit(bit)] => constants.get(&group(*bit)).copied(),
                    _ => None,
                };
                match value {
                    Some(false) => low |= 1u64 << position,
                    Some(true) => high |= 1u64 << position,
                    None => {}
                }
            }
            let cache = specializations
                .entry((shape.cell, low, high))
                .or_insert_with(|| {
                    let mut tied = HashMap::new();
                    for (i, name) in next.inputs.iter().enumerate() {
                        let mask = 1u64 << i;
                        if (high | low) & mask != 0 {
                            tied.insert(name.clone(), high & mask != 0);
                        }
                    }
                    let cell = &lib.cells[shape.cell];
                    // Prove the role from the live ties even for assembled
                    // registers: rewiring a control can change what D captures.
                    let data = classify_tied_ff_data(&next.expr, &tied, &next.inputs)
                        .and_then(|name| netlist.interner.get(name));
                    let mut priced = HashSet::new();
                    for pin in &cell.pins {
                        let Some(port) = netlist.interner.get(lib.resolve_string(&pin.name)) else {
                            continue;
                        };
                        if (Some(port) == data
                            || matches!(
                                shape.pins.get(&port),
                                Some(Role::Output { state: true, .. })
                            ))
                            && crate::netlist::sta::has_complete_register_pin_timing(
                                lib, cell, pin, &tied,
                            )
                        {
                            priced.insert(port);
                        }
                    }
                    (data, priced)
                });
            &*cache
        });
        for (connection, conn) in inst.connections.iter().enumerate() {
            let [BitSource::Bit(bit)] = conn.bits.as_slice() else {
                continue;
            };
            match shape.pins.get(&conn.port) {
                Some(Role::Output { state: true, .. })
                    if coverage.is_none_or(|(_, priced)| !priced.contains(&conn.port)) =>
                {
                    blocked[group(*bit).index()] = true
                }
                Some(Role::Input {
                    can_change_connection,
                    ..
                }) if coverage.is_none_or(|(data, priced)| {
                    *data != Some(conn.port) || !priced.contains(&conn.port)
                }) =>
                {
                    // Without both active setup edges this physical D can be
                    // absent from otherwise successful whole-module STA.
                    control.push(group(*bit));
                    if *can_change_connection {
                        fixed_capture_pins.insert(PinConnection {
                            instance: inst.raw_index,
                            connection,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    // A generated clock/reset may reach its control pin through ordinary
    // inverters or buffers. Preserve the combinational fan-in to that pin.
    let mut visited = HashSet::new();
    while let Some(g) = control.pop() {
        if !visited.insert(g) {
            continue;
        }
        if let Some(&i) = combinational_driver.get(&g) {
            let inst = &n.instances[i];
            let shape = &catalog[&inst.type_name];
            for conn in &inst.connections {
                if matches!(shape.pins.get(&conn.port), Some(Role::Input { .. })) {
                    for bit in &conn.bits {
                        if let BitSource::Bit(b) = bit {
                            control.push(group(*b));
                        }
                    }
                }
            }
        }
    }
    drivers.retain(|g, _| !blocked[g.index()]);
    for (group, driver) in &mut drivers {
        driver.in_control_fanin = visited.contains(group);
        driver.fixed_load = driver.electrical.load.capacitance;
    }
    let selected = drivers.keys().copied().collect();
    let pins = wiring.index_pins(&selected, |id: PinConnection| {
        let inst = &n.instances[id.instance.0];
        let conn = &inst.connections[id.connection];
        match catalog[&inst.type_name].pins.get(&conn.port) {
            Some(Role::Input { .. }) => Some(PinDirection::Input),
            Some(Role::Output { .. }) => Some(PinDirection::Output),
            _ => None,
        }
    });
    let mut roots = Vec::with_capacity(drivers.len());
    for (g, mut driver) in drivers {
        let Some(signal) = pins.signal(g) else {
            continue;
        };
        if signal.drivers.as_slice() != [driver.binding] {
            continue;
        }
        let measured = SignalElectrical::analyze(
            signal,
            driver.electrical.load.output_ports,
            module_output_load,
            driver.electrical.limits,
            driver.electrical.characterization,
            None,
            |id| {
                let inst = &n.instances[id.instance.0];
                let conn = &inst.connections[id.connection];
                let Some(Role::Input {
                    can_change_connection,
                    cap,
                    fanout_load,
                    ..
                }) = catalog[&inst.type_name].pins.get(&conn.port)
                else {
                    return None;
                };
                if conn.bits.len() != 1 {
                    return None;
                }
                let load = InputLoad {
                    capacitance: *cap,
                    fanout: *fanout_load,
                };
                if *can_change_connection && !fixed_capture_pins.contains(&id) {
                    driver.sinks.push(EditablePin {
                        binding: id,
                        original: netlist.module.instances[id.instance.0].connections
                            [id.connection]
                            .1
                            .clone(),
                        capacitance: *cap,
                        fanout_load: *fanout_load,
                    });
                } else {
                    driver.fixed_load[0] += cap[0];
                    driver.fixed_load[1] += cap[1];
                    driver.fixed_sink_count += 1;
                    driver.fixed_fanout_load += fanout_load;
                }
                Some(load)
            },
        )
        .ok_or_else(|| anyhow!("incomplete electrical characterization for editable output"))?;
        driver.electrical = measured;
        roots.push(driver);
    }
    roots.sort_by_key(|root| (root.binding.instance.0, root.binding.connection));
    Ok(EditableOutputs { roots, launches })
}

/// Whether a characterized pin carries editable data, a control, or an output.
enum Role {
    Input {
        can_change_connection: bool,
        control: bool,
        cap: [f64; 2],
        fanout_load: f64,
    },
    Output {
        state: bool,
        constant: Option<bool>,
        limits: PinLimits,
        characterization: OutputCharacterization,
    },
    Unknown,
}

/// Decoded functional and pin roles for one used Liberty master.
struct Shape {
    cell: usize,
    combinational: bool,
    pins: HashMap<PortId, Role>,
    conditional: Option<ConditionalData>,
}

/// Next-state expression and physical inputs for per-instance tied-control
/// checks.
struct ConditionalData {
    expr: Term,
    inputs: Vec<String>,
    positions: HashMap<PortId, usize>,
}

fn indexed_cells(lib: &Library) -> Result<HashMap<&str, usize>> {
    let mut cells = HashMap::with_capacity(lib.cells.len());
    for (i, cell) in lib.cells.iter().enumerate() {
        ensure!(
            cells.insert(cell.name.as_str(), i).is_none(),
            "duplicate Liberty master {}",
            cell.name
        );
    }
    Ok(cells)
}

/// Decode each used master's pin functions and characterization once per graph.
fn pin_roles(
    netlist: &MappedNetlist,
    lib: &Library,
    assembled: Option<AssembledRegisters<'_>>,
) -> Result<HashMap<PortId, Shape>> {
    let cells = indexed_cells(lib)?;
    let mut shapes = HashMap::new();
    for inst in &netlist.module.instances {
        if shapes.contains_key(&inst.type_name) {
            continue;
        }
        let name = netlist
            .interner
            .resolve(inst.type_name)
            .ok_or_else(|| anyhow!("unresolved cell type"))?;
        let i = *cells
            .get(name)
            .ok_or_else(|| anyhow!("missing Liberty cell {name}"))?;
        let cell = &lib.cells[i];
        let valid = cell.validate_functional_semantics().is_ok();
        let clock = crate::netlist::sequential_liberty::get_sequential_ff_clock_spec(cell, lib)
            .ok()
            .flatten();
        let next = clock
            .as_ref()
            .and_then(|_| parse_formula(&cell.sequential[0].next_state).ok());
        let mut controls = HashSet::new();
        if let Some(clock) = &clock {
            controls.insert(clock.pin_name.clone());
        }
        if let Some(gate) = &cell.clock_gate {
            controls.insert(gate.clock_pin.clone());
            controls.extend(gate.enable_pins.iter().chain(&gate.test_pins).cloned());
        }
        for seq in &cell.sequential {
            for text in [&seq.clock_expr, &seq.clear_expr, &seq.preset_expr] {
                if let Ok(expr) = parse_formula(text) {
                    controls.extend(expr.inputs());
                }
            }
        }
        let combinational = cell.sequential.is_empty();
        let data_inputs: HashSet<_> = if valid
            && cell.clock_gate.is_none()
            && combinational
            && !cell.pins.iter().any(|p| p.is_clocking_pin)
        {
            cell.pins
                .iter()
                .filter(|p| p.direction == PinDirection::Output as i32)
                .filter_map(|p| parse_formula(lib.resolve_string(&p.function)).ok())
                .flat_map(|term| term.inputs())
                .collect()
        } else {
            HashSet::new()
        };
        let known_cell =
            assembled.is_some_and(|a| a.plan.cells.iter().any(|m| m.cell_name == cell.name));
        let mut pins = HashMap::new();
        for pin in &cell.pins {
            let name = lib.resolve_string(&pin.name);
            let Some(sym) = netlist.interner.get(name) else {
                continue;
            };
            let role = if pin.direction == PinDirection::Output as i32 {
                let state = cell.clock_gate.is_none()
                    && next.is_some()
                    && if known_cell {
                        assembled.is_some_and(|a| {
                            a.plan
                                .cells
                                .iter()
                                .any(|m| m.cell_name == cell.name && m.output_pin == name)
                        })
                    } else {
                        state_output(lib, cell, pin)
                    };
                Role::Output {
                    state,
                    constant: if combinational && valid && cell.clock_gate.is_none() {
                        parse_formula(lib.resolve_string(&pin.function))
                            .ok()
                            .and_then(|term| term.evaluate_partial(&HashMap::new()))
                    } else {
                        None
                    },
                    limits: PinLimits::for_pin(pin),
                    characterization: if state {
                        OutputCharacterization::for_pin(lib, pin)
                    } else {
                        OutputCharacterization::default()
                    },
                }
            } else if pin.direction == PinDirection::Input as i32 {
                let control = pin.is_clocking_pin || controls.contains(name);
                if let Some(load) = InputLoad::for_pin(pin) {
                    let cap = load.capacitance;
                    let fanout_load = load.fanout;
                    // FF inputs are candidates here. The instance-specific
                    // check fixes every pin without proven data and setup arcs.
                    let can_change_connection = valid
                        && cell.clock_gate.is_none()
                        && !control
                        && (data_inputs.contains(name) || next.is_some());
                    Role::Input {
                        can_change_connection,
                        control,
                        cap,
                        fanout_load,
                    }
                } else {
                    Role::Unknown
                }
            } else {
                Role::Unknown
            };
            ensure!(
                pins.insert(sym, role).is_none(),
                "duplicate Liberty pin {name} on {}",
                cell.name
            );
        }
        let conditional = next.and_then(|expr| {
            let inputs: Vec<_> = cell
                .pins
                .iter()
                .filter(|p| p.direction == PinDirection::Input as i32)
                .map(|p| lib.resolve_string(&p.name).to_owned())
                .collect();
            if inputs.len() > u64::BITS as usize {
                return None;
            }
            let positions = inputs
                .iter()
                .enumerate()
                .filter_map(|(i, name)| netlist.interner.get(name.as_str()).map(|s| (s, i)))
                .collect();
            Some(ConditionalData {
                expr,
                inputs,
                positions,
            })
        });
        shapes.insert(
            inst.type_name,
            Shape {
                cell: i,
                combinational,
                pins,
                conditional,
            },
        );
    }
    Ok(shapes)
}

/// The output has to expose Q or its complement. The FF may implement a scan
/// mux in next_state; that changes its captured value, not its state outputs.
fn state_output(lib: &Library, cell: &Cell, pin: &Pin) -> bool {
    let seq = &cell.sequential[0];
    if seq.state_var.is_empty() {
        return false;
    }
    let Ok(term) = parse_formula(lib.resolve_string(&pin.function)) else {
        return false;
    };
    let signal = match &term {
        Term::Input(name) => name,
        Term::Negate(inner) => match inner.as_ref() {
            Term::Input(name) => name,
            _ => return false,
        },
        _ => return false,
    };
    signal == &seq.state_var || seq.complementary_state_var.as_deref() == Some(signal.as_str())
}

#[cfg(test)]
mod tests;
