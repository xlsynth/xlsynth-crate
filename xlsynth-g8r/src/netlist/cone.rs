// SPDX-License-Identifier: Apache-2.0

//! Cone traversal for gate-level netlists.
//!
//! This module provides a small API for traversing the fanin/fanout cone around
//! a particular instance in a parsed gate-level netlist. Callers supply:
//!
//! - A `NetlistModule` plus the global `nets` array and `interner`.
//! - A Liberty `Library` describing per-cell pin directions.
//! - A start instance name, optional start pin list, traversal direction, and a
//!   stop condition.
//!
//! The traversal walks instance-to-instance and invokes a user callback on each
//! visited `(instance_type, instance_name, traversal_pin)` triple in a
//! deterministic order.

use crate::liberty::descriptor::liberty_descriptor_pool;
use crate::liberty_proto::{Library, Pin, PinDirection};
use crate::netlist::io::{ParsedNetlist, parse_netlist_from_path};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistModule, NetlistPort, PortDirection, PortId};
use anyhow::{anyhow};
use prost::Message;
use prost_reflect::DynamicMessage;
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryFrom;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Direction to traverse the cone from the starting instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    Fanin,
    Fanout,
}

/// Condition that bounds how far the cone traversal proceeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopCondition {
    /// Stop once instances beyond this graph distance (in edges) would be
    /// reached. Level 0 is the starting instance.
    Levels(u32),
    /// Stop at DFF instances as inferred from Liberty; do not traverse beyond
    /// them, but include them in the visit stream.
    AtDff,
    /// Stop at module ports (primary inputs/outputs); do not traverse beyond
    /// the block boundary, but include instances that connect to the port.
    AtBlockPort,
}

/// One visit in the cone traversal.
///
/// - `instance_type` is the Liberty cell name (e.g. `INVX1`).
/// - `instance_name` is the Verilog instance label (e.g. `u123`).
/// - `traversal_pin` is the pin name on this instance through which the cone
///   traversal reaches or departs this instance, depending on direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConeVisit {
    pub instance_type: String,
    pub instance_name: String,
    pub traversal_pin: String,
}

/// Error type for cone traversal.
#[derive(Debug)]
pub enum ConeError {
    MissingInstance { name: String },
    AmbiguousInstance { name: String, count: usize },
    UnknownCellType { cell: String },
    UnknownCellPin { cell: String, pin: String },
    NoModulesParsed { path: String },
    ModuleNotFound { name: String },
    NetlistParse(String),
    Liberty(String),
    Invariant(String),
}

impl std::fmt::Display for ConeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConeError::MissingInstance { name } => {
                write!(f, "start instance '{}' was not found in the module", name)
            }
            ConeError::AmbiguousInstance { name, count } => write!(
                f,
                "start instance '{}' is ambiguous: found {} instances with this name",
                name, count
            ),
            ConeError::UnknownCellType { cell } => {
                write!(
                    f,
                    "cell type '{}' is not present in the Liberty library",
                    cell
                )
            }
            ConeError::UnknownCellPin { cell, pin } => write!(
                f,
                "pin '{}' is not present on cell type '{}' in the Liberty library",
                pin, cell
            ),
            ConeError::NoModulesParsed { path } => write!(
                f,
                "no modules parsed from '{}'; expected at least one module",
                path
            ),
            ConeError::ModuleNotFound { name } => {
                write!(f, "module '{}' was not found in the netlist", name)
            }
            ConeError::NetlistParse(msg) => write!(f, "netlist parse error: {}", msg),
            ConeError::Liberty(msg) => write!(f, "liberty parse error: {}", msg),
            ConeError::Invariant(msg) => write!(f, "cone traversal invariant failed: {}", msg),
        }
    }
}

impl std::error::Error for ConeError {}

type ConeResult<T> = std::result::Result<T, ConeError>;

/// Lightweight per-port summary used during traversal.
struct InstancePort {
    port: PortId,
    dir: PinDirection,
    nets: Vec<NetIndex>,
}

/// Per-module context with adjacency maps and cached metadata.
struct ModuleConeContext<'a> {
    module: &'a NetlistModule,
    nets: &'a [Net],
    interner: &'a StringInterner<StringBackend<SymbolU32>>,
    /// For each instance index in `module.instances`, the list of its ports.
    instance_ports: Vec<Vec<InstancePort>>,
    /// For each `NetIndex`, the instance ports that drive the net.
    net_drivers: Vec<Vec<(usize, PortId)>>,
    /// For each `NetIndex`, the instance ports that load the net.
    net_loads: Vec<Vec<(usize, PortId)>>,
    /// For each `NetIndex`, whether it corresponds to a module port and, if so,
    /// in which direction.
    net_port_direction: Vec<Option<PortDirection>>,
    /// Set of cell type symbols classified as DFFs (used for
    /// StopCondition::AtDff).
    dff_types: HashSet<PortId>,
}

impl<'a> ModuleConeContext<'a> {
    fn new(
        module: &'a NetlistModule,
        nets: &'a [Net],
        interner: &'a StringInterner<StringBackend<SymbolU32>>,
        lib: &'a Library,
        dff_cell_names: &HashSet<String>,
    ) -> ConeResult<Self> {
        // Build a map from Liberty cell name to &Cell.
        let mut cell_by_name: HashMap<&str, &crate::liberty_proto::Cell> = HashMap::new();
        for cell in &lib.cells {
            cell_by_name.insert(cell.name.as_str(), cell);
        }

        // Map from module port net symbol -> PortDirection.
        let mut port_dir_by_sym: HashMap<SymbolU32, PortDirection> = HashMap::new();
        for NetlistPort {
            direction, name, ..
        } in &module.ports
        {
            port_dir_by_sym.insert(*name, direction.clone());
        }

        let mut instance_ports: Vec<Vec<InstancePort>> = Vec::with_capacity(module.instances.len());
        instance_ports.resize_with(module.instances.len(), || Vec::new());
        let mut net_drivers: Vec<Vec<(usize, PortId)>> = vec![Vec::new(); nets.len()];
        let mut net_loads: Vec<Vec<(usize, PortId)>> = vec![Vec::new(); nets.len()];

        // Precompute dff_types as a set of type-name symbols used by instances.
        let mut dff_types: HashSet<PortId> = HashSet::new();

        for (inst_idx, inst) in module.instances.iter().enumerate() {
            let type_sym = inst.type_name;
            let type_name = resolve_to_string(interner, type_sym);
            let cell = cell_by_name
                .get(type_name.as_str())
                .copied()
                .ok_or(ConeError::UnknownCellType { cell: type_name })?;

            if dff_cell_names.contains(&cell.name) {
                dff_types.insert(type_sym);
            }

            // Build per-port summary and net driver/load maps.
            // First, build a lookup of Liberty pins by name for this cell.
            let mut pins_by_name: HashMap<&str, &Pin> = HashMap::new();
            for pin in &cell.pins {
                pins_by_name.insert(pin.name.as_str(), pin);
            }

            let mut ports_for_instance: HashMap<PortId, InstancePort> = HashMap::new();

            for (port_sym, netref) in &inst.connections {
                let port_name = resolve_to_string(interner, *port_sym);
                let pin = pins_by_name.get(port_name.as_str()).copied().ok_or(
                    ConeError::UnknownCellPin {
                        cell: cell.name.clone(),
                        pin: port_name,
                    },
                )?;

                let dir = PinDirection::try_from(pin.direction).unwrap_or(PinDirection::Invalid);
                let entry = ports_for_instance
                    .entry(*port_sym)
                    .or_insert_with(|| InstancePort {
                        port: *port_sym,
                        dir,
                        nets: Vec::new(),
                    });

                let mut net_indices: Vec<NetIndex> = Vec::new();
                collect_net_indices(netref, &mut net_indices);
                for idx in net_indices {
                    if idx.0 >= nets.len() {
                        return Err(ConeError::Invariant(format!(
                            "NetIndex({}) out of bounds for nets (len={})",
                            idx.0,
                            nets.len()
                        )));
                    }
                    entry.nets.push(idx);
                    match dir {
                        PinDirection::Output => {
                            net_drivers[idx.0].push((inst_idx, *port_sym));
                        }
                        PinDirection::Input => {
                            net_loads[idx.0].push((inst_idx, *port_sym));
                        }
                        PinDirection::Invalid => {
                            // Treat invalid as both to avoid silently dropping connectivity.
                            net_drivers[idx.0].push((inst_idx, *port_sym));
                            net_loads[idx.0].push((inst_idx, *port_sym));
                        }
                    }
                }
            }

            instance_ports[inst_idx] = ports_for_instance.into_values().collect();
        }

        // Map nets to module port directions (if any).
        let mut net_port_direction: Vec<Option<PortDirection>> = vec![None; nets.len()];
        for (idx, net) in nets.iter().enumerate() {
            if let Some(dir) = port_dir_by_sym.get(&net.name) {
                net_port_direction[idx] = Some(dir.clone());
            }
        }

        Ok(ModuleConeContext {
            module,
            nets,
            interner,
            instance_ports,
            net_drivers,
            net_loads,
            net_port_direction,
            dff_types,
        })
    }
}

fn resolve_to_string(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: SymbolU32,
) -> String {
    interner
        .resolve(sym)
        .expect("symbol should always resolve in interner")
        .to_string()
}

fn collect_net_indices(netref: &NetRef, out: &mut Vec<NetIndex>) {
    match netref {
        NetRef::Simple(idx) | NetRef::BitSelect(idx, _) | NetRef::PartSelect(idx, _, _) => {
            out.push(*idx);
        }
        NetRef::Concat(elems) => {
            for e in elems {
                collect_net_indices(e, out);
            }
        }
        NetRef::Literal(_) | NetRef::Unconnected => {}
    }
}

/// Load a Liberty proto (binary or textproto) into a `Library`.
pub fn load_liberty_for_cone(path: &Path) -> ConeResult<Library> {
    let mut buf: Vec<u8> = Vec::new();
    File::open(path)
        .map_err(|e| {
            ConeError::Liberty(format!("opening liberty proto '{}': {}", path.display(), e))
        })?
        .read_to_end(&mut buf)
        .map_err(|e| {
            ConeError::Liberty(format!("reading liberty proto '{}': {}", path.display(), e))
        })?;

    let lib = Library::decode(&buf[..]).or_else(|_| {
        let descriptor_pool = liberty_descriptor_pool();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .ok_or_else(|| anyhow!("missing liberty.Library descriptor"))?;
        let dyn_msg = DynamicMessage::parse_text_format(msg_desc, std::str::from_utf8(&buf)?)?;
        let encoded = dyn_msg.encode_to_vec();
        Ok::<Library, anyhow::Error>(Library::decode(&encoded[..])?)
    });

    match lib {
        Ok(v) => Ok(v),
        Err(e) => Err(ConeError::Liberty(format!(
            "decoding liberty proto '{}': {}",
            path.display(),
            e
        ))),
    }
}

/// Traverse the cone around `start_instance` in `module`, calling `on_visit`
/// for each visited `(instance_type, instance_name, traversal_pin)` triple.
///
/// This is the core traversal routine; callers that need file I/O should use
/// [`visit_cone_from_paths`] instead.
pub fn visit_module_cone<F>(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    lib: &Library,
    dff_cell_names: &HashSet<String>,
    start_instance: &str,
    start_pins: Option<&[String]>,
    direction: TraversalDirection,
    stop: StopCondition,
    mut on_visit: F,
) -> ConeResult<()>
where
    F: FnMut(&ConeVisit) -> ConeResult<()>,
{
    let ctx = ModuleConeContext::new(module, nets, interner, lib, dff_cell_names)?;

    // Resolve the starting instance index.
    let mut matches: Vec<usize> = Vec::new();
    for (idx, inst) in module.instances.iter().enumerate() {
        let name = resolve_to_string(interner, inst.instance_name);
        if name == start_instance {
            matches.push(idx);
        }
    }
    if matches.is_empty() {
        return Err(ConeError::MissingInstance {
            name: start_instance.to_string(),
        });
    }
    if matches.len() > 1 {
        return Err(ConeError::AmbiguousInstance {
            name: start_instance.to_string(),
            count: matches.len(),
        });
    }
    let start_idx = matches[0];

    // Build a map from PortId to InstancePort for the starting instance to
    // resolve and validate start pins.
    let mut ports_by_sym: HashMap<PortId, &InstancePort> = HashMap::new();
    let mut ports_by_name: HashMap<String, PortId> = HashMap::new();
    for p in &ctx.instance_ports[start_idx] {
        ports_by_sym.insert(p.port, p);
        let name = resolve_to_string(interner, p.port);
        ports_by_name.insert(name, p.port);
    }

    let mut chosen_ports: Vec<PortId> = Vec::new();
    match start_pins {
        Some(list) => {
            for pin_name in list {
                let port_sym = ports_by_name
                    .get(pin_name)
                    .ok_or(ConeError::Invariant(format!(
                        "start pin '{}' not found on instance '{}'",
                        pin_name, start_instance
                    )))?;
                let p = ports_by_sym
                    .get(port_sym)
                    .ok_or(ConeError::Invariant(format!(
                        "internal error: missing port_sym for pin '{}' on instance '{}'",
                        pin_name, start_instance
                    )))?;
                match direction {
                    TraversalDirection::Fanin => {
                        if p.dir != PinDirection::Input {
                            return Err(ConeError::Invariant(format!(
                                "start pin '{}' on instance '{}' is not an INPUT pin for fanin traversal",
                                pin_name, start_instance
                            )));
                        }
                    }
                    TraversalDirection::Fanout => {
                        if p.dir != PinDirection::Output {
                            return Err(ConeError::Invariant(format!(
                                "start pin '{}' on instance '{}' is not an OUTPUT pin for fanout traversal",
                                pin_name, start_instance
                            )));
                        }
                    }
                }
                chosen_ports.push(*port_sym);
            }
        }
        None => {
            for p in &ctx.instance_ports[start_idx] {
                match direction {
                    TraversalDirection::Fanin => {
                        if p.dir == PinDirection::Input {
                            chosen_ports.push(p.port);
                        }
                    }
                    TraversalDirection::Fanout => {
                        if p.dir == PinDirection::Output {
                            chosen_ports.push(p.port);
                        }
                    }
                }
            }
        }
    }

    // Emit visits for the starting instance pins.
    let start_inst = &module.instances[start_idx];
    let inst_type_str = resolve_to_string(interner, start_inst.type_name);
    let inst_name_str = resolve_to_string(interner, start_inst.instance_name);

    // Ensure we only emit each (instance, port) pair once.
    let mut emitted_ports: HashSet<(usize, PortId)> = HashSet::new();
    for port_sym in &chosen_ports {
        let port_name = resolve_to_string(interner, *port_sym);
        if emitted_ports.insert((start_idx, *port_sym)) {
            let visit = ConeVisit {
                instance_type: inst_type_str.clone(),
                instance_name: inst_name_str.clone(),
                traversal_pin: port_name,
            };
            on_visit(&visit)?;
        }
    }

    // BFS over instances, bounded by StopCondition.
    #[derive(Clone, Copy)]
    struct QueueEntry {
        inst_idx: usize,
        level: u32,
    }

    let mut visited_instances: HashSet<usize> = HashSet::new();
    visited_instances.insert(start_idx);

    let mut queue: VecDeque<QueueEntry> = VecDeque::new();
    queue.push_back(QueueEntry {
        inst_idx: start_idx,
        level: 0,
    });

    while let Some(QueueEntry { inst_idx, level }) = queue.pop_front() {
        let ports = &ctx.instance_ports[inst_idx];

        for port in ports {
            // Only traverse through pins consistent with the global direction.
            let traverse_through = match direction {
                TraversalDirection::Fanin => port.dir == PinDirection::Input,
                TraversalDirection::Fanout => port.dir == PinDirection::Output,
            };
            if !traverse_through {
                continue;
            }

            for net_idx in &port.nets {
                if net_idx.0 >= ctx.nets.len() {
                    return Err(ConeError::Invariant(format!(
                        "NetIndex({}) out of bounds for nets (len={}) during traversal",
                        net_idx.0,
                        ctx.nets.len()
                    )));
                }

                // If we are stopping at block ports, do not traverse across
                // module boundary nets.
                if matches!(stop, StopCondition::AtBlockPort) {
                    if ctx.net_port_direction[net_idx.0].is_some() {
                        continue;
                    }
                }

                let neighbor_level = level + 1;
                if let StopCondition::Levels(max) = stop {
                    if neighbor_level > max {
                        continue;
                    }
                }

                let neighbors = match direction {
                    TraversalDirection::Fanin => &ctx.net_drivers[net_idx.0],
                    TraversalDirection::Fanout => &ctx.net_loads[net_idx.0],
                };

                for (nbr_inst_idx, nbr_port_sym) in neighbors {
                    // Skip self-loops; the starting instance has already been emitted.
                    if *nbr_inst_idx == inst_idx {
                        continue;
                    }

                    let nbr_inst = &ctx.module.instances[*nbr_inst_idx];
                    let nbr_type_str = resolve_to_string(&ctx.interner, nbr_inst.type_name);
                    let nbr_name_str = resolve_to_string(&ctx.interner, nbr_inst.instance_name);
                    let nbr_port_name = resolve_to_string(&ctx.interner, *nbr_port_sym);

                    if emitted_ports.insert((*nbr_inst_idx, *nbr_port_sym)) {
                        let visit = ConeVisit {
                            instance_type: nbr_type_str,
                            instance_name: nbr_name_str,
                            traversal_pin: nbr_port_name,
                        };
                        on_visit(&visit)?;
                    }

                    // Decide whether to enqueue this neighbor for further
                    // expansion.
                    let is_dff = ctx.dff_types.contains(&nbr_inst.type_name);
                    match stop {
                        StopCondition::Levels(max) => {
                            if neighbor_level < max && visited_instances.insert(*nbr_inst_idx) {
                                queue.push_back(QueueEntry {
                                    inst_idx: *nbr_inst_idx,
                                    level: neighbor_level,
                                });
                            }
                        }
                        StopCondition::AtDff => {
                            if is_dff {
                                // Include the DFF in the output but do not
                                // traverse beyond it.
                                continue;
                            }
                            if visited_instances.insert(*nbr_inst_idx) {
                                queue.push_back(QueueEntry {
                                    inst_idx: *nbr_inst_idx,
                                    level: neighbor_level,
                                });
                            }
                        }
                        StopCondition::AtBlockPort => {
                            if visited_instances.insert(*nbr_inst_idx) {
                                queue.push_back(QueueEntry {
                                    inst_idx: *nbr_inst_idx,
                                    level: neighbor_level,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// High-level convenience entry point: parse the netlist and Liberty proto from
/// disk, select a module, and run cone traversal.
pub fn visit_cone_from_paths<F>(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    module_name: Option<&str>,
    start_instance: &str,
    start_pins: Option<&[String]>,
    direction: TraversalDirection,
    stop: StopCondition,
    dff_cell_names: &HashSet<String>,
    on_visit: F,
) -> ConeResult<()>
where
    F: FnMut(&ConeVisit) -> ConeResult<()>,
{
    let parsed: ParsedNetlist = parse_netlist_from_path(netlist_path)
        .map_err(|e| ConeError::NetlistParse(format!("{}", e)))?;
    let lib = load_liberty_for_cone(liberty_proto_path)?;

    // Select the target module.
    let module = match module_name {
        Some(name) => {
            let mut found: Option<&NetlistModule> = None;
            for m in &parsed.modules {
                let m_name = resolve_to_string(&parsed.interner, m.name);
                if m_name == name {
                    found = Some(m);
                    break;
                }
            }
            found.ok_or_else(|| ConeError::ModuleNotFound {
                name: name.to_string(),
            })?
        }
        None => {
            if parsed.modules.len() != 1 {
                return Err(ConeError::Invariant(format!(
                    "netlist contains {} modules; specify --module_name to disambiguate",
                    parsed.modules.len()
                )));
            }
            &parsed.modules[0]
        }
    };

    visit_module_cone(
        module,
        &parsed.nets,
        &parsed.interner,
        &lib,
        dff_cell_names,
        start_instance,
        start_pins,
        direction,
        stop,
        on_visit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_proto::{Cell, Library};
    use crate::netlist::parse::{
        Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
    };
    use pretty_assertions::assert_eq;
    use std::collections::HashSet;

    fn make_simple_inverter_lib() -> Library {
        Library {
            cells: vec![Cell {
                name: "INVX1".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "A".to_string(),
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        function: "(!A)".to_string(),
                        name: "Y".to_string(),
                    },
                ],
                area: 1.0,
            }],
        }
    }

    #[test]
    fn simple_fanout_levels() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n1 = interner.get_or_intern("n1");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INVX1");
        let u1 = interner.get_or_intern("u1");
        let u2 = interner.get_or_intern("u2");
        let top = interner.get_or_intern("top");

        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: n1,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];

        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];

        let instances = vec![
            NetlistInstance {
                type_name: invx1,
                instance_name: u1,
                connections: vec![
                    (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                    (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            NetlistInstance {
                type_name: invx1,
                instance_name: u2,
                connections: vec![
                    (interner.get_or_intern("A"), NetRef::Simple(NetIndex(1))),
                    (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(2))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports,
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2)],
            instances,
        };

        let lib = make_simple_inverter_lib();

        let mut visits: Vec<ConeVisit> = Vec::new();
        let dff_cells: HashSet<String> = HashSet::new();
        visit_module_cone(
            &module,
            &nets,
            &interner,
            &lib,
            &dff_cells,
            "u1",
            None,
            TraversalDirection::Fanout,
            StopCondition::Levels(1),
            |v| {
                visits.push(v.clone());
                Ok(())
            },
        )
        .unwrap();

        let rendered: String = {
            let mut rows: Vec<String> = Vec::new();
            for v in &visits {
                rows.push(format!(
                    "{},{},{}",
                    v.instance_type, v.instance_name, v.traversal_pin
                ));
            }
            rows.join("\n")
        };

        let want = "INVX1,u1,Y\nINVX1,u2,A";
        assert_eq!(rendered, want);
    }

    #[test]
    fn fanout_to_two_pins_on_same_gate() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n1 = interner.get_or_intern("n1");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INVX1");
        let and2x1 = interner.get_or_intern("AND2X1");
        let u1 = interner.get_or_intern("u1");
        let u_and = interner.get_or_intern("u_and");
        let top = interner.get_or_intern("top");

        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: n1,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];

        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];

        // Topology:
        //   a -> INVX1 u1 -> n1
        //   n1 drives both A and B pins of AND2X1 u_and -> y
        let instances = vec![
            NetlistInstance {
                type_name: invx1,
                instance_name: u1,
                connections: vec![
                    (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                    (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            NetlistInstance {
                type_name: and2x1,
                instance_name: u_and,
                connections: vec![
                    (interner.get_or_intern("A"), NetRef::Simple(NetIndex(1))),
                    (interner.get_or_intern("B"), NetRef::Simple(NetIndex(1))),
                    (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(2))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports,
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2)],
            instances,
        };

        // Liberty library with INVX1 and AND2X1 cells.
        let lib = Library {
            cells: vec![
                Cell {
                    name: "INVX1".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "A".to_string(),
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "(!A)".to_string(),
                            name: "Y".to_string(),
                        },
                    ],
                    area: 1.0,
                },
                Cell {
                    name: "AND2X1".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "A".to_string(),
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "B".to_string(),
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "(A*B)".to_string(),
                            name: "Y".to_string(),
                        },
                    ],
                    area: 2.0,
                },
            ],
        };

        let mut visits: Vec<ConeVisit> = Vec::new();
        let dff_cells: HashSet<String> = HashSet::new();
        visit_module_cone(
            &module,
            &nets,
            &interner,
            &lib,
            &dff_cells,
            "u1",
            None,
            TraversalDirection::Fanout,
            StopCondition::Levels(1),
            |v| {
                visits.push(v.clone());
                Ok(())
            },
        )
        .unwrap();

        let rendered: String = {
            let mut rows: Vec<String> = Vec::new();
            for v in &visits {
                rows.push(format!(
                    "{},{},{}",
                    v.instance_type, v.instance_name, v.traversal_pin
                ));
            }
            rows.join("\n")
        };

        let want = "INVX1,u1,Y\nAND2X1,u_and,A\nAND2X1,u_and,B";
        assert_eq!(rendered, want);
    }
}
