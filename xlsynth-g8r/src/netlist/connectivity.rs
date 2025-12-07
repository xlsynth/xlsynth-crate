// SPDX-License-Identifier: Apache-2.0

//! Connectivity helpers for gate-level netlists.
//!
//! This module provides a small API for reasoning about which instances
//! drive or load each net in a `NetlistModule`, combining information from
//! the parsed netlist with Liberty pin directions via `IndexedLibrary`.

use crate::liberty::IndexedLibrary;
use crate::liberty_proto::PinDirection;
use crate::netlist::parse::{InstIndex, Net, NetIndex, NetlistModule, PortDirection, PortId};
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Instances and ports that connect to a net.
pub struct NetNeighbors {
    /// Instances and ports that drive this net.
    pub drivers: Vec<(InstIndex, PortId)>,
    /// Instances and ports that load this net.
    pub loads: Vec<(InstIndex, PortId)>,
}

/// Lightweight per-port summary used when reasoning about instance-level
/// connectivity.
pub struct InstancePortInfo {
    /// Logical port identifier on the instance (symbol for the Liberty pin
    /// name, e.g. `A` or `Y`).
    pub port: PortId,
    /// Direction of this pin on the cell (input/output), as derived from the
    /// Liberty library.
    pub dir: PinDirection,
    /// All `NetIndex` values reachable from this port's connection expression.
    ///
    /// This may contain multiple nets when the Verilog connection is a concat
    /// (e.g. `{a, b[5], c[7:0], 1'b0}`) or when a bus slice spans several
    /// underlying nets.
    pub nets: Vec<NetIndex>,
}

/// Block-level attachment for a net that touches one or more module ports.
///
/// This structure is stored sparsely: only nets that are attached to at least
/// one module port get an entry in the `NetlistConnectivity` map.
pub struct BlockPortBoundary {
    /// True if this net is attached to at least one module input port.
    pub has_input: bool,
    /// True if this net is attached to at least one module output port.
    pub has_output: bool,
    /// True if this net is attached to at least one module inout port.
    pub has_inout: bool,
}

/// Per-module connectivity derived from a `NetlistModule` and `IndexedLibrary`.
pub struct NetlistConnectivity<'a> {
    pub module: &'a NetlistModule,
    pub nets: &'a [Net],
    pub lib: &'a IndexedLibrary,
    /// For each `NetIndex.0`, the instances and ports that drive/load it.
    pub net_neighbors: Vec<NetNeighbors>,
    /// Sparse map from nets that touch module ports to their block-port roles
    /// (whether they are attached to input / output / inout ports).
    pub block_port_nets: HashMap<NetIndex, BlockPortBoundary>,
    /// For each instance index in `module.instances`, the list of its ports,
    /// pin directions, and connected nets. Ports for each instance are stored
    /// in a deterministic order by port name.
    instance_ports: Vec<Vec<InstancePortInfo>>,
}

impl<'a> NetlistConnectivity<'a> {
    /// Builds connectivity information for `module` using the given Liberty
    /// library. This constructor iterates all instances once and classifies
    /// their connections as drivers or loads for each net.
    pub fn new(
        module: &'a NetlistModule,
        nets: &'a [Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        lib: &'a IndexedLibrary,
    ) -> Self {
        // Map from module port net symbol -> PortDirection for sparse
        // block-port attachment metadata.
        let mut port_dir_by_sym: HashMap<SymbolU32, PortDirection> = HashMap::new();
        for port in &module.ports {
            port_dir_by_sym.insert(port.name, port.direction.clone());
        }

        // Sparse per-net module-port attachment (only nets that touch block
        // ports get entries).
        let mut block_port_nets: HashMap<NetIndex, BlockPortBoundary> = HashMap::new();
        for (idx, net) in nets.iter().enumerate() {
            if let Some(dir) = port_dir_by_sym.get(&net.name) {
                let entry =
                    block_port_nets
                        .entry(NetIndex(idx))
                        .or_insert_with(|| BlockPortBoundary {
                            has_input: false,
                            has_output: false,
                            has_inout: false,
                        });
                match dir {
                    PortDirection::Input => entry.has_input = true,
                    PortDirection::Output => entry.has_output = true,
                    PortDirection::Inout => entry.has_inout = true,
                }
            }
        }

        let mut net_neighbors: Vec<NetNeighbors> = Vec::with_capacity(nets.len());
        net_neighbors.resize_with(nets.len(), || NetNeighbors {
            drivers: Vec::new(),
            loads: Vec::new(),
        });

        let mut instance_ports: Vec<Vec<InstancePortInfo>> =
            Vec::with_capacity(module.instances.len());
        instance_ports.resize_with(module.instances.len(), Vec::new);

        for (inst_idx_raw, inst) in module.instances.iter().enumerate() {
            let inst_idx = InstIndex(inst_idx_raw);

            // Resolve cell and pin directions once per instance.
            let type_name = resolve_to_string(interner, inst.type_name);
            let cell = match lib.get_cell(type_name.as_str()) {
                Some(c) => c,
                None => {
                    // Missing cell types are considered an invariant violation;
                    // skip connectivity for this instance.
                    continue;
                }
            };

            // Build pin-name -> direction map for this cell.
            let mut dir_by_pin: HashMap<&str, PinDirection> = HashMap::new();
            for pin in &cell.pins {
                let dir_val = pin.direction;
                if dir_val == PinDirection::Input as i32 {
                    dir_by_pin.insert(pin.name.as_str(), PinDirection::Input);
                } else if dir_val == PinDirection::Output as i32 {
                    dir_by_pin.insert(pin.name.as_str(), PinDirection::Output);
                } else {
                    dir_by_pin.insert(pin.name.as_str(), PinDirection::Invalid);
                }
            }

            let mut ports_for_instance: HashMap<PortId, InstancePortInfo> = HashMap::new();

            for (port_sym, netref) in &inst.connections {
                let port_name = resolve_to_string(interner, *port_sym);
                let dir = *dir_by_pin
                    .get(port_name.as_str())
                    .unwrap_or(&PinDirection::Invalid);

                let entry =
                    ports_for_instance
                        .entry(*port_sym)
                        .or_insert_with(|| InstancePortInfo {
                            port: *port_sym,
                            dir,
                            nets: Vec::new(),
                        });

                let mut net_indices: Vec<NetIndex> = Vec::new();
                netref.collect_net_indices(&mut net_indices);

                for idx in net_indices {
                    if idx.0 >= nets.len() {
                        continue;
                    }
                    let nn_entry = &mut net_neighbors[idx.0];
                    match dir {
                        PinDirection::Output => nn_entry.drivers.push((inst_idx, *port_sym)),
                        PinDirection::Input => nn_entry.loads.push((inst_idx, *port_sym)),
                        PinDirection::Invalid => {
                            nn_entry.drivers.push((inst_idx, *port_sym));
                            nn_entry.loads.push((inst_idx, *port_sym));
                        }
                    }
                    entry.nets.push(idx);
                }
            }

            let mut ports_vec: Vec<InstancePortInfo> = ports_for_instance.into_values().collect();
            ports_vec.sort_by_key(|p| resolve_to_string(interner, p.port));
            instance_ports[inst_idx_raw] = ports_vec;
        }

        NetlistConnectivity {
            module,
            nets,
            lib,
            net_neighbors,
            block_port_nets,
            instance_ports,
        }
    }

    /// Returns the instances and ports that drive `net`.
    pub fn drivers_for_net(&self, net: NetIndex) -> &[(InstIndex, PortId)] {
        &self.net_neighbors[net.0].drivers
    }

    /// Returns the instances and ports that load `net`.
    pub fn loads_for_net(&self, net: NetIndex) -> &[(InstIndex, PortId)] {
        &self.net_neighbors[net.0].loads
    }

    /// Returns true if `net` is attached to at least one module port (input,
    /// output, or inout).
    pub fn is_block_port_net(&self, net: NetIndex) -> bool {
        self.block_port_nets.contains_key(&net)
    }

    /// Returns the sparse block-port boundary metadata for `net`, if any.
    pub fn net_block_port_boundary(&self, net: NetIndex) -> Option<&BlockPortBoundary> {
        self.block_port_nets.get(&net)
    }

    /// Returns the per-port view for `inst_idx`, including pin direction and
    /// connected nets.
    pub fn instance_ports(
        &self,
        inst_idx: InstIndex,
        _interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> &[InstancePortInfo] {
        &self.instance_ports[inst_idx.0]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_proto::{Cell, Library, Pin};
    use crate::netlist::parse::{
        Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
    };

    fn make_simple_module_and_lib() -> (
        NetlistModule,
        Vec<Net>,
        StringInterner<StringBackend<SymbolU32>>,
        IndexedLibrary,
    ) {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n1 = interner.get_or_intern("n1");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INVX1");
        let u1 = interner.get_or_intern("u1");
        let u2 = interner.get_or_intern("u2");

        let nets = vec![
            Net { name: a, width: None },
            Net { name: n1, width: None },
            Net { name: y, width: None },
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
            name: interner.get_or_intern("top"),
            ports,
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2)],
            instances,
        };

        let lib_proto = Library {
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
        };
        let indexed = IndexedLibrary::new(lib_proto);

        (module, nets, interner, indexed)
    }

    #[test]
    fn connectivity_identifies_simple_drivers_and_loads() {
        let (module, nets, interner, indexed) = make_simple_module_and_lib();
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed);

        // Net 0: driven by no instance (primary input), loaded by u1.A.
        let loads_n0 = conn.loads_for_net(NetIndex(0));
        assert_eq!(loads_n0.len(), 1);

        // Net 1: driven by u1.Y, loaded by u2.A.
        let drivers_n1 = conn.drivers_for_net(NetIndex(1));
        let loads_n1 = conn.loads_for_net(NetIndex(1));
        assert_eq!(drivers_n1.len(), 1);
        assert_eq!(loads_n1.len(), 1);

        // Net 2: driven by u2.Y, no loads.
        let drivers_n2 = conn.drivers_for_net(NetIndex(2));
        let loads_n2 = conn.loads_for_net(NetIndex(2));
        assert_eq!(drivers_n2.len(), 1);
        assert!(loads_n2.is_empty());
    }

    #[test]
    fn block_port_nets_match_module_ports() {
        let (module, nets, interner, indexed) = make_simple_module_and_lib();
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed);

        // Net 0 corresponds to primary input "a".
        assert!(conn.is_block_port_net(NetIndex(0)));
        let b0 = conn
            .net_block_port_boundary(NetIndex(0))
            .expect("net 0 should have block-port metadata");
        assert!(b0.has_input);
        assert!(!b0.has_output);
        assert!(!b0.has_inout);

        // Net 1 is internal-only.
        assert!(!conn.is_block_port_net(NetIndex(1)));
        assert!(conn.net_block_port_boundary(NetIndex(1)).is_none());

        // Net 2 corresponds to primary output "y".
        assert!(conn.is_block_port_net(NetIndex(2)));
        let b2 = conn
            .net_block_port_boundary(NetIndex(2))
            .expect("net 2 should have block-port metadata");
        assert!(!b2.has_input);
        assert!(b2.has_output);
        assert!(!b2.has_inout);
    }

    #[test]
    fn instance_ports_reports_directions_and_nets() {
        let (module, nets, interner, indexed) = make_simple_module_and_lib();
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed);

        // Instance 0: INVX1 u1
        let ports_u1 = conn.instance_ports(InstIndex(0), &interner);
        let mut rendered_u1: Vec<(String, PinDirection, Vec<NetIndex>)> = Vec::new();
        for p in ports_u1.iter() {
            let name = interner
                .resolve(p.port)
                .expect("port symbol should resolve")
                .to_string();
            rendered_u1.push((name, p.dir, p.nets.clone()));
        }

        // Expect a deterministic ordering by port name.
        assert_eq!(rendered_u1.len(), 2);
        assert_eq!(rendered_u1[0].0, "A");
        assert_eq!(rendered_u1[0].1, PinDirection::Input);
        assert_eq!(rendered_u1[0].2, vec![NetIndex(0)]);
        assert_eq!(rendered_u1[1].0, "Y");
        assert_eq!(rendered_u1[1].1, PinDirection::Output);
        assert_eq!(rendered_u1[1].2, vec![NetIndex(1)]);

        // Instance 1: INVX1 u2
        let ports_u2 = conn.instance_ports(InstIndex(1), &interner);
        let mut rendered_u2: Vec<(String, PinDirection, Vec<NetIndex>)> = Vec::new();
        for p in ports_u2.iter() {
            let name = interner
                .resolve(p.port)
                .expect("port symbol should resolve")
                .to_string();
            rendered_u2.push((name, p.dir, p.nets.clone()));
        }

        assert_eq!(rendered_u2.len(), 2);
        assert_eq!(rendered_u2[0].0, "A");
        assert_eq!(rendered_u2[0].1, PinDirection::Input);
        assert_eq!(rendered_u2[0].2, vec![NetIndex(1)]);
        assert_eq!(rendered_u2[1].0, "Y");
        assert_eq!(rendered_u2[1].1, PinDirection::Output);
        assert_eq!(rendered_u2[1].2, vec![NetIndex(2)]);
    }
}
