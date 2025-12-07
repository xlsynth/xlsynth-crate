// SPDX-License-Identifier: Apache-2.0

//! Connectivity helpers for gate-level netlists.
//!
//! This module provides a small API for reasoning about which instances
//! drive or load each net in a `NetlistModule`, combining information from
//! the parsed netlist with Liberty pin directions via `IndexedLibrary`.

use crate::liberty::IndexedLibrary;
use crate::liberty_proto::PinDirection;
use crate::netlist::parse::{InstIndex, Net, NetIndex, NetlistModule, PortId};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use std::collections::HashMap;

/// Instances and ports that connect to a net.
pub struct NetNeighbors {
    /// Instances and ports that drive this net.
    pub drivers: Vec<(InstIndex, PortId)>,
    /// Instances and ports that load this net.
    pub loads: Vec<(InstIndex, PortId)>,
}

/// Per-module connectivity derived from a `NetlistModule` and `IndexedLibrary`.
pub struct NetlistConnectivity<'a> {
    pub module: &'a NetlistModule,
    pub nets: &'a [Net],
    pub lib: &'a IndexedLibrary,
    /// For each `NetIndex.0`, the instances and ports that drive/load it.
    pub net_neighbors: Vec<NetNeighbors>,
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
        let mut net_neighbors: Vec<NetNeighbors> = Vec::with_capacity(nets.len());
        net_neighbors.resize_with(nets.len(), || NetNeighbors {
            drivers: Vec::new(),
            loads: Vec::new(),
        });

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

            for (port_sym, netref) in &inst.connections {
                let port_name = resolve_to_string(interner, *port_sym);
                let dir = *dir_by_pin
                    .get(port_name.as_str())
                    .unwrap_or(&PinDirection::Invalid);

                let mut net_indices: Vec<NetIndex> = Vec::new();
                netref.collect_net_indices(&mut net_indices);

                for idx in net_indices {
                    if idx.0 >= nets.len() {
                        continue;
                    }
                    let entry = &mut net_neighbors[idx.0];
                    match dir {
                        PinDirection::Output => entry.drivers.push((inst_idx, *port_sym)),
                        PinDirection::Input => entry.loads.push((inst_idx, *port_sym)),
                        PinDirection::Invalid => {
                            entry.drivers.push((inst_idx, *port_sym));
                            entry.loads.push((inst_idx, *port_sym));
                        }
                    }
                }
            }
        }

        NetlistConnectivity {
            module,
            nets,
            lib,
            net_neighbors,
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

    fn make_simple_module_and_lib()
        -> (NetlistModule, Vec<Net>, StringInterner<StringBackend<SymbolU32>>, IndexedLibrary)
    {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n1 = interner.get_or_intern("n1");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INVX1");
        let u1 = interner.get_or_intern("u1");
        let u2 = interner.get_or_intern("u2");

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
}




