// SPDX-License-Identifier: Apache-2.0

//! Connectivity helpers for gate-level netlists.
//!
//! This module derives per-bit instance and assign connectivity from the
//! normalized netlist frontend. Parsed Verilog syntax such as concat, selects,
//! continuous assigns, and plain `tran` aliases is lowered before cone-like
//! consumers walk the graph.

use crate::liberty::IndexedLibrary;
use crate::liberty_proto::PinDirection;
use crate::netlist::normalized::{BitIndex, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{InstIndex, Net, NetIndex, NetlistModule, PortDirection, PortId};
use anyhow::Result;
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Instances and ports that connect to one normalized net bit.
pub struct BitNeighbors {
    /// Instances and ports that drive this bit.
    pub drivers: Vec<(InstIndex, PortId)>,
    /// Instances and ports that load this bit.
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
    /// All normalized net bits reachable from this port's connection
    /// expression. Literal, unknown, and unconnected sources are omitted.
    pub bits: Vec<BitIndex>,
}

/// Block-level attachment for a normalized net bit that touches one or more
/// module ports.
pub struct BlockPortBoundary {
    /// True if this bit is attached to at least one module input port.
    pub has_input: bool,
    /// True if this bit is attached to at least one module output port.
    pub has_output: bool,
    /// True if this bit is attached to at least one module inout port.
    pub has_inout: bool,
}

/// Per-module connectivity derived from a normalized `NetlistModule` and
/// `IndexedLibrary`.
pub struct NetlistConnectivity<'a> {
    pub module: &'a NetlistModule,
    pub nets: &'a [Net],
    pub lib: &'a IndexedLibrary,
    normalized: NormalizedNetlistModule<'a>,
    /// For each normalized bit index, the instances and ports that drive/load
    /// it.
    pub bit_neighbors: Vec<BitNeighbors>,
    /// For each normalized bit index, assign-source bits that feed it.
    pub assign_fanin_bits: Vec<Vec<BitIndex>>,
    /// For each normalized bit index, assign-destination bits fed by it.
    pub assign_fanout_bits: Vec<Vec<BitIndex>>,
    /// Sparse map from normalized bits that touch module ports to their
    /// block-port roles.
    pub block_port_bits: HashMap<BitIndex, BlockPortBoundary>,
    /// For each instance index in `module.instances`, the list of its ports,
    /// pin directions, and connected normalized bits. Ports for each instance
    /// are stored in a deterministic order by port name.
    instance_ports: Vec<Vec<InstancePortInfo>>,
}

impl<'a> NetlistConnectivity<'a> {
    /// Builds connectivity information for `module` using the given Liberty
    /// library. This constructor classifies instance pin bindings and lowers
    /// normalized assign dependencies onto the bit graph.
    pub fn new(
        module: &'a NetlistModule,
        nets: &'a [Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        lib: &'a IndexedLibrary,
        module_port_dirs: Option<&HashMap<PortId, HashMap<PortId, PinDirection>>>,
    ) -> Result<Self> {
        let normalized = NormalizedNetlistModule::new(module, nets, interner)?;
        let block_port_bits = build_block_port_bits(&normalized);
        let mut bit_neighbors = create_empty_bit_neighbors(normalized.bit_count());
        let (assign_fanin_bits, assign_fanout_bits) = build_assign_connectivity(&normalized);
        let instance_ports = build_instance_connectivity(
            &normalized,
            interner,
            lib,
            module_port_dirs,
            bit_neighbors.as_mut_slice(),
        );

        Ok(Self {
            module,
            nets,
            lib,
            normalized,
            bit_neighbors,
            assign_fanin_bits,
            assign_fanout_bits,
            block_port_bits,
            instance_ports,
        })
    }

    /// Returns the instances and ports that drive normalized `bit`.
    pub fn drivers_for_bit(&self, bit: BitIndex) -> &[(InstIndex, PortId)] {
        &self.bit_neighbors[bit].drivers
    }

    /// Returns the instances and ports that load normalized `bit`.
    pub fn loads_for_bit(&self, bit: BitIndex) -> &[(InstIndex, PortId)] {
        &self.bit_neighbors[bit].loads
    }

    /// Returns assign-source bits that feed normalized `bit`.
    pub fn assign_fanin_bits(&self, bit: BitIndex) -> &[BitIndex] {
        self.assign_fanin_bits[bit].as_slice()
    }

    /// Returns assign-destination bits fed by normalized `bit`.
    pub fn assign_fanout_bits(&self, bit: BitIndex) -> &[BitIndex] {
        self.assign_fanout_bits[bit].as_slice()
    }

    /// Returns true if normalized `bit` is attached to at least one module
    /// port (input, output, or inout).
    pub fn is_block_port_bit(&self, bit: BitIndex) -> bool {
        self.block_port_bits.contains_key(&bit)
    }

    /// Returns the sparse block-port boundary metadata for normalized `bit`,
    /// if any.
    pub fn bit_block_port_boundary(&self, bit: BitIndex) -> Option<&BlockPortBoundary> {
        self.block_port_bits.get(&bit)
    }

    /// Returns the per-port view for `inst_idx`, including pin direction and
    /// connected normalized bits.
    pub fn instance_ports(
        &self,
        inst_idx: InstIndex,
        _interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> &[InstancePortInfo] {
        &self.instance_ports[inst_idx.0]
    }

    /// Renders one normalized bit through the underlying parsed netlist names.
    pub fn render_bit(
        &self,
        bit: BitIndex,
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> String {
        self.normalized.render_bit(bit, self.nets, interner)
    }

    /// Returns the parsed net containing normalized `bit`.
    pub fn net_for_bit(&self, bit: BitIndex) -> NetIndex {
        self.normalized.bit(bit).net
    }
}

fn build_block_port_bits(
    normalized: &NormalizedNetlistModule<'_>,
) -> HashMap<BitIndex, BlockPortBoundary> {
    let mut block_port_bits = HashMap::new();
    for port in &normalized.ports {
        for bit in &port.bits {
            let entry = block_port_bits
                .entry(*bit)
                .or_insert_with(|| BlockPortBoundary {
                    has_input: false,
                    has_output: false,
                    has_inout: false,
                });
            match port.direction {
                PortDirection::Input => entry.has_input = true,
                PortDirection::Output => entry.has_output = true,
                PortDirection::Inout => entry.has_inout = true,
            }
        }
    }
    block_port_bits
}

fn create_empty_bit_neighbors(bits_len: usize) -> Vec<BitNeighbors> {
    let mut bit_neighbors = Vec::with_capacity(bits_len);
    bit_neighbors.resize_with(bits_len, || BitNeighbors {
        drivers: Vec::new(),
        loads: Vec::new(),
    });
    bit_neighbors
}

fn build_assign_connectivity(
    normalized: &NormalizedNetlistModule<'_>,
) -> (Vec<Vec<BitIndex>>, Vec<Vec<BitIndex>>) {
    let mut assign_fanin_bits = vec![Vec::new(); normalized.bit_count()];
    let mut assign_fanout_bits = vec![Vec::new(); normalized.bit_count()];
    for assign in &normalized.assigns {
        for (lhs_bit, rhs_expr) in assign.lhs_bits.iter().copied().zip(&assign.rhs_bits) {
            let mut rhs_bits = Vec::new();
            rhs_expr.collect_source_bits(&mut rhs_bits);
            rhs_bits.sort_unstable();
            rhs_bits.dedup();
            for rhs_bit in rhs_bits {
                extend_unique_bit(&mut assign_fanin_bits[lhs_bit], rhs_bit);
                extend_unique_bit(&mut assign_fanout_bits[rhs_bit], lhs_bit);
            }
        }
    }
    (assign_fanin_bits, assign_fanout_bits)
}

fn build_instance_connectivity(
    normalized: &NormalizedNetlistModule<'_>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    lib: &IndexedLibrary,
    module_port_dirs: Option<&HashMap<PortId, HashMap<PortId, PinDirection>>>,
    bit_neighbors: &mut [BitNeighbors],
) -> Vec<Vec<InstancePortInfo>> {
    let mut instance_ports: Vec<Vec<InstancePortInfo>> =
        Vec::with_capacity(normalized.instances.len());
    instance_ports.resize_with(normalized.instances.len(), Vec::new);

    for inst in &normalized.instances {
        let inst_idx = inst.raw_index;
        let type_sym = inst.type_name;
        let type_name = resolve_to_string(interner, type_sym);

        let Some(dir_by_pin) =
            build_dir_by_pin_for_instance(lib, module_port_dirs, interner, type_sym, type_name)
        else {
            // Unknown instance type: skip connectivity as before.
            continue;
        };

        let mut ports_for_instance: HashMap<PortId, InstancePortInfo> = HashMap::new();
        for connection in &inst.connections {
            let port_name = resolve_to_string(interner, connection.port);
            let dir = *dir_by_pin
                .get(port_name.as_str())
                .unwrap_or(&PinDirection::Invalid);
            let entry = ports_for_instance
                .entry(connection.port)
                .or_insert_with(|| InstancePortInfo {
                    port: connection.port,
                    dir,
                    bits: Vec::new(),
                });
            let mut bit_indices = connection
                .bits
                .iter()
                .filter_map(|source| match source {
                    BitSource::Bit(bit_idx) => Some(*bit_idx),
                    BitSource::Literal(_) | BitSource::Unknown => None,
                })
                .collect::<Vec<_>>();
            bit_indices.sort_unstable();
            bit_indices.dedup();
            for bit_idx in bit_indices {
                let neighbors = &mut bit_neighbors[bit_idx];
                match dir {
                    PinDirection::Output => {
                        extend_unique_endpoint(&mut neighbors.drivers, (inst_idx, connection.port))
                    }
                    PinDirection::Input => {
                        extend_unique_endpoint(&mut neighbors.loads, (inst_idx, connection.port))
                    }
                    PinDirection::Invalid => {
                        extend_unique_endpoint(&mut neighbors.drivers, (inst_idx, connection.port));
                        extend_unique_endpoint(&mut neighbors.loads, (inst_idx, connection.port));
                    }
                }
                entry.bits.push(bit_idx);
            }
            entry.bits.sort_unstable();
            entry.bits.dedup();
        }

        let mut ports_vec: Vec<InstancePortInfo> = ports_for_instance.into_values().collect();
        ports_vec.sort_by_key(|port| resolve_to_string(interner, port.port));
        instance_ports[inst_idx.0] = ports_vec;
    }

    instance_ports
}

fn extend_unique_endpoint(endpoints: &mut Vec<(InstIndex, PortId)>, endpoint: (InstIndex, PortId)) {
    if !endpoints.contains(&endpoint) {
        endpoints.push(endpoint);
    }
}

fn extend_unique_bit(bits: &mut Vec<BitIndex>, bit: BitIndex) {
    if !bits.contains(&bit) {
        bits.push(bit);
    }
}

fn build_dir_by_pin_for_instance(
    lib: &IndexedLibrary,
    module_port_dirs: Option<&HashMap<PortId, HashMap<PortId, PinDirection>>>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    type_sym: PortId,
    type_name: String,
) -> Option<HashMap<String, PinDirection>> {
    // Build pin-name -> direction map for this instance type.
    //
    // Preference order:
    // - Liberty cell pin directions when a matching cell is present.
    // - Module port directions when the type matches a parsed module name and
    //   `module_port_dirs` is provided.
    // - Otherwise, skip connectivity for this instance (maintaining historical
    //   behavior where unknown cell types are treated as an invariant violation).
    let mut dir_by_pin: HashMap<String, PinDirection> = HashMap::new();

    if let Some(cell) = lib.get_cell(type_name.as_str()) {
        for pin in &cell.pins {
            let dir_val = pin.direction;
            let dir = if dir_val == PinDirection::Input as i32 {
                PinDirection::Input
            } else if dir_val == PinDirection::Output as i32 {
                PinDirection::Output
            } else {
                PinDirection::Invalid
            };
            dir_by_pin.insert(pin.name.clone(), dir);
        }
        Some(dir_by_pin)
    } else if let Some(module_maps) = module_port_dirs {
        if let Some(port_dirs) = module_maps.get(&type_sym) {
            for (port_sym, dir) in port_dirs {
                let port_name = resolve_to_string(interner, *port_sym);
                dir_by_pin.insert(port_name, *dir);
            }
            Some(dir_by_pin)
        } else {
            None
        }
    } else {
        None
    }
}

/// Builds a mapping from module name symbol to its port directions expressed
/// in Liberty `PinDirection` terms.
///
/// This helper allows connectivity and cone-traversal code to treat instances
/// whose type is another module (rather than a Liberty cell) as having
/// well-defined pin directions derived from the module's port list.
pub fn build_module_port_directions(
    modules: &[NetlistModule],
) -> HashMap<PortId, HashMap<PortId, PinDirection>> {
    let mut result: HashMap<PortId, HashMap<PortId, PinDirection>> = HashMap::new();

    for m in modules {
        let mut port_dirs: HashMap<PortId, PinDirection> = HashMap::new();
        for p in &m.ports {
            let dir = match p.direction {
                PortDirection::Input => PinDirection::Input,
                PortDirection::Output => PinDirection::Output,
                // Treat inout ports conservatively as "both directions".
                PortDirection::Inout => PinDirection::Invalid,
            };
            port_dirs.insert(p.name, dir);
        }
        result.insert(m.name, port_dirs);
    }

    result
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
        let invx1 = interner.get_or_intern("INV");
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
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2)],
            assigns: vec![],
            instances,
        };

        let lib_proto = Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "A".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        function: "(!A)".to_string(),
                        name: "Y".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let indexed = IndexedLibrary::new(lib_proto);

        (module, nets, interner, indexed)
    }

    #[test]
    fn connectivity_identifies_simple_drivers_and_loads() {
        let (module, nets, interner, indexed) = make_simple_module_and_lib();
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed, None)
            .expect("connectivity");
        let bit_n0 = conn.normalized.net_bits(NetIndex(0))[0];
        let bit_n1 = conn.normalized.net_bits(NetIndex(1))[0];
        let bit_n2 = conn.normalized.net_bits(NetIndex(2))[0];

        // Bit 0: driven by no instance (primary input), loaded by u1.A.
        let loads_n0 = conn.loads_for_bit(bit_n0);
        assert_eq!(loads_n0.len(), 1);

        // Bit 1: driven by u1.Y, loaded by u2.A.
        let drivers_n1 = conn.drivers_for_bit(bit_n1);
        let loads_n1 = conn.loads_for_bit(bit_n1);
        assert_eq!(drivers_n1.len(), 1);
        assert_eq!(loads_n1.len(), 1);

        // Bit 2: driven by u2.Y, no loads.
        let drivers_n2 = conn.drivers_for_bit(bit_n2);
        let loads_n2 = conn.loads_for_bit(bit_n2);
        assert_eq!(drivers_n2.len(), 1);
        assert!(loads_n2.is_empty());
    }

    #[test]
    fn block_port_bits_match_module_ports() {
        let (module, nets, interner, indexed) = make_simple_module_and_lib();
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed, None)
            .expect("connectivity");
        let bit_n0 = conn.normalized.net_bits(NetIndex(0))[0];
        let bit_n1 = conn.normalized.net_bits(NetIndex(1))[0];
        let bit_n2 = conn.normalized.net_bits(NetIndex(2))[0];

        // Bit 0 corresponds to primary input "a".
        assert!(conn.is_block_port_bit(bit_n0));
        let b0 = conn
            .bit_block_port_boundary(bit_n0)
            .expect("bit 0 should have block-port metadata");
        assert!(b0.has_input);
        assert!(!b0.has_output);
        assert!(!b0.has_inout);

        // Bit 1 is internal-only.
        assert!(!conn.is_block_port_bit(bit_n1));
        assert!(conn.bit_block_port_boundary(bit_n1).is_none());

        // Bit 2 corresponds to primary output "y".
        assert!(conn.is_block_port_bit(bit_n2));
        let b2 = conn
            .bit_block_port_boundary(bit_n2)
            .expect("bit 2 should have block-port metadata");
        assert!(!b2.has_input);
        assert!(b2.has_output);
        assert!(!b2.has_inout);
    }

    #[test]
    fn instance_ports_reports_directions_and_bits() {
        let (module, nets, interner, indexed) = make_simple_module_and_lib();
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed, None)
            .expect("connectivity");
        let bit_n0 = conn.normalized.net_bits(NetIndex(0))[0];
        let bit_n1 = conn.normalized.net_bits(NetIndex(1))[0];
        let bit_n2 = conn.normalized.net_bits(NetIndex(2))[0];

        // Instance 0: INVX1 u1
        let ports_u1 = conn.instance_ports(InstIndex(0), &interner);
        let mut rendered_u1: Vec<(String, PinDirection, Vec<BitIndex>)> = Vec::new();
        for p in ports_u1.iter() {
            let name = interner
                .resolve(p.port)
                .expect("port symbol should resolve")
                .to_string();
            rendered_u1.push((name, p.dir, p.bits.clone()));
        }

        // Expect a deterministic ordering by port name.
        assert_eq!(rendered_u1.len(), 2);
        assert_eq!(rendered_u1[0].0, "A");
        assert_eq!(rendered_u1[0].1, PinDirection::Input);
        assert_eq!(rendered_u1[0].2, vec![bit_n0]);
        assert_eq!(rendered_u1[1].0, "Y");
        assert_eq!(rendered_u1[1].1, PinDirection::Output);
        assert_eq!(rendered_u1[1].2, vec![bit_n1]);

        // Instance 1: INVX1 u2
        let ports_u2 = conn.instance_ports(InstIndex(1), &interner);
        let mut rendered_u2: Vec<(String, PinDirection, Vec<BitIndex>)> = Vec::new();
        for p in ports_u2.iter() {
            let name = interner
                .resolve(p.port)
                .expect("port symbol should resolve")
                .to_string();
            rendered_u2.push((name, p.dir, p.bits.clone()));
        }

        assert_eq!(rendered_u2.len(), 2);
        assert_eq!(rendered_u2[0].0, "A");
        assert_eq!(rendered_u2[0].1, PinDirection::Input);
        assert_eq!(rendered_u2[0].2, vec![bit_n1]);
        assert_eq!(rendered_u2[1].0, "Y");
        assert_eq!(rendered_u2[1].1, PinDirection::Output);
        assert_eq!(rendered_u2[1].2, vec![bit_n2]);
    }

    #[test]
    fn connectivity_tracks_normalized_assign_dependencies() {
        let (mut module, nets, interner, indexed) = make_simple_module_and_lib();
        module.assigns.push(crate::netlist::parse::NetlistAssign {
            kind: crate::netlist::parse::NetlistAssignKind::Continuous,
            lhs: NetRef::Simple(NetIndex(2)),
            rhs: crate::netlist::parse::AssignExpr::Leaf(NetRef::Simple(NetIndex(1))),
            span: crate::netlist::parse::Span {
                start: crate::netlist::parse::Pos {
                    lineno: 1,
                    colno: 1,
                },
                limit: crate::netlist::parse::Pos {
                    lineno: 1,
                    colno: 15,
                },
            },
        });
        let conn = NetlistConnectivity::new(&module, &nets, &interner, &indexed, None)
            .expect("connectivity");
        let bit_n1 = conn.normalized.net_bits(NetIndex(1))[0];
        let bit_n2 = conn.normalized.net_bits(NetIndex(2))[0];
        assert_eq!(conn.assign_fanout_bits(bit_n1), &[bit_n2]);
        assert_eq!(conn.assign_fanin_bits(bit_n2), &[bit_n1]);
    }
}
