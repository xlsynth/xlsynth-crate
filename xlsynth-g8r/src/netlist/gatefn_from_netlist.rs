// SPDX-License-Identifier: Apache-2.0

//! Project a parsed netlist and Liberty proto into a GateFn.

use crate::gate::AigBitVector;
use crate::gate::AigOperand;
use crate::gate::GateFn;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty_proto::Library;
use crate::netlist::parse::{Net, NetIndex, NetlistModule};
use std::collections::HashMap;
use std::collections::HashSet;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

fn build_cell_formula_map(
    liberty_lib: &Library,
) -> HashMap<(String, String), (crate::liberty::cell_formula::Term, String)> {
    let mut cell_formula_map = HashMap::new();
    for cell in &liberty_lib.cells {
        for pin in &cell.pins {
            if pin.direction == 1 && !pin.function.is_empty() {
                let original_formula_string = pin.function.clone();
                match crate::liberty::cell_formula::parse_formula(&pin.function) {
                    Ok(term) => {
                        cell_formula_map.insert(
                            (cell.name.clone(), pin.name.clone()),
                            (term, original_formula_string),
                        );
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to parse formula for cell '{}', pin '{}' (formula: \"{}\"): {}",
                            cell.name,
                            pin.name,
                            original_formula_string,
                            e
                        );
                    }
                }
            }
        }
    }
    cell_formula_map
}

fn check_undriven_nets(
    used_as_input: &HashSet<SymbolU32>,
    driven: &HashSet<SymbolU32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
) {
    for net in used_as_input {
        if !driven.contains(net) {
            let net_name = interner.resolve(*net).unwrap_or("<unknown>");
            panic!(
                "Net '{}' is used as an input but is never driven by any instance or module input!",
                net_name
            );
        }
    }
}

fn process_instance_outputs(
    inst: &crate::netlist::parse::NetlistInstance,
    type_name: &str,
    inst_name: &str,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    gb: &mut GateBuilder,
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
    dff_cells: &std::collections::HashSet<String>,
    cell_formula_map: &HashMap<(String, String), (crate::liberty::cell_formula::Term, String)>,
    input_map: &HashMap<String, AigOperand>,
    port_map: &HashMap<String, String>,
) -> bool {
    let mut processed_any_output = false;
    for (port, netref) in &inst.connections {
        let port_name = interner.resolve(*port).unwrap();
        let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
        if pin_dir == 1 {
            if handle_dff_identity_override(
                type_name, inst_name, port_name, inst, interner, gb, net_to_bv, dff_cells,
            ) {
                processed_any_output = true;
                continue;
            }
            let key = (type_name.to_string(), port_name.to_string());
            let (formula_ast, original_formula_str) = match cell_formula_map.get(&key) {
                Some(x) => x,
                None => continue,
            };
            let context = crate::liberty::cell_formula::EmitContext {
                cell_name: type_name,
                original_formula: original_formula_str.as_str(),
                instance_name: Some(inst_name),
                port_map: Some(port_map),
            };
            let out_op = formula_ast
                .emit_formula_term(gb, input_map, &context)
                .unwrap();
            match netref {
                crate::netlist::parse::NetRef::Simple(net_idx) => {
                    if net_to_bv.contains_key(net_idx) {
                        panic!("Output net '{}' already assigned", net_idx.0);
                    }
                    net_to_bv.insert(*net_idx, AigBitVector::from_bit(out_op));
                }
                crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => {
                    let width = nets[net_idx.0]
                        .width
                        .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                        .unwrap_or(1);
                    let mut bv = net_to_bv.remove(net_idx).unwrap_or_else(|| {
                        AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width])
                    });
                    if (*bit as usize) >= width {
                        panic!(
                            "Bit-select out of range for output net '{}' (width {}) in instance '{}' port '{}', net_idx={:?}, bit={}",
                            interner.resolve(nets[net_idx.0].name).unwrap(),
                            width,
                            inst_name,
                            port_name,
                            net_idx,
                            bit
                        );
                    }
                    bv.set_lsb(*bit as usize, out_op);
                    net_to_bv.insert(*net_idx, bv);
                }
                _ => {
                    // Only simple/bitselect supported for output
                }
            }
            processed_any_output = true;
        }
    }
    processed_any_output
}

pub fn project_gatefn_from_netlist_and_liberty(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    dff_cells: &std::collections::HashSet<String>,
) -> Result<GateFn, String> {
    let cell_formula_map = build_cell_formula_map(liberty_lib);
    let module_name = interner.resolve(module.name).unwrap();
    let mut gb = GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt());
    let mut net_to_bv: HashMap<NetIndex, AigBitVector> = HashMap::new();
    collect_module_io_nets(module, nets, interner, &mut gb, &mut net_to_bv);
    let mut used_as_input = HashSet::new();
    let mut driven = HashSet::new();
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Input {
            driven.insert(port.name);
        }
    }
    for inst in &module.instances {
        let type_name = interner.resolve(inst.type_name).unwrap();
        let cell = liberty_lib
            .cells
            .iter()
            .find(|c| c.name == type_name)
            .unwrap();
        let mut pin_directions = std::collections::HashMap::new();
        for pin in &cell.pins {
            pin_directions.insert(pin.name.as_str(), pin.direction);
        }
        for (port, netref) in &inst.connections {
            let port_name = interner.resolve(*port).unwrap();
            let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
            match netref {
                crate::netlist::parse::NetRef::Simple(net_idx)
                | crate::netlist::parse::NetRef::BitSelect(net_idx, _) => {
                    if pin_dir == 1 {
                        driven.insert(nets[net_idx.0].name);
                    } else if pin_dir == 2 {
                        used_as_input.insert(nets[net_idx.0].name);
                    }
                }
                _ => {}
            }
        }
    }
    check_undriven_nets(&used_as_input, &driven, interner);
    let mut unprocessed: Vec<_> = module.instances.iter().collect();
    let mut processed_any = true;
    while !unprocessed.is_empty() && processed_any {
        processed_any = false;
        let mut i = 0;
        while i < unprocessed.len() {
            let inst = unprocessed[i];
            let type_name = interner.resolve(inst.type_name).unwrap();
            let inst_name = interner.resolve(inst.instance_name).unwrap();
            let cell = liberty_lib
                .cells
                .iter()
                .find(|c| c.name == type_name)
                .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
            let mut pin_directions = HashMap::new();
            for pin in &cell.pins {
                pin_directions.insert(pin.name.as_str(), pin.direction);
            }
            let (input_map, missing_inputs, port_map) =
                build_instance_input_map(inst, &pin_directions, interner, nets, &net_to_bv);
            if !missing_inputs.is_empty() {
                log::trace!(
                    "Skipping instance '{}' (cell '{}') due to missing input nets: {:?}",
                    inst_name,
                    type_name,
                    missing_inputs
                );
                i += 1;
                continue;
            }
            let processed = process_instance_outputs(
                inst,
                type_name,
                inst_name,
                &pin_directions,
                interner,
                nets,
                &mut gb,
                &mut net_to_bv,
                dff_cells,
                &cell_formula_map,
                &input_map,
                &port_map,
            );
            if processed {
                unprocessed.remove(i);
                processed_any = true;
            } else {
                i += 1;
            }
        }
        if !processed_any && !unprocessed.is_empty() {
            return Err(format!(
                "Could not resolve all instance dependencies (possible cycle or missing driver). Remaining instances: {}",
                unprocessed.len()
            ));
        }
    }
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Output {
            let net_idx = nets
                .iter()
                .position(|n| n.name == port.name)
                .map(NetIndex)
                .ok_or("output port net not found")?;
            let net = &nets[net_idx.0];
            let net_name = interner.resolve(net.name).unwrap();
            let width = net
                .width
                .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                .unwrap_or(1);
            let bv = net_to_bv
                .get(&net_idx)
                .ok_or_else(|| format!("No value for output net '{}'", net_name))?;
            assert_eq!(
                bv.get_bit_count(),
                width,
                "Output net '{}' width mismatch",
                net_name
            );
            gb.add_output(net_name.to_string(), bv.clone());
        }
    }
    Ok(gb.build())
}

fn collect_module_io_nets(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
) {
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Input {
            let net_idx = nets
                .iter()
                .position(|n| n.name == port.name)
                .map(NetIndex)
                .expect("input port net not found");
            let net = &nets[net_idx.0];
            let net_name = interner.resolve(net.name).unwrap();
            let width = net
                .width
                .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                .unwrap_or(1);
            let bv = gb.add_input(net_name.to_string(), width);
            net_to_bv.insert(net_idx, bv.clone());
        }
    }
}

fn build_instance_input_map(
    inst: &crate::netlist::parse::NetlistInstance,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    net_to_bv: &HashMap<NetIndex, AigBitVector>,
) -> (
    HashMap<String, AigOperand>,
    Vec<String>,
    HashMap<String, String>,
) {
    let mut input_map = HashMap::new();
    let mut missing_inputs = Vec::new();
    let mut port_map = HashMap::new();
    for (port, netref) in &inst.connections {
        let port_name = interner.resolve(*port).unwrap();
        let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
        let net_name_str = match netref {
            crate::netlist::parse::NetRef::Simple(net_idx) => interner
                .resolve(nets[net_idx.0].name)
                .unwrap_or("<unknown>")
                .to_string(),
            crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => format!(
                "{}[{}]",
                interner
                    .resolve(nets[net_idx.0].name)
                    .unwrap_or("<unknown>"),
                bit
            ),
            crate::netlist::parse::NetRef::PartSelect(net_idx, msb, lsb) => format!(
                "{}[{}:{}]",
                interner
                    .resolve(nets[net_idx.0].name)
                    .unwrap_or("<unknown>"),
                msb,
                lsb
            ),
            crate::netlist::parse::NetRef::Literal(bits) => format!("{}", bits),
        };
        port_map.insert(port_name.to_string(), net_name_str);
        if pin_dir == 2 {
            match netref {
                crate::netlist::parse::NetRef::Simple(net_idx) => {
                    if let Some(bv) = net_to_bv.get(net_idx) {
                        assert_eq!(bv.get_bit_count(), 1);
                        input_map.insert(port_name.to_string(), *bv.get_lsb(0));
                    } else {
                        let net_name = interner
                            .resolve(nets[net_idx.0].name)
                            .unwrap_or("<unknown>");
                        missing_inputs.push(format!(
                            "{} (NetIndex({}), name='{}')",
                            port_name, net_idx.0, net_name
                        ));
                    }
                }
                crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => {
                    if let Some(bv) = net_to_bv.get(net_idx) {
                        if (*bit as usize) >= bv.get_bit_count() {
                            missing_inputs.push(format!(
                                "{} (NetIndex({}), name='{}', bit={})",
                                port_name,
                                net_idx.0,
                                interner
                                    .resolve(nets[net_idx.0].name)
                                    .unwrap_or("<unknown>"),
                                bit
                            ));
                        } else {
                            input_map.insert(port_name.to_string(), *bv.get_lsb(*bit as usize));
                        }
                    } else {
                        let net_name = interner
                            .resolve(nets[net_idx.0].name)
                            .unwrap_or("<unknown>");
                        missing_inputs.push(format!(
                            "{} (NetIndex({}), name='{}', bit={})",
                            port_name, net_idx.0, net_name, bit
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    (input_map, missing_inputs, port_map)
}

fn handle_dff_identity_override(
    type_name: &str,
    inst_name: &str,
    port_name: &str,
    inst: &crate::netlist::parse::NetlistInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
    dff_cells: &std::collections::HashSet<String>,
) -> bool {
    if !dff_cells.contains(type_name) {
        return false;
    }
    // Find D input and Q output
    let d_input = inst.connections.iter().find_map(|(p, nref)| {
        let pname = interner.resolve(*p).unwrap();
        if pname.eq_ignore_ascii_case("d") {
            Some(nref)
        } else {
            None
        }
    });
    if let Some(d_netref) = d_input {
        log::trace!(
            "DFF identity override: cell '{}' (instance '{}'), wiring D->Q for output '{}'",
            type_name,
            inst_name,
            port_name
        );
        match port_name {
            q if q.eq_ignore_ascii_case("q") => {
                // Get the bitvector for the D input netref
                let d_bv = match d_netref {
                    crate::netlist::parse::NetRef::Simple(d_netidx) => {
                        net_to_bv.get(d_netidx).cloned()
                    }
                    crate::netlist::parse::NetRef::BitSelect(d_netidx, bit) => {
                        net_to_bv.get(d_netidx).map(|bv| {
                            let mut bv1 =
                                AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); 1]);
                            bv1.set_lsb(0, *bv.get_lsb(*bit as usize));
                            bv1
                        })
                    }
                    crate::netlist::parse::NetRef::PartSelect(d_netidx, msb, lsb) => {
                        net_to_bv.get(d_netidx).map(|bv| {
                            let width = (*msb as usize) - (*lsb as usize) + 1;
                            let mut bv_part =
                                AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width]);
                            for i in 0..width {
                                bv_part.set_lsb(i, *bv.get_lsb(*lsb as usize + i));
                            }
                            bv_part
                        })
                    }
                    crate::netlist::parse::NetRef::Literal(_) => None, // Not supported for D
                };
                if let Some(d_bv) = d_bv {
                    for (p, q_ref) in &inst.connections {
                        let pname = interner.resolve(*p).unwrap();
                        if pname.eq_ignore_ascii_case("q") {
                            match q_ref {
                                crate::netlist::parse::NetRef::Simple(q_netidx) => {
                                    net_to_bv.insert(*q_netidx, d_bv.clone());
                                }
                                crate::netlist::parse::NetRef::BitSelect(q_netidx, bit) => {
                                    let width = (*bit as usize) + 1;
                                    let mut bv = net_to_bv.remove(q_netidx).unwrap_or_else(|| {
                                        AigBitVector::from_lsb_is_index_0(&vec![
                                            gb.get_false();
                                            width
                                        ])
                                    });
                                    if bv.get_bit_count() <= *bit as usize {
                                        let mut new_bv = AigBitVector::from_lsb_is_index_0(&vec![
                                                gb.get_false();
                                                width
                                            ]);
                                        for i in 0..bv.get_bit_count() {
                                            new_bv.set_lsb(i, *bv.get_lsb(i));
                                        }
                                        bv = new_bv;
                                    }
                                    bv.set_lsb(*bit as usize, *d_bv.get_lsb(0));
                                    net_to_bv.insert(*q_netidx, bv);
                                }
                                crate::netlist::parse::NetRef::PartSelect(q_netidx, msb, lsb) => {
                                    let width = (*msb as usize) - (*lsb as usize) + 1;
                                    let mut bv = net_to_bv.remove(q_netidx).unwrap_or_else(|| {
                                        AigBitVector::from_lsb_is_index_0(&vec![
                                            gb.get_false();
                                            width
                                        ])
                                    });
                                    if bv.get_bit_count() < width {
                                        let mut new_bv = AigBitVector::from_lsb_is_index_0(&vec![
                                                gb.get_false();
                                                width
                                            ]);
                                        for i in 0..bv.get_bit_count() {
                                            new_bv.set_lsb(i, *bv.get_lsb(i));
                                        }
                                        bv = new_bv;
                                    }
                                    for i in 0..width {
                                        bv.set_lsb(i, *d_bv.get_lsb(i));
                                    }
                                    net_to_bv.insert(*q_netidx, bv);
                                }
                                crate::netlist::parse::NetRef::Literal(_) => {
                                    log::warn!(
                                        "DFF identity override: Q output as literal not supported for cell '{}' instance '{}'",
                                        type_name,
                                        inst_name
                                    );
                                }
                            }
                        }
                    }
                } else {
                    log::warn!(
                        "DFF identity override: D net not found for cell '{}' instance '{}'",
                        type_name,
                        inst_name
                    );
                }
                return true;
            }
            _ => {
                log::warn!(
                    "DFF identity override: unexpected D/Q pin mapping for cell '{}' instance '{}' output '{}'",
                    type_name,
                    inst_name,
                    port_name
                );
            }
        }
    } else {
        log::warn!(
            "DFF identity override: D input not found for cell '{}' instance '{}'",
            type_name,
            inst_name
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::parse::{
        Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
    };
    use string_interner::{StringInterner, backend::StringBackend};

    #[test]
    fn test_inverter_projection() {
        // Build a minimal netlist and Liberty proto for an inverter
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INVX1");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
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
        let instances = vec![NetlistInstance {
            type_name: invx1,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("A"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Y"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(1)),
                ),
            ],
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            ports,
            wires: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "INVX1".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(!A)".to_string(),
                    },
                ],
                area: 1.0,
            }],
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_bitselect_output_projection() {
        // Build a netlist and Liberty proto for a buffer with bit-select output
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n = interner.get_or_intern("n");
        let buf = interner.get_or_intern("BUFX1");
        let u1 = interner.get_or_intern("u1");
        // n is a 4-bit net
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: n,
                width: Some((3, 0)),
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
                width: Some((3, 0)),
                name: n,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: buf,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("A"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Y"),
                    NetRef::BitSelect(crate::netlist::parse::NetIndex(1), 3),
                ),
            ],
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            ports,
            wires: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "BUFX1".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "A".to_string(),
                    },
                ],
                area: 1.0,
            }],
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
        )
        .expect("BitSelect output should be supported");
        let s = gate_fn.to_string();
        // n[3] should be assigned to a nonzero node (not literal false)
        assert!(
            s.contains("n[3] = %1") || s.contains("n[3] = %"),
            "GateFn output: {}",
            s
        );
    }

    #[test]
    fn test_dff_identity_override_simple() {
        // D and Q are both Simple netrefs
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("d"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("q"), NetRef::Simple(NetIndex(1))),
            ],
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            ports,
            wires: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFF".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "d".to_string(),
                        direction: 2,
                        function: "".to_string(),
                    },
                    crate::liberty_proto::Pin {
                        name: "q".to_string(),
                        direction: 1,
                        function: "d".to_string(),
                    },
                ],
                area: 1.0,
            }],
        };
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());
        // Set up the input value for D
        let mut gb = GateBuilder::new("top".to_string(), GateBuilderOptions::no_opt());
        let d_bv = gb.add_input("d".to_string(), 1);
        let mut net_to_bv = HashMap::new();
        net_to_bv.insert(NetIndex(0), d_bv.clone());
        // Run the projection
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
        )
        .unwrap();
        // The output should be wired directly from D
        let q_input = gate_fn.inputs.iter().find(|i| i.name == "d").unwrap();
        let q_output = gate_fn.outputs.iter().find(|o| o.name == "q").unwrap();
        assert_eq!(
            q_input.bit_vector.get_lsb(0),
            q_output.bit_vector.get_lsb(0)
        );
    }
}
