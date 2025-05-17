// SPDX-License-Identifier: Apache-2.0

//! Project a parsed netlist and Liberty proto into a GateFn.

use crate::gate::AigBitVector;
use crate::gate::GateFn;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty::cell_formula;
use crate::liberty_proto::Library;
use crate::netlist::parse::{Net, NetIndex, NetlistModule};
use std::collections::HashMap;
use std::collections::HashSet;
use string_interner::symbol::SymbolU32;
use string_interner::{backend::StringBackend, StringInterner};

pub fn project_gatefn_from_netlist_and_liberty(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
) -> Result<GateFn, String> {
    // Validate: only output pins have functions defined
    for cell in &liberty_lib.cells {
        for pin in &cell.pins {
            // 1 = OUTPUT, 2 = INPUT
            if pin.direction != 1 && !pin.function.is_empty() {
                return Err(format!(
                    "Liberty cell '{}' pin '{}' is not an output but has a function defined: {}",
                    cell.name, pin.name, pin.function
                ));
            }
        }
    }
    // Build cell formula map: (cell_name, pin_name) -> Term
    type CellPinKey = (String, String);
    let mut cell_formula_map: HashMap<CellPinKey, (crate::liberty::cell_formula::Term, String)> =
        HashMap::new();
    for cell in &liberty_lib.cells {
        for pin in &cell.pins {
            if pin.direction == 1 && !pin.function.is_empty() {
                let original_formula_string = pin.function.clone(); // Keep original string
                let term = cell_formula::parse_formula(&pin.function).map_err(|e| {
                    format!(
                        "Failed to parse formula for cell '{}', pin '{}' (formula: \\\"{}\\\"): {}",
                        cell.name, pin.name, original_formula_string, e
                    )
                })?;
                cell_formula_map.insert(
                    (cell.name.clone(), pin.name.clone()),
                    (term, original_formula_string), // Store Term and original string
                );
            }
        }
    }
    let module_name = interner.resolve(module.name).unwrap();
    let mut gb = GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt());
    let mut net_to_bv: HashMap<NetIndex, AigBitVector> = HashMap::new();
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Input {
            let net_idx = nets
                .iter()
                .position(|n| n.name == port.name)
                .map(NetIndex)
                .ok_or("input port net not found")?;
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
    // Pre-pass: check that all nets used as inputs are driven
    let mut used_as_input = HashSet::new();
    let mut driven = HashSet::new();
    // Module input ports are drivers
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Input {
            driven.insert(port.name);
        }
    }
    // Instance outputs are drivers; inputs are users
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
                        // OUTPUT
                        driven.insert(nets[net_idx.0].name);
                    } else if pin_dir == 2 {
                        // INPUT
                        used_as_input.insert(nets[net_idx.0].name);
                    }
                }
                _ => {}
            }
        }
    }
    // Now check for undriven nets
    for net in &used_as_input {
        if !driven.contains(net) {
            let net_name = interner.resolve(*net).unwrap_or("<unknown>");
            panic!(
                "Net '{}' is used as an input but is never driven by any instance or module input!",
                net_name
            );
        }
    }
    // Worklist-based instance processing
    let mut unprocessed: Vec<_> = module.instances.iter().collect();
    let mut processed_any = true;
    while !unprocessed.is_empty() && processed_any {
        processed_any = false;
        let mut i = 0;
        while i < unprocessed.len() {
            let inst = unprocessed[i];
            let type_name = interner.resolve(inst.type_name).unwrap();
            let inst_name = interner.resolve(inst.instance_name).unwrap();
            // Build a map from pin name to direction for this cell
            let cell = liberty_lib
                .cells
                .iter()
                .find(|c| c.name == type_name)
                .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
            let mut pin_directions = HashMap::new();
            for pin in &cell.pins {
                pin_directions.insert(pin.name.as_str(), pin.direction);
            }
            let mut input_map = HashMap::new();
            let mut missing_inputs = Vec::new();
            // First pass: gather all input pins
            for (port, netref) in &inst.connections {
                let port_name = interner.resolve(*port).unwrap();
                let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
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
                                    return Err(format!(
                                        "Bit-select out of range for net '{}' (width {}) in instance '{}' port '{}', net_idx={:?}, bit={}",
                                        interner.resolve(nets[net_idx.0].name).unwrap(),
                                        bv.get_bit_count(),
                                        inst_name,
                                        port_name,
                                        net_idx,
                                        bit
                                    ));
                                }
                                input_map.insert(port_name.to_string(), *bv.get_lsb(*bit as usize));
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
            // Second pass: process all output pins
            let mut processed_any_output = false;
            for (port, netref) in &inst.connections {
                let port_name = interner.resolve(*port).unwrap();
                let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
                if pin_dir == 1 {
                    let key = (type_name.to_string(), port_name.to_string());
                    let (formula_ast, original_formula_str) =
                        cell_formula_map.get(&key).ok_or_else(|| {
                            format!("No formula for cell '{}', pin '{}'", type_name, port_name)
                        })?;

                    let context = crate::liberty::cell_formula::EmitContext {
                        cell_name: type_name,
                        // Use the stored original string
                        original_formula: original_formula_str.as_str(),
                    };
                    let out_op = formula_ast.emit_formula_term(&mut gb, &input_map, &context)?;
                    match netref {
                        crate::netlist::parse::NetRef::Simple(net_idx) => {
                            if net_to_bv.contains_key(net_idx) {
                                return Err(format!("Output net '{}' already assigned", net_idx.0));
                            }
                            net_to_bv.insert(*net_idx, AigBitVector::from_bit(out_op));
                            log::trace!("Assigned output net '{}' (NetIndex({})) in instance '{}' (cell '{}')", interner.resolve(nets[net_idx.0].name).unwrap_or("<unknown>"), net_idx.0, inst_name, type_name);
                            assert!(
                                net_to_bv.contains_key(net_idx),
                                "After assignment, output net not present in net_to_bv"
                            );
                        }
                        crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => {
                            let width = nets[net_idx.0]
                                .width
                                .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                                .unwrap_or(1);
                            let mut bv = net_to_bv.remove(net_idx).unwrap_or_else(|| {
                                // If not present, create a vector of false
                                AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width])
                            });
                            if (*bit as usize) >= width {
                                return Err(format!(
                                    "Bit-select out of range for output net '{}' (width {}) in instance '{}' port '{}', net_idx={:?}, bit={}",
                                    interner.resolve(nets[net_idx.0].name).unwrap(),
                                    width,
                                    inst_name,
                                    port_name,
                                    net_idx,
                                    bit
                                ));
                            }
                            bv.set_lsb(*bit as usize, out_op);
                            net_to_bv.insert(*net_idx, bv);
                            log::trace!("Assigned output net '{}' (NetIndex({})) bit {} in instance '{}' (cell '{}')", interner.resolve(nets[net_idx.0].name).unwrap_or("<unknown>"), net_idx.0, bit, inst_name, type_name);
                            assert!(
                                net_to_bv.contains_key(net_idx),
                                "After assignment, output net not present in net_to_bv"
                            );
                        }
                        _ => {
                            log::info!(
                                "Unsupported output netref: {:?} in instance '{}' (cell '{}'). All connections:",
                                netref, inst_name, type_name
                            );
                            for (p, nref) in &inst.connections {
                                let pname = interner.resolve(*p).unwrap();
                                log::info!("  .{}({:?})", pname, nref);
                            }
                            let mut msg = format!(
                                "Only simple netrefs supported for output in instance '{}', cell type '{}', netref: {:?}\nAll connections for this instance:",
                                inst_name, type_name, netref
                            );
                            for (p, nref) in &inst.connections {
                                let pname = interner.resolve(*p).unwrap();
                                msg.push_str(&format!("\n  .{}({:?})", pname, nref));
                            }
                            return Err(msg);
                        }
                    }
                    processed_any_output = true;
                }
            }
            if processed_any_output {
                unprocessed.remove(i);
                processed_any = true;
            } else {
                i += 1;
            }
        }
        if !processed_any && !unprocessed.is_empty() {
            return Err(format!("Could not resolve all instance dependencies (possible cycle or missing driver). Remaining instances: {}", unprocessed.len()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::parse::{
        Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
    };
    use string_interner::{backend::StringBackend, StringInterner};

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
        let gate_fn =
            project_gatefn_from_netlist_and_liberty(&module, &nets, &interner, &liberty_lib)
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
        let gate_fn =
            project_gatefn_from_netlist_and_liberty(&module, &nets, &interner, &liberty_lib)
                .expect("BitSelect output should be supported");
        let s = gate_fn.to_string();
        // n[3] should be assigned to a nonzero node (not literal false)
        assert!(
            s.contains("n[3] = %1") || s.contains("n[3] = %"),
            "GateFn output: {}",
            s
        );
    }
}
