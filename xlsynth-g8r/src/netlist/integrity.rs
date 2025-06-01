// SPDX-License-Identifier: Apache-2.0

//! Basic integrity checks for parsed netlists.
//!
//! These checks look for simple wiring issues such as inputs that are never
//! used, outputs that are never driven, and wires that are declared but never
//! connected.

use std::collections::{HashMap, HashSet};

use crate::liberty_proto::{Library, PinDirection};
use crate::netlist::parse::{
    Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
};
use string_interner::symbol::SymbolU32;
use string_interner::{backend::StringBackend, StringInterner};

/// A specific integrity problem found during checking.
#[derive(Debug, PartialEq, Eq)]
pub enum IntegrityFinding {
    /// Input port was declared but never used by any instance.
    UnusedInput(String),
    /// Output port was declared but never driven by any instance.
    UndrivenOutput(String),
    /// A wire was declared but never driven by any instance.
    UndrivenWire(String),
    /// A wire was declared but never used by any instance.
    UnusedWire(String),
}

/// Result of running the integrity checker over a module.
#[derive(Debug, PartialEq, Eq)]
pub enum IntegritySummary {
    /// No issues were found.
    Clean,
    /// One or more problems were detected.
    Findings(Vec<IntegrityFinding>),
}

/// Check a parsed module for simple wiring issues.
///
/// `module`    - The module to check.
/// `nets`      - The global list of nets referenced by `module`.
/// `interner`  - The interner used when parsing the netlist.
/// `lib`       - Liberty library providing pin directions for instances.
pub fn check_module(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    lib: &Library,
) -> IntegritySummary {
    // Build mapping of (cell name -> {pin name -> direction})
    let mut dir_map: HashMap<&str, HashMap<&str, i32>> = HashMap::new();
    for cell in &lib.cells {
        let mut pins = HashMap::new();
        for pin in &cell.pins {
            pins.insert(pin.name.as_str(), pin.direction);
        }
        dir_map.insert(cell.name.as_str(), pins);
    }

    let mut used_as_input: HashSet<SymbolU32> = HashSet::new();
    let mut driven: HashSet<SymbolU32> = HashSet::new();

    // Module ports contribute to driving/using sets.
    for NetlistPort {
        direction, name, ..
    } in &module.ports
    {
        match direction {
            PortDirection::Input => {
                driven.insert(*name);
            }
            PortDirection::Output => {
                used_as_input.insert(*name); // environment observes the output
            }
            PortDirection::Inout => {
                driven.insert(*name);
                used_as_input.insert(*name);
            }
        }
    }

    // Walk instances and classify connections according to pin directions.
    for NetlistInstance {
        type_name,
        connections,
        ..
    } in &module.instances
    {
        let Some(type_str) = interner.resolve(*type_name) else {
            continue;
        };
        let pin_dirs = dir_map.get(type_str);
        for (port, netref) in connections {
            let Some(port_str) = interner.resolve(*port) else {
                continue;
            };
            let dir = pin_dirs
                .and_then(|m| m.get(port_str))
                .copied()
                .unwrap_or(PinDirection::Invalid as i32);
            let net_sym = match netref {
                NetRef::Simple(idx) | NetRef::BitSelect(idx, _) | NetRef::PartSelect(idx, _, _) => {
                    nets[idx.0].name
                }
                NetRef::Literal(_) => continue,
            };
            if dir == PinDirection::Output as i32 {
                driven.insert(net_sym);
            } else if dir == PinDirection::Input as i32 {
                used_as_input.insert(net_sym);
            } else {
                // Unknown or inout pin - treat as both directions.
                driven.insert(net_sym);
                used_as_input.insert(net_sym);
            }
        }
    }

    let mut findings = Vec::new();

    // Inputs should be used somewhere.
    for port in &module.ports {
        if port.direction == PortDirection::Input && !used_as_input.contains(&port.name) {
            let name = interner
                .resolve(port.name)
                .unwrap_or("<unknown>")
                .to_string();
            findings.push(IntegrityFinding::UnusedInput(name));
        }
    }

    // Outputs must be driven.
    for port in &module.ports {
        if port.direction == PortDirection::Output && !driven.contains(&port.name) {
            let name = interner
                .resolve(port.name)
                .unwrap_or("<unknown>")
                .to_string();
            findings.push(IntegrityFinding::UndrivenOutput(name));
        }
    }

    // Every declared wire should be driven and used.
    for net_idx in &module.wires {
        let sym = nets[net_idx.0].name;
        if !driven.contains(&sym) {
            let name = interner.resolve(sym).unwrap_or("<unknown>").to_string();
            findings.push(IntegrityFinding::UndrivenWire(name.clone()));
        }
        if !used_as_input.contains(&sym) {
            let name = interner.resolve(sym).unwrap_or("<unknown>").to_string();
            findings.push(IntegrityFinding::UnusedWire(name));
        }
    }

    if findings.is_empty() {
        IntegritySummary::Clean
    } else {
        IntegritySummary::Findings(findings)
    }
}
