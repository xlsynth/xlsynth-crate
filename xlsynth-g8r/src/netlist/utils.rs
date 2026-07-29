// SPDX-License-Identifier: Apache-2.0

//! Utility routines for netlist analysis and reporting.

use crate::netlist::parse::{
    AssignExpr, Net, NetRef, NetlistAssignKind, NetlistModule, PortDirection,
};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, HashSet};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Validates literal tie-offs to scalar outputs or individual packed bits.
pub(crate) fn validate_constant_output_assignments(
    module: &NetlistModule,
    nets: &[Net],
) -> Result<()> {
    let mut output_nets = HashSet::new();
    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        let net = module
            .find_net_index(port.name, nets)
            .ok_or_else(|| anyhow!("constant-output assignment refers to an unknown port net"))?;
        output_nets.insert(net.0);
    }

    let mut assigned_bits = HashSet::new();
    for assign in &module.assigns {
        if assign.kind != NetlistAssignKind::Continuous {
            return Err(anyhow!(
                "mapped-netlist emission only supports continuous output tie-offs"
            ));
        }
        let AssignExpr::Leaf(NetRef::Literal(value)) = &assign.rhs else {
            return Err(anyhow!(
                "mapped-netlist emission only supports constant output assignments"
            ));
        };

        match &assign.lhs {
            NetRef::Simple(index) => {
                if !output_nets.contains(&index.0) {
                    return Err(anyhow!(
                        "mapped-netlist emission only supports assignments directly to module outputs"
                    ));
                }
                let net = nets
                    .get(index.0)
                    .ok_or_else(|| anyhow!("constant output net index is out of bounds"))?;
                let width = net.width_bits();
                if value.get_bit_count() != width {
                    return Err(anyhow!(
                        "constant output assignment width {} does not match output width {}",
                        value.get_bit_count(),
                        width
                    ));
                }
                for bit in 0..width {
                    if !assigned_bits.insert((index.0, bit)) {
                        return Err(anyhow!(
                            "mapped-netlist emission found multiple assignments to one output bit"
                        ));
                    }
                }
            }
            NetRef::BitSelect(index, bit) => {
                if !output_nets.contains(&index.0) {
                    return Err(anyhow!(
                        "mapped-netlist emission only supports assignments directly to module outputs"
                    ));
                }
                let net = nets
                    .get(index.0)
                    .ok_or_else(|| anyhow!("constant output net index is out of bounds"))?;
                let offset = net.bit_offset(*bit).ok_or_else(|| {
                    anyhow!("constant output bit {bit} is outside its declared packed width")
                })?;
                if value.get_bit_count() != 1 {
                    return Err(anyhow!(
                        "a constant packed-output bit requires a one-bit tie-off"
                    ));
                }
                if !assigned_bits.insert((index.0, offset)) {
                    return Err(anyhow!(
                        "mapped-netlist emission found multiple assignments to one output bit"
                    ));
                }
            }
            _ => {
                return Err(anyhow!(
                    "mapped-netlist emission requires scalar or packed output assignment destinations"
                ));
            }
        }
    }
    Ok(())
}

/// Validates scalar constant assignments that directly drive module outputs.
pub(crate) fn scalar_constant_output_assignments(
    module: &NetlistModule,
    nets: &[Net],
) -> Result<BTreeMap<usize, bool>> {
    let mut output_nets = HashSet::new();
    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        let net = module
            .find_net_index(port.name, nets)
            .ok_or_else(|| anyhow!("constant-output assignment refers to an unknown port net"))?;
        output_nets.insert(net.0);
    }

    let mut assignments = BTreeMap::new();
    for assign in &module.assigns {
        if assign.kind != NetlistAssignKind::Continuous {
            return Err(anyhow!(
                "mapped-netlist optimization only supports continuous output tie-offs"
            ));
        }
        let NetRef::Simple(net) = &assign.lhs else {
            return Err(anyhow!(
                "mapped-netlist optimization requires scalar output assignment destinations"
            ));
        };
        if !output_nets.contains(&net.0) {
            return Err(anyhow!(
                "mapped-netlist optimization only supports assignments directly to module outputs"
            ));
        }
        let AssignExpr::Leaf(NetRef::Literal(bits)) = &assign.rhs else {
            return Err(anyhow!(
                "mapped-netlist optimization only supports constant output assignments"
            ));
        };
        if bits.get_bit_count() != 1 {
            return Err(anyhow!(
                "mapped-netlist optimization requires one-bit output tie-offs"
            ));
        }
        let value = bits.get_bit(0).map_err(|error| anyhow!("{error}"))?;
        if assignments.insert(net.0, value).is_some() {
            return Err(anyhow!(
                "mapped-netlist optimization found multiple assignments to one output"
            ));
        }
    }
    Ok(assignments)
}

/// Returns a Vec of (instance_name, cell_type) for all instances in the
/// modules, using the interner for resolution.
pub fn instance_names_and_types(
    modules: &[NetlistModule],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for module in modules {
        for inst in &module.instances {
            let inst_str = interner
                .resolve(inst.instance_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            let type_str = interner
                .resolve(inst.type_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            pairs.push((inst_str, type_str));
        }
    }
    pairs
}

/// Returns a Vec of (module_name, instance_name, cell_type) for all instances
/// in the modules, using the interner for resolution.
pub fn module_instance_names_and_types(
    modules: &[NetlistModule],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Vec<(String, String, String)> {
    let mut triples: Vec<(String, String, String)> = Vec::new();
    for module in modules {
        let module_str = interner
            .resolve(module.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<unknown>".to_owned());
        for inst in &module.instances {
            let inst_str = interner
                .resolve(inst.instance_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            let type_str = interner
                .resolve(inst.type_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            triples.push((module_str.clone(), inst_str, type_str));
        }
    }
    triples
}
