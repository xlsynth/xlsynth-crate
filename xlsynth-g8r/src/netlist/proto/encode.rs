// SPDX-License-Identifier: Apache-2.0

use super::{MAX_CONCAT_DEPTH, pb, validate_declarations, validate_identifier, validate_lvalue};
use crate::netlist::io::{ParsedNetlist, resolve_symbol};
use crate::netlist::parse::{
    AssignExpr, NetIndex, NetRef, NetlistAssignKind, NetlistInstance, NetlistModule, PortDirection,
    PortId,
};
use anyhow::{Context, Result, bail, ensure};
use prost::Message;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;

/// Assigns each distinct (cell type, ordered pin list) its index at first use.
#[derive(Default)]
struct Interfaces {
    indices: HashMap<(PortId, Vec<PortId>), u32>,
    records: Vec<pb::CellInterface>,
}

impl Interfaces {
    fn intern(&mut self, parsed: &ParsedNetlist, instance: &NetlistInstance) -> Result<u32> {
        let pins = instance.connections.iter().map(|(pin, _)| *pin).collect();
        match self.indices.entry((instance.type_name, pins)) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let index =
                    u32::try_from(self.records.len()).context("too many cell interfaces")?;
                let (cell, pins) = entry.key();
                self.records.push(pb::CellInterface {
                    cell: encode_name(parsed, *cell, "cell type")?,
                    pins: pins
                        .iter()
                        .map(|pin| encode_name(parsed, *pin, "pin name"))
                        .collect::<Result<_>>()?,
                });
                entry.insert(index);
                Ok(index)
            }
        }
    }
}

fn encode_name(parsed: &ParsedNetlist, symbol: PortId, what: &str) -> Result<String> {
    let name = resolve_symbol(&parsed.interner, symbol, what)?;
    validate_identifier(&name)?;
    Ok(name)
}

/// Converts a file-global net index to a checked module-local index.
fn local_net(index: NetIndex, range: &Range<usize>) -> Result<u32> {
    ensure!(
        range.contains(&index.0),
        "net index {} outside module net range {range:?}",
        index.0
    );
    u32::try_from(index.0 - range.start).context("module net index exceeds u32")
}

fn encode_range(width: Option<(u32, u32)>) -> Option<pb::Range> {
    width.map(|(left, right)| pb::Range { left, right })
}

/// Encodes a reference, counting enclosing concatenations to ensure
/// decodability.
fn encode_ref(reference: &NetRef, range: &Range<usize>, concat_depth: usize) -> Result<pb::Ref> {
    ensure!(
        concat_depth <= MAX_CONCAT_DEPTH,
        "net reference exceeds the protobuf nesting limit of {MAX_CONCAT_DEPTH} enclosing concatenations; flatten the concatenation"
    );
    let kind = match reference {
        NetRef::Simple(index) => pb::r#ref::Kind::Net(local_net(*index, range)?),
        NetRef::BitSelect(index, left) => pb::r#ref::Kind::Select(pb::Select {
            net: local_net(*index, range)?,
            left: *left,
            right: None,
        }),
        NetRef::PartSelect(index, left, right) => pb::r#ref::Kind::Select(pb::Select {
            net: local_net(*index, range)?,
            left: *left,
            right: Some(*right),
        }),
        NetRef::Literal(bits) => {
            ensure!(bits.get_bit_count() > 0, "zero-width Verilog literal");
            pb::r#ref::Kind::Literal(pb::Literal {
                width: u64::try_from(bits.get_bit_count())?,
                little_endian: bits.to_bytes(),
            })
        }
        NetRef::UnknownLiteral(0) => bail!("zero-width Verilog literal"),
        NetRef::UnknownLiteral(width) => pb::r#ref::Kind::UnknownWidth(u64::try_from(*width)?),
        NetRef::Concat(parts) if parts.is_empty() => bail!("empty Verilog concatenation"),
        NetRef::Concat(parts) => pb::r#ref::Kind::Concat(pb::Concat {
            parts: parts
                .iter()
                .map(|part| encode_ref(part, range, concat_depth + 1))
                .collect::<Result<_>>()?,
        }),
        NetRef::Unconnected => bail!("only a whole pin connection may be open"),
    };
    Ok(pb::Ref { kind: Some(kind) })
}

/// Encodes a module in declaration order and resolves references in its
/// namespace.
fn encode_module(
    parsed: &ParsedNetlist,
    module: &NetlistModule,
    interfaces: &mut Interfaces,
) -> Result<pb::Module> {
    let range = &module.net_index_range;
    let local_nets = parsed
        .nets
        .get(range.clone())
        .with_context(|| format!("invalid module net range {range:?}"))?;
    let mut result = pb::Module {
        name: encode_name(parsed, module.name, "module name")?,
        ..Default::default()
    };

    let mut net_by_name = HashMap::with_capacity(local_nets.len());
    for (index, net) in local_nets.iter().enumerate() {
        let index = u32::try_from(index).context("module net index exceeds u32")?;
        let name = encode_name(parsed, net.name, "net name")?;
        net_by_name.insert(net.name, index);
        result.nets.push(pb::Net {
            name,
            range: encode_range(net.width),
        });
    }
    for port in &module.ports {
        let net = *net_by_name
            .get(&port.name)
            .context("port has no net in its module")?;
        let direction = match port.direction {
            PortDirection::Input => pb::port::Direction::Input,
            PortDirection::Output => pb::port::Direction::Output,
            PortDirection::Inout => pb::port::Direction::Inout,
        };
        result.ports.push(pb::Port {
            net,
            direction: direction as i32,
            range: encode_range(port.width),
        });
    }
    result.wires = module
        .wires
        .iter()
        .map(|wire| local_net(*wire, range))
        .collect::<Result<_>>()?;

    for assign in &module.assigns {
        let AssignExpr::Leaf(rhs) = &assign.rhs else {
            bail!("mapped netlists permit only wiring references in assignments")
        };
        let lhs_ref = encode_ref(&assign.lhs, range, 0)?;
        let rhs_ref = encode_ref(rhs, range, 0)?;
        validate_lvalue(&assign.lhs)?;
        if assign.kind == NetlistAssignKind::Tran {
            validate_lvalue(rhs)?;
        }
        result.assigns.push(pb::Assignment {
            lhs: Some(lhs_ref),
            rhs: Some(rhs_ref),
            tran: assign.kind == NetlistAssignKind::Tran,
        });
    }
    for instance in &module.instances {
        result
            .instances
            .push(encode_instance(parsed, instance, interfaces, range)?);
    }
    Ok(result)
}

/// Packs whole-net connections as plain indices and preserves other pin
/// expressions.
fn encode_instance(
    parsed: &ParsedNetlist,
    instance: &NetlistInstance,
    interfaces: &mut Interfaces,
    range: &Range<usize>,
) -> Result<pb::Instance> {
    let mut encoded = pb::Instance {
        name: encode_name(parsed, instance.instance_name, "instance name")?,
        interface: interfaces.intern(parsed, instance)?,
        ..Default::default()
    };
    let all_whole_nets = instance
        .connections
        .iter()
        .all(|(_, reference)| matches!(reference, NetRef::Simple(_)));
    for (_, reference) in &instance.connections {
        match reference {
            NetRef::Simple(index) if all_whole_nets => {
                encoded.nets.push(local_net(*index, range)?);
            }
            NetRef::Unconnected => encoded.connections.push(pb::Ref {
                kind: Some(pb::r#ref::Kind::Open(pb::Open {})),
            }),
            _ => encoded.connections.push(encode_ref(reference, range, 0)?),
        }
    }
    Ok(encoded)
}

/// Encodes the parsed modules as one binary mapped-netlist protobuf.
pub fn encode_netlist(parsed: &ParsedNetlist) -> Result<Vec<u8>> {
    let mut interfaces = Interfaces::default();
    let modules = parsed
        .modules
        .iter()
        .map(|module| encode_module(parsed, module, &mut interfaces))
        .collect::<Result<_>>()?;
    let design = pb::Netlist {
        interfaces: interfaces.records,
        modules,
    };
    validate_declarations(&design)?;
    Ok(design.encode_to_vec())
}
