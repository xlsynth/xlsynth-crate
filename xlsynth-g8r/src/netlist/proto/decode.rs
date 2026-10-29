// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use anyhow::{Context, Result, bail, ensure};
use prost::Message;
use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};
use xlsynth_pir::IrBits;

use super::pb::r#ref::Kind;
use super::{normalize_declarations, pb, validate_identifier, validate_lvalue};
use crate::netlist::io::ParsedNetlist;
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistAssignKind, NetlistInstance,
    NetlistModule, NetlistPort, PortDirection, Pos, Span,
};

// As with netlists constructed by the mapper, no source text positions exist.
const SYNTHETIC_POS: Pos = Pos {
    lineno: 0,
    colno: 0,
};

type Interner = StringInterner<StringBackend<SymbolU32>>;

/// Resolved type and ordered pin names shared by instances of an interface.
struct CellInterface {
    cell: SymbolU32,
    pins: Vec<SymbolU32>,
}

/// Decodes a binary mapped netlist into the same model used by GV readers.
pub fn decode_netlist(bytes: &[u8]) -> Result<ParsedNetlist> {
    let mut message = pb::Netlist::decode(bytes).context("decoding netlist protobuf")?;
    normalize_declarations(&mut message)?;
    let mut interner = Interner::new();
    let mut interfaces = Vec::with_capacity(message.interfaces.len());
    for interface in message.interfaces {
        let cell = intern(&mut interner, &interface.cell)?;
        let pins = interface
            .pins
            .iter()
            .map(|pin| intern(&mut interner, pin))
            .collect::<Result<_>>()?;
        interfaces.push(CellInterface { cell, pins });
    }

    let mut modules = Vec::with_capacity(message.modules.len());
    let mut nets = Vec::new();
    for module in message.modules {
        modules.push(
            decode_module(&module, &interfaces, &mut nets, &mut interner)
                .with_context(|| format!("decoding module {:?}", module.name))?,
        );
    }
    Ok(ParsedNetlist {
        modules,
        nets,
        interner,
    })
}

fn intern(interner: &mut Interner, name: &str) -> Result<SymbolU32> {
    validate_identifier(name)?;
    Ok(interner.get_or_intern(name))
}

/// Restores the module's local net namespace in the file-global net array.
fn decode_module(
    module: &pb::Module,
    interfaces: &[CellInterface],
    nets: &mut Vec<Net>,
    interner: &mut Interner,
) -> Result<NetlistModule> {
    let name = intern(interner, &module.name)?;
    let start = nets.len();
    for net in &module.nets {
        let name = intern(interner, &net.name)?;
        nets.push(Net {
            name,
            width: net.range.as_ref().map(|r| (r.left, r.right)),
        });
    }

    let net_index_range = start..nets.len();
    let mut ports = Vec::with_capacity(module.ports.len());
    for port in &module.ports {
        let index = net_index(&net_index_range, port.net)?;
        let net = &nets[index.0];
        let direction = match pb::port::Direction::try_from(port.direction)? {
            pb::port::Direction::Input => PortDirection::Input,
            pb::port::Direction::Output => PortDirection::Output,
            pb::port::Direction::Inout => PortDirection::Inout,
        };
        let width = port.range.as_ref().map(|r| (r.left, r.right));
        ports.push(NetlistPort {
            direction,
            width,
            name: net.name,
        });
    }
    let wires = module
        .wires
        .iter()
        .map(|&wire| net_index(&net_index_range, wire))
        .collect::<Result<_>>()?;

    let mut assigns = Vec::with_capacity(module.assigns.len());
    for assign in &module.assigns {
        let lhs = assign
            .lhs
            .as_ref()
            .context("assignment is missing its LHS")?;
        let rhs = assign
            .rhs
            .as_ref()
            .context("assignment is missing its RHS")?;
        let lhs = decode_ref(lhs, &net_index_range)?;
        let rhs = decode_ref(rhs, &net_index_range)?;
        validate_lvalue(&lhs)?;
        if assign.tran {
            validate_lvalue(&rhs)?;
        }
        assigns.push(NetlistAssign {
            kind: if assign.tran {
                NetlistAssignKind::Tran
            } else {
                NetlistAssignKind::Continuous
            },
            lhs,
            rhs: AssignExpr::Leaf(rhs),
            span: Span {
                start: SYNTHETIC_POS,
                limit: SYNTHETIC_POS,
            },
        });
    }

    let mut instances = Vec::with_capacity(module.instances.len());
    for instance in &module.instances {
        let instance_name = intern(interner, &instance.name)?;
        let interface = interfaces
            .get(instance.interface as usize)
            .with_context(|| {
                format!(
                    "instance {:?} uses out-of-range interface index {}",
                    instance.name, instance.interface
                )
            })?;
        let connections = decode_connections(instance, interface, &net_index_range)?;
        instances.push(NetlistInstance {
            type_name: interface.cell,
            instance_name,
            connections,
            inst_lineno: SYNTHETIC_POS.lineno,
            inst_colno: SYNTHETIC_POS.colno,
        });
    }
    Ok(NetlistModule {
        name,
        net_index_range,
        ports,
        wires,
        assigns,
        instances,
    })
}

/// Resolves either a packed net list or a full list of pin expressions.
fn decode_connections(
    instance: &pb::Instance,
    interface: &CellInterface,
    range: &Range<usize>,
) -> Result<Vec<(SymbolU32, NetRef)>> {
    ensure!(
        instance.nets.is_empty() || instance.connections.is_empty(),
        "instance {:?} specifies both nets and connections",
        instance.name
    );
    let count = instance.nets.len() + instance.connections.len();
    ensure!(
        interface.pins.len() == count,
        "instance {:?} has {} connections for an interface with {} pins",
        instance.name,
        count,
        interface.pins.len()
    );
    let mut connections = Vec::with_capacity(count);
    for (&pin, &net) in interface.pins.iter().zip(&instance.nets) {
        connections.push((pin, NetRef::Simple(net_index(range, net)?)));
    }
    for (&pin, reference) in interface.pins.iter().zip(&instance.connections) {
        let connection = match &reference.kind {
            Some(Kind::Open(_)) => NetRef::Unconnected,
            _ => decode_ref(reference, range)?,
        };
        connections.push((pin, connection));
    }
    Ok(connections)
}

/// Translates a local protobuf net ID, rejecting references outside the module.
fn net_index(range: &Range<usize>, local: u32) -> Result<NetIndex> {
    let local = local as usize;
    ensure!(
        local < range.len(),
        "net index {local} out of range ({} nets)",
        range.len()
    );
    Ok(NetIndex(range.start + local))
}

/// Reconstructs a wiring expression, which must have a value in this context.
fn decode_ref(reference: &pb::Ref, range: &Range<usize>) -> Result<NetRef> {
    let kind = reference
        .kind
        .as_ref()
        .context("missing net reference kind")?;
    Ok(match kind {
        Kind::Net(n) => NetRef::Simple(net_index(range, *n)?),
        Kind::Select(select) => {
            let index = net_index(range, select.net)?;
            match select.right {
                Some(right) => NetRef::PartSelect(index, select.left, right),
                None => NetRef::BitSelect(index, select.left),
            }
        }
        Kind::Literal(literal) => {
            ensure!(literal.width > 0, "zero-width Verilog literal");
            let width = usize::try_from(literal.width).context("literal width exceeds usize")?;
            NetRef::Literal(IrBits::from_le_bytes(width, &literal.little_endian)?)
        }
        Kind::UnknownWidth(0) => bail!("zero-width Verilog literal"),
        Kind::UnknownWidth(width) => NetRef::UnknownLiteral(
            usize::try_from(*width).context("unknown literal width exceeds usize")?,
        ),
        Kind::Concat(concat) if concat.parts.is_empty() => bail!("empty Verilog concatenation"),
        Kind::Concat(concat) => NetRef::Concat(
            concat
                .parts
                .iter()
                .map(|part| decode_ref(part, range))
                .collect::<Result<_>>()?,
        ),
        Kind::Open(_) => bail!("open connection outside an instance pin"),
    })
}
