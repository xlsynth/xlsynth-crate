// SPDX-License-Identifier: Apache-2.0

//! Binary protobuf interchange for parsed technology-mapped netlists.
//! Cell interfaces and net indices are local to a file and module respectively.
//! Decoding reconstructs connectivity without a Liberty library. Source
//! locations, comments and attributes are outside the structural model.

use std::collections::HashSet;

use anyhow::{Context, Result, ensure};

use super::emit::validate_identifier;
use super::parse::NetRef;

mod decode;
mod encode;

/// Generated wire types for the public netlist interchange schema.
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/xlsynth.netlist.rs"));
}

pub use decode::decode_netlist;
pub use encode::encode_netlist;

// Prost allows 100 nested messages below the Netlist. Module, Assignment (or
// Instance), Ref, and a leaf payload such as Literal consume four. Each
// enclosing concatenation adds a Concat and a Ref for its selected part.
pub(super) const MAX_CONCAT_DEPTH: usize = (100 - 4) / 2;

/// Records which declarations supply a net's direction, type and packed range.
#[derive(Clone, Copy, Default)]
struct Declarations {
    port: bool,
    wire: bool,
    packed_range: bool,
}

/// Checks declaration namespaces and removes synthetic implicit-net ranges.
fn normalize_declarations(design: &mut pb::Netlist) -> Result<()> {
    for interface in &design.interfaces {
        let mut pins = HashSet::with_capacity(interface.pins.len());
        for pin in &interface.pins {
            ensure!(
                pins.insert(pin),
                "duplicate pin {pin:?} on cell {:?}",
                interface.cell
            );
        }
    }
    let mut modules = HashSet::with_capacity(design.modules.len());
    for module in &mut design.modules {
        normalize_module_declarations(module)
            .with_context(|| format!("declarations in module {:?}", module.name))?;
        ensure!(
            modules.insert(&module.name),
            "duplicate module {:?}",
            module.name
        );
    }
    Ok(())
}

/// Ports may redeclare their nets once as wires. Instances share the net
/// namespace.
fn normalize_module_declarations(module: &mut pb::Module) -> Result<()> {
    let mut names = HashSet::with_capacity(module.nets.len() + module.instances.len());
    for net in &module.nets {
        ensure!(names.insert(&net.name), "duplicate net {:?}", net.name);
    }
    for instance in &module.instances {
        ensure!(
            names.insert(&instance.name),
            "duplicate name {:?} for instance",
            instance.name
        );
    }

    let mut declared = vec![Declarations::default(); module.nets.len()];
    for port in &module.ports {
        let net = module
            .nets
            .get(port.net as usize)
            .context("port net index out of range")?;
        let decl = &mut declared[port.net as usize];
        ensure!(!decl.port, "duplicate port {:?}", net.name);
        decl.port = true;
        if let Some(range) = &port.range {
            ensure!(
                net.range.as_ref() == Some(range),
                "port range disagrees with net {}",
                net.name
            );
            decl.packed_range = true;
        }
    }
    for &wire in &module.wires {
        let decl = declared
            .get_mut(wire as usize)
            .context("wire net index out of range")?;
        ensure!(
            !decl.wire,
            "duplicate wire {:?}",
            module.nets[wire as usize].name
        );
        decl.wire = true;
    }
    for (net, declared) in module.nets.iter_mut().zip(declared) {
        if !declared.wire && !declared.packed_range {
            // The GV reader uses [0:0] for inferred scalars. Only an
            // explicit declaration can give such a net a packed range.
            ensure!(
                matches!(net.range, None | Some(pb::Range { left: 0, right: 0 })),
                "packed net {:?} has no range declaration",
                net.name
            );
            net.range = None;
        }
    }
    Ok(())
}

/// An assignment target or bidirectional terminal must be a net lvalue.
fn validate_lvalue(reference: &NetRef) -> Result<()> {
    match reference {
        NetRef::Simple(_) | NetRef::BitSelect(..) | NetRef::PartSelect(..) => Ok(()),
        NetRef::Concat(parts) => parts.iter().try_for_each(validate_lvalue),
        _ => anyhow::bail!("expected a net lvalue (net, select or concatenation of nets)"),
    }
}

#[cfg(test)]
mod tests;
