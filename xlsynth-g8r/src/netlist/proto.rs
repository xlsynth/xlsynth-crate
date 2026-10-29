// SPDX-License-Identifier: Apache-2.0

//! Binary protobuf interchange for parsed technology-mapped netlists.
//! Cell interfaces and net indices are local to a file and module respectively.
//! Decoding reconstructs connectivity without a Liberty library. Source
//! locations, comments and attributes are outside the structural model.

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

/// Checks that emitted port and wire declarations account for every packed net.
fn validate_declarations(module: &pb::Module) -> Result<()> {
    let mut declared = vec![false; module.nets.len()];
    for port in &module.ports {
        let net = module
            .nets
            .get(port.net as usize)
            .context("port net index out of range")?;
        if let Some(range) = &port.range {
            ensure!(
                net.range.as_ref() == Some(range),
                "port range disagrees with net {}",
                net.name
            );
            declared[port.net as usize] = true;
        }
    }
    for &wire in &module.wires {
        *declared
            .get_mut(wire as usize)
            .context("wire net index out of range")? = true;
    }
    for (net, declared) in module.nets.iter().zip(declared) {
        // The GV reader uses [0:0] for implicit scalar wires.
        ensure!(
            declared || matches!(net.range, None | Some(pb::Range { left: 0, right: 0 })),
            "packed net {:?} has no range declaration",
            net.name
        );
    }
    Ok(())
}

/// A continuous assignment must have a net lvalue, including within a
/// concatenation.
fn validate_lvalue(reference: &NetRef) -> Result<()> {
    match reference {
        NetRef::Simple(_) | NetRef::BitSelect(..) | NetRef::PartSelect(..) => Ok(()),
        NetRef::Concat(parts) => {
            ensure!(!parts.is_empty(), "empty assignment lvalue");
            parts.iter().try_for_each(validate_lvalue)
        }
        _ => anyhow::bail!("assignment target must be a net, select or concatenation of nets"),
    }
}

#[cfg(test)]
mod tests;
