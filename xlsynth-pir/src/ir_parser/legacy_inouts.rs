// SPDX-License-Identifier: Apache-2.0

//! Normalize historical port and instance accesses onto block-local inout nets.

use std::collections::BTreeMap;

use super::ParseError;
use crate::ir::{self, Block};
use crate::ir_block::InoutNetNames;

/// Identifies information implied by older inout node spellings.
pub(super) enum LegacyInout {
    Combined {
        drive: ir::NodeRef,
        output_enable: ir::NodeRef,
        sv_type: Option<String>,
    },
    PublicPort,
    PrivateInstance {
        instantiation: String,
        port_name: String,
    },
}

/// Resolves references to public aliases and shares a private wire across all
/// legacy accesses of each child pin. All user names are known at this point.
pub(super) fn resolve_legacy_inouts(
    block: &mut Block,
    accesses: Vec<(ir::NodeRef, LegacyInout)>,
    added: &mut Vec<ir::Node>,
) -> Result<(), ParseError> {
    let mut private = BTreeMap::<(String, String), String>::new();
    let public: BTreeMap<_, _> = block
        .inout_nets
        .iter()
        .enumerate()
        .flat_map(|(index, net)| net.ports.iter().cloned().map(move |alias| (alias, index)))
        .collect();
    let mut names = None;
    for (reference, access) in accesses {
        let target = match access {
            LegacyInout::PublicPort => {
                let node = block.get_node(reference);
                let (ir::NodePayload::ExtInoutRead { net }
                | ir::NodePayload::ExtInoutDrive { net, .. }
                | ir::NodePayload::ExtInstantiationInout { net, .. }) = &node.payload
                else {
                    unreachable!("legacy public references read, drive, or bind a pin")
                };
                let index = public.get(net).ok_or_else(|| {
                    ParseError::new(format!(
                        "{} requires parent inout '{net}' in block '{}'",
                        node.payload.get_operator(),
                        block.name
                    ))
                })?;
                block.inout_nets[*index].name.clone()
            }
            LegacyInout::PrivateInstance {
                instantiation,
                port_name,
            } => match private.entry((instantiation, port_name)) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.get().clone(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let (instance, port) = entry.key();
                    let name = names
                        .get_or_insert_with(|| InoutNetNames::new(block))
                        .allocate(&format!("{instance}_{port}"));
                    let access = block.get_node(reference);
                    let ty = match &access.payload {
                        ir::NodePayload::ExtInoutRead { .. } => access.ty.clone(),
                        ir::NodePayload::ExtInoutDrive { drive, .. } => {
                            block.get_node_ty(*drive).clone()
                        }
                        _ => unreachable!("legacy child access is a read or a drive"),
                    };
                    added.push(ir::Node {
                        text_id: 0,
                        name: None,
                        ty: ir::Type::nil(),
                        payload: ir::NodePayload::ExtInstantiationInout {
                            instantiation: instance.clone(),
                            port_name: port.clone(),
                            net: name.clone(),
                        },
                        pos: access.pos.clone(),
                    });
                    block.inout_nets.push(ir::InoutNet {
                        name: name.clone(),
                        ty,
                        ports: vec![],
                        sv_type: None,
                    });
                    entry.insert(name).clone()
                }
            },
            LegacyInout::Combined { .. } => {
                unreachable!("combined pins are declared during parsing")
            }
        };
        let (ir::NodePayload::ExtInoutRead { net }
        | ir::NodePayload::ExtInoutDrive { net, .. }
        | ir::NodePayload::ExtInstantiationInout { net, .. }) =
            &mut block.nodes[reference.index].payload
        else {
            unreachable!("legacy accesses are lowered to net operations")
        };
        *net = target;
    }
    Ok(())
}

/// Assigns package-unique IDs to the extra drives and bindings produced by
/// legacy syntax.
pub(super) fn append_legacy_nodes(
    block: &mut Block,
    nodes: Vec<ir::Node>,
    last_id: &mut usize,
) -> Result<(), ParseError> {
    for mut node in nodes {
        *last_id = last_id.checked_add(1).ok_or_else(|| {
            ParseError::new("node ID overflow while expanding legacy inouts".into())
        })?;
        node.text_id = *last_id;
        block.nodes.push(node);
    }
    Ok(())
}
