// SPDX-License-Identifier: Apache-2.0

//! Connect flattened child bodies to all sibling instances before resolving
//! the sources of individual ports.

use std::collections::{HashMap, HashSet};

use super::uniquify_node_name;
use crate::block_feedback::projection::is_empty_data;
use crate::ir::{Block, BlockReset, Instantiation, Node, NodePayload, NodeRef, Register, Unop};
use crate::ir_utils::remap_payload_with;
use crate::ir_value_utils::zero_ir_value_for_type;

/// Parent driver for each formal input and all uses of each child output.
#[derive(Default)]
struct Ports<'a> {
    inputs: HashMap<&'a str, NodeRef>,
    outputs: Vec<(&'a str, NodeRef)>,
}

/// Bindings and module-scoped names for one parent. Each original port can
/// initially refer to a sibling output whose driver has yet to be copied.
pub(super) struct Wiring<'a> {
    ports: HashMap<&'a str, Ports<'a>>,
    aliases: Vec<Option<NodeRef>>,
    resets: Vec<(&'a str, BlockReset)>,
    node_names: HashMap<String, usize>,
    register_names: HashMap<String, usize>,
}

impl<'a> Wiring<'a> {
    /// Indexes the original parent once, before instance bodies are appended.
    pub(super) fn new(parent: &'a Block) -> Result<Self, String> {
        let mut ports = HashMap::<&str, Ports>::new();
        for (index, node) in parent.nodes.iter().enumerate() {
            match &node.payload {
                NodePayload::InstantiationInput {
                    instantiation,
                    port_name,
                    arg,
                } => {
                    if ports
                        .entry(instantiation)
                        .or_default()
                        .inputs
                        .insert(port_name, *arg)
                        .is_some()
                    {
                        return Err(format!(
                            "block_inline: duplicate input '{port_name}' for instantiation '{instantiation}'"
                        ));
                    }
                }
                NodePayload::InstantiationOutput {
                    instantiation,
                    port_name,
                } => {
                    ports
                        .entry(instantiation)
                        .or_default()
                        .outputs
                        .push((port_name, NodeRef { index }));
                }
                _ => {} // Ordinary operations have explicit operands.
            }
        }
        Ok(Self {
            ports,
            aliases: vec![None; parent.nodes.len()],
            resets: Vec::new(),
            node_names: parent
                .nodes
                .iter()
                .filter_map(|n| n.name.clone().map(|name| (name, 1)))
                .collect(),
            register_names: parent
                .registers
                .iter()
                .map(|r| (r.name.clone(), 1))
                .collect(),
        })
    }

    /// Copies one instance and records the parent drivers of its outputs and
    /// reset.
    pub(super) fn copy_instance(
        &mut self,
        parent: &mut Block,
        instance: &'a Instantiation,
        child: &Block,
        last_id: &mut usize,
    ) -> Result<(), String> {
        let name = instance.name.as_str();
        let bindings = self.ports.remove(name).unwrap_or_default();
        let mapping = self.copy_body(parent, name, child, &bindings.inputs, last_id)?;
        if let Some(reset) = &child.reset {
            let port = mapping
                .get(reset.port.index)
                .copied()
                .ok_or_else(|| format!("block_inline: invalid reset port in '{}'", child.name))?;
            self.resets.push((
                name,
                BlockReset {
                    port,
                    ..reset.clone()
                },
            ));
        }
        let mut outputs = HashMap::new();
        for port in child.output_ports() {
            let Some(NodePayload::OutputPort { name, arg, .. }) =
                child.nodes.get(port.index).map(|n| &n.payload)
            else {
                return Err(format!(
                    "block_inline: invalid output port in '{}'",
                    child.name
                ));
            };
            if outputs.insert(name.as_str(), mapping[arg.index]).is_some() {
                return Err(format!("block_inline: duplicate output '{name}'"));
            }
        }
        for (port, target) in bindings.outputs {
            let &driver = outputs.get(port).ok_or_else(|| {
                format!("block_inline: instantiation '{name}' missing output '{port}'")
            })?;
            if parent.get_node_ty(driver) != parent.get_node_ty(target) {
                return Err(format!(
                    "block_inline: instantiation '{name}' has a type mismatch on output '{port}'"
                ));
            }
            self.aliases[target.index] = Some(driver);
        }
        Ok(())
    }

    /// Copies child nodes and registers, using the parent bindings for formals.
    fn copy_body(
        &mut self,
        parent: &mut Block,
        name: &str,
        child: &Block,
        inputs: &HashMap<&str, NodeRef>,
        last_id: &mut usize,
    ) -> Result<Vec<NodeRef>, String> {
        if !child.registers.is_empty() && parent.clock_port_name().is_none() {
            return Err(format!(
                "block_inline: instantiation '{name}' has registers but the parent has no clock port"
            ));
        }
        let mut registers = HashMap::new();
        for reg in &child.registers {
            let scoped =
                uniquify_node_name(&format!("{name}__{}", reg.name), &mut self.register_names);
            registers.insert(reg.name.as_str(), scoped.clone());
            parent.registers.push(Register {
                name: scoped,
                ..reg.clone()
            });
        }

        let mapping = destinations(child, name, inputs, parent.nodes.len())?;
        for node in &child.nodes {
            let (payload, local_name) = match &node.payload {
                NodePayload::Nil | NodePayload::InputPort { .. } => continue,
                // Preserve the unit value of output sinks independently of the
                // data value consumed outside the instance.
                NodePayload::OutputPort { .. } => (NodePayload::Tuple(vec![]), None),
                NodePayload::InstantiationInput { .. }
                | NodePayload::InstantiationOutput { .. } => {
                    return Err(format!(
                        "block_inline: child '{}' still has instance nodes",
                        child.name
                    ));
                }
                payload => {
                    // Each source definition passed operand bounds checking
                    // before wiring.
                    let mut mapped = remap_payload_with(payload, |(_, arg)| mapping[arg.index]);
                    if let NodePayload::RegisterRead { register }
                    | NodePayload::RegisterWrite { register, .. } = &mut mapped
                    {
                        *register = registers
                            .get(register.as_str())
                            .ok_or_else(|| {
                                format!(
                                    "block_inline: undeclared register '{register}' in '{}'",
                                    child.name
                                )
                            })?
                            .clone();
                    }
                    let local_name = node
                        .assigned_name()
                        .map(|n| uniquify_node_name(&format!("{name}__{n}"), &mut self.node_names));
                    (mapped, local_name)
                }
            };
            *last_id = last_id
                .checked_add(1)
                .ok_or("block_inline: node IDs overflow usize")?;
            parent.nodes.push(Node {
                text_id: *last_id,
                name: local_name,
                ty: node.ty.clone(),
                payload,
                pos: node.pos.clone(),
            });
        }
        Ok(mapping)
    }

    /// Collapses chains of directly wired ports, then remaps every operand
    /// once.
    pub(super) fn finish(mut self, parent: &mut Block) -> Result<(), String> {
        if let Some(name) = self.ports.keys().min() {
            return Err(format!(
                "block_inline: connections reference unknown instance '{name}'"
            ));
        }
        self.collapse_aliases(parent);
        let driver = |arg: NodeRef| {
            self.aliases
                .get(arg.index)
                .copied()
                .flatten()
                .unwrap_or(arg)
        };
        for (i, node) in parent.nodes.iter_mut().enumerate() {
            if self.aliases.get(i).is_some_and(Option::is_some) {
                node.payload = NodePayload::Nil;
            } else if matches!(node.payload, NodePayload::InstantiationInput { .. }) {
                // An ordinary user of a connection sink still sees its unit
                // value.
                node.replace_payload(NodePayload::Tuple(vec![]));
            } else {
                node.payload = remap_payload_with(&node.payload, |(_, arg)| driver(arg));
            }
        }

        let inputs: HashSet<_> = if self.resets.is_empty() {
            HashSet::new()
        } else {
            parent.input_ports().collect()
        };
        for (instance, mut reset) in self.resets {
            reset.port = driver(reset.port);
            if !inputs.contains(&reset.port) {
                return Err(format!(
                    "block_inline: instantiation '{instance}' reset must connect directly to a parent input port"
                ));
            }
            if parent
                .reset
                .as_ref()
                .is_some_and(|existing| existing != &reset)
            {
                return Err(format!(
                    "block_inline: instantiation '{instance}' has an incompatible reset port, polarity, or timing"
                ));
            }
            parent.reset = Some(reset);
        }
        parent.instantiations.clear();
        Ok(())
    }

    /// Path-compresses whole-port aliases. A closed chain of empty data has
    /// one possible value. Other loops keep a self-edge for cycle diagnosis.
    fn collapse_aliases(&mut self, parent: &mut Block) {
        let mut state = vec![0u8; self.aliases.len()];
        let mut path = Vec::new();
        for start in 0..self.aliases.len() {
            if self.aliases[start].is_none() || state[start] != 0 {
                continue;
            }
            let mut node = NodeRef { index: start };
            while let Some(driver) = self.aliases.get(node.index).copied().flatten() {
                match state[node.index] {
                    0 => {
                        state[node.index] = 1;
                        path.push(node);
                        node = driver;
                    }
                    1 => {
                        let root = parent.get_node_mut(node);
                        let payload = if is_empty_data(&root.ty) {
                            NodePayload::Literal(zero_ir_value_for_type(&root.ty))
                        } else {
                            NodePayload::Unop(Unop::Identity, node)
                        };
                        root.replace_payload(payload);
                        break;
                    }
                    _ => {
                        node = driver;
                        break;
                    }
                }
            }
            for alias in path.drain(..) {
                state[alias.index] = 2;
                self.aliases[alias.index] = (alias != node).then_some(node);
            }
        }
    }
}

/// Binds inputs and reserves all other destinations before remapping operands.
/// This also permits forward references in an as-yet unspecialized child.
fn destinations(
    child: &Block,
    name: &str,
    inputs: &HashMap<&str, NodeRef>,
    mut next: usize,
) -> Result<Vec<NodeRef>, String> {
    let mut mapping = Vec::with_capacity(child.nodes.len());
    for node in &child.nodes {
        let dest = match &node.payload {
            NodePayload::InputPort { name: port, .. } => {
                *inputs.get(port.as_str()).ok_or_else(|| {
                    format!("block_inline: instantiation '{name}' missing input '{port}'")
                })?
            }
            NodePayload::Nil => NodeRef { index: 0 },
            _ => {
                let dest = NodeRef { index: next };
                next += 1;
                dest
            }
        };
        mapping.push(dest);
    }
    Ok(mapping)
}
