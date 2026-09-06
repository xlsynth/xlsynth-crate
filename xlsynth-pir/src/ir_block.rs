// SPDX-License-Identifier: Apache-2.0

//! Blocks have graph-resident data ports and explicit sequential resources.

use std::ops::{Deref, DerefMut};

use crate::ir::{Fn, Instantiation, Node, NodeGraph, NodePayload, NodeRef, Register, Type};

/// One port in declaration order; a clock is not a graph value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockPort {
    Input(NodeRef),
    Output(NodeRef),
    Clock(String),
}

/// Reset behavior attached to a block's one-bit input port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockReset {
    pub port: NodeRef,
    pub asynchronous: bool,
    pub active_low: bool,
}

/// A hardware block, with no function parameters or synthetic return value.
///
/// Input and output names, IDs, annotations, and data dependencies live on
/// their actual graph nodes. `ports` records their declaration order alongside
/// the optional clock; it does not duplicate the data-port payloads.
#[derive(Debug, Clone)]
pub struct Block {
    pub graph: NodeGraph,
    pub ports: Vec<BlockPort>,
    pub reset: Option<BlockReset>,
    pub registers: Vec<Register>,
    pub instantiations: Vec<Instantiation>,
}

impl Deref for Block {
    type Target = NodeGraph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Block {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl Block {
    /// Converts a function interface into actual block input and output nodes.
    ///
    /// Without explicit names, the entire return value becomes one `out` port,
    /// including tuple values. One explicit name has the same behavior;
    /// multiple names split the return tuple, and an empty list requires a
    /// unit return. Package-dependent calls are retained and should be
    /// verified in the destination package. No sequential resources are
    /// inferred.
    pub fn from_function(function: Fn, output_names: Option<&[String]>) -> Result<Self, String> {
        function.check_pir_layout_invariants()?;
        crate::ir_verify::verify_graph_unique_node_ids(&function)?;
        crate::ir_verify::verify_graph_operand_indices_in_bounds(&function)?;
        crate::ir_utils::verify_no_cycle(&function)?;
        if function.nodes.iter().any(|node| {
            matches!(
                node.payload,
                NodePayload::InputPort { .. } | NodePayload::OutputPort { .. }
            )
        }) {
            return Err("function already contains block port nodes".to_string());
        }
        let return_ref = function
            .ret_node_ref
            .ok_or("function has no return value")?;
        let return_node = function
            .nodes
            .get(return_ref.index)
            .filter(|node| !matches!(node.payload, NodePayload::Nil))
            .ok_or("function return references an invalid node")?;
        if return_node.ty != function.ret_ty {
            return Err("function return type disagrees with its return node".to_string());
        }
        let names = output_names
            .map(<[String]>::to_vec)
            .unwrap_or_else(|| vec!["out".to_string()]);
        if names.len() != 1 {
            match &function.ret_ty {
                Type::Tuple(elements) if elements.len() == names.len() => {}
                _ => {
                    return Err(format!(
                        "{} output names require a return tuple of that arity",
                        names.len()
                    ));
                }
            }
        }
        let mut block = Self {
            graph: function.graph,
            ports: Vec::new(),
            reset: None,
            registers: Vec::new(),
            instantiations: Vec::new(),
        };
        for (index, param) in function.params.into_iter().enumerate() {
            let node_ref = NodeRef { index: index + 1 };
            let node = block.get_node_mut(node_ref);
            if node.ty != param.ty || node.name.as_deref() != Some(param.name.as_str()) {
                return Err(format!(
                    "parameter '{}' disagrees with its graph node",
                    param.name
                ));
            }
            node.payload = NodePayload::InputPort {
                name: param.name,
                sv_type: None,
            };
            block.ports.push(BlockPort::Input(node_ref));
        }
        if block.nodes.iter().any(|node| {
            matches!(
                node.payload,
                NodePayload::GetParam(_)
                    | NodePayload::RegisterRead { .. }
                    | NodePayload::RegisterWrite { .. }
                    | NodePayload::InstantiationInput { .. }
                    | NodePayload::InstantiationOutput { .. }
                    | NodePayload::OutputPort { .. }
            )
        }) {
            return Err(
                "function contains nodes incompatible with a combinational block interface"
                    .to_string(),
            );
        }
        let outputs = if names.len() == 1 {
            vec![return_ref]
        } else if let NodePayload::Tuple(elements) = &block.get_node(return_ref).payload {
            let Type::Tuple(types) = &function.ret_ty else {
                unreachable!()
            };
            if elements.len() != types.len()
                || elements
                    .iter()
                    .zip(types)
                    .any(|(element, ty)| block.get_node_ty(*element) != ty.as_ref())
            {
                return Err("return tuple operands disagree with its type".to_string());
            }
            elements.clone()
        } else {
            let Type::Tuple(types) = &function.ret_ty else {
                unreachable!()
            };
            let types = types.clone();
            let mut outputs = Vec::with_capacity(types.len());
            for (index, ty) in types.into_iter().enumerate() {
                let node_ref = NodeRef {
                    index: block.nodes.len(),
                };
                let text_id = block.next_node_id()?;
                block.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: *ty,
                    payload: NodePayload::TupleIndex {
                        tuple: return_ref,
                        index,
                    },
                    pos: None,
                });
                outputs.push(node_ref);
            }
            outputs
        };
        for (name, source) in names.iter().zip(outputs) {
            block.add_output_port(name, source)?;
        }
        // A tuple used only to bundle multiple function outputs is not part of
        // the block interface. Keep it if another operation still needs it.
        if names.len() != 1
            && matches!(block.get_node(return_ref).payload, NodePayload::Tuple(_))
            && !block
                .nodes
                .iter()
                .any(|node| crate::ir_utils::operands(&node.payload).contains(&return_ref))
        {
            block.get_node_mut(return_ref).payload = NodePayload::Nil;
            block.compact_and_toposort()?;
        }
        Ok(block)
    }

    /// Creates an empty block with the graph's reserved Nil sentinel.
    pub fn new(name: &str) -> Self {
        Self {
            graph: NodeGraph::new(name),
            ports: Vec::new(),
            reset: None,
            registers: Vec::new(),
            instantiations: Vec::new(),
        }
    }

    /// Iterates input nodes in port declaration order.
    pub fn input_ports(&self) -> impl Iterator<Item = NodeRef> + '_ {
        self.ports.iter().filter_map(|port| match port {
            BlockPort::Input(node) => Some(*node),
            _ => None,
        })
    }

    /// Iterates output nodes in port declaration order.
    pub fn output_ports(&self) -> impl Iterator<Item = NodeRef> + '_ {
        self.ports.iter().filter_map(|port| match port {
            BlockPort::Output(node) => Some(*node),
            _ => None,
        })
    }

    /// Returns the optional clock's declared name.
    pub fn clock_port_name(&self) -> Option<&str> {
        self.ports.iter().find_map(|port| match port {
            BlockPort::Clock(name) => Some(name.as_str()),
            _ => None,
        })
    }

    /// Returns the semantic name of an input or output node.
    pub fn port_name(&self, port: NodeRef) -> &str {
        match &self.get_node(port).payload {
            NodePayload::InputPort { name, .. } | NodePayload::OutputPort { name, .. } => name,
            _ => panic!("node {} is not a data port", port.index),
        }
    }

    /// Returns the external data type; an output node itself has type `()`.
    pub fn port_type(&self, port: NodeRef) -> &Type {
        let node = self.get_node(port);
        match node.payload {
            NodePayload::InputPort { .. } => &node.ty,
            NodePayload::OutputPort { arg, .. } => self.get_node_ty(arg),
            _ => panic!("node {} is not a data port", port.index),
        }
    }

    /// Returns the data value driving an output port.
    pub fn output_value(&self, port: NodeRef) -> NodeRef {
        match self.get_node(port).payload {
            NodePayload::OutputPort { arg, .. } => arg,
            _ => panic!("node {} is not an output port", port.index),
        }
    }

    /// Finds an input port by its semantic name.
    pub fn get_input_port(&self, name: &str) -> Option<NodeRef> {
        self.input_ports()
            .find(|port| self.port_name(*port) == name)
    }

    /// Finds an output port by its semantic name.
    pub fn get_output_port(&self, name: &str) -> Option<NodeRef> {
        self.output_ports()
            .find(|port| self.port_name(*port) == name)
    }

    fn check_new_port_name(&self, name: &str) -> Result<(), String> {
        if !crate::ir_builder::is_valid_identifier(name) {
            return Err(format!("invalid block port name {name:?}"));
        }
        if self.clock_port_name() == Some(name)
            || self.nodes.iter().any(|node| {
                node.name.as_deref() == Some(name)
                    || matches!(&node.payload,
                        NodePayload::InputPort { name: existing, .. }
                        | NodePayload::OutputPort { name: existing, .. } if existing == name)
            })
        {
            return Err(format!("duplicate block port or node name {name:?}"));
        }
        Ok(())
    }

    fn next_node_id(&self) -> Result<usize, String> {
        self.nodes
            .iter()
            .map(|node| node.text_id)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| "block node IDs overflow usize".to_string())
    }

    /// Appends an input node and records it at the end of the port order.
    pub fn add_input_port(&mut self, name: &str, ty: Type) -> Result<NodeRef, String> {
        self.check_new_port_name(name)?;
        ty.checked_bit_count()
            .ok_or_else(|| "input type width overflows usize".to_string())?;
        let text_id = self.next_node_id()?;
        let port = NodeRef {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            text_id,
            name: Some(name.to_string()),
            ty,
            payload: NodePayload::InputPort {
                name: name.to_string(),
                sv_type: None,
            },
            pos: None,
        });
        self.ports.push(BlockPort::Input(port));
        Ok(port)
    }

    /// Appends an output sink; its value is never flattened into multiple
    /// ports.
    pub fn add_output_port(&mut self, name: &str, value: NodeRef) -> Result<NodeRef, String> {
        self.check_new_port_name(name)?;
        if self
            .nodes
            .get(value.index)
            .is_none_or(|node| matches!(node.payload, NodePayload::Nil))
        {
            return Err(format!("output source node {} is invalid", value.index));
        }
        let text_id = self.next_node_id()?;
        let port = NodeRef {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            text_id,
            name: Some(name.to_string()),
            ty: Type::nil(),
            payload: NodePayload::OutputPort {
                name: name.to_string(),
                arg: value,
                sv_type: None,
            },
            pos: None,
        });
        self.ports.push(BlockPort::Output(port));
        Ok(port)
    }

    /// Adds the single clock declaration without creating a data node.
    pub fn add_clock_port(&mut self, name: &str) -> Result<(), String> {
        self.check_new_port_name(name)?;
        if self.clock_port_name().is_some() {
            return Err("a block may have only one clock port".to_string());
        }
        self.ports.push(BlockPort::Clock(name.to_string()));
        Ok(())
    }

    /// Compacts and topologically orders the graph together with its interface.
    ///
    /// Missing or deleted interface nodes are rejected before mutation. On
    /// success, ordered ports and reset references follow the reordered nodes.
    pub fn compact_and_toposort(&mut self) -> Result<(), String> {
        let port_nodes = self.ports.iter().filter_map(|port| match port {
            BlockPort::Input(node) | BlockPort::Output(node) => Some(*node),
            BlockPort::Clock(_) => None,
        });
        for node in port_nodes.chain(self.reset.iter().map(|reset| reset.port)) {
            if self
                .nodes
                .get(node.index)
                .is_none_or(|node| matches!(node.payload, NodePayload::Nil))
            {
                return Err(format!(
                    "block port or reset node {} is missing or deleted",
                    node.index
                ));
            }
        }
        let mapping =
            crate::ir_utils::compact_graph_and_toposort_with_mapping_in_place(&mut self.graph)?;
        // Every interface reference was checked above and compaction retains
        // all non-Nil nodes, so every reference has a mapping.
        self.remap_node_refs(&mapping)
    }

    /// Updates port/reset references after the graph has been compacted.
    ///
    /// A data port or reset input cannot be silently removed. Failure leaves
    /// the block's port order and reset descriptor unchanged.
    pub fn remap_node_refs(&mut self, mapping: &[Option<NodeRef>]) -> Result<(), String> {
        let remap = |node: NodeRef| {
            mapping
                .get(node.index)
                .copied()
                .flatten()
                .ok_or_else(|| format!("block port or reset node {} was removed", node.index))
        };
        let ports = self
            .ports
            .iter()
            .map(|port| match port {
                BlockPort::Input(node) => remap(*node).map(BlockPort::Input),
                BlockPort::Output(node) => remap(*node).map(BlockPort::Output),
                BlockPort::Clock(name) => Ok(BlockPort::Clock(name.clone())),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reset = self
            .reset
            .as_ref()
            .map(|reset| -> Result<BlockReset, String> {
                Ok(BlockReset {
                    port: remap(reset.port)?,
                    asynchronous: reset.asynchronous,
                    active_low: reset.active_low,
                })
            })
            .transpose()?;
        self.ports = ports;
        self.reset = reset;
        Ok(())
    }
}

impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::ir_parser::emit_block(self, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    #[test]
    fn function_conversion_has_explicit_aggregate_output_arity() {
        let function = Parser::new(
            r#"fn pair(data: (bits[8], bits[3]) id=1) -> (bits[8], bits[3]) {
  ret data: (bits[8], bits[3]) = param(name=data, id=1)
}"#,
        )
        .parse_fn()
        .unwrap();
        let aggregate = Block::from_function(function.clone(), None).unwrap();
        assert_eq!(aggregate.output_ports().count(), 1);
        assert_eq!(
            aggregate.to_string(),
            r#"block pair(data: (bits[8], bits[3]), out: (bits[8], bits[3])) {
  data: (bits[8], bits[3]) = input_port(name=data, id=1)
  out: () = output_port(data, name=out, id=2)
}"#
        );
        let split =
            Block::from_function(function, Some(&["low".to_string(), "high".to_string()])).unwrap();
        assert_eq!(
            split.to_string(),
            r#"block pair(data: (bits[8], bits[3]), low: bits[8], high: bits[3]) {
  data: (bits[8], bits[3]) = input_port(name=data, id=1)
  tuple_index.2: bits[8] = tuple_index(data, index=0, id=2)
  tuple_index.3: bits[3] = tuple_index(data, index=1, id=3)
  low: () = output_port(tuple_index.2, name=low, id=4)
  high: () = output_port(tuple_index.3, name=high, id=5)
}"#
        );
    }

    #[test]
    fn function_conversion_discards_only_an_unused_return_bundler() {
        let function = Parser::new(
            r#"fn pair(a: bits[8] id=1, b: bits[3] id=2) -> (bits[8], bits[3]) {
  ret tuple.3: (bits[8], bits[3]) = tuple(a, b, id=3)
}"#,
        )
        .parse_fn()
        .unwrap();
        let block =
            Block::from_function(function, Some(&["low".to_string(), "high".to_string()])).unwrap();
        assert_eq!(block.nodes.len(), 5);
        assert!(
            !block
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Tuple(_)))
        );
        let function = Parser::new("fn empty() -> () { ret tuple.1: () = tuple(id=1) }")
            .parse_fn()
            .unwrap();
        let block = Block::from_function(function, Some(&[])).unwrap();
        assert_eq!(block.nodes.len(), 1);
        assert_eq!(block.to_string(), "block empty() {\n}");
    }

    #[test]
    fn function_conversion_rejects_mismatched_output_arity_and_names() {
        let function = Parser::new("fn identity(data: bits[8] id=1) -> bits[8] { ret data: bits[8] = param(name=data, id=1) }").parse_fn().unwrap();
        assert!(Block::from_function(function.clone(), Some(&[])).is_err());
        assert!(
            Block::from_function(function.clone(), Some(&["a".to_string(), "b".to_string()]))
                .is_err()
        );
        assert!(Block::from_function(function, Some(&["data".to_string()])).is_err());
    }

    #[test]
    fn function_conversion_rejects_undeclared_parameters_and_block_only_nodes() {
        let function = Parser::new("fn identity(data: bits[8] id=1) -> bits[8] { ret data: bits[8] = param(name=data, id=1) }").parse_fn().unwrap();
        let input = NodeRef { index: 1 };
        for payload in [
            NodePayload::GetParam(crate::ir::ParamId::new(2)),
            NodePayload::InputPort {
                name: "extra".to_string(),
                sv_type: None,
            },
            NodePayload::OutputPort {
                name: "extra".to_string(),
                arg: input,
                sv_type: None,
            },
            NodePayload::RegisterRead {
                register: "value".to_string(),
            },
            NodePayload::RegisterWrite {
                register: "value".to_string(),
                arg: input,
                load_enable: None,
                reset: None,
            },
            NodePayload::InstantiationInput {
                instantiation: "child".to_string(),
                port_name: "data".to_string(),
                arg: input,
            },
            NodePayload::InstantiationOutput {
                instantiation: "child".to_string(),
                port_name: "result".to_string(),
            },
        ] {
            let mut invalid = function.clone();
            invalid.nodes.push(Node {
                text_id: 2,
                name: None,
                ty: Type::Bits(8),
                payload,
                pos: None,
            });
            assert!(Block::from_function(invalid, None).is_err());
        }
    }

    #[test]
    fn compaction_rejects_removed_ports_without_mutating_the_block() {
        let mut block = Block::new("b");
        let input = block.add_input_port("data", Type::Bits(8)).unwrap();
        let output = block.add_output_port("result", input).unwrap();
        block.get_node_mut(output).payload = NodePayload::Nil;
        let before = format!("{block:?}");
        assert_eq!(
            block.compact_and_toposort().unwrap_err(),
            "block port or reset node 2 is missing or deleted"
        );
        assert_eq!(format!("{block:?}"), before);
    }

    #[test]
    fn compaction_rejects_missing_reset_without_mutating_the_block() {
        let mut block = Block::new("b");
        block.add_input_port("rst", Type::Bits(1)).unwrap();
        block.reset = Some(BlockReset {
            port: NodeRef { index: 7 },
            asynchronous: false,
            active_low: false,
        });
        let before = format!("{block:?}");
        assert_eq!(
            block.compact_and_toposort().unwrap_err(),
            "block port or reset node 7 is missing or deleted"
        );
        assert_eq!(format!("{block:?}"), before);
    }

    #[test]
    fn data_ports_live_in_the_graph_and_clock_does_not() {
        let mut block = Block::new("passthrough");
        let input = block.add_input_port("data", Type::Bits(129)).unwrap();
        block.add_clock_port("clk").unwrap();
        let output = block.add_output_port("result", input).unwrap();
        assert_eq!(block.nodes.len(), 3);
        assert_eq!(
            block.ports,
            vec![
                BlockPort::Input(input),
                BlockPort::Clock("clk".to_string()),
                BlockPort::Output(output)
            ]
        );
        assert_eq!(block.get_node_ty(output), &Type::nil());
        assert_eq!(block.port_type(output), &Type::Bits(129));
        assert_eq!(block.output_value(output), input);
        assert_eq!(block.get_input_port("data"), Some(input));
        assert_eq!(block.get_output_port("result"), Some(output));
    }

    #[test]
    fn tuple_typed_output_is_one_port_without_synthetic_return() {
        let mut block = Block::new("aggregate");
        let ty = Type::Tuple(vec![Box::new(Type::Bits(8)), Box::new(Type::Bits(3))]);
        let input = block.add_input_port("data", ty.clone()).unwrap();
        let output = block.add_output_port("result", input).unwrap();
        assert_eq!(block.output_ports().count(), 1);
        assert_eq!(block.port_type(output), &ty);
        assert!(
            !block
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Tuple(_)))
        );
    }

    #[test]
    fn invalid_port_additions_and_remapping_leave_interface_unchanged() {
        let mut block = Block::new("checked");
        let input = block.add_input_port("rst", Type::Bits(1)).unwrap();
        block.reset = Some(BlockReset {
            port: input,
            asynchronous: false,
            active_low: true,
        });
        assert!(block.add_output_port("rst", input).is_err());
        assert!(block.add_output_port("out", NodeRef { index: 99 }).is_err());
        assert_eq!(block.nodes.len(), 2);
        let old_ports = block.ports.clone();
        let old_reset = block.reset.clone();
        assert!(block.remap_node_refs(&[None, None]).is_err());
        assert_eq!(block.ports, old_ports);
        assert_eq!(block.reset, old_reset);
    }
}
