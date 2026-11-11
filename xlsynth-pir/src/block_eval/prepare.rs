// SPDX-License-Identifier: Apache-2.0

//! Elaborate one validated block hierarchy into a reusable combinational PIR
//! function.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::IrValue;
use crate::ir::{
    Block, Fn, InstantiationKind, Node, NodeGraph, NodePayload, NodeRef, Package, PackageMember,
    Type, Unop,
};
use crate::ir_utils::{compact_and_toposort_with_mapping_in_place, remap_payload_with};

use super::BlockEvalField;

/// One function and the metadata needed to translate between its tuple and a
/// block cycle.
pub(super) struct Prepared {
    pub(super) function: Fn,
    pub(super) inputs: Vec<BlockEvalField>,
    pub(super) outputs: Vec<BlockEvalField>,
    pub(super) registers: Vec<BlockEvalField>,
    pub(super) updates: Vec<RegisterUpdate>,
    pub(super) origins: Vec<Option<NodeOrigin>>,
}

/// Original site of a projected node, including its instance in the block
/// hierarchy.
#[derive(Clone, Debug)]
pub(super) struct NodeOrigin {
    pub(super) instance: Arc<str>,
    pub(super) node_ref: NodeRef,
    pub(super) text_id: usize,
}

/// All writes to one register and its shared reset, expressed as cycle result
/// tuple indices.
#[derive(Default)]
pub(super) struct RegisterUpdate {
    pub(super) reset: Option<(usize, bool, IrValue)>,
    pub(super) writes: Vec<Write>,
}

/// The data and optional enable of a register write in the cycle result tuple.
pub(super) struct Write {
    pub(super) data: usize,
    pub(super) enable: Option<usize>,
    pub(super) node_id: usize,
}

/// One occurrence of a block with offsets for its private copy of nodes and
/// register state.
struct Instance<'a> {
    block: &'a Block,
    path: String,
    parent: Option<(usize, &'a str)>,
    children: HashMap<&'a str, usize>,
    node_base: usize,
    register_base: usize,
    register_index: HashMap<&'a str, usize>,
    child_inputs: HashMap<(&'a str, &'a str), NodeRef>,
    output_data: HashMap<&'a str, NodeRef>,
}

impl Instance<'_> {
    /// Returns the projected function reference for a node of this instance
    /// before sorting.
    fn projected(&self, node: NodeRef) -> NodeRef {
        NodeRef {
            index: self.node_base + node.index,
        }
    }
}

/// Escaping each segment lets state files distinguish a literal dot from a
/// hierarchy separator.
fn qualified(path: &str, segment: &str) -> String {
    let segment = segment.replace('$', "$$").replace('.', "$d");
    if path.is_empty() {
        segment
    } else {
        format!("{path}.{segment}")
    }
}

/// Enumerates reachable instances in declaration preorder, allocating
/// independent register slots.
fn elaborate<'a>(package: &'a Package, top: &'a Block) -> Result<Vec<Instance<'a>>, String> {
    let blocks: HashMap<_, _> = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Block(block) => Some((block.name.as_str(), block)),
            PackageMember::Function(_) => None,
        })
        .collect();
    let mut instances: Vec<Instance<'a>> = Vec::new();
    let mut pending = vec![(top, String::new(), None::<(usize, &'a str)>)];
    let mut node_base = 1;
    let mut register_base = 0;
    while let Some((block, path, parent)) = pending.pop() {
        let index = instances.len();
        if let Some((parent_index, name)) = parent {
            instances[parent_index].children.insert(name, index);
        }
        if block.reset.as_ref().is_some_and(|reset| reset.asynchronous) {
            return Err(format!(
                "block '{}' at instance '{path}' has an unsupported asynchronous reset",
                block.name
            ));
        }
        let clock = block.clock_port_name();
        for inst in block.instantiations.iter().rev() {
            let child_path = qualified(&path, &inst.name);
            if inst.kind == InstantiationKind::Extern {
                return Err(format!(
                    "unsupported extern instance '{child_path}' in block '{}'",
                    block.name
                ));
            }
            let child = blocks.get(inst.block.as_str()).ok_or_else(|| {
                format!(
                    "instance '{child_path}' references missing block '{}'",
                    inst.block
                )
            })?;
            if clock.is_none() && child.clock_port_name().is_some() {
                return Err(format!(
                    "clocked instance '{child_path}' requires a clock on parent block '{}'",
                    block.name
                ));
            }
            // The package verifier requires a child to precede its parent,
            // ruling out recursion.
            pending.push((*child, child_path, Some((index, inst.name.as_str()))));
        }
        let register_index = block
            .registers
            .iter()
            .enumerate()
            .map(|(i, r)| (r.name.as_str(), i))
            .collect();
        let child_inputs = block
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::InstantiationInput {
                    instantiation,
                    port_name,
                    arg,
                } => Some(((instantiation.as_str(), port_name.as_str()), *arg)),
                _ => None,
            })
            .collect();
        let output_data = block
            .output_ports()
            .map(|port| (block.port_name(port), block.output_value(port)))
            .collect();
        instances.push(Instance {
            block,
            path,
            parent,
            children: HashMap::new(),
            node_base,
            register_base,
            register_index,
            child_inputs,
            output_data,
        });
        node_base += block.nodes.len();
        register_base += block.registers.len();
    }
    Ok(instances)
}

/// Tuple elements 0..output_count are visible outputs; subsequent elements
/// drive register writes.
struct CycleProjection<'a, 'pkg> {
    instances: &'a [Instance<'pkg>],
    top_inputs: HashMap<&'pkg str, NodeRef>,
    register_params: usize,
    result: Vec<NodeRef>,
    updates: Vec<RegisterUpdate>,
}

impl CycleProjection<'_, '_> {
    /// Projects a wire into the return tuple and gives the interpreter its
    /// tuple position.
    fn project_result(&mut self, value: NodeRef) -> usize {
        let index = self.result.len();
        self.result.push(value);
        index
    }

    /// Resolves hierarchy connections as data dependencies, leaving Q reads
    /// independent of D.
    fn lower(&mut self, instance_index: usize, node: &Node) -> Result<NodePayload, String> {
        let inst = &self.instances[instance_index];
        match &node.payload {
            NodePayload::InputPort { name, .. } => {
                let value = if let Some((parent_index, child_name)) = inst.parent {
                    let parent = &self.instances[parent_index];
                    let arg = parent
                        .child_inputs
                        .get(&(child_name, name))
                        .ok_or_else(|| {
                            format!(
                                "missing connection to input '{name}' of instance '{}'",
                                inst.path
                            )
                        })?;
                    parent.projected(*arg)
                } else {
                    self.top_inputs[name.as_str()]
                };
                Ok(NodePayload::Unop(Unop::Identity, value))
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } => {
                let child = &self.instances[inst.children[instantiation.as_str()]];
                let arg = child.output_data[port_name.as_str()];
                Ok(NodePayload::Unop(Unop::Identity, child.projected(arg)))
            }
            NodePayload::RegisterRead { register } => {
                let index = inst.register_base + inst.register_index[register.as_str()];
                Ok(NodePayload::Unop(
                    Unop::Identity,
                    NodeRef {
                        index: self.register_params + index,
                    },
                ))
            }
            NodePayload::RegisterWrite {
                arg,
                register,
                load_enable,
                reset,
            } => {
                let local_index = inst.register_index[register.as_str()];
                let reg_index = inst.register_base + local_index;
                if self.updates[reg_index].reset.is_none()
                    && let Some(reset) = reset
                {
                    let reset_bit = self.project_result(inst.projected(*reset));
                    let metadata = inst
                        .block
                        .reset
                        .as_ref()
                        .ok_or("register write lacks block reset metadata")?;
                    let value = inst.block.registers[local_index]
                        .reset_value
                        .clone()
                        .ok_or("register write lacks reset value")?;
                    self.updates[reg_index].reset = Some((reset_bit, metadata.active_low, value));
                }
                let data = self.project_result(inst.projected(*arg));
                let enable = load_enable.map(|e| self.project_result(inst.projected(e)));
                self.updates[reg_index].writes.push(Write {
                    data,
                    enable,
                    node_id: node.text_id,
                });
                Ok(NodePayload::Tuple(vec![]))
            }
            // Wiring and register writes have unit graph values; their data
            // edges are represented by outputs, child input aliases or register
            // update records.
            NodePayload::OutputPort { .. }
            | NodePayload::InstantiationInput { .. }
            | NodePayload::Nil => Ok(NodePayload::Tuple(vec![])),
            NodePayload::Assert {
                token,
                activate,
                message,
                label,
            } => {
                let site = if label.is_empty() {
                    format!("node {}", node.text_id)
                } else {
                    label.clone()
                };
                Ok(NodePayload::Assert {
                    token: inst.projected(*token),
                    activate: inst.projected(*activate),
                    message: message.clone(),
                    label: if inst.path.is_empty() {
                        site
                    } else {
                        format!("{}.{site}", inst.path)
                    },
                })
            }
            payload => Ok(remap_payload_with(payload, |(_, dep)| inst.projected(dep))),
        }
    }
}

/// Prepares a single top block from a package validated by the public
/// constructor.
pub(super) fn prepare(package: &Package, top: &str) -> Result<Prepared, String> {
    let block = package
        .get_block(top)
        .ok_or_else(|| format!("block '{top}' not found"))?;
    let instances = elaborate(package, block)?;
    let fields = |port| BlockEvalField {
        name: block.port_name(port).to_owned(),
        ty: block.port_type(port).clone(),
    };
    let inputs: Vec<_> = block.input_ports().map(fields).collect();
    let outputs: Vec<_> = block.output_ports().map(fields).collect();
    let registers: Vec<_> = instances
        .iter()
        .flat_map(|inst| {
            inst.block.registers.iter().map(|reg| BlockEvalField {
                name: qualified(&inst.path, &reg.name),
                ty: reg.ty.clone(),
            })
        })
        .collect();
    let param_base = 1 + instances
        .iter()
        .map(|inst| inst.block.nodes.len())
        .sum::<usize>();
    let mut projection = CycleProjection {
        instances: &instances,
        top_inputs: block
            .input_ports()
            .enumerate()
            .map(|(i, port)| {
                (
                    block.port_name(port),
                    NodeRef {
                        index: param_base + i,
                    },
                )
            })
            .collect(),
        register_params: param_base + inputs.len(),
        result: block
            .output_ports()
            .map(|port| instances[0].projected(block.output_value(port)))
            .collect(),
        updates: (0..registers.len())
            .map(|_| RegisterUpdate::default())
            .collect(),
    };
    let mut graph = NodeGraph::new(top);
    let mut origins = vec![None];
    // Projected text IDs must never collide with real function callees, whose
    // observers use their own node-index space and whose cover counters are
    // keyed by text ID.
    let used_ids: HashSet<_> = package
        .members
        .iter()
        .flat_map(|member| member.graph().nodes.iter().map(|node| node.text_id))
        .collect();
    let mut ids = (1..=usize::MAX).filter(|id| !used_ids.contains(id));
    let mut fresh_id = || ids.next().ok_or("exhausted PIR node IDs");
    for (index, inst) in instances.iter().enumerate() {
        let path: Arc<str> = Arc::from(inst.path.as_str());
        for (source_index, node) in inst.block.nodes.iter().enumerate() {
            graph.nodes.push(Node {
                name: None,
                ty: node.ty.clone(),
                text_id: fresh_id()?,
                pos: node.pos.clone(),
                // Rewrites may leave removed nodes with their original
                // annotated types. Preserve those tombstones until compaction
                // checks and removes them.
                payload: if source_index != 0 && matches!(node.payload, NodePayload::Nil) {
                    NodePayload::Nil
                } else {
                    projection.lower(index, node)?
                },
            });
            origins.push(Some(NodeOrigin {
                instance: Arc::clone(&path),
                node_ref: NodeRef {
                    index: source_index,
                },
                text_id: node.text_id,
            }));
        }
    }
    let mut params = Vec::new();
    for (index, field) in inputs.iter().chain(&registers).enumerate() {
        params.push(NodeRef {
            index: graph.nodes.len(),
        });
        graph.nodes.push(Node {
            name: Some(format!("p{index}")),
            ty: field.ty.clone(),
            text_id: fresh_id()?,
            pos: None,
            payload: NodePayload::Param,
        });
        origins.push(None);
    }
    let ret_ty = Type::Tuple(
        projection
            .result
            .iter()
            .map(|r| Box::new(graph.get_node_ty(*r).clone()))
            .collect(),
    );
    let ret_node_ref = Some(NodeRef {
        index: graph.nodes.len(),
    });
    graph.nodes.push(Node {
        name: None,
        ty: ret_ty.clone(),
        payload: NodePayload::Tuple(projection.result),
        text_id: fresh_id()?,
        pos: None,
    });
    origins.push(None);
    let mut function = Fn {
        graph,
        params,
        ret_ty,
        ret_node_ref,
    };
    // A parent and a child can each be acyclic alone, but their wiring can
    // close a combinational loop. This also schedules reused instances for a
    // single pass per edge.
    let mapping = compact_and_toposort_with_mapping_in_place(&mut function)
        .map_err(|e| format!("cannot schedule block '{top}': {e}"))?;
    let mut sorted = vec![None; function.nodes.len()];
    for (origin, new_ref) in origins.into_iter().zip(mapping) {
        if let Some(new_ref) = new_ref {
            sorted[new_ref.index] = origin;
        }
    }
    Ok(Prepared {
        function,
        inputs,
        outputs,
        registers,
        updates: projection.updates,
        origins: sorted,
    })
}

#[cfg(test)]
mod tests {
    use crate::IrValue;
    use crate::block_eval::{BlockEvalError, BlockEvaluator, RegisterInit};
    use crate::ir_parser::Parser;

    #[test]
    fn instance_paths_distinguish_literal_separators_and_attribute_child_failures() {
        let ir = r#"package test

block leaf(clk: clock) {
  reg q(bits[1])
  current: bits[1] = register_read(register=q, id=1)
  hold: () = register_write(current, register=q, id=2)
  start: token = after_all(id=3)
  check: token = assert(start, current, message="q must be one", id=4)
}

block nested(clk: clock) {
  instantiation b(block=leaf, kind=block)
}

block unrelated(clk: clock, rst: bits[1]) {
  #![reset(port="rst", asynchronous=true, active_low=false)]
  rst: bits[1] = input_port(name=rst, id=5)
}

top block dut(clk: clock) {
  instantiation a(block=nested, kind=block)
  instantiation direct(block=leaf, kind=block)
  instantiation escaped(block=leaf, kind=block)
}
"#;
        let mut pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        // Direct Rust producers can use punctuation in these names.
        let dut = pkg.get_block_mut("dut").unwrap();
        dut.instantiations[1].name = "a.b".to_string();
        dut.instantiations[2].name = "a$db".to_string();

        let evaluator = BlockEvaluator::prepare(&pkg, "dut").unwrap();
        let names: Vec<_> = evaluator
            .registers()
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(names, ["a.b.q", "a$db.q", "a$$db.q"]);
        let state = evaluator
            .initial_state(RegisterInit::Supplied(vec![
                IrValue::bool(true),
                IrValue::bool(false),
                IrValue::bool(true),
            ]))
            .unwrap();
        let BlockEvalError::Runtime(failure) = evaluator.step(&state, &[]).unwrap_err() else {
            panic!("expected an assertion failure in the second instance")
        };
        assert_eq!(failure.assertion_failures.len(), 1);
        assert_eq!(failure.assertion_failures[0].label, "a$db.node 4");
    }

    #[test]
    fn unsupported_instances_only_fail_when_reachable_from_the_selected_top() {
        let ir = r#"package test
fn stub() -> bits[1] {
  ret zero: bits[1] = literal(value=0, id=1)
}
block foreign() {
  instantiation external(foreign_function=stub, kind=extern)
}
block clocked(clk: clock) {
}
block missing_clock() {
  instantiation child(block=clocked, kind=block)
}
top block empty() {
}
"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        BlockEvaluator::prepare(&pkg, "empty").unwrap();
        for (top, expected) in [
            (
                "foreign",
                "unsupported extern instance 'external' in block 'foreign'",
            ),
            (
                "missing_clock",
                "clocked instance 'child' requires a clock on parent block 'missing_clock'",
            ),
        ] {
            match BlockEvaluator::prepare(&pkg, top) {
                Err(BlockEvalError::Preparation(actual)) => assert_eq!(actual, expected),
                _ => panic!("expected a preparation error for {top}"),
            }
        }
    }

    #[test]
    fn hierarchy_feedback_uses_old_q_and_each_blocks_reset_polarity() {
        let ir = r#"package test

block leaf(clk: clock, rst_n: bits[1], d: bits[1], out: bits[1]) {
  #![reset(port="rst_n", asynchronous=false, active_low=true)]
  reg q(bits[1], reset_value=1)
  rst_n: bits[1] = input_port(name=rst_n, id=1)
  d: bits[1] = input_port(name=d, id=2)
  old: bits[1] = register_read(register=q, id=3)
  w: () = register_write(d, register=q, reset=rst_n, id=4)
  out: () = output_port(old, name=out, id=5)
}

top block feedback(clk: clock, rst: bits[1], out: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg q(bits[1], reset_value=0)
  instantiation child(block=leaf, kind=block)
  rst: bits[1] = input_port(name=rst, id=6)
  q1: bits[1] = register_read(register=q, id=7)
  q2: bits[1] = instantiation_output(instantiation=child, port_name=out, id=8)
  invert: bits[1] = not(q2, id=9)
  w1: () = register_write(q2, register=q, reset=rst, id=10)
  w2: () = instantiation_input(invert, instantiation=child, port_name=d, id=11)
  connect_reset: () = instantiation_input(rst, instantiation=child, port_name=rst_n, id=12)
  out: () = output_port(q1, name=out, id=13)
}
"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let evaluator = BlockEvaluator::prepare(&pkg, "feedback").unwrap();
        let low = IrValue::bool(false);
        let high = IrValue::bool(true);
        let start = evaluator
            .initial_state(RegisterInit::Supplied(vec![low.clone(), high.clone()]))
            .unwrap();
        let reset_child = evaluator.step(&start, std::slice::from_ref(&low)).unwrap();
        assert_eq!(reset_child.outputs, [low.clone()]);
        assert_eq!(
            reset_child.next_state.values(),
            [high.clone(), high.clone()]
        );
        let reset_parent = evaluator
            .step(&reset_child.next_state, std::slice::from_ref(&high))
            .unwrap();
        assert_eq!(reset_parent.outputs, [high]);
        assert_eq!(reset_parent.next_state.values(), [low.clone(), low]);

        // Bypassing the child's register closes a cycle through the parent's
        // inverter.
        let invalid = ir.replace("output_port(old, name=out", "output_port(d, name=out");
        let pkg = Parser::new(&invalid).parse_and_validate_package().unwrap();
        match BlockEvaluator::prepare(&pkg, "feedback") {
            Err(BlockEvalError::Preparation(error)) => {
                assert!(error.contains("cycle detected"), "{error}")
            }
            _ => panic!("expected a combinational cycle during preparation"),
        }
    }
}
