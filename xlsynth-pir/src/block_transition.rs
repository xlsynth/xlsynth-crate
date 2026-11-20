// SPDX-License-Identifier: Apache-2.0

//! Word-level, rising-edge transitions for synchronous XLS blocks.

use std::collections::HashMap;

use crate::block_inline::{flatten_block_hierarchy, uniquify_node_name};
use crate::dce::remove_dead_nodes;
use crate::ir::{
    self, Block, NodeGraph, NodePayload, NodeRef, Package, PackageMember, Register, Type,
};
use crate::ir_rebase_ids::package_max_emitted_node_id;
use crate::ir_verify::{verify_function_in_package, verify_package};

/// A data port in the original block's declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionPort {
    pub name: String,
    pub ty: Type,
}

/// The physical reset input and its asserted level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionReset {
    pub name: String,
    pub active_low: bool,
}

/// A combinational function describing one clock edge of a flattened block.
///
/// Pass the data inputs (including reset) followed by the current register
/// values. The return is always a tuple containing the sampled outputs followed
/// by the next register values. All outputs and writes see the same pre-edge
/// state. Each field retains its IR type, including aggregate and empty values.
/// Evaluate calls in the original package with
/// [`crate::ir_eval::eval_fn_in_package`]. For symbolic execution or IR
/// serialization, [`crate::ir_inline::package_with_reachable_callees`] builds
/// a standalone package containing the transition and its callees.
#[derive(Debug, Clone)]
pub struct BlockTransition {
    pub function: ir::Fn,
    pub inputs: Vec<TransitionPort>,
    pub outputs: Vec<TransitionPort>,
    pub registers: Vec<Register>,
    pub reset: Option<TransitionReset>,
}

/// Lowers the selected block and its child instances, preserving calls to
/// package functions. Asynchronous resets and multiple writes to a register
/// require semantics outside this single-edge model.
pub fn lower_block_transition(
    package: &Package,
    block_name: &str,
) -> Result<BlockTransition, String> {
    let block = package
        .get_block(block_name)
        .ok_or_else(|| format!("block '{block_name}' not found"))?;
    if block.instantiations.is_empty() {
        verify_package(package).map_err(|e| e.to_string())?;
        return project_transition(package, block);
    }
    let hierarchy = flatten_block_hierarchy(package, block_name)?;
    let block = hierarchy
        .get_top_block()
        .ok_or("flattened package has no top block")?;
    project_transition(&hierarchy, block)
}

/// Retains the data and controls of one register-write sink during projection.
struct RegisterWrite {
    data: NodeRef,
    enable: Option<NodeRef>,
    reset: Option<NodeRef>,
}

/// Reuses each input and register-read node as a parameter. Write and output
/// sinks still evaluate to unit when another IR node consumes their value.
fn project_transition(package: &Package, block: &Block) -> Result<BlockTransition, String> {
    block.reject_inouts("block transition projection")?;
    if block.reset.as_ref().is_some_and(|reset| reset.asynchronous) {
        return Err(format!(
            "cannot lower asynchronous reset in block '{}' to a rising-edge transition",
            block.name
        ));
    }
    let port_info = |port| TransitionPort {
        name: block.port_name(port).to_owned(),
        ty: block.port_type(port).clone(),
    };
    let inputs = block.input_ports().map(port_info).collect();
    let outputs = block.output_ports().map(port_info).collect();
    let mut params: Vec<_> = block.input_ports().collect();
    let mut results: Vec<_> = block
        .output_ports()
        .map(|port| block.output_value(port))
        .collect();
    let mut function_names = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Function(function) => Some((function.name.clone(), 1)),
            PackageMember::Block(_) => None,
        })
        .collect();
    let mut graph = NodeGraph::new(uniquify_node_name(&block.name, &mut function_names));
    graph.nodes = block.nodes.clone();
    let mut used_names: HashMap<String, usize> = block
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            !matches!(
                node.payload,
                NodePayload::InputPort { .. }
                    | NodePayload::OutputPort { .. }
                    | NodePayload::RegisterRead { .. }
                    | NodePayload::RegisterWrite { .. }
            )
        })
        .map(|(index, _)| (ir::node_textual_id(block, NodeRef { index }), 1))
        .collect();
    for &port in &params {
        let node = graph.get_node_mut(port);
        node.name = Some(uniquify_node_name(block.port_name(port), &mut used_names));
        node.payload = NodePayload::Param;
    }
    let mut reads = HashMap::new();
    let mut writes = HashMap::new();
    for (index, node) in graph.nodes.iter_mut().enumerate() {
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                reads.insert(register.clone(), NodeRef { index });
                node.name = Some(uniquify_node_name(
                    &format!("{register}__q"),
                    &mut used_names,
                ));
                node.payload = NodePayload::Param;
            }
            NodePayload::RegisterWrite {
                arg,
                register,
                load_enable,
                reset,
            } => {
                let write = RegisterWrite {
                    data: *arg,
                    enable: *load_enable,
                    reset: *reset,
                };
                if writes.insert(register.clone(), write).is_some() {
                    return Err(format!(
                        "block '{}' has multiple register writes for '{register}'",
                        block.name
                    ));
                }
                node.name = None;
                node.payload = NodePayload::Tuple(vec![]);
            }
            NodePayload::OutputPort { .. } => {
                node.name = None;
                node.payload = NodePayload::Tuple(vec![]);
            }
            _ => {
                // Other nodes already have their combinational function
                // meaning.
            }
        }
    }
    let mut last_id = package_max_emitted_node_id(package);
    let mut append = |ty: Type, payload: NodePayload| -> Result<NodeRef, String> {
        last_id = last_id
            .checked_add(1)
            .ok_or("transition node IDs overflow usize")?;
        let index = graph.nodes.len();
        graph.nodes.push(ir::Node {
            text_id: last_id,
            name: None,
            ty,
            payload,
            pos: None,
        });
        Ok(NodeRef { index })
    };
    for register in &block.registers {
        let current = *reads
            .get(&register.name)
            .ok_or_else(|| format!("register '{}' has no read", register.name))?;
        let write = writes
            .get(&register.name)
            .ok_or_else(|| format!("register '{}' has no write", register.name))?;
        params.push(current);
        let mut next = write.data;
        if let Some(selector) = write.enable {
            next = append(
                register.ty.clone(),
                NodePayload::Sel {
                    selector,
                    cases: vec![current, next],
                    default: None,
                },
            )?;
        }
        if let Some(selector) = write.reset {
            let reset = block
                .reset
                .as_ref()
                .ok_or("reset write requires block reset metadata")?;
            let reset_value = register
                .reset_value
                .clone()
                .ok_or("reset write requires a register reset value")?;
            let literal = append(register.ty.clone(), NodePayload::Literal(reset_value))?;
            let cases = if reset.active_low {
                vec![literal, next]
            } else {
                vec![next, literal]
            };
            next = append(
                register.ty.clone(),
                NodePayload::Sel {
                    selector,
                    cases,
                    default: None,
                },
            )?;
        }
        results.push(next);
    }
    let ret_ty = Type::Tuple(
        block
            .output_ports()
            .map(|port| block.port_type(port))
            .chain(block.registers.iter().map(|register| &register.ty))
            .cloned()
            .map(Box::new)
            .collect(),
    );
    let result = append(ret_ty.clone(), NodePayload::Tuple(results))?;
    let function = ir::Fn {
        graph,
        params,
        ret_ty,
        ret_node_ref: Some(result),
    };
    verify_function_in_package(&function, package).map_err(|e| e.to_string())?;
    Ok(BlockTransition {
        // Keep output/state cones and observable effects, dropping unit-valued
        // sinks unless a user actually consumes them.
        function: remove_dead_nodes(&function),
        inputs,
        outputs,
        registers: block.registers.clone(),
        reset: block.reset.as_ref().map(|reset| TransitionReset {
            name: block.port_name(reset.port).to_owned(),
            active_low: reset.active_low,
        }),
    })
}

#[cfg(test)]
mod tests {
    use crate::IrValue;
    use crate::ir_eval::{FnEvalResult, eval_fn_in_package};
    use crate::ir_inline::package_with_reachable_callees;
    use crate::ir_parser::Parser;

    use super::*;

    const SWAP: &str = r#"package swap
top block swap(clk: clock, rst: bits[1], en: bits[1], b: bits[129], a: bits[129], observed: (())) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg first(bits[129], reset_value=7)
  reg second(bits[129])
  en: bits[1] = input_port(name=en, id=1)
  rst: bits[1] = input_port(name=rst, id=2)
  second_q: bits[129] = register_read(register=second, id=3)
  first_q: bits[129] = register_read(register=first, id=4)
  first_w: () = register_write(second_q, register=first, load_enable=en, reset=rst, id=5)
  second_w: () = register_write(first_q, register=second, id=6)
  a: () = output_port(first_q, name=a, id=7)
  b: () = output_port(second_q, name=b, id=8)
  unit: (()) = tuple(first_w, id=9)
  observed: () = output_port(unit, name=observed, id=10)
}
"#;

    #[test]
    fn register_updates_sample_simultaneously_and_reset_precedes_enable() {
        let first = IrValue::parse_typed("bits[129]:0x1ffffffffffffffffffffffffffffffff").unwrap();
        let second = IrValue::parse_typed("bits[129]:0x10000000000000000").unwrap();
        let reset_value = IrValue::make_ubits(129, 7).unwrap();
        let unit = IrValue::make_tuple(&[IrValue::make_tuple(&[])]);
        for active_low in [false, true] {
            let text = SWAP.replace("active_low=false", &format!("active_low={active_low}"));
            let package = Parser::new(&text).parse_and_validate_package().unwrap();
            let transition = lower_block_transition(&package, "swap").unwrap();
            assert_eq!(
                transition
                    .inputs
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>(),
                ["rst", "en"]
            );
            assert_eq!(
                transition
                    .outputs
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>(),
                ["b", "a", "observed"]
            );
            assert_eq!(
                transition
                    .registers
                    .iter()
                    .map(|r| r.name.as_str())
                    .collect::<Vec<_>>(),
                ["first", "second"]
            );
            assert_eq!(
                transition.reset,
                Some(TransitionReset {
                    name: "rst".to_owned(),
                    active_low
                })
            );
            for (asserted, enable, next_first) in [
                (false, false, &first),
                (false, true, &second),
                (true, false, &reset_value),
                (true, true, &reset_value),
            ] {
                let args = [
                    IrValue::make_ubits(1, u64::from(asserted ^ active_low)).unwrap(),
                    IrValue::make_ubits(1, u64::from(enable)).unwrap(),
                    first.clone(),
                    second.clone(),
                ];
                let FnEvalResult::Success(result) =
                    eval_fn_in_package(&package, &transition.function, &args)
                else {
                    panic!("transition failed to evaluate");
                };
                assert_eq!(
                    result.value,
                    IrValue::make_tuple(&[
                        second.clone(),
                        first.clone(),
                        unit.clone(),
                        next_first.clone(),
                        first.clone(),
                    ])
                );
            }
        }
    }

    #[test]
    fn instantiated_register_with_map_invoke_and_counted_for() {
        let text = r#"package calls
fn sum(i: bits[4] id=1, acc: bits[4] id=2) -> bits[4] {
  ret plus: bits[4] = add(i, acc, id=3)
}
fn loop(x: bits[4] id=4) -> bits[4] {
  ret updated: bits[4] = counted_for(x, trip_count=3, stride=1, body=sum, id=5)
}
fn parent(x: bits[4] id=6) -> bits[4] {
  ret updated: bits[4] = invoke(x, to_apply=loop, id=7)
}
block child(clk: clock, xs: bits[4][2], out: bits[4][2]) {
  reg r(bits[4][2])
  xs: bits[4][2] = input_port(name=xs, id=8)
  r_q: bits[4][2] = register_read(register=r, id=9)
  mapped: bits[4][2] = map(xs, to_apply=parent, id=10)
  r_w: () = register_write(mapped, register=r, id=11)
  out: () = output_port(r_q, name=out, id=12)
}
top block parent(clock: clock, xs: bits[4][2], delayed: bits[4][2]) {
  instantiation leaf(block=child, kind=block)
  xs: bits[4][2] = input_port(name=xs, id=13)
  drive: () = instantiation_input(xs, instantiation=leaf, port_name=xs, id=14)
  q: bits[4][2] = instantiation_output(instantiation=leaf, port_name=out, id=15)
  delayed: () = output_port(q, name=delayed, id=16)
}
"#;
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let transition = lower_block_transition(&package, "parent").unwrap();
        assert_eq!(transition.registers[0].name, "leaf__r");
        let proof_package = package_with_reachable_callees(&package, &transition.function).unwrap();
        verify_package(&proof_package).unwrap();
        let xs = IrValue::parse_typed("[bits[4]:3, bits[4]:15]").unwrap();
        let state = IrValue::parse_typed("[bits[4]:0, bits[4]:9]").unwrap();
        let FnEvalResult::Success(result) =
            eval_fn_in_package(&package, &transition.function, &[xs, state.clone()])
        else {
            panic!("transition with helper calls failed to evaluate");
        };
        let next = IrValue::parse_typed("[bits[4]:6, bits[4]:2]").unwrap();
        assert_eq!(result.value, IrValue::make_tuple(&[state, next]));
    }

    #[test]
    fn empty_transition_returns_an_empty_tuple() {
        let package = Parser::new("package empty\ntop block empty() {}")
            .parse_and_validate_package()
            .unwrap();
        let transition = lower_block_transition(&package, "empty").unwrap();
        let FnEvalResult::Success(result) = eval_fn_in_package(&package, &transition.function, &[])
        else {
            panic!("empty transition failed to evaluate");
        };
        assert_eq!(result.value, IrValue::make_tuple(&[]));
    }

    #[test]
    fn reports_asynchronous_reset_and_ambiguous_writes() {
        let text = SWAP.replace("asynchronous=false", "asynchronous=true");
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        assert_eq!(
            lower_block_transition(&package, "swap").unwrap_err(),
            "cannot lower asynchronous reset in block 'swap' to a rising-edge transition"
        );
        let text = SWAP.replace("  second_w:", "  extra_w: () = register_write(first_q, register=first, load_enable=en, reset=rst, id=11)\n  second_w:");
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        assert_eq!(
            lower_block_transition(&package, "swap").unwrap_err(),
            "block 'swap' has multiple register writes for 'first'"
        );
    }
}
