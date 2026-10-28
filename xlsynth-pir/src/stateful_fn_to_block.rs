// SPDX-License-Identifier: Apache-2.0

//! Interpret a function's register-parameter suffix and outer return tuple as
//! the current state, outputs, and next state of a clocked block.

use std::collections::HashSet;

use crate::ir::{
    Block, BlockPort, BlockReset, Fn, Node, NodePayload, NodeRef, Package, PackageMember, Register,
    Type,
};
use crate::ir_builder::is_valid_identifier;
use crate::ir_rebase_ids::package_max_emitted_node_id;
use crate::ir_value_utils::zero_ir_value_for_type;
use crate::ir_verify::{verify_function_in_package, verify_package};

const REGISTER_PREFIX: &str = "__reg_";

/// Replaces the named function with the package's top block.
///
/// Each trailing `__reg_NAME` parameter reads register `NAME`. The return
/// must be a tuple whose final elements supply those registers' next values
/// in parameter order. Earlier elements drive `out0`, `out1`, etc. Registers
/// synchronously reset to zero on `rst`. Other package members are preserved
/// and all references are checked against the resulting package.
pub fn stateful_fn_to_block(mut package: Package, top: &str) -> Result<Package, String> {
    let index = package
        .members
        .iter()
        .position(|member| matches!(member, PackageMember::Function(f) if f.name == top))
        .ok_or_else(|| format!("function {top:?} not found in package"))?;
    let PackageMember::Function(source) = &package.members[index] else {
        unreachable!()
    };
    verify_function_in_package(source, &package).map_err(|error| error.to_string())?;
    let last_id = package_max_emitted_node_id(&package);
    let PackageMember::Function(source) = package.members.remove(index) else {
        unreachable!()
    };
    let block = convert_function(source, last_id)?;
    package.members.insert(index, PackageMember::Block(block));
    package.set_top_block(top)?;
    verify_package(&package).map_err(|error| format!("invalid converted package: {error}"))?;
    Ok(package)
}

/// The original graph is moved intact, so replacing a parameter's payload
/// changes every use, including a direct return of the parameter.
fn convert_function(function: Fn, mut last_id: usize) -> Result<Block, String> {
    let first_register = function
        .param_nodes()
        .position(|param| param.param_name().starts_with(REGISTER_PREFIX))
        .unwrap_or(function.params.len());
    let register_count = function.params.len() - first_register;
    let Type::Tuple(elements) = function.ret_ty else {
        return Err("a stateful function must return a tuple".to_string());
    };
    let output_count = elements.len().checked_sub(register_count).ok_or_else(|| {
        format!(
            "return tuple has {} elements for {register_count} registers",
            elements.len()
        )
    })?;
    let output_names: Vec<_> = (0..output_count).map(|n| format!("out{n}")).collect();
    let generated_names: HashSet<_> = ["clk", "rst"]
        .into_iter()
        .chain(output_names.iter().map(String::as_str))
        .collect();
    let return_ref = function
        .ret_node_ref
        .ok_or("function has no return value")?;
    let mut block = Block {
        graph: function.graph,
        ports: vec![BlockPort::Clock("clk".into())],
        reset: None,
        registers: Vec::new(),
        instantiations: Vec::new(),
    };

    // Body names are optional. Free any that the generated interface needs
    // while retaining the body's textual node IDs and source positions.
    for node in &mut block.nodes {
        if !matches!(node.payload, NodePayload::Param)
            && node
                .name
                .as_deref()
                .is_some_and(|name| generated_names.contains(name))
        {
            node.name = None;
        }
    }
    let rst = append_node(
        &mut block,
        &mut last_id,
        Type::Bits(1),
        NodePayload::InputPort {
            name: "rst".into(),
            sv_type: None,
        },
    )?;
    block.ports.push(BlockPort::Input(rst));
    block.reset = Some(BlockReset {
        port: rst,
        asynchronous: false,
        active_low: false,
    });
    for (i, param) in function.params.into_iter().enumerate() {
        let node = block.get_node_mut(param);
        let name = node.param_name().to_owned();
        if i < first_register {
            if generated_names.contains(name.as_str()) {
                return Err(format!(
                    "input {name:?} conflicts with a generated block port"
                ));
            }
            node.payload = NodePayload::InputPort {
                name,
                sv_type: None,
            };
            block.ports.push(BlockPort::Input(param));
        } else {
            let register = name.strip_prefix(REGISTER_PREFIX).ok_or_else(|| format!(
                "register parameters must form a contiguous suffix; found ordinary parameter {name:?}"
            ))?;
            if !is_valid_identifier(register) {
                return Err(format!(
                    "invalid register name {register:?} from parameter {name:?}"
                ));
            }
            let next_type = &elements[output_count + i - first_register];
            if node.ty != **next_type {
                return Err(format!(
                    "register {register:?} has type {}, but its next value has type {next_type}",
                    node.ty
                ));
            }
            node.payload = NodePayload::RegisterRead {
                register: register.into(),
            };
            let ty = node.ty.clone();
            block.registers.push(Register {
                name: register.into(),
                reset_value: Some(zero_ir_value_for_type(&ty)),
                ty,
            });
        }
    }

    // A function can also return an aggregate parameter or call result.
    // Project exactly one level in that case, including for a singleton tuple.
    let direct_elements = match &block.get_node(return_ref).payload {
        NodePayload::Tuple(args) => Some(args.clone()),
        _ => None,
    };
    for (index, element) in elements.into_iter().enumerate() {
        let arg = if let Some(args) = &direct_elements {
            args[index]
        } else {
            append_node(
                &mut block,
                &mut last_id,
                *element,
                NodePayload::TupleIndex {
                    tuple: return_ref,
                    index,
                },
            )?
        };
        if index < output_count {
            let port = append_node(
                &mut block,
                &mut last_id,
                Type::nil(),
                NodePayload::OutputPort {
                    name: output_names[index].clone(),
                    arg,
                    sv_type: None,
                },
            )?;
            block.ports.push(BlockPort::Output(port));
        } else {
            let register = block.registers[index - output_count].name.clone();
            append_node(
                &mut block,
                &mut last_id,
                Type::nil(),
                NodePayload::RegisterWrite {
                    register,
                    arg,
                    load_enable: None,
                    reset: Some(rst),
                },
            )?;
        }
    }
    if direct_elements.is_some()
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

/// Allocates only new IDs; existing nodes keep their package-wide identities.
fn append_node(
    block: &mut Block,
    last_id: &mut usize,
    ty: Type,
    payload: NodePayload,
) -> Result<NodeRef, String> {
    *last_id = last_id
        .checked_add(1)
        .ok_or("block node IDs overflow usize")?;
    let name = match &payload {
        NodePayload::InputPort { name, .. } | NodePayload::OutputPort { name, .. } => {
            Some(name.clone())
        }
        _ => None,
    };
    let node = NodeRef {
        index: block.nodes.len(),
    };
    block.nodes.push(Node {
        text_id: *last_id,
        name,
        ty,
        payload,
        pos: None,
    });
    Ok(node)
}
