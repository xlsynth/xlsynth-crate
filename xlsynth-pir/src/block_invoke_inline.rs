// SPDX-License-Identifier: Apache-2.0

//! Inline IR function bodies at `invoke` nodes inside a block.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::IrValue;
use crate::ir::{Block, Fn, Node, NodePayload, NodeRef, Package, PackageMember, Type, Unop};
use crate::ir_rebase_ids::package_max_emitted_node_id;
use crate::ir_utils::remap_payload_with;
use crate::ir_verify::{
    verify_package, verify_package_inout_hierarchies, verify_package_structure,
};

/// Inlines ordinary calls in every block, retaining the package's functions,
/// block hierarchy, and register boundaries. Checks hierarchy dependencies
/// after expansion so an ignored argument cannot introduce a spurious cycle.
pub fn inline_block_invokes_in_package(package: Package) -> Result<Package, String> {
    let block_names: Vec<_> = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Block(block) => Some(block.name.clone()),
            PackageMember::Function(_) => None,
        })
        .collect();
    inline_block_invokes_in_blocks(package, &block_names)
}

/// Inlines calls in the named blocks in order, then verifies the entire
/// package. Unselected blocks can still contain invokes.
pub fn inline_block_invokes_in_blocks(
    mut package: Package,
    block_names: &[String],
) -> Result<Package, String> {
    verify_package_structure(&package).map_err(|error| error.to_string())?;
    let mut inliner = BlockInvokeInliner::from_verified_package(&package);
    let mut replacements = Vec::new();
    for name in block_names {
        if let Cow::Owned(block) = inliner.inline_block(name)? {
            replacements.push((inliner.blocks[name.as_str()].0, block));
        }
    }
    if !replacements.is_empty() {
        for (index, block) in replacements {
            package.members[index] = PackageMember::Block(block);
        }
        verify_package_structure(&package).map_err(|error| error.to_string())?;
    }
    verify_package_inout_hierarchies(&package).map_err(|error| error.to_string())?;
    Ok(package)
}

/// Resolves calls against one verified package and allocates fresh IDs across
/// its expanded blocks.
pub struct BlockInvokeInliner<'a> {
    /// Foreign functions have no inlineable implementation.
    functions: HashMap<&'a str, Option<&'a Fn>>,
    blocks: HashMap<&'a str, (usize, &'a Block)>,
    last_id: usize,
}

impl<'a> BlockInvokeInliner<'a> {
    /// Verifies the package, including callee signatures and the absence of
    /// recursive calls.
    pub fn new(package: &'a Package) -> Result<Self, String> {
        verify_package(package).map_err(|error| error.to_string())?;
        Ok(Self::from_verified_package(package))
    }

    /// Indexes callees after verification has established valid call targets
    /// and IDs.
    fn from_verified_package(package: &'a Package) -> Self {
        let mut functions = HashMap::new();
        let mut blocks = HashMap::new();
        for (index, member) in package.members.iter().enumerate() {
            match member {
                PackageMember::Function(function) => {
                    let foreign = function.ffi_proto().is_some();
                    functions.insert(function.name.as_str(), (!foreign).then_some(function));
                }
                PackageMember::Block(block) => {
                    blocks.insert(block.name.as_str(), (index, block));
                }
            }
        }
        Self {
            functions,
            blocks,
            last_id: package_max_emitted_node_id(package),
        }
    }

    /// Inlines the callee at each `invoke` node, borrowing the original block
    /// when it has none. Foreign functions require explicit extern instances.
    /// Block instances, maps, and counted loops retain their IR representation.
    pub fn inline_block(&mut self, name: &str) -> Result<Cow<'a, Block>, String> {
        self.expand_calls(name, None)
    }

    /// Also expands maps and counted loops, including those in nested callees.
    /// The budget bounds the total number of IR nodes added to this block.
    pub fn inline_block_with_unrolling(
        &mut self,
        name: &str,
        max_added_nodes: usize,
    ) -> Result<Cow<'a, Block>, String> {
        self.expand_calls(name, Some(max_added_nodes))
    }

    fn expand_calls(&mut self, name: &str, limit: Option<usize>) -> Result<Cow<'a, Block>, String> {
        let &(_, block) = self
            .blocks
            .get(name)
            .ok_or_else(|| format!("block '{name}' not found"))?;
        let Some(mut index) = block
            .nodes
            .iter()
            .position(|node| is_expandable(&node.payload, limit.is_some()))
        else {
            return Ok(Cow::Borrowed(block));
        };
        let mut expanded = block.clone();
        let mut names = block_names(block);
        let mut last_id = self.last_id;
        // Appended callees may contain further calls. Each node is visited
        // once; restore operand-before-user order only after the last
        // call is expanded.
        while index < expanded.nodes.len() {
            if !is_expandable(&expanded.nodes[index].payload, limit.is_some()) {
                index += 1;
                continue;
            }
            let call = expanded.nodes[index].clone();
            let (target, iteration_nodes) = match &call.payload {
                NodePayload::Invoke { to_apply, .. } => (to_apply, None),
                NodePayload::Map { to_apply, .. } => {
                    let Type::Array(layout) = &call.ty else {
                        unreachable!("verified map type");
                    };
                    // Index literal, array read, callback invoke.
                    let additional = layout
                        .element_count
                        .checked_mul(3)
                        .ok_or("map expansion size overflow")?;
                    (to_apply, Some(additional))
                }
                NodePayload::CountedFor {
                    body, trip_count, ..
                } => {
                    // Induction literal and body invoke.
                    let additional = trip_count
                        .checked_mul(2)
                        .ok_or("counted_for expansion size overflow")?;
                    (body, Some(additional))
                }
                _ => unreachable!("filtered to expandable calls"),
            };
            let Some(function) = self.functions[target.as_str()] else {
                return Err(format!(
                    "block '{name}' invokes foreign function '{target}'; use an extern instantiation to preserve its implementation"
                ));
            };
            // Newly added invokes and their nested bodies are visited by this
            // same traversal. An invoke copies neither parameters nor nil.
            let addition =
                iteration_nodes.unwrap_or(function.nodes.len() - function.params.len() - 1);
            if let Some(limit) = limit
                && addition > limit.saturating_sub(expanded.nodes.len() - block.nodes.len())
            {
                return Err(format!(
                    "expanding calls in block '{name}' exceeds the budget of {limit} additional IR nodes"
                ));
            }
            let replacement = if let NodePayload::Invoke { operands, .. } = &call.payload {
                let result = copy_callee(
                    function,
                    operands,
                    call.text_id,
                    &mut expanded.nodes,
                    &mut names,
                    &mut last_id,
                )?;
                NodePayload::Unop(Unop::Identity, result)
            } else {
                expand_iterations(&call, function, &mut expanded.nodes, &mut last_id)?
            };
            // Retain user-assigned names. Explicit operator.<id> spellings
            // must follow the new operator instead.
            let call = &mut expanded.nodes[index];
            if call
                .name
                .as_deref()
                .and_then(|name| name.strip_prefix(call.payload.get_operator()))
                .and_then(|suffix| suffix.strip_prefix('.'))
                .is_some_and(|digits| {
                    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
                })
            {
                call.name = None;
            }
            call.payload = replacement;
            index += 1;
        }
        expanded.compact_and_toposort()?;
        self.last_id = last_id;
        Ok(Cow::Owned(expanded))
    }
}

fn is_expandable(payload: &NodePayload, unroll: bool) -> bool {
    matches!(payload, NodePayload::Invoke { .. })
        || unroll
            && matches!(
                payload,
                NodePayload::Map { .. } | NodePayload::CountedFor { .. }
            )
}

/// Builds the constant induction variable or array index of an unrolled call.
fn index_literal(index: usize, width: usize) -> Result<NodePayload, String> {
    let index = u64::try_from(index).map_err(|e| e.to_string())?;
    IrValue::make_ubits(width, index)
        .map(NodePayload::Literal)
        .map_err(|e| e.to_string())
}

/// Appends ordinary invokes in iteration order without changing the register
/// graph.
fn expand_iterations(
    call: &Node,
    function: &Fn,
    nodes: &mut Vec<Node>,
    last_id: &mut usize,
) -> Result<NodePayload, String> {
    let mut append = |ty: Type, payload: NodePayload| -> Result<NodeRef, String> {
        *last_id = last_id
            .checked_add(1)
            .ok_or("node IDs overflow while unrolling block calls")?;
        let index = nodes.len();
        nodes.push(Node {
            text_id: *last_id,
            ty,
            payload,
            name: None,
            pos: call.pos.clone(),
        });
        Ok(NodeRef { index })
    };
    let invoke = |operands| NodePayload::Invoke {
        to_apply: function.name.clone(),
        operands,
    };
    match &call.payload {
        NodePayload::Map { array, .. } => {
            let Type::Array(layout) = &call.ty else {
                unreachable!("verified map type");
            };
            let index_width = crate::math::ceil_log2(layout.element_count).max(1);
            let mut values = Vec::with_capacity(layout.element_count);
            for i in 0..layout.element_count {
                let index = append(Type::Bits(index_width), index_literal(i, index_width)?)?;
                let element = append(
                    function.get_param(0).ty.clone(),
                    NodePayload::ArrayIndex {
                        array: *array,
                        indices: vec![index],
                        assumed_in_bounds: true,
                    },
                )?;
                values.push(append(function.ret_ty.clone(), invoke(vec![element]))?);
            }
            Ok(NodePayload::Array(values))
        }
        NodePayload::CountedFor {
            init,
            trip_count,
            stride,
            invariant_args,
            ..
        } => {
            let Type::Bits(width) = function.get_param(0).ty else {
                unreachable!("verified induction type");
            };
            let mut carry = *init;
            for i in 0..*trip_count {
                let induction = i
                    .checked_mul(*stride)
                    .ok_or("counted_for induction overflow")?;
                let index = append(Type::Bits(width), index_literal(induction, width)?)?;
                let mut operands = Vec::with_capacity(2 + invariant_args.len());
                operands.extend([index, carry]);
                operands.extend(invariant_args);
                carry = append(function.ret_ty.clone(), invoke(operands))?;
            }
            Ok(NodePayload::Unop(Unop::Identity, carry))
        }
        _ => unreachable!("unrolling a map or counted_for"),
    }
}

/// Copies the complete body, including effects whose tokens are outside the
/// return cone.
fn copy_callee(
    function: &Fn,
    operands: &[NodeRef],
    call_id: usize,
    nodes: &mut Vec<Node>,
    names: &mut HashSet<String>,
    last_id: &mut usize,
) -> Result<NodeRef, String> {
    // Bind parameters first, including when the callee returns one directly.
    let mut mapping = vec![NodeRef { index: 0 }; function.nodes.len()];
    for (&parameter, &actual) in function.params.iter().zip(operands) {
        mapping[parameter.index] = actual;
    }
    // Verification guarantees that each operand already has a destination.
    for (index, source) in function
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !matches!(node.payload, NodePayload::Nil | NodePayload::Param))
    {
        mapping[index] = NodeRef { index: nodes.len() };
        *last_id = last_id
            .checked_add(1)
            .ok_or("node IDs overflow while inlining invoke")?;
        let mut payload = remap_payload_with(&source.payload, |(_, arg)| mapping[arg.index]);
        if let NodePayload::Assert { label, .. } | NodePayload::Cover { label, .. } = &mut payload
            && !label.is_empty()
        {
            *label = call_scoped_name(names, call_id, label);
        }
        nodes.push(Node {
            text_id: *last_id,
            name: source
                .name
                .as_ref()
                .map(|name| call_scoped_name(names, call_id, name)),
            ty: source.ty.clone(),
            payload,
            pos: source.pos.clone(),
        });
    }
    Ok(mapping[function
        .ret_node_ref
        .expect("verified function has a return node")
        .index])
}

/// Reserves user-visible names before assigning names to copied operations and
/// debug labels.
fn block_names(block: &Block) -> HashSet<String> {
    let mut names: HashSet<_> = block
        .nodes
        .iter()
        .filter_map(|node| node.name.clone())
        .collect();
    names.extend(block.clock_port_name().map(str::to_owned));
    names.extend(block.registers.iter().map(|reg| reg.name.clone()));
    names.extend(
        block
            .instantiations
            .iter()
            .map(|instance| instance.name.clone()),
    );
    names.extend(block.inout_ports().map(str::to_owned));
    names.extend(block.inout_nets.iter().map(|net| net.name.clone()));
    for node in &block.nodes {
        match &node.payload {
            NodePayload::InputPort { name, .. }
            | NodePayload::OutputPort { name, .. }
            | NodePayload::Assert { label: name, .. }
            | NodePayload::Cover { label: name, .. } => {
                names.insert(name.clone());
            }
            _ => {
                // Other operations have no additional module-scoped names.
            }
        }
    }
    names
}

/// Qualifies local names by the unique call site, including labels that will
/// become RTL identifiers.
fn call_scoped_name(names: &mut HashSet<String>, call_id: usize, local: &str) -> String {
    let local = local.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_");
    let base = format!("invoke_{call_id}_{local}");
    let mut candidate = base.clone();
    let mut suffix = 0;
    while !names.insert(candidate.clone()) {
        suffix += 1;
        candidate = format!("{base}__{suffix}");
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_transition::lower_block_transition;
    use crate::ir_eval::{FnEvalResult, eval_fn_in_package};
    use crate::ir_parser::Parser;

    #[test]
    fn unrolling_keeps_zero_trip_carry_and_strided_wide_induction() -> Result<(), String> {
        let ir = r#"package iterations
fn body(i: bits[81] id=1, acc: bits[81] id=2, x: bits[81] id=3) -> bits[81] {
  biased: bits[81] = add(acc, x, id=4)
  ret updated: bits[81] = add(biased, i, id=5)
}
top block sample(init: bits[81], x: bits[81], a: bits[81], b: bits[81]) {
  init: bits[81] = input_port(name=init, id=6)
  x: bits[81] = input_port(name=x, id=7)
  empty: bits[81] = counted_for(init, trip_count=0, stride=3, body=body, invariant_args=[x], id=8)
  loop_result: bits[81] = counted_for(init, trip_count=3, stride=7, body=body, invariant_args=[x], id=9)
  a: () = output_port(empty, name=a, id=10)
  b: () = output_port(loop_result, name=b, id=11)
}
"#;
        let mut package = Parser::new(ir)
            .parse_and_validate_package()
            .map_err(|e| e.to_string())?;
        let mut inliner = BlockInvokeInliner::new(&package)?;
        let block = inliner
            .inline_block_with_unrolling("sample", 12)?
            .into_owned();
        package.replace_block("sample", block)?;
        let transition = lower_block_transition(&package, "sample")?;
        let init =
            IrValue::parse_typed("bits[81]:0x100000000000000000000").map_err(|e| e.to_string())?;
        let bias = IrValue::make_ubits(81, 5).map_err(|e| e.to_string())?;
        let FnEvalResult::Success(result) =
            eval_fn_in_package(&package, &transition.function, &[init.clone(), bias])
        else {
            panic!("expanded iterations failed");
        };
        let expected =
            IrValue::parse_typed("bits[81]:0x100000000000000000024").map_err(|e| e.to_string())?;
        assert_eq!(result.value, IrValue::make_tuple(&[init, expected]));
        Ok(())
    }

    #[test]
    fn package_stage_expands_nested_calls_in_all_blocks() {
        let ir = r#"package calls

fn seen(x: bits[1] id=1) -> bits[1] {
  event: () = cover(x, label="was_seen", id=2)
  ret x: bits[1] = param(name=x, id=1)
}

fn nested(x: bits[1] id=3) -> bits[1] {
  ret delegated: bits[1] = invoke(x, to_apply=seen, id=4)
}

block first(x: bits[1], invoke_6_was_seen: bits[1]) {
  x: bits[1] = input_port(name=x, id=5)
  called: bits[1] = invoke(x, to_apply=seen, id=6)
  out: () = output_port(called, name=invoke_6_was_seen, id=7)
}

block second(x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=8)
  called: bits[1] = invoke(x, to_apply=nested, id=9)
  out: () = output_port(called, name=out, id=10)
}

block untouched() {
}
"#;
        let mut package = Parser::new(ir).parse_and_validate_package().unwrap();
        let call = package
            .get_block_mut("first")
            .unwrap()
            .nodes
            .iter_mut()
            .find(|node| node.text_id == 6)
            .unwrap();
        call.name = Some("invoke.6".to_string());
        let mut inliner = BlockInvokeInliner::new(&package).unwrap();
        assert!(matches!(
            inliner.inline_block("untouched").unwrap(),
            Cow::Borrowed(_)
        ));
        let expanded = inline_block_invokes_in_package(package).unwrap();
        let first = expanded.get_block("first").unwrap();
        let labels: Vec<_> = first
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::Cover { label, .. } => Some(label.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(labels, ["invoke_6_was_seen__1"]);
        let second = expanded.get_block("second").unwrap();
        assert!(
            !second
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Invoke { .. }))
        );

        let text = expanded.to_string();
        let reparsed = Parser::new(&text).parse_and_validate_package().unwrap();
        assert_eq!(
            inline_block_invokes_in_package(reparsed)
                .unwrap()
                .to_string(),
            text
        );
    }

    #[test]
    fn hierarchy_cycles_use_inlined_pin_driver_dependencies() {
        let ir = r#"package pins
fn drive(sensed: bits[1] id=1) -> bits[1] {
  ret zero: bits[1] = literal(value=0, id=2)
}
block child(data: bits[1], io: bits[1]) {
  ext_inout_net pad(bits[1], ports=[io])
  data: bits[1] = input_port(name=data, id=3)
  on: bits[1] = literal(value=1, id=4)
  driven: bits[1] = invoke(data, to_apply=drive, id=5)
  drive_pad: () = ext_inout_drive(driven, on, net=pad, id=6)
}
top block parent(pin: bits[1]) {
  ext_inout_net bus(bits[1], ports=[pin])
  instantiation cell(block=child, kind=block)
  bind_pin: () = ext_instantiation_inout(instantiation=cell, port_name=io, net=bus, id=7)
  sample: bits[1] = ext_inout_read(net=bus, id=8)
  sense: () = instantiation_input(sample, instantiation=cell, port_name=data, id=9)
}
"#;
        let package = Parser::new(ir).parse_package().unwrap();
        // The standalone verifier conservatively considers the call to read
        // data.
        assert!(
            verify_package(&package)
                .unwrap_err()
                .to_string()
                .contains("combinational cycle")
        );
        inline_block_invokes_in_package(package).unwrap();

        let feedback = ir.replace("literal(value=0, id=2)", "identity(sensed, id=2)");
        let package = Parser::new(&feedback).parse_package().unwrap();
        let error = inline_block_invokes_in_package(package).unwrap_err();
        assert!(error.contains("combinational cycle"), "{error}");
    }

    #[test]
    fn nested_foreign_function_is_rejected_without_committing_ids() {
        let ir = r#"package foreign_calls

#[ // a foreign cell with a behavioral body
  ffi_proto("""code_template: "custom_cell {fn} (.x({x}), .out({return}));"
""")]
fn foreign(x: bits[1] id=1) -> bits[1] {
  ret x: bits[1] = param(name=x, id=1)
}

fn nested(x: bits[1] id=2) -> bits[1] {
  ret delegated: bits[1] = invoke(x, to_apply=foreign, id=3)
}

block caller(x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=4)
  called: bits[1] = invoke(x, to_apply=nested, id=5)
  out: () = output_port(called, name=out, id=6)
}
"#;
        let package = Parser::new(ir).parse_and_validate_package().unwrap();
        let mut inliner = BlockInvokeInliner::new(&package).unwrap();
        assert_eq!(
            inliner.inline_block("caller").unwrap_err(),
            "block 'caller' invokes foreign function 'foreign'; use an extern instantiation to preserve its implementation"
        );
        assert_eq!(inliner.last_id, package_max_emitted_node_id(&package));
    }
}
