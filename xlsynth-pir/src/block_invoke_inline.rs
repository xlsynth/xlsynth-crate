// SPDX-License-Identifier: Apache-2.0

//! Inline IR function bodies at `invoke` nodes inside a block.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::ir::{Block, Fn, Node, NodePayload, NodeRef, Package, PackageMember, Unop};
use crate::ir_rebase_ids::package_max_emitted_node_id;
use crate::ir_utils::remap_payload_with;
use crate::ir_verify::verify_package;

/// Resolves calls against one verified package and allocates fresh IDs across
/// its expanded blocks.
pub struct BlockInvokeInliner<'a> {
    /// Foreign functions have no inlineable implementation.
    functions: HashMap<&'a str, Option<&'a Fn>>,
    blocks: HashMap<&'a str, &'a Block>,
    last_id: usize,
}

impl<'a> BlockInvokeInliner<'a> {
    /// Verifies the package, including callee signatures and the absence of
    /// recursive calls.
    pub fn new(package: &'a Package) -> Result<Self, String> {
        verify_package(package).map_err(|error| error.to_string())?;
        let mut functions = HashMap::new();
        let mut blocks = HashMap::new();
        for member in &package.members {
            match member {
                PackageMember::Function(function) => {
                    let foreign = function.ffi_proto().is_some();
                    functions.insert(function.name.as_str(), (!foreign).then_some(function));
                }
                PackageMember::Block(block) => {
                    blocks.insert(block.name.as_str(), block);
                }
            }
        }
        Ok(Self {
            functions,
            blocks,
            last_id: package_max_emitted_node_id(package),
        })
    }

    /// Inlines the callee at each `invoke` node, borrowing the original block
    /// when it has none. Foreign functions require explicit extern instances.
    /// Block instances, maps, and counted loops retain their IR representation.
    pub fn inline_block(&mut self, name: &str) -> Result<Cow<'a, Block>, String> {
        let &block = self
            .blocks
            .get(name)
            .ok_or_else(|| format!("block '{name}' not found"))?;
        let Some(mut index) = block
            .nodes
            .iter()
            .position(|node| matches!(node.payload, NodePayload::Invoke { .. }))
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
            if let NodePayload::Invoke { to_apply, operands } = &expanded.nodes[index].payload {
                let Some(function) = self.functions[to_apply.as_str()] else {
                    return Err(format!(
                        "block '{name}' invokes foreign function '{to_apply}'; use an extern instantiation to preserve its implementation"
                    ));
                };
                let call_id = expanded.nodes[index].text_id;
                let operands = operands.clone();
                let result = copy_callee(
                    function,
                    &operands,
                    call_id,
                    &mut expanded.nodes,
                    &mut names,
                    &mut last_id,
                )?;
                // Retain the call site's user-assigned name and existing
                // references. An explicit invoke.<id> spelling
                // must follow the new operator instead.
                let call = &mut expanded.nodes[index];
                if call
                    .name
                    .as_deref()
                    .and_then(|name| name.strip_prefix("invoke."))
                    .is_some_and(|digits| {
                        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
                    })
                {
                    call.name = None;
                }
                call.payload = NodePayload::Unop(Unop::Identity, result);
            }
            index += 1;
        }
        expanded.compact_and_toposort()?;
        self.last_id = last_id;
        Ok(Cow::Owned(expanded))
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
    use crate::ir_parser::Parser;

    #[test]
    fn expanded_blocks_can_be_reassembled_in_one_package() {
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
        let first = inliner.inline_block("first").unwrap().into_owned();
        let labels: Vec<_> = first
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::Cover { label, .. } => Some(label.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(labels, ["invoke_6_was_seen__1"]);
        let second = inliner.inline_block("second").unwrap().into_owned();
        assert!(
            !second
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Invoke { .. }))
        );

        let mut expanded = package.clone();
        expanded.replace_block("first", first).unwrap();
        expanded.replace_block("second", second).unwrap();
        verify_package(&expanded).unwrap();
        Parser::new(&expanded.to_string())
            .parse_and_validate_package()
            .unwrap();
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
