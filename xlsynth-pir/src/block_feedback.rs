// SPDX-License-Identifier: Apache-2.0

//! Combinational feedback analysis shared by hierarchical validation and
//! function projection. Whole-value searches find candidates. The projection
//! rules trace the selected bits and fields using the controls at each binding.

pub(crate) mod controls;
pub(crate) mod projection;

use std::collections::BTreeMap;

use crate::ir::{Block, Instantiation, NodeGraph, NodePayload, NodeRef};

/// Block-local instance definitions and the sink nodes driving their inputs.
pub(crate) struct InstanceConnections<'a> {
    pub(crate) definitions: BTreeMap<&'a str, &'a Instantiation>,
    pub(crate) inputs: BTreeMap<&'a str, BTreeMap<&'a str, NodeRef>>,
}

impl<'a> InstanceConnections<'a> {
    /// Indexes connections once, independently of the widths of the ports.
    pub(crate) fn new(block: &'a Block) -> Self {
        let mut inputs: BTreeMap<&str, BTreeMap<&str, NodeRef>> = BTreeMap::new();
        for (index, node) in block.nodes.iter().enumerate() {
            if let NodePayload::InstantiationInput {
                instantiation,
                port_name,
                ..
            } = &node.payload
            {
                inputs
                    .entry(instantiation)
                    .or_default()
                    .insert(port_name, NodeRef { index });
            }
        }
        let definitions = block
            .instantiations
            .iter()
            .map(|instance| (instance.name.as_str(), instance))
            .collect();
        Self {
            definitions,
            inputs,
        }
    }

    /// Returns the driver of a connected instance input.
    pub(crate) fn driver(&self, graph: &NodeGraph, instance: &str, port: &str) -> Option<NodeRef> {
        let sink = *self.inputs.get(instance)?.get(port)?;
        let NodePayload::InstantiationInput { arg, .. } = graph.get_node(sink).payload else {
            unreachable!("indexed instance input")
        };
        Some(arg)
    }
}

/// One node on the active dependency path and its next operand to visit.
pub(crate) struct Frame {
    pub(crate) node: usize,
    next: usize,
}

/// Visits each dependency once. A back edge identifies a cycle in the current
/// path, and finishing order also supplies the first pass of Kosaraju's
/// algorithm.
pub(crate) fn walk_dependencies<'a, E>(
    len: usize,
    operands: impl Fn(usize) -> &'a [NodeRef],
    missing: impl Fn(usize) -> E,
    mut back_edge: impl FnMut(&[Frame]) -> Result<(), E>,
    mut finish: impl FnMut(usize),
) -> Result<(), E> {
    const UNSEEN: usize = usize::MAX;
    const FINISHED: usize = usize::MAX - 1;
    let mut depth = vec![UNSEEN; len];
    let mut path = Vec::new();
    for root in 0..len {
        if depth[root] != UNSEEN {
            continue;
        }
        depth[root] = 0;
        path.push(Frame {
            node: root,
            next: 0,
        });
        while let Some(frame) = path.last_mut() {
            if let Some(arg) = operands(frame.node).get(frame.next) {
                frame.next += 1;
                let slot = depth.get_mut(arg.index).ok_or_else(|| missing(arg.index))?;
                match *slot {
                    UNSEEN => {
                        *slot = path.len();
                        path.push(Frame {
                            node: arg.index,
                            next: 0,
                        });
                    }
                    FINISHED => {} // All dependencies of this node were visited.
                    start => back_edge(&path[start..])?,
                }
            } else {
                let node = frame.node;
                path.pop();
                depth[node] = FINISHED;
                finish(node);
            }
        }
    }
    Ok(())
}

/// Marks exactly the nodes in a nontrivial strongly connected component or
/// with a self-edge. Consumers that only depend on feedback stay unmarked.
pub(crate) fn cyclic_nodes(edges: &[Vec<NodeRef>]) -> Result<Vec<bool>, String> {
    let mut order = Vec::with_capacity(edges.len());
    let mut has_cycle = false;
    walk_dependencies(
        edges.len(),
        |i| &edges[i],
        |i| format!("dependency references missing node {i}"),
        |_| {
            has_cycle = true;
            Ok(())
        },
        |node| order.push(node),
    )?;
    let mut cyclic = vec![false; edges.len()];
    if !has_cycle {
        return Ok(cyclic);
    }
    let mut users = vec![Vec::new(); edges.len()];
    for (node, args) in edges.iter().enumerate() {
        for arg in args {
            users[arg.index].push(node);
        }
    }
    let mut visited = vec![false; edges.len()];
    let mut component = Vec::new();
    for root in order.into_iter().rev() {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        component.clear();
        component.push(root);
        let mut i = 0;
        while i < component.len() {
            for &user in &users[component[i]] {
                if !visited[user] {
                    visited[user] = true;
                    component.push(user);
                }
            }
            i += 1;
        }
        if component.len() > 1 || edges[root].contains(&NodeRef { index: root }) {
            for &node in &component {
                cyclic[node] = true;
            }
        }
    }
    Ok(cyclic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_membership_matches_nonempty_paths_back_to_each_node() {
        for mask in 0..1usize << 9 {
            let edges: Vec<Vec<_>> = (0..3)
                .map(|from| {
                    (0..3)
                        .filter(|to| mask & (1 << (3 * from + to)) != 0)
                        .map(|index| NodeRef { index })
                        .collect()
                })
                .collect();
            let expected: Vec<_> = (0..3)
                .map(|root| {
                    let mut frontier = vec![root];
                    // A shortest cycle through root uses at most three edges.
                    for _ in 0..3 {
                        frontier = frontier
                            .iter()
                            .flat_map(|&i| edges[i].iter().map(|n| n.index))
                            .collect();
                        if frontier.contains(&root) {
                            return true;
                        }
                    }
                    false
                })
                .collect();
            assert_eq!(cyclic_nodes(&edges).unwrap(), expected, "edges {edges:?}");
        }
    }
}
