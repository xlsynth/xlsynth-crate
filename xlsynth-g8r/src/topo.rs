// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef};
use std::collections::{HashMap, HashSet, VecDeque};

/// Returns a postorder traversal of the AIG nodes reachable from aig_ref (dedup
/// by node).
pub fn postorder_for_aig_ref(
    aig_ref: &AigRef,
    nodes: &[AigNode],
    cache: &HashMap<AigRef, impl Sized>,
) -> Vec<AigRef> {
    let mut worklist = VecDeque::new();
    let mut visited = HashSet::new();
    let mut postorder = Vec::new();
    worklist.push_back(*aig_ref);
    while let Some(current) = worklist.pop_back() {
        if cache.contains_key(&current) || visited.contains(&current) {
            continue;
        }
        let node = &nodes[current.id];
        let mut all_deps_visited = true;
        for dep in node.get_operands() {
            if !cache.contains_key(&dep.node) && !visited.contains(&dep.node) {
                worklist.push_back(current); // Revisit after dependencies
                worklist.push_back(dep.node);
                all_deps_visited = false;
                break;
            }
        }
        if all_deps_visited {
            visited.insert(current);
            postorder.push(current);
        }
    }
    postorder
}

/// Returns a postorder traversal of the AIG operands reachable from a set of
/// outputs (dedup by operand).
pub fn post_order_operands(
    starts: &[AigOperand],
    nodes: &[AigNode],
    discard_inputs: bool,
) -> Vec<AigOperand> {
    // Assert starts is not empty (degenerate case)
    debug_assert!(
        !starts.is_empty(),
        "post_order_operands: starts is empty (no outputs or degenerate graph)"
    );
    let mut worklist = VecDeque::new();
    let mut visited = HashSet::new();
    let mut postorder = Vec::new();
    for &start in starts {
        debug_assert!(
            start.node.id < nodes.len(),
            "post_order_operands: start operand node index out of bounds: {} (nodes.len() = {})",
            start.node.id,
            nodes.len()
        );
        worklist.push_back(start);
    }
    while let Some(current) = worklist.pop_back() {
        if visited.contains(&current) {
            continue;
        }
        debug_assert!(
            current.node.id < nodes.len(),
            "post_order_operands: operand node index out of bounds: {} (nodes.len() = {})",
            current.node.id,
            nodes.len()
        );
        let node = &nodes[current.node.id];
        let mut all_deps_visited = true;
        for dep in node.get_operands() {
            if !visited.contains(&dep) {
                debug_assert!(
                    dep.node.id < nodes.len(),
                    "post_order_operands: dependency operand node index out of bounds: {:?} (nodes.len() = {})",
                    dep.node.id,
                    nodes.len()
                );
                worklist.push_back(current); // Revisit after dependencies
                worklist.push_back(dep);
                all_deps_visited = false;
                break;
            }
        }
        if all_deps_visited {
            let should_push = match node {
                AigNode::Input { .. } => current.negated || !discard_inputs,
                _ => true,
            };
            if should_push {
                postorder.push(current);
            }
            visited.insert(current);
        }
    }
    postorder
}

/// Postorder traversal that deduplicates only by AigRef (node id), ignoring
/// operand negation.
pub fn postorder_for_aig_refs_node_only(
    aig_refs: &[AigRef],
    nodes: &[AigNode],
    cache: &HashMap<AigRef, impl Sized>,
) -> Vec<AigRef> {
    let mut worklist = VecDeque::new();
    let mut visited = HashSet::new();
    let mut postorder = Vec::new();
    worklist.extend(aig_refs);
    while let Some(current) = worklist.pop_back() {
        if cache.contains_key(&current) || visited.contains(&current) {
            continue;
        }
        let node = &nodes[current.id];
        let mut all_deps_visited = true;
        for dep in node.get_operands() {
            if !cache.contains_key(&dep.node) && !visited.contains(&dep.node) {
                worklist.push_back(current); // Revisit after dependencies
                worklist.push_back(dep.node);
                all_deps_visited = false;
                break;
            }
        }
        if all_deps_visited {
            visited.insert(current);
            postorder.push(current);
        }
    }
    postorder
}

/// Extracts the combined transitive fan-in cone for a set of nodes.
///
/// Returns:
/// * the set of all gates within the cones (this is given in a deterministic
///   topological ordering, as that can often be more useful than getting the
///   set and the caller needing to reconstruct the ordering).
/// * the set of primary inputs feeding the cones.
pub fn extract_cone(start_nodes: &[AigRef], gates: &[AigNode]) -> (Vec<AigRef>, HashSet<AigRef>) {
    let mut cone_gates_set = HashSet::new();
    let mut cone_gates = Vec::new();
    let mut cone_inputs = HashSet::new();
    let mut visited = HashSet::new();
    let mut worklist: Vec<AigRef> = start_nodes.to_vec();

    let mut add_cone_gate = |aig_ref: AigRef| {
        if cone_gates_set.insert(aig_ref) {
            cone_gates.push(aig_ref);
        }
    };

    while let Some(current_ref) = worklist.pop() {
        if !visited.insert(current_ref) {
            // Already visited or WIP
            continue;
        }

        let node = &gates[current_ref.id];

        match node {
            AigNode::Input { .. } => {
                cone_inputs.insert(current_ref);
            }
            AigNode::Literal(_) => {
                add_cone_gate(current_ref);
            }
            AigNode::And2 { a, b, .. } => {
                add_cone_gate(current_ref);
                worklist.push(a.node);
                worklist.push(b.node);
            }
        }
    }

    (cone_gates, cone_inputs)
}

/// Returns (topological order, None) if acyclic, or (partial order,
/// Some(not_visited_nodes)) if a cycle is detected.
pub fn topo_order_and_cycle_check(nodes: &[AigNode]) -> (Vec<AigRef>, Option<Vec<usize>>) {
    let gate_count = nodes.len();
    let mut indegree = vec![0usize; gate_count];
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); gate_count];
    for (i, node) in nodes.iter().enumerate() {
        if let AigNode::And2 { a, b, .. } = node {
            indegree[i] = 2;
            parents[a.node.id].push(i);
            parents[b.node.id].push(i);
        }
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..gate_count {
        if indegree[i] == 0 {
            queue.push_back(i);
        }
    }
    let mut topo: Vec<AigRef> = Vec::with_capacity(gate_count);
    while let Some(node_id) = queue.pop_front() {
        topo.push(AigRef { id: node_id });
        for &parent in &parents[node_id] {
            indegree[parent] -= 1;
            if indegree[parent] == 0 {
                queue.push_back(parent);
            }
        }
    }
    if topo.len() != gate_count {
        let not_visited: Vec<usize> = (0..gate_count)
            .filter(|id| !topo.iter().any(|r| r.id == *id))
            .collect();
        (topo, Some(not_visited))
    } else {
        (topo, None)
    }
}

/// Returns a topological order (children before parents) of all nodes in the
/// graph.
pub fn topo_sort_refs(nodes: &[AigNode]) -> Vec<AigRef> {
    let (order, cycle) = topo_order_and_cycle_check(nodes);
    if let Some(not_visited) = cycle {
        panic!(
            "Cycle detected in AIG graph: topological sort visited {} of {} nodes; not visited: {:?}",
            nodes.len() - not_visited.len(),
            nodes.len(),
            not_visited
        );
    }
    order
}

pub fn debug_assert_no_cycles(nodes: &[AigNode], context: &str) {
    if !cfg!(debug_assertions) {
        return;
    }
    let (_order, cycle) = topo_order_and_cycle_check(nodes);
    if let Some(not_visited) = cycle {
        log::error!(
            "[{}] Cycle detected! Not visited: {:?}",
            context,
            not_visited
        );
        for id in &not_visited {
            log::error!("[{}] Node %{}: {:?}", context, id, nodes[*id]);
        }
        panic!(
            "Cycle detected in graph (context: {}) after transformation: topological sort visited {} of {} nodes. See logs for details.",
            context,
            nodes.len() - not_visited.len(),
            nodes.len()
        );
    }
}
