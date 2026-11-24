// SPDX-License-Identifier: Apache-2.0

//! Optional lowering of structural Block IR into a flattened whole-node DAG.
//!
//! This pass is intended for node-at-a-time consumers such as `block2fn`.
//! Emission and hierarchy-preserving transformations never invoke it
//! implicitly.

mod projection;

use std::collections::{HashMap, HashSet};

use crate::ir::{Block, NaryOp, Node, NodeGraph, NodePayload, NodeRef, Package, Type, Unop};
use crate::ir_builder::{
    BuilderError, verify_block_combinational_cycles, verify_block_wiring_cycles,
};
use crate::ir_rebase_ids::package_max_emitted_node_id;
use crate::ir_utils::{operands, remap_payload_with, verify_no_cycle};
use projection::{Demand, Recipe, Selection, needs_aggregate_expansion, project};

/// Produces a flattened, node-acyclic copy of the selected block hierarchy.
///
/// Structural feedback is resolved by projecting only selected wiring ranges
/// and fields. Gates remain whole operations. Actual combinational cycles and
/// opaque external instances are rejected; the input package is never changed.
/// Simplification, when desired, is a separate preceding pass. Array reads
/// and updates use canonical XLS bounds semantics; this is not equivalent to
/// unchecked SV array indexing for out-of-bounds indices. A conservative
/// hierarchy-expansion preflight and proof-helper limits reject impractically
/// large lowerings before they allocate the corresponding graphs.
pub fn legalize_block_hierarchy(package: &Package, top: &str) -> Result<Package, String> {
    check_flatten_budget(package, top)?;
    match crate::block_inline::flatten_block_hierarchy(package, top) {
        Ok(flattened) => Ok(flattened),
        Err(original) => legalize_after_inline_failure(package, top, original),
    }
}

const MAX_EXPANSION_NODES: usize = 1_000_000;

/// Bounds prospective flattened bodies before either inlining path allocates
/// them. Estimates are memoized by definition and charged once per instance;
/// generated intermediate definitions count toward the same allocation limit.
pub(crate) fn check_flatten_budget(package: &Package, top: &str) -> Result<(), String> {
    check_expansion_budget(
        package,
        top,
        ExpansionCost::FlattenedBodies,
        MAX_EXPANSION_NODES,
    )
}

/// Bounds the one fully projected instance graph without charging temporary
/// flattened definitions that evaluation never constructs.
pub(crate) fn check_projection_budget(package: &Package, top: &str) -> Result<(), String> {
    check_expansion_budget(
        package,
        top,
        ExpansionCost::ProjectedHierarchy,
        MAX_EXPANSION_NODES,
    )
}

/// Distinguishes the live intermediate bodies of flattening from a single
/// elaborated projection of the selected hierarchy.
#[derive(Clone, Copy)]
enum ExpansionCost {
    FlattenedBodies,
    ProjectedHierarchy,
}

impl ExpansionCost {
    /// Counts nodes owned by one instance, including simulator state parameters
    /// and the top-level input parameters and result tuple.
    fn local_nodes(self, block: &Block, root: bool) -> Option<usize> {
        match self {
            Self::FlattenedBodies => Some(block.nodes.len()),
            Self::ProjectedHierarchy => {
                let nodes = block.nodes.len().checked_add(block.registers.len())?;
                if root {
                    nodes
                        .checked_add(block.input_ports().count())?
                        .checked_add(2)
                } else {
                    Some(nodes)
                }
            }
        }
    }

    /// Reports the specific prospective allocation that exceeded its limit.
    fn error(self, limit: usize) -> String {
        let kind = match self {
            Self::FlattenedBodies => "flattened-node",
            Self::ProjectedHierarchy => "projected-node",
        };
        format!("block expansion exceeds its {kind} budget ({limit} estimated nodes)")
    }
}

/// Estimates each definition once and charges its expanded size per instance.
fn check_expansion_budget(
    package: &Package,
    top: &str,
    cost: ExpansionCost,
    limit: usize,
) -> Result<(), String> {
    let root = package
        .get_block(top)
        .ok_or_else(|| format!("block '{top}' not found"))?;
    if cost
        .local_nodes(root, true)
        .is_none_or(|nodes| nodes > limit)
    {
        return Err(cost.error(limit));
    }
    if root.instantiations.is_empty() {
        return Ok(());
    }
    let blocks: HashMap<_, _> = package
        .members
        .iter()
        .filter_map(|member| match member {
            crate::ir::PackageMember::Block(block) => Some((block.name.as_str(), block)),
            _ => None,
        })
        .collect();
    let mut states = HashMap::<&str, u8>::new();
    let mut sizes = HashMap::<&str, usize>::new();
    let mut pending = vec![(top, false)];
    let mut allocated = 0usize;
    while let Some((name, complete)) = pending.pop() {
        let block = blocks
            .get(name)
            .ok_or_else(|| format!("missing child block '{name}'"))?;
        if complete {
            let mut size = cost
                .local_nodes(block, name == top)
                .ok_or_else(|| cost.error(limit))?;
            for instance in &block.instantiations {
                let Some(child) = instance.referenced_block() else {
                    continue;
                };
                size = size
                    .checked_add(sizes[child])
                    .ok_or_else(|| cost.error(limit))?;
                if size > limit {
                    return Err(cost.error(limit));
                }
            }
            if matches!(cost, ExpansionCost::FlattenedBodies) && !block.instantiations.is_empty() {
                allocated = allocated
                    .checked_add(size)
                    .ok_or_else(|| cost.error(limit))?;
                if allocated > limit {
                    return Err(cost.error(limit));
                }
            }
            sizes.insert(name, size);
            states.insert(name, 2);
            continue;
        }
        match states.get(name) {
            Some(2) => continue,
            Some(1) => return Err(format!("recursive block instantiation involving '{name}'")),
            _ => {}
        }
        states.insert(name, 1);
        pending.push((name, true));
        pending.extend(
            block
                .instantiations
                .iter()
                .rev()
                .filter_map(|instance| instance.referenced_block().map(|child| (child, false))),
        );
    }
    Ok(())
}

/// Refines a failed ordinary inline only when its selected hierarchy has a node
/// cycle.
pub(crate) fn legalize_after_inline_failure(
    package: &Package,
    top: &str,
    original: String,
) -> Result<Package, String> {
    let block = cycle_candidate(package, top, original)?;
    verify_block_wiring_cycles(block, package).map_err(|error| error.to_string())?;
    finish_legalization(package, top, |_, _| Ok(()))
}

/// Applies consumer-specific bindings and simplification before resolving
/// feedback. The transform mutates only a temporary flattened graph; the source
/// hierarchy is unchanged.
pub(crate) fn legalize_after_inline_failure_with_preprocess(
    package: &Package,
    top: &str,
    original: String,
    preprocess: impl FnOnce(&mut Block, &mut usize) -> Result<(), String>,
) -> Result<Package, String> {
    cycle_candidate(package, top, original)?;
    finish_legalization(package, top, preprocess)
}

/// Returns the original selected top only when failed inlining encountered a
/// node cycle.
fn cycle_candidate<'a>(
    package: &'a Package,
    top: &str,
    original: String,
) -> Result<&'a Block, String> {
    let Some(block) = package.get_block(top) else {
        return Err(original);
    };
    match verify_block_combinational_cycles(block, package) {
        Err(BuilderError::CombinationalCycle { .. }) => Ok(block),
        _ => Err(original),
    }
}

/// Builds, optionally transforms, and validates a proof-only flat wiring graph.
fn finish_legalization(
    package: &Package,
    top: &str,
    preprocess: impl FnOnce(&mut Block, &mut usize) -> Result<(), String>,
) -> Result<Package, String> {
    check_flatten_budget(package, top)?;
    let mut result = crate::block_inline::flatten_block_hierarchy_for_legalization(package, top)?;
    let mut last_id = package_max_emitted_node_id(&result);
    let flattened = result.get_block_mut(top).ok_or("flattened block missing")?;
    preprocess(flattened, &mut last_id)?;
    legalize_node_graph(&mut flattened.graph, last_id)?;
    flattened
        .compact_and_toposort()
        .map_err(|e| format!("block legalization: {e}"))?;
    crate::ir_verify::verify_package(&result)
        .map_err(|e| format!("invalid legalized package: {e}"))?;
    Ok(result)
}

/// Rewrites structural feedback in place without compacting or renumbering
/// existing nodes. Interface owners retain responsibility for sorting the graph
/// and remapping their own metadata.
pub(crate) fn legalize_node_graph(block: &mut NodeGraph, mut last_id: usize) -> Result<(), String> {
    let mut budget = Budget::default();
    while verify_no_cycle(block).is_err() {
        let cyclic = cyclic_nodes(block)?;
        if cut_empty_operands(block, &cyclic, &mut last_id, &mut budget)? {
            continue;
        }
        let mut plan = plan_edits(block, &cyclic, last_id, budget, Expansion::Deferred)?;
        if plan.edits.is_empty() {
            plan = plan_edits(block, &cyclic, last_id, budget, Expansion::Allowed)?;
        }
        let EditPlan {
            pending,
            edits,
            last_id: next_id,
            budget: next_budget,
        } = plan;
        if edits.is_empty() {
            return Err(format!(
                "block '{}': combinational cycle or wiring that cannot be legalized into a whole-node DAG",
                block.name
            ));
        }
        last_id = next_id;
        budget = next_budget;
        block.nodes.extend(pending);
        for (index, payload) in edits {
            block.nodes[index].replace_payload(payload);
        }
    }
    Ok(())
}

const MAX_LEGALIZATION_HELPERS: usize = 100_000;
const MAX_LEGALIZATION_HELPER_BITS: usize = 16 * 1024 * 1024;
const MAX_LEGALIZATION_DEMANDS: usize = 100_000;

/// Limits the size and cumulative bit width of newly materialized proof nodes.
#[derive(Clone, Copy, Default)]
struct Budget {
    nodes: usize,
    bits: usize,
}

impl Budget {
    /// Reserves one helper's storage and proof-relevant bit width.
    fn reserve(&mut self, ty: &Type) -> Result<(), String> {
        let bits = ty
            .checked_bit_count()
            .ok_or("legalization helper bit width overflows usize")?;
        let next_bits = self
            .bits
            .checked_add(bits)
            .ok_or("legalization helper bit budget overflow")?;
        if self.nodes >= MAX_LEGALIZATION_HELPERS || next_bits > MAX_LEGALIZATION_HELPER_BITS {
            return Err(format!(
                "legalization exceeds its proof-graph budget ({} helper nodes, {} cumulative helper bits)",
                MAX_LEGALIZATION_HELPERS, MAX_LEGALIZATION_HELPER_BITS
            ));
        }
        self.nodes += 1;
        self.bits = next_bits;
        Ok(())
    }
}

/// Whether a projection may materialize the elements of a mixed array.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Expansion {
    Deferred,
    Allowed,
}

/// Helper nodes and original-node edits planned against one graph snapshot.
struct EditPlan {
    pending: Vec<Node>,
    edits: Vec<(usize, NodePayload)>,
    last_id: usize,
    budget: Budget,
}

/// Plans rewrites against an immutable snapshot, postponing aggregate
/// expansion.
fn plan_edits(
    block: &NodeGraph,
    cyclic: &[bool],
    last_id: usize,
    budget: Budget,
    expand: Expansion,
) -> Result<EditPlan, String> {
    let mut resolver = Resolver::new(block, cyclic, last_id, budget, expand);
    let mut edits = Vec::new();
    for (index, &in_cycle) in cyclic.iter().enumerate() {
        if !in_cycle {
            continue;
        }
        let root = Demand::whole(block, NodeRef { index });
        if expand == Expansion::Deferred && needs_aggregate_expansion(block, &root) {
            continue;
        }
        let checkpoint = resolver.checkpoint();
        let Some(recipe) = checked_project(block, &root)? else {
            continue;
        };
        let mut ready = true;
        for demand in recipe.inputs() {
            if !resolver.resolve(demand.clone())? {
                ready = false;
                break;
            }
        }
        if !ready {
            resolver.rollback(checkpoint);
            continue;
        }
        let payload = resolver.payload(&root, recipe)?;
        if payload != block.nodes[index].payload {
            edits.push((index, payload));
        } else {
            resolver.rollback(checkpoint);
        }
    }
    Ok(EditPlan {
        pending: resolver.pending,
        edits,
        last_id: resolver.last_id,
        budget: resolver.budget,
    })
}

/// Applies the same explicit array-expansion budget as structural cycle
/// checking.
fn checked_project(block: &NodeGraph, demand: &Demand) -> Result<Option<Recipe>, String> {
    if needs_aggregate_expansion(block, demand)
        && let Some(Type::Array(array)) = demand.part.selected_type(block.get_node_ty(demand.node))
        && array.element_count > crate::ir_builder::MAX_WIRING_ARRAY_ELEMENTS
    {
        return Err(format!(
            "array legalization would expand {} elements (limit {})",
            array.element_count,
            crate::ir_builder::MAX_WIRING_ARRAY_ELEMENTS
        ));
    }
    Ok(project(block, demand))
}

/// Marks exactly the nodes in a cyclic strongly connected component.
fn cyclic_nodes(block: &NodeGraph) -> Result<Vec<bool>, String> {
    let edges: Vec<_> = block.nodes.iter().map(|n| operands(&n.payload)).collect();
    let mut visited = vec![false; edges.len()];
    let mut order = Vec::with_capacity(edges.len());
    for root in 0..edges.len() {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        let mut stack = vec![(root, 0)];
        while let Some((node, next)) = stack.last_mut() {
            if let Some(arg) = edges[*node].get(*next) {
                *next += 1;
                let slot = visited
                    .get_mut(arg.index)
                    .ok_or_else(|| format!("missing node {}", arg.index))?;
                if !*slot {
                    *slot = true;
                    stack.push((arg.index, 0));
                }
            } else {
                order.push(*node);
                stack.pop();
            }
        }
    }
    let mut users = vec![Vec::new(); edges.len()];
    for (node, args) in edges.iter().enumerate() {
        for arg in args {
            users[arg.index].push(node);
        }
    }
    visited.fill(false);
    let mut cyclic = vec![false; edges.len()];
    for root in order.into_iter().rev() {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        let mut component = vec![root];
        let mut cursor = 0;
        while cursor < component.len() {
            for &user in &users[component[cursor]] {
                if !visited[user] {
                    visited[user] = true;
                    component.push(user);
                }
            }
            cursor += 1;
        }
        if component.len() > 1 || edges[root].contains(&NodeRef { index: root }) {
            for node in component {
                cyclic[node] = true;
            }
        }
    }
    Ok(cyclic)
}

/// Replaces empty data dependencies without discarding their producers or
/// effects.
fn cut_empty_operands(
    block: &mut NodeGraph,
    cyclic: &[bool],
    last_id: &mut usize,
    budget: &mut Budget,
) -> Result<bool, String> {
    let mut zeros = HashMap::<Type, NodeRef>::new();
    let original_len = block.nodes.len();
    for index in 0..original_len {
        if !cyclic[index] {
            continue;
        }
        let args = operands(&block.nodes[index].payload);
        for &arg in &args {
            let ty = block.get_node_ty(arg);
            if cyclic[arg.index]
                && !ty.contains_token()
                && ty.checked_bit_count() == Some(0)
                && !zeros.contains_key(ty)
            {
                let ty = ty.clone();
                budget.reserve(&ty)?;
                *last_id = last_id
                    .checked_add(1)
                    .ok_or("legalization node IDs overflow usize")?;
                let nr = NodeRef {
                    index: block.nodes.len(),
                };
                block.nodes.push(Node {
                    text_id: *last_id,
                    name: None,
                    pos: None,
                    payload: NodePayload::Literal(zero_value_for_legalization(&ty)?),
                    ty: ty.clone(),
                });
                zeros.insert(ty, nr);
            }
        }
        let payload = remap_payload_with(&block.nodes[index].payload, |(_, arg)| {
            if cyclic[arg.index] {
                zeros.get(block.get_node_ty(arg)).copied().unwrap_or(arg)
            } else {
                arg
            }
        });
        block.nodes[index].replace_payload(payload);
    }
    Ok(!zeros.is_empty())
}

/// Materializes a zero value only when its aggregate representation is bounded.
pub(crate) fn zero_value_for_legalization(ty: &Type) -> Result<crate::IrValue, String> {
    fn nodes(ty: &Type) -> Option<usize> {
        match ty {
            Type::Bits(_) => Some(1),
            Type::Tuple(fields) => fields
                .iter()
                .try_fold(1usize, |sum, field| sum.checked_add(nodes(field)?)),
            Type::Array(array) => nodes(&array.element_type)?
                .checked_mul(array.element_count)?
                .checked_add(1),
            _ => None,
        }
    }
    if nodes(ty).is_none_or(|count| count > 256) {
        return Err(
            "legalization would materialize an excessively large empty aggregate".to_string(),
        );
    }
    Ok(crate::ir_value_utils::zero_ir_value_for_type(ty))
}

/// Memoizes exact selections against an immutable graph snapshot.
struct Resolver<'a> {
    block: &'a NodeGraph,
    cyclic: &'a [bool],
    cache: HashMap<Demand, NodeRef>,
    pending: Vec<Node>,
    last_id: usize,
    budget: Budget,
    expand: Expansion,
}

impl<'a> Resolver<'a> {
    fn new(
        block: &'a NodeGraph,
        cyclic: &'a [bool],
        last_id: usize,
        budget: Budget,
        expand: Expansion,
    ) -> Self {
        Self {
            block,
            cyclic,
            cache: HashMap::new(),
            pending: Vec::new(),
            last_id,
            budget,
            expand,
        }
    }

    /// Saves the staged helper boundary before planning an individual rewrite.
    fn checkpoint(&self) -> (usize, usize, Budget) {
        (self.pending.len(), self.last_id, self.budget)
    }

    /// Discards helpers that belong only to an unsuccessful or unchanged
    /// rewrite.
    fn rollback(&mut self, (len, id, budget): (usize, usize, Budget)) {
        if self.pending.len() == len {
            return;
        }
        self.pending.truncate(len);
        self.last_id = id;
        self.budget = budget;
        let end = self.block.nodes.len() + len;
        self.cache.retain(|_, node| node.index < end);
    }

    /// Resolves selected operands iteratively, avoiding call-stack depth
    /// limits.
    fn resolve(&mut self, root: Demand) -> Result<bool, String> {
        let mut active = HashSet::new();
        let mut stack = vec![(root, None::<Recipe>)];
        while let Some((demand, ready)) = stack.pop() {
            if self.cache.contains_key(&demand) {
                continue;
            }
            if let Some(recipe) = ready {
                active.remove(&demand);
                let node = if let Recipe::Forward(arg) = recipe {
                    self.cache[&arg]
                } else {
                    let payload = self.payload(&demand, recipe)?;
                    self.push(
                        demand.node,
                        demand
                            .part
                            .selected_type(self.block.get_node_ty(demand.node))
                            .ok_or("invalid wiring selection")?,
                        payload,
                    )?
                };
                self.cache.insert(demand, node);
                continue;
            }
            if demand.part == Selection::whole(self.block.get_node_ty(demand.node)) {
                self.cache.insert(demand.clone(), demand.node);
                continue;
            }
            if self.expand == Expansion::Deferred && needs_aggregate_expansion(self.block, &demand)
            {
                return Ok(false);
            }
            if self.cyclic[demand.node.index]
                && let Some(recipe) = checked_project(self.block, &demand)?
            {
                if self.cache.len().saturating_add(active.len()) >= MAX_LEGALIZATION_DEMANDS {
                    return Err(format!(
                        "legalization exceeds its selection budget ({MAX_LEGALIZATION_DEMANDS} demands)"
                    ));
                }
                if !active.insert(demand.clone()) {
                    return Err(format!(
                        "block '{}': cyclic wiring selection at node {}",
                        self.block.name, demand.node.index
                    ));
                }
                let inputs: Vec<_> = recipe.inputs().to_vec();
                stack.push((demand, Some(recipe)));
                for input in inputs.into_iter().rev() {
                    stack.push((input, None));
                }
            } else {
                let node = self.select(&demand)?;
                self.cache.insert(demand, node);
            }
        }
        Ok(true)
    }

    /// Converts a wiring recipe into a payload referring to resolved operands.
    fn payload(&mut self, demand: &Demand, recipe: Recipe) -> Result<NodePayload, String> {
        Ok(match recipe {
            Recipe::Forward(arg) => NodePayload::Unop(Unop::Identity, self.cache[&arg]),
            Recipe::Concat(args) => NodePayload::Nary(
                NaryOp::Concat,
                args.iter().map(|arg| self.cache[arg]).collect(),
            ),
            Recipe::Reverse(arg) => NodePayload::Unop(Unop::Reverse, self.cache[&arg]),
            Recipe::Zero(width) => {
                if width <= 4096 {
                    NodePayload::Literal(crate::IrValue::Bits(crate::IrBits::zero(width)))
                } else {
                    let zero = self.push(
                        demand.node,
                        Type::Bits(1),
                        NodePayload::Literal(crate::IrValue::Bits(crate::IrBits::zero(1))),
                    )?;
                    NodePayload::ZeroExt {
                        arg: zero,
                        new_bit_count: width,
                    }
                }
            }
            Recipe::Extend { arg, signed, width } => {
                if signed {
                    NodePayload::SignExt {
                        arg: self.cache[&arg],
                        new_bit_count: width,
                    }
                } else {
                    NodePayload::ZeroExt {
                        arg: self.cache[&arg],
                        new_bit_count: width,
                    }
                }
            }
            Recipe::Tuple(args) => {
                NodePayload::Tuple(args.iter().map(|arg| self.cache[arg]).collect())
            }
            Recipe::Array(args) => {
                NodePayload::Array(args.iter().map(|arg| self.cache[arg]).collect())
            }
        })
    }

    /// Materializes a selection without distributing a gate or dynamic
    /// operation.
    fn select(&mut self, demand: &Demand) -> Result<NodeRef, String> {
        let mut current = demand.node;
        let mut ty = self.block.get_node_ty(current).clone();
        for &index in &demand.part.path {
            let (next_ty, payload) = match &ty {
                Type::Tuple(fields) => (
                    fields
                        .get(index)
                        .ok_or("tuple field out of bounds")?
                        .as_ref()
                        .clone(),
                    NodePayload::TupleIndex {
                        tuple: current,
                        index,
                    },
                ),
                Type::Array(array) if index < array.element_count => {
                    let width = (usize::BITS - index.leading_zeros()).max(1) as usize;
                    let offset = self.push(
                        demand.node,
                        Type::Bits(width),
                        NodePayload::Literal(
                            crate::IrValue::make_ubits(width, index as u64)
                                .map_err(|e| e.to_string())?,
                        ),
                    )?;
                    (
                        array.element_type.as_ref().clone(),
                        NodePayload::ArrayIndex {
                            array: current,
                            indices: vec![offset],
                            assumed_in_bounds: true,
                        },
                    )
                }
                _ => return Err("invalid aggregate selection".to_string()),
            };
            current = self.push(demand.node, next_ty.clone(), payload)?;
            ty = next_ty;
        }
        if let Some((lo, hi)) = demand.part.bits {
            let Type::Bits(width) = ty else {
                return Err("invalid bit selection".to_string());
            };
            if hi > width || lo > hi {
                return Err("bit selection out of bounds".to_string());
            }
            if lo != 0 || hi != width {
                current = self.push(
                    demand.node,
                    Type::Bits(hi - lo),
                    NodePayload::BitSlice {
                        arg: current,
                        start: lo,
                        width: hi - lo,
                    },
                )?;
            }
        }
        Ok(current)
    }

    /// Appends an unnamed helper with a package-wide unique textual ID.
    fn push(&mut self, origin: NodeRef, ty: Type, payload: NodePayload) -> Result<NodeRef, String> {
        self.budget.reserve(&ty)?;
        self.last_id = self
            .last_id
            .checked_add(1)
            .ok_or("legalization node IDs overflow usize")?;
        let node = NodeRef {
            index: self.block.nodes.len() + self.pending.len(),
        };
        self.pending.push(Node {
            text_id: self.last_id,
            name: None,
            ty,
            payload,
            pos: self.block.get_node(origin).pos.clone(),
        });
        Ok(node)
    }
}

#[cfg(test)]
mod expansion_tests {
    use super::*;
    use crate::{BlockBuilder, ir::Type};

    /// A unary hierarchy has one small projected graph but many flattened
    /// intermediate bodies that must be charged only to flattening.
    #[test]
    fn projection_does_not_charge_flattening_intermediates() {
        let mut leaf = BlockBuilder::new("level0");
        let d = leaf.input_port("d", Type::Bits(1)).unwrap();
        leaf.output_port("q", d).unwrap();
        let mut package = leaf.build_package("unary").unwrap();
        assert!(
            check_expansion_budget(&package, "level0", ExpansionCost::ProjectedHierarchy, 3)
                .is_err()
        );
        for depth in 1..=10 {
            let mut parent = BlockBuilder::new(&format!("level{depth}"));
            let d = parent.input_port("d", Type::Bits(1)).unwrap();
            let child = parent
                .instantiate(
                    "u",
                    package.get_block(&format!("level{}", depth - 1)).unwrap(),
                )
                .unwrap();
            parent.instantiation_input(child, "d", d).unwrap();
            let q = parent.instantiation_output(child, "q").unwrap();
            parent.output_port("q", q).unwrap();
            parent.build_into_package(&mut package).unwrap();
        }
        assert!(
            check_expansion_budget(&package, "level10", ExpansionCost::FlattenedBodies, 100)
                .is_err()
        );
        check_expansion_budget(&package, "level10", ExpansionCost::ProjectedHierarchy, 100)
            .unwrap();
    }
}
