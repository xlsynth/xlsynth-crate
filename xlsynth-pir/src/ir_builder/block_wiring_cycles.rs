// SPDX-License-Identifier: Apache-2.0

//! Structural feedback analysis for block emission. Unlike the ordinary PIR
//! cycle checker, this checker can follow individual wires within a value.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::block_cycles::InoutTopology;
use super::{BlockCycleConnection, BlockCycleStep, BuilderError};
use crate::ir::{Block, NaryOp, NodePayload, NodeRef, Package, Type, Unop};
use crate::ir_utils::operands;

/// Checks compositional block connectivity at wiring granularity.
///
/// The ordinary node-level checker remains the fast path and the validity
/// requirement for consumers that evaluate complete PIR values. If that check
/// finds a cycle, this analysis refines pure wiring operations into their
/// selected bit ranges and aggregate fields. All other operations
/// conservatively depend on their complete operands. Registers and opaque
/// external instances cut traversal; modeled externs use their behavioral
/// block. The input hierarchy is inspected without flattening or specializing
/// blocks.
pub fn verify_block_wiring_cycles(block: &Block, package: &Package) -> Result<(), BuilderError> {
    verify_block_wiring_cycles_with_options(block, package, &BlockWiringCycleOptions::default())
}

/// Array indexing semantics used by the circuit consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockWiringCycleOptions {
    /// Whether out-of-bounds indices clamp to the final array element.
    pub array_index_bounds_checking: bool,
}

impl Default for BlockWiringCycleOptions {
    fn default() -> Self {
        Self {
            array_index_bounds_checking: true,
        }
    }
}

/// Checks wiring cycles using the array-index semantics of the target backend.
/// When bounds checking is disabled, out-of-bounds indices are conservative.
pub fn verify_block_wiring_cycles_with_options(
    block: &Block,
    package: &Package,
    options: &BlockWiringCycleOptions,
) -> Result<(), BuilderError> {
    match super::block_cycles::verify_block_combinational_cycles(block, package) {
        Ok(()) => Ok(()),
        Err(original @ BuilderError::CombinationalCycle { .. }) => {
            BlockWiringHierarchy::new(*options).check(block, package, &original)
        }
        Err(error) => Err(error),
    }
}

/// One distinct block definition, with its instance connections indexed.
struct Context<'a> {
    block: &'a Block,
    inputs: HashMap<(&'a str, &'a str), NodeRef>,
    children: HashMap<&'a str, usize>,
    externs: HashSet<&'a str>,
    inouts: Option<InoutTopology<'a>>,
    net_drives: Vec<Vec<NodeRef>>,
    net_children: Vec<Vec<(&'a str, &'a str)>>,
    net_witnesses: Vec<Option<NodeRef>>,
}

type SummaryKey = (usize, String, Part);
type InputDependency = (String, Part);

/// Pending work for one lazily computed selected child-output summary.
struct SummaryFrame {
    key: SummaryKey,
    root: Demand,
    pending: Vec<(Demand, usize)>,
    visited: HashSet<Demand>,
    inputs: BTreeSet<InputDependency>,
}

/// Summaries map selected child outputs to selected formal inputs. They are
/// independent of parent tie-offs, so each definition can be analyzed once.
struct Hierarchy<'a> {
    contexts: Vec<Context<'a>>,
    summaries: HashMap<SummaryKey, Vec<InputDependency>>,
    options: BlockWiringCycleOptions,
    budget: RefinementBudget,
    cached_dependencies: usize,
}

/// Caches refined contexts and boundary summaries across package verification.
pub(crate) struct BlockWiringHierarchy<'a> {
    hierarchy: Hierarchy<'a>,
    definitions: HashMap<&'a str, usize>,
    state: Vec<u8>,
}

impl<'a> BlockWiringHierarchy<'a> {
    /// Creates a lazy refinement cache; ordinary checking need not construct
    /// it.
    pub(crate) fn new(options: BlockWiringCycleOptions) -> Self {
        Self {
            hierarchy: Hierarchy {
                contexts: Vec::new(),
                summaries: HashMap::new(),
                options,
                budget: RefinementBudget::default(),
                cached_dependencies: 0,
            },
            definitions: HashMap::new(),
            state: Vec::new(),
        }
    }

    /// Checks one newly encountered root, reusing previously checked
    /// definitions.
    pub(crate) fn check(
        &mut self,
        root: &'a Block,
        package: &'a Package,
        original: &BuilderError,
    ) -> Result<(), BuilderError> {
        self.hierarchy.budget.begin_root();
        let root_index = self.index_root(root, package)?;
        self.state.resize(self.hierarchy.contexts.len(), 0);
        self.hierarchy
            .check_context(root_index, &mut self.state, original)
    }

    /// Indexes each reachable definition and its modeled instance bindings
    /// once.
    fn index_root(&mut self, root: &'a Block, package: &'a Package) -> Result<usize, BuilderError> {
        if let Some(&index) = self.definitions.get(root.name.as_str()) {
            return Ok(index);
        }
        let first = self.hierarchy.contexts.len();
        self.definitions.insert(root.name.as_str(), first);
        self.hierarchy.contexts.push(Context::new(root));
        let mut index = first;
        while index < self.hierarchy.contexts.len() {
            let block = self.hierarchy.contexts[index].block;
            for instance in &block.instantiations {
                let Some(child_name) = instance.referenced_block() else {
                    self.hierarchy.contexts[index]
                        .externs
                        .insert(&instance.name);
                    continue;
                };
                let child = package
                    .get_block(child_name)
                    .ok_or_else(|| invalid(format!("missing child block '{child_name}'")))?;
                if child.clock_port_name().is_some() && block.clock_port_name().is_none() {
                    return Err(invalid(format!(
                        "clocked child '{}' requires a clock on parent block '{}'",
                        child.name, block.name
                    )));
                }
                let child_index = if let Some(&id) = self.definitions.get(child.name.as_str()) {
                    id
                } else {
                    let id = self.hierarchy.contexts.len();
                    self.definitions.insert(child.name.as_str(), id);
                    self.hierarchy.contexts.push(Context::new(child));
                    id
                };
                self.hierarchy.contexts[index]
                    .children
                    .insert(&instance.name, child_index);
            }
            index += 1;
        }
        Ok(first)
    }
}

const MAX_REFINEMENT_STATES: usize = 100_000;
const MAX_REFINEMENT_EDGES: usize = 1_000_000;
const MAX_CACHED_SUMMARIES: usize = 100_000;
const MAX_CACHED_DEPENDENCIES: usize = 1_000_000;

/// Bounds selected-state work independently for each analyzed root.
struct RefinementBudget {
    states: usize,
    edges: usize,
    max_states: usize,
    max_edges: usize,
}

impl Default for RefinementBudget {
    fn default() -> Self {
        Self {
            states: 0,
            edges: 0,
            max_states: MAX_REFINEMENT_STATES,
            max_edges: MAX_REFINEMENT_EDGES,
        }
    }
}

impl RefinementBudget {
    /// Starts an independent per-root work budget while preserving cache
    /// limits.
    fn begin_root(&mut self) {
        self.states = 0;
        self.edges = 0;
    }

    /// Charges one newly explored selected node or boundary summary.
    fn state(&mut self) -> Result<(), BuilderError> {
        self.states = self
            .states
            .checked_add(1)
            .ok_or_else(|| invalid("wiring refinement state count overflow"))?;
        if self.states > self.max_states {
            return Err(invalid(format!(
                "wiring cycle refinement exceeds its selected-state budget ({})",
                self.max_states
            )));
        }
        Ok(())
    }

    /// Refuses excessive fan-in before allocating or cloning selected edges.
    fn preflight_edges(&self, count: usize) -> Result<(), BuilderError> {
        if self
            .edges
            .checked_add(count)
            .is_none_or(|next| next > self.max_edges)
        {
            return Err(invalid(format!(
                "wiring cycle refinement exceeds its selected-edge budget ({})",
                self.max_edges
            )));
        }
        Ok(())
    }

    /// Charges selected-edge work after checking its allocation bound.
    fn edges(&mut self, count: usize) -> Result<(), BuilderError> {
        self.preflight_edges(count)?;
        self.edges += count;
        Ok(())
    }
}

impl<'a> Hierarchy<'a> {
    /// Checks child definitions before parents without consuming the Rust
    /// stack.
    fn check_context(
        &mut self,
        root: usize,
        state: &mut [u8],
        original: &BuilderError,
    ) -> Result<(), BuilderError> {
        let mut pending = vec![(root, false)];
        while let Some((context, complete)) = pending.pop() {
            if complete {
                let block = self.contexts[context].block;
                if let Err(reason) = crate::ir_utils::verify_no_cycle(block) {
                    if matches!(original, BuilderError::CombinationalCycle { block: name, .. } if name == &block.name)
                    {
                        return Err(original.clone());
                    }
                    return Err(invalid(reason));
                }
                self.index_inouts(context)?;
                self.check_graph(context)?;
                state[context] = 2;
                continue;
            }
            match state[context] {
                2 => continue,
                1 => {
                    return Err(invalid(format!(
                        "recursive block instantiation involving '{}'",
                        self.contexts[context].block.name
                    )));
                }
                _ => {}
            }
            state[context] = 1;
            pending.push((context, true));
            let scope = &self.contexts[context];
            for instance in scope.block.instantiations.iter().rev() {
                if let Some(&child) = scope.children.get(instance.name.as_str()) {
                    pending.push((child, false));
                }
            }
        }
        Ok(())
    }

    /// Indexes child aliases, directed local drivers, and modeled child drives.
    fn index_inouts(&mut self, context: usize) -> Result<(), BuilderError> {
        let scope = &self.contexts[context];
        let block = scope.block;
        let definitions = block
            .instantiations
            .iter()
            .map(|instance| (instance.name.as_str(), instance))
            .collect::<BTreeMap<_, _>>();
        let topology = InoutTopology::new(block, &definitions, |instance, port| {
            let child = *scope.children.get(instance.name.as_str())?;
            self.contexts[child].inouts.as_ref()?.canonical_port(port)
        })?;
        let class_count = topology.classes.len();
        let mut drives = vec![Vec::new(); class_count];
        let mut children = vec![Vec::new(); class_count];
        let mut witnesses = vec![None; class_count];
        let mut inputs = Vec::new();
        for (index, node) in block.nodes.iter().enumerate() {
            let reference = NodeRef { index };
            match &node.payload {
                NodePayload::ExtInoutRead { net } | NodePayload::ExtInoutDrive { net, .. } => {
                    let class = *topology
                        .local_nets
                        .get(net.as_str())
                        .ok_or_else(|| invalid(format!("undeclared inout net '{net}'")))?;
                    witnesses[class].get_or_insert(reference);
                    if matches!(node.payload, NodePayload::ExtInoutDrive { .. }) {
                        drives[class].push(reference);
                    }
                }
                NodePayload::ExtInstantiationInout {
                    instantiation,
                    port_name,
                    net,
                } => {
                    let class = *topology
                        .local_nets
                        .get(net.as_str())
                        .ok_or_else(|| invalid(format!("undeclared inout net '{net}'")))?;
                    witnesses[class].get_or_insert(reference);
                    if let Some(&child) = scope.children.get(instantiation.as_str())
                        && self.contexts[child]
                            .inouts
                            .as_ref()
                            .and_then(|t| t.canonical_port(port_name))
                            == Some(port_name.as_str())
                    {
                        children[class].push((instantiation.as_str(), port_name.as_str()));
                        inputs.push((
                            (instantiation.as_str(), port_name.as_str()),
                            NodeRef {
                                index: block.nodes.len() + class,
                            },
                        ));
                    }
                }
                _ => {}
            }
        }
        let scope = &mut self.contexts[context];
        scope.inputs.extend(inputs);
        scope.inouts = Some(topology);
        scope.net_drives = drives;
        scope.net_children = children;
        scope.net_witnesses = witnesses;
        Ok(())
    }

    /// Refines instance outputs and inout reads/nets in conservative feedback
    /// components.
    fn check_graph(&mut self, context: usize) -> Result<(), BuilderError> {
        let block = self.contexts[context].block;
        let mut done = HashSet::new();
        let mut active = HashMap::new();
        let mut path = Vec::<Frame>::new();
        for index in self.feedback_outputs(context) {
            let ty = if let Some(node) = block.nodes.get(index) {
                &node.ty
            } else {
                self.contexts[context]
                    .inouts
                    .as_ref()
                    .expect("inouts indexed")
                    .classes[index - block.nodes.len()]
                .ty
            };
            let root = Demand::value(context, NodeRef { index }, Part::whole(ty));
            if done.contains(&root) {
                continue;
            }
            self.budget.state()?;
            active.insert(root.clone(), 0);
            path.push(Frame::new(root, self)?);
            while let Some(frame) = path.last_mut() {
                if let Some(next) = frame.dependencies.next() {
                    if done.contains(&next) {
                        continue;
                    }
                    if let Some(&start) = active.get(&next) {
                        return Err(self.cycle_error(block, &path[start..]));
                    }
                    self.budget.state()?;
                    active.insert(next.clone(), path.len());
                    path.push(Frame::new(next, self)?);
                } else {
                    let finished = path.pop().expect("active frame").demand;
                    active.remove(&finished);
                    done.insert(finished);
                }
            }
        }
        Ok(())
    }

    /// Finds conservative feedback components with linear-size instance hubs
    /// and one vertex per passively resolved inout net.
    fn feedback_outputs(&self, context: usize) -> Vec<usize> {
        let scope = &self.contexts[context];
        let block = scope.block;
        let real_nodes = block.nodes.len();
        let classes = scope.inouts.as_ref().expect("inouts indexed").classes.len();
        let mut hubs = HashMap::new();
        for instance in &block.instantiations {
            if instance.referenced_block().is_some() {
                let index = real_nodes + classes + hubs.len();
                hubs.insert(instance.name.as_str(), index);
            }
        }
        let count = real_nodes + classes + hubs.len();
        let mut graph = vec![Vec::new(); count];
        let mut reverse = vec![Vec::new(); count];
        let mut edge = |from: usize, to: usize| {
            graph[from].push(to);
            reverse[to].push(from);
        };
        for (index, node) in block.nodes.iter().enumerate() {
            for operand in operands(&node.payload) {
                edge(index, operand.index);
            }
            match &node.payload {
                NodePayload::InstantiationOutput { instantiation, .. } => {
                    if let Some(&hub) = hubs.get(instantiation.as_str()) {
                        edge(index, hub);
                    }
                }
                NodePayload::InstantiationInput { instantiation, .. } => {
                    if let Some(&hub) = hubs.get(instantiation.as_str()) {
                        edge(hub, index);
                    }
                }
                NodePayload::ExtInoutRead { net } => {
                    let class =
                        scope.inouts.as_ref().expect("inouts indexed").local_nets[net.as_str()];
                    edge(index, real_nodes + class);
                }
                NodePayload::ExtInoutDrive { net, .. } => {
                    let class =
                        scope.inouts.as_ref().expect("inouts indexed").local_nets[net.as_str()];
                    edge(real_nodes + class, index);
                }
                _ => {}
            }
        }
        for (class, bindings) in scope.net_children.iter().enumerate() {
            for &(instance, _) in bindings {
                if let Some(&hub) = hubs.get(instance) {
                    edge(real_nodes + class, hub);
                    edge(hub, real_nodes + class);
                }
            }
        }
        let mut visited = vec![false; count];
        let mut order = Vec::with_capacity(count);
        for root in 0..count {
            if visited[root] {
                continue;
            }
            visited[root] = true;
            let mut stack = vec![(root, 0)];
            while let Some((node, next)) = stack.last_mut() {
                if let Some(&child) = graph[*node].get(*next) {
                    *next += 1;
                    if !visited[child] {
                        visited[child] = true;
                        stack.push((child, 0));
                    }
                } else {
                    order.push(*node);
                    stack.pop();
                }
            }
        }
        let mut component = vec![usize::MAX; count];
        let mut sizes = Vec::new();
        for root in order.into_iter().rev() {
            if component[root] != usize::MAX {
                continue;
            }
            let id = sizes.len();
            let mut size = 0;
            let mut stack = vec![root];
            component[root] = id;
            while let Some(node) = stack.pop() {
                size += 1;
                for &child in &reverse[node] {
                    if component[child] == usize::MAX {
                        component[child] = id;
                        stack.push(child);
                    }
                }
            }
            sizes.push(size);
        }
        (0..real_nodes + classes)
            .filter(|&index| {
                let candidate = index >= real_nodes
                    || matches!(
                        block.nodes[index].payload,
                        NodePayload::InstantiationOutput { .. } | NodePayload::ExtInoutRead { .. }
                    );
                candidate && (sizes[component[index]] > 1 || graph[index].contains(&index))
            })
            .collect()
    }

    /// Memoizes boundary input parts used by a selected ordinary output or
    /// locally driven public inout. Only the initial zero-length path from an
    /// inout to itself is excluded; selected paths through its drivers remain.
    fn summary(
        &mut self,
        context: usize,
        port: &str,
        part: &Part,
    ) -> Result<Vec<InputDependency>, BuilderError> {
        let key = (context, port.to_owned(), part.clone());
        if let Some(result) = self.summaries.get(&key) {
            self.budget.edges(result.len())?;
            return Ok(result.clone());
        }

        let mut frames = vec![self.summary_frame(key)?];
        loop {
            let frame = frames.last_mut().expect("active summary frame");
            let Some((demand, cursor)) = frame.pending.pop() else {
                let frame = frames.pop().expect("completed summary frame");
                self.budget.preflight_edges(frame.inputs.len())?;
                let result: Vec<_> = frame.inputs.into_iter().collect();
                if self.summaries.len() >= MAX_CACHED_SUMMARIES
                    || self.cached_dependencies.saturating_add(result.len())
                        > MAX_CACHED_DEPENDENCIES
                {
                    self.summaries.clear();
                    self.cached_dependencies = 0;
                }
                self.cached_dependencies += result.len();
                self.summaries.insert(frame.key, result.clone());
                if frames.is_empty() {
                    self.budget.edges(result.len())?;
                    return Ok(result);
                }
                continue;
            };
            if cursor == 0 {
                if !frame.visited.insert(demand.clone()) {
                    continue;
                }
                // Charge the selected demand before descending into a child's
                // summary so the state budget also bounds pending frames.
                self.budget.state()?;
            }
            // Child summaries must be ready before dependencies() visits the
            // demand. Keep the binding cursor so wide inout nets are scanned
            // once while their children are prepared.
            if let Some((key, next)) = self.missing_summary(&demand, cursor) {
                frames
                    .last_mut()
                    .expect("active summary frame")
                    .pending
                    .push((demand, next));
                frames.push(self.summary_frame(key)?);
                continue;
            }
            // Computing a child can evict older cache entries. Recheck the
            // completed prefix before calling dependencies(), which expects
            // every child summary to be available without recursive work.
            if cursor != 0
                && let Some((key, next)) = self.missing_summary(&demand, 0)
            {
                frames
                    .last_mut()
                    .expect("active summary frame")
                    .pending
                    .push((demand, next));
                frames.push(self.summary_frame(key)?);
                continue;
            }
            let frame = frames.last_mut().expect("active summary frame");
            let scope = &self.contexts[demand.context];
            let block = scope.block;
            if let Some(node) = block.nodes.get(demand.node.index) {
                if let NodePayload::InputPort { name, .. } = &node.payload {
                    frame.inputs.insert((name.clone(), demand.part));
                    continue;
                }
            } else {
                let class = demand.node.index - block.nodes.len();
                if demand != frame.root
                    && let Some(name) =
                        scope.inouts.as_ref().expect("inouts indexed").classes[class].public_name
                {
                    frame.inputs.insert((name.to_owned(), demand.part.clone()));
                }
            }
            let dependencies = self.dependencies(&demand)?;
            frames
                .last_mut()
                .expect("active summary frame")
                .pending
                .extend(dependencies.into_iter().map(|demand| (demand, 0)));
        }
    }

    /// Starts an explicit work frame for an ordinary output or public inout.
    fn summary_frame(&self, key: SummaryKey) -> Result<SummaryFrame, BuilderError> {
        let (context, port, part) = &key;
        let scope = &self.contexts[*context];
        let block = scope.block;
        let root_net = scope
            .inouts
            .as_ref()
            .expect("inouts indexed")
            .public_nets
            .get(port.as_str())
            .copied();
        let root = if let Some(class) = root_net {
            Demand::value(
                *context,
                NodeRef {
                    index: block.nodes.len() + class,
                },
                part.clone(),
            )
        } else {
            let sink = block.get_output_port(port).ok_or_else(|| {
                invalid(format!("unknown output '{port}' on child '{}'", block.name))
            })?;
            Demand::driver(*context, sink, part.clone())
        };
        Ok(SummaryFrame {
            key,
            pending: vec![(root.clone(), 0)],
            root,
            visited: HashSet::new(),
            inputs: BTreeSet::new(),
        })
    }

    /// Finds the next uncached child boundary needed to evaluate one demand.
    /// Keep this preflight in sync with the child-summary calls in
    /// `dependencies_inner`; those calls must hit the cache during a summary.
    fn missing_summary(&self, demand: &Demand, cursor: usize) -> Option<(SummaryKey, usize)> {
        let scope = &self.contexts[demand.context];
        if demand.node.index >= scope.block.nodes.len() {
            let class = demand.node.index - scope.block.nodes.len();
            for (index, &(instance, port)) in
                scope.net_children[class].iter().enumerate().skip(cursor)
            {
                let child = scope.children[instance];
                let key = (child, port.to_owned(), demand.part.clone());
                if !self.summaries.contains_key(&key) {
                    return Some((key, index + 1));
                }
            }
        } else if cursor == 0
            && !demand.driver
            && !is_empty(&scope.block.nodes[demand.node.index].ty, &demand.part)
            && let NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } = &scope.block.nodes[demand.node.index].payload
            && let Some(&child) = scope.children.get(instantiation.as_str())
        {
            let key = (child, port_name.clone(), demand.part.clone());
            if !self.summaries.contains_key(&key) {
                return Some((key, 1));
            }
        }
        None
    }

    /// Maps a child's ordinary input or inout boundary to its parent driver.
    fn connected_input(
        &self,
        context: usize,
        instance: &str,
        port: &str,
        part: Part,
    ) -> Result<Demand, BuilderError> {
        let scope = &self.contexts[context];
        let reference = *scope.inputs.get(&(instance, port)).ok_or_else(|| {
            invalid(format!(
                "unconnected input '{port}' on instance '{instance}'"
            ))
        })?;
        Ok(if reference.index < scope.block.nodes.len() {
            Demand::driver(context, reference, part)
        } else {
            Demand::value(context, reference, part)
        })
    }

    /// Returns the structural drivers for a specific part of one value.
    fn dependencies(&mut self, demand: &Demand) -> Result<Vec<Demand>, BuilderError> {
        let dependencies = self.dependencies_inner(demand)?;
        self.budget.edges(dependencies.len())?;
        Ok(dependencies)
    }

    /// Computes exact wiring dependencies while leaving gates whole.
    fn dependencies_inner(&mut self, demand: &Demand) -> Result<Vec<Demand>, BuilderError> {
        let block = self.contexts[demand.context].block;
        if demand.node.index >= block.nodes.len() {
            let class = demand.node.index - block.nodes.len();
            let scope = &self.contexts[demand.context];
            let driver_count = scope
                .net_drives
                .get(class)
                .ok_or_else(|| invalid("invalid inout class"))?
                .len();
            let binding_count = scope.net_children[class].len();
            self.budget
                .edges(driver_count.saturating_add(binding_count))?;
            let drives = scope.net_drives[class].clone();
            let bindings = scope.net_children[class].clone();
            let mut dependencies = drives
                .into_iter()
                .map(|node| Demand::driver(demand.context, node, demand.part.clone()))
                .collect::<Vec<_>>();
            for (instance, port) in bindings {
                let child = self.contexts[demand.context].children[instance];
                let child_inputs = self.summary(child, port, &demand.part)?;
                self.budget
                    .preflight_edges(dependencies.len().saturating_add(child_inputs.len()))?;
                for (input, selected) in child_inputs {
                    dependencies.push(self.connected_input(
                        demand.context,
                        instance,
                        &input,
                        selected,
                    )?);
                }
            }
            return Ok(dependencies);
        }
        let node = &block.nodes[demand.node.index];
        if demand.driver {
            return match node.payload {
                NodePayload::OutputPort { arg, .. }
                | NodePayload::InstantiationInput { arg, .. } => Ok(vec![demand.at(arg)]),
                NodePayload::ExtInoutDrive {
                    drive,
                    output_enable,
                    ..
                } => Ok(vec![
                    demand.at(drive),
                    Demand::value(
                        demand.context,
                        output_enable,
                        Part::whole(block.get_node_ty(output_enable)),
                    ),
                ]),
                _ => Err(invalid("expected an input/output sink or inout drive")),
            };
        }
        if is_empty(&node.ty, &demand.part) {
            return Ok(Vec::new());
        }
        let part = &demand.part;
        let at = |node| demand.at(node);
        let whole = |node: NodeRef| {
            Demand::value(demand.context, node, Part::whole(block.get_node_ty(node)))
        };
        let make = |node, part| Demand::value(demand.context, node, part);
        let dependencies = match &node.payload {
            NodePayload::InputPort { .. } => Vec::new(),
            NodePayload::ExtInoutRead { net } => {
                let class = self.contexts[demand.context]
                    .inouts
                    .as_ref()
                    .expect("inouts indexed")
                    .local_nets[net.as_str()];
                vec![make(
                    NodeRef {
                        index: block.nodes.len() + class,
                    },
                    part.clone(),
                )]
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } => {
                if let Some(&child) = self.contexts[demand.context]
                    .children
                    .get(instantiation.as_str())
                {
                    let inputs = self.summary(child, port_name, part)?;
                    self.budget.preflight_edges(inputs.len())?;
                    inputs
                        .into_iter()
                        .map(|(input, selected)| {
                            self.connected_input(demand.context, instantiation, &input, selected)
                        })
                        .collect::<Result<Vec<_>, BuilderError>>()?
                } else if self.contexts[demand.context]
                    .externs
                    .contains(instantiation.as_str())
                {
                    Vec::new()
                } else {
                    return Err(invalid(format!(
                        "unknown instance '{instantiation}' in block '{}'",
                        block.name
                    )));
                }
            }
            NodePayload::Unop(Unop::Identity, arg) => vec![at(*arg)],
            NodePayload::Unop(Unop::Reverse, arg) if part.path.is_empty() => {
                let (lo, hi) = part.bits.expect("reverse is bits-typed");
                let Type::Bits(width) = block.get_node_ty(*arg) else {
                    unreachable!()
                };
                vec![make(*arg, Part::bits(width - hi, width - lo))]
            }
            NodePayload::BitSlice { arg, start, .. } if part.path.is_empty() => {
                let (lo, hi) = part.bits.expect("bit_slice is bits-typed");
                vec![make(*arg, Part::bits(start + lo, start + hi))]
            }
            NodePayload::Nary(NaryOp::Concat, args) if part.path.is_empty() => {
                let (lo, hi) = part.bits.expect("concat is bits-typed");
                let mut base = 0;
                let mut result = Vec::new();
                for &arg in args.iter().rev() {
                    let Type::Bits(width) = block.get_node_ty(arg) else {
                        unreachable!()
                    };
                    let end = base + width;
                    let start = lo.max(base);
                    let limit = hi.min(end);
                    if start < limit {
                        self.budget
                            .preflight_edges(result.len().saturating_add(1))?;
                        result.push(make(arg, Part::bits(start - base, limit - base)));
                    }
                    if end >= hi {
                        break;
                    }
                    base = end;
                }
                result
            }
            NodePayload::ZeroExt { arg, .. } | NodePayload::SignExt { arg, .. }
                if part.path.is_empty() =>
            {
                let (lo, hi) = part.bits.expect("extension is bits-typed");
                let Type::Bits(width) = block.get_node_ty(*arg) else {
                    unreachable!()
                };
                let mut result = Vec::new();
                if lo < *width {
                    result.push(make(*arg, Part::bits(lo, hi.min(*width))));
                }
                if hi > *width && *width != 0 && matches!(node.payload, NodePayload::SignExt { .. })
                {
                    result.push(make(*arg, Part::bits(width - 1, *width)));
                }
                result
            }
            NodePayload::Tuple(values) | NodePayload::Array(values) => {
                if let Some((&index, rest)) = part.path.split_first() {
                    vec![make(
                        values[index],
                        Part {
                            path: rest.to_vec(),
                            bits: part.bits,
                        },
                    )]
                } else {
                    self.budget.preflight_edges(values.len())?;
                    values.iter().copied().map(whole).collect()
                }
            }
            NodePayload::TupleIndex { tuple, index } => vec![make(*tuple, part.prepend(*index))],
            NodePayload::ArrayConcat(arrays) => {
                if let Some((&index, rest)) = part.path.split_first() {
                    let mut offset = index;
                    let mut found = None;
                    for &array in arrays {
                        let Type::Array(ty) = block.get_node_ty(array) else {
                            unreachable!()
                        };
                        if offset < ty.element_count {
                            let mut path = vec![offset];
                            path.extend_from_slice(rest);
                            found = Some(make(
                                array,
                                Part {
                                    path,
                                    bits: part.bits,
                                },
                            ));
                            break;
                        }
                        offset -= ty.element_count;
                    }
                    vec![
                        found.ok_or_else(|| {
                            invalid("array concatenation selection is out of bounds")
                        })?,
                    ]
                } else {
                    self.budget.preflight_edges(arrays.len())?;
                    arrays.iter().copied().map(whole).collect()
                }
            }
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => {
                if let LiteralIndices::Known(indices) = literal_indices(
                    block,
                    *array,
                    indices,
                    *assumed_in_bounds || !self.options.array_index_bounds_checking,
                    true,
                ) {
                    let mut path = indices;
                    path.extend_from_slice(&part.path);
                    vec![make(
                        *array,
                        Part {
                            path,
                            bits: part.bits,
                        },
                    )]
                } else {
                    self.whole_operands(demand.context, &node.payload)?
                }
            }
            NodePayload::ArraySlice { array, start, .. } => {
                if let Some(start) = literal_index(block, *start) {
                    if let Some((&index, rest)) = part.path.split_first() {
                        let Type::Array(ty) = block.get_node_ty(*array) else {
                            unreachable!()
                        };
                        let last = ty
                            .element_count
                            .checked_sub(1)
                            .ok_or_else(|| invalid("array slice source is empty"))?;
                        let source = start.saturating_add(index).min(last);
                        let mut path = vec![source];
                        path.extend_from_slice(rest);
                        vec![make(
                            *array,
                            Part {
                                path,
                                bits: part.bits,
                            },
                        )]
                    } else {
                        self.split_array(demand)?
                    }
                } else {
                    self.whole_operands(demand.context, &node.payload)?
                }
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                assumed_in_bounds,
            } => match literal_indices(block, *array, indices, *assumed_in_bounds, false) {
                LiteralIndices::Known(indices) => {
                    if indices.iter().zip(&part.path).any(|(a, b)| a != b) {
                        vec![at(*array)]
                    } else if part.path.len() < indices.len() {
                        self.split_array(demand)?
                    } else {
                        vec![make(
                            *value,
                            Part {
                                path: part.path[indices.len()..].to_vec(),
                                bits: part.bits,
                            },
                        )]
                    }
                }
                LiteralIndices::OutOfBounds => vec![at(*array)],
                LiteralIndices::Unknown => self.whole_operands(demand.context, &node.payload)?,
            },
            // A sink produces unit; its driver is followed only when traversing
            // the hierarchy connection itself.
            NodePayload::OutputPort { .. } | NodePayload::InstantiationInput { .. } => Vec::new(),
            _ => self.whole_operands(demand.context, &node.payload)?,
        };
        Ok(dependencies)
    }

    /// Collects conservative whole-operand dependencies after preflighting
    /// fan-in.
    fn whole_operands(
        &self,
        context: usize,
        payload: &NodePayload,
    ) -> Result<Vec<Demand>, BuilderError> {
        self.budget.preflight_edges(variable_fan_in(payload))?;
        let block = self.contexts[context].block;
        Ok(operands(payload)
            .into_iter()
            .map(|node| Demand::value(context, node, Part::whole(block.get_node_ty(node))))
            .collect())
    }

    /// Splits an array only when mixed element drivers require it during
    /// refinement.
    fn split_array(&self, demand: &Demand) -> Result<Vec<Demand>, BuilderError> {
        let block = self.contexts[demand.context].block;
        let selected = selected_type(block.get_node_ty(demand.node), &demand.part.path)
            .ok_or_else(|| invalid("invalid aggregate selection"))?;
        let Type::Array(ty) = selected else {
            return Err(invalid("expected an array selection"));
        };
        if ty.element_count > super::MAX_WIRING_ARRAY_ELEMENTS {
            return Err(invalid(format!(
                "array wiring refinement would expand {} elements (limit {})",
                ty.element_count,
                super::MAX_WIRING_ARRAY_ELEMENTS
            )));
        }
        self.budget.preflight_edges(ty.element_count)?;
        Ok((0..ty.element_count)
            .map(|index| {
                let mut part = Part::whole(&ty.element_type);
                part.path = demand.part.path.clone();
                part.path.push(index);
                Demand::value(demand.context, demand.node, part)
            })
            .collect())
    }

    /// Reports the block-local feedback and implicit instance connections.
    fn cycle_error(&self, block: &Block, path: &[Frame]) -> BuilderError {
        let local: Vec<_> = path.iter().collect();
        let steps = local
            .iter()
            .zip(local.iter().cycle().skip(1))
            .map(|(first, second)| {
                let scope = &self.contexts[first.demand.context];
                let node = scope.witness(first.demand.node);
                let next = scope.witness(second.demand.node);
                let connection = match (&node.payload, &next.payload) {
                    (
                        NodePayload::InstantiationOutput {
                            instantiation,
                            port_name: output_port,
                        },
                        NodePayload::InstantiationInput {
                            port_name: input_port,
                            ..
                        },
                    ) => Some(BlockCycleConnection {
                        instance: instantiation.clone(),
                        input_port: input_port.clone(),
                        output_port: output_port.clone(),
                        input_node_id: next.text_id,
                    }),
                    _ => None,
                };
                BlockCycleStep {
                    node_id: node.text_id,
                    connection,
                }
            })
            .collect();
        BuilderError::CombinationalCycle {
            block: block.name.clone(),
            steps,
        }
    }
}

impl<'a> Context<'a> {
    /// Reports a graph node witnessing a synthetic resolved-net dependency.
    fn witness(&self, reference: NodeRef) -> &'a crate::ir::Node {
        let reference = if reference.index < self.block.nodes.len() {
            reference
        } else {
            self.net_witnesses[reference.index - self.block.nodes.len()]
                .expect("a net in a directed cycle has a read, drive, or child binding")
        };
        self.block.get_node(reference)
    }

    fn new(block: &'a Block) -> Self {
        let inputs = block
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| match &node.payload {
                NodePayload::InstantiationInput {
                    instantiation,
                    port_name,
                    ..
                } => Some((
                    (instantiation.as_str(), port_name.as_str()),
                    NodeRef { index },
                )),
                _ => None,
            })
            .collect();
        Self {
            block,
            inputs,
            children: HashMap::new(),
            externs: HashSet::new(),
            inouts: None,
            net_drives: Vec::new(),
            net_children: Vec::new(),
            net_witnesses: Vec::new(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct Part {
    path: Vec<usize>,
    bits: Option<(usize, usize)>,
}

impl Part {
    fn whole(ty: &Type) -> Self {
        match ty {
            Type::Bits(width) => Self::bits(0, *width),
            _ => Self {
                path: Vec::new(),
                bits: None,
            },
        }
    }
    fn bits(lo: usize, hi: usize) -> Self {
        Self {
            path: Vec::new(),
            bits: Some((lo, hi)),
        }
    }
    fn prepend(&self, index: usize) -> Self {
        let mut path = Vec::with_capacity(self.path.len() + 1);
        path.push(index);
        path.extend_from_slice(&self.path);
        Self {
            path,
            bits: self.bits,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct Demand {
    context: usize,
    node: NodeRef,
    part: Part,
    driver: bool,
}
impl Demand {
    fn value(context: usize, node: NodeRef, part: Part) -> Self {
        Self {
            context,
            node,
            part,
            driver: false,
        }
    }
    fn driver(context: usize, node: NodeRef, part: Part) -> Self {
        Self {
            context,
            node,
            part,
            driver: true,
        }
    }
    fn at(&self, node: NodeRef) -> Self {
        Self::value(self.context, node, self.part.clone())
    }
}
struct Frame {
    demand: Demand,
    dependencies: std::vec::IntoIter<Demand>,
}
impl Frame {
    fn new(demand: Demand, hierarchy: &mut Hierarchy<'_>) -> Result<Self, BuilderError> {
        let dependencies = hierarchy.dependencies(&demand)?.into_iter();
        Ok(Self {
            demand,
            dependencies,
        })
    }
}

/// Preflights operations whose explicit operand list can have unbounded fan-in.
fn variable_fan_in(payload: &NodePayload) -> usize {
    match payload {
        NodePayload::Tuple(args)
        | NodePayload::Array(args)
        | NodePayload::ArrayConcat(args)
        | NodePayload::AfterAll(args)
        | NodePayload::Nary(_, args) => args.len(),
        NodePayload::ArrayUpdate { indices, .. } => indices.len().saturating_add(2),
        NodePayload::ArrayIndex { indices, .. } => indices.len().saturating_add(1),
        NodePayload::ExtNaryAdd { terms, .. } => terms.len(),
        NodePayload::Trace { operands, .. } => operands.len().saturating_add(2),
        NodePayload::Invoke { operands, .. } => operands.len(),
        NodePayload::PrioritySel { cases, default, .. }
        | NodePayload::Sel { cases, default, .. } => cases
            .len()
            .saturating_add(1 + usize::from(default.is_some())),
        NodePayload::OneHotSel { cases, .. } => cases.len().saturating_add(1),
        NodePayload::CountedFor { invariant_args, .. } => invariant_args.len().saturating_add(1),
        // Bounded operations can allocate at most a handful of operands; their
        // exact edge count is charged immediately after construction.
        _ => 0,
    }
}

fn selected_type<'a>(mut ty: &'a Type, path: &[usize]) -> Option<&'a Type> {
    for &index in path {
        ty = match ty {
            Type::Tuple(fields) => fields.get(index)?,
            Type::Array(array) if index < array.element_count => &array.element_type,
            _ => return None,
        };
    }
    Some(ty)
}
fn is_empty(ty: &Type, part: &Part) -> bool {
    if let Some((lo, hi)) = part.bits {
        return lo == hi;
    }
    selected_type(ty, &part.path)
        .is_some_and(|t| t.checked_bit_count() == Some(0) && !t.contains_token())
}
fn literal_index(block: &Block, node: NodeRef) -> Option<usize> {
    let NodePayload::Literal(value) = &block.get_node(node).payload else {
        return None;
    };
    let bits = value.as_bits().ok()?;
    Some(crate::ir_value_utils::ir_bits_to_usize(bits).unwrap_or(usize::MAX))
}
enum LiteralIndices {
    Known(Vec<usize>),
    OutOfBounds,
    Unknown,
}

/// Resolves locally literal indices using the operation's out-of-bounds rules.
fn literal_indices(
    block: &Block,
    array: NodeRef,
    indices: &[NodeRef],
    assumed: bool,
    clamp: bool,
) -> LiteralIndices {
    let mut ty = block.get_node_ty(array);
    let mut result = Vec::new();
    for &node in indices {
        let Type::Array(dim) = ty else {
            return LiteralIndices::Unknown;
        };
        let Some(raw) = literal_index(block, node) else {
            return LiteralIndices::Unknown;
        };
        if raw >= dim.element_count {
            if assumed {
                return LiteralIndices::Unknown;
            }
            if !clamp {
                return LiteralIndices::OutOfBounds;
            }
        }
        let Some(last) = dim.element_count.checked_sub(1) else {
            return LiteralIndices::Unknown;
        };
        result.push(raw.min(last));
        ty = &dim.element_type;
    }
    LiteralIndices::Known(result)
}
fn invalid(reason: impl Into<String>) -> BuilderError {
    BuilderError::InvalidOperation(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    const WIRING: &str = r#"package example
block leaf(d: bits[2], q: bits[2]) {
  d: bits[2] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[2]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[2] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[1] = bit_slice(q, start=0, width=1, id=5)
  d: bits[2] = concat(low, seed, id=6)
  drive: () = instantiation_input(d, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}"#;

    #[test]
    fn strict_cycle_check_and_structural_check_have_distinct_contracts() {
        let package = Parser::new(WIRING).parse_and_verify_package().unwrap();
        let top = package.get_top_block().unwrap();
        assert!(matches!(
            super::super::block_cycles::verify_block_combinational_cycles(top, &package),
            Err(BuilderError::CombinationalCycle { .. })
        ));
        verify_block_wiring_cycles(top, &package).unwrap();
        let source = WIRING.replace("start=0", "start=1");
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let error =
            verify_block_wiring_cycles(package.get_top_block().unwrap(), &package).unwrap_err();
        let BuilderError::CombinationalCycle { block, steps } = error else {
            panic!("expected cycle")
        };
        assert_eq!(block, "root");
        assert!(
            steps
                .iter()
                .any(|step| step.connection.as_ref().is_some_and(|c| c.instance == "u"))
        );
    }

    /// Replaces the leaf's output wiring while retaining the same parent loop.
    fn with_leaf(body: &str, index: usize, low_feedback: bool) -> String {
        let mut source = WIRING.replace("q: () = output_port(d, name=q, id=2)", body);
        source = source.replace(
            "start=0, width=1, id=5",
            &format!("start={index}, width=1, id=5"),
        );
        if low_feedback {
            source = source.replace("concat(low, seed, id=6)", "concat(seed, low, id=6)");
        }
        source
    }

    /// Asserts that the strict checker finds a node cycle, then tests
    /// refinement.
    fn refined(source: &str, cycle: bool) {
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        let block = package.get_top_block().unwrap();
        assert!(matches!(
            super::super::verify_block_combinational_cycles(block, &package),
            Err(BuilderError::CombinationalCycle { .. })
        ));
        let result = verify_block_wiring_cycles(block, &package);
        assert_eq!(result.is_err(), cycle, "{result:?}\n{source}");
    }

    #[test]
    fn refines_static_bit_and_aggregate_wiring() {
        let identity_cases = [
            r#"lo: bits[1] = bit_slice(d, start=0, width=1, id=101)
  hi: bits[1] = bit_slice(d, start=1, width=1, id=102)
  pair: (bits[1], bits[1]) = tuple(hi, lo, id=103)
  a: bits[1] = tuple_index(pair, index=0, id=104)
  b: bits[1] = tuple_index(pair, index=1, id=105)
  both: bits[2] = concat(a, b, id=106)
  q: () = output_port(both, name=q, id=2)"#,
            r#"lo: bits[1] = bit_slice(d, start=0, width=1, id=101)
  hi: bits[1] = bit_slice(d, start=1, width=1, id=102)
  a: bits[1][1] = array(lo, id=103)
  b: bits[1][1] = array(hi, id=104)
  pair: bits[1][2] = array_concat(a, b, id=105)
  zero: bits[2] = literal(value=0, id=106)
  one: bits[2] = literal(value=1, id=107)
  oob: bits[2] = literal(value=3, id=108)
  untouched: bits[1][2] = array_update(pair, hi, indices=[oob], id=109)
  first: bits[1] = array_index(untouched, indices=[zero], id=110)
  last: bits[1] = array_index(untouched, indices=[one], id=111)
  both: bits[2] = concat(last, first, id=112)
  q: () = output_port(both, name=q, id=2)"#,
        ];
        for body in identity_cases {
            refined(&with_leaf(body, 0, false), false);
            refined(&with_leaf(body, 1, false), true);
        }
        let reverse =
            "back: bits[2] = reverse(d, id=101)\n  q: () = output_port(back, name=q, id=2)";
        refined(&with_leaf(reverse, 1, false), false);
        refined(&with_leaf(reverse, 0, false), true);
        let zero = "lo: bits[1] = bit_slice(d, start=0, width=1, id=101)\n  ext: bits[2] = zero_ext(lo, new_bit_count=2, id=102)\n  q: () = output_port(ext, name=q, id=2)";
        refined(&with_leaf(zero, 1, false), false);
        refined(&with_leaf(zero, 0, true), true);
        let sign = zero.replace("zero_ext", "sign_ext");
        refined(&with_leaf(&sign, 1, false), false);
        refined(&with_leaf(&sign, 1, true), true);
        let clamped = r#"lo: bits[1] = bit_slice(d, start=0, width=1, id=101)
  hi: bits[1] = bit_slice(d, start=1, width=1, id=102)
  pair: bits[1][2] = array(lo, hi, id=103)
  oob: bits[2] = literal(value=3, id=104)
  sliced: bits[1][1] = array_slice(pair, oob, width=1, id=105)
  zero: bits[1] = literal(value=0, id=106)
  last: bits[1] = array_index(sliced, indices=[zero], id=107)
  both: bits[2] = concat(lo, last, id=108)
  q: () = output_port(both, name=q, id=2)"#;
        refined(&with_leaf(clamped, 1, false), false);
        refined(&with_leaf(clamped, 0, false), true);
        let directly_indexed = clamped.replace(
            "array_index(sliced, indices=[zero]",
            "array_index(pair, indices=[oob]",
        );
        refined(&with_leaf(&directly_indexed, 1, false), false);
        refined(&with_leaf(&directly_indexed, 0, false), true);
        let unchecked = directly_indexed.replace("value=3", "value=2");
        let source = with_leaf(&unchecked, 0, true);
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let top = package.get_top_block().unwrap();
        verify_block_wiring_cycles(top, &package).unwrap();
        let options = BlockWiringCycleOptions {
            array_index_bounds_checking: false,
        };
        assert!(matches!(
            verify_block_wiring_cycles_with_options(top, &package, &options),
            Err(BuilderError::CombinationalCycle { .. })
        ));
    }

    #[test]
    fn unsimplified_logic_and_dynamic_operations_remain_conservative() {
        for body in [
            "zero: bits[2] = literal(value=0, id=101)\n  value: bits[2] = and(d, zero, id=102)\n  q: () = output_port(value, name=q, id=2)",
            "value: bits[2] = shrl(d, d, id=101)\n  q: () = output_port(value, name=q, id=2)",
        ] {
            refined(&with_leaf(body, 0, false), true);
        }
    }

    #[test]
    fn externs_and_registers_are_opaque_during_refinement() {
        let source = WIRING
            .replace("block leaf", "fn foreign(arg: bits[2] id=20) -> bits[2] {\n  ret arg: bits[2] = param(name=arg, id=20)\n}\nblock leaf")
            .replace("top block root(seed", "top block root(clk: clock, seed")
            .replace("  instantiation u(block=leaf, kind=block)", "  reg state(bits[2])\n  instantiation u(block=leaf, kind=block)\n  instantiation ext(foreign_function=foreign, kind=extern)")
            .replace("  seed: bits[1]", "  e: bits[2] = instantiation_output(instantiation=ext, port_name=return, id=21)\n  feedback: () = instantiation_input(e, instantiation=ext, port_name=arg, id=22)\n  stored: bits[2] = register_read(register=state, id=23)\n  update: () = register_write(stored, register=state, id=24)\n  seed: bits[1]");
        refined(&source, false);
    }

    #[test]
    fn repeated_hierarchy_is_indexed_by_definition() {
        let mut prefix = String::from("block bush0() {}\n");
        for depth in 1..=20 {
            prefix.push_str(&format!("block bush{depth}() {{\n  instantiation left(block=bush{}, kind=block)\n  instantiation right(block=bush{}, kind=block)\n}}\n", depth - 1, depth - 1));
        }
        let source = WIRING.replace("block leaf", &format!("{prefix}block leaf")).replace("  instantiation u(block=leaf, kind=block)", "  instantiation u(block=leaf, kind=block)\n  instantiation dense(block=bush20, kind=block)");
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let block = package.get_top_block().unwrap();
        let mut cache = BlockWiringHierarchy::new(BlockWiringCycleOptions::default());
        cache.index_root(block, &package).unwrap();
        assert_eq!(cache.hierarchy.contexts.len(), 23);
        verify_block_wiring_cycles(block, &package).unwrap();
    }

    #[test]
    fn ignores_huge_unrelated_aggregate_and_finds_disconnected_feedback() {
        let source = WIRING
            .replace("root(seed: bits[1], result", "root(seed: bits[1], bulk: token[100000000], atom: token, result")
            .replace("  seed: bits[1]", "  bulk: token[100000000] = input_port(name=bulk, id=30)\n  atom: token = input_port(name=atom, id=31)\n  i: bits[1] = literal(value=0, id=32)\n  unused: token[100000000] = array_update(bulk, atom, indices=[i], id=33)\n  seed: bits[1]");
        refined(&source, false);
        let disconnected = source
            .replace("  instantiation u(block=leaf, kind=block)", "  instantiation u(block=leaf, kind=block)\n  instantiation v(block=leaf, kind=block)")
            .replace("  result: () = output_port", "  vq: bits[2] = instantiation_output(instantiation=v, port_name=q, id=34)\n  vd: () = instantiation_input(vq, instantiation=v, port_name=d, id=35)\n  result: () = output_port");
        refined(&disconnected, true);
    }
    /// Builds a source hierarchy with valid resolved-net feedback in its leaf.
    fn inout_chain(depth: usize) -> String {
        let mut source = String::from("package chain\n");
        source.push_str("block layer0(seed: bits[1], result: bits[2]) {\n  ext_inout_net bus(bits[2])\n  seed: bits[1] = input_port(name=seed, id=1)\n  one: bits[1] = literal(value=1, id=2)\n  sample: bits[2] = ext_inout_read(net=bus, id=3)\n  lower: bits[1] = bit_slice(sample, start=0, width=1, id=4)\n  data: bits[2] = concat(lower, seed, id=5)\n  drive: () = ext_inout_drive(data, one, net=bus, id=6)\n  result: () = output_port(sample, name=result, id=7)\n}\n");
        let mut id = 8;
        for layer in 1..depth {
            let top = if layer + 1 == depth { "top " } else { "" };
            source.push_str(&format!("{top}block layer{layer}(seed: bits[1], result: bits[2]) {{\n  instantiation u(block=layer{}, kind=block)\n  seed: bits[1] = input_port(name=seed, id={id})\n  q: bits[2] = instantiation_output(instantiation=u, port_name=result, id={})\n  bind: () = instantiation_input(seed, instantiation=u, port_name=seed, id={})\n  result: () = output_port(q, name=result, id={})\n}}\n", layer - 1, id + 1, id + 2, id + 3));
            id += 4;
        }
        source
    }

    #[test]
    fn deep_inout_chain_reuses_each_refined_context_once() {
        let source = inout_chain(64);
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let leaf = package.get_block("layer0").unwrap();
        let original =
            super::super::block_cycles::verify_block_combinational_cycles(leaf, &package)
                .unwrap_err();
        let mut cache = BlockWiringHierarchy::new(BlockWiringCycleOptions::default());
        for layer in 0..64 {
            let block = package.get_block(&format!("layer{layer}")).unwrap();
            cache.check(block, &package, &original).unwrap();
        }
        assert_eq!(cache.hierarchy.contexts.len(), 64);
        assert!(cache.state.iter().all(|&state| state == 2));
        assert!(cache.hierarchy.budget.states < 64);
    }

    /// A selected public inout can immediately request its child's matching
    /// summary, so budget accounting must stop descent before visiting leaves.
    #[test]
    fn summary_state_budget_bounds_pending_child_frames() {
        let depth = 8;
        let mut source = String::from(
            r#"package passive
block layer0(bus: bits[1]) {
  ext_inout_net net(bits[1])
  bus: () = ext_inout_port(name=bus, net=net, id=1)
}
"#,
        );
        let mut id = 2;
        for layer in 1..depth {
            let top = if layer + 1 == depth { "top " } else { "" };
            let child = layer - 1;
            let binding_id = id + 1;
            source.push_str(&format!(
                r#"{top}block layer{layer}(bus: bits[1]) {{
  ext_inout_net net(bits[1])
  instantiation u(block=layer{child}, kind=block)
  bus: () = ext_inout_port(name=bus, net=net, id={id})
  bind: () = ext_instantiation_inout(instantiation=u, port_name=bus, net=net, id={binding_id})
}}
"#
            ));
            id += 2;
        }
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let mut cache = BlockWiringHierarchy::new(BlockWiringCycleOptions::default());
        let root = cache
            .index_root(package.get_top_block().unwrap(), &package)
            .unwrap();
        for context in (0..cache.hierarchy.contexts.len()).rev() {
            cache.hierarchy.index_inouts(context).unwrap();
        }
        cache.hierarchy.budget.max_states = 2;
        let error = cache
            .hierarchy
            .summary(root, "bus", &Part::bits(0, 1))
            .err()
            .expect("summary descent must exhaust its state budget");
        assert!(error.to_string().contains("selected-state budget"));
        assert!(
            cache.hierarchy.summaries.is_empty(),
            "the traversal reached a leaf before exhausting its state budget"
        );
    }

    #[test]
    fn refinement_budget_fails_with_an_actionable_diagnostic() {
        let package = Parser::new(WIRING).parse_and_verify_package().unwrap();
        let block = package.get_top_block().unwrap();
        let original =
            super::super::block_cycles::verify_block_combinational_cycles(block, &package)
                .unwrap_err();
        let mut cache = BlockWiringHierarchy::new(BlockWiringCycleOptions::default());
        cache.hierarchy.budget.max_states = 1;
        let error = cache
            .check(block, &package, &original)
            .unwrap_err()
            .to_string();
        assert!(error.contains("selected-state budget"), "{error}");
    }
    #[test]
    fn independent_roots_receive_independent_traversal_budgets() {
        let (leaf, body) = WIRING.split_once("top block root").unwrap();
        let mut second = body.to_owned();
        for id in 3..=8 {
            second = second.replace(&format!("id={id})"), &format!("id={})", id + 10));
        }
        let source = format!("{leaf}block left{body}\ntop block right{second}");
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let left = package.get_block("left").unwrap();
        let right = package.get_block("right").unwrap();
        let original =
            super::super::block_cycles::verify_block_combinational_cycles(left, &package)
                .unwrap_err();
        let mut cache = BlockWiringHierarchy::new(BlockWiringCycleOptions::default());
        cache.check(left, &package, &original).unwrap();
        let per_root_limit = cache.hierarchy.budget.states;
        cache.hierarchy.budget.max_states = per_root_limit;
        cache.check(right, &package, &original).unwrap();
        assert!(cache.hierarchy.budget.states <= per_root_limit);
    }
    #[test]
    fn narrow_tuple_and_concat_selections_only_preflight_selected_operands() {
        use crate::{BlockBuilder, ir::Type};
        let mut builder = BlockBuilder::new("root");
        let bit = builder.input_port("bit", Type::Bits(1)).unwrap();
        let many = vec![bit; 256];
        builder.tuple(&many).unwrap();
        builder.concat(&many).unwrap();
        builder.output_port("result", bit).unwrap();
        let package = builder.build_package("large_fanin").unwrap();
        let block = package.get_top_block().unwrap();
        let tuple = NodeRef {
            index: block
                .nodes
                .iter()
                .position(
                    |n| matches!(n.payload, NodePayload::Tuple(ref fields) if fields.len() == 256),
                )
                .unwrap(),
        };
        let concat = NodeRef { index: block.nodes.iter().position(|n| matches!(n.payload, NodePayload::Nary(NaryOp::Concat, ref fields) if fields.len() == 256)).unwrap() };
        let mut cache = BlockWiringHierarchy::new(BlockWiringCycleOptions::default());
        let context = cache.index_root(block, &package).unwrap();
        cache.hierarchy.budget.max_edges = 1;
        let selected_field = Demand::value(
            context,
            tuple,
            Part {
                path: vec![42],
                bits: Some((0, 1)),
            },
        );
        assert_eq!(
            cache.hierarchy.dependencies(&selected_field).unwrap().len(),
            1
        );
        cache.hierarchy.budget.begin_root();
        let selected_bit = Demand::value(context, concat, Part::bits(0, 1));
        assert_eq!(
            cache.hierarchy.dependencies(&selected_bit).unwrap().len(),
            1
        );
        cache.hierarchy.budget.begin_root();
        let whole_tuple = Demand::value(context, tuple, Part::whole(block.get_node_ty(tuple)));
        assert!(
            cache
                .hierarchy
                .dependencies(&whole_tuple)
                .err()
                .unwrap()
                .to_string()
                .contains("selected-edge budget")
        );
    }
}
