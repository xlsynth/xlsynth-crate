// SPDX-License-Identifier: Apache-2.0

//! Transactional scalar graph used only by the bounded fanout optimizer.

use super::*;
use crate::netlist::sta::fanout::{CachedCell, SharedTiming, Source};
use crate::netlist::sta::{EdgeTiming, SignalTiming};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// A Boolean-only connectivity cycle requires full STA's finer dependencies.
#[derive(Debug)]
pub(super) struct UnsupportedTopology;

impl std::fmt::Display for UnsupportedTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("connected-pin graph contains a cycle outside timing dependencies")
    }
}
impl std::error::Error for UnsupportedTopology {}

/// Compact instance, preserving original connection order for exact load sums.
#[derive(Clone)]
struct Node {
    cell: usize,
    connections: Vec<(usize, Source)>,
    rank: usize,
}
/// A cached net's connectivity, edge loads, timing and fixed output load count.
#[derive(Clone, Default)]
struct CachedNet {
    driver: Option<(usize, usize)>,
    sinks: BTreeSet<(usize, usize)>,
    load: [f64; 2],
    ports: usize,
    timing: Option<SignalTiming>,
    constant: Option<bool>,
    coverage_constant: Option<bool>,
}
/// Cached electrical table bounds and pin constraints.
struct Bounds {
    load: Vec<f64>,
    slew: Vec<f64>,
}
/// Sparse replacements and append-only topology additions for a trial.
pub(super) struct Edit {
    changed: BTreeMap<usize, Node>,
    added: Vec<Node>,
    new_nets: usize,
}
/// Work counters exclude the full-STA validation oracle.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Work {
    pub nodes: usize,
    pub arcs: usize,
    pub changed_nets: usize,
    pub load_nets: usize,
    pub topology_seconds: f64,
    pub propagation_seconds: f64,
    pub metrics_seconds: f64,
}
/// Undo only touched topology, timing values and per-cell diagnostics.
struct Undo {
    node_count: usize,
    net_count: usize,
    nodes: BTreeMap<usize, Node>,
    nets: BTreeMap<usize, CachedNet>,
    timing: BTreeMap<usize, (Option<SignalTiming>, Option<bool>)>,
    counts: BTreeMap<usize, TimingQueryDiagnosticCounts>,
}
/// One reusable graph and compiled Liberty cache, shared by every candidate.
pub(super) struct Engine<'a> {
    library: &'a Library,
    shared: SharedTiming<'a>,
    options: &'a FanoutOptimizationOptions,
    kernels: Vec<Option<CachedCell>>,
    bounds: Vec<Bounds>,
    nodes: Vec<Node>,
    nets: Vec<CachedNet>,
    roots: Vec<Source>,
    counts: Vec<TimingQueryDiagnosticCounts>,
    cell_counts: Vec<usize>,
    lexical_cells: Vec<usize>,
    endpoints: Vec<Source>,
}
impl<'a> Engine<'a> {
    /// Builds topology once from an already full-STA-validated scalar netlist.
    pub fn new(
        netlist: &MappedNetlist,
        library: &'a Library,
        options: &'a FanoutOptimizationOptions,
        choices: &[CellChoice],
        sta: &StaReport,
    ) -> Result<Self> {
        let shared = SharedTiming::new(library)?;
        let mut needed: BTreeSet<_> = choices.iter().map(|c| c.index).collect();
        let names: BTreeMap<_, _> = library
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.as_str(), i))
            .collect();
        for inst in &netlist.module.instances {
            needed
                .insert(names[resolve_symbol(&netlist.interner, inst.type_name, "cell")?.as_str()]);
        }
        let mut kernels = (0..library.cells.len()).map(|_| None).collect::<Vec<_>>();
        for i in needed {
            kernels[i] = Some(CachedCell::new(&shared, i)?);
        }
        let bounds = library
            .cells
            .iter()
            .map(|c| Bounds {
                load: c
                    .pins
                    .iter()
                    .map(|p| table_bound(library, p, None, true))
                    .collect(),
                slew: c
                    .pins
                    .iter()
                    .map(|p| {
                        c.pins
                            .iter()
                            .filter(|o| o.direction == PinDirection::Output as i32)
                            .map(|o| {
                                table_bound(
                                    library,
                                    o,
                                    Some(library.resolve_string(&p.name)),
                                    false,
                                )
                            })
                            .fold(f64::INFINITY, f64::min)
                    })
                    .collect(),
            })
            .collect();
        let mut aliases = BTreeMap::new();
        for a in &netlist.module.assigns {
            let NetRef::Simple(lhs) = a.lhs else {
                return Err(anyhow!("scalar alias expected"));
            };
            let AssignExpr::Leaf(rhs) = &a.rhs else {
                return Err(anyhow!("wire alias expected"));
            };
            aliases.insert(lhs.0, Self::raw_source(rhs)?);
        }
        let mut roots = Vec::new();
        for i in 0..netlist.nets.len() {
            let mut s = Source::Net(i);
            let mut seen = BTreeSet::new();
            while let Source::Net(n) = s {
                let Some(next) = aliases.get(&n) else {
                    break;
                };
                ensure!(seen.insert(n), "alias cycle");
                s = *next;
            }
            roots.push(s);
        }
        let mut nodes = Vec::new();
        for inst in &netlist.module.instances {
            let ci = names[resolve_symbol(&netlist.interner, inst.type_name, "cell")?.as_str()];
            let c = kernels[ci].as_ref().unwrap();
            let mut connections = Vec::new();
            for (p, r) in &inst.connections {
                let pn = resolve_symbol(&netlist.interner, *p, "pin")?;
                let pi = c
                    .names
                    .iter()
                    .position(|n| *n == pn)
                    .ok_or_else(|| anyhow!("missing cached pin"))?;
                let s = match Self::raw_source(r)? {
                    Source::Net(n) if c.inputs[pi] => roots[n],
                    s => s,
                };
                connections.push((pi, s));
            }
            nodes.push(Node {
                cell: ci,
                connections,
                rank: 0,
            });
        }
        let mut nets = vec![CachedNet::default(); netlist.nets.len()];
        let mut endpoints = Vec::new();
        let mut net_by_name = HashMap::with_capacity(netlist.nets.len());
        for (i, n) in netlist.nets.iter().enumerate() {
            net_by_name.entry(n.name).or_insert(i);
        }
        for p in &netlist.module.ports {
            let n = *net_by_name
                .get(&p.name)
                .ok_or_else(|| anyhow!("missing port"))?;
            if p.direction == PortDirection::Output {
                endpoints.push(roots[n]);
                if let Source::Net(n) = roots[n] {
                    nets[n].ports += 1;
                }
            }
        }
        for (i, n) in nets.iter_mut().enumerate() {
            n.timing = sta.net_timing[i];
        }
        let mut lexical_cells = (0..library.cells.len()).collect::<Vec<_>>();
        lexical_cells.sort_by_key(|i| &library.cells[*i].name);
        let count = nodes.len();
        let mut e = Self {
            library,
            shared,
            options,
            kernels,
            bounds,
            nodes,
            nets,
            roots,
            counts: vec![TimingQueryDiagnosticCounts::default(); count],
            cell_counts: vec![0; library.cells.len()],
            lexical_cells,
            endpoints,
        };
        for i in 0..e.nodes.len() {
            e.add_connections(i);
        }
        for n in 0..e.nets.len() {
            e.reload(n);
        }
        e.rank()?;
        e.initialize_coverage_constants();
        // Establish per-cell query counts with the same compiled kernel used in
        // trials; the full report remains the independent initial oracle.
        let mut order = (0..e.nodes.len()).collect::<Vec<_>>();
        order.sort_by_key(|i| (e.nodes[*i].rank, *i));
        for i in order {
            let (values, counts, _) = e.evaluate_node(i)?;
            e.counts[i] = counts;
            for (n, t, constant) in values {
                e.nets[n].timing = Some(t);
                e.nets[n].coverage_constant = constant;
            }
        }
        e.check(sta)?;
        Ok(e)
    }
    /// Reaches the same Boolean fixed point as full STA before timing checks,
    /// including an output wired back to an untimed input of its own instance.
    fn initialize_coverage_constants(&mut self) {
        let mut queue: VecDeque<_> = (0..self.nodes.len()).collect();
        let mut queued = vec![true; self.nodes.len()];
        while let Some(i) = queue.pop_front() {
            queued[i] = false;
            let node = &self.nodes[i];
            let cell = self.kernels[node.cell]
                .as_ref()
                .expect("cached instantiated cell");
            let known: HashMap<_, _> = node
                .connections
                .iter()
                .filter_map(|&(pin, source)| {
                    if !cell.inputs[pin] {
                        return None;
                    }
                    let value = match source {
                        Source::Net(net) => self.nets[net].coverage_constant,
                        Source::Literal(value) => Some(value),
                        Source::Empty => None,
                    };
                    value.map(|value| (cell.names[pin].clone(), value))
                })
                .collect();
            let values = cell.coverage_constants(&self.shared, &known);
            for &(pin, source) in &node.connections {
                if cell.inputs[pin] {
                    continue;
                }
                let Source::Net(net) = source else {
                    continue;
                };
                if self.nets[net].coverage_constant.is_none() && values[pin].is_some() {
                    self.nets[net].coverage_constant = values[pin];
                    for &(sink, _) in &self.nets[net].sinks {
                        if !queued[sink] {
                            queued[sink] = true;
                            queue.push_back(sink);
                        }
                    }
                }
            }
        }
    }

    /// Resolves scalar references without accepting implicit bus expansion.
    fn raw_source(r: &NetRef) -> Result<Source> {
        Ok(match r {
            NetRef::Simple(n) => Source::Net(n.0),
            NetRef::Literal(v) if v.get_bit_count() == 1 => {
                Source::Literal(v.get_bit(0).map_err(anyhow::Error::msg)?)
            }
            NetRef::Unconnected => Source::Empty,
            _ => return Err(anyhow!("unsupported cached scalar source")),
        })
    }
    /// Adds one node to ordered sink sets and output-driver identities.
    fn add_connections(&mut self, i: usize) {
        let node = &self.nodes[i];
        let cell = self.kernels[node.cell].as_ref().unwrap();
        self.cell_counts[node.cell] += 1;
        for (j, (p, s)) in node.connections.iter().enumerate() {
            if let Source::Net(n) = s {
                if cell.inputs[*p] {
                    self.nets[*n].sinks.insert((i, j));
                } else {
                    self.nets[*n].driver = Some((i, *p));
                    self.nets[*n].constant = cell.constants[*p];
                }
            }
        }
    }
    /// Removes one node before a transactional replacement.
    fn remove_connections(&mut self, i: usize) {
        let node = &self.nodes[i];
        let cell = self.kernels[node.cell].as_ref().unwrap();
        self.cell_counts[node.cell] -= 1;
        for (j, (p, s)) in node.connections.iter().enumerate() {
            if let Source::Net(n) = s {
                if cell.inputs[*p] {
                    self.nets[*n].sinks.remove(&(i, j));
                } else {
                    self.nets[*n].driver = None;
                    self.nets[*n].constant = None;
                }
            }
        }
    }
    /// Resums only changed loads in original instance/connection order. This
    /// avoids subtract/add rounding drift, including repeated output aliases.
    fn reload(&mut self, n: usize) {
        let mut load = [0.0; 2];
        for &(i, j) in &self.nets[n].sinks {
            let node = &self.nodes[i];
            let p = node.connections[j].0;
            let cap = self.kernels[node.cell]
                .as_ref()
                .expect("cached instantiated cell")
                .caps[p];
            load[0] += cap[0];
            load[1] += cap[1];
        }
        for _ in 0..self.nets[n].ports {
            load[0] += self.options.module_output_load;
            load[1] += self.options.module_output_load;
        }
        self.nets[n].load = load;
    }
    /// Reestablishes spaced topological ranks once after an accepted topology.
    fn rank(&mut self) -> Result<()> {
        let mut degree = vec![0; self.nodes.len()];
        let mut successors = vec![BTreeSet::new(); self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            for (p, s) in &node.connections {
                if !self.kernels[node.cell]
                    .as_ref()
                    .expect("cached instantiated cell")
                    .inputs[*p]
                {
                    continue;
                }
                if let Source::Net(n) = s {
                    if let Some((d, _)) = self.nets[*n].driver {
                        if d != i && successors[d].insert(i) {
                            degree[i] += 1;
                        }
                    }
                }
            }
        }
        let mut q = BinaryHeap::new();
        let mut levels = vec![1; self.nodes.len()];
        let mut seen = 0;
        for (i, &d) in degree.iter().enumerate() {
            if d == 0 {
                q.push(Reverse(i));
            }
        }
        while let Some(Reverse(i)) = q.pop() {
            seen += 1;
            self.nodes[i].rank = levels[i] * 4;
            for &s in &successors[i] {
                levels[s] = levels[s].max(levels[i] + 1);
                degree[s] -= 1;
                if degree[s] == 0 {
                    q.push(Reverse(s));
                }
            }
        }
        if seen != self.nodes.len() {
            return Err(UnsupportedTopology.into());
        }
        Ok(())
    }
    /// Returns ideal literal timing, matching the full STA boundary model.
    fn zero() -> SignalTiming {
        SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
        }
    }
    /// Reads a cached net or ideal literal timing.
    fn signal(&self, s: Source) -> Option<SignalTiming> {
        match s {
            Source::Net(n) => self.nets[n].timing,
            Source::Literal(_) | Source::Empty => Some(Self::zero()),
        }
    }
    /// Evaluates one instance with shared STA kernels and coverage constants.
    fn evaluate_node(
        &self,
        i: usize,
    ) -> Result<(
        Vec<(usize, SignalTiming, Option<bool>)>,
        TimingQueryDiagnosticCounts,
        usize,
    )> {
        let node = &self.nodes[i];
        let c = self.kernels[node.cell]
            .as_ref()
            .expect("cached instantiated cell");
        let mut inputs = vec![None; c.names.len()];
        let mut known = HashMap::new();
        let mut coverage_known = HashMap::new();
        let mut loads = vec![[0.0; 2]; c.names.len()];
        for &(p, s) in &node.connections {
            if c.inputs[p] {
                inputs[p] = self.signal(s);
                let constant = match s {
                    Source::Literal(v) => Some(v),
                    Source::Net(n) => self.nets[n].constant,
                    Source::Empty => None,
                };
                if let Some(v) = constant {
                    known.insert(c.names[p].clone(), v);
                }
                let coverage = match s {
                    Source::Literal(v) => Some(v),
                    Source::Net(n) => self.nets[n].coverage_constant,
                    Source::Empty => None,
                };
                if let Some(v) = coverage {
                    coverage_known.insert(c.names[p].clone(), v);
                }
            } else if let Source::Net(n) = s {
                loads[p] = self.nets[n].load;
            }
        }
        let result = c.evaluate(
            &self.shared,
            &inputs,
            &known,
            &coverage_known,
            &loads,
            &node.connections,
        )?;
        let mut outputs = Vec::new();
        for &(p, s) in &node.connections {
            if !c.inputs[p] {
                if let Source::Net(n) = s {
                    outputs.push((
                        n,
                        result.timing[p].ok_or_else(|| anyhow!("incomplete output timing"))?,
                        result.coverage_constants[p],
                    ));
                }
            }
        }
        Ok((outputs, result.diagnostics, result.arcs))
    }
    /// Forms a sparse candidate without cloning the module, names or library.
    pub fn edit(&self, d: &Driver, kind: &FanoutEdit, choices: &[CellChoice]) -> Result<Edit> {
        let mut edit = Edit {
            changed: BTreeMap::new(),
            added: Vec::new(),
            new_nets: 0,
        };
        let original = &self.nodes[d.instance];
        match kind {
            FanoutEdit::Upsize { cell } => {
                let c = choices
                    .iter()
                    .find(|c| &c.name == cell)
                    .ok_or_else(|| anyhow!("replacement missing"))?;
                let mut node = original.clone();
                node.cell = c.index;
                for (j, (p, _)) in node.connections.iter_mut().enumerate() {
                    let name = if j == d.output_connection {
                        c.output.as_str()
                    } else {
                        self.kernels[original.cell]
                            .as_ref()
                            .expect("cached original cell")
                            .names[*p]
                            .as_str()
                    };
                    *p = self.kernels[c.index]
                        .as_ref()
                        .expect("cached candidate cell")
                        .names
                        .iter()
                        .position(|n| n == name)
                        .ok_or_else(|| anyhow!("replacement pin missing"))?;
                }
                edit.changed.insert(d.instance, node);
            }
            FanoutEdit::Replicate { groups } => {
                for part in partition(&d.sinks, *groups, d.fixed_output_load)
                    .iter()
                    .skip(1)
                {
                    if part.is_empty() {
                        continue;
                    }
                    let n = self.nets.len() + edit.new_nets;
                    edit.new_nets += 1;
                    let mut node = original.clone();
                    node.connections[d.output_connection].1 = Source::Net(n);
                    edit.added.push(node);
                    self.rewire(&mut edit, part, n);
                }
            }
            FanoutEdit::Buffer {
                cell,
                groups,
                levels,
            } => {
                let c = choices
                    .iter()
                    .find(|c| &c.name == cell)
                    .ok_or_else(|| anyhow!("buffer missing"))?;
                let k = self.kernels[c.index]
                    .as_ref()
                    .expect("cached candidate cell");
                let a = k.names.iter().position(|n| n == &c.inputs[0]).unwrap();
                let y = k.names.iter().position(|n| n == &c.output).unwrap();
                let mut parents = vec![d.net; *groups];
                if *levels == 2 {
                    for parity in 0..2 {
                        let n = self.nets.len() + edit.new_nets;
                        edit.new_nets += 1;
                        edit.added.push(Node {
                            cell: c.index,
                            connections: vec![(a, Source::Net(d.net)), (y, Source::Net(n))],
                            rank: original.rank + 1,
                        });
                        for j in (parity..*groups).step_by(2) {
                            parents[j] = n;
                        }
                    }
                }
                for (j, part) in partition(&d.sinks, *groups, 0.0).iter().enumerate() {
                    if part.is_empty() {
                        continue;
                    }
                    let n = self.nets.len() + edit.new_nets;
                    edit.new_nets += 1;
                    edit.added.push(Node {
                        cell: c.index,
                        connections: vec![(a, Source::Net(parents[j])), (y, Source::Net(n))],
                        rank: original.rank + levels,
                    });
                    self.rewire(&mut edit, part, n);
                }
            }
        }
        Ok(edit)
    }
    /// Sums candidate cell counts in the same lexical order as area reports.
    pub fn candidate_area(&self, e: &Edit) -> f64 {
        let mut counts = self.cell_counts.clone();
        for (&i, n) in &e.changed {
            counts[self.nodes[i].cell] -= 1;
            counts[n.cell] += 1;
        }
        for n in &e.added {
            counts[n.cell] += 1;
        }
        let mut area = 0.0;
        for &c in &self.lexical_cells {
            if counts[c] > 0 {
                area += self.library.cells[c].area * counts[c] as f64;
            }
        }
        area
    }
    /// Copies only changed sink instances into the sparse edit.
    fn rewire(&self, e: &mut Edit, part: &[Sink], n: usize) {
        for s in part {
            e.changed
                .entry(s.instance)
                .or_insert_with(|| self.nodes[s.instance].clone())
                .connections[s.connection]
                .1 = Source::Net(n);
        }
    }
    /// Installs sparse topology and seeds changed cells plus every changed-load
    /// driver, including upstream drivers and their sibling consumers.
    fn install(&mut self, e: Edit) -> (Undo, BinaryHeap<Reverse<(usize, usize)>>, usize) {
        let mut u = Undo {
            node_count: self.nodes.len(),
            net_count: self.nets.len(),
            nodes: BTreeMap::new(),
            nets: BTreeMap::new(),
            timing: BTreeMap::new(),
            counts: BTreeMap::new(),
        };
        let mut touched = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        for (&i, new) in &e.changed {
            for (_, s) in self.nodes[i].connections.iter().chain(&new.connections) {
                if let Source::Net(n) = s {
                    touched.insert(*n);
                }
            }
            u.nodes.insert(i, self.nodes[i].clone());
            seeds.insert(i);
        }
        for new in &e.added {
            for (_, s) in &new.connections {
                if let Source::Net(n) = s {
                    touched.insert(*n);
                }
            }
        }
        for &n in &touched {
            if n < self.nets.len() {
                u.nets.insert(n, self.nets[n].clone());
            }
        }
        self.nets
            .resize_with(self.nets.len() + e.new_nets, CachedNet::default);
        for (i, node) in e.changed {
            self.remove_connections(i);
            self.nodes[i] = node;
            self.add_connections(i);
        }
        for node in e.added {
            let i = self.nodes.len();
            self.nodes.push(node);
            self.counts.push(TimingQueryDiagnosticCounts::default());
            self.add_connections(i);
            seeds.insert(i);
        }
        for &n in &touched {
            let previous = self.nets[n].load;
            self.reload(n);
            if previous != self.nets[n].load {
                if let Some((i, _)) = self.nets[n].driver {
                    seeds.insert(i);
                }
            }
        }
        let queue = seeds
            .into_iter()
            .map(|i| Reverse((self.nodes[i].rank, i)))
            .collect();
        (u, queue, touched.len())
    }
    /// Propagates in topological order until timing and coverage constants stop
    /// changing.
    fn propagate(
        &mut self,
        u: &mut Undo,
        mut queue: BinaryHeap<Reverse<(usize, usize)>>,
        work: &mut Work,
    ) -> Result<()> {
        let mut last = None;
        while let Some(Reverse((rank, i))) = queue.pop() {
            if last == Some((rank, i)) {
                continue;
            }
            last = Some((rank, i));
            let (values, counts, arcs) = self.evaluate_node(i).map_err(CandidateTimingError)?;
            work.nodes += 1;
            work.arcs += arcs;
            u.counts.entry(i).or_insert(self.counts[i]);
            self.counts[i] = counts;
            for (n, t, constant) in values {
                // No epsilon: compare all four finite rise/fall arrival/slew
                // values exactly. The prototype has no predecessor cache.
                if self.nets[n].timing != Some(t) || self.nets[n].coverage_constant != constant {
                    u.timing
                        .entry(n)
                        .or_insert((self.nets[n].timing, self.nets[n].coverage_constant));
                    self.nets[n].timing = Some(t);
                    self.nets[n].coverage_constant = constant;
                    work.changed_nets += 1;
                    for &(s, _) in &self.nets[n].sinks {
                        // Intra-instance sources were updated during the cell
                        // evaluation, in full STA's output-pin order.
                        if s == i {
                            continue;
                        }
                        ensure!(
                            self.nodes[s].rank > rank,
                            "invalid incremental topological rank"
                        );
                        queue.push(Reverse((self.nodes[s].rank, s)));
                    }
                }
            }
        }
        Ok(())
    }
    /// Restores touched connectivity, timing, coverage constants and
    /// diagnostics.
    fn rollback(&mut self, u: Undo) {
        for i in u.node_count..self.nodes.len() {
            self.cell_counts[self.nodes[i].cell] -= 1;
        }
        for (&i, old) in &u.nodes {
            self.cell_counts[self.nodes[i].cell] -= 1;
            self.cell_counts[old.cell] += 1;
        }
        self.nodes.truncate(u.node_count);
        self.nets.truncate(u.net_count);
        self.counts.truncate(u.node_count);
        for (i, node) in u.nodes {
            self.nodes[i] = node;
        }
        for (n, net) in u.nets {
            self.nets[n] = net;
        }
        for (n, t) in u.timing {
            if n < u.net_count {
                self.nets[n].timing = t.0;
                self.nets[n].coverage_constant = t.1;
            }
        }
        for (i, c) in u.counts {
            if i < u.node_count {
                self.counts[i] = c;
            }
        }
    }
    /// Computes compact reductions from cached data, without library/table
    /// lookup or graph rebuilding. This O(N) reporting cost is measured.
    pub fn metrics(&self) -> Result<FanoutMetrics> {
        let mut m = FanoutMetrics {
            area: 0.0,
            cell_count: self.nodes.len(),
            worst_delay: 0.0,
            max_slew: 0.0,
            max_load: 0.0,
            max_fanout: 0,
            output_load_above_table_max: 0,
            input_slew_above_table_max: 0,
            max_capacitance_violations: 0,
            max_transition_violations: 0,
            timing_queries: TimingQueryDiagnosticCounts::default(),
        };
        for &c in &self.lexical_cells {
            if self.cell_counts[c] > 0 {
                m.area += self.library.cells[c].area * self.cell_counts[c] as f64;
            }
        }
        m.worst_delay = self
            .endpoints
            .iter()
            .filter_map(|s| self.signal(*s))
            .map(|t| t.rise.arrival.max(t.fall.arrival))
            .reduce(f64::max)
            .unwrap_or(0.0);
        for counts in &self.counts {
            m.timing_queries += *counts;
        }
        for net in &self.nets {
            let Some((i, p)) = net.driver else {
                continue;
            };
            let node = &self.nodes[i];
            let pin = &self.library.cells[node.cell].pins[p];
            let timing = net.timing.ok_or_else(|| anyhow!("missing cached net"))?;
            let slew = timing.rise.transition.max(timing.fall.transition);
            let load = net.load[0].max(net.load[1]);
            m.max_slew = m.max_slew.max(slew);
            m.max_load = m.max_load.max(load);
            m.max_fanout = m.max_fanout.max(net.sinks.len());
            m.output_load_above_table_max += usize::from(load > self.bounds[node.cell].load[p]);
            m.max_capacitance_violations +=
                usize::from(pin.max_capacitance.is_some_and(|b| load > b));
            m.max_transition_violations +=
                usize::from(pin.max_transition.is_some_and(|b| slew > b));
            for &(i, j) in &net.sinks {
                let n = &self.nodes[i];
                m.input_slew_above_table_max +=
                    usize::from(slew > self.bounds[n.cell].slew[n.connections[j].0]);
            }
        }
        Ok(m)
    }
    /// Runs a trial and optional oracle while temporary state is installed.
    /// Errors and rejected trials restore all touched state before returning.
    pub fn trial<F>(&mut self, e: Edit, oracle: F) -> Result<(FanoutMetrics, Work)>
    where
        F: FnOnce(&Self, &FanoutMetrics) -> Result<()>,
    {
        let start = Instant::now();
        let (mut undo, q, loads) = self.install(e);
        let mut work = Work {
            load_nets: loads,
            topology_seconds: start.elapsed().as_secs_f64(),
            ..Default::default()
        };
        let start = Instant::now();
        let result = self.propagate(&mut undo, q, &mut work);
        work.propagation_seconds = start.elapsed().as_secs_f64();
        let result = result.and_then(|()| {
            let start = Instant::now();
            let m = self.metrics()?;
            work.metrics_seconds = start.elapsed().as_secs_f64();
            oracle(self, &m)?;
            Ok((m, work))
        });
        self.rollback(undo);
        result
    }
    /// Commits a previously checked edit and rebuilds ranks for the next round.
    pub fn commit(&mut self, e: Edit) -> Result<()> {
        let (mut undo, q, _) = self.install(e);
        if let Err(err) = self.propagate(&mut undo, q, &mut Work::default()) {
            self.rollback(undo);
            return Err(err);
        }
        self.roots
            .extend((self.roots.len()..self.nets.len()).map(Source::Net));
        self.rank()
    }
    /// Materializes a report only once per accepted edit, for search targeting.
    pub fn report(&self) -> Result<StaReport> {
        let metrics = self.metrics()?;
        Ok(StaReport {
            net_timing: self.roots.iter().map(|s| self.signal(*s)).collect(),
            worst_output_arrival: metrics.worst_delay,
            worst_output_breakdown: None,
            worst_register_input_arrival: 0.0,
            worst_register_input_breakdown: None,
            register_input_arrivals: vec![None; self.nodes.len()],
            register_input_breakdowns: vec![None; self.nodes.len()],
            timing_query_diagnostic_counts: metrics.timing_queries,
            cell_levels: self
                .endpoints
                .iter()
                .filter_map(|s| {
                    if let Source::Net(n) = s {
                        self.nets[*n].driver.map(|(i, _)| self.nodes[i].rank / 4)
                    } else {
                        None
                    }
                })
                .max()
                .unwrap_or(0),
        })
    }
    /// Checks every driven edge load against an independently rebuilt graph.
    pub fn check_loads(&self, g: &Graph) -> Result<()> {
        for d in &g.drivers {
            ensure!(
                self.nets[d.net].load == d.load,
                "oracle load at net {}",
                d.net
            );
        }
        Ok(())
    }
    /// Exact per-net rise/fall/slew, endpoint maximum and query-count oracle.
    pub fn check(&self, full: &StaReport) -> Result<()> {
        ensure!(full.net_timing.len() == self.nets.len(), "oracle net count");
        for (n, expected) in full.net_timing.iter().enumerate() {
            let s = self.roots.get(n).copied().unwrap_or(Source::Net(n));
            ensure!(
                self.signal(s) == *expected,
                "oracle timing mismatch at net {n}: {:?} vs {:?}",
                self.signal(s),
                expected
            );
        }
        let m = self.metrics()?;
        ensure!(
            m.worst_delay == full.worst_output_arrival,
            "oracle endpoint maximum"
        );
        ensure!(
            m.timing_queries == full.timing_query_diagnostic_counts,
            "oracle query diagnostics: {:?} vs {:?}",
            m.timing_queries,
            full.timing_query_diagnostic_counts
        );
        Ok(())
    }
}
