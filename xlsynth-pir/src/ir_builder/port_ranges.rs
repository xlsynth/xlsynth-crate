// SPDX-License-Identifier: Apache-2.0

//! Split electrical and dataflow ranges only at boundaries observable by a
//! connection.

use super::{BlockCycleConnection, BlockCycleStep, BuilderError};
use crate::ir::{Block, NodePayload, NodeRef};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// An ascending range on a local signal or a named interface port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Span<S> {
    pub signal: S,
    pub start: usize,
    pub width: usize,
}

impl<S: Copy> Span<S> {
    pub(super) fn part(self, start: usize, width: usize) -> Self {
        Self {
            signal: self.signal,
            start: self.start + start,
            width,
        }
    }
}

/// Ties collapse passive connections. Aligned dependencies preserve bit
/// positions. Opaque operations conservatively depend on all bits of each
/// operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub(super) enum Link {
    Tie,
    Aligned,
    All,
}

/// The cached electrical aliases and directed dependencies of a block
/// interface.
#[derive(Default)]
pub(super) struct Summary {
    pub inouts: BTreeMap<String, crate::ir::Type>,
    pub links: Vec<(Span<String>, Link, Span<String>)>,
}

/// One signal's contribution to an interface summary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Direction {
    Input,
    Output,
    Inout,
}

/// Unpartitioned signals and their span-to-span connections.
#[derive(Default)]
pub(super) struct Network {
    widths: Vec<usize>,
    witnesses: Vec<Option<NodeRef>>,
    pub ports: BTreeMap<String, (Span<usize>, Direction)>,
    links: Vec<(Span<usize>, Link, Span<usize>)>,
}

impl Network {
    pub fn signal(&mut self, width: usize, witness: Option<NodeRef>) -> Span<usize> {
        let signal = self.widths.len();
        self.widths.push(width);
        self.witnesses.push(witness);
        Span {
            signal,
            start: 0,
            width,
        }
    }

    pub fn witness(&mut self, s: Span<usize>, node: NodeRef) {
        self.witnesses[s.signal].get_or_insert(node);
    }

    pub fn connect(
        &mut self,
        a: Span<usize>,
        kind: Link,
        b: Span<usize>,
    ) -> Result<(), BuilderError> {
        for s in [a, b] {
            if s.start
                .checked_add(s.width)
                .is_none_or(|end| end > self.widths[s.signal])
            {
                return Err(super::block_cycles::invalid("out-of-range dependency span"));
            }
        }
        if kind != Link::All && a.width != b.width {
            return Err(super::block_cycles::invalid(
                "incompatible dependency widths",
            ));
        }
        if a.width != 0 && b.width != 0 {
            self.links.push((a, kind, b));
        }
        Ok(())
    }

    /// Propagate each new cut along offset-preserving mappings in both
    /// directions. An unfragmented bus costs one vertex, independent of its bit
    /// width.
    fn cuts(&self) -> Vec<Vec<usize>> {
        let mut cuts: Vec<_> = self
            .widths
            .iter()
            .map(|&w| BTreeSet::from([0, w]))
            .collect();
        let mut maps = vec![Vec::new(); cuts.len()];
        let mut pending = VecDeque::new();
        let mut new_cuts = vec![BTreeSet::new(); cuts.len()];
        let mut add_cut = |signal: usize,
                           cut: usize,
                           queue: &mut VecDeque<usize>,
                           new: &mut [BTreeSet<usize>]| {
            if cuts[signal].insert(cut) {
                if new[signal].is_empty() {
                    queue.push_back(signal);
                }
                new[signal].insert(cut);
            }
        };
        for &(a, kind, b) in &self.links {
            for (source, target) in [(a, b), (b, a)] {
                add_cut(source.signal, source.start, &mut pending, &mut new_cuts);
                add_cut(
                    source.signal,
                    source.start + source.width,
                    &mut pending,
                    &mut new_cuts,
                );
                if kind != Link::All {
                    maps[source.signal].push((source, target));
                }
            }
        }
        // Batch all pending cuts on a signal. Disjoint selects on a wide bus
        // visit each incident mapping once per batch, rather than once per bit.
        while let Some(signal) = pending.pop_front() {
            let batch = std::mem::take(&mut new_cuts[signal]);
            for &(source, target) in &maps[signal] {
                for &cut in batch.range(source.start + 1..source.start + source.width) {
                    add_cut(
                        target.signal,
                        target.start + (cut - source.start),
                        &mut pending,
                        &mut new_cuts,
                    );
                }
            }
        }
        cuts.into_iter().map(|c| c.into_iter().collect()).collect()
    }

    /// Split signals on matching cut points, collapse passive aliases, and
    /// install directed combinational edges on the resulting representatives.
    pub fn finish(&self) -> Graph {
        let cuts = self.cuts();
        let mut count = 0;
        let bases: Vec<_> = cuts
            .iter()
            .map(|c| {
                let start = count;
                count += c.len() - 1;
                start
            })
            .collect();
        let vertices = |s: Span<usize>| {
            let c = &cuts[s.signal];
            let base = bases[s.signal];
            base + c.binary_search(&s.start).unwrap()
                ..base + c.binary_search(&(s.start + s.width)).unwrap()
        };
        let mut parents: Vec<usize> = (0..count).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while i != p[i] {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for &(a, kind, b) in &self.links {
            if kind == Link::Tie {
                for (left, right) in vertices(a).zip(vertices(b)) {
                    let l = root(&mut parents, left);
                    let r = root(&mut parents, right);
                    parents[r] = l;
                }
            }
        }
        for i in 0..count {
            parents[i] = root(&mut parents, i);
        }
        let mut graph = Graph {
            edges: vec![Vec::new(); count],
            witness: vec![None; count],
            ports: Vec::new(),
        };
        for (i, witness) in self.witnesses.iter().enumerate() {
            if let Some(w) = witness {
                for &id in &parents[bases[i]..bases[i] + cuts[i].len() - 1] {
                    graph.witness[id].get_or_insert(*w);
                }
            }
        }
        for &(a, kind, b) in &self.links {
            match kind {
                Link::Tie => { /* The endpoints already share vertices. */ }
                Link::Aligned => {
                    for (l, r) in vertices(a).zip(vertices(b)) {
                        graph.edges[parents[l]].push((parents[r], kind));
                    }
                }
                Link::All => {
                    let ls = vertices(a);
                    let rs = vertices(b);
                    // A shared intermediate vertex avoids a Cartesian product
                    // for fragmented operands.
                    if ls.len() > 1 && rs.len() > 1 {
                        let hub = graph.edges.len();
                        graph.edges.push(rs.map(|r| (parents[r], kind)).collect());
                        graph.witness.push(self.witnesses[a.signal]);
                        for l in ls {
                            graph.edges[parents[l]].push((hub, kind));
                        }
                    } else {
                        for l in ls {
                            for r in rs.clone() {
                                graph.edges[parents[l]].push((parents[r], kind));
                            }
                        }
                    }
                }
            }
        }
        for (name, (s, dir)) in &self.ports {
            for (i, w) in cuts[s.signal].windows(2).enumerate() {
                graph.ports.push((
                    Span {
                        signal: name.clone(),
                        start: w[0],
                        width: w[1] - w[0],
                    },
                    *dir,
                    parents[bases[s.signal] + i],
                ));
            }
        }
        graph
    }
}

/// Partitioned graph with one representative per passive electrical class.
pub(super) struct Graph {
    edges: Vec<Vec<(usize, Link)>>,
    witness: Vec<Option<NodeRef>>,
    ports: Vec<(Span<String>, Direction, usize)>,
}

impl Graph {
    /// Check every component for cycles and report the IR operations on the
    /// cycle.
    pub fn check(&self, block: &Block) -> Result<(), BuilderError> {
        let mut colors = vec![0u8; self.edges.len()];
        for root in 0..self.edges.len() {
            if colors[root] != 0 {
                continue;
            }
            let mut stack = vec![(root, 0)];
            colors[root] = 1;
            while let Some((v, i)) = stack.last_mut() {
                if let Some(&(next, _)) = self.edges[*v].get(*i) {
                    *i += 1;
                    if colors[next] == 1 {
                        let start = stack.iter().position(|&(v, _)| v == next).unwrap();
                        let path = &stack[start..];
                        let steps = path
                            .iter()
                            .zip(path.iter().cycle().skip(1))
                            .map(|(&(a, _), &(b, _))| {
                                let node = block
                                    .get_node(self.witness[a].expect("cycle vertex has an access"));
                                let dep = block
                                    .get_node(self.witness[b].expect("cycle vertex has an access"));
                                let connection = match (&node.payload, &dep.payload) {
                                    (
                                        NodePayload::InstantiationOutput {
                                            instantiation,
                                            port_name,
                                        },
                                        NodePayload::InstantiationInput {
                                            port_name: input, ..
                                        },
                                    ) => Some(BlockCycleConnection {
                                        instance: instantiation.clone(),
                                        output_port: port_name.clone(),
                                        input_port: input.clone(),
                                        input_node_id: dep.text_id,
                                    }),
                                    _ => None,
                                };
                                BlockCycleStep {
                                    node_id: node.text_id,
                                    connection,
                                }
                            })
                            .collect();
                        return Err(BuilderError::CombinationalCycle {
                            block: block.name.clone(),
                            steps,
                        });
                    }
                    if colors[next] == 0 {
                        colors[next] = 1;
                        stack.push((next, 0));
                    }
                } else {
                    colors[*v] = 2;
                    stack.pop();
                }
            }
        }
        Ok(())
    }

    /// Summarize one output cone at a time. A path through any opaque edge
    /// summarizes to all-to-all; otherwise corresponding bit positions suffice.
    pub fn summarize(self, block: &Block) -> Summary {
        let mut summary = Summary {
            inouts: block
                .inout_ports()
                .map(|p| (block.port_name(p).to_owned(), block.port_type(p).clone()))
                .collect(),
            links: Vec::new(),
        };
        let mut inputs = BTreeMap::new();
        let mut outputs = Vec::new();
        for (span, dir, v) in self.ports {
            if dir == Direction::Inout
                && let Some(first) = inputs.get(&v)
            {
                summary.links.push((span, Link::Tie, Span::clone(first)));
                continue;
            }
            if dir != Direction::Output {
                inputs.insert(v, span.clone());
            }
            if dir != Direction::Input {
                outputs.push((v, span));
            }
        }
        let mut visited = vec![(usize::MAX, false); self.edges.len()];
        for (root, span) in outputs {
            let mut pending = vec![(root, false)];
            let mut deps = BTreeMap::<usize, bool>::new();
            while let Some((v, opaque)) = pending.pop() {
                let (seen, all) = visited[v];
                if seen == root && (all || !opaque) {
                    continue;
                }
                visited[v] = (root, opaque);
                if v != root && inputs.contains_key(&v) {
                    deps.entry(v).and_modify(|p| *p |= opaque).or_insert(opaque);
                }
                for &(dep, kind) in &self.edges[v] {
                    pending.push((dep, opaque || kind == Link::All));
                }
            }
            for (v, all) in deps {
                summary.links.push((
                    span.clone(),
                    if all { Link::All } else { Link::Aligned },
                    inputs[&v].clone(),
                ));
            }
        }
        summary
    }
}
