// SPDX-License-Identifier: Apache-2.0

//! Track combinational dependencies between bit ranges of ordinary block ports.

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

/// Aligned dependencies preserve bit positions. Opaque operations
/// conservatively depend on all bits of each operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub(super) enum Link {
    Aligned,
    All,
}

/// Cached dependencies from ordinary outputs to ordinary inputs of a block.
pub(super) type Summary = Vec<(Span<String>, Link, Span<String>)>;

/// One signal's contribution to an interface summary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Direction {
    Input,
    Output,
}

/// Unpartitioned signals and their span-to-span connections.
#[derive(Default)]
pub(super) struct Network {
    widths: Vec<usize>,
    witnesses: Vec<NodeRef>,
    pub ports: BTreeMap<String, (Span<usize>, Direction)>,
    links: Vec<(Span<usize>, Link, Span<usize>)>,
}

impl Network {
    pub fn signal(&mut self, width: usize, witness: NodeRef) -> Span<usize> {
        let signal = self.widths.len();
        self.widths.push(width);
        self.witnesses.push(witness);
        Span {
            signal,
            start: 0,
            width,
        }
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

    /// Split signals on matching cut points and add combinational dependencies.
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
        let mut graph = Graph {
            edges: vec![Vec::new(); count],
            witness: self
                .witnesses
                .iter()
                .zip(&cuts)
                .flat_map(|(&node, c)| std::iter::repeat_n(node, c.len() - 1))
                .collect(),
            ports: Vec::new(),
        };
        for &(a, kind, b) in &self.links {
            match kind {
                Link::Aligned => {
                    for (l, r) in vertices(a).zip(vertices(b)) {
                        graph.edges[l].push((r, kind));
                    }
                }
                Link::All => {
                    let ls = vertices(a);
                    let rs = vertices(b);
                    // A shared intermediate vertex avoids a Cartesian product
                    // for fragmented operands.
                    if ls.len() > 1 && rs.len() > 1 {
                        let hub = graph.edges.len();
                        graph.edges.push(rs.map(|r| (r, kind)).collect());
                        graph.witness.push(self.witnesses[a.signal]);
                        for l in ls {
                            graph.edges[l].push((hub, kind));
                        }
                    } else {
                        for l in ls {
                            for r in rs.clone() {
                                graph.edges[l].push((r, kind));
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
                    bases[s.signal] + i,
                ));
            }
        }
        graph
    }
}

/// Partitioned combinational dependencies, with IR nodes for cycle diagnostics.
pub(super) struct Graph {
    edges: Vec<Vec<(usize, Link)>>,
    witness: Vec<NodeRef>,
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
                                let node = block.get_node(self.witness[a]);
                                let dep = block.get_node(self.witness[b]);
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
    pub fn summarize(self) -> Summary {
        let mut summary = Vec::new();
        let mut inputs = BTreeMap::new();
        let mut outputs = Vec::new();
        for (span, dir, v) in self.ports {
            match dir {
                Direction::Input => {
                    inputs.insert(v, span);
                }
                Direction::Output => outputs.push((v, span)),
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
                if inputs.contains_key(&v) {
                    deps.entry(v).and_modify(|p| *p |= opaque).or_insert(opaque);
                }
                for &(dep, kind) in &self.edges[v] {
                    pending.push((dep, opaque || kind == Link::All));
                }
            }
            for (v, all) in deps {
                summary.push((
                    span.clone(),
                    if all { Link::All } else { Link::Aligned },
                    inputs[&v].clone(),
                ));
            }
        }
        summary
    }
}
