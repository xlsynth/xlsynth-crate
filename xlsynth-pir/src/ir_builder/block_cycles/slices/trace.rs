// SPDX-License-Identifier: Apache-2.0

//! The suspended DFS for one block output or function return.

use std::collections::{BTreeSet, HashSet};
use std::vec::IntoIter;

use super::{
    CheckError, Checked, Checker, Demand, Inputs, NodeGraph, NodePayload, ScopeId, cycle_error,
    is_empty_data,
};

pub(super) struct TraceRequest<'a> {
    pub(super) graph: &'a NodeGraph,
    pub(super) scope: ScopeId,
    pub(super) root: Demand,
}

/// A dependency query either has an answer or needs another member's summary.
pub(super) enum Lookup<'a, T> {
    Ready(T),
    Pending(TraceRequest<'a>),
}

impl<'a, T> Lookup<'a, T> {
    /// Maps a completed result while forwarding a pending summary request.
    pub(super) fn map<U>(self, f: impl FnOnce(T) -> U) -> Lookup<'a, U> {
        match self {
            Self::Ready(value) => Lookup::Ready(f(value)),
            Self::Pending(request) => Lookup::Pending(request),
        }
    }
}

pub(super) struct Trace<'a> {
    pub(super) request: TraceRequest<'a>,
    // None means the node's edges still need to be resolved.
    path: Vec<(Demand, Option<IntoIter<Demand>>)>,
    active: HashSet<Demand>,
    done: HashSet<Demand>,
    inputs: Inputs,
    missing: BTreeSet<Demand>,
}

impl<'a> Trace<'a> {
    /// Starts a member-local traversal at the requested selection.
    pub(super) fn new(request: TraceRequest<'a>) -> Self {
        Self {
            path: vec![(request.root.clone(), None)],
            active: HashSet::from([request.root.clone()]),
            done: HashSet::new(),
            inputs: Inputs::new(),
            missing: BTreeSet::new(),
            request,
        }
    }

    /// Advances to the next missing callee summary or completes this trace.
    /// A recurrence is a cycle only if the same selection is still active.
    pub(super) fn advance(&mut self, checker: &mut Checker<'a>) -> Checked<Lookup<'a, Inputs>> {
        let graph = self.request.graph;
        while let Some((visit, edges)) = self.path.last_mut() {
            let Some(edges) = edges else {
                *edges = Some(
                    match checker.dependencies(graph, visit, self.request.scope) {
                        Ok(Lookup::Pending(request)) => return Ok(Lookup::Pending(request)),
                        Ok(Lookup::Ready(deps)) => deps.into_iter(),
                        Err(CheckError::Formals(missing)) => {
                            // Another branch of this same cone may need
                            // different formals. Collect both before asking the
                            // caller.
                            self.missing.extend(missing);
                            Vec::new().into_iter()
                        }
                        Err(error) => return Err(error),
                    },
                );
                continue;
            };
            if let Some(next) = edges.next() {
                if self.done.contains(&next) {
                    continue;
                }
                if !self.active.insert(next.clone()) {
                    let path = self
                        .path
                        .iter()
                        .skip_while(|(v, _)| *v != next)
                        .map(|(v, _)| v.node().index)
                        .collect::<Vec<_>>();
                    return Err(cycle_error(graph, &path).into());
                }
                self.path.push((next, None));
            } else {
                if matches!(
                    graph.get_node(visit.node()).payload,
                    NodePayload::InputPort { .. } | NodePayload::Param
                ) && !is_empty_data(graph.get_node_ty(visit.node()))
                {
                    self.inputs.insert(visit.clone());
                }
                self.active.remove(visit);
                self.done
                    .insert(self.path.pop().expect("active DFS node").0);
            }
        }
        if self.missing.is_empty() {
            Ok(Lookup::Ready(std::mem::take(&mut self.inputs)))
        } else {
            Err(CheckError::Formals(std::mem::take(&mut self.missing)))
        }
    }
}
