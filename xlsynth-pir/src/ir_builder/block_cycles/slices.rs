// SPDX-License-Identifier: Apache-2.0

//! Demand-driven wiring dependencies for a suspected whole-node cycle.

mod trace;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use smallvec::SmallVec;

use super::{BuilderError, Candidates, cycle_error, invalid};
use crate::block_feedback::controls::{ControlValues, ScopeId};
use crate::block_feedback::projection::{
    AggregateProjection, BitRange, Demand, Selection, element_type, is_empty_data,
    project_aggregate,
};
use crate::ir::{Block, InstantiationKind, NodeGraph, NodePayload, NodeRef, Package, Type, Unop};
use crate::ir_utils::operands;
use trace::{Lookup, Trace, TraceRequest};

/// Selections of the formal inputs needed for one selected result.
type Inputs = BTreeSet<Demand>;

/// A refinement can request a control from its caller or diagnose invalid IR.
#[derive(Clone)]
enum CheckError {
    Formals(BTreeSet<Demand>),
    Invalid(BuilderError),
}

impl From<BuilderError> for CheckError {
    fn from(value: BuilderError) -> Self {
        Self::Invalid(value)
    }
}

type Checked<T> = Result<T, CheckError>;

/// Caches selected output dependencies and internal checks per specialization.
pub(super) struct Checker<'a> {
    package: &'a Package,
    offsets: ControlValues<'a>,
    summaries: HashMap<(ScopeId, Demand), Checked<Rc<Inputs>>>,
    interiors: HashMap<ScopeId, Checked<()>>,
}

impl<'a> Checker<'a> {
    /// Creates demand-driven analysis state for the checked hierarchy.
    pub(super) fn new(package: &'a Package) -> Self {
        Self {
            package,
            offsets: ControlValues::new(package),
            summaries: HashMap::new(),
            interiors: HashMap::new(),
        }
    }

    /// Checks candidate paths, specializing only the controls they actually
    /// use.
    pub(super) fn check(
        &mut self,
        block: &'a Block,
        candidates: &BTreeMap<String, Candidates>,
    ) -> Result<(), BuilderError> {
        match self.with_bindings(
            block,
            |_, _| None,
            |this, scope| this.check_interior(block, scope, candidates),
        ) {
            Ok(()) => Ok(()),
            Err(CheckError::Invalid(error)) => Err(error),
            Err(CheckError::Formals(_)) => unreachable!("unbound root inputs are runtime"),
        }
    }

    /// Includes hidden loops within instances, even when none of their outputs
    /// are visible at the top.
    fn check_interior(
        &mut self,
        block: &'a Block,
        scope: ScopeId,
        candidates: &BTreeMap<String, Candidates>,
    ) -> Checked<()> {
        if let Some(result) = self.interiors.get(&scope) {
            return result.clone();
        }
        self.offsets.register_block(block);
        let candidate = &candidates[&block.name];
        let result = (|| {
            let mut missing = BTreeSet::new();
            // Independent traces can request different formal values. Gather
            // their requests before restarting in a specialized scope.
            let mut record = |result: Checked<()>| match result {
                Err(CheckError::Formals(formals)) => {
                    missing.extend(formals);
                    Ok(())
                }
                result => result,
            };
            for &index in &candidate.roots {
                let request = TraceRequest {
                    graph: block,
                    scope,
                    root: Demand::whole_connection(block, NodeRef { index }),
                };
                record(self.trace(request))?;
            }
            for &index in &candidate.children {
                let instance = &block.instantiations[index];
                let child = self
                    .package
                    .get_block(&instance.block)
                    .expect("indexed child");
                record(self.with_bindings(
                    child,
                    |this, formal| {
                        this.offsets
                            .connections(block)
                            .driver(block, &instance.name, child.port_name(formal))
                            .map(|arg| (scope, arg))
                    },
                    |this, inner| this.check_interior(child, inner, candidates),
                ))?;
            }
            if missing.is_empty() {
                Ok(())
            } else {
                Err(CheckError::Formals(missing))
            }
        })();
        self.interiors.insert(scope, result.clone());
        result
    }

    /// If an analysis needs a formal selection, resolve its actual driver in
    /// the caller. Other ports, including unrelated data cones, stay untouched.
    fn with_bindings<T>(
        &mut self,
        graph: &'a NodeGraph,
        actual: impl Fn(&Self, NodeRef) -> Option<(ScopeId, NodeRef)>,
        mut run: impl FnMut(&mut Self, ScopeId) -> Checked<T>,
    ) -> Checked<T> {
        let mut scope = self.offsets.unbound(graph);
        loop {
            match run(self, scope) {
                Err(CheckError::Formals(formals)) => {
                    let mut bindings = Vec::with_capacity(formals.len());
                    let mut missing = BTreeSet::new();
                    for formal in formals {
                        let Some((parent, arg)) = actual(self, formal.node()) else {
                            bindings.push((formal, None));
                            continue;
                        };
                        match self.offsets.value(parent, formal.at(arg)) {
                            Ok(value) => bindings.push((formal, value)),
                            Err(outer) => {
                                missing.insert(outer);
                            }
                        }
                    }
                    if !missing.is_empty() {
                        return Err(CheckError::Formals(missing));
                    }
                    scope = self.offsets.bind_all(scope, bindings);
                }
                result => return result,
            }
        }
    }

    /// Summarizes a child output or function return under its necessary control
    /// bindings.
    fn summary(
        &mut self,
        graph: &'a NodeGraph,
        root: Demand,
        actual: impl Fn(&Self, NodeRef) -> Option<(ScopeId, NodeRef)>,
    ) -> Checked<Lookup<'a, Rc<Inputs>>> {
        self.with_bindings(graph, actual, |this, scope| {
            let key = (scope, root.clone());
            match this.summaries.get(&key) {
                Some(result) => result.clone().map(Lookup::Ready),
                None => Ok(Lookup::Pending(TraceRequest {
                    graph,
                    scope,
                    root: root.clone(),
                })),
            }
        })
    }

    /// Runs member-local DFS frames on a heap worklist. A callee can suspend
    /// its caller at the exact node whose dependencies need its summary.
    fn trace(&mut self, request: TraceRequest<'a>) -> Checked<()> {
        let mut stack = vec![Trace::new(request)];
        loop {
            let frame = stack.last_mut().expect("root or pending callee");
            let result = match frame.advance(self) {
                Ok(Lookup::Pending(callee)) => {
                    stack.push(Trace::new(callee));
                    continue;
                }
                Ok(Lookup::Ready(inputs)) => Ok(Rc::new(inputs)),
                Err(error) => Err(error),
            };
            let completed = stack.pop().expect("completed trace");
            if stack.is_empty() {
                return result.map(|_| ());
            }
            self.summaries
                .insert((completed.request.scope, completed.request.root), result);
        }
    }

    /// Traces wires, selected fields and bit ranges. Other operations
    /// conservatively depend on their whole operands.
    fn dependencies(
        &mut self,
        graph: &'a NodeGraph,
        visit: &Demand,
        scope: ScopeId,
    ) -> Checked<Lookup<'a, Vec<Demand>>> {
        let payload = &graph.get_node(visit.node()).payload;
        if let Some(driver) = visit.driver(graph) {
            return Ok(Lookup::Ready(vec![visit.at(driver)]));
        }
        if is_empty_data(graph.get_node_ty(visit.node())) {
            return Ok(Lookup::Ready(Vec::new()));
        }
        let mut missing = None;
        let mut read_control =
            |node: NodeRef| match self.offsets.value(scope, Demand::whole(graph, node)) {
                Ok(value) => value,
                Err(formal) => {
                    missing = Some(formal);
                    None
                }
            };
        let selection = &visit.part;
        let deps = match payload {
            NodePayload::Invoke { to_apply, operands }
                if let Some(function) = self.package.get_fn(to_apply)
                    && function.ffi_proto().is_none() =>
            {
                let result = function
                    .ret_node_ref
                    .ok_or_else(|| invalid("invoked function has no return value"))?;
                let inputs = self.summary(function, visit.at(result), |_, formal| {
                    function.param_index(formal).map(|i| (scope, operands[i]))
                })?;
                return Ok(inputs.map(|inputs| {
                    inputs
                        .iter()
                        .map(|input| {
                            let i = function
                                .param_index(input.node())
                                .expect("verified function parameter");
                            input.at(operands[i])
                        })
                        .collect()
                }));
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } => {
                return self.instance_dependencies(
                    graph,
                    instantiation,
                    port_name,
                    selection,
                    scope,
                );
            }
            NodePayload::Unop(Unop::Identity, arg) => vec![visit.at(*arg)],
            _ if let Some(projection) =
                project_aggregate(graph, visit.node(), &selection.path, &mut read_control) =>
            {
                match projection {
                    AggregateProjection::Forward { node, path } => vec![Demand::value(
                        node,
                        Selection {
                            path,
                            bits: selection.bits,
                        },
                    )],
                    AggregateProjection::Or(arms) => {
                        arms.into_iter().map(|node| visit.at(node)).collect()
                    }
                    AggregateProjection::Elements(count) => {
                        let Some(Type::Array(dim)) =
                            element_type(graph.get_node_ty(visit.node()), &selection.path)
                        else {
                            unreachable!("only an array splits into elements");
                        };
                        (0..count)
                            .map(|index| {
                                let mut part = Selection::whole(&dim.element_type);
                                part.path = selection.path.clone();
                                part.path.push(index);
                                Demand::value(visit.node(), part)
                            })
                            .collect()
                    }
                }
            }
            _ if selection.path.is_empty()
                && let Some((lo, hi)) = selection.bits
                && let Some(projection) = (BitRange {
                    node: visit.node(),
                    lo,
                    hi,
                })
                .project(graph, read_control) =>
            {
                projection
                    .inputs()
                    .iter()
                    .map(|part| {
                        Demand::value(
                            part.node,
                            Selection {
                                path: SmallVec::new(),
                                bits: Some((part.lo, part.hi)),
                            },
                        )
                    })
                    .collect()
            }
            _ => operands(payload)
                .into_iter()
                .map(|arg| Demand::whole(graph, arg))
                .collect(),
        };
        match missing {
            Some(formal) => Err(CheckError::Formals(BTreeSet::from([formal]))),
            None => Ok(Lookup::Ready(deps)),
        }
    }

    /// Projects a child's output back onto this instance's input drivers.
    fn instance_dependencies(
        &mut self,
        parent: &'a NodeGraph,
        instance: &'a str,
        port: &str,
        selection: &Selection,
        scope: ScopeId,
    ) -> Checked<Lookup<'a, Vec<Demand>>> {
        let bindings = self.offsets.connections(parent);
        let definition = bindings
            .definitions
            .get(instance)
            .ok_or_else(|| invalid(format!("unknown instance '{instance}'")))?;
        if definition.kind == InstantiationKind::Extern {
            return Ok(Lookup::Ready(Vec::new()));
        }
        let target = definition.block.as_str();
        let child = self
            .package
            .get_block(target)
            .ok_or_else(|| invalid(format!("missing child block '{target}'")))?;
        let output = child
            .get_output_port(port)
            .ok_or_else(|| invalid(format!("unknown output '{port}' on child '{target}'")))?;

        self.offsets.register_block(child);
        let inputs = self.summary(
            child,
            Demand::port_driver(output, selection.clone()),
            |this, formal| {
                this.offsets
                    .connections(parent)
                    .driver(parent, instance, child.port_name(formal))
                    .map(|arg| (scope, arg))
            },
        )?;
        let drivers = self.offsets.connections(parent).inputs.get(instance);
        let inputs = match inputs {
            Lookup::Pending(request) => return Ok(Lookup::Pending(request)),
            Lookup::Ready(inputs) => inputs,
        };
        inputs
            .iter()
            .map(|input| {
                let name = child.port_name(input.node());
                let node = drivers.and_then(|ports| ports.get(name)).ok_or_else(|| {
                    invalid(format!(
                        "unconnected input '{name}' on instance '{instance}'"
                    ))
                })?;
                Ok(Demand::port_driver(*node, input.part.clone()))
            })
            .collect::<Checked<_>>()
            .map(Lookup::Ready)
    }
}
