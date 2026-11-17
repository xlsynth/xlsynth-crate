// SPDX-License-Identifier: Apache-2.0

//! Static control values selected through hierarchical wiring and local calls.

use std::collections::{BTreeMap, HashMap};
use std::ptr;
use std::rc::Rc;

use smallvec::SmallVec;

use super::InstanceConnections;
use super::projection::{
    AggregateProjection, BitProjection, BitRange, Demand, Selection, element_type,
    project_aggregate,
};
use crate::ir::{
    Block, InstantiationKind, Node, NodeGraph, NodePayload, NodeRef, Package, Type, Unop,
};
use crate::ir_eval::eval_pure_if_supported;
use crate::ir_utils::operands;
use crate::ir_value_utils::{deep_or_ir_values_for_type, zero_ir_value_for_type};
use crate::{IrBits, IrValue};

pub(crate) type ScopeId = usize;

type Bindings = BTreeMap<Demand, Option<Rc<IrValue>>>;

/// A scope describes just the formal selections whose values were required.
/// Equal values reuse one definition's results across call and instance sites.
struct Scope<'a> {
    graph: &'a NodeGraph,
    bindings: Rc<Bindings>,
}

/// A selected value in a definition with specific control bindings.
#[derive(Clone, Eq, Hash, PartialEq)]
struct Query {
    scope: ScopeId,
    demand: Demand,
}

impl Query {
    fn operand(&self, node: NodeRef) -> Self {
        Self {
            scope: self.scope,
            demand: self.demand.at(node),
        }
    }
}

/// A callee may request the selected bits of a formal before its result can
/// be classified as constant or dependent on runtime data.
#[derive(Clone)]
enum Answer {
    Value(Rc<IrValue>),
    Formal(Demand),
    Runtime,
}

type Pending = Vec<Query>;
type Evaluation = Result<Answer, Pending>;

/// Resolves static wiring under lazily supplied formal bindings. Queries live
/// at member definitions, so repeated calls can share the same specialized
/// result without expanding the call tree.
pub(crate) struct ControlValues<'a> {
    package: &'a Package,
    // Borrowed graphs keep stable addresses throughout the analysis. Node
    // indices belong to a particular graph, even when a block and a function
    // share a name.
    connections: HashMap<*const NodeGraph, InstanceConnections<'a>>,
    scopes: Vec<Scope<'a>>,
    ids: HashMap<(*const NodeGraph, Rc<Bindings>), ScopeId>,
    calls: HashMap<Query, ScopeId>,
    answers: HashMap<Query, Answer>,
    empty: Rc<Bindings>,
}

impl<'a> ControlValues<'a> {
    /// Starts a lookup for a package with a possibly independent root block.
    pub(crate) fn new(package: &'a Package) -> Self {
        Self {
            package,
            connections: HashMap::new(),
            scopes: Vec::new(),
            ids: HashMap::new(),
            calls: HashMap::new(),
            answers: HashMap::new(),
            empty: Rc::default(),
        }
    }

    /// Registers complete local port bindings, independent of declaration
    /// order.
    pub(crate) fn register_block(&mut self, block: &'a Block) {
        if !block.instantiations.is_empty() {
            self.connections
                .entry(ptr::from_ref(&block.graph))
                .or_insert_with(|| InstanceConnections::new(block));
        }
    }

    /// Returns the instance wiring shared with selected-range analysis.
    pub(crate) fn connections(&self, parent: &NodeGraph) -> &InstanceConnections<'a> {
        &self.connections[&ptr::from_ref(parent)]
    }

    /// Interns the exact formal values used by a control query.
    fn specialize(&mut self, graph: &'a NodeGraph, bindings: Rc<Bindings>) -> ScopeId {
        *self
            .ids
            .entry((ptr::from_ref(graph), bindings.clone()))
            .or_insert_with(|| {
                let id = self.scopes.len();
                self.scopes.push(Scope { graph, bindings });
                id
            })
    }

    /// Returns the definition's scope before a caller supplies any selections.
    pub(crate) fn unbound(&mut self, graph: &'a NodeGraph) -> ScopeId {
        self.specialize(graph, self.empty.clone())
    }

    /// Records the caller's answer for the exact formal selection requested.
    pub(crate) fn bind(
        &mut self,
        scope: ScopeId,
        formal: Demand,
        value: Option<Rc<IrValue>>,
    ) -> ScopeId {
        self.bind_all(scope, [(formal, value)])
    }

    /// Interns one specialization after binding all independently requested
    /// formal selections in the caller.
    pub(crate) fn bind_all(
        &mut self,
        scope: ScopeId,
        values: impl IntoIterator<Item = (Demand, Option<Rc<IrValue>>)>,
    ) -> ScopeId {
        let graph = self.scopes[scope].graph;
        let mut bindings = (*self.scopes[scope].bindings).clone();
        bindings.extend(values);
        self.specialize(graph, Rc::new(bindings))
    }

    /// Reads a control in the root's wiring context, preserving every mask bit.
    pub(crate) fn root_value(
        &mut self,
        graph: &'a NodeGraph,
        node: NodeRef,
    ) -> Option<Rc<IrValue>> {
        let scope = self.unbound(graph);
        self.value(scope, Demand::whole(graph, node)).ok().flatten()
    }

    /// Resolves only the selected upstream dependencies. A query in progress
    /// temporarily yields runtime if feedback reaches that same selection.
    pub(crate) fn value(
        &mut self,
        scope: ScopeId,
        demand: Demand,
    ) -> Result<Option<Rc<IrValue>>, Demand> {
        let root = Query { scope, demand };
        let mut stack = vec![(root.clone(), false)];
        while let Some((at, resume)) = stack.pop() {
            if !resume {
                if self.answers.contains_key(&at) {
                    continue;
                }
                self.answers.insert(at.clone(), Answer::Runtime);
            }
            match self.evaluate(&at) {
                Ok(answer) => {
                    self.answers.insert(at, answer);
                }
                Err(pending) => {
                    stack.push((at, true));
                    stack.extend(pending.into_iter().rev().map(|query| (query, false)));
                }
            }
        }
        match &self.answers[&root] {
            Answer::Value(value) => Ok(Some(value.clone())),
            Answer::Formal(formal) => Err(formal.clone()),
            Answer::Runtime => Ok(None),
        }
    }

    /// Fetches a previously completed prerequisite, or schedules it.
    fn read(&self, query: Query) -> Evaluation {
        self.answers.get(&query).cloned().ok_or_else(|| vec![query])
    }

    /// Supplies callee formals only when its selected result needs them.
    fn callee(
        &mut self,
        at: &Query,
        graph: &'a NodeGraph,
        result: Demand,
        actual: impl Fn(&Self, NodeRef) -> Option<NodeRef>,
    ) -> Evaluation {
        let mut scope = self
            .calls
            .get(at)
            .copied()
            .unwrap_or_else(|| self.unbound(graph));
        loop {
            let next = Query {
                scope,
                demand: result.clone(),
            };
            let formal = match self.read(next)? {
                Answer::Formal(formal) => formal,
                result => return Ok(result),
            };
            let Some(driver) = actual(self, formal.node()) else {
                return Ok(Answer::Runtime);
            };
            match self.read(Query {
                scope: at.scope,
                demand: formal.at(driver),
            })? {
                Answer::Value(value) => {
                    scope = self.bind(scope, formal, Some(value));
                    self.calls.insert(at.clone(), scope);
                }
                other => return Ok(other),
            }
        }
    }

    /// Resolves selected ports, local calls, and the common projection rules.
    fn evaluate(&mut self, at: &Query) -> Evaluation {
        let scope = &self.scopes[at.scope];
        let graph = scope.graph;
        let node = at.demand.node();
        let op = graph.get_node(node);
        let part = &at.demand.part;
        if let Some(driver) = at.demand.driver(graph) {
            return self.read(at.operand(driver));
        }
        if matches!(part.bits, Some((lo, hi)) if lo == hi) {
            return Ok(Answer::Value(Rc::new(IrValue::Bits(IrBits::zero(0)))));
        }
        match &op.payload {
            NodePayload::Literal(value) => return Ok(part.value(value)),
            NodePayload::Param | NodePayload::InputPort { .. } => {
                if let Some(value) = scope.bindings.get(&at.demand) {
                    return Ok(value.clone().map_or(Answer::Runtime, Answer::Value));
                }
                let whole = Demand::whole(graph, node);
                return Ok(match scope.bindings.get(&whole) {
                    Some(Some(value)) => part.value(value),
                    _ => Answer::Formal(at.demand.clone()),
                });
            }
            NodePayload::Unop(Unop::Identity, arg) => return self.read(at.operand(*arg)),
            NodePayload::OutputPort { .. } | NodePayload::InstantiationInput { .. } => {
                return Ok(Answer::Value(Rc::new(IrValue::make_tuple(&[]))));
            }
            NodePayload::Invoke { to_apply, operands }
                if let Some(function) = self.package.get_fn(to_apply)
                    && function.ffi_proto().is_none()
                    && let Some(result) = function.ret_node_ref =>
            {
                return self.callee(at, function, at.demand.at(result), |_, formal| {
                    function
                        .param_index(formal)
                        .and_then(|i| operands.get(i))
                        .copied()
                });
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } if let Some(instance) = self
                .connections(graph)
                .definitions
                .get(instantiation.as_str())
                && instance.kind == InstantiationKind::Block
                && let Some(child) = self.package.get_block(&instance.block)
                && let Some(output) = child.get_output_port(port_name) =>
            {
                self.register_block(child);
                return self.callee(
                    at,
                    child,
                    Demand::port_driver(output, part.clone()),
                    |this, formal| {
                        this.connections(graph).driver(
                            graph,
                            instantiation,
                            child.port_name(formal),
                        )
                    },
                );
            }
            _ => { /* Local structural operators use the same selections as the cycle checker. */ }
        }

        let mut needed = None;
        let mut control_value = |control: NodeRef| match self.read(Query {
            scope: at.scope,
            demand: Demand::whole(graph, control),
        }) {
            Ok(Answer::Value(value)) => Some(value),
            other => {
                needed = Some(other);
                None
            }
        };
        let aggregate = project_aggregate(graph, node, &part.path, &mut control_value);
        let bits = if aggregate.is_none() && part.path.is_empty() {
            part.bits
                .filter(|(lo, hi)| lo < hi)
                .and_then(|(lo, hi)| BitRange { node, lo, hi }.project(graph, &mut control_value))
        } else {
            None
        };
        if let Some(answer) = needed {
            return answer;
        }
        if let Some(projection) = aggregate {
            return match projection {
                AggregateProjection::Forward { node, path } => self.read(Query {
                    scope: at.scope,
                    demand: Demand::value(
                        node,
                        Selection {
                            path,
                            bits: part.bits,
                        },
                    ),
                }),
                AggregateProjection::Or(arms) => {
                    let ty = match part.bits {
                        Some((lo, hi)) => Type::Bits(hi - lo),
                        None => element_type(&op.ty, &part.path)
                            .expect("verified selection")
                            .clone(),
                    };
                    let inputs = arms.into_iter().map(|node| at.operand(node));
                    self.with_values(inputs, |values| {
                        Some(values.iter().fold(zero_ir_value_for_type(&ty), |acc, v| {
                            deep_or_ir_values_for_type(&ty, &acc, v)
                        }))
                    })
                }
                AggregateProjection::Elements(count) => {
                    let Some(Type::Array(dim)) = element_type(&op.ty, &part.path) else {
                        return Ok(Answer::Runtime);
                    };
                    let queries = (0..count).map(|i| {
                        let mut field = Selection::whole(&dim.element_type);
                        field.path = part.path.clone();
                        field.path.push(i);
                        Query {
                            scope: at.scope,
                            demand: Demand::value(node, field),
                        }
                    });
                    self.with_values(queries, |values| {
                        IrValue::make_array_typed(
                            (*dim.element_type).clone(),
                            &values.iter().map(|v| (**v).clone()).collect::<Vec<_>>(),
                        )
                        .ok()
                    })
                }
            };
        }
        if let Some(projection) = bits {
            let parts = projection
                .inputs()
                .iter()
                .map(|p| Query {
                    scope: at.scope,
                    demand: Demand::value(
                        p.node,
                        Selection {
                            path: SmallVec::new(),
                            bits: Some((p.lo, p.hi)),
                        },
                    ),
                })
                .collect::<Vec<_>>();
            let (lo, hi) = part.bits.expect("bit projection");
            return self.with_values(parts.into_iter(), |args| {
                eval_projection(projection, hi - lo, args)
            });
        }
        // An unprojected whole tuple or array requires every constituent.
        // Unknown controls and violated bounds in routing operators stay
        // dynamic.
        if matches!(op.payload, NodePayload::Array(_) | NodePayload::Tuple(_))
            && part.path.is_empty()
        {
            let args = operands(&op.payload).into_iter().map(|node| Query {
                scope: at.scope,
                demand: Demand::whole(graph, node),
            });
            return self.with_values(args, |args| eval_pure_if_supported(op, args));
        }
        Ok(Answer::Runtime)
    }

    /// Schedules all missing operands in one pass, including at large fan-in.
    fn with_values(
        &self,
        queries: impl Iterator<Item = Query>,
        eval: impl FnOnce(&[&IrValue]) -> Option<IrValue>,
    ) -> Evaluation {
        let mut args = Vec::new();
        let mut pending = Vec::new();
        for query in queries {
            match self.answers.get(&query) {
                Some(Answer::Value(value)) => args.push(value.as_ref()),
                Some(unknown) => return Ok(unknown.clone()),
                None => pending.push(query),
            }
        }
        if !pending.is_empty() {
            return Err(pending);
        }
        Ok(eval(&args).map_or(Answer::Runtime, |v| Answer::Value(Rc::new(v))))
    }
}

impl Selection {
    /// Extracts the exact formal or literal selection, including wide bits.
    fn value(&self, mut value: &IrValue) -> Answer {
        for &index in &self.path {
            let Ok(elements) = value.as_elements() else {
                return Answer::Runtime;
            };
            let Some(element) = elements.get(index) else {
                return Answer::Runtime;
            };
            value = element;
        }
        if let Some((lo, hi)) = self.bits {
            let Ok(bits) = value.as_bits() else {
                return Answer::Runtime;
            };
            if lo != 0 || hi != bits.get_bit_count() {
                return Answer::Value(Rc::new(IrValue::Bits(
                    bits.width_slice(lo as i64, (hi - lo) as i64),
                )));
            }
        }
        Answer::Value(Rc::new(value.clone()))
    }
}

/// Evaluates an existing range recipe on its selected operands. Local node
/// references in the temporary operation are placeholders for the supplied
/// evaluator arguments.
fn eval_projection(
    projection: BitProjection,
    width: usize,
    inputs: &[&IrValue],
) -> Option<IrValue> {
    let mut arg_index = 0;
    let mut padding = None;
    let Ok(payload) = projection.into_payload(
        width,
        |_| {
            let index = arg_index;
            arg_index += 1;
            NodeRef { index }
        },
        |zeros| {
            padding = Some(IrValue::Bits(IrBits::zero(zeros)));
            Ok::<_, std::convert::Infallible>(NodeRef {
                index: inputs.len(),
            })
        },
    );
    let mut padded;
    let args = if let Some(value) = padding.as_ref() {
        padded = inputs.to_vec();
        padded.push(value);
        &padded[..]
    } else {
        inputs
    };
    let op = Node {
        text_id: 0,
        name: None,
        pos: None,
        ty: Type::Bits(width),
        payload,
    };
    eval_pure_if_supported(&op, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;
    use std::fmt::Write;

    #[test]
    fn routed_controls_preserve_high_bits_and_only_read_selected_inputs() {
        let source = r#"package static_controls

fn extract(x: bits[132] id=1) -> bits[2] {
  ret hi: bits[2] = bit_slice(x, start=128, width=2, id=2)
}
block source(result: bits[132]) {
  wide: bits[132] = literal(value=0xa00000000000000000000000000000003, id=3)
  result: () = output_port(wide, name=result, id=4)
}
block wrapped(input_word: bits[132], result: bits[2]) {
  input_word: bits[132] = input_port(name=input_word, id=5)
  hi: bits[2] = invoke(input_word, to_apply=extract, id=6)
  result: () = output_port(hi, name=result, id=7)
}
top block controls(runtime: bits[2], direct: bits[2], nested: bits[2], low: bits[2], chosen: bits[2]) {
  instantiation use_word(block=wrapped, kind=block)
  instantiation value(block=source, kind=block)
  received: bits[132] = instantiation_output(instantiation=value, port_name=result, id=8)
  alias: bits[132] = identity(received, id=9)
  drive: () = instantiation_input(alias, instantiation=use_word, port_name=input_word, id=10)
  field: bits[2] = instantiation_output(instantiation=use_word, port_name=result, id=11)
  direct_field: bits[2] = bit_slice(received, start=128, width=2, id=12)
  runtime: bits[2] = input_port(name=runtime, id=13)
  both: bits[4] = concat(runtime, field, id=14)
  first: bits[2] = bit_slice(both, start=0, width=2, id=15)
  one: bits[1] = literal(value=1, id=16)
  selected: bits[2] = sel(one, cases=[runtime, field], id=17)
  direct: () = output_port(direct_field, name=direct, id=18)
  nested: () = output_port(field, name=nested, id=19)
  low: () = output_port(first, name=low, id=20)
  chosen: () = output_port(selected, name=chosen, id=21)
}
"#;
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        let block = package.get_top_block().unwrap();
        let mut lookup = ControlValues::new(&package);
        lookup.register_block(block);
        for (index, node) in block.nodes.iter().enumerate() {
            if matches!(
                node.payload,
                NodePayload::OutputPort { .. } | NodePayload::InstantiationInput { .. }
            ) {
                assert_eq!(
                    lookup.root_value(block, NodeRef { index }).as_deref(),
                    Some(&IrValue::make_tuple(&[]))
                );
            }
        }
        for output in block.output_ports() {
            let NodePayload::OutputPort { arg, name, .. } = &block.get_node(output).payload else {
                unreachable!()
            };
            assert_eq!(
                lookup.root_value(block, *arg).as_deref(),
                Some(&IrValue::make_ubits(2, 2).unwrap()),
                "{name}"
            );
        }
    }

    #[test]
    fn repeated_structural_calls_share_specialized_results() {
        let mut source = String::from(
            "package shared_controls\nfn f0(x: bits[4] id=1) -> bits[4] {\n ret r: bits[4] = reverse(x, id=2)\n}\n",
        );
        let mut id = 3;
        let depth = 24;
        for level in 1..=depth {
            let sub = level - 1;
            writeln!(source,"fn f{level}(x: bits[4] id={id}) -> bits[4] {{\n a: bits[4] = invoke(x, to_apply=f{sub}, id={})\n ret b: bits[4] = invoke(a, to_apply=f{sub}, id={})\n}}",id+1,id+2).unwrap();
            id += 3;
        }
        writeln!(source,"top block test(out: bits[4]) {{\n x: bits[4] = literal(value=3, id={id})\n result: bits[4] = invoke(x, to_apply=f{depth}, id={})\n out: () = output_port(result, name=out, id={})\n}}",id+1,id+2).unwrap();
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let block = package.get_top_block().unwrap();
        let mut lookup = ControlValues::new(&package);
        let out = block.get_output_port("out").unwrap();
        let NodePayload::OutputPort { arg, .. } = block.get_node(out).payload else {
            unreachable!()
        };
        assert_eq!(
            lookup.root_value(block, arg).as_deref(),
            Some(&IrValue::make_ubits(4, 3).unwrap())
        );
        // The expanded tree has over sixteen million leaves.
        assert!(lookup.scopes.len() <= 3 * (depth + 2));
    }
}
