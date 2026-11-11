// SPDX-License-Identifier: Apache-2.0

//! Opt-in stage inference for BX block outputs.
//!
//! Each driver records where a name gets its value. Once the drivers are
//! indexed, the checker walks backwards from each annotated output.
//! For `q + r`, it follows `q` to its register write, looks up input `r`,
//! and checks that both reach the adder at the same stage. The pending steps
//! run in that order. Each read leaves its stage on the value stack, and the
//! adder replaces its two operand stages with one result.

use std::cell::{Cell, RefCell};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

mod summary;
use summary::{Builder, FixedInputs, Rule, Summary};

use super::{StructTable, output_ports};
use crate::ast::{BinaryOp, OutputCycles, Type};
use crate::typed::{CheckedBlock, CheckedStmt, TypedExpr, TypedExprKind as K};

const BLOCK_SCOPE: usize = 0;

/// The elaborator orders concrete blocks child-first, including
/// specializations.
pub(super) fn check_blocks(
    blocks: &mut [CheckedBlock],
    structs: &StructTable<'_>,
) -> Result<(), String> {
    // Unannotated modules with no timed caller stay outside the analysis.
    let mut needed = BTreeSet::new();
    for block in blocks.iter().rev() {
        if block.output_cycles.is_some() || needed.contains(&block.name) {
            for stmt in &block.statements {
                if let CheckedStmt::Inst { block, .. } = stmt {
                    needed.insert(block.clone());
                }
            }
        }
    }
    let mut children: BTreeMap<&str, Child<'_>> = BTreeMap::new();
    for block in blocks {
        if block.output_cycles.is_none() && !needed.contains(&block.name) {
            continue;
        }
        let inferred = if let Some(expected) = &block.output_cycles {
            check_outputs(block, expected, structs, &children).map(|()| BTreeMap::new())
        } else {
            infer_outputs(block, structs, &children)
        };
        // Lowering gives a more precise diagnostic for feedback through a
        // deferred wire, even if an unannotated wrapper sits between the
        // blocks.
        let defer_to_lowering = block.statements.iter().any(|stmt| match stmt {
            CheckedStmt::Wire { .. } => true,
            CheckedStmt::Inst { block, .. } => children
                .get(block.as_str())
                .is_some_and(|child| child.defer_to_lowering),
            _ => false,
        });
        let inferred = match inferred {
            Ok(ports) => ports,
            Err(error) => {
                if defer_to_lowering {
                    // Let block IR diagnose feedback in this block or its child
                    // first.
                    block.timing_error = Some(error);
                    BTreeMap::new()
                } else if block.name != block.source_name {
                    return Err(format!(
                        "block '{}' ({}): {error}",
                        block.source_name, block.name
                    ));
                } else {
                    return Err(format!("block '{}': {error}", block.name));
                }
            }
        };
        let fixed_inputs = if block.output_cycles.is_some() {
            fixed_inputs(block)
        } else {
            Rc::from([])
        };
        let defer_to_lowering =
            defer_to_lowering && (block.output_cycles.is_none() || block.timing_error.is_some());
        children.insert(
            block.name.as_str(),
            Child {
                block,
                inferred,
                fixed_inputs,
                defer_to_lowering,
            },
        );
    }
    Ok(())
}

/// Concrete child definitions retain either their declaration or an inferred
/// relation per physical output. Unsupported paths only affect callers that
/// read them.
struct Child<'a> {
    block: &'a CheckedBlock,
    inferred: PortSummaries,
    fixed_inputs: FixedInputs,
    defer_to_lowering: bool,
}

type PortSummaries = BTreeMap<String, Result<Rc<Summary>, String>>;

/// Compute a child interface with input stages left as variables.
fn infer_outputs(
    block: &CheckedBlock,
    structs: &StructTable<'_>,
    children: &BTreeMap<&str, Child<'_>>,
) -> Result<PortSummaries, String> {
    let checker = Checker::new(block, structs, children)?;
    let aggregate = matches!(block.result, Type::Named(_) | Type::Tuple(_));
    let mut results = BTreeMap::new();
    let mut shared = BTreeMap::new();
    for (index, port) in output_ports(&block.result, &structs.definitions)?
        .into_iter()
        .enumerate()
    {
        let field = aggregate.then_some(index);
        let summary = checker
            .infer(&block.output, field.as_slice())
            .and_then(|stage| {
                let mut builder = checker.builder.borrow_mut();
                let result = match checker.input_offset {
                    Some(offset) => builder.apply(Rule::RequireOffset, &[offset, stage])?,
                    None => stage,
                };
                let summary = match shared.entry(result) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(Rc::new(builder.summary(result)?)),
                };
                Ok(Rc::clone(summary))
            });
        results.insert(port.name, summary);
    }
    Ok(results)
}

/// Check a declared stage against each physical output of this block.
fn check_outputs(
    block: &CheckedBlock,
    expected: &OutputCycles,
    structs: &StructTable<'_>,
    children: &BTreeMap<&str, Child<'_>>,
) -> Result<(), String> {
    let checker = Checker::new(block, structs, children)?;
    let check = |fields: &[usize], expected| checker.infer(&block.output, fields)?.check(expected);
    let aggregate = matches!(block.result, Type::Named(_) | Type::Tuple(_));
    if !aggregate && let OutputCycles::Uniform(cycle) = expected {
        return check(&[], *cycle);
    }
    let ports = output_ports(&block.result, &structs.definitions)?;
    if let OutputCycles::ByPort(expected) = expected {
        let port_names: BTreeSet<_> = ports.iter().map(|p| p.name.as_str()).collect();
        if let Some(port) = expected
            .keys()
            .find(|port| !port_names.contains(port.as_str()))
        {
            return Err(format!("unknown output port '{port}' in cycle annotation"));
        }
        if let Some(port) = port_names
            .into_iter()
            .find(|port| !expected.contains_key(*port))
        {
            return Err(format!("missing cycle annotation for output port '{port}'"));
        }
    }

    for (index, port) in ports.iter().enumerate() {
        // Suppose `saved` is field 0. In `@(live: 0, saved: 1)`, its stage
        // is 1. Use its name for the annotation and its index for the value.
        let cycle = match expected {
            OutputCycles::Uniform(cycle) => *cycle,
            OutputCycles::ByPort(cycles) => cycles[&port.name],
        };
        let field = aggregate.then_some(index);
        check(field.as_slice(), cycle).map_err(|err| format!("output '{}': {err}", port.name))?;
    }
    Ok(())
}

/// What the checker knows about the stage of an expression.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Cycle {
    // A translated instance origin can be negative. Observable stages fit u64.
    At(i128),
    Input(usize),
    Expr(usize),
    /// A literal such as `u8:1` can join `q + u8:1` at whatever stage `q` has.
    Independent,
    /// The checker has no stage to compare against the output annotation.
    Unknown,
}

impl Cycle {
    fn check(self, expected: u64) -> Result<(), String> {
        match self {
            Self::Unknown => Err("cannot infer output cycle for this expression".into()),
            Self::At(actual) if actual != i128::from(expected) => {
                Err(format!("expected cycle {expected}, got cycle {actual}"))
            }
            Self::At(_) | Self::Independent => Ok(()),
            Self::Input(_) | Self::Expr(_) => unreachable!("checked output has concrete inputs"),
        }
    }

    /// A register adds one clock edge. A constant alone supplies no stage to
    /// count from.
    fn registered(self, name: &str) -> Result<Self, String> {
        let Self::At(stage) = self else {
            return Ok(Self::Unknown);
        };
        let stage = stage + 1;
        u64::try_from(stage).map_err(|_| format!("cycle offset overflow at '{name}'"))?;
        Ok(Self::At(stage))
    }
}

/// A name either supplies an input stage or has an RHS to follow.
enum Binding<'a> {
    Input(Cycle),
    Let(Driver<'a>),
    Register(Driver<'a>),
}

/// The RHS and its scope are fixed after indexing. The DFS memoizes each
/// selected field separately because `p.saved` and `p.live` can differ.
struct Driver<'a> {
    rhs: &'a TypedExpr,
    scope: usize,
    reads: RefCell<BTreeMap<Vec<usize>, Option<Cycle>>>,
}

/// An if body owns its local bindings. Its condition is evaluated in the
/// parent.
#[derive(Default)]
struct Scope<'a> {
    parent: Option<(usize, &'a TypedExpr)>,
    guard_cycle: Cell<Option<Cycle>>,
    names: BTreeMap<&'a str, Binding<'a>>,
}

/// A call uses either a declared common offset or a saved relation for the
/// selected output port. Both are memoized per instance.
struct Instance<'a> {
    name: Rc<str>,
    inputs: Vec<&'a TypedExpr>,
    child: &'a Child<'a>,
    offset: Cell<InstanceOffset>,
    reads: RefCell<BTreeMap<String, Option<Cycle>>>,
}

/// A declared child shares a single translated stage origin across its ports.
#[derive(Clone, Copy)]
enum InstanceOffset {
    Unseen,
    Visiting,
    Done(Cycle),
}

/// Data ports in connection order, with unlabelled inputs at local stage zero.
fn fixed_inputs(block: &CheckedBlock) -> FixedInputs {
    block
        .params
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                block.input_cycles.get(&p.name).copied().unwrap_or(0),
            )
        })
        .collect()
}

/// Build the scopes once, then follow only the drivers reached from the
/// outputs.
struct Checker<'a> {
    scopes: Vec<Scope<'a>>,
    structs: &'a StructTable<'a>,
    instances: BTreeMap<&'a str, Instance<'a>>,
    builder: RefCell<Builder>,
    input_offset: Option<Cycle>,
}

impl<'a> Checker<'a> {
    fn new(
        block: &'a CheckedBlock,
        structs: &'a StructTable<'a>,
        children: &'a BTreeMap<&str, Child<'a>>,
    ) -> Result<Self, String> {
        let mut root = Scope::default();
        let mut builder = Builder::default();
        let symbolic = block.output_cycles.is_none();
        let name: Rc<str> = Rc::from(block.name.as_str());
        // If the author specified input stages, keep their relative alignment
        // even when the output stage is inferred. Other data inputs default to
        // 0.
        let input_offset = if symbolic && !block.input_cycles.is_empty() {
            let vars: Vec<_> = (0..block.params.len()).map(Cycle::Input).collect();
            Some(builder.apply(
                Rule::Offset {
                    name: name.clone(),
                    inputs: fixed_inputs(block),
                },
                &vars,
            )?)
        } else {
            None
        };
        for (i, param) in block.params.iter().enumerate() {
            let local = block.input_cycles.get(&param.name).copied().unwrap_or(0);
            let stage = if let Some(offset) = input_offset {
                builder.apply(
                    Rule::Translate {
                        name: name.clone(),
                        port: param.name.clone(),
                        local,
                    },
                    &[offset],
                )?
            } else if symbolic {
                Cycle::Input(i)
            } else {
                Cycle::At(i128::from(local))
            };
            root.names.insert(&param.name, Binding::Input(stage));
        }
        // An explicit connection may precede its inst declaration. Both it
        // and a let-call connection have already been checked for a single
        // driver.
        let connections: BTreeMap<_, _> = block
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                CheckedStmt::Connect {
                    instance,
                    port,
                    value,
                    ..
                } => Some(((instance.as_str(), port.as_str()), value)),
                _ => None,
            })
            .collect();
        let mut instances = BTreeMap::new();
        for stmt in &block.statements {
            if let CheckedStmt::Inst { name, block } = stmt
                && let Some(child) = children.get(block.as_str())
                && child.block.timing_error.is_none()
            {
                let inputs = child
                    .block
                    .params
                    .iter()
                    .map(|p| connections[&(name.as_str(), p.name.as_str())])
                    .collect();
                instances.insert(
                    name.as_str(),
                    Instance {
                        name: Rc::from(name.as_str()),
                        inputs,
                        child,
                        offset: Cell::new(InstanceOffset::Unseen),
                        reads: RefCell::default(),
                    },
                );
            }
        }
        let mut checker = Self {
            scopes: vec![root],
            structs,
            instances,
            input_offset,
            builder: RefCell::new(builder),
        };
        checker.collect_drivers(&block.statements, BLOCK_SCOPE);
        Ok(checker)
    }

    /// Index writes first so an output can follow a register read to a later
    /// write. In `if en { let d = x; q := d; }`, `q` is visible throughout
    /// the block. Its RHS uses the inner scope to find `d`.
    fn collect_drivers(&mut self, statements: &'a [CheckedStmt], scope: usize) {
        let driver = |rhs| Driver {
            rhs,
            scope,
            reads: RefCell::default(),
        };
        for stmt in statements {
            let (owner, name, binding) = match stmt {
                CheckedStmt::Let { name, value } | CheckedStmt::DriveWire { name, value, .. } => {
                    (scope, name, Binding::Let(driver(value)))
                }
                CheckedStmt::Write { name, value } => {
                    (BLOCK_SCOPE, name, Binding::Register(driver(value)))
                }
                CheckedStmt::If { condition, body } => {
                    let child = self.scopes.len();
                    self.scopes.push(Scope {
                        parent: Some((scope, condition)),
                        ..Scope::default()
                    });
                    self.collect_drivers(body, child);
                    continue;
                }
                // Instance connections are indexed separately from lexical
                // names.
                _ => continue,
            };
            self.scopes[owner].names.insert(name, binding);
        }
    }

    /// A read inside an if body can use a name from any enclosing scope.
    fn binding(&self, mut scope: usize, name: &str) -> Option<&Binding<'a>> {
        loop {
            if let Some(binding) = self.scopes[scope].names.get(name) {
                return Some(binding);
            }
            scope = self.scopes[scope].parent?.0;
        }
    }

    /// If the selected register value is itself a tuple or struct, read its
    /// fields separately before merging their stages. Arrays stay a single
    /// value.
    fn register_field_count(
        &self,
        mut ty: &'a Type,
        path: &[usize],
    ) -> Result<Option<usize>, String> {
        for &index in path.iter().rev() {
            ty = match ty {
                Type::Tuple(items) => &items[index],
                Type::Named(name) => &self.structs.definition(name)?.fields[index].ty,
                _ => unreachable!("field projection has a checked aggregate type"),
            };
        }
        Ok(match ty {
            Type::Tuple(items) => Some(items.len()),
            Type::Named(name) => Some(self.structs.definition(name)?.fields.len()),
            _ => None,
        })
    }

    /// Walk backwards from one output. Each step either schedules its inputs
    /// or uses stages those inputs have already left on the value stack.
    fn infer(&self, output: &'a TypedExpr, fields: &[usize]) -> Result<Cycle, String> {
        let mut walk = Walk {
            pending: vec![Step::Expr(output, BLOCK_SCOPE, fields.to_vec())],
            values: Vec::new(),
        };
        let result = self.run(&mut walk);
        if result.is_err() {
            // A failed port must release its active reads before the next
            // port starts. Completed reads are still valid.
            for step in walk.pending {
                match step {
                    Step::SaveRead(driver, fields) => {
                        driver.reads.borrow_mut().remove(&fields);
                    }
                    Step::SaveOffset(instance) => instance.offset.set(InstanceOffset::Unseen),
                    Step::ApplySummary(instance, port, _) => {
                        instance.reads.borrow_mut().remove(port);
                    }
                    _ => { /* Only the three cache writes own visiting markers. */ }
                }
            }
        }
        result
    }

    /// Evaluate pending drivers and joins until the requested output is known.
    fn run<'s>(&'s self, walk: &mut Walk<'s>) -> Result<Cycle, String> {
        while let Some(step) = walk.pending.pop() {
            match step {
                Step::Expr(expr, scope, fields) => self.visit_expr(walk, expr, scope, fields),
                Step::Read(name, binding, fields) => {
                    self.visit_read(walk, name, binding, fields)?
                }
                Step::Output(instance, port) => self.visit_output(walk, instance, port)?,
                Step::SaveOffset(instance) => {
                    instance.offset.set(InstanceOffset::Unseen);
                    let start = walk.values.len() - instance.inputs.len();
                    let offset = self.builder.borrow_mut().apply(
                        Rule::Offset {
                            name: instance.name.clone(),
                            inputs: instance.child.fixed_inputs.clone(),
                        },
                        &walk.values[start..],
                    )?;
                    walk.values.truncate(start);
                    instance.offset.set(InstanceOffset::Done(offset));
                }
                Step::ApplySummary(instance, port, summary) => {
                    instance.reads.borrow_mut().remove(port);
                    let start = walk.values.len() - summary.inputs.len();
                    let stage = summary
                        .instantiate(&walk.values[start..], &mut self.builder.borrow_mut())
                        .map_err(|e| format!("instance '{}.{port}': {e}", instance.name))?;
                    walk.values.truncate(start);
                    walk.values.push(stage);
                    instance
                        .reads
                        .borrow_mut()
                        .insert(port.to_owned(), Some(stage));
                }
                Step::Guard(scope) => {
                    let body = &self.scopes[scope];
                    if let Some(stage) = body.guard_cycle.get() {
                        walk.values.push(stage);
                    } else if let Some((parent, condition)) = body.parent {
                        // `if a { if b { ... } }` enables a write with a && b.
                        walk.schedule([
                            Step::Guard(parent),
                            Step::Expr(condition, parent, vec![]),
                            Step::Join(Join::Guards),
                            Step::SaveGuard(scope),
                        ]);
                    } else {
                        walk.values.push(Cycle::Independent);
                    }
                }
                Step::Join(rule) => {
                    let right = walk.values.pop().expect("right operand evaluated");
                    let left = walk.values.last_mut().expect("left operand evaluated");
                    *left = self
                        .builder
                        .borrow_mut()
                        .apply(Rule::Join(rule), &[*left, right])?;
                }
                Step::SaveRead(driver, fields) => {
                    let stage = *walk.values.last().expect("driver evaluated");
                    driver.reads.borrow_mut().insert(fields, Some(stage));
                }
                Step::SaveGuard(scope) => {
                    let stage = *walk.values.last().expect("enable evaluated");
                    self.scopes[scope].guard_cycle.set(Some(stage));
                }
            }
        }
        debug_assert_eq!(walk.values.len(), 1);
        Ok(walk.values.pop().expect("output evaluated"))
    }

    /// Read a declared output relative to the instance origin, or bind the
    /// contributing inputs of an inferred output to its saved timing rule.
    fn visit_output<'s>(
        &'s self,
        walk: &mut Walk<'s>,
        instance: &'s Instance<'a>,
        port: &'s str,
    ) -> Result<(), String> {
        if let Some(outputs) = &instance.child.block.output_cycles {
            match instance.offset.get() {
                InstanceOffset::Unseen => {
                    instance.offset.set(InstanceOffset::Visiting);
                    walk.pending.push(Step::Output(instance, port));
                    walk.pending.push(Step::SaveOffset(instance));
                    for value in instance.inputs.iter().rev() {
                        walk.pending.push(Step::Expr(value, BLOCK_SCOPE, vec![]));
                    }
                }
                InstanceOffset::Visiting => {
                    return Err(format!(
                        "timing feedback through instance '{}' is unsupported",
                        instance.name
                    ));
                }
                InstanceOffset::Done(offset) => {
                    let local = match outputs {
                        OutputCycles::Uniform(stage) => *stage,
                        OutputCycles::ByPort(stages) => stages[port],
                    };
                    let stage = self.builder.borrow_mut().apply(
                        Rule::Translate {
                            name: instance.name.clone(),
                            port: port.to_owned(),
                            local,
                        },
                        &[offset],
                    )?;
                    walk.values.push(stage);
                }
            }
            return Ok(());
        }
        let cached = instance.reads.borrow().get(port).copied();
        if let Some(read) = cached {
            walk.values.push(read.ok_or_else(|| {
                format!(
                    "timing feedback through instance '{}.{port}' is unsupported",
                    instance.name
                )
            })?);
        } else {
            let summary = instance.child.inferred[port]
                .as_ref()
                .map_err(|e| format!("instance '{}.{port}': {e}", instance.name))?;
            instance.reads.borrow_mut().insert(port.to_owned(), None);
            walk.pending
                .push(Step::ApplySummary(instance, port, summary));
            for &index in summary.inputs.iter().rev() {
                walk.pending
                    .push(Step::Expr(instance.inputs[index], BLOCK_SCOPE, vec![]));
            }
        }
        Ok(())
    }

    /// Follow the driver in its defining scope. Cache each field separately so
    /// `q.saved` can depend on `q.live` without reporting feedback.
    fn visit_read<'s>(
        &'s self,
        walk: &mut Walk<'s>,
        name: &'s str,
        binding: &'s Binding<'a>,
        fields: Vec<usize>,
    ) -> Result<(), String> {
        let driver = match binding {
            Binding::Input(stage) => {
                walk.values.push(*stage);
                return Ok(());
            }
            Binding::Let(driver) | Binding::Register(driver) => driver,
        };
        // In `q + q` the second read reuses the finished stage. In
        // `q := q + x` it finds the first read still in progress.
        if let Some(stage) = driver.reads.borrow().get(&fields) {
            walk.values.push(
                stage.ok_or_else(|| format!("timing feedback through '{name}' is unsupported"))?,
            );
            return Ok(());
        }
        driver.reads.borrow_mut().insert(fields.clone(), None);
        walk.pending.push(Step::SaveRead(driver, fields.clone()));
        if let Binding::Let(_) = binding {
            walk.pending
                .push(Step::Expr(driver.rhs, driver.scope, fields));
        } else if let Some(count) = self.register_field_count(&driver.rhs.ty, &fields)? {
            // Reading all of q has the same timing as rebuilding it from its
            // fields.
            walk.aggregate((0..count).map(|index| {
                let mut member = Vec::with_capacity(fields.len() + 1);
                member.push(index);
                member.extend_from_slice(&fields);
                Step::Read(name, binding, member)
            }));
        } else {
            // In `if en { q := x; }`, x and en meet before the clock edge.
            walk.schedule([
                Step::Expr(driver.rhs, driver.scope, fields),
                Step::Guard(driver.scope),
                Step::Join(Join::Register(name.to_owned())),
            ]);
        }
        Ok(())
    }

    /// Schedule operands in BX evaluation order, followed by the timing rule
    /// that consumes them. A projected tuple needs only its selected member.
    fn visit_expr<'s>(
        &'s self,
        walk: &mut Walk<'s>,
        expr: &'s TypedExpr,
        scope: usize,
        mut fields: Vec<usize>,
    ) {
        let operand = |value| Step::Expr(value, scope, vec![]);
        let mut read = |name: &'s str, fields| match self.binding(scope, name) {
            Some(binding) => walk.pending.push(Step::Read(name, binding, fields)),
            None => walk.values.push(Cycle::Unknown),
        };
        match &expr.kind {
            K::Name(name) => read(name, fields),
            K::InstanceOutput { instance, port } => match self.instances.get(instance.as_str()) {
                Some(child) => walk.pending.push(Step::Output(child, port)),
                None => walk.values.push(Cycle::Unknown),
            },
            K::BindingTupleIndex { name, index } => {
                // `let (head, _) = pair;` reads the same field as `pair.0`.
                fields.push(*index);
                read(name, fields);
            }
            K::Literal(_) => walk.values.push(Cycle::Independent),
            K::Unary { value, .. } | K::Cast { value } => {
                walk.pending.push(Step::Expr(value, scope, fields));
            }
            K::TupleIndex { value, index } => {
                // For p.header.id, push id and then header. The tuple for p
                // pops header first.
                fields.push(*index);
                walk.pending.push(Step::Expr(value, scope, fields));
            }
            K::Tuple(members) => {
                if let Some(index) = fields.pop() {
                    walk.pending
                        .push(Step::Expr(&members[index], scope, fields));
                } else {
                    walk.aggregate(members.iter().map(operand));
                }
            }
            K::Binary { op, left, right } => {
                walk.schedule([
                    operand(left),
                    operand(right),
                    Step::Join(Join::Operator(*op)),
                ]);
            }
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                // `if sel { x } else { y }` is a mux. Check its data arms
                // together, then check that their stage agrees with sel.
                walk.schedule([
                    operand(condition),
                    Step::Expr(then_value, scope, fields.clone()),
                    Step::Expr(else_value, scope, fields),
                    Step::Join(Join::Branches),
                    Step::Join(Join::Select),
                ]);
            }
            _ => walk.values.push(Cycle::Unknown),
        }
    }
}

/// Pending DFS work and stages produced so far for one output.
struct Walk<'a> {
    pending: Vec<Step<'a>>,
    values: Vec<Cycle>,
}

impl<'a> Walk<'a> {
    /// Callers list steps in execution order. The stack pops the first one
    /// next.
    fn schedule<const N: usize>(&mut self, steps: [Step<'a>; N]) {
        self.pending.extend(steps.into_iter().rev());
    }

    /// An empty aggregate is stage-independent. Each member joins the running
    /// stage.
    fn aggregate(&mut self, members: impl DoubleEndedIterator<Item = Step<'a>>) {
        self.values.push(Cycle::Independent);
        for member in members.rev() {
            self.schedule([member, Step::Join(Join::Fields)]);
        }
    }
}

/// A pending expression, enable, or operation on already evaluated operands.
enum Step<'a> {
    Expr(&'a TypedExpr, usize, Vec<usize>),
    Output(&'a Instance<'a>, &'a str),
    SaveOffset(&'a Instance<'a>),
    ApplySummary(&'a Instance<'a>, &'a str, &'a Summary),
    Read(&'a str, &'a Binding<'a>, Vec<usize>),
    Guard(usize),
    Join(Join),
    SaveRead(&'a Driver<'a>, Vec<usize>),
    SaveGuard(usize),
}

/// The hardware operation where two operand stages meet.
#[derive(Clone)]
enum Join {
    Operator(BinaryOp),
    Fields,
    Branches,
    Select,
    Guards,
    Register(String),
}

impl Join {
    /// Check alignment first. Only a register advances the result to the next
    /// stage.
    fn apply(&self, left: Cycle, right: Cycle) -> Result<Cycle, String> {
        let stage = match (left, right) {
            (Cycle::Unknown, _) | (_, Cycle::Unknown) => Cycle::Unknown,
            (Cycle::Independent, stage) | (stage, Cycle::Independent) => stage,
            (Cycle::At(a), Cycle::At(b)) if a != b => return Err(self.mismatch(a, b)),
            (Cycle::At(_), Cycle::At(_)) => left,
            _ => unreachable!("symbolic join evaluated before substitution"),
        };
        match self {
            Self::Register(name) => stage.registered(name),
            _ => Ok(stage),
        }
    }

    fn mismatch(&self, a: i128, b: i128) -> String {
        match self {
            Self::Operator(op) => format!("operator '{op}' combines cycle {a} with cycle {b}"),
            Self::Fields => format!("aggregate combines cycle {a} with cycle {b}"),
            Self::Branches => format!("if branches combine cycle {a} with cycle {b}"),
            Self::Select => {
                format!("if expression has condition at cycle {a} and data at cycle {b}")
            }
            Self::Guards => format!("nested write guards combine cycle {a} with cycle {b}"),
            Self::Register(name) => {
                format!("write to '{name}' has guard at cycle {b} and data at cycle {a}")
            }
        }
    }
}
