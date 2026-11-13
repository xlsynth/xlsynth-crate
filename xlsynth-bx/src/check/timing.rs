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
use std::collections::{BTreeMap, BTreeSet};

use super::StructTable;
use crate::ast::{BinaryOp, OutputCycles, Param, Type};
use crate::typed::{
    CheckedBlock, CheckedStmt, TypedExpr, TypedExprKind as K, is_constexpr_only_name,
};

const BLOCK_SCOPE: usize = 0;

/// Check annotated blocks in child-first order so parents can use each
/// child's verified output declaration.
pub(super) fn check_blocks(
    blocks: &mut [CheckedBlock],
    structs: &StructTable<'_>,
    invariant_harnesses: &BTreeSet<&str>,
) -> Result<(), String> {
    // In BX, `#[invariant] ... -> bool @0` promises stage 0 for the
    // returned Boolean. The harness exposes it as tuple field 0, followed
    // by the lemmas. Passing just "out" makes check_outputs visit field 0
    // even if a lemma reads a stage-1 register.
    let predicate_port = [Param {
        name: "out".into(),
        ty: Type::Bool,
    }];
    let mut children: BTreeMap<&str, &CheckedBlock> = BTreeMap::new();
    for block in blocks {
        let Some(expected) = &block.output_cycles else {
            // The caller needs an output annotation even if all inputs have
            // stages.
            continue;
        };
        let invariant = invariant_harnesses.contains(block.name.as_str());
        let ports = if invariant {
            &predicate_port[..]
        } else {
            &block.output_ports
        };
        if let Err(error) = check_outputs(block, expected, ports, structs, &children) {
            let deferred = block.statements.iter().any(|stmt| match stmt {
                CheckedStmt::Wire { .. } => true,
                CheckedStmt::Inst { block, .. } => children
                    .get(block.as_str())
                    .is_some_and(|child| child.timing_error.is_some()),
                _ => false,
            });
            if deferred {
                // Lowering can point to the wire that closes a combinational
                // loop. Report that first, or this timing error if IR is valid.
                block.timing_error = Some(error);
            } else if invariant {
                return Err(format!("invariant '{}': {error}", block.source_name));
            } else if block.name != block.source_name {
                return Err(format!(
                    "block '{}' ({}): {error}",
                    block.source_name, block.name
                ));
            } else {
                return Err(format!("block '{}': {error}", block.name));
            }
        }
        // Keep deferred errors visible so callers can defer in turn. Only
        // successful children supply stage information to an instance.
        children.insert(block.name.as_str(), block);
    }
    Ok(())
}

/// Check the declared stages of the source output ports in projection order.
fn check_outputs(
    block: &CheckedBlock,
    expected: &OutputCycles,
    ports: &[Param],
    structs: &StructTable<'_>,
    children: &BTreeMap<&str, &CheckedBlock>,
) -> Result<(), String> {
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

    let checker = Checker::new(block, structs, children);
    let check = |fields: &[usize], expected| checker.infer(&block.output, fields)?.check(expected);
    let aggregate = matches!(block.result, Type::Named(_) | Type::Tuple(_));
    if !aggregate && let OutputCycles::Uniform(cycle) = expected {
        return check(&[], *cycle);
    }
    for (index, port) in ports.iter().enumerate() {
        // Suppose `saved` is field 0. In `@(live: 0, saved: 1)`, its stage
        // is 1. Use its name for the annotation and its index for the value.
        let cycle = expected.stage_for_port(&port.name);
        let field = aggregate.then_some(index);
        check(field.as_slice(), cycle).map_err(|err| format!("output '{}': {err}", port.name))?;
    }
    Ok(())
}

impl OutputCycles {
    /// Look up a local stage after validating the declared port names.
    fn stage_for_port(&self, port: &str) -> u64 {
        match self {
            Self::Uniform(stage) => *stage,
            Self::ByPort(stages) => stages[port],
        }
    }
}

/// What the checker knows about the stage of an expression.
#[derive(Clone, Copy)]
enum Cycle {
    /// A known stage in the block whose expression is being checked.
    At(u64),
    /// A literal such as `u8:1` can join `q + u8:1` at whatever stage `q` has.
    Independent,
    /// The checker has no stage to compare against the output annotation.
    Unknown,
}

impl Cycle {
    fn check(self, expected: u64) -> Result<(), String> {
        match self {
            Self::Unknown => Err("cannot infer output cycle for this expression".into()),
            Self::At(actual) if actual != expected => {
                Err(format!("expected cycle {expected}, got cycle {actual}"))
            }
            Self::At(_) | Self::Independent => Ok(()),
        }
    }

    /// A register adds one clock edge. A constant alone supplies no stage to
    /// count from.
    fn registered(self, name: &str) -> Result<Self, String> {
        let Self::At(stage) = self else {
            return Ok(Self::Unknown);
        };
        stage
            .checked_add(1)
            .map(Self::At)
            .ok_or_else(|| format!("cycle offset overflow at '{name}'"))
    }
}

/// A name either supplies an input stage or has an RHS to follow.
enum Binding<'a> {
    /// A data port supplies its declared stage, defaulting to zero.
    Input(u64),
    /// A let or a deferred wire takes its driver's stage directly.
    Let(Driver<'a>),
    /// A register read is one stage past the data and enable at its write.
    Register(Driver<'a>),
}

/// The RHS and its scope are fixed after indexing. The DFS memoizes each
/// selected field separately because `p.saved` and `p.live` can differ.
struct Driver<'a> {
    rhs: &'a TypedExpr,
    scope: usize,
    /// A missing key starts a read. `None` marks an active read for feedback
    /// detection. `Some(Unknown)` is a completed read with no provable stage.
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

/// A parent expression connected to a child port with a fixed local stage.
struct InstanceInput<'a> {
    port: &'a str,
    local_stage: u64,
    parent_expr: &'a TypedExpr,
}

/// All outputs of one instance share the same translation into parent stages.
struct Instance<'a> {
    name: &'a str,
    inputs: Vec<InstanceInput<'a>>,
    outputs: &'a OutputCycles,
    offset: Cell<InstanceOffset>,
}

/// DFS state for one instance's shared offset (`parent_stage - child_stage`).
#[derive(Clone, Copy)]
enum InstanceOffset {
    /// The first output read will schedule the connected input expressions.
    Unseen,
    /// Those inputs are being evaluated. Reading this instance again on
    /// that path is feedback, for example through a deferred wire.
    Visiting,
    /// Every output can reuse the computed offset. `None` means an input
    /// stage was unknown or no staged input established an offset.
    Done(Option<i128>),
}

impl Instance<'_> {
    /// Each staged input must agree on parent stage minus child stage.
    fn infer_offset(
        &self,
        parent_stages: impl Iterator<Item = Cycle>,
    ) -> Result<Option<i128>, String> {
        let mut offset = None;
        let mut has_unknown_input = false;
        for (input, stage) in self.inputs.iter().zip(parent_stages) {
            match stage {
                Cycle::At(parent) => {
                    // Child x @0 connected to parent x @4 sets offset +4.
                    // A second child port at @1 must then connect at parent @5.
                    let local = i128::from(input.local_stage);
                    let candidate = i128::from(parent) - local;
                    let agreed = *offset.get_or_insert(candidate);
                    if agreed != candidate {
                        return Err(format!(
                            "instance '{}' input '{}': expected parent cycle {}, got {parent}",
                            self.name,
                            input.port,
                            local + agreed
                        ));
                    }
                }
                Cycle::Independent => { /* A constant fits any stage and supplies no offset. */ }
                // Keep scanning so a mismatch between other known inputs
                // still gets reported, regardless of port order.
                Cycle::Unknown => has_unknown_input = true,
            }
        }
        Ok(if has_unknown_input { None } else { offset })
    }

    /// Translate a verified child output into the stage seen by its caller.
    fn output_cycle(&self, port: &str, offset: Option<i128>) -> Result<Cycle, String> {
        let Some(offset) = offset else {
            return Ok(Cycle::Unknown);
        };
        // Connecting child @3 to parent @0 gives offset -3. Only the
        // translated port stage has to fit the unsigned BX stage range.
        let translated = i128::from(self.outputs.stage_for_port(port)) + offset;
        let stage = u64::try_from(translated).map_err(|_| {
            format!(
                "instance '{}.{port}': translated cycle {translated} is out of range",
                self.name
            )
        })?;
        Ok(Cycle::At(stage))
    }
}

/// Build the scopes once, then follow only the drivers reached from the
/// outputs.
struct Checker<'a> {
    scopes: Vec<Scope<'a>>,
    structs: &'a StructTable<'a>,
    instances: BTreeMap<&'a str, Instance<'a>>,
}

impl<'a> Checker<'a> {
    fn new(
        block: &'a CheckedBlock,
        structs: &'a StructTable<'a>,
        children: &BTreeMap<&str, &'a CheckedBlock>,
    ) -> Self {
        let mut root = Scope::default();
        for param in &block.params {
            let stage = block.input_cycles.get(&param.name).copied().unwrap_or(0);
            root.names.insert(&param.name, Binding::Input(stage));
        }
        // Body checking emits an Inst for each child call and a Connect for
        // each argument. Index those connections by instance and port.
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
                && child.timing_error.is_none()
                && let Some(outputs) = &child.output_cycles
            {
                let inputs = child
                    .params
                    .iter()
                    .map(|p| InstanceInput {
                        port: &p.name,
                        local_stage: child.input_cycles.get(&p.name).copied().unwrap_or(0),
                        // Body checking requires exactly one driver per data
                        // port.
                        parent_expr: connections[&(name.as_str(), p.name.as_str())],
                    })
                    .collect();
                instances.insert(
                    name.as_str(),
                    Instance {
                        name,
                        inputs,
                        outputs,
                        offset: Cell::new(InstanceOffset::Unseen),
                    },
                );
            }
        }
        let mut checker = Self {
            scopes: vec![root],
            structs,
            instances,
        };
        checker.collect_drivers(&block.statements, BLOCK_SCOPE);
        checker
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
                CheckedStmt::Let { name, .. } if is_constexpr_only_name(name) => continue,
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
        while let Some(step) = walk.pending.pop() {
            match step {
                Step::Expr(expr, scope, fields) => {
                    self.visit_expr(&mut walk, expr, scope, fields)?
                }
                Step::Read(name, binding, fields) => {
                    self.visit_read(&mut walk, name, binding, fields)?
                }
                Step::FinishInstanceOutput(instance, port) => {
                    // In `a + lane.out`, keep a's stage below the input stages
                    // so the pending adder can consume it after this read.
                    let start = walk.values.len() - instance.inputs.len();
                    let offset = instance.infer_offset(walk.values.drain(start..))?;
                    instance.offset.set(InstanceOffset::Done(offset));
                    walk.values.push(instance.output_cycle(port, offset)?);
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
                    *left = rule.apply(*left, right)?;
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
                walk.values.push(Cycle::At(*stage));
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
                Step::Join(Join::Register(name)),
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
    ) -> Result<(), String> {
        let operand = |value| Step::Expr(value, scope, vec![]);
        let mut read = |name: &'s str, fields| match self.binding(scope, name) {
            Some(binding) => walk.pending.push(Step::Read(name, binding, fields)),
            None => walk.values.push(Cycle::Unknown),
        };
        match &expr.kind {
            K::Name(name) => read(name, fields),
            K::InstanceOutput { instance, port } => match self.instances.get(instance.as_str()) {
                Some(child) => walk.read_instance_output(child, port)?,
                // Externs and children without a verified output annotation
                // have no timing interface to use here.
                None => walk.values.push(Cycle::Unknown),
            },
            K::BindingTupleIndex { name, index } => {
                // `let (head, _) = pair;` reads the same field as `pair.0`.
                fields.push(*index);
                read(name, fields);
            }
            K::Literal(_) => walk.values.push(Cycle::Independent),
            K::Unary { value, .. } | K::Cast { value } | K::Sequence { result: value, .. } => {
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
        Ok(())
    }
}

/// Pending DFS work and stages produced so far for one output.
struct Walk<'a> {
    pending: Vec<Step<'a>>,
    values: Vec<Cycle>,
}

impl<'a> Walk<'a> {
    /// Read a child output using the instance offset, evaluating its inputs
    /// on the first read. A later read of another port uses the same offset.
    fn read_instance_output(
        &mut self,
        instance: &'a Instance<'a>,
        port: &'a str,
    ) -> Result<(), String> {
        match instance.offset.get() {
            InstanceOffset::Unseen => {
                instance.offset.set(InstanceOffset::Visiting);
                self.pending
                    .push(Step::FinishInstanceOutput(instance, port));
                // The stack pops inputs in declaration order, which is the
                // order infer_offset uses to pair each stage with its port.
                for input in instance.inputs.iter().rev() {
                    self.pending
                        .push(Step::Expr(input.parent_expr, BLOCK_SCOPE, vec![]));
                }
            }
            InstanceOffset::Visiting => {
                return Err(format!(
                    "timing feedback through instance '{}' is unsupported",
                    instance.name
                ));
            }
            InstanceOffset::Done(offset) => self.values.push(instance.output_cycle(port, offset)?),
        }
        Ok(())
    }

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

/// One piece of pending DFS work. Operand steps leave stages on the value
/// stack for the step that consumes them.
enum Step<'a> {
    /// Evaluate an expression in its scope. The last field index is the
    /// next projection to follow (e.g. `header` in `p.header.id`).
    Expr(&'a TypedExpr, usize, Vec<usize>),
    /// After the connected inputs are evaluated, cache the instance offset
    /// and push this port's translated stage. For offset +4 and `saved @1`,
    /// the cache holds +4 and the value stack receives `At(5)`.
    FinishInstanceOutput(&'a Instance<'a>, &'a str),
    /// Follow the driver of a bound name, selecting only the requested fields.
    Read(&'a str, &'a Binding<'a>, Vec<usize>),
    /// Evaluate the enclosing write guards. The outermost scope is
    /// unconditional.
    Guard(usize),
    /// Consume two operand stages and push the result of their timing rule.
    Join(Join<'a>),
    /// Cache the completed stage for this driver's field path (empty for the
    /// whole value).
    SaveRead(&'a Driver<'a>, Vec<usize>),
    /// Cache the combined guard stage for other writes in this scope.
    SaveGuard(usize),
}

/// The hardware operation where two operand stages meet.
enum Join<'a> {
    /// The two inputs of a combinational operator such as `q + r`.
    Operator(BinaryOp),
    /// Members of a struct or tuple read as one value.
    Fields,
    /// The two data arms of an `if` expression.
    Branches,
    /// The condition and the data stage established by `Branches`.
    Select,
    /// An enclosing write guard and the next nested `if` condition.
    Guards,
    /// A register's data and load enable meet before the edge advances the
    /// stage.
    Register(&'a str),
}

impl Join<'_> {
    /// Check alignment first. Only a register advances the result to the next
    /// stage.
    fn apply(self, left: Cycle, right: Cycle) -> Result<Cycle, String> {
        let stage = match (left, right) {
            (Cycle::Unknown, _) | (_, Cycle::Unknown) => Cycle::Unknown,
            (Cycle::Independent, stage) | (stage, Cycle::Independent) => stage,
            (Cycle::At(a), Cycle::At(b)) if a != b => return Err(self.mismatch(a, b)),
            (Cycle::At(_), Cycle::At(_)) => left,
        };
        match self {
            Self::Register(name) => stage.registered(name),
            _ => Ok(stage),
        }
    }

    fn mismatch(&self, a: u64, b: u64) -> String {
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
