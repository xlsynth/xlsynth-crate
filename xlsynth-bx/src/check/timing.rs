// SPDX-License-Identifier: Apache-2.0

//! Opt-in stage inference for BX block outputs.
//!
//! Annotated blocks are checked child-first. For fixed-stage annotations, the
//! checker indexes inputs, assignments, and child calls, then follows each
//! output backwards to infer its stage. A case-dependent output runs that same
//! walk once per control case, also tracking values known at each stage.
//! For `q + r`, it follows `q` to its register write, looks up input `r`,
//! and checks that they meet at the same stage.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use xlsynth_pir::IrValue;

use super::StructTable;
use crate::ast::{BinaryOp, OutputCycles, Param, Type};
use crate::typed::{
    AvailableFunctions, CheckedBlock, CheckedStmt, TypedExpr, TypedExprKind as K, TypedMatchArm,
    is_constexpr_only_name,
};

mod cases;

const BLOCK_SCOPE: usize = 0;

/// Check annotated blocks in child-first order so parents can use each
/// child's verified output declaration.
pub(super) fn check_blocks(
    blocks: &mut [CheckedBlock],
    structs: &StructTable<'_>,
    invariant_harnesses: &BTreeSet<&str>,
    functions: &AvailableFunctions,
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
        let result = match expected {
            OutputCycles::CaseCall { function, inputs } => {
                cases::enumerate(block, function, inputs, functions, &structs.enums).and_then(
                    |cases| {
                        for case in cases {
                            let stage = OutputCycles::Uniform(case.stage);
                            let label = case.label.clone();
                            check_outputs(block, &stage, ports, structs, &children, Some(case))
                                .map_err(|error| format!("case ({label}): {error}"))?;
                        }
                        Ok(())
                    },
                )
            }
            _ => check_outputs(block, expected, ports, structs, &children, None),
        };
        if let Err(error) = result {
            report_timing_error(block, error, invariant, &children)?;
        }
        // Keep deferred errors visible so callers can defer in turn. Only
        // successful children supply stage information to an instance.
        children.insert(block.name.as_str(), block);
    }
    Ok(())
}

/// Let lowering diagnose a possible combinational loop first; otherwise report
/// the timing error against the original source block or invariant.
fn report_timing_error(
    block: &mut CheckedBlock,
    error: String,
    invariant: bool,
    children: &BTreeMap<&str, &CheckedBlock>,
) -> Result<(), String> {
    let deferred = block.statements.iter().any(|stmt| match stmt {
        CheckedStmt::Wire { .. } => true,
        CheckedStmt::Inst { block, .. } => children
            .get(block.as_str())
            .is_some_and(|child| child.timing_error.is_some()),
        _ => false,
    });
    if deferred {
        block.timing_error = Some(error);
        return Ok(());
    }
    Err(if invariant {
        format!("invariant '{}': {error}", block.source_name)
    } else if block.name != block.source_name {
        format!("block '{}' ({}): {error}", block.source_name, block.name)
    } else {
        format!("block '{}': {error}", block.name)
    })
}

/// Check the declared stages of the source output ports in projection order.
fn check_outputs(
    block: &CheckedBlock,
    expected: &OutputCycles,
    ports: &[Param],
    structs: &StructTable<'_>,
    children: &BTreeMap<&str, &CheckedBlock>,
    case: Option<cases::Case<'_>>,
) -> Result<(), String> {
    if let OutputCycles::ByPort(expected) = expected {
        // If Ports has `saved` and `live`, `@(saved: 1)` is incomplete
        // and `@(saved: 1, typo: 0)` names a nonexistent port. Validate
        // the contract before trying to infer either field's stage.
        let mut remaining: BTreeSet<_> = ports.iter().map(|p| p.name.as_str()).collect();
        for port in expected.keys() {
            if !remaining.remove(port.as_str()) {
                return Err(format!("unknown output port '{port}' in cycle annotation"));
            }
        }
        if let Some(port) = remaining.first() {
            return Err(format!("missing cycle annotation for output port '{port}'"));
        }
    }

    let checker = StageChecker::new(block, structs, children, case);
    // `[]` checks the whole output; `[i]` checks just field i.
    let check = |fields: &[usize], expected| checker.infer(&block.output, fields)?.check(expected);
    let has_field_ports = matches!(block.result, Type::Named(_) | Type::Tuple(_));
    // `-> u8 @1`: infer the whole output and check that its stage is 1.
    // `-> Ports @1` is also Uniform(1), so exclude structs and tuples here:
    // their individual output ports must be checked in the loop below.
    if !has_field_ports && let OutputCycles::Uniform(cycle) = expected {
        return check(&[], *cycle);
    }
    // `-> Ports @1`: check every output port at stage 1.
    // `-> Ports @(saved: 1, live: 0)`: check each port at its declared stage.
    for (index, port) in ports.iter().enumerate() {
        // Match the annotation by port name, then infer the corresponding
        // output field by index. Annotation and field order may differ.
        let cycle = expected
            .stage_for_port(&port.name)
            .expect("fixed output stage");
        let field = has_field_ports.then_some(index);
        check(field.as_slice(), cycle).map_err(|err| format!("output '{}': {err}", port.name))?;
    }
    Ok(())
}

impl OutputCycles {
    /// Look up a local stage after validating the declared port names.
    fn stage_for_port(&self, port: &str) -> Option<u64> {
        match self {
            Self::Uniform(stage) => Some(*stage),
            Self::ByPort(stages) => stages.get(port).copied(),
            Self::CaseCall { .. } => None,
        }
    }
}

/// What the checker knows about the stage of an expression.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// A known stage in the block whose expression is being checked.
    At(u64),
    /// A literal such as `u8:1` can join `q + u8:1` at whatever stage `q` has.
    Independent,
    /// The checker has no stage to compare against the output annotation.
    Unknown,
}

/// A structural stage and, during case checking, a value known at that stage.
/// A held register keeps its stage but loses its known value on the hold path.
#[derive(Clone)]
struct TimedValue {
    stage: Stage,
    known: Option<IrValue>,
}

impl TimedValue {
    fn new(stage: Stage) -> Self {
        Self { stage, known: None }
    }

    fn boolean(&self) -> Option<bool> {
        self.known.as_ref()?.to_bool().ok()
    }
}

impl Stage {
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

/// Progress of a memoized inference. Re-entering a visiting value is feedback;
/// a completed inference may still contain an unknown stage or offset.
#[derive(Clone, Copy)]
enum Inference<T> {
    /// No dependencies have been visited.
    Unseen,
    /// Dependencies are being evaluated; another visit is feedback.
    Visiting,
    /// Evaluation finished, even if the resulting fact is unknown.
    Done(T),
}

/// Where a named value comes from: an input port, a let/wire assignment, or a
/// register assignment.
enum ValueSource<'a> {
    /// A data port supplies its declared stage, defaulting to zero.
    Input(u64),
    /// A let or a deferred wire takes its assignment's stage directly.
    Let(Assignment<'a>),
    /// A register read is one stage past the data and enable at its write.
    Register(Assignment<'a>),
}

/// The RHS of an assignment such as `let d = x + y` or `q := x + y`, along
/// with its scope and the inferred stages of any fields read from it.
struct Assignment<'a> {
    rhs: &'a TypedExpr,
    scope: usize,
    /// Each field is inferred separately. A missing key is unseen.
    reads: RefCell<BTreeMap<Vec<usize>, Inference<TimedValue>>>,
}

/// The condition that creates an if-body scope, evaluated in its parent.
struct EnclosingIf<'a> {
    scope: usize,
    condition: &'a TypedExpr,
}

/// An if body owns its local names. Its condition is evaluated in the
/// parent.
#[derive(Default)]
struct Scope<'a> {
    enclosing_if: Option<EnclosingIf<'a>>,
    guard_stage: RefCell<Option<TimedValue>>,
    names: BTreeMap<&'a str, ValueSource<'a>>,
}

/// A checked child and its parent arguments, in the child's parameter order.
/// All outputs of the instance share one translation into parent stages.
struct Instance<'a> {
    name: &'a str,
    child: &'a CheckedBlock,
    inputs: Vec<&'a TypedExpr>,
    offset: Cell<Inference<StageOffset>>,
}

/// Translation from a child's stages into its parent's stages. An offset can
/// be negative, unlike a stage; unknown inputs cannot establish one.
#[derive(Clone, Copy)]
enum StageOffset {
    /// The signed difference `parent_stage - child_stage`.
    Known(i128),
    /// An input was unknown, or no staged input could establish an offset.
    Unknown,
}

impl<'a> Instance<'a> {
    /// Index only children with verified fixed-stage interfaces. Unchecked
    /// children and externs remain unknown to the timing walk.
    fn collect(
        block: &'a CheckedBlock,
        children: &BTreeMap<&str, &'a CheckedBlock>,
    ) -> BTreeMap<&'a str, Self> {
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
                && matches!(
                    &child.output_cycles,
                    Some(OutputCycles::Uniform(_) | OutputCycles::ByPort(_))
                )
            {
                // Body checking guarantees one connection for each data port.
                let inputs = child
                    .params
                    .iter()
                    .map(|p| connections[&(name.as_str(), p.name.as_str())])
                    .collect();
                instances.insert(
                    name.as_str(),
                    Self {
                        name,
                        child,
                        inputs,
                        offset: Cell::new(Inference::Unseen),
                    },
                );
            }
        }
        instances
    }

    /// Each staged input must agree on parent stage minus child stage.
    fn infer_offset(
        &self,
        parent_stages: impl Iterator<Item = Stage>,
    ) -> Result<StageOffset, String> {
        let mut offset = None;
        let mut has_unknown_input = false;
        let input_cycles = &self.child.input_cycles;
        for (param, stage) in self.child.params.iter().zip(parent_stages) {
            match stage {
                Stage::At(parent) => {
                    // Child x @0 connected to parent x @4 sets offset +4.
                    // A second child port at @1 must then connect at parent @5.
                    let local = i128::from(input_cycles.get(&param.name).copied().unwrap_or(0));
                    let candidate = i128::from(parent) - local;
                    let agreed = *offset.get_or_insert(candidate);
                    if agreed != candidate {
                        return Err(format!(
                            "instance '{}' input '{}': expected parent cycle {}, got {parent}",
                            self.name,
                            param.name,
                            local + agreed
                        ));
                    }
                }
                Stage::Independent => { /* A constant fits any stage and supplies no offset. */ }
                // Keep scanning so a mismatch between other known inputs
                // still gets reported, regardless of port order.
                Stage::Unknown => has_unknown_input = true,
            }
        }
        Ok(match (has_unknown_input, offset) {
            (false, Some(offset)) => StageOffset::Known(offset),
            _ => StageOffset::Unknown,
        })
    }

    /// Translate a verified child output into the stage seen by its caller.
    fn output_stage(&self, port: &str, offset: StageOffset) -> Result<Stage, String> {
        let local = self
            .child
            .output_cycles
            .as_ref()
            .and_then(|outputs| outputs.stage_for_port(port));
        let (StageOffset::Known(offset), Some(local)) = (offset, local) else {
            return Ok(Stage::Unknown);
        };
        // Connecting child @3 to parent @0 gives offset -3. Only the
        // translated port stage has to fit the unsigned BX stage range.
        let translated = i128::from(local) + offset;
        let stage = u64::try_from(translated).map_err(|_| {
            format!(
                "instance '{}.{port}': translated cycle {translated} is out of range",
                self.name
            )
        })?;
        Ok(Stage::At(stage))
    }
}

/// Build the scopes once, then follow only the drivers reached from the
/// outputs.
struct StageChecker<'a> {
    scopes: Vec<Scope<'a>>,
    structs: &'a StructTable<'a>,
    instances: BTreeMap<&'a str, Instance<'a>>,
    case: Option<cases::Case<'a>>,
}

impl<'a> StageChecker<'a> {
    fn new(
        block: &'a CheckedBlock,
        structs: &'a StructTable<'a>,
        children: &BTreeMap<&str, &'a CheckedBlock>,
        case: Option<cases::Case<'a>>,
    ) -> Self {
        let mut root = Scope::default();
        for param in &block.params {
            let stage = block.input_cycles.get(&param.name).copied().unwrap_or(0);
            root.names.insert(&param.name, ValueSource::Input(stage));
        }
        let mut checker = Self {
            scopes: vec![root],
            structs,
            instances: Instance::collect(block, children),
            case,
        };
        checker.collect_assignments(&block.statements, BLOCK_SCOPE);
        checker
    }

    /// Index writes first so an output can follow a register read to a later
    /// write. In `if en { let d = x; q := d; }`, `q` is visible throughout
    /// the block. Its RHS uses the inner scope to find `d`.
    fn collect_assignments(&mut self, statements: &'a [CheckedStmt], scope: usize) {
        let assignment = |rhs| Assignment {
            rhs,
            scope,
            reads: RefCell::default(),
        };
        for stmt in statements {
            let (owner, name, source) = match stmt {
                CheckedStmt::Let { name, .. } if is_constexpr_only_name(name) => continue,
                CheckedStmt::Let { name, value } | CheckedStmt::DriveWire { name, value, .. } => {
                    (scope, name, ValueSource::Let(assignment(value)))
                }
                CheckedStmt::Write { name, value } => {
                    (BLOCK_SCOPE, name, ValueSource::Register(assignment(value)))
                }
                CheckedStmt::If { condition, body } => {
                    let child = self.scopes.len();
                    self.scopes.push(Scope {
                        enclosing_if: Some(EnclosingIf { scope, condition }),
                        ..Scope::default()
                    });
                    self.collect_assignments(body, child);
                    continue;
                }
                // Instance connections are indexed separately from lexical
                // names.
                _ => continue,
            };
            self.scopes[owner].names.insert(name, source);
        }
    }

    /// A read inside an if body can use a name from any enclosing scope.
    fn value_source(&self, mut scope: usize, name: &str) -> Option<&ValueSource<'a>> {
        loop {
            if let Some(source) = self.scopes[scope].names.get(name) {
                return Some(source);
            }
            scope = self.scopes[scope].enclosing_if.as_ref()?.scope;
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

    /// Choose only a statically known arm. The selector still meets the
    /// selected data at the mux, so a stage-0 opcode cannot select stage-1
    /// data.
    fn choose<'s>(
        &'s self,
        walk: &mut Walk<'s>,
        expr: &'s TypedExpr,
        scope: usize,
        fields: Vec<usize>,
    ) -> Result<(), String> {
        let subject = walk.values.last().expect("selector evaluated");
        match &expr.kind {
            K::If {
                then_value,
                else_value,
                ..
            } => {
                if same_value(then_value, else_value) {
                    // `if en { q } else { q }` is just q.
                    walk.values.pop();
                    walk.pending.push(Step::Expr(then_value, scope, fields));
                } else if let Some(bit) = subject.boolean() {
                    let value = if bit { then_value } else { else_value };
                    walk.schedule([
                        Step::Expr(value, scope, fields),
                        Step::Join(Join::Selected("if")),
                    ]);
                } else {
                    walk.schedule([
                        Step::Expr(then_value, scope, fields.clone()),
                        Step::Expr(else_value, scope, fields),
                        Step::Join(Join::Branches("if")),
                        Step::Join(Join::Select("if")),
                    ]);
                }
            }
            K::Match {
                subject: source,
                arms,
            } => {
                let selected = subject
                    .known
                    .as_ref()
                    .map(|value| selected_match_arm(arms, value, &source.ty))
                    .transpose()?;
                if let Some(first) = arms.first()
                    && arms.iter().all(|arm| same_value(&first.value, &arm.value))
                {
                    // `match en { true => q, false => q }` is just q.
                    walk.values.pop();
                    walk.pending.push(Step::Expr(&first.value, scope, fields));
                } else if let Some(value) = selected {
                    walk.schedule([
                        Step::Expr(value, scope, fields),
                        Step::Join(Join::Selected("match")),
                    ]);
                } else {
                    walk.pending.push(Step::Join(Join::Select("match")));
                    for (index, arm) in arms.iter().enumerate().rev() {
                        if index != 0 {
                            walk.pending.push(Step::Join(Join::Branches("match")));
                        }
                        walk.pending
                            .push(Step::Expr(&arm.value, scope, fields.clone()));
                    }
                }
            }
            _ => unreachable!("choose follows an if or match selector"),
        }
        Ok(())
    }

    /// Walk backwards from one output. Each step either schedules its inputs
    /// or uses stages those inputs have already left on the value stack.
    fn infer(&self, output: &'a TypedExpr, fields: &[usize]) -> Result<Stage, String> {
        let mut walk = Walk {
            pending: vec![Step::Expr(output, BLOCK_SCOPE, fields.to_vec())],
            values: Vec::new(),
        };
        while let Some(step) = walk.pending.pop() {
            match step {
                Step::Expr(expr, scope, fields) => {
                    self.visit_expr(&mut walk, expr, scope, fields)?
                }
                Step::Read(name, source, fields) => {
                    self.visit_read(&mut walk, name, source, fields)?
                }
                Step::ComputeCallOutputStage(instance, port) => {
                    walk.compute_call_output_stage(instance, port)?
                }
                Step::Choose(expr, scope, fields) => self.choose(&mut walk, expr, scope, fields)?,
                Step::Unary(expr) => {
                    let value = walk.values.pop().expect("unary operand evaluated");
                    let known = self
                        .case
                        .as_ref()
                        .expect("case unary step")
                        .fold_operation(expr, &[&value])?;
                    walk.values.push(TimedValue {
                        stage: value.stage,
                        known,
                    });
                }
                Step::Binary(expr) => {
                    let right = walk.values.pop().expect("right operand evaluated");
                    let left = walk.values.pop().expect("left operand evaluated");
                    let case = self.case.as_ref().expect("case binary step");
                    walk.values.push(case.binary(expr, left, right)?);
                }
                Step::Aggregate(count) => walk.finish_aggregate(count, self.case.is_some())?,
                Step::Guard(scope) => {
                    let body = &self.scopes[scope];
                    if let Some(value) = body.guard_stage.borrow().clone() {
                        walk.values.push(value);
                    } else if let Some(enclosing) = &body.enclosing_if {
                        // `if a { if b { ... } }` enables a write with a && b.
                        walk.schedule([
                            Step::Guard(enclosing.scope),
                            Step::Expr(enclosing.condition, enclosing.scope, vec![]),
                            Step::Join(Join::Guards),
                            Step::SaveGuard(scope),
                        ]);
                    } else {
                        walk.values.push(TimedValue {
                            stage: Stage::Independent,
                            known: self.case.as_ref().map(|_| IrValue::bool(true)),
                        });
                    }
                }
                Step::Join(rule) => {
                    let right = walk.values.pop().expect("right operand evaluated");
                    let left = walk.values.pop().expect("left operand evaluated");
                    let value = match &self.case {
                        Some(case) => case.join(rule, left, right)?,
                        None => TimedValue::new(rule.apply(left.stage, right.stage)?),
                    };
                    walk.values.push(value);
                }
                Step::SaveRead(assignment, fields) => {
                    let stage = walk.values.last().expect("assignment evaluated").clone();
                    assignment
                        .reads
                        .borrow_mut()
                        .insert(fields, Inference::Done(stage));
                }
                Step::SaveGuard(scope) => {
                    let stage = walk.values.last().expect("enable evaluated").clone();
                    *self.scopes[scope].guard_stage.borrow_mut() = Some(stage);
                }
            }
        }
        debug_assert_eq!(walk.values.len(), 1);
        Ok(walk.values.pop().expect("output evaluated").stage)
    }

    /// Follow the assignment in its defining scope. Cache each field separately
    /// so `q.saved` can depend on `q.live` without reporting feedback.
    fn visit_read<'s>(
        &'s self,
        walk: &mut Walk<'s>,
        name: &'s str,
        source: &'s ValueSource<'a>,
        fields: Vec<usize>,
    ) -> Result<(), String> {
        let assignment = match source {
            ValueSource::Input(stage) => {
                walk.values.push(TimedValue {
                    stage: Stage::At(*stage),
                    known: self
                        .case
                        .as_ref()
                        .and_then(|case| case.inputs.get(name))
                        .cloned(),
                });
                return Ok(());
            }
            ValueSource::Let(assignment) | ValueSource::Register(assignment) => assignment,
        };
        // In `q + q` the second read reuses the finished stage. In
        // `q := q + x` it finds the first read still in progress.
        let inference = assignment
            .reads
            .borrow()
            .get(&fields)
            .cloned()
            .unwrap_or(Inference::Unseen);
        match inference {
            Inference::Visiting => {
                return Err(format!("timing feedback through '{name}' is unsupported"));
            }
            Inference::Done(stage) => {
                walk.values.push(stage);
                return Ok(());
            }
            Inference::Unseen => {
                assignment
                    .reads
                    .borrow_mut()
                    .insert(fields.clone(), Inference::Visiting);
            }
        }
        walk.pending
            .push(Step::SaveRead(assignment, fields.clone()));
        if let ValueSource::Let(_) = source {
            walk.pending
                .push(Step::Expr(assignment.rhs, assignment.scope, fields));
        } else if let Some(count) = self.register_field_count(&assignment.rhs.ty, &fields)? {
            // Reading all of q has the same timing as rebuilding it from its
            // fields.
            walk.aggregate((0..count).map(|index| {
                let mut member = Vec::with_capacity(fields.len() + 1);
                member.push(index);
                member.extend_from_slice(&fields);
                Step::Read(name, source, member)
            }));
        } else {
            // In `if en { q := x; }`, x and en meet before the clock edge.
            walk.schedule([
                Step::Expr(assignment.rhs, assignment.scope, fields),
                Step::Guard(assignment.scope),
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
        let mut read = |name: &'s str, fields| match self.value_source(scope, name) {
            Some(source) => walk.pending.push(Step::Read(name, source, fields)),
            None => walk.values.push(TimedValue::new(Stage::Unknown)),
        };
        match &expr.kind {
            K::Name(name) => read(name, fields),
            K::InstanceOutput { instance, port } => match self.instances.get(instance.as_str()) {
                Some(child) => walk.read_instance_output(child, port)?,
                // Externs and children without a verified output annotation
                // have no timing interface to use here.
                None => walk.values.push(TimedValue::new(Stage::Unknown)),
            },
            K::BindingTupleIndex { name, index } => {
                // `let (head, _) = pair;` reads the same field as `pair.0`.
                fields.push(*index);
                read(name, fields);
            }
            K::Literal(value) => walk.values.push(TimedValue {
                stage: Stage::Independent,
                known: self.case.as_ref().map(|_| value.clone()),
            }),
            K::Unary { value, .. } | K::Cast { value } if self.case.is_some() => {
                walk.schedule([Step::Expr(value, scope, fields), Step::Unary(expr)]);
            }
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
                let combine = if self.case.is_some() {
                    Step::Binary(expr)
                } else {
                    Step::Join(Join::Operator(*op))
                };
                walk.schedule([operand(left), operand(right), combine]);
            }
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                // `if sel { x } else { y }` is a mux. Check its data arms
                // together, then check that their stage agrees with sel.
                if self.case.is_some() {
                    walk.schedule([operand(condition), Step::Choose(expr, scope, fields)]);
                } else {
                    walk.schedule([
                        operand(condition),
                        Step::Expr(then_value, scope, fields.clone()),
                        Step::Expr(else_value, scope, fields),
                        Step::Join(Join::Branches("if")),
                        Step::Join(Join::Select("if")),
                    ]);
                }
            }
            K::Match { subject, .. } => {
                walk.schedule([operand(subject), Step::Choose(expr, scope, fields)]);
            }
            _ => walk.values.push(TimedValue::new(Stage::Unknown)),
        }
        Ok(())
    }
}

/// Find the first source-ordered match arm for this concrete case.
fn selected_match_arm<'a>(
    arms: &'a [TypedMatchArm],
    value: &IrValue,
    ty: &Type,
) -> Result<&'a TypedExpr, String> {
    for arm in arms {
        for pattern in &arm.patterns {
            if pattern.matches_value(value, ty)? {
                return Ok(&arm.value);
            }
        }
    }
    Err("stage case has no matching arm".into())
}

/// Recognize identical leaves without proving arbitrary expressions equivalent.
fn same_value(a: &TypedExpr, b: &TypedExpr) -> bool {
    match (&a.kind, &b.kind) {
        (K::Name(a), K::Name(b)) => a == b,
        (K::Literal(a), K::Literal(b)) => a == b,
        (
            K::InstanceOutput {
                instance: a,
                port: ap,
            },
            K::InstanceOutput {
                instance: b,
                port: bp,
            },
        ) => a == b && ap == bp,
        (
            K::BindingTupleIndex { name: a, index: ai },
            K::BindingTupleIndex { name: b, index: bi },
        ) => a == b && ai == bi,
        _ => false,
    }
}

/// The work needed to infer one output's stage. `pending` holds instructions;
/// `values` holds the stages those instructions have calculated.
struct Walk<'a> {
    pending: Vec<Step<'a>>,
    values: Vec<TimedValue>,
}

impl<'a> Walk<'a> {
    /// Find the stage of a child output such as `lane.saved`. On the first
    /// read, evaluate the arguments connected to the child before calculating
    /// its output stage. Other outputs of the same instance reuse the offset.
    fn read_instance_output(
        &mut self,
        instance: &'a Instance<'a>,
        port: &'a str,
    ) -> Result<(), String> {
        match instance.offset.get() {
            Inference::Unseen => {
                instance.offset.set(Inference::Visiting);
                // Leave this step pending until all argument stages are known.
                self.pending
                    .push(Step::ComputeCallOutputStage(instance, port));
                // The stack pops inputs in declaration order, which is the
                // order infer_offset uses to pair each stage with its port.
                for &input in instance.inputs.iter().rev() {
                    self.pending.push(Step::Expr(input, BLOCK_SCOPE, vec![]));
                }
            }
            Inference::Visiting => {
                return Err(format!(
                    "timing feedback through instance '{}' is unsupported",
                    instance.name
                ));
            }
            Inference::Done(offset) => self
                .values
                .push(TimedValue::new(instance.output_stage(port, offset)?)),
        }
        Ok(())
    }

    /// Once the call's argument stages are known, calculate its shared
    /// offset and replace those stages with the requested output's stage.
    fn compute_call_output_stage(
        &mut self,
        instance: &Instance<'_>,
        port: &str,
    ) -> Result<(), String> {
        // In `a + lane.out`, a's stage is already on the stack. Consume only
        // lane's argument stages so the pending addition can still use a.
        let input_start = self.values.len() - instance.inputs.len();
        let offset = instance.infer_offset(self.values.drain(input_start..).map(|v| v.stage))?;
        instance.offset.set(Inference::Done(offset));
        self.values
            .push(TimedValue::new(instance.output_stage(port, offset)?));
        Ok(())
    }

    /// Callers list steps in execution order. The stack pops the first one
    /// next.
    fn schedule<const N: usize>(&mut self, steps: [Step<'a>; N]) {
        self.pending.extend(steps.into_iter().rev());
    }

    /// Evaluate the members in order, then combine them into one tuple.
    fn aggregate(
        &mut self,
        members: impl DoubleEndedIterator<Item = Step<'a>> + ExactSizeIterator,
    ) {
        self.pending.push(Step::Aggregate(members.len()));
        self.pending.extend(members.rev());
    }

    /// Join the field stages and build the immutable tuple once, rather than
    /// copying its accumulated fields after each member.
    fn finish_aggregate(&mut self, count: usize, case: bool) -> Result<(), String> {
        let start = self.values.len() - count;
        let mut stage = Stage::Independent;
        let mut known = case.then(|| Vec::with_capacity(count));
        for field in self.values.drain(start..) {
            stage = Join::Fields.apply(stage, field.stage)?;
            if let Some(fields) = &mut known {
                if let Some(value) = field.known {
                    fields.push(value);
                } else {
                    known = None;
                }
            }
        }
        self.values.push(TimedValue {
            stage,
            known: known.map(|fields| IrValue::Tuple(fields.into())),
        });
        Ok(())
    }
}

/// An instruction for the timing walk. Some instructions discover more work;
/// others use stages calculated by earlier instructions. For `q + y`, we
/// schedule Expr(q), Expr(y), then Join(+). The join runs after both stages
/// are on `Walk::values`.
enum Step<'a> {
    /// Infer an expression's stage, e.g. `q + y`. The scope tells us where
    /// to look up names. An empty field path means the whole expression;
    /// for nested fields, the last index is the next field to follow.
    Expr(&'a TypedExpr, usize, Vec<usize>),
    /// Pick a case's matching arm, then check its selector against the arm.
    Choose(&'a TypedExpr, usize, Vec<usize>),
    /// Fold a known unary operand without changing its structural stage.
    Unary(&'a TypedExpr),
    /// Combine the stages and, if possible, the known values of two operands.
    Binary(&'a TypedExpr),
    /// Combine the evaluated tuple fields and their stages.
    Aggregate(usize),
    /// Calculate a block call's output stage after evaluating its arguments.
    /// If `delay` declares input `x @0` and output `saved @1`, calling it
    /// with an argument at stage 4 establishes offset +4, so `saved` is at
    /// stage 5. Cache the offset for the call's other output ports.
    ComputeCallOutputStage(&'a Instance<'a>, &'a str),
    /// Look up the stage of a name such as `q`. An input supplies its stage
    /// directly; a let or register follows its assignment's RHS. A read of
    /// `p.saved` follows only that field.
    Read(&'a str, &'a ValueSource<'a>, Vec<usize>),
    /// Infer the enable stage for a register write. For
    /// `if en { if ready { q := x; } }`, check that en and ready agree in
    /// stage. An unconditional write has a stage-independent enable.
    Guard(usize),
    /// Combine the top two stages using the rule for the hardware operation.
    /// An adder requires matching stages; a register also advances by one.
    Join(Join<'a>),
    /// Remember the result of a name read so `q + q` only infers q once.
    /// Cache fields separately because `p.saved` and `p.live` can differ.
    SaveRead(&'a Assignment<'a>, Vec<usize>),
    /// Cache the enable stage for other writes under the same if.
    SaveGuard(usize),
}

/// The hardware operation where two operand stages meet.
#[derive(Clone, Copy)]
enum Join<'a> {
    /// The two inputs of a combinational operator such as `q + r`.
    Operator(BinaryOp),
    /// Members of a struct or tuple read as one value.
    Fields,
    /// The two data arms of an `if` expression.
    Branches(&'static str),
    /// The condition and the data stage established by `Branches`.
    Select(&'static str),
    /// A known selector and its one selected data arm.
    Selected(&'static str),
    /// An enclosing write guard and the next nested `if` condition.
    Guards,
    /// A register's data and load enable meet before the edge advances the
    /// stage.
    Register(&'a str),
}

impl Join<'_> {
    /// Check alignment first. Only a register advances the result to the next
    /// stage.
    fn apply(self, left: Stage, right: Stage) -> Result<Stage, String> {
        let stage = match (left, right) {
            (Stage::Unknown, _) | (_, Stage::Unknown) => Stage::Unknown,
            (Stage::Independent, stage) | (stage, Stage::Independent) => stage,
            (Stage::At(a), Stage::At(b)) if a != b => return Err(self.mismatch(a, b)),
            (Stage::At(_), Stage::At(_)) => left,
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
            Self::Branches(kind) => format!("{kind} branches combine cycle {a} with cycle {b}"),
            Self::Select(kind) | Self::Selected(kind) => {
                format!("{kind} expression has condition at cycle {a} and data at cycle {b}")
            }
            Self::Guards => format!("nested write guards combine cycle {a} with cycle {b}"),
            Self::Register(name) => {
                format!("write to '{name}' has guard at cycle {b} and data at cycle {a}")
            }
        }
    }
}
