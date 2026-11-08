// SPDX-License-Identifier: Apache-2.0

//! Opt-in stage inference for BX block outputs.
//!
//! Each driver records where a name gets its value. Once the drivers are
//! indexed, the checker walks backwards from each annotated output.
//! For `q + r`, it follows `q` to its register write, looks up input `r`,
//! and checks that both reach the adder at the same stage.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

use super::MAX_ELABORATION_DEPTH;
use crate::ast::{OutputCycles, Param, Type};
use crate::typed::{CheckedBlock, CheckedStmt, TypedExpr, TypedExprKind as K};

const BLOCK_SCOPE: usize = 0;

/// `ports` lists the block's physical outputs in declaration order.
/// `port_names` contains those same names for validating a per-port annotation.
pub(super) fn check_outputs(
    block: &CheckedBlock,
    ports: &[Param],
    port_names: &BTreeSet<String>,
) -> Result<(), String> {
    let Some(expected) = &block.output_cycles else {
        return Ok(());
    };
    if let OutputCycles::ByPort(expected) = expected {
        if let Some(port) = expected.keys().find(|port| !port_names.contains(*port)) {
            return Err(format!("unknown output port '{port}' in cycle annotation"));
        }
        if let Some(port) = port_names.iter().find(|port| !expected.contains_key(*port)) {
            return Err(format!("missing cycle annotation for output port '{port}'"));
        }
    }

    let checker = Checker::new(block);
    let check = |fields: &[usize], expected| {
        checker
            .infer(&block.output, BLOCK_SCOPE, fields, 0)?
            .check(expected)
    };
    let aggregate = matches!(block.result, Type::Named(_) | Type::Tuple(_));
    if !aggregate && let OutputCycles::Uniform(cycle) = expected {
        return check(&[], *cycle);
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
#[derive(Clone, Copy)]
enum Cycle {
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

    /// Operands of a combinational operation meet at one stage.
    fn merge(self, other: Self, mismatch: impl FnOnce(u64, u64) -> String) -> Result<Self, String> {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Ok(Self::Unknown),
            (Self::Independent, cycle) | (cycle, Self::Independent) => Ok(cycle),
            (Self::At(a), Self::At(b)) if a == b => Ok(self),
            (Self::At(a), Self::At(b)) => Err(mismatch(a, b)),
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
    Input(u64),
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

/// Build the scopes once, then follow only the drivers reached from the
/// outputs.
struct Checker<'a> {
    scopes: Vec<Scope<'a>>,
}

impl<'a> Checker<'a> {
    fn new(block: &'a CheckedBlock) -> Self {
        let mut root = Scope::default();
        for param in &block.params {
            let stage = block.input_cycles.get(&param.name).copied().unwrap_or(0);
            root.names.insert(&param.name, Binding::Input(stage));
        }
        let mut checker = Self { scopes: vec![root] };
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
                CheckedStmt::Let { name, value } => (scope, name, Binding::Let(driver(value))),
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
                // Declarations and instance connections supply no inferred
                // stage.
                _ => continue,
            };
            self.scopes[owner].names.insert(name, binding);
        }
    }

    /// In `if a { if b { q := x; } }`, the inner write uses `a && b`. Writes in
    /// the same body share this result. The root enable is implicitly true.
    fn guard_cycle(&self, scope: usize, depth: usize) -> Result<Cycle, String> {
        let scope = &self.scopes[scope];
        let Some((parent, condition)) = scope.parent else {
            return Ok(Cycle::Independent);
        };
        if let Some(cycle) = scope.guard_cycle.get() {
            return Ok(cycle);
        }
        let outer = self.guard_cycle(parent, depth)?;
        let inner = self.infer(condition, parent, &[], depth)?;
        let cycle = outer.merge(inner, |a, b| {
            format!("nested write guards combine cycle {a} with cycle {b}")
        })?;
        scope.guard_cycle.set(Some(cycle));
        Ok(cycle)
    }

    /// An empty field path reads the whole value. For `p.header.id` the walk
    /// pushes `id`, then `header`. At the tuple for `p` it selects `header`
    /// first.
    fn infer(
        &self,
        expr: &'a TypedExpr,
        scope: usize,
        fields: &[usize],
        depth: usize,
    ) -> Result<Cycle, String> {
        // Depth counts DFS steps through operands and drivers. A register
        // write adds a hardware stage.
        if depth > MAX_ELABORATION_DEPTH {
            return Err(format!(
                "timing dependency depth exceeds {MAX_ELABORATION_DEPTH}"
            ));
        }
        let operand = |value, fields: &[usize]| self.infer(value, scope, fields, depth + 1);
        match &expr.kind {
            K::Name(name) => self.infer_name(name, scope, fields, depth + 1),
            K::Literal(_) => Ok(Cycle::Independent),
            K::Unary { value, .. } | K::Cast { value } => operand(value, fields),
            K::TupleIndex { value, index } => {
                let mut path = Vec::with_capacity(fields.len() + 1);
                path.extend_from_slice(fields);
                path.push(*index);
                operand(value, &path)
            }
            K::Tuple(members) => {
                if let Some((&index, rest)) = fields.split_last() {
                    return operand(&members[index], rest);
                }
                let mut cycle = Cycle::Independent;
                for member in members {
                    cycle = cycle.merge(operand(member, &[])?, |a, b| {
                        format!("aggregate combines cycle {a} with cycle {b}")
                    })?;
                }
                Ok(cycle)
            }
            K::Binary { op, left, right } => {
                let left = operand(left, &[])?;
                let right = operand(right, &[])?;
                left.merge(right, |a, b| {
                    format!("operator '{op}' combines cycle {a} with cycle {b}")
                })
            }
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                // This is a data mux, as in `if sel { x } else { y }`.
                // A statement such as `if en { q := x; }` drives an enable.
                let select = operand(condition, &[])?;
                let left = operand(then_value, fields)?;
                let right = operand(else_value, fields)?;
                let data = left.merge(right, |a, b| {
                    format!("if branches combine cycle {a} with cycle {b}")
                })?;
                select.merge(data, |a, b| {
                    format!("if expression has condition at cycle {a} and data at cycle {b}")
                })
            }
            _ => Ok(Cycle::Unknown),
        }
    }

    /// Resolve the name where it was read, then evaluate its driver where it
    /// was written. Input stages are already known.
    fn infer_name(
        &self,
        name: &str,
        mut scope: usize,
        fields: &[usize],
        depth: usize,
    ) -> Result<Cycle, String> {
        let binding = loop {
            if let Some(binding) = self.scopes[scope].names.get(name) {
                break binding;
            }
            let Some((parent, _)) = self.scopes[scope].parent else {
                return Ok(Cycle::Unknown);
            };
            scope = parent;
        };
        let driver = match binding {
            Binding::Input(stage) => return Ok(Cycle::At(*stage)),
            Binding::Let(driver) | Binding::Register(driver) => driver,
        };
        // In `q + q` the second read reuses the finished stage. In `q := q + x`
        // the second read finds None because the first is still in progress.
        if let Some(stage) = driver.reads.borrow().get(fields) {
            return stage.ok_or_else(|| format!("timing feedback through '{name}' is unsupported"));
        }
        driver.reads.borrow_mut().insert(fields.to_vec(), None);
        let mut stage = self.infer(driver.rhs, driver.scope, fields, depth)?;
        if let Binding::Register(_) = binding {
            // In `if en { q := x; }`, en and x meet before the clock edge.
            // If both are @0, q is @1 even when its enable holds the old value.
            let enable = self.guard_cycle(driver.scope, depth)?;
            stage = enable
                .merge(stage, |en, data| {
                    format!("write to '{name}' has guard at cycle {en} and data at cycle {data}")
                })?
                .registered(name)?;
        }
        driver
            .reads
            .borrow_mut()
            .insert(fields.to_vec(), Some(stage));
        Ok(stage)
    }
}
