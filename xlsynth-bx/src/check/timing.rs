// SPDX-License-Identifier: Apache-2.0

//! Opt-in stage inference for BX block outputs.
//!
//! Each driver records where a name gets its value. Once the drivers are
//! indexed, the checker walks backwards from each annotated output.
//! For `q + r`, it follows `q` to its register write, looks up input `r`,
//! and checks that both reach the adder at the same stage.

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

    let mut checker = Checker::new(block);
    let mut check = |fields: &[usize], expected| {
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
        let fields = if aggregate {
            std::slice::from_ref(&index)
        } else {
            &[]
        };
        check(fields, cycle).map_err(|err| format!("output '{}': {err}", port.name))?;
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

/// Lets use their defining scope. Registers are block-wide but their RHS and
/// enable are evaluated in the scope of the write.
#[derive(Clone, Copy)]
enum Driver<'a> {
    Let(&'a TypedExpr),
    Register {
        rhs: &'a TypedExpr,
        write_scope: usize,
    },
}

/// Revisiting an active read detects feedback. A completed read supplies its
/// cached stage.
enum Visit {
    Visiting,
    Inferred(Cycle),
}

/// A lexical scope owns its names, their cached field reads, and its enclosing
/// guard.
#[derive(Default)]
struct Scope<'a> {
    /// The enclosing scope's index and the condition that introduced this body.
    /// The condition can refer to lets visible in the enclosing scope.
    parent: Option<(usize, &'a TypedExpr)>,
    guard_cycle: Option<Cycle>,
    drivers: BTreeMap<&'a str, Driver<'a>>,
    reads: BTreeMap<(&'a str, Vec<usize>), Visit>,
}

/// Drives a backwards DFS from the annotated output ports.
struct Checker<'a> {
    inputs: BTreeMap<&'a str, Cycle>,
    scopes: Vec<Scope<'a>>,
}

impl<'a> Checker<'a> {
    fn new(block: &'a CheckedBlock) -> Self {
        let mut inputs = BTreeMap::new();
        for param in &block.params {
            let stage = block.input_cycles.get(&param.name).copied().unwrap_or(0);
            inputs.insert(param.name.as_str(), Cycle::At(stage));
        }
        let mut checker = Self {
            inputs,
            scopes: vec![Scope::default()],
        };
        checker.collect_drivers(&block.statements, BLOCK_SCOPE);
        checker
    }

    /// Index drivers before following reads, since a register's write can
    /// appear later.
    fn collect_drivers(&mut self, statements: &'a [CheckedStmt], scope: usize) {
        for stmt in statements {
            let (binding_scope, name, driver) = match stmt {
                CheckedStmt::Let { name, value } => (scope, name, Driver::Let(value)),
                // In `if en { let d = x; q := d; }`, a later output can read
                // `q` at block scope. Follow the write back into the if body
                // to resolve `d` and find the enable for that write.
                CheckedStmt::Write { name, value } => (
                    BLOCK_SCOPE,
                    name,
                    Driver::Register {
                        rhs: value,
                        write_scope: scope,
                    },
                ),
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
            self.scopes[binding_scope].drivers.insert(name, driver);
        }
    }

    /// The effective enable is the AND of this scope's condition and its
    /// parent's enable.
    fn guard_cycle(&mut self, scope: usize, depth: usize) -> Result<Cycle, String> {
        let Some((parent, condition)) = self.scopes[scope].parent else {
            // An unconditional write has an implicit true enable at any stage.
            return Ok(Cycle::Independent);
        };
        if let Some(cycle) = self.scopes[scope].guard_cycle {
            return Ok(cycle);
        }
        // In `if a { if b { q := x; } }`, the inner write uses `a && b`.
        // Infer `b` in the outer body, where its name was in scope.
        let outer = self.guard_cycle(parent, depth)?;
        let inner = self.infer(condition, parent, &[], depth)?;
        let cycle = outer.merge(inner, |a, b| {
            format!("nested write guards combine cycle {a} with cycle {b}")
        })?;
        self.scopes[scope].guard_cycle = Some(cycle);
        Ok(cycle)
    }

    /// An empty field path reads the whole value. For `p.header.id` the walk
    /// pushes `id`, then `header`. At the tuple for `p` it selects `header`
    /// first.
    fn infer(
        &mut self,
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
        let mut operand = |value, fields: &[usize]| self.infer(value, scope, fields, depth + 1);
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
                // A statement such as `if en { q := x; }` uses guard_cycle.
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

    /// Find the binding, then follow its RHS. Revisiting an active field read
    /// is feedback.
    fn infer_name(
        &mut self,
        name: &'a str,
        mut scope: usize,
        fields: &[usize],
        depth: usize,
    ) -> Result<Cycle, String> {
        let driver = loop {
            if let Some(&driver) = self.scopes[scope].drivers.get(name) {
                break driver;
            }
            let Some((parent, _)) = self.scopes[scope].parent else {
                return Ok(self.inputs.get(name).copied().unwrap_or(Cycle::Unknown));
            };
            scope = parent;
        };

        // Include the field path in the cache key because `p.saved` and
        // `p.live` can come from different stages of the same struct.
        let read = (name, fields.to_vec());
        let reads = &mut self.scopes[scope].reads;
        // With `q := x`, the second read in `q + q` reuses a finished result.
        // With `q := q + x`, the RHS reaches `q` while its read is active.
        match reads.get(&read) {
            Some(Visit::Inferred(cycle)) => return Ok(*cycle),
            Some(Visit::Visiting) => {
                return Err(format!("timing feedback through '{name}' is unsupported"));
            }
            None => {
                reads.insert(read.clone(), Visit::Visiting);
            }
        }
        let cycle = match driver {
            Driver::Let(rhs) => self.infer(rhs, scope, fields, depth)?,
            Driver::Register { rhs, write_scope } => {
                // For `if en { q := x; }`, compare `en` and `x` on the input
                // side of the edge. If both are @0, `q` is @1. A low enable
                // holds the previous value in that same stage.
                let data = self.infer(rhs, write_scope, fields, depth)?;
                let enable = self.guard_cycle(write_scope, depth)?;
                let input = enable.merge(data, |en, data| {
                    format!("write to '{name}' has guard at cycle {en} and data at cycle {data}")
                })?;
                input.registered(name)?
            }
        };
        self.scopes[scope]
            .reads
            .insert(read, Visit::Inferred(cycle));
        Ok(cycle)
    }
}
