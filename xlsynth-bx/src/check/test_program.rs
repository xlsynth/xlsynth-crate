// SPDX-License-Identifier: Apache-2.0

//! Check test expressions in their declaring module and compile the circuit
//! once. The runner drives typed inputs for stimulus, loop rows and snapshots.

use std::collections::BTreeMap;

use super::unroll::Shadowed;
use super::{
    BodyChecker, CheckedBinding, CheckedBlock, CheckedFunction, CheckedStmt, K, TypedExpr,
    TypedIterable,
};
use crate::ast::{BindingPattern, Param, TestDecl, TestKind, TestStmt, TestStmtKind, Type};
use crate::test_runner::{TestAction, TestIterable, TestProgram, TestStep, TestSuite, input_name};

/// The synthetic harness interface and the source registers that can be driven.
struct TestPorts {
    inputs: Vec<Param>,
    outputs: Vec<TypedExpr>,
    drivers: BTreeMap<String, usize>,
}

impl TestPorts {
    fn input(&mut self, ty: Type) -> (usize, TypedExpr) {
        let index = self.inputs.len();
        let name = input_name(index);
        self.inputs.push(Param {
            name: name.clone(),
            ty: ty.clone(),
        });
        (
            index,
            TypedExpr {
                ty,
                kind: K::Name(name),
            },
        )
    }

    fn output(&mut self, expr: TypedExpr) -> usize {
        let index = self.outputs.len();
        self.outputs.push(expr);
        index
    }

    /// Keep a reached pure statement's lexical initializers and operands in
    /// one IR cone, including initializers unused by its final value.
    fn group_pure_outputs(
        &mut self,
        kind: TestKind,
        first: usize,
        bindings: &[CheckedStmt],
    ) -> Option<usize> {
        if kind != TestKind::Function || bindings.is_empty() {
            return None;
        }
        let operands = &self.outputs[first..];
        let bindings: Vec<_> = bindings
            .iter()
            .filter_map(|binding| {
                let CheckedStmt::Let { name, value } = binding else {
                    unreachable!("pure test expressions cannot elaborate hardware");
                };
                if operands
                    .iter()
                    .any(|e| matches!(&e.kind, K::Name(n) if n == name))
                {
                    return None;
                }
                Some(TypedExpr {
                    ty: value.ty.clone(),
                    kind: K::Name(name.clone()),
                })
            })
            .collect();
        if bindings.is_empty() {
            return None;
        }
        let mut fields = self.outputs.split_off(first);
        fields.extend(bindings);
        let ty = Type::Tuple(fields.iter().map(|e| e.ty.clone()).collect());
        Some(self.output(TypedExpr {
            ty,
            kind: K::Tuple(fields),
        }))
    }

    fn check_pattern(&self, pattern: &BindingPattern) -> Result<(), String> {
        let mut collision = None;
        pattern.visit_names(&mut |name| {
            if self.drivers.contains_key(name) {
                collision = Some(name.to_owned());
            }
        });
        if let Some(name) = collision {
            return Err(format!("cannot shadow test stimulus register '{name}'"));
        }
        Ok(())
    }
}

impl<'m> BodyChecker<'_, 'm> {
    /// Use normal BX type inference, specialization and block call elaboration
    /// for test code. Only the sequence of actions is interpreted separately.
    pub(super) fn check_test(
        &mut self,
        test: &'m TestDecl,
        harness: String,
        suite: &mut TestSuite,
        functions: &mut Vec<CheckedFunction>,
        blocks: &mut Vec<CheckedBlock>,
    ) -> Result<(), String> {
        let mut ports = TestPorts {
            inputs: Vec::new(),
            outputs: Vec::new(),
            drivers: BTreeMap::new(),
        };
        let steps = self.test_steps(test, &test.statements, &mut ports, true)?;
        let result = Type::Tuple(
            ports
                .outputs
                .iter()
                .map(|output| output.ty.clone())
                .collect(),
        );
        let output = TypedExpr {
            ty: result.clone(),
            kind: K::Tuple(ports.outputs),
        };
        let statements = std::mem::take(&mut self.statements);
        match test.kind {
            TestKind::Function => {
                let bindings = statements
                    .into_iter()
                    .map(|stmt| match stmt {
                        CheckedStmt::Let { name, value } => CheckedBinding { name, value },
                        _ => unreachable!("pure test expressions cannot elaborate hardware"),
                    })
                    .collect();
                functions.push(CheckedFunction {
                    name: harness.clone(),
                    public: false,
                    params: ports.inputs,
                    result,
                    bindings,
                    output,
                });
            }
            TestKind::Block => blocks.push(CheckedBlock {
                name: harness.clone(),
                source_name: test.name.clone(),
                source: None,
                public: false,
                clock: "clk".into(),
                reset: Some("rst".into()),
                params: ports.inputs,
                input_cycles: BTreeMap::new(),
                result,
                output_cycles: None,
                timing_error: None,
                statements,
                output,
            }),
        }
        suite.programs.push(TestProgram {
            name: test.name.clone(),
            position: test.position,
            kind: test.kind,
            harness,
            steps,
        });
        Ok(())
    }

    /// A clocked test's initial declarations form a static circuit. Lets in
    /// the subsequent procedure sample their initializer when control reaches
    /// it.
    fn test_steps(
        &mut self,
        test: &TestDecl,
        statements: &[TestStmt],
        ports: &mut TestPorts,
        mut setup: bool,
    ) -> Result<Vec<TestStep>, String> {
        let mut saved = Vec::new();
        let mut steps = Vec::new();
        for stmt in statements {
            let at_stmt = |message: String| {
                format!(
                    "{}:{}: test '{}': {message}",
                    stmt.position.line, stmt.position.column, test.name
                )
            };
            setup &= matches!(stmt.kind, TestStmtKind::Let(_) | TestStmtKind::Reg { .. });
            let first_binding = self.statements.len();
            let first_output = ports.outputs.len();
            let group_outputs = |checker: &Self, ports: &mut TestPorts| {
                ports.group_pure_outputs(
                    test.kind,
                    first_output,
                    &checker.statements[first_binding..],
                )
            };
            let (action, group) = match &stmt.kind {
                TestStmtKind::For {
                    pattern,
                    iterable,
                    body,
                } => {
                    ports.check_pattern(pattern).map_err(at_stmt)?;
                    self.validate_source_pattern(pattern).map_err(at_stmt)?;
                    let iteration = self.counted_iteration(iterable, None).map_err(at_stmt)?;
                    let iterable = match iteration.iterable {
                        TypedIterable::Range { start } => TestIterable::Range(start),
                        TypedIterable::Array(array) => TestIterable::Array(ports.output(array)),
                    };
                    let group = group_outputs(self, ports);
                    let (port, element) = ports.input(iteration.ty);
                    let mut loop_scope = Vec::new();
                    self.bind_pattern(pattern, element, iteration.widths, &mut loop_scope)
                        .map_err(at_stmt)?;
                    let body = self.test_steps(test, body, ports, false)?;
                    self.leave_scope(loop_scope);
                    (
                        TestAction::For {
                            port,
                            iterable,
                            count: iteration.count,
                            body,
                        },
                        group,
                    )
                }
                TestStmtKind::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let condition =
                        ports.output(self.expr(condition, Some(&Type::Bool)).map_err(at_stmt)?);
                    let group = group_outputs(self, ports);
                    (
                        TestAction::If {
                            condition,
                            then_body: self.test_steps(test, then_body, ports, false)?,
                            else_body: self.test_steps(test, else_body, ports, false)?,
                        },
                        group,
                    )
                }
                kind => match self
                    .test_action(test.kind, kind, ports, setup, &mut saved)
                    .map_err(at_stmt)?
                {
                    Some(action) => (action, group_outputs(self, ports)),
                    None => continue,
                },
            };
            steps.push(TestStep {
                position: stmt.position,
                action,
                group,
            });
        }
        self.leave_scope(saved);
        Ok(steps)
    }

    fn test_action(
        &mut self,
        kind: TestKind,
        stmt: &TestStmtKind,
        ports: &mut TestPorts,
        setup: bool,
        saved: &mut Vec<Shadowed>,
    ) -> Result<Option<TestAction>, String> {
        Ok(Some(match stmt {
            TestStmtKind::Reg { name, ty, init } => {
                if kind != TestKind::Block || !setup {
                    return Err("declare stimulus registers in the initial block test setup".into());
                }
                if name == "_" || self.names.contains(name) {
                    return Err(format!("invalid or duplicate stimulus register '{name}'"));
                }
                let ty = self.source_type(ty)?;
                let initial = match init {
                    Some(expr) => self.expr(expr, Some(&ty))?,
                    None => TypedExpr {
                        kind: self.intrinsic_zero(&ty)?,
                        ty: ty.clone(),
                    },
                };
                let value = ports.output(initial);
                let (port, input) = ports.input(ty);
                ports.drivers.insert(name.clone(), port);
                self.bind_scoped_value(name, input, None, saved);
                TestAction::Drive { port, value }
            }
            TestStmtKind::Assign { name, value } => {
                let port = *ports
                    .drivers
                    .get(name)
                    .ok_or_else(|| format!("'{name}' is not a stimulus register"))?;
                let expr = self.expr(value, Some(&ports.inputs[port].ty))?;
                TestAction::Drive {
                    port,
                    value: ports.output(expr),
                }
            }
            TestStmtKind::Let(binding) => {
                ports.check_pattern(&binding.pattern)?;
                let mut init =
                    self.pattern_initializer(binding, setup && kind == TestKind::Block)?;
                if setup && kind == TestKind::Block {
                    self.bind_pattern(&binding.pattern, init.value, init.widths, saved)?;
                    return Ok(None);
                }
                if kind == TestKind::Function
                    || init.value.has_constant_inputs(&self.constexpr_names)
                {
                    // Destructuring a tuple constructor preserves the constexpr
                    // status of each field, even when another field varies.
                    let value = if matches!(&binding.pattern, BindingPattern::Tuple(_))
                        && matches!(&init.value.kind, K::Tuple(_))
                    {
                        init.value
                    } else {
                        self.share_scoped_value(init.value)
                    };
                    let output = ports.output(value.clone());
                    self.bind_pattern(&binding.pattern, value, init.widths, saved)?;
                    return Ok(Some(TestAction::Evaluate(output)));
                }
                let (port, input) = ports.input(init.value.ty.clone());
                let value = if matches!(&binding.pattern, BindingPattern::Tuple(_)) {
                    self.snapshot_pattern_value(&mut init.value, input)
                } else {
                    input
                };
                self.bind_pattern(&binding.pattern, value, init.widths, saved)?;
                TestAction::Drive {
                    port,
                    value: ports.output(init.value),
                }
            }
            TestStmtKind::Tick { reset } => {
                if kind != TestKind::Block {
                    return Err("tick! requires a block test".into());
                }
                TestAction::Tick(
                    reset
                        .as_ref()
                        .map(|r| self.expr(r, Some(&Type::Bool)).map(|e| ports.output(e)))
                        .transpose()?,
                )
            }
            TestStmtKind::Assert { predicate, label } => TestAction::Assert {
                predicate: ports.output(self.expr(predicate, Some(&Type::Bool))?),
                label: label.clone(),
            },
            TestStmtKind::AssertEq { actual, expected } => {
                let (a, b) = self.comparison_pair(actual, expected)?;
                TestAction::AssertEq {
                    actual: ports.output(a),
                    expected: ports.output(b),
                }
            }
            TestStmtKind::For { .. } | TestStmtKind::If { .. } => {
                unreachable!("nested statements have their own scopes")
            }
        }))
    }

    /// Keep constexpr fields of a destructured tuple and read its dynamic
    /// fields from the single snapshot taken at the let statement.
    fn snapshot_pattern_value(&mut self, source: &mut TypedExpr, snapshot: TypedExpr) -> TypedExpr {
        if let K::Tuple(fields) = &mut source.kind {
            let shared = self.share_scoped_value(snapshot);
            let K::Name(name) = shared.kind else {
                unreachable!("snapshot inputs and their shared projections are names");
            };
            let fields = fields
                .iter_mut()
                .enumerate()
                .map(|(index, field)| {
                    let projection = TypedExpr {
                        ty: field.ty.clone(),
                        kind: K::BindingTupleIndex {
                            name: name.clone(),
                            index,
                        },
                    };
                    self.snapshot_pattern_value(field, projection)
                })
                .collect();
            return TypedExpr {
                ty: source.ty.clone(),
                kind: K::Tuple(fields),
            };
        }
        if source.has_constant_inputs(&self.constexpr_names) {
            let value = self.share_scoped_value(std::mem::replace(source, snapshot));
            *source = value.clone();
            value
        } else {
            snapshot
        }
    }
}
