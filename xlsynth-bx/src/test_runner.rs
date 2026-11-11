// SPDX-License-Identifier: Apache-2.0

//! Execute test control flow over typed BX expressions and clocked hardware.

mod simulation;

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use xlsynth_pir::ir::{Fn, NodePayload, NodeRef, Package, PackageMember};
use xlsynth_pir::ir_eval::{AssumptionFailureKind, FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_fn_cone_extract::{SinkSelector, extract_fn_cone_to_params};
use xlsynth_pir::ir_value_utils::zero_ir_value_for_type;
use xlsynth_pir::{IrBits, IrValue};

use crate::ast::{SourcePosition, TestKind};
use crate::{BxWarning, ImportOptions, ResetBehavior, bx_import, check, lower_module};
use simulation::Simulation;

/// Outcome of a single test, in source declaration order.
#[derive(Clone, Debug)]
pub struct BxTestResult {
    pub name: String,
    pub failure: Option<String>,
}

/// Results of running the tests declared in the requested BX file.
#[derive(Clone, Debug)]
pub struct BxTestReport {
    pub tests: Vec<BxTestResult>,
    pub warnings: Vec<BxWarning>,
}

/// Test-only IR names, source locations and sequenced instructions.
pub(crate) struct TestSuite {
    reserved: BTreeSet<String>,
    pub(crate) programs: Vec<TestProgram>,
}

impl TestSuite {
    pub(crate) fn reserve_names(&mut self, names: &BTreeSet<String>) {
        self.reserved.extend(names.iter().cloned());
    }

    /// Allocate a legal symbol disjoint from the source and linked IR names.
    pub(crate) fn harness_name(&mut self, test: &str) -> String {
        let mut name = format!(
            "bx_test_{}",
            test.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_")
        );
        while !self.reserved.insert(name.clone()) {
            name.push('_');
        }
        name
    }
}

/// Expressions are output positions on the compiled function or block.
pub(crate) struct TestProgram {
    pub(crate) name: String,
    pub(crate) position: SourcePosition,
    pub(crate) kind: TestKind,
    pub(crate) harness: String,
    pub(crate) steps: Vec<TestStep>,
}

/// One located action in the imperative test program.
pub(crate) struct TestStep {
    pub(crate) position: SourcePosition,
    pub(crate) action: TestAction,
    /// When present, this output is a tuple containing the action's operands
    /// (indexed from this position), followed by its lexical initializers.
    pub(crate) group: Option<usize>,
}

/// Runtime control over the inputs and outputs of a test harness.
pub(crate) enum TestAction {
    Evaluate(usize),
    Drive {
        port: usize,
        value: usize,
    },
    Tick(Option<usize>),
    Assert {
        predicate: usize,
        label: Option<String>,
    },
    AssertEq {
        actual: usize,
        expected: usize,
    },
    For {
        port: usize,
        iterable: TestIterable,
        count: usize,
        body: Vec<TestStep>,
    },
    If {
        condition: usize,
        then_body: Vec<TestStep>,
        else_body: Vec<TestStep>,
    },
}

pub(crate) enum TestIterable {
    Array(usize),
    Range(IrBits),
}

pub(crate) fn input_name(index: usize) -> String {
    format!("bx_test_input_{index}")
}

/// Check the file and its imports, then execute each root test independently.
/// A failed assertion is a test result. Parse, type and link errors are file
/// errors.
pub fn run_tests(
    source: &str,
    path: &Path,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    warnings_as_errors: bool,
) -> Result<BxTestReport, String> {
    let mut suite = TestSuite {
        reserved: package
            .members
            .iter()
            .map(|m| m.graph().name.clone())
            .collect(),
        programs: Vec::new(),
    };
    let at_file = |message: String| {
        // Parser and test-statement errors already begin with a line and
        // column.
        let space = if message.starts_with(|c: char| c.is_ascii_digit()) {
            ""
        } else {
            " "
        };
        format!("{}:{space}{message}", path.display())
    };
    let (root, imports) = bx_import::read_for_tests(source, options).map_err(at_file)?;
    let mut checked = check::check_sources_with_tests(&root, imports, options, Some(&mut suite))
        .map_err(bx_import::diagnostic)
        .map_err(at_file)?;
    for block in &mut checked.blocks {
        if let Some(source) = &mut block.source {
            source.path.get_or_insert_with(|| path.to_owned());
        }
    }
    let package = lower_module(&checked, package, reset).map_err(at_file)?;
    let warnings = checked.warnings;
    if warnings_as_errors && !warnings.is_empty() {
        let messages = warnings
            .iter()
            .map(|w| w.display_with_root(path).to_string())
            .collect::<Vec<_>>();
        return Err(format!(
            "{}\nBX warnings treated as errors",
            messages.join("\n")
        ));
    }
    let tests = suite
        .programs
        .into_iter()
        .map(|program| {
            let result = run_program(&program, &package, reset.active_low)
                .err()
                .map(|failure| {
                    format!(
                        "{}:{}:{}: test '{}'{}: {}",
                        path.display(),
                        failure.position.line,
                        failure.position.column,
                        program.name,
                        if program.kind == TestKind::Block {
                            format!(" at tick {}", failure.tick)
                        } else {
                            String::new()
                        },
                        failure.message
                    )
                });
            BxTestResult {
                name: program.name,
                failure: result,
            }
        })
        .collect();
    Ok(BxTestReport { tests, warnings })
}

/// An expression's IR dependencies and the harness input ports they read.
struct PureExpression {
    function: Fn,
    ports: Vec<usize>,
    last: Option<(Vec<IrValue>, IrValue)>,
}

impl PureExpression {
    /// Extract the selected output once. The original package supplies its
    /// callees.
    fn extract(harness: &Fn, node: NodeRef) -> Result<Self, String> {
        let sink = SinkSelector::TextId(harness.get_node(node).text_id);
        let mut cone = extract_fn_cone_to_params(harness, None, sink, false)?;
        let ports = cone
            .used_params
            .iter()
            .map(|p| harness.param_index(*p).expect("extracted harness input"))
            .collect();
        let Some(PackageMember::Function(function)) = cone.package.members.pop() else {
            return Err("missing extracted test expression".into());
        };
        Ok(Self {
            function,
            ports,
            last: None,
        })
    }

    /// Cache the expression's value only after its contracts pass.
    fn evaluate(&mut self, package: &Package, inputs: &[IrValue]) -> Result<IrValue, String> {
        let args = self
            .ports
            .iter()
            .map(|&i| inputs[i].clone())
            .collect::<Vec<_>>();
        if let Some((previous, result)) = &self.last
            && previous == &args
        {
            return Ok(result.clone());
        }
        let value = match eval_fn_in_package(package, &self.function, &args) {
            FnEvalResult::Success(success) => success.value,
            FnEvalResult::Failure(failure) => {
                if let Some(assertion) = failure.assertion_failures.first() {
                    return Err(format!(
                        "IR assertion '{}' failed: {}",
                        assertion.label, assertion.message
                    ));
                }
                let assumption = failure
                    .assumption_failures
                    .first()
                    .expect("failed evaluation contains an assertion or assumption failure");
                let operation = match assumption.kind {
                    AssumptionFailureKind::ArrayIndexOutOfBounds => "array_index",
                    AssumptionFailureKind::ArrayUpdateOutOfBounds => "array_update",
                };
                return Err(format!(
                    "IR {operation} at node {} was out of bounds despite assumed_in_bounds=true",
                    assumption.node_text_id
                ));
            }
        };
        self.last = Some((args, value.clone()));
        Ok(value)
    }
}

/// Pure tests evaluate selected expressions. Block tests sample a live circuit.
enum Harness<'a> {
    Function {
        package: &'a Package,
        function: &'a Fn,
        inputs: Vec<IrValue>,
        outputs: &'a [NodeRef],
        expressions: HashMap<NodeRef, PureExpression>,
    },
    Block(Box<Simulation>),
}

impl Harness<'_> {
    fn drive(&mut self, port: usize, value: IrValue) -> Result<(), String> {
        match self {
            Self::Function { inputs, .. } => {
                inputs[port] = value;
                Ok(())
            }
            Self::Block(sim) => sim.drive(&input_name(port), &value),
        }
    }

    fn output(&mut self, index: usize) -> Result<IrValue, String> {
        match self {
            Self::Function {
                package,
                function,
                inputs,
                outputs,
                expressions,
            } => {
                let node = outputs[index];
                if let NodePayload::Literal(value) = &function.get_node(node).payload {
                    return Ok(value.clone());
                }
                let expression = match expressions.entry(node) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(PureExpression::extract(function, node)?),
                };
                expression.evaluate(package, inputs)
            }
            Self::Block(sim) => sim.output(index),
        }
    }
}

/// Runtime errors retain the innermost statement and edge count.
struct Failure {
    position: SourcePosition,
    tick: u64,
    message: String,
}

/// Program counter state shared by nested loops and conditionals.
struct Execution<'a> {
    harness: Harness<'a>,
    tick: u64,
    active_low: bool,
}

fn run_program(program: &TestProgram, package: &Package, active_low: bool) -> Result<(), Failure> {
    let at_declaration = |message| Failure {
        position: program.position,
        tick: 0,
        message,
    };
    let harness = match program.kind {
        TestKind::Function => {
            let function = package.get_fn(&program.harness).ok_or_else(|| {
                at_declaration(format!("missing test function '{}'", program.harness))
            })?;
            let result = function.ret_node_ref.expect("test harness return value");
            let NodePayload::Tuple(outputs) = &function.get_node(result).payload else {
                return Err(at_declaration(
                    "test harness must return its expression tuple".into(),
                ));
            };
            Harness::Function {
                package,
                function,
                inputs: function
                    .param_nodes()
                    .map(|p| zero_ir_value_for_type(&p.ty))
                    .collect(),
                outputs,
                expressions: HashMap::new(),
            }
        }
        TestKind::Block => Harness::Block(Box::new(
            Simulation::new(package, &program.harness).map_err(at_declaration)?,
        )),
    };
    Execution {
        harness,
        tick: 0,
        active_low,
    }
    .execute(&program.steps)
}

impl Execution<'_> {
    /// Evaluate only the selected test statements, keeping input drives across
    /// edges.
    fn execute(&mut self, steps: &[TestStep]) -> Result<(), Failure> {
        for step in steps {
            let at_step = |message| Failure {
                position: step.position,
                tick: self.tick,
                message,
            };
            let group = step
                .group
                .map(|index| self.harness.output(index).map(|values| (index, values)))
                .transpose()
                .map_err(at_step)?;
            let mut output = |index: usize| {
                match &group {
                    Some((first, values)) => {
                        values.get_element(index - first).map_err(|e| e.to_string())
                    }
                    None => self.harness.output(index),
                }
                .map_err(at_step)
            };
            match &step.action {
                TestAction::Evaluate(index) => {
                    output(*index)?;
                }
                TestAction::Drive { port, value } => {
                    let value = output(*value)?;
                    self.harness.drive(*port, value).map_err(at_step)?;
                }
                TestAction::Tick(reset) => {
                    let asserted = reset
                        .map(output)
                        .transpose()?
                        .is_some_and(|v| v.bits_equals_u64_value(1));
                    let Harness::Block(sim) = &mut self.harness else {
                        return Err(at_step("tick! requires a block test".into()));
                    };
                    sim.drive("rst", &IrValue::bool(asserted ^ self.active_low))
                        .map_err(at_step)?;
                    sim.tick().map_err(at_step)?;
                    if asserted {
                        sim.drive("rst", &IrValue::bool(self.active_low))
                            .map_err(at_step)?;
                    }
                    self.tick += 1;
                }
                TestAction::Assert { predicate, label } => {
                    if !output(*predicate)?.bits_equals_u64_value(1) {
                        let message = label.as_ref().map_or_else(
                            || "assertion failed".to_owned(),
                            |label| format!("assertion failed: {label}"),
                        );
                        return Err(at_step(message));
                    }
                }
                TestAction::AssertEq { actual, expected } => {
                    let actual = output(*actual)?;
                    let expected = output(*expected)?;
                    if actual != expected {
                        return Err(at_step(format!(
                            "assert_eq failed: actual {actual}, expected {expected}"
                        )));
                    }
                }
                TestAction::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let condition = output(*condition)?;
                    self.execute(if condition.bits_equals_u64_value(1) {
                        then_body
                    } else {
                        else_body
                    })?;
                }
                TestAction::For {
                    port,
                    iterable,
                    count,
                    body,
                } => {
                    let array = match iterable {
                        TestIterable::Array(index) => Some(output(*index)?),
                        TestIterable::Range(_) => None,
                    };
                    for ordinal in 0..*count {
                        let at_iteration = |message| Failure {
                            position: step.position,
                            tick: self.tick,
                            message,
                        };
                        let element = match iterable {
                            TestIterable::Array(_) => array
                                .as_ref()
                                .expect("array at loop entry")
                                .get_element(ordinal)
                                .map_err(|e| at_iteration(e.to_string()))?,
                            TestIterable::Range(start) => {
                                let offset =
                                    IrBits::make_ubits(start.get_bit_count(), ordinal as u64)
                                        .map_err(|e| at_iteration(e.to_string()))?;
                                IrValue::Bits(start.add(&offset))
                            }
                        };
                        self.harness.drive(*port, element).map_err(at_iteration)?;
                        self.execute(body)?;
                    }
                }
            }
        }
        Ok(())
    }
}
