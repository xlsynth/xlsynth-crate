// SPDX-License-Identifier: Apache-2.0

//! Execute test control flow over typed BX expressions and clocked hardware.

pub(crate) mod invariant;
mod invariant_trials;
#[cfg(test)]
mod mixed_suite_tests;
mod proof;
mod proof_domain;
mod prove;
mod quickcheck;
mod simulation;

pub use prove::{BxProofCase, BxProofSuite, BxQuickCheckProof, prepare_quickcheck_proofs};

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::path::Path;

use xlsynth_pir::ir::{Fn, NodePayload, NodeRef, Package, PackageMember};
use xlsynth_pir::ir_eval::{
    AssumptionFailureKind, FnEvalFailure, FnEvalResult, eval_fn_in_package,
};
use xlsynth_pir::ir_fn_cone_extract::{SinkSelector, extract_fn_cone_to_params};
use xlsynth_pir::ir_value_utils::zero_ir_value_for_type;
use xlsynth_pir::{IrBits, IrValue};

use crate::ast::{SourcePosition, TestKind};
use crate::{
    BxWarning, CheckedModule, ImportOptions, ResetBehavior, bx_import, check, lower_module,
};
use simulation::{Observed, Simulation};

/// Concrete trials per clocked property, reproducible from the same seed.
#[derive(Clone, Copy, Debug)]
pub struct QuickCheckOptions {
    pub trials: NonZeroUsize,
    pub seed: u64,
    pub cross_check_vastly: bool,
}

impl Default for QuickCheckOptions {
    fn default() -> Self {
        Self {
            trials: NonZeroUsize::new(100).unwrap(),
            seed: 0,
            cross_check_vastly: false,
        }
    }
}

/// Concrete test settings, including how many cycles to sample in each
/// invariant trial.
#[derive(Clone, Copy, Debug)]
pub struct BxTestOptions {
    pub quickcheck: QuickCheckOptions,
    pub invariant_cycles: NonZeroUsize,
}

impl Default for BxTestOptions {
    fn default() -> Self {
        Self {
            quickcheck: QuickCheckOptions::default(),
            invariant_cycles: NonZeroUsize::new(16).unwrap(),
        }
    }
}

/// Outcomes from starting the RTL at X, independent of the concrete trials.
#[derive(Clone, Debug, Default)]
pub struct FourValuedReport {
    pub passed: usize,
    pub failed: usize,
    pub indeterminate: usize,
    pub first_false: Option<String>,
    pub first_indeterminate: Option<String>,
}

impl FourValuedReport {
    /// Retain the first witness for each four-state outcome. A simulator error
    /// prevents classification and must be reported by the enclosing test.
    fn record_failure(
        &mut self,
        error: &ExecutionError,
        witness: impl FnOnce() -> String,
    ) -> Option<String> {
        let first = match error {
            ExecutionError::Assertion(_) => {
                self.failed += 1;
                &mut self.first_false
            }
            ExecutionError::Unknown(_) => {
                self.indeterminate += 1;
                &mut self.first_indeterminate
            }
            ExecutionError::Runtime(_) => return Some(witness()),
        };
        first.get_or_insert_with(witness);
        None
    }
}

/// Trial count and seed for one property, with optional four-valued outcomes.
#[derive(Clone, Debug)]
pub struct QuickCheckReport {
    pub seed: u64,
    pub trials: usize,
    pub four_valued: Option<FourValuedReport>,
}

/// Outcome of a single test, in source declaration order.
#[derive(Clone, Debug)]
pub struct BxTestResult {
    pub name: String,
    pub failure: Option<String>,
    pub quickcheck: Option<QuickCheckReport>,
}

/// Results of running the tests declared in the requested BX file.
#[derive(Clone, Debug)]
pub struct BxTestReport {
    pub tests: Vec<BxTestResult>,
    pub warnings: Vec<BxWarning>,
}

/// Concrete test results and the names of persistent clocked invariants in this
/// run.
#[derive(Clone, Debug)]
pub struct BxTestRun {
    pub tests: Vec<BxTestResult>,
    pub warnings: Vec<BxWarning>,
    pub invariant_names: BTreeSet<String>,
}

/// Test-only IR names, source locations and sequenced instructions.
pub(crate) struct TestSuite {
    reserved: BTreeSet<String>,
    pub(crate) programs: Vec<TestProgram>,
    pub(crate) invariants: Vec<InvariantProgram>,
}

/// Checked input domains and named observations of a persistent invariant
/// harness.
#[derive(Clone)]
pub(crate) struct InvariantProgram {
    pub(crate) name: String,
    pub(crate) position: SourcePosition,
    pub(crate) harness: String,
    pub(crate) params: Vec<crate::ast::Param>,
    pub(crate) lemmas: Vec<crate::ast::InvariantLemma>,
}

impl TestSuite {
    /// A separate hardware lowering is needed only when a clocked harness
    /// reaches a helper whose discarded fields are observed by a pure test.
    /// The pure harnesses themselves can then be omitted from that lowering.
    fn pure_harnesses_for_clocked_lowering(
        &self,
        checked: &crate::CheckedModule,
    ) -> Option<BTreeSet<&str>> {
        let mut clocked = self
            .programs
            .iter()
            .filter(|p| p.kind == TestKind::Block)
            .map(|p| p.harness.clone())
            .chain(self.invariants.iter().map(|p| p.harness.clone()))
            .peekable();
        if checked.functions_with_test_discards.is_empty() || clocked.peek().is_none() {
            return None;
        }
        let pure_harnesses: BTreeSet<_> = self
            .programs
            .iter()
            .filter(|p| p.kind == TestKind::Function)
            .map(|p| p.harness.as_str())
            .collect();
        if checked
            .functions_with_test_discards
            .iter()
            .all(|name| pure_harnesses.contains(name.as_str()))
        {
            return None;
        }
        checked
            .blocks_reach_test_discards(clocked)
            .then_some(pure_harnesses)
    }

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
    pub(crate) quickcheck_params: Option<Vec<crate::ast::Param>>,
    pub(crate) stimulus_names: BTreeMap<usize, String>,
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
    run_tests_with_quickcheck(
        source,
        path,
        package,
        reset,
        options,
        warnings_as_errors,
        QuickCheckOptions::default(),
    )
}

/// Execute unit tests and concrete QuickCheck trials with the selected seed and
/// backends.
pub fn run_tests_with_quickcheck(
    source: &str,
    path: &Path,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    warnings_as_errors: bool,
    quickcheck_options: QuickCheckOptions,
) -> Result<BxTestReport, String> {
    let run = run_tests_with_options(
        source,
        path,
        package,
        reset,
        options,
        warnings_as_errors,
        BxTestOptions {
            quickcheck: quickcheck_options,
            ..Default::default()
        },
    )?;
    Ok(BxTestReport {
        tests: run.tests,
        warnings: run.warnings,
    })
}

/// Execute the root file's unit tests, procedural QuickChecks, and invariant
/// properties.
pub fn run_tests_with_options(
    source: &str,
    path: &Path,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    warnings_as_errors: bool,
    test_options: BxTestOptions,
) -> Result<BxTestRun, String> {
    let CompiledTests {
        suite,
        checked,
        package,
        clocked_package,
        warnings,
    } = compile_tests(source, path, package, reset, options, warnings_as_errors)?;
    let clocked = clocked_package.as_ref().unwrap_or(&package);
    let mut tests: Vec<_> = suite
        .programs
        .into_iter()
        .map(|program| {
            let package = match program.kind {
                TestKind::Function => &package,
                TestKind::Block => clocked,
            };
            if let Some(params) = &program.quickcheck_params {
                return (
                    program.position,
                    quickcheck::run(
                        &program,
                        params,
                        package,
                        &checked,
                        path,
                        reset.active_low,
                        test_options.quickcheck,
                    ),
                );
            }
            let result = run_program(&program, package, reset.active_low)
                .err()
                .map(|f| f.diagnostic(path, &program));
            (
                program.position,
                BxTestResult {
                    name: program.name,
                    failure: result,
                    quickcheck: None,
                },
            )
        })
        .collect();
    tests.extend(suite.invariants.iter().map(|program| {
        (
            program.position,
            invariant_trials::run(program, &checked, clocked, path, test_options),
        )
    }));
    tests.sort_by_key(|(position, _)| *position);
    Ok(BxTestRun {
        invariant_names: suite
            .invariants
            .into_iter()
            .map(|program| program.name)
            .collect(),
        tests: tests.into_iter().map(|(_, result)| result).collect(),
        warnings,
    })
}

/// Checked testbench actions and their combinational and clocked harnesses.
struct CompiledTests {
    suite: TestSuite,
    checked: CheckedModule,
    package: Package,
    clocked_package: Option<Package>,
    warnings: Vec<BxWarning>,
}

fn compile_tests(
    source: &str,
    path: &Path,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    warnings_as_errors: bool,
) -> Result<CompiledTests, String> {
    let mut suite = TestSuite {
        reserved: package
            .members
            .iter()
            .map(|m| m.graph().name.clone())
            .collect(),
        programs: Vec::new(),
        invariants: Vec::new(),
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
    let clocked_seed = suite
        .pure_harnesses_for_clocked_lowering(&checked)
        .map(|pure_harnesses| (package.clone(), pure_harnesses));
    let package = lower_module(&checked, package, reset).map_err(at_file)?;
    let warnings = std::mem::take(&mut checked.warnings);
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
    let clocked_package = if let Some((seed, pure_harnesses)) = clocked_seed {
        checked
            .functions
            .retain(|f| !pure_harnesses.contains(f.name.as_str()));
        checked.functions_with_test_discards.clear();
        Some(lower_module(&checked, seed, reset).map_err(at_file)?)
    } else {
        None
    };
    Ok(CompiledTests {
        suite,
        checked,
        package,
        clocked_package,
        warnings,
    })
}

/// Pure calls check assertions when evaluated. Clocked hardware checks them on
/// the active edge, including assertions inside linked function calls.
#[derive(Clone, Copy)]
enum AssertionTiming {
    Expression,
    ClockEdge,
}

/// Reports the first linked IR assertion triggered in one evaluation.
fn check_ir_assertions(failure: &FnEvalFailure) -> Result<(), String> {
    if let Some(assertion) = failure.assertion_failures.first() {
        return Err(format!(
            "IR assertion '{}' failed: {}",
            assertion.label, assertion.message
        ));
    }
    Ok(())
}

/// An expression's IR dependencies and the harness input ports they read.
struct TestExpression {
    function: Fn,
    ports: Vec<usize>,
    assertions: AssertionTiming,
    last: Option<(Vec<IrValue>, IrValue)>,
}

impl TestExpression {
    /// Extract the selected output once. The original package supplies its
    /// callees.
    fn extract(harness: &Fn, node: NodeRef, assertions: AssertionTiming) -> Result<Self, String> {
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
            assertions,
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
                if matches!(self.assertions, AssertionTiming::Expression) {
                    check_ir_assertions(&failure)?;
                }
                if let Some(assumption) = failure.assumption_failures.first() {
                    let operation = match assumption.kind {
                        AssumptionFailureKind::ArrayIndexOutOfBounds => "array_index",
                        AssumptionFailureKind::ArrayUpdateOutOfBounds => "array_update",
                    };
                    return Err(format!(
                        "IR {operation} at node {} was out of bounds despite assumed_in_bounds=true",
                        assumption.node_text_id
                    ));
                }
                failure.value
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
        expressions: HashMap<NodeRef, TestExpression>,
    },
    Block(&'a mut dyn ClockedHarness),
}

/// A clocked circuit driven and sampled by the procedural test interpreter.
trait ClockedHarness {
    fn drive(&mut self, name: &str, value: &IrValue) -> Result<(), String>;
    fn tick(&mut self) -> Result<(), String>;
    fn sample(&mut self, index: usize) -> Result<Observed, String>;
}

impl ClockedHarness for Simulation {
    fn drive(&mut self, name: &str, value: &IrValue) -> Result<(), String> {
        Simulation::drive(self, name, value)
    }
    fn tick(&mut self) -> Result<(), String> {
        Simulation::tick(self)
    }
    fn sample(&mut self, index: usize) -> Result<Observed, String> {
        self.output_observed(index)
    }
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

    fn output(&mut self, index: usize) -> Result<IrValue, ExecutionError> {
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
                    Entry::Vacant(entry) => entry.insert(TestExpression::extract(
                        function,
                        node,
                        AssertionTiming::Expression,
                    )?),
                };
                Ok(expression.evaluate(package, inputs)?)
            }
            Self::Block(sim) => match sim.sample(index)? {
                Observed::Known(value) => Ok(value),
                Observed::Unknown(message) => Err(ExecutionError::Unknown(message)),
            },
        }
    }
}

/// Runtime errors retain the innermost statement and edge count.
#[derive(PartialEq)]
struct Failure {
    position: SourcePosition,
    tick: u64,
    error: ExecutionError,
}

/// Distinguish property outcomes from simulator or linkage errors.
#[derive(PartialEq)]
enum ExecutionError {
    Runtime(String),
    Assertion(String),
    Unknown(String),
}

impl From<String> for ExecutionError {
    fn from(message: String) -> Self {
        Self::Runtime(message)
    }
}

impl TestProgram {
    fn at_declaration(&self, message: String) -> Failure {
        Failure {
            position: self.position,
            tick: 0,
            error: ExecutionError::Runtime(message),
        }
    }
}

impl Failure {
    fn diagnostic(&self, path: &Path, program: &TestProgram) -> String {
        let message = match &self.error {
            ExecutionError::Runtime(m)
            | ExecutionError::Assertion(m)
            | ExecutionError::Unknown(m) => m,
        };
        let edge = if program.kind == TestKind::Block {
            format!(" at tick {}", self.tick)
        } else {
            String::new()
        };
        format!(
            "{}:{}:{}: test '{}'{edge}: {message}",
            path.display(),
            self.position.line,
            self.position.column,
            program.name
        )
    }
}

/// Program counter state shared by nested loops and conditionals.
struct Execution<'a> {
    harness: Harness<'a>,
    tick: u64,
    active_low: bool,
    trace: Option<(&'a BTreeMap<usize, String>, &'a mut quickcheck::TrialTrace)>,
}

fn run_program(program: &TestProgram, package: &Package, active_low: bool) -> Result<(), Failure> {
    let at_declaration = |message| program.at_declaration(message);
    let mut simulation;
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
        TestKind::Block => {
            simulation = Simulation::new(package, &program.harness).map_err(at_declaration)?;
            Harness::Block(&mut simulation)
        }
    };
    Execution {
        harness,
        tick: 0,
        active_low,
        trace: None,
    }
    .execute(&program.steps)
}

impl Execution<'_> {
    /// Evaluate only the selected test statements, keeping input drives across
    /// edges.
    fn execute(&mut self, steps: &[TestStep]) -> Result<(), Failure> {
        for step in steps {
            let at_step = |error| Failure {
                position: step.position,
                tick: self.tick,
                error,
            };
            let group = step
                .group
                .map(|index| self.harness.output(index).map(|values| (index, values)))
                .transpose()
                .map_err(at_step)?;
            let mut output = |index: usize| {
                match &group {
                    Some((first, values)) => values
                        .get_element(index - first)
                        .map_err(|e| ExecutionError::Runtime(e.to_string())),
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
                    if let Some((names, trace)) = &mut self.trace
                        && names.contains_key(port)
                    {
                        trace.drive(*port, &value);
                    }
                    self.harness
                        .drive(*port, value)
                        .map_err(|e| at_step(e.into()))?;
                }
                TestAction::Tick(reset) => {
                    let asserted = reset
                        .map(output)
                        .transpose()?
                        .is_some_and(|v| v.bits_equals_u64_value(1));
                    let Harness::Block(sim) = &mut self.harness else {
                        return Err(at_step(ExecutionError::Runtime(
                            "tick! requires a block test".into(),
                        )));
                    };
                    sim.drive("rst", &IrValue::bool(asserted ^ self.active_low))
                        .map_err(|e| at_step(e.into()))?;
                    if let Some((_, trace)) = &mut self.trace {
                        trace.tick(asserted);
                    }
                    sim.tick().map_err(|e| at_step(e.into()))?;
                    if asserted {
                        sim.drive("rst", &IrValue::bool(self.active_low))
                            .map_err(|e| at_step(e.into()))?;
                    }
                    self.tick += 1;
                }
                TestAction::Assert { predicate, label } => {
                    if !output(*predicate)?.bits_equals_u64_value(1) {
                        let message = label.as_ref().map_or_else(
                            || "assertion failed".to_owned(),
                            |label| format!("assertion failed: {label}"),
                        );
                        return Err(at_step(ExecutionError::Assertion(message)));
                    }
                }
                TestAction::AssertEq { actual, expected } => {
                    let actual = output(*actual)?;
                    let expected = output(*expected)?;
                    if actual != expected {
                        return Err(at_step(ExecutionError::Assertion(format!(
                            "assert_eq failed: actual {actual}, expected {expected}"
                        ))));
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
                            error: ExecutionError::Runtime(message),
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
