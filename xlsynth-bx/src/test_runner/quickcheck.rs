// SPDX-License-Identifier: Apache-2.0

//! Seeded concrete execution of clocked BX properties.

use std::collections::{BTreeMap, HashMap, hash_map::Entry};
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use xlsynth_pir::block_inline::flatten_block_hierarchy;
use xlsynth_pir::block_transition::{BlockTransition, lower_block_transition};
use xlsynth_pir::ir::{NodePayload, NodeRef, Package, Type as IrType};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_value_utils::zero_ir_value_for_type;
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;
use xlsynth_pir::{IrBits, IrValue};

use super::simulation::{Observed, Simulation};
use super::{
    AssertionTiming, BxTestResult, ClockedHarness, Execution, ExecutionError, Failure,
    FourValuedReport, Harness, QuickCheckOptions, QuickCheckReport, TestExpression, TestProgram,
    check_ir_assertions, input_name,
};
use crate::ast::{Param, Type};
use crate::typed::CheckedModule;

/// Legal QuickCheck stimulus values, including enum members inside aggregates.
pub(super) enum Stimulus {
    Uniform(IrType),
    Enum(Vec<IrBits>),
    Tuple(Vec<Stimulus>),
    Array {
        size: usize,
        element_type: IrType,
        element: Box<Stimulus>,
    },
}

impl Stimulus {
    /// Resolve BX stimulus domains against the flattened harness input layout.
    pub(super) fn for_params(
        params: &[Param],
        transition: &BlockTransition,
        checked: &CheckedModule,
    ) -> Result<Vec<Self>, String> {
        let inputs: HashMap<_, _> = transition
            .inputs
            .iter()
            .map(|p| (p.name.as_str(), &p.ty))
            .collect();
        params
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let ty = inputs
                    .get(input_name(i).as_str())
                    .ok_or_else(|| format!("missing input for parameter '{}'", param.name))?;
                Self::new(&param.ty, ty, checked)
            })
            .collect()
    }

    pub(super) fn new(ty: &Type, ir: &IrType, module: &CheckedModule) -> Result<Self, String> {
        let tuple = |types: Vec<&Type>| -> Result<Self, String> {
            let IrType::Tuple(fields) = ir else {
                return Err(format!("expected tuple layout for {ty}"));
            };
            if fields.len() != types.len() {
                return Err(format!("invalid tuple layout for {ty}"));
            }
            Ok(Self::Tuple(
                types
                    .into_iter()
                    .zip(fields)
                    .map(|(t, f)| Self::new(t, f, module))
                    .collect::<Result<_, _>>()?,
            ))
        };
        match ty {
            Type::Bits { .. } | Type::Bool | Type::Array { size: 0, .. } => {
                Ok(Self::Uniform(ir.clone()))
            }
            Type::Enum { name, .. } => {
                let definition = module
                    .enums()
                    .get(name)
                    .ok_or_else(|| format!("missing enum '{name}'"))?;
                if definition.members.is_empty() {
                    return Err(format!("cannot generate a value of empty enum '{name}'"));
                }
                Ok(Self::Enum(definition.members.values().cloned().collect()))
            }
            Type::Tuple(fields) => tuple(fields.iter().collect()),
            Type::Named(name) => {
                let definition = module
                    .structs()
                    .iter()
                    .find(|s| &s.name == name)
                    .ok_or_else(|| format!("missing struct '{name}'"))?;
                tuple(definition.fields.iter().map(|f| &f.ty).collect())
            }
            Type::Array { element, size } => {
                let IrType::Array(layout) = ir else {
                    return Err(format!("expected array layout for {ty}"));
                };
                Ok(Self::Array {
                    size: *size,
                    element_type: (*layout.element_type).clone(),
                    element: Box::new(Self::new(element, &layout.element_type, module)?),
                })
            }
            _ => Err(format!("unresolved QuickCheck parameter type '{ty}'")),
        }
    }

    pub(super) fn draw(&self, rng: &mut Pcg64Mcg) -> Result<IrValue, String> {
        match self {
            Self::Uniform(ty) => Ok(generate_uniform_value_with_rng(rng, ty)),
            Self::Enum(values) => Ok(IrValue::Bits(
                values[rng.gen_range(0..values.len())].clone(),
            )),
            Self::Tuple(fields) => Ok(IrValue::make_tuple(
                &fields
                    .iter()
                    .map(|f| f.draw(rng))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Self::Array {
                size,
                element_type,
                element,
            } => {
                let values = (0..*size)
                    .map(|_| element.draw(rng))
                    .collect::<Result<Vec<_>, _>>()?;
                IrValue::make_array_typed(element_type.clone(), &values).map_err(|e| e.to_string())
            }
        }
    }
}

/// A prepared PIR transition evaluated with persistent drives and simultaneous
/// register updates.
pub(super) struct PirCircuit<'a> {
    package: &'a Package,
    transition: BlockTransition,
    inputs: BTreeMap<String, usize>,
    args: Vec<IrValue>,
    settled: Option<FnEvalResult>,
    expressions: HashMap<NodeRef, TestExpression>,
}

impl<'a> PirCircuit<'a> {
    fn new(package: &'a Package, top: &str) -> Result<Self, String> {
        let transition = lower_block_transition(package, top)?;
        Ok(Self::from_transition(package, transition))
    }

    pub(super) fn from_transition(package: &'a Package, transition: BlockTransition) -> Self {
        let inputs = transition
            .inputs
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();
        let args = transition
            .inputs
            .iter()
            .map(|p| zero_ir_value_for_type(&p.ty))
            .collect();
        Self {
            package,
            transition,
            inputs,
            args,
            settled: None,
            expressions: HashMap::new(),
        }
    }

    pub(super) fn reset_trial(&mut self, state: &BTreeMap<String, IrValue>) {
        self.args.clear();
        self.args.extend(
            self.transition
                .inputs
                .iter()
                .map(|p| zero_ir_value_for_type(&p.ty)),
        );
        if let Some(reset) = &self.transition.reset {
            self.args[self.inputs[&reset.name]] = IrValue::bool(reset.active_low);
        }
        self.args.extend(
            self.transition
                .registers
                .iter()
                .map(|r| state[&r.name].clone()),
        );
        self.settled = None;
    }

    fn settle(&mut self) -> &FnEvalResult {
        self.settled.get_or_insert_with(|| {
            eval_fn_in_package(self.package, &self.transition.function, &self.args)
        })
    }

    /// The circuit also computes expressions for later or skipped test
    /// statements. If an assumption failed, check the dependencies of the
    /// sampled port or register update before attributing that failure to the
    /// current statement.
    fn sample(&mut self, index: usize) -> Result<IrValue, String> {
        let value = match self.settle() {
            FnEvalResult::Success(result) => Some(&result.value),
            FnEvalResult::Failure(failure) if failure.assumption_failures.is_empty() => {
                Some(&failure.value)
            }
            FnEvalResult::Failure(_) => None,
        };
        if let Some(value) = value {
            return value.get_element(index).map_err(|e| e.to_string());
        }
        let function = &self.transition.function;
        let ret = function.ret_node_ref.expect("transition result");
        let NodePayload::Tuple(fields) = &function.get_node(ret).payload else {
            return Err("expected transition result tuple".into());
        };
        let node = fields[index];
        let expr = match self.expressions.entry(node) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(TestExpression::extract(
                function,
                node,
                AssertionTiming::ClockEdge,
            )?),
        };
        expr.evaluate(self.package, &self.args)
    }

    fn drive(&mut self, name: &str, value: &IrValue) -> Result<(), String> {
        let index = *self
            .inputs
            .get(name)
            .ok_or_else(|| format!("unknown test input '{name}'"))?;
        if value.type_() != self.transition.inputs[index].ty {
            return Err(format!("invalid value for test input '{name}'"));
        }
        if &self.args[index] != value {
            self.args[index] = value.clone();
            self.settled = None;
        }
        Ok(())
    }

    fn tick(&mut self) -> Result<(), String> {
        let first = self.transition.outputs.len();
        let count = self.transition.registers.len();
        // The emitted RTL clocks all assertions in the flattened block and
        // disables them while its physical reset is asserted.
        let in_reset = self.transition.reset.as_ref().is_some_and(|reset| {
            self.args[self.inputs[&reset.name]].bits_equals_u64_value(u64::from(!reset.active_low))
        });
        if !in_reset && let FnEvalResult::Failure(failure) = self.settle() {
            check_ir_assertions(failure)?;
        }
        let next = (first..first + count)
            .map(|i| self.sample(i))
            .collect::<Result<Vec<_>, _>>()?;
        self.args.truncate(self.transition.inputs.len());
        self.args.extend(next);
        self.settled = None;
        Ok(())
    }
}

/// Drive the two concrete engines in lockstep and compare each sampled
/// expression.
pub(super) struct Concrete<'a, 'p> {
    pub(super) pir: &'a mut PirCircuit<'p>,
    pub(super) rtl: Option<&'a mut Simulation>,
}

impl ClockedHarness for Concrete<'_, '_> {
    fn drive(&mut self, name: &str, value: &IrValue) -> Result<(), String> {
        self.pir.drive(name, value)?;
        if let Some(rtl) = &mut self.rtl {
            rtl.drive(name, value)?;
        }
        Ok(())
    }
    fn tick(&mut self) -> Result<(), String> {
        self.pir.tick()?;
        if let Some(rtl) = &mut self.rtl {
            rtl.tick()?;
        }
        Ok(())
    }
    fn sample(&mut self, index: usize) -> Result<Observed, String> {
        let value = self.pir.sample(index)?;
        if let Some(rtl) = &mut self.rtl {
            match rtl.output_observed(index)? {
                Observed::Known(other) if value != other => {
                    return Err(format!(
                        "Vastly cross-check mismatch: PIR {value}, Vastly {other}"
                    ));
                }
                Observed::Unknown(reason) => {
                    return Err(format!(
                        "Vastly cross-check mismatch: PIR {value}, Vastly {reason}"
                    ));
                }
                Observed::Known(_) => { /* Both concrete samples agree. */ }
            }
        }
        Ok(Observed::Known(value))
    }
}

/// Source stimulus assignments and active edges in execution order.
#[derive(Default)]
pub(super) struct TrialTrace(Vec<TraceEvent>);

enum TraceEvent {
    Drive(usize, IrValue),
    Tick(bool),
}

impl TrialTrace {
    pub(super) fn drive(&mut self, port: usize, value: &IrValue) {
        self.0.push(TraceEvent::Drive(port, value.clone()));
    }
    pub(super) fn tick(&mut self, reset: bool) {
        self.0.push(TraceEvent::Tick(reset));
    }
    fn describe(&self, names: &BTreeMap<usize, String>) -> String {
        self.0
            .iter()
            .map(|event| match event {
                TraceEvent::Drive(port, value) => format!("{} := {value}", names[port]),
                TraceEvent::Tick(true) => "tick!(rst: true)".to_owned(),
                TraceEvent::Tick(false) => "tick!()".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// One reproducible input sequence and starting DUT state.
struct Trial<'a> {
    program: &'a TestProgram,
    params: &'a [Param],
    args: Vec<IrValue>,
    registers: BTreeMap<String, IrValue>,
    active_low: bool,
}

impl Trial<'_> {
    fn execute(
        &self,
        circuit: &mut dyn ClockedHarness,
        trace: Option<&mut TrialTrace>,
    ) -> Result<(), Failure> {
        for (i, value) in self.args.iter().enumerate() {
            circuit
                .drive(&input_name(i), value)
                .map_err(|e| self.program.at_declaration(e))?;
        }
        Execution {
            harness: Harness::Block(circuit),
            tick: 0,
            active_low: self.active_low,
            trace: trace.map(|t| (&self.program.stimulus_names, t)),
        }
        .execute(&self.program.steps)
    }

    fn execute_concrete(
        &self,
        pir: &mut PirCircuit,
        rtl: Option<&mut Simulation>,
        trace: Option<&mut TrialTrace>,
    ) -> Result<(), Failure> {
        pir.reset_trial(&self.registers);
        let mut concrete = Concrete { pir, rtl };
        if let Some(sim) = &mut concrete.rtl {
            sim.reset_trial(Some(&self.registers))
                .map_err(|e| self.program.at_declaration(e))?;
        }
        self.execute(&mut concrete, trace)
    }

    fn execute_from_x(
        &self,
        sim: &mut Simulation,
        trace: Option<&mut TrialTrace>,
    ) -> Result<(), Failure> {
        sim.reset_trial(None)
            .map_err(|e| self.program.at_declaration(e))?;
        self.execute(sim, trace)
    }

    /// Replay only the first reportable failure, so passing trials need no
    /// trace storage.
    fn witness(
        &self,
        path: &Path,
        origin: (u64, usize),
        failure: &Failure,
        from_x: bool,
        replay: impl FnOnce(&mut TrialTrace) -> Result<(), Failure>,
    ) -> String {
        let mut trace = TrialTrace::default();
        let replayed = replay(&mut trace);
        let mut message = format!(
            "{}\n  seed {}, trial {}\n{}",
            failure.diagnostic(path, self.program),
            origin.0,
            origin.1,
            self.describe_stimulus(from_x, &trace)
        );
        if replayed.as_ref().err() != Some(failure) {
            message.push_str("\n  failure changed during trace replay");
        }
        message
    }

    /// Format the driven values and edges shared by random and solver
    /// witnesses.
    fn describe_stimulus(&self, from_x: bool, trace: &TrialTrace) -> String {
        let inputs = self
            .params
            .iter()
            .zip(&self.args)
            .map(|(p, v)| format!("{}={v}", p.name))
            .collect::<Vec<_>>()
            .join(", ");
        let initial = if from_x {
            "initial registers: X".to_owned()
        } else {
            let registers = self
                .registers
                .iter()
                .map(|(n, v)| format!("{n}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("concrete initial registers: [{registers}]")
        };
        format!(
            "  inputs: [{inputs}]\n  {initial}\n  stimulus trace: [{}]",
            trace.describe(&self.program.stimulus_names)
        )
    }

    /// Accumulate the all-X outcomes independently of the concrete property.
    fn observe_from_x(
        &self,
        sim: &mut Simulation,
        path: &Path,
        origin: (u64, usize),
        report: &mut FourValuedReport,
    ) -> Option<String> {
        let Err(failure) = self.execute_from_x(sim, None) else {
            report.passed += 1;
            return None;
        };
        report.record_failure(&failure.error, || {
            self.witness(path, origin, &failure, true, |trace| {
                self.execute_from_x(sim, Some(trace))
            })
        })
    }
}

/// Replay a model with the same transitions and source actions as a concrete
/// trial.
pub(super) fn replay_counterexample(
    program: &TestProgram,
    domains: &[Stimulus],
    package: &Package,
    path: &Path,
    active_low: bool,
    inputs: &[IrValue],
) -> Result<String, String> {
    let params = program
        .quickcheck_params
        .as_deref()
        .ok_or("expected a block QuickCheck")?;
    let mut pir = PirCircuit::new(package, &program.harness)?;
    if inputs.len() != params.len() + pir.transition.registers.len() {
        return Err(format!(
            "invalid counterexample arity for '{}'",
            program.name
        ));
    }
    let (args, state) = inputs.split_at(params.len());
    for (i, arg) in args.iter().enumerate() {
        let port = pir
            .inputs
            .get(&input_name(i))
            .ok_or("missing QuickCheck parameter port")?;
        if arg.type_() != pir.transition.inputs[*port].ty {
            return Err(format!(
                "invalid counterexample type for parameter '{}'",
                params[i].name
            ));
        }
        if !domains[i].admits(arg) {
            return Err(format!(
                "counterexample is outside the declared stimulus domain for '{}'",
                params[i].name
            ));
        }
    }
    let registers = pir
        .transition
        .registers
        .iter()
        .zip(state)
        .map(|(reg, value)| {
            if value.type_() != reg.ty {
                return Err(format!(
                    "invalid counterexample type for register '{}'",
                    reg.name
                ));
            }
            Ok((reg.name.clone(), value.clone()))
        })
        .collect::<Result<_, _>>()?;
    let trial = Trial {
        program,
        params,
        args: args.to_vec(),
        registers,
        active_low,
    };
    let mut trace = TrialTrace::default();
    match trial.execute_concrete(&mut pir, None, Some(&mut trace)) {
        Err(failure) if matches!(failure.error, ExecutionError::Assertion(_)) => Ok(format!(
            "{}\n{}",
            failure.diagnostic(path, program),
            trial.describe_stimulus(false, &trace)
        )),
        Err(failure) => Err(failure.diagnostic(path, program)),
        Ok(()) => Err(format!(
            "solver counterexample passed concrete replay of '{}'",
            program.name
        )),
    }
}

/// Generate trials and execute each property against its prepared transition.
pub(super) fn run(
    program: &TestProgram,
    params: &[Param],
    package: &Package,
    checked: &CheckedModule,
    path: &Path,
    active_low: bool,
    options: QuickCheckOptions,
) -> BxTestResult {
    let mut report = QuickCheckReport {
        seed: options.seed,
        trials: 0,
        four_valued: options.cross_check_vastly.then(FourValuedReport::default),
    };
    let result = (|| -> Result<Option<String>, String> {
        let package = flatten_block_hierarchy(package, &program.harness)?;
        let mut pir = PirCircuit::new(&package, &program.harness)?;
        let mut rtl = options
            .cross_check_vastly
            .then(|| Simulation::from_flattened(&package, &program.harness))
            .transpose()?;
        let stimuli = Stimulus::for_params(params, &pir.transition, checked)?;
        let mut rng = Pcg64Mcg::seed_from_u64(options.seed);
        for number in 1..=options.trials.get() {
            let args = stimuli
                .iter()
                .map(|s| s.draw(&mut rng))
                .collect::<Result<Vec<_>, _>>()?;
            let registers = pir
                .transition
                .registers
                .iter()
                .map(|r| {
                    (
                        r.name.clone(),
                        generate_uniform_value_with_rng(&mut rng, &r.ty),
                    )
                })
                .collect();
            let trial = Trial {
                program,
                params,
                args,
                registers,
                active_low,
            };
            report.trials = number;
            let concrete = trial.execute_concrete(&mut pir, rtl.as_mut(), None);
            let x_error = match (&mut rtl, &mut report.four_valued) {
                (Some(sim), Some(four)) => {
                    trial.observe_from_x(sim, path, (options.seed, number), four)
                }
                _ => None,
            };
            if let Err(failure) = concrete {
                let mut message =
                    trial.witness(path, (options.seed, number), &failure, false, |trace| {
                        trial.execute_concrete(&mut pir, rtl.as_mut(), Some(trace))
                    });
                if let Some(error) = x_error {
                    message.push_str(&format!("\n  four-valued execution error: {error}"));
                }
                return Ok(Some(message));
            }
            if x_error.is_some() {
                return Ok(x_error);
            }
        }
        Ok(None)
    })();
    let failure =
        result.unwrap_or_else(|e| Some(program.at_declaration(e).diagnostic(path, program)));
    BxTestResult {
        name: program.name.clone(),
        failure,
        quickcheck: Some(report),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImportOptions, ResetBehavior, compile, empty_package, run_tests_with_quickcheck};
    use xlsynth_pir::ir_parser::parse_xls_ir_package;

    #[test]
    fn linked_block_assertions_sample_only_on_unmasked_rising_edges() -> Result<(), String> {
        let ir = r#"package linked
block guarded(clk: clock, rst: bits[1], good: bits[1], out: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  rst: bits[1] = input_port(name=rst, id=1)
  good: bits[1] = input_port(name=good, id=2)
  t: token = after_all(id=3)
  checked: token = assert(t, good, message="bad value", label="value_ok", id=4)
  out: () = output_port(good, name=out, id=5)
}
block resetless(clk: clock, good: bits[1], out: bits[1]) {
  good: bits[1] = input_port(name=good, id=6)
  t: token = after_all(id=7)
  checked: token = assert(t, good, message="resetless value", label="resetless_ok", id=8)
  out: () = output_port(good, name=out, id=9)
}"#;
        let source = r#"extern block guarded(clk, rst, good: bool) -> bool;
extern block resetless(clk, good: bool) -> bool;
#[quickcheck] block between_edges() -> bool {
    reg good: bool;
    let value = guarded(good);
    assert_eq(value, false);
    good := true;
    tick!();
    good := false;
    !value
}
#[quickcheck] block during_reset() -> bool {
    let a = guarded(false);
    let b = resetless(false);
    tick!(rst: true);
    !(a || b)
}
#[quickcheck] block after_reset() -> bool {
    let value = guarded(false);
    tick!(rst: true);
    tick!();
    value
}"#;
        for active_low in [false, true] {
            let ir = ir.replace("active_low=false", &format!("active_low={active_low}"));
            let package = parse_xls_ir_package(&ir).map_err(|e| e.to_string())?;
            let report = run_tests_with_quickcheck(
                source,
                Path::new("edges.bx"),
                package,
                ResetBehavior {
                    asynchronous: false,
                    active_low,
                },
                &ImportOptions::default(),
                true,
                QuickCheckOptions {
                    trials: std::num::NonZeroUsize::MIN,
                    ..Default::default()
                },
            )?;
            assert_eq!(report.tests[0].failure, None);
            assert_eq!(report.tests[1].failure, None);
            assert_eq!(
                report.tests[2]
                    .failure
                    .as_deref()
                    .and_then(|f| f.lines().next()),
                Some(
                    "edges.bx:21:5: test 'after_reset' at tick 1: IR assertion 'value_ok' failed: bad value"
                )
            );
        }
        Ok(())
    }

    #[test]
    fn clocked_samples_scope_bounds_even_when_linked_calls_also_assert() -> Result<(), String> {
        let package = parse_xls_ir_package(
            r#"package linked
fn lookup(xs: bits[8][2] id=1, i: bits[8] id=2) -> bits[8] {
  t: token = after_all(id=3)
  bad: bits[1] = literal(value=0, id=4)
  checked: token = assert(t, bad, message="bad call", label="called", id=5)
  ret selected: bits[8] = array_index(xs, indices=[i], assumed_in_bounds=true, id=6)
}
block select_word(clk: clock, i: bits[8], out: bits[8]) {
  i: bits[8] = input_port(name=i, id=7)
  xs: bits[8][2] = literal(value=[17, 29], id=8)
  value: bits[8] = invoke(xs, i, to_apply=lookup, id=9)
  out: () = output_port(value, name=out, id=10)
}"#,
        )
        .map_err(|e| e.to_string())?;
        let source = r#"extern block select_word(clk, i: u8) -> u8;
block store(clk, i: u8) -> u8 { let d = select_word(i); reg q: u8; q := d; q }
#[quickcheck] block valid_sample() -> bool {
    let good = select_word(u8:0);
    let bad = select_word(u8:2);
    if false { assert_eq(bad, u8:29); }
    good == u8:17
}
#[quickcheck] block bad_sample() -> bool {
    let bad = select_word(u8:2);
    bad == u8:29
}
#[quickcheck] block bad_update() -> bool {
    let q = store(u8:2);
    tick!(rst: true);
    true
}"#;
        let report = run_tests_with_quickcheck(
            source,
            Path::new("contracts.bx"),
            package,
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
            QuickCheckOptions {
                trials: std::num::NonZeroUsize::MIN,
                ..Default::default()
            },
        )?;
        let diagnostics: Vec<_> = report
            .tests
            .iter()
            .map(|test| {
                test.failure
                    .as_deref()
                    .and_then(|failure| failure.lines().next())
            })
            .collect();
        assert_eq!(
            diagnostics,
            [
                None,
                Some(
                    "contracts.bx:11:5: test 'bad_sample' at tick 0: IR array_index at node 6 was out of bounds despite assumed_in_bounds=true"
                ),
                Some(
                    "contracts.bx:15:5: test 'bad_update' at tick 0: IR array_index at node 6 was out of bounds despite assumed_in_bounds=true"
                ),
            ]
        );
        Ok(())
    }

    #[test]
    fn only_the_observed_expression_or_register_update_checks_its_bounds() -> Result<(), String> {
        let package = parse_xls_ir_package(
            r#"package linked
fn lookup(a: bits[8][2] id=1, i: bits[8] id=2) -> bits[8] {
  ret selected: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=3)
}"#,
        )
        .map_err(|e| e.to_string())?;
        let source = r#"extern fn lookup(a: u8[2], i: u8) -> u8;
const DATA: u8[2] = [u8:17, u8:29];
enum Empty : u2 {}
block read(clk, idx: u8) -> u8 {
    reg q: u8;
    q := lookup(DATA, idx);
    q
}
#[quickcheck] block skipped(xs: Empty[0]) -> bool {
    if false { assert_eq(lookup(DATA, u8:2), u8:29); }
    xs == xs
}
#[quickcheck] block before_drive() -> bool {
    reg idx: u8 = u8:2;
    tick!();
    idx := u8:0;
    lookup(DATA, idx) == u8:17
}
#[quickcheck] block reached() -> bool {
    lookup(DATA, u8:2) == u8:29
}
#[quickcheck] block write_contract() -> bool {
    let q = read(u8:2);
    tick!();
    true
}
#[quickcheck] block write_on_reset() -> bool {
    reg idx: u8;
    let q = read(idx);
    tick!();
    idx := u8:2;
    tick!(rst: true);
    true
}"#;
        let options = QuickCheckOptions {
            trials: std::num::NonZeroUsize::MIN,
            ..Default::default()
        };
        let report = run_tests_with_quickcheck(
            source,
            Path::new("bounds.bx"),
            package,
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
            options,
        )?;
        assert_eq!(report.tests[0].failure, None);
        assert_eq!(report.tests[1].failure, None);
        let traces = [
            "  stimulus trace: []",
            "  stimulus trace: [tick!()]",
            "  stimulus trace: [idx := bits[8]:0, tick!(), idx := bits[8]:2, tick!(rst: true)]",
        ];
        assert_eq!(report.tests.len(), 2 + traces.len());
        for (test, trace) in report.tests[2..].iter().zip(traces) {
            let failure = test
                .failure
                .as_deref()
                .expect("reached out-of-bounds access");
            assert!(
                failure.contains("despite assumed_in_bounds=true"),
                "{failure}"
            );
            assert_eq!(failure.lines().last(), Some(trace));
        }
        Ok(())
    }

    #[test]
    fn concrete_cross_check_detects_a_different_pre_edge_sample() -> Result<(), String> {
        let package = compile(
            "pub block sample(clk, d: u8) -> u8 { reg q: u8; q := d; q }",
            empty_package("sample"),
            ResetBehavior::default(),
        )?;
        let flat = flatten_block_hierarchy(&package, "sample")?;
        let mut pir = PirCircuit::new(&flat, "sample")?;
        let mut rtl = Simulation::from_flattened(&flat, "sample")?;
        let zero = IrValue::make_ubits(8, 0).map_err(|e| e.to_string())?;
        let one = IrValue::make_ubits(8, 1).map_err(|e| e.to_string())?;
        pir.reset_trial(&BTreeMap::from([("q".to_owned(), zero)]));
        rtl.reset_trial(Some(&BTreeMap::from([("q".to_owned(), one)])))?;
        let mut circuit = Concrete {
            pir: &mut pir,
            rtl: Some(&mut rtl),
        };
        assert_eq!(
            circuit.sample(0).err().as_deref(),
            Some("Vastly cross-check mismatch: PIR bits[8]:0, Vastly bits[8]:1")
        );
        circuit.tick()?;
        assert!(
            matches!(circuit.sample(0)?, Observed::Known(value) if value.bits_equals_u64_value(0))
        );
        Ok(())
    }
}
