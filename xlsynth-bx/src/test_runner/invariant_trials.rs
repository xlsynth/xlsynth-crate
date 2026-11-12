// SPDX-License-Identifier: Apache-2.0

//! Sample a persistent BX invariant on each cycle of a concrete trace.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::flatten_block_hierarchy;
use xlsynth_pir::ir::Package;
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;

use super::InvariantProgram;
use super::invariant::BxInvariantProof;
use super::quickcheck::{Concrete, PirCircuit};
use super::simulation::{Observed, Simulation};
use super::{
    BxTestOptions, BxTestResult, ClockedHarness, ExecutionError, FourValuedReport, QuickCheckReport,
};
use crate::CheckedModule;
use crate::ast::SourcePosition;

/// A validated trace over the same positional inputs as a bounded base query.
struct Trace<'a> {
    registers: BTreeMap<String, IrValue>,
    reset_inputs: Option<&'a [IrValue]>,
    samples: &'a [IrValue],
    cycles: usize,
    frame_width: usize,
}

/// An initialization or sampling failure with a source location and trace
/// prefix.
struct TrialFailure {
    position: SourcePosition,
    cycle: Option<usize>,
    error: ExecutionError,
}

impl BxInvariantProof {
    fn model_size(&self, cycles: usize) -> Result<usize, String> {
        let data = self.domains.len();
        let reset = usize::from(self.transition.reset.is_some());
        let total = cycles
            .checked_mul(data + reset)
            .and_then(|v| v.checked_add(data * reset))
            .and_then(|v| v.checked_add(self.transition.registers.len()));
        total.ok_or_else(|| "invariant trace length overflow".into())
    }

    /// Generate a power-up state, initial reset drives and fresh inputs at
    /// every sample.
    fn draw_model(&self, cycles: usize, rng: &mut Pcg64Mcg) -> Result<Vec<IrValue>, String> {
        let mut model = Vec::new();
        model
            .try_reserve_exact(self.model_size(cycles)?)
            .map_err(|e| e.to_string())?;
        model.extend(
            self.transition
                .registers
                .iter()
                .map(|r| generate_uniform_value_with_rng(rng, &r.ty)),
        );
        let reset = &self.transition.reset;
        if reset.is_some() {
            for domain in &self.domains {
                model.push(domain.draw(rng)?);
            }
        }
        for _ in 0..cycles {
            for domain in &self.domains {
                model.push(domain.draw(rng)?);
            }
            if let Some(reset) = reset {
                // Occasional resets exercise reinitialization and leave room
                // for longer runs.
                model.push(IrValue::bool(rng.gen_ratio(1, 16) ^ reset.active_low));
            }
        }
        Ok(model)
    }

    /// Replay an initialized bounded counterexample and identify its first
    /// failing sample.
    pub(crate) fn replay(&self, depth: usize, model: &[IrValue]) -> Result<String, String> {
        let trace = Trace::from_model(self, depth, model)?;
        let mut pir = PirCircuit::from_transition(&self.package, self.transition.clone());
        pir.reset_trial(&trace.registers);
        match trace.execute(
            self,
            &mut Concrete {
                pir: &mut pir,
                rtl: None,
            },
        ) {
            Err(failure) if matches!(failure.error, ExecutionError::Assertion(_)) => {
                Ok(trace.describe(self, &failure, false))
            }
            Err(failure) => Err(trace.describe(self, &failure, false)),
            Ok(()) => Err("solver counterexample passed concrete invariant replay".into()),
        }
    }
}

impl<'a> Trace<'a> {
    fn from_model(
        proof: &BxInvariantProof,
        cycles: usize,
        model: &'a [IrValue],
    ) -> Result<Self, String> {
        let expected = proof.model_size(cycles)?;
        if model.len() != expected {
            return Err(format!(
                "invariant model expects {expected} values, got {}",
                model.len()
            ));
        }
        let (initial, mut samples) = model.split_at(proof.transition.registers.len());
        let mut registers = BTreeMap::new();
        for (reg, value) in proof.transition.registers.iter().zip(initial) {
            if reg.ty != value.type_() {
                return Err(format!("invalid initial value for register '{}'", reg.name));
            }
            registers.insert(reg.name.clone(), value.clone());
        }
        let data = proof.domains.len();
        let reset_inputs = if proof.transition.reset.is_some() {
            let (init, rest) = samples.split_at(data);
            samples = rest;
            Some(init)
        } else {
            None
        };
        let trace = Self {
            registers,
            reset_inputs,
            samples,
            cycles,
            frame_width: data + usize::from(reset_inputs.is_some()),
        };
        for frame in trace
            .reset_inputs
            .into_iter()
            .chain((0..cycles).map(|i| trace.frame(i)))
        {
            for (j, (value, domain)) in frame.iter().zip(&proof.domains).enumerate() {
                if value.type_() != proof.transition.inputs[proof.input_indices[j]].ty
                    || !domain.admits(value)
                {
                    return Err(format!(
                        "invalid invariant input '{}': {value}",
                        proof.program.params[j].name
                    ));
                }
            }
            if let Some(level) = frame.get(data)
                && level.type_() != xlsynth_pir::ir::Type::Bits(1)
            {
                return Err(format!("invalid invariant reset input: {level}"));
            }
        }
        Ok(trace)
    }

    fn frame(&self, cycle: usize) -> &[IrValue] {
        &self.samples[cycle * self.frame_width..(cycle + 1) * self.frame_width]
    }

    fn drive(
        proof: &BxInvariantProof,
        circuit: &mut dyn ClockedHarness,
        frame: &[IrValue],
    ) -> Result<(), String> {
        for (param, value) in proof.program.params.iter().zip(frame) {
            circuit.drive(&param.name, value)?;
        }
        if let Some(reset) = &proof.transition.reset {
            let level = frame
                .get(proof.domains.len())
                .cloned()
                .unwrap_or_else(|| IrValue::bool(!reset.active_low));
            circuit.drive(&reset.name, &level)?;
        }
        Ok(())
    }

    /// All checks observe pre-edge values for this frame. An unknown RTL sample
    /// is remembered while the trace continues looking for a definite failure.
    fn execute(
        &self,
        proof: &BxInvariantProof,
        circuit: &mut dyn ClockedHarness,
    ) -> Result<(), TrialFailure> {
        let program = &proof.program;
        let runtime = |cycle, message| TrialFailure {
            position: program.position,
            cycle,
            error: ExecutionError::Runtime(message),
        };
        if let Some(frame) = self.reset_inputs {
            Self::drive(proof, circuit, frame)
                .and_then(|()| circuit.tick())
                .map_err(|e| runtime(None, e))?;
        }
        let mut unknown = None;
        for i in 0..self.cycles {
            let cycle = Some(i);
            Self::drive(proof, circuit, self.frame(i)).map_err(|e| runtime(cycle, e))?;
            for index in (1..=program.lemmas.len()).chain(std::iter::once(0)) {
                let (label, position) = if index == 0 {
                    ("property", program.position)
                } else {
                    let lemma = &program.lemmas[index - 1];
                    (lemma.name.as_str(), lemma.position)
                };
                match circuit.sample(index).map_err(|e| runtime(cycle, e))? {
                    Observed::Known(value) if value.bits_equals_u64_value(1) => {
                        // This observation holds; check the rest of the
                        // predicates.
                    }
                    Observed::Known(_) => {
                        let message = if index == 0 {
                            "property is false".to_owned()
                        } else {
                            format!("lemma '{label}' is false")
                        };
                        return Err(TrialFailure {
                            position,
                            cycle,
                            error: ExecutionError::Assertion(message),
                        });
                    }
                    Observed::Unknown(message) => {
                        unknown.get_or_insert_with(|| TrialFailure {
                            position,
                            cycle,
                            error: ExecutionError::Unknown(format!("{label}: {message}")),
                        });
                    }
                }
            }
            circuit.tick().map_err(|e| runtime(cycle, e))?;
        }
        unknown.map_or(Ok(()), Err)
    }

    fn describe(&self, proof: &BxInvariantProof, failure: &TrialFailure, from_x: bool) -> String {
        let program = &proof.program;
        let where_at = failure
            .cycle
            .map_or_else(|| "initial reset edge".to_owned(), |i| format!("cycle {i}"));
        let message = match &failure.error {
            ExecutionError::Runtime(s)
            | ExecutionError::Assertion(s)
            | ExecutionError::Unknown(s) => s,
        };
        let mut text = format!(
            "{}:{}:{}: invariant '{}' at {where_at}: {message}",
            proof.path.display(),
            failure.position.line,
            failure.position.column,
            program.name
        );
        let regs = if from_x {
            "X".into()
        } else {
            self.registers
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        write!(text, "\n  power-up registers: [{regs}]").unwrap();
        let mut append = |label: &str, values: &[IrValue]| {
            let mut fields = program
                .params
                .iter()
                .zip(values)
                .map(|(p, v)| format!("{}={v}", p.name))
                .collect::<Vec<_>>();
            if let Some(reset) = &proof.transition.reset {
                let asserted = values
                    .get(proof.domains.len())
                    .is_none_or(|level| level.bits_equals_u64_value(u64::from(!reset.active_low)));
                fields.push(format!("rst={asserted}"));
            }
            write!(text, "\n  {label}: [{}]", fields.join(", ")).unwrap();
        };
        if let Some(frame) = self.reset_inputs {
            append("initial reset edge", frame);
        }
        if let Some(end) = failure.cycle {
            for i in 0..=end {
                append(&format!("cycle {i}"), self.frame(i));
            }
        }
        text
    }
}

/// Check all observations with independently generated state and inputs per
/// trial.
pub(super) fn run(
    program: &InvariantProgram,
    checked: &CheckedModule,
    hardware: &Package,
    path: &Path,
    options: BxTestOptions,
) -> BxTestResult {
    let qc = options.quickcheck;
    let mut report = QuickCheckReport {
        seed: qc.seed,
        trials: 0,
        four_valued: qc.cross_check_vastly.then(FourValuedReport::default),
    };
    let result = (|| -> Result<(), String> {
        let proof = BxInvariantProof::prepare(program, checked, hardware, path)?;
        let mut pir = PirCircuit::from_transition(&proof.package, proof.transition.clone());
        let mut rtl = if qc.cross_check_vastly {
            let flat = flatten_block_hierarchy(hardware, &program.harness)?;
            Some(Simulation::from_flattened(&flat, &program.harness)?)
        } else {
            None
        };
        let mut rng = Pcg64Mcg::seed_from_u64(qc.seed);
        let cycles = options.invariant_cycles.get();
        for trial in 1..=qc.trials.get() {
            let model = proof.draw_model(cycles, &mut rng)?;
            let trace = Trace::from_model(&proof, cycles, &model)?;
            report.trials = trial;
            pir.reset_trial(&trace.registers);
            if let Some(sim) = &mut rtl {
                sim.reset_trial(Some(&trace.registers))?;
            }
            let describe = |failure: &TrialFailure, x| {
                format!(
                    "{}\n  seed {}, trial {trial}",
                    trace.describe(&proof, failure, x),
                    qc.seed
                )
            };
            let concrete = trace.execute(
                &proof,
                &mut Concrete {
                    pir: &mut pir,
                    rtl: rtl.as_mut(),
                },
            );
            let failure = concrete.err().map(|e| describe(&e, false));
            if let (Some(sim), Some(four)) = (&mut rtl, &mut report.four_valued) {
                sim.reset_trial(None)?;
                match trace.execute(&proof, sim) {
                    Ok(()) => four.passed += 1,
                    Err(x_failure) => {
                        if let Some(message) =
                            four.record_failure(&x_failure.error, || describe(&x_failure, true))
                        {
                            return Err(match failure {
                                Some(concrete) => {
                                    format!("{concrete}\n  four-valued execution error: {message}")
                                }
                                None => message,
                            });
                        }
                    }
                }
            }
            if let Some(message) = failure {
                return Err(message);
            }
        }
        Ok(())
    })();
    BxTestResult {
        name: program.name.clone(),
        failure: result.err(),
        quickcheck: Some(report),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::path::Path;

    use super::*;
    use crate::{
        ImportOptions, QuickCheckOptions, ResetBehavior, empty_package, prepare_invariant_proofs,
        run_tests_with_options,
    };

    const COUNTER: &str = r#"#[invariant]
block count_bound(clk, rst, enable: bool) -> bool {
    reg count: u2 = u2:0;
    if enable { count := count + u2:1; }
    #[lemma] let below_two = count < u2:2;
    true
}"#;

    #[test]
    fn base_replay_locates_a_later_false_lemma_for_both_reset_polarities() -> Result<(), String> {
        for active_low in [false, true] {
            let suite = prepare_invariant_proofs(
                COUNTER,
                Path::new("counter.bx"),
                empty_package("counter"),
                ResetBehavior {
                    asynchronous: false,
                    active_low,
                },
                &ImportOptions::default(),
                true,
            )?;
            let proof = suite.properties.into_iter().next().unwrap().proof?;
            // Arbitrary pre-reset count, reset-edge enable, three post-reset
            // enable/physical-reset pairs. Count reaches two on the third
            // sample.
            let initial = IrValue::parse_typed("bits[2]:3").map_err(|e| e.to_string())?;
            let mut model = vec![initial, IrValue::bool(false)];
            for _ in 0..3 {
                model.extend([IrValue::bool(true), IrValue::bool(active_low)]);
            }
            assert_eq!(
                proof.replay(2, &model[..6]).unwrap_err(),
                "solver counterexample passed concrete invariant replay"
            );
            let failure = proof.replay(3, &model)?;
            assert!(failure.starts_with("counter.bx:5:"));
            assert_eq!(
                failure.split_once(": invariant").unwrap().1,
                " 'count_bound' at cycle 2: lemma 'below_two' is false\n  power-up registers: [count=bits[2]:3]\n  initial reset edge: [enable=bits[1]:0, rst=true]\n  cycle 0: [enable=bits[1]:1, rst=false]\n  cycle 1: [enable=bits[1]:1, rst=false]\n  cycle 2: [enable=bits[1]:1, rst=false]"
            );
        }
        Ok(())
    }

    #[test]
    fn named_lemmas_sample_fresh_legal_inputs_and_persist_child_state() -> Result<(), String> {
        let source = r#"
enum Mode : u2 { Low = 1, High = 3 }
block delay(clk, b: bool) -> bool {
    reg last: bool = false;
    last := b;
    last
}
#[test] fn first() { assert!(true); }
#[invariant]
block delayed(clk, rst, modes: Mode[2]) -> bool {
    let data = modes[0] == Mode::High;
    let output = delay(data);
    reg model: bool = false;
    model := data;
    #[lemma] let valid_inputs = (modes[1] == Mode::Low) || (modes[1] == Mode::High);
    output == model
}
#[test] fn last() { assert!(true); }"#;
        let options = BxTestOptions {
            quickcheck: QuickCheckOptions {
                trials: NonZeroUsize::new(3).unwrap(),
                seed: 17,
                cross_check_vastly: true,
            },
            invariant_cycles: NonZeroUsize::new(8).unwrap(),
        };
        let report = run_tests_with_options(
            source,
            Path::new("delayed.bx"),
            empty_package("delayed"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
            options,
        )?;
        assert_eq!(
            report
                .tests
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["first", "delayed", "last"]
        );
        for test in &report.tests {
            assert!(test.failure.is_none(), "{:?}", test.failure);
        }
        assert_eq!(
            report
                .invariant_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["delayed"]
        );
        let report = report.tests[1].quickcheck.as_ref().unwrap();
        assert_eq!(report.trials, 3);
        assert_eq!(report.four_valued.as_ref().unwrap().passed, 3);
        Ok(())
    }

    #[test]
    fn a_changing_stimulus_can_falsify_a_property_after_its_first_sample() -> Result<(), String> {
        let source = r#"#[invariant]
block input_stays(clk, rst, data: bool) -> bool {
    reg previous: bool;
    reg seen: bool = false;
    previous := data;
    seen := true;
    !seen || data == previous
}"#;
        let run = |cycles| {
            let options = BxTestOptions {
                quickcheck: QuickCheckOptions {
                    trials: NonZeroUsize::MIN,
                    seed: 42,
                    cross_check_vastly: false,
                },
                invariant_cycles: NonZeroUsize::new(cycles).unwrap(),
            };
            run_tests_with_options(
                source,
                Path::new("changing.bx"),
                empty_package("changing"),
                ResetBehavior::default(),
                &ImportOptions::default(),
                true,
                options,
            )
        };
        assert!(run(1)?.tests[0].failure.is_none());
        let failure = run(16)?.tests.remove(0).failure.unwrap();
        assert!(failure.contains("property is false"), "{failure}");
        assert!(failure.contains("data=bits[1]:0") && failure.contains("data=bits[1]:1"));
        Ok(())
    }

    #[test]
    fn random_check_samples_unreset_state_and_reports_reproducible_traces() -> Result<(), String> {
        let source = r#"#[invariant]
block check_init(clk, rst, data: bool) -> bool {
    reg q: u8;
    if !rst && data { q := u8:0; }
    q == u8:0
}"#;
        let options = BxTestOptions {
            quickcheck: QuickCheckOptions {
                trials: NonZeroUsize::new(4).unwrap(),
                seed: 17,
                cross_check_vastly: true,
            },
            invariant_cycles: NonZeroUsize::new(3).unwrap(),
        };
        let run = || {
            run_tests_with_options(
                source,
                Path::new("powerup.bx"),
                empty_package("powerup"),
                ResetBehavior::default(),
                &ImportOptions::default(),
                true,
                options,
            )
        };
        let mut first = run()?.tests.remove(0);
        let second = run()?.tests.remove(0);
        assert_eq!(first.failure, second.failure);
        let failure = first.failure.take().unwrap();
        assert!(
            failure.contains("invariant 'check_init' at cycle 0: property is false"),
            "{failure}"
        );
        assert!(failure.contains("seed 17, trial 1"));
        let four = first.quickcheck.unwrap().four_valued.unwrap();
        assert_eq!(four.indeterminate, 1);
        assert!(
            four.first_indeterminate
                .unwrap()
                .contains("power-up registers: [X]")
        );
        Ok(())
    }
}
