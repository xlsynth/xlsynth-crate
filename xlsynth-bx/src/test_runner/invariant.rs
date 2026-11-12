// SPDX-License-Identifier: Apache-2.0

//! Word-level base and induction obligations for persistent BX invariant
//! blocks.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_transition::{BlockTransition, lower_block_transition};
use xlsynth_pir::ir::{Package, Type};
use xlsynth_pir::ir_builder::{BValue, BuilderError, FnBuilder, InvokeSignature};
use xlsynth_pir::ir_inline::package_with_reachable_callees;

use super::proof::{MAX_PROOF_WORK, check_block_expansion, expanded_node_visits};
use super::proof_domain::legal_stimulus;
use super::prove::check_transition_contracts;
use super::quickcheck::Stimulus;
use super::{InvariantProgram, compile_tests};
use crate::{BxWarning, CheckedModule, ImportOptions, ResetBehavior};

/// The persistent invariants declared in the root BX file, in source order.
pub struct BxInvariantSuite {
    pub warnings: Vec<BxWarning>,
    pub properties: Vec<BxInvariantCase>,
}

/// A named invariant and the result of preparing its transition system.
pub struct BxInvariantCase {
    pub name: String,
    pub proof: Result<BxInvariantProof, String>,
}

/// A flattened transition system with BX source names and legal stimulus
/// domains.
pub struct BxInvariantProof {
    pub(super) program: InvariantProgram,
    pub(super) transition: BlockTransition,
    pub(super) package: Package,
    pub(super) domains: Vec<Stimulus>,
    pub(super) input_indices: Vec<usize>,
    pub(super) reset_index: Option<usize>,
    pub(super) path: PathBuf,
    frame_work: usize,
}

/// Base checks the initialized trace. The two step obligations start in an
/// arbitrary state, with the proved lemma group available to the main step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BxInvariantObligation {
    Base,
    LemmaStep,
    PropertyStep,
}

/// Compile each invariant's circuit once for all requested proof depths.
pub fn prepare_invariant_proofs(
    source: &str,
    path: &Path,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    warnings_as_errors: bool,
) -> Result<BxInvariantSuite, String> {
    let compiled = compile_tests(source, path, package, reset, options, warnings_as_errors)?;
    let hardware = compiled
        .clocked_package
        .as_ref()
        .unwrap_or(&compiled.package);
    let properties = compiled
        .suite
        .invariants
        .iter()
        .map(|program| {
            let proof = BxInvariantProof::prepare(program, &compiled.checked, hardware, path)
                .and_then(|proof| {
                    check_transition_contracts(&proof.package)?;
                    Ok(proof)
                })
                .map_err(|e| {
                    format!(
                        "{}:{}:{}: invariant '{}': {e}",
                        path.display(),
                        program.position.line,
                        program.position.column,
                        program.name
                    )
                });
            BxInvariantCase {
                name: program.name.clone(),
                proof,
            }
        })
        .collect();
    Ok(BxInvariantSuite {
        properties,
        warnings: compiled.warnings,
    })
}

impl BxInvariantProof {
    /// Resolve source-order input and observation names against the flattened
    /// block interface.
    pub(super) fn prepare(
        program: &InvariantProgram,
        checked: &CheckedModule,
        hardware: &Package,
        path: &Path,
    ) -> Result<Self, String> {
        check_block_expansion(hardware, &program.harness)?;
        let transition = lower_block_transition(hardware, &program.harness)?;
        let package = package_with_reachable_callees(hardware, &transition.function)?;
        let ports: HashMap<_, _> = transition
            .inputs
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();
        let port_index = |name: &str| {
            ports
                .get(name)
                .copied()
                .ok_or_else(|| format!("missing invariant input port '{name}'"))
        };
        let reset_index = transition
            .reset
            .as_ref()
            .map(|reset| port_index(&reset.name))
            .transpose()?;
        let input_indices = program
            .params
            .iter()
            .map(|p| port_index(&p.name))
            .collect::<Result<Vec<_>, _>>()?;
        if input_indices.len() + usize::from(reset_index.is_some()) != transition.inputs.len() {
            return Err("invalid invariant input interface".into());
        }
        if transition.outputs.len() != program.lemmas.len() + 1
            || transition.outputs.iter().any(|p| p.ty != Type::Bits(1))
        {
            return Err("invalid invariant observation interface".into());
        }
        let domains = program
            .params
            .iter()
            .zip(&input_indices)
            .map(|(p, i)| Stimulus::new(&p.ty, &transition.inputs[*i].ty, checked))
            .collect::<Result<_, _>>()?;
        let frame_work = expanded_node_visits(&package, &transition.function.name)?
            .saturating_add(transition.inputs.len())
            .saturating_add(transition.outputs.len())
            .saturating_add(transition.registers.len())
            .saturating_add(1);
        Ok(Self {
            program: program.clone(),
            transition,
            package,
            domains,
            input_indices,
            reset_index,
            path: path.to_owned(),
            frame_work,
        })
    }

    pub(crate) fn has_lemmas(&self) -> bool {
        !self.program.lemmas.is_empty()
    }

    /// Construct a Boolean function over arbitrary power-up bits and fresh
    /// inputs at each edge. Only the base starts by applying a reset edge.
    pub(crate) fn query(
        &self,
        obligation: BxInvariantObligation,
        depth: usize,
    ) -> Result<Package, String> {
        if depth == 0 {
            return Err("invariant proof depth must be positive".into());
        }
        if obligation == BxInvariantObligation::LemmaStep && !self.has_lemmas() {
            return Err("expected a nonempty invariant lemma group".into());
        }
        let base = obligation == BxInvariantObligation::Base;
        let starts_with_reset = base && self.reset_index.is_some();
        let frame_count = depth.saturating_add(usize::from(!base || starts_with_reset));
        let work = frame_count.saturating_mul(self.frame_work);
        let mut budget = MAX_PROOF_WORK.checked_sub(work).ok_or_else(|| format!(
            "invariant depth {depth} exceeds the symbolic unrolling limit of {MAX_PROOF_WORK} work units"
        ))?;
        let mut name = format!("bx_prove_{}", self.program.harness);
        while self.package.get_fn(&name).is_some() {
            name.push('_');
        }
        let mut builder = FnBuilder::new(&name);
        let build = |error: BuilderError| error.to_string();
        let mut state = self
            .transition
            .registers
            .iter()
            .enumerate()
            .map(|(i, r)| builder.param(&format!("state_{i}"), r.ty.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(build)?;
        // Declare all parameters before emitting any body nodes. Counterexample
        // replay consumes this same order, including reset-edge data inputs.
        let mut frames = Vec::with_capacity(frame_count);
        for t in 0..frame_count {
            let data = self
                .input_indices
                .iter()
                .enumerate()
                .map(|(i, index)| {
                    builder.param(
                        &format!("frame_{t}_input_{i}"),
                        self.transition.inputs[*index].ty.clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(build)?;
            let reset = if self.reset_index.is_some() && !(starts_with_reset && t == 0) {
                Some(
                    builder
                        .param(&format!("frame_{t}_reset"), Type::Bits(1))
                        .map_err(build)?,
                )
            } else {
                None
            };
            frames.push((data, reset));
        }
        let signature = InvokeSignature::from_function(&self.transition.function).map_err(build)?;
        let mut assumptions = Vec::new();
        let mut checks = Vec::new();
        for (t, (data, reset)) in frames.into_iter().enumerate() {
            let legal = legal_stimulus(
                &mut builder,
                &self.program.params,
                &data,
                &self.domains,
                &mut budget,
            )?;
            assumptions.push(legal);
            let mut args = vec![None; self.transition.inputs.len()];
            for (i, value) in self.input_indices.iter().zip(data) {
                args[*i] = Some(value);
            }
            if let (Some(i), Some(info)) = (self.reset_index, &self.transition.reset) {
                args[i] = Some(match reset {
                    Some(raw) => raw,
                    None => builder
                        .literal(IrValue::bool(!info.active_low))
                        .map_err(build)?,
                });
            }
            let mut args: Vec<_> = args
                .into_iter()
                .collect::<Option<_>>()
                .ok_or("missing invariant input")?;
            args.extend_from_slice(&state);
            let sampled = builder
                .invoke_with_signature(&signature, &args)
                .map_err(build)?;
            if t + 1 < frame_count {
                for (i, q) in state.iter_mut().enumerate() {
                    *q = builder
                        .tuple_index(sampled, self.transition.outputs.len() + i)
                        .map_err(build)?;
                }
            }
            if starts_with_reset && t == 0 {
                continue;
            }
            let last = t + 1 == frame_count;
            for i in 0..self.transition.outputs.len() {
                let target = match (obligation, i, last) {
                    (BxInvariantObligation::Base, ..) => &mut checks,
                    (BxInvariantObligation::LemmaStep, 0, _) => continue,
                    (BxInvariantObligation::LemmaStep, _, true)
                    | (BxInvariantObligation::PropertyStep, 0, true) => &mut checks,
                    _ => &mut assumptions,
                };
                target.push(builder.tuple_index(sampled, i).map_err(build)?);
            }
        }
        let premise = conjunction(&mut builder, &assumptions).map_err(build)?;
        let conclusion = conjunction(&mut builder, &checks).map_err(build)?;
        let skipped = builder.not(premise).map_err(build)?;
        let pred = builder.or(skipped, conclusion).map_err(build)?;
        let function = builder
            .build_in_package(pred, &self.package)
            .map_err(build)?;
        package_with_reachable_callees(&self.package, &function)
    }
}

fn conjunction(builder: &mut FnBuilder, terms: &[BValue]) -> Result<BValue, BuilderError> {
    match terms {
        [] => builder.literal(IrValue::bool(true)),
        [one] => Ok(*one),
        many => builder
            .concat(many)
            .and_then(|bits| builder.and_reduce(bits)),
    }
}

#[cfg(test)]
mod tests {
    use super::BxInvariantObligation::{Base, LemmaStep, PropertyStep};
    use super::*;
    use crate::empty_package;
    use xlsynth_pir::IrBits;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};

    #[test]
    fn obligation_truth_tables_match_the_state_machine() -> Result<(), String> {
        let source = r#"
enum Op : u2 { Clear = 1, Set = 2 }
#[invariant]
block check_op(clk, rst, op: Op) -> bool {
    reg q: Op;
    if rst { q := op; }
    #[lemma] let ready = q == Op::Clear || op == Op::Set;
    q == op || op == Op::Clear
}
"#;
        for active_low in [false, true] {
            let suite = prepare_invariant_proofs(
                source,
                Path::new("enum.bx"),
                empty_package("enum_proof"),
                ResetBehavior {
                    asynchronous: false,
                    active_low,
                },
                &ImportOptions::default(),
                true,
            )?;
            let proof = suite.properties.into_iter().next().unwrap().proof?;
            for (kind, k) in [(Base, 2), (LemmaStep, 1), (PropertyStep, 1)] {
                let query = proof.query(kind, k)?;
                let top = query.get_top_fn().unwrap();
                let bits = |n| IrValue::Bits(IrBits::make_ubits(2, n).unwrap());
                let legal = |op| (1..=2).contains(&op);
                // q has no reset initializer. On any asserted reset edge it
                // loads that edge's data input through an ordinary enable.
                for power_up in 0..4 {
                    for reset_op in 0..(if kind == Base { 4 } else { 1 }) {
                        for op0 in 0..4 {
                            for op1 in 0..4 {
                                for raw_reset in [false, true] {
                                    let q0 = if kind == Base { reset_op } else { power_up };
                                    let q1 = if raw_reset != active_low { op0 } else { q0 };
                                    let h0 = q0 == 1 || op0 == 2;
                                    let h1 = q1 == 1 || op1 == 2;
                                    let p0 = q0 == op0 || op0 == 1;
                                    let p1 = q1 == op1 || op1 == 1;
                                    let holds = match kind {
                                        Base => h0 && p0 && h1 && p1,
                                        LemmaStep => !h0 || h1,
                                        PropertyStep => !(h0 && p0 && h1) || p1,
                                    };
                                    let in_domain = legal(op0)
                                        && legal(op1)
                                        && (kind != Base || legal(reset_op));
                                    let mut args = vec![bits(power_up)];
                                    if kind == Base {
                                        args.push(bits(reset_op));
                                    }
                                    args.extend([
                                        bits(op0),
                                        IrValue::bool(raw_reset),
                                        bits(op1),
                                        IrValue::bool(active_low),
                                    ]);
                                    let FnEvalResult::Success(result) =
                                        eval_fn_in_package(&query, top, &args)
                                    else {
                                        return Err("invariant obligation did not evaluate".into());
                                    };
                                    assert_eq!(
                                        result.value,
                                        IrValue::bool(!in_domain || holds),
                                        "{kind:?} {args:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
