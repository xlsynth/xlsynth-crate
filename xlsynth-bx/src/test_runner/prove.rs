// SPDX-License-Identifier: Apache-2.0

//! Prepare word-level, fixed-schedule properties for a function prover.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use xlsynth_pir::IrValue;
use xlsynth_pir::block_transition::lower_block_transition;
use xlsynth_pir::ir::{Binop, NodePayload, Package};
use xlsynth_pir::ir_inline::package_with_reachable_callees;

use super::{TestProgram, compile_tests, proof, quickcheck};
use crate::{BxWarning, ImportOptions, ResetBehavior};

/// The clocked properties declared in a BX file, in source order.
pub struct BxProofSuite {
    pub warnings: Vec<BxWarning>,
    pub properties: Vec<BxProofCase>,
}

/// A named property and either its proof input or a source diagnostic
/// explaining why it could not be prepared.
pub struct BxProofCase {
    pub name: String,
    pub proof: Result<BxQuickCheckProof, String>,
}

/// A Boolean PIR top and the concrete testbench used to replay solver models.
pub struct BxQuickCheckProof {
    pub package: Package,
    program: TestProgram,
    domains: Vec<quickcheck::Stimulus>,
    hardware: Arc<Package>,
    path: PathBuf,
    active_low: bool,
}

impl BxQuickCheckProof {
    /// Replay the solver's positional parameters and report the first failing
    /// check.
    pub fn replay(&self, inputs: &[IrValue]) -> Result<String, String> {
        quickcheck::replay_counterexample(
            &self.program,
            &self.domains,
            &self.hardware,
            &self.path,
            self.active_low,
            inputs,
        )
    }
}

/// Lower each block QuickCheck to a word-level predicate over stimulus and
/// initial state.
pub fn prepare_quickcheck_proofs(
    source: &str,
    path: &Path,
    package: Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
    warnings_as_errors: bool,
) -> Result<BxProofSuite, String> {
    let compiled = compile_tests(source, path, package, reset, options, warnings_as_errors)?;
    let hardware = Arc::new(compiled.clocked_package.unwrap_or(compiled.package));
    let mut properties = Vec::new();
    for program in compiled.suite.programs {
        let Some(params) = &program.quickcheck_params else {
            continue;
        };
        let prepare = || -> Result<_, String> {
            proof::check_block_expansion(&hardware, &program.harness)?;
            let transition = lower_block_transition(&hardware, &program.harness)?;
            let callees = package_with_reachable_callees(&hardware, &transition.function)?;
            check_transition_contracts(&callees)?;
            let domains = quickcheck::Stimulus::for_params(params, &transition, &compiled.checked)?;
            let predicate = proof::lower_property(&program, &transition, &domains, &callees)?;
            Ok((
                package_with_reachable_callees(&callees, &predicate)?,
                domains,
            ))
        };
        let name = program.name.clone();
        let proof = prepare()
            .map_err(|message| {
                if message.starts_with(|c: char| c.is_ascii_digit()) {
                    format!(
                        "{}:{message} (QuickCheck '{}')",
                        path.display(),
                        program.name
                    )
                } else {
                    format!(
                        "{}:{}:{}: QuickCheck '{}': {message}",
                        path.display(),
                        program.position.line,
                        program.position.column,
                        program.name
                    )
                }
            })
            .map(|(package, domains)| BxQuickCheckProof {
                package,
                program,
                domains,
                hardware: Arc::clone(&hardware),
                path: path.to_owned(),
                active_low: reset.active_low,
            });
        properties.push(BxProofCase { name, proof });
    }
    Ok(BxProofSuite {
        properties,
        warnings: compiled.warnings,
    })
}

/// The transition's returned values do not encode clock-edge assertion events
/// or array bounds promises. Refuse those until they are part of the predicate.
pub(super) fn check_transition_contracts(package: &Package) -> Result<(), String> {
    for member in &package.members {
        let graph = member.graph();
        for node in &graph.nodes {
            let reason = match &node.payload {
                NodePayload::Assert { label, .. } => format!("hardware assertion '{label}'"),
                NodePayload::ArrayIndex {
                    assumed_in_bounds: true,
                    ..
                }
                | NodePayload::ArrayUpdate {
                    assumed_in_bounds: true,
                    ..
                } => "assumed_in_bounds=true array operation".to_owned(),
                NodePayload::Binop(Binop::Umulp | Binop::Smulp, ..) => {
                    "partial-product multiply".to_owned()
                }
                _ => continue,
            };
            return Err(format!(
                "bounded proof does not yet support reachable {reason} in '{}'",
                graph.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empty_package;
    use xlsynth_pir::IrBits;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};

    #[test]
    fn enum_domain_constrains_stimulus_but_allows_every_initial_register_pattern()
    -> Result<(), String> {
        let source = r#"
enum Op : u2 { Hold = 1, Load = 2 }
block store(clk, rst, input: Op) -> Op {
    reg q: Op;
    q := input;
    q
}
#[quickcheck] block valid_at_power_up(input: Op) -> bool {
    let q = store(input);
    q == Op::Hold || q == Op::Load
}
"#;
        let suite = prepare_quickcheck_proofs(
            source,
            Path::new("enum.bx"),
            empty_package("enum_state"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
        )?;
        let proof = suite.properties.into_iter().next().unwrap().proof?;
        let one = IrValue::Bits(IrBits::make_ubits(2, 1).unwrap());
        let zero = IrValue::Bits(IrBits::make_ubits(2, 0).unwrap());
        for (stimulus, holds) in [(one, false), (zero.clone(), true)] {
            let model = [stimulus, zero.clone()];
            let FnEvalResult::Success(result) =
                eval_fn_in_package(&proof.package, proof.package.get_top_fn().unwrap(), &model)
            else {
                return Err("symbolic predicate failed to evaluate".into());
            };
            assert_eq!(result.value, IrValue::bool(holds));
            if holds {
                assert_eq!(
                    proof.replay(&model).unwrap_err(),
                    "counterexample is outside the declared stimulus domain for 'input'"
                );
            } else {
                assert!(
                    proof
                        .replay(&model)?
                        .contains("at tick 0: assertion failed")
                );
            }
        }
        Ok(())
    }

    /// Exhaust the legal inputs and power-up states for a small state machine.
    #[test]
    fn predicate_and_concrete_replay_agree_for_every_initial_state() -> Result<(), String> {
        let source = r#"
block store(clk, rst, write: bool, data: bool) -> bool {
    reg value: bool;
    if write { value := data; }
    value
}
#[quickcheck]
block preserves(write: bool, data: bool) -> bool {
    reg before: bool;
    let out = store(write, data);
    before := out;
    tick!();
    before == out
}
"#;
        let suite = prepare_quickcheck_proofs(
            source,
            Path::new("store.bx"),
            empty_package("store"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
        )?;
        let proof = suite.properties.into_iter().next().unwrap().proof?;
        let predicate = proof.package.get_top_fn().unwrap();
        for write in [false, true] {
            for data in [false, true] {
                for initial in [false, true] {
                    let model = [write, data, initial].map(IrValue::bool);
                    let FnEvalResult::Success(result) =
                        eval_fn_in_package(&proof.package, predicate, &model)
                    else {
                        return Err("symbolic predicate failed to evaluate".into());
                    };
                    let holds = !write || initial == data;
                    assert_eq!(result.value, IrValue::bool(holds));
                    let replay = proof.replay(&model);
                    if holds {
                        assert_eq!(
                            replay.unwrap_err(),
                            "solver counterexample passed concrete replay of 'preserves'"
                        );
                    } else {
                        assert!(replay?.contains("at tick 1: assertion failed"));
                    }
                }
            }
        }
        Ok(())
    }
}
