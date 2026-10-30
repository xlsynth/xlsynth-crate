// SPDX-License-Identifier: Apache-2.0

//! Scripted decomposition and SAT discharge of independent stage obligations.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::Deserialize;
use xlsynth_prover::dslx_tactics::script::{Selector, syntax::ScriptStep};

use crate::aig::topo::extract_cone;
use crate::aig::{AigOperand, GateFn};
use crate::prove_gate_fn_equiv_sat::{
    CadicalSat, GateFormalOptions, IncrementalSat, SatSolveResult, ValidationError,
    build_sat_clauses,
};

use super::stage_graph::StageMiters;

/// A gate-level tactic that partitions a stage's ordered mismatch bits.
#[derive(Deserialize)]
enum StageTactic {
    SplitOutputs,
}

/// Commands accepted by the gate-level stage proof runner.
#[derive(Deserialize)]
enum StageCommand {
    Apply(StageTactic),
    Solve,
}

/// A complete plan of selected leaves for one stage's combinational proof.
#[derive(Clone, Debug)]
pub struct StageProofScript {
    leaves: Vec<Selector>,
}

impl Default for StageProofScript {
    fn default() -> Self {
        Self {
            leaves: vec![vec!["root".to_owned()]],
        }
    }
}

impl StageProofScript {
    /// Parses an array of Apply/Solve steps, requiring one Solve for every
    /// leaf.
    pub fn from_json(input: &str) -> Result<Self, String> {
        let steps: Vec<ScriptStep<StageCommand>> = serde_json::from_str(input)
            .map_err(|err| format!("invalid stage proof script: {err}"))?;
        let mut open = BTreeSet::from([vec!["root".to_owned()]]);
        let mut leaves = Vec::new();
        for step in steps {
            if !open.remove(&step.selector) {
                return Err(format!(
                    "selector {:?} does not name an open leaf",
                    step.selector
                ));
            }
            match step.command {
                StageCommand::Solve => leaves.push(step.selector),
                StageCommand::Apply(StageTactic::SplitOutputs) => {
                    for branch in ["low", "high"] {
                        let mut child = step.selector.clone();
                        child.push(branch.to_owned());
                        open.insert(child);
                    }
                }
            }
        }
        if let Some(selector) = open.first() {
            return Err(format!("stage proof script leaves {selector:?} unsolved"));
        }
        Ok(Self { leaves })
    }

    /// Instantiates the selected leaves at the actual stage width.
    fn ranges(&self, width: usize) -> Result<Vec<Range<usize>>, String> {
        self.leaves
            .iter()
            .map(|leaf| {
                let mut bits = 0..width;
                for (depth, branch) in leaf.iter().enumerate().skip(1) {
                    if bits.len() < 2 {
                        return Err(format!(
                            "SplitOutputs requires at least two bits at {:?}, found {}",
                            &leaf[..depth],
                            bits.len()
                        ));
                    }
                    let middle = bits.start + bits.len() / 2;
                    // Only low/high selectors created by SplitOutputs reach
                    // here.
                    if branch == "low" {
                        bits.end = middle;
                    } else {
                        bits.start = middle;
                    }
                }
                Ok(bits)
            })
            .collect()
    }
}

/// Conclusion of all selected stage proofs in specification order.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum StageSolveOutcome {
    Proved { leaf_counts: Vec<usize> },
    Counterexample { location: String },
    Inconclusive { location: String },
}

/// A single selected range in the shared graph's stage mismatch vectors.
struct LeafTask<'a> {
    stage: &'a str,
    selector: &'a Selector,
    bits: &'a [AigOperand],
}

impl LeafTask<'_> {
    /// Identifies the stage and, for a split proof, its selected child.
    fn location(&self) -> String {
        if self.selector.len() == 1 {
            format!("'{}'", self.stage)
        } else {
            format!("'{}' at {}", self.stage, self.selector.join("/"))
        }
    }
}

/// Asserts the disjunction of the selected mismatches in their fan-in cone.
fn solve_leaf(
    graph: &GateFn,
    bits: &[AigOperand],
    options: GateFormalOptions,
) -> Result<SatSolveResult, ValidationError> {
    let mut solver = CadicalSat::new_with_options(options)?;
    let roots: Vec<_> = bits.iter().map(|bit| bit.node).collect();
    let (cone, inputs) = extract_cone(&roots, &graph.gates);
    let literals = build_sat_clauses(&mut solver, &cone, &inputs, &graph.gates);
    let clause: Vec<_> = bits
        .iter()
        .map(|bit| {
            let literal = literals[&bit.node];
            if bit.negated { !literal } else { literal }
        })
        .collect();
    // A stage with no mismatch bits has an empty (unsatisfiable) disjunction.
    solver.sat_add_clause(&clause);
    solver.sat_solve_assuming(&[])
}

/// Validates all stage plans before sharing one worker budget across their
/// leaves.
pub(super) fn run_stage_proofs(
    miters: &StageMiters,
    stage_names: &[String],
    scripts: &BTreeMap<String, StageProofScript>,
    jobs: usize,
    options: GateFormalOptions,
) -> Result<StageSolveOutcome, String> {
    if jobs == 0 {
        return Err("stage proof worker count must be positive".into());
    }
    if stage_names.is_empty() || stage_names.len() != miters.mismatches.len() {
        return Err("stage names must match the nonempty list of stage obligations".into());
    }
    let default = StageProofScript::default();
    let mut tasks = Vec::new();
    let mut leaf_counts = Vec::new();
    for (stage, bits) in stage_names.iter().zip(&miters.mismatches) {
        let script = scripts.get(stage).unwrap_or(&default);
        let ranges = script
            .ranges(bits.len())
            .map_err(|err| format!("stage '{stage}': {err}"))?;
        leaf_counts.push(ranges.len());
        for (selector, range) in script.leaves.iter().zip(ranges) {
            tasks.push(LeafTask {
                stage,
                selector,
                bits: &bits[range],
            });
        }
    }
    let solve = |task: &LeafTask| match solve_leaf(&miters.graph, task.bits, options) {
        Ok(SatSolveResult::Unsat) => None,
        Ok(SatSolveResult::Sat) => Some(Ok(StageSolveOutcome::Counterexample {
            location: task.location(),
        })),
        Err(ValidationError::CadicalSolveInterrupted) => {
            Some(Ok(StageSolveOutcome::Inconclusive {
                location: task.location(),
            }))
        }
        Err(err) => Some(Err(format!("stage {}: {err}", task.location()))),
    };
    let workers = jobs.min(tasks.len());
    // Preserve stage and script order. After a failure, skip later leaves
    // that have not started. SAT calls already in flight finish normally.
    let first_failure = if workers > 1
        && let Ok(pool) = ThreadPoolBuilder::new().num_threads(workers).build()
    {
        pool.install(|| tasks.par_iter().find_map_first(solve))
    } else {
        // The worker budget is a maximum. A host at its thread limit can
        // still discharge every obligation on the calling thread.
        tasks.iter().find_map(solve)
    };
    first_failure.unwrap_or(Ok(StageSolveOutcome::Proved { leaf_counts }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    const SPLIT: &str = r#"[
        {"selector":["root"], "command":{"Apply":"SplitOutputs"}},
        {"selector":["root","high"], "command":{"Apply":"SplitOutputs"}},
        {"selector":["root","high","low"], "command":"Solve"},
        {"selector":["root","low"], "command":"Solve"},
        {"selector":["root","high","high"], "command":"Solve"}
    ]"#;

    #[test]
    fn nested_splits_cover_each_bit_once_including_an_odd_width() {
        let script = StageProofScript::from_json(SPLIT).unwrap();
        assert_eq!(script.ranges(5).unwrap(), vec![2..3, 0..2, 3..5]);
        assert_eq!(
            script.ranges(2).unwrap_err(),
            r#"SplitOutputs requires at least two bits at ["root", "high"], found 1"#
        );
    }

    #[test]
    fn incomplete_duplicate_and_unknown_actions_are_rejected() {
        let incomplete = r#"[
            {"selector":["root"], "command":{"Apply":"SplitOutputs"}},
            {"selector":["root","low"], "command":"Solve"}
        ]"#;
        assert_eq!(
            StageProofScript::from_json(incomplete).unwrap_err(),
            r#"stage proof script leaves ["root", "high"] unsolved"#
        );
        let duplicate = r#"[
            {"selector":["root"], "command":"Solve"},
            {"selector":["root"], "command":"Solve"}
        ]"#;
        assert_eq!(
            StageProofScript::from_json(duplicate).unwrap_err(),
            r#"selector ["root"] does not name an open leaf"#
        );
        for invalid in [
            r#"[]"#,
            r#"[{"selector":["root","low"], "command":"Solve"}]"#,
            r#"[{"selector":["root"], "command":"Skip"}]"#,
            r#"[{"selector":["root"], "command":{"Apply":"Focus"}}]"#,
        ] {
            assert!(StageProofScript::from_json(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn split_and_unscripted_stages_discharge_all_bits_with_any_worker_count() {
        let mut builder = GateBuilder::new("stage_miters".into(), GateBuilderOptions::opt());
        let input = builder.add_input("x".into(), 1);
        let bit = *input.get_lsb(0);
        let zero = builder.get_false();
        builder.add_output("out".into(), input);
        let mut miters = StageMiters {
            graph: builder.build(),
            mismatches: vec![vec![zero; 5], Vec::new(), vec![zero]],
        };
        let stages = ["first".into(), "empty".into(), "last".into()];
        let scripts =
            BTreeMap::from([("first".into(), StageProofScript::from_json(SPLIT).unwrap())]);
        let options = GateFormalOptions::default();
        for jobs in [1, 2, 8] {
            assert_eq!(
                run_stage_proofs(&miters, &stages, &scripts, jobs, options).unwrap(),
                StageSolveOutcome::Proved {
                    leaf_counts: vec![3, 1, 1]
                }
            );
        }
        // Two stages have satisfiable leaves. Even when the last stage finishes
        // first, the earlier stage's refutation must be reported.
        miters.mismatches[0][4] = bit.negate();
        miters.mismatches[2][0] = bit;
        for jobs in [1, 2, 8] {
            assert_eq!(
                run_stage_proofs(&miters, &stages, &scripts, jobs, options).unwrap(),
                StageSolveOutcome::Counterexample {
                    location: "'first' at root/high/high".into()
                }
            );
        }
        miters.mismatches[0][4] = zero;
        assert_eq!(
            run_stage_proofs(&miters, &stages, &scripts, 2, options).unwrap(),
            StageSolveOutcome::Counterexample {
                location: "'last'".into()
            }
        );
        let invalid_limit = options.with_cadical_terminate_limit(u32::MAX);
        assert!(run_stage_proofs(&miters, &stages, &scripts, 2, invalid_limit).is_err());
    }
}
