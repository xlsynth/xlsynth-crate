// SPDX-License-Identifier: Apache-2.0
#![no_main]

//! Fuzzes the prover scheduler for correctness of reduction semantics and
//! invariance across worker counts.
//!
//! From a fuzzed `FuzzPlanNode` we build a `ProverPlan` that contains only
//! `Fake` leaf tasks and `Group` combinators {`All`, `Any`, `First`}.
//! We then compute an abstract expected result by locally interpreting the
//! tree (`eval`):
//! - `All`: true if all children true; false if any child false; else
//!   indeterminate.
//! - `Any`: true if any child true; false if all children false; else
//!   indeterminate.
//! - `First`: determinate only when all children agree; otherwise indeterminate
//!   (models potential nondeterminism due to scheduling/ordering).
//!
//! The plan is executed via `run_prover_plan` with 1..=4 workers. Whenever the
//! abstract eval is determinate (`Some(v)`), we require the scheduler's
//! `report.success == v` for every worker count. If the abstract eval is
//! indeterminate (`None`), either outcome is acceptable. This catches
//! ordering/parallelism bugs and ensures the scheduler matches the intended
//! semantics of these combinators.
//!
//! TODO: Currently, we don't have a timeout in our prover plan. This will be
//! left as future work.

use libfuzzer_sys::fuzz_target;

use xlsynth_driver::prover::run_prover_plan;
use xlsynth_driver::prover_config::{GroupKind, ProverPlan, ProverTask};

#[derive(Clone, Copy, Debug)]
struct Eval {
    outcome: Option<bool>,
}

fn eval(plan: &ProverPlan) -> Eval {
    match plan {
        ProverPlan::Task { task } => match task {
            ProverTask::Fake { config } => Eval {
                outcome: Some(config.success),
            },
            _ => panic!(
                "non-Fake task encountered in fuzz plan; builder should only emit Fake tasks"
            ),
        },
        ProverPlan::Group { kind, tasks } => {
            let kids: Vec<Eval> = tasks.iter().map(eval).collect();
            match kind {
                GroupKind::All => {
                    if kids.iter().any(|k| k.outcome == Some(false)) {
                        Eval {
                            outcome: Some(false),
                        }
                    } else if kids.iter().all(|k| k.outcome == Some(true)) {
                        Eval {
                            outcome: Some(true),
                        }
                    } else {
                        Eval { outcome: None }
                    }
                }
                GroupKind::Any => {
                    if kids.iter().any(|k| k.outcome == Some(true)) {
                        Eval {
                            outcome: Some(true),
                        }
                    } else if kids.iter().all(|k| k.outcome == Some(false)) {
                        Eval {
                            outcome: Some(false),
                        }
                    } else {
                        Eval { outcome: None }
                    }
                }
                GroupKind::First => {
                    // Do not assume earliest completion; only determinate if all children share the
                    // same outcome.
                    if kids.iter().any(|k| k.outcome.is_none()) {
                        Eval { outcome: None }
                    } else if kids.iter().all(|k| k.outcome == Some(true)) {
                        Eval {
                            outcome: Some(true),
                        }
                    } else if kids.iter().all(|k| k.outcome == Some(false)) {
                        Eval {
                            outcome: Some(false),
                        }
                    } else {
                        Eval { outcome: None }
                    }
                }
            }
        }
    }
}

fuzz_target!(|root: xlsynth_driver_fuzz::FuzzPlanNode| {
    // Build a ProverPlan from fuzz input with bounds.
    let Some(plan) = xlsynth_driver_fuzz::build_plan_from_fuzz(root) else {
        return;
    };

    // Compute expected outcome with First support; None means both possible.
    let expected = eval(&plan).outcome;

    // Run the scheduler across 1..=4 cores. If expected is Some(v), all must match
    // v.
    for cores in 1..=4 {
        let report = match run_prover_plan(plan.clone(), cores) {
            Ok(r) => r,
            Err(_) => return,
        };
        if let Some(v) = expected {
            assert_eq!(report.success, v);
        }
    }
});
