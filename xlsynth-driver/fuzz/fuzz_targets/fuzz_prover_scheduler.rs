// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

use xlsynth_driver::prover::run_prover_plan;
use xlsynth_driver::prover_config::{GroupKind, ProverPlan, ProverTask};

#[derive(Clone, Copy, Debug)]
struct Eval { outcome: Option<bool> }

fn eval(plan: &ProverPlan) -> Eval {
    match plan {
        ProverPlan::Task { task } => match task {
            ProverTask::Fake { config } => Eval { outcome: Some(config.success) },
            _ => panic!("non-Fake task encountered in fuzz plan; builder should only emit Fake tasks"),
        },
        ProverPlan::Group { kind, tasks } => {
            let kids: Vec<Eval> = tasks.iter().map(eval).collect();
            match kind {
                GroupKind::All => {
                    if kids.iter().any(|k| k.outcome == Some(false)) {
                        Eval { outcome: Some(false) }
                    } else if kids.iter().all(|k| k.outcome == Some(true)) {
                        Eval { outcome: Some(true) }
                    } else {
                        Eval { outcome: None }
                    }
                }
                GroupKind::Any => {
                    if kids.iter().any(|k| k.outcome == Some(true)) {
                        Eval { outcome: Some(true) }
                    } else if kids.iter().all(|k| k.outcome == Some(false)) {
                        Eval { outcome: Some(false) }
                    } else {
                        Eval { outcome: None }
                    }
                }
                GroupKind::First => {
                    // Do not assume earliest completion; only determinate if all children share the same outcome.
                    if kids.iter().any(|k| k.outcome.is_none()) {
                        Eval { outcome: None }
                    } else if kids.iter().all(|k| k.outcome == Some(true)) {
                        Eval { outcome: Some(true) }
                    } else if kids.iter().all(|k| k.outcome == Some(false)) {
                        Eval { outcome: Some(false) }
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
    let Some(plan) = xlsynth_driver_fuzz::build_plan_from_fuzz(root) else { return; };

    // Compute expected outcome with First support; None means both possible.
    let expected = eval(&plan).outcome;


    // Run the scheduler across 1..=4 cores. If expected is Some(v), all must match v.
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
