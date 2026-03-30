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

use xlsynth_driver::prover::{run_prover_plan, ProverReportNode, TaskOutcome};
use xlsynth_driver::prover_config::ProverPlan;
use xlsynth_driver_fuzz::eval_fuzz;

fuzz_target!(|root: xlsynth_driver_fuzz::FuzzPlanNode| {
    let mut budget = 64usize;
    let Some(root) = root.clamp(&mut budget, 5) else {
        return;
    };
    // Build a ProverPlan from normalized fuzz input (clamp already normalizes).
    let plan = xlsynth_driver_fuzz::build_plan_from_fuzz(root.clone());

    // Compute expected outcome on the normalized tree; None means both possible.
    let expected = eval_fuzz(&root).outcome;

    // Helper: verify the tasks that must be kept running are not canceled.
    fn assert_keep_running_corridor(plan: &ProverPlan, report: &ProverReportNode) {
        match (plan, report) {
            (ProverPlan::Task { .. }, ProverReportNode::Task { outcome, .. }) => match outcome {
                Some(TaskOutcome::Normal { .. }) | Some(TaskOutcome::Indefinite { .. }) => {}
                _ => panic!("leaf in keep-running corridor has no outcome"),
            },
            (
                ProverPlan::Group {
                    kind,
                    tasks,
                    keep_running_till_finish,
                    ..
                },
                ProverReportNode::Group {
                    kind: report_kind,
                    tasks: report_tasks,
                    ..
                },
            ) => {
                assert!(kind == report_kind, "group kind mismatch");
                assert!(report_tasks.len() == tasks.len(), "report shape mismatch");
                if !(*keep_running_till_finish || tasks.len() == 1) {
                    return;
                }
                for (t, r) in tasks.iter().zip(report_tasks.iter()) {
                    assert_keep_running_corridor(t, r);
                }
            }
            _ => {}
        }
    }

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

        assert_keep_running_corridor(&plan, &report.plan);
    }
});
