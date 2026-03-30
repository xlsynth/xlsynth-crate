// SPDX-License-Identifier: Apache-2.0
use arbitrary::Arbitrary;
use xlsynth_driver::prover_config::{GroupKind, ProverPlan, ProverTask};

#[derive(Debug, Clone, Arbitrary)]
pub enum FuzzNodeKind {
    Task,
    GroupAll,
    GroupAny,
    GroupFirst,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzFakeTaskCfg {
    pub delay_ms: u32,
    pub success: bool,
    pub stdout_len: u8,
    pub stderr_len: u8,
    pub timeout_ms: Option<u32>,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzPlanNode {
    pub kind: FuzzNodeKind,
    pub children: Vec<FuzzPlanNode>,
    pub fake: Option<FuzzFakeTaskCfg>,
    pub keep_running_till_finish: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Eval {
    pub outcome: Option<bool>,
}

impl FuzzPlanNode {
    pub fn clamp(mut self, max_nodes: &mut usize, max_depth: usize) -> Option<Self> {
        if *max_nodes == 0 {
            return None;
        }
        *max_nodes -= 1;
        if max_depth == 0 {
            // Force to task
            self.kind = FuzzNodeKind::Task;
            self.children.clear();
        }
        // Limit fanout
        if self.children.len() > 4 {
            self.children.truncate(4);
        }
        // Recurse with depth budget
        let mut new_children = Vec::new();
        for c in self.children.into_iter() {
            if let Some(kid) = c.clamp(max_nodes, max_depth.saturating_sub(1)) {
                new_children.push(kid);
            }
            if new_children.len() >= 4 {
                break;
            }
        }
        self.children = new_children;
        match self.kind {
            FuzzNodeKind::Task => {
                if self.fake.is_some() {
                    Some(self)
                } else {
                    None
                }
            }
            FuzzNodeKind::GroupAll | FuzzNodeKind::GroupAny | FuzzNodeKind::GroupFirst => {
                if self.children.is_empty() {
                    None
                } else {
                    Some(self)
                }
            }
        }
    }
}

pub fn decode_fuzz_plan_node(data: &[u8]) -> arbitrary::Result<FuzzPlanNode> {
    let mut unstructured = arbitrary::Unstructured::new(data);
    FuzzPlanNode::arbitrary(&mut unstructured)
}

pub fn eval_fuzz(node: &FuzzPlanNode) -> Eval {
    match node.kind {
        FuzzNodeKind::Task => {
            if let Some(f) = &node.fake {
                match f.timeout_ms {
                    Some(_) => Eval { outcome: None },
                    None => Eval {
                        outcome: Some(f.success),
                    },
                }
            } else {
                Eval { outcome: None }
            }
        }
        FuzzNodeKind::GroupAll | FuzzNodeKind::GroupAny | FuzzNodeKind::GroupFirst => {
            let kids: Vec<Eval> = node.children.iter().map(eval_fuzz).collect();
            match node.kind {
                FuzzNodeKind::GroupAll => {
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
                FuzzNodeKind::GroupAny => {
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
                FuzzNodeKind::GroupFirst => {
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
                FuzzNodeKind::Task => unreachable!(),
            }
        }
    }
}

pub fn build_plan_from_fuzz(root: FuzzPlanNode) -> ProverPlan {
    fn to_plan(n: &FuzzPlanNode) -> ProverPlan {
        match n.kind {
            FuzzNodeKind::Task => {
                // Assume normalized: Task must have a fake config.
                let f = n.fake.as_ref().expect(
                    "build_plan_from_fuzz: Task node missing fake config after normalization",
                );
                let cfg = xlsynth_driver::prover_config::FakeTaskConfig {
                    // If a timeout is present, make the task indefinite so it deterministically
                    // times out. Otherwise, use a bounded small delay so fuzz runs complete.
                    delay_ms: if f.timeout_ms.is_some() {
                        None
                    } else {
                        Some((f.delay_ms % 120).max(1))
                    },
                    success: f.success,
                    stdout_len: f.stdout_len as u16,
                    stderr_len: f.stderr_len as u16,
                };
                let timeout_ms = f.timeout_ms.map(|t| (t % 120) as u64); // keep small
                ProverPlan::Task {
                    task: ProverTask::Fake { config: cfg },
                    timeout_ms,
                    task_id: None,
                }
            }
            FuzzNodeKind::GroupAll | FuzzNodeKind::GroupAny | FuzzNodeKind::GroupFirst => {
                let mut tasks = Vec::new();
                for c in &n.children {
                    // Assume normalized: children should always convert.
                    let p = to_plan(c);
                    tasks.push(p);
                }
                assert!(
                    !tasks.is_empty(),
                    "build_plan_from_fuzz: Group has zero children after normalization"
                );
                let kind = match n.kind {
                    FuzzNodeKind::GroupAll => GroupKind::All,
                    FuzzNodeKind::GroupAny => GroupKind::Any,
                    FuzzNodeKind::GroupFirst => GroupKind::First,
                    _ => unreachable!(),
                };
                ProverPlan::Group {
                    kind,
                    tasks,
                    keep_running_till_finish: n.keep_running_till_finish,
                }
            }
        }
    }
    to_plan(&root)
}

#[cfg(test)]
mod tests {
    use super::{build_plan_from_fuzz, decode_fuzz_plan_node, eval_fuzz};
    use xlsynth_driver::prover::run_prover_plan;

    #[test]
    fn crashing_scheduler_artifact_single_success_task_regression() {
        let bytes = [0x27, 0x27, 0x27, 0x27, 0x32, 0x27, 0x0b, 0x27, 0x27, 0x27, 0x0b];
        let root = decode_fuzz_plan_node(&bytes).expect("decode");
        let mut budget = 64usize;
        let root = root.clamp(&mut budget, 5).expect("clamp");
        assert_eq!(eval_fuzz(&root), super::Eval { outcome: Some(true) });
        let plan = build_plan_from_fuzz(root);
        for cores in 1..=4 {
            let report = run_prover_plan(plan.clone(), cores).expect("run");
            assert!(
                report.success,
                "single successful fake task should succeed for {cores} cores; report={:#?}",
                report.plan
            );
        }
    }
}
