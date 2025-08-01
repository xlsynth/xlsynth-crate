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
    pub min_delay_ms: u32,
    pub max_delay_ms: u32,
    pub success: bool,
    pub stdout_len: u8,
    pub stderr_len: u8,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzPlanNode {
    pub kind: FuzzNodeKind,
    pub children: Vec<FuzzPlanNode>,
    pub fake: Option<FuzzFakeTaskCfg>,
}

impl FuzzPlanNode {
    fn clamp(mut self, max_nodes: &mut usize, max_depth: usize) -> Option<Self> {
        if *max_nodes == 0 { return None; }
        *max_nodes -= 1;
        if max_depth == 0 {
            // Force to task
            self.kind = FuzzNodeKind::Task;
            self.children.clear();
        }
        // Limit fanout
        if self.children.len() > 4 { self.children.truncate(4); }
        // Recurse with depth budget
        let mut new_children = Vec::new();
        for c in self.children.into_iter() {
            if let Some(kid) = c.clamp(max_nodes, max_depth.saturating_sub(1)) {
                new_children.push(kid);
            }
            if new_children.len() >= 4 { break; }
        }
        self.children = new_children;
        Some(self)
    }
}

pub fn build_plan_from_fuzz(root: FuzzPlanNode) -> Option<ProverPlan> {
    let mut budget = 64usize;
    let root = root.clamp(&mut budget, 5)?;
    fn to_plan(n: &FuzzPlanNode) -> Option<ProverPlan> {
        match n.kind {
            FuzzNodeKind::Task => {
                // Build a fake task; default values if missing.
                let f = n.fake.as_ref()?;
                let cfg = xlsynth_driver::prover_config::FakeTaskConfig {
                    min_delay_ms: (f.min_delay_ms % 20).min(f.max_delay_ms.max(1)),
                    max_delay_ms: (f.max_delay_ms % 40).max(1),
                    success: f.success,
                    stdout_len: f.stdout_len as u16,
                    stderr_len: f.stderr_len as u16,
                };
                Some(ProverPlan::Task {
                    task: ProverTask::Fake { config: cfg },
                })
            }
            FuzzNodeKind::GroupAll | FuzzNodeKind::GroupAny | FuzzNodeKind::GroupFirst => {
                let mut tasks = Vec::new();
                for c in &n.children {
                    if let Some(p) = to_plan(c) { tasks.push(p); }
                }
                if tasks.is_empty() { return None; }
                let kind = match n.kind {
                    FuzzNodeKind::GroupAll => GroupKind::All,
                    FuzzNodeKind::GroupAny => GroupKind::Any,
                    FuzzNodeKind::GroupFirst => GroupKind::First,
                    _ => unreachable!(),
                };
                Some(ProverPlan::Group { kind, tasks })
            }
        }
    }
    to_plan(&root)
}
