// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::proofs::obligations::{ObligationPayload, ProverObligation};
use crate::prover_config::{
    DslxEquivConfig, GroupKind, ProveQuickcheckConfig, ProverPlan, ProverTask,
};
use regex::escape;
use xlsynth_prover::prover::types::AssertionSemantics;
use xlsynth_prover::prover::SolverChoice;

fn write_text(
    dir: &tempfile::TempDir,
    name: &str,
    text: &str,
) -> Result<std::path::PathBuf, String> {
    let path = dir.path().join(name);
    std::fs::write(&path, text).map_err(|e| format!("write {}: {}", name, e))?;
    Ok(path)
}

fn cfg_with_paths(cfg: &mut DslxEquivConfig, stdlib: Option<&Path>, paths: &[&Path]) {
    if let Some(p) = stdlib {
        cfg.dslx_stdlib_path = Some(p.to_path_buf());
    }
    if !paths.is_empty() {
        cfg.dslx_path = Some(paths.iter().map(|p| (*p).to_path_buf()).collect());
    }
}

pub struct ObligationPlan {
    pub plan: ProverPlan,
    tempdir: tempfile::TempDir,
}

impl ObligationPlan {
    pub fn into_parts(self) -> (ProverPlan, tempfile::TempDir) {
        (self.plan, self.tempdir)
    }
}

pub fn build_plan_from_obligations(
    obligations: &[(String, ProverObligation)],
    dslx_stdlib_path: Option<&Path>,
    dslx_paths: &[&Path],
    solver: Option<SolverChoice>,
    timeout_ms: Option<u64>,
) -> Result<ObligationPlan, String> {
    let tempdir = tempfile::TempDir::new().map_err(|e| format!("tempdir: {e}"))?;
    let mut tasks: Vec<ProverPlan> = Vec::new();

    for (i, (task_id, ob)) in obligations.iter().enumerate() {
        match &ob.payload {
            ObligationPayload::Lec(lec) => {
                let lhs_text = lec.lhs.file.text.clone();
                let rhs_text = lec.rhs.file.text.clone();
                let lhs_path = write_text(&tempdir, &format!("ob{}_lhs.x", i), &lhs_text)?;
                let rhs_path = write_text(&tempdir, &format!("ob{}_rhs.x", i), &rhs_text)?;

                let lhs_uf: Vec<String> = lec
                    .lhs
                    .uf_map
                    .iter()
                    .map(|(f, u)| format!("{}:{}", f, u))
                    .collect();
                let rhs_uf: Vec<String> = lec
                    .rhs
                    .uf_map
                    .iter()
                    .map(|(f, u)| format!("{}:{}", f, u))
                    .collect();

                let mut cfg = DslxEquivConfig {
                    lhs_dslx_file: lhs_path,
                    rhs_dslx_file: rhs_path,
                    lhs_dslx_top: Some(lec.lhs.top_func.clone()),
                    rhs_dslx_top: Some(lec.rhs.top_func.clone()),
                    solver: solver.clone(),
                    flatten_aggregates: Some(false),
                    assertion_semantics: Some(AssertionSemantics::Ignore),
                    json: None,
                    ..Default::default()
                };
                cfg_with_paths(&mut cfg, dslx_stdlib_path, dslx_paths);
                if !lhs_uf.is_empty() {
                    cfg.lhs_uf = Some(lhs_uf);
                }
                if !rhs_uf.is_empty() {
                    cfg.rhs_uf = Some(rhs_uf);
                }

                tasks.push(ProverPlan::Task {
                    task: ProverTask::DslxEquiv { config: cfg },
                    timeout_ms,
                    task_id: Some(task_id.clone()),
                });
            }
            ObligationPayload::QuickCheck(qc) => {
                let qc_text = qc.file.text.clone();
                let qc_path = write_text(&tempdir, &format!("ob{}_qc.x", i), &qc_text)?;

                let test_filter = if qc.tests.is_empty() {
                    None
                } else {
                    let mut tests = qc.tests.clone();
                    tests.sort();
                    let escaped: Vec<String> = tests.iter().map(|t| escape(t)).collect();
                    let pattern = if escaped.len() == 1 {
                        format!("^{}$", escaped[0])
                    } else {
                        format!("^({})$", escaped.join("|"))
                    };
                    Some(pattern)
                };

                let uf: Vec<String> = qc
                    .uf_map
                    .iter()
                    .map(|(f, u)| format!("{}:{}", f, u))
                    .collect();

                let cfg = ProveQuickcheckConfig {
                    dslx_input_file: qc_path,
                    dslx_path: if dslx_paths.is_empty() {
                        None
                    } else {
                        Some(dslx_paths.iter().map(|p| (*p).to_path_buf()).collect())
                    },
                    dslx_stdlib_path: dslx_stdlib_path.map(|p| p.to_path_buf()),
                    test_filter,
                    solver: solver.clone(),
                    assertion_semantics: None,
                    uf: if uf.is_empty() { None } else { Some(uf) },
                    assert_label_filter: None,
                    json: None,
                };

                tasks.push(ProverPlan::Task {
                    task: ProverTask::ProveQuickcheck { config: cfg },
                    timeout_ms,
                    task_id: Some(task_id.clone()),
                });
            }
        }
    }

    let plan = ProverPlan::Group {
        kind: GroupKind::All,
        tasks,
        keep_running_till_finish: false,
    };
    if let ProverPlan::Group { tasks, .. } = &plan {
        log::debug!("plan_tasks={}", tasks.len());
    }
    Ok(ObligationPlan { plan, tempdir })
}
