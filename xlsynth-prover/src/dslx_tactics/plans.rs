// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::dslx_tactics::obligations::{
    FileWithHistory, ObligationPayload, ProverObligation, SourceFile,
};
use crate::proof_plan::config::{
    DslxEquivConfig, GroupKind, ProveQuickcheckConfig, ProverPlan, ProverTask,
};
use crate::prover::SolverChoice;
use regex::escape;

/// Retains module identity while isolating sources with identical basenames.
fn write_obligation_source(
    dir: &tempfile::TempDir,
    subdir: &str,
    fallback_name: &str,
    source: &FileWithHistory,
) -> Result<std::path::PathBuf, String> {
    let name = match &source.base_source {
        SourceFile::Path(path) => path
            .file_name()
            .ok_or_else(|| format!("source path has no filename: {}", path.display()))?,
        SourceFile::Text(_) => std::ffi::OsStr::new(fallback_name),
    };
    let directory = dir.path().join(subdir);
    std::fs::create_dir(&directory).map_err(|e| format!("create source directory: {e}"))?;
    let path = directory.join(name);
    std::fs::write(&path, &source.text).map_err(|e| format!("write source: {e}"))?;
    Ok(path)
}

/// Executable obligation tasks and the source files they reference.
pub struct ObligationPlan {
    pub plan: ProverPlan,
    tempdir: tempfile::TempDir,
}

impl ObligationPlan {
    pub fn into_parts(self) -> (ProverPlan, tempfile::TempDir) {
        (self.plan, self.tempdir)
    }
}

/// Converts selected obligations into a process plan with retained source
/// files.
pub fn build_plan_from_obligations(
    obligations: &[(String, ProverObligation)],
    dslx_stdlib_path: Option<&Path>,
    dslx_paths: &[&Path],
    solver: Option<SolverChoice>,
    timeout_ms: Option<u64>,
) -> Result<ObligationPlan, String> {
    let tempdir = tempfile::TempDir::new().map_err(|e| format!("tempdir: {e}"))?;
    let mut tasks: Vec<ProverPlan> = Vec::new();
    // The parent has already resolved these roots, including any explicit
    // clear.
    let dslx_paths: Vec<_> = dslx_paths.iter().map(|p| p.to_path_buf()).collect();

    for (i, (task_id, ob)) in obligations.iter().enumerate() {
        match &ob.payload {
            ObligationPayload::Lec(lec) => {
                let lhs_path = write_obligation_source(
                    &tempdir,
                    &format!("ob{i}_lhs"),
                    "lhs.x",
                    &lec.lhs.file,
                )?;
                let rhs_path = write_obligation_source(
                    &tempdir,
                    &format!("ob{i}_rhs"),
                    "rhs.x",
                    &lec.rhs.file,
                )?;

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
                    dslx_path: Some(dslx_paths.clone()),
                    dslx_stdlib_path: dslx_stdlib_path.map(Path::to_path_buf),
                    solver: solver.clone(),
                    flatten_aggregates: Some(lec.options.flatten_aggregates),
                    assertion_semantics: Some(lec.options.assertion_semantics),
                    assume_enum_in_bound: Some(lec.options.assume_enum_in_bound),
                    drop_params: Some(lec.options.drop_params.clone()),
                    assert_label_filter: lec.options.assert_label_filter.clone(),
                    lhs_fixed_implicit_activation: Some(lec.lhs.fixed_implicit_activation),
                    rhs_fixed_implicit_activation: Some(lec.rhs.fixed_implicit_activation),
                    parallelism_strategy: Some(lec.options.parallelism),
                    json: None,
                    ..Default::default()
                };
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
                let qc_path =
                    write_obligation_source(&tempdir, &format!("ob{i}_qc"), "qc.x", &qc.file)?;

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
                    dslx_path: Some(dslx_paths.clone()),
                    dslx_stdlib_path: dslx_stdlib_path.map(|p| p.to_path_buf()),
                    test_filter,
                    solver: solver.clone(),
                    assertion_semantics: Some(qc.assertion_semantics),
                    uf: if uf.is_empty() { None } else { Some(uf) },
                    assert_label_filter: qc.assert_label_filter.clone(),
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
