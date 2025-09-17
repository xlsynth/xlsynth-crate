// SPDX-License-Identifier: Apache-2.0

//! Prover configuration helpers.
//!
//! This module defines configuration structs for invoking xlsynth-driver
//! subcommands programmatically. Each configuration translates to a complete
//! command line for the driver, suitable for running in parallel processes.
//!
//! JSON DSL overview
//! - A single task is an object with a `kind` discriminator (`ir-equiv`,
//!   `dslx-equiv`, `prove-quickcheck`).
//! - Tasks can be composed into groups with `kind` = `all` | `any` | `first`
//!   and a `tasks` array.
//!
//! Example (single task):
//!
//! ```json
//! { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir", "top": "main" }
//! ```
//!
//! See the driver README section "Prover configuration JSON (task-spec DSL)"
//! for the full schema and examples.

use std::path::PathBuf;
use std::process::Command;

use crate::parallelism::ParallelismStrategy;
use crate::solver_choice::SolverChoice;
use serde::{Deserialize, Serialize};
use xlsynth_prover::types::AssertionSemantics;
use xlsynth_prover::types::QuickCheckAssertionSemantics;

fn add_flag(cmd: &mut Command, name: &str, value: &str) {
    cmd.arg(format!("--{}", name)).arg(value);
}

fn add_bool(cmd: &mut Command, name: &str, value: Option<bool>) {
    if let Some(v) = value {
        add_flag(cmd, name, if v { "true" } else { "false" });
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IrEquivConfig {
    pub lhs_ir_file: PathBuf,
    pub rhs_ir_file: PathBuf,
    /// --top if both sides have same name, otherwise use lhs_ir_top/rhs_ir_top
    pub top: Option<String>,
    pub lhs_ir_top: Option<String>,
    pub rhs_ir_top: Option<String>,

    pub solver: Option<SolverChoice>,
    pub flatten_aggregates: Option<bool>,
    pub drop_params: Option<Vec<String>>, // comma joined
    pub parallelism_strategy: Option<ParallelismStrategy>,
    pub assertion_semantics: Option<AssertionSemantics>,
    pub lhs_fixed_implicit_activation: Option<bool>,
    pub rhs_fixed_implicit_activation: Option<bool>,
    /// Include only assertions whose label matches this regex.
    pub assert_label_filter: Option<String>,

    pub json: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DslxEquivConfig {
    pub lhs_dslx_file: PathBuf,
    pub rhs_dslx_file: PathBuf,

    /// Either set `dslx_top` or both `lhs_dslx_top` and `rhs_dslx_top`.
    pub dslx_top: Option<String>,
    pub lhs_dslx_top: Option<String>,
    pub rhs_dslx_top: Option<String>,

    // Optional search paths
    pub dslx_path: Option<Vec<PathBuf>>, // joined with ';'
    pub dslx_stdlib_path: Option<PathBuf>,

    // Behavior flags
    pub solver: Option<SolverChoice>,
    pub flatten_aggregates: Option<bool>,
    pub drop_params: Option<Vec<String>>, // comma joined
    pub parallelism_strategy: Option<ParallelismStrategy>,
    pub assertion_semantics: Option<AssertionSemantics>,
    pub lhs_fixed_implicit_activation: Option<bool>,
    pub rhs_fixed_implicit_activation: Option<bool>,
    pub assume_enum_in_bound: Option<bool>,
    pub type_inference_v2: Option<bool>, // external toolchain only
    /// Include only assertions whose label matches this regex.
    pub assert_label_filter: Option<String>,

    /// Treat DSLX function on the LHS/RHS as uninterpreted function (repeated).
    /// Each entry is "func_name:uf_name".
    pub lhs_uf: Option<Vec<String>>, // each entry is "func_name:uf_name"
    pub rhs_uf: Option<Vec<String>>, // each entry is "func_name:uf_name"

    pub json: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProveQuickcheckConfig {
    pub dslx_input_file: PathBuf,
    pub test_filter: Option<String>,
    pub solver: Option<SolverChoice>,
    pub assertion_semantics: Option<QuickCheckAssertionSemantics>,
    /// Treat DSLX function as uninterpreted function (repeated), entries
    /// "func_name:uf_name".
    pub uf: Option<Vec<String>>,
    /// Include only assertions whose label matches this regex.
    pub assert_label_filter: Option<String>,
    pub json: Option<bool>,
}

pub trait ToDriverCommand {
    fn to_command(&self) -> Command;
}

impl ToDriverCommand for IrEquivConfig {
    fn to_command(&self) -> Command {
        let exe = std::env::current_exe().expect("resolve current exe");
        let mut cmd = Command::new(exe);
        cmd.arg("ir-equiv");
        cmd.arg(self.lhs_ir_file.as_os_str());
        cmd.arg(self.rhs_ir_file.as_os_str());
        if let Some(top) = &self.top {
            add_flag(&mut cmd, "top", top);
        }
        if let Some(lhs_top) = &self.lhs_ir_top {
            add_flag(&mut cmd, "lhs_ir_top", lhs_top);
        }
        if let Some(rhs_top) = &self.rhs_ir_top {
            add_flag(&mut cmd, "rhs_ir_top", rhs_top);
        }
        if let Some(solver) = &self.solver {
            add_flag(&mut cmd, "solver", &solver.to_string());
        }
        add_bool(&mut cmd, "flatten_aggregates", self.flatten_aggregates);
        if let Some(drop) = &self.drop_params {
            if !drop.is_empty() {
                add_flag(&mut cmd, "drop_params", &drop.join(","));
            }
        }
        if let Some(strategy) = &self.parallelism_strategy {
            add_flag(&mut cmd, "parallelism-strategy", &strategy.to_string());
        }
        if let Some(sem) = &self.assertion_semantics {
            add_flag(&mut cmd, "assertion-semantics", &sem.to_string());
        }
        if let Some(pat) = &self.assert_label_filter {
            add_flag(&mut cmd, "assert-label-filter", pat);
        }
        add_bool(
            &mut cmd,
            "lhs_fixed_implicit_activation",
            self.lhs_fixed_implicit_activation,
        );
        add_bool(
            &mut cmd,
            "rhs_fixed_implicit_activation",
            self.rhs_fixed_implicit_activation,
        );
        add_bool(&mut cmd, "json", self.json);
        cmd
    }
}

impl ToDriverCommand for DslxEquivConfig {
    fn to_command(&self) -> Command {
        let exe = std::env::current_exe().expect("resolve current exe");
        let mut cmd = Command::new(exe);
        cmd.arg("dslx-equiv");
        cmd.arg(self.lhs_dslx_file.as_os_str());
        cmd.arg(self.rhs_dslx_file.as_os_str());

        if let Some(top) = &self.dslx_top {
            add_flag(&mut cmd, "dslx_top", top);
        }
        if let Some(lhs_top) = &self.lhs_dslx_top {
            add_flag(&mut cmd, "lhs_dslx_top", lhs_top);
        }
        if let Some(rhs_top) = &self.rhs_dslx_top {
            add_flag(&mut cmd, "rhs_dslx_top", rhs_top);
        }
        if let Some(stdlib) = &self.dslx_stdlib_path {
            add_flag(&mut cmd, "dslx_stdlib_path", &stdlib.display().to_string());
        }
        if let Some(paths) = &self.dslx_path {
            if !paths.is_empty() {
                let joined = paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(";");
                add_flag(&mut cmd, "dslx_path", &joined);
            }
        }

        if let Some(solver) = &self.solver {
            add_flag(&mut cmd, "solver", &solver.to_string());
        }
        add_bool(&mut cmd, "flatten_aggregates", self.flatten_aggregates);
        if let Some(drop) = &self.drop_params {
            if !drop.is_empty() {
                add_flag(&mut cmd, "drop_params", &drop.join(","));
            }
        }
        if let Some(strategy) = &self.parallelism_strategy {
            add_flag(&mut cmd, "parallelism-strategy", &strategy.to_string());
        }
        if let Some(sem) = &self.assertion_semantics {
            add_flag(&mut cmd, "assertion-semantics", &sem.to_string());
        }
        if let Some(pat) = &self.assert_label_filter {
            add_flag(&mut cmd, "assert-label-filter", pat);
        }
        add_bool(
            &mut cmd,
            "lhs_fixed_implicit_activation",
            self.lhs_fixed_implicit_activation,
        );
        add_bool(
            &mut cmd,
            "rhs_fixed_implicit_activation",
            self.rhs_fixed_implicit_activation,
        );
        add_bool(&mut cmd, "assume-enum-in-bound", self.assume_enum_in_bound);
        add_bool(&mut cmd, "type_inference_v2", self.type_inference_v2);

        if let Some(list) = &self.lhs_uf {
            for entry in list {
                add_flag(&mut cmd, "lhs_uf", entry);
            }
        }
        if let Some(list) = &self.rhs_uf {
            for entry in list {
                add_flag(&mut cmd, "rhs_uf", entry);
            }
        }

        add_bool(&mut cmd, "json", self.json);
        cmd
    }
}

impl ToDriverCommand for ProveQuickcheckConfig {
    fn to_command(&self) -> Command {
        let exe = std::env::current_exe().expect("resolve current exe");
        let mut cmd = Command::new(exe);
        cmd.arg("prove-quickcheck");
        add_flag(
            &mut cmd,
            "dslx_input_file",
            &self.dslx_input_file.display().to_string(),
        );
        if let Some(filter) = &self.test_filter {
            add_flag(&mut cmd, "test_filter", filter);
        }
        if let Some(solver) = &self.solver {
            add_flag(&mut cmd, "solver", &solver.to_string());
        }
        if let Some(sem) = &self.assertion_semantics {
            add_flag(&mut cmd, "assertion-semantics", &sem.to_string());
        }
        if let Some(list) = &self.uf {
            for entry in list {
                add_flag(&mut cmd, "uf", entry);
            }
        }
        if let Some(pat) = &self.assert_label_filter {
            add_flag(&mut cmd, "assert-label-filter", pat);
        }
        add_bool(&mut cmd, "json", self.json);
        cmd
    }
}

// -------------------------------
// Fake task definition (available for fuzzing, tests, and local use)
// -------------------------------
#[cfg(feature = "enable-fake-task")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FakeTaskConfig {
    /// Optional delay before the fake task reports a result.
    /// - When `Some(d)`, sleeps for `d` milliseconds then exits and writes
    ///   JSON.
    /// - When `None`, sleeps indefinitely and never finishes (until
    ///   canceled/timeout).
    pub delay_ms: Option<u32>,
    /// Whether the task should report success.
    pub success: bool,
    /// Number of bytes to write to stdout.
    pub stdout_len: u16,
    /// Number of bytes to write to stderr.
    pub stderr_len: u16,
}

#[cfg(feature = "enable-fake-task")]
impl ToDriverCommand for FakeTaskConfig {
    fn to_command(&self) -> Command {
        // Use a POSIX shell snippet to sleep and then locate the --output_json path
        // from argv.
        let script = r#"
delay_ms="${FAKE_DELAY_MS:-}"
# If delay_ms is empty, sleep indefinitely (never complete).
if [ -z "$delay_ms" ]; then
  # Sleep forever to simulate a non-terminating task; will be killed on timeout/cancel.
  tail -f /dev/null
fi
# Otherwise, sleep for the requested milliseconds.
secs=$(printf "%s" "$delay_ms" | awk '{ printf("%.3f", $1/1000.0) }')
sleep "$secs"
# Emit stdout noise
ol=${FAKE_STDOUT_LEN:-0}
if [ "$ol" -gt 0 ]; then head -c "$ol" < /dev/zero | tr '\0' 'X'; echo; fi
# Emit stderr noise
el=${FAKE_STDERR_LEN:-0}
if [ "$el" -gt 0 ]; then (head -c "$el" < /dev/zero | tr '\0' 'Y'; echo) >&2; fi
# Parse --output_json DEST from arguments
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--output_json" ]; then
    shift
    out="$1"
    break
  fi
  shift
done
if [ -n "$out" ]; then
  if [ "${FAKE_SUCCESS:-1}" -eq 1 ]; then
    printf '{\"success\":true}\n' > "$out"
  else
    printf '{\"success\":false}\n' > "$out"
  fi
fi
"#;
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg(script).arg("sh");
        if let Some(ms) = self.delay_ms {
            cmd.env("FAKE_DELAY_MS", ms.to_string());
        }
        cmd.env("FAKE_SUCCESS", if self.success { "1" } else { "0" });
        cmd.env("FAKE_STDOUT_LEN", self.stdout_len.to_string());
        cmd.env("FAKE_STDERR_LEN", self.stderr_len.to_string());
        cmd
    }
}

// -----------------------------------------------------------------------------
// Unified enum for collections of prover tasks
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ProverTask {
    #[serde(rename = "ir-equiv")]
    IrEquiv {
        #[serde(flatten)]
        config: IrEquivConfig,
    },
    #[serde(rename = "dslx-equiv")]
    DslxEquiv {
        #[serde(flatten)]
        config: DslxEquivConfig,
    },
    #[serde(rename = "prove-quickcheck")]
    ProveQuickcheck {
        #[serde(flatten)]
        config: ProveQuickcheckConfig,
    },
    #[cfg(feature = "enable-fake-task")]
    #[serde(rename = "fake")]
    Fake {
        #[serde(flatten)]
        config: FakeTaskConfig,
    },
}

// No common fields on ProverTask variants; common per-task metadata lives on
// the ProverPlan::Task node for consolidation.

impl ToDriverCommand for ProverTask {
    fn to_command(&self) -> Command {
        match self {
            ProverTask::IrEquiv { config, .. } => config.to_command(),
            ProverTask::DslxEquiv { config, .. } => config.to_command(),
            ProverTask::ProveQuickcheck { config, .. } => config.to_command(),
            #[cfg(feature = "enable-fake-task")]
            ProverTask::Fake { config, .. } => config.to_command(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GroupKind {
    All,
    Any,
    First,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProverPlan {
    Task {
        #[serde(flatten)]
        task: ProverTask,
        /// Optional timeout for this task node.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        /// Optional user-provided identifier for tracking/reporting.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<String>,
    },
    Group {
        kind: GroupKind,
        tasks: Vec<ProverPlan>,
        // When true, do not cancel or prune sibling tasks upon this group's
        // resolution; allow them to continue running until they naturally
        // complete. Default is false.
        #[serde(default)]
        keep_running_till_finish: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_of(cmd: Command) -> Vec<String> {
        cmd.get_args()
            .map(|s| s.to_string_lossy().into_owned())
            .collect()
    }

    fn head_of(plan: &ProverPlan) -> String {
        match plan {
            ProverPlan::Task { task, .. } => args_of(task.to_command())[0].clone(),
            ProverPlan::Group { kind, .. } => match kind {
                GroupKind::All => "all".to_string(),
                GroupKind::Any => "any".to_string(),
                GroupKind::First => "first".to_string(),
            },
        }
    }

    #[test]
    fn test_ir_equiv_to_command_args() {
        let cfg = IrEquivConfig {
            lhs_ir_file: "lhs.ir".into(),
            rhs_ir_file: "rhs.ir".into(),
            top: Some("main".to_string()),
            solver: Some(SolverChoice::Toolchain),
            parallelism_strategy: Some(ParallelismStrategy::OutputBits),
            assertion_semantics: Some(AssertionSemantics::Same),
            flatten_aggregates: Some(true),
            drop_params: Some(vec!["p0".into(), "p1".into()]),
            json: Some(true),
            ..Default::default()
        };
        let got = args_of(cfg.to_command());
        assert_eq!(
            got,
            vec![
                "ir-equiv",
                "lhs.ir",
                "rhs.ir",
                "--top",
                "main",
                "--solver",
                "toolchain",
                "--flatten_aggregates",
                "true",
                "--drop_params",
                "p0,p1",
                "--parallelism-strategy",
                "output-bits",
                "--assertion-semantics",
                "same",
                "--json",
                "true",
            ]
        );
    }

    #[test]
    fn test_dslx_equiv_to_command_args() {
        let cfg = DslxEquivConfig {
            lhs_dslx_file: "lhs.x".into(),
            rhs_dslx_file: "rhs.x".into(),
            dslx_top: Some("foo".into()),
            dslx_path: Some(vec!["a".into(), "b".into()]),
            dslx_stdlib_path: Some("stdlib".into()),
            solver: Some(SolverChoice::Toolchain),
            assertion_semantics: Some(AssertionSemantics::Assume),
            parallelism_strategy: Some(ParallelismStrategy::SingleThreaded),
            flatten_aggregates: Some(false),
            lhs_fixed_implicit_activation: Some(true),
            rhs_fixed_implicit_activation: Some(false),
            assume_enum_in_bound: Some(true),
            json: Some(true),
            ..Default::default()
        };
        let got = args_of(cfg.to_command());
        assert_eq!(
            got,
            vec![
                "dslx-equiv",
                "lhs.x",
                "rhs.x",
                "--dslx_top",
                "foo",
                "--dslx_stdlib_path",
                "stdlib",
                "--dslx_path",
                "a;b",
                "--solver",
                "toolchain",
                "--flatten_aggregates",
                "false",
                "--parallelism-strategy",
                "single-threaded",
                "--assertion-semantics",
                "assume",
                "--lhs_fixed_implicit_activation",
                "true",
                "--rhs_fixed_implicit_activation",
                "false",
                "--assume-enum-in-bound",
                "true",
                "--json",
                "true",
            ]
        );
    }

    #[test]
    fn test_prove_quickcheck_to_command_args() {
        let cfg = ProveQuickcheckConfig {
            dslx_input_file: "qc.x".into(),
            test_filter: Some(".*mytest".into()),
            solver: Some(SolverChoice::Toolchain),
            assertion_semantics: Some(QuickCheckAssertionSemantics::Assume),
            json: Some(true),
            ..Default::default()
        };
        let got = args_of(cfg.to_command());
        assert_eq!(
            got,
            vec![
                "prove-quickcheck",
                "--dslx_input_file",
                "qc.x",
                "--test_filter",
                ".*mytest",
                "--solver",
                "toolchain",
                "--assertion-semantics",
                "assume",
                "--json",
                "true",
            ]
        );
    }

    #[test]
    fn test_serde_ir_equiv_parse() {
        let json = r#"{
            "lhs_ir_file": "lhs.ir",
            "rhs_ir_file": "rhs.ir",
            "top": "main",
            "solver": "toolchain",
            "parallelism_strategy": "input-bit-split",
            "assertion_semantics": "never",
            "json": true
        }"#;
        let cfg: IrEquivConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.solver, Some(SolverChoice::Toolchain));
        assert_eq!(
            cfg.parallelism_strategy,
            Some(ParallelismStrategy::InputBitSplit)
        );
        assert_eq!(cfg.assertion_semantics, Some(AssertionSemantics::Never));
        assert_eq!(cfg.json, Some(true));
    }

    #[test]
    fn test_prover_task_json_roundtrip() {
        let json = r#"[
          {
            "kind": "ir-equiv",
            "lhs_ir_file": "lhs.ir",
            "rhs_ir_file": "rhs.ir",
            "top": "main",
            "solver": "toolchain",
            "json": true
          },
          {
            "kind": "dslx-equiv",
            "lhs_dslx_file": "lhs.x",
            "rhs_dslx_file": "rhs.x",
            "dslx_top": "foo",
            "solver": "toolchain",
            "json": true
          },
          {
            "kind": "prove-quickcheck",
            "dslx_input_file": "qc.x",
            "assertion_semantics": "assume",
            "solver": "toolchain",
            "json": true
          }
        ]"#;
        let tasks: Vec<ProverTask> = serde_json::from_str(json).unwrap();
        assert_eq!(tasks.len(), 3);
        // Smoke test command arg heads
        let head: Vec<String> = args_of(tasks[0].to_command());
        assert_eq!(head[0], "ir-equiv");
        let head: Vec<String> = args_of(tasks[1].to_command());
        assert_eq!(head[0], "dslx-equiv");
        let head: Vec<String> = args_of(tasks[2].to_command());
        assert_eq!(head[0], "prove-quickcheck");
    }

    // -----------------------
    // ProverPlan tests
    // -----------------------

    #[test]
    fn test_prover_plan_all_deserialize_and_heads() {
        let json = r#"{
          "kind": "all",
          "tasks": [
            { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir" },
            { "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "dslx_top": "foo" },
            { "kind": "prove-quickcheck", "dslx_input_file": "qc.x" }
          ]
        }"#;
        let plan: ProverPlan = serde_json::from_str(json).unwrap();
        match plan {
            ProverPlan::Group {
                kind: GroupKind::All,
                tasks,
                keep_running_till_finish,
            } => {
                assert_eq!(tasks.len(), 3);
                let heads: Vec<String> = tasks.iter().map(head_of).collect();
                assert_eq!(heads, vec!["ir-equiv", "dslx-equiv", "prove-quickcheck"]);
                assert!(!keep_running_till_finish);
            }
            _ => panic!("expected ProverPlan::Group(All)"),
        }
    }

    #[test]
    fn test_prover_plan_any_deserialize_and_heads() {
        let json = r#"{
          "kind": "any",
          "tasks": [
            { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir" },
            { "kind": "prove-quickcheck", "dslx_input_file": "qc.x" }
          ],
          "keep_running_till_finish": true
        }"#;
        let plan: ProverPlan = serde_json::from_str(json).unwrap();
        match plan {
            ProverPlan::Group {
                kind: GroupKind::Any,
                tasks,
                keep_running_till_finish,
            } => {
                assert_eq!(tasks.len(), 2);
                let heads: Vec<String> = tasks.iter().map(head_of).collect();
                assert_eq!(heads, vec!["ir-equiv", "prove-quickcheck"]);
                assert!(keep_running_till_finish);
            }
            _ => panic!("expected ProverPlan::Group(Any)"),
        }
    }

    #[test]
    fn test_prover_plan_first_deserialize_and_heads() {
        let json = r#"{
          "kind": "first",
          "tasks": [
            { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir" },
            { "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "dslx_top": "foo" }
          ]
        }"#;
        let plan: ProverPlan = serde_json::from_str(json).unwrap();
        match plan {
            ProverPlan::Group {
                kind: GroupKind::First,
                tasks,
                ..
            } => {
                assert_eq!(tasks.len(), 2);
                let heads: Vec<String> = tasks.iter().map(head_of).collect();
                assert_eq!(heads, vec!["ir-equiv", "dslx-equiv"]);
            }
            _ => panic!("expected ProverPlan::Group(First)"),
        }
    }

    #[test]
    fn test_prover_plan_all_serialize_shape() {
        let plan = ProverPlan::Group {
            kind: GroupKind::All,
            tasks: vec![ProverPlan::Task {
                task: ProverTask::IrEquiv {
                    config: IrEquivConfig {
                        lhs_ir_file: "lhs.ir".into(),
                        rhs_ir_file: "rhs.ir".into(),
                        ..Default::default()
                    },
                },
                timeout_ms: None,
                task_id: None,
            }],
            keep_running_till_finish: false,
        };
        let v = serde_json::to_value(&plan).unwrap();
        assert_eq!(v["kind"], "all");
        assert!(v["tasks"].is_array());
        assert_eq!(v["tasks"][0]["kind"], "ir-equiv");
        assert_eq!(v["tasks"][0]["lhs_ir_file"], "lhs.ir");
        assert_eq!(v["tasks"][0]["rhs_ir_file"], "rhs.ir");
    }

    #[test]
    fn test_prover_plan_task_with_metadata_serde() {
        // Build a task plan with metadata and ensure it serializes and parses
        // correctly.
        let plan = ProverPlan::Task {
            task: ProverTask::IrEquiv {
                config: IrEquivConfig {
                    lhs_ir_file: "lhs.ir".into(),
                    rhs_ir_file: "rhs.ir".into(),
                    ..Default::default()
                },
            },
            timeout_ms: Some(1234),
            task_id: Some("task-xyz".to_string()),
        };
        let v = serde_json::to_value(&plan).unwrap();
        assert_eq!(v["kind"], "ir-equiv");
        assert_eq!(v["lhs_ir_file"], "lhs.ir");
        assert_eq!(v["rhs_ir_file"], "rhs.ir");
        assert_eq!(v["timeout_ms"], 1234);
        assert_eq!(v["task_id"], "task-xyz");

        // Round-trip back to ProverPlan and verify fields.
        let parsed: ProverPlan = serde_json::from_value(v).unwrap();
        match parsed {
            ProverPlan::Task {
                task: ProverTask::IrEquiv { .. },
                timeout_ms,
                task_id,
            } => {
                assert_eq!(timeout_ms, Some(1234));
                assert_eq!(task_id.as_deref(), Some("task-xyz"));
            }
            _ => panic!("expected ProverPlan::Task::IrEquiv"),
        }
    }

    #[test]
    fn test_prover_plan_task_parse_metadata_from_json() {
        let json = r#"{
            "kind": "ir-equiv",
            "lhs_ir_file": "lhs.ir",
            "rhs_ir_file": "rhs.ir",
            "timeout_ms": 5000,
            "task_id": "tid-123"
        }"#;
        let plan: ProverPlan = serde_json::from_str(json).unwrap();
        match plan {
            ProverPlan::Task {
                task: ProverTask::IrEquiv { .. },
                timeout_ms,
                task_id,
            } => {
                assert_eq!(timeout_ms, Some(5000));
                assert_eq!(task_id.as_deref(), Some("tid-123"));
            }
            _ => panic!("expected ProverPlan::Task::IrEquiv"),
        }
    }

    #[test]
    fn test_prover_plan_task_serialize_and_parse_behavior() {
        let plan = ProverPlan::Task {
            task: ProverTask::IrEquiv {
                config: IrEquivConfig {
                    lhs_ir_file: "lhs.ir".into(),
                    rhs_ir_file: "rhs.ir".into(),
                    ..Default::default()
                },
            },
            timeout_ms: None,
            task_id: None,
        };
        let v = serde_json::to_value(&plan).unwrap();
        // The inner task tag is used on serialization.
        assert_eq!(v["kind"], "ir-equiv");
        // It can be parsed as a ProverTask...
        let task: ProverTask = serde_json::from_value(v.clone()).unwrap();
        match task {
            ProverTask::IrEquiv { .. } => {}
            _ => panic!("expected ProverTask::IrEquiv"),
        }
        // ...and can be parsed back as a ProverPlan::Task as well.
        let plan2: ProverPlan = serde_json::from_value(v.clone()).unwrap();
        match plan2 {
            ProverPlan::Task {
                task: ProverTask::IrEquiv { .. },
                ..
            } => {}
            _ => panic!("expected ProverPlan::Task::IrEquiv"),
        }
    }
}
