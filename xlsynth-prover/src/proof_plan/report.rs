// SPDX-License-Identifier: Apache-2.0

//! Typed outcomes and captured diagnostics from proof-plan execution.

use serde::{Deserialize, Serialize};

use crate::proof_outcome::{ProofResultHeader, ProofStatus};

use super::config::GroupKind;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Explains why a proof task did not produce a definite logical result.
pub enum IndefiniteReason {
    GroupCriteriaAlreadyMet,
    IndefiniteChildren,
    Timeout,
    Cleanup,
    /// The solver returned no proof or counterexample.
    Inconclusive,
    /// The proof command reported a failure to execute the proof.
    ExecutionError,
    /// The command did not produce a valid, internally consistent result.
    ProtocolError,
    /// The subprocess did not terminate with the result's required exit code.
    ProcessFailure,
}

impl IndefiniteReason {
    /// Whether this reason indicates a failure to execute or report a proof.
    pub(crate) fn is_execution_error(self) -> bool {
        match self {
            Self::ExecutionError | Self::ProtocolError | Self::ProcessFailure => true,
            Self::GroupCriteriaAlreadyMet
            | Self::IndefiniteChildren
            | Self::Timeout
            | Self::Cleanup
            | Self::Inconclusive => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// A task's definite proof result or reason for an indefinite result.
pub enum TaskOutcome {
    Normal { success: bool },
    Indefinite { reason: IndefiniteReason },
}

impl TaskOutcome {
    /// Returns the logical status without treating infrastructure failures as
    /// disproofs.
    pub fn proof_status(&self) -> ProofStatus {
        match self {
            Self::Normal { success: true } => ProofStatus::Proved,
            Self::Normal { success: false } => ProofStatus::Disproved,
            Self::Indefinite { reason } if reason.is_execution_error() => {
                ProofStatus::ExecutionError
            }
            Self::Indefinite { .. } => ProofStatus::Inconclusive,
        }
    }

    pub(super) fn normal(success: bool) -> Self {
        TaskOutcome::Normal { success }
    }

    pub(super) fn indefinite(reason: IndefiniteReason) -> Self {
        TaskOutcome::Indefinite { reason }
    }

    pub(super) fn success() -> Self {
        TaskOutcome::normal(true)
    }

    pub(super) fn failed() -> Self {
        TaskOutcome::normal(false)
    }

    #[allow(dead_code)]
    pub(super) fn timeout() -> Self {
        TaskOutcome::indefinite(IndefiniteReason::Timeout)
    }

    pub(super) fn is_success(&self) -> bool {
        match self {
            TaskOutcome::Normal { success } => *success,
            _ => false,
        }
    }

    pub(super) fn is_failed(&self) -> bool {
        match self {
            TaskOutcome::Normal { success } => !*success,
            _ => false,
        }
    }

    pub(super) fn is_indefinite(&self) -> bool {
        match self {
            TaskOutcome::Indefinite { .. } => true,
            _ => false,
        }
    }
}

impl From<ProofStatus> for TaskOutcome {
    fn from(status: ProofStatus) -> Self {
        match status {
            ProofStatus::Proved => Self::success(),
            ProofStatus::Disproved => Self::failed(),
            ProofStatus::Inconclusive => Self::indefinite(IndefiniteReason::Inconclusive),
            ProofStatus::ExecutionError => Self::indefinite(IndefiniteReason::ExecutionError),
        }
    }
}

impl serde::Serialize for TaskOutcome {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let label: &str = match self {
            TaskOutcome::Normal { success: true } => "Success",
            TaskOutcome::Normal { success: false } => "Failed",
            TaskOutcome::Indefinite { reason } => match reason {
                IndefiniteReason::GroupCriteriaAlreadyMet => "GroupCriteriaAlreadyMet",
                IndefiniteReason::IndefiniteChildren => "IndefiniteChildren",
                IndefiniteReason::Timeout => "Timeout",
                IndefiniteReason::Cleanup => "Cleanup",
                IndefiniteReason::Inconclusive => "Inconclusive",
                IndefiniteReason::ExecutionError => "ExecutionError",
                IndefiniteReason::ProtocolError => "ProtocolError",
                IndefiniteReason::ProcessFailure => "ProcessFailure",
            },
        };
        serializer.serialize_str(label)
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
/// A proof-plan node with its outcome and captured task output.
pub enum ProverReportNode {
    Task {
        task_id: Option<String>,
        cmdline: Option<String>,
        outcome: Option<TaskOutcome>,
        stdout: Option<String>,
        stderr: Option<String>,
    },
    Group {
        kind: GroupKind,
        outcome: Option<TaskOutcome>,
        tasks: Vec<ProverReportNode>,
    },
}

impl ProverReportNode {
    /// Returns the resolved logical result, preserving errors in unresolved
    /// groups.
    pub fn proof_status(&self) -> ProofStatus {
        let (outcome, children) = match self {
            Self::Task { outcome, .. } => (outcome, &[][..]),
            Self::Group { outcome, tasks, .. } => (outcome, tasks.as_slice()),
        };
        let status = outcome
            .map(|o| o.proof_status())
            .unwrap_or(ProofStatus::Inconclusive);
        if status == ProofStatus::Inconclusive
            && children
                .iter()
                .any(|child| child.proof_status() == ProofStatus::ExecutionError)
        {
            ProofStatus::ExecutionError
        } else {
            status
        }
    }

    /// Finds a task by its caller-assigned identifier in deterministic tree
    /// order.
    pub fn find_task_node<'a>(&'a self, task_id: &str) -> Option<&'a ProverReportNode> {
        match self {
            ProverReportNode::Task { task_id: tid, .. } => {
                if let Some(t) = tid {
                    if t == task_id {
                        return Some(self);
                    }
                }
                None
            }
            ProverReportNode::Group { tasks, .. } => {
                for t in tasks {
                    if let Some(found) = t.find_task_node(task_id) {
                        return Some(found);
                    }
                }
                None
            }
        }
    }
}

/// Overall plan outcome together with the complete report tree.
pub struct ProverReport {
    pub plan: ProverReportNode,
}

impl Serialize for ProverReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        /// Wire report with the status derived from the resolved plan.
        struct Report<'a> {
            #[serde(flatten)]
            conclusion: ProofResultHeader,
            plan: &'a ProverReportNode,
        }
        Report {
            conclusion: self.plan.proof_status().into(),
            plan: &self.plan,
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_setup_error_is_preserved_when_children_have_no_outcome() {
        let report = ProverReport {
            plan: ProverReportNode::Group {
                kind: GroupKind::All,
                outcome: Some(ProofStatus::ExecutionError.into()),
                tasks: vec![ProverReportNode::Task {
                    task_id: Some("root".into()),
                    cmdline: None,
                    outcome: None,
                    stdout: None,
                    stderr: None,
                }],
            },
        };
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::json!({
                "status": "execution-error",
                "success": false,
                "plan": {
                    "kind": "all",
                    "outcome": "ExecutionError",
                    "tasks": [{
                        "task_id": "root", "cmdline": null, "outcome": null,
                        "stdout": null, "stderr": null
                    }]
                }
            })
        );
    }
}
