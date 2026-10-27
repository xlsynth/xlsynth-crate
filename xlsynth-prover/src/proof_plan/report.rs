// SPDX-License-Identifier: Apache-2.0

//! Typed outcomes and captured diagnostics from proof-plan execution.

use serde::Serialize;

use super::config::GroupKind;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
/// Explains why a proof task did not produce a definite logical result.
pub enum IndefiniteReason {
    GroupCriteriaAlreadyMet,
    IndefiniteChildren,
    Timeout,
    Cleanup,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// A task's definite proof result or reason for an indefinite result.
pub enum TaskOutcome {
    Normal { success: bool },
    Indefinite { reason: IndefiniteReason },
}

impl TaskOutcome {
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

#[derive(Serialize)]
/// Overall plan outcome together with the complete report tree.
pub struct ProverReport {
    pub success: bool,
    pub plan: ProverReportNode,
}
