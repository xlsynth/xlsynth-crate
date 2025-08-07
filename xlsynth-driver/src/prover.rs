// SPDX-License-Identifier: Apache-2.0

//! Process-based scheduler for executing ProverPlan trees.
//!
//! - Executes leaf ProverTasks as external processes using their driver
//!   cmdlines.
//! - Respects a global concurrency limit ("cores").
//! - Group semantics:
//!   - First: resolve on first child completion (success or failure); cancel
//!     siblings.
//!   - Any:   resolve true on first successful child; cancel siblings; resolve
//!     false if all children complete and none succeeded.
//!   - All:   resolve when all children succeed; if any fails, resolve failure
//!     and cancel siblings.
//!
//! Implementation detail: strictly uses processes; no threads. Unix-only.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::prover_config::{GroupKind, ProverPlan, ProverTask, ToDriverCommand};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

use log::{debug, info, trace, warn};
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::low_level as siglow;
use std::os::unix::process::CommandExt;
use std::process::Stdio;

use serde::Serialize;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
pub enum TaskOutcome {
    Success,
    Failed,
    Canceled,
}

impl TaskOutcome {
    fn is_success(self) -> bool {
        matches!(self, TaskOutcome::Success)
    }

    fn finish(success: bool) -> Self {
        if success {
            TaskOutcome::Success
        } else {
            TaskOutcome::Failed
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum TaskState {
    NotStarted,
    Running { pid: i32 },
    Completed { outcome: TaskOutcome },
}

#[derive(Debug)]
struct TaskNode {
    task: ProverTask,
    state: TaskState,
    // Captured outputs after task completion; None until completed.
    completed_stdout: Option<String>,
    completed_stderr: Option<String>,
}

#[derive(Debug)]
struct GroupNode {
    kind: GroupKind,
    children: Vec<NodeId>,
    outcome: Option<TaskOutcome>,
    // When true, resolved groups do not cancel or prune sibling subtrees; all
    // descendants are allowed to finish naturally.
    keep_running_till_finish: bool,
    // For GroupKind::First when keep_running_till_finish=true, remember the
    // first child's outcome, then finalize only after all children complete.
    first_winner_outcome: Option<TaskOutcome>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct NodeId(usize);

#[derive(Debug)]
enum NodeInner {
    Task(TaskNode),
    Group(GroupNode),
}

#[derive(Debug)]
struct Node {
    id: NodeId,
    parent: Option<NodeId>,
    inner: NodeInner,
}

// Cancellation is scoped per run via an Arc<AtomicBool> passed into the
// scheduler.

#[derive(Debug)]
/// Invariants:
/// - `nodes` indexed by `NodeId.0` (stable for lifetime).
/// - `root` is a valid index into `nodes`.
/// - `running ⊆ task_ids`; contains only tasks in `TaskState::Running`.
/// - `pid_to_node` maps PIDs of spawned, unreaped children to their task; may
///   include canceled-but-not-yet-reaped tasks.
/// - `round_robin` contains each task at most once; no terminated tasks.
/// - `max_procs >= 1`.
/// - `json_files` holds a temp file for a task between spawn and reap.
pub struct Scheduler {
    nodes: Vec<Node>,
    root: NodeId,
    // For quick scanning of tasks to (re)fill slots.
    task_ids: Vec<NodeId>,
    // Running task ids set for fast membership.
    running: HashSet<NodeId>,
    // pid -> task node mapping for efficient waitpid handling.
    pid_to_node: HashMap<i32, NodeId>,
    // Queue to iterate scheduling in a fair-ish manner.
    round_robin: VecDeque<NodeId>,
    // Concurrency limit.
    max_procs: usize,
    // Temp JSON files for task results, held to ensure lifetime until read.
    json_files: HashMap<NodeId, tempfile::NamedTempFile>,
    // Temp stdout/stderr files and cmdline for each running task.
    outputs: HashMap<NodeId, (tempfile::NamedTempFile, tempfile::NamedTempFile, String)>,
    // Persistent record of cmdlines for tasks (even after outputs are dropped).
    cmdlines: HashMap<NodeId, String>,
    // Run-scoped cancellation flag, toggled by CLI signal handlers or callers.
    cancel_flag: Arc<AtomicBool>,
}

#[derive(Debug)]
pub enum ProverReportNode {
    Task {
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

impl serde::Serialize for ProverReportNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        match self {
            ProverReportNode::Task {
                cmdline,
                outcome,
                stdout,
                stderr,
            } => {
                let mut s = serializer.serialize_struct("Task", 4)?;
                s.serialize_field("cmdline", cmdline)?;
                s.serialize_field("outcome", outcome)?;
                s.serialize_field("stdout", stdout)?;
                s.serialize_field("stderr", stderr)?;
                s.end()
            }
            ProverReportNode::Group {
                kind,
                outcome,
                tasks,
            } => {
                let mut s = serializer.serialize_struct("Group", 3)?;
                s.serialize_field("kind", kind)?;
                s.serialize_field("outcome", outcome)?;
                s.serialize_field("tasks", tasks)?;
                s.end()
            }
        }
    }
}

#[derive(Serialize)]
pub struct ProverReport {
    pub success: bool,
    pub plan: ProverReportNode,
}

impl Scheduler {
    /// Postconditions:
    /// - `max_procs` is clamped to at least 1.
    /// - All nodes built; `task_ids` and `round_robin` contain every task once.
    /// - `running`, `pid_to_node`, and `json_files` are empty.
    /// - `root` indexes a node inside `nodes`.
    pub fn new(plan: ProverPlan, max_procs: usize, cancel_flag: Arc<AtomicBool>) -> Scheduler {
        let mut builder = PlanBuilder::default();
        let root = builder.build_plan(None, plan);
        let nodes = builder.nodes;
        // Populate task index and rr queue.
        let mut task_ids = Vec::new();
        let mut round_robin = VecDeque::new();
        for n in &nodes {
            if matches!(n.inner, NodeInner::Task(_)) {
                task_ids.push(n.id);
                round_robin.push_back(n.id);
            }
        }
        let mut sched = Scheduler {
            nodes,
            root,
            task_ids,
            running: HashSet::new(),
            pid_to_node: HashMap::new(),
            round_robin,
            max_procs: max_procs.max(1),
            json_files: HashMap::new(),
            outputs: HashMap::new(),
            cmdlines: HashMap::new(),
            cancel_flag,
        };
        // Precompute planned cmdlines for all tasks so short-circuited tasks still
        // report one.
        for n in &sched.nodes {
            if let NodeInner::Task(t) = &n.inner {
                let cmd = t.task.to_command();
                let program = cmd.get_program().to_string_lossy().into_owned();
                let args = cmd
                    .get_args()
                    .map(|s| s.to_string_lossy().into_owned())
                    .collect::<Vec<_>>();
                let cmdline = if args.is_empty() {
                    program.clone()
                } else {
                    format!("{} {}", program, args.join(" "))
                };
                sched.cmdlines.insert(n.id, cmdline);
            }
        }
        info!(
            "prover: scheduler initialized: tasks={}, max_procs={}",
            sched.task_ids.len(),
            sched.max_procs
        );
        sched
    }

    /// Behavior:
    /// - Schedules up to `max_procs` concurrent tasks.
    /// - Returns when the root plan resolves or no running tasks remain.
    /// - On cancellation, cancels all tasks and returns `Ok(false)`.
    /// - Maintains `running.len() <= max_procs` at all times.
    pub fn run(&mut self) -> io::Result<bool> {
        loop {
            // Check for cancellation request.
            if self.cancel_flag.load(Ordering::Relaxed) {
                warn!("prover: cancellation requested; cleaning up and exiting");
                let _ = self.cleanup_tasks();
                return Ok(false);
            }

            // If the root is already resolved, finish immediately.
            if let Some(outcome) = self.node_outcome(self.root) {
                let done = outcome.is_success();
                info!("prover: plan resolved: success={}", done);
                self.cleanup_tasks()?;
                return Ok(done);
            }

            // Fill available slots next.
            if self.running.len() < self.max_procs {
                self.fill_slots()?;
            }

            if self.running.is_empty() {
                // No running processes; if root not resolved and no runnable left, consider
                // failure.
                info!("prover: no running tasks remain and root unresolved; failing");
                return Ok(false);
            }

            // Block until any child process changes state; handle one completion at a time.
            if let Err(_e) = self.wait_for_one_child_and_handle() {
                return Err(io::Error::new(io::ErrorKind::Other, "waitpid failed"));
            }
        }
    }

    /// Returns true iff the node is a `Task` in `NotStarted`.
    /// Note: Group cancellation/resolution removes descendant tasks from
    /// queues, so ancestor state is enforced by queue pruning rather than
    /// checked here.
    fn is_task_runnable(&self, task_id: NodeId) -> bool {
        let node = &self.nodes[task_id.0];
        match &node.inner {
            NodeInner::Task(t) => matches!(t.state, TaskState::NotStarted),
            _ => false,
        }
    }

    /// Returns true iff the task is `Completed`.
    fn is_task_completed(&self, task_id: NodeId) -> bool {
        let node = &self.nodes[task_id.0];
        match &node.inner {
            NodeInner::Task(t) => matches!(t.state, TaskState::Completed { .. }),
            _ => false,
        }
    }

    // Removed is_task_success and is_task_failed; prefer using
    // node_outcome(...).is_success()

    fn is_task_canceled(&self, task_id: NodeId) -> bool {
        let node = &self.nodes[task_id.0];
        match &node.inner {
            NodeInner::Task(t) => matches!(
                t.state,
                TaskState::Completed {
                    outcome: TaskOutcome::Canceled,
                }
            ),
            _ => false,
        }
    }

    /// Postconditions:
    /// - Schedules runnable tasks in round-robin order until `running ==
    ///   max_procs` or no more runnable tasks exist at this moment.
    /// - Each task is spawned at most once.
    fn fill_slots(&mut self) -> io::Result<()> {
        let len = self.round_robin.len();
        for _ in 0..len {
            if let Some(tid) = self.round_robin.pop_front() {
                if self.is_task_runnable(tid) {
                    self.spawn_task(tid)?;
                } else {
                    // This happens only when inconsistent state is left by the cleanup.
                    assert!(!self.is_task_completed(tid));
                    self.round_robin.push_back(tid);
                }
                if self.running.len() >= self.max_procs {
                    break;
                }
            } else {
                break;
            }
        }
        trace!(
            "prover: scheduling iteration complete; running={}/{}",
            self.running.len(),
            self.max_procs
        );
        Ok(())
    }

    /// Precondition: `task_id` is a `Task` in `NotStarted`.
    /// Postconditions:
    /// - Child process spawned; becomes its own process group leader.
    /// - Task state becomes `Running { pid }`.
    /// - `running` contains `task_id`; `pid_to_node[pid] == task_id`.
    /// - A temp JSON file is created and tracked in `json_files`.
    fn spawn_task(&mut self, task_id: NodeId) -> io::Result<()> {
        let node = &mut self.nodes[task_id.0];
        // Build command and tempfiles; only insert into maps after successful spawn.
        let (mut cmd, stdout_tmp, stderr_tmp, json_tmp, json_path, cmdline) = match &mut node.inner
        {
            NodeInner::Task(t) => match t.state {
                TaskState::NotStarted => {
                    let mut cmd = t.task.to_command();
                    // Create a temp path for JSON results and pass it to the child.
                    let json_tmp = tempfile::Builder::new().suffix(".json").tempfile()?;
                    let json_path = json_tmp.path().to_path_buf();
                    cmd.arg("--output_json").arg(&json_path);
                    let json_path = json_path.display().to_string();

                    // Create temp files to capture stdout/stderr; pass fds to child.
                    let stdout_tmp = tempfile::Builder::new().suffix(".stdout").tempfile()?;
                    let stderr_tmp = tempfile::Builder::new().suffix(".stderr").tempfile()?;
                    let stdout_file = stdout_tmp.as_file().try_clone()?;
                    let stderr_file = stderr_tmp.as_file().try_clone()?;
                    cmd.stdout(Stdio::from(stdout_file));
                    cmd.stderr(Stdio::from(stderr_file));

                    // Build a human-readable cmdline snapshot.
                    let program = cmd.get_program().to_string_lossy().into_owned();
                    let args = cmd
                        .get_args()
                        .map(|s| s.to_string_lossy().into_owned())
                        .collect::<Vec<_>>();
                    let cmdline = if args.is_empty() {
                        program.clone()
                    } else {
                        format!("{} {}", program, args.join(" "))
                    };

                    // Ensure child is leader of its own process group so we can kill the
                    // subtree.
                    unsafe {
                        cmd.pre_exec(|| {
                            let rc = libc::setpgid(0, 0);
                            if rc != 0 {
                                return Err(io::Error::new(
                                    io::ErrorKind::Other,
                                    "Failed to setpgid",
                                ));
                            }
                            Ok(())
                        })
                    };

                    (cmd, stdout_tmp, stderr_tmp, json_tmp, json_path, cmdline)
                }
                _ => return Ok(()),
            },
            _ => return Ok(()),
        };

        let child = cmd.spawn()?;
        let pid_i32 = child.id() as i32;

        // Now that spawn succeeded, store tempfiles and cmdline.
        self.outputs
            .insert(task_id, (stdout_tmp, stderr_tmp, cmdline.clone()));
        self.json_files.insert(task_id, json_tmp);

        if let NodeInner::Task(t) = &mut self.nodes[task_id.0].inner {
            t.state = TaskState::Running { pid: pid_i32 };
            debug!(
                "prover: spawned task, cmdline=\"{}\" pid={} json_out={}",
                self.cmdlines.get(&task_id).unwrap(),
                pid_i32,
                json_path
            );
        }
        self.running.insert(task_id);
        self.pid_to_node.insert(pid_i32, task_id);
        Ok(())
    }

    /// Ensures none of `tids` remain in `round_robin` nor `task_ids`.
    fn remove_tasks_from_queues(&mut self, tids: &[NodeId]) {
        use std::collections::HashSet;
        let to_remove: HashSet<NodeId> = tids.iter().copied().collect();
        // Rebuild round-robin without removed tasks.
        let mut new_rr = VecDeque::with_capacity(self.round_robin.len());
        while let Some(id) = self.round_robin.pop_front() {
            if !to_remove.contains(&id) {
                new_rr.push_back(id);
            }
        }
        self.round_robin = new_rr;
        // Filter task_ids.
        self.task_ids.retain(|id| !to_remove.contains(id));
    }

    /// Convenience wrapper around `remove_tasks_from_queues`.
    fn remove_task_from_queues(&mut self, tid: NodeId) {
        self.remove_tasks_from_queues(&[tid]);
    }

    /// Appends all descendant task ids of `id` into `out`. No duplicates.
    fn collect_subtree_tasks_rec(&self, id: NodeId, out: &mut Vec<NodeId>) {
        match &self.nodes[id.0].inner {
            NodeInner::Task(_) => out.push(id),
            NodeInner::Group(g) => {
                for &cid in &g.children {
                    self.collect_subtree_tasks_rec(cid, out);
                }
            }
        }
    }

    /// Removes all descendant tasks of `id` from scheduling queues only.
    fn prune_subtree_tasks(&mut self, id: NodeId) {
        let mut tids = Vec::new();
        self.collect_subtree_tasks_rec(id, &mut tids);
        self.remove_tasks_from_queues(&tids);
        trace!("prover: pruned {} tasks from queues", tids.len());
    }

    /// Reaps and processes at most one child exit.
    /// Postconditions on a reaped (non-canceled) task:
    /// - Task state -> `Completed { outcome }`.
    /// - Removed from `running`, scheduling queues, and `json_files`.
    /// - Group resolution is bubbled to ancestors as needed.
    /// Note: For canceled tasks, result does not propagate.
    fn wait_for_one_child_and_handle(&mut self) -> io::Result<()> {
        let mut status: libc::c_int = 0;
        let pid = unsafe { libc::waitpid(-1, &mut status as *mut libc::c_int, 0) };
        if pid < 0 {
            let err = io::Error::last_os_error();
            if let Some(raw) = err.raw_os_error() {
                if raw == libc::ECHILD {
                    warn!("prover: waitpid returned ECHILD; no child processes remain");
                    return Ok(());
                }
                if raw == libc::EINTR {
                    // Interrupted by a signal; give the outer loop a chance to observe CANCEL.
                    return Ok(());
                }
            }
            return Err(io::Error::new(io::ErrorKind::Other, "waitpid failed"));
        }

        // New: extract terminating signal (if any) for diagnostics
        let term_signal = if libc::WIFSIGNALED(status) {
            Some(libc::WTERMSIG(status))
        } else {
            None
        };
        if let Some(tid) = self.pid_to_node.remove(&pid) {
            // If the task was already canceled, do not convert it to Completed or bubble
            // up.
            let was_canceled = self.is_task_canceled(tid);
            if was_canceled {
                debug!("prover: reaped canceled task pid={}", pid);
                return Ok(());
            }
            // If terminated by signal, log it with context
            if let Some(sig) = term_signal {
                warn!(
                    "prover: task terminated by signal {} pid={} cmdline=\"{}\"",
                    sig,
                    pid,
                    self.cmdlines.get(&tid).unwrap()
                );
            }
            let success = self.read_success_from_json(tid);

            // Read captured stdout/stderr for this task; store for final report printing.
            let mut captured_stdout: Option<String> = None;
            let mut captured_stderr: Option<String> = None;
            if let Some((stdout_tmp, stderr_tmp, _cmdline)) = self.outputs.remove(&tid) {
                let read_lossy = |p: &std::path::Path| -> String {
                    match std::fs::read(p) {
                        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                        Err(_) => String::new(),
                    }
                };
                captured_stdout = Some(read_lossy(stdout_tmp.path()));
                captured_stderr = Some(read_lossy(stderr_tmp.path()));
            }

            if let NodeInner::Task(t) = &mut self.nodes[tid.0].inner {
                t.state = TaskState::Completed {
                    outcome: TaskOutcome::finish(success),
                };
                t.completed_stdout = captured_stdout;
                t.completed_stderr = captured_stderr;
                debug!(
                    "prover: task completed, cmdline=\"{}\" pid={} success={}",
                    self.cmdlines.get(&tid).unwrap(),
                    pid,
                    success
                );
            }
            self.remove_task_bookkeeping(tid);
            self.on_child_plan_finished(tid, success)?;
        }
        Ok(())
    }

    /// Attempts to read a task's logical success from its JSON output.
    /// Returns `false` if no file or parse error; does not mutate state.
    fn read_success_from_json(&self, tid: NodeId) -> bool {
        let path = match self.json_files.get(&tid) {
            Some(tf) => tf.path(),
            None => return false,
        };
        let data = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let v: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return false,
        };

        v.get("success").and_then(|b| b.as_bool()).unwrap_or(false)
    }

    /// Bubbles child completion into ancestor groups, possibly resolving them,
    /// canceling sibling subtrees, and pruning queues.
    fn on_child_plan_finished(&mut self, node_id: NodeId, success: bool) -> io::Result<()> {
        // Climb up towards root, resolving groups as needed.
        let mut cur_id = node_id;
        let mut parent = self.nodes[cur_id.0].parent;
        let mut cur_success = success;
        while let Some(pid) = parent {
            self.resolve_group_child(
                pid,
                cur_id,
                if cur_success {
                    TaskOutcome::Success
                } else {
                    TaskOutcome::Failed
                },
            )?;
            match self.node_outcome(pid) {
                None => break,
                Some(TaskOutcome::Success) => cur_success = true,
                Some(TaskOutcome::Failed) => cur_success = false,
                Some(TaskOutcome::Canceled) => {
                    panic!("Should not happen for now.");
                }
            }
            cur_id = pid;
            parent = self.nodes[pid.0].parent;
        }
        Ok(())
    }

    /// Precondition: `child_id` is a direct child of `gid` whose result is
    /// known. Behavior: Inspects current child outcomes, possibly resolves
    /// the group, cancels siblings according to `kind`, and prunes queues.
    fn resolve_group_child(
        &mut self,
        gid: NodeId,
        child_id: NodeId,
        child_outcome: TaskOutcome,
    ) -> io::Result<()> {
        let child_result = child_outcome.is_success();
        let (kind, idx, all_done, any_success, all_success, keep_running) = {
            match &self.nodes[gid.0].inner {
                NodeInner::Group(g) => {
                    if g.outcome.is_some() {
                        return Ok(());
                    }
                    let idx = g
                        .children
                        .iter()
                        .position(|&id| id == child_id)
                        .expect("child must be in group's children");
                    let mut all_done = true;
                    let mut any_success = false;
                    let mut all_success = true;
                    for &cid in &g.children {
                        let child_outcomes = self.node_outcome(cid);
                        match child_outcomes {
                            Some(TaskOutcome::Success) => any_success = true,
                            Some(TaskOutcome::Failed) => all_success = false,
                            Some(TaskOutcome::Canceled) => panic!("Should not happen for now."),
                            None => all_done = false,
                        }
                    }
                    (
                        g.kind,
                        idx,
                        all_done,
                        any_success,
                        all_success,
                        g.keep_running_till_finish,
                    )
                }
                _ => unreachable!(),
            }
        };
        match kind {
            GroupKind::First => {
                if !keep_running {
                    info!("prover: group resolved kind=first result={}", child_result);
                    self.finalize_group_resolution(gid, child_result, Some(idx))
                } else {
                    if let NodeInner::Group(g) = &mut self.nodes[gid.0].inner {
                        g.first_winner_outcome.get_or_insert(child_outcome);
                        if all_done {
                            let winner = g.first_winner_outcome.unwrap();
                            info!(
                                "prover: group resolved kind=first result={}",
                                winner.is_success()
                            );
                            self.finalize_group_resolution(gid, winner.is_success(), None)
                        } else {
                            Ok(())
                        }
                    } else {
                        unreachable!();
                    }
                }
            }
            GroupKind::Any => {
                if !keep_running && child_result {
                    info!("prover: group resolved kind=any result=true");
                    return self.finalize_group_resolution(gid, true, Some(idx));
                }
                if all_done {
                    info!("prover: group resolved kind=any result={}", any_success);
                    return self.finalize_group_resolution(gid, any_success, None);
                }
                Ok(())
            }
            GroupKind::All => {
                if !keep_running && !child_result {
                    info!("prover: group resolved kind=all result=false");
                    return self.finalize_group_resolution(gid, false, Some(idx));
                }
                if all_done {
                    info!("prover: group resolved kind=all result={}", all_success);
                    return self.finalize_group_resolution(gid, all_success, None);
                }
                Ok(())
            }
        }
    }

    fn finalize_group_resolution(
        &mut self,
        gid: NodeId,
        resolved: bool,
        cancel_siblings_of: Option<usize>,
    ) -> io::Result<()> {
        if let NodeInner::Group(g) = &mut self.nodes[gid.0].inner {
            g.outcome = Some(TaskOutcome::finish(resolved));
        }
        if let Some(idx) = cancel_siblings_of {
            self.cancel_group_siblings(gid, idx)?;
        }
        // Group resolved; prune all tasks in this subtree from queues.
        self.prune_subtree_tasks(gid);
        Ok(())
    }

    /// Cancels all child subtrees of the group except `exclude_child_idx`.
    ///
    /// Post: No descendant of canceled siblings remains runnable; running
    /// children are killed.
    fn cancel_group_siblings(&mut self, gid: NodeId, exclude_child_idx: usize) -> io::Result<()> {
        // Capture ids to cancel, then perform cancellations.
        let cancel_ids: Vec<NodeId> = match &self.nodes[gid.0].inner {
            NodeInner::Group(g) => g
                .children
                .iter()
                .enumerate()
                .filter_map(|(i, &cid)| {
                    if i != exclude_child_idx {
                        Some(cid)
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        };
        if !cancel_ids.is_empty() {
            warn!(
                "prover: canceling {} sibling subtree(s) for group (excluding child idx={})",
                cancel_ids.len(),
                exclude_child_idx
            );
        }
        for cid in cancel_ids.iter().copied() {
            self.cancel_subtree(cid)?;
        }
        Ok(())
    }

    fn set_node_outcome(&mut self, id: NodeId, outcome: TaskOutcome) {
        match &mut self.nodes[id.0].inner {
            NodeInner::Task(t) => t.state = TaskState::Completed { outcome },
            NodeInner::Group(g) => g.outcome = Some(outcome),
        }
    }

    /// Cancels an entire subtree rooted at `id`.
    /// Tasks:
    /// - NotStarted -> mark `Completed { Canceled }` and remove from queues.
    /// - Running -> send SIGKILL to process group, mark `Completed { Canceled
    ///   }`, remove from queues; `pid_to_node` entry remains until reaped.
    /// - Completed (Success/Failed) -> no-op bookkeeping.
    /// Groups: mark `canceled`, cancel children, then prune their tasks from
    /// queues.
    fn cancel_subtree(&mut self, id: NodeId) -> io::Result<()> {
        match &self.nodes[id.0].inner {
            NodeInner::Task(t) => {
                let prior_state = t.state.clone();
                if matches!(prior_state, TaskState::Completed { .. }) {
                    return Ok(());
                }
                self.set_node_outcome(id, TaskOutcome::Canceled);
                match prior_state {
                    TaskState::NotStarted => {
                        debug!("prover: canceled (not-started) task");
                    }
                    TaskState::Running { pid } => {
                        unsafe {
                            let _ = libc::kill(-pid, libc::SIGKILL);
                        }
                        info!("prover: canceled running task pid={}", pid);
                    }
                    TaskState::Completed { .. } => {
                        // Already completed; no-op.
                    }
                }
                self.remove_task_bookkeeping(id);
            }
            NodeInner::Group(g) => {
                let prior_outcome = g.outcome.clone();
                if matches!(prior_outcome, Some(..)) {
                    return Ok(());
                }
                let kind = g.kind;
                let children = g.children.clone();
                for cid in children {
                    self.cancel_subtree(cid)?;
                }
                self.set_node_outcome(id, TaskOutcome::Canceled);
                // Prune entire group subtree tasks from queues.
                self.prune_subtree_tasks(id);
                warn!("prover: canceled group subtree kind={:?} ", kind);
            }
        }
        Ok(())
    }

    /// Returns the outcome if the node is resolved:
    /// - Task: `Some(outcome)` when in `Completed` (may be Success, Failed, or
    ///   Canceled).
    /// - Group: `Some(outcome)` when `outcome.is_some()` (may be Success,
    ///   Failed, or Canceled).
    /// Otherwise, `None`.
    fn node_outcome(&self, id: NodeId) -> Option<TaskOutcome> {
        match &self.nodes[id.0].inner {
            NodeInner::Task(t) => match t.state {
                TaskState::Completed { outcome } => Some(outcome),
                _ => None,
            },
            NodeInner::Group(g) => g.outcome,
        }
    }

    // Small helper to clean task-related bookkeeping.
    /// Removes `id` from `running` and scheduling queues; drops any temp JSON
    /// file. Does not remove any `pid_to_node` entry; that occurs on reap.
    fn remove_task_bookkeeping(&mut self, id: NodeId) {
        self.running.remove(&id);
        self.remove_task_from_queues(id);
        let _ = self.json_files.remove(&id);
        let _ = self.outputs.remove(&id);
    }

    fn build_report_node(&self, id: NodeId) -> ProverReportNode {
        match &self.nodes[id.0].inner {
            NodeInner::Task(t) => {
                let cmdline = self.cmdlines.get(&id).cloned();
                let outcome = self.node_outcome(id);
                let stdout = t.completed_stdout.clone();
                let stderr = t.completed_stderr.clone();
                ProverReportNode::Task {
                    cmdline,
                    outcome,
                    stdout,
                    stderr,
                }
            }
            NodeInner::Group(g) => {
                let outcome = g.outcome;
                let tasks = g
                    .children
                    .iter()
                    .map(|&cid| self.build_report_node(cid))
                    .collect();
                ProverReportNode::Group {
                    kind: g.kind,
                    outcome,
                    tasks,
                }
            }
        }
    }

    // --- New cleanup helpers for unexpected OS errors or abrupt shutdowns. ---
    /// Best-effort teardown:
    /// - Sets global cancel flag.
    /// - Cancels the entire plan subtree.
    /// - Reaps all remaining children until none remain (blocking).
    /// After return, no child processes should remain.
    pub fn cleanup_tasks(&mut self) -> io::Result<()> {
        // Prevent any further scheduling if this instance somehow continues.
        self.cancel_flag.store(true, Ordering::Relaxed);
        info!("prover: cleanup starting (best-effort) ");
        let _ = self.cancel_subtree(self.root);
        while !self.pid_to_node.is_empty() {
            let mut status: libc::c_int = 0;
            let pid = unsafe { libc::waitpid(-1, &mut status as *mut libc::c_int, 0) };
            if pid > 0 {
                // Remove from mapping if we tracked it.
                self.pid_to_node.remove(&pid);
                trace!("prover: reaped child pid={} during cleanup", pid);
                continue;
            }
            if pid == 0 {
                // Should not happen without WNOHANG; just continue.
                continue;
            }
            // pid < 0: error
            let err = io::Error::last_os_error();
            if let Some(raw) = err.raw_os_error() {
                if raw == libc::ECHILD {
                    // No child processes remain; we're done.
                    break;
                }
                if raw == libc::EINTR {
                    // Interrupted by signal; retry.
                    continue;
                }
            }
            warn!("prover: waitpid during cleanup failed: {}", err);
            break;
        }
        warn!("prover: cleanup complete");
        Ok(())
    }
}

#[derive(Default)]
/// Invariant: `nodes[id.0].id == NodeId(id.0)` holds for all nodes.
struct PlanBuilder {
    nodes: Vec<Node>,
}

impl PlanBuilder {
    fn push(&mut self, parent: Option<NodeId>, inner: NodeInner) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { id, parent, inner });
        id
    }

    /// Builds the tree for `plan` under `parent`.
    /// Post: For `Group` nodes, `children.len()` is set to the number of tasks.
    fn build_plan(&mut self, parent: Option<NodeId>, plan: ProverPlan) -> NodeId {
        match plan {
            ProverPlan::Task { task } => self.push(
                parent,
                NodeInner::Task(TaskNode {
                    task,
                    state: TaskState::NotStarted,
                    completed_stdout: None,
                    completed_stderr: None,
                }),
            ),
            ProverPlan::Group {
                kind,
                tasks,
                keep_running_till_finish,
            } => {
                let gid = self.push(
                    parent,
                    NodeInner::Group(GroupNode {
                        kind,
                        children: Vec::new(),
                        outcome: None,
                        keep_running_till_finish,
                        first_winner_outcome: None,
                    }),
                );
                let child_ids: Vec<NodeId> = tasks
                    .into_iter()
                    .map(|t| self.build_plan(Some(gid), t))
                    .collect();
                if let NodeInner::Group(g) = &mut self.nodes[gid.0].inner {
                    g.children = child_ids;
                }
                gid
            }
        }
    }
}

/// Convenience entry point: run a `ProverPlan` with up to `max_procs`
/// concurrent processes.
/// On internal error, performs cleanup before returning `Err`.
pub fn run_prover_plan(plan: ProverPlan, max_procs: usize) -> io::Result<ProverReport> {
    // Create a run-scoped cancellation flag and wire OS signals to it.
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut sig_ids = Vec::new();
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        let flag = Arc::clone(&cancel_flag);
        // Set the flag from signal handler; store SigId so we can unregister on exit.
        match unsafe { siglow::register(sig, move || flag.store(true, Ordering::Relaxed)) } {
            Ok(id) => sig_ids.push(id),
            Err(e) => warn!("prover: failed to register signal {}: {}", sig, e),
        }
    }

    let mut sched = Scheduler::new(plan, max_procs, cancel_flag);
    info!("prover: run starting");
    let run_res = sched.run();

    // Always unregister signal handlers before returning.
    for id in sig_ids.drain(..) {
        let _ = siglow::unregister(id);
    }

    let result = match run_res {
        Ok(v) => v,
        Err(e) => {
            let _ = sched.cleanup_tasks();
            return Err(e);
        }
    };
    let plan_node = sched.build_report_node(sched.root);
    Ok(ProverReport {
        success: result,
        plan: plan_node,
    })
}

/// Implements the `prover` sub-command.
/// CLI behavior:
/// - Installs signal handlers that set a global cancel flag.
/// - Reads the plan JSON from a file or stdin.
/// - Parses and executes the plan with the requested concurrency.
/// Process exits with status 0 on success, 1 on failure.
pub fn handle_prover(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let cores: usize = matches
        .get_one::<String>("cores")
        .map(|s| s.parse::<usize>().unwrap_or(1))
        .unwrap_or(1)
        .max(1);
    let plan_path = matches
        .get_one::<String>("plan_json_file")
        .expect("plan_json_file arg missing");
    let output_json_path = matches
        .get_one::<String>("output_json")
        .map(|s| s.to_string());

    info!(
        "prover: starting with cores={} plan_source= {}",
        cores,
        if plan_path == "-" { "stdin" } else { plan_path }
    );

    let plan_json = if plan_path == "-" {
        use std::io::Read;
        let mut buf = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
            let err_msg = e.to_string();
            report_cli_error_and_exit(
                "Failed to read plan from stdin",
                Some("prover"),
                vec![("error", &err_msg)],
            );
        }
        buf
    } else {
        match std::fs::read_to_string(plan_path) {
            Ok(s) => s,
            Err(e) => {
                let err_msg = e.to_string();
                report_cli_error_and_exit(
                    "Failed to read plan JSON file",
                    Some("prover"),
                    vec![("path", plan_path), ("error", &err_msg)],
                );
            }
        }
    };

    let plan: ProverPlan = match serde_json::from_str(&plan_json) {
        Ok(p) => p,
        Err(e) => {
            let err_msg = e.to_string();
            report_cli_error_and_exit(
                "Failed to parse ProverPlan JSON",
                Some("prover"),
                vec![("error", &err_msg)],
            );
        }
    };

    match run_prover_plan(plan, cores) {
        Ok(report) => {
            if let Some(path) = &output_json_path {
                // Write full report including plan tree and captured outputs; do not print
                // per-task outputs.
                let s = serde_json::to_string(&report).unwrap();
                if let Err(e) = std::fs::write(path, s) {
                    warn!("prover: failed writing output_json to {}: {}", path, e);
                }
            } else {
                // No JSON requested; print all task outputs together at the end in DFS order.
                fn print_task_outputs(node: &ProverReportNode) {
                    match node {
                        ProverReportNode::Task {
                            cmdline,
                            stdout,
                            stderr,
                            ..
                        } => {
                            println!("{}", "-".repeat(80));
                            println!("{}", cmdline.as_deref().unwrap_or(""));
                            let out = stdout.as_deref().unwrap_or("");
                            let err = stderr.as_deref().unwrap_or("");
                            if out.is_empty() {
                                println!(">>>> stdout:");
                            } else {
                                println!(">>>> stdout:\n{}", out);
                            }
                            if err.is_empty() {
                                println!(">>>> stderr:");
                            } else {
                                println!(">>>> stderr:\n{}", err);
                            }
                            println!("{}", "-".repeat(80));
                        }
                        ProverReportNode::Group { tasks, .. } => {
                            for t in tasks {
                                print_task_outputs(t);
                            }
                        }
                    }
                }
                print_task_outputs(&report.plan);
            }
            if report.success {
                println!("Overall: success");
                std::process::exit(0)
            } else {
                println!("Overall: failure");
                std::process::exit(1)
            }
        }
        Err(e) => {
            let err_msg = e.to_string();
            report_cli_error_and_exit(
                "prover run failed",
                Some("prover"),
                vec![("error", &err_msg)],
            )
        }
    }
}
