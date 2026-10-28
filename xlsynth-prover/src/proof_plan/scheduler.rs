// SPDX-License-Identifier: Apache-2.0

//! Process-based scheduler for executing ProverPlan trees.
//!
//! - Executes leaf ProverTasks as external processes using their driver
//!   cmdlines.
//! - Respects a global concurrency limit ("cores").
//! - Group semantics:
//!   - First: resolve on the first proved or disproved child; indefinite
//!     attempts do not win the race.
//!   - Any:   resolve true on first successful child; cancel siblings; resolve
//!     false only if all children are disproved.
//!   - All:   resolve when all children succeed; if any fails, resolve failure
//!     and cancel siblings.
//!
//! Implementation detail: strictly uses processes; no threads. Unix-only.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::proof_outcome::ProofResultHeader;

use super::config::{GroupKind, ProofDriver, ProverPlan, ProverTask, resolve_driver_exe};
use super::report::{IndefiniteReason, ProverReport, ProverReportNode, TaskOutcome};

use log::{debug, info, trace, warn};
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::low_level as siglow;
use std::os::unix::process::CommandExt;
use std::process::Stdio;

use std::path::Path;

/// A validated result or an explanation of why it cannot establish a proof.
struct ProcessResult {
    outcome: TaskOutcome,
    diagnostic: Option<String>,
}

impl ProcessResult {
    /// Records a failure of the subprocess or its result protocol.
    fn error(reason: IndefiniteReason, message: impl Into<String>) -> Self {
        Self {
            outcome: TaskOutcome::indefinite(reason),
            diagnostic: Some(message.into()),
        }
    }
}

#[derive(Copy, Clone, Debug)]
/// Lifecycle state of a task's subprocess.
enum TaskState {
    NotStarted,
    Running { pid: i32 },
    Completed { outcome: TaskOutcome },
}

#[derive(Debug)]
/// A subprocess task and its scheduling state and captured output.
struct TaskNode {
    task: ProverTask,
    timeout_ms: Option<u64>,
    task_id: Option<String>,
    state: TaskState,
    // Captured outputs after task completion; None until completed.
    completed_stdout: Option<String>,
    completed_stderr: Option<String>,
}

#[derive(Debug)]
/// A group of child nodes together with its completion policy.
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
/// Stable index into the scheduler's node arena.
struct NodeId(usize);

#[derive(Debug)]
/// Task-specific or group-specific data stored at a plan node.
enum NodeInner {
    Task(TaskNode),
    Group(GroupNode),
}

#[derive(Debug)]
/// A stable plan-tree node with its parent and task or group state.
struct Node {
    id: NodeId,
    parent: Option<NodeId>,
    inner: NodeInner,
}

// Cancellation is scoped per run via an Arc<AtomicBool> passed into the
// scheduler.

#[derive(Debug)]
/// Executes a proof plan as bounded concurrent subprocesses.
///
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
    deadlines: BTreeMap<Instant, Vec<NodeId>>,
    driver: ProofDriver,
}

impl Scheduler {
    /// Postconditions:
    /// - `max_procs` is clamped to at least 1.
    /// - All nodes built; `task_ids` and `round_robin` contain every task once.
    /// - `running`, `pid_to_node`, and `json_files` are empty.
    /// - `root` indexes a node inside `nodes`.
    pub fn new(plan: ProverPlan, max_procs: usize, cancel_flag: Arc<AtomicBool>) -> Scheduler {
        Self::new_with_driver(plan, max_procs, cancel_flag, &resolve_driver_exe())
    }

    /// Builds a scheduler whose proof commands use the supplied driver
    /// executable.
    pub fn new_with_driver(
        plan: ProverPlan,
        max_procs: usize,
        cancel_flag: Arc<AtomicBool>,
        driver: impl Into<ProofDriver>,
    ) -> Scheduler {
        log::info!("prover: new scheduler");
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
            deadlines: BTreeMap::new(),
            driver: driver.into(),
        };
        // Precompute planned cmdlines for all tasks so short-circuited tasks
        // still report one.
        for n in &sched.nodes {
            if let NodeInner::Task(t) = &n.inner {
                let Ok(cmd) = sched.driver.command_for(&t.task) else {
                    // An unrepresentable request has no command line to
                    // display. Report its setup error if scheduled.
                    continue;
                };
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
    /// - On cancellation, cancels unresolved tasks and reports their outcomes.
    /// - Maintains `running.len() <= max_procs` at all times.
    pub fn run(&mut self) -> io::Result<ProverReport> {
        loop {
            // Check for cancellation request.
            if self.cancel_flag.load(Ordering::Relaxed) {
                warn!("prover: cancellation requested; cleaning up and exiting");
                self.cleanup_tasks()?;
                break;
            }

            // Check for timeouts.
            self.check_and_handle_timeouts()?;

            // If the root is already resolved, finish immediately.
            if self.node_outcome(self.root).is_some() {
                self.cleanup_tasks()?;
                break;
            }

            // Fill available slots next.
            if self.running.len() < self.max_procs {
                self.fill_slots()?;
            }

            // Launch failures can resolve the plan without creating a child.
            if self.node_outcome(self.root).is_some() {
                self.cleanup_tasks()?;
                break;
            }

            if self.running.is_empty() {
                // No running processes; if root not resolved and no runnable
                // left, consider failure.
                panic!("prover: no running tasks remain and root unresolved; should not happen");
            }

            // Block until any child process changes state; handle one
            // completion at a time.
            self.wait_for_one_child_and_handle()?;
        }
        Ok(ProverReport {
            plan: self.build_report_node(self.root),
        })
    }

    fn check_and_handle_timeouts(&mut self) -> io::Result<()> {
        while let Some(entry) = self.deadlines.first_entry() {
            let now = Instant::now();
            if now < *entry.key() {
                break;
            }
            let tids = entry.get().iter().copied().collect::<Vec<_>>();
            entry.remove();
            for tid in tids {
                log::info!("timeout task={:?}", tid);
                self.cancel_subtree(tid, IndefiniteReason::Timeout)?;
                self.on_child_plan_finished(tid)?;
            }
        }
        Ok(())
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

    fn is_task_indefinite(&self, task_id: NodeId) -> bool {
        let node = &self.nodes[task_id.0];
        match &node.inner {
            NodeInner::Task(t) => matches!(
                t.state,
                TaskState::Completed {
                    outcome: TaskOutcome::Indefinite { .. },
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
                    if let Err(error) = self.spawn_task(tid) {
                        self.finish_task(
                            tid,
                            ProcessResult::error(
                                IndefiniteReason::ExecutionError,
                                format!("failed to launch proof command: {error}"),
                            ),
                        )?;
                    }
                } else {
                    // This happens only when inconsistent state is left by the
                    // cleanup.
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
        // Build command and tempfiles; only insert into maps after successful
        // spawn.
        let (mut cmd, stdout_tmp, stderr_tmp, json_tmp, json_path, timeout_ms, cmdline) =
            match &mut node.inner {
                NodeInner::Task(t) => match t.state {
                    TaskState::NotStarted => {
                        let mut cmd = self.driver.command_for(&t.task)?;
                        // Create a temp path for JSON results and pass it to
                        // the child.
                        let json_tmp = tempfile::Builder::new().suffix(".json").tempfile()?;
                        let json_path = json_tmp.path().to_path_buf();
                        cmd.arg("--output_json").arg(&json_path);
                        let json_path = json_path.display().to_string();

                        // Create temp files to capture stdout/stderr; pass fds
                        // to child.
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

                        // Ensure child is leader of its own process group so we
                        // can kill the subtree.
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

                        let timeout_ms = t.timeout_ms;

                        (
                            cmd, stdout_tmp, stderr_tmp, json_tmp, json_path, timeout_ms, cmdline,
                        )
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
        if let Some(timeout_ms) = timeout_ms {
            let deadline = Instant::now() + Duration::from_millis(timeout_ms);
            self.deadlines.entry(deadline).or_default().push(task_id);
        }

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
        // Poll only our known children; never use waitpid(-1) to avoid reaping
        // unrelated children of the enclosing test process.
        let pids: Vec<i32> = self.pid_to_node.keys().copied().collect();
        let mut handled_any = false;

        for pid in pids {
            let mut status: libc::c_int = 0;
            let rc = unsafe { libc::waitpid(pid, &mut status as *mut libc::c_int, libc::WNOHANG) };
            if rc == 0 {
                continue; // still running
            }
            if rc < 0 {
                let err = io::Error::last_os_error();
                if let Some(raw) = err.raw_os_error() {
                    if raw == libc::EINTR {
                        continue;
                    }
                    if raw == libc::ECHILD {
                        if let Some(tid) = self.pid_to_node.remove(&pid) {
                            if self.is_task_indefinite(tid) {
                                self.capture_task_output(tid, None);
                                self.remove_task_bookkeeping(tid);
                            } else {
                                self.finish_task(tid, ProcessResult::error(
                                    IndefiniteReason::ProcessFailure,
                                    "proof command was reaped without an observable exit status",
                                ))?;
                            }
                        }
                        handled_any = true;
                        continue;
                    }
                }
                return Err(err);
            }

            // rc > 0: child state available; complete bookkeeping for this pid.
            if let Some(tid) = self.pid_to_node.remove(&pid) {
                // If the task was already indefinite, do not convert it to
                // normal or bubble up.
                let was_indefinite = self.is_task_indefinite(tid);
                if was_indefinite {
                    debug!("prover: reaped indefinite task pid={}", pid);
                    self.capture_task_output(tid, None);
                    self.remove_task_bookkeeping(tid);
                    handled_any = true;
                    continue;
                }

                let result = self.read_task_result(tid, status);
                self.finish_task(tid, result)?;
                handled_any = true;
                break; // handle one completion per call (preserve prior pacing)
            }
        }

        if !handled_any {
            // Back off briefly if no child changed state.
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 100 * 1000000,
            };
            unsafe { libc::nanosleep(&ts, &mut ts) };
        }

        Ok(())
    }

    /// Accepts a logical result only when JSON and process termination agree.
    fn read_task_result(&self, tid: NodeId, wait_status: libc::c_int) -> ProcessResult {
        if libc::WIFSIGNALED(wait_status) {
            return ProcessResult::error(
                IndefiniteReason::ProcessFailure,
                format!(
                    "proof command terminated by signal {}",
                    libc::WTERMSIG(wait_status)
                ),
            );
        }
        if !libc::WIFEXITED(wait_status) {
            return ProcessResult::error(
                IndefiniteReason::ProcessFailure,
                "proof command did not exit normally",
            );
        }
        let exit_code = libc::WEXITSTATUS(wait_status);
        if exit_code != 0 && exit_code != 1 {
            return ProcessResult::error(
                IndefiniteReason::ProcessFailure,
                format!("proof command exited with unexpected code {exit_code}"),
            );
        }
        let path = match self.json_files.get(&tid) {
            Some(tf) => tf.path(),
            None => {
                return ProcessResult::error(
                    IndefiniteReason::ProtocolError,
                    "proof command has no result file",
                );
            }
        };
        let data = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(error) => {
                return ProcessResult::error(
                    IndefiniteReason::ProtocolError,
                    format!("cannot read proof result: {error}"),
                );
            }
        };
        let report: ProofResultHeader = match serde_json::from_str(&data) {
            Ok(report) => report,
            Err(error) => {
                return ProcessResult::error(
                    IndefiniteReason::ProtocolError,
                    format!("invalid proof result JSON: {error}"),
                );
            }
        };
        if exit_code != report.status.exit_code() {
            return ProcessResult::error(
                IndefiniteReason::ProcessFailure,
                format!(
                    "proof result status {:?} contradicts exit code {exit_code}",
                    report.status
                ),
            );
        }
        ProcessResult {
            outcome: report.status.into(),
            diagnostic: None,
        }
    }

    /// Saves captured streams after reaping, including timeout/cancellation
    /// output.
    fn capture_task_output(&mut self, tid: NodeId, diagnostic: Option<String>) {
        let NodeInner::Task(task) = &mut self.nodes[tid.0].inner else {
            unreachable!("only tasks have process output");
        };
        if let Some((stdout, stderr, _)) = self.outputs.remove(&tid) {
            let read_lossy = |path: &Path| {
                std::fs::read(path)
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    .unwrap_or_else(|error| {
                        format!("prover: cannot read captured output: {error}\n")
                    })
            };
            task.completed_stdout = Some(read_lossy(stdout.path()));
            task.completed_stderr = Some(read_lossy(stderr.path()));
        }
        if let Some(diagnostic) = diagnostic {
            let stderr = task.completed_stderr.get_or_insert_with(String::new);
            if !stderr.is_empty() && !stderr.ends_with('\n') {
                stderr.push('\n');
            }
            stderr.push_str(&format!("prover: {diagnostic}\n"));
        }
    }

    /// Completes an attempt and propagates only its validated logical outcome.
    fn finish_task(&mut self, tid: NodeId, result: ProcessResult) -> io::Result<()> {
        self.set_outcome(tid, result.outcome);
        self.capture_task_output(tid, result.diagnostic);
        self.remove_task_bookkeeping(tid);
        self.on_child_plan_finished(tid)
    }

    /// Bubbles child completion into ancestor groups, possibly resolving them,
    /// canceling sibling subtrees, and pruning queues.
    fn on_child_plan_finished(&mut self, node_id: NodeId) -> io::Result<()> {
        // Climb up towards root, resolving groups as needed.
        let mut cur_id = node_id;
        let mut parent = self.nodes[cur_id.0].parent;
        while let Some(pid) = parent {
            self.resolve_group_child(pid, cur_id)?;
            cur_id = pid;
            parent = self.nodes[pid.0].parent;
        }
        Ok(())
    }

    fn get_group_node(&self, id: NodeId) -> &GroupNode {
        match &self.nodes[id.0].inner {
            NodeInner::Group(g) => g,
            _ => panic!("not a group node"),
        }
    }

    fn get_group_node_mut(&mut self, id: NodeId) -> &mut GroupNode {
        match &mut self.nodes[id.0].inner {
            NodeInner::Group(g) => g,
            _ => panic!("not a group node"),
        }
    }

    /// Precondition: `child_id` is a direct child of `gid` whose result is
    /// known. Behavior: Inspects current child outcomes, possibly resolves
    /// the group, cancels siblings according to `kind`, and prunes queues.
    fn resolve_group_child(&mut self, gid: NodeId, child_id: NodeId) -> io::Result<()> {
        let (kind, idx, num_children, finished, success, failed, indefinite, keep_running) = {
            let g = self.get_group_node(gid);
            if g.outcome.is_some() {
                return Ok(());
            }
            let num_children = g.children.len();
            let idx = g
                .children
                .iter()
                .position(|&id| id == child_id)
                .expect("child must be in group's children");
            let mut finished = 0;
            let mut success = 0;
            let mut failed = 0;
            let mut indefinite = 0;
            for &cid in &g.children {
                let child_outcomes = self.node_outcome(cid);
                match child_outcomes {
                    Some(TaskOutcome::Normal { success: true }) => {
                        finished += 1;
                        success += 1;
                    }
                    Some(TaskOutcome::Normal { success: false }) => {
                        finished += 1;
                        failed += 1;
                    }
                    Some(TaskOutcome::Indefinite { .. }) => {
                        finished += 1;
                        indefinite += 1;
                    }
                    None => {}
                }
            }
            (
                g.kind,
                idx,
                num_children,
                finished,
                success,
                failed,
                indefinite,
                g.keep_running_till_finish,
            )
        };

        // Only consider the direct child's actual resolved outcome (if any) for
        // early resolution.
        let child_outcome = self.node_outcome(child_id);

        match kind {
            GroupKind::First => {
                let g = self.get_group_node_mut(gid);
                if let Some(co) = child_outcome {
                    if !co.is_indefinite() {
                        g.first_winner_outcome.get_or_insert(co);
                    }
                }
                if finished == num_children {
                    if let Some(winner) = g.first_winner_outcome {
                        info!("prover: group resolved kind=first outcome={:?}", winner);
                        return self.finalize_group_resolution(gid, winner, None);
                    } else {
                        return self.finalize_group_resolution(
                            gid,
                            TaskOutcome::indefinite(IndefiniteReason::IndefiniteChildren),
                            None,
                        );
                    }
                }
                if !keep_running {
                    if let Some(co) = child_outcome {
                        if !co.is_indefinite() {
                            info!("prover: group resolved kind=first result={:?}", co);
                            return self.finalize_group_resolution(gid, co, Some(idx));
                        }
                    }
                }
                return Ok(());
            }
            GroupKind::Any => {
                if finished == num_children {
                    let outcome = if success > 0 {
                        TaskOutcome::success()
                    } else if indefinite > 0 {
                        TaskOutcome::indefinite(IndefiniteReason::IndefiniteChildren)
                    } else {
                        TaskOutcome::failed()
                    };
                    info!("prover: group resolved kind=any result={:?}", outcome);
                    return self.finalize_group_resolution(gid, outcome, None);
                }
                if !keep_running {
                    if let Some(co) = child_outcome {
                        if co.is_success() {
                            info!("prover: group resolved kind=any result=true");
                            return self.finalize_group_resolution(
                                gid,
                                TaskOutcome::success(),
                                Some(idx),
                            );
                        }
                    }
                }
                Ok(())
            }
            GroupKind::All => {
                if finished == num_children {
                    let outcome = if failed > 0 {
                        TaskOutcome::failed()
                    } else if indefinite > 0 {
                        TaskOutcome::indefinite(IndefiniteReason::IndefiniteChildren)
                    } else {
                        TaskOutcome::success()
                    };
                    info!("prover: group resolved kind=all result={:?}", outcome);
                    return self.finalize_group_resolution(gid, outcome, None);
                }
                if !keep_running {
                    if let Some(co) = child_outcome {
                        if co.is_failed() {
                            info!("prover: group resolved kind=all result=false");
                            return self.finalize_group_resolution(
                                gid,
                                TaskOutcome::failed(),
                                Some(idx),
                            );
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn finalize_group_resolution(
        &mut self,
        gid: NodeId,
        outcome: TaskOutcome,
        cancel_siblings_of: Option<usize>,
    ) -> io::Result<()> {
        if let NodeInner::Group(g) = &mut self.nodes[gid.0].inner {
            g.outcome = Some(outcome);
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
            self.cancel_subtree(cid, IndefiniteReason::GroupCriteriaAlreadyMet)?;
        }
        Ok(())
    }

    fn set_outcome(&mut self, id: NodeId, outcome: TaskOutcome) {
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
    fn cancel_subtree(
        &mut self,
        id: NodeId,
        indefinite_reason: IndefiniteReason,
    ) -> io::Result<()> {
        match &self.nodes[id.0].inner {
            NodeInner::Task(t) => {
                let prior_state = t.state.clone();
                if matches!(prior_state, TaskState::Completed { .. }) {
                    return Ok(());
                }
                self.set_outcome(id, TaskOutcome::indefinite(indefinite_reason));
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
                    self.cancel_subtree(cid, indefinite_reason)?;
                }
                self.set_outcome(id, TaskOutcome::indefinite(indefinite_reason));
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
        // Keep output files alive until reap so killed tasks retain
        // diagnostics.
    }

    fn build_report_node(&self, id: NodeId) -> ProverReportNode {
        match &self.nodes[id.0].inner {
            NodeInner::Task(t) => {
                let task_id = t.task_id.clone();
                let cmdline = self.cmdlines.get(&id).cloned();
                let outcome = self.node_outcome(id);
                let stdout = t.completed_stdout.clone();
                let stderr = t.completed_stderr.clone();
                ProverReportNode::Task {
                    task_id,
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
        let _ = self.cancel_subtree(self.root, IndefiniteReason::Cleanup);

        // Reap only our tracked children until none remain.
        while !self.pid_to_node.is_empty() {
            let pids: Vec<i32> = self.pid_to_node.keys().copied().collect();
            let mut reaped_any = false;

            for pid in pids {
                let mut status: libc::c_int = 0;
                let rc =
                    unsafe { libc::waitpid(pid, &mut status as *mut libc::c_int, libc::WNOHANG) };
                if rc == 0 {
                    continue;
                }
                if rc < 0 {
                    let err = io::Error::last_os_error();
                    if let Some(raw) = err.raw_os_error() {
                        if raw == libc::EINTR {
                            continue;
                        }
                        if raw == libc::ECHILD {
                            // Already reaped elsewhere; drop bookkeeping.
                            if let Some(tid) = self.pid_to_node.remove(&pid) {
                                self.capture_task_output(tid, None);
                            }
                            reaped_any = true;
                            continue;
                        }
                    }
                    warn!("prover: waitpid during cleanup failed: {}", err);
                    // Drop from bookkeeping to avoid getting stuck.
                    if let Some(tid) = self.pid_to_node.remove(&pid) {
                        self.capture_task_output(
                            tid,
                            Some(format!("waitpid during cleanup failed: {err}")),
                        );
                    }
                    reaped_any = true;
                    continue;
                }

                // rc > 0
                if let Some(tid) = self.pid_to_node.remove(&pid) {
                    self.capture_task_output(tid, None);
                }
                trace!("prover: reaped child pid={} during cleanup", pid);
                reaped_any = true;
            }

            if !reaped_any {
                // Brief sleep to avoid busy-wait; children were SIGKILLed in
                // cancel_subtree.
                let mut ts = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 100 * 1000000,
                };
                unsafe { libc::nanosleep(&ts, &mut ts) };
            }
        }

        warn!("prover: cleanup complete");
        Ok(())
    }
}

#[derive(Default)]
/// Builds an arena with `nodes[id.0].id == NodeId(id.0)` for every plan node.
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
            ProverPlan::Task {
                task,
                timeout_ms,
                task_id,
            } => self.push(
                parent,
                NodeInner::Task(TaskNode {
                    task,
                    timeout_ms,
                    task_id,
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
    run_prover_plan_with_driver(plan, max_procs, &resolve_driver_exe())
}

/// Runs a proof plan with an explicit driver executable and cleans up on
/// internal errors.
pub fn run_prover_plan_with_driver(
    plan: ProverPlan,
    max_procs: usize,
    driver: impl Into<ProofDriver>,
) -> io::Result<ProverReport> {
    // Create a run-scoped cancellation flag and wire OS signals to it.
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut sig_ids = Vec::new();
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        let flag = Arc::clone(&cancel_flag);
        // Set the flag from signal handler; store SigId so we can unregister on
        // exit.
        match unsafe { siglow::register(sig, move || flag.store(true, Ordering::Relaxed)) } {
            Ok(id) => sig_ids.push(id),
            Err(e) => warn!("prover: failed to register signal {}: {}", sig, e),
        }
    }

    let mut sched = Scheduler::new_with_driver(plan, max_procs, cancel_flag, driver);
    info!("prover: run starting");
    let run_res = sched.run();

    // Always unregister signal handlers before returning.
    for id in sig_ids.drain(..) {
        let _ = siglow::unregister(id);
    }

    if run_res.is_err() {
        let _ = sched.cleanup_tasks();
    }
    run_res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_plan::config::IrEquivConfig;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    /// Keeps the fake driver alive and reaps its processes even after a panic.
    struct TestScheduler {
        scheduler: Scheduler,
        _directory: tempfile::TempDir,
    }

    impl TestScheduler {
        /// Builds a scheduler using a shell script as its only proof backend.
        fn new(script: &str, plan: ProverPlan) -> io::Result<Self> {
            let directory = tempfile::tempdir()?;
            let driver = directory.path().join("driver.sh");
            fs::write(&driver, script)?;
            fs::set_permissions(&driver, fs::Permissions::from_mode(0o755))?;
            Ok(Self {
                scheduler: Scheduler::new_with_driver(
                    plan,
                    2,
                    Arc::new(AtomicBool::new(false)),
                    &driver,
                ),
                _directory: directory,
            })
        }
    }

    impl Drop for TestScheduler {
        fn drop(&mut self) {
            let _ = self.scheduler.cleanup_tasks();
        }
    }

    /// Selects the fixture behavior through the first positional input path.
    fn task(mode: &str) -> ProverPlan {
        ProverPlan::Task {
            task: ProverTask::IrEquiv {
                config: IrEquivConfig {
                    lhs_ir_file: PathBuf::from(mode),
                    rhs_ir_file: PathBuf::from("unused.ir"),
                    ..Default::default()
                },
            },
            timeout_ms: None,
            task_id: Some(mode.to_string()),
        }
    }

    /// Waits for this specific fixture child without changing signal handling.
    fn wait_for_pid(pid: i32, options: i32) -> io::Result<libc::c_int> {
        loop {
            let mut status = 0;
            let result = unsafe { libc::waitpid(pid, &mut status, options) };
            if result == pid {
                return Ok(status);
            }
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::EINTR) {
                return Err(error);
            }
        }
    }

    #[test]
    fn externally_reaped_task_completes_as_process_failure() {
        let mut fixture = TestScheduler::new(
            r#"#!/bin/sh
set -eu
while [ "$1" != --output_json ]; do shift; done
shift
printf '%s\n' '{"status":"proved","success":true}' > "$1"
printf 'task stdout\n'
printf 'task stderr\n' >&2
"#,
            task("result"),
        )
        .unwrap();
        let scheduler = &mut fixture.scheduler;
        scheduler.fill_slots().unwrap();
        let pid = *scheduler.pid_to_node.keys().next().unwrap();
        let status = wait_for_pid(pid, 0).unwrap();
        assert!(libc::WIFEXITED(status));
        assert_eq!(libc::WEXITSTATUS(status), 0);

        // The scheduler now gets ECHILD and cannot validate the process status,
        // even though its result file contains an otherwise valid proof.
        scheduler.wait_for_one_child_and_handle().unwrap();
        assert_eq!(
            scheduler.node_outcome(scheduler.root),
            Some(TaskOutcome::Indefinite {
                reason: IndefiniteReason::ProcessFailure,
            })
        );
        assert!(scheduler.running.is_empty());
        assert!(scheduler.pid_to_node.is_empty());
        assert!(scheduler.round_robin.is_empty());
        assert!(scheduler.outputs.is_empty());
        assert!(scheduler.json_files.is_empty());
        let report = scheduler.run().unwrap();
        assert_eq!(
            report.plan.proof_status(),
            crate::proof_outcome::ProofStatus::ExecutionError
        );
        let ProverReportNode::Task { stdout, stderr, .. } = report.plan else {
            panic!("expected a task report");
        };
        assert_eq!(stdout.as_deref(), Some("task stdout\n"));
        assert_eq!(
            stderr.as_deref(),
            Some(
                "task stderr\nprover: proof command was reaped without an observable exit status\n"
            )
        );
    }

    #[test]
    fn first_group_cancellation_retains_running_sibling_output_and_reason() {
        let mut fixture = TestScheduler::new(
            r#"#!/bin/sh
set -eu
mode="$2"
while [ "$1" != --output_json ]; do shift; done
shift
output="$1"
printf '%s stdout\n' "$mode"
printf '%s stderr\n' "$mode" >&2
kill -STOP "$$"
printf '%s\n' '{"status":"proved","success":true}' > "$output"
"#,
            ProverPlan::Group {
                kind: GroupKind::First,
                tasks: vec![task("winner"), task("canceled")],
                keep_running_till_finish: false,
            },
        )
        .unwrap();
        let scheduler = &mut fixture.scheduler;
        scheduler.fill_slots().unwrap();
        let children = scheduler.get_group_node(scheduler.root).children.clone();
        let mut winner_pid = None;
        for child in &children {
            let NodeInner::Task(TaskNode {
                state: TaskState::Running { pid },
                ..
            }) = scheduler.nodes[child.0].inner
            else {
                panic!("expected a running fixture child");
            };
            // A stopped child has flushed both streams and remains alive until
            // explicitly resumed or canceled. No timing assumptions are needed.
            let status = wait_for_pid(pid, libc::WUNTRACED).unwrap();
            assert!(libc::WIFSTOPPED(status));
            if *child == children[0] {
                winner_pid = Some(pid);
            }
        }
        assert_eq!(unsafe { libc::kill(winner_pid.unwrap(), libc::SIGCONT) }, 0);
        let report = scheduler.run().unwrap().plan;
        assert!(report.proof_status().is_proved());
        assert!(scheduler.pid_to_node.is_empty());
        assert!(scheduler.outputs.is_empty());

        let ProverReportNode::Task {
            outcome,
            stdout,
            stderr,
            ..
        } = report.find_task_node("canceled").unwrap()
        else {
            panic!("expected the canceled sibling's task report");
        };
        assert_eq!(
            *outcome,
            Some(TaskOutcome::Indefinite {
                reason: IndefiniteReason::GroupCriteriaAlreadyMet,
            })
        );
        assert_eq!(stdout.as_deref(), Some("canceled stdout\n"));
        assert_eq!(stderr.as_deref(), Some("canceled stderr\n"));
    }

    #[test]
    fn externally_reaped_cancellation_keeps_its_reason_and_output() {
        let mut fixture = TestScheduler::new(
            r#"#!/bin/sh
printf 'task stdout\n'
printf 'task stderr\n' >&2
kill -STOP "$$"
"#,
            task("result"),
        )
        .unwrap();
        let scheduler = &mut fixture.scheduler;
        scheduler.fill_slots().unwrap();
        let pid = *scheduler.pid_to_node.keys().next().unwrap();
        let stopped = wait_for_pid(pid, libc::WUNTRACED).unwrap();
        assert!(libc::WIFSTOPPED(stopped));
        scheduler
            .cancel_subtree(scheduler.root, IndefiniteReason::GroupCriteriaAlreadyMet)
            .unwrap();
        let killed = wait_for_pid(pid, 0).unwrap();
        assert!(libc::WIFSIGNALED(killed));
        assert_eq!(libc::WTERMSIG(killed), libc::SIGKILL);

        scheduler.wait_for_one_child_and_handle().unwrap();
        assert!(scheduler.pid_to_node.is_empty());
        assert!(scheduler.outputs.is_empty());
        let ProverReportNode::Task {
            outcome,
            stdout,
            stderr,
            ..
        } = scheduler.build_report_node(scheduler.root)
        else {
            panic!("expected a task report");
        };
        assert_eq!(
            outcome,
            Some(TaskOutcome::Indefinite {
                reason: IndefiniteReason::GroupCriteriaAlreadyMet,
            })
        );
        assert_eq!(stdout.as_deref(), Some("task stdout\n"));
        assert_eq!(stderr.as_deref(), Some("task stderr\n"));
    }
}
