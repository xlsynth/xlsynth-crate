// SPDX-License-Identifier: Apache-2.0

//! Measured phases of a proof invocation, including work done by its
//! partitions.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::proof_timing::micros;

/// Sums wall times for each kind of work across all participating workers.
/// An absent duration means the phase was not observed. A measured phase may
/// take less than one microsecond.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ProofPhases {
    /// Front-end work, including DSLX and IR preparation or optimization.
    pub prepare_work_micros: Option<u64>,
    /// Building the SMT query, up to and including asserting its predicate.
    pub smt_encode_work_micros: Option<u64>,
    /// Wall time within calls to the solver's satisfiability check.
    pub solver_check_work_micros: Option<u64>,
    /// Number of observed checks, including failed and inconclusive calls.
    pub solver_check_count: u64,
    /// Reading the model after a satisfiable check, including a failed read.
    pub counterexample_read_work_micros: Option<u64>,
    /// Wall time for launched XLS proof tools, whose internal phases are
    /// opaque.
    pub external_tool_work_micros: Option<u64>,
}

impl ProofPhases {
    /// Includes the measured work of another property or partition.
    pub fn merge(&mut self, other: Self) {
        for (total, observed) in [
            (&mut self.prepare_work_micros, other.prepare_work_micros),
            (
                &mut self.smt_encode_work_micros,
                other.smt_encode_work_micros,
            ),
            (
                &mut self.solver_check_work_micros,
                other.solver_check_work_micros,
            ),
            (
                &mut self.counterexample_read_work_micros,
                other.counterexample_read_work_micros,
            ),
            (
                &mut self.external_tool_work_micros,
                other.external_tool_work_micros,
            ),
        ] {
            if let Some(value) = observed {
                *total = Some(total.unwrap_or(0).saturating_add(value));
            }
        }
        self.solver_check_count = self
            .solver_check_count
            .saturating_add(other.solver_check_count);
    }
}

/// Measurements for one invocation, shared by reference with its partition
/// workers.
#[derive(Default)]
pub struct PhaseCollector {
    totals: Mutex<ProofPhases>,
}

/// Each interval belongs to exactly one phase within a solver worker.
#[derive(Clone, Copy)]
pub(crate) enum Phase {
    Prepare,
    SmtEncode,
    SolverCheck,
    CounterexampleRead,
    ExternalTool,
}

impl PhaseCollector {
    /// Times preparation until the guard drops, including an early error
    /// return.
    pub fn start_prepare(&self) -> impl Drop + '_ {
        self.start(Phase::Prepare)
    }

    /// Measures front-end preparation, including failed preparation.
    pub fn prepare<T>(&self, work: impl FnOnce() -> T) -> T {
        let _timer = self.start(Phase::Prepare);
        work()
    }

    /// Returns all completed measurements.
    pub fn snapshot(&self) -> ProofPhases {
        *self.totals.lock().unwrap()
    }

    pub(crate) fn start(&self, phase: Phase) -> PhaseTimer<'_> {
        PhaseTimer {
            collector: self,
            phase,
            start: Instant::now(),
        }
    }

    pub(crate) fn record(&self, phase: Phase, duration: Duration) {
        let mut totals = self.totals.lock().unwrap();
        let slot = match phase {
            Phase::Prepare => &mut totals.prepare_work_micros,
            Phase::SmtEncode => &mut totals.smt_encode_work_micros,
            Phase::SolverCheck => {
                totals.solver_check_count = totals.solver_check_count.saturating_add(1);
                &mut totals.solver_check_work_micros
            }
            Phase::CounterexampleRead => &mut totals.counterexample_read_work_micros,
            Phase::ExternalTool => &mut totals.external_tool_work_micros,
        };
        *slot = Some(slot.unwrap_or(0).saturating_add(micros(duration)));
    }
}

/// Records elapsed time on every exit, including early returns from a fallible
/// phase.
#[must_use]
pub(crate) struct PhaseTimer<'a> {
    collector: &'a PhaseCollector,
    phase: Phase,
    start: Instant,
}

impl Drop for PhaseTimer<'_> {
    fn drop(&mut self) {
        self.collector.record(self.phase, self.start.elapsed());
    }
}
