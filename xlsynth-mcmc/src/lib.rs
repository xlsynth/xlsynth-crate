// SPDX-License-Identifier: Apache-2.0

use rand::RngCore;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod multichain;

/// Shared best-so-far candidate across threads.
pub struct Best<T> {
    pub cost: AtomicUsize,
    pub value: Mutex<T>,
}

impl<T: Clone> Best<T> {
    pub fn new(initial_cost: usize, value: T) -> Self {
        Self {
            cost: AtomicUsize::new(initial_cost),
            value: Mutex::new(value),
        }
    }

    /// Attempts to update the best-so-far candidate.
    ///
    /// Returns `true` if this call updated the global best, `false` otherwise.
    pub fn try_update(&self, new_cost: usize, new_value: T) -> bool {
        let mut current = self.cost.load(Ordering::SeqCst);
        while new_cost < current {
            match self
                .cost
                .compare_exchange(current, new_cost, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => {
                    let mut v = self.value.lock().unwrap();
                    *v = new_value;
                    return true;
                }
                Err(v) => current = v,
            }
        }
        false
    }

    pub fn get(&self) -> T {
        self.value.lock().unwrap().clone()
    }
}

/// Holds MCMC iteration statistics.
#[derive(Debug)]
pub struct McmcStats<K> {
    pub accepted_overall: usize,
    pub rejected_apply_fail: usize,
    pub rejected_candidate_fail: usize,
    pub rejected_oracle: usize,
    pub rejected_metro: usize,
    pub oracle_verified: usize,
    pub total_oracle_time_micros: u128,
    pub accepted_edits_by_kind: HashMap<K, usize>,
    pub rejected_sim_fail: usize,
    pub total_sim_time_micros: u128,
}

impl<K> Default for McmcStats<K> {
    fn default() -> Self {
        McmcStats {
            accepted_overall: 0,
            rejected_apply_fail: 0,
            rejected_candidate_fail: 0,
            rejected_oracle: 0,
            rejected_metro: 0,
            oracle_verified: 0,
            total_oracle_time_micros: 0,
            accepted_edits_by_kind: HashMap::new(),
            rejected_sim_fail: 0,
            total_sim_time_micros: 0,
        }
    }
}

impl<K> McmcStats<K>
where
    K: Eq + Hash,
{
    /// Merges `other` into `self` (useful for multi-segment / multi-chain
    /// runs).
    pub fn merge_from(&mut self, other: McmcStats<K>) {
        self.accepted_overall += other.accepted_overall;
        self.rejected_apply_fail += other.rejected_apply_fail;
        self.rejected_candidate_fail += other.rejected_candidate_fail;
        self.rejected_oracle += other.rejected_oracle;
        self.rejected_metro += other.rejected_metro;
        self.oracle_verified += other.oracle_verified;
        self.total_oracle_time_micros += other.total_oracle_time_micros;
        self.rejected_sim_fail += other.rejected_sim_fail;
        self.total_sim_time_micros += other.total_sim_time_micros;
        for (k, v) in other.accepted_edits_by_kind.into_iter() {
            *self.accepted_edits_by_kind.entry(k).or_insert(0) += v;
        }
    }
}

/// Details of what occurred during a single MCMC iteration attempt.
pub enum IterationOutcomeDetails<K> {
    CandidateFailure,
    ApplyFailure,
    SimFailure,
    OracleFailure,
    MetropolisReject,
    Accepted { kind: K },
}

/// Output of a single MCMC iteration.
pub struct McmcIterationOutput<S, C, K> {
    pub output_state: S,
    pub output_cost: C,
    pub best_updated: bool,
    pub outcome: IterationOutcomeDetails<K>,
    pub transform_always_equivalent: bool,
    pub transform: Option<K>,
    /// Time spent in oracle or simulation, 0 if not run.
    pub oracle_time_micros: u128,
}

/// Engine-level options for MCMC runs.
#[derive(Clone, Debug)]
pub struct McmcOptions {
    pub sat_reset_interval: u64,
    pub initial_temperature: f64,
    /// Global starting iteration index (for multi-segment runs).
    pub start_iteration: u64,
    /// Total planned iterations for the *entire* run (across segments).
    /// If `None`, temperature remains constant (no cooling).
    pub total_iters: Option<u64>,
}

impl<K> McmcStats<K>
where
    K: Eq + Hash + Ord + Clone + fmt::Debug,
{
    /// Update statistics based on the outcome of a single iteration.
    ///
    /// `iteration_index` is the human-readable global iteration number used
    /// only for panic messages in paranoid mode.
    pub fn update_for_iteration<S, C>(
        &mut self,
        iteration: &McmcIterationOutput<S, C, K>,
        paranoid: bool,
        iteration_index: u64,
    ) {
        self.total_oracle_time_micros += iteration.oracle_time_micros;

        match &iteration.outcome {
            IterationOutcomeDetails::Accepted { kind } => {
                self.accepted_overall += 1;
                *self.accepted_edits_by_kind.entry(kind.clone()).or_insert(0) += 1;
                if iteration.oracle_time_micros > 0 {
                    self.oracle_verified += 1;
                }
            }
            IterationOutcomeDetails::CandidateFailure => {
                self.rejected_candidate_fail += 1;
            }
            IterationOutcomeDetails::ApplyFailure => {
                self.rejected_apply_fail += 1;
            }
            IterationOutcomeDetails::SimFailure => {
                self.rejected_sim_fail += 1;
                self.total_sim_time_micros += iteration.oracle_time_micros;
            }
            IterationOutcomeDetails::OracleFailure => {
                self.rejected_oracle += 1;
                if paranoid && iteration.transform_always_equivalent {
                    panic!(
                        "[mcmc] equivalence failure for always-equivalent transform at iteration {}; transform: {:?} should always be equivalent",
                        iteration_index, iteration.transform
                    );
                }
            }
            IterationOutcomeDetails::MetropolisReject => {
                self.rejected_metro += 1;
                if iteration.oracle_time_micros > 0 {
                    self.oracle_verified += 1;
                }
            }
        }
    }
}

/// Minimum allowed relative temperature (as a ratio of the initial
/// temperature) to avoid underflow and numeric issues during cooling.
pub const MIN_TEMPERATURE_RATIO: f64 = 0.00001;

/// Decide whether to accept a candidate move under the Metropolis rule.
///
/// `current_metric` and `new_metric` are scalar objective values (lower is
/// better).  When `new_metric < current_metric`, the move is always accepted.
/// Otherwise it is accepted with probability `exp((current - new) / temp)`.
pub fn metropolis_accept<R: RngCore + ?Sized>(
    current_metric: f64,
    new_metric: f64,
    temp: f64,
    rng: &mut R,
) -> bool {
    if new_metric < current_metric {
        return true;
    }

    let accept_prob = ((current_metric - new_metric) / temp).exp();
    let raw = rng.next_u64();

    // Generate a uniform floating-point value in [0, 1) with correct IEEE-754
    // semantics.
    //
    // Using `(raw as f64) / (u64::MAX as f64)` can produce `1.0` when `raw ==
    // u64::MAX`, which introduces a tiny rejection bias in Metropolis-Hastings.
    //
    // We instead take the top 53 bits (the precision of f64’s mantissa) and scale
    // by 2^-53, yielding values in [0, 1).
    let u01 = (raw >> 11) as f64 * 2.0_f64.powi(-53);
    u01 < accept_prob
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    struct FixedU64Rng {
        v: u64,
    }

    impl RngCore for FixedU64Rng {
        fn next_u32(&mut self) -> u32 {
            (self.v >> 32) as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.v
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for (i, b) in dest.iter_mut().enumerate() {
                *b = (self.v >> ((i % 8) * 8)) as u8;
            }
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    #[test]
    fn u01_conversion_never_reaches_one() {
        let mut rng = FixedU64Rng { v: u64::MAX };
        let raw = rng.next_u64();
        let u01 = (raw >> 11) as f64 * 2.0_f64.powi(-53);
        assert!(u01 < 1.0, "u01 must be in [0,1), got {u01}");
    }

    #[test]
    fn u01_conversion_zero_is_zero() {
        let mut rng = FixedU64Rng { v: 0 };
        let raw = rng.next_u64();
        let u01 = (raw >> 11) as f64 * 2.0_f64.powi(-53);
        assert_eq!(u01, 0.0);
    }
}
