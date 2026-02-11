// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::thread;

use crate::McmcStats;

/// Strategy for running multiple MCMC chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainStrategy {
    /// Run chains independently and pick the best at the end.
    Independent,
    /// Run one explorer chain (high temperature) and N-1 exploit chains.
    ///
    /// The orchestrator runs the chains in lockstep segments of
    /// `checkpoint_iters` so that the reduction and any "jump to global
    /// best" behavior is deterministic.
    ExploreExploit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainRole {
    Explorer,
    Exploit,
}

/// Parameters for running a single segment of a single chain.
#[derive(Clone, Copy, Debug)]
pub struct SegmentRunParams {
    pub chain_no: usize,
    pub role: ChainRole,
    pub iter_offset: u64,
    pub segment_iters: u64,
    pub total_iters: u64,
    /// The orchestrator supplies deterministic per-chain, per-segment seeds.
    pub seed: u64,
}

/// Outcome of running a segment.
pub struct SegmentOutcome<S, C, K> {
    pub end_state: S,
    pub end_cost: C,
    pub best_state: S,
    pub best_cost: C,
    pub stats: McmcStats<K>,
}

pub trait SegmentRunner<S, C, K>: Send + Sync {
    type Error: Send + 'static;

    fn run_segment(
        &self,
        start_state: S,
        params: SegmentRunParams,
    ) -> Result<SegmentOutcome<S, C, K>, Self::Error>;
}

fn choose_better<S, C>(
    objective_metric: impl Fn(&C) -> u128,
    tiebreak_key: impl Fn(&S) -> String,
    a: (&S, &C),
    b: (&S, &C),
) -> bool {
    let a_m = objective_metric(a.1);
    let b_m = objective_metric(b.1);
    if a_m != b_m {
        return a_m < b_m;
    }
    tiebreak_key(a.0) < tiebreak_key(b.0)
}

/// Runs multiple MCMC chains in parallel and returns the best result.
///
/// Determinism: for a fixed `(seed, threads, strategy, checkpoint_iters)` and a
/// deterministic `SegmentRunner`, the returned `(best_state, best_cost)` is
/// deterministic.
pub fn run_multichain<S, C, K, R>(
    start_state: S,
    total_iters: u64,
    seed: u64,
    threads: usize,
    strategy: ChainStrategy,
    checkpoint_iters: u64,
    runner: Arc<R>,
    objective_metric: impl Fn(&C) -> u128 + Copy + Send + Sync + 'static,
    tiebreak_key: impl Fn(&S) -> String + Copy + Send + Sync + 'static,
    should_jump_to_best: impl Fn(&C, &C) -> bool + Copy + Send + Sync + 'static,
) -> Result<(S, C, McmcStats<K>), R::Error>
where
    S: Send + Clone + 'static,
    C: Send + Copy + 'static,
    K: Send + 'static + std::hash::Hash + Eq,
    R: SegmentRunner<S, C, K> + 'static,
{
    let thread_count = threads.max(1);
    let seg_size = checkpoint_iters.max(1);

    match strategy {
        ChainStrategy::Independent => {
            let mut handles = Vec::with_capacity(thread_count);
            for chain_no in 0..thread_count {
                let st = start_state.clone();
                let seed_i = seed ^ chain_no as u64;
                let runner = runner.clone();
                handles.push(thread::spawn(move || {
                    runner
                        .run_segment(
                            st,
                            SegmentRunParams {
                                chain_no,
                                role: ChainRole::Exploit,
                                iter_offset: 0,
                                segment_iters: total_iters,
                                total_iters,
                                seed: seed_i,
                            },
                        )
                        .map(|o| (chain_no, o))
                }));
            }

            let mut outs: Vec<(usize, SegmentOutcome<S, C, K>)> = Vec::with_capacity(thread_count);
            for h in handles {
                let out = h.join().expect("chain thread panicked")?;
                outs.push(out);
            }
            outs.sort_by_key(|(i, _)| *i);

            let mut best_idx = 0usize;
            for (i, o) in outs.iter().enumerate() {
                if choose_better(
                    objective_metric,
                    tiebreak_key,
                    (&o.1.best_state, &o.1.best_cost),
                    (&outs[best_idx].1.best_state, &outs[best_idx].1.best_cost),
                ) {
                    best_idx = i;
                }
            }

            let best = outs.swap_remove(best_idx).1;
            Ok((best.best_state, best.best_cost, best.stats))
        }
        ChainStrategy::ExploreExploit => {
            if thread_count == 1 {
                return run_multichain(
                    start_state,
                    total_iters,
                    seed,
                    1,
                    ChainStrategy::Independent,
                    checkpoint_iters,
                    runner,
                    objective_metric,
                    tiebreak_key,
                    should_jump_to_best,
                );
            }

            let mut local_state: Vec<S> = vec![start_state.clone(); thread_count];
            let mut local_best_state: Vec<S> = vec![start_state.clone(); thread_count];
            // This assumes the runner will produce a consistent initial best on first
            // segment.
            let mut local_best_cost: Vec<Option<C>> = vec![None; thread_count];
            let mut local_stats: Vec<McmcStats<K>> = std::iter::repeat_with(McmcStats::default)
                .take(thread_count)
                .collect();

            let mut global_best_state: Option<S> = None;
            let mut global_best_cost: Option<C> = None;

            let mut iter_offset = 0u64;
            while iter_offset < total_iters {
                let seg = std::cmp::min(seg_size, total_iters - iter_offset);

                let mut handles = Vec::with_capacity(thread_count);
                for chain_no in 0..thread_count {
                    let st = local_state[chain_no].clone();
                    let role = if chain_no == 0 {
                        ChainRole::Explorer
                    } else {
                        ChainRole::Exploit
                    };
                    let seed_seg = (seed ^ chain_no as u64).wrapping_add(iter_offset);
                    let runner = runner.clone();
                    handles.push(thread::spawn(move || {
                        runner
                            .run_segment(
                                st,
                                SegmentRunParams {
                                    chain_no,
                                    role,
                                    iter_offset,
                                    segment_iters: seg,
                                    total_iters,
                                    seed: seed_seg,
                                },
                            )
                            .map(|o| (chain_no, o))
                    }));
                }

                let mut outs: Vec<(usize, SegmentOutcome<S, C, K>)> =
                    Vec::with_capacity(thread_count);
                for h in handles {
                    outs.push(h.join().expect("chain thread panicked")?);
                }
                outs.sort_by_key(|(i, _)| *i);

                // Merge segment outcomes deterministically by chain index.
                for (chain_no, o) in outs.into_iter() {
                    local_stats[chain_no].merge_from(o.stats);
                    local_state[chain_no] = o.end_state;

                    match local_best_cost[chain_no] {
                        None => {
                            local_best_state[chain_no] = o.best_state;
                            local_best_cost[chain_no] = Some(o.best_cost);
                        }
                        Some(cur_cost) => {
                            if choose_better(
                                objective_metric,
                                tiebreak_key,
                                (&o.best_state, &o.best_cost),
                                (&local_best_state[chain_no], &cur_cost),
                            ) {
                                local_best_state[chain_no] = o.best_state;
                                local_best_cost[chain_no] = Some(o.best_cost);
                            }
                        }
                    }
                }

                // Update global best deterministically across chain bests.
                for chain_no in 0..thread_count {
                    let c = local_best_cost[chain_no].expect("best cost must be set");
                    match (&global_best_state, global_best_cost) {
                        (None, None) => {
                            global_best_state = Some(local_best_state[chain_no].clone());
                            global_best_cost = Some(c);
                        }
                        (Some(gs), Some(gc)) => {
                            if choose_better(
                                objective_metric,
                                tiebreak_key,
                                (&local_best_state[chain_no], &c),
                                (gs, &gc),
                            ) {
                                global_best_state = Some(local_best_state[chain_no].clone());
                                global_best_cost = Some(c);
                            }
                        }
                        _ => unreachable!("global best state/cost must be set together"),
                    }
                }

                // Allow exploit chains to jump to global best deterministically.
                let gb = global_best_state
                    .as_ref()
                    .expect("global best must be set")
                    .clone();
                let gbc = global_best_cost.expect("global best cost must be set");
                for chain_no in 1..thread_count {
                    let cur_cost = local_best_cost[chain_no].expect("best cost must be set");
                    if should_jump_to_best(&cur_cost, &gbc) {
                        local_state[chain_no] = gb.clone();
                    }
                }

                iter_offset += seg;
            }

            // Pick which chain's stats to return: the chain whose best matches global best.
            let gb = global_best_state.expect("global best must be set");
            let gbc = global_best_cost.expect("global best cost must be set");

            let mut best_chain = 0usize;
            for chain_no in 0..thread_count {
                let c = local_best_cost[chain_no].expect("best cost must be set");
                if choose_better(
                    objective_metric,
                    tiebreak_key,
                    (&local_best_state[chain_no], &c),
                    (
                        &local_best_state[best_chain],
                        &local_best_cost[best_chain].unwrap(),
                    ),
                ) {
                    best_chain = chain_no;
                }
            }

            Ok((gb, gbc, std::mem::take(&mut local_stats[best_chain])))
        }
    }
}
