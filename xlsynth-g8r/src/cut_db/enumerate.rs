// SPDX-License-Identifier: Apache-2.0

//! Exact enumeration of 4-input truth tables under AND + free-NOT AIGs.
//!
//! This enumerator works in the full `u16` truth-table space (all 65536
//! functions under a fixed input ordering). We later compress to canonical NPN
//! representatives for on-disk storage.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::cut_db::fragment::{FIRST_NODE_ID, FragmentNode, GateFnFragment, Lit};
use crate::cut_db::pareto::{ParetoPoint, dominates, same_cost};
use crate::cut_db::tt16::TruthTable16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateKey {
    tt: TruthTable16,
    ands: u16,
    depth: u16,
    q_index: usize,
    lhs_neg: bool,
    rhs_neg: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EnumerateOptions {
    /// Optional maximum AND-count to explore.
    ///
    /// If `None`, enumeration continues until a fixed point is reached
    /// (worklist empty).
    pub max_ands: Option<u16>,

    /// If set, emit `log::info!` progress every N worklist pops.
    pub progress_every_pops: Option<u64>,

    /// Chunk size for parallelizing the `p × all_points` expansion.
    pub parallel_chunk_size: usize,
}

impl Default for EnumerateOptions {
    fn default() -> Self {
        Self {
            max_ands: None,
            progress_every_pops: None,
            parallel_chunk_size: 4096,
        }
    }
}

/// Holds the full-space enumeration results indexed by `u16` truth table.
pub struct FullSpaceDb {
    /// `frontiers[tt]` is the Pareto frontier for that exact `tt` under the
    /// fixed input ordering.
    pub frontiers: Vec<Vec<ParetoPoint>>,
    pub covered_count: usize,
}

impl FullSpaceDb {
    pub fn new_empty() -> Self {
        Self {
            frontiers: vec![Vec::new(); 65536],
            covered_count: 0,
        }
    }

    pub fn is_covered(&self) -> bool {
        self.frontiers.iter().all(|v| !v.is_empty())
    }

    pub fn covered_count(&self) -> usize {
        self.covered_count
    }
}

fn insert_pareto(frontier: &mut Vec<ParetoPoint>, cand: ParetoPoint) -> bool {
    // Reject if dominated or cost-duplicate.
    for p in frontier.iter() {
        if dominates(p, &cand) || same_cost(p, &cand) {
            return false;
        }
    }
    // Remove anything dominated by candidate.
    frontier.retain(|p| !dominates(&cand, p));
    frontier.push(cand);
    true
}

fn can_insert_pareto(frontier: &[ParetoPoint], ands: u16, depth: u16) -> bool {
    for p in frontier {
        if (p.ands <= ands && p.depth <= depth) || (p.ands == ands && p.depth == depth) {
            return false;
        }
    }
    true
}

fn negate_tt(tt: TruthTable16, neg: bool) -> TruthTable16 {
    if neg { tt.not() } else { tt }
}

fn combine_and(
    lhs: &GateFnFragment,
    rhs: &GateFnFragment,
    lhs_neg: bool,
    rhs_neg: bool,
) -> GateFnFragment {
    let lhs_nodes_len = lhs.nodes.len() as u16;
    let rhs_nodes_len = rhs.nodes.len() as u16;
    let offset = lhs_nodes_len;

    let mut nodes: Vec<FragmentNode> = Vec::with_capacity(lhs.nodes.len() + rhs.nodes.len() + 1);
    nodes.extend_from_slice(&lhs.nodes);

    let remap_rhs_lit = |lit: Lit| -> Lit {
        if lit.id >= FIRST_NODE_ID {
            Lit::new(lit.id + offset, lit.negated)
        } else {
            lit
        }
    };

    for node in &rhs.nodes {
        match *node {
            FragmentNode::And2 { a, b } => nodes.push(FragmentNode::And2 {
                a: remap_rhs_lit(a),
                b: remap_rhs_lit(b),
            }),
        }
    }

    let mut a = lhs.output;
    if lhs_neg {
        a = a.negate();
    }
    let mut b = remap_rhs_lit(rhs.output);
    if rhs_neg {
        b = b.negate();
    }

    let new_id = FIRST_NODE_ID + lhs_nodes_len + rhs_nodes_len;
    nodes.push(FragmentNode::And2 { a, b });

    GateFnFragment {
        nodes,
        output: Lit::new(new_id, false),
    }
}

/// Enumerates Pareto-optimal AIG fragments for all 4-input truth tables.
///
/// Returns a `FullSpaceDb` indexed by exact `u16` truth tables.
pub fn enumerate_full_space(options: EnumerateOptions) -> FullSpaceDb {
    let mut db = FullSpaceDb::new_empty();

    // Flattened list of all Pareto points currently known (for joining).
    let mut all_points: Vec<ParetoPoint> = Vec::new();
    let mut worklist: VecDeque<ParetoPoint> = VecDeque::new();

    // Seed with const0 (const1 is represented by output negation in later steps).
    {
        let frag = GateFnFragment::const0();
        let p = ParetoPoint {
            tt: TruthTable16::const0(),
            ands: 0,
            depth: 0,
            frag,
        };
        let idx = p.tt.0 as usize;
        let was_empty = db.frontiers[idx].is_empty();
        if insert_pareto(&mut db.frontiers[idx], p.clone()) {
            if was_empty {
                db.covered_count += 1;
            }
            all_points.push(p.clone());
            worklist.push_back(p);
        }
    }

    // Seed with primary inputs.
    for i in 0..4usize {
        let frag = GateFnFragment::input(i as u16);
        let tt = TruthTable16::var(i);
        let p = ParetoPoint {
            tt,
            ands: 0,
            depth: 0,
            frag,
        };
        let idx = tt.0 as usize;
        let was_empty = db.frontiers[idx].is_empty();
        if insert_pareto(&mut db.frontiers[idx], p.clone()) {
            if was_empty {
                db.covered_count += 1;
            }
            all_points.push(p.clone());
            worklist.push_back(p);
        }
    }

    let mut pops: u64 = 0;
    while let Some(p) = worklist.pop_front() {
        pops += 1;
        if let Some(every) = options.progress_every_pops {
            if pops % every == 0 {
                log::info!(
                    "enumerate_full_space: pops={} covered={}/65536 all_points={} worklist={}",
                    pops,
                    db.covered_count,
                    all_points.len(),
                    worklist.len()
                );
            }
        }

        if let Some(max_ands) = options.max_ands {
            if p.ands >= max_ands {
                continue;
            }
        }

        // Snapshot current all-points length to avoid infinite growth within this pop.
        let current_points_len = all_points.len();

        // Generate small candidate keys in parallel over chunks of `all_points`.
        // We intentionally do NOT build fragments in the parallel phase; fragments
        // are expensive to allocate and most candidates will be dominated. Instead
        // we build fragments only for candidates that survive the fast Pareto check
        // during the deterministic sequential merge phase.
        let chunk_size = core::cmp::max(1, options.parallel_chunk_size);
        let frontiers_snapshot = &db.frontiers;
        let chunk_results: Vec<(usize, Vec<CandidateKey>)> = all_points[..current_points_len]
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut v: Vec<CandidateKey> = Vec::with_capacity(chunk.len() * 4);
                let base = chunk_idx * chunk_size;
                for (j, q) in chunk.iter().enumerate() {
                    let qi = base + j;
                    let cand_ands = p.ands + q.ands + 1;
                    if let Some(max_ands) = options.max_ands {
                        if cand_ands > max_ands {
                            continue;
                        }
                    }
                    let cand_depth = 1 + core::cmp::max(p.depth, q.depth);

                    for lhs_neg in [false, true] {
                        let p_tt = negate_tt(p.tt, lhs_neg);
                        for rhs_neg in [false, true] {
                            let q_tt = negate_tt(q.tt, rhs_neg);
                            let cand_tt = p_tt.and(q_tt);
                            let idx = cand_tt.0 as usize;
                            // Safe pruning: if dominated by the current frontier snapshot,
                            // it will remain dominated as frontiers only ever improve.
                            if !can_insert_pareto(&frontiers_snapshot[idx], cand_ands, cand_depth) {
                                continue;
                            }
                            v.push(CandidateKey {
                                tt: cand_tt,
                                ands: cand_ands,
                                depth: cand_depth,
                                q_index: qi,
                                lhs_neg,
                                rhs_neg,
                            });
                        }
                    }
                }
                (chunk_idx, v)
            })
            .collect();

        // Deterministic merge order across rayon scheduling:
        // process chunks in ascending chunk index, and candidates in the order they
        // were generated within the chunk.
        let mut chunk_results = chunk_results;
        chunk_results.sort_by_key(|(chunk_idx, _)| *chunk_idx);

        for (_chunk_idx, keys) in chunk_results {
            for key in keys {
                let idx = key.tt.0 as usize;
                if !can_insert_pareto(&db.frontiers[idx], key.ands, key.depth) {
                    continue;
                }

                let q = &all_points[key.q_index];
                let cand_frag = combine_and(&p.frag, &q.frag, key.lhs_neg, key.rhs_neg);

                debug_assert_eq!(cand_frag.and_count(), key.ands);
                debug_assert_eq!(cand_frag.depth(), key.depth);
                debug_assert_eq!(cand_frag.eval_tt16(), key.tt);

                let cand = ParetoPoint {
                    tt: key.tt,
                    ands: key.ands,
                    depth: key.depth,
                    frag: cand_frag,
                };

                let was_empty = db.frontiers[idx].is_empty();
                if insert_pareto(&mut db.frontiers[idx], cand.clone()) {
                    if was_empty {
                        db.covered_count += 1;
                    }
                    all_points.push(cand.clone());
                    worklist.push_back(cand);
                }
            }
        }
    }

    log::info!(
        "enumerate_full_space: done pops={} covered={}/65536 all_points={}",
        pops,
        db.covered_count,
        all_points.len()
    );

    db
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_seeds_cover_constants_and_vars() {
        let db = enumerate_full_space(EnumerateOptions {
            max_ands: Some(0),
            progress_every_pops: None,
            parallel_chunk_size: 64,
        });
        assert!(!db.frontiers[TruthTable16::const0().0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(0).0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(1).0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(2).0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(3).0 as usize].is_empty());
        // const1 not directly seeded; it is reachable by output negation as a
        // literal. (We don't include NOT as a node in this basis.)
    }

    #[test]
    fn test_insert_pareto_prunes_dominated() {
        let base = ParetoPoint {
            tt: TruthTable16(0xBEEF),
            ands: 2,
            depth: 2,
            frag: GateFnFragment::const0(),
        };
        let dominated = ParetoPoint {
            tt: TruthTable16(0xBEEF),
            ands: 3,
            depth: 2,
            frag: GateFnFragment::const0(),
        };
        let mut f = Vec::new();
        assert!(insert_pareto(&mut f, base.clone()));
        assert!(!insert_pareto(&mut f, dominated));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].ands, base.ands);
        assert_eq!(f[0].depth, base.depth);
    }
}
