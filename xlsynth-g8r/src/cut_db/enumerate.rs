// SPDX-License-Identifier: Apache-2.0

//! Exact enumeration of 4-input truth tables under AND + free-NOT AIGs.
//!
//! This enumerator works in the full `u16` truth-table space (all 65536
//! functions under a fixed input ordering). We later compress to canonical NPN
//! representatives for on-disk storage.

use std::collections::VecDeque;

use crate::cut_db::fragment::{FragmentNode, GateFnFragment, Lit, FIRST_NODE_ID};
use crate::cut_db::pareto::{dominates, same_cost, ParetoPoint};
use crate::cut_db::tt16::TruthTable16;

#[derive(Debug, Clone, Copy)]
pub struct EnumerateOptions {
    /// Optional maximum AND-count to explore.
    ///
    /// If `None`, enumeration continues until a fixed point is reached
    /// (worklist empty).
    pub max_ands: Option<u16>,

    /// If set, emit `log::info!` progress every N worklist pops.
    pub progress_every_pops: Option<u64>,
}

impl Default for EnumerateOptions {
    fn default() -> Self {
        Self {
            max_ands: None,
            progress_every_pops: None,
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

fn negate_tt(tt: TruthTable16, neg: bool) -> TruthTable16 {
    if neg { tt.not() } else { tt }
}

fn combine_and(lhs: &GateFnFragment, rhs: &GateFnFragment, lhs_neg: bool, rhs_neg: bool) -> GateFnFragment {
    let lhs_nodes_len = lhs.nodes.len() as u16;
    let rhs_nodes_len = rhs.nodes.len() as u16;
    let offset = lhs_nodes_len;

    let mut nodes: Vec<FragmentNode> =
        Vec::with_capacity(lhs.nodes.len() + rhs.nodes.len() + 1);
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
        for qi in 0..current_points_len {
            let q = all_points[qi].clone();
            // Enumerate operand polarity combos (free NOT on edges).
            for lhs_neg in [false, true] {
                let p_tt = negate_tt(p.tt, lhs_neg);
                for rhs_neg in [false, true] {
                    let q_tt = negate_tt(q.tt, rhs_neg);
                    let cand_tt = p_tt.and(q_tt);

                    // AND-count/depth composition (tree-like composition).
                    // Note: this is exact for the constructed witness; the enumerator
                    // maintains a Pareto set and will prune dominated higher-cost
                    // candidates.
                    let cand_ands = p.ands + q.ands + 1;
                    if let Some(max_ands) = options.max_ands {
                        if cand_ands > max_ands {
                            continue;
                        }
                    }
                    let cand_depth = 1 + core::cmp::max(p.depth, q.depth);

                    // Build witness fragment by concatenation + new AND node.
                    let cand_frag = combine_and(&p.frag, &q.frag, lhs_neg, rhs_neg);

                    debug_assert_eq!(cand_frag.and_count(), cand_ands);
                    debug_assert_eq!(cand_frag.depth(), cand_depth);

                    // Quick semantic consistency check in debug builds.
                    debug_assert_eq!(cand_frag.eval_tt16(), cand_tt);

                    let cand = ParetoPoint {
                        tt: cand_tt,
                        ands: cand_ands,
                        depth: cand_depth,
                        frag: cand_frag,
                    };

                    let idx = cand_tt.0 as usize;
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
        });
        assert!(!db.frontiers[TruthTable16::const0().0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(0).0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(1).0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(2).0 as usize].is_empty());
        assert!(!db.frontiers[TruthTable16::var(3).0 as usize].is_empty());
        // const1 not directly seeded; it is reachable by output negation as a literal.
        // (We don't include NOT as a node in this basis.)
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


