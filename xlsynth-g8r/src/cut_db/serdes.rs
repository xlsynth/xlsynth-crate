// SPDX-License-Identifier: Apache-2.0

//! Canonical on-disk database representation and conversions.

use serde::{Deserialize, Serialize};

use crate::cut_db::enumerate::FullSpaceDb;
use crate::cut_db::npn::canon_tt16;
use crate::cut_db::pareto::{dominates, same_cost, ParetoPoint};
use crate::cut_db::tt16::TruthTable16;

const CUT_DB_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonDbOnDisk {
    pub version: u32,
    pub entries: Vec<CanonEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonEntry {
    pub canon_tt: TruthTable16,
    pub pareto: Vec<ParetoPoint>,
}

fn insert_pareto(frontier: &mut Vec<ParetoPoint>, cand: ParetoPoint) -> bool {
    for p in frontier.iter() {
        if dominates(p, &cand) || same_cost(p, &cand) {
            return false;
        }
    }
    frontier.retain(|p| !dominates(&cand, p));
    frontier.push(cand);
    true
}

/// Converts a full-space DB (indexed by exact `u16`) into an NPN-canonical DB
/// suitable for on-disk storage.
///
/// Canonicalization rewrites each witness fragment into the canonical input
/// space; AND-count and depth are preserved.
pub fn canon_from_full_space(full: &FullSpaceDb) -> CanonDbOnDisk {
    let mut canon_frontiers: Vec<Vec<ParetoPoint>> = vec![Vec::new(); 65536];

    for tt_u in 0u32..=0xFFFF {
        let idx = tt_u as usize;
        for p in &full.frontiers[idx] {
            let (canon_tt, xform) = canon_tt16(p.tt);
            let canon_frag = p.frag.apply_npn(xform.inverse());
            debug_assert_eq!(canon_frag.eval_tt16(), canon_tt);
            let cand = ParetoPoint {
                tt: canon_tt,
                ands: p.ands,
                depth: p.depth,
                frag: canon_frag,
            };
            let canon_idx = canon_tt.0 as usize;
            insert_pareto(&mut canon_frontiers[canon_idx], cand);
        }
    }

    let mut entries: Vec<CanonEntry> = canon_frontiers
        .into_iter()
        .enumerate()
        .filter_map(|(canon_u, pareto)| {
            if pareto.is_empty() {
                None
            } else {
                Some(CanonEntry {
                    canon_tt: TruthTable16(canon_u as u16),
                    pareto,
                })
            }
        })
        .collect();

    // Deterministic ordering.
    entries.sort_by_key(|e| e.canon_tt.0);

    CanonDbOnDisk {
        version: CUT_DB_VERSION,
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut_db::enumerate::{enumerate_full_space, EnumerateOptions};
    use crate::cut_db::loader::CutDb;

    #[test]
    fn test_canon_db_bincode_round_trip_and_loader_smoke() {
        // Use a small AND bound so this stays fast; this test only checks
        // serialization invariants and loader behavior, not full coverage.
        let full = enumerate_full_space(EnumerateOptions {
            max_ands: Some(2),
            progress_every_pops: None,
            parallel_chunk_size: 64,
        });
        let canon = canon_from_full_space(&full);

        let bytes = bincode::serialize(&canon).unwrap();
        // Loader requires a complete canonical DB (so it can build a total dense map).
        // With a small AND bound we expect the DB to be incomplete; that's not a bug
        // in serialization, and we want this test to be fast.
        match CutDb::load_from_reader(bytes.as_slice()) {
            Ok(_) => panic!("expected load failure due to incomplete canonical DB"),
            Err(crate::cut_db::loader::LoadError::MissingCanonicalEntry { .. }) => {}
            Err(other) => panic!("expected MissingCanonicalEntry, got: {:?}", other),
        };
    }
}
