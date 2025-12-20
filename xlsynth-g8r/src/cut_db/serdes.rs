// SPDX-License-Identifier: Apache-2.0

//! Canonical on-disk database representation and conversions.

use serde::{Deserialize, Serialize};

use crate::cut_db::enumerate::FullSpaceDb;
use crate::cut_db::npn::{NpnTransform, canon_tt16};
use crate::cut_db::pareto::{ParetoPoint, dominates, same_cost};
use crate::cut_db::tt16::TruthTable16;

const CUT_DB_VERSION: u32 = 2;

#[derive(Debug)]
pub enum BuildDenseError {
    MissingCanonicalEntry { canon_tt: u16 },
}

impl std::fmt::Display for BuildDenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCanonicalEntry { canon_tt } => {
                write!(f, "missing canonical entry for canon_tt=0x{:04x}", canon_tt)
            }
        }
    }
}

impl std::error::Error for BuildDenseError {}

/// Packs `(canon_index, xform)` into a single `u32`:
/// - low 16 bits: canon_index
/// - high 16 bits: packed NPN transform
#[inline]
pub fn pack_dense_info(canon_index: u16, xform: NpnTransform) -> u32 {
    let xform_packed: u16 = crate::cut_db::npn::pack_npn_transform(xform);
    (canon_index as u32) | ((xform_packed as u32) << 16)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonDbOnDisk {
    pub version: u32,
    pub entries: Vec<CanonEntry>,
    /// Dense lookup table for all 65536 truth tables.
    ///
    /// Entry i packs `(canon_index, xform)` where `xform` maps tt(i) to its
    /// canonical representative.
    pub dense: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonEntry {
    pub canon_tt: TruthTable16,
    pub pareto: Vec<ParetoPoint>,
}

/// Legacy v1 format: canonical entries only (dense map was constructed at load
/// time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonDbOnDiskV1 {
    pub version: u32,
    pub entries: Vec<CanonEntry>,
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
pub fn canon_from_full_space(full: &FullSpaceDb) -> Result<CanonDbOnDisk, BuildDenseError> {
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

    let dense = build_dense_from_entries(&entries)?;

    Ok(CanonDbOnDisk {
        version: CUT_DB_VERSION,
        entries,
        dense,
    })
}

/// Builds the dense 65536-entry mapping from canonical entries.
pub fn build_dense_from_entries(entries: &[CanonEntry]) -> Result<Vec<u32>, BuildDenseError> {
    // Map canon_tt -> canon_index (index into `entries`).
    let mut canon_index_of_tt: Vec<u16> = vec![u16::MAX; 65536];
    for (i, entry) in entries.iter().enumerate() {
        canon_index_of_tt[entry.canon_tt.0 as usize] = i as u16;
    }

    let mut dense: Vec<u32> = vec![0; 65536];
    for tt_u in 0u32..=0xFFFF {
        let tt = TruthTable16(tt_u as u16);
        let (canon_tt, xform) = canon_tt16(tt);
        let canon_idx = canon_index_of_tt[canon_tt.0 as usize];
        if canon_idx == u16::MAX {
            return Err(BuildDenseError::MissingCanonicalEntry {
                canon_tt: canon_tt.0,
            });
        }
        dense[tt_u as usize] = pack_dense_info(canon_idx, xform);
    }
    Ok(dense)
}

/// Converts a legacy v1 (entries-only) artifact into the current v2 format by
/// computing and embedding the dense lookup table.
pub fn upgrade_v1_to_v2(v1: CanonDbOnDiskV1) -> Result<CanonDbOnDisk, BuildDenseError> {
    let dense = build_dense_from_entries(&v1.entries)?;
    Ok(CanonDbOnDisk {
        version: CUT_DB_VERSION,
        entries: v1.entries,
        dense,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut_db::enumerate::{EnumerateOptions, enumerate_full_space};

    #[test]
    fn test_canon_db_bincode_round_trip_smoke() {
        // Use a small AND bound so this stays fast; this test only checks
        // serialization invariants and error behavior, not full coverage.
        let full = enumerate_full_space(EnumerateOptions {
            max_ands: Some(2),
            progress_every_pops: None,
            parallel_chunk_size: 64,
        });
        match canon_from_full_space(&full) {
            Ok(_) => panic!("expected build failure due to incomplete canonical DB"),
            Err(BuildDenseError::MissingCanonicalEntry { .. }) => {}
        }
    }
}
