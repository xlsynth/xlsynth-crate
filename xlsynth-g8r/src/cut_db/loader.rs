// SPDX-License-Identifier: Apache-2.0

//! Runtime loader for the 4-input canonical cut database.

use std::io;

use serde::{Deserialize, Serialize};

use crate::cut_db::npn::{NpnTransform, unpack_npn_transform};
use crate::cut_db::serdes::{CanonDbOnDisk, CanonDbOnDiskV1, CanonEntry, upgrade_v1_to_v2};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
use crate::cut_db::npn::canon_tt16;
#[cfg(test)]
use crate::cut_db::tt16::TruthTable16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseInfo {
    pub canon_index: u16,
    pub xform: NpnTransform,
}

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    Bincode(Box<bincode::ErrorKind>),
    UnsupportedVersion {
        got: u32,
    },
    MissingCanonicalEntry {
        canon_tt: u16,
    },
    DenseNotFullyCovered {
        filled: usize,
    },
    DenseWrongLength {
        got: usize,
    },
    DenseBadCanonIndex {
        canon_index: u16,
        entries_len: usize,
    },
}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<Box<bincode::ErrorKind>> for LoadError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        Self::Bincode(e)
    }
}

pub struct CutDb {
    canon_entries: Vec<CanonEntry>,
    dense: Vec<DenseInfo>, // length 65536
}

impl CutDb {
    /// Loads the in-tree default cut DB artifact (embedded via
    /// `include_bytes!`), returning a process-wide cached `Arc`.
    pub fn load_default() -> Arc<Self> {
        static DEFAULT_DB: OnceLock<Arc<CutDb>> = OnceLock::new();
        DEFAULT_DB
            .get_or_init(|| {
                static CUT_DB_BYTES: &[u8] =
                    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/cut_db_v1.bin"));
                let db = CutDb::load_from_reader(CUT_DB_BYTES)
                    .expect("failed to load embedded cut_db_v1.bin");
                Arc::new(db)
            })
            .clone()
    }

    pub fn load_from_reader(mut r: impl io::Read) -> Result<Self, LoadError> {
        // Prefer v2 format (contains dense table). If that fails, try legacy v1 and
        // upgrade (slow; only for compatibility).
        let bytes: Vec<u8> = {
            let mut v = Vec::new();
            r.read_to_end(&mut v)?;
            v
        };

        let v2: Result<CanonDbOnDisk, Box<bincode::ErrorKind>> =
            bincode::deserialize(bytes.as_slice());
        let on_disk = match v2 {
            Ok(db) => db,
            Err(_) => {
                let v1: CanonDbOnDiskV1 = bincode::deserialize(bytes.as_slice())?;
                upgrade_v1_to_v2(v1).map_err(|e| {
                    // Preserve the historical error flavor for partial DBs.
                    match e {
                        crate::cut_db::serdes::BuildDenseError::MissingCanonicalEntry {
                            canon_tt,
                        } => LoadError::MissingCanonicalEntry { canon_tt },
                    }
                })?
            }
        };

        if on_disk.version != 2 {
            return Err(LoadError::UnsupportedVersion {
                got: on_disk.version,
            });
        }

        if on_disk.dense.len() != 65536 {
            return Err(LoadError::DenseWrongLength {
                got: on_disk.dense.len(),
            });
        }

        let canon_entries = on_disk.entries;
        let entries_len = canon_entries.len();

        let mut dense: Vec<DenseInfo> = Vec::with_capacity(65536);
        for packed in on_disk.dense {
            let canon_index = (packed & 0xFFFF) as u16;
            let xform_packed = (packed >> 16) as u16;
            if (canon_index as usize) >= entries_len {
                return Err(LoadError::DenseBadCanonIndex {
                    canon_index,
                    entries_len,
                });
            }
            dense.push(DenseInfo {
                canon_index,
                xform: unpack_npn_transform(xform_packed),
            });
        }

        Ok(Self {
            canon_entries,
            dense,
        })
    }

    pub fn lookup(&self, tt: u16) -> (NpnTransform, &[crate::cut_db::pareto::ParetoPoint]) {
        let info = self.dense[tt as usize];
        let entry = &self.canon_entries[info.canon_index as usize];
        (info.xform, entry.pareto.as_slice())
    }
}

#[cfg(test)]
impl CutDb {
    pub(crate) fn from_raw_for_test(canon_entries: Vec<CanonEntry>, dense: Vec<DenseInfo>) -> Self {
        assert_eq!(dense.len(), 65536);
        Self {
            canon_entries,
            dense,
        }
    }

    /// Creates a `CutDb` suitable for tests from a partial set of canonical
    /// entries.
    ///
    /// Any lookup whose canonical truth table is not present in `entries` will
    /// return an empty pareto list. This keeps unit tests fast without
    /// requiring a full 65536-entry database.
    pub fn new_for_test_partial(mut entries: Vec<CanonEntry>) -> Self {
        // Ensure there is at least one entry so we have a valid canon_index to
        // point at for missing canonical truth tables.
        if entries.is_empty() {
            entries.push(CanonEntry {
                canon_tt: TruthTable16(0),
                pareto: Vec::new(),
            });
        }

        // Deterministic ordering by canon_tt.
        entries.sort_by_key(|e| e.canon_tt.0);

        let mut canon_index_of_tt: Vec<u16> = vec![u16::MAX; 65536];
        for (i, entry) in entries.iter().enumerate() {
            canon_index_of_tt[entry.canon_tt.0 as usize] = i as u16;
        }

        let mut dense: Vec<DenseInfo> = vec![
            DenseInfo {
                canon_index: 0,
                xform: NpnTransform::identity(),
            };
            65536
        ];

        for tt_u in 0u32..=0xFFFF {
            let tt = TruthTable16(tt_u as u16);
            let (canon_tt, xform) = canon_tt16(tt);
            let canon_idx = canon_index_of_tt[canon_tt.0 as usize];
            if canon_idx != u16::MAX {
                dense[tt_u as usize] = DenseInfo {
                    canon_index: canon_idx,
                    xform,
                };
            } else {
                // Missing canonical entry => map to entry 0 which should be empty.
                dense[tt_u as usize] = DenseInfo {
                    canon_index: 0,
                    xform,
                };
            }
        }

        Self {
            canon_entries: entries,
            dense,
        }
    }
}
