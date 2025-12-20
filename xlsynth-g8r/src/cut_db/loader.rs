// SPDX-License-Identifier: Apache-2.0

//! Runtime loader for the 4-input canonical cut database.

use std::io;

use serde::{Deserialize, Serialize};

use crate::cut_db::npn::{canon_tt16, NpnTransform};
use crate::cut_db::serdes::{CanonDbOnDisk, CanonEntry};
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
    UnsupportedVersion { got: u32 },
    MissingCanonicalEntry { canon_tt: u16 },
    DenseNotFullyCovered { filled: usize },
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
    pub fn load_from_reader(mut r: impl io::Read) -> Result<Self, LoadError> {
        let on_disk: CanonDbOnDisk = bincode::deserialize_from(&mut r)?;
        if on_disk.version != 1 {
            return Err(LoadError::UnsupportedVersion {
                got: on_disk.version,
            });
        }

        let canon_entries = on_disk.entries;

        // Build a dense mapping from canon_tt to canon_index.
        let mut canon_index_of_tt: Vec<u16> = vec![u16::MAX; 65536];
        for (i, entry) in canon_entries.iter().enumerate() {
            let idx = entry.canon_tt.0 as usize;
            canon_index_of_tt[idx] = i as u16;
        }

        let mut dense: Vec<DenseInfo> = vec![
            DenseInfo {
                canon_index: 0,
                xform: NpnTransform::identity(),
            };
            65536
        ];

        let mut filled: usize = 0;
        for tt_u in 0u32..=0xFFFF {
            let tt = TruthTable16(tt_u as u16);
            let (canon_tt, xform) = canon_tt16(tt);
            let canon_idx = canon_index_of_tt[canon_tt.0 as usize];
            if canon_idx == u16::MAX {
                return Err(LoadError::MissingCanonicalEntry {
                    canon_tt: canon_tt.0,
                });
            }
            dense[tt_u as usize] = DenseInfo {
                canon_index: canon_idx,
                xform,
            };
            filled += 1;
        }

        if filled != 65536 {
            return Err(LoadError::DenseNotFullyCovered { filled });
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
