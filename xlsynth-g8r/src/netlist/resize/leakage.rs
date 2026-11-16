// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// Per-master scalar leakage in nW at one operating condition, keyed by the
/// master names in the associated timing library. Missing cells are unknown.
#[derive(Debug, Deserialize)]
pub struct RecoveryCellLeakage {
    liberty_proto_sha256: String,
    cell_leakage_nw: BTreeMap<String, f64>,
}

impl RecoveryCellLeakage {
    /// Check independently supplied leakage against the source identity of an
    /// already decoded timing model.
    pub fn from_library_digest(sidecar: &Path, digest: &str, library_path: &Path) -> Result<Self> {
        let sidecar_file =
            File::open(sidecar).with_context(|| format!("reading {}", sidecar.display()))?;
        let data: Self = serde_json::from_reader(BufReader::new(sidecar_file))
            .with_context(|| format!("parsing cell leakage in {}", sidecar.display()))?;
        data.validate(digest, library_path)
    }

    /// Validate the same sidecar bytes a caller hashed for its provenance
    /// report. `digest` identifies the exact timing library, in lowercase
    /// SHA-256 hex.
    pub fn from_bytes(bytes: &[u8], digest: &str, library_path: &Path) -> Result<Self> {
        let table: Self = serde_json::from_slice(bytes).context("parsing cell leakage table")?;
        table.validate(digest, library_path)
    }

    fn validate(self, digest: &str, timing_library: &Path) -> Result<Self> {
        ensure!(
            self.liberty_proto_sha256 == digest,
            "cell leakage table does not match timing library {}",
            timing_library.display()
        );
        ensure!(!self.cell_leakage_nw.is_empty(), "empty cell leakage table");
        for (cell, leakage) in &self.cell_leakage_nw {
            ensure!(
                leakage.is_finite() && *leakage >= 0.0,
                "invalid nominal leakage for {cell}"
            );
        }
        Ok(self)
    }

    pub(super) fn get(&self, cell: &str) -> Option<f64> {
        self.cell_leakage_nw.get(cell).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::RecoveryCellLeakage;
    use sha2::{Digest, Sha256};
    use std::fs;

    // Distinguish a characterized zero from a missing cell. Negative prices
    // and prices for a different timing-library digest must fail validation.
    #[test]
    fn scalar_table_is_bound_to_its_original_library() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let lib = dir.path().join("timing");
        let table = dir.path().join("leakage.json");
        let bytes = b"two area-tied cells";
        let digest = format!("{:x}", Sha256::digest(bytes));
        let characterize = |leakage| {
            serde_json::json!({
                "liberty_proto_sha256": digest, "cell_leakage_nw": {"INV": leakage}
            })
        };
        fs::write(&table, characterize(0.0).to_string())?;
        let valid = RecoveryCellLeakage::from_library_digest(&table, &digest, &lib)?;
        assert_eq!(valid.get("INV"), Some(0.0));
        assert_eq!(valid.get("BUF"), None);
        let recorded = fs::read(&table)?;
        assert_eq!(
            RecoveryCellLeakage::from_bytes(&recorded, &digest, &lib)?.get("INV"),
            Some(0.0)
        );
        assert!(RecoveryCellLeakage::from_bytes(&recorded, "wrong library", &lib).is_err());
        fs::write(&table, characterize(-1.0).to_string())?;
        assert!(RecoveryCellLeakage::from_library_digest(&table, &digest, &lib).is_err());
        fs::write(&table, characterize(3.0).to_string())?;
        let replacement = format!("{:x}", Sha256::digest(b"different cells"));
        assert!(RecoveryCellLeakage::from_library_digest(&table, &replacement, &lib).is_err());
        Ok(())
    }
}
