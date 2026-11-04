// SPDX-License-Identifier: Apache-2.0

//! Records standalone campaign outcomes and their replay inputs.

use crate::CheckFailure;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use xlsynth_prover::prover::types::EquivResult;

/// Outcome ledger and optional input retention for a fuzz process.
pub struct Campaign {
    proof_results: Option<Mutex<File>>,
    inconclusive_dir: Option<PathBuf>,
    findings_dir: Option<PathBuf>,
}

/// Atomically publishes a replay input or the first diagnostic for its class.
fn retain(directory: &Path, name: &str, data: &[u8]) -> std::io::Result<()> {
    let path = directory.join(name);
    if path.exists() {
        return Ok(());
    }
    let mut unit = tempfile::NamedTempFile::new_in(directory)?;
    unit.write_all(data)?;
    match unit.persist_noclobber(path) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.error),
    }
}

impl Campaign {
    /// Opens campaign outputs requested in the environment.
    pub fn from_env(standalone: bool) -> Self {
        let proof_results = std::env::var_os("DSLX_PROOF_RESULTS").map(|p| {
            Mutex::new(
                File::options()
                    .create(true)
                    .append(true)
                    .open(p)
                    .expect("open DSLX_PROOF_RESULTS"),
            )
        });
        let inconclusive_dir = std::env::var_os("DSLX_INCONCLUSIVE_DIR").map(PathBuf::from);
        if let Some(directory) = &inconclusive_dir {
            std::fs::create_dir_all(directory).expect("create DSLX_INCONCLUSIVE_DIR");
        }
        let findings_dir = std::env::var_os("DSLX_FINDINGS_DIR").map(PathBuf::from);
        if let Some(directory) = &findings_dir {
            assert!(standalone, "DSLX_FINDINGS_DIR requires DSLX_FUZZ_CONVERTER");
            for kind in ["disproved", "converter-rejected"] {
                std::fs::create_dir_all(directory.join(kind)).expect("create finding directory");
            }
        }
        Self {
            proof_results,
            inconclusive_dir,
            findings_dir,
        }
    }

    /// Retains a finding before recording its outcome. Checker errors are
    /// fatal.
    pub fn record(
        &self,
        data: &[u8],
        family: Option<&str>,
        result: Result<EquivResult, CheckFailure>,
    ) {
        let mark = match result {
            Ok(EquivResult::Proved) => b'P',
            Ok(EquivResult::Inconclusive(_)) => {
                // A solver limit leaves this sample unresolved; retain it for a
                // longer replay.
                if let Some(directory) = &self.inconclusive_dir {
                    retain(directory, &format!("{:x}", Sha256::digest(data)), data)
                        .expect("save inconclusive input");
                }
                b'I'
            }
            Err(failure) => {
                let Some(directory) = &self.findings_dir else {
                    panic!("DSLX semantics check failed: {}", failure.into_message());
                };
                let (kind, mark, message) = match failure {
                    CheckFailure::Disproved(message) => ("disproved", b'D', message),
                    CheckFailure::ConverterRejected(message) => {
                        ("converter-rejected", b'R', message)
                    }
                    CheckFailure::InvalidOutput(message) => ("invalid-output", b'X', message),
                    CheckFailure::StructuralGrowth(message) => ("structural-growth", b'G', message),
                    CheckFailure::Checker(message) => {
                        panic!("DSLX semantics check failed: {message}")
                    }
                };
                let digest = format!("{:x}", Sha256::digest(data));
                let directory = directory.join(kind);
                let directory = match family {
                    Some(family) => directory.join(family),
                    None => directory,
                };
                std::fs::create_dir_all(&directory).expect("create finding directory");
                retain(&directory, &digest, data).expect("save failing input");
                retain(
                    &directory,
                    "first.txt",
                    format!("{digest}\n{message}").as_bytes(),
                )
                .expect("save first diagnostic");
                mark
            }
            _ => unreachable!("check_prepared reports prover errors and mismatches as failures"),
        };
        if let Some(results) = &self.proof_results {
            results
                .lock()
                .unwrap()
                .write_all(&[mark])
                .expect("write proof outcome");
        }
    }
}
