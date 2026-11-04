// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, generate_with_grammar};
use xlsynth_dslx_fuzzer_fuzz::{
    CheckFailure, check_prepared, cli::classify_cli, native::NativeChecker,
};
use xlsynth_prover::prover::types::EquivResult;

/// Process-wide generator, converter, and optional corpus recording settings.
struct Config {
    grammar: Grammar,
    native: Option<NativeChecker>,
    converter: Option<PathBuf>,
    stdlib: PathBuf,
    cli_profile: Option<PathBuf>,
    proof_results: Option<Mutex<File>>,
    inconclusive_dir: Option<PathBuf>,
    findings_dir: Option<PathBuf>,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

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

impl Config {
    /// Reads the generator profile and optional experiment paths once at
    /// startup.
    fn from_env() -> Self {
        let grammar = std::env::var_os("DSLX_FUZZ_GRAMMAR")
            .unwrap_or_else(|| "mission".into())
            .into_string()
            .expect("DSLX_FUZZ_GRAMMAR must be UTF-8")
            .parse::<Grammar>()
            .expect("invalid DSLX_FUZZ_GRAMMAR");
        let native = NativeChecker::from_build().expect("native DSLX configuration failed");
        if let Some(checker) = &native {
            eprintln!(
                "Native DSLX feedback: {} inline 8-bit counters",
                checker.counter_snapshot().len()
            );
        }
        let converter = std::env::var_os("DSLX_FUZZ_CONVERTER").map(|p| {
            std::fs::canonicalize(p).expect("DSLX_FUZZ_CONVERTER must name an executable file")
        });
        let stdlib = std::env::var_os("DSLX_STDLIB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| xlsynth::default_dslx_stdlib_path().to_owned());
        let stdlib = std::path::absolute(stdlib).expect("invalid DSLX_STDLIB_PATH");
        let cli_profile = std::env::var_os("DSLX_CLI_PROFILE")
            .map(|p| std::path::absolute(p).expect("invalid DSLX_CLI_PROFILE"));
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
            assert!(
                converter.is_some(),
                "DSLX_FINDINGS_DIR requires DSLX_FUZZ_CONVERTER"
            );
            for kind in ["disproved", "converter-rejected"] {
                std::fs::create_dir_all(directory.join(kind)).expect("create finding directory");
            }
        }
        Self {
            grammar,
            native,
            converter,
            stdlib,
            cli_profile,
            proof_results,
            inconclusive_dir,
            findings_dir,
        }
    }

    /// Checks one mutation and records its outcome after any requested input
    /// retention.
    fn check(&self, data: &[u8]) {
        let sample = generate_with_grammar(data, GenerateOptions::default(), self.grammar)
            .unwrap_or_else(|error| panic!("generation failed: {error}"));
        let result = if let Some(converter) = &self.converter {
            classify_cli(
                &sample,
                self.grammar,
                converter,
                &self.stdlib,
                self.cli_profile.as_deref(),
                self.native.as_ref(),
            )
        } else {
            if let Some(checker) = &self.native {
                let modules = sample.sources().expect("source formatting failed");
                checker.check(&modules).unwrap_or_else(|error| {
                    panic!("native DSLX check failed: {error}\n{modules:#?}")
                });
            }
            let prepared = sample.prepare().unwrap_or_else(|error| {
                panic!(
                    "DSLX preparation failed: {error}\nmodules: {:#?}\nreference: {:#?}",
                    sample.sources(),
                    sample.expected_ir()
                )
            });
            check_prepared(&prepared, self.grammar).map_err(CheckFailure::Checker)
        };

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
                    CheckFailure::Checker(message) => {
                        panic!("DSLX semantics check failed: {message}")
                    }
                };
                let digest = format!("{:x}", Sha256::digest(data));
                let directory = directory.join(kind);
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

fuzz_target!(init: {
    assert!(CONFIG.set(Config::from_env()).is_ok(), "fuzzer initialized twice");
}, |data: &[u8]| {
    CONFIG.get().expect("fuzzer not initialized").check(data);
});
