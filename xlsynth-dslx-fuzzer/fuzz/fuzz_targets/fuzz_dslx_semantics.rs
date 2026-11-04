// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use std::sync::OnceLock;
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, generate_with_grammar};
use xlsynth_dslx_fuzzer_fuzz::{
    CheckFailure, campaign::Campaign, check_prepared, cli::classify_cli, native::NativeChecker,
};

/// Process-wide generator, converter, and optional corpus recording settings.
struct Config {
    grammar: Grammar,
    native: Option<NativeChecker>,
    converter: Option<PathBuf>,
    stdlib: PathBuf,
    cli_profile: Option<PathBuf>,
    campaign: Campaign,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

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
        let campaign = Campaign::from_env(converter.is_some());
        Self {
            grammar,
            native,
            converter,
            stdlib,
            cli_profile,
            campaign,
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

        self.campaign.record(data, None, result);
    }
}

fuzz_target!(init: {
    assert!(CONFIG.set(Config::from_env()).is_ok(), "fuzzer initialized twice");
}, |data: &[u8]| {
    CONFIG.get().expect("fuzzer not initialized").check(data);
});
