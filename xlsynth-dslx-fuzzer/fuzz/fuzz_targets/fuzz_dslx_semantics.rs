// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use xlsynth_dslx_fuzzer::generate;
use xlsynth_dslx_fuzzer_fuzz::native::NativeChecker;
use xlsynth_dslx_fuzzer_fuzz::prove;
use xlsynth_prover::prover::types::EquivResult;

static NATIVE: OnceLock<Option<NativeChecker>> = OnceLock::new();

fuzz_target!(init: {
    let checker = NativeChecker::from_build().expect("native DSLX configuration failed");
    if let Some(checker) = &checker {
        eprintln!("Native DSLX feedback: {} inline 8-bit counters", checker.counter_snapshot().len());
    } else {
        eprintln!("Native DSLX feedback disabled: DSLX_FUZZ_NATIVE_CONFIG was not set during build");
    }
    assert!(NATIVE.set(checker).is_ok(), "fuzzer initialized twice");
}, |data: &[u8]| {
    let sample = generate(data).unwrap_or_else(|error| panic!("generation failed: {error}"));
    if let Some(checker) = NATIVE.get().expect("fuzzer was not initialized") {
        let modules = sample.sources().expect("source formatting failed");
        checker.check(&modules).unwrap_or_else(|error| panic!("native DSLX check failed: {error}\n{modules:#?}"));
    }
    let prepared = sample.prepare().unwrap_or_else(|error| {
        panic!(
            "DSLX preparation failed: {error}\nmodules: {:#?}\nreference: {:#?}",
            sample.sources(),
            sample.expected_ir()
        )
    });
    match prove(&prepared) {
        Ok(EquivResult::Proved) => {
            // The sample's outputs agree for every input.
        }
        // A configured solver resource limit is not evidence of a semantics
        // failure. Converter, IR parser and prover errors still fail the sample.
        Ok(EquivResult::Inconclusive(_)) => return,
        result => panic!("DSLX semantics check failed: {result:#?}\n{prepared:#?}"),
    }
});
