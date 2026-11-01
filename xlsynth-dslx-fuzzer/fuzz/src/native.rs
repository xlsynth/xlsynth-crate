// SPDX-License-Identifier: Apache-2.0

//! Optional in-process C++ checking with counters registered into libFuzzer.

#[cfg(xlsynth_dslx_native)]
use std::ffi::{CStr, CString, c_char, c_int};
use xlsynth_dslx_fuzzer::SourceModule;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};

#[cfg(xlsynth_dslx_native)]
unsafe extern "C" {
    fn dslx_fuzz_native_check(
        stdlib: *const c_char,
        entry: *const c_char,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn dslx_fuzz_native_counters(output: *mut u8, capacity: usize) -> usize;
}

/// A validated native checker, present only when configured during Cargo build.
pub struct NativeChecker {
    #[cfg(xlsynth_dslx_native)]
    stdlib: CString,
    _private: (),
}

impl NativeChecker {
    /// Validates native instrumentation and stdlib configuration at startup.
    pub fn from_build() -> Result<Option<Self>, String> {
        #[cfg(xlsynth_dslx_native)]
        {
            let stdlib =
                CString::new(env!("DSLX_FUZZ_NATIVE_STDLIB")).map_err(|error| error.to_string())?;
            // SAFETY: a null output requests only the immutable section size.
            if unsafe { dslx_fuzz_native_counters(std::ptr::null_mut(), 0) } == 0 {
                return Err("native DSLX library has no inline 8-bit counters; rebuild it with native/build.py".to_string());
            }
            Ok(Some(Self {
                stdlib,
                _private: (),
            }))
        }
        #[cfg(not(xlsynth_dslx_native))]
        {
            Ok(None)
        }
    }

    /// Parses/typechecks fresh modules; constexpr evaluation executes C++
    /// bytecode.
    pub fn check(&self, modules: &[SourceModule]) -> Result<(), String> {
        #[cfg(xlsynth_dslx_native)]
        {
            let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
            let mut names = std::collections::BTreeSet::new();
            for module in modules {
                if !module
                    .name
                    .as_bytes()
                    .first()
                    .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
                    || !module
                        .name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                    || !names.insert(module.name.clone())
                {
                    return Err("native modules require unique identifier names".to_string());
                }
                std::fs::write(
                    directory.path().join(format!("{}.x", module.name)),
                    &module.text,
                )
                .map_err(|error| error.to_string())?;
            }
            if !names.contains("sample") {
                return Err("native environment requires a sample module".to_string());
            }
            let entry = directory.path().join("sample.x");
            let entry = CString::new(entry.as_os_str().as_encoded_bytes())
                .map_err(|error| error.to_string())?;
            let mut error = vec![0u8; 16 * 1024];
            // SAFETY: both paths are NUL-terminated and live through the call;
            // the error allocation is writable for exactly the supplied size.
            let status = unsafe {
                dslx_fuzz_native_check(
                    self.stdlib.as_ptr(),
                    entry.as_ptr(),
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                // The native ABI always terminates a nonempty error buffer.
                let message =
                    CStr::from_bytes_until_nul(&error).map_err(|error| error.to_string())?;
                Err(message.to_string_lossy().into_owned())
            }
        }
        #[cfg(not(xlsynth_dslx_native))]
        {
            let _ = modules;
            Err("native DSLX feedback was not configured at build time".to_string())
        }
    }

    /// Snapshots the same native counters libFuzzer reads and clears per input.
    pub fn counter_snapshot(&self) -> Vec<u8> {
        #[cfg(xlsynth_dslx_native)]
        {
            // SAFETY: the library remains loaded and its section size is fixed.
            let count = unsafe { dslx_fuzz_native_counters(std::ptr::null_mut(), 0) };
            let mut counters = vec![0; count];
            // SAFETY: the allocation covers the entire counter section.
            unsafe {
                dslx_fuzz_native_counters(counters.as_mut_ptr(), counters.len());
            }
            counters
        }
        #[cfg(not(xlsynth_dslx_native))]
        {
            vec![]
        }
    }
}

/// Typed evidence that native edges vary with samples and the oracle still
/// runs.
#[derive(Debug)]
pub struct FeedbackReport {
    pub native_counters: usize,
    pub covered_before: usize,
    pub covered_after: usize,
    pub inputs_with_new_edges: usize,
    pub proved_samples: usize,
    pub wrong_reference_disproved: bool,
}

/// Checks native feedback, error propagation, and released-libxls isolation.
pub fn probe_feedback(inputs: &[Vec<u8>]) -> Result<FeedbackReport, String> {
    let checker = NativeChecker::from_build()?
        .ok_or("set DSLX_FUZZ_NATIVE_CONFIG during Cargo build to enable native feedback")?;
    let mut seen: Vec<_> = checker
        .counter_snapshot()
        .into_iter()
        .map(|value| value != 0)
        .collect();
    let mut report = FeedbackReport {
        native_counters: seen.len(),
        covered_before: seen.iter().filter(|value| **value).count(),
        covered_after: 0,
        inputs_with_new_edges: 0,
        proved_samples: 0,
        wrong_reference_disproved: false,
    };
    for bytes in inputs {
        let sample = xlsynth_dslx_fuzzer::generate_with_options(
            bytes,
            xlsynth_dslx_fuzzer::GenerateOptions { max_width: 8 },
        )?;
        checker.check(&sample.sources()?)?;
        let counters = checker.counter_snapshot();
        let mut new_edges = 0;
        for (seen, value) in seen.iter_mut().zip(&counters) {
            if !*seen && *value != 0 {
                *seen = true;
                new_edges += 1;
            }
        }
        report.inputs_with_new_edges += usize::from(new_edges != 0);
        let prepared = sample.prepare()?;
        let result = crate::prove(&prepared)?;
        if !matches!(result, xlsynth_prover::prover::types::EquivResult::Proved) {
            return Err(format!("oracle probe did not prove sample: {result:?}"));
        }
        if checker.counter_snapshot() != counters {
            return Err(
                "released libxls/prover unexpectedly changed the isolated native DSLX counters"
                    .to_string(),
            );
        }
        report.proved_samples += 1;
    }
    report.covered_after = seen.iter().filter(|value| **value).count();
    if report.inputs_with_new_edges < 2 {
        return Err(
            "probe requires multiple inputs that discover different native edges".to_string(),
        );
    }
    let invalid = [SourceModule {
        name: "sample".to_string(),
        text: "fn main(x: u8) -> u8 { x + true }".to_string(),
    }];
    if checker.check(&invalid).is_ok() {
        return Err("native checker accepted an invalid bits/bool addition".to_string());
    }
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(65),
    };
    let mut correct = Body::new(ty)?;
    correct.push(Fragment::Binary {
        op: BinaryOp::Add,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })?;
    let correct = xlsynth_dslx_fuzzer::Sample::new(vec![], vec![correct])?;
    checker.check(&correct.sources()?)?;
    let native_before_oracle = checker.counter_snapshot();
    let mut prepared = correct.prepare()?;
    let mut wrong = Body::new(ty)?;
    wrong.push(Fragment::Binary {
        op: BinaryOp::Sub,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })?;
    prepared.expected_ir = xlsynth_dslx_fuzzer::Sample::new(vec![], vec![wrong])?
        .expected_ir()?
        .to_string();
    report.wrong_reference_disproved = matches!(
        crate::prove(&prepared)?,
        xlsynth_prover::prover::types::EquivResult::Disproved { .. }
    );
    if !report.wrong_reference_disproved || checker.counter_snapshot() != native_before_oracle {
        return Err("negative oracle/isolation probe failed after native checking".to_string());
    }
    Ok(report)
}
