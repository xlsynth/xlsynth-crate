// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Compares native block emission with independently generated stock-XLS RTL.

use libfuzzer_sys::fuzz_target;
use std::{path::PathBuf, sync::OnceLock};
use xlsynth_codegen_fuzz::coverage::record_progress;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{Profile, input::FuzzCase};
static ARTIFACTS: OnceLock<Option<PathBuf>> = OnceLock::new();

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::StockXls) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let profile = Profile::StockXls;
    let Ok(case) = FuzzCase::decode(data, profile) else {
        // An unknown explicit input-format version is not a generated RTL bug.
        return;
    };
    let directory = ARTIFACTS.get_or_init(|| {
        std::env::args().find_map(|arg| {
            arg.strip_prefix("-artifact_prefix=")
                .map(|p| PathBuf::from(p).join("current"))
        })
    });
    let checked = case.check_with_artifacts(directory.as_deref());
    if let Some(report) = record_progress(
        &case,
        checked.coverage_outcome(),
        checked.checked_trace(),
    ) {
        eprintln!("codegen-coverage {report}");
    }
});
