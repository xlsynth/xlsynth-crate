// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks deterministic functional behavior across supported codegen options.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{Profile, input::FuzzCase, top_block};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let Ok(mut case) = FuzzCase::decode(data, Profile::NativeSemantics) else {
        // An unknown explicit input-format version is not a generated RTL bug.
        return;
    };
    let flags = data.get(42).copied().unwrap_or_default();
    if flags & 1 != 0 {
        case.options.top = Some(top_block(&case.package).0.name.clone());
    }
    if flags & 2 != 0 {
        case.options.module_name = Some(format!("public_module_{flags}"));
    }
    // Bounds checks remain enabled: omitting a required clamp intentionally
    // changes XLS semantics, so it is not an equivalent codegen option.
    let checked = case.check();
    if let Some(report) = xlsynth_codegen_fuzz::coverage::record_progress(
        &case, checked.coverage_outcome(), checked.checked_trace(),
    ) {
        eprintln!("codegen-coverage {report}");
    }
});
