// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../tests/common/lean_bundle.rs"]
mod lean_bundle;
#[path = "../../tests/common/parametric_cases.rs"]
mod parametric_cases;
#[path = "../../tests/common/parametric_check.rs"]
mod parametric_check;

fuzz_target!(init: {
    std::env::var_os("XLSYNTH_LEAN")
        .expect("set XLSYNTH_LEAN to the actual Lean executable");
}, |data: [u8; 11]| {
    let case = parametric_cases::generate(data);
    parametric_check::check_case(&case).unwrap_or_else(|error| {
        panic!("parametric caller mismatch for {data:?}: {error:#}\n{}", case.source)
    });
});
