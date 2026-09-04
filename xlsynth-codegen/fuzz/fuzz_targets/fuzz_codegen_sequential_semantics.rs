// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Compares register updates and visible outputs across deterministic clock
//! cycles.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{assert_sequential_semantics, block_options, generate};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let package = generate(data, &block_options(false, true));
    xlsynth_codegen_fuzz::fuzz_tool!(assert_sequential_semantics(
        &package,
        &BlockCodegenOptions::default(),
    ));
});
