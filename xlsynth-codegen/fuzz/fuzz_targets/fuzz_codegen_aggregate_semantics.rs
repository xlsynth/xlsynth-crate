// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks array and tuple flattening against independent typed block
//! evaluation.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{assert_combinational_semantics, block_options, generate};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let package = generate(data, &block_options(true, false));
    xlsynth_codegen_fuzz::fuzz_tool!(assert_combinational_semantics(
        &package,
        &BlockCodegenOptions::default(),
    ));
});
