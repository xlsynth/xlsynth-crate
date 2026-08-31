// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks stable block emission before and after a public PIR roundtrip.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::{block_options, emit, generate};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{BlockTopology, RandomBlockResetTiming};

fuzz_target!(|data: &[u8]| {
    let mut generation = block_options(true, true);
    generation.topology = if data.first().copied().unwrap_or(0) & 1 == 0 {
        BlockTopology::Combinational
    } else {
        BlockTopology::GeneralSequential
    };
    generation.reset_timing = RandomBlockResetTiming::Either;
    let package = generate(data, &generation);
    let options = BlockCodegenOptions::default();
    let expected = emit(&package, &options);
    assert_eq!(
        emit(&package, &options),
        expected,
        "repeated emission changed"
    );

    let ir = package.to_string();
    let roundtrip = Parser::new(&ir)
        .parse_and_validate_package()
        .unwrap_or_else(|error| panic!("generated block should roundtrip:\n{ir}\n{error}"));
    assert_eq!(
        emit(&roundtrip, &options),
        expected,
        "SystemVerilog changed after a PIR roundtrip:\n{ir}"
    );
});
