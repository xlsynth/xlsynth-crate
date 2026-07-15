// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use pretty_assertions::assert_eq;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{
    DepletableBytes, RandomBlockOptions, RandomFnOptions, StopPolicy, generate_block_package,
};

fuzz_target!(|data: &[u8]| {
    let mut entropy = DepletableBytes::new(data);
    let options = RandomBlockOptions {
        max_input_ports: 6,
        max_output_ports: 4,
        max_registers: 4,
        function_options: RandomFnOptions {
            max_nodes: 64,
            max_bit_width: 16,
            allow_arbitrary_width_multiply: true,
            allow_gate: true,
            allow_assumed_in_bounds: true,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    let generated = generate_block_package(
        &mut entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed random block options should construct a valid package");
    let ir_text = generated.package.to_string();
    let reparsed = Parser::new(&ir_text)
        .parse_and_validate_package()
        .expect("generated random block package should parse and validate");

    assert_eq!(reparsed.to_string(), ir_text);
});
