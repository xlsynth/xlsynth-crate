// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_random::{generate_fn, DepletableBytes, RandomFnOptions, StopPolicy};
use xlsynth_pir::structural_similarity::structurally_equivalent_ir;

fuzz_target!(|data: &[u8]| {
    log::debug!("Testing random PIR function through libxls IR roundtrip");
    let _ = env_logger::builder().is_test(true).try_init();

    // 1) Generate valid PIR directly, then parse and re-emit it through
    // libxls so the Rust parser continues to see XLS-emitted IR.
    let mut entropy = DepletableBytes::new(data);
    let options = RandomFnOptions {
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        ..RandomFnOptions::default()
    };
    let generated = generate_fn(
        &mut entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed random PIR options should construct a valid function");
    let pir_text = generated.into_top_package("fuzz_pkg").to_string();
    let xls_pkg = xlsynth::IrPackage::parse_ir(&pir_text, None)
        .expect("PIR-emitted standard XLS IR should parse in libxls");
    let pkg_text = xls_pkg.to_string();

    // 2) Parse the libxls-emitted text back via our Rust parser.
    let parsed_pkg = ir_parser::Parser::new(&pkg_text)
        .parse_and_validate_package()
        .expect("libxls-emitted IR failed to parse/validate in Rust parser");

    // 3) Verify and obtain the top function
    let parsed_top = parsed_pkg
        .get_top_fn()
        .expect("generator should set a top function");

    // 4) Emit the function text via Display and parse the function alone, too
    let func_text = parsed_top.to_string();
    let reparsed_top = ir_parser::Parser::new(&func_text)
        .parse_fn()
        .expect("Function pretty-printer emitted text that failed to reparse");

    // 5) Sanity: re-serialize package and reparse to ensure package-level printer
    //    is sound
    let reparsed_pkg = ir_parser::Parser::new(&parsed_pkg.to_string())
        .parse_and_validate_package()
        .expect("Package pretty-printer emitted IR that failed to reparse/validate");
    let reparsed_pkg_top = reparsed_pkg
        .get_top_fn()
        .expect("package pretty-printer should preserve top function");

    // 6) Structural equivalence checks between original parsed top and both
    //    reparsed variants
    assert!(structurally_equivalent_ir(parsed_top, &reparsed_top));
    assert!(structurally_equivalent_ir(parsed_top, reparsed_pkg_top));
});
