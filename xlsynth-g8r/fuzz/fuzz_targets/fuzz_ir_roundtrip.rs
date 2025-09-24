// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_fuzz::{FuzzSample, generate_ir_fn};
use xlsynth_pir::structural_similarity::structurally_equivalent_ir;
use xlsynth_pir::{ir, ir_parser};

fuzz_target!(|sample: FuzzSample| {
    log::debug!("Testing FuzzSample IR roundtrip");
    // Skip degenerate samples early.
    if sample.ops.is_empty() {
        // Degenerate generator inputs (no ops or zero-width inputs) are not
        // interesting for this target and can arise frequently. We intentionally
        // skip rather than crash to avoid biasing the corpus toward trivial cases.
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // 1) Generate XLS IR via C++ bindings into a package
    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg")
        .expect("IrPackage::new should not fail; treat as infra error");
    if let Err(_) = generate_ir_fn(sample.ops.clone(), &mut pkg, None) {
        // The generator can deliberately produce edge-case or temporarily unsupported
        // constructs; those are not failures of this roundtrip target. Skip to let
        // the generator evolve without turning such cases into crashes here.
        return;
    }
    log::debug!("FuzzSample generated valid IR");

    // 2) Serialize to text and parse back via our Rust parser
    let pkg_text = pkg.to_string();
    let parsed_pkg = ir_parser::Parser::new(&pkg_text)
        .parse_and_validate_package()
        .expect("C++-emitted IR failed to parse/validate in Rust parser");

    // 3) Verify and obtain the top function
    let parsed_top = parsed_pkg
        .get_top()
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
        .get_top()
        .expect("package pretty-printer should preserve top function");

    // 6) Structural equivalence checks between original parsed top and both
    //    reparsed variants
    assert!(structurally_equivalent_ir(parsed_top, &reparsed_top));
    assert!(structurally_equivalent_ir(parsed_top, reparsed_pkg_top));
});
