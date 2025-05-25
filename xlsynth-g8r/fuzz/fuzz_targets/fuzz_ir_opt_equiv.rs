// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::ir_equiv_boolector;
use xlsynth_g8r::xls_ir::ir_parser;
use xlsynth_test_helpers::ir_fuzz::{generate_ir_fn, FuzzSample};

fuzz_target!(|sample: FuzzSample| {
    // Ensure XLSYNTH_TOOLS is set for equivalence checking
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }

    if sample.ops.is_empty() || sample.input_bits == 0 {
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // Build an XLS IR package from the fuzz sample
    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg").unwrap();
    if let Err(e) = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut pkg) {
        log::info!("IR generation failed: {}", e);
        return;
    }

    let top_fn = pkg.get_function("fuzz_test").unwrap();

    // Optimize the IR
    let optimized_pkg = match xlsynth::optimize_ir(&pkg, top_fn.get_name().as_str()) {
        Ok(p) => p,
        Err(e) => {
            log::error!("optimize_ir failed: {}", e);
            return;
        }
    };

    // Parse both packages using the xlsynth-g8r parser
    let orig_pkg = ir_parser::Parser::new(&pkg.to_string())
        .parse_package()
        .unwrap();
    let opt_pkg = ir_parser::Parser::new(&optimized_pkg.to_string())
        .parse_package()
        .unwrap();

    let orig_fn = orig_pkg.get_top().unwrap();
    let opt_fn = opt_pkg.get_top().unwrap();

    // Check equivalence using the external tool first, specifying the top function
    let orig_ir = pkg.to_string();
    let opt_ir = optimized_pkg.to_string();
    let top_fn_name = "fuzz_test";
    let ext_equiv =
        check_equivalence::check_equivalence_with_top(&orig_ir, &opt_ir, Some(top_fn_name));
    let boolector_result = ir_equiv_boolector::prove_ir_fn_equiv(orig_fn, opt_fn);
    match ext_equiv {
        Ok(()) => {
            // External tool says equivalent, Boolector should agree
            match boolector_result {
                ir_equiv_boolector::EquivResult::Proved => (),
                ir_equiv_boolector::EquivResult::Disproved(cex) => {
                    log::info!("==== IR disagreement detected ====");
                    log::info!("Original IR:\n{}", orig_ir);
                    log::info!("Optimized IR:\n{}", opt_ir);
                    panic!(
                        "Disagreement: external tool says equivalent, Boolector disproves: {:?}",
                        cex
                    );
                }
            }
        }
        Err(ext_err) => {
            // External tool says not equivalent, check Boolector
            match boolector_result {
                ir_equiv_boolector::EquivResult::Proved => {
                    log::info!("==== IR disagreement detected ====");
                    log::info!("Original IR:\n{}", orig_ir);
                    log::info!("Optimized IR:\n{}", opt_ir);
                    panic!(
                        "Disagreement: external tool says NOT equivalent, Boolector proves equivalence. External error: {}",
                        ext_err
                    );
                }
                ir_equiv_boolector::EquivResult::Disproved(_cex) => (), // Both agree not equivalent
            }
        }
    }
});
