// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::check_equivalence;
use xlsynth_pir::aug_opt::{run_aug_opt_over_ir_text_with_stats, AugOptOptions};
use xlsynth_pir::ir_fuzz::{generate_ir_fn, FuzzSample};

fuzz_target!(|sample: FuzzSample| {
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }

    if sample.ops.is_empty() {
        // Empty op lists cannot form a function body, so they are not
        // informative for rewrite equivalence.
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg").unwrap();
    if let Err(e) = generate_ir_fn(sample.ops.clone(), &mut pkg, None) {
        // The generator can intentionally skip unsupported combos; treat as
        // non-actionable for rewrite equivalence.
        log::info!("IR generation failed: {}", e);
        return;
    }

    let orig_ir = pkg.to_string();
    let top_fn_name = "fuzz_test";

    let aug_result = match run_aug_opt_over_ir_text_with_stats(
        &orig_ir,
        Some(top_fn_name),
        AugOptOptions {
            enable: true,
            rounds: 1,
            run_xlsynth_opt_before: true,
            run_xlsynth_opt_after: true,
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            log::error!("aug_opt failed: {}", err);
            panic!("aug_opt failed unexpectedly for input:\n{}", orig_ir);
        }
    };

    if !aug_result.rewrote() {
        // Only check equivalence when the aug-opt rewrites actually fired.
        return;
    }

    let rewritten_ir = aug_result.output_text;
    let equiv =
        check_equivalence::check_equivalence_with_top(&orig_ir, &rewritten_ir, Some(top_fn_name), false);
    if let Err(err) = equiv {
        log::error!("aug_opt equivalence check failed: {}", err);
        log::info!("Original IR:\n{}", orig_ir);
        log::info!("Rewritten IR:\n{}", rewritten_ir);
        panic!("aug_opt rewrites are not equivalent: {}", err);
    }
});
