// SPDX-License-Identifier: Apache-2.0

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::{AtomicU64, Ordering};
use xlsynth_pir::aug_opt::{run_aug_opt_over_ir_text_with_stats, AugOptOptions};
use xlsynth_pir::ir_fuzz::{generate_ir_fn, FuzzSample};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::ir_equiv::{run_ir_equiv, IrEquivRequest, IrModule};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::SolverChoice;

static RUN_COUNT: AtomicU64 = AtomicU64::new(0);
static TOTAL_REWRITES: AtomicU64 = AtomicU64::new(0);
static POW2_MSB_TIEBREAK_REWRITES: AtomicU64 = AtomicU64::new(0);

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let sample = match FuzzSample::arbitrary(&mut u) {
        Ok(s) => s,
        Err(_) => return,
    };
    let run_idx = RUN_COUNT.fetch_add(1, Ordering::Relaxed).saturating_add(1);

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
        log::debug!("IR generation failed: {}", e);
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
            mode: xlsynth_pir::aug_opt::AugOptMode::PirOnly,
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            log::error!("aug_opt failed: {}", err);
            panic!("aug_opt failed unexpectedly for input:\n{}", orig_ir);
        }
    };

    let total_rewrites = u64::try_from(aug_result.total_rewrites).unwrap_or(u64::MAX);
    let pow2_rewrites =
        u64::try_from(aug_result.rewrite_stats.pow2_msb_compare_with_eq_tiebreak).unwrap_or(u64::MAX);
    TOTAL_REWRITES.fetch_add(total_rewrites, Ordering::Relaxed);
    POW2_MSB_TIEBREAK_REWRITES.fetch_add(pow2_rewrites, Ordering::Relaxed);
    if run_idx % 1000 == 0 {
        log::info!(
            "fuzz_aug_opt_equiv: runs={} total_rewrites={} pow2_msb_tiebreak_rewrites={}",
            run_idx,
            TOTAL_REWRITES.load(Ordering::Relaxed),
            POW2_MSB_TIEBREAK_REWRITES.load(Ordering::Relaxed)
        );
    }

    if !aug_result.rewrote() {
        // Only check equivalence when the aug-opt rewrites actually fired.
        return;
    }

    let rewritten_ir = aug_result.output_text;

    #[cfg(not(feature = "has-bitwuzla"))]
    {
        panic!(
            "fuzz_aug_opt_equiv requires an in-process solver; \
             build with --features=with-bitwuzla-system (or built)"
        );
    }

    #[cfg(feature = "has-bitwuzla")]
    {
        let request = IrEquivRequest::new(
            IrModule::new(&orig_ir).with_top(Some(top_fn_name)),
            IrModule::new(&rewritten_ir).with_top(Some(top_fn_name)),
        )
        .with_solver(Some(SolverChoice::Bitwuzla));
        match run_ir_equiv(&request) {
            Ok(report) => {
                if !report.is_success() {
                    let err = report
                        .error_str()
                        .unwrap_or_else(|| "unknown equivalence failure".to_string());
                    log::error!("aug_opt equivalence check failed: {}", err);
                    log::info!("Original IR:\n{}", orig_ir);
                    log::info!("Rewritten IR:\n{}", rewritten_ir);
                    panic!("aug_opt rewrites are not equivalent: {}", err);
                }
            }
            Err(err) => {
                log::error!("aug_opt equivalence check failed: {}", err);
                log::info!("Original IR:\n{}", orig_ir);
                log::info!("Rewritten IR:\n{}", rewritten_ir);
                panic!("aug_opt rewrites are not equivalent: {}", err);
            }
        }
    }
});
