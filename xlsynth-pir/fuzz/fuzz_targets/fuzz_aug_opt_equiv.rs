// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::atomic::{AtomicU64, Ordering};
use xlsynth_pir_fuzz::{fuzz_solver_limits, generate_upstream_formal_random_pir_package};
use xlsynth_pir::aug_opt::{run_aug_opt_over_ir_text_with_stats, AugOptOptions};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::SolverChoice;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::types::EquivResult;

static RUN_COUNT: AtomicU64 = AtomicU64::new(0);
static TOTAL_REWRITES: AtomicU64 = AtomicU64::new(0);
static EQ_SHLL_SLICE_LITERAL_REWRITES: AtomicU64 = AtomicU64::new(0);
static POW2_MSB_TIEBREAK_REWRITES: AtomicU64 = AtomicU64::new(0);
static SELECTED_OPPOSITE_SUBTRACT_REWRITES: AtomicU64 = AtomicU64::new(0);

fuzz_target!(|data: &[u8]| {
    let run_idx = RUN_COUNT.fetch_add(1, Ordering::Relaxed).saturating_add(1);

    let _ = env_logger::builder().is_test(true).try_init();

    let pkg = generate_upstream_formal_random_pir_package(data, "fuzz_pkg");
    let orig_ir = pkg.to_string();
    let top_fn_name = "random_fn";

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
    let eq_shll_slice_literal_rewrites =
        u64::try_from(aug_result.rewrite_stats.eq_shll_slice_literal).unwrap_or(u64::MAX);
    let pow2_rewrites = u64::try_from(aug_result.rewrite_stats.pow2_msb_compare_with_eq_tiebreak)
        .unwrap_or(u64::MAX);
    let selected_opposite_subtract_rewrites =
        u64::try_from(aug_result.rewrite_stats.selected_opposite_subtracts).unwrap_or(u64::MAX);
    TOTAL_REWRITES.fetch_add(total_rewrites, Ordering::Relaxed);
    EQ_SHLL_SLICE_LITERAL_REWRITES.fetch_add(eq_shll_slice_literal_rewrites, Ordering::Relaxed);
    POW2_MSB_TIEBREAK_REWRITES.fetch_add(pow2_rewrites, Ordering::Relaxed);
    SELECTED_OPPOSITE_SUBTRACT_REWRITES
        .fetch_add(selected_opposite_subtract_rewrites, Ordering::Relaxed);
    if run_idx % 1000 == 0 {
        log::info!(
            "fuzz_aug_opt_equiv: runs={} total_rewrites={} eq_shll_slice_literal_rewrites={} pow2_msb_tiebreak_rewrites={} selected_opposite_subtract_rewrites={}",
            run_idx,
            TOTAL_REWRITES.load(Ordering::Relaxed),
            EQ_SHLL_SLICE_LITERAL_REWRITES.load(Ordering::Relaxed),
            POW2_MSB_TIEBREAK_REWRITES.load(Ordering::Relaxed),
            SELECTED_OPPOSITE_SUBTRACT_REWRITES.load(Ordering::Relaxed)
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
        .with_solver(Some(SolverChoice::Bitwuzla))
        .with_solver_limits(fuzz_solver_limits());
        match run_ir_equiv(&request) {
            Ok(report) => {
                if let EquivResult::Inconclusive(msg) = &report.result {
                    log::debug!("aug-opt equivalence inconclusive: {}", msg);
                    // Early-return rationale: a configured solver resource
                    // limit is expected fuzzing noise, not a rewrite failure.
                    return;
                }
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
