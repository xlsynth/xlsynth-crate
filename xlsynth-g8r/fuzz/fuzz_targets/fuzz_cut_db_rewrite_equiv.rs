// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

use std::sync::OnceLock;

use xlsynth_g8r::aig::cut_db_rewrite::{rewrite_gatefn_with_cut_db, RewriteOptions};
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::prove_gate_fn_equiv_common::{EquivResult, GateFormalBackend};
use xlsynth_g8r::prove_gate_fn_equiv_sat::{
    ValidationError, prove_gate_fn_equiv_with_backend_and_options,
};
use xlsynth_g8r_fuzz::{FuzzGraph, build_graph, fuzz_gate_formal_options};

static CUT_DB_BYTES: &[u8] = include_bytes!("../../data/cut_db_v1.bin");
static CUT_DB: OnceLock<CutDb> = OnceLock::new();

fuzz_target!(|graph: FuzzGraph| {
    // Enable logs inside fuzz runs for diagnostics when something fails.
    let _ = env_logger::builder()
        .is_test(true)
        // Default to WARN so we don't drown libFuzzer in per-iteration INFO logs.
        // If detailed logging is needed, set `RUST_LOG` explicitly.
        .filter_level(log::LevelFilter::Warn)
        .try_init();

    let Some(orig_g) = build_graph(&graph) else {
        // Degenerate/unbuildable graph samples are not informative for rewrite
        // soundness.
        return;
    };

    let db = CUT_DB.get_or_init(|| {
        CutDb::load_from_reader(CUT_DB_BYTES).unwrap_or_else(|e| {
            panic!(
                "failed to load vendored cut DB xlsynth-g8r/data/cut_db_v1.bin: {:?}",
                e
            )
        })
    });

    // Conservative settings to keep fuzzing fast and avoid timeouts.
    let rewritten = rewrite_gatefn_with_cut_db(
        &orig_g,
        &db,
        RewriteOptions {
            max_cuts_per_node: 32,
            max_iterations: 2,
            verify_area_costing: true,
            verify_delay_costing: true,
            ..RewriteOptions::default()
        },
    );

    let equiv_result = match prove_gate_fn_equiv_with_backend_and_options(
        &orig_g,
        &rewritten,
        GateFormalBackend::Cadical,
        fuzz_gate_formal_options(),
    ) {
        Ok(result) => result,
        Err(ValidationError::CadicalSolveInterrupted) => {
            // A configured solver timeout is expected to make some fuzz
            // samples inconclusive; those samples are not rewrite failures.
            return;
        }
        Err(err) => panic!("Cadical gate equivalence failed: {err}"),
    };
    match equiv_result {
        EquivResult::Proved => {}
        EquivResult::Disproved(cex) => {
            panic!(
                "cut-db rewrite broke equivalence; counterexample inputs: {:?}",
                cex
            );
        }
    }
});
