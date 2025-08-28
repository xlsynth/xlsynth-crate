// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use xlsynth_g8r::prove_gate_fn_equiv_common::EquivResult;
use xlsynth_g8r::prove_gate_fn_equiv_varisat::{
    prove_gate_fn_equiv as prove_sat, Ctx as VarisatCtx,
};
use xlsynth_g8r::transforms::{self, transform_trait::TransformDirection};
use xlsynth_g8r_fuzz::{build_graph, FuzzGraph};

#[cfg(feature = "z3")]
use xlsynth_g8r::prove_gate_fn_equiv_z3::{prove_gate_fn_equiv as prove_z3, Ctx as Z3Ctx};

const NUM_STEPS: usize = 32;

fuzz_target!(|graph: FuzzGraph| {
    // Ensure INFO logging is enabled for visibility inside fuzz runs.
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    // Build an initial GateFn from the fuzzed graph description.
    let Some(mut cur_g) = build_graph(&graph) else {
        log::info!("Rejected graph: could not build");
        return;
    };

    let orig_g = cur_g.clone();

    // Prepare the transform set & RNG.
    let mut transforms = transforms::get_all_transforms();
    if transforms.is_empty() {
        log::info!("No transforms available – skipping input");
        return;
    }
    let mut rng = thread_rng();

    let mut attempts = 0usize;
    while attempts < NUM_STEPS {
        // Pick a random transform.
        let t = transforms.iter_mut().choose(&mut rng).unwrap();
        let cand_locs = t.find_candidates(&cur_g, TransformDirection::Forward);
        if cand_locs.is_empty() {
            log::debug!("Transform {}: no applicable candidates", t.display_name());
            continue;
        }
        let cand = cand_locs.iter().choose(&mut rng).unwrap();

        // Apply to produce next_g.
        let mut next_g = cur_g.clone();
        if let Err(e) = t.apply(&mut next_g, cand, TransformDirection::Forward) {
            log::info!(
                "Transform {} apply failed (skipping): {}",
                t.display_name(),
                e
            );
            continue;
        }

        attempts += 1; // Count this attempted (and successful) application

        // Cross-check equivalence solvers.
        #[cfg(feature = "z3")]
        {
            let mut varisat_ctx = VarisatCtx::new();
            let mut z3_ctx = Z3Ctx::new();
            let sat_orig_cur = prove_sat(&orig_g, &cur_g, &mut varisat_ctx);
            let z3_orig_cur = prove_z3(&orig_g, &cur_g, &mut z3_ctx);
            assert_eq!(
                matches!(sat_orig_cur, EquivResult::Proved),
                matches!(z3_orig_cur, EquivResult::Proved),
                "Disagreement between SAT and Z3 on orig vs cur (transform {}): {:?} vs {:?}",
                t.display_name(),
                sat_orig_cur,
                z3_orig_cur
            );

            let mut varisat_ctx2 = VarisatCtx::new();
            let mut z3_ctx2 = Z3Ctx::new();
            let sat_cur_next = prove_sat(&cur_g, &next_g, &mut varisat_ctx2);
            let z3_cur_next = prove_z3(&cur_g, &next_g, &mut z3_ctx2);
            assert_eq!(
                matches!(sat_cur_next, EquivResult::Proved),
                matches!(z3_cur_next, EquivResult::Proved),
                "Disagreement between SAT and Z3 on cur vs next (transform {}): {:?} vs {:?}",
                t.display_name(),
                sat_cur_next,
                z3_cur_next
            );

            // If the transform is claimed to be always-equivalent, equivalence must hold.
            if t.always_equivalent() {
                if !matches!(sat_cur_next, EquivResult::Proved) {
                    log::info!(
                        "ALWAYS-EQUIV transform {} produced inequivalence; panicking",
                        t.display_name()
                    );
                    panic!(
                        "Transform {} is marked always_equivalent but produced inequivalence",
                        t.display_name()
                    );
                }
            }
        }

        // Advance.
        cur_g = next_g;
    }
});
