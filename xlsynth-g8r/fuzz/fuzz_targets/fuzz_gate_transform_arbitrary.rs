// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;

#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use xlsynth_g8r::prove_gate_fn_equiv_common::EquivResult;
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use xlsynth_g8r::prove_gate_fn_equiv_sat::{
    Ctx as VarisatCtx, prove_gate_fn_equiv as prove_sat,
};
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use xlsynth_g8r::prove_gate_fn_equiv_z3::{Ctx as Z3Ctx, prove_gate_fn_equiv as prove_z3};
use xlsynth_g8r::transforms::{self, transform_trait::TransformDirection};
use xlsynth_g8r_fuzz::{FuzzGraph, build_graph};

const NUM_STEPS: usize = 32;
const MAX_TRANSFORM_DRAWS: usize = NUM_STEPS * 32;

fn make_rng(graph: &FuzzGraph) -> StdRng {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[graph.num_inputs]);
    hasher.update(&[graph.input_width]);
    hasher.update(&[graph.num_ops]);
    hasher.update(&[graph.num_outputs]);
    hasher.update(&[u8::from(graph.use_opt)]);
    hasher.update(&(graph.ops.len() as u64).to_le_bytes());
    for op in &graph.ops {
        hasher.update(&op.lhs.to_le_bytes());
        hasher.update(&op.rhs.to_le_bytes());
        hasher.update(&[u8::from(op.lhs_neg), u8::from(op.rhs_neg)]);
    }
    StdRng::from_seed(*hasher.finalize().as_bytes())
}

fuzz_target!(|graph: FuzzGraph| {
    // Respect `RUST_LOG` so callers can choose quiet or verbose fuzz runs.
    let _ = env_logger::Builder::from_env(env_logger::Env::default())
        .is_test(true)
        .try_init();

    // Build an initial GateFn from the fuzzed graph description.
    let Some(mut cur_g) = build_graph(&graph) else {
        log::info!("Rejected graph: could not build");
        return;
    };

    #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
    let orig_g = cur_g.clone();

    // Prepare the transform set & RNG.
    let mut transforms = transforms::get_all_transforms();
    if transforms.is_empty() {
        log::info!("No transforms available – skipping input");
        return;
    }
    let mut rng = make_rng(&graph);

    let mut attempts = 0usize;
    for _draw in 0..MAX_TRANSFORM_DRAWS {
        if attempts >= NUM_STEPS {
            break;
        }

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
        #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
        {
            let mut varisat_ctx = VarisatCtx::new();
            let sat_orig_cur = prove_sat(&orig_g, &cur_g, &mut varisat_ctx);
            let mut z3_ctx = Z3Ctx::new();
            let z3_orig_cur = prove_z3(&orig_g, &cur_g, &mut z3_ctx);
            assert_eq!(
                matches!(sat_orig_cur, EquivResult::Proved),
                matches!(z3_orig_cur, EquivResult::Proved),
                "Disagreement between SAT and Z3 on orig vs cur (transform {}): {:?} vs {:?}",
                t.display_name(),
                sat_orig_cur,
                z3_orig_cur
            );
        }

        #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
        {
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
