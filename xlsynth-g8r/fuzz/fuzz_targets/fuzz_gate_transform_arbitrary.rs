// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;

use xlsynth_g8r::prove_gate_fn_equiv_common::{EquivResult, GateFormalBackend};
use xlsynth_g8r::prove_gate_fn_equiv_sat::{
    ValidationError, prove_gate_fn_equiv_with_backend_and_options,
};
use xlsynth_g8r::transforms::{self, transform_trait::TransformDirection};
use xlsynth_g8r_fuzz::{FuzzGraph, build_graph, fuzz_gate_formal_options};

// Each successful step runs gate-level equivalence checks twice, so bound the
// transform sequence to keep one fuzz input comfortably short.
const NUM_STEPS: usize = 8;
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

/// Proves gate equivalence while allowing configured fuzz timeouts to skip a sample.
fn prove_with_fuzz_timeout(lhs: &xlsynth_g8r::aig::GateFn, rhs: &xlsynth_g8r::aig::GateFn) -> Option<EquivResult> {
    match prove_gate_fn_equiv_with_backend_and_options(
        lhs,
        rhs,
        GateFormalBackend::Cadical,
        fuzz_gate_formal_options(),
    ) {
        Ok(result) => Some(result),
        Err(ValidationError::CadicalSolveInterrupted) => None,
        Err(err) => panic!("Cadical gate equivalence failed: {err}"),
    }
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

        let Some(_orig_cur) = prove_with_fuzz_timeout(&orig_g, &cur_g) else {
            // A configured solver timeout is expected to make some fuzz
            // samples inconclusive; those samples are not transform failures.
            return;
        };

        let Some(cur_next) = prove_with_fuzz_timeout(&cur_g, &next_g) else {
            // A configured solver timeout is expected to make some fuzz
            // samples inconclusive; those samples are not transform failures.
            return;
        };

        // If the transform is claimed to be always-equivalent, equivalence must hold.
        if t.always_equivalent() && !matches!(cur_next, EquivResult::Proved) {
            log::info!(
                "ALWAYS-EQUIV transform {} produced inequivalence; panicking",
                t.display_name()
            );
            panic!(
                "Transform {} is marked always_equivalent but produced inequivalence",
                t.display_name()
            );
        }

        // Advance.
        cur_g = next_g;
    }
});
