// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use rand::seq::IteratorRandom;
use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gate_random::RandomGateOptions;
use xlsynth_g8r::transforms::{self, transform_trait::TransformDirection};
use xlsynth_g8r_fuzz::direct_g8r::generate_combinational;

fuzz_target!(|graph: &[u8]| {
    // External equivalence tools are infrastructure, not a property of the
    // sample.
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();
    let g = generate_combinational(graph, &RandomGateOptions::default())
        .expect("valid bounded combinational generator options")
        .graph;
    let mut transforms = transforms::get_equiv_transforms();
    // No registered transform means there is no rewrite property to exercise.
    if transforms.is_empty() {
        return;
    }
    let mut rng = StdRng::from_seed(*blake3::hash(graph).as_bytes());
    let t = transforms.iter_mut().choose(&mut rng).unwrap();
    let candidates = t.find_candidates(&g, TransformDirection::Forward);
    // This transform need not be applicable to every valid random graph.
    if candidates.is_empty() {
        return;
    }
    let cand = candidates.iter().choose(&mut rng).unwrap();
    let mut new_g = g.clone();
    // A transform may reject an inapplicable candidate without changing
    // semantics.
    if t.apply(&mut new_g, cand, TransformDirection::Forward)
        .is_err()
    {
        return;
    }
    if let Err(e) = check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&g, &new_g) {
        panic!("Transform {} broke equivalence: {}", t.display_name(), e);
    }
});
