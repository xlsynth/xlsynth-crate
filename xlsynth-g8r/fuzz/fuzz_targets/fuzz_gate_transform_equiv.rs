// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use rand::seq::IteratorRandom;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::transforms::{self, transform_trait::TransformDirection};
use xlsynth_g8r_fuzz::{build_graph, FuzzGraph};

fuzz_target!(|graph: FuzzGraph| {
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();
    let Some(g) = build_graph(&graph) else {
        return;
    };
    let mut transforms = transforms::get_equiv_transforms();
    if transforms.is_empty() {
        return;
    }
    let mut rng = rand::thread_rng();
    let t = transforms.iter_mut().choose(&mut rng).unwrap();
    let candidates = t.find_candidates(&g, TransformDirection::Forward);
    if candidates.is_empty() {
        return;
    }
    let cand = candidates.iter().choose(&mut rng).unwrap();
    let mut new_g = g.clone();
    if t.apply(&mut new_g, cand, TransformDirection::Forward)
        .is_err()
    {
        return;
    }
    if let Err(e) = check_equivalence::validate_same_gate_fn(&g, &new_g) {
        panic!("Transform {} broke equivalence: {}", t.display_name(), e);
    }
});
