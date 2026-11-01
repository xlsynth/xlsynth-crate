// SPDX-License-Identifier: Apache-2.0

#![cfg(all(xlsynth_dslx_native, feature = "has-bitwuzla"))]

use libfuzzer_sys as _;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth_dslx_fuzzer_fuzz::native::probe_feedback;

#[test]
fn native_feedback_and_equivalence_oracle_are_live() {
    let mut rng = StdRng::from_seed([71; 32]);
    let mut inputs = vec![vec![]];
    for _ in 0..16 {
        let mut bytes = vec![0; 1024];
        rng.fill_bytes(&mut bytes);
        inputs.push(bytes);
    }
    let report = probe_feedback(&inputs).unwrap();
    assert!(report.native_counters > 0);
    assert!(report.covered_after > report.covered_before);
    assert!(report.inputs_with_new_edges > 1);
    assert_eq!(report.proved_samples, inputs.len());
    assert!(report.wrong_reference_disproved);
}
