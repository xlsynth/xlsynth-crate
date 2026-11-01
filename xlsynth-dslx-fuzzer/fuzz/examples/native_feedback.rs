// SPDX-License-Identifier: Apache-2.0

use libfuzzer_sys as _;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth_dslx_fuzzer_fuzz::native::probe_feedback;

/// Verifies native edge discovery and the independent equivalence oracle.
fn main() -> Result<(), String> {
    let mut rng = StdRng::from_seed([71; 32]);
    let mut inputs = vec![vec![]];
    for _ in 0..16 {
        let mut bytes = vec![0; 1024];
        rng.fill_bytes(&mut bytes);
        inputs.push(bytes);
    }
    let report = probe_feedback(&inputs)?;
    println!(
        r#"{{"native_counters":{},"covered_before":{},"covered_after":{},"inputs_with_new_edges":{},"proved_samples":{},"wrong_reference_disproved":{}}}"#,
        report.native_counters,
        report.covered_before,
        report.covered_after,
        report.inputs_with_new_edges,
        report.proved_samples,
        report.wrong_reference_disproved
    );
    Ok(())
}
