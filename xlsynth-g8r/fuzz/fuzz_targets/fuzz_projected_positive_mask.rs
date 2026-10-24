// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r_fuzz::projected_positive_mask::{KINDS, check_case, coverage_audit};

fuzz_target!(init: {
    let proofs = coverage_audit();
    assert_eq!(proofs, 632);
    eprintln!("Projected positive-mask coverage: {proofs} equivalence proofs with fire/decline assertions");
}, |data: [u8; 4]| {
    let width = 3 + usize::from(data[0]);
    check_case(
        width,
        usize::from(data[1]) % KINDS.len(),
        data[2] & 1 != 0,
        data[3] & 1 != 0,
    );
});
