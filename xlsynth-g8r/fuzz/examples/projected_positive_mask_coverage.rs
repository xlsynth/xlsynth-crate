// SPDX-License-Identifier: Apache-2.0

fn main() {
    let proofs = xlsynth_g8r_fuzz::projected_positive_mask::coverage_audit();
    assert_eq!(proofs, 632);
    println!(
        "Projected positive-mask coverage: {proofs} equivalence proofs with fire/decline assertions"
    );
}
