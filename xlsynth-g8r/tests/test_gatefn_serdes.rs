// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::test_utils::{load_bf16_add_sample, structurally_equivalent};

#[test]
fn test_gatefn_bincode_roundtrip() {
    let sample = load_bf16_add_sample(xlsynth_g8r::test_utils::Opt::Yes).gate_fn;
    let bin_a = bincode::serialize(&sample).expect("serialize GateFn");
    let unbin: GateFn = bincode::deserialize(&bin_a).expect("deserialize GateFn");
    let bin_b = bincode::serialize(&unbin).expect("serialize GateFn again");
    assert_eq!(bin_a, bin_b, "bincode serdes is not lossless");
    assert!(
        structurally_equivalent(&sample, &unbin),
        "GateFn not structurally equivalent after serdes roundtrip"
    );
}
