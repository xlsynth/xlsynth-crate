// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::test_utils::{
    interesting_ir_roundtrip_cases, load_bf16_add_sample, load_interesting_ir_roundtrip_case,
    structurally_equivalent,
};

fn assert_gatefn_bincode_roundtrip(case_name: &str, sample: &GateFn) {
    let bin_a = bincode::serialize(sample).expect("serialize GateFn");
    let unbin: GateFn = bincode::deserialize(&bin_a).expect("deserialize GateFn");
    let bin_b = bincode::serialize(&unbin).expect("serialize GateFn again");
    assert_eq!(
        bin_a, bin_b,
        "bincode serdes is not lossless for {case_name}"
    );
    assert!(
        structurally_equivalent(sample, &unbin),
        "GateFn not structurally equivalent after serdes roundtrip for {case_name}"
    );
}

#[test]
fn test_gatefn_bincode_roundtrip_interesting_signatures() {
    for case in interesting_ir_roundtrip_cases() {
        let sample = load_interesting_ir_roundtrip_case(case);
        assert_gatefn_bincode_roundtrip(case.name, &sample.gate_fn);
    }
}

#[test]
fn test_gatefn_bincode_roundtrip_bf16_add() {
    let sample = load_bf16_add_sample(xlsynth_g8r::test_utils::Opt::Yes).gate_fn;
    assert_gatefn_bincode_roundtrip("bf16_add", &sample);
}
