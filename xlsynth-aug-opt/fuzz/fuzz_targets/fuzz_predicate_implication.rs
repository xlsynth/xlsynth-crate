// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_aug_opt_fuzz::predicate_implication::{FAMILIES, check_input};
fuzz_target!(|data: &[u8]| {
    let result = check_input(data).unwrap_or_else(|error| panic!("{error}\nbytes={data:02x?}"));
    if std::env::var_os("XLSYNTH_FUZZ_REPORT_SAMPLES").is_some() {
        eprintln!(
            "implication_sample bytes={data:02x?} family={} pipeline={} width={} prefix={} forced={} accepted={} proofs={} features={:?}",
            FAMILIES[result.family],
            result.pipeline,
            result.width,
            result.prefix,
            result.forced,
            result.accepted,
            result.proofs,
            result.features
        );
    }
});
