// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r_fuzz::shift_nonzero::{check_case, coverage_audit};
fuzz_target!(init: {
    let (accepted, rejected) = coverage_audit();
    assert!(accepted > 100 && rejected > 100);
    eprintln!("Shift survival startup: {accepted} accepted, {rejected} rejected; {} proofs", 2 * (accepted + rejected));
}, |data: [u8; 4]| {
    let accepted = check_case(data);
    if std::env::var_os("XLSYNTH_FUZZ_REPORT_SAMPLES").is_some() {
        eprintln!("shift_survival data={data:?} accepted={accepted} proofs=2");
    }
});
