// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r_fuzz::serdes_roundtrip::check_binary_aiger_roundtrip;

fuzz_target!(|data: &[u8]| {
    check_binary_aiger_roundtrip(data);
});
