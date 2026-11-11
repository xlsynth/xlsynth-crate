// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Err(error) = xlsynth_bx_fuzz::check_clocked_test(data) {
        panic!("{error}");
    }
});
