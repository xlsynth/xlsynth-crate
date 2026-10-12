// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r_fuzz::sequential_cleanup::check_cleanup;

fuzz_target!(|data: &[u8]| {
    check_cleanup(data);
});
