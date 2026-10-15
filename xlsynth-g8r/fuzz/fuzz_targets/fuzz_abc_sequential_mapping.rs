// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r_fuzz::abc_mapping::{check, preflight};

fuzz_target!(init: {
    if let Err(error) = preflight() {
        eprintln!("ABC fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    check(data);
});
