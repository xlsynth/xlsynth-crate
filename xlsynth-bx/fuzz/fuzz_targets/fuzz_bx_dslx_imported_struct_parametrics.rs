// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Err(message) = xlsynth_bx_fuzz::check_imported_struct_parametrics(data) {
        panic!("{message}");
    }
});
