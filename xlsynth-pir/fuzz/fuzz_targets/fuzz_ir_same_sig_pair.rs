// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir_fuzz::generate_full_random_pir_pair;

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let (first, second) = generate_full_random_pir_pair(data);
    assert_eq!(first.get_type(), second.get_type());
});
