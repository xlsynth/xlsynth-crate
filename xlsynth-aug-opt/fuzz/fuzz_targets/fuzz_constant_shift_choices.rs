// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::sync::{Mutex, Once, OnceLock};

use libfuzzer_sys::fuzz_target;
use xlsynth_aug_opt_fuzz::constant_shift_choices::{FuzzStats, check_input};

static LOGGER: Once = Once::new();
static TOTALS: OnceLock<Mutex<FuzzStats>> = OnceLock::new();

fuzz_target!(|data: &[u8]| {
    LOGGER.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });
    let stats = check_input(data);
    let mut totals = TOTALS
        .get_or_init(|| Mutex::new(FuzzStats::default()))
        .lock()
        .unwrap();
    totals.accumulate(stats);
    if totals.samples == 1 || totals.samples % 256 == 0 {
        // Campaign evidence remains visible without depending on RUST_LOG.
        eprintln!("fuzz_constant_shift_choices {totals:?}");
    }
});
