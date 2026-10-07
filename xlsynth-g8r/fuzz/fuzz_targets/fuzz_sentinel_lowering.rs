// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::sync::{Mutex, Once, OnceLock};

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r_fuzz::sentinel_lowering::{Coverage, check_input};

static LOGGER: Once = Once::new();
static REPORT_SAMPLES: OnceLock<bool> = OnceLock::new();
static TOTALS: OnceLock<Mutex<Coverage>> = OnceLock::new();

fuzz_target!(|data: &[u8]| {
    LOGGER.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });
    let report = check_input(data).unwrap_or_else(|error| {
        panic!("sentinel-lowering fuzz failure: {error}\nBytes: {data:02x?}")
    });
    if *REPORT_SAMPLES.get_or_init(|| std::env::var_os("XLSYNTH_FUZZ_REPORT_SAMPLES").is_some()) {
        // Optional bounded-campaign evidence includes the bytes for exact
        // replay and distinguishes mutation hits from the starting
        // corpus.
        eprintln!(
            "sentinel_sample bytes={data:02x?} cases={:?} features={:?} counts={:?}",
            report.families.keys().collect::<Vec<_>>(),
            report.features.keys().collect::<Vec<_>>(),
            report.total
        );
    }
    let mut totals = TOTALS
        .get_or_init(|| Mutex::new(Coverage::default()))
        .lock()
        .unwrap();
    totals.accumulate(report);
    if totals.total.samples == 1 || totals.total.samples % 256 == 0 {
        eprintln!("fuzz_sentinel_lowering {totals:?}");
    }
});
