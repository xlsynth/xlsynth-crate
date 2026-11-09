// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use xlsynth_dslx_fuzzer::audit_contracts::generate_v7;
use xlsynth_dslx_fuzzer_fuzz::{bx::check_audit, campaign::Campaign};

static CAMPAIGN: OnceLock<Campaign> = OnceLock::new();

fuzz_target!(|data: &[u8]| {
    let case = generate_v7(data).expect("generating BX converter contract");
    let family = case.family();
    let result = check_audit(case);
    CAMPAIGN
        .get_or_init(|| Campaign::from_env(true))
        .record_bx(data, family, result);
});
