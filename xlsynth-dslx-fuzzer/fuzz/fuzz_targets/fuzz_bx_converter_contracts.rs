// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use std::sync::OnceLock;
use xlsynth_dslx_fuzzer::audit_contracts::{self, AuditCase};
use xlsynth_dslx_fuzzer_fuzz::{bx::check_audit, campaign::Campaign};

/// DSLX compiler used to triage BX discrepancies, and the campaign ledger.
struct Config {
    converter: PathBuf,
    stdlib: PathBuf,
    campaign: Campaign,
    generate: fn(&[u8]) -> Result<AuditCase, String>,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

fuzz_target!(init: {
    let converter = std::env::var_os("DSLX_FUZZ_CONVERTER").expect("set DSLX_FUZZ_CONVERTER");
    let converter = std::fs::canonicalize(converter).expect("invalid DSLX_FUZZ_CONVERTER");
    let stdlib = std::env::var_os("DSLX_STDLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| xlsynth::default_dslx_stdlib_path().to_owned());
    let stdlib = std::fs::canonicalize(stdlib).expect("invalid DSLX_STDLIB_PATH");
    let generate: fn(&[u8]) -> Result<AuditCase, String> = match std::env::var_os("DSLX_FUZZ_CONTRACT_GRAMMAR").as_deref() {
        None => audit_contracts::generate_v7,
        Some(name) if name == "audit-v7" => audit_contracts::generate_v7,
        Some(name) if name == "audit-v8" => audit_contracts::generate_v8,
        Some(name) if name == "audit-v9" => audit_contracts::generate_v9,
        Some(name) => panic!("unknown BX contract grammar {name:?}"),
    };
    let config = Config { converter, stdlib, campaign: Campaign::from_env(true), generate };
    assert!(CONFIG.set(config).is_ok(), "fuzzer initialized twice");
}, |data: &[u8]| {
    let config = CONFIG.get().expect("fuzzer not initialized");
    let case = (config.generate)(data).expect("generating BX converter contract");
    let family = case.family();
    let result = check_audit(case, &config.converter, &config.stdlib);
    config.campaign.record_bx(data, family, result);
});
