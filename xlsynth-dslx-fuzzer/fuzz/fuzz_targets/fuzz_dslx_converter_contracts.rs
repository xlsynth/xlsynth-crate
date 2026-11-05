// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use std::sync::OnceLock;
use xlsynth_dslx_fuzzer::{audit_contracts, converter_contracts};
use xlsynth_dslx_fuzzer_fuzz::{
    campaign::Campaign,
    cli::{check_audit, check_contract},
};

/// Production converter, import path, and campaign outputs shared by mutations.
struct Config {
    converter: PathBuf,
    stdlib: PathBuf,
    campaign: Campaign,
    audit_v1: bool,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

fuzz_target!(init: {
    let converter = std::env::var_os("DSLX_FUZZ_CONVERTER").expect("set DSLX_FUZZ_CONVERTER");
    let converter = std::fs::canonicalize(converter).expect("invalid DSLX_FUZZ_CONVERTER");
    let stdlib = std::env::var_os("DSLX_STDLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| xlsynth::default_dslx_stdlib_path().to_owned());
    let stdlib = std::path::absolute(stdlib).expect("invalid DSLX_STDLIB_PATH");
    let audit_v1 = match std::env::var_os("DSLX_FUZZ_CONTRACT_GRAMMAR").as_deref() {
        None => false,
        Some(name) if name == "audit-v1" => true,
        Some(name) => panic!("unknown DSLX_FUZZ_CONTRACT_GRAMMAR {name:?}"),
    };
    let config = Config { converter, stdlib, campaign: Campaign::from_env(true), audit_v1 };
    assert!(CONFIG.set(config).is_ok(), "fuzzer initialized twice");
}, |data: &[u8]| {
    let config = CONFIG.get().expect("fuzzer not initialized");
    let (family, result) = if config.audit_v1 {
        let case = audit_contracts::generate(data).expect("generating audit contract");
        (case.family(), check_audit(case, &config.converter, &config.stdlib))
    } else {
        let case = converter_contracts::generate(data).expect("generating converter contract");
        (case.family, check_contract(case, &config.converter, &config.stdlib))
    };
    config.campaign.record(data, Some(family), result);
});
