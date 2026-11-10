// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use std::sync::OnceLock;
use xlsynth_dslx_fuzzer::audit_contracts::{self, AuditCase};
use xlsynth_dslx_fuzzer::converter_contracts;
use xlsynth_dslx_fuzzer_fuzz::{campaign::Campaign, cli::check_audit_with_optimizer};

/// Production converter, import path, and campaign outputs shared by mutations.
struct Config {
    converter: PathBuf,
    optimizer: PathBuf,
    stdlib: PathBuf,
    campaign: Campaign,
    generate: fn(&[u8]) -> Result<AuditCase, String>,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

fuzz_target!(init: {
    let converter = std::env::var_os("DSLX_FUZZ_CONVERTER").expect("set DSLX_FUZZ_CONVERTER");
    let converter = std::fs::canonicalize(converter).expect("invalid DSLX_FUZZ_CONVERTER");
    let optimizer = std::env::var_os("DSLX_FUZZ_OPTIMIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| converter.with_file_name("opt_main"));
    let stdlib = std::env::var_os("DSLX_STDLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| xlsynth::default_dslx_stdlib_path().to_owned());
    let stdlib = std::path::absolute(stdlib).expect("invalid DSLX_STDLIB_PATH");
    let generate: fn(&[u8]) -> Result<AuditCase, String> = match std::env::var_os("DSLX_FUZZ_CONTRACT_GRAMMAR").as_deref() {
        None => |data| converter_contracts::generate(data).map(AuditCase::Semantic),
        Some(name) if name == "audit-v1" => audit_contracts::generate,
        Some(name) if name == "audit-v2" => audit_contracts::generate_v2,
        Some(name) if name == "audit-v3" => audit_contracts::generate_v3,
        Some(name) if name == "audit-v4" => audit_contracts::generate_v4,
        Some(name) if name == "audit-v5" => audit_contracts::generate_v5,
        Some(name) if name == "audit-v6" => audit_contracts::generate_v6,
        Some(name) if name == "audit-v7" => audit_contracts::generate_v7,
        Some(name) if name == "audit-v8" => audit_contracts::generate_v8,
        Some(name) if name == "audit-v9" => audit_contracts::generate_v9,
        Some(name) if name == "audit-v10" => audit_contracts::generate_v10,
        Some(name) if name == "audit-v11" => audit_contracts::generate_v11,
        Some(name) if name == "audit-v12" => audit_contracts::generate_v12,
        Some(name) if name == "audit-v13" => audit_contracts::generate_v13,
        Some(name) => panic!("unknown DSLX_FUZZ_CONTRACT_GRAMMAR {name:?}"),
    };
    let config = Config { converter, optimizer, stdlib, campaign: Campaign::from_env(true), generate };
    assert!(CONFIG.set(config).is_ok(), "fuzzer initialized twice");
}, |data: &[u8]| {
    let config = CONFIG.get().expect("fuzzer not initialized");
    let case = (config.generate)(data).expect("generating converter contract");
    let family = case.family();
    let result = check_audit_with_optimizer(case, &config.converter, &config.optimizer, &config.stdlib);
    config.campaign.record(data, Some(family), result);
});
