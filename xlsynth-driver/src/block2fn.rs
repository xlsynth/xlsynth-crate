// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use xlsynth::{IrBits, IrValue};
use xlsynth_pir::block2fn::{block_package_to_fn, Block2FnOptions};
use xlsynth_pir::ir_parser::Parser;

pub fn handle_block2fn(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("block_ir").unwrap();
    let input_path = std::path::Path::new(input_file);
    let _ = config;
    let block_ir_text =
        std::fs::read_to_string(input_path).expect("read block IR file should succeed");
    let mut parser = Parser::new(&block_ir_text);
    let pkg = parser
        .parse_package()
        .expect("parse block IR package should succeed");
    let tie_input_ports = matches
        .get_one::<String>("tie_input_ports")
        .map(|s| parse_csv_kv(s))
        .transpose()
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse --tie-input-ports: {e}");
            std::process::exit(1);
        })
        .unwrap_or_default();
    let drop_output_ports = matches
        .get_one::<String>("drop_output_ports")
        .map(|s| parse_csv_set(s))
        .transpose()
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse --drop-output-ports: {e}");
            std::process::exit(1);
        })
        .unwrap_or_default();
    let clock_port = matches.get_one::<String>("clock_port").cloned();

    let options = Block2FnOptions {
        tie_input_ports,
        drop_output_ports,
        clock_port,
    };
    let result = block_package_to_fn(&pkg, &options).unwrap_or_else(|e| {
        eprintln!("Failed to convert block IR to function: {e}");
        std::process::exit(1);
    });
    println!("{}", result.function);
}

fn parse_csv_kv(input: &str) -> Result<BTreeMap<String, IrBits>, String> {
    let mut out = BTreeMap::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("expected KEY=VALUE, got '{part}'"))?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return Err(format!("invalid KEY=VALUE entry '{part}'"));
        }
        let bits = parse_literal_bits(value)?;
        if out.insert(key.to_string(), bits).is_some() {
            return Err(format!("duplicate entry for '{key}'"));
        }
    }
    Ok(out)
}

fn parse_csv_set(input: &str) -> Result<BTreeSet<String>, String> {
    let mut out = BTreeSet::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if !out.insert(part.to_string()) {
            return Err(format!("duplicate entry '{part}'"));
        }
    }
    Ok(out)
}

fn parse_literal_bits(value: &str) -> Result<IrBits, String> {
    let trimmed = value.trim();
    if !trimmed.starts_with("bits[") {
        return Err(format!(
            "tie-input-ports literal '{trimmed}' must include a bits[N]: prefix"
        ));
    }
    let typed = trimmed.to_string();
    let ir_value =
        IrValue::parse_typed(&typed).map_err(|e| format!("parse literal '{trimmed}': {e}"))?;
    ir_value
        .to_bits()
        .map_err(|e| format!("literal '{trimmed}' is not bits: {e}"))
}
