// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::gate_fn_equiv_report;

use std::fs;
use std::path::Path;

use crate::toolchain_config::ToolchainConfig;

fn load_gate_fn(path: &Path) -> Result<GateFn, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    if path.extension().map(|e| e == "g8rbin").unwrap_or(false) {
        bincode::deserialize(&bytes).map_err(|e| {
            format!(
                "failed to deserialize GateFn from {}: {}",
                path.display(),
                e
            )
        })
    } else {
        let txt = String::from_utf8(bytes)
            .map_err(|e| format!("failed to decode utf8 from {}: {}", path.display(), e))?;
        GateFn::try_from(txt.as_str())
            .map_err(|e| format!("failed to parse GateFn from {}: {}", path.display(), e))
    }
}

pub fn handle_g8r_equiv(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let lhs_path = Path::new(matches.get_one::<String>("lhs_g8r_file").unwrap());
    let rhs_path = Path::new(matches.get_one::<String>("rhs_g8r_file").unwrap());

    let lhs = match load_gate_fn(lhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };
    let rhs = match load_gate_fn(rhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };

    let report = gate_fn_equiv_report::prove_gate_fn_equiv_report(&lhs, &rhs);
    serde_json::to_writer(std::io::stdout(), &report).unwrap();
    println!();

    if report.results.values().any(|r| !r.is_equiv()) {
        std::process::exit(1);
    }
}
