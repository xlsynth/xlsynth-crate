// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;

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

pub fn handle_g8r2ir(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let g8r_input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let g8r_path = Path::new(g8r_input_file);

    let gate_fn = match load_gate_fn(g8r_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r2ir error: {}", e);
            std::process::exit(2);
        }
    };

    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = match gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type) {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!(
                "g8r2ir error: failed to convert GateFn to XLS IR for {}: {}",
                g8r_input_file, e
            );
            std::process::exit(2);
        }
    };

    println!("{}", ir_pkg.to_string());
}
