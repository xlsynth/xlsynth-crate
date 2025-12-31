// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;
use std::path::Path;
use xlsynth_pir::bool_cones::BoolConeExtractOptions;

pub fn handle_ir_fn_extract_bool_cones(
    matches: &clap::ArgMatches,
    _config: &Option<ToolchainConfig>,
) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let top = matches
        .get_one::<String>("ir_top")
        .expect("--top must be specified");
    let out_dir = matches
        .get_one::<String>("out_dir")
        .expect("--out-dir must be specified");
    let max_depth = *matches
        .get_one::<usize>("max_depth")
        .expect("--max-depth must be specified");
    let max_params = *matches
        .get_one::<usize>("max_params")
        .expect("--max-params must be specified");

    let input_path = Path::new(input_file);
    let mut pkg = match xlsynth_pir::ir_parser::parse_and_validate_path_to_package(input_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "Failed to parse/validate IR package '{}': {}",
                input_file, e
            );
            std::process::exit(1);
        }
    };

    if let Err(e) = pkg.set_top_fn(top) {
        eprintln!("Failed to select --top='{}': {}", top, e);
        std::process::exit(1);
    }
    let f = match pkg.get_top_fn() {
        Some(f) => f,
        None => {
            eprintln!("Top member '{}' is not a function", top);
            std::process::exit(1);
        }
    };

    let opts = BoolConeExtractOptions {
        max_depth_exclusive: max_depth,
        max_params_exclusive: max_params,
    };
    let (cones, stats) = match xlsynth_pir::bool_cones::extract_bool_cones_from_fn(f, &opts) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Cone extraction failed: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("Failed to create out dir '{}': {}", out_dir, e);
        std::process::exit(1);
    }

    for cone in cones.iter() {
        let out_path = Path::new(out_dir).join(format!("{}.ir", cone.sha256_hex));
        if let Err(e) = std::fs::write(&out_path, &cone.fn_text) {
            eprintln!("Failed to write '{}': {}", out_path.display(), e);
            std::process::exit(1);
        }
    }

    println!(
        "roots={} extracted_unique={} pruned_by_depth={} pruned_by_params={} skipped_unsupported={}",
        stats.roots,
        stats.extracted_unique,
        stats.pruned_by_depth,
        stats.pruned_by_params,
        stats.skipped_unsupported
    );
}
