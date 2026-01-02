// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_opt_main;
use sha2::Digest;
use std::path::Path;
use xlsynth::IrPackage;
use xlsynth_pir::bool_cones::BoolConeExtractOptions;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_validate;

fn count_nontrivial_ops_in_cone_fn(f: &xlsynth_pir::ir::Fn) -> usize {
    f.nodes
        .iter()
        .filter(|n| match &n.payload {
            NodePayload::Nil | NodePayload::GetParam(_) | NodePayload::Literal(_) => false,
            _ => true,
        })
        .count()
}

pub fn handle_ir_fn_extract_bool_cones(
    matches: &clap::ArgMatches,
    config: &Option<ToolchainConfig>,
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

    let mut emitted_unique: usize = 0;
    let mut skipped_trivial: usize = 0;

    for cone in cones.iter() {
        // Optimize each extracted cone before content-addressing it.
        //
        // This ensures that we dedupe on optimized form, and that we can drop cones
        // that optimize to a trivial `ret param` or `ret literal`.
        let optimized_text: String =
            if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
                // External tool expects a file path; write the cone to a temp file.
                let temp_dir = tempfile::TempDir::new().unwrap();
                let cone_path = temp_dir.path().join("cone.ir");
                std::fs::write(&cone_path, &cone.fn_text).unwrap();
                run_opt_main(&cone_path, Some("cone"), tool_path)
            } else {
                let ir_pkg = IrPackage::parse_ir(&cone.fn_text, Some("bool_cone.ir"))
                    .unwrap_or_else(|e| panic!("Failed to parse extracted cone as IrPackage: {e}"));
                let optimized = xlsynth::optimize_ir(&ir_pkg, "cone")
                    .unwrap_or_else(|e| panic!("Failed to optimize extracted cone: {e}"));
                optimized.to_string()
            };

        // Parse+validate optimized text using xlsynth-pir so we can inspect the return
        // node.
        let mut parser = Parser::new(&optimized_text);
        let opt_pkg = match parser.parse_package() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Optimized cone failed to parse as XLS IR package: {}", e);
                std::process::exit(1);
            }
        };
        if let Err(e) = ir_validate::validate_package(&opt_pkg) {
            eprintln!("Optimized cone failed validation: {}", e);
            std::process::exit(1);
        }

        let opt_f = match opt_pkg.get_top_fn() {
            Some(f) => f,
            None => {
                eprintln!("Optimized cone did not have a top function");
                std::process::exit(1);
            }
        };
        let ret_ref = opt_f
            .ret_node_ref
            .expect("optimized cone fn must have a return node");
        let ret_node = opt_f.get_node(ret_ref);
        match &ret_node.payload {
            NodePayload::GetParam(_) | NodePayload::Literal(_) => {
                skipped_trivial += 1;
                continue;
            }
            _ => {}
        }

        // Also skip cones that are a single nontrivial operation after optimization
        // (e.g. `and(a,b)` or `not(a)`). We only want cones that have at least
        // two nontrivial ops in them.
        let nontrivial_op_count = count_nontrivial_ops_in_cone_fn(opt_f);
        if nontrivial_op_count <= 1 {
            skipped_trivial += 1;
            continue;
        }

        // Content-address by optimized text.
        let digest = sha2::Sha256::digest(optimized_text.as_bytes());
        let sha256_hex = format!("{digest:x}");
        let out_path = Path::new(out_dir).join(format!("{}.ir", sha256_hex));

        // Best-effort skip if already present (dedupe in output directory).
        if out_path.exists() {
            continue;
        }
        {
            if let Err(e) = std::fs::write(&out_path, &optimized_text) {
                eprintln!("Failed to write '{}': {}", out_path.display(), e);
                std::process::exit(1);
            }
        }
        emitted_unique += 1;
    }

    println!(
        "roots={} extracted_unique_preopt={} emitted_unique={} pruned_by_depth={} pruned_by_params={} skipped_unsupported={} skipped_trivial={}",
        stats.roots,
        stats.extracted_unique,
        emitted_unique,
        stats.pruned_by_depth,
        stats.pruned_by_params,
        stats.skipped_unsupported,
        skipped_trivial
    );
}
