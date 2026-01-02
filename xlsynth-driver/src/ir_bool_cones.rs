// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use serde::Serialize;
use std::io::Write;
use xlsynth_pir::ir_bool_cones::{
    enumerate_bool_cone_specs, extract_bool_cone, is_trivial_literal_return_cone,
    is_trivial_param_return_cone, BoolConeConfig,
};
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::{classify_trivial_fn_body, TrivialFnBody};

#[derive(Serialize)]
struct ConeManifestLine {
    sha256: String,
    sink_node_index: usize,
    frontier_leaf_indices: Vec<usize>,
    frontier_non_literal_count: usize,
    included_node_count: usize,
}

pub fn handle_ir_bool_cones(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let k: usize = matches
        .get_one::<String>("k")
        .expect("--k is required")
        .parse()
        .unwrap_or_else(|_| panic!("Invalid --k value"));
    let output_dir = matches.get_one::<String>("output_dir").unwrap();

    let max_cuts_per_node: usize = matches
        .get_one::<String>("max_cuts_per_node")
        .map(|s| s.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    if max_cuts_per_node == 0 {
        panic!("--max_cuts_per_node must be > 0");
    }

    let max_cones: Option<usize> = matches
        .get_one::<String>("max_cones")
        .map(|s| s.parse::<usize>().unwrap_or(0))
        .map(|n| if n == 0 { None } else { Some(n) })
        .unwrap_or(None);

    let emit_pos_data = match matches
        .get_one::<String>("emit_pos_data")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };

    let manifest_jsonl_flag = matches.get_one::<String>("manifest_jsonl");

    let cfg = BoolConeConfig {
        k,
        max_cuts_per_node,
        max_cones,
        emit_pos_data,
    };

    let parse_options = ir_parser::ParseOptions {
        retain_pos_data: emit_pos_data,
    };
    let file_content = std::fs::read_to_string(input_file)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_file, e));
    let mut parser = ir_parser::Parser::new_with_options(&file_content, parse_options);
    let mut pkg = parser
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("Failed to parse/validate IR package: {}", e));

    if let Some(top) = matches.get_one::<String>("ir_top") {
        pkg.set_top_fn(top)
            .unwrap_or_else(|e| panic!("Failed to set --top: {}", e));
    }
    let top_fn = pkg
        .get_top_fn()
        .unwrap_or_else(|| panic!("No top function found in package"));

    let specs = enumerate_bool_cone_specs(top_fn, &cfg);

    let out_dir_path = std::path::Path::new(output_dir);
    std::fs::create_dir_all(out_dir_path)
        .unwrap_or_else(|e| panic!("Failed to create output_dir {}: {}", output_dir, e));

    let manifest_path = match manifest_jsonl_flag {
        Some(p) => std::path::PathBuf::from(p),
        None => out_dir_path.join("manifest.jsonl"),
    };

    let mut manifest_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&manifest_path)
        .unwrap_or_else(|e| panic!("Failed to open manifest {}: {}", manifest_path.display(), e));

    let mut emitted = 0usize;
    for spec in specs.iter() {
        let extracted = extract_bool_cone(top_fn, Some(&pkg.file_table), spec, &cfg);
        if is_trivial_param_return_cone(&extracted) {
            continue;
        }
        if is_trivial_literal_return_cone(&extracted) {
            continue;
        }
        // Also skip “structural-only” functions like bit_slice/tuple_index over a
        // single input param; these are effectively zero-cost in downstream gate
        // metrics and are typically uninteresting cones.
        if extracted.package.members.len() == 1 {
            match &extracted.package.members[0] {
                xlsynth_pir::ir::PackageMember::Function(f) => {
                    if matches!(
                        classify_trivial_fn_body(f),
                        Some(
                            TrivialFnBody::Constant
                                | TrivialFnBody::SingleParamStructural { .. }
                                | TrivialFnBody::SingleBoolGate { .. }
                        )
                    ) {
                        continue;
                    }
                }
                xlsynth_pir::ir::PackageMember::Block { .. } => {}
            }
        }
        let out_path = out_dir_path.join(format!("{}.ir", extracted.sha256_hex));

        if !out_path.exists() {
            let mut f = std::fs::File::create(&out_path)
                .unwrap_or_else(|e| panic!("Failed to create {}: {}", out_path.display(), e));
            f.write_all(extracted.package.to_string().as_bytes())
                .unwrap_or_else(|e| panic!("Failed to write {}: {}", out_path.display(), e));
        }

        let line = ConeManifestLine {
            sha256: extracted.sha256_hex,
            sink_node_index: extracted.sink,
            frontier_leaf_indices: extracted.cut.leaves,
            frontier_non_literal_count: extracted.frontier_non_literal_count,
            included_node_count: extracted.included_node_count,
        };
        let json = serde_json::to_string(&line).expect("manifest JSON serialization must succeed");
        writeln!(manifest_file, "{}", json).expect("manifest write must succeed");

        emitted += 1;
        if let Some(max) = cfg.max_cones {
            if emitted >= max {
                break;
            }
        }
    }

    eprintln!(
        "ir-bool-cones: emitted {} cones into {}; manifest: {}",
        emitted,
        out_dir_path.display(),
        manifest_path.display()
    );
}
