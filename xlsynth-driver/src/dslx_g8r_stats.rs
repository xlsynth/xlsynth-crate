// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};
use crate::tools::{run_ir_converter_main, run_opt_main};
use tempfile::NamedTempFile;
use xlsynth::DslxConvertOptions;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::process_ir_path::{process_ir_path, Options as G8rOptions};

pub fn handle_dslx_g8r_stats(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("dslx_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let dslx_top = matches.get_one::<String>("dslx_top").unwrap();

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path = dslx_stdlib_path.as_deref();

    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());
    let enable_warnings = config.as_ref().and_then(|c| c.enable_warnings.as_deref());
    let disable_warnings = config.as_ref().and_then(|c| c.disable_warnings.as_deref());

    let type_inference_v2 = matches
        .get_one::<String>("type_inference_v2")
        .map(|s| s == "true");

    if type_inference_v2 == Some(true) && tool_path.is_none() {
        eprintln!("error: --type_inference_v2 is only supported when using --toolchain (external tool path)");
        std::process::exit(1);
    }

    let ir_text = if let Some(tool_path) = tool_path {
        let mut output = run_ir_converter_main(
            input_path,
            Some(dslx_top),
            dslx_stdlib_path,
            dslx_path,
            tool_path,
            enable_warnings,
            disable_warnings,
            type_inference_v2,
        );
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &output).unwrap();
        output = run_opt_main(temp_file.path(), None, tool_path);
        output
    } else {
        if type_inference_v2 == Some(true) {
            eprintln!("error: --type_inference_v2 is only supported when using --toolchain (external tool path)");
            std::process::exit(1);
        }
        let dslx_contents = std::fs::read_to_string(input_path).expect("failed to read DSLX input");
        let stdlib_path = dslx_stdlib_path.map(|s| std::path::Path::new(s));
        let additional_paths: Vec<&std::path::Path> = dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let convert_result = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            input_path,
            &DslxConvertOptions {
                dslx_stdlib_path: stdlib_path,
                additional_search_paths: additional_paths,
                enable_warnings,
                disable_warnings,
            },
        )
        .expect("successful conversion");
        for warning in convert_result.warnings {
            log::warn!("DSLX warning for {}: {}", input_file, warning);
        }
        let ir_pkg = xlsynth::IrPackage::parse_ir(&convert_result.ir, None).unwrap();
        let ir_top =
            xlsynth::mangle_dslx_name(input_path.file_stem().unwrap().to_str().unwrap(), dslx_top)
                .unwrap();
        let opt_pkg = xlsynth::optimize_ir(&ir_pkg, &ir_top).unwrap();
        opt_pkg.to_string()
    };

    let temp_ir = NamedTempFile::new().unwrap();
    std::fs::write(temp_ir.path(), &ir_text).unwrap();

    let stats = process_ir_path(
        temp_ir.path(),
        &G8rOptions {
            check_equivalence: false,
            fold: true,
            hash: true,
            adder_mapping: AdderMapping::default(),
            fraig: true,
            fraig_max_iterations: None,
            fraig_sim_samples: None,
            quiet: true,
            emit_netlist: false,
            toggle_sample_count: 0,
            toggle_sample_seed: 0,
            compute_graph_logical_effort: true,
            graph_logical_effort_beta1: 1.0,
            graph_logical_effort_beta2: 0.0,
        },
    );

    serde_json::to_writer(std::io::stdout(), &stats).unwrap();
    println!();
}
