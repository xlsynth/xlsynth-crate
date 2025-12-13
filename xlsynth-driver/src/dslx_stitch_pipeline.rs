// SPDX-License-Identifier: Apache-2.0

use crate::common::{ensure_trailing_newline, get_dslx_paths};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_g8r::verilog_version::VerilogVersion;

/// Handles the `dslx-stitch-pipeline` subcommand (stitch pipeline stages).
pub fn handle_dslx_stitch_pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let paths = get_dslx_paths(matches, config);
    let path_refs = paths.search_path_views();

    let input = matches.get_one::<String>("dslx_input_file").unwrap();
    let dslx_top = matches.get_one::<String>("dslx_top");
    let dslx = std::fs::read_to_string(input).unwrap_or_else(|e| {
        report_cli_error_and_exit("could not read DSLX input", Some(&e.to_string()), vec![]);
    });
    let use_system_verilog = matches
        .get_one::<String>("use_system_verilog")
        .map(|s| s == "true")
        .unwrap_or(crate::flag_defaults::CODEGEN_USE_SYSTEM_VERILOG);
    let verilog_version = if use_system_verilog {
        VerilogVersion::SystemVerilog
    } else {
        VerilogVersion::Verilog
    };

    let stage_list: Option<Vec<String>> = matches
        .get_one::<String>("stages")
        .map(|csv| csv.split(',').map(|s| s.trim().to_string()).collect());
    let output_module_name_opt = matches.get_one::<String>("output_module_name");

    // Enforce mutual exclusion and required combinations.
    if stage_list.is_some() && dslx_top.is_some() {
        report_cli_error_and_exit(
            "--dslx_top is mutually exclusive with --stages",
            None,
            vec![],
        );
    }
    if stage_list.is_some() && output_module_name_opt.is_none() {
        report_cli_error_and_exit(
            "--output_module_name is required when --stages is provided",
            None,
            vec![],
        );
    }
    if stage_list.is_none() && dslx_top.is_none() {
        report_cli_error_and_exit(
            "one of --dslx_top or --stages must be provided",
            None,
            vec![],
        );
    }
    let input_valid_signal_opt = matches
        .get_one::<String>("input_valid_signal")
        .map(|s| s.as_str());
    let output_valid_signal_opt = matches
        .get_one::<String>("output_valid_signal")
        .map(|s| s.as_str());
    let reset_opt = matches.get_one::<String>("reset").map(|s| s.as_str());
    let reset_active_low = matches
        .get_one::<String>("reset_active_low")
        .map(|s| s == "true")
        .unwrap_or(false);

    // Determine whether to add invariant assertions based on toolchain config
    // (default false).
    let add_invariant_assertions = config
        .as_ref()
        .and_then(|c| c.codegen.as_ref())
        .and_then(|cg| cg.add_invariant_assertions)
        .unwrap_or(crate::flag_defaults::CODEGEN_ADD_INVARIANT_ASSERTIONS);

    // Determine whether to emit array-index bounds checking (default true).
    let array_index_bounds_checking = matches
        .get_one::<String>("array_index_bounds_checking")
        .map(|s| s == "true")
        .or_else(|| {
            config
                .as_ref()
                .and_then(|c| c.codegen.as_ref())
                .and_then(|cg| cg.array_index_bounds_checking)
        })
        .unwrap_or(crate::flag_defaults::CODEGEN_ARRAY_INDEX_BOUNDS_CHECKING);

    let flop_inputs = matches
        .get_one::<String>("flop_inputs")
        .map(|s| s == "true")
        .unwrap_or(crate::flag_defaults::CODEGEN_FLOP_INPUTS);
    let flop_outputs = matches
        .get_one::<String>("flop_outputs")
        .map(|s| s == "true")
        .unwrap_or(crate::flag_defaults::CODEGEN_FLOP_OUTPUTS);

    // Determine:
    //  * stage-discovery prefix: only used for implicit discovery of
    //    `<prefix>_cycleN` when `--stages` is not provided (from `--dslx_top`).
    //  * wrapper module name: the emitted outer module name (from
    //    `--output_module_name`, or defaults to `--dslx_top` when using implicit
    //    discovery).
    let stage_discovery_prefix_opt = dslx_top.as_deref();
    let wrapper_name = output_module_name_opt
        .as_deref()
        .or(stage_discovery_prefix_opt)
        .expect("validated combinations above");

    let output_unopt_ir = matches.get_one::<String>("output_unopt_ir");
    let output_opt_ir = matches.get_one::<String>("output_opt_ir");

    let options = xlsynth_g8r::dslx_stitch_pipeline::StitchPipelineOptions {
        verilog_version,
        explicit_stages: stage_list,
        stdlib_path: paths.stdlib_path.as_deref(),
        search_paths: path_refs.clone(),
        flop_inputs,
        flop_outputs,
        input_valid_signal: input_valid_signal_opt,
        output_valid_signal: output_valid_signal_opt,
        reset_signal: reset_opt,
        reset_active_low,
        add_invariant_assertions,
        array_index_bounds_checking,
        output_module_name: wrapper_name,
    };

    // Library `top` parameter is the implicit stage-discovery prefix; when stages
    // are provided explicitly, this value is unused by discovery but we pass
    // the wrapper name.
    let top_for_library = stage_discovery_prefix_opt.unwrap_or(wrapper_name);

    if output_unopt_ir.is_some() || output_opt_ir.is_some() {
        let mut pkgs = xlsynth_g8r::dslx_stitch_pipeline::stitch_pipeline_ir_packages(
            &dslx,
            std::path::Path::new(input),
            top_for_library,
            &options,
        )
        .unwrap_or_else(|e| {
            report_cli_error_and_exit("stitch error (ir output)", Some(&e.0), vec![])
        });

        if let Some(path) = output_unopt_ir {
            let unopt_text = ensure_trailing_newline(std::mem::take(&mut pkgs.unopt_ir));
            std::fs::write(path, unopt_text).unwrap_or_else(|e| {
                report_cli_error_and_exit(
                    "could not write unoptimized IR output",
                    Some(&e.to_string()),
                    vec![("path", path)],
                );
            });
        }
        if let Some(path) = output_opt_ir {
            let opt_text = ensure_trailing_newline(std::mem::take(&mut pkgs.opt_ir));
            std::fs::write(path, opt_text).unwrap_or_else(|e| {
                report_cli_error_and_exit(
                    "could not write optimized IR output",
                    Some(&e.to_string()),
                    vec![("path", path)],
                );
            });
        }
    }

    let result = xlsynth_g8r::dslx_stitch_pipeline::stitch_pipeline(
        &dslx,
        std::path::Path::new(input),
        top_for_library,
        &options,
    );
    match result {
        Ok(sv) => println!("{}", sv),
        Err(e) => report_cli_error_and_exit("stitch error", Some(&e.0), vec![]),
    }
}
