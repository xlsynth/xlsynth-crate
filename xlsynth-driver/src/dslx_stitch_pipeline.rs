// SPDX-License-Identifier: Apache-2.0

use crate::common::get_dslx_paths;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;
use tar::Builder;
use xlsynth_g8r::dslx_stitch_pipeline::IrPackageTextFile;
use xlsynth_g8r::verilog_version::VerilogVersion;

/// Writes the given IR files to a tar.gz file.
///
/// The filenames come from the `file_name` recorded on the IrPackageTextFile
/// struct.
fn write_ir_tgz(tar_gz_path: &str, ir_files: &[IrPackageTextFile]) -> Result<(), String> {
    let file = std::fs::File::create(tar_gz_path)
        .map_err(|e| format!("could not create tar.gz output file '{tar_gz_path}': {e}"))?;
    let enc = GzEncoder::new(file, Compression::default());
    let mut tar = Builder::new(enc);

    for f in ir_files {
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(f.ir_text.as_bytes().len() as u64);
        header.set_cksum();
        tar.append_data(&mut header, f.file_name.as_str(), f.ir_text.as_bytes())
            .map_err(|e| format!("could not append '{}' to tarball: {e}", f.file_name))?;
    }

    let enc = tar
        .into_inner()
        .map_err(|e| format!("could not finish tarball '{tar_gz_path}': {e}"))?;
    let mut file = enc
        .finish()
        .map_err(|e| format!("could not finish gzip stream '{tar_gz_path}': {e}"))?;
    file.flush()
        .map_err(|e| format!("could not flush gzip file '{tar_gz_path}': {e}"))?;
    Ok(())
}

/// Handles the `dslx-stitch-pipeline` subcommand.
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
    let output_unopt_ir_tgz = matches
        .get_one::<String>("output_unopt_ir_tgz")
        .map(|s| s.as_str());
    let output_opt_ir_tgz = matches
        .get_one::<String>("output_opt_ir_tgz")
        .map(|s| s.as_str());

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
    let result = xlsynth_g8r::dslx_stitch_pipeline::stitch_pipeline(
        &dslx,
        std::path::Path::new(input),
        top_for_library,
        &options,
    );
    match result {
        Ok(stitch_output) => {
            if let Some(unopt_ir_tgz_path) = output_unopt_ir_tgz {
                let unopt_files = vec![IrPackageTextFile {
                    file_name: "unopt.ir".to_string(),
                    ir_text: stitch_output.unopt_ir_text.clone(),
                }];
                if let Err(msg) = write_ir_tgz(unopt_ir_tgz_path, &unopt_files) {
                    report_cli_error_and_exit(
                        "could not write unoptimized IR tarball",
                        Some(&msg),
                        vec![],
                    );
                }
            }
            if let Some(opt_ir_tgz_path) = output_opt_ir_tgz {
                if let Err(msg) = write_ir_tgz(opt_ir_tgz_path, &stitch_output.opt_ir_files) {
                    report_cli_error_and_exit(
                        "could not write optimized IR tarball",
                        Some(&msg),
                        vec![],
                    );
                }
            }

            println!("{}", stitch_output.sv_text)
        }
        Err(e) => report_cli_error_and_exit("stitch error", Some(&e.0), vec![]),
    }
}
