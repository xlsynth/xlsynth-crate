// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::ArgMatches;
use xlsynth::XlsynthError;
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
use xlsynth_g8r::dslx_stitch_pipeline::StitchPipelineSpec;
use xlsynth_g8r::dslx_stitch_pipeline::g8r_pipeline::{
    StitchG8rPipelineOptions, stitch_g8r_pipeline,
};
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;

use crate::common::{get_dslx_paths, write_stdout};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

/// Handles the `dslx-stitch-g8r-pipeline` subcommand.
pub fn handle_dslx_stitch_g8r_pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let mut lowering = crate::g8r_cli::parse_g8r_cli_options(matches);
    if !cfg!(feature = "has-bitwuzla")
        && matches
            .get_one::<String>("enable-formal-array-alias-analysis")
            .is_none()
    {
        lowering.enable_formal_array_alias_analysis = false;
    }
    let design = match matches.get_one::<String>("spec") {
        Some(path) => StitchPipelineSpec::load(Path::new(path))
            .and_then(|spec| spec.resolve())
            .and_then(|resolved| resolved.lower_to_g8r(&lowering)),
        None => stitch_from_flags(matches, config, &lowering),
    }
    .unwrap_or_else(|error| {
        report_cli_error_and_exit("stitch error", Some(&error.0), vec![]);
    });

    let text = emit_g8r(&design).unwrap_or_else(|error| {
        report_cli_error_and_exit("could not serialize g8r text", Some(&error), vec![]);
    });
    write_stdout(&text);
    if let Some(bin_path) = matches.get_one::<String>("bin_out") {
        let binary = encode_g8r_binary(&design).unwrap_or_else(|error| {
            report_cli_error_and_exit("could not serialize g8r binary", Some(&error), vec![]);
        });
        std::fs::write(bin_path, binary).unwrap_or_else(|error| {
            report_cli_error_and_exit(
                "could not write g8r binary",
                Some(&error.to_string()),
                vec![("path", bin_path)],
            );
        });
    }
}

/// Lowers stages using the original CLI's native g8r name and clock
/// conventions.
fn stitch_from_flags(
    matches: &ArgMatches,
    config: &Option<ToolchainConfig>,
    lowering: &CanonicalG8rOptions,
) -> Result<SequentialGateFn, XlsynthError> {
    let paths = get_dslx_paths(matches, config);
    let path_refs = paths.search_path_views();
    let input = matches.get_one::<String>("dslx_input_file").unwrap();
    let dslx = std::fs::read_to_string(input).unwrap_or_else(|error| {
        report_cli_error_and_exit(
            "could not read DSLX input",
            Some(&error.to_string()),
            vec![],
        );
    });
    let dslx_top = matches.get_one::<String>("dslx_top");
    let stage_list = matches.get_one::<String>("stages").map(|csv| {
        csv.split(',')
            .map(|stage| stage.trim().to_string())
            .collect::<Vec<String>>()
    });
    let output_design_name = matches.get_one::<String>("output_design_name");
    if stage_list.is_some() && dslx_top.is_some() {
        report_cli_error_and_exit(
            "--dslx_top is mutually exclusive with --stages",
            None,
            vec![],
        );
    }
    if stage_list.is_some() && output_design_name.is_none() {
        report_cli_error_and_exit(
            "--output_design_name is required when --stages is provided",
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

    let stage_discovery_prefix = dslx_top.map(String::as_str);
    let design_name = output_design_name
        .map(String::as_str)
        .or(stage_discovery_prefix)
        .expect("validated pipeline name arguments above");
    let top_for_library = stage_discovery_prefix.unwrap_or(design_name);
    let options = StitchG8rPipelineOptions {
        explicit_stages: stage_list,
        stdlib_path: paths.stdlib_path.as_deref(),
        search_paths: path_refs,
        flop_inputs: matches
            .get_one::<String>("flop_inputs")
            .map(|value| value == "true")
            .unwrap_or(crate::flag_defaults::CODEGEN_FLOP_INPUTS),
        flop_outputs: matches
            .get_one::<String>("flop_outputs")
            .map(|value| value == "true")
            .unwrap_or(crate::flag_defaults::CODEGEN_FLOP_OUTPUTS),
        input_valid_signal: matches
            .get_one::<String>("input_valid_signal")
            .map(String::as_str),
        output_valid_signal: matches
            .get_one::<String>("output_valid_signal")
            .map(String::as_str),
        reset_signal: matches.get_one::<String>("reset").map(String::as_str),
        reset_active_low: matches
            .get_one::<String>("reset_active_low")
            .map(|value| value == "true")
            .unwrap_or(false),
        output_design_name: design_name,
        clock_name: matches
            .get_one::<String>("clock_name")
            .map(String::as_str)
            .unwrap_or("clk"),
    };
    stitch_g8r_pipeline(&dslx, Path::new(input), top_for_library, &options, lowering)
}
