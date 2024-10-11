// SPDX-License-Identifier: Apache-2.0

//! This is a command line driver program that integrates various XLS
//! capabilities.
//!
//! It can either work via the runtime APIs or by invoking the XLS tools present
//! in a directory (e.g. as downloaded from a release).
//!
//! Commands are given like:
//!
//! ```text
//! xlsynth-driver <global-options> <command> <command-args-and-options>
//! ```
//!
//! Commands are:
//!
//! - dslx2sv: Converts a DSLX file to SystemVerilog.
//! - dslx2ir: Converts a DSLX file to the XLS IR.
//! - ir2opt: Converts an XLS IR file to an optimized XLS IR file.
//!
//! Sample usage:
//!
//! ```shell
//! $ cargo run -- --tool_path=/home/cdleary/opt/xlsynth/latest/ dslx2ir ../sample-usage/src/sample.x
//! ```

use clap::{App, Arg, ArgMatches, SubCommand};
use std::process;
use std::process::Command;

fn main() {
    let matches = App::new("xlsynth-driver")
        .version("0.1.0")
        .about("Command line driver for XLS/xlsynth capabilities")
        .arg(
            Arg::with_name("tool_path")
                .long("tool_path")
                .value_name("TOOL_PATH")
                .help("Path to a directory containing binary tools to use in lieu of runtime APIs")
                .takes_value(true),
        )
        .subcommand(
            SubCommand::with_name("dslx2sv")
                .about("Converts DSLX to SystemVerilog")
                .arg(
                    Arg::with_name("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::with_name("TOP")
                        .help("The top-level entry point (optional)")
                        .required(false)
                        .index(2),
                )
                .arg(
                    Arg::with_name("dslx_stdlib_path")
                        .long("dslx_stdlib_path")
                        .value_name("DSLX_STDLIB_PATH")
                        .help("Path to the DSLX standard library")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("dslx_path")
                        .long("dslx_path")
                        .value_name("DSLX_PATH_SEMI_SEPARATED")
                        .help("Semi-separated paths for DSLX")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("dslx2ir")
                .about("Converts DSLX to IR")
                .arg(
                    Arg::with_name("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::with_name("TOP")
                        .help("The top-level entry point (optional)")
                        .required(false)
                        .index(2),
                )
                .arg(
                    Arg::with_name("dslx_stdlib_path")
                        .long("dslx_stdlib_path")
                        .value_name("DSLX_STDLIB_PATH")
                        .help("Path to the DSLX standard library")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("dslx_path")
                        .long("dslx_path")
                        .value_name("DSLX_PATH_SEMI_SEPARATED")
                        .help("Semi-separated paths for DSLX")
                        .takes_value(true),
                ),
        )
        // ir2opt subcommand requires a top symbol
        .subcommand(
            SubCommand::with_name("ir2opt")
                .about("Converts IR to optimized IR")
                .arg(
                    Arg::with_name("INPUT_FILE")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                // Top is given as a (non-positional) flag for symmetry but it is required.
                .arg(
                    Arg::with_name("TOP")
                        .long("top")
                        .value_name("TOP")
                        .help("The top-level entry point")
                        .required(true)
                        .takes_value(true),
                ),
        )
        // ir2pipeline subcommand
        // requires a delay model flag
        // takes either a --clock_period_ps or pipeline_stages flag
        .subcommand(
            SubCommand::with_name("ir2pipeline")
                .about("Converts IR to a pipeline")
                .arg(
                    Arg::with_name("INPUT_FILE")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::with_name("DELAY_MODEL")
                        .long("delay_model")
                        .value_name("DELAY_MODEL")
                        .help("The delay model to use")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("pipeline_stages")
                        .long("pipeline_stages")
                        .value_name("PIPELINE_STAGES")
                        .help("The number of pipeline stages to use")
                        .required(false)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("clock_period_ps")
                        .long("clock_period_ps")
                        .value_name("CLOCK_PERIOD_PS")
                        .help("The clock period in picoseconds")
                        .required(false)
                        .takes_value(true),
                ),
        )
        .get_matches();

    let tool_path = matches.value_of("tool_path");

    if let Some(matches) = matches.subcommand_matches("dslx2sv") {
        handle_dslx2sv(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("dslx2ir") {
        handle_dslx2ir(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("ir2opt") {
        handle_ir2opt(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("ir2pipeline") {
        handle_ir2pipeline(matches, tool_path);
    } else {
        eprintln!("No valid subcommand provided.");
        process::exit(1);
    }
}

enum PipelineSpec {
    Stages(u64),
    ClockPeriodPs(u64),
}

fn handle_ir2pipeline(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let delay_model = matches.value_of("DELAY_MODEL").unwrap();

    // See which of pipeline_stages or clock_period_ps we're using.
    let pipeline_spec = if let Some(pipeline_stages) = matches.value_of("pipeline_stages") {
        PipelineSpec::Stages(pipeline_stages.parse().unwrap())
    } else if let Some(clock_period_ps) = matches.value_of("clock_period_ps") {
        PipelineSpec::ClockPeriodPs(clock_period_ps.parse().unwrap())
    } else {
        eprintln!("Must provide either --pipeline_stages or --clock_period_ps");
        process::exit(1);
    };

    ir2pipeline(input_file, delay_model, &pipeline_spec, tool_path);
}

fn handle_dslx2sv(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let top = matches.value_of("TOP");
    let dslx_stdlib_path = matches.value_of("dslx_stdlib_path");
    let dslx_path = matches.value_of("dslx_path");

    // Stub function for DSLX to SV conversion
    dslx2sv(input_file, top, dslx_stdlib_path, dslx_path, tool_path);
}

fn handle_dslx2ir(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let top = matches.value_of("TOP");
    let dslx_stdlib_path = matches.value_of("dslx_stdlib_path");
    let dslx_path = matches.value_of("dslx_path");

    // Stub function for DSLX to IR conversion
    dslx2ir(input_file, top, dslx_stdlib_path, dslx_path, tool_path);
}

fn handle_ir2opt(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let top = matches.value_of("TOP").unwrap();
    ir2opt(input_file, top, tool_path);
}

/// To convert an IR file to a pipeline we run the codegen_main command and give
/// it a number of pipeline stages.
fn ir2pipeline(
    input_file: &str,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    tool_path: Option<&str>,
) {
    if let Some(tool_path) = tool_path {
        // Give an error if the codegen_main tool is not found.
        let codegen_main_path = format!("{}/codegen_main", tool_path);
        if !std::path::Path::new(&codegen_main_path).exists() {
            eprintln!("codegen_main tool not found at: {}", codegen_main_path);
            process::exit(1);
        }

        let mut command = Command::new(codegen_main_path);
        command
            .arg(input_file)
            .arg("--delay_model")
            .arg(delay_model);

        let command = match pipeline_spec {
            PipelineSpec::Stages(stages) => {
                command.arg("--pipeline_stages").arg(stages.to_string())
            }
            PipelineSpec::ClockPeriodPs(clock_period_ps) => command
                .arg("--clock_period_ps")
                .arg(clock_period_ps.to_string()),
        };

        // We run the codegen_main tool on the given input file.
        let output = command.output().expect("Failed to execute codegen_main");

        if !output.status.success() {
            eprintln!("Pipeline generation failed with status: {}", output.status);
            eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            process::exit(1);
        }

        println!(
            "Pipeline generation output: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    } else {
        todo!("ir2pipeline subcommand using runtime APIs")
    }
}

fn ir2opt(input_file: &str, top: &str, tool_path: Option<&str>) {
    if let Some(tool_path) = tool_path {
        // Give an error if the ir_opt tool is not found.
        let ir_opt_path = format!("{}/opt_main", tool_path);
        if !std::path::Path::new(&ir_opt_path).exists() {
            eprintln!("IR optimization tool not found at: {}", ir_opt_path);
            process::exit(1);
        }

        // We run the ir_opt tool on the given input file.
        let output = Command::new(ir_opt_path)
            .arg(input_file)
            .arg("--top")
            .arg(top)
            .output()
            .expect("Failed to execute IR optimization");

        if !output.status.success() {
            eprintln!("IR optimization failed with status: {}", output.status);
            eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            process::exit(1);
        }

        println!("{}", String::from_utf8_lossy(&output.stdout));
    } else {
        todo!("ir2opt subcommand using runtime APIs")
    }
}

fn dslx2sv(
    input_file: &str,
    top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: Option<&str>,
) {
    todo!("dslx2sv subcommand")
}

fn dslx2ir(
    input_file: &str,
    top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: Option<&str>,
) {
    if let Some(tool_path) = tool_path {
        // Give an error if the ir_convert tool is not found.
        let ir_convert_path = format!("{}/ir_converter_main", tool_path);
        if !std::path::Path::new(&ir_convert_path).exists() {
            eprintln!("IR conversion tool not found at: {}", ir_convert_path);
            process::exit(1);
        }

        // We run the ir_convert tool on the given input file.
        // If top is specified we pass --top, and similarly for dslx_stdlib_path and
        // dslx_path.
        let mut command = Command::new(ir_convert_path);
        command.arg(input_file);

        if let Some(top) = top {
            command.arg("--top").arg(top);
        }

        if let Some(dslx_stdlib_path) = dslx_stdlib_path {
            command.arg("--dslx_stdlib_path").arg(dslx_stdlib_path);
        }

        if let Some(dslx_path) = dslx_path {
            command.arg("--dslx_path").arg(dslx_path);
        }

        let output = command.output().expect("Failed to execute ir_convert");

        if !output.status.success() {
            eprintln!("IR conversion failed with status: {}", output.status);
            eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            process::exit(1);
        }

        println!("{}", String::from_utf8_lossy(&output.stdout));
    } else {
        todo!("dslx2ir subcommand using runtime APIs")
    }
}
