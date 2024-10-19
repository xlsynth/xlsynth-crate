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
//! - dslx2pipeline: Converts a DSLX entry point to a SystemVerilog pipeline.
//! - dslx2ir: Converts a DSLX file to the XLS IR.
//! - ir2opt: Converts an XLS IR file to an optimized XLS IR file.
//! - dslx2sv-types: Convert type definitions in a DSLX file to SystemVerilog
//!   types.
//!
//! Sample usage:
//!
//! ```shell
//! $ cargo run -- --tool_path=/home/cdleary/opt/xlsynth/latest/ \
//!     dslx2ir ../sample-usage/src/sample.x
//! $ cargo run -- --tool_path=/home/cdleary/opt/xlsynth/latest/ \
//!     dslx2pipeline ../sample-usage/src/sample.x add1 \
//!     --delay_model=asap7 --pipeline_stages=2
//! $ cargo run -- \
//!     dslx2sv-types ../tests/structure_zoo.x
//! ```

use clap::{App, Arg, ArgMatches, SubCommand};
use std::process;
use std::process::Command;

trait AppExt {
    fn add_delay_model_arg(self) -> Self;
    fn add_pipeline_args(self) -> Self;
}

impl AppExt for App<'_, '_> {
    fn add_delay_model_arg(self) -> Self {
        (self as App).arg(
            Arg::with_name("DELAY_MODEL")
                .long("delay_model")
                .value_name("DELAY_MODEL")
                .help("The delay model to use")
                .required(true)
                .takes_value(true),
        )
    }

    fn add_pipeline_args(self) -> Self {
        (self as App)
            .arg(
                Arg::with_name("pipeline_stages")
                    .long("pipeline_stages")
                    .value_name("PIPELINE_STAGES")
                    .help("Number of pipeline stages")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name("clock_period_ps")
                    .long("clock_period_ps")
                    .value_name("CLOCK_PERIOD_PS")
                    .help("Clock period in picoseconds")
                    .takes_value(true),
            )
    }
}

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
            SubCommand::with_name("dslx2pipeline")
                .about("Converts DSLX to SystemVerilog")
                .arg(
                    Arg::with_name("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::with_name("TOP")
                        .help("The top-level entry point")
                        .required(true)
                        .index(2),
                )
                .add_delay_model_arg()
                .add_pipeline_args()
                // --keep_temps flag to keep temporary files
                .arg(
                    Arg::with_name("keep_temps")
                        .long("keep_temps")
                        .help("Keep temporary files")
                        .takes_value(false),
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
        // dslx2sv-types converts all the definitions in the .x file to SV types
        .subcommand(
            SubCommand::with_name("dslx2sv-types")
                .about("Converts DSLX type definitions to SystemVerilog")
                .arg(
                    Arg::with_name("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
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
                .add_delay_model_arg()
                .add_pipeline_args(),
        )
        .get_matches();

    let tool_path = matches.value_of("tool_path");

    if let Some(matches) = matches.subcommand_matches("dslx2pipeline") {
        handle_dslx2pipeline(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("dslx2ir") {
        handle_dslx2ir(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("ir2opt") {
        handle_ir2opt(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("ir2pipeline") {
        handle_ir2pipeline(matches, tool_path);
    } else if let Some(matches) = matches.subcommand_matches("dslx2sv-types") {
        handle_dslx2sv_types(matches, tool_path);
    } else {
        eprintln!("No valid subcommand provided.");
        process::exit(1);
    }
}

enum PipelineSpec {
    Stages(u64),
    ClockPeriodPs(u64),
}

fn extract_pipeline_spec(matches: &ArgMatches) -> PipelineSpec {
    if let Some(pipeline_stages) = matches.value_of("pipeline_stages") {
        PipelineSpec::Stages(pipeline_stages.parse().unwrap())
    } else if let Some(clock_period_ps) = matches.value_of("clock_period_ps") {
        PipelineSpec::ClockPeriodPs(clock_period_ps.parse().unwrap())
    } else {
        eprintln!("Must provide either --pipeline_stages or --clock_period_ps");
        process::exit(1);
    }
}

fn handle_ir2pipeline(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.value_of("DELAY_MODEL").unwrap();

    // See which of pipeline_stages or clock_period_ps we're using.
    let pipeline_spec = extract_pipeline_spec(matches);

    ir2pipeline(input_path, delay_model, &pipeline_spec, tool_path);
}

fn handle_dslx2pipeline(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = matches.value_of("TOP").unwrap();
    let dslx_stdlib_path = matches.value_of("dslx_stdlib_path");
    let dslx_path = matches.value_of("dslx_path");
    let pipeline_spec = extract_pipeline_spec(matches);
    let delay_model = matches.value_of("DELAY_MODEL").unwrap();
    let keep_temps = matches.is_present("keep_temps");

    // Stub function for DSLX to SV conversion
    dslx2pipeline(
        input_path,
        top,
        &pipeline_spec,
        dslx_stdlib_path,
        dslx_path,
        delay_model,
        keep_temps,
        tool_path,
    );
}

fn handle_dslx2ir(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = matches.value_of("TOP");
    let dslx_stdlib_path = matches.value_of("dslx_stdlib_path");
    let dslx_path = matches.value_of("dslx_path");

    // Stub function for DSLX to IR conversion
    dslx2ir(input_path, top, dslx_stdlib_path, dslx_path, tool_path);
}

fn handle_ir2opt(matches: &ArgMatches, tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let top = matches.value_of("TOP").unwrap();
    let input_path = std::path::Path::new(input_file);

    ir2opt(input_path, top, tool_path);
}

fn handle_dslx2sv_types(matches: &ArgMatches, _tool_path: Option<&str>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);

    // Stub function for DSLX to SV type conversion
    dslx2sv_types(input_path);
}

fn run_codegen_pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    tool_path: &str,
) -> String {
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
        PipelineSpec::Stages(stages) => command.arg("--pipeline_stages").arg(stages.to_string()),
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

    String::from_utf8_lossy(&output.stdout).to_string()
}

/// To convert an IR file to a pipeline we run the codegen_main command and give
/// it a number of pipeline stages.
fn ir2pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    tool_path: Option<&str>,
) {
    if let Some(tool_path) = tool_path {
        let output = run_codegen_pipeline(input_file, delay_model, pipeline_spec, tool_path);
        println!("{}", output);
    } else {
        todo!("ir2pipeline subcommand using runtime APIs")
    }
}

fn run_opt_main(input_file: &std::path::Path, top: Option<&str>, tool_path: &str) -> String {
    let opt_main_path = format!("{}/opt_main", tool_path);
    if !std::path::Path::new(&opt_main_path).exists() {
        eprintln!("IR optimization tool not found at: {}", opt_main_path);
        process::exit(1);
    }

    let mut command = Command::new(opt_main_path);
    command.arg(input_file);
    if top.is_some() {
        command.arg("--top").arg(top.unwrap());
    }

    let output = command.output().expect("Failed to execute IR optimization");

    if !output.status.success() {
        eprintln!("IR optimization failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

fn ir2opt(input_file: &std::path::Path, top: &str, tool_path: Option<&str>) {
    if let Some(tool_path) = tool_path {
        let output = run_opt_main(input_file, Some(top), tool_path);
        println!("{}", output);
    } else {
        todo!("ir2opt subcommand using runtime APIs")
    }
}

fn dslx2sv_types(input_file: &std::path::Path) {
    let dslx = std::fs::read_to_string(input_file).unwrap();
    let mut import_data = xlsynth::dslx::ImportData::default();
    let mut builder = xlsynth::sv_bridge_builder::SvBridgeBuilder::new();
    xlsynth::dslx_bridge::convert_leaf_module(&mut import_data, &dslx, input_file, &mut builder)
        .unwrap();
    let sv = builder.build();
    println!("{}", sv);
}

fn dslx2pipeline(
    input_file: &std::path::Path,
    top: &str,
    pipeline_spec: &PipelineSpec,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    delay_model: &str,
    keep_temps: bool,
    tool_path: Option<&str>,
) {
    if let Some(tool_path) = tool_path {
        let temp_dir = tempfile::TempDir::new().unwrap();

        let module_name = xlsynth::dslx_path_to_module_name(input_file).unwrap();

        let unopt_ir = run_ir_converter_main(
            input_file,
            Some(top),
            dslx_stdlib_path,
            dslx_path,
            tool_path,
        );
        let unopt_ir_path = temp_dir.path().join("unopt.ir");
        std::fs::write(&unopt_ir_path, unopt_ir).unwrap();

        let ir_top = xlsynth::mangle_dslx_name(module_name, top).unwrap();

        let opt_ir = run_opt_main(&unopt_ir_path, Some(&ir_top), tool_path);
        let opt_ir_path = temp_dir.path().join("opt.ir");
        std::fs::write(&opt_ir_path, opt_ir).unwrap();

        let sv = run_codegen_pipeline(&opt_ir_path, delay_model, pipeline_spec, tool_path);
        let sv_path = temp_dir.path().join("output.sv");
        std::fs::write(&sv_path, &sv).unwrap();

        if keep_temps {
            eprintln!(
                "Pipeline generation successful. Output written to: {}",
                temp_dir.into_path().to_str().unwrap()
            );
        }
        println!("{}", sv);
    } else {
        todo!("dslx2pipeline subcommand using runtime APIs")
    }
}

fn run_ir_converter_main(
    input_file: &std::path::Path,
    top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: &str,
) -> String {
    let ir_convert_path = format!("{}/ir_converter_main", tool_path);
    if !std::path::Path::new(&ir_convert_path).exists() {
        eprintln!("IR conversion tool not found at: {}", ir_convert_path);
        process::exit(1);
    }

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

    String::from_utf8_lossy(&output.stdout).to_string()
}

fn dslx2ir(
    input_file: &std::path::Path,
    top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: Option<&str>,
) {
    if let Some(tool_path) = tool_path {
        let output = run_ir_converter_main(input_file, top, dslx_stdlib_path, dslx_path, tool_path);
        println!("{}", output);
    } else {
        todo!("dslx2ir subcommand using runtime APIs")
    }
}
