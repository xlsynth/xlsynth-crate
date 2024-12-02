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
//! $ cargo run -- --toolchain=$HOME/xlsynth-toolchain.toml \
//!     dslx2ir ../sample-usage/src/sample.x
//! $ cargo run -- --toolchain=$HOME/xlsynth-toolchain.toml \
//!     dslx2pipeline ../sample-usage/src/sample.x add1 \
//!     --delay_model=asap7 --pipeline_stages=2 \
//!     --input_valid_signal=input_valid \
//!     --output_valid_signal=output_valid
//! $ cargo run -- \
//!     dslx2sv-types ../tests/structure_zoo.x
//! ```

use clap;
use clap::{Arg, ArgAction, ArgMatches};
use serde::Deserialize;
use std::process;
use std::process::Command;
use xlsynth::DslxConvertOptions;

#[derive(Deserialize)]
struct ToolchainConfig {
    /// Path to the DSLX standard library.
    dslx_stdlib_path: Option<String>,

    /// Additional paths to use in the DSLX module search, i.e. as roots for
    /// import statements.
    dslx_path: Vec<String>,

    /// Directory path for the XLS toolset, e.g. codegen_main, opt_main, etc.
    tool_path: Option<String>,
}

#[derive(Deserialize)]
struct XlsynthToolchain {
    toolchain: ToolchainConfig,
}

trait AppExt {
    fn add_delay_model_arg(self) -> Self;
    fn add_pipeline_args(self) -> Self;
    fn add_dslx_path_arg(self) -> Self;
    fn add_dslx_stdlib_path_arg(self) -> Self;
    fn add_codegen_args(self) -> Self;
}

impl AppExt for clap::Command {
    fn add_delay_model_arg(self) -> Self {
        (self as clap::Command).arg(
            Arg::new("DELAY_MODEL")
                .long("delay_model")
                .value_name("DELAY_MODEL")
                .help("The delay model to use")
                .required(true)
                .action(ArgAction::Set),
        )
    }

    fn add_pipeline_args(self) -> Self {
        (self as clap::Command)
            .arg(
                Arg::new("pipeline_stages")
                    .long("pipeline_stages")
                    .value_name("PIPELINE_STAGES")
                    .help("Number of pipeline stages")
                    .action(ArgAction::Set),
            )
            .arg(
                Arg::new("clock_period_ps")
                    .long("clock_period_ps")
                    .value_name("CLOCK_PERIOD_PS")
                    .help("Clock period in picoseconds")
                    .action(ArgAction::Set),
            )
    }

    fn add_dslx_path_arg(self) -> Self {
        (self as clap::Command).arg(
            Arg::new("dslx_path")
                .long("dslx_path")
                .value_name("DSLX_PATH_SEMI_SEPARATED")
                .help("Semi-separated paths for DSLX")
                .action(ArgAction::Set),
        )
    }

    fn add_dslx_stdlib_path_arg(self) -> Self {
        (self as clap::Command).arg(
            Arg::new("dslx_stdlib_path")
                .long("dslx_stdlib_path")
                .value_name("DSLX_STDLIB_PATH")
                .help("Path to the DSLX standard library")
                .action(ArgAction::Set),
        )
    }

    fn add_codegen_args(self) -> Self {
        (self as clap::Command)
            .arg(
                Arg::new("input_valid_signal")
                    .long("input_valid_signal")
                    .value_name("INPUT_VALID_SIGNAL")
                    .help("Load enable signal for pipeline registers"),
            )
            .arg(
                Arg::new("output_valid_signal")
                    .long("output_valid_signal")
                    .value_name("OUTPUT_VALID_SIGNAL")
                    .help("Output port holding pipelined valid signal"),
            )
            .arg(
                Arg::new("flop_inputs")
                    .long("flop_inputs")
                    .value_name("BOOL")
                    .action(ArgAction::Set)
                    .value_parser(["true", "false"])
                    .num_args(0)
                    .help("Flop input ports"),
            )
            .arg(
                Arg::new("flop_outputs")
                    .long("flop_outputs")
                    .value_name("BOOL")
                    .action(ArgAction::Set)
                    .value_parser(["true", "false"])
                    .num_args(0)
                    .help("Flop output ports"),
            )
            .arg(
                Arg::new("add_idle_output")
                    .long("add_idle_output")
                    .value_name("BOOL")
                    .action(ArgAction::Set)
                    .value_parser(["true", "false"])
                    .num_args(0)
                    .help("Add an idle output port"),
            )
            .arg(
                Arg::new("module_name")
                    .long("module_name")
                    .value_name("MODULE_NAME")
                    .help("Name of the generated module"),
            )
            .arg(
                Arg::new("array_index_bounds_checking")
                    .long("array_index_bounds_checking")
                    .value_name("BOOL")
                    .action(ArgAction::Set)
                    .value_parser(["true", "false"])
                    .num_args(0)
                    .help("Array index bounds checking"),
            )
            .arg(
                Arg::new("separate_lines")
                    .long("separate_lines")
                    .value_name("BOOL")
                    .action(ArgAction::Set)
                    .value_parser(["true", "false"])
                    .num_args(0)
                    .help("Separate lines in generated code"),
            )
    }
}

fn main() {
    let _ = env_logger::try_init();

    let matches = clap::Command::new("xlsynth-driver")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Command line driver for XLS/xlsynth capabilities")
        .arg(
            Arg::new("toolchain")
                .long("toolchain")
                .value_name("TOOLCHAIN")
                .help("Path to a xlsynth-toolchain.toml file")
                .action(ArgAction::Set),
        )
        .subcommand(clap::Command::new("version").about("Prints the version of the driver"))
        .subcommand(
            clap::Command::new("dslx2pipeline")
                .about("Converts DSLX to SystemVerilog")
                .arg(
                    Arg::new("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("TOP")
                        .help("The top-level entry point")
                        .required(true)
                        .index(2),
                )
                .add_delay_model_arg()
                .add_pipeline_args()
                .add_codegen_args()
                // --keep_temps flag to keep temporary files
                .arg(
                    Arg::new("keep_temps")
                        .long("keep_temps")
                        .help("Keep temporary files")
                        .action(ArgAction::Set),
                ),
        )
        .subcommand(
            clap::Command::new("dslx2ir")
                .about("Converts DSLX to IR")
                .arg(
                    Arg::new("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("TOP")
                        .help("The top-level entry point (optional)")
                        .required(false)
                        .index(2),
                )
                .add_dslx_stdlib_path_arg()
                .add_dslx_path_arg(),
        )
        // dslx2sv-types converts all the definitions in the .x file to SV types
        .subcommand(
            clap::Command::new("dslx2sv-types")
                .about("Converts DSLX type definitions to SystemVerilog")
                .arg(
                    Arg::new("INPUT_FILE")
                        .help("The input DSLX file")
                        .required(true)
                        .index(1),
                )
                .add_dslx_stdlib_path_arg()
                .add_dslx_path_arg(),
        )
        // ir2opt subcommand requires a top symbol
        .subcommand(
            clap::Command::new("ir2opt")
                .about("Converts IR to optimized IR")
                .arg(
                    Arg::new("INPUT_FILE")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                // Top is given as a (non-positional) flag for symmetry but it is required.
                .arg(
                    Arg::new("TOP")
                        .long("top")
                        .value_name("TOP")
                        .help("The top-level entry point")
                        .required(true)
                        .action(ArgAction::Set),
                ),
        )
        // ir2pipeline subcommand
        // requires a delay model flag
        // takes either a --clock_period_ps or pipeline_stages flag
        .subcommand(
            clap::Command::new("ir2pipeline")
                .about("Converts IR to a pipeline")
                .arg(
                    Arg::new("INPUT_FILE")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .add_delay_model_arg()
                .add_codegen_args()
                .add_pipeline_args(),
        )
        .subcommand(
            clap::Command::new("ir2delayinfo")
                .about("Converts IR entry point to delay info output")
                .add_delay_model_arg()
                .arg(
                    Arg::new("INPUT_FILE")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("TOP")
                        .help("The top-level entry point")
                        .required(true)
                        .index(2),
                ),
        )
        .get_matches();

    let toml_path = matches.get_one::<String>("toolchain");
    let toml_value: Option<toml::Value> = toml_path.map(|path| {
        // If we got a toolchain toml file, read/parse it.
        let toml_str = std::fs::read_to_string(path).expect("Failed to read toolchain toml file");
        toml::from_str(&toml_str).expect("Failed to parse toolchain toml file")
    });
    let config = toml_value.map(|v| {
        let toolchain_config = v
            .clone()
            .try_into::<XlsynthToolchain>()
            .expect(&format!("Failed to parse toolchain config; value: {}", v));
        toolchain_config.toolchain
    });

    if let Some(matches) = matches.subcommand_matches("dslx2pipeline") {
        handle_dslx2pipeline(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx2ir") {
        handle_dslx2ir(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2opt") {
        handle_ir2opt(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2pipeline") {
        handle_ir2pipeline(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx2sv-types") {
        handle_dslx2sv_types(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2delayinfo") {
        handle_ir2delayinfo(matches, &config);
    } else if let Some(_matches) = matches.subcommand_matches("version") {
        println!("{}", env!("CARGO_PKG_VERSION"));
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
    if let Some(pipeline_stages) = matches.get_one::<String>("pipeline_stages") {
        PipelineSpec::Stages(pipeline_stages.parse().unwrap())
    } else if let Some(clock_period_ps) = matches.get_one::<String>("clock_period_ps") {
        PipelineSpec::ClockPeriodPs(clock_period_ps.parse().unwrap())
    } else {
        eprintln!("Must provide either --pipeline_stages or --clock_period_ps");
        process::exit(1)
    }
}

/// Extracts flags that we pass to the "codegen" step of the process (i.e.
/// generating lowered Verilog).
fn extract_codegen_flags(matches: &ArgMatches) -> CodegenFlags {
    CodegenFlags {
        input_valid_signal: matches
            .get_one::<String>("input_valid_signal")
            .map(|s| s.to_string()),
        output_valid_signal: matches
            .get_one::<String>("output_valid_signal")
            .map(|s| s.to_string()),
        use_system_verilog: matches
            .get_one::<String>("use_system_verilog")
            .map(|s| s == "true"),
        flop_inputs: matches
            .get_one::<String>("flop_inputs")
            .map(|s| s == "true"),
        flop_outputs: matches
            .get_one::<String>("flop_outputs")
            .map(|s| s == "true"),
        add_idle_output: matches
            .get_one::<String>("add_idle_output")
            .map(|s| s == "true"),
        module_name: matches
            .get_one::<String>("module_name")
            .map(|s| s.to_string()),
        array_index_bounds_checking: matches
            .get_one::<String>("array_index_bounds_checking")
            .map(|s| s == "true"),
        separate_lines: matches
            .get_one::<String>("separate_lines")
            .map(|s| s == "true"),
    }
}

fn handle_ir2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();

    // See which of pipeline_stages or clock_period_ps we're using.
    let pipeline_spec = extract_pipeline_spec(matches);

    let codegen_flags = extract_codegen_flags(matches);

    ir2pipeline(
        input_path,
        delay_model,
        &pipeline_spec,
        &codegen_flags,
        config,
    );
}

fn handle_dslx2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = matches.get_one::<String>("TOP").unwrap();
    let pipeline_spec = extract_pipeline_spec(matches);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();
    let keep_temps = matches.get_flag("keep_temps");
    let codegen_flags = extract_codegen_flags(matches);

    // Stub function for DSLX to SV conversion
    dslx2pipeline(
        input_path,
        top,
        &pipeline_spec,
        &codegen_flags,
        delay_model,
        keep_temps,
        config,
    );
}

/// Helper for extracting the DSLX standard library path from the command line
/// flag, if specified, or the toolchain config if it's present and the cmdline
/// flag isn't specified.
fn get_dslx_stdlib_path(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Option<String> {
    let dslx_stdlib_path = matches.get_one::<String>("dslx_stdlib_path");
    if let Some(dslx_stdlib_path) = dslx_stdlib_path {
        Some(dslx_stdlib_path.to_string())
    } else if let Some(config) = config {
        config.dslx_stdlib_path.clone()
    } else {
        None
    }
}

/// Helper for retrieving supplemental DSLX search paths from the command line
/// flag, if specified, or the toolchain config if it's present and the cmdline
/// flag isn't specified.
fn get_dslx_path(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Option<String> {
    let dslx_path = matches.get_one::<String>("dslx_path");
    if let Some(dslx_path) = dslx_path {
        Some(dslx_path.to_string())
    } else if let Some(config) = config {
        Some(config.dslx_path.join(";"))
    } else {
        None
    }
}

fn handle_dslx2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = if let Some(top) = matches.get_one::<String>("TOP") {
        Some(top.to_string())
    } else {
        None
    };
    let top = top.as_deref();
    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path = dslx_stdlib_path.as_deref();

    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    // Stub function for DSLX to IR conversion
    dslx2ir(input_path, top, dslx_stdlib_path, dslx_path, tool_path);
}

fn handle_ir2opt(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("INPUT_FILE").unwrap();
    let top = matches.get_one::<String>("TOP").unwrap();
    let input_path = std::path::Path::new(input_file);

    ir2opt(input_path, top, config);
}

fn handle_dslx2sv_types(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path: Option<std::path::PathBuf> =
        dslx_stdlib_path.map(|s| std::path::Path::new(&s).to_path_buf());
    let dslx_stdlib_path = dslx_stdlib_path.as_ref().map(|p| p.as_path());

    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    // Stub function for DSLX to SV type conversion
    dslx2sv_types(input_path, dslx_stdlib_path, dslx_path);
}

fn handle_ir2delayinfo(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("INPUT_FILE").unwrap();
    let top = matches.get_one::<String>("TOP").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();

    ir2delayinfo(input_path, top, delay_model, config);
}

struct CodegenFlags {
    input_valid_signal: Option<String>,
    output_valid_signal: Option<String>,
    use_system_verilog: Option<bool>,
    flop_inputs: Option<bool>,
    flop_outputs: Option<bool>,
    add_idle_output: Option<bool>,
    module_name: Option<String>,
    array_index_bounds_checking: Option<bool>,
    separate_lines: Option<bool>,
}

/// Adds the given code-generation flags to the command in command-line-arg
/// form.
fn add_codegen_flags(command: &mut Command, codegen_flags: &CodegenFlags) {
    if let Some(use_system_verilog) = codegen_flags.use_system_verilog {
        command.arg(format!("--use_system_verilog={use_system_verilog}"));
    }
    if let Some(input_valid_signal) = &codegen_flags.input_valid_signal {
        command.arg("--input_valid_signal").arg(input_valid_signal);
    }
    if let Some(output_valid_signal) = &codegen_flags.output_valid_signal {
        command
            .arg("--output_valid_signal")
            .arg(output_valid_signal);
    }
    if let Some(flop_inputs) = codegen_flags.flop_inputs {
        command.arg(format!("--flop_inputs={flop_inputs}"));
    }
    if let Some(flop_outputs) = codegen_flags.flop_outputs {
        command.arg(format!("--flop_outputs={flop_outputs}"));
    }
    if let Some(add_idle_output) = codegen_flags.add_idle_output {
        command.arg(format!("--add_idle_output={add_idle_output}"));
    }
    if let Some(module_name) = &codegen_flags.module_name {
        command.arg("--module_name").arg(module_name);
    }
    if let Some(array_index_bounds_checking) = codegen_flags.array_index_bounds_checking {
        command.arg(format!(
            "--array_index_bounds_checking={array_index_bounds_checking}"
        ));
    }
    if let Some(separate_lines) = codegen_flags.separate_lines {
        command.arg(format!("--separate_lines={separate_lines}"));
    }
}

fn run_codegen_pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
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

    add_codegen_flags(&mut command, codegen_flags);

    let command = match pipeline_spec {
        PipelineSpec::Stages(stages) => command.arg("--pipeline_stages").arg(stages.to_string()),
        PipelineSpec::ClockPeriodPs(clock_period_ps) => command
            .arg("--clock_period_ps")
            .arg(clock_period_ps.to_string()),
    };

    log::info!("Running command: {:?}", command);

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
    codegen_flags: &CodegenFlags,
    config: &Option<ToolchainConfig>,
) {
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_codegen_pipeline(
            input_file,
            delay_model,
            pipeline_spec,
            codegen_flags,
            tool_path,
        );
        println!("{}", output);
    } else {
        todo!("ir2pipeline subcommand using runtime APIs")
    }
}

/// Runs the IR optimization command line tool and returns the output.
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

fn ir2opt(input_file: &std::path::Path, top: &str, config: &Option<ToolchainConfig>) {
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_opt_main(input_file, Some(top), tool_path);
        println!("{}", output);
    } else {
        todo!("ir2opt subcommand using runtime APIs")
    }
}

fn run_delay_info_main(
    input_file: &std::path::Path,
    top: Option<&str>,
    delay_model: &str,
    tool_path: &str,
) -> String {
    let delay_info_path = format!("{}/delay_info_main", tool_path);
    if !std::path::Path::new(&delay_info_path).exists() {
        eprintln!("Delay info tool not found at: {}", delay_info_path);
        process::exit(1);
    }

    let mut command = Command::new(delay_info_path);
    command.arg(input_file);
    command.arg("--delay_model").arg(delay_model);
    if top.is_some() {
        command.arg("--top").arg(top.unwrap());
    }

    let output = command.output().expect("Failed to execute delay_info_main");

    if !output.status.success() {
        eprintln!("Delay info failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

fn ir2delayinfo(
    input_file: &std::path::Path,
    top: &str,
    delay_model: &str,
    config: &Option<ToolchainConfig>,
) {
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_delay_info_main(input_file, Some(top), delay_model, tool_path);
        println!("{}", output);
    } else {
        todo!("ir2delayinfo subcommand using runtime APIs")
    }
}

fn dslx2sv_types(
    input_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    dslx_path: Option<&str>,
) {
    let dslx = std::fs::read_to_string(input_file).unwrap();

    let dslx_stdlib_path_buf: Option<std::path::PathBuf> =
        dslx_stdlib_path.map(|s| std::path::Path::new(s).to_path_buf());
    let dslx_stdlib_path = dslx_stdlib_path_buf.as_ref().map(|p| p.as_path());

    let mut additional_search_path_bufs: Vec<std::path::PathBuf> = vec![];
    if let Some(dslx_path) = dslx_path {
        for path in dslx_path.split(';') {
            additional_search_path_bufs.push(std::path::Path::new(path).to_path_buf());
        }
    }

    // We need the `Path` view type instead of `PathBuf`.
    let additional_search_path_views: Vec<&std::path::Path> = additional_search_path_bufs
        .iter()
        .map(|p| p.as_path())
        .collect::<Vec<_>>();

    let mut import_data =
        xlsynth::dslx::ImportData::new(dslx_stdlib_path, &additional_search_path_views);
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
    codegen_flags: &CodegenFlags,
    delay_model: &str,
    keep_temps: bool,
    config: &Option<ToolchainConfig>,
) {
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let temp_dir = tempfile::TempDir::new().unwrap();

        let module_name = xlsynth::dslx_path_to_module_name(input_file).unwrap();

        let dslx_stdlib_path = config.as_ref().and_then(|c| c.dslx_stdlib_path.as_deref());
        let dslx_path = config.as_ref().and_then(|c| Some(c.dslx_path.join(":")));
        let dslx_path_ref = dslx_path.as_ref().map(|s| s.as_str());

        let unopt_ir = run_ir_converter_main(
            input_file,
            Some(top),
            dslx_stdlib_path,
            dslx_path_ref,
            tool_path,
        );
        let unopt_ir_path = temp_dir.path().join("unopt.ir");
        std::fs::write(&unopt_ir_path, unopt_ir).unwrap();

        let ir_top = xlsynth::mangle_dslx_name(module_name, top).unwrap();

        let opt_ir = run_opt_main(&unopt_ir_path, Some(&ir_top), tool_path);
        let opt_ir_path = temp_dir.path().join("opt.ir");
        std::fs::write(&opt_ir_path, opt_ir).unwrap();

        let sv = run_codegen_pipeline(
            &opt_ir_path,
            delay_model,
            pipeline_spec,
            codegen_flags,
            tool_path,
        );
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
        let dslx = std::fs::read_to_string(input_file).unwrap();

        let dslx_path = config.as_ref().and_then(|c| Some(&c.dslx_path));
        let dslx_path_vec = match dslx_path {
            Some(entries) => entries
                .iter()
                .map(|p| std::path::Path::new(p))
                .collect::<Vec<_>>(),
            None => vec![],
        };
        let dslx_stdlib_path = config.as_ref().and_then(|c| c.dslx_stdlib_path.as_deref());
        let convert_options = xlsynth::DslxConvertOptions {
            dslx_stdlib_path: dslx_stdlib_path.map(|p| std::path::Path::new(p)),
            additional_search_paths: dslx_path_vec,
        };
        let ir = xlsynth::convert_dslx_to_ir(&dslx, input_file, &convert_options)
            .expect("successful conversion");

        let opt_ir = xlsynth::optimize_ir(&ir, top).unwrap();

        let mut sched_opt_lines = vec![format!("delay_model: \"{}\"", delay_model)];
        match pipeline_spec {
            PipelineSpec::Stages(stages) => {
                sched_opt_lines.push(format!("pipeline_stages: {}", stages))
            }
            PipelineSpec::ClockPeriodPs(clock_period_ps) => {
                sched_opt_lines.push(format!("clock_period_ps: {}", clock_period_ps))
            }
        }
        let scheduling_options_flags_proto = sched_opt_lines.join("\n");
        let codegen_flags_proto = "register_merge_strategy: STRATEGY_IDENTITY_ONLY
generator: GENERATOR_KIND_PIPELINE";
        let codegen_result = xlsynth::schedule_and_codegen(
            &opt_ir,
            &scheduling_options_flags_proto,
            &codegen_flags_proto,
        )
        .unwrap();
        let sv = codegen_result.get_verilog_text().unwrap();
        println!("{}", sv);
    }
}

/// Runs the IR converter command line tool and returns the output.
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
        let dslx_contents = std::fs::read_to_string(input_file).expect("file read successful");
        let dslx_stdlib_path: Option<&std::path::Path> =
            dslx_stdlib_path.map(|s| std::path::Path::new(s));
        let additional_search_paths: Vec<&std::path::Path> = dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let output = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            input_file,
            &DslxConvertOptions {
                dslx_stdlib_path,
                additional_search_paths,
            },
        )
        .expect("successful conversion");
        println!("{}", output);
    }
}
