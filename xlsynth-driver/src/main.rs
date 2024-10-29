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

    fn add_dslx_path_arg(self) -> Self {
        (self as App).arg(
            Arg::with_name("dslx_path")
                .long("dslx_path")
                .value_name("DSLX_PATH_SEMI_SEPARATED")
                .help("Semi-separated paths for DSLX")
                .takes_value(true),
        )
    }

    fn add_dslx_stdlib_path_arg(self) -> Self {
        (self as App).arg(
            Arg::with_name("dslx_stdlib_path")
                .long("dslx_stdlib_path")
                .value_name("DSLX_STDLIB_PATH")
                .help("Path to the DSLX standard library")
                .takes_value(true),
        )
    }
}

fn main() {
    let matches = App::new("xlsynth-driver")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Command line driver for XLS/xlsynth capabilities")
        .arg(
            Arg::with_name("toolchain")
                .long("toolchain")
                .value_name("TOOLCHAIN")
                .help("Path to a xlsynth-toolchain.toml file")
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
                .add_dslx_stdlib_path_arg()
                .add_dslx_path_arg()
                // --keep_temps flag to keep temporary files
                .arg(
                    Arg::with_name("keep_temps")
                        .long("keep_temps")
                        .help("Keep temporary files")
                        .takes_value(false),
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
                .add_dslx_stdlib_path_arg()
                .add_dslx_path_arg(),
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
                )
                .add_dslx_stdlib_path_arg()
                .add_dslx_path_arg(),
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

    let toml_path = matches.value_of("toolchain");
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
        process::exit(1)
    }
}

fn handle_ir2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.value_of("DELAY_MODEL").unwrap();

    // See which of pipeline_stages or clock_period_ps we're using.
    let pipeline_spec = extract_pipeline_spec(matches);

    ir2pipeline(input_path, delay_model, &pipeline_spec, config);
}

fn handle_dslx2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
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
        config,
    );
}

/// Helper for extracting the DSLX standard library path from the command line
/// flag, if specified, or the toolchain config if it's present and the cmdline
/// flag isn't specified.
fn get_dslx_stdlib_path(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Option<String> {
    let dslx_stdlib_path = matches.value_of("dslx_stdlib_path");
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
    let dslx_path = matches.value_of("dslx_path");
    if let Some(dslx_path) = dslx_path {
        Some(dslx_path.to_string())
    } else if let Some(config) = config {
        Some(config.dslx_path.join(";"))
    } else {
        None
    }
}

fn handle_dslx2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = matches.value_of("TOP");
    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path = dslx_stdlib_path.as_deref();

    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    // Stub function for DSLX to IR conversion
    dslx2ir(input_path, top, dslx_stdlib_path, dslx_path, tool_path);
}

fn handle_ir2opt(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
    let top = matches.value_of("TOP").unwrap();
    let input_path = std::path::Path::new(input_file);

    ir2opt(input_path, top, config);
}

fn handle_dslx2sv_types(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.value_of("INPUT_FILE").unwrap();
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
    config: &Option<ToolchainConfig>,
) {
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_codegen_pipeline(input_file, delay_model, pipeline_spec, tool_path);
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

fn run_dslx2pipeline_via_tools(
    input_file: &std::path::Path,
    top: &str,
    pipeline_spec: &PipelineSpec,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    delay_model: &str,
    keep_temps: bool,
    tool_path: &str,
) {
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
}

fn run_dslx2pipeline_via_apis(
    _input_file: &std::path::Path,
    _top: &str,
    _pipeline_spec: &PipelineSpec,
    _dslx_stdlib_path: Option<&str>,
    _dslx_path: Option<&str>,
    _delay_model: &str,
    _keep_temps: bool,
    _config: &Option<ToolchainConfig>,
) {
    // First we convert to IR.
    // Then we optimize IR.
    // Then we schedule-and-codegen.
    todo!("dslx2pipeline subcommand using runtime APIs")
}

fn dslx2pipeline(
    input_file: &std::path::Path,
    top: &str,
    pipeline_spec: &PipelineSpec,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    delay_model: &str,
    keep_temps: bool,
    config: &Option<ToolchainConfig>,
) {
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        run_dslx2pipeline_via_tools(
            input_file,
            top,
            pipeline_spec,
            dslx_stdlib_path,
            dslx_path,
            delay_model,
            keep_temps,
            tool_path,
        );
    } else {
        run_dslx2pipeline_via_apis(
            input_file,
            top,
            pipeline_spec,
            dslx_stdlib_path,
            dslx_path,
            delay_model,
            keep_temps,
            config,
        );
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
        let output = xlsynth::xls_convert_dslx_to_ir(
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
