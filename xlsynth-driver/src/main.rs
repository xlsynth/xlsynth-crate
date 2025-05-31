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

mod common;
mod dslx2ir;
mod dslx2pipeline;
mod dslx2sv_types;
mod gv2ir;
mod ir2delayinfo;
mod ir2gates;
mod ir2opt;
mod ir2pipeline;
mod ir_equiv;
mod ir_ged;
mod lib2proto;
mod report_cli_error;
mod toolchain_config;
mod tools;

use crate::toolchain_config::ToolchainConfig;
use clap;
use clap::{Arg, ArgAction};
use report_cli_error::report_cli_error_and_exit;
use serde::Deserialize;

#[derive(Deserialize)]
struct XlsynthToolchain {
    toolchain: ToolchainConfig,
}

trait AppExt {
    fn add_delay_model_arg(self) -> Self;
    fn add_pipeline_args(self) -> Self;
    fn add_dslx_input_args(self, include_top: bool) -> Self;
    fn add_codegen_args(self) -> Self;
    fn add_bool_arg(self, long: &'static str, help: &'static str) -> Self;
    fn add_ir_top_arg(self, required: bool) -> Self;
    fn add_ir2g8r_flags(self) -> Self;
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

    fn add_dslx_input_args(self, include_top: bool) -> Self {
        let mut command = (self as clap::Command).arg(
            Arg::new("dslx_input_file")
                .long("dslx_input_file")
                .value_name("DSLX_INPUT_FILE")
                .help("The input DSLX file")
                .required(true)
                .action(ArgAction::Set),
        );
        if include_top {
            command = command.arg(
                Arg::new("dslx_top")
                    .long("dslx_top")
                    .value_name("DSLX_TOP")
                    .help("The top-level entry point")
                    .required(true),
            )
        }
        command = command
            .arg(
                Arg::new("dslx_path")
                    .long("dslx_path")
                    .value_name("DSLX_PATH_SEMI_SEPARATED")
                    .help("Semi-separated paths for DSLX")
                    .action(ArgAction::Set),
            )
            .arg(
                Arg::new("dslx_stdlib_path")
                    .long("dslx_stdlib_path")
                    .value_name("DSLX_STDLIB_PATH")
                    .help("Path to the DSLX standard library")
                    .action(ArgAction::Set),
            );
        command.add_bool_arg("warnings_as_errors", "Treat warnings as errors")
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

    /// Adds a boolean argument to the command -- the helper ensures we have a
    /// uniform uniform style/handling for boolean arguments.
    fn add_bool_arg(self, long: &'static str, help: &'static str) -> Self {
        (self as clap::Command).arg(
            Arg::new(long)
                .long(long)
                .value_name("BOOL")
                .action(ArgAction::Set)
                .value_parser(["true", "false"])
                .num_args(1)
                .help(help),
        )
    }

    fn add_codegen_args(self) -> Self {
        let result = (self as clap::Command)
            .arg(
                Arg::new("module_name")
                    .long("module_name")
                    .value_name("MODULE_NAME")
                    .help("Name of the generated module"),
            )
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
            );
        result
            .add_bool_arg(
                "flop_inputs",
                "Whether to flop input ports (vs leaving combinational delay into the I/Os)",
            )
            .add_bool_arg(
                "flop_outputs",
                "Whether to flop output ports (vs leaving combinational delay into the I/Os)",
            )
            .add_bool_arg("add_idle_output", "Add an idle output port")
            .add_bool_arg("array_index_bounds_checking", "Array index bounds checking")
            .add_bool_arg("separate_lines", "Separate lines in generated code")
            .add_bool_arg(
                "use_system_verilog",
                "Whether to emit System Verilog instead of Verilog",
            )
            .arg(
                Arg::new("reset")
                    .long("reset")
                    .value_name("RESET")
                    .help("Reset signal name"),
            )
            .add_bool_arg("reset_asynchronous", "Reset is asynchronous")
            .add_bool_arg("reset_active_low", "Reset is active low")
            .add_bool_arg(
                "reset_data_path",
                "Reset datapath registers as well as valid signals",
            )
    }

    fn add_ir_top_arg(self, required: bool) -> Self {
        (self as clap::Command).arg(
            Arg::new("ir_top")
                .long("top")
                .value_name("TOP")
                .help("The top-level entry point to use for the IR")
                .required(required)
                .action(ArgAction::Set),
        )
    }

    fn add_ir2g8r_flags(self) -> Self {
        (self as clap::Command)
            .add_bool_arg("fold", "Fold the gate representation")
            .add_bool_arg("hash", "Hash the gate representation")
            .add_bool_arg("fraig", "Run fraig optimization")
            .arg(
                clap::Arg::new("fraig_max_iterations")
                    .long("fraig-max-iterations")
                    .value_name("N")
                    .help("Maximum number of iterations for fraig optimization")
                    .action(clap::ArgAction::Set),
            )
            .arg(
                clap::Arg::new("fraig_sim_samples")
                    .long("fraig-sim-samples")
                    .value_name("N")
                    .help("Number of samples to use for fraig optimization")
                    .action(clap::ArgAction::Set),
            )
            .add_bool_arg(
                "compute_graph_logical_effort",
                "Compute the graph logical effort worst case delay",
            )
            .arg(
                clap::Arg::new("graph_logical_effort_beta1")
                    .long("graph-logical-effort-beta1")
                    .value_name("BETA1")
                    .help("Beta1 value for graph logical effort computation (default 1.0)")
                    .default_value("1.0")
                    .action(clap::ArgAction::Set),
            )
            .arg(
                clap::Arg::new("graph_logical_effort_beta2")
                    .long("graph-logical-effort-beta2")
                    .value_name("BETA2")
                    .help("Beta2 value for graph logical effort computation (default 0.0)")
                    .default_value("0.0")
                    .action(clap::ArgAction::Set),
            )
            .arg(
                Arg::new("toggle_sample_count")
                    .long("toggle-sample-count")
                    .value_name("N")
                    .help("If > 0, generate N random input samples and print toggle stats.")
                    .default_value("0")
                    .action(ArgAction::Set),
            )
            .arg(
                Arg::new("toggle_sample_seed")
                    .long("toggle-seed")
                    .value_name("SEED")
                    .help("Seed for random toggle stimulus (default 0)")
                    .default_value("0")
                    .action(ArgAction::Set),
            )
    }
}

fn main() {
    let _ = env_logger::try_init();

    log::info!(
        "xlsynth-driver starting; version: {}",
        env!("CARGO_PKG_VERSION")
    );

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
                .add_delay_model_arg()
                .add_dslx_input_args(true)
                .add_pipeline_args()
                .add_codegen_args()
                .add_bool_arg("keep_temps", "Keep temporary files"),
        )
        .subcommand(
            clap::Command::new("dslx2ir")
                .about("Converts DSLX to IR")
                .add_dslx_input_args(true)
                .add_bool_arg("opt", "Optimize the IR we emit as well"),
        )
        // dslx2sv-types converts all the definitions in the .x file to SV types
        .subcommand(
            clap::Command::new("dslx2sv-types")
                .about("Converts DSLX type definitions to SystemVerilog")
                .add_dslx_input_args(false),
        )
        // ir2opt subcommand requires a top symbol
        .subcommand(
            clap::Command::new("ir2opt")
                .about("Converts IR to optimized IR")
                .arg(
                    Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                // Top is given as a (non-positional) flag for symmetry but it is required.
                .arg(
                    Arg::new("ir_top")
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
                    Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .add_delay_model_arg()
                .add_ir_top_arg(false)
                .add_codegen_args()
                .add_pipeline_args()
                .add_bool_arg("opt", "Optimize the IR before scheduling pipeline")
                .add_bool_arg("keep_temps", "Keep temporary files"),
        )
        .subcommand(
            clap::Command::new("ir2delayinfo")
                .about("Converts IR entry point to delay info output")
                .add_delay_model_arg()
                .arg(
                    Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("ir_top")
                        .help("The top-level entry point")
                        .required(true)
                        .index(2),
                ),
        )
        .subcommand(
            clap::Command::new("ir-equiv")
                .about("Checks if two IRs are equivalent")
                .arg(
                    Arg::new("lhs_ir_file")
                        .help("The left-hand side IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("rhs_ir_file")
                        .help("The right-hand side IR file")
                        .required(true)
                        .index(2),
                )
                .add_ir_top_arg(false)
                .arg(
                    Arg::new("lhs_ir_top")
                        .long("lhs_ir_top")
                        .help("The top-level entry point for the left-hand side IR"),
                )
                .arg(
                    Arg::new("rhs_ir_top")
                        .long("rhs_ir_top")
                        .help("The top-level entry point for the right-hand side IR"),
                )
                .add_bool_arg(
                    "boolector",
                    "Use Boolector for equivalence checking (requires --features=with-boolector-built or --features=with-boolector-system)",
                )
                .add_bool_arg(
                    "flatten_aggregates",
                    "Flatten tuple and array types to bits for equivalence checking",
                )
                .arg(
                    Arg::new("drop_params")
                        .long("drop_params")
                        .help("Comma-separated list of parameter names to drop from both functions before equivalence checking")
                        .action(ArgAction::Set),
                ),
        )
        .subcommand(
            clap::Command::new("ir-ged")
                .about("Tells the Graph Edit Distance between two IR functions")
                .arg(
                    Arg::new("lhs_ir_file")
                        .help("The left-hand side IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("rhs_ir_file")
                        .help("The right-hand side IR file")
                        .required(true)
                        .index(2),
                )
                .arg(
                    Arg::new("lhs_ir_top")
                        .long("lhs_ir_top")
                        .help("The top-level entry point for the left-hand side IR"),
                )
                .arg(
                    Arg::new("rhs_ir_top")
                        .long("rhs_ir_top")
                        .help("The top-level entry point for the right-hand side IR"),
                )
                .add_bool_arg("json", "Output in JSON format"),
        )
        .subcommand(
            clap::Command::new("ir2gates")
                .about("Converts IR to gates")
                .arg(
                    Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .add_bool_arg("quiet", "Quiet mode")
                .add_ir2g8r_flags(),
        )
        .subcommand(
            clap::Command::new("ir2g8r")
                .about("Converts IR to GateFn and emits it to stdout; optionally writes .g8rbin, netlist, and stats")
                .arg(
                    clap::Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .add_ir2g8r_flags()
                .arg(
                    clap::Arg::new("bin_out")
                        .long("bin-out")
                        .value_name("PATH")
                        .help("Path to write the .g8rbin file")
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("stats_out")
                        .long("stats-out")
                        .value_name("PATH")
                        .help("Path to write the JSON summary statistics")
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("netlist_out")
                        .long("netlist-out")
                        .value_name("PATH")
                        .help("Path to write the gate-level netlist (human-readable)")
                        .action(clap::ArgAction::Set),
                )
        )
        .subcommand(
            clap::Command::new("lib2proto")
                .about("Converts Liberty file(s) to proto or textproto")
                .arg(
                    Arg::new("liberty_files")
                        .help("Liberty file(s)")
                        .required(true)
                        .num_args(1..)
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .help("Output file (.proto or .textproto)")
                        .required(true)
                        .action(ArgAction::Set),
                ),
        )
        .subcommand(
            clap::Command::new("gv2ir")
                .about("Converts a gate-level netlist and Liberty proto to XLS IR")
                .arg(
                    Arg::new("netlist")
                        .long("netlist")
                        .help("Input gate-level netlist (.gv)")
                        .required(true)
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("liberty_proto")
                        .long("liberty_proto")
                        .help("Input Liberty proto (.proto or .textproto)")
                        .required(true)
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("dff_cells")
                        .long("dff_cells")
                        .help("Comma-separated list of DFF cell names to treat as identity (D->Q)")
                        .action(ArgAction::Set),
                )
        )
        .subcommand(
            clap::Command::new("g8r2v")
                .about("Converts a .g8r or .g8rbin file to a .ugv netlist on stdout, optionally adding a clock port as the first input.")
                .arg(
                    clap::Arg::new("g8r_input_file")
                        .help("The input .g8r or .g8rbin file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    clap::Arg::new("add_clk_port")
                        .long("add-clk-port")
                        .value_name("NAME")
                        .help("Insert an (unused) clock port with this name as the first input; if given with no value, defaults to 'clk'")
                        .num_args(0..=1)
                        .require_equals(true),
                )
        )
        .get_matches();

    let mut toml_path: Option<String> = matches
        .get_one::<String>("toolchain")
        .map(|s| s.to_string());

    // If there is no toolchain flag specified, but there is a
    // xlsynth-toolchain.toml in the current directory, use that.
    if toml_path.is_none() {
        let cwd = std::env::current_dir().unwrap();
        let cwd_toml_path = cwd.join("xlsynth-toolchain.toml");
        if cwd_toml_path.exists() {
            log::info!(
                "Using xlsynth-toolchain.toml in current directory: {}",
                cwd_toml_path.display()
            );
            toml_path = Some(cwd_toml_path.to_str().unwrap().to_string());
        }
    }

    let toml_value: Option<toml::Value> = toml_path.map(|path| {
        // If we got a toolchain toml file, read/parse it.
        if !std::path::Path::new(&path).exists() {
            report_cli_error_and_exit(
                "toolchain toml file does not exist",
                None,
                vec![
                    ("path", &path),
                    (
                        "working directory",
                        &std::env::current_dir().unwrap().display().to_string(),
                    ),
                ],
            );
        }
        let toml_str =
            std::fs::read_to_string(path).expect("read toolchain toml file should succeed");
        toml::from_str(&toml_str).expect("parse toolchain toml file should succeed")
    });
    let config = toml_value.map(|v| {
        let toolchain_config = v.clone().try_into::<XlsynthToolchain>().expect(&format!(
            "parse toolchain config should succeed; value: {}",
            v
        ));
        toolchain_config.toolchain
    });

    if let Some(matches) = matches.subcommand_matches("dslx2pipeline") {
        dslx2pipeline::handle_dslx2pipeline(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx2ir") {
        dslx2ir::handle_dslx2ir(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2opt") {
        ir2opt::handle_ir2opt(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2pipeline") {
        ir2pipeline::handle_ir2pipeline(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx2sv-types") {
        dslx2sv_types::handle_dslx2sv_types(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2delayinfo") {
        ir2delayinfo::handle_ir2delayinfo(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-equiv") {
        ir_equiv::handle_ir_equiv(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-ged") {
        ir_ged::handle_ir_ged(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2gates") {
        ir2gates::handle_ir2gates(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2g8r") {
        ir2gates::handle_ir2g8r(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("lib2proto") {
        lib2proto::handle_lib2proto(matches);
    } else if let Some(matches) = matches.subcommand_matches("gv2ir") {
        gv2ir::handle_gv2ir(matches);
    } else if let Some(matches) = matches.subcommand_matches("g8r2v") {
        // TODO: implement g8r2v handler
        unimplemented!("g8r2v handler not yet implemented");
    } else if let Some(_matches) = matches.subcommand_matches("version") {
        println!("{}", env!("CARGO_PKG_VERSION"));
    } else {
        report_cli_error_and_exit("No valid subcommand provided.", None, vec![]);
    }
}
