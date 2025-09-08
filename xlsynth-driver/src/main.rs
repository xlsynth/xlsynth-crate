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
mod dslx_equiv;
mod dslx_g8r_stats;
mod dslx_show;
mod dslx_stitch_pipeline;
mod flag_defaults;
mod g8r2v;
mod g8r_equiv;
mod gv2ir;
mod gv_read_stats;
mod ir2combo;
mod ir2delayinfo;
mod ir2gates;
mod ir2opt;
mod ir2pipeline;
mod ir_equiv;
mod ir_equiv_blocks;
mod ir_fn_eval;
mod ir_fn_to_block;
mod ir_ged;
mod ir_localized_eco;
mod ir_round_trip;
mod ir_strip_pos_data;
mod ir_structural_similarity;
mod lib2proto;
mod parallelism;
mod prove_quickcheck;
mod prover;
mod prover_config;
mod report_cli_error;
mod run_verilog_pipeline;
mod solver_choice;
mod toolchain_config;
mod tools;

use crate::toolchain_config::ToolchainConfig;
use clap;
use clap::{Arg, ArgAction};
use once_cell::sync::Lazy;
use report_cli_error::report_cli_error_and_exit;
use serde::Deserialize;
use xlsynth_g8r::equiv::prove_equiv::AssertionSemantics;
use xlsynth_g8r::equiv::prove_quickcheck::QuickCheckAssertionSemantics;

static DEFAULT_ADDER_MAPPING: Lazy<String> =
    Lazy::new(|| xlsynth_g8r::ir2gate_utils::AdderMapping::default().to_string());

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

// TODO: Change flags from using strings to using clap::ValueEnum.
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
            .add_bool_arg(
                "add_invariant_assertions",
                "Add assertions for invariants in generated code",
            )
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
            .arg(
                Arg::new("output_schedule_path")
                    .long("output_schedule_path")
                    .value_name("OUTPUT_SCHEDULE_PATH")
                    .help("Write schedule proto text to this path"),
            )
            .arg(
                Arg::new("output_verilog_line_map_path")
                    .long("output_verilog_line_map_path")
                    .value_name("OUTPUT_VERILOG_LINE_MAP_PATH")
                    .help("Write Verilog line map textproto to this path"),
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
            .arg(
                clap::Arg::new("adder_mapping")
                    .long("adder-mapping")
                    .value_name("ADDER_MAPPING")
                    .help("The adder mapping strategy to use (default: brent-kung).")
                    .value_parser(["ripple-carry", "brent-kung", "kogge-stone"])
                    .default_value(DEFAULT_ADDER_MAPPING.as_str())
                    .action(clap::ArgAction::Set),
            )
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
                .arg(
                    clap::Arg::new("output_unopt_ir")
                        .long("output_unopt_ir")
                        .value_name("PATH")
                        .help("Path to write the unoptimized IR (package) output")
                        .required(false)
                        .action(ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("output_opt_ir")
                        .long("output_opt_ir")
                        .value_name("PATH")
                        .help("Path to write the optimized IR (package) output")
                        .required(false)
                        .action(ArgAction::Set),
                )
                .add_delay_model_arg()
                .add_dslx_input_args(true)
                .add_pipeline_args()
                .add_codegen_args()
                .add_bool_arg("keep_temps", "Keep temporary files")
                .add_bool_arg(
                    "type_inference_v2",
                    "Enable the experimental type-inference v2 algorithm",
                ),
        )
        .subcommand(
            clap::Command::new("dslx-stitch-pipeline")
                .about("Stitches DSLX pipeline stages")
                .add_dslx_input_args(true)
                .add_bool_arg(
                    "use_system_verilog",
                    "Whether to emit SystemVerilog (default true; set to false for plain Verilog)",
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
                )
                .arg(
                    Arg::new("reset")
                        .long("reset")
                        .value_name("RESET")
                        .help("Reset signal name"),
                )
                .add_bool_arg(
                    "reset_active_low",
                    "Reset is active low",
                )
                .arg(
                    Arg::new("stages")
                        .long("stages")
                        .value_name("CSV")
                        .help("Comma-separated explicit stage names in order (overrides automatic _cycle indexing)")
                        .action(ArgAction::Set),
                )
                .add_bool_arg(
                    "flop_inputs",
                    "Whether to insert input pipeline flops (default true)",
                )
                .add_bool_arg(
                    "flop_outputs",
                    "Whether to insert output pipeline flops (default true)",
                )
                .add_bool_arg(
                    "array_index_bounds_checking",
                    "Whether to emit array index bounds checking",
                ),
        )
        .subcommand(
            clap::Command::new("dslx2ir")
                .about("Converts DSLX to IR")
                .add_dslx_input_args(false)
                .arg(
                    Arg::new("dslx_top")
                        .long("dslx_top")
                        .value_name("DSLX_TOP")
                        .help("The top-level entry point")
                        .required_if_eq("opt", "true"),
                )
                .arg(
                    Arg::new("opt")
                        .long("opt")
                        .value_name("BOOL")
                        .action(ArgAction::Set)
                        .value_parser(["true", "false"])
                        .num_args(1)
                        .help("Optimize the IR we emit as well")
                )
                .add_bool_arg(
                    "type_inference_v2",
                    "Enable the experimental type-inference v2 algorithm",
                )
                .add_bool_arg(
                    "convert_tests",
                    "Convert test procs/functions to IR",
                )
        )
        // dslx2sv-types converts all the definitions in the .x file to SV types
        .subcommand(
            clap::Command::new("dslx-show")
                .about("Resolve and print a DSLX symbol definition (enums/structs/type aliases/constants/functions/quickchecks)")
                .arg(
                    clap::Arg::new("dslx_input_file")
                        .long("dslx_input_file")
                        .value_name("DSLX_INPUT_FILE")
                        .help("Optional input DSLX file - if omitted, symbol must be qualified like 'path.with.dots::Name'")
                        .required(false)
                        .action(ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("dslx_path")
                        .long("dslx_path")
                        .value_name("DSLX_PATH_SEMI_SEPARATED")
                        .help("Semi-separated paths for DSLX lookup (used for imported/library symbols)")
                        .action(ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("dslx_stdlib_path")
                        .long("dslx_stdlib_path")
                        .value_name("DSLX_STDLIB_PATH")
                        .help("Path to the DSLX standard library")
                        .action(ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("symbol")
                        .value_name("SYMBOL")
                        .help("Symbol to show; supports dotted module path + member like 'foo.bar.baz::Name', or just 'Name' with --dslx_input_file")
                        .required(true)
                        .index(1)
                        .action(ArgAction::Set),
                ),
        )
        .subcommand(
            clap::Command::new("dslx2sv-types")
                .about("Converts DSLX type definitions to SystemVerilog")
                .add_dslx_input_args(false),
        )
        .subcommand(
            clap::Command::new("dslx-g8r-stats")
                .about("Emit gate-level summary stats for a DSLX entry point")
                .add_dslx_input_args(true)
                .add_bool_arg(
                    "type_inference_v2",
                    "Enable the experimental type-inference v2 algorithm",
                ),
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
            clap::Command::new("ir2combo")
                .about("Converts IR to combinational SystemVerilog")
                .arg(
                    Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .add_delay_model_arg()
                .add_ir_top_arg(false)
                .add_codegen_args()
                .add_bool_arg("opt", "Optimize the IR before codegen")
                .add_bool_arg("keep_temps", "Keep temporary files"),
        )
        .subcommand(
            clap::Command::new("ir-fn-to-block")
                .about("Converts an IR function to Block IR (requires external toolchain)")
                .arg(
                    clap::Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .add_ir_top_arg(true),
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
                .arg(
                    Arg::new("solver")
                        .long("solver")
                        .value_name("SOLVER")
                        .help("Use the specified solver for equivalence checking (requires --features=with-easy-smt)")
                        .value_parser([
                            #[cfg(feature = "has-easy-smt")]
                            "z3-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "bitwuzla-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "boolector-binary",
                            #[cfg(feature = "has-bitwuzla")]
                            "bitwuzla",
                            #[cfg(feature = "has-boolector")]
                            "boolector",
                            "toolchain",
                        ])
                        .default_value("toolchain")
                        .action(ArgAction::Set),
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
                )
                .arg(
                    Arg::new("parallelism_strategy")
                        .long("parallelism-strategy")
                        .value_name("STRATEGY")
                        .help("Parallelism strategy")
                        .value_parser(["single-threaded", "output-bits", "input-bit-split"])
                        .default_value("single-threaded")
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("assertion_semantics")
                        .long("assertion-semantics")
                        .value_name("SEMANTICS")
                        .help("Assertion semantics")
                        .value_parser(clap::value_parser!(AssertionSemantics))
                        .default_value("same")
                        .action(ArgAction::Set),
                )
                .add_bool_arg(
                    "lhs_fixed_implicit_activation",
                    "Fix the implicit activation bit to true for the LHS IR, useful when only LHS or RHS has implicit token",
                )
                .add_bool_arg(
                    "rhs_fixed_implicit_activation",
                    "Fix the implicit activation bit to true for the RHS IR, useful when only LHS or RHS has implicit token",
                )
                .arg(
                    clap::Arg::new("output_json")
                        .long("output_json")
                        .value_name("PATH")
                        .help("Write the JSON result to PATH")
                        .action(clap::ArgAction::Set),
                ),
        )
        .subcommand(
            clap::Command::new("ir-equiv-blocks")
                .about("Checks if two IR blocks are equivalent")
                .arg(
                    Arg::new("lhs_ir_file")
                        .help("The left-hand side IR block file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("rhs_ir_file")
                        .help("The right-hand side IR block file")
                        .required(true)
                        .index(2),
                )
                .arg(
                    Arg::new("lhs_top")
                        .long("lhs_top")
                        .help("Top-level block name for the left-hand side IR"),
                )
                .arg(
                    Arg::new("rhs_top")
                        .long("rhs_top")
                        .help("Top-level block name for the right-hand side IR"),
                )
                .arg(
                    Arg::new("top")
                        .long("top")
                        .help("Top-level block name for both IRs"),
                )
                .arg(
                    Arg::new("solver")
                        .long("solver")
                        .value_name("SOLVER")
                        .help("Use the specified solver for equivalence checking (requires --features=with-easy-smt)")
                        .value_parser([
                            #[cfg(feature = "has-easy-smt")]
                            "z3-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "bitwuzla-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "boolector-binary",
                            #[cfg(feature = "has-bitwuzla")]
                            "bitwuzla",
                            #[cfg(feature = "has-boolector")]
                            "boolector",
                            "toolchain",
                        ])
                        .default_value("toolchain")
                        .action(ArgAction::Set),
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
                )
                .arg(
                    Arg::new("parallelism_strategy")
                        .long("parallelism-strategy")
                        .value_name("STRATEGY")
                        .help("Parallelism strategy")
                        .value_parser(["single-threaded", "output-bits", "input-bit-split"])
                        .default_value("single-threaded")
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("assertion_semantics")
                        .long("assertion-semantics")
                        .value_name("SEMANTICS")
                        .help("Assertion semantics")
                        .value_parser(clap::value_parser!(AssertionSemantics))
                        .default_value("same")
                        .action(ArgAction::Set),
                )
                .add_bool_arg(
                    "lhs_fixed_implicit_activation",
                    "Fix the implicit activation bit to true for the LHS IR, useful when only LHS or RHS has implicit token",
                )
                .add_bool_arg(
                    "rhs_fixed_implicit_activation",
                    "Fix the implicit activation bit to true for the RHS IR, useful when only LHS or RHS has implicit token",
                )
                .arg(
                    clap::Arg::new("output_json")
                        .long("output_json")
                        .value_name("PATH")
                        .help("Write the JSON result to PATH")
                        .action(clap::ArgAction::Set),
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
            clap::Command::new("ir-structural-similarity")
                .about("Computes a depth-to-discrepancy histogram between two IR functions")
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
                .arg(
                    Arg::new("output_dir")
                        .long("output-dir")
                        .value_name("DIR")
                        .help("Directory to write outputs (original IR copies). If omitted, a temp directory is created and printed."),
                )
                .add_bool_arg(
                    "show_discrepancies",
                    "Show per-depth discrepancy signatures in verbose form",
                ),
        )
        .subcommand(
            clap::Command::new("ir-localized-eco")
                .about("Computes a localized ECO diff (old → new) and emits JSON edits and a summary")
                .arg(
                    Arg::new("old_ir_file")
                        .help("The old/original IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("new_ir_file")
                        .help("The new/target IR file")
                        .required(true)
                        .index(2),
                )
                .arg(
                    Arg::new("old_ir_top")
                        .long("old_ir_top")
                        .help("Top-level entry point for the old IR"),
                )
                .arg(
                    Arg::new("new_ir_top")
                        .long("new_ir_top")
                        .help("Top-level entry point for the new IR"),
                )
                .arg(
                    Arg::new("json_out")
                        .long("json_out")
                        .value_name("PATH")
                        .help("Write the JSON report to PATH; if omitted a temp file is created and its path is printed"),
                )
                .arg(
                    Arg::new("output_dir")
                        .long("output_dir")
                        .value_name("DIR")
                        .help("Directory to write outputs (JSON, patched .ir). If omitted, a temp directory is created and printed."),
                )
                .arg(
                    Arg::new("compute_text_diff")
                        .long("compute-text-diff")
                        .value_name("BOOL")
                        .help("Compute IR/RTL text diffs (expensive)")
                        .value_parser(["true", "false"]).num_args(1)
                        .default_value("false")
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("sanity_samples")
                        .long("sanity-samples")
                        .value_name("N")
                        .help("If > 0, run N randomized interpreter samples (in addition to all-zeros and all-ones) to sanity-check patched(old) vs new.")
                        .default_value("0"),
                )
                .arg(
                    Arg::new("sanity_seed")
                        .long("sanity-seed")
                        .value_name("SEED")
                        .help("Seed for randomized interpreter samples (default 0)")
                        .default_value("0"),
                ),
        )
        .subcommand(
            clap::Command::new("ir-round-trip")
                .about("Parses an IR file and writes it back out to stdout")
                .arg(
                    clap::Arg::new("ir_input_file")
                        .help("The input IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    clap::Arg::new("strip_pos_attrs")
                        .long("strip-pos-attrs")
                        .value_name("BOOL")
                        .action(ArgAction::Set)
                        .value_parser(["true", "false"])
                        .num_args(1)
                        .help("If true, strip file_number and (future) node pos attributes from output"),
                ),
        )
        .subcommand(
            clap::Command::new("ir2gates")
                .about("Converts IR to GateFn and emits it to stdout as JSON")
                .arg(
                    clap::Arg::new("ir_input_file")
                        .value_name("IR_INPUT_FILE")
                        .help("The input IR file")
                        .required(true)
                        .action(ArgAction::Set),
                )
                .add_bool_arg("quiet", "Quiet mode")
                .arg(
                    clap::Arg::new("output_json")
                        .long("output_json")
                        .value_name("PATH")
                        .help("Write the JSON summary to PATH")
                        .action(clap::ArgAction::Set),
                )
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
                .arg(
                    Arg::new("dff_cell_formula")
                        .long("dff_cell_formula")
                        .help("If set, any cell with an output pin function exactly matching this formula string is treated as a DFF for D->Q identity override.")
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("dff_cell_invert_formula")
                        .long("dff_cell_invert_formula")
                        .help("If set, any cell with an output pin function exactly matching this formula string is treated as a DFF with inverted output (QN = NOT(D)).")
                        .action(ArgAction::Set),
                )
        )
        .subcommand(
            clap::Command::new("gv-read-stats")
                .about("Reads a gate-level netlist and prints summary statistics")
                .arg(
                    clap::Arg::new("netlist")
                        .help("Input gate-level netlist (.gv or .gv.gz)")
                        .required(true)
                        .index(1),
                ),
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
                    clap::Arg::new("add-clk-port")
                        .long("add-clk-port")
                        .value_name("NAME")
                        .help("Name for the clock port. Mandatory if --flop-inputs or --flop-outputs is used. If specified without flopping, adds a clock port with this name.")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("flop-inputs")
                        .long("flop-inputs")
                        .help("Add a layer of flops for all inputs.")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    clap::Arg::new("flop-outputs")
                        .long("flop-outputs")
                        .help("Add a layer of flops for all outputs.")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    clap::Arg::new("use-system-verilog")
                        .long("use-system-verilog")
                        .help("Emit SystemVerilog instead of Verilog.")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    clap::Arg::new("module-name")
                        .long("module-name")
                        .value_name("MODULE_NAME")
                        .help("Name of the generated module"),
                )
        )
        .subcommand(
            clap::Command::new("g8r-equiv")
                .about("Checks if two GateFns are equivalent using available engines")
                .arg(
                    clap::Arg::new("lhs_g8r_file")
                        .help("The left-hand side GateFn file (.g8r or .g8rbin)")
                        .required(true)
                        .index(1),
                )
                .arg(
                    clap::Arg::new("rhs_g8r_file")
                        .help("The right-hand side GateFn file (.g8r or .g8rbin)")
                        .required(true)
                        .index(2),
                )
        )
        .subcommand(
            clap::Command::new("ir-fn-eval")
                .about("Interprets an IR function with the provided argument tuple")
                .arg(
                    clap::Arg::new("ir_file")
                        .help("Path to the IR file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    clap::Arg::new("entry_fn")
                        .help("Name of the function to invoke")
                        .required(true)
                        .index(2),
                )
                .arg(
                    clap::Arg::new("arg_tuple")
                        .help("Tuple of typed IR values for the function arguments")
                        .required(true)
                        .index(3),
                )
        )
        .subcommand(
            clap::Command::new("ir-strip-pos-data")
                .about("Reads an .ir file and emits the same IR with all position data removed (file table and pos= attributes)")
                .arg(
                    clap::Arg::new("ir_file")
                        .help("Path to the IR file")
                        .required(true)
                        .index(1),
                ),
        )
        .subcommand(
            clap::Command::new("run-verilog-pipeline")
                .about("Runs a SystemVerilog pipeline via iverilog with a single input value")
                .long_about("Runs a SystemVerilog pipeline simulation using iverilog.\n\nUsage: xlsynth-driver run-verilog-pipeline <SV_PATH> [INPUT_VALUE]\n  SV_PATH: Path to SystemVerilog file (or '-' for stdin)\n  INPUT_VALUE: XLS IR typed value (e.g., 'bits[32]:5', 'tuple(bits[8]:1, bits[16]:2)')\n               If not provided, zero values will be used and displayed.")
                .arg(
                    Arg::new("input_valid_signal")
                        .long("input_valid_signal")
                        .value_name("NAME")
                        .help("Input-valid signal name"),
                )
                .arg(
                    Arg::new("output_valid_signal")
                        .long("output_valid_signal")
                        .value_name("NAME")
                        .help("Output-valid signal name"),
                )
                .arg(
                    Arg::new("reset")
                        .long("reset")
                        .value_name("NAME")
                        .help("Reset signal name"),
                )
                .add_bool_arg("reset_active_low", "Reset is active low")
                .arg(
                    Arg::new("clk")
                        .long("clk")
                        .value_name("NAME")
                        .help("Clock signal name for the DUT (default 'clk')"),
                )
                .arg(
                    Arg::new("latency")
                        .long("latency")
                        .value_name("CYCLES")
                        .help("Latency in cycles (required if output_valid_signal is not provided)"),
                )
                .arg(
                    Arg::new("waves")
                        .long("waves")
                        .value_name("PATH")
                        .help("Write VCD dump to PATH"),
                )
                .arg(
                    Arg::new("sv_path")
                        .help("Path to SystemVerilog pipeline source (use '-' for stdin)")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("input_value")
                        .help("XLS IR typed value used as input (e.g., 'bits[32]:5', 'tuple(bits[8]:1, bits[16]:2)'). If not provided, zero values will be used.")
                        .required(false)
                        .index(2),
                ),
        )
        .subcommand(
            clap::Command::new("prove-quickcheck")
                .about("Prove that DSLX #[quickcheck] functions always return true")
                .add_dslx_input_args(false)
                .arg(
                    clap::Arg::new("test_filter")
                        .long("test_filter")
                        .value_name("FILTER")
                        .help("Regular expression; prove only quickcheck functions whose name fully matches the pattern"),
                )
                .arg(
                    clap::Arg::new("solver")
                        .long("solver")
                        .value_name("SOLVER")
                        .help("Select solver backend")
                        .value_parser([
                            #[cfg(feature = "has-easy-smt")]
                            "z3-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "bitwuzla-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "boolector-binary",
                            #[cfg(feature = "has-bitwuzla")]
                            "bitwuzla",
                            #[cfg(feature = "has-boolector")]
                            "boolector",
                            "toolchain",
                        ])
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("assertion_semantics")
                        .long("assertion-semantics")
                        .value_name("SEM")
                        .help("Assertion semantics")
                        .value_parser(clap::value_parser!(QuickCheckAssertionSemantics))
                        .default_value("ignore")
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("output_json")
                        .long("output_json")
                        .value_name("PATH")
                        .help("Write the JSON result to PATH")
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("uf")
                        .long("uf")
                        .value_name("func_name:uf_name")
                        .help("Treat DSLX function as uninterpreted: format <func_name>:<uf_name> (repeatable). Functions sharing the same uf_name are assumed equivalent; assertions inside them are ignored.")
                        .action(clap::ArgAction::Append),
                ),
        )
        .subcommand(
            clap::Command::new("prover")
                .about("Run a prover plan with a process-based scheduler")
                .arg(
                    clap::Arg::new("cores")
                        .long("cores")
                        .value_name("N")
                        .help("Maximum concurrent processes to run")
                        .default_value("1")
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("plan_json_file")
                        .long("plan_json_file")
                        .value_name("PATH_OR_-")
                        .help("Path to ProverPlan JSON file or '-' for stdin")
                        .required(true)
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("output_json")
                        .long("output_json")
                        .value_name("PATH")
                        .help("Write the overall result to PATH as JSON {\"success\": <bool>}")
                        .action(clap::ArgAction::Set),
                ),
        )
        .subcommand(
            clap::Command::new("dslx-equiv")
                .about("Checks if two DSLX functions are equivalent")
                .arg(
                    clap::Arg::new("lhs_dslx_file")
                        .help("The left-hand side DSLX file")
                        .required(true)
                        .index(1),
                )
                .arg(
                    clap::Arg::new("rhs_dslx_file")
                        .help("The right-hand side DSLX file")
                        .required(true)
                        .index(2),
                )
                .arg(
                    clap::Arg::new("dslx_top")
                        .long("dslx_top")
                        .value_name("TOP")
                        .help("Shared top-level function name (applies to both LHS and RHS if provided)"),
                )
                .arg(
                    clap::Arg::new("lhs_dslx_top")
                        .long("lhs_dslx_top")
                        .value_name("LHS_TOP")
                        .help("Top-level function name for the LHS DSLX file"),
                )
                .arg(
                    clap::Arg::new("rhs_dslx_top")
                        .long("rhs_dslx_top")
                        .value_name("RHS_TOP")
                        .help("Top-level function name for the RHS DSLX file"),
                )
                .arg(
                    clap::Arg::new("dslx_path")
                        .long("dslx_path")
                        .value_name("DSLX_PATH_SEMI_SEPARATED")
                        .help("Semi-separated search paths for DSLX modules"),
                )
                .arg(
                    clap::Arg::new("dslx_stdlib_path")
                        .long("dslx_stdlib_path")
                        .value_name("DSLX_STDLIB_PATH")
                        .help("Path to the DSLX standard library"),
                )
                .add_bool_arg(
                    "type_inference_v2",
                    "Enable the experimental type-inference v2 algorithm (external toolchain only)",
                )
                .arg(
                    clap::Arg::new("solver")
                        .long("solver")
                        .value_name("SOLVER")
                        .help("Use the specified solver for equivalence checking")
                        .value_parser([
                            #[cfg(feature = "has-easy-smt")]
                            "z3-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "bitwuzla-binary",
                            #[cfg(feature = "has-easy-smt")]
                            "boolector-binary",
                            #[cfg(feature = "has-bitwuzla")]
                            "bitwuzla",
                            #[cfg(feature = "has-boolector")]
                            "boolector",
                            #[cfg(feature = "has-boolector")]
                            "boolector-legacy",
                            "toolchain",
                        ])
                        .default_value("toolchain")
                        .action(clap::ArgAction::Set),
                )
                .add_bool_arg(
                    "flatten_aggregates",
                    "Flatten tuple and array types to bits for equivalence checking",
                )
                .arg(
                    clap::Arg::new("drop_params")
                        .long("drop_params")
                        .value_name("CSV")
                        .help("Comma-separated list of parameter names to drop prior to equivalence checking"),
                )
                .arg(
                    clap::Arg::new("parallelism_strategy")
                        .long("parallelism-strategy")
                        .value_name("STRATEGY")
                        .help("Parallelism strategy")
                        .value_parser(["single-threaded", "output-bits", "input-bit-split"])
                        .default_value("single-threaded")
                        .action(clap::ArgAction::Set),
                )
                .arg(
                    clap::Arg::new("assertion_semantics")
                        .long("assertion-semantics")
                        .value_name("SEMANTICS")
                        .help("Assertion semantics")
                        .value_parser(["ignore", "never", "same", "assume", "implies"])
                        .default_value("same")
                        .action(clap::ArgAction::Set),
                )
                .add_bool_arg(
                    "lhs_fixed_implicit_activation",
                    "Fix the implicit activation bit to true for the LHS IR, useful when only LHS or RHS has implicit token",
                )
                .add_bool_arg(
                    "rhs_fixed_implicit_activation",
                    "Fix the implicit activation bit to true for the RHS IR, useful when only LHS or RHS has implicit token",
                )
                .add_bool_arg(
                    "assume-enum-in-bound",
                    "Constrain enum-typed parameters to their defined values during equivalence proving",
                )
                .arg(
                    clap::Arg::new("lhs_uf")
                        .long("lhs_uf")
                        .value_name("func_name:uf_name")
                        .help("Treat LHS DSLX function as uninterpreted: format <func_name>:<uf_name> (repeatable). Mappings to the same uf_name across sides are assumed equivalent; assertions inside them are ignored.")
                        .action(clap::ArgAction::Append),
                )
                .arg(
                    clap::Arg::new("rhs_uf")
                        .long("rhs_uf")
                        .value_name("func_name:uf_name")
                        .help("Treat RHS DSLX function as uninterpreted: format <func_name>:<uf_name> (repeatable). Mappings to the same uf_name across sides are assumed equivalent; assertions inside them are ignored.")
                        .action(clap::ArgAction::Append),
                )
                .arg(
                    clap::Arg::new("output_json")
                        .long("output_json")
                        .value_name("PATH")
                        .help("Write the JSON result to PATH")
                        .action(clap::ArgAction::Set),
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
    } else if let Some(matches) = matches.subcommand_matches("dslx-stitch-pipeline") {
        dslx_stitch_pipeline::handle_dslx_stitch_pipeline(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx2ir") {
        dslx2ir::handle_dslx2ir(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2opt") {
        ir2opt::handle_ir2opt(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2pipeline") {
        ir2pipeline::handle_ir2pipeline(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx2sv-types") {
        dslx2sv_types::handle_dslx2sv_types(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx-show") {
        dslx_show::handle_dslx_show(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx-g8r-stats") {
        dslx_g8r_stats::handle_dslx_g8r_stats(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2delayinfo") {
        ir2delayinfo::handle_ir2delayinfo(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-equiv") {
        ir_equiv::handle_ir_equiv(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-equiv-blocks") {
        ir_equiv_blocks::handle_ir_equiv_blocks(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("dslx-equiv") {
        dslx_equiv::handle_dslx_equiv(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-ged") {
        ir_ged::handle_ir_ged(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-fn-to-block") {
        ir_fn_to_block::handle_ir_fn_to_block(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-structural-similarity") {
        ir_structural_similarity::handle_ir_structural_similarity(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-localized-eco") {
        ir_localized_eco::handle_ir_localized_eco(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir-round-trip") {
        ir_round_trip::handle_ir_round_trip(matches);
    } else if let Some(matches) = matches.subcommand_matches("ir2gates") {
        ir2gates::handle_ir2gates(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2g8r") {
        ir2gates::handle_ir2g8r(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("lib2proto") {
        lib2proto::handle_lib2proto(matches);
    } else if let Some(matches) = matches.subcommand_matches("gv2ir") {
        gv2ir::handle_gv2ir(matches);
    } else if let Some(matches) = matches.subcommand_matches("gv-read-stats") {
        gv_read_stats::handle_gv_read_stats(matches);
    } else if let Some(matches) = matches.subcommand_matches("g8r2v") {
        if let Err(e) = g8r2v::handle_g8r2v(matches) {
            report_cli_error::report_cli_error_and_exit(&e, None, vec![]);
        }
    } else if let Some(matches) = matches.subcommand_matches("ir-fn-eval") {
        ir_fn_eval::handle_ir_fn_eval(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("g8r-equiv") {
        g8r_equiv::handle_g8r_equiv(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("run-verilog-pipeline") {
        run_verilog_pipeline::handle_run_verilog_pipeline(matches);
    } else if let Some(matches) = matches.subcommand_matches("ir-strip-pos-data") {
        ir_strip_pos_data::handle_ir_strip_pos_data(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("prove-quickcheck") {
        prove_quickcheck::handle_prove_quickcheck(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("prover") {
        prover::handle_prover(matches, &config);
    } else if let Some(matches) = matches.subcommand_matches("ir2combo") {
        ir2combo::handle_ir2combo(matches, &config);
    } else if let Some(_matches) = matches.subcommand_matches("version") {
        println!("{}", env!("CARGO_PKG_VERSION"));
    } else {
        report_cli_error_and_exit("No valid subcommand provided.", None, vec![]);
    }
}
