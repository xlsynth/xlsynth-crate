// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;
use xlsynth_g8r::equiv::solver_interface::Solver;

use xlsynth_g8r::equiv::prove_equiv::{
    prove_ir_fn_equiv, prove_ir_fn_equiv_output_bits_parallel, prove_ir_fn_equiv_split_input_bit,
    EquivResult,
};
use xlsynth_g8r::xls_ir::ir_parser;

#[derive(Clone, Copy)]
enum ParallelismStrategy {
    SingleThreaded,
    OutputBits,
    InputBitSplit,
}

#[cfg(feature = "has-easy-smt")]
impl std::str::FromStr for ParallelismStrategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "single-threaded" => Ok(Self::SingleThreaded),
            "output-bits" => Ok(Self::OutputBits),
            "input-bit-split" => Ok(Self::InputBitSplit),
            _ => Err(format!("invalid parallelism strategy: {}", s)),
        }
    }
}

#[derive(Clone, Copy)]
enum SolverChoice {
    #[cfg(feature = "has-easy-smt")]
    Z3,
    #[cfg(feature = "has-easy-smt")]
    Bitwuzla,
    #[cfg(feature = "has-easy-smt")]
    Boolector,
    #[cfg(feature = "has-bitwuzla")]
    BitwuzlaStatic,
}

impl std::str::FromStr for SolverChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            #[cfg(feature = "has-easy-smt")]
            "z3" => Ok(Self::Z3),
            #[cfg(feature = "has-easy-smt")]
            "bitwuzla" => Ok(Self::Bitwuzla),
            #[cfg(feature = "has-easy-smt")]
            "boolector" => Ok(Self::Boolector),
            #[cfg(feature = "has-bitwuzla")]
            "bitwuzla-static" => Ok(Self::BitwuzlaStatic),
            _ => Err(format!("invalid solver: {}", s)),
        }
    }
}

const SUBCOMMAND: &str = "ir-equiv";

fn ir_equiv(
    lhs: &std::path::Path,
    rhs: &std::path::Path,
    lhs_top: Option<&str>,
    rhs_top: Option<&str>,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir-equiv");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        if lhs_top != rhs_top {
            eprintln!("Error: --top flag is required for tool-based equivalence checking");
            std::process::exit(1);
        }
        let top = lhs_top;
        let output = run_check_ir_equivalence_main(lhs, rhs, top, tool_path);
        match output {
            Ok(stdout) => {
                println!("success: {}", stdout);
            }
            Err(output) => {
                // Note: the details of the counterexample come on stdout, not stderr.
                let mut message = String::from_utf8_lossy(&output.stdout);
                if message.is_empty() {
                    // Try for stderr if stdout was empty, e.g. in case of other kinds of errors.
                    message = String::from_utf8_lossy(&output.stderr);
                }
                report_cli_error_and_exit(
                    &format!("failure: {}", message),
                    Some(SUBCOMMAND),
                    vec![
                        ("lhs_ir_file", lhs.to_str().unwrap()),
                        ("rhs_ir_file", rhs.to_str().unwrap()),
                        (
                            "stdout",
                            &format!("{:?}", String::from_utf8_lossy(&output.stdout)),
                        ),
                        (
                            "stderr",
                            &format!("{:?}", String::from_utf8_lossy(&output.stderr)),
                        ),
                    ],
                );
            }
        }
    } else {
        report_cli_error_and_exit(
            "a tool path was not provided (e.g. via a xlsynth-toolchain.toml file)",
            Some(SUBCOMMAND),
            vec![("toolchain-config", format!("{:?}", config).as_str())],
        );
    }
}

fn run_easy_smt_equiv_check<S: Solver>(
    lhs_path: &std::path::Path,
    rhs_path: &std::path::Path,
    lhs_top: Option<&str>,
    rhs_top: Option<&str>,
    flatten_aggregates: bool,
    drop_params: &[String],
    strategy: ParallelismStrategy,
    solver_config: &S::Config,
) -> ! {
    // Parse both IR files to xls_ir::ir::Package
    let lhs_pkg = match ir_parser::parse_path_to_package(lhs_path) {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!("Failed to parse lhs IR file: {}", e);
            std::process::exit(1);
        }
    };
    let rhs_pkg = match ir_parser::parse_path_to_package(rhs_path) {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!("Failed to parse rhs IR file: {}", e);
            std::process::exit(1);
        }
    };
    // Select the function to check (top or first)
    let lhs_fn = if let Some(top_name) = lhs_top {
        lhs_pkg.get_fn(top_name).cloned().unwrap_or_else(|| {
            eprintln!("Top function '{}' not found in lhs IR file", top_name);
            std::process::exit(1);
        })
    } else {
        lhs_pkg.get_top().cloned().unwrap_or_else(|| {
            eprintln!("No top function found in lhs IR file");
            std::process::exit(1);
        })
    };
    let rhs_fn = if let Some(top_name) = rhs_top {
        rhs_pkg.get_fn(top_name).cloned().unwrap_or_else(|| {
            eprintln!("Top function '{}' not found in rhs IR file", top_name);
            std::process::exit(1);
        })
    } else {
        rhs_pkg.get_top().cloned().unwrap_or_else(|| {
            eprintln!("No top function found in rhs IR file");
            std::process::exit(1);
        })
    };
    // Drop parameters by name and check for usage
    let lhs_fn = lhs_fn
        .drop_params(drop_params)
        .expect("Dropped parameter is used in the function body!");
    let rhs_fn = rhs_fn
        .drop_params(drop_params)
        .expect("Dropped parameter is used in the function body!");
    let start_time = std::time::Instant::now();
    let result = match strategy {
        ParallelismStrategy::SingleThreaded => {
            prove_ir_fn_equiv::<S>(solver_config, &lhs_fn, &rhs_fn, flatten_aggregates)
        }
        ParallelismStrategy::OutputBits => prove_ir_fn_equiv_output_bits_parallel::<S>(
            solver_config,
            &lhs_fn,
            &rhs_fn,
            flatten_aggregates,
        ),
        // Split on the first parameter and the first bit for now.
        // TODO: introduce better heuristics like picking the maximal-fan-out bit or
        // divide-and-conquer dynamically on more and more bits.
        ParallelismStrategy::InputBitSplit => prove_ir_fn_equiv_split_input_bit::<S>(
            solver_config,
            &lhs_fn,
            &rhs_fn,
            0,
            0,
            flatten_aggregates,
        ),
    };
    let end_time = std::time::Instant::now();
    println!("Time taken: {:?}", end_time.duration_since(start_time));

    match result {
        EquivResult::Proved => {
            println!("success: Solver proved equivalence");
            std::process::exit(0);
        }
        EquivResult::Disproved {
            inputs: cex,
            outputs: (lhs_bits, rhs_bits),
        } => {
            println!("failure: Solver found counterexample: {:?}", cex);
            println!("    output LHS: {:?}", lhs_bits);
            println!("    output RHS: {:?}", rhs_bits);
            std::process::exit(1);
        }
    }
}

/// Implements the "ir-equiv" subcommand.
pub fn handle_ir_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_equiv");
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();

    let mut lhs_top = matches.get_one::<String>("lhs_ir_top").map(|s| s.as_str());
    let mut rhs_top = matches.get_one::<String>("rhs_ir_top").map(|s| s.as_str());

    let top = if let Some(top) = matches.get_one::<String>("ir_top") {
        Some(top.as_str())
    } else {
        None
    };

    if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        eprintln!("Error: --ir_top and --lhs_ir_top/--rhs_ir_top cannot be used together");
        std::process::exit(1);
    }

    if lhs_top.is_some() ^ rhs_top.is_some() {
        eprintln!("Error: --lhs_ir_top and --rhs_ir_top must be used together");
        std::process::exit(1);
    }

    if top.is_some() {
        lhs_top = top;
        rhs_top = top;
    }

    let lhs_path = std::path::Path::new(lhs);
    let rhs_path = std::path::Path::new(rhs);

    let solver: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());
    if let Some(solver) = solver {
        // If the feature is not enabled, error out.
        #[cfg(not(feature = "has-easy-smt"))]
        {
            eprintln!("Error: --solver requested but this binary was not built with the has-easy-smt feature (enabled by with-easy-smt).");
            std::process::exit(1);
        }

        #[cfg(feature = "has-easy-smt")]
        #[allow(unreachable_code)]
        {
            let flatten_aggregates = matches
                .get_one::<String>("flatten_aggregates")
                .map(|s| s == "true")
                .unwrap_or(false);
            let drop_params: Vec<String> = matches
                .get_one::<String>("drop_params")
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                .unwrap_or_else(Vec::new);
            let strategy = matches
                .get_one::<String>("parallelism_strategy")
                .map(|s| s.parse().unwrap())
                .unwrap_or(ParallelismStrategy::SingleThreaded);
            match solver {
                #[cfg(feature = "has-easy-smt")]
                SolverChoice::Z3 | SolverChoice::Bitwuzla | SolverChoice::Boolector => {
                    use xlsynth_g8r::equiv::easy_smt_backend::{EasySMTConfig, EasySMTSolver};
                    let solver_config = match solver {
                        SolverChoice::Z3 => EasySMTConfig::z3(),
                        SolverChoice::Bitwuzla => EasySMTConfig {
                            ..EasySMTConfig::bitwuzla()
                        },
                        SolverChoice::Boolector => EasySMTConfig::boolector(),
                        _ => unreachable!(),
                    };
                    log::info!("run_easy_smt_equiv_check");
                    run_easy_smt_equiv_check::<EasySMTSolver>(
                        lhs_path,
                        rhs_path,
                        lhs_top,
                        rhs_top,
                        flatten_aggregates,
                        &drop_params,
                        strategy,
                        &solver_config,
                    );
                }
                #[cfg(feature = "has-bitwuzla")]
                SolverChoice::BitwuzlaStatic => {
                    use xlsynth_g8r::equiv::bitwuzla_backend::BitwuzlaSolver;

                    log::info!("run_easy_smt_equiv_check");
                    run_easy_smt_equiv_check::<BitwuzlaSolver>(
                        lhs_path,
                        rhs_path,
                        lhs_top,
                        rhs_top,
                        flatten_aggregates,
                        &drop_params,
                        strategy,
                        (),
                    );
                }
            }
            unreachable!();
        }
    }
    // Default: use the toolchain-based equivalence check
    ir_equiv(lhs_path, rhs_path, lhs_top, rhs_top, config);
}
