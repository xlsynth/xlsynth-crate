// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::solver_choice::SolverChoice;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;
use xlsynth_g8r::equiv::solver_interface::Solver;

use xlsynth_g8r::equiv::prove_equiv::{
    prove_ir_fn_equiv, prove_ir_fn_equiv_output_bits_parallel, prove_ir_fn_equiv_split_input_bit,
    AssertionSemantics, EquivResult, IrFn,
};
use xlsynth_g8r::xls_ir::ir_parser;

#[derive(Clone, Copy)]
enum ParallelismStrategy {
    SingleThreaded,
    OutputBits,
    InputBitSplit,
}

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

#[cfg(feature = "has-boolector")]
fn run_boolector_equiv_check(
    lhs_path: &std::path::Path,
    rhs_path: &std::path::Path,
    lhs_top: Option<&str>,
    rhs_top: Option<&str>,
    flatten_aggregates: bool,
    drop_params: &[String],
    strategy: ParallelismStrategy,
) -> ! {
    // Parse both IR files to xls_ir::ir::Package

    use xlsynth_g8r::ir_equiv_boolector;
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
            xlsynth_g8r::ir_equiv_boolector::prove_ir_fn_equiv(&lhs_fn, &rhs_fn, flatten_aggregates)
        }
        ParallelismStrategy::OutputBits => {
            xlsynth_g8r::ir_equiv_boolector::prove_ir_fn_equiv_output_bits_parallel(
                &lhs_fn,
                &rhs_fn,
                flatten_aggregates,
            )
        }
        // Split on the first parameter and the first bit for now.
        // TODO: introduce better heuristics like picking the maximal-fan-out bit or
        // divide-and-conquer dynamically on more and more bits.
        ParallelismStrategy::InputBitSplit => {
            xlsynth_g8r::ir_equiv_boolector::prove_ir_fn_equiv_split_input_bit(
                &lhs_fn,
                &rhs_fn,
                0,
                0,
                flatten_aggregates,
            )
        }
    };
    let end_time = std::time::Instant::now();
    println!("Time taken: {:?}", end_time.duration_since(start_time));

    match result {
        ir_equiv_boolector::EquivResult::Proved => {
            println!("success: Solver proved equivalence");
            std::process::exit(0);
        }
        ir_equiv_boolector::EquivResult::Disproved {
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

fn run_equiv_check<S: Solver>(
    solver_config: &S::Config,
    lhs_path: &std::path::Path,
    rhs_path: &std::path::Path,
    lhs_top: Option<&str>,
    rhs_top: Option<&str>,
    lhs_fixed_implicit_activation: bool,
    rhs_fixed_implicit_activation: bool,
    flatten_aggregates: bool,
    drop_params: &[String],
    strategy: ParallelismStrategy,
    assertion_semantics: AssertionSemantics,
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
    let lhs_fn = IrFn {
        fn_ref: &lhs_fn
            .drop_params(drop_params)
            .expect("Dropped parameter is used in the function body!"),
        fixed_implicit_activation: lhs_fixed_implicit_activation,
    };
    let rhs_fn = IrFn {
        fn_ref: &rhs_fn
            .drop_params(drop_params)
            .expect("Dropped parameter is used in the function body!"),
        fixed_implicit_activation: rhs_fixed_implicit_activation,
    };

    let start_time = std::time::Instant::now();
    let result = match strategy {
        ParallelismStrategy::SingleThreaded => prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_fn,
            &rhs_fn,
            assertion_semantics,
            flatten_aggregates,
        ),
        ParallelismStrategy::OutputBits => prove_ir_fn_equiv_output_bits_parallel::<S>(
            solver_config,
            &lhs_fn,
            &rhs_fn,
            assertion_semantics,
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
            assertion_semantics,
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
    let assertion_semantics = matches
        .get_one::<String>("assertion_semantics")
        .map(|s| s.parse().unwrap())
        .unwrap_or(AssertionSemantics::Same);

    let solver: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());

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
    let lhs_fixed_implicit_activation = matches
        .get_one::<String>("lhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);
    let rhs_fixed_implicit_activation = matches
        .get_one::<String>("rhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);

    if let Some(solver) = solver {
        match solver {
            #[cfg(feature = "has-boolector")]
            SolverChoice::Boolector => {
                use xlsynth_g8r::equiv::boolector_backend::{Boolector, BoolectorConfig};
                let config = BoolectorConfig::new();
                run_equiv_check::<Boolector>(
                    &config,
                    lhs_path,
                    rhs_path,
                    lhs_top,
                    rhs_top,
                    lhs_fixed_implicit_activation,
                    rhs_fixed_implicit_activation,
                    flatten_aggregates,
                    &drop_params,
                    strategy,
                    assertion_semantics,
                );
            }
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::Z3Binary
            | SolverChoice::BitwuzlaBinary
            | SolverChoice::BoolectorBinary => {
                use xlsynth_g8r::equiv::easy_smt_backend::{EasySmtConfig, EasySmtSolver};
                let config = match solver {
                    SolverChoice::Z3Binary => EasySmtConfig::z3(),
                    SolverChoice::BitwuzlaBinary => EasySmtConfig::bitwuzla(),
                    SolverChoice::BoolectorBinary => EasySmtConfig::boolector(),
                    _ => unreachable!(),
                };
                run_equiv_check::<EasySmtSolver>(
                    &config,
                    lhs_path,
                    rhs_path,
                    lhs_top,
                    rhs_top,
                    lhs_fixed_implicit_activation,
                    rhs_fixed_implicit_activation,
                    flatten_aggregates,
                    &drop_params,
                    strategy,
                    assertion_semantics,
                );
            }
            #[cfg(feature = "has-bitwuzla")]
            SolverChoice::Bitwuzla => {
                use xlsynth_g8r::equiv::bitwuzla_backend::{Bitwuzla, BitwuzlaOptions};
                let options = BitwuzlaOptions::new();
                let config = options;
                run_equiv_check::<Bitwuzla>(
                    &config,
                    lhs_path,
                    rhs_path,
                    lhs_top,
                    rhs_top,
                    lhs_fixed_implicit_activation,
                    rhs_fixed_implicit_activation,
                    flatten_aggregates,
                    &drop_params,
                    strategy,
                    assertion_semantics,
                );
            }
            #[cfg(feature = "has-boolector")]
            SolverChoice::BoolectorLegacy => {
                log::info!("run_boolector_equiv_check");
                if lhs_fixed_implicit_activation || rhs_fixed_implicit_activation {
                    eprintln!("Error: --lhs_fixed_implicit_activation and --rhs_fixed_implicit_activation are not supported for boolector-legacy solver");
                    std::process::exit(1);
                }
                if assertion_semantics != AssertionSemantics::Same {
                    eprintln!(
                        "Error: --assertion_semantics is not supported for boolector-legacy solver"
                    );
                    std::process::exit(1);
                }
                run_boolector_equiv_check(
                    lhs_path,
                    rhs_path,
                    lhs_top,
                    rhs_top,
                    flatten_aggregates,
                    &drop_params,
                    strategy,
                );
            }
            SolverChoice::Toolchain => {
                ir_equiv(lhs_path, rhs_path, lhs_top, rhs_top, config);
            }
        }
    } else {
        // Default: use the toolchain-based equivalence check
        ir_equiv(lhs_path, rhs_path, lhs_top, rhs_top, config);
    }
}
