// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::ir_equiv_boolector;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::xls_ir::ir_parser;

#[cfg(feature = "has-boolector")]
#[derive(Clone, Copy)]
enum ParallelismStrategy {
    SingleThreaded,
    OutputBits,
}

#[cfg(feature = "has-boolector")]
impl std::str::FromStr for ParallelismStrategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "single-threaded" => Ok(Self::SingleThreaded),
            "output-bits" => Ok(Self::OutputBits),
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
    let result = match strategy {
        ParallelismStrategy::SingleThreaded => {
            if flatten_aggregates {
                xlsynth_g8r::ir_equiv_boolector::prove_ir_equiv_flattened(&lhs_fn, &rhs_fn)
            } else {
                xlsynth_g8r::ir_equiv_boolector::prove_ir_fn_equiv(&lhs_fn, &rhs_fn)
            }
        }
        ParallelismStrategy::OutputBits => {
            xlsynth_g8r::ir_equiv_boolector::prove_ir_fn_equiv_output_bits_parallel(
                &lhs_fn,
                &rhs_fn,
                flatten_aggregates,
            )
        }
    };

    match result {
        ir_equiv_boolector::EquivResult::Proved => {
            println!("success: Boolector proved equivalence");
            std::process::exit(0);
        }
        ir_equiv_boolector::EquivResult::Disproved(cex) => {
            println!("failure: Boolector found counterexample: {:?}", cex);
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

    let use_boolector = matches
        .get_one::<String>("boolector")
        .map(|s| s == "true")
        .unwrap_or(false);
    if use_boolector {
        // If the feature is not enabled, error out.
        #[cfg(not(feature = "has-boolector"))]
        {
            eprintln!("Error: --boolector requested but this binary was not built with the has-boolector feature (enabled by with-boolector-system or with-boolector-built).");
            std::process::exit(1);
        }
        #[cfg(feature = "has-boolector")]
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

            log::info!("run_boolector_equiv_check");
            run_boolector_equiv_check(
                lhs_path,
                rhs_path,
                lhs_top,
                rhs_top,
                flatten_aggregates,
                &drop_params,
                strategy,
            );
            unreachable!();
        }
    }
    // Default: use the toolchain-based equivalence check
    ir_equiv(lhs_path, rhs_path, lhs_top, rhs_top, config);
}
