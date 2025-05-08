// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;

use xlsynth_g8r::ir_equiv_boolector;
use xlsynth_g8r::xls_ir::ir_parser;

const SUBCOMMAND: &str = "ir-equiv";

fn ir_equiv(
    lhs: &std::path::Path,
    rhs: &std::path::Path,
    top: Option<&str>,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir-equiv");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
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

#[cfg(feature = "with-boolector")]
fn run_boolector_equiv_check(
    lhs_path: &std::path::Path,
    rhs_path: &std::path::Path,
    top: Option<&str>,
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
    let lhs_fn = if let Some(top_name) = top {
        lhs_pkg.get_fn(top_name).unwrap_or_else(|| {
            eprintln!("Top function '{}' not found in lhs IR file", top_name);
            std::process::exit(1);
        })
    } else {
        lhs_pkg.get_top().unwrap_or_else(|| {
            eprintln!("No top function found in lhs IR file");
            std::process::exit(1);
        })
    };
    let rhs_fn = if let Some(top_name) = top {
        rhs_pkg.get_fn(top_name).unwrap_or_else(|| {
            eprintln!("Top function '{}' not found in rhs IR file", top_name);
            std::process::exit(1);
        })
    } else {
        rhs_pkg.get_top().unwrap_or_else(|| {
            eprintln!("No top function found in rhs IR file");
            std::process::exit(1);
        })
    };
    // Run Boolector equivalence check
    match ir_equiv_boolector::check_equiv(lhs_fn, rhs_fn) {
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

pub fn handle_ir_equiv(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_equiv");
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let top = if let Some(top) = matches.get_one::<String>("ir_top") {
        Some(top.as_str())
    } else {
        None
    };
    let lhs_path = std::path::Path::new(lhs);
    let rhs_path = std::path::Path::new(rhs);

    let use_boolector = matches
        .get_one::<String>("boolector")
        .map(|s| s == "true")
        .unwrap_or(false);
    if use_boolector {
        // If the feature is not enabled, error out.
        #[cfg(not(feature = "with-boolector"))]
        {
            eprintln!("Error: --boolector requested but this binary was not built with --features=with-boolector");
            std::process::exit(1);
        }
        #[cfg(feature = "with-boolector")]
        #[allow(unreachable_code)]
        {
            log::info!("run_boolector_equiv_check");
            run_boolector_equiv_check(lhs_path, rhs_path, top);
            unreachable!();
        }
    }
    // Default: use the toolchain-based equivalence check
    ir_equiv(lhs_path, rhs_path, top, config);
}
