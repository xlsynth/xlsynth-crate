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

use crate::parallelism::ParallelismStrategy;

const SUBCOMMAND: &str = "ir-equiv";

// -----------------------------------------------------------------------------
// Shared equivalence input arguments (solver choice handled outside)
// -----------------------------------------------------------------------------

pub struct EquivInputs<'a> {
    pub lhs_ir_text: &'a str,
    pub rhs_ir_text: &'a str,
    pub lhs_top: Option<&'a str>,
    pub rhs_top: Option<&'a str>,
    pub flatten_aggregates: bool,
    pub drop_params: &'a [String],
    pub strategy: ParallelismStrategy,
    pub assertion_semantics: AssertionSemantics,
    pub lhs_fixed_implicit_activation: bool,
    pub rhs_fixed_implicit_activation: bool,
    pub subcommand: &'a str,
    pub lhs_origin: &'a str,
    pub rhs_origin: &'a str,
}

// Helper: parse IR text, pick top (explicit or package top), drop params.
fn parse_and_prepare_fn(
    ir_text: &str,
    top: Option<&str>,
    drop_params: &[String],
    subcommand: &str,
    origin: &str,
    side: &str,
) -> xlsynth_g8r::xls_ir::ir::Fn {
    let pkg = match ir_parser::Parser::new(ir_text).parse_package() {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!(
                "[{}] Failed to parse {} IR ({}): {}",
                subcommand, side, origin, e
            );
            std::process::exit(1);
        }
    };
    let fn_owned = if let Some(top_name) = top {
        pkg.get_fn(top_name).cloned().unwrap_or_else(|| {
            eprintln!(
                "[{}] Top function '{}' not found in {} IR (origin: {})",
                subcommand, top_name, side, origin
            );
            std::process::exit(1);
        })
    } else {
        pkg.get_top().cloned().unwrap_or_else(|| {
            eprintln!(
                "[{}] No top function found in {} IR (origin: {})",
                subcommand, side, origin
            );
            std::process::exit(1);
        })
    };
    fn_owned
        .drop_params(drop_params)
        .expect("Dropped parameter used in function body")
}

/// Toolchain top name unifier (exposed so DSLX path can reuse identical logic).
pub fn unify_toolchain_tops<'a>(
    lhs_ir: &'a str,
    rhs_ir: &'a str,
    lhs_top: &str,
    rhs_top: &str,
) -> (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>, String) {
    if lhs_top == rhs_top {
        return (
            std::borrow::Cow::Borrowed(lhs_ir),
            std::borrow::Cow::Borrowed(rhs_ir),
            lhs_top.to_string(),
        );
    }
    let unified = lhs_top.to_string();
    let rhs_rewritten = rhs_ir.replace(rhs_top, &unified);
    (
        std::borrow::Cow::Borrowed(lhs_ir),
        std::borrow::Cow::Owned(rhs_rewritten),
        unified,
    )
}

/// Internal helper used by the native runner for solvers (non-legacy path)
fn run_equiv_check_native<S: Solver>(solver_config: &S::Config, inputs: &EquivInputs) -> ! {
    let lhs_fn_dropped = parse_and_prepare_fn(
        inputs.lhs_ir_text,
        inputs.lhs_top,
        inputs.drop_params,
        inputs.subcommand,
        inputs.lhs_origin,
        "LHS",
    );
    let rhs_fn_dropped = parse_and_prepare_fn(
        inputs.rhs_ir_text,
        inputs.rhs_top,
        inputs.drop_params,
        inputs.subcommand,
        inputs.rhs_origin,
        "RHS",
    );

    let lhs_ir_fn = IrFn {
        fn_ref: &lhs_fn_dropped,
        fixed_implicit_activation: inputs.lhs_fixed_implicit_activation,
    };
    let rhs_ir_fn = IrFn {
        fn_ref: &rhs_fn_dropped,
        fixed_implicit_activation: inputs.rhs_fixed_implicit_activation,
    };

    let start_time = std::time::Instant::now();
    let result = match inputs.strategy {
        ParallelismStrategy::SingleThreaded => prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_ir_fn,
            &rhs_ir_fn,
            inputs.assertion_semantics,
            inputs.flatten_aggregates,
        ),
        ParallelismStrategy::OutputBits => prove_ir_fn_equiv_output_bits_parallel::<S>(
            solver_config,
            &lhs_ir_fn,
            &rhs_ir_fn,
            inputs.assertion_semantics,
            inputs.flatten_aggregates,
        ),
        ParallelismStrategy::InputBitSplit => prove_ir_fn_equiv_split_input_bit::<S>(
            solver_config,
            &lhs_ir_fn,
            &rhs_ir_fn,
            0,
            0,
            inputs.assertion_semantics,
            inputs.flatten_aggregates,
        ),
    };
    let end_time = std::time::Instant::now();
    println!(
        "[{}] Time taken: {:?}",
        inputs.subcommand,
        end_time.duration_since(start_time)
    );

    match result {
        EquivResult::Proved => {
            println!("[{}] success: Solver proved equivalence", inputs.subcommand);
            std::process::exit(0);
        }
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => {
            println!(
                "[{}] failure: Solver found counterexample:",
                inputs.subcommand
            );
            println!("  inputs LHS: {:?}", lhs_inputs);
            println!("  inputs RHS: {:?}", rhs_inputs);
            println!("  output LHS: {:?}", lhs_output);
            println!("  output RHS: {:?}", rhs_output);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "has-boolector")]
fn run_boolector_legacy_native(inputs: &EquivInputs) -> ! {
    use xlsynth_g8r::ir_equiv_boolector;

    if inputs.lhs_fixed_implicit_activation || inputs.rhs_fixed_implicit_activation {
        eprintln!(
            "[{}] Error: fixed implicit activation flags not supported for boolector-legacy solver",
            inputs.subcommand
        );
        std::process::exit(1);
    }
    if inputs.assertion_semantics != AssertionSemantics::Same {
        eprintln!("[{}] Error: assertion semantics other than 'same' not supported for boolector-legacy solver", inputs.subcommand);
        std::process::exit(1);
    }

    let lhs_fn_dropped = parse_and_prepare_fn(
        inputs.lhs_ir_text,
        inputs.lhs_top,
        inputs.drop_params,
        inputs.subcommand,
        inputs.lhs_origin,
        "LHS",
    );
    let rhs_fn_dropped = parse_and_prepare_fn(
        inputs.rhs_ir_text,
        inputs.rhs_top,
        inputs.drop_params,
        inputs.subcommand,
        inputs.rhs_origin,
        "RHS",
    );

    let start_time = std::time::Instant::now();
    let result = match inputs.strategy {
        ParallelismStrategy::SingleThreaded => ir_equiv_boolector::prove_ir_fn_equiv(
            &lhs_fn_dropped,
            &rhs_fn_dropped,
            inputs.flatten_aggregates,
        ),
        ParallelismStrategy::OutputBits => {
            ir_equiv_boolector::prove_ir_fn_equiv_output_bits_parallel(
                &lhs_fn_dropped,
                &rhs_fn_dropped,
                inputs.flatten_aggregates,
            )
        }
        ParallelismStrategy::InputBitSplit => {
            ir_equiv_boolector::prove_ir_fn_equiv_split_input_bit(
                &lhs_fn_dropped,
                &rhs_fn_dropped,
                0,
                0,
                inputs.flatten_aggregates,
            )
        }
    };
    let end_time = std::time::Instant::now();
    println!(
        "[{}] Time taken: {:?}",
        inputs.subcommand,
        end_time.duration_since(start_time)
    );

    match result {
        ir_equiv_boolector::EquivResult::Proved => {
            println!("[{}] success: Solver proved equivalence", inputs.subcommand);
            std::process::exit(0);
        }
        ir_equiv_boolector::EquivResult::Disproved {
            inputs: cex,
            outputs: (lhs_bits, rhs_bits),
        } => {
            println!(
                "[{}] failure: Solver found counterexample: {:?}",
                inputs.subcommand, cex
            );
            println!("    output LHS: {:?}", lhs_bits);
            println!("    output RHS: {:?}", rhs_bits);
            std::process::exit(1);
        }
    }
}

pub fn run_ir_equiv_native(solver_choice: SolverChoice, inputs: &EquivInputs) -> ! {
    match solver_choice {
        #[cfg(feature = "has-boolector")]
        SolverChoice::Boolector => {
            use xlsynth_g8r::equiv::boolector_backend::{Boolector, BoolectorConfig};
            let cfg = BoolectorConfig::new();
            run_equiv_check_native::<Boolector>(&cfg, inputs);
        }
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::Z3Binary | SolverChoice::BitwuzlaBinary | SolverChoice::BoolectorBinary => {
            use xlsynth_g8r::equiv::easy_smt_backend::{EasySmtConfig, EasySmtSolver};
            let cfg = match solver_choice {
                SolverChoice::Z3Binary => EasySmtConfig::z3(),
                SolverChoice::BitwuzlaBinary => EasySmtConfig::bitwuzla(),
                SolverChoice::BoolectorBinary => EasySmtConfig::boolector(),
                _ => unreachable!(),
            };
            run_equiv_check_native::<EasySmtSolver>(&cfg, inputs);
        }
        #[cfg(feature = "has-bitwuzla")]
        SolverChoice::Bitwuzla => {
            use xlsynth_g8r::equiv::bitwuzla_backend::{Bitwuzla, BitwuzlaOptions};
            let opts = BitwuzlaOptions::new();
            run_equiv_check_native::<Bitwuzla>(&opts, inputs);
        }
        #[cfg(feature = "has-boolector")]
        SolverChoice::BoolectorLegacy => {
            run_boolector_legacy_native(inputs);
        }
        SolverChoice::Toolchain => {
            eprintln!("Internal error: run_ir_equiv_native called with Toolchain solver");
            std::process::exit(1);
        }
    }
}

pub fn run_toolchain_ir_equiv_text(
    lhs_ir: &str,
    rhs_ir: &str,
    top: &str,
    tool_path: &str,
    subcommand: &str,
    lhs_origin: &str,
    rhs_origin: &str,
) -> ! {
    let lhs_tmp = tempfile::NamedTempFile::new().unwrap();
    let rhs_tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(lhs_tmp.path(), lhs_ir).unwrap();
    std::fs::write(rhs_tmp.path(), rhs_ir).unwrap();
    let output =
        run_check_ir_equivalence_main(lhs_tmp.path(), rhs_tmp.path(), Some(top), tool_path);
    match output {
        Ok(stdout) => {
            println!("[{}] success: {}", subcommand, stdout.trim());
            std::process::exit(0);
        }
        Err(output) => {
            let mut message = String::from_utf8_lossy(&output.stdout);
            if message.is_empty() {
                message = String::from_utf8_lossy(&output.stderr);
            }
            report_cli_error_and_exit(
                &format!("failure: {}", message),
                Some(subcommand),
                vec![
                    ("lhs_origin", lhs_origin),
                    ("rhs_origin", rhs_origin),
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
}

// -----------------------------------------------------------------------------
// Unified dispatch function (toolchain vs native) for reuse by both subcommands
// -----------------------------------------------------------------------------

pub fn dispatch_ir_equiv(
    solver_choice: Option<SolverChoice>,
    tool_path: Option<&str>,
    inputs: &EquivInputs,
) -> ! {
    if solver_choice.is_none() && tool_path.is_none() {
        eprintln!(
            "[{}] Error: no solver specified and no toolchain path configured (need --solver or toolchain config)",
            inputs.subcommand
        );
        std::process::exit(1);
    }
    let use_toolchain = match solver_choice {
        Some(SolverChoice::Toolchain) => true,
        Some(_) => false,
        None => tool_path.is_some(),
    };

    if use_toolchain {
        let tool_path = tool_path.expect("tool_path required for toolchain path");
        match (inputs.lhs_top, inputs.rhs_top) {
            (Some(lt), Some(rt)) => {
                if lt != rt {
                    let (lhs_use, rhs_use, unified_top) =
                        unify_toolchain_tops(inputs.lhs_ir_text, inputs.rhs_ir_text, lt, rt);
                    run_toolchain_ir_equiv_text(
                        &lhs_use,
                        &rhs_use,
                        &unified_top,
                        tool_path,
                        inputs.subcommand,
                        inputs.lhs_origin,
                        inputs.rhs_origin,
                    );
                } else {
                    run_toolchain_ir_equiv_text(
                        inputs.lhs_ir_text,
                        inputs.rhs_ir_text,
                        lt,
                        tool_path,
                        inputs.subcommand,
                        inputs.lhs_origin,
                        inputs.rhs_origin,
                    );
                }
            }
            _ => {
                eprintln!(
                    "[{}] Error: top function(s) must be specified for toolchain equivalence",
                    inputs.subcommand
                );
                std::process::exit(1);
            }
        }
    } else {
        let solver_choice = solver_choice.expect("Non-toolchain solver must be specified");
        run_ir_equiv_native(solver_choice, inputs);
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
        rhs_top = lhs_top;
    }

    let assertion_semantics = matches
        .get_one::<AssertionSemantics>("assertion_semantics")
        .unwrap_or(&AssertionSemantics::Same);

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

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    let lhs_ir_text = std::fs::read_to_string(lhs).unwrap_or_else(|e| {
        eprintln!("Failed to read lhs IR file: {}", e);
        std::process::exit(1)
    });
    let rhs_ir_text = std::fs::read_to_string(rhs).unwrap_or_else(|e| {
        eprintln!("Failed to read rhs IR file: {}", e);
        std::process::exit(1)
    });

    let inputs = EquivInputs {
        lhs_ir_text: &lhs_ir_text,
        rhs_ir_text: &rhs_ir_text,
        lhs_top,
        rhs_top,
        flatten_aggregates,
        drop_params: &drop_params,
        strategy,
        assertion_semantics: *assertion_semantics,
        lhs_fixed_implicit_activation,
        rhs_fixed_implicit_activation,
        subcommand: SUBCOMMAND,
        lhs_origin: lhs,
        rhs_origin: rhs,
    };

    dispatch_ir_equiv(solver, tool_path, &inputs);
}
