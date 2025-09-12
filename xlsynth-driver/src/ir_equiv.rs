// SPDX-License-Identifier: Apache-2.0

// Use some pragmas since when the configuration does not have certain
// configured engines on we would get a bunch of warnings.
#![allow(unused)]

use crate::common::{infer_uf_signature, merge_uf_signature};
use crate::solver_choice::SolverChoice;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;
use xlsynth_pir::ir;
use xlsynth_prover::solver_interface::Solver;

use std::collections::HashMap;
use xlsynth::IrValue;
use xlsynth_pir::ir_parser;
use xlsynth_prover::prove_equiv::{
    prove_ir_fn_equiv_full, prove_ir_fn_equiv_output_bits_parallel,
    prove_ir_fn_equiv_split_input_bit,
};
use xlsynth_prover::prover::{ExternalProver, Prover};
use xlsynth_prover::types::{AssertionSemantics, EquivResult, EquivSide, IrFn};

use crate::parallelism::ParallelismStrategy;

const SUBCOMMAND: &str = "ir-equiv";
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EquivOutcome {
    pub time_micros: u128,
    pub success: bool,
    pub counterexample: Option<String>,
}

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
    pub lhs_param_domains: Option<HashMap<String, Vec<IrValue>>>,
    pub rhs_param_domains: Option<HashMap<String, Vec<IrValue>>>,
    pub lhs_uf_map: HashMap<String, String>,
    pub rhs_uf_map: HashMap<String, String>,
}

// Helper: parse IR text into a Package, pick top (explicit or package top),
// drop params. Returns the parsed package and an owned function (potentially
// modified by drop_params). It is okay to keep the unmodified package as we do
// not allow recursion in the IR.
fn parse_and_prepare_fn(
    ir_text: &str,
    top: Option<&str>,
    drop_params: &[String],
    subcommand: &str,
    origin: &str,
    side: &str,
) -> (ir::Package, ir::Fn) {
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
    let fn_owned = fn_owned
        .drop_params(drop_params)
        .expect("Dropped parameter used in function body");
    (pkg, fn_owned)
}

/// Internal helper that runs equivalence via a provided Prover (native path)
fn run_equiv_with_prover(prover: &dyn Prover, inputs: &EquivInputs) -> EquivOutcome {
    let (lhs_pkg, lhs_fn_dropped) = parse_and_prepare_fn(
        inputs.lhs_ir_text,
        inputs.lhs_top,
        inputs.drop_params,
        inputs.subcommand,
        inputs.lhs_origin,
        "LHS",
    );
    let (rhs_pkg, rhs_fn_dropped) = parse_and_prepare_fn(
        inputs.rhs_ir_text,
        inputs.rhs_top,
        inputs.drop_params,
        inputs.subcommand,
        inputs.rhs_origin,
        "RHS",
    );

    let lhs_ir_fn = IrFn {
        fn_ref: &lhs_fn_dropped,
        pkg_ref: Some(&lhs_pkg),
        fixed_implicit_activation: inputs.lhs_fixed_implicit_activation,
    };
    let rhs_ir_fn = IrFn {
        fn_ref: &rhs_fn_dropped,
        pkg_ref: Some(&rhs_pkg),
        fixed_implicit_activation: inputs.rhs_fixed_implicit_activation,
    };

    let start_time = std::time::Instant::now();
    let result = match inputs.strategy {
        ParallelismStrategy::SingleThreaded => {
            let lhs_uf_sigs = infer_uf_signature(&lhs_pkg, &inputs.lhs_uf_map);
            let rhs_uf_sigs = infer_uf_signature(&rhs_pkg, &inputs.rhs_uf_map);
            let uf_sigs = merge_uf_signature(lhs_uf_sigs, &rhs_uf_sigs);

            let lhs_side = EquivSide {
                ir_fn: &lhs_ir_fn,
                domains: inputs.lhs_param_domains.clone(),
                uf_map: inputs.lhs_uf_map.clone(),
            };
            let rhs_side = EquivSide {
                ir_fn: &rhs_ir_fn,
                domains: inputs.rhs_param_domains.clone(),
                uf_map: inputs.rhs_uf_map.clone(),
            };
            prover.prove_ir_fn_equiv_full(
                &lhs_side,
                &rhs_side,
                inputs.assertion_semantics,
                inputs.flatten_aggregates,
                &uf_sigs,
            )
        }
        ParallelismStrategy::OutputBits => prover.prove_ir_fn_equiv_output_bits_parallel(
            &lhs_ir_fn,
            &rhs_ir_fn,
            inputs.assertion_semantics,
            inputs.flatten_aggregates,
        ),
        ParallelismStrategy::InputBitSplit => prover.prove_ir_fn_equiv_split_input_bit(
            &lhs_ir_fn,
            &rhs_ir_fn,
            0,
            0,
            inputs.assertion_semantics,
            inputs.flatten_aggregates,
        ),
    };
    let micros = start_time.elapsed().as_micros();

    match result {
        EquivResult::Proved => EquivOutcome {
            time_micros: micros,
            success: true,
            counterexample: None,
        },
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => {
            let cex_str = format!(
                "lhs_inputs: {:?}, rhs_inputs: {:?}, lhs_output: {:?}, rhs_output: {:?}",
                lhs_inputs, rhs_inputs, lhs_output, rhs_output
            );
            EquivOutcome {
                time_micros: micros,
                success: false,
                counterexample: Some(cex_str),
            }
        }
        EquivResult::Error(msg) => {
            eprintln!("[{}] Error: {}", inputs.subcommand, msg);
            std::process::exit(1);
        }
        EquivResult::ToolchainDisproved(msg) => EquivOutcome {
            time_micros: micros,
            success: false,
            counterexample: Some(msg),
        },
    }
}

// -----------------------------------------------------------------------------
// Unified dispatch function (toolchain vs native) for reuse by both subcommands
// -----------------------------------------------------------------------------

pub fn dispatch_ir_equiv(
    solver_choice: Option<SolverChoice>,
    tool_path: Option<&str>,
    inputs: &EquivInputs,
) -> EquivOutcome {
    log::info!(
        "dispatch_ir_equiv; solver_choice: {:?}, tool_path: {:?}",
        solver_choice,
        tool_path
    );
    // Construct a Prover for the selected native solver via unified interface.
    match solver_choice {
        #[cfg(feature = "has-boolector")]
        Some(SolverChoice::Boolector) => {
            use xlsynth_prover::boolector_backend::BoolectorConfig;
            let prover = BoolectorConfig::new();
            run_equiv_with_prover(&prover, inputs)
        }
        #[cfg(feature = "has-easy-smt")]
        Some(SolverChoice::Z3Binary)
        | Some(SolverChoice::BitwuzlaBinary)
        | Some(SolverChoice::BoolectorBinary) => {
            use xlsynth_prover::easy_smt_backend::EasySmtConfig;
            let cfg = match solver_choice {
                Some(SolverChoice::Z3Binary) => EasySmtConfig::z3(),
                Some(SolverChoice::BitwuzlaBinary) => EasySmtConfig::bitwuzla(),
                Some(SolverChoice::BoolectorBinary) => EasySmtConfig::boolector(),
                _ => unreachable!(),
            };
            run_equiv_with_prover(&cfg, inputs)
        }
        #[cfg(feature = "has-bitwuzla")]
        Some(SolverChoice::Bitwuzla) => {
            use xlsynth_prover::bitwuzla_backend::BitwuzlaOptions;
            let opts = BitwuzlaOptions::new();
            run_equiv_with_prover(&opts, inputs)
        }
        Some(SolverChoice::Toolchain) => {
            let prover = match tool_path {
                Some(p) => {
                    let pb = std::path::Path::new(p);
                    if pb.is_dir() {
                        ExternalProver::ToolDir(pb.to_path_buf())
                    } else {
                        ExternalProver::ToolExe(pb.to_path_buf())
                    }
                }
                None => ExternalProver::Toolchain,
            };
            run_equiv_with_prover(&prover, inputs)
        }
        None => {
            let prover = xlsynth_prover::prover::auto_selected_prover();
            run_equiv_with_prover(&*prover, inputs)
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

    let top = matches.get_one::<String>("ir_top").map(|s| s.as_str());

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
    let output_json = matches.get_one::<String>("output_json");

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
        lhs_param_domains: None,
        rhs_param_domains: None,
        lhs_uf_map: std::collections::HashMap::new(),
        rhs_uf_map: std::collections::HashMap::new(),
    };

    let outcome = dispatch_ir_equiv(solver, tool_path, &inputs);
    if let Some(path) = output_json {
        std::fs::write(path, serde_json::to_string(&outcome).unwrap()).unwrap();
    }
    let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
    if outcome.success {
        println!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
        println!("[{}] success: Solver proved equivalence", SUBCOMMAND);
        std::process::exit(0);
    } else {
        eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
        if let Some(cex) = outcome.counterexample {
            eprintln!("[{}] failure: {}", SUBCOMMAND, cex);
        } else {
            eprintln!("[{}] failure", SUBCOMMAND);
        }
        std::process::exit(1);
    }
}
