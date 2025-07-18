// SPDX-License-Identifier: Apache-2.0

use crate::parallelism::ParallelismStrategy;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::solver_choice::SolverChoice;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};
use crate::tools::{run_check_ir_equivalence_main, run_ir_converter_main, run_opt_main};
use xlsynth::{mangle_dslx_name, DslxConvertOptions, IrPackage};
use xlsynth_g8r::equiv::prove_equiv::{
    prove_ir_fn_equiv, prove_ir_fn_equiv_output_bits_parallel, prove_ir_fn_equiv_split_input_bit,
    AssertionSemantics, EquivResult, IrFn,
};
use xlsynth_g8r::equiv::solver_interface::Solver;

const SUBCOMMAND: &str = "dslx-equiv";

fn run_dslx_toolchain_equiv(lhs_ir: &str, rhs_ir: &str, top: &str, tool_path: &str) -> ! {
    // Write IRs to temp files then invoke external equivalence checker.
    let lhs_tmp = tempfile::NamedTempFile::new().unwrap();
    let rhs_tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(lhs_tmp.path(), lhs_ir).unwrap();
    std::fs::write(rhs_tmp.path(), rhs_ir).unwrap();
    let output =
        run_check_ir_equivalence_main(lhs_tmp.path(), rhs_tmp.path(), Some(top), tool_path);
    match output {
        Ok(stdout) => {
            println!("success: {}", stdout.trim());
            std::process::exit(0);
        }
        Err(output) => {
            // stdout contains counterexample details.
            let mut message = String::from_utf8_lossy(&output.stdout);
            if message.is_empty() {
                message = String::from_utf8_lossy(&output.stderr);
            }
            report_cli_error_and_exit(
                &format!("failure: {}", message),
                Some(SUBCOMMAND),
                vec![
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

fn build_ir_for_dslx(
    input_path: &std::path::Path,
    dslx_top: &str,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    enable_warnings: Option<&[String]>,
    disable_warnings: Option<&[String]>,
    tool_path: Option<&str>,
    type_inference_v2: Option<bool>,
    opt: bool,
) -> (String, String) {
    // Returns (optimized_ir_text, mangled_top_name)
    let module_name = input_path.file_stem().unwrap().to_str().unwrap();
    let mangled_top = mangle_dslx_name(module_name, dslx_top).unwrap();

    if let Some(tool_path) = tool_path {
        // Use external tool to convert; optionally optimize.
        let mut ir_text = run_ir_converter_main(
            input_path,
            Some(dslx_top),
            dslx_stdlib_path,
            dslx_path,
            tool_path,
            enable_warnings,
            disable_warnings,
            type_inference_v2,
        );
        if opt {
            // Optimize using external tool.
            let tmp = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(tmp.path(), &ir_text).unwrap();
            ir_text = run_opt_main(tmp.path(), Some(&mangled_top), tool_path);
        }
        (ir_text, mangled_top)
    } else {
        if type_inference_v2 == Some(true) {
            eprintln!("error: --type_inference_v2 is only supported with external toolchain");
            std::process::exit(1);
        }
        let dslx_contents = std::fs::read_to_string(input_path).unwrap_or_else(|e| {
            eprintln!("Failed to read DSLX file {}: {}", input_path.display(), e);
            std::process::exit(1);
        });
        let dslx_stdlib_path: Option<&std::path::Path> =
            dslx_stdlib_path.map(|s| std::path::Path::new(s));
        let additional_search_paths: Vec<&std::path::Path> = dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let result = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            input_path,
            &DslxConvertOptions {
                dslx_stdlib_path,
                additional_search_paths,
                enable_warnings,
                disable_warnings,
            },
        )
        .expect("successful DSLX->IR conversion");
        let ir_text = if opt {
            let pkg = IrPackage::parse_ir(&result.ir, Some(&mangled_top)).unwrap();
            let optimized = xlsynth::optimize_ir(&pkg, &mangled_top).unwrap();
            optimized.to_string()
        } else {
            result.ir
        };
        (ir_text, mangled_top)
    }
}

fn prove_equiv_internal<S: Solver>(
    solver_cfg: &S::Config,
    lhs_ir_text: &str,
    rhs_ir_text: &str,
    lhs_top: &str,
    rhs_top: &str,
    lhs_fixed_implicit_activation: bool,
    rhs_fixed_implicit_activation: bool,
    flatten_aggregates: bool,
    drop_params: &[String],
    strategy: ParallelismStrategy,
    assertion_semantics: AssertionSemantics,
) -> ! {
    // Parse both IR packages through g8r parser so we can re-use existing proving
    // code.
    let lhs_pkg = xlsynth_g8r::xls_ir::ir_parser::Parser::new(lhs_ir_text)
        .parse_package()
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse LHS IR: {}", e);
            std::process::exit(1);
        });
    let rhs_pkg = xlsynth_g8r::xls_ir::ir_parser::Parser::new(rhs_ir_text)
        .parse_package()
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse RHS IR: {}", e);
            std::process::exit(1);
        });
    let lhs_fn_owned = lhs_pkg.get_fn(lhs_top).cloned().unwrap_or_else(|| {
        eprintln!("Top function '{}' not found in LHS IR", lhs_top);
        std::process::exit(1);
    });
    let rhs_fn_owned = rhs_pkg.get_fn(rhs_top).cloned().unwrap_or_else(|| {
        eprintln!("Top function '{}' not found in RHS IR", rhs_top);
        std::process::exit(1);
    });
    let lhs_fn_dropped = lhs_fn_owned
        .drop_params(drop_params)
        .expect("Dropped parameter used in LHS body");
    let rhs_fn_dropped = rhs_fn_owned
        .drop_params(drop_params)
        .expect("Dropped parameter used in RHS body");
    let lhs_ir_fn = IrFn {
        fn_ref: &lhs_fn_dropped,
        fixed_implicit_activation: lhs_fixed_implicit_activation,
    };
    let rhs_ir_fn = IrFn {
        fn_ref: &rhs_fn_dropped,
        fixed_implicit_activation: rhs_fixed_implicit_activation,
    };

    let start = std::time::Instant::now();
    let result = match strategy {
        ParallelismStrategy::SingleThreaded => prove_ir_fn_equiv::<S>(
            solver_cfg,
            &lhs_ir_fn,
            &rhs_ir_fn,
            assertion_semantics,
            flatten_aggregates,
        ),
        ParallelismStrategy::OutputBits => prove_ir_fn_equiv_output_bits_parallel::<S>(
            solver_cfg,
            &lhs_ir_fn,
            &rhs_ir_fn,
            assertion_semantics,
            flatten_aggregates,
        ),
        ParallelismStrategy::InputBitSplit => prove_ir_fn_equiv_split_input_bit::<S>(
            solver_cfg,
            &lhs_ir_fn,
            &rhs_ir_fn,
            0,
            0,
            assertion_semantics,
            flatten_aggregates,
        ),
    };
    let end = std::time::Instant::now();
    println!("Time taken: {:?}", end.duration_since(start));
    match result {
        EquivResult::Proved => {
            println!("success: Solver proved equivalence");
            std::process::exit(0);
        }
        EquivResult::Disproved {
            inputs,
            outputs: (lhs_bits, rhs_bits),
        } => {
            println!("failure: Solver found counterexample: {:?}", inputs);
            println!("    output LHS: {:?}", lhs_bits);
            println!("    output RHS: {:?}", rhs_bits);
            std::process::exit(1);
        }
    }
}

pub fn handle_dslx_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let lhs_file = matches.get_one::<String>("lhs_dslx_file").unwrap();
    let rhs_file = matches.get_one::<String>("rhs_dslx_file").unwrap();

    let mut lhs_top = matches
        .get_one::<String>("lhs_dslx_top")
        .map(|s| s.as_str());
    let mut rhs_top = matches
        .get_one::<String>("rhs_dslx_top")
        .map(|s| s.as_str());

    let shared_top = matches.get_one::<String>("dslx_top").map(|s| s.as_str());

    if shared_top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        eprintln!("Error: --top and --lhs_dslx_top/--rhs_dslx_top cannot be used together");
        std::process::exit(1);
    }
    if lhs_top.is_some() ^ rhs_top.is_some() {
        eprintln!("Error: --lhs_dslx_top and --rhs_dslx_top must be used together");
        std::process::exit(1);
    }
    if shared_top.is_some() {
        lhs_top = shared_top;
        rhs_top = shared_top;
    }

    if lhs_top.is_none() || rhs_top.is_none() {
        eprintln!("Error: a top function must be specified via --top or both --lhs_dslx_top/--rhs_dslx_top");
        std::process::exit(1);
    }

    let lhs_path = std::path::Path::new(lhs_file);
    let rhs_path = std::path::Path::new(rhs_file);

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path = dslx_stdlib_path.as_deref();
    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    let enable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.enable_warnings.as_deref());
    let disable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.disable_warnings.as_deref());

    let type_inference_v2 = matches
        .get_one::<String>("type_inference_v2")
        .map(|s| s == "true")
        .or_else(|| {
            config
                .as_ref()
                .and_then(|c| c.dslx.as_ref()?.type_inference_v2)
        });

    let opt = matches
        .get_one::<String>("opt")
        .map(|s| s == "true")
        .unwrap_or(false);

    let assertion_semantics = matches
        .get_one::<String>("assertion_semantics")
        .map(|s| s.parse().unwrap())
        .unwrap_or(AssertionSemantics::Same);

    let solver_choice: Option<SolverChoice> = matches
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

    // Build IRs (possibly using external converter) always; if solver is toolchain
    // we may re-use optimized IRs for temp files.
    let (lhs_ir_text_raw, lhs_mangled_top) = build_ir_for_dslx(
        lhs_path,
        lhs_top.unwrap(),
        dslx_stdlib_path,
        dslx_path,
        enable_warnings,
        disable_warnings,
        tool_path,
        type_inference_v2,
        opt,
    );
    let (rhs_ir_text_raw, rhs_mangled_top) = build_ir_for_dslx(
        rhs_path,
        rhs_top.unwrap(),
        dslx_stdlib_path,
        dslx_path,
        enable_warnings,
        disable_warnings,
        tool_path,
        type_inference_v2,
        opt,
    );

    // Always optimize before proving equivalence to normalize IR and remove
    // constructs the g8r parser / equivalence engine may not yet support (e.g.
    // counted_for).
    let lhs_ir_text = if opt {
        lhs_ir_text_raw.clone()
    } else if let Some(tp) = tool_path {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &lhs_ir_text_raw).unwrap();
        run_opt_main(tmp.path(), Some(&lhs_mangled_top), tp)
    } else {
        let pkg = IrPackage::parse_ir(&lhs_ir_text_raw, Some(&lhs_mangled_top)).unwrap();
        xlsynth::optimize_ir(&pkg, &lhs_mangled_top)
            .unwrap()
            .to_string()
    };
    let rhs_ir_text = if opt {
        rhs_ir_text_raw.clone()
    } else if let Some(tp) = tool_path {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &rhs_ir_text_raw).unwrap();
        run_opt_main(tmp.path(), Some(&rhs_mangled_top), tp)
    } else {
        let pkg = IrPackage::parse_ir(&rhs_ir_text_raw, Some(&rhs_mangled_top)).unwrap();
        xlsynth::optimize_ir(&pkg, &rhs_mangled_top)
            .unwrap()
            .to_string()
    };

    if let Some(solver) = solver_choice {
        match solver {
            #[cfg(feature = "has-boolector")]
            SolverChoice::Boolector => {
                use xlsynth_g8r::equiv::boolector_backend::{Boolector, BoolectorConfig};
                let cfg = BoolectorConfig::new();
                prove_equiv_internal::<Boolector>(
                    &cfg,
                    &lhs_ir_text,
                    &rhs_ir_text,
                    &lhs_mangled_top,
                    &rhs_mangled_top,
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
                let cfg = match solver {
                    SolverChoice::Z3Binary => EasySmtConfig::z3(),
                    SolverChoice::BitwuzlaBinary => EasySmtConfig::bitwuzla(),
                    SolverChoice::BoolectorBinary => EasySmtConfig::boolector(),
                    _ => unreachable!(),
                };
                prove_equiv_internal::<EasySmtSolver>(
                    &cfg,
                    &lhs_ir_text,
                    &rhs_ir_text,
                    &lhs_mangled_top,
                    &rhs_mangled_top,
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
                let opts = BitwuzlaOptions::new();
                prove_equiv_internal::<Bitwuzla>(
                    &opts,
                    &lhs_ir_text,
                    &rhs_ir_text,
                    &lhs_mangled_top,
                    &rhs_mangled_top,
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
                if lhs_fixed_implicit_activation || rhs_fixed_implicit_activation {
                    eprintln!("Error: fixed implicit activation flags not supported for boolector-legacy solver");
                    std::process::exit(1);
                }
                if assertion_semantics != AssertionSemantics::Same {
                    eprintln!("Error: assertion semantics other than 'same' not supported for boolector-legacy solver");
                    std::process::exit(1);
                }
                use xlsynth_g8r::ir_equiv_boolector;
                // Use legacy path directly with parsed IR (boolector legacy expects flattened
                // style similar to ir_equiv.rs)
                let lhs_pkg = xlsynth_g8r::xls_ir::ir_parser::Parser::new(&lhs_ir_text)
                    .parse_package()
                    .unwrap();
                let rhs_pkg = xlsynth_g8r::xls_ir::ir_parser::Parser::new(&rhs_ir_text)
                    .parse_package()
                    .unwrap();
                let lhs_fn_owned = lhs_pkg.get_fn(&lhs_mangled_top).cloned().unwrap();
                let rhs_fn_owned = rhs_pkg.get_fn(&rhs_mangled_top).cloned().unwrap();
                let lhs_fn_dropped = lhs_fn_owned
                    .drop_params(&drop_params)
                    .expect("Dropped parameter used in LHS body");
                let rhs_fn_dropped = rhs_fn_owned
                    .drop_params(&drop_params)
                    .expect("Dropped parameter used in RHS body");
                let start = std::time::Instant::now();
                let result = match strategy {
                    ParallelismStrategy::SingleThreaded => ir_equiv_boolector::prove_ir_fn_equiv(
                        &lhs_fn_dropped,
                        &rhs_fn_dropped,
                        flatten_aggregates,
                    ),
                    ParallelismStrategy::OutputBits => {
                        ir_equiv_boolector::prove_ir_fn_equiv_output_bits_parallel(
                            &lhs_fn_dropped,
                            &rhs_fn_dropped,
                            flatten_aggregates,
                        )
                    }
                    ParallelismStrategy::InputBitSplit => {
                        ir_equiv_boolector::prove_ir_fn_equiv_split_input_bit(
                            &lhs_fn_dropped,
                            &rhs_fn_dropped,
                            0,
                            0,
                            flatten_aggregates,
                        )
                    }
                };
                let end = std::time::Instant::now();
                println!("Time taken: {:?}", end.duration_since(start));
                match result {
                    ir_equiv_boolector::EquivResult::Proved => {
                        println!("success: Solver proved equivalence");
                        std::process::exit(0);
                    }
                    ir_equiv_boolector::EquivResult::Disproved {
                        inputs,
                        outputs: (lhs_bits, rhs_bits),
                    } => {
                        println!("failure: Solver found counterexample: {:?}", inputs);
                        println!("    output LHS: {:?}", lhs_bits);
                        println!("    output RHS: {:?}", rhs_bits);
                        std::process::exit(1);
                    }
                }
            }
            SolverChoice::Toolchain => {
                let tool_path = tool_path.expect("tool_path required for toolchain solver");
                if lhs_mangled_top != rhs_mangled_top {
                    eprintln!("Error: with toolchain solver, LHS/RHS mangled top names must match (module and function must be same)");
                    std::process::exit(1);
                }
                run_dslx_toolchain_equiv(&lhs_ir_text, &rhs_ir_text, &lhs_mangled_top, tool_path);
            }
        }
    } else {
        // Default to toolchain if available.
        if let Some(tool_path) = tool_path {
            if lhs_mangled_top != rhs_mangled_top {
                eprintln!("Error: with toolchain solver, LHS/RHS mangled top names must match (module and function must be same)");
                std::process::exit(1);
            }
            run_dslx_toolchain_equiv(&lhs_ir_text, &rhs_ir_text, &lhs_mangled_top, tool_path);
        } else {
            report_cli_error_and_exit(
                "No solver specified and no toolchain available",
                Some(SUBCOMMAND),
                vec![],
            );
        }
    }
}
