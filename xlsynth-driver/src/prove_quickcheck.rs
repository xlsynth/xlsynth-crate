// SPDX-License-Identifier: Apache-2.0

//! Implements the `prove-quickcheck` sub-command – prove that every
//! `#[quickcheck]` function in a DSLX file (or a selected one) always
//! returns `true` for every possible input using an SMT solver.

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};

use crate::solver_choice::SolverChoice;
use crate::tools::run_prove_quickcheck_main;
use regex::Regex;
use xlsynth::{mangle_dslx_name, DslxConvertOptions};
use xlsynth_g8r::equiv::prove_equiv::IrFn;
use xlsynth_g8r::equiv::prove_quickcheck::{
    prove_ir_fn_always_true, BoolPropertyResult, QuickCheckAssertionSemantics,
};
use xlsynth_g8r::equiv::solver_interface::Solver;

/// Implements the `prove-quickcheck` sub-command.
pub fn handle_prove_quickcheck(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file_str = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file arg missing");
    let input_path = std::path::Path::new(input_file_str);

    let test_filter = matches.get_one::<String>("test_filter").map(|s| s.as_str());

    // Compile regex filter if provided; we enforce full-name match by anchoring.
    let filter_regex: Option<Regex> =
        test_filter.map(|pat| Regex::new(&format!("^{}$", pat)).unwrap());

    // DSLX search/stdlib path handling.
    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_path = get_dslx_path(matches, config);

    let dslx_contents = match std::fs::read_to_string(input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read DSLX file {}: {}", input_path.display(), e);
            std::process::exit(1);
        }
    };

    // Gather quickcheck function names via parse+type-check.
    let module_name = input_path.file_stem().unwrap().to_str().unwrap();
    let mut import_data = {
        let stdlib_opt = dslx_stdlib_path.as_deref().map(|p| std::path::Path::new(p));
        let addl_paths: Vec<&std::path::Path> = dslx_path
            .as_deref()
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        xlsynth::dslx::ImportData::new(stdlib_opt, &addl_paths)
    };
    let tcm = match xlsynth::dslx::parse_and_typecheck(
        &dslx_contents,
        input_path.to_str().unwrap(),
        module_name,
        &mut import_data,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("DSLX parse/type-check failed: {}", e);
            std::process::exit(1);
        }
    };
    let module = tcm.get_module();
    let mut quickchecks: Vec<String> = Vec::new();
    for idx in 0..module.get_member_count() {
        if let Some(xlsynth::dslx::MatchableModuleMember::Quickcheck(qc)) =
            module.get_member(idx).to_matchable()
        {
            let fn_ident = qc.get_function().get_identifier();
            if filter_regex
                .as_ref()
                .map(|re| re.is_match(&fn_ident))
                .unwrap_or(true)
            {
                quickchecks.push(fn_ident);
            }
        }
    }
    if quickchecks.is_empty() {
        report_cli_error_and_exit(
            "No matching quickcheck functions found",
            Some("prove-quickcheck"),
            vec![("file", input_file_str)],
        );
    }

    // Convert whole module to IR text once.
    let options = DslxConvertOptions {
        dslx_stdlib_path: dslx_stdlib_path.as_deref().map(|p| std::path::Path::new(p)),
        additional_search_paths: dslx_path
            .as_deref()
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default(),
        enable_warnings: None,
        disable_warnings: None,
    };
    let ir_text_result =
        match xlsynth::convert_dslx_to_ir_text(&dslx_contents, input_path, &options) {
            Ok(r) => r.ir,
            Err(e) => {
                eprintln!("DSLX->IR conversion failed: {}", e);
                std::process::exit(1);
            }
        };

    // Solver selection.
    let solver_choice_opt: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    // If user did not specify --solver we default to toolchain (external prover) if
    // available.
    let solver_choice = solver_choice_opt.unwrap_or_else(|| {
        if tool_path.is_some() {
            SolverChoice::Toolchain
        } else {
            report_cli_error_and_exit(
                "No solver specified and no toolchain available",
                Some("prove-quickcheck"),
                vec![],
            );
        }
    });

    // Assertion semantics.
    let assertion_semantics = matches
        .get_one::<QuickCheckAssertionSemantics>("assertion_semantics")
        .unwrap_or(&QuickCheckAssertionSemantics::Ignore);

    // Helper closure that runs proof for a given solver type.
    fn run_for_solver<S: Solver>(
        config: &S::Config,
        qc_names: &[String],
        ir_text: &str,
        module_name: &str,
        semantics: QuickCheckAssertionSemantics,
    ) -> ! {
        let mut all_passed = true;
        for qc_name in qc_names {
            let mangled = mangle_dslx_name(module_name, qc_name).unwrap();
            // Optimize IR with this QC as top.
            let base_pkg = xlsynth::IrPackage::parse_ir(ir_text, None).unwrap();
            let optimized_pkg = xlsynth::optimize_ir(&base_pkg, &mangled).unwrap();
            let optimized_text = optimized_pkg.to_string();
            let g8_pkg = xlsynth_g8r::xls_ir::ir_parser::Parser::new(&optimized_text)
                .parse_package()
                .unwrap();
            let ir_fn_ref = g8_pkg.get_fn(&mangled).unwrap();

            let ir_fn = IrFn {
                fn_ref: ir_fn_ref,
                fixed_implicit_activation: false,
            };
            let res = prove_ir_fn_always_true::<S>(config, &ir_fn, semantics.clone());
            match res {
                BoolPropertyResult::Proved => {
                    println!("QuickCheck '{}' proved", qc_name);
                }
                BoolPropertyResult::Disproved { inputs, output } => {
                    all_passed = false;
                    println!("QuickCheck '{}' disproved", qc_name);
                    println!("  Inputs: {:?}", inputs);
                    println!("  Output: {:?}", output);
                }
            }
        }
        if all_passed {
            println!("Success: All QuickChecks proved");
            std::process::exit(0);
        } else {
            println!("Failure: Some QuickChecks disproved");
            std::process::exit(1);
        }
    }

    match solver_choice {
        SolverChoice::Toolchain => {
            let tool_path = tool_path.expect("tool_path required for Toolchain solver");
            let mut all_passed = true;
            for qc_name in &quickchecks {
                match run_prove_quickcheck_main(input_path, Some(qc_name), tool_path) {
                    Ok(stdout) => {
                        println!("QuickCheck '{}' proved: {}", qc_name, stdout.trim());
                    }
                    Err(output) => {
                        all_passed = false;
                        println!(
                            "QuickCheck '{}' disproved: {}",
                            qc_name,
                            String::from_utf8_lossy(&output.stderr)
                        );
                    }
                }
            }
            if all_passed {
                println!("Success: All QuickChecks proved");
                std::process::exit(0);
            } else {
                println!("Failure: Some QuickChecks disproved");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "has-boolector")]
        SolverChoice::Boolector | SolverChoice::BoolectorLegacy => {
            use xlsynth_g8r::equiv::boolector_backend::{Boolector, BoolectorConfig};
            let cfg = BoolectorConfig::new();
            run_for_solver::<Boolector>(
                &cfg,
                &quickchecks,
                &ir_text_result,
                module_name,
                *assertion_semantics,
            );
        }
        #[cfg(feature = "has-bitwuzla")]
        SolverChoice::Bitwuzla => {
            use xlsynth_g8r::equiv::bitwuzla_backend::{Bitwuzla, BitwuzlaOptions};
            let opts = BitwuzlaOptions::new();
            run_for_solver::<Bitwuzla>(
                &opts,
                &quickchecks,
                &ir_text_result,
                module_name,
                *assertion_semantics,
            );
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
            run_for_solver::<EasySmtSolver>(
                &cfg,
                &quickchecks,
                &ir_text_result,
                module_name,
                *assertion_semantics,
            );
        }
    }
}
