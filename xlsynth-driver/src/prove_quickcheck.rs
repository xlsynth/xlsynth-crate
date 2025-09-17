// SPDX-License-Identifier: Apache-2.0

//! Implements the `prove-quickcheck` sub-command – prove that every
//! `#[quickcheck]` function in a DSLX file (or a selected one) always
//! returns `true` for every possible input using an SMT solver.

// use std::collections::HashMap;

use crate::common::parse_uf_spec;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};

use crate::solver_choice::SolverChoice;
use regex::Regex;
use serde::Serialize;
use std::path::PathBuf;
use xlsynth_prover::prover::Prover;
use xlsynth_prover::types::{BoolPropertyResult, QuickCheckAssertionSemantics};

#[derive(Debug, Clone, Serialize)]
pub struct QuickCheckTestOutcome {
    pub name: String,
    pub time_micros: u128,
    pub success: bool,
    pub counterexample: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuickCheckOutcome {
    pub success: bool,
    pub tests: Vec<QuickCheckTestOutcome>,
}

/// Implements the `prove-quickcheck` sub-command.
pub fn handle_prove_quickcheck(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file_str = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file arg missing");
    let input_path = std::path::Path::new(input_file_str);

    // Gather optional assertion-label filter regex.
    let assert_label_filter = matches.get_one::<String>("assert_label_filter").cloned();

    let test_filter = matches.get_one::<String>("test_filter").map(|s| s.as_str());

    // Compile regex filter if provided; we enforce full-name match by anchoring.
    let filter_regex: Option<Regex> =
        test_filter.map(|pat| Regex::new(&format!("^{}$", pat)).unwrap());

    // DSLX search/stdlib path handling.
    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path_buf = dslx_stdlib_path.as_deref().map(PathBuf::from);
    let dslx_path = get_dslx_path(matches, config);
    let additional_search_paths: Vec<PathBuf> = dslx_path
        .as_deref()
        .map(|s| {
            s.split(';')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default();

    let additional_search_paths_refs: Vec<&std::path::Path> = additional_search_paths
        .iter()
        .map(|p| p.as_path())
        .collect();

    // Solver selection (optional). If omitted, use auto-selected prover.
    let solver_choice_opt: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());

    // Assertion semantics.
    let assertion_semantics = matches
        .get_one::<QuickCheckAssertionSemantics>("assertion_semantics")
        .unwrap_or(&QuickCheckAssertionSemantics::Ignore);

    let module_name = input_path.file_stem().unwrap().to_str().unwrap();
    let uf_map = parse_uf_spec(module_name, matches.get_many::<String>("uf"));
    // UF semantics: functions mapped to the same <uf_name> are treated as the
    // same uninterpreted symbol (assumed equivalent) and assertions inside them
    // are ignored during proving.
    // Helper: run proofs for a given prover over all quickchecks, using the
    // trait's DSLX-based entry point which handles implicit-token mangling.
    fn run_for_prover(
        prover: &dyn Prover,
        entry_file: &std::path::Path,
        semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &std::collections::HashMap<String, String>,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&Regex>,
    ) -> Vec<QuickCheckTestOutcome> {
        let runs = prover.prove_dslx_quickcheck_full(
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
            semantics,
            assert_label_filter,
            uf_map,
        );

        runs.into_iter()
            .map(|run| {
                let (success, counterexample) = match run.result {
                    BoolPropertyResult::Proved => (true, None),
                    BoolPropertyResult::Disproved { inputs, output } => {
                        let cex_str = format!("inputs: {:?}, output: {:?}", inputs, output);
                        (false, Some(cex_str))
                    }
                    BoolPropertyResult::ToolchainDisproved(msg) => (false, Some(msg)),
                };

                QuickCheckTestOutcome {
                    name: run.name,
                    time_micros: run.duration.as_micros(),
                    success,
                    counterexample,
                }
            })
            .collect()
    }

    let results: Vec<QuickCheckTestOutcome> = match solver_choice_opt {
        None => {
            let prover = xlsynth_prover::prover::auto_selected_prover();
            run_for_prover(
                &*prover,
                input_path,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
                filter_regex.as_ref(),
            )
        }
        Some(SolverChoice::Toolchain) => {
            let prover = xlsynth_prover::prover::ExternalProver::Toolchain;
            run_for_prover(
                &prover,
                input_path,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
                filter_regex.as_ref(),
            )
        }
        #[cfg(feature = "has-boolector")]
        Some(SolverChoice::Boolector) => {
            use xlsynth_prover::boolector_backend::BoolectorConfig;
            let prover = BoolectorConfig::new();
            run_for_prover(
                &prover,
                input_path,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
                filter_regex.as_ref(),
            )
        }
        #[cfg(feature = "has-bitwuzla")]
        Some(SolverChoice::Bitwuzla) => {
            use xlsynth_prover::bitwuzla_backend::BitwuzlaOptions;
            let prover = BitwuzlaOptions::new();
            run_for_prover(
                &prover,
                input_path,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
                filter_regex.as_ref(),
            )
        }
        #[cfg(feature = "has-easy-smt")]
        Some(SolverChoice::Z3Binary)
        | Some(SolverChoice::BitwuzlaBinary)
        | Some(SolverChoice::BoolectorBinary) => {
            use xlsynth_prover::easy_smt_backend::EasySmtConfig;
            let prover = match solver_choice_opt.unwrap() {
                SolverChoice::Z3Binary => EasySmtConfig::z3(),
                SolverChoice::BitwuzlaBinary => EasySmtConfig::bitwuzla(),
                SolverChoice::BoolectorBinary => EasySmtConfig::boolector(),
                _ => unreachable!(),
            };
            run_for_prover(
                &prover,
                input_path,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
                filter_regex.as_ref(),
            )
        }
    };

    if results.is_empty() {
        report_cli_error_and_exit(
            "No matching quickcheck functions found",
            Some("prove-quickcheck"),
            vec![("file", input_file_str)],
        );
    }

    let mut all_passed = true;
    for r in &results {
        if r.success {
            println!("QuickCheck '{}' proved", r.name);
        } else {
            all_passed = false;
            println!("QuickCheck '{}' disproved", r.name);
            if let Some(ref cex) = r.counterexample {
                println!("  {}", cex);
            }
        }
    }

    let outcome = QuickCheckOutcome {
        success: all_passed,
        tests: results,
    };

    let output_json = matches.get_one::<String>("output_json");
    if let Some(path) = output_json {
        std::fs::write(path, serde_json::to_string(&outcome).unwrap()).unwrap();
    }

    if all_passed {
        println!("Success: All QuickChecks proved");
        std::process::exit(0);
    } else {
        println!("Failure: Some QuickChecks disproved");
        std::process::exit(1);
    }
}
