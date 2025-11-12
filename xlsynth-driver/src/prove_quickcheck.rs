// SPDX-License-Identifier: Apache-2.0

//! Implements the `prove-quickcheck` sub-command – prove that every
//! `#[quickcheck]` function in a DSLX file (or a selected one) always
//! returns `true` for every possible input using an SMT solver.

use crate::common::parse_uf_spec;
use crate::proofs::obligations::{
    FileWithHistory, ObligationPayload, ProverObligation, QcObligation,
};
use crate::proofs::script::{
    execute_script, read_script_steps_from_json_path, read_script_steps_from_jsonl_path, OblTree,
    OblTreeConfig,
};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use xlsynth_prover::prover::types::{BoolPropertyResult, QuickCheckAssertionSemantics};
use xlsynth_prover::prover::{discover_quickcheck_tests, Prover, SolverChoice};

const SUBCOMMAND: &str = "prove-quickcheck";

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

    let tactic_json_path = matches.get_one::<String>("tactic_json").cloned();
    let tactic_jsonl_path = matches.get_one::<String>("tactic_jsonl").cloned();
    if tactic_json_path.is_some() && tactic_jsonl_path.is_some() {
        eprintln!(
            "[{}] Error: --tactic_json and --tactic_jsonl cannot be used together",
            SUBCOMMAND
        );
        std::process::exit(1);
    }
    let output_json = matches.get_one::<String>("output_json").cloned();

    if tactic_json_path.is_some() || tactic_jsonl_path.is_some() {
        let quickcheck_tests = match discover_quickcheck_tests(
            input_path,
            dslx_stdlib_path_buf.as_deref(),
            &additional_search_paths_refs,
            test_filter,
        ) {
            Ok(tests) => tests,
            Err(e) => {
                report_cli_error_and_exit(&e, Some(SUBCOMMAND), vec![("file", input_file_str)]);
            }
        };

        if quickcheck_tests.is_empty() {
            report_cli_error_and_exit(
                "No matching quickcheck functions found",
                Some(SUBCOMMAND),
                vec![("file", input_file_str)],
            );
        }

        let input_path_buf = input_path.to_path_buf();
        let qc_uf_map: BTreeMap<String, String> =
            uf_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let root_qc = QcObligation {
            file: FileWithHistory::from_path(&input_path_buf),
            tests: quickcheck_tests.clone(),
            uf_map: qc_uf_map,
        };

        let cfg = OblTreeConfig {
            dslx_stdlib_path: dslx_stdlib_path_buf.clone(),
            dslx_paths: additional_search_paths.clone(),
            solver: solver_choice_opt.clone(),
            timeout_ms: None,
        };

        let root_obligation = ProverObligation {
            selector_segment: String::new(),
            description: None,
            payload: ObligationPayload::QuickCheck(root_qc),
        };
        let mut tree = OblTree::new(root_obligation, cfg);

        let steps = if let Some(path) = tactic_json_path.as_ref() {
            match read_script_steps_from_json_path(path) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[{}] {}", SUBCOMMAND, e);
                    std::process::exit(2);
                }
            }
        } else if let Some(path) = tactic_jsonl_path.as_ref() {
            match read_script_steps_from_jsonl_path(path) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[{}] {}", SUBCOMMAND, e);
                    std::process::exit(2);
                }
            }
        } else {
            unreachable!("tactic script presence already checked");
        };

        let start = std::time::Instant::now();
        let report = match execute_script(&mut tree, &steps) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{}] execute_script error: {}", SUBCOMMAND, e);
                std::process::exit(2);
            }
        };
        let dur = start.elapsed();
        let success = report.failed.is_empty() && report.indefinite.is_empty();
        if let Some(path) = output_json.as_ref() {
            let json = serde_json::json!({
                "success": success,
                "report": report,
            });
            std::fs::write(path, serde_json::to_string(&json).unwrap()).unwrap();
        }
        if success {
            println!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
            println!("[{}] success: QuickCheck obligations proved", SUBCOMMAND);
            std::process::exit(0);
        } else {
            eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
            eprintln!("[{}] failure", SUBCOMMAND);
            std::process::exit(1);
        }
    }
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
        test_filter: Option<&str>,
    ) -> Vec<QuickCheckTestOutcome> {
        let runs = prover.prove_dslx_quickcheck(
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
                    BoolPropertyResult::ToolchainDisproved(msg)
                    | BoolPropertyResult::Error(msg) => (false, Some(msg)),
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

    let solver_choice = solver_choice_opt.unwrap_or(SolverChoice::Auto);
    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());
    let prover = xlsynth_prover::prover::prover_for_choice(
        solver_choice,
        tool_path.map(std::path::Path::new),
    );
    let results: Vec<QuickCheckTestOutcome> = run_for_prover(
        &*prover,
        input_path,
        *assertion_semantics,
        assert_label_filter.as_deref(),
        &uf_map,
        dslx_stdlib_path_buf.as_deref(),
        &additional_search_paths_refs,
        test_filter.as_deref(),
    );

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

    if let Some(path) = output_json.as_ref() {
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
