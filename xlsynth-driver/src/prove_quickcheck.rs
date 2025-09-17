// SPDX-License-Identifier: Apache-2.0

//! Implements the `prove-quickcheck` sub-command – prove that every
//! `#[quickcheck]` function in a DSLX file (or a selected one) always
//! returns `true` for every possible input using an SMT solver.

// use std::collections::HashMap;

use crate::common::{infer_uf_signature, parse_uf_spec};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};

use crate::solver_choice::SolverChoice;
use regex::Regex;
use serde::Serialize;
use std::path::PathBuf;
use xlsynth::DslxConvertOptions;
use xlsynth_pir::ir_parser;
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

    let dslx_contents = match std::fs::read_to_string(input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read DSLX file {}: {}", input_path.display(), e);
            std::process::exit(1);
        }
    };

    // Gather quickcheck function names via parse+type-check.
    let module_name = input_path.file_stem().unwrap().to_str().unwrap();
    let additional_search_paths_refs: Vec<&std::path::Path> = additional_search_paths
        .iter()
        .map(|p| p.as_path())
        .collect();
    let mut import_data = xlsynth::dslx::ImportData::new(
        dslx_stdlib_path_buf.as_deref(),
        &additional_search_paths_refs,
    );
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
    let type_info = tcm.get_type_info();
    let mut quickchecks: Vec<(String, bool)> = Vec::new();
    for idx in 0..module.get_member_count() {
        if let Some(xlsynth::dslx::MatchableModuleMember::Quickcheck(qc)) =
            module.get_member(idx).to_matchable()
        {
            let function = qc.get_function();
            let fn_ident = function.get_identifier();
            if filter_regex
                .as_ref()
                .map(|re| re.is_match(&fn_ident))
                .unwrap_or(true)
            {
                let requires_itok = type_info
                    .requires_implicit_token(&function)
                    .expect("requires_implicit_token query");
                quickchecks.push((fn_ident, requires_itok));
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
        dslx_stdlib_path: dslx_stdlib_path_buf.as_deref(),
        additional_search_paths: additional_search_paths_refs.clone(),
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
    let pkg = ir_parser::Parser::new(&ir_text_result)
        .parse_package()
        .unwrap();

    let uf_sigs = infer_uf_signature(&pkg, &uf_map);

    // Helper: run proofs for a given prover over all quickchecks, using the
    // trait's DSLX-based entry point which handles implicit-token mangling.
    fn run_for_prover(
        prover: &dyn Prover,
        entry_file: &std::path::Path,
        qc_names: &[String],
        semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &std::collections::HashMap<String, String>,
        uf_sigs: &std::collections::HashMap<String, xlsynth_prover::types::UfSignature>,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
    ) -> Vec<QuickCheckTestOutcome> {
        let mut results: Vec<QuickCheckTestOutcome> = Vec::with_capacity(qc_names.len());
        for qc_name in qc_names {
            let start_time = std::time::Instant::now();
            let res = prover.prove_dslx_quickcheck_full(
                entry_file,
                dslx_stdlib_path,
                additional_search_paths,
                qc_name,
                semantics,
                assert_label_filter,
                uf_map,
                uf_sigs,
            );

            let micros = start_time.elapsed().as_micros();
            match res {
                BoolPropertyResult::Proved => results.push(QuickCheckTestOutcome {
                    name: qc_name.clone(),
                    time_micros: micros,
                    success: true,
                    counterexample: None,
                }),
                BoolPropertyResult::Disproved { inputs, output } => {
                    let cex_str = format!("inputs: {:?}, output: {:?}", inputs, output);
                    results.push(QuickCheckTestOutcome {
                        name: qc_name.clone(),
                        time_micros: micros,
                        success: false,
                        counterexample: Some(cex_str),
                    });
                }
                BoolPropertyResult::ToolchainDisproved(msg) => {
                    results.push(QuickCheckTestOutcome {
                        name: qc_name.clone(),
                        time_micros: micros,
                        success: false,
                        counterexample: Some(msg),
                    })
                }
            }
        }
        results
    }

    // (unused now; external quickchecks go via Prover::prove_dslx_quickcheck)

    let qc_names: Vec<String> = quickchecks.iter().map(|(n, _)| n.clone()).collect();

    let results: Vec<QuickCheckTestOutcome> = match solver_choice_opt {
        None => {
            let prover = xlsynth_prover::prover::auto_selected_prover();
            run_for_prover(
                &*prover,
                input_path,
                &qc_names,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                &uf_sigs,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
            )
        }
        Some(SolverChoice::Toolchain) => {
            let prover = xlsynth_prover::prover::ExternalProver::Toolchain;
            run_for_prover(
                &prover,
                input_path,
                &qc_names,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                &uf_sigs,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
            )
        }
        #[cfg(feature = "has-boolector")]
        Some(SolverChoice::Boolector) => {
            use xlsynth_prover::boolector_backend::BoolectorConfig;
            let prover = BoolectorConfig::new();
            run_for_prover(
                &prover,
                input_path,
                &qc_names,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                &uf_sigs,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
            )
        }
        #[cfg(feature = "has-bitwuzla")]
        Some(SolverChoice::Bitwuzla) => {
            use xlsynth_prover::bitwuzla_backend::BitwuzlaOptions;
            let prover = BitwuzlaOptions::new();
            run_for_prover(
                &prover,
                input_path,
                &qc_names,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                &uf_sigs,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
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
                &qc_names,
                *assertion_semantics,
                assert_label_filter.as_deref(),
                &uf_map,
                &uf_sigs,
                dslx_stdlib_path_buf.as_deref(),
                &additional_search_paths_refs,
            )
        }
    };

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
