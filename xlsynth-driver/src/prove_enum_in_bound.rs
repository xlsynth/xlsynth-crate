// SPDX-License-Identifier: Apache-2.0

use crate::common::get_function_enum_param_domains;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};

use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use xlsynth::dslx::ImportData;
use xlsynth::{
    mangle_dslx_name_with_calling_convention, DslxCallingConvention, DslxConvertOptions,
};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::enum_in_bound::{prove_enum_in_bound, ASSERT_LABEL_PREFIX};
use xlsynth_prover::prover::{prover_for_choice, SolverChoice};
use xlsynth_prover::types::{BoolPropertyResult, FnInput, FnOutput};

const SUBCOMMAND: &str = "prove-enum-in-bound";

#[derive(Debug, Serialize)]
struct SerializedInput {
    name: String,
    value: String,
}

#[derive(Debug, Serialize)]
struct SerializedOutput {
    value: String,
    assertion_label: Option<String>,
    assertion_message: Option<String>,
}

#[derive(Debug, Serialize)]
struct SerializedCounterexample {
    inputs: Vec<SerializedInput>,
    output: SerializedOutput,
}

#[derive(Debug, Serialize)]
struct EnumInBoundOutcome {
    success: bool,
    counterexample: Option<SerializedCounterexample>,
    assert_label_prefix: &'static str,
}

fn serialize_inputs(inputs: &[FnInput]) -> Vec<SerializedInput> {
    inputs
        .iter()
        .map(|input| SerializedInput {
            name: input.name.clone(),
            value: format!("{:?}", input.value),
        })
        .collect()
}

fn serialize_output(output: &FnOutput) -> SerializedOutput {
    let (assertion_label, assertion_message) = output
        .assertion_violation
        .as_ref()
        .map(|vio| (Some(vio.label.clone()), Some(vio.message.clone())))
        .unwrap_or((None, None));
    SerializedOutput {
        value: format!("{:?}", output.value),
        assertion_label,
        assertion_message,
    }
}

fn collect_additional_search_paths(dslx_path: Option<&str>) -> Vec<PathBuf> {
    dslx_path
        .map(|s| {
            s.split(';')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_targets(matches: &clap::ArgMatches) -> Vec<String> {
    matches
        .get_many::<String>("target")
        .map(|vals| vals.map(|s| s.trim().to_string()).collect())
        .unwrap_or_default()
}

pub fn handle_prove_enum_in_bound(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file arg missing");
    let input_path = Path::new(input_file);

    let dslx_top = matches
        .get_one::<String>("dslx_top")
        .expect("dslx_top argument missing");

    let targets = parse_targets(matches);
    if targets.is_empty() {
        report_cli_error_and_exit(
            "At least one --target must be specified",
            Some(SUBCOMMAND),
            vec![("file", input_file.as_str())],
        );
    }

    let solver_choice = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap())
        .unwrap_or(SolverChoice::Auto);

    if let SolverChoice::Toolchain = solver_choice {
        report_cli_error_and_exit(
            "Solver 'toolchain' is not supported for this command",
            Some(SUBCOMMAND),
            vec![("solver", "toolchain")],
        );
    }

    let dslx_stdlib_path_opt = get_dslx_stdlib_path(matches, config);
    let dslx_path_opt = get_dslx_path(matches, config);

    let additional_search_paths = collect_additional_search_paths(dslx_path_opt.as_deref());
    let additional_search_path_refs: Vec<&Path> = additional_search_paths
        .iter()
        .map(|p| p.as_path())
        .collect();

    let enable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.enable_warnings.as_deref());
    let disable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.disable_warnings.as_deref());

    let dslx_contents = std::fs::read_to_string(input_path).unwrap_or_else(|err| {
        report_cli_error_and_exit(
            &format!("Failed to read DSLX file: {}", err),
            Some(SUBCOMMAND),
            vec![("file", input_file.as_str())],
        );
    });

    let module_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("module name");

    let stdlib_path_ref = dslx_stdlib_path_opt.as_deref().map(Path::new);
    let mut import_data = ImportData::new(stdlib_path_ref, &additional_search_path_refs);
    let typechecked = xlsynth::dslx::parse_and_typecheck(
        &dslx_contents,
        input_path
            .to_str()
            .expect("DSLX input path must be valid UTF-8"),
        module_name,
        &mut import_data,
    )
    .unwrap_or_else(|err| {
        report_cli_error_and_exit(
            &format!("DSLX parse/typecheck failed: {}", err),
            Some(SUBCOMMAND),
            vec![("file", input_file.as_str())],
        );
    });

    let mut target_domains: HashMap<String, xlsynth_prover::types::ParamDomains> = HashMap::new();
    for target in &targets {
        let domains = get_function_enum_param_domains(&typechecked, target).unwrap_or_else(|err| {
            report_cli_error_and_exit(&err, Some(SUBCOMMAND), vec![("target", target.as_str())]);
        });
        let mangled = mangle_dslx_name_with_calling_convention(
            module_name,
            target,
            DslxCallingConvention::ImplicitToken,
        )
        .unwrap_or_else(|err| {
            report_cli_error_and_exit(
                &format!("Failed to mangle function name '{}': {}", target, err),
                Some(SUBCOMMAND),
                vec![("target", target.as_str())],
            );
        });
        target_domains.insert(mangled, domains);
    }

    let options = DslxConvertOptions {
        dslx_stdlib_path: stdlib_path_ref,
        additional_search_paths: additional_search_path_refs.clone(),
        enable_warnings,
        disable_warnings,
        force_implicit_token_calling_convention: true,
    };

    let ir_text = xlsynth::convert_dslx_to_ir_text(&dslx_contents, input_path, &options)
        .unwrap_or_else(|err| {
            report_cli_error_and_exit(
                &format!("DSLX->IR conversion failed: {}", err),
                Some(SUBCOMMAND),
                vec![("file", input_file.as_str())],
            );
        })
        .ir;

    let mut parser = Parser::new(&ir_text);
    let package = parser.parse_package().unwrap_or_else(|err| {
        report_cli_error_and_exit(
            &format!("Failed to parse IR package: {}", err),
            Some(SUBCOMMAND),
            vec![("file", input_file.as_str())],
        );
    });

    let mangled_top = mangle_dslx_name_with_calling_convention(
        module_name,
        dslx_top,
        DslxCallingConvention::ImplicitToken,
    )
    .unwrap_or_else(|err| {
        report_cli_error_and_exit(
            &format!("Failed to mangle top function '{}': {}", dslx_top, err),
            Some(SUBCOMMAND),
            vec![("top", dslx_top.as_str())],
        );
    });

    let prover = prover_for_choice(solver_choice, None);

    let top_domains =
        get_function_enum_param_domains(&typechecked, dslx_top).unwrap_or_else(|err| {
            report_cli_error_and_exit(&err, Some(SUBCOMMAND), vec![("top", dslx_top.as_str())]);
        });

    let proof_result = prove_enum_in_bound(
        &*prover,
        &package,
        &mangled_top,
        &target_domains,
        Some(&top_domains),
    );

    let output_json_path = matches.get_one::<String>("output_json");

    match proof_result {
        BoolPropertyResult::Proved => {
            println!(
                "Success: All instrumentation assertions with label prefix '{}' hold",
                ASSERT_LABEL_PREFIX
            );
            if let Some(path) = output_json_path {
                let outcome = EnumInBoundOutcome {
                    success: true,
                    counterexample: None,
                    assert_label_prefix: ASSERT_LABEL_PREFIX,
                };
                std::fs::write(path, serde_json::to_string_pretty(&outcome).unwrap())
                    .unwrap_or_else(|err| {
                        report_cli_error_and_exit(
                            &format!("Failed to write JSON output: {}", err),
                            Some(SUBCOMMAND),
                            vec![("path", path.as_str())],
                        );
                    });
            }
            std::process::exit(0);
        }
        BoolPropertyResult::Disproved { inputs, output } => {
            println!("Failure: Found counterexample for enum bounds");
            for input in &inputs {
                println!("  {} = {:?}", input.name, input.value);
            }
            if let Some(vio) = &output.assertion_violation {
                println!(
                    "  Violated assertion label '{}': {}",
                    vio.label, vio.message
                );
            }

            if let Some(path) = output_json_path {
                let outcome = EnumInBoundOutcome {
                    success: false,
                    counterexample: Some(SerializedCounterexample {
                        inputs: serialize_inputs(&inputs),
                        output: serialize_output(&output),
                    }),
                    assert_label_prefix: ASSERT_LABEL_PREFIX,
                };
                std::fs::write(path, serde_json::to_string_pretty(&outcome).unwrap())
                    .unwrap_or_else(|err| {
                        report_cli_error_and_exit(
                            &format!("Failed to write JSON output: {}", err),
                            Some(SUBCOMMAND),
                            vec![("path", path.as_str())],
                        );
                    });
            }

            std::process::exit(1);
        }
        BoolPropertyResult::ToolchainDisproved(msg) => {
            println!("Failure: Toolchain reported failure: {}", msg);
            if let Some(path) = output_json_path {
                let outcome = EnumInBoundOutcome {
                    success: false,
                    counterexample: None,
                    assert_label_prefix: ASSERT_LABEL_PREFIX,
                };
                std::fs::write(path, serde_json::to_string_pretty(&outcome).unwrap())
                    .unwrap_or_else(|err| {
                        report_cli_error_and_exit(
                            &format!("Failed to write JSON output: {}", err),
                            Some(SUBCOMMAND),
                            vec![("path", path.as_str())],
                        );
                    });
            }
            std::process::exit(1);
        }
        BoolPropertyResult::Error(msg) => {
            report_cli_error_and_exit(&msg, Some(SUBCOMMAND), Vec::new());
        }
    }
}
