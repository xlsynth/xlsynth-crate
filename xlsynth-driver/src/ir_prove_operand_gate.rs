// SPDX-License-Identifier: Apache-2.0

//! CLI adapter for the library operand-gate proof.

use clap::ArgMatches;
use serde::{Deserialize, Serialize};
use xlsynth_pir::IrFormatPreference;
use xlsynth_pir::ir_operand_gate::OperandGateSite;
use xlsynth_prover::ir_operand_gate::{PredicateReachability, prove_operand_gate};
use xlsynth_prover::prover::SolverLimits;
use xlsynth_prover::prover::types::EquivResult;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonSite {
    consumer: String,
    operand: usize,
    start: Option<usize>,
    width: Option<usize>,
    clamp: String,
}

#[derive(Serialize)]
struct InputValue {
    name: String,
    value: String,
}

#[derive(Serialize)]
struct Output {
    status: &'static str,
    function: String,
    predicate_reachability: &'static str,
    vacuous: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    inputs: Option<Vec<InputValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gated_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

/// Reads either a single site from flags or a JSON array of simultaneous sites.
fn sites_from_matches(matches: &ArgMatches) -> Result<Vec<OperandGateSite>, String> {
    if let Some(path) = matches.get_one::<String>("sites_json") {
        if ["consumer", "operand", "start", "width", "clamp"]
            .iter()
            .any(|key| matches.contains_id(key) && matches.value_source(key).is_some())
        {
            return Err("--sites-json cannot be combined with single-site flags".to_string());
        }
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read sites JSON {path:?}: {error}"))?;
        let specs: Vec<JsonSite> = serde_json::from_str(&text)
            .map_err(|error| format!("invalid sites JSON {path:?}: {error}"))?;
        return Ok(specs
            .into_iter()
            .map(|site| OperandGateSite {
                consumer: site.consumer,
                operand: site.operand,
                start: site.start,
                width: site.width,
                clamp: site.clamp,
            })
            .collect());
    }
    let consumer = matches
        .get_one::<String>("consumer")
        .ok_or("supply --consumer, --operand, and --clamp, or use --sites-json")?;
    let operand = matches
        .get_one::<usize>("operand")
        .ok_or("--operand is required for a single site")?;
    let clamp = matches
        .get_one::<String>("clamp")
        .ok_or("--clamp is required for a single site")?;
    Ok(vec![OperandGateSite {
        consumer: consumer.clone(),
        operand: *operand,
        start: matches.get_one::<usize>("start").copied(),
        width: matches.get_one::<usize>("width").copied(),
        clamp: clamp.clone(),
    }])
}

/// Handles `ir-prove-operand-gate` and exits according to its proof status.
pub fn handle_ir_prove_operand_gate(matches: &ArgMatches) {
    match run(matches) {
        Ok((output, code)) => {
            if matches
                .get_one::<String>("format")
                .is_some_and(|format| format == "json")
            {
                println!(
                    "{}",
                    serde_json::to_string(&output).expect("serializable proof output")
                );
            } else {
                println!(
                    "ir-prove-operand-gate: {} ({})",
                    output.status, output.function
                );
                if let Some(reason) = &output.reason {
                    println!("reason: {reason}");
                }
                if let Some(inputs) = &output.inputs {
                    for input in inputs {
                        println!("  {} = {}", input.name, input.value);
                    }
                    println!(
                        "  original = {}",
                        output.original_output.as_deref().unwrap_or("")
                    );
                    println!("  gated = {}", output.gated_output.as_deref().unwrap_or(""));
                }
                println!("predicate reachability: {}", output.predicate_reachability);
                println!("vacuous: {}", output.vacuous);
            }
            std::process::exit(code);
        }
        Err(error) => {
            eprintln!("ir-prove-operand-gate: {error}");
            std::process::exit(2);
        }
    }
}

fn run(matches: &ArgMatches) -> Result<(Output, i32), String> {
    let path = matches
        .get_one::<String>("ir_input_file")
        .expect("required IR path");
    let source = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read IR {path:?}: {error}"))?;
    let sites = sites_from_matches(matches)?;
    let limit = matches.get_one::<u64>("time_limit_ms").copied();
    if limit == Some(0) {
        return Err("--time-limit-ms must be positive".to_string());
    }
    let proof = prove_operand_gate(
        &source,
        matches.get_one::<String>("top").map(String::as_str),
        matches
            .get_one::<String>("when")
            .expect("required predicate"),
        &sites,
        SolverLimits {
            time_limit_per_ms: limit,
            memory_limit_mb: None,
        },
    )?;
    let reachability = match proof.predicate_reachability {
        PredicateReachability::Reachable => "reachable",
        PredicateReachability::Unreachable => "unreachable",
        PredicateReachability::Unknown => "unknown",
    };
    let mut output = Output {
        status: "proved",
        function: proof.function,
        predicate_reachability: reachability,
        vacuous: proof.predicate_reachability == PredicateReachability::Unreachable,
        inputs: None,
        original_output: None,
        gated_output: None,
        reason: None,
    };
    let code = match proof.result {
        EquivResult::Proved => 0,
        EquivResult::Disproved {
            lhs_inputs,
            lhs_output,
            rhs_output,
            ..
        } => {
            output.status = "counterexample";
            output.inputs = Some(
                lhs_inputs
                    .into_iter()
                    .map(|input| InputValue {
                        name: input.name,
                        value: input.value.to_string_fmt(IrFormatPreference::Hex),
                    })
                    .collect(),
            );
            output.original_output = Some(lhs_output.value.to_string_fmt(IrFormatPreference::Hex));
            output.gated_output = Some(rhs_output.value.to_string_fmt(IrFormatPreference::Hex));
            output.vacuous = false;
            1
        }
        EquivResult::Inconclusive(reason) => {
            output.status = "unknown";
            output.reason = Some(reason);
            output.vacuous = false;
            3
        }
        EquivResult::Error(message) | EquivResult::ToolchainDisproved(message) => {
            return Err(message);
        }
    };
    Ok((output, code))
}
