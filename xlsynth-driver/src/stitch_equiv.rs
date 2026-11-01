// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use crate::dslx_stitch_pipeline::resolve_stitch_spec;
use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use xlsynth_g8r::prove_gate_fn_equiv_sat::GateFormalOptions;
use xlsynth_g8r::stitch_equiv::{
    StageProofScript, StitchEquivResult, StitchProofMethod, StitchProofMode, StitchProofOptions,
    StitchProofSolver,
};

/// Identifies the same behavioral spec, mapped top and cell library for both
/// stitch checks.
pub(crate) fn design_args() -> [Arg; 4] {
    [
        Arg::new("spec")
            .long("spec")
            .required(true)
            .value_name("PATH")
            .value_parser(value_parser!(PathBuf))
            .help("TOML stitch specification"),
        Arg::new("netlist")
            .long("netlist")
            .required(true)
            .value_name("PATH")
            .value_parser(value_parser!(PathBuf))
            .help("Mapped gate-level Verilog netlist"),
        Arg::new("liberty-proto")
            .long("liberty-proto")
            .required(true)
            .value_name("PATH")
            .value_parser(value_parser!(PathBuf))
            .help("Liberty proto describing the mapped cells"),
        Arg::new("module-name")
            .long("module-name")
            .value_name("NAME")
            .help("Mapped top module (defaults to the stitch specification's wrapper name)"),
    ]
}

/// Defines the clocked equivalence check between a stitch spec and mapped
/// gates.
pub fn command() -> Command {
    Command::new("stitch-equiv")
        .about("Proves output equivalence of a DSLX stitch pipeline and mapped gates under the spec's valid and reset protocol")
        .args(design_args())
        .arg(Arg::new("solver").long("solver").value_name("SOLVER")
            .value_parser(["bitwuzla", "cadical"])
            .default_value(match StitchProofSolver::default() {
                StitchProofSolver::Bitwuzla { .. } => "bitwuzla",
                StitchProofSolver::Cadical(_) => "cadical",
            })
            .help("Compare behavioral XLS IR (Bitwuzla) or a gatified reference (CaDiCaL) against the mapped cells"))
        .arg(Arg::new("proof-mode").long("proof-mode").value_name("MODE")
            .value_parser(["auto", "stages", "whole-pipeline"]).default_value("auto")
            .help("Try stage proofs with whole-pipeline fallback, require stage proofs, or check the whole pipeline directly"))
        .arg(Arg::new("jobs").long("jobs").value_name("N").value_parser(value_parser!(NonZeroUsize))
            .help("Maximum concurrent stage proof leaves (default available CPUs)"))
        .arg(Arg::new("stage-script").long("stage-script").value_name("STAGE=PATH").action(ArgAction::Append)
            .help("JSON tactic script for one DSLX stage (repeat for other stages)"))
        .arg(Arg::new("show-proof-plan").long("show-proof-plan").action(ArgAction::SetTrue)
            .help("Report the proof method, discharged leaf counts or fallback reason, and solver on stderr"))
        .arg(Arg::new("timeout-seconds").long("timeout-seconds").value_name("SECONDS")
            .value_parser(value_parser!(u64).range(1..)).help("Timeout per solver check (default unlimited)"))
}

/// Loads the reference and reports the proof result or a replayable
/// counterexample.
pub fn handle(matches: &ArgMatches) -> Result<()> {
    let reference = resolve_stitch_spec(matches.get_one::<PathBuf>("spec").unwrap())?;
    let timeout = matches
        .get_one::<u64>("timeout-seconds")
        .copied()
        .map(Duration::from_secs);
    let mut options = StitchProofOptions {
        solver: match matches.get_one::<String>("solver").unwrap().as_str() {
            "bitwuzla" => StitchProofSolver::Bitwuzla { timeout },
            "cadical" => StitchProofSolver::Cadical(GateFormalOptions {
                cadical_timeout: timeout,
                ..GateFormalOptions::default()
            }),
            _ => unreachable!("clap validates the stitch solver"),
        },
        mode: match matches.get_one::<String>("proof-mode").map(String::as_str) {
            Some("stages") => StitchProofMode::Stages,
            Some("whole-pipeline") => StitchProofMode::WholePipeline,
            _ => StitchProofMode::Auto,
        },
        ..StitchProofOptions::default()
    };
    if let Some(jobs) = matches.get_one::<NonZeroUsize>("jobs") {
        options.jobs = *jobs;
    }
    for arg in matches
        .get_many::<String>("stage-script")
        .into_iter()
        .flatten()
    {
        let (name, path) = arg
            .split_once('=')
            .filter(|(name, path)| !name.is_empty() && !path.is_empty())
            .ok_or_else(|| anyhow!("--stage-script expects STAGE=PATH, got {arg:?}"))?;
        if options.stage_scripts.contains_key(name) {
            bail!("more than one script supplied for stage '{name}'");
        }
        let context = || format!("stage '{name}' script '{path}'");
        let source = std::fs::read_to_string(path).with_context(context)?;
        let script = StageProofScript::from_json(&source)
            .map_err(anyhow::Error::msg)
            .with_context(context)?;
        options.stage_scripts.insert(name.to_owned(), script);
    }
    let module = matches.get_one::<String>("module-name").map(String::as_str);
    let report = reference.prove_equivalent_to_gv_with_options(
        matches.get_one::<PathBuf>("netlist").unwrap(),
        matches.get_one::<PathBuf>("liberty-proto").unwrap(),
        module,
        &options,
    )?;
    if matches.get_flag("show-proof-plan") {
        match &report.method {
            StitchProofMethod::Stages { stages } => {
                for stage in stages {
                    eprintln!(
                        "stage {}: {} proved {}",
                        stage.name,
                        stage.leaves,
                        if stage.leaves == 1 { "leaf" } else { "leaves" }
                    );
                }
            }
            StitchProofMethod::WholePipeline {
                fallback_reason: Some(reason),
            } => eprintln!("whole-pipeline fallback: {reason}"),
            StitchProofMethod::WholePipeline {
                fallback_reason: None,
            } => eprintln!("whole-pipeline proof"),
        }
        match options.solver {
            StitchProofSolver::Bitwuzla { .. } => {
                eprintln!("solver bitwuzla: behavioral XLS IR vs. mapped cells")
            }
            StitchProofSolver::Cadical(_) => {
                eprintln!("solver cadical: gatified reference vs. mapped cells")
            }
        }
    }
    let name = reference.name();
    let mapped = module.unwrap_or(name);
    let latency = reference.latency();
    let edges = if latency == 1 { "edge" } else { "edges" };
    match report.result {
        StitchEquivResult::Proved if reference.has_valid_signal() => println!(
            "Proved {name} equivalent to {mapped} under the stitch valid protocol ({latency} rising clock {edges} of latency)."
        ),
        StitchEquivResult::Proved => {
            println!("Proved {name} equivalent to {mapped} after {latency} rising clock {edges}.")
        }
        StitchEquivResult::Disproved(trace) if reference.has_valid_signal() => {
            bail!("valid-qualified output mismatch\n{trace}")
        }
        StitchEquivResult::Disproved(trace) => bail!("post-fill output mismatch\n{trace}"),
    }
    Ok(())
}
