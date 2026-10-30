// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use crate::dslx_stitch_pipeline::resolve_stitch_spec;
use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use xlsynth_g8r::stitch_equiv::{
    StageProofScript, StitchEquivResult, StitchProofMethod, StitchProofMode, StitchProofOptions,
};

/// Defines the clocked equivalence check between a stitch spec and mapped
/// gates.
pub fn command() -> Command {
    Command::new("stitch-equiv")
        .about("Proves post-fill output equivalence of a free-running DSLX stitch pipeline and mapped gates")
        .arg(Arg::new("spec").long("spec").required(true).value_name("PATH")
            .value_parser(value_parser!(PathBuf)).help("TOML stitch specification"))
        .arg(Arg::new("netlist").long("netlist").required(true).value_name("PATH")
            .value_parser(value_parser!(PathBuf)).help("Mapped gate-level Verilog netlist"))
        .arg(Arg::new("liberty-proto").long("liberty-proto").required(true).value_name("PATH")
            .value_parser(value_parser!(PathBuf)).help("Liberty proto describing the mapped cells"))
        .arg(Arg::new("module-name").long("module-name").value_name("NAME")
            .help("Mapped top module (defaults to the stitch specification's wrapper name)"))
        .arg(Arg::new("proof-mode").long("proof-mode").value_name("MODE")
            .value_parser(["auto", "stages", "whole-pipeline"]).default_value("auto")
            .help("Try stage proofs with post-fill fallback, require stage proofs, or prove the whole pipeline directly"))
        .arg(Arg::new("jobs").long("jobs").value_name("N").value_parser(value_parser!(NonZeroUsize))
            .help("Maximum concurrent stage proof leaves (default available CPUs)"))
        .arg(Arg::new("stage-script").long("stage-script").value_name("STAGE=PATH").action(ArgAction::Append)
            .help("JSON tactic script for one DSLX stage (repeat for other stages)"))
        .arg(Arg::new("show-proof-plan").long("show-proof-plan").action(ArgAction::SetTrue)
            .help("Report the proof method and discharged leaf counts or fallback reason on stderr"))
        .arg(Arg::new("timeout-seconds").long("timeout-seconds").value_name("SECONDS")
            .value_parser(value_parser!(u64).range(1..)).help("Timeout per CaDiCaL solve (default unlimited)"))
}

/// Loads the reference and reports the proof result or a replayable
/// counterexample.
pub fn handle(matches: &ArgMatches) -> Result<()> {
    let reference = resolve_stitch_spec(matches.get_one::<PathBuf>("spec").unwrap())?;
    let mut options = StitchProofOptions {
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
    if let Some(seconds) = matches.get_one::<u64>("timeout-seconds") {
        options.formal = options
            .formal
            .with_cadical_timeout(Duration::from_secs(*seconds));
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
    }
    match report.result {
        StitchEquivResult::Proved => println!(
            "Proved {} equivalent to {} after {} rising clock edge{}.",
            reference.name(),
            module.unwrap_or(reference.name()),
            reference.latency(),
            if reference.latency() == 1 { "" } else { "s" },
        ),
        StitchEquivResult::Disproved(trace) => bail!("post-fill output mismatch\n{trace}"),
    }
    Ok(())
}
