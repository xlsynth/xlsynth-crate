// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::Serialize;
use xlsynth_g8r::aig_serdes::g8r::load_gate_fn_from_path;
use xlsynth_lean::{LeanProofSources, prepare_equivalence_target, run_equivalence_proof};

use crate::common::write_stdout;

/// Defines the command that prepares an AAG-backed Lean equivalence target.
pub fn prepare_command() -> Command {
    Command::new("g8r-lean-prepare")
        .about("Emits two AAG files and one Lean equivalence target using the shared AAG loader")
        .arg(
            Arg::new("lhs")
                .long("lhs")
                .required(true)
                .value_name("PATH")
                .help("Left combinational .g8r or .g8rbin design"),
        )
        .arg(
            Arg::new("rhs")
                .long("rhs")
                .required(true)
                .value_name("PATH")
                .help("Right combinational .g8r or .g8rbin design"),
        )
        .arg(
            Arg::new("output-dir")
                .long("output-dir")
                .required(true)
                .value_name("DIR")
                .help("Directory for the generated Lean target bundle"),
        )
}

/// Defines the command that runs a user proof against a prepared target.
pub fn check_command() -> Command {
    Command::new("g8r-lean-check")
        .about("Runs a supplied Lean proof against a prepared g8r equivalence target; no proof audit is performed")
        .arg(Arg::new("target-dir").long("target-dir").required(true).value_name("DIR").help("Target bundle written by g8r-lean-prepare"))
        .arg(Arg::new("proof").long("proof").required(true).value_name("PATH").help("File containing a complete Lean proof term for Target.Equivalent, such as by ..."))
        .arg(Arg::new("proof-root").long("proof-root").value_name("DIR").help("Root used for helper paths; defaults to the proof body's directory"))
        .arg(Arg::new("proof-source").long("proof-source").value_name("PATH").action(ArgAction::Append).help("Lean helper module imported by the generated proof wrapper; repeat for multiple modules"))
        .arg(Arg::new("lean").long("lean").required(true).value_name("PATH").help("Explicit Lean executable path"))
        .arg(Arg::new("work-dir").long("work-dir").value_name("DIR").help("Empty proof work directory; defaults to a temporary directory"))
        .arg(Arg::new("output-json").long("output-json").value_name("PATH").help("Write a machine-readable result and captured diagnostics"))
}

/// Checks graph interfaces and writes AAG files with one shared-loader target.
pub fn handle_prepare(matches: &ArgMatches) -> Result<(), String> {
    let lhs = load_gate_fn_from_path(Path::new(matches.get_one::<String>("lhs").unwrap()))?;
    let rhs = load_gate_fn_from_path(Path::new(matches.get_one::<String>("rhs").unwrap()))?;
    prepare_equivalence_target(&lhs, &rhs)
        .map_err(|error| error.to_string())?
        .write_to(Path::new(matches.get_one::<String>("output-dir").unwrap()))
        .map_err(|error| error.to_string())
}

/// Machine-readable outcome of a Lean proof invocation.
#[derive(Serialize)]
struct CheckReport {
    status: &'static str,
    audit_performed: bool,
    lean_version: Option<String>,
    elapsed_ms: u128,
    stdout: String,
    stderr: String,
    error: Option<String>,
}

/// Stages proof inputs, invokes Lean, and emits optional JSON on either
/// outcome.
pub fn handle_check(matches: &ArgMatches) -> Result<(), String> {
    let proof = PathBuf::from(matches.get_one::<String>("proof").unwrap());
    let proof_root = matches
        .get_one::<String>("proof-root")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            proof
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let sources = LeanProofSources {
        root: proof_root,
        body: proof,
        modules: matches
            .get_many::<String>("proof-source")
            .into_iter()
            .flatten()
            .map(PathBuf::from)
            .collect(),
    };
    let temporary = if matches.get_one::<String>("work-dir").is_none() {
        Some(
            tempfile::Builder::new()
                .prefix("xlsynth-lean-check-")
                .tempdir()
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let work_root = matches
        .get_one::<String>("work-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| temporary.as_ref().unwrap().path().to_path_buf());
    let started = Instant::now();
    let outcome = run_equivalence_proof(
        Path::new(matches.get_one::<String>("lean").unwrap()),
        Path::new(matches.get_one::<String>("target-dir").unwrap()),
        &sources,
        &work_root,
    );
    let report = match outcome {
        Ok(output) => CheckReport {
            status: "accepted_by_lean",
            audit_performed: false,
            lean_version: Some(output.lean_version),
            elapsed_ms: started.elapsed().as_millis(),
            stdout: output.stdout,
            stderr: output.stderr,
            error: None,
        },
        Err(error) => CheckReport {
            status: "failed",
            audit_performed: false,
            lean_version: None,
            elapsed_ms: started.elapsed().as_millis(),
            stdout: String::new(),
            stderr: String::new(),
            error: Some(error.to_string()),
        },
    };
    if let Some(path) = matches.get_one::<String>("output-json") {
        let json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
        std::fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("writing Lean proof report '{path}': {error}"))?;
    }
    if let Some(error) = report.error {
        return Err(error);
    }
    write_stdout(&report.stdout);
    if !report.stderr.is_empty() {
        eprint!("{}", report.stderr);
    }
    Ok(())
}
