// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::time::Duration;

use crate::dslx_stitch_pipeline::resolve_stitch_spec;
use anyhow::{Result, bail};
use clap::{Arg, ArgMatches, Command, value_parser};
use xlsynth_g8r::prove_gate_fn_equiv_sat::GateFormalOptions;
use xlsynth_g8r::stitch_equiv::StitchEquivResult;

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
        .arg(Arg::new("timeout-seconds").long("timeout-seconds").value_name("SECONDS")
            .value_parser(value_parser!(u64).range(1..)).help("CaDiCaL solve timeout (default unlimited)"))
}

/// Loads the reference and reports the proof result or a replayable
/// counterexample.
pub fn handle(matches: &ArgMatches) -> Result<()> {
    let reference = resolve_stitch_spec(matches.get_one::<PathBuf>("spec").unwrap())?;
    let mut options = GateFormalOptions::default();
    if let Some(seconds) = matches.get_one::<u64>("timeout-seconds") {
        options = options.with_cadical_timeout(Duration::from_secs(*seconds));
    }
    let module = matches.get_one::<String>("module-name").map(String::as_str);
    match reference.prove_equivalent_to_gv(
        matches.get_one::<PathBuf>("netlist").unwrap(),
        matches.get_one::<PathBuf>("liberty-proto").unwrap(),
        module,
        options,
    )? {
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
