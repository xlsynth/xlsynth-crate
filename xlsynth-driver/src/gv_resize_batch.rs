// SPDX-License-Identifier: Apache-2.0

//! Reuse a timing library across sequential requests to the ordinary gv-resize
//! CLI.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use xlsynth_g8r::liberty::load::TimingLibrarySnapshot;
use xlsynth_g8r::netlist::resize::RecoveryCellLeakage;

use crate::gv_resize;
use crate::mapped_netlist_output::write_batch_outputs;

/// A single gv-resize command line and the destination for its emitted gate
/// text.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    args: Vec<String>,
    output: PathBuf,
}

/// A flushed acknowledgement denotes a completely written netlist and resizing
/// report.
#[derive(Serialize)]
struct Completion {
    #[serde(skip_serializing_if = "Option::is_none")]
    process_cpu_s: Option<f64>,
}

pub(crate) fn command() -> Command {
    Command::new("gv-resize-batch")
        .about(
            "Run sequential gv-resize requests from JSON lines against one immutable Liberty file",
        )
        .arg(
            Arg::new("liberty-proto")
                .long("liberty-proto")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("report-process-cpu")
                .long("report-process-cpu")
                .help("Include cumulative process CPU seconds in each acknowledgement (Unix)")
                .action(ArgAction::SetTrue),
        )
}

#[cfg(unix)]
pub(crate) fn process_cpu_seconds() -> Result<f64> {
    let usage = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)?;
    let cpu = usage.user_time() + usage.system_time();
    Ok(nix::sys::time::TimeValLike::num_microseconds(&cpu) as f64 * 1e-6)
}

#[cfg(not(unix))]
pub(crate) fn process_cpu_seconds() -> Result<f64> {
    anyhow::bail!("process CPU accounting requires Unix")
}

/// A fresh argument parse, netlist, optimizer and output are used for every
/// request.
pub(crate) fn handle(matches: &ArgMatches) -> Result<()> {
    let path = matches.get_one::<PathBuf>("liberty-proto").unwrap();
    let report_cpu = matches.get_flag("report-process-cpu");
    if report_cpu {
        process_cpu_seconds()?;
    }
    let snapshot = TimingLibrarySnapshot::from_path(path)?;
    let mut output = io::stdout().lock();
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line = line?;
        let process = || -> Result<()> {
            let request: Request = serde_json::from_str(&line)?;
            let args = gv_resize::command()
                .no_binary_name(true)
                .try_get_matches_from(request.args)?;
            ensure!(
                Some(path.as_path())
                    == args
                        .get_one::<String>("liberty_proto")
                        .map(std::path::Path::new),
                "request must use the batch Liberty path {}",
                path.display()
            );
            // Validate each request's table against the resident model. The
            // path can be replaced during the batch, while the worker keeps
            // using its snapshot.
            let leakage = args
                .get_one::<PathBuf>("terminal-leakage-table")
                .map(|sidecar| {
                    RecoveryCellLeakage::from_library_digest(sidecar, &snapshot.sha256, path)
                })
                .transpose()?;
            let output = gv_resize::resize_text(&args, &snapshot.library, leakage)
                .map_err(anyhow::Error::msg)?;
            write_batch_outputs(
                (&request.output, output.verilog.into_bytes()),
                output.report,
            )
        };
        process().with_context(|| format!("gv-resize-batch request {}", index + 1))?;
        let ack = Completion {
            process_cpu_s: report_cpu.then(process_cpu_seconds).transpose()?,
        };
        serde_json::to_writer(&mut output, &ack)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
