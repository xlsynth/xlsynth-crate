// SPDX-License-Identifier: Apache-2.0

//! Reuse one timing library across native mapping requests. Output files are
//! distinct for each request.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use xlsynth_g8r::liberty::load::TimingLibrarySnapshot;
use xlsynth_g8r::techmap::postprocess::NativeCoverLibrary;

use crate::{g8r_cli, g8r2gv, gv_resize_batch::process_cpu_seconds};

/// One complete g8r2gv argument vector, including the subcommand name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    argv: Vec<String>,
}

/// A response is flushed after every request, including a final failure.
#[derive(Serialize)]
struct Response<'a> {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    process_cpu_s: Option<f64>,
}

pub(crate) fn command() -> Command {
    Command::new("g8r2gv-batch")
        .about("Run native g8r2gv commands against one SHA-pinned timing library")
        .arg(
            Arg::new("liberty-proto")
                .long("liberty-proto")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("liberty-sha256")
                .long("liberty-sha256")
                .required(true),
        )
        .arg(
            Arg::new("report-process-cpu")
                .long("report-process-cpu")
                .action(ArgAction::SetTrue),
        )
}

/// Use the exact ordinary argument parser and sequential mapping operation.
fn map_one(raw: &str, lib_path: &Path, cache: &mut NativeCoverLibrary, digest: &str) -> Result<()> {
    let request: Request = serde_json::from_str(raw).context("parsing map request")?;
    ensure!(
        request.argv.first().is_some_and(|x| x == "g8r2gv"),
        "request must start with g8r2gv"
    );
    let args = g8r2gv::command().try_get_matches_from(request.argv)?;
    ensure!(
        args.get_one::<String>("mapper")
            .is_some_and(|x| x == "native"),
        "batch requires --mapper native"
    );
    let requested_lib = Path::new(args.get_one::<String>("liberty-proto").unwrap());
    ensure!(
        requested_lib == lib_path,
        "request must use the batch Liberty path {}",
        lib_path.display()
    );
    let input = Path::new(args.get_one::<String>("g8r_input_file").unwrap());
    ensure!(
        input.is_absolute(),
        "batch G8R input must have an absolute path"
    );
    let output = args.get_one::<PathBuf>("output").unwrap();
    let reports = ["refinement-report", "adaptive-resize-report"]
        .iter()
        .filter_map(|name| args.get_one::<String>(name).map(Path::new));
    for path in std::iter::once(output.as_path()).chain(reports) {
        ensure!(
            path.is_absolute(),
            "batch output must have an absolute path: {}",
            path.display()
        );
    }
    g8r2gv::map_design_with_library(
        &args,
        || g8r_cli::read_netlist_design(&args).map_err(anyhow::Error::msg),
        Some((cache, digest)),
    )
}

/// Keep one decoded library and at most one delay index alive until EOF.
pub(crate) fn handle(matches: &ArgMatches) -> Result<()> {
    let lib_path = matches.get_one::<PathBuf>("liberty-proto").unwrap();
    ensure!(
        lib_path.is_absolute(),
        "batch Liberty must have an absolute path"
    );
    let expected = matches.get_one::<String>("liberty-sha256").unwrap();
    let report_cpu = matches.get_flag("report-process-cpu");
    if report_cpu {
        process_cpu_seconds()?;
    }
    let snapshot = TimingLibrarySnapshot::from_path(lib_path)
        .context("mapping/resizing requires a timing-enabled --liberty-proto")?;
    ensure!(
        snapshot.sha256 == *expected,
        "batch Liberty SHA-256 mismatch"
    );
    // Reuse only library-dependent matches. Each request still owns its graph,
    // timing work and output files, and resolves its own leakage table.
    let mut cache = NativeCoverLibrary::new(&snapshot.library);
    let mut output = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let result = map_one(&line?, lib_path, &mut cache, &snapshot.sha256);
        let error = result.as_ref().err().map(|e| format!("{e:#}"));
        let response = Response {
            ok: result.is_ok(),
            error: error.as_deref(),
            process_cpu_s: report_cpu.then(process_cpu_seconds).transpose()?,
        };
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        output.flush()?;
        result?;
    }
    Ok(())
}
