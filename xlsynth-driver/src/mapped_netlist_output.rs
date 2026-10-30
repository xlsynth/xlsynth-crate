// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, ValueEnum};
use std::io::{self, Write};
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::io::ParsedNetlist;
use xlsynth_g8r::netlist::proto::encode_netlist;

/// Output formats accepted by the technology mapping commands.
#[derive(Clone, Copy, ValueEnum)]
pub enum Format {
    Gv,
    Pb,
}

pub fn format_arg() -> Arg {
    Arg::new("netlist-format")
        .long("netlist-format")
        .value_name("FORMAT")
        .value_parser(clap::value_parser!(Format))
        .default_value("gv")
        .help("Mapped netlist format: gate-level Verilog or binary netlist protobuf")
}

impl Format {
    pub fn from_matches(matches: &ArgMatches) -> Self {
        *matches.get_one::<Self>("netlist-format").unwrap()
    }

    /// Serializes the mapper's final netlist directly in the selected format.
    pub fn encode(self, netlist: &ParsedNetlist) -> Result<Vec<u8>> {
        match self {
            Self::Gv => Ok(emit_netlist_as_text(netlist)
                .context("failed to emit mapped netlist text")?
                .into_bytes()),
            Self::Pb => encode_netlist(netlist).context("failed to encode mapped netlist protobuf"),
        }
    }
}

/// Flushes the entire netlist before continuing to sidecar output or exiting.
pub fn write_stdout(bytes: &[u8]) -> Result<()> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(bytes)
        .and_then(|()| stdout.flush())
        .context("writing netlist to stdout")
}
