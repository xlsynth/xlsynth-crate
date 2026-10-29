// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, Command, value_parser};
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::read_gv_for_roundtrip_from_path;
use xlsynth_g8r::netlist::proto::{decode_netlist, encode_netlist};

/// Declares the file and stdout interfaces shared by the two netlist
/// conversions.
pub fn commands() -> [Command; 2] {
    [
        (
            "gv2pb",
            "Encode mapped gate-level Verilog as a binary netlist protobuf",
            "Input gate-level Verilog (.gv, .v, or .gz)",
        ),
        (
            "pb2gv",
            "Emit gate-level Verilog from a binary netlist protobuf",
            "Input binary netlist protobuf",
        ),
    ]
    .map(|(name, about, input_help)| {
        Command::new(name)
            .about(about)
            .arg(
                Arg::new("input")
                    .value_name("INPUT")
                    .value_parser(value_parser!(PathBuf))
                    .required(true)
                    .help(input_help),
            )
            .arg(
                Arg::new("output")
                    .value_name("PATH")
                    .value_parser(value_parser!(PathBuf))
                    .short('o')
                    .long("output")
                    .help("Write the result to PATH (default: stdout)"),
            )
    })
}

/// Writes a completed conversion to the requested destination.
fn write_output(matches: &ArgMatches, bytes: &[u8]) -> Result<()> {
    if let Some(path) = matches.get_one::<PathBuf>("output") {
        std::fs::write(path, bytes).with_context(|| format!("writing '{}'", path.display()))
    } else {
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(bytes)
            .and_then(|()| stdout.flush())
            .context("writing stdout")
    }
}

pub fn handle_gv2pb(matches: &ArgMatches) -> Result<()> {
    let input = matches.get_one::<PathBuf>("input").unwrap();
    let parsed = read_gv_for_roundtrip_from_path(input)?;
    write_output(matches, &encode_netlist(&parsed)?)
}

pub fn handle_pb2gv(matches: &ArgMatches) -> Result<()> {
    let input = matches.get_one::<PathBuf>("input").unwrap();
    let bytes = std::fs::read(input).with_context(|| format!("reading '{}'", input.display()))?;
    let parsed =
        decode_netlist(&bytes).with_context(|| format!("decoding '{}'", input.display()))?;
    let mut text = String::new();
    for module in &parsed.modules {
        text.push_str(&emit_module_as_netlist_text(
            module,
            &parsed.nets,
            &parsed.interner,
        )?);
    }
    write_output(matches, text.as_bytes())
}
