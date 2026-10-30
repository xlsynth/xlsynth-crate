// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, Command, value_parser};
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
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
        crate::mapped_netlist_output::write_stdout(bytes)
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
    write_output(matches, emit_netlist_as_text(&parsed)?.as_bytes())
}
