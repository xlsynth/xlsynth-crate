// SPDX-License-Identifier: Apache-2.0

//! Lower convention-based stateful functions to block IR and readable Lean.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use clap::{Arg, ArgAction, ArgMatches, Command};
use xlsynth_lean::block;

use crate::toolchain_config::ToolchainConfig;

pub fn command() -> Command {
    Command::new("dslx2lean")
        .about("Compiles a DSLX function with trailing __reg_ parameters to readable Lean")
        .arg(Arg::new("input_file").value_name("SOURCE").required(true))
        .arg(
            Arg::new("top")
                .long("top")
                .value_name("NAME")
                .help("Stateful DSLX function to compile")
                .required(true),
        )
        .arg(
            Arg::new("output_dir")
                .long("output-dir")
                .value_name("NEW_DIR")
                .help("Write the five Lean modules to a new directory")
                .required(true),
        )
        .arg(stdlib_arg())
        .arg(search_path_arg())
}

fn stdlib_arg() -> Arg {
    Arg::new("stdlib")
        .long("dslx-stdlib-path")
        .value_name("DIR")
        .help("Override the native DSLX standard library directory")
}

fn search_path_arg() -> Arg {
    Arg::new("search_path")
        .long("dslx-search-path")
        .value_name("DIR")
        .help("Additional DSLX import directory; may be repeated")
        .action(ArgAction::Append)
}

/// Both block-producing commands print a verified IR package to stdout.
pub fn to_block_command(dslx: bool) -> Command {
    let (name, about) = if dslx {
        (
            "dslx-stateful-fn-to-block",
            "Lower a stateful DSLX function to block IR",
        )
    } else {
        (
            "ir-stateful-fn-to-block",
            "Convert a stateful IR function to block IR",
        )
    };
    let command = Command::new(name)
        .about(about)
        .arg(Arg::new("input_file").required(true))
        .arg(
            Arg::new("top")
                .long("top")
                .value_name("NAME")
                .required(true),
        );
    if dslx {
        command.arg(stdlib_arg()).arg(search_path_arg())
    } else {
        command
    }
}

fn search_paths(matches: &ArgMatches) -> Vec<&Path> {
    matches
        .get_many::<String>("search_path")
        .into_iter()
        .flatten()
        .map(Path::new)
        .collect()
}

fn compile_stateful(
    matches: &ArgMatches,
    source: &str,
    path: &Path,
) -> Result<block::CompiledStatefulFunction> {
    block::compile_stateful_function(
        source,
        path,
        matches.get_one::<String>("top").unwrap(),
        matches.get_one::<String>("stdlib").map(Path::new),
        &search_paths(matches),
    )
}

pub fn handle_to_block(
    name: &str,
    matches: &ArgMatches,
    config: &Option<ToolchainConfig>,
) -> Result<()> {
    ensure!(
        config.is_none(),
        "{name} uses the linked native compiler and explicit options; --toolchain and an automatically loaded xlsynth-toolchain.toml are unsupported"
    );
    let path = Path::new(matches.get_one::<String>("input_file").unwrap());
    let source =
        std::fs::read_to_string(path).with_context(|| format!("reading '{}'", path.display()))?;
    if name == "dslx-stateful-fn-to-block" {
        let compiled = compile_stateful(matches, &source, path)?;
        print!("{}", compiled.block_ir());
    } else {
        let top = matches.get_one::<String>("top").unwrap();
        print!("{}", block::stateful_ir_to_block(&source, top)?);
    }
    Ok(())
}

/// Generate the complete bundle before creating output; never invoke Lean.
///
/// Compilation and validation errors leave outputs untouched. Filesystem write
/// errors can leave a partial new directory.
pub fn handle_dslx2lean(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    ensure!(
        config.is_none(),
        "dslx2lean uses the linked native compiler and explicit options; --toolchain and an automatically loaded xlsynth-toolchain.toml are unsupported"
    );
    let source_path = Path::new(matches.get_one::<String>("input_file").unwrap());
    let source = std::fs::read_to_string(source_path)
        .with_context(|| format!("reading DSLX source '{}'", source_path.display()))?;
    let bundle = compile_stateful(matches, &source, source_path)?.emit_readable()?;
    let output_dir = Path::new(matches.get_one::<String>("output_dir").unwrap());
    std::fs::create_dir(output_dir)
        .with_context(|| format!("creating new output directory '{}'", output_dir.display()))?;
    for (name, text) in bundle.files {
        let path = output_dir.join(name);
        std::fs::write(&path, text)
            .with_context(|| format!("writing generated Lean module '{}'", path.display()))?;
    }
    Ok(())
}
