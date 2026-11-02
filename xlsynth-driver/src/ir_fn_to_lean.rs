// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command, value_parser};
use xlsynth_lean::{LeanModuleName, export_ir_function};
use xlsynth_pir::ir_parser::Parser;

pub fn command() -> Command {
    Command::new("ir-fn-to-lean")
        .about("Exports a pure XLS IR function and its callees as executable word-level Lean")
        .arg(
            Arg::new("ir")
                .required(true)
                .value_name("IR")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("top")
                .long("top")
                .value_name("FUNCTION")
                .help("Override the package top or sole function"),
        )
        .arg(
            Arg::new("module")
                .long("module")
                .value_name("NAME")
                .default_value("Generated.Function")
                .help("Dotted Lean module name"),
        )
        .arg(
            Arg::new("output-dir")
                .long("output-dir")
                .value_name("DIR")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .help("Directory for the Lean module, support source, and toolchain pin"),
        )
}

pub fn handle(matches: &ArgMatches) -> Result<(), String> {
    let path = matches.get_one::<PathBuf>("ir").expect("required path");
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let package = Parser::new(&text)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    let module = LeanModuleName::new(matches.get_one::<String>("module").expect("module default"))
        .map_err(|e| e.to_string())?;
    let function = export_ir_function(
        &package,
        matches.get_one::<String>("top").map(String::as_str),
        &module,
    )
    .map_err(|e| e.to_string())?;
    function
        .write_to(
            matches
                .get_one::<PathBuf>("output-dir")
                .expect("required directory"),
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}
