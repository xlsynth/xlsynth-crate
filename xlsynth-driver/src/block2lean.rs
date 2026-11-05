// SPDX-License-Identifier: Apache-2.0

//! File and CLI adaptation for the library-owned Lean block translation.
//!
//! Input validation and model generation finish before any output is opened.

use std::io::Write;

use anyhow::{Context, Result};
use clap::ArgMatches;
use xlsynth::IrPackage;
use xlsynth_lean::block;
use xlsynth_pir::ir_parser::Parser;

/// Generate all Lean text before creating or truncating the output file.
pub fn handle_block2lean(matches: &ArgMatches) -> Result<()> {
    let input_path = matches
        .get_one::<String>("input_file")
        .expect("clap requires an input file");
    let input = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read block IR file '{input_path}'"))?;
    let top = matches.get_one::<String>("top").map(String::as_str);
    IrPackage::parse_ir(&input, None)
        .context("native XLS package parse")?
        .verify()
        .context("native XLS package verification")?;
    let package = Parser::new(&input)
        .parse_and_validate_package()
        .context("typed PIR package parse/validation")?;
    let selected = block::select_block(&package, top)?;
    let text = block::emit(&package, selected)?;
    if let Some(path) = matches.get_one::<String>("output") {
        std::fs::write(path, text)
            .with_context(|| format!("failed to write generated Lean file '{path}'"))?;
    } else {
        std::io::stdout()
            .lock()
            .write_all(text.as_bytes())
            .context("failed to write generated Lean to stdout")?;
    }
    Ok(())
}
