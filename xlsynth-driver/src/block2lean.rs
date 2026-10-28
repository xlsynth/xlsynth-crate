// SPDX-License-Identifier: Apache-2.0

//! File and CLI adaptation for the library-owned Lean block translation.
//!
//! Input validation and model generation finish before any output is opened.

use std::io::Write;

use anyhow::{Context, Result};
use clap::ArgMatches;
use xlsynth_lean::block;

/// Generate all Lean text before creating or truncating the output file.
pub fn handle_block2lean(matches: &ArgMatches) -> Result<()> {
    let input_path = matches
        .get_one::<String>("input_file")
        .expect("clap requires an input file");
    let input = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read block IR file '{input_path}'"))?;
    let top = matches.get_one::<String>("top").map(String::as_str);
    let parsed = block::parse(&input, top)?;
    let text = block::emit(&parsed)?;
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
