// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use anyhow::Context;
use clap::ArgMatches;
use xlsynth_pir::block_simplify::{simplify_block_hierarchy, simplify_block_package};
use xlsynth_pir::ir::{MemberType, PackageMember};
use xlsynth_pir::ir_parser::Parser;

/// Simplifies a Block IR package and writes the transformed package to stdout.
pub fn handle_block_simplify(matches: &ArgMatches) -> anyhow::Result<()> {
    let path = matches
        .get_one::<String>("input_file")
        .expect("clap requires an input file");
    let input = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read block IR file '{path}'"))?;
    let package = Parser::new(&input)
        .parse_package()
        .map_err(|err| anyhow::anyhow!("failed to parse block IR: {err}"))?;
    let explicit = matches.get_one::<String>("top").map(String::as_str);
    let block_names: Vec<_> = package
        .members
        .iter()
        .filter_map(|m| match m {
            PackageMember::Block(b) => Some(b.name.as_str()),
            _ => None,
        })
        .collect();
    anyhow::ensure!(
        !block_names.is_empty(),
        "package contains no blocks to simplify"
    );
    let selected = explicit.or_else(|| match &package.top {
        Some((name, MemberType::Block)) => Some(name.as_str()),
        _ => match block_names.as_slice() {
            [name] => Some(*name),
            _ => None,
        },
    });
    let mut simplified = match selected {
        Some(top) => simplify_block_hierarchy(&package, top),
        None => simplify_block_package(&package),
    }
    .map_err(anyhow::Error::msg)?;
    if let Some(top) = explicit {
        simplified.set_top_block(top).map_err(anyhow::Error::msg)?;
    }
    let output = simplified.to_string();
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(output.as_bytes())?;
    if !output.ends_with('\n') {
        stdout.write_all(b"\n")?;
    }
    Ok(())
}
