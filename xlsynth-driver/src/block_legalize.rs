// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use anyhow::Context;
use clap::ArgMatches;
use xlsynth_pir::block_legalize::legalize_block_hierarchy;
use xlsynth_pir::ir::{MemberType, PackageMember};
use xlsynth_pir::ir_parser::Parser;

/// Lowers the selected Block IR hierarchy into a flattened, schedulable block.
pub fn handle_block_legalize(matches: &ArgMatches) -> anyhow::Result<()> {
    let path = matches
        .get_one::<String>("input_file")
        .expect("clap requires an input file");
    let input = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read block IR file '{path}'"))?;
    let package = Parser::new(&input)
        .parse_package()
        .map_err(|error| anyhow::anyhow!("failed to parse block IR: {error}"))?;
    let explicit = matches.get_one::<String>("top").map(String::as_str);
    let marked = package
        .top
        .as_ref()
        .and_then(|(name, kind)| (*kind == MemberType::Block).then_some(name.as_str()));
    let selected = match explicit.or(marked) {
        Some(name) => name,
        None => {
            let mut blocks = package.members.iter().filter_map(|member| match member {
                PackageMember::Block(block) => Some(block.name.as_str()),
                _ => None,
            });
            let only = blocks
                .next()
                .ok_or_else(|| anyhow::anyhow!("package contains no blocks to legalize"))?;
            anyhow::ensure!(
                blocks.next().is_none(),
                "package has multiple blocks; select a top with --top"
            );
            only
        }
    };
    let legalized = legalize_block_hierarchy(&package, selected).map_err(anyhow::Error::msg)?;
    let output = legalized.to_string();
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(output.as_bytes())?;
    if !output.ends_with('\n') {
        stdout.write_all(b"\n")?;
    }
    Ok(())
}
