// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use anyhow::Context;
use clap::ArgMatches;
use xlsynth_codegen::{BlockCodegenOptions, Layout, emit_system_verilog};
use xlsynth_pir::ir_parser::Parser;

/// Parses package-form block IR and emits its selected block as SystemVerilog.
pub fn handle_block2sv(matches: &ArgMatches) -> anyhow::Result<()> {
    let input_path = matches
        .get_one::<String>("input_file")
        .expect("clap requires an input file");
    let input = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read block IR file '{input_path}'"))?;
    let package = Parser::new(&input)
        .parse_package()
        .map_err(|error| anyhow::anyhow!("failed to parse block IR: {error}"))?;

    let mut options = BlockCodegenOptions::default();
    options.top = matches.get_one::<String>("top").cloned();
    options.module_name = matches.get_one::<String>("module_name").cloned();
    options.layout = match matches
        .get_one::<String>("layout")
        .expect("clap supplies a layout default")
        .as_str()
    {
        "none" => Layout::None,
        "pipeline" => Layout::Pipeline,
        _ => unreachable!("clap validates the layout value"),
    };

    if let Some(value) = matches.get_one::<bool>("array_index_bounds_checking") {
        options.array_index_bounds_checking = *value;
    }
    if let Some(value) = matches.get_one::<bool>("separate_lines") {
        options.separate_lines = *value;
    }
    if let Some(value) = matches.get_one::<usize>("max_inline_depth") {
        options.max_inline_depth = *value;
    }
    if let Some(value) = matches.get_one::<bool>("emit_sv_types") {
        options.emit_sv_types = *value;
    }
    if let Some(value) = matches.get_one::<bool>("add_invariant_assertions") {
        options.add_invariant_assertions = *value;
    }
    if let Some(path) = matches.get_one::<String>("register_codegen_options") {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read register codegen options '{path}'"))?;
        options.register_codegen_options = Some(
            toml::from_str(&text)
                .with_context(|| format!("failed to parse register codegen options '{path}'"))?,
        );
    }

    let output = emit_system_verilog(&package, &options)?;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    stdout.write_all(output.system_verilog.as_bytes())?;
    if !output.system_verilog.ends_with('\n') {
        stdout.write_all(b"\n")?;
    }
    Ok(())
}
