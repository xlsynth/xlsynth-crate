// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, anyhow};
use clap::ArgMatches;
use std::path::Path;
use xlsynth_g8r::netlist::buffer::BufferOptions;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{
    load_liberty_with_timing_data_from_path, parse_netlist_from_path, select_module,
};
use xlsynth_g8r::netlist::optimize::{
    NetlistOptimizationOptions, NetlistOptimizationStats, optimize_mapped_netlist,
};
use xlsynth_g8r::netlist::resize::ResizeOptions;
use xlsynth_g8r::netlist::sta::StaOptions;

/// Buffers and resizes an existing timing-complete mapped netlist.
pub fn handle_gv_optimize(matches: &ArgMatches) -> Result<()> {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist is required");
    let liberty_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required");
    let output_path = matches
        .get_one::<String>("netlist_out")
        .expect("netlist_out is required");
    let module_name = matches.get_one::<String>("module_name").map(String::as_str);

    let mut parsed = parse_netlist_from_path(Path::new(netlist_path))
        .with_context(|| format!("failed to parse mapped netlist '{netlist_path}'"))?;
    let selected = select_module(&parsed, module_name)
        .context("failed to select the mapped module to optimize")?;
    let module_index = parsed
        .modules
        .iter()
        .position(|candidate| std::ptr::eq(candidate, selected))
        .ok_or_else(|| anyhow!("selected mapped module is missing from the parsed netlist"))?;
    let mut module = parsed.modules.remove(module_index);

    let library = load_liberty_with_timing_data_from_path(Path::new(liberty_path))
        .with_context(|| format!("failed to load timing-enabled Liberty proto '{liberty_path}'"))?;
    let sta_options = StaOptions {
        primary_input_transition: *matches
            .get_one::<f64>("primary_input_transition")
            .expect("primary_input_transition has a default"),
        module_output_load: *matches
            .get_one::<f64>("module_output_load")
            .expect("module_output_load has a default"),
    };
    let options = NetlistOptimizationOptions {
        sta_options,
        buffer_options: matches
            .get_one::<bool>("buffer")
            .copied()
            .expect("buffer has a default")
            .then(|| BufferOptions {
                max_fanout: *matches
                    .get_one::<usize>("max_fanout")
                    .expect("max_fanout has a default"),
                target_load: matches.get_one::<f64>("buffer_target_load").copied(),
                module_output_load: sta_options.module_output_load,
                buffer_primary_inputs: matches.get_flag("buffer_primary_inputs"),
            }),
        resize_options: matches
            .get_one::<bool>("resize")
            .copied()
            .expect("resize has a default")
            .then(|| ResizeOptions {
                sta_options,
                max_iterations: *matches
                    .get_one::<usize>("resize_iterations")
                    .expect("resize_iterations has a default"),
                max_area_iterations: *matches
                    .get_one::<usize>("resize_area_iterations")
                    .expect("resize_area_iterations has a default"),
                max_evaluations_per_iteration: *matches
                    .get_one::<usize>("resize_max_evaluations")
                    .expect("resize_max_evaluations has a default"),
                ..ResizeOptions::default()
            }),
    };
    let stats = optimize_mapped_netlist(
        &mut module,
        &mut parsed.nets,
        &mut parsed.interner,
        &library,
        &options,
    )
    .context("mapped-netlist buffering and resizing failed")?;
    let text = emit_module_as_netlist_text(&module, &parsed.nets, &parsed.interner)
        .context("failed to emit optimized mapped netlist")?;
    if output_path == "-" {
        print!("{text}");
    } else {
        std::fs::write(output_path, text)
            .with_context(|| format!("failed to write optimized netlist '{output_path}'"))?;
    }
    if let Some(json_path) = matches.get_one::<String>("json_out") {
        write_stats_json(json_path, &stats)?;
    }

    eprintln!(
        "gv-optimize: area={} -> {}, delay={} -> {}, buffers={}, upsizes={}, downsizes={}",
        stats.initial_area,
        stats.final_area,
        stats.initial_delay,
        stats.final_delay,
        stats
            .buffer_stats
            .as_ref()
            .map_or(0, |buffer| buffer.buffers_inserted),
        stats
            .resize_stats
            .as_ref()
            .map_or(0, |resize| resize.upsizes),
        stats
            .resize_stats
            .as_ref()
            .map_or(0, |resize| resize.downsizes),
    );
    Ok(())
}

/// Writes complete, independently verified optimization diagnostics.
fn write_stats_json(path: &str, stats: &NetlistOptimizationStats) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create optimization JSON '{path}'"))?;
    serde_json::to_writer_pretty(file, stats)
        .with_context(|| format!("failed to write optimization JSON '{path}'"))
}
