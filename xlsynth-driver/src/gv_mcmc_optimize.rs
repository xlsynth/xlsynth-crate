// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, anyhow};
use clap::ArgMatches;
use std::path::Path;
use std::sync::Arc;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{
    load_liberty_with_timing_data_from_path, parse_netlist_from_path, select_module,
};
use xlsynth_g8r::netlist::mcmc::{
    NetlistMcmcObjective, NetlistMcmcOptions, optimize_mapped_netlist_mcmc,
};
use xlsynth_g8r::netlist::sta::StaOptions;
use xlsynth_g8r::netlist::timing_buffer::BufferTimingConstraints;

/// Explores equivalent cell sizes and buffer trees around a mapped netlist.
pub fn handle_gv_mcmc_optimize(matches: &ArgMatches) -> Result<()> {
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
        .context("failed to select the mapped module for MCMC exploration")?;
    let module_index = parsed
        .modules
        .iter()
        .position(|candidate| std::ptr::eq(candidate, selected))
        .ok_or_else(|| anyhow!("selected mapped module is missing from the parsed netlist"))?;
    let mut module = parsed.modules.remove(module_index);
    let library = Arc::new(
        load_liberty_with_timing_data_from_path(Path::new(liberty_path)).with_context(|| {
            format!("failed to load timing-enabled Liberty proto '{liberty_path}'")
        })?,
    );

    let options = NetlistMcmcOptions {
        objective: match matches
            .get_one::<String>("objective")
            .expect("objective has a default")
            .as_str()
        {
            "delay" => NetlistMcmcObjective::Delay,
            "area" => NetlistMcmcObjective::Area,
            objective => return Err(anyhow!("unsupported mapped MCMC objective '{objective}'")),
        },
        iterations: *matches
            .get_one::<u64>("iterations")
            .expect("iterations has a default"),
        time_limit_seconds: matches.get_one::<u64>("time_limit_seconds").copied(),
        threads: *matches
            .get_one::<usize>("threads")
            .expect("threads has a default"),
        seed: *matches.get_one::<u64>("seed").expect("seed has a default"),
        initial_temperature: *matches
            .get_one::<f64>("temperature")
            .expect("temperature has a default"),
        checkpoint_iterations: *matches
            .get_one::<u64>("checkpoint_iterations")
            .expect("checkpoint_iterations has a default"),
        sta_options: StaOptions {
            primary_input_transition: *matches
                .get_one::<f64>("primary_input_transition")
                .expect("primary_input_transition has a default"),
            module_output_load: *matches
                .get_one::<f64>("module_output_load")
                .expect("module_output_load has a default"),
        },
        timing_constraints: BufferTimingConstraints {
            clock_period: matches.get_one::<f64>("clock_period").copied(),
            ..BufferTimingConstraints::default()
        },
        delay_limit: matches.get_one::<f64>("delay_limit").copied(),
        max_area_growth: matches.get_one::<f64>("max_area_growth").copied(),
        enable_sizing: *matches
            .get_one::<bool>("sizing")
            .expect("sizing has a default"),
        enable_pin_swaps: *matches
            .get_one::<bool>("pin_swaps")
            .expect("pin_swaps has a default"),
        enable_buffer_moves: *matches
            .get_one::<bool>("buffer")
            .expect("buffer has a default"),
        enable_remap: *matches
            .get_one::<bool>("remap")
            .expect("remap has a default"),
        max_remap_leaves: *matches
            .get_one::<usize>("remap_max_leaves")
            .expect("remap_max_leaves has a default"),
        remap_relax_evaluations: *matches
            .get_one::<usize>("remap_relax_evaluations")
            .expect("remap_relax_evaluations has a default"),
        buffer_primary_inputs: matches.get_flag("buffer_primary_inputs"),
        max_buffer_fanout: *matches
            .get_one::<usize>("max_buffer_fanout")
            .expect("max_buffer_fanout has a default"),
        critical_window: *matches
            .get_one::<f64>("critical_window")
            .expect("critical_window has a default"),
    };
    let stats = optimize_mapped_netlist_mcmc(
        &mut module,
        &mut parsed.nets,
        &mut parsed.interner,
        library,
        &options,
    )
    .context("mapped-netlist MCMC exploration failed")?;
    let text = emit_module_as_netlist_text(&module, &parsed.nets, &parsed.interner)
        .context("failed to emit MCMC-optimized mapped netlist")?;
    if output_path == "-" {
        print!("{text}");
    } else {
        std::fs::write(output_path, text)
            .with_context(|| format!("failed to write optimized netlist '{output_path}'"))?;
    }
    if let Some(path) = matches.get_one::<String>("json_out") {
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create MCMC diagnostics '{path}'"))?;
        serde_json::to_writer_pretty(file, &stats)
            .with_context(|| format!("failed to write MCMC diagnostics '{path}'"))?;
    }
    eprintln!(
        "gv-mcmc-optimize: area={} -> {}, delay={} -> {}, accepted={}/{}, incremental-evals={}, complete-evals={}, seconds={:.3}",
        stats.initial_area,
        stats.final_area,
        stats.initial_delay,
        stats.final_delay,
        stats.accepted_moves,
        stats.attempted_moves,
        stats.incremental_timing_evaluations,
        stats.complete_timing_evaluations,
        stats.elapsed_seconds,
    );
    Ok(())
}
