// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::ArgMatches;
use std::path::Path;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{
    ParsedNetlist, load_liberty_with_timing_data_from_path, parse_netlist_from_path, select_module,
};
use xlsynth_g8r::netlist::parse::NetlistModule;
use xlsynth_g8r::netlist::resize::{
    AreaResizeOptions, AreaResizeSummary, DelayResizeOptions, DelayResizeSummary,
    resize_for_minimum_area_under_register_delay, resize_for_minimum_register_delay,
};
use xlsynth_g8r::netlist::sta::StaOptions;

pub fn handle_gv_resize(matches: &ArgMatches) -> Result<()> {
    let netlist_path = Path::new(
        matches
            .get_one::<String>("netlist")
            .expect("required netlist"),
    );
    let liberty_proto_path = Path::new(
        matches
            .get_one::<String>("liberty_proto")
            .expect("required liberty_proto"),
    );
    let netlist_out = Path::new(
        matches
            .get_one::<String>("netlist_out")
            .expect("required netlist_out"),
    );
    let objective = matches
        .get_one::<String>("objective")
        .expect("objective has default");
    let module_name = matches.get_one::<String>("module_name").map(String::as_str);
    let primary_input_transition = *matches
        .get_one::<f64>("primary_input_transition")
        .expect("primary_input_transition has default");
    let module_output_load = *matches
        .get_one::<f64>("module_output_load")
        .expect("module_output_load has default");
    let max_iterations = *matches
        .get_one::<usize>("max_iterations")
        .expect("max_iterations has default");
    let max_candidate_paths = *matches
        .get_one::<usize>("max_candidate_paths")
        .expect("max_candidate_paths has default");
    let max_evaluations_per_iteration = *matches
        .get_one::<usize>("max_evaluations_per_iteration")
        .expect("max_evaluations_per_iteration has default");
    let max_cell_candidates_per_instance = *matches
        .get_one::<usize>("max_cell_candidates_per_instance")
        .expect("max_cell_candidates_per_instance has default");
    let improvement_epsilon = *matches
        .get_one::<f64>("improvement_epsilon")
        .expect("improvement_epsilon has default");
    let area_epsilon = *matches
        .get_one::<f64>("area_epsilon")
        .expect("area_epsilon has default");
    let resize_sequential_cells = *matches
        .get_one::<bool>("resize_sequential_cells")
        .expect("resize_sequential_cells has default");
    let dont_use_patterns = matches
        .get_many::<String>("dont_use")
        .map(|values| values.cloned().collect())
        .unwrap_or_default();

    let mut parsed = parse_netlist_from_path(netlist_path)
        .with_context(|| format!("failed to parse netlist '{}'", netlist_path.display()))?;
    let module = select_module(&parsed, module_name)?.clone();
    let liberty =
        load_liberty_with_timing_data_from_path(liberty_proto_path).with_context(|| {
            format!(
                "failed to load timing-enabled Liberty proto '{}'",
                liberty_proto_path.display()
            )
        })?;
    let sta_options = StaOptions {
        primary_input_transition,
        module_output_load,
    };
    match objective.as_str() {
        "delay" => {
            let result = resize_for_minimum_register_delay(
                &module,
                parsed.nets.as_slice(),
                &mut parsed.interner,
                &liberty,
                &DelayResizeOptions {
                    sta_options,
                    max_iterations,
                    max_candidate_paths,
                    max_evaluations_per_iteration,
                    max_cell_candidates_per_instance,
                    improvement_epsilon,
                    resize_sequential_cells,
                    dont_use_patterns,
                },
            )?;
            write_netlist(&result.module, &parsed, netlist_out)?;
            render_delay_summary(&result.summary);
            write_json_summary(matches, &result.summary)?;
        }
        "area-under-delay" => {
            let result = resize_for_minimum_area_under_register_delay(
                &module,
                parsed.nets.as_slice(),
                &mut parsed.interner,
                &liberty,
                &AreaResizeOptions {
                    sta_options,
                    max_iterations,
                    max_evaluations_per_iteration,
                    max_cell_candidates_per_instance,
                    delay_epsilon: improvement_epsilon,
                    area_epsilon,
                    resize_sequential_cells,
                    dont_use_patterns,
                },
            )?;
            write_netlist(&result.module, &parsed, netlist_out)?;
            render_area_summary(&result.summary);
            write_json_summary(matches, &result.summary)?;
        }
        _ => unreachable!("clap restricts objective values"),
    }
    Ok(())
}

fn write_netlist(module: &NetlistModule, parsed: &ParsedNetlist, netlist_out: &Path) -> Result<()> {
    let netlist_text =
        emit_module_as_netlist_text(module, parsed.nets.as_slice(), &parsed.interner)?;
    std::fs::write(netlist_out, netlist_text)
        .with_context(|| format!("failed writing resized netlist '{}'", netlist_out.display()))
}

fn write_json_summary<T: serde::Serialize>(matches: &ArgMatches, summary: &T) -> Result<()> {
    if let Some(json_out) = matches.get_one::<String>("json_out") {
        let file = std::fs::File::create(json_out)
            .with_context(|| format!("failed creating JSON summary '{}'", json_out))?;
        serde_json::to_writer_pretty(file, summary)
            .with_context(|| format!("failed writing JSON summary '{}'", json_out))?;
    }
    Ok(())
}

fn render_delay_summary(summary: &DelayResizeSummary) {
    println!("initial_delay: {:.6}", summary.initial_delay);
    println!("final_delay: {:.6}", summary.final_delay);
    println!(
        "delay_improvement: {:.6}",
        summary.initial_delay - summary.final_delay
    );
    println!("initial_area: {:.6}", summary.initial_area);
    println!("final_area: {:.6}", summary.final_area);
    println!("evaluations: {}", summary.evaluations);
    println!("failed_evaluations: {}", summary.failed_evaluations);
    println!("replacements: {}", summary.replacements.len());
    for replacement in &summary.replacements {
        println!(
            "  iteration={} instance={} old_cell={} new_cell={} delay_before={:.6} delay_after={:.6}",
            replacement.iteration,
            replacement.instance_name,
            replacement.old_cell,
            replacement.new_cell,
            replacement.delay_before,
            replacement.delay_after
        );
    }
}

fn render_area_summary(summary: &AreaResizeSummary) {
    println!("delay_limit: {:.6}", summary.delay_limit);
    println!("initial_delay: {:.6}", summary.initial_delay);
    println!("final_delay: {:.6}", summary.final_delay);
    println!("initial_area: {:.6}", summary.initial_area);
    println!("final_area: {:.6}", summary.final_area);
    println!(
        "area_improvement: {:.6}",
        summary.initial_area - summary.final_area
    );
    println!("evaluations: {}", summary.evaluations);
    println!("failed_evaluations: {}", summary.failed_evaluations);
    println!("replacements: {}", summary.replacements.len());
    for replacement in &summary.replacements {
        println!(
            "  iteration={} instance={} old_cell={} new_cell={} area_before={:.6} area_after={:.6} delay_before={:.6} delay_after={:.6}",
            replacement.iteration,
            replacement.instance_name,
            replacement.old_cell,
            replacement.new_cell,
            replacement.area_before,
            replacement.area_after,
            replacement.delay_before,
            replacement.delay_after
        );
    }
}
