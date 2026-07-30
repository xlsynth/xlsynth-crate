// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, anyhow};
use clap::ArgMatches;
use std::collections::BTreeMap;
use std::path::Path;
use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;
use xlsynth_g8r::aig_serdes::load_abc_choice_aiger::load_abc_choice_aiger_auto_from_path;
use xlsynth_g8r::netlist::buffer::BufferOptions;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::load_liberty_with_timing_data_from_path;
use xlsynth_g8r::netlist::resize::ResizeOptions;
use xlsynth_g8r::techmap::{
    SequentialTechMapConstraints, TechMapOptions, TechMapTimingConstraints, TechMapTimingModel,
    map_choice_aig_to_netlist, map_sequential_choice_aig_to_netlist,
};

/// Parses a finite, strictly positive sequential clock period.
pub(crate) fn parse_positive_finite_clock_period(raw: &str) -> std::result::Result<f64, String> {
    let clock_period = raw
        .parse::<f64>()
        .map_err(|_| format!("clock period must be a finite positive time, got '{raw}'"))?;
    if !clock_period.is_finite() || clock_period <= 0.0 {
        return Err(format!(
            "clock period must be a finite positive time, got '{raw}'"
        ));
    }
    Ok(clock_period)
}

/// Runs the clean-sheet, final-only choice-AIG technology mapper.
pub fn handle_choice_aig_tech_map(matches: &ArgMatches) -> Result<()> {
    let aig_input_file = matches
        .get_one::<String>("aig_input_file")
        .expect("aig_input_file is required");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required");
    let netlist_out = matches
        .get_one::<String>("netlist_out")
        .expect("netlist_out is required");
    let sequential_design = matches.get_one::<String>("sequential_design");
    let clock_period = matches.get_one::<f64>("clock_period").copied();
    let choice_aig = load_abc_choice_aiger_auto_from_path(Path::new(aig_input_file))
        .map_err(|error| anyhow!("failed to load AIG '{}': {}", aig_input_file, error))?;
    let library = load_liberty_with_timing_data_from_path(Path::new(liberty_proto_path))
        .with_context(|| {
            format!(
                "failed to load timing-enabled Liberty proto '{}'",
                liberty_proto_path
            )
        })?;
    let primary_input_arrivals =
        parse_named_times(matches, "primary_input_arrival", "--primary-input-arrival")?;
    let primary_output_required = parse_named_times(
        matches,
        "primary_output_required",
        "--primary-output-required",
    )?;
    let options = TechMapOptions {
        module_name: matches.get_one::<String>("module_name").cloned(),
        max_cut_size: *matches
            .get_one::<usize>("max_cut_size")
            .expect("max_cut_size has a default"),
        max_cuts_per_node: *matches
            .get_one::<usize>("max_cuts_per_node")
            .expect("max_cuts_per_node has a default"),
        max_frontier_size: *matches
            .get_one::<usize>("max_frontier_size")
            .expect("max_frontier_size has a default"),
        primary_input_transition: *matches
            .get_one::<f64>("primary_input_transition")
            .expect("primary_input_transition has a default"),
        module_output_load: *matches
            .get_one::<f64>("module_output_load")
            .expect("module_output_load has a default"),
        timing_model: match matches
            .get_one::<String>("timing_model")
            .expect("timing_model has a default")
            .as_str()
        {
            "balanced" => TechMapTimingModel::Balanced,
            "nf-unit" => TechMapTimingModel::NfUnit,
            "nf-liberty" => TechMapTimingModel::NfLiberty,
            "buffered-liberty" => TechMapTimingModel::BufferedLiberty,
            value => return Err(anyhow!("unsupported mapping timing model '{value}'")),
        },
        buffer_options: matches
            .get_one::<bool>("buffer")
            .copied()
            .expect("buffer has a default")
            .then(|| BufferOptions {
                max_fanout: *matches
                    .get_one::<usize>("max_fanout")
                    .expect("max_fanout has a default"),
                target_load: matches.get_one::<f64>("buffer_target_load").copied(),
                module_output_load: *matches
                    .get_one::<f64>("module_output_load")
                    .expect("module_output_load has a default"),
                buffer_primary_inputs: matches.get_flag("buffer_primary_inputs"),
            }),
        resize_options: matches
            .get_one::<bool>("resize")
            .copied()
            .expect("resize has a default")
            .then(|| ResizeOptions {
                max_outer_iterations: *matches
                    .get_one::<usize>("resize_rounds")
                    .expect("resize_rounds has a default"),
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
    let mapped = if let Some(design_path) = sequential_design {
        let design =
            load_sequential_gate_fn_from_path(Path::new(design_path)).map_err(|error| {
                anyhow!(
                    "failed to load sequential design '{}': {}",
                    design_path,
                    error
                )
            })?;
        let constraints = SequentialTechMapConstraints {
            primary_input_arrivals,
            primary_output_required,
            clock_period,
        };
        map_sequential_choice_aig_to_netlist(&design, &choice_aig, &library, &constraints, &options)
            .context("sequential choice-AIG technology mapping failed")?
    } else {
        let constraints = TechMapTimingConstraints {
            primary_input_arrivals,
            primary_output_required,
        };
        map_choice_aig_to_netlist(&choice_aig, &library, &constraints, &options)
            .context("final choice-AIG technology mapping failed")?
    };
    let text =
        emit_module_as_netlist_text(&mapped.module, mapped.nets.as_slice(), &mapped.interner)
            .context("failed to emit mapped netlist text")?;
    if netlist_out == "-" {
        print!("{}", text);
    } else {
        std::fs::write(netlist_out, text)
            .with_context(|| format!("failed to write mapped netlist '{}'", netlist_out))?;
    }
    let representative_timing = match (
        mapped.stats.representative_input_transition,
        mapped.stats.representative_output_load,
    ) {
        (Some(transition), Some(load)) => {
            format!(", representative-slew={transition}, representative-load={load}")
        }
        _ => String::new(),
    };
    let sequential_diagnostics = if sequential_design.is_some() {
        let mut diagnostics = format!(
            ", registers={}, register-area={}",
            mapped.stats.sequential_instance_count, mapped.stats.sequential_area
        );
        if let Some(clock_period) = mapped.stats.clock_period {
            diagnostics.push_str(&format!(", clock-period={clock_period}"));
        }
        if let Some(slack) = mapped.stats.worst_register_slack {
            diagnostics.push_str(&format!(", worst-register-slack={slack}"));
        }
        diagnostics
    } else {
        String::new()
    };
    eprintln!(
        "choice-aig-tech-map: {} instances, area={}, delay={}, choices={}, cuts={}, candidates={}, buffers={}, upsizes={}, downsizes={}, timing-model={}{}{}",
        mapped.stats.selected_instance_count,
        mapped.stats.selected_area,
        mapped.stats.worst_estimated_output_arrival,
        mapped.stats.choice_link_count,
        mapped.stats.enumerated_cut_count,
        mapped.stats.matched_candidate_count,
        mapped
            .stats
            .buffer_stats
            .as_ref()
            .map_or(0, |stats| stats.buffers_inserted),
        mapped
            .stats
            .resize_stats
            .as_ref()
            .map_or(0, |stats| stats.upsizes),
        mapped
            .stats
            .resize_stats
            .as_ref()
            .map_or(0, |stats| stats.downsizes),
        match mapped.stats.selected_timing_model {
            TechMapTimingModel::Balanced => "balanced",
            TechMapTimingModel::NfUnit => "nf-unit",
            TechMapTimingModel::NfLiberty => "nf-liberty",
            TechMapTimingModel::BufferedLiberty => "buffered-liberty",
        },
        representative_timing,
        sequential_diagnostics,
    );
    Ok(())
}

fn parse_named_times(
    matches: &ArgMatches,
    argument_name: &str,
    flag_name: &str,
) -> Result<BTreeMap<String, f64>> {
    let mut values = BTreeMap::new();
    let Some(raw_values) = matches.get_many::<String>(argument_name) else {
        return Ok(values);
    };
    for raw in raw_values {
        let (name, value) = raw
            .split_once('=')
            .ok_or_else(|| anyhow!("{} expects NAME=TIME, got '{}'", flag_name, raw))?;
        if name.is_empty() {
            return Err(anyhow!("{} has an empty port name in '{}'", flag_name, raw));
        }
        let value: f64 = value
            .parse()
            .map_err(|_| anyhow!("{} has invalid time in '{}'", flag_name, raw))?;
        if !value.is_finite() {
            return Err(anyhow!(
                "{} requires a finite time, got '{}'",
                flag_name,
                raw
            ));
        }
        if values.insert(name.to_string(), value).is_some() {
            return Err(anyhow!(
                "{} specifies primary port '{}' more than once",
                flag_name,
                name
            ));
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::parse_positive_finite_clock_period;

    #[test]
    fn clock_period_parser_accepts_positive_finite_times() {
        assert_eq!(parse_positive_finite_clock_period("0.01"), Ok(0.01));
        assert_eq!(parse_positive_finite_clock_period("100"), Ok(100.0));
    }

    #[test]
    fn clock_period_parser_rejects_non_positive_and_non_finite_times() {
        for invalid in ["0", "-0", "-1", "NaN", "inf", "-inf", "not-a-time"] {
            let error = parse_positive_finite_clock_period(invalid)
                .expect_err("clock periods must be positive and finite");
            assert!(
                error.contains("finite positive"),
                "unexpected error for '{invalid}': {error}"
            );
        }
    }
}
