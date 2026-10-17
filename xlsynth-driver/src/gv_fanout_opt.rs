// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::{Arg, ArgAction, ArgMatches, Command};
use xlsynth_g8r::liberty::Library;
use xlsynth_g8r::liberty::io_defaults::default_output_load;
use xlsynth_g8r::liberty::load::load_library_with_timing_data_from_path;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::fanout_opt::{FanoutOptimizationOptions, optimize_fanout};
use xlsynth_g8r::netlist::io::{ParsedNetlist, read_gv_from_path, read_gv_from_str, select_module};
use xlsynth_g8r::netlist::mapped::MappedNetlist;

/// Exposes bounded timing-driven fanout optimization independently of mapping.
pub fn command() -> Command {
    let defaults = FanoutOptimizationOptions::default();
    Command::new("gv-fanout-opt")
        .about(
            "Improve modeled timing through buffering, replication and upsizing; emit GV on stdout",
        )
        .arg(Arg::new("netlist").long("netlist").required(true))
        .arg(
            Arg::new("liberty_proto")
                .long("liberty_proto")
                .required(true),
        )
        .arg(Arg::new("module_name").long("module_name"))
        .arg(
            Arg::new("primary_input_transition")
                .long("primary_input_transition")
                .help(format!(
                    "Primary-input transition [default: {}]",
                    defaults.primary_input_transition
                ))
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("module_output_load")
                .long("module_output_load")
                .help("Output capacitance; defaults to the library default_output_load")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("max-area-growth")
                .long("max-area-growth")
                .help(format!(
                    "Fractional area allowance [default: {}]",
                    defaults.area_growth_limit
                ))
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("area-weight")
                .long("area-weight")
                .help(format!(
                    "Area-growth weight in the normalized delay/area score [default: {}]",
                    defaults.area_weight
                ))
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("rounds")
                .long("rounds")
                .help(format!(
                    "Search rounds, at most 32 [default: {}]",
                    defaults.max_rounds
                ))
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("drivers")
                .long("drivers")
                .help(format!(
                    "Candidate drivers per round, 1 through 8 [default: {}]",
                    defaults.drivers_per_round
                ))
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("fanout-trigger")
                .long("fanout-trigger")
                .help(format!(
                    "Sink count that prioritizes a driver [default: {}]",
                    defaults.fanout_trigger
                ))
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("full-sta")
                .long("full-sta")
                .help("Evaluate each candidate with full STA instead of incremental STA")
                .action(ArgAction::SetTrue)
                .conflicts_with("verify-incremental"),
        )
        .arg(
            Arg::new("verify-incremental")
                .long("verify-incremental")
                .help("Compare each incremental trial with independent full STA")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("report-json")
                .long("report-json")
                .help("Write deterministic decisions, metrics and work counts"),
        )
        .arg(
            Arg::new("profile-json")
                .long("profile-json")
                .help("Write the report with nondeterministic runtime measurements"),
        )
}

/// Isolates the selected module so added nets belong to a contiguous owned
/// range.
fn owned_module(parsed: &ParsedNetlist, name: Option<&str>) -> Result<MappedNetlist, String> {
    let module = select_module(parsed, name).map_err(|e| e.to_string())?;
    let text = emit_module_as_netlist_text(module, &parsed.nets, &parsed.interner)
        .map_err(|e| e.to_string())?;
    let mut isolated = read_gv_from_str(&text).map_err(|e| e.to_string())?;
    Ok(MappedNetlist {
        module: isolated.modules.remove(0),
        nets: isolated.nets,
        interner: isolated.interner,
    })
}

/// Reads one flat mapped module and delegates optimization to the library.
pub fn handle(matches: &ArgMatches) -> Result<(), String> {
    let library = load_library_with_timing_data_from_path(Path::new(
        matches.get_one::<String>("liberty_proto").unwrap(),
    ))
    .map_err(|e| e.to_string())?;
    let parsed = read_gv_from_path(Path::new(matches.get_one::<String>("netlist").unwrap()))
        .map_err(|e| e.to_string())?;
    let mut netlist = owned_module(
        &parsed,
        matches.get_one::<String>("module_name").map(String::as_str),
    )?;
    let options = optimization_options(matches, &library)?;
    let report = optimize_fanout(&mut netlist, &library, options).map_err(|e| e.to_string())?;
    let verilog = emit_module_as_netlist_text(&netlist.module, &netlist.nets, &netlist.interner)
        .map_err(|e| e.to_string())?;
    let mut json = serde_json::to_value(&report).map_err(|e| e.to_string())?;
    if let Some(path) = matches.get_one::<String>("profile-json") {
        write_report(path, &json)?;
    }
    if let Some(path) = matches.get_one::<String>("report-json") {
        remove_runtime(&mut json);
        write_report(path, &json)?;
    }
    print!("{verilog}");
    Ok(())
}

/// Applies explicit overrides to library-owned search defaults.
fn optimization_options(
    matches: &ArgMatches,
    library: &Library,
) -> Result<FanoutOptimizationOptions, String> {
    let defaults = FanoutOptimizationOptions::default();
    Ok(FanoutOptimizationOptions {
        full_sta: matches.get_flag("full-sta"),
        verify_incremental: matches.get_flag("verify-incremental"),
        primary_input_transition: matches
            .get_one::<f64>("primary_input_transition")
            .copied()
            .unwrap_or(defaults.primary_input_transition),
        module_output_load: match matches.get_one::<f64>("module_output_load") {
            Some(load) => *load,
            None => default_output_load(library)?,
        },
        area_growth_limit: matches
            .get_one::<f64>("max-area-growth")
            .copied()
            .unwrap_or(defaults.area_growth_limit),
        area_weight: matches
            .get_one::<f64>("area-weight")
            .copied()
            .unwrap_or(defaults.area_weight),
        max_rounds: matches
            .get_one::<usize>("rounds")
            .copied()
            .unwrap_or(defaults.max_rounds),
        drivers_per_round: matches
            .get_one::<usize>("drivers")
            .copied()
            .unwrap_or(defaults.drivers_per_round),
        fanout_trigger: matches
            .get_one::<usize>("fanout-trigger")
            .copied()
            .unwrap_or(defaults.fanout_trigger),
    })
}

/// Removes explicitly named elapsed-time fields while preserving deterministic
/// work.
fn remove_runtime(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => {
            fields.retain(|key, _| !key.ends_with("_seconds"));
            for value in fields.values_mut() {
                remove_runtime(value);
            }
        }
        serde_json::Value::Array(values) => values.iter_mut().for_each(remove_runtime),
        _ => {}
    }
}

/// Writes newline-terminated report JSON without mixing diagnostics into GV.
fn write_report(path: &str, value: &serde_json::Value) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, format!("{json}\n"))
        .map_err(|e| format!("writing fanout optimization report: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_inherit_defaults_and_allow_explicit_zero_load() -> anyhow::Result<()> {
        let mut library = Library::default();
        library.default_output_load = Some(4.0);
        let args = [
            "gv-fanout-opt",
            "--netlist",
            "input.gv",
            "--liberty_proto",
            "cells.proto",
        ];
        let matches = command().try_get_matches_from(args)?;
        let options = optimization_options(&matches, &library).map_err(anyhow::Error::msg)?;
        let defaults = FanoutOptimizationOptions::default();
        assert_eq!(
            options.primary_input_transition,
            defaults.primary_input_transition
        );
        assert_eq!(options.module_output_load, 4.0);
        assert_eq!(options.max_rounds, defaults.max_rounds);
        assert_eq!(options.drivers_per_round, defaults.drivers_per_round);
        assert_eq!(options.area_growth_limit, defaults.area_growth_limit);
        assert_eq!(options.area_weight, defaults.area_weight);
        assert_eq!(options.fanout_trigger, defaults.fanout_trigger);
        library.default_output_load = None;
        assert!(optimization_options(&matches, &library).is_err());
        let matches = command().try_get_matches_from(args.into_iter().chain([
            "--primary_input_transition",
            "0",
            "--module_output_load",
            "0",
            "--max-area-growth",
            "0",
            "--area-weight",
            "0",
            "--rounds",
            "0",
            "--drivers",
            "1",
            "--fanout-trigger",
            "1",
            "--verify-incremental",
        ]))?;
        let options = optimization_options(&matches, &library).map_err(anyhow::Error::msg)?;
        assert_eq!(options.primary_input_transition, 0.0);
        assert_eq!(options.module_output_load, 0.0);
        assert_eq!(options.area_growth_limit, 0.0);
        assert_eq!(options.area_weight, 0.0);
        assert_eq!(options.max_rounds, 0);
        assert_eq!(options.drivers_per_round, 1);
        assert_eq!(options.fanout_trigger, 1);
        assert!(options.verify_incremental);
        Ok(())
    }

    #[test]
    fn selected_module_owns_its_nets_and_report_omits_runtime() -> anyhow::Result<()> {
        let parsed = read_gv_from_str(
            r#"
module first(a,y); input a; output y; assign y=a; endmodule
module second(a,y); input a; output y; assign y=a; endmodule
"#,
        )?;
        assert!(owned_module(&parsed, None).is_err());
        let netlist = owned_module(&parsed, Some("second")).map_err(anyhow::Error::msg)?;
        assert_eq!(netlist.module.net_index_range, (0..netlist.nets.len()));
        let text = emit_module_as_netlist_text(&netlist.module, &netlist.nets, &netlist.interner)?;
        assert!(text.contains("module second"));
        assert!(!text.contains("module first"));
        let mut report = serde_json::json!({"optimization_seconds": 1.0, "before": {"area": 2.0}, "incremental": {"candidates": [{"nodes": 3, "propagation_seconds": 0.5}]}});
        remove_runtime(&mut report);
        assert_eq!(
            report,
            serde_json::json!({"before": {"area": 2.0}, "incremental": {"candidates": [{"nodes": 3}]}})
        );
        Ok(())
    }
}
