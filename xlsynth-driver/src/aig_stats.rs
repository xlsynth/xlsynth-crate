// SPDX-License-Identifier: Apache-2.0

//! Reads an AIGER file and reports structural + logical-effort statistics.

use std::collections::BTreeMap;
use std::path::Path;

use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    analyze_graph_logical_effort, GraphLogicalEffortOptions,
};
use xlsynth_g8r::aig::logical_effort::{self, compute_logical_effort_min_delay};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;

#[derive(Debug, serde::Serialize)]
struct AigStatsOutput {
    and_nodes: usize,
    depth: usize,
    fanout_histogram: BTreeMap<usize, usize>,
    logical_effort_deepest_path_min_delay: f64,
    graph_logical_effort_worst_case_delay: Option<f64>,
}

fn parse_bool(matches: &ArgMatches, name: &str, default: bool) -> bool {
    match matches.get_one::<String>(name).map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => default,
    }
}

fn format_fanout_histogram(hist: &BTreeMap<usize, usize>) -> String {
    let mut s = String::new();
    s.push('{');
    for (i, (fanout, count)) in hist.iter().enumerate() {
        if i != 0 {
            s.push(',');
        }
        s.push_str(&format!("{}:{}", fanout, count));
    }
    s.push('}');
    s
}

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

pub fn handle_aig_stats(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let output_json = matches.get_one::<String>("output_json");
    let quiet = parse_bool(matches, "quiet", false);
    let compute_graph_logical_effort = parse_bool(matches, "compute_graph_logical_effort", true);
    let graph_logical_effort_beta1 = *matches
        .get_one::<f64>("graph_logical_effort_beta1")
        .expect("graph_logical_effort_beta1 has a default and is parsed by clap");
    let graph_logical_effort_beta2 = *matches
        .get_one::<f64>("graph_logical_effort_beta2")
        .expect("graph_logical_effort_beta2 has a default and is parsed by clap");

    let gate_fn = match load_aig_gate_fn(Path::new(input_file)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("aig-stats error: {}", e);
            std::process::exit(2);
        }
    };

    let stats = get_aig_stats(&gate_fn);
    let logical_effort_deepest_path_min_delay =
        compute_logical_effort_min_delay(&gate_fn, &logical_effort::Options::default());
    let graph_logical_effort_worst_case_delay = if compute_graph_logical_effort {
        Some(
            analyze_graph_logical_effort(
                &gate_fn,
                &GraphLogicalEffortOptions {
                    beta1: graph_logical_effort_beta1,
                    beta2: graph_logical_effort_beta2,
                },
            )
            .delay,
        )
    } else {
        None
    };

    let out = AigStatsOutput {
        and_nodes: stats.and_nodes,
        depth: stats.max_depth,
        fanout_histogram: stats.fanout_histogram,
        logical_effort_deepest_path_min_delay,
        graph_logical_effort_worst_case_delay,
    };

    if quiet {
        serde_json::to_writer(std::io::stdout(), &out).unwrap();
        println!();
    } else {
        println!(
            "aig stats: and_nodes={} depth={} fanout_hist={}",
            out.and_nodes,
            out.depth,
            format_fanout_histogram(&out.fanout_histogram)
        );
        println!(
            "== Logical effort deepest path min delay: {:.4} (FO4 units)",
            out.logical_effort_deepest_path_min_delay
        );
        if let Some(delay) = out.graph_logical_effort_worst_case_delay {
            println!(
                "== Graph logical effort worst case delay: {:.4} (FO4 units)",
                delay
            );
        }
    }

    if let Some(path) = output_json.map(|s| Path::new(s)) {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {}", path.display(), e));
        serde_json::to_writer_pretty(file, &out)
            .unwrap_or_else(|e| panic!("Failed to write JSON: {}", e));
    }
}
