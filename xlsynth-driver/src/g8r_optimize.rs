// SPDX-License-Identifier: Apache-2.0

//! CLI shim for the reusable post-gatification GateFn optimizer.

use std::path::Path;

use clap::ArgMatches;
use serde::Serialize;
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig::fraig::FraigPassStat;
use xlsynth_g8r::aig::get_summary_stats::{AigStats, get_aig_stats};
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::emit_netlist;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary, load_gate_fn_from_path};
use xlsynth_g8r::gate_fn_optimize::optimize_gate_fn;

use crate::common::{parse_bool_flag_or, write_stdout};

#[derive(Serialize)]
struct AigMetrics {
    and_nodes: usize,
    levels: usize,
}

impl From<&AigStats> for AigMetrics {
    fn from(stats: &AigStats) -> Self {
        Self {
            and_nodes: stats.and_nodes,
            levels: stats.max_depth,
        }
    }
}

#[derive(Serialize)]
struct G8rOptimizeStats {
    input: AigMetrics,
    output: AigMetrics,
    fraig_pass_stat: Option<FraigPassStat>,
}

/// Runs post-gatification optimizations on a combinational native g8r design.
pub fn handle_g8r_optimize(matches: &ArgMatches) -> Result<(), String> {
    let input_file = matches
        .get_one::<String>("g8r_input_file")
        .expect("clap requires a g8r input file");
    let input_path = Path::new(input_file);
    let gate_fn = load_gate_fn_from_path(input_path).map_err(|error| {
        format!(
            "g8r-optimize input '{}' must be clockless and register-free: {error}",
            input_path.display()
        )
    })?;
    let input_stats = get_aig_stats(&gate_fn);
    let options = crate::g8r_cli::parse_gate_fn_optimize_options(matches);
    let outcome = optimize_gate_fn(gate_fn, &options)?;
    let output_stats = get_aig_stats(&outcome.gate_fn);
    let stats = G8rOptimizeStats {
        input: AigMetrics::from(&input_stats),
        output: AigMetrics::from(&output_stats),
        fraig_pass_stat: outcome.fraig_pass_stat,
    };
    let design = SequentialGateFn::from_gate_fn(outcome.gate_fn);

    if !parse_bool_flag_or(matches, "quiet", false) {
        write_stdout(&emit_g8r(&design));
    }

    if let Some(path) = matches.get_one::<String>("bin_out") {
        let bytes = encode_g8r_binary(&design)
            .map_err(|error| format!("could not serialize optimized g8r binary: {error}"))?;
        std::fs::write(path, bytes)
            .map_err(|error| format!("could not write optimized g8r binary '{path}': {error}"))?;
    }

    if let Some(path) = matches.get_one::<String>("aiger_out") {
        let gate_fn = design
            .clone()
            .try_into_gate_fn()
            .expect("g8r-optimize produces a combinational design");
        let is_binary = Path::new(path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("aig"));
        if is_binary {
            let bytes = emit_aiger_binary(&gate_fn, true)
                .map_err(|error| format!("could not emit binary AIGER: {error}"))?;
            std::fs::write(path, bytes)
                .map_err(|error| format!("could not write binary AIGER '{path}': {error}"))?;
        } else {
            let text = emit_aiger(&gate_fn, true)
                .map_err(|error| format!("could not emit ASCII AIGER: {error}"))?;
            std::fs::write(path, text)
                .map_err(|error| format!("could not write ASCII AIGER '{path}': {error}"))?;
        }
    }

    if let Some(path) = matches.get_one::<String>("stats_out") {
        let mut bytes = serde_json::to_vec_pretty(&stats)
            .map_err(|error| format!("could not serialize g8r optimization stats: {error}"))?;
        bytes.push(b'\n');
        std::fs::write(path, bytes)
            .map_err(|error| format!("could not write g8r optimization stats '{path}': {error}"))?;
    }

    if let Some(path) = matches.get_one::<String>("netlist_out") {
        let netlist = emit_netlist::emit_netlist(&design, false)
            .map_err(|error| format!("could not emit optimized gate-level netlist: {error}"))?;
        std::fs::write(path, netlist)
            .map_err(|error| format!("could not write optimized netlist '{path}': {error}"))?;
    }

    Ok(())
}
