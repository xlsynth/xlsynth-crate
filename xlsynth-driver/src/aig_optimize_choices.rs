// SPDX-License-Identifier: Apache-2.0

//! Thin CLI shim for ABC Boolean optimization and canonical choice export.

use anyhow::{Result, anyhow};
use clap::ArgMatches;
use std::path::{Path, PathBuf};
use xlsynth_g8r::aig::abc_choice::{
    AbcChoiceOptimizationFlow, AbcChoiceOptimizationOptions, optimize_aiger_with_abc_choices,
};

/// Optimizes one AIG through ABC while preserving its full scalar interface.
pub fn handle_aig_optimize_choices(matches: &ArgMatches) -> Result<()> {
    let input = matches
        .get_one::<String>("aig_input_file")
        .expect("aig_input_file is required");
    let output = matches
        .get_one::<String>("aiger_out")
        .expect("aiger_out is required");
    let abc = matches.get_one::<String>("abc").expect("abc is required");
    let liberty_files = matches
        .get_many::<String>("liberty")
        .into_iter()
        .flatten()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let mut options = AbcChoiceOptimizationOptions::new(abc, liberty_files);
    options.constraints_file = matches.get_one::<String>("constraints").map(PathBuf::from);
    options.flow = AbcChoiceOptimizationFlow {
        rounds: *matches
            .get_one::<usize>("rounds")
            .expect("rounds has a default"),
        initial_choice_command: matches
            .get_one::<String>("dch_command")
            .expect("dch_command has a default")
            .clone(),
        synthesis_command: matches
            .get_one::<String>("syn_command")
            .expect("syn_command has a default")
            .clone(),
        lut_mapping_command: matches
            .get_one::<String>("if_command")
            .expect("if_command has a default")
            .clone(),
        choice_command: matches
            .get_one::<String>("synch_command")
            .expect("synch_command has a default")
            .clone(),
        intermediate_mapping_command: matches
            .get_one::<String>("nf_command")
            .expect("nf_command has a default")
            .clone(),
        prefix_commands: matches
            .get_many::<String>("prefix_command")
            .into_iter()
            .flatten()
            .cloned()
            .collect(),
        suffix_commands: matches
            .get_many::<String>("suffix_command")
            .into_iter()
            .flatten()
            .cloned()
            .collect(),
    };

    let result = optimize_aiger_with_abc_choices(Path::new(input), Path::new(output), &options)
        .map_err(|error| anyhow!("ABC choice optimization failed: {error}"))?;
    eprintln!(
        "aig-optimize-choices: inputs={}, latches={}, outputs={}, ANDs={} -> {}, choices={}",
        result.input_interface.inputs,
        result.input_interface.latches,
        result.input_interface.outputs,
        result.input_interface.and_nodes,
        result.output_interface.and_nodes,
        result.choice_links
    );
    Ok(())
}
