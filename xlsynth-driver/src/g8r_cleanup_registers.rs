// SPDX-License-Identifier: Apache-2.0

//! Thin CLI for ordinary-AIG sequential register cleanup.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use clap::ArgMatches;
use xlsynth_g8r::aig::{
    GateBuilderOptions, RegisterInitializationPolicy, SequentialCleanupOptions,
    cleanup_sequential_gate_fn, cleanup_sequential_transition,
};
use xlsynth_g8r::aig_serdes::g8r::{
    emit_g8r, encode_g8r_binary, load_sequential_gate_fn_from_path,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;

use crate::common::{parse_bool_flag_or, write_stdout};
use crate::ir2gates::{encode_aiger_for_path, write_aiger_atomically};

/// Rebinds, cleans, and emits a consistently reduced sequential transition.
pub fn handle_g8r_cleanup_registers(matches: &ArgMatches) -> Result<()> {
    let design_path = Path::new(
        matches
            .get_one::<String>("g8r_input_file")
            .expect("clap requires a native sequential input"),
    );
    let design = load_sequential_gate_fn_from_path(design_path)
        .map_err(|error| anyhow!("failed to load '{}': {error}", design_path.display()))?;
    let initialization_policy = match matches
        .get_one::<String>("initialization_policy")
        .expect("initialization policy has a clap default")
        .as_str()
    {
        "preserve-cycle-zero" => RegisterInitializationPolicy::PreserveCycleZero,
        "uninitialized-dont-care" => RegisterInitializationPolicy::UninitializedDontCare,
        value => {
            return Err(anyhow!(
                "unsupported register initialization policy '{value}'"
            ));
        }
    };
    let options = SequentialCleanupOptions {
        initialization_policy,
    };

    let outcome = if let Some(path) = matches.get_one::<String>("optimized_transition") {
        let path = Path::new(path);
        let optimized = load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
            .map_err(|error| anyhow!("failed to load optimized '{}': {error}", path.display()))?;
        cleanup_sequential_transition(&design, &optimized.gate_fn, &options)
            .map_err(|error| anyhow!("failed to clean optimized sequential transition: {error}"))?
    } else {
        cleanup_sequential_gate_fn(&design, &options)
            .map_err(|error| anyhow!("failed to clean sequential transition: {error}"))?
    };

    let transition_artifact = matches
        .get_one::<String>("transition_aiger_out")
        .map(|path| {
            let path = Path::new(path);
            encode_aiger_for_path(&outcome.design.transition, path)
                .map(|bytes| (path, bytes))
                .map_err(|error| {
                    anyhow!(
                        "failed to encode cleaned transition '{}': {error}",
                        path.display()
                    )
                })
        })
        .transpose()?;
    let binary_artifact = matches
        .get_one::<String>("bin_out")
        .map(|path| {
            encode_g8r_binary(&outcome.design)
                .map(|bytes| (Path::new(path), bytes))
                .map_err(|error| anyhow!("failed to encode cleaned native design: {error}"))
        })
        .transpose()?;
    let stats_artifact = matches
        .get_one::<String>("stats_out")
        .map(|path| {
            let mut bytes = serde_json::to_vec_pretty(&outcome.stats)
                .context("failed to encode register-cleanup statistics")?;
            bytes.push(b'\n');
            Ok::<_, anyhow::Error>((Path::new(path), bytes))
        })
        .transpose()?;

    if let Some((path, bytes)) = &transition_artifact {
        write_aiger_atomically(path, bytes)
            .map_err(|error| anyhow!("failed to write cleaned transition: {error}"))?;
    }
    if let Some((path, bytes)) = &binary_artifact {
        std::fs::write(path, bytes)
            .with_context(|| format!("failed to write cleaned design '{}'", path.display()))?;
    }
    if let Some((path, bytes)) = &stats_artifact {
        std::fs::write(path, bytes)
            .with_context(|| format!("failed to write cleanup statistics '{}'", path.display()))?;
    }

    if !parse_bool_flag_or(matches, "quiet", false) {
        write_stdout(&emit_g8r(&outcome.design));
    }

    eprintln!(
        "g8r-cleanup-registers: register-bits={} -> {}, dead={}, constant={}, merged={}, and-nodes={} -> {}, rounds={}, initialization={}",
        outcome.stats.initial_register_bits,
        outcome.stats.final_register_bits,
        outcome.stats.dead_register_bits,
        outcome.stats.constant_register_bits,
        outcome.stats.merged_register_bits,
        outcome.stats.initial_and_nodes,
        outcome.stats.final_and_nodes,
        outcome.stats.iterations,
        matches
            .get_one::<String>("initialization_policy")
            .expect("initialization policy has a clap default")
    );
    Ok(())
}
