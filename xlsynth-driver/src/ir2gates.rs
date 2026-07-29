// SPDX-License-Identifier: Apache-2.0

//! Accepts input IR and then performs the g8r IR-to-gates mapping on it.

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use xlsynth_g8r::aig::{GateFn, SequentialGateFn};
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::emit_netlist;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::process_ir_path;
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, Ir2GatesSummaryStats, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::ir::{MemberType, Package, PackageMember};
use xlsynth_pir::ir_parser::Parser;

enum Ir2G8rLowering {
    Function {
        design: SequentialGateFn,
        stats: Ir2GatesSummaryStats,
    },
    Block {
        design: SequentialGateFn,
    },
}

impl Ir2G8rLowering {
    fn design(&self) -> &SequentialGateFn {
        match self {
            Self::Function { design, .. } | Self::Block { design } => design,
        }
    }

    fn stats(&self) -> Option<&Ir2GatesSummaryStats> {
        match self {
            Self::Function { stats, .. } => Some(stats),
            Self::Block { .. } => None,
        }
    }
}

fn select_ir2g8r_member(
    ir_text: &str,
    requested_top: Option<&str>,
) -> Result<(Package, String, MemberType), String> {
    let mut parser = Parser::new(ir_text);
    let mut package = parser
        .parse_and_validate_package()
        .map_err(|e| format!("PIR parse/validate failed: {e}"))?;
    let (name, member_type) = match requested_top {
        Some(name) => match (
            package.get_fn(name).is_some(),
            package.get_block(name).is_some(),
        ) {
            (true, false) => (name.to_string(), MemberType::Function),
            (false, true) => (name.to_string(), MemberType::Block),
            (true, true) => {
                return Err(format!(
                    "PIR package has both a function and block named '{name}'; --top is ambiguous"
                ));
            }
            (false, false) => {
                return Err(format!(
                    "PIR package has no function or block named '{name}'"
                ));
            }
        },
        None => match package.top.clone() {
            Some(top) => top,
            None => {
                let candidates = package
                    .members
                    .iter()
                    .map(|member| match member {
                        PackageMember::Function(function) => {
                            (function.name.clone(), MemberType::Function)
                        }
                        PackageMember::Block { func, .. } => (func.name.clone(), MemberType::Block),
                    })
                    .collect::<Vec<(String, MemberType)>>();
                match candidates.as_slice() {
                    [] => {
                        return Err("PIR package has no function or block to lower".to_string());
                    }
                    [candidate] => candidate.clone(),
                    _ => {
                        let member_names = candidates
                            .iter()
                            .map(|(name, member_type)| match member_type {
                                MemberType::Function => format!("function '{name}'"),
                                MemberType::Block => format!("block '{name}'"),
                            })
                            .collect::<Vec<String>>()
                            .join(", ");
                        return Err(format!(
                            "PIR package has no declared top and multiple lowerable members ({member_names}); provide --top to select one"
                        ));
                    }
                }
            }
        },
    };
    package.top = Some((name.clone(), member_type.clone()));
    Ok((package, name, member_type))
}

fn block_gatify_options(options: &CanonicalG8rOptions) -> GatifyOptions {
    GatifyOptions {
        fold: options.fold,
        hash: options.hash,
        track_pir_node_ids: options.track_pir_node_ids,
        adder_mapping: options.adder_mapping,
        mul_adder_mapping: options.mul_adder_mapping,
        enable_rewrite_carry_out: options.enable_rewrite_carry_out,
        enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
        enable_rewrite_nary_add: options.enable_rewrite_nary_add,
        enable_rewrite_mask_low: options.enable_rewrite_mask_low,
        enable_rewrite_normalize_left: options.enable_rewrite_normalize_left,
        unsafe_gatify_gate_operation: options.unsafe_gatify_gate_operation,
        ..GatifyOptions::all_opts_disabled()
    }
}

fn lower_ir2g8r_design(
    ir_text: &str,
    requested_top: Option<&str>,
    lowering_options: &CanonicalG8rOptions,
) -> Result<Ir2G8rLowering, String> {
    let (package, top_name, member_type) = select_ir2g8r_member(ir_text, requested_top)?;
    match member_type {
        MemberType::Function => {
            let artifacts = canonical_ir_text_to_g8r_lowering_artifacts(
                ir_text,
                Some(&top_name),
                lowering_options,
            )?;
            Ok(Ir2G8rLowering::Function {
                design: SequentialGateFn::from_gate_fn(artifacts.gate_fn),
                stats: artifacts.stats,
            })
        }
        MemberType::Block => {
            let design = block_package_to_sequential_gate_fn(
                &package,
                block_gatify_options(lowering_options),
            )
            .map_err(|e| format!("failed to lower selected IR block '{top_name}': {e}"))?;
            Ok(Ir2G8rLowering::Block { design })
        }
    }
}

/// Encodes a combinational graph using the AIGER format selected by its path.
fn encode_aiger_for_path(gate_fn: &GateFn, output_path: &Path) -> Result<Vec<u8>, String> {
    let is_binary_aig = output_path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("aig"))
        .unwrap_or(false);

    if is_binary_aig {
        emit_aiger_binary(gate_fn, true)
            .map_err(|error| format!("Failed to emit binary AIGER: {error}"))
    } else {
        emit_aiger(gate_fn, true)
            .map(String::into_bytes)
            .map_err(|error| format!("Failed to emit ASCII AIGER: {error}"))
    }
}

/// Atomically replaces an AIGER output only after its complete bytes are ready.
fn write_aiger_atomically(output_path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary_file = tempfile::Builder::new()
        .prefix(".xlsynth-aiger-")
        .tempfile_in(parent)
        .map_err(|error| {
            format!(
                "failed to create a temporary AIGER file in '{}': {error}",
                parent.display()
            )
        })?;
    temporary_file
        .write_all(bytes)
        .map_err(|error| format!("failed to write temporary AIGER output: {error}"))?;
    temporary_file
        .flush()
        .map_err(|error| format!("failed to flush temporary AIGER output: {error}"))?;
    temporary_file.persist(output_path).map_err(|error| {
        format!(
            "failed to persist AIGER output '{}': {}",
            output_path.display(),
            error.error
        )
    })?;
    Ok(())
}

pub fn handle_ir2gates(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let quiet = match matches.get_one::<String>("quiet").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    let emit_independent_op_stats = match matches
        .get_one::<String>("emit-independent-op-stats")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    let output_json = matches.get_one::<String>("output_json");
    let prepared_ir_out = matches.get_one::<String>("prepared_ir_out");

    let input_path = std::path::Path::new(input_file);
    let options = crate::g8r_cli::build_process_ir_path_options_for_cli(
        matches,
        quiet,
        /* emit_netlist= */ false,
        emit_independent_op_stats,
        ir_top,
        prepared_ir_out.map(|s| std::path::Path::new(s)),
    );
    let stats = process_ir_path::process_ir_path_for_cli(input_path, &options);
    if quiet {
        serde_json::to_writer(std::io::stdout(), &stats).unwrap();
        println!();
    }
    if let Some(path) = output_json.map(|s| std::path::Path::new(s)) {
        let file = File::create(path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {}", path.display(), e));
        serde_json::to_writer_pretty(file, &stats)
            .unwrap_or_else(|e| panic!("Failed to write JSON: {}", e));
    }
}

pub fn handle_ir_prep_for_gates(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let input_path = std::path::Path::new(input_file);
    let ir_text = std::fs::read_to_string(input_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_path.display(), e));
    let lowering_options = crate::g8r_cli::parse_g8r_cli_options(matches);
    let prepared_ir = process_ir_path::canonical_ir_text_to_prepared_gatify_ir(
        &ir_text,
        ir_top,
        &lowering_options,
    )
    .unwrap_or_else(|err| {
        eprintln!("Error encountered preparing IR for gates: {}", err);
        std::process::exit(1);
    });
    print!("{}", prepared_ir);
}

pub fn handle_ir2g8r(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let bin_out = matches.get_one::<String>("bin_out");
    let aiger_out = matches.get_one::<String>("aiger_out");
    let transition_aiger_out = matches.get_one::<String>("transition_aiger_out");
    let stats_out = matches.get_one::<String>("stats_out");
    let netlist_out = matches.get_one::<String>("netlist_out");
    let input_path = std::path::Path::new(input_file);
    let ir_text = std::fs::read_to_string(input_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_path.display(), e));
    let lowering_options = crate::g8r_cli::parse_g8r_cli_options(matches);
    let lowering = lower_ir2g8r_design(&ir_text, ir_top, &lowering_options).unwrap_or_else(|err| {
        eprintln!("Error encountered lowering IR to gates: {}", err);
        std::process::exit(1);
    });
    let design = lowering.design();
    // Encode every requested AIGER artifact before exposing stdout or creating
    // another output. In particular, --aiger-out must continue to reject a
    // registered design rather than silently exporting only its transition.
    let aiger_bytes = aiger_out.map(|path| {
        let gate_fn = design.clone().try_into_gate_fn().unwrap_or_else(|error| {
            eprintln!(
                "Failed to emit AIGER: --aiger-out requires a clockless, register-free design: {}",
                error
            );
            std::process::exit(1);
        });
        encode_aiger_for_path(&gate_fn, Path::new(path)).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        })
    });
    let transition_aiger_bytes = transition_aiger_out.map(|path| {
        encode_aiger_for_path(&design.transition, Path::new(path)).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        })
    });
    let stats = stats_out.map(|_| {
        lowering.stats().unwrap_or_else(|| {
            eprintln!(
                "Failed to emit stats: --stats-out is currently supported only when the selected IR member is a function"
            );
            std::process::exit(1);
        })
    });
    // Always print the native sequential design representation to stdout.
    println!("{}", emit_g8r(design));
    // If --bin-out is given, write the native binary representation.
    if let Some(bin_path) = bin_out {
        let bin = encode_g8r_binary(design).expect("Failed to serialize g8r design");
        let mut f = File::create(bin_path).expect("Failed to create bin_out file");
        f.write_all(&bin).expect("Failed to write bin_out file");
    }
    // --aiger-out represents a complete, strictly combinational design.
    if let (Some(aiger_path), Some(bytes)) = (aiger_out, aiger_bytes.as_deref()) {
        write_aiger_atomically(Path::new(aiger_path), bytes).unwrap_or_else(|error| {
            eprintln!("Failed to write --aiger-out: {error}");
            std::process::exit(1);
        });
    }
    // --transition-aiger-out represents just the combinational transition,
    // whose register boundary metadata is retained by --bin-out or stdout.
    if let (Some(aiger_path), Some(bytes)) =
        (transition_aiger_out, transition_aiger_bytes.as_deref())
    {
        write_aiger_atomically(Path::new(aiger_path), bytes).unwrap_or_else(|error| {
            eprintln!("Failed to write --transition-aiger-out: {error}");
            std::process::exit(1);
        });
    }
    // If --stats-out is given, write the stats as JSON
    if let (Some(stats_path), Some(stats)) = (stats_out, stats) {
        let json = serde_json::to_string_pretty(&stats).expect("Failed to serialize stats to JSON");
        let mut f = File::create(stats_path).expect("Failed to create stats_out file");
        f.write_all(json.as_bytes())
            .expect("Failed to write stats_out file");
    }
    // If --netlist-out is given, write the gate-level netlist (human-readable)
    if let Some(netlist_path) = netlist_out {
        let netlist = match emit_netlist::emit_netlist(design, false) {
            Ok(netlist) => netlist,
            Err(e) => {
                eprintln!("Failed to emit netlist: {}", e);
                std::process::exit(1);
            }
        };
        let mut f = File::create(netlist_path).expect("Failed to create netlist_out file");
        f.write_all(netlist.as_bytes())
            .expect("Failed to write netlist_out file");
    }
}
