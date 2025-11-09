// SPDX-License-Identifier: Apache-2.0

use crate::ir_equiv::{dispatch_ir_equiv, IrEquivRequest, IrModule};
use crate::toolchain_config::ToolchainConfig;
use xlsynth_prover::types::EquivParallelism;

use xlsynth_pir::ir::{
    self as ir_mod, BlockPortInfo, FileTable, MemberType, Package, PackageMember,
};
use xlsynth_pir::ir_parser;
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::types::AssertionSemantics;

use std::path::Path;

const SUBCOMMAND: &str = "ir-equiv-blocks";

pub fn handle_ir_equiv_blocks(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_equiv_blocks");
    let lhs_path = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs_path = matches.get_one::<String>("rhs_ir_file").unwrap();

    let mut lhs_top = matches.get_one::<String>("lhs_top").map(|s| s.as_str());
    let mut rhs_top = matches.get_one::<String>("rhs_top").map(|s| s.as_str());
    let top = matches.get_one::<String>("top").map(|s| s.as_str());

    if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        eprintln!("Error: --top and --lhs_top/--rhs_top cannot be used together");
        std::process::exit(1);
    }
    if lhs_top.is_some() ^ rhs_top.is_some() {
        eprintln!("Error: --lhs_top and --rhs_top must be used together");
        std::process::exit(1);
    }
    if let Some(t) = top {
        lhs_top = Some(t);
        rhs_top = lhs_top;
    }

    let assertion_semantics = matches
        .get_one::<AssertionSemantics>("assertion_semantics")
        .unwrap_or(&AssertionSemantics::Same);
    let solver: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());
    let flatten_aggregates = matches
        .get_one::<String>("flatten_aggregates")
        .map(|s| s == "true")
        .unwrap_or(false);
    let drop_params: Vec<String> = matches
        .get_one::<String>("drop_params")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(Vec::new);
    let strategy = matches
        .get_one::<String>("parallelism_strategy")
        .map(|s| s.parse().unwrap())
        .unwrap_or(EquivParallelism::SingleThreaded);
    let lhs_fixed_implicit_activation = matches
        .get_one::<String>("lhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);
    let rhs_fixed_implicit_activation = matches
        .get_one::<String>("rhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);
    let output_json = matches.get_one::<String>("output_json");

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    // Parse each input as a package and select a block member (prefer the top
    // block).
    let lhs_pkg_parsed = match ir_parser::parse_path_to_package(std::path::Path::new(lhs_path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "[{}] Failed to parse LHS IR package ({}): {}",
                SUBCOMMAND, lhs_path, e
            );
            std::process::exit(1);
        }
    };
    let rhs_pkg_parsed = match ir_parser::parse_path_to_package(std::path::Path::new(rhs_path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "[{}] Failed to parse RHS IR package ({}): {}",
                SUBCOMMAND, rhs_path, e
            );
            std::process::exit(1);
        }
    };

    fn select_block_from_package<'a>(
        pkg: &'a ir_mod::Package,
        name_opt: Option<&str>,
    ) -> Option<(&'a ir_mod::Fn, &'a BlockPortInfo)> {
        if let Some(name) = name_opt {
            for m in pkg.members.iter() {
                if let PackageMember::Block { func, port_info } = m {
                    if func.name == name {
                        return Some((func, port_info));
                    }
                }
            }
            return None;
        }
        if let Some((top_name, MemberType::Block)) = &pkg.top {
            for m in pkg.members.iter() {
                if let PackageMember::Block { func, port_info } = m {
                    if &func.name == top_name {
                        return Some((func, port_info));
                    }
                }
            }
        }
        for m in pkg.members.iter() {
            if let PackageMember::Block { func, port_info } = m {
                return Some((func, port_info));
            }
        }
        None
    }

    let (lhs_fn_ref, _lhs_ports) = match select_block_from_package(&lhs_pkg_parsed, lhs_top) {
        Some(pair) => pair,
        None => {
            eprintln!(
                "[{}] No block member found in LHS package (or name not found)",
                SUBCOMMAND
            );
            std::process::exit(1);
        }
    };
    let (rhs_fn_ref, _rhs_ports) = match select_block_from_package(&rhs_pkg_parsed, rhs_top) {
        Some(pair) => pair,
        None => {
            eprintln!(
                "[{}] No block member found in RHS package (or name not found)",
                SUBCOMMAND
            );
            std::process::exit(1);
        }
    };

    // Clone selected blocks as functions and build single-fn packages for
    // equivalence.
    let mut lhs_fn = lhs_fn_ref.clone();
    let mut rhs_fn = rhs_fn_ref.clone();

    if let Some(name) = lhs_top {
        lhs_fn.name = name.to_string();
    } else {
        lhs_top = Some(lhs_fn.name.as_str());
    }
    if let Some(name) = rhs_top {
        rhs_fn.name = name.to_string();
    } else {
        rhs_top = Some(rhs_fn.name.as_str());
    }

    let lhs_pkg = Package {
        name: "lhs_pkg".to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(lhs_fn.clone())],
        top: Some((lhs_fn.name.clone(), MemberType::Block)),
    };
    let rhs_pkg = Package {
        name: "rhs_pkg".to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(rhs_fn.clone())],
        top: Some((rhs_fn.name.clone(), MemberType::Block)),
    };
    let lhs_pkg_text = lhs_pkg.to_string();
    let rhs_pkg_text = rhs_pkg.to_string();

    // Need owned Strings for tops
    let lhs_top_owned = lhs_top.unwrap().to_string();
    let rhs_top_owned = rhs_top.unwrap().to_string();

    let tool_path_ref = tool_path.map(Path::new);

    let request = IrEquivRequest::new(
        IrModule::new(&lhs_pkg_text)
            .with_top(Some(lhs_top_owned.as_str()))
            .with_fixed_implicit_activation(lhs_fixed_implicit_activation),
        IrModule::new(&rhs_pkg_text)
            .with_top(Some(rhs_top_owned.as_str()))
            .with_fixed_implicit_activation(rhs_fixed_implicit_activation),
    )
    .with_drop_params(&drop_params)
    .with_flatten_aggregates(flatten_aggregates)
    .with_parallelism(strategy)
    .with_assertion_semantics(*assertion_semantics)
    .with_solver(solver)
    .with_tool_path(tool_path_ref);

    let outcome = dispatch_ir_equiv(&request, SUBCOMMAND);
    if let Some(path) = output_json {
        std::fs::write(path, serde_json::to_string(&outcome).unwrap()).unwrap();
    }
    let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
    if outcome.success {
        println!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
        println!("[{}] success: Solver proved equivalence", SUBCOMMAND);
        std::process::exit(0);
    } else {
        eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
        if let Some(err) = outcome.error_str.as_ref() {
            eprintln!("[{}] failure: {}", SUBCOMMAND, err);
        } else {
            eprintln!("[{}] failure", SUBCOMMAND);
        }
        std::process::exit(1);
    }
}
