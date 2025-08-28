// SPDX-License-Identifier: Apache-2.0

use crate::ir_equiv::{dispatch_ir_equiv, EquivInputs};
use crate::parallelism::ParallelismStrategy;
use crate::solver_choice::SolverChoice;
use crate::toolchain_config::ToolchainConfig;

use xlsynth_g8r::equiv::prove_equiv::AssertionSemantics;
use xlsynth_g8r::xls_ir::ir::{FileTable, Package, PackageMember};
use xlsynth_g8r::xls_ir::ir_parser;

use std::collections::HashMap;

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
        .unwrap_or(ParallelismStrategy::SingleThreaded);
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

    let lhs_block_text = std::fs::read_to_string(lhs_path).unwrap_or_else(|e| {
        eprintln!("Failed to read lhs IR file: {}", e);
        std::process::exit(1);
    });
    let rhs_block_text = std::fs::read_to_string(rhs_path).unwrap_or_else(|e| {
        eprintln!("Failed to read rhs IR file: {}", e);
        std::process::exit(1);
    });

    let mut lhs_parser = ir_parser::Parser::new(&lhs_block_text);
    let mut rhs_parser = ir_parser::Parser::new(&rhs_block_text);
    let mut lhs_fn = match lhs_parser.parse_block_to_fn() {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "[{}] Failed to parse LHS block ({}): {}",
                SUBCOMMAND, lhs_path, e
            );
            std::process::exit(1);
        }
    };
    let mut rhs_fn = match rhs_parser.parse_block_to_fn() {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "[{}] Failed to parse RHS block ({}): {}",
                SUBCOMMAND, rhs_path, e
            );
            std::process::exit(1);
        }
    };

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
        top_name: Some(lhs_fn.name.clone()),
    };
    let rhs_pkg = Package {
        name: "rhs_pkg".to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(rhs_fn.clone())],
        top_name: Some(rhs_fn.name.clone()),
    };
    let lhs_pkg_text = lhs_pkg.to_string();
    let rhs_pkg_text = rhs_pkg.to_string();

    // Need owned Strings for tops
    let lhs_top_owned = lhs_top.unwrap().to_string();
    let rhs_top_owned = rhs_top.unwrap().to_string();

    let inputs = EquivInputs {
        lhs_ir_text: &lhs_pkg_text,
        rhs_ir_text: &rhs_pkg_text,
        lhs_top: Some(&lhs_top_owned),
        rhs_top: Some(&rhs_top_owned),
        flatten_aggregates,
        drop_params: &drop_params,
        strategy,
        assertion_semantics: *assertion_semantics,
        lhs_fixed_implicit_activation,
        rhs_fixed_implicit_activation,
        subcommand: SUBCOMMAND,
        lhs_origin: lhs_path,
        rhs_origin: rhs_path,
        lhs_param_domains: None,
        rhs_param_domains: None,
        lhs_uf_map: HashMap::new(),
        rhs_uf_map: HashMap::new(),
    };

    let outcome = dispatch_ir_equiv(solver, tool_path, &inputs);
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
        if let Some(cex) = outcome.counterexample {
            eprintln!("[{}] failure: {}", SUBCOMMAND, cex);
        } else {
            eprintln!("[{}] failure", SUBCOMMAND);
        }
        std::process::exit(1);
    }
}
