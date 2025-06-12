// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use std::time::Instant;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::ir_equiv_boolector::{self, Ctx};
#[cfg(feature = "has-boolector")]
use xlsynth_g8r::xls_ir::ir_parser;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::ir_equiv_boolector::EquivResult;

/// Runs a Boolector equivalence check using the given context.
#[cfg(feature = "has-boolector")]
fn run_check_with_ctx(
    lhs_path: &std::path::Path,
    rhs_path: &std::path::Path,
    lhs_top: Option<&str>,
    rhs_top: Option<&str>,
    flatten_aggregates: bool,
    drop_params: &[String],
    ctx: &Ctx,
) -> bool {
    let start = Instant::now();
    let lhs_pkg = match ir_parser::parse_path_to_package(lhs_path) {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!("Failed to parse lhs IR file: {}", e);
            return false;
        }
    };
    let rhs_pkg = match ir_parser::parse_path_to_package(rhs_path) {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!("Failed to parse rhs IR file: {}", e);
            return false;
        }
    };
    let lhs_fn = if let Some(top_name) = lhs_top {
        lhs_pkg.get_fn(top_name).cloned().unwrap_or_else(|| {
            eprintln!("Top function '{}' not found in lhs IR file", top_name);
            std::process::exit(1);
        })
    } else {
        lhs_pkg.get_top().cloned().unwrap_or_else(|| {
            eprintln!("No top function found in lhs IR file");
            std::process::exit(1);
        })
    };
    let rhs_fn = if let Some(top_name) = rhs_top {
        rhs_pkg.get_fn(top_name).cloned().unwrap_or_else(|| {
            eprintln!("Top function '{}' not found in rhs IR file", top_name);
            std::process::exit(1);
        })
    } else {
        rhs_pkg.get_top().cloned().unwrap_or_else(|| {
            eprintln!("No top function found in rhs IR file");
            std::process::exit(1);
        })
    };
    let lhs_fn = lhs_fn
        .drop_params(drop_params)
        .expect("Dropped parameter is used in the function body!");
    let rhs_fn = rhs_fn
        .drop_params(drop_params)
        .expect("Dropped parameter is used in the function body!");
    let result = if flatten_aggregates {
        xlsynth_g8r::ir_equiv_boolector::prove_ir_equiv_flattened_with_ctx(&lhs_fn, &rhs_fn, ctx)
    } else {
        xlsynth_g8r::ir_equiv_boolector::prove_ir_fn_equiv_with_ctx(&lhs_fn, &rhs_fn, ctx)
    };
    let duration = start.elapsed();
    match result {
        EquivResult::Proved => {
            println!(
                "success: Boolector proved equivalence for {}:{} vs {}:{} (in {:?})",
                lhs_path.display(),
                lhs_fn.name,
                rhs_path.display(),
                rhs_fn.name,
                duration
            );
            true
        }
        EquivResult::Disproved(cex) => {
            println!(
                "failure: Boolector found counterexample for {}:{} vs {}:{}: {:?} (in {:?})",
                lhs_path.display(),
                lhs_fn.name,
                rhs_path.display(),
                rhs_fn.name,
                cex,
                duration
            );
            false
        }
    }
}

pub fn handle_ir_equiv_batch(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    #[cfg(not(feature = "has-boolector"))]
    {
        eprintln!("error: ir-equiv-batch requires --features=with-boolector-built or --features=with-boolector-system");
        std::process::exit(1);
    }
    #[cfg(feature = "has-boolector")]
    {
        let files: Vec<String> = matches
            .get_many::<String>("ir_files")
            .unwrap()
            .map(|s| s.to_string())
            .collect();
        if files.len() % 2 != 0 {
            eprintln!("Error: expected an even number of IR files");
            std::process::exit(1);
        }
        let mut lhs_top = matches.get_one::<String>("lhs_ir_top").map(|s| s.as_str());
        let mut rhs_top = matches.get_one::<String>("rhs_ir_top").map(|s| s.as_str());
        let top = matches.get_one::<String>("ir_top");
        if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
            eprintln!("Error: --ir_top and --lhs_ir_top/--rhs_ir_top cannot be used together");
            std::process::exit(1);
        }
        if lhs_top.is_some() ^ rhs_top.is_some() {
            eprintln!("Error: --lhs_ir_top and --rhs_ir_top must be used together");
            std::process::exit(1);
        }
        if let Some(t) = top {
            lhs_top = Some(t.as_str());
            rhs_top = Some(t.as_str());
        }
        let flatten_aggregates = matches
            .get_one::<String>("flatten_aggregates")
            .map(|s| s == "true")
            .unwrap_or(false);
        let drop_params: Vec<String> = matches
            .get_one::<String>("drop_params")
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_else(Vec::new);
        let stop_on_failure = matches
            .get_one::<String>("stop_on_failure")
            .map(|s| s == "true")
            .unwrap_or(false);
        let ctx = Ctx::new();
        let mut all_equiv = true;
        for pair in files.chunks(2) {
            let lhs_path = std::path::Path::new(&pair[0]);
            let rhs_path = std::path::Path::new(&pair[1]);
            let ok = run_check_with_ctx(
                lhs_path,
                rhs_path,
                lhs_top,
                rhs_top,
                flatten_aggregates,
                &drop_params,
                &ctx,
            );
            if !ok {
                all_equiv = false;
                if stop_on_failure {
                    std::process::exit(1);
                }
            }
        }
        if all_equiv {
            std::process::exit(0);
        } else {
            std::process::exit(1);
        }
    }
}
