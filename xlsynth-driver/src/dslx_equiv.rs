// SPDX-License-Identifier: Apache-2.0

use crate::common::{dslx_to_ir, DslxIrBuildOptions};
use crate::ir_equiv::{dispatch_ir_equiv, EquivInputs};
use crate::parallelism::ParallelismStrategy;
use crate::solver_choice::SolverChoice;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};
use xlsynth_g8r::equiv::prove_equiv::AssertionSemantics;

const SUBCOMMAND: &str = "dslx-equiv";

pub fn handle_dslx_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx_equiv");
    let lhs_file = matches.get_one::<String>("lhs_dslx_file").unwrap();
    let rhs_file = matches.get_one::<String>("rhs_dslx_file").unwrap();

    let mut lhs_top = matches
        .get_one::<String>("lhs_dslx_top")
        .map(|s| s.as_str());
    let mut rhs_top = matches
        .get_one::<String>("rhs_dslx_top")
        .map(|s| s.as_str());

    let top = matches.get_one::<String>("dslx_top").map(|s| s.as_str());

    if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        eprintln!("Error: --dslx_top and --lhs_dslx_top/--rhs_dslx_top cannot be used together");
        std::process::exit(1);
    }
    if lhs_top.is_some() ^ rhs_top.is_some() {
        eprintln!("Error: --lhs_dslx_top and --rhs_dslx_top must be used together");
        std::process::exit(1);
    }
    if top.is_some() {
        lhs_top = top;
        rhs_top = top;
    }

    if lhs_top.is_none() || rhs_top.is_none() {
        eprintln!("Error: a top function must be specified via --dslx_top or both --lhs_dslx_top/--rhs_dslx_top");
        std::process::exit(1);
    }

    let lhs_path = std::path::Path::new(lhs_file);
    let rhs_path = std::path::Path::new(rhs_file);

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path = dslx_stdlib_path.as_deref();
    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    let enable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.enable_warnings.as_deref());
    let disable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.disable_warnings.as_deref());

    let type_inference_v2 = matches
        .get_one::<String>("type_inference_v2")
        .map(|s| s == "true")
        .or_else(|| {
            config
                .as_ref()
                .and_then(|c| c.dslx.as_ref()?.type_inference_v2)
        });

    let assertion_semantics = matches
        .get_one::<String>("assertion_semantics")
        .map(|s| s.parse().unwrap())
        .unwrap_or(AssertionSemantics::Same);

    let solver_choice: Option<SolverChoice> = matches
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

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    // Build artifacts (always optimized) for both sides.
    let lhs_artifacts = dslx_to_ir(&DslxIrBuildOptions {
        input_path: lhs_path,
        dslx_top: lhs_top.unwrap(),
        dslx_stdlib_path,
        dslx_path,
        enable_warnings,
        disable_warnings,
        tool_path,
        type_inference_v2,
        optimize: true,
    });
    let rhs_artifacts = dslx_to_ir(&DslxIrBuildOptions {
        input_path: rhs_path,
        dslx_top: rhs_top.unwrap(),
        dslx_stdlib_path,
        dslx_path,
        enable_warnings,
        disable_warnings,
        tool_path,
        type_inference_v2,
        optimize: true,
    });

    let inputs = EquivInputs {
        lhs_ir_text: lhs_artifacts
            .optimized_ir
            .as_deref()
            .unwrap_or(&lhs_artifacts.raw_ir),
        rhs_ir_text: rhs_artifacts
            .optimized_ir
            .as_deref()
            .unwrap_or(&rhs_artifacts.raw_ir),
        lhs_top: Some(&lhs_artifacts.mangled_top),
        rhs_top: Some(&rhs_artifacts.mangled_top),
        flatten_aggregates,
        drop_params: &drop_params,
        strategy,
        assertion_semantics,
        lhs_fixed_implicit_activation,
        rhs_fixed_implicit_activation,
        subcommand: SUBCOMMAND,
        lhs_origin: lhs_file,
        rhs_origin: rhs_file,
    };

    dispatch_ir_equiv(solver_choice, tool_path, &inputs);
}
