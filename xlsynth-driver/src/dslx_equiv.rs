// SPDX-License-Identifier: Apache-2.0

use crate::ir_equiv::{dispatch_ir_equiv, EquivInputs};
use crate::parallelism::ParallelismStrategy;
use crate::solver_choice::SolverChoice;
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};
use crate::tools::{run_ir_converter_main, run_opt_main};
use xlsynth::{mangle_dslx_name, DslxConvertOptions, IrPackage};
use xlsynth_g8r::equiv::prove_equiv::AssertionSemantics;

const SUBCOMMAND: &str = "dslx-equiv";

pub struct OptimizedIrText {
    pub ir_text: String,
    pub mangled_top: String,
}

/// Builds (always optimized) IR text plus mangled top name for a DSLX module.
fn build_ir_for_dslx(
    input_path: &std::path::Path,
    dslx_top: &str,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    enable_warnings: Option<&[String]>,
    disable_warnings: Option<&[String]>,
    tool_path: Option<&str>,
    type_inference_v2: Option<bool>,
) -> OptimizedIrText {
    let module_name = input_path.file_stem().unwrap().to_str().unwrap();
    let mangled_top = mangle_dslx_name(module_name, dslx_top).unwrap();

    if let Some(tool_path) = tool_path {
        // Convert via external tool then always optimize.
        let mut ir_text = run_ir_converter_main(
            input_path,
            Some(dslx_top),
            dslx_stdlib_path,
            dslx_path,
            tool_path,
            enable_warnings,
            disable_warnings,
            type_inference_v2,
        );
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &ir_text).unwrap();
        ir_text = run_opt_main(tmp.path(), Some(&mangled_top), tool_path);
        OptimizedIrText {
            ir_text,
            mangled_top,
        }
    } else {
        if type_inference_v2 == Some(true) {
            eprintln!("error: --type_inference_v2 is only supported with external toolchain");
            std::process::exit(1);
        }
        let dslx_contents = std::fs::read_to_string(input_path).unwrap_or_else(|e| {
            eprintln!("Failed to read DSLX file {}: {}", input_path.display(), e);
            std::process::exit(1);
        });
        let dslx_stdlib_path: Option<&std::path::Path> =
            dslx_stdlib_path.map(|s| std::path::Path::new(s));
        let additional_search_paths: Vec<&std::path::Path> = dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let result = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            input_path,
            &DslxConvertOptions {
                dslx_stdlib_path,
                additional_search_paths,
                enable_warnings,
                disable_warnings,
            },
        )
        .expect("successful DSLX->IR conversion");
        let pkg = IrPackage::parse_ir(&result.ir, Some(&mangled_top)).unwrap();
        let optimized = xlsynth::optimize_ir(&pkg, &mangled_top).unwrap();
        OptimizedIrText {
            ir_text: optimized.to_string(),
            mangled_top,
        }
    }
}

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

    // Retrieve but ignore deprecated --opt flag if provided.
    if let Some(val) = matches.get_one::<String>("opt") {
        if val == "false" {
            eprintln!(
                "[warning] --opt=false ignored: dslx-equiv always optimizes IR before proving"
            );
        }
    }

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

    let OptimizedIrText {
        ir_text: lhs_ir_text,
        mangled_top: lhs_mangled_top,
    } = build_ir_for_dslx(
        lhs_path,
        lhs_top.unwrap(),
        dslx_stdlib_path,
        dslx_path,
        enable_warnings,
        disable_warnings,
        tool_path,
        type_inference_v2,
    );
    let OptimizedIrText {
        ir_text: rhs_ir_text,
        mangled_top: rhs_mangled_top,
    } = build_ir_for_dslx(
        rhs_path,
        rhs_top.unwrap(),
        dslx_stdlib_path,
        dslx_path,
        enable_warnings,
        disable_warnings,
        tool_path,
        type_inference_v2,
    );

    let inputs = EquivInputs {
        lhs_ir_text: &lhs_ir_text,
        rhs_ir_text: &rhs_ir_text,
        lhs_top: Some(&lhs_mangled_top),
        rhs_top: Some(&rhs_mangled_top),
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
