// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::ArgMatches;
use xlsynth::{mangle_dslx_name, DslxConvertOptions};
use xlsynth_prover::ir_equiv::{run_ir_equiv, IrEquivRequest, IrModule};

use crate::common::parse_bool_flag_or;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

const SUBCOMMAND: &str = "ir-fn-to-dslx";

pub fn handle_ir_fn_to_dslx(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let ir_top = matches
        .get_one::<String>("ir_top")
        .expect("--top must be specified");
    let verify_roundtrip = parse_bool_flag_or(matches, "verify_roundtrip", false);

    let ir_text = std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to read IR input file: {}", e),
            Some(SUBCOMMAND),
            vec![("ir_input_file", input_file)],
        )
    });

    let translated =
        xlsynth_pir::ir_fn_to_dslx::convert_ir_package_fn_to_dslx(&ir_text, Some(ir_top))
            .unwrap_or_else(|e| {
                report_cli_error_and_exit(
                    &format!("Failed to translate IR function to DSLX: {}", e),
                    Some(SUBCOMMAND),
                    vec![("ir_top", ir_top)],
                )
            });

    if verify_roundtrip {
        verify_roundtrip_equivalence(
            &ir_text,
            input_file,
            ir_top,
            &translated.function_name,
            &translated.dslx_text,
            config,
        )
        .unwrap_or_else(|e| {
            report_cli_error_and_exit(
                &format!("Roundtrip equivalence check failed: {}", e),
                Some(SUBCOMMAND),
                vec![("ir_top", ir_top)],
            )
        });
    }

    print!("{}", translated.dslx_text);
}

fn verify_roundtrip_equivalence(
    lhs_ir_text: &str,
    lhs_ir_path: &str,
    lhs_ir_top: &str,
    dslx_top: &str,
    dslx_text: &str,
    config: &Option<ToolchainConfig>,
) -> Result<(), String> {
    let fake_dslx_path = Path::new("ir_fn_to_dslx_roundtrip.x");
    let dslx_convert =
        xlsynth::convert_dslx_to_ir_text(dslx_text, fake_dslx_path, &DslxConvertOptions::default())
            .map_err(|e| format!("DSLX to IR conversion failed: {}", e))?;

    let module_name = fake_dslx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Unable to derive module name from fake DSLX path".to_string())?;
    let rhs_ir_top = mangle_dslx_name(module_name, dslx_top)
        .map_err(|e| format!("Failed to mangle DSLX top name: {}", e))?;

    let tool_path = config
        .as_ref()
        .and_then(|c| c.tool_path.as_deref())
        .map(Path::new);

    let request = IrEquivRequest::new(
        IrModule::new(lhs_ir_text)
            .with_path(Some(Path::new(lhs_ir_path)))
            .with_top(Some(lhs_ir_top)),
        IrModule::new(&dslx_convert.ir)
            .with_path(Some(fake_dslx_path))
            .with_top(Some(&rhs_ir_top)),
    )
    .with_tool_path(tool_path);

    let report = run_ir_equiv(&request).map_err(|e| e.to_string())?;
    if report.is_success() {
        Ok(())
    } else {
        Err(report
            .error_str()
            .unwrap_or_else(|| "solver reported non-equivalence".to_string()))
    }
}
