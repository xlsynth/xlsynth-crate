// SPDX-License-Identifier: Apache-2.0

//! Helper functions for running tools from the XLS toolchain directory that can
//! be provided to the driver.

use crate::common::{add_codegen_flags, CodegenFlags, PipelineSpec};
use std::process;
use std::process::Command;

pub fn run_codegen_pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
    tool_path: &str,
) -> String {
    log::info!("run_codegen_pipeline");
    // Give an error if the codegen_main tool is not found.
    let codegen_main_path = format!("{}/codegen_main", tool_path);
    if !std::path::Path::new(&codegen_main_path).exists() {
        eprintln!("codegen_main tool not found at: {}", codegen_main_path);
        process::exit(1);
    }

    let mut command = Command::new(codegen_main_path);
    command
        .arg(input_file)
        .arg("--delay_model")
        .arg(delay_model);

    add_codegen_flags(&mut command, codegen_flags);

    let command = match pipeline_spec {
        PipelineSpec::Stages(stages) => command.arg("--pipeline_stages").arg(stages.to_string()),
        PipelineSpec::ClockPeriodPs(clock_period_ps) => command
            .arg("--clock_period_ps")
            .arg(clock_period_ps.to_string()),
    };

    log::info!("Running command: {:?}", command);

    // We run the codegen_main tool on the given input file.
    let output = command.output().expect("codegen_main should succeed");

    if !output.status.success() {
        eprintln!("Pipeline generation failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Runs the IR optimization command line tool and returns the output.
pub fn run_opt_main(input_file: &std::path::Path, ir_top: Option<&str>, tool_path: &str) -> String {
    log::info!("run_opt_main; ir_top: {:?}", ir_top);
    let opt_main_path = format!("{}/opt_main", tool_path);
    if !std::path::Path::new(&opt_main_path).exists() {
        eprintln!("IR optimization tool not found at: {}", opt_main_path);
        process::exit(1);
    }

    let mut command = Command::new(opt_main_path);
    command.arg(input_file);
    if ir_top.is_some() {
        command.arg("--top").arg(ir_top.unwrap());
    }

    let output = command.output().expect("opt_main should succeed");

    if !output.status.success() {
        eprintln!("IR optimization failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Runs the IR converter command line tool and returns the output.
pub fn run_ir_converter_main(
    input_file: &std::path::Path,
    dslx_top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: &str,
    enable_warnings: Option<&[String]>,
    disable_warnings: Option<&[String]>,
    type_inference_v2: Option<bool>,
) -> String {
    log::info!(
        "run_ir_converter_main; enable_warnings: {:?}; disable_warnings: {:?}",
        enable_warnings,
        disable_warnings
    );
    let ir_convert_path = format!("{}/ir_converter_main", tool_path);
    if !std::path::Path::new(&ir_convert_path).exists() {
        eprintln!("IR conversion tool not found at: {}", ir_convert_path);
        process::exit(1);
    }

    let mut command = Command::new(ir_convert_path);
    command.arg(input_file);

    if let Some(dslx_top) = dslx_top {
        command.arg("--top").arg(dslx_top);
    }

    if let Some(dslx_stdlib_path) = dslx_stdlib_path {
        command.arg("--dslx_stdlib_path").arg(dslx_stdlib_path);
    }

    if let Some(dslx_path) = dslx_path {
        command.arg("--dslx_path").arg(dslx_path);
    }

    if let Some(enable_warnings) = enable_warnings {
        command
            .arg("--enable_warnings")
            .arg(enable_warnings.join(","));
    }

    if let Some(disable_warnings) = disable_warnings {
        command
            .arg("--disable_warnings")
            .arg(disable_warnings.join(","));
    }

    // Pass through the experimental type inference flag if requested.
    if let Some(true) = type_inference_v2 {
        command.arg("--type_inference_v2");
    }

    log::info!("command: {:?}", command);
    let output = command.output().expect("ir_converter_main should succeed");

    if !output.status.success() {
        eprintln!("IR conversion failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    let ir_text = String::from_utf8_lossy(&output.stdout).to_string();
    log::info!("ir_text: {}", ir_text);
    ir_text
}

pub fn run_check_ir_equivalence_main(
    lhs: &std::path::Path,
    rhs: &std::path::Path,
    top: Option<&str>,
    tool_path: &str,
) -> Result<String, std::process::Output> {
    log::info!("run_check_ir_equivalence_main");
    let irequiv_path = format!("{}/check_ir_equivalence_main", tool_path);
    if !std::path::Path::new(&irequiv_path).exists() {
        eprintln!("IR equivalence tool not found at: {}", irequiv_path);
        process::exit(1);
    }

    let mut command = Command::new(irequiv_path);
    command.arg(lhs);
    command.arg(rhs);
    if let Some(top) = top {
        command.arg("--top").arg(top);
    }

    log::info!("command: {:?}", command);
    let output = command
        .output()
        .expect("check_ir_equivalence_main should succeed");

    if !output.status.success() {
        log::info!("IR equivalence check failed with status: {}", output.status);
        log::info!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        log::info!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        return Err(output);
    }

    log::info!(
        "IR equivalence check succeeded so ignoring stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Runs the prove_quickcheck_main tool shipped in the external toolchain.
///
/// `entry_file` – DSLX file path; `quickcheck_name` – the QC function to prove.
/// Returns stdout on success, or the full `Output` on failure (caller decides).
pub fn run_prove_quickcheck_main(
    entry_file: &std::path::Path,
    test_filter: Option<&str>,
    tool_path: &str,
) -> Result<String, std::process::Output> {
    log::info!(
        "run_prove_quickcheck_main entry_file={:?} test_filter={:?} tool_path={} ",
        entry_file,
        test_filter,
        tool_path
    );

    let qc_path = format!("{}/prove_quickcheck_main", tool_path);
    if !std::path::Path::new(&qc_path).exists() {
        eprintln!("prove_quickcheck_main tool not found at: {}", qc_path);
        std::process::exit(1);
    }

    let mut command = Command::new(qc_path);
    if let Some(test_filter) = test_filter {
        command.arg("--test_filter").arg(test_filter);
    }
    command.arg(entry_file);

    log::info!("command: {:?}", command);

    let output = command.output().expect("prove_quickcheck_main should run");

    if !output.status.success() {
        log::info!(
            "prove_quickcheck_main failed – status: {} stdout: {} stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(output);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn run_codegen_combinational(
    input_file: &std::path::Path,
    delay_model: &str,
    codegen_flags: &CodegenFlags,
    tool_path: &str,
) -> String {
    log::info!("run_codegen_combinational");
    let codegen_main_path = format!("{}/codegen_main", tool_path);
    if !std::path::Path::new(&codegen_main_path).exists() {
        eprintln!("codegen_main tool not found at: {}", codegen_main_path);
        process::exit(1);
    }

    let mut command = Command::new(codegen_main_path);
    command
        .arg(input_file)
        .arg("--delay_model")
        .arg(delay_model)
        .arg("--generator=combinational");

    add_codegen_flags(&mut command, codegen_flags);

    log::info!("Running command: {:?}", command);
    let output = command
        .output()
        .expect("codegen_main (combinational) should succeed");

    if !output.status.success() {
        eprintln!(
            "Combinational codegen failed with status: {}",
            output.status
        );
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}
