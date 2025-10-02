// SPDX-License-Identifier: Apache-2.0

//! Helper functions for running tools from the XLS toolchain directory that can
//! be provided to the driver.

use crate::common::{add_codegen_flags, CodegenFlags, PipelineSpec};
use std::process;
use std::process::Command;

/// Constructs the path to `binary_name` inside `tool_dir` and exits the process
/// with an error if the binary does not exist.
pub fn tool_path_or_exit(tool_dir: &str, binary_name: &str, pretty_name: &str) -> String {
    let path = format!("{}/{}", tool_dir, binary_name);
    if !std::path::Path::new(&path).exists() {
        eprintln!("{} tool not found at: {}", pretty_name, path);
        process::exit(1);
    }
    path
}

pub fn run_codegen_pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
    tool_path: &str,
) -> String {
    log::info!("run_codegen_pipeline");
    let codegen_main_path = tool_path_or_exit(tool_path, "codegen_main", "codegen_main");
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
    let opt_main_path = tool_path_or_exit(tool_path, "opt_main", "IR optimization");
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
    convert_tests: bool,
) -> String {
    log::info!(
        "run_ir_converter_main; enable_warnings: {:?}; disable_warnings: {:?}",
        enable_warnings,
        disable_warnings
    );
    let ir_convert_path = tool_path_or_exit(tool_path, "ir_converter_main", "IR conversion");
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

    if convert_tests {
        command.arg("--convert_tests=true");
    } else {
        command.arg("--convert_tests=false");
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
    let irequiv_path = tool_path_or_exit(tool_path, "check_ir_equivalence_main", "IR equivalence");
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
    let qc_path = tool_path_or_exit(tool_path, "prove_quickcheck_main", "prove_quickcheck_main");
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
    let codegen_main_path = tool_path_or_exit(tool_path, "codegen_main", "codegen_main");
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

/// Runs codegen_main in combinational mode to emit block IR for a specific
/// function top, capturing the block IR text from a requested output path.
pub fn run_codegen_block_ir_to_string(
    input_file: &std::path::Path,
    top: &str,
    tool_path: &str,
    output_block_ir_path: &std::path::Path,
) -> String {
    log::info!("run_codegen_block_ir_to_string");
    let codegen_main_path = tool_path_or_exit(tool_path, "codegen_main", "codegen_main");
    let mut command = Command::new(codegen_main_path);
    command
        .arg(input_file)
        .arg("--delay_model")
        .arg("unit")
        .arg("--generator=combinational")
        .arg("--top")
        .arg(top)
        .arg("--output_block_ir_path")
        .arg(output_block_ir_path);

    log::info!("Running command: {:?}", command);
    let output = command
        .output()
        .expect("codegen_main (block IR) should succeed");

    if !output.status.success() {
        eprintln!("Block IR generation failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    std::fs::read_to_string(output_block_ir_path).expect("reading output block IR should succeed")
}

/// Runs block_to_verilog_main to emit Verilog from a block IR file, honoring
/// the same codegen flags used by codegen_main.
pub fn run_block_to_verilog(
    block_ir_file: &std::path::Path,
    codegen_flags: &CodegenFlags,
    tool_path: &str,
) -> String {
    log::info!("run_block_to_verilog");
    let b2v_path = tool_path_or_exit(tool_path, "block_to_verilog_main", "block_to_verilog_main");
    let mut command = Command::new(b2v_path);
    command.arg(block_ir_file);

    // Reuse the same flag population as codegen_main.
    add_codegen_flags(&mut command, codegen_flags);
    log::info!("Running command: {:?}", command);
    let output = command
        .output()
        .expect("block_to_verilog_main should succeed");

    if !output.status.success() {
        eprintln!(
            "block_to_verilog_main failed with status: {}",
            output.status
        );
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}
