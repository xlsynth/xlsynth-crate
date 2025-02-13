// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::process;
use std::process::Command;

// By default in the driver we treat warnings as errors.
pub const DEFAULT_WARNINGS_AS_ERRORS: bool = true;

// Specification for a pipeline generation can be either stages-based or
// clock-period-based.
pub enum PipelineSpec {
    Stages(u64),
    ClockPeriodPs(u64),
}

pub fn extract_pipeline_spec(matches: &ArgMatches) -> PipelineSpec {
    if let Some(pipeline_stages) = matches.get_one::<String>("pipeline_stages") {
        PipelineSpec::Stages(pipeline_stages.parse().unwrap())
    } else if let Some(clock_period_ps) = matches.get_one::<String>("clock_period_ps") {
        PipelineSpec::ClockPeriodPs(clock_period_ps.parse().unwrap())
    } else {
        eprintln!("Must provide either --pipeline_stages or --clock_period_ps");
        process::exit(1)
    }
}

/// Extracts flags that we pass to the "codegen" step of the process (i.e.
/// generating lowered Verilog).
pub fn extract_codegen_flags(matches: &ArgMatches) -> CodegenFlags {
    let result = CodegenFlags {
        input_valid_signal: matches
            .get_one::<String>("input_valid_signal")
            .map(|s| s.to_string()),
        output_valid_signal: matches
            .get_one::<String>("output_valid_signal")
            .map(|s| s.to_string()),
        use_system_verilog: matches
            .get_one::<String>("use_system_verilog")
            .map(|s| s == "true"),
        flop_inputs: matches
            .get_one::<String>("flop_inputs")
            .map(|s| s == "true"),
        flop_outputs: matches
            .get_one::<String>("flop_outputs")
            .map(|s| s == "true"),
        add_idle_output: matches
            .get_one::<String>("add_idle_output")
            .map(|s| s == "true"),
        module_name: matches
            .get_one::<String>("module_name")
            .map(|s| s.to_string()),
        array_index_bounds_checking: matches
            .get_one::<String>("array_index_bounds_checking")
            .map(|s| s == "true"),
        separate_lines: matches
            .get_one::<String>("separate_lines")
            .map(|s| s == "true"),
    };
    result
}

pub fn codegen_flags_to_textproto(codegen_flags: &CodegenFlags) -> String {
    let mut pieces = vec![];
    if let Some(input_valid_signal) = &codegen_flags.input_valid_signal {
        pieces.push(format!("input_valid_signal: \"{input_valid_signal}\""));
    }
    if let Some(output_valid_signal) = &codegen_flags.output_valid_signal {
        pieces.push(format!("output_valid_signal: \"{output_valid_signal}\""));
    }
    if let Some(use_system_verilog) = codegen_flags.use_system_verilog {
        pieces.push(format!("use_system_verilog: {use_system_verilog}"));
    }
    if let Some(flop_inputs) = codegen_flags.flop_inputs {
        pieces.push(format!("flop_inputs: {flop_inputs}"));
    }
    if let Some(flop_outputs) = codegen_flags.flop_outputs {
        pieces.push(format!("flop_outputs: {flop_outputs}"));
    }
    if let Some(add_idle_output) = codegen_flags.add_idle_output {
        pieces.push(format!("add_idle_output: {add_idle_output}"));
    }
    if let Some(module_name) = &codegen_flags.module_name {
        pieces.push(format!("module_name: \"{module_name}\""));
    }
    if let Some(array_index_bounds_checking) = codegen_flags.array_index_bounds_checking {
        pieces.push(format!(
            "array_index_bounds_checking: {array_index_bounds_checking}"
        ));
    }
    if let Some(separate_lines) = codegen_flags.separate_lines {
        pieces.push(format!("separate_lines: {separate_lines}"));
    }
    pieces.join("\n")
}

#[derive(Debug)]
pub struct CodegenFlags {
    input_valid_signal: Option<String>,
    output_valid_signal: Option<String>,
    use_system_verilog: Option<bool>,
    flop_inputs: Option<bool>,
    flop_outputs: Option<bool>,
    add_idle_output: Option<bool>,
    module_name: Option<String>,
    array_index_bounds_checking: Option<bool>,
    separate_lines: Option<bool>,
}

/// Adds the given code-generation flags to the command in command-line-arg
/// form.
pub fn add_codegen_flags(command: &mut Command, codegen_flags: &CodegenFlags) {
    log::info!("add_codegen_flags");
    if let Some(use_system_verilog) = codegen_flags.use_system_verilog {
        command.arg(format!("--use_system_verilog={use_system_verilog}"));
    }
    if let Some(input_valid_signal) = &codegen_flags.input_valid_signal {
        command.arg("--input_valid_signal").arg(input_valid_signal);
    }
    if let Some(output_valid_signal) = &codegen_flags.output_valid_signal {
        command
            .arg("--output_valid_signal")
            .arg(output_valid_signal);
    }
    if let Some(flop_inputs) = codegen_flags.flop_inputs {
        command.arg(format!("--flop_inputs={flop_inputs}"));
    }
    if let Some(flop_outputs) = codegen_flags.flop_outputs {
        command.arg(format!("--flop_outputs={flop_outputs}"));
    }
    if let Some(add_idle_output) = codegen_flags.add_idle_output {
        command.arg(format!("--add_idle_output={add_idle_output}"));
    }
    if let Some(module_name) = &codegen_flags.module_name {
        command.arg("--module_name").arg(module_name);
    }
    if let Some(array_index_bounds_checking) = codegen_flags.array_index_bounds_checking {
        command.arg(format!(
            "--array_index_bounds_checking={array_index_bounds_checking}"
        ));
    }
    if let Some(separate_lines) = codegen_flags.separate_lines {
        command.arg(format!("--separate_lines={separate_lines}"));
    }
}
