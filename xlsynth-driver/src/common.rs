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

#[derive(Debug)]
pub struct CodegenFlags {
    input_valid_signal: Option<String>,
    output_valid_signal: Option<String>,
    use_system_verilog: Option<bool>,
    flop_inputs: Option<bool>,
    flop_outputs: Option<bool>,
    add_idle_output: Option<bool>,
    add_invariant_assertions: Option<bool>,
    module_name: Option<String>,
    array_index_bounds_checking: Option<bool>,
    separate_lines: Option<bool>,
    reset: Option<String>,
    reset_active_low: Option<bool>,
    reset_asynchronous: Option<bool>,
    reset_data_path: Option<bool>,
    gate_format: Option<String>,
    assert_format: Option<String>,
    output_schedule_path: Option<String>,
    output_verilog_line_map_path: Option<String>,
}

/// Extracts flags that we pass to the "codegen" step of the process (i.e.
/// generating lowered Verilog).
pub fn extract_codegen_flags(
    matches: &ArgMatches,
    toolchain_config: Option<&crate::toolchain_config::ToolchainConfig>,
) -> CodegenFlags {
    let (gate_format, assert_format) = if let Some(config) = toolchain_config {
        (
            config.codegen.as_ref().and_then(|c| c.gate_format.clone()),
            config
                .codegen
                .as_ref()
                .and_then(|c| c.assert_format.clone()),
        )
    } else {
        (None, None)
    };
    CodegenFlags {
        input_valid_signal: matches
            .get_one::<String>("input_valid_signal")
            .map(|s| s.to_string()),
        output_valid_signal: matches
            .get_one::<String>("output_valid_signal")
            .map(|s| s.to_string()),
        use_system_verilog: matches
            .get_one::<String>("use_system_verilog")
            .map(|s| s == "true")
            .or_else(|| toolchain_config.and_then(|c| c.codegen.as_ref()?.use_system_verilog)),
        flop_inputs: matches
            .get_one::<String>("flop_inputs")
            .map(|s| s == "true"),
        flop_outputs: matches
            .get_one::<String>("flop_outputs")
            .map(|s| s == "true"),
        add_idle_output: matches
            .get_one::<String>("add_idle_output")
            .map(|s| s == "true"),
        add_invariant_assertions: matches
            .get_one::<String>("add_invariant_assertions")
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
        reset: matches.get_one::<String>("reset").map(|s| s.to_string()),
        reset_active_low: matches
            .get_one::<String>("reset_active_low")
            .map(|s| s == "true"),
        reset_asynchronous: matches
            .get_one::<String>("reset_asynchronous")
            .map(|s| s == "true"),
        reset_data_path: matches
            .get_one::<String>("reset_data_path")
            .map(|s| s == "true"),
        gate_format,
        assert_format,
        output_schedule_path: matches
            .get_one::<String>("output_schedule_path")
            .map(|s| s.to_string()),
        output_verilog_line_map_path: matches
            .get_one::<String>("output_verilog_line_map_path")
            .map(|s| s.to_string()),
    }
}

pub fn codegen_flags_to_textproto(codegen_flags: &CodegenFlags) -> String {
    log::debug!(
        "codegen_flags_to_textproto; codegen_flags: {:?}",
        codegen_flags
    );
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
    if let Some(add_invariant_assertions) = codegen_flags.add_invariant_assertions {
        pieces.push(format!(
            "add_invariant_assertions: {add_invariant_assertions}"
        ));
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
    if let Some(reset) = &codegen_flags.reset {
        pieces.push(format!("reset: \"{reset}\""));
    }
    if let Some(reset_active_low) = codegen_flags.reset_active_low {
        pieces.push(format!("reset_active_low: {reset_active_low}"));
    }
    if let Some(reset_asynchronous) = codegen_flags.reset_asynchronous {
        pieces.push(format!("reset_asynchronous: {reset_asynchronous}"));
    }
    if let Some(reset_data_path) = codegen_flags.reset_data_path {
        pieces.push(format!("reset_data_path: {reset_data_path}"));
    }
    if let Some(gate_format) = &codegen_flags.gate_format {
        pieces.push(format!("gate_format: {gate_format:?}"));
    }
    if let Some(assert_format) = &codegen_flags.assert_format {
        pieces.push(format!("assert_format: {assert_format:?}"));
    }
    if let Some(output_schedule_path) = &codegen_flags.output_schedule_path {
        pieces.push(format!("output_schedule_path: \"{output_schedule_path}\""));
    }
    if let Some(output_verilog_line_map_path) = &codegen_flags.output_verilog_line_map_path {
        pieces.push(format!(
            "output_verilog_line_map_path: \"{output_verilog_line_map_path}\""
        ));
    }
    pieces.push(format!("assertion_macro_names: \"ASSERT_ON\""));
    pieces.join("\n")
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
    if let Some(add_invariant_assertions) = codegen_flags.add_invariant_assertions {
        command.arg(format!(
            "--add_invariant_assertions={add_invariant_assertions}"
        ));
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
    if let Some(reset) = &codegen_flags.reset {
        command.arg(format!("--reset={reset}"));
    }
    if let Some(reset_active_low) = codegen_flags.reset_active_low {
        command.arg(format!("--reset_active_low={reset_active_low}"));
    }
    if let Some(reset_asynchronous) = codegen_flags.reset_asynchronous {
        command.arg(format!("--reset_asynchronous={reset_asynchronous}"));
    }
    if let Some(reset_data_path) = codegen_flags.reset_data_path {
        command.arg(format!("--reset_data_path={reset_data_path}"));
    }
    if let Some(gate_format) = &codegen_flags.gate_format {
        command.arg(format!("--gate_format={gate_format}"));
    }
    if let Some(assert_format) = &codegen_flags.assert_format {
        command.arg(format!("--assert_format={assert_format}"));
    }
    if let Some(output_schedule_path) = &codegen_flags.output_schedule_path {
        command
            .arg("--output_schedule_path")
            .arg(output_schedule_path);
    }
    if let Some(output_verilog_line_map_path) = &codegen_flags.output_verilog_line_map_path {
        command
            .arg("--output_verilog_line_map_path")
            .arg(output_verilog_line_map_path);
    }
}

/// Builds the textproto string for `SchedulingOptionsProto` given the delay
/// model and pipeline specification.
pub fn scheduling_options_proto(delay_model: &str, pipeline_spec: &PipelineSpec) -> String {
    let mut lines = vec![format!("delay_model: \"{}\"", delay_model)];
    match pipeline_spec {
        PipelineSpec::Stages(stages) => lines.push(format!("pipeline_stages: {}", stages)),
        PipelineSpec::ClockPeriodPs(clock_period_ps) => {
            lines.push(format!("clock_period_ps: {}", clock_period_ps))
        }
    }
    lines.join("\n")
}

/// Builds the textproto string for `CodegenFlagsProto` used by pipeline
/// generation.
pub fn pipeline_codegen_flags_proto(codegen_flags: &CodegenFlags) -> String {
    format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\ngenerator: GENERATOR_KIND_PIPELINE\n{}",
        codegen_flags_to_textproto(codegen_flags)
    )
}

/// Gathers additional DSLX module search paths from:
///   • the `--dslx_path` command-line flag (semi-colon separated)
///   • `[toolchain.dslx].dslx_path` array in the toolchain config.
/// Returns a vector of unique `PathBuf`s in the order first–seen.
pub fn collect_dslx_search_paths(
    matches: &ArgMatches,
    config: &Option<crate::toolchain_config::ToolchainConfig>,
) -> Vec<std::path::PathBuf> {
    use std::collections::HashSet;
    use std::path::PathBuf;

    let mut out: Vec<PathBuf> = Vec::new();

    // --dslx_path flag: semicolon-separated list like "dirA;dirB" (kept for
    // consistency with other subcommands).
    if let Some(flag_value) = matches.get_one::<String>("dslx_path") {
        for entry in flag_value.split(';').filter(|s| !s.is_empty()) {
            out.push(PathBuf::from(entry));
        }
    }

    // toolchain.toml paths: Vec<String>.
    if let Some(cfg) = config {
        if let Some(dslx_cfg) = &cfg.dslx {
            if let Some(vec) = &dslx_cfg.dslx_path {
                for p in vec {
                    if !p.is_empty() {
                        out.push(PathBuf::from(p));
                    }
                }
            }
        }
    }

    // Deduplicate preserving the first occurrence.
    let mut seen: HashSet<std::path::PathBuf> = HashSet::new();
    out.retain(|p| seen.insert(p.clone()));
    out
}

/// Holds resolved locations for DSLX support files.
pub struct DslxPaths {
    /// Optional alternative stdlib root.
    pub stdlib_path: Option<std::path::PathBuf>,
    /// Additional search directories beyond the source file’s dir.
    pub search_paths: Vec<std::path::PathBuf>,
}

impl DslxPaths {
    /// Returns the additional search paths as borrowed `&Path` slice for APIs
    /// like `ImportData::new` or `DslxConvertOptions`.
    pub fn search_path_views(&self) -> Vec<&std::path::Path> {
        self.search_paths.iter().map(|p| p.as_path()).collect()
    }
}

/// Builds the `DslxPaths` structure from CLI flags / toolchain configuration.
pub fn get_dslx_paths(
    matches: &ArgMatches,
    config: &Option<crate::toolchain_config::ToolchainConfig>,
) -> DslxPaths {
    use crate::toolchain_config::get_dslx_stdlib_path;

    let stdlib_path_opt =
        get_dslx_stdlib_path(matches, config).map(|s| std::path::PathBuf::from(s));

    let search_paths = collect_dslx_search_paths(matches, config);

    DslxPaths {
        stdlib_path: stdlib_path_opt,
        search_paths,
    }
}

/// Locates an executable in PATH and verifies it can actually be executed.
///
/// This function:
/// 1. Uses `which` to find the executable in PATH
/// 2. Verifies the file has executable permissions (on Unix systems)
/// 3. Returns the path if valid, or a descriptive error if not
pub fn find_and_verify_executable(
    name: &str,
    install_hint: &str,
) -> anyhow::Result<std::path::PathBuf> {
    use std::path::PathBuf;

    // Find the executable in PATH
    let exe_path = which::which(name).map_err(|_| {
        anyhow::anyhow!("`{}` executable not found in PATH. {}", name, install_hint)
    })?;

    // Verify the binary is actually executable
    if let Ok(metadata) = std::fs::metadata(&exe_path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = metadata.permissions();
            if perms.mode() & 0o111 == 0 {
                return Err(anyhow::anyhow!(
                    "{} binary at '{}' exists but is not executable (permissions: {:o})",
                    name,
                    exe_path.display(),
                    perms.mode()
                ));
            }
        }
    }

    Ok(exe_path)
}

/// Executes a command and provides a descriptive error message if execution
/// fails.
pub fn execute_command_with_context(
    mut cmd: std::process::Command,
    context: &str,
) -> anyhow::Result<std::process::Output> {
    log::debug!("execute_command_with_context: About to execute command");
    let result = cmd.output().map_err(|e| {
        anyhow::anyhow!(
            "{}: {}. This could indicate missing dynamic libraries or other execution issues.",
            context,
            e
        )
    });
    match &result {
        Ok(output) => log::debug!(
            "execute_command_with_context: Command completed with status: {}",
            output.status
        ),
        Err(e) => log::debug!("execute_command_with_context: Command failed: {}", e),
    }
    result
}

// -----------------------------------------------------------------------------
// Shared DSLX -> IR conversion / optimization artifacts
// -----------------------------------------------------------------------------

pub struct DslxIrArtifacts {
    pub raw_ir: String,
    pub optimized_ir: Option<String>,
    pub mangled_top: String,
}

pub struct DslxIrBuildOptions<'a> {
    pub input_path: &'a std::path::Path,
    pub dslx_top: &'a str,
    pub dslx_stdlib_path: Option<&'a str>,
    pub dslx_path: Option<&'a str>, // semicolon separated when Some
    pub enable_warnings: Option<&'a [String]>,
    pub disable_warnings: Option<&'a [String]>,
    pub tool_path: Option<&'a str>,
    pub type_inference_v2: Option<bool>,
    pub optimize: bool,
}

/// Generalized builder that produces raw IR and (optionally) optimized IR for a
/// DSLX module. If `optimize` is false, `optimized_ir` will be `None`.
pub fn dslx_to_ir(opts: &DslxIrBuildOptions) -> DslxIrArtifacts {
    let module_name = opts
        .input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("valid module name");
    let mangled_top =
        xlsynth::mangle_dslx_name(module_name, opts.dslx_top).expect("mangling succeeds");

    if let Some(tool_path) = opts.tool_path {
        // External tool path conversion.
        let raw_ir = crate::tools::run_ir_converter_main(
            opts.input_path,
            Some(opts.dslx_top),
            opts.dslx_stdlib_path,
            opts.dslx_path,
            tool_path,
            opts.enable_warnings,
            opts.disable_warnings,
            opts.type_inference_v2,
        );
        let optimized_ir = if opts.optimize {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(tmp.path(), &raw_ir).unwrap();
            Some(crate::tools::run_opt_main(
                tmp.path(),
                Some(&mangled_top),
                tool_path,
            ))
        } else {
            None
        };
        DslxIrArtifacts {
            raw_ir,
            optimized_ir,
            mangled_top,
        }
    } else {
        if opts.type_inference_v2 == Some(true) {
            eprintln!("error: --type_inference_v2 is only supported with external toolchain");
            std::process::exit(1);
        }
        // Internal conversion path.
        let dslx_contents = std::fs::read_to_string(opts.input_path).unwrap_or_else(|e| {
            eprintln!(
                "Failed to read DSLX file {}: {}",
                opts.input_path.display(),
                e
            );
            std::process::exit(1);
        });
        let stdlib_path = opts.dslx_stdlib_path.map(|p| std::path::Path::new(p));
        let additional_search_paths: Vec<&std::path::Path> = opts
            .dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let convert_result = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            opts.input_path,
            &xlsynth::DslxConvertOptions {
                dslx_stdlib_path: stdlib_path,
                additional_search_paths,
                enable_warnings: opts.enable_warnings,
                disable_warnings: opts.disable_warnings,
            },
        )
        .expect("successful DSLX->IR conversion");
        for w in &convert_result.warnings {
            log::warn!("DSLX warning: {}", w);
        }
        let raw_ir = convert_result.ir;
        let optimized_ir = if opts.optimize {
            let pkg = xlsynth::IrPackage::parse_ir(&raw_ir, Some(&mangled_top)).unwrap();
            Some(
                xlsynth::optimize_ir(&pkg, &mangled_top)
                    .unwrap()
                    .to_string(),
            )
        } else {
            None
        };
        DslxIrArtifacts {
            raw_ir,
            optimized_ir,
            mangled_top,
        }
    }
}
