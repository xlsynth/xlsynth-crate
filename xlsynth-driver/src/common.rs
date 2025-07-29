// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::collections::HashSet;
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
    let mut out: Vec<std::path::PathBuf> = Vec::new();

    // --dslx_path flag: semicolon-separated list like "dirA;dirB" (kept for
    // consistency with other subcommands).
    if let Some(flag_value) = matches.get_one::<String>("dslx_path") {
        for entry in flag_value.split(';').filter(|s| !s.is_empty()) {
            out.push(std::path::PathBuf::from(entry));
        }
    }

    // toolchain.toml paths: Vec<String>.
    if let Some(cfg) = config {
        if let Some(dslx_cfg) = &cfg.dslx {
            if let Some(vec) = &dslx_cfg.dslx_path {
                for p in vec {
                    if !p.is_empty() {
                        out.push(std::path::PathBuf::from(p));
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
// DSLX helpers shared across subcommands
// -----------------------------------------------------------------------------

use xlsynth_g8r::equiv::prove_equiv::ParamDomains;

pub fn get_enum_domain(
    tcm: &xlsynth::dslx::TypecheckedModule,
    enum_def: &xlsynth::dslx::EnumDef,
) -> Vec<xlsynth::IrValue> {
    let mut values = Vec::new();
    for mi in 0..enum_def.get_member_count() {
        let m = enum_def.get_member(mi);
        let expr = m.get_value();
        let owner_module = expr.get_owner_module();
        let owner_type_info = tcm
            .get_type_info_for_module(&owner_module)
            .expect("imported type info");
        let interp = owner_type_info.get_const_expr(&expr).expect("constexpr");
        let ir_val = interp.convert_to_ir().expect("convert");
        values.push(ir_val);
    }

    values
}

pub fn get_function_enum_param_domains(
    tcm: &xlsynth::dslx::TypecheckedModule,
    dslx_top: &str,
) -> ParamDomains {
    let module = tcm.get_module();
    let type_info = tcm.get_type_info();
    let mut domains: ParamDomains = std::collections::HashMap::new();

    for i in 0..module.get_member_count() {
        if let Some(xlsynth::dslx::MatchableModuleMember::Function(f)) =
            module.get_member(i).to_matchable()
        {
            if f.get_identifier() == dslx_top {
                for pidx in 0..f.get_param_count() {
                    let p = f.get_param(pidx);
                    let name = p.get_name();
                    let ta = p.get_type_annotation();
                    let ty = type_info.get_type_for_type_annotation(&ta);
                    if ty.is_enum() {
                        let enum_def = ty.get_enum_def().unwrap();
                        let values = get_enum_domain(tcm, &enum_def);
                        domains.insert(name, values);
                    }
                }
            }
        }
    }

    domains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_function_enum_param_domains_cross_module() {
        use xlsynth::dslx;

        // Temp directory housing the modules.
        let tmpdir = xlsynth_test_helpers::make_test_tmpdir("xlsynth_driver_dslx_test");
        let dir = tmpdir.path();

        // imported.x defines an enum whose member uses a local const.
        let imported_path = dir.join("imported.x");
        let imported_dslx = r#"
            const K = u3:5;
            pub enum ImpE : u3 { Z = 0, P = K }
        "#;
        std::fs::write(&imported_path, imported_dslx).expect("write imported.x");

        // main.x imports `imported` and exposes a top that takes the imported enum.
        let main_path = dir.join("main.x");
        let main_dslx = r#"
            import imported;
            pub fn top(x: imported::ImpE) -> u3 { u3:0 }
        "#;
        std::fs::write(&main_path, main_dslx).expect("write main.x");

        // Parse/typecheck the main module; imported will be resolved via search path.
        let mut import_data = dslx::ImportData::new(None, &[dir]);
        let tcm = dslx::parse_and_typecheck(
            main_dslx,
            main_path.to_str().unwrap(),
            "main",
            &mut import_data,
        )
        .expect("parse_and_typecheck success");

        // Exercise the helper under test.
        let domains = get_function_enum_param_domains(&tcm, "top");
        assert!(domains.contains_key("x"));
        let values = domains.get("x").unwrap();
        assert_eq!(values.len(), 2);
        assert!(values.contains(&xlsynth::IrValue::make_ubits(3, 0).unwrap()));
        assert!(values.contains(&xlsynth::IrValue::make_ubits(3, 5).unwrap()));
    }
}
