// SPDX-License-Identifier: Apache-2.0

//! Runs configurable ABC optimization while preserving structural AIG choices.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::aig_serdes::load_abc_choice_aiger::load_abc_choice_aiger_auto_from_path;

/// A reproducible ABC Boolean-optimization and choice-generation schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbcChoiceOptimizationFlow {
    /// Number of Liberty-assisted optimization and choice-generation rounds.
    pub rounds: usize,
    /// Initial structural-choice preparation command.
    pub initial_choice_command: String,
    /// AIG restructuring command used in every optimization round.
    pub synthesis_command: String,
    /// LUT-balancing command used in every optimization round.
    pub lut_mapping_command: String,
    /// Structural-choice synthesis command used in every optimization round.
    pub choice_command: String,
    /// Temporary Liberty-mapping command used between optimization rounds.
    pub intermediate_mapping_command: String,
    /// Classic-network commands issued before entering the ABC GIA network.
    pub prefix_commands: Vec<String>,
    /// GIA commands issued after the last optimization round.
    pub suffix_commands: Vec<String>,
}

impl AbcChoiceOptimizationFlow {
    /// Returns the conventional five-round delay-oriented ABC speed schedule.
    pub fn baseline() -> Self {
        Self {
            rounds: 5,
            initial_choice_command: "&dch".to_string(),
            synthesis_command: "&syn2".to_string(),
            lut_mapping_command: "&if -g -K 6".to_string(),
            choice_command: "&synch2".to_string(),
            intermediate_mapping_command: "&nf".to_string(),
            prefix_commands: Vec::new(),
            suffix_commands: Vec::new(),
        }
    }

    /// Preserves AIG logic level during each Liberty-assisted synthesis round.
    pub fn level_preserving() -> Self {
        let mut flow = Self::baseline();
        flow.synthesis_command = "&resyn3".to_string();
        flow
    }
}

impl Default for AbcChoiceOptimizationFlow {
    fn default() -> Self {
        Self::level_preserving()
    }
}

/// Inputs needed to run ABC before native choice-AIG technology mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbcChoiceOptimizationOptions {
    /// Path to an ABC-compatible executable.
    pub abc_executable: PathBuf,
    /// Raw Liberty files used by intermediate ABC optimization rounds.
    pub liberty_files: Vec<PathBuf>,
    /// Optional ABC input-drive and output-load constraint file.
    pub constraints_file: Option<PathBuf>,
    /// Boolean-optimization and structural-choice-generation schedule.
    pub flow: AbcChoiceOptimizationFlow,
}

impl AbcChoiceOptimizationOptions {
    /// Constructs options using the default choice-preserving ABC schedule.
    pub fn new(abc_executable: impl Into<PathBuf>, liberty_files: Vec<PathBuf>) -> Self {
        Self {
            abc_executable: abc_executable.into(),
            liberty_files,
            constraints_file: None,
            flow: AbcChoiceOptimizationFlow::default(),
        }
    }
}

/// Scalar interface and structural counts recorded in an AIGER header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbcAigerInterface {
    /// Number of scalar primary inputs.
    pub inputs: usize,
    /// Number of AIGER latch bindings.
    pub latches: usize,
    /// Number of scalar primary outputs.
    pub outputs: usize,
    /// Number of serialized AND nodes, including structural alternatives.
    pub and_nodes: usize,
}

/// Diagnostics returned after successful ABC optimization and choice export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbcChoiceOptimizationResult {
    /// Original scalar interface and AND-node count.
    pub input_interface: AbcAigerInterface,
    /// Optimized scalar interface and serialized AND-node count.
    pub output_interface: AbcAigerInterface,
    /// Validated canonical structural-choice sibling-link count.
    pub choice_links: usize,
    /// Reproducible ABC command script used for the optimization.
    pub script: String,
    /// Captured ABC standard output; library routines do not print it.
    pub stdout: String,
    /// Captured ABC standard error; library routines do not print it.
    pub stderr: String,
}

/// Renders one deterministic ABC program ending in canonical q-AIGER export.
pub fn render_abc_choice_optimization_script(
    input_aiger: &Path,
    output_aiger: &Path,
    options: &AbcChoiceOptimizationOptions,
) -> Result<String, String> {
    if options.liberty_files.is_empty() && options.flow.rounds != 0 {
        return Err("ABC choice optimization requires at least one Liberty file".to_string());
    }

    let flow = &options.flow;
    for command in [
        flow.initial_choice_command.as_str(),
        flow.synthesis_command.as_str(),
        flow.lut_mapping_command.as_str(),
        flow.choice_command.as_str(),
        flow.intermediate_mapping_command.as_str(),
    ] {
        validate_abc_command(command)?;
    }
    for command in flow.prefix_commands.iter().chain(&flow.suffix_commands) {
        validate_abc_command(command)?;
    }

    let mut commands = vec![format!("read_aiger {}", quote_abc_path(input_aiger)?)];
    for (index, liberty_file) in options.liberty_files.iter().enumerate() {
        let merge = if index == 0 { "" } else { " -m" };
        commands.push(format!(
            "read_lib{merge} -w {}",
            quote_abc_path(liberty_file)?
        ));
    }
    if let Some(constraints_file) = &options.constraints_file {
        commands.push(format!(
            "read_constr -v {}",
            quote_abc_path(constraints_file)?
        ));
    }
    commands.extend(flow.prefix_commands.iter().cloned());
    commands.push("&get -n".to_string());
    commands.push("&st".to_string());
    commands.push(flow.initial_choice_command.clone());

    if flow.rounds != 0 {
        commands.push(flow.intermediate_mapping_command.clone());
        for round in 0..flow.rounds {
            commands.push("&st".to_string());
            commands.push(flow.synthesis_command.clone());
            commands.push(flow.lut_mapping_command.clone());
            commands.push(flow.choice_command.clone());
            if round + 1 != flow.rounds {
                commands.push(flow.intermediate_mapping_command.clone());
            }
        }
    }

    commands.extend(flow.suffix_commands.iter().cloned());
    commands.push("&ps".to_string());
    commands.push("&dfs -c".to_string());
    commands.push(format!("&write -s {}", quote_abc_path(output_aiger)?));
    Ok(format!("{}\n", commands.join("\n")))
}

/// Optimizes an AIG through ABC and writes its validated canonical choice AIG.
pub fn optimize_aiger_with_abc_choices(
    input_aiger: &Path,
    output_aiger: &Path,
    options: &AbcChoiceOptimizationOptions,
) -> Result<AbcChoiceOptimizationResult, String> {
    let input_aiger = input_aiger.canonicalize().map_err(|error| {
        format!(
            "failed to resolve input AIGER '{}': {error}",
            input_aiger.display()
        )
    })?;
    let input_interface = read_aiger_interface(&input_aiger)?;
    if input_interface.outputs == 0 || input_interface.and_nodes == 0 {
        let choices = load_abc_choice_aiger_auto_from_path(&input_aiger)
            .map_err(|error| format!("failed to validate degenerate input AIGER: {error}"))?;
        let bytes = fs::read(&input_aiger)
            .map_err(|error| format!("failed to read degenerate input AIGER: {error}"))?;
        fs::write(output_aiger, bytes).map_err(|error| {
            format!(
                "failed to write output AIGER '{}': {error}",
                output_aiger.display()
            )
        })?;
        return Ok(AbcChoiceOptimizationResult {
            input_interface,
            output_interface: input_interface,
            choice_links: choices.sibling_link_count(),
            script: String::new(),
            stdout: String::new(),
            stderr: String::new(),
        });
    }

    let mut resolved = options.clone();
    resolved.liberty_files = options
        .liberty_files
        .iter()
        .map(|path| {
            path.canonicalize().map_err(|error| {
                format!(
                    "failed to resolve ABC Liberty file '{}': {error}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    resolved.constraints_file = options
        .constraints_file
        .as_ref()
        .map(|path| {
            path.canonicalize().map_err(|error| {
                format!(
                    "failed to resolve ABC constraints file '{}': {error}",
                    path.display()
                )
            })
        })
        .transpose()?;

    let temporary = tempfile::Builder::new()
        .prefix("xlsynth-abc-choices-")
        .tempdir()
        .map_err(|error| format!("failed to create temporary ABC directory: {error}"))?;
    let temporary_output = temporary.path().join("optimized.choices.aig");
    let script = render_abc_choice_optimization_script(&input_aiger, &temporary_output, &resolved)?;
    let script_path = temporary.path().join("optimize.abc");
    fs::write(&script_path, &script)
        .map_err(|error| format!("failed to write temporary ABC script: {error}"))?;

    let output = Command::new(&options.abc_executable)
        .current_dir(temporary.path())
        .arg("-f")
        .arg(&script_path)
        .output()
        .map_err(|error| {
            format!(
                "failed to run ABC executable '{}': {error}",
                options.abc_executable.display()
            )
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return Err(format!(
            "ABC choice optimization failed with {}\nscript:\n{}stdout:\n{}\nstderr:\n{}",
            output.status, script, stdout, stderr
        ));
    }
    if abc_reported_failure(&stdout) || abc_reported_failure(&stderr) {
        return Err(format!(
            "ABC reported an optimization error despite a successful exit status\nscript:\n{}stdout:\n{}\nstderr:\n{}",
            script, stdout, stderr
        ));
    }

    let output_interface = read_aiger_interface(&temporary_output)?;
    if (
        input_interface.inputs,
        input_interface.latches,
        input_interface.outputs,
    ) != (
        output_interface.inputs,
        output_interface.latches,
        output_interface.outputs,
    ) {
        return Err(format!(
            "ABC changed the AIGER interface: input PI/latches/PO={}/{}/{}; output PI/latches/PO={}/{}/{}",
            input_interface.inputs,
            input_interface.latches,
            input_interface.outputs,
            output_interface.inputs,
            output_interface.latches,
            output_interface.outputs
        ));
    }
    let choices = load_abc_choice_aiger_auto_from_path(&temporary_output)
        .map_err(|error| format!("ABC produced a noncanonical choice AIG: {error}"))?;
    let bytes = fs::read(&temporary_output)
        .map_err(|error| format!("failed to read ABC choice AIG: {error}"))?;
    fs::write(output_aiger, bytes).map_err(|error| {
        format!(
            "failed to write optimized AIGER '{}': {error}",
            output_aiger.display()
        )
    })?;

    Ok(AbcChoiceOptimizationResult {
        input_interface,
        output_interface,
        choice_links: choices.sibling_link_count(),
        script,
        stdout,
        stderr,
    })
}

/// Detects ABC diagnostics that its executable sometimes exits successfully on.
fn abc_reported_failure(output: &str) -> bool {
    output.lines().any(|line| {
        let diagnostic = line.trim_start().to_ascii_lowercase();
        diagnostic.starts_with("error:")
            || diagnostic.contains("unknown command")
            || diagnostic.contains("unknown option")
    })
}

/// Reads the complete scalar interface from an ASCII or binary AIGER header.
pub fn read_aiger_interface(path: &Path) -> Result<AbcAigerInterface, String> {
    let file = fs::File::open(path)
        .map_err(|error| format!("failed to open AIGER '{}': {error}", path.display()))?;
    let mut line = String::new();
    BufReader::new(file)
        .read_line(&mut line)
        .map_err(|error| format!("failed to read AIGER header '{}': {error}", path.display()))?;
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 6 || !matches!(fields[0], "aig" | "aag") {
        return Err(format!(
            "invalid AIGER header in '{}': {line:?}",
            path.display()
        ));
    }
    let parse = |index: usize, name: &str| {
        fields[index].parse::<usize>().map_err(|error| {
            format!(
                "invalid AIGER {name} count '{}' in '{}': {error}",
                fields[index],
                path.display()
            )
        })
    };
    Ok(AbcAigerInterface {
        inputs: parse(2, "input")?,
        latches: parse(3, "latch")?,
        outputs: parse(4, "output")?,
        and_nodes: parse(5, "AND")?,
    })
}

/// Rejects multi-line commands so every configured action stays explicit.
fn validate_abc_command(command: &str) -> Result<(), String> {
    if command.trim().is_empty() {
        return Err("ABC optimization commands cannot be empty".to_string());
    }
    if command.contains(['\n', '\r']) {
        return Err("ABC optimization commands must fit on one line".to_string());
    }
    Ok(())
}

/// Quotes one filesystem path for ABC's script parser.
fn quote_abc_path(path: &Path) -> Result<String, String> {
    let value = path
        .to_str()
        .ok_or_else(|| format!("ABC path is not valid UTF-8: {}", path.display()))?;
    if value.contains(['\n', '\r']) {
        return Err(format!("ABC path contains a newline: {}", path.display()));
    }
    Ok(format!(
        "\"{}\"",
        value.replace('\\', "\\\\").replace('"', "\\\"")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> AbcChoiceOptimizationOptions {
        AbcChoiceOptimizationOptions::new("abc", vec![PathBuf::from("cells.lib")])
    }

    #[test]
    fn baseline_script_matches_five_round_choice_preserving_schedule() {
        let mut options = options();
        options.flow = AbcChoiceOptimizationFlow::baseline();
        let script = render_abc_choice_optimization_script(
            Path::new("input.aig"),
            Path::new("output.aig"),
            &options,
        )
        .unwrap();
        assert_eq!(
            script,
            r#"read_aiger "input.aig"
read_lib -w "cells.lib"
&get -n
&st
&dch
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&ps
&dfs -c
&write -s "output.aig"
"#
        );
    }

    #[test]
    fn default_schedule_preserves_logic_level_during_all_five_rounds() {
        let options = options();
        assert_eq!(options.flow, AbcChoiceOptimizationFlow::level_preserving());

        let script = render_abc_choice_optimization_script(
            Path::new("input.aig"),
            Path::new("output.aig"),
            &options,
        )
        .expect("the default level-preserving schedule should render");
        let restructuring = script
            .lines()
            .filter(|line| *line == "&resyn3" || *line == "&syn2")
            .collect::<Vec<_>>();

        assert_eq!(restructuring, vec!["&resyn3"; 5]);
    }

    #[test]
    fn custom_schedule_keeps_liberty_order_constraints_and_choice_export() {
        let mut options = options();
        options.liberty_files.push(PathBuf::from("extra.lib"));
        options.constraints_file = Some(PathBuf::from("timing constraints.txt"));
        options.flow.rounds = 1;
        options.flow.initial_choice_command = "&dch -f".to_string();
        options.flow.synthesis_command = "&syn2 -d -R 0".to_string();
        options.flow.lut_mapping_command = "&if -gx -K 7".to_string();
        options.flow.choice_command = "&synch2 -R 0".to_string();
        options.flow.prefix_commands.push("rewrite -z".to_string());
        options.flow.suffix_commands.push("&dch -f".to_string());

        let script = render_abc_choice_optimization_script(
            Path::new("input graph.aig"),
            Path::new("output graph.aig"),
            &options,
        )
        .unwrap();
        assert_eq!(
            script,
            r#"read_aiger "input graph.aig"
read_lib -w "cells.lib"
read_lib -m -w "extra.lib"
read_constr -v "timing constraints.txt"
rewrite -z
&get -n
&st
&dch -f
&nf
&st
&syn2 -d -R 0
&if -gx -K 7
&synch2 -R 0
&dch -f
&ps
&dfs -c
&write -s "output graph.aig"
"#
        );
    }

    #[test]
    fn zero_round_schedule_needs_no_liberty_and_preserves_initial_choices() {
        let mut options = options();
        options.liberty_files.clear();
        options.flow.rounds = 0;
        assert_eq!(
            render_abc_choice_optimization_script(
                Path::new("input.aig"),
                Path::new("output.aig"),
                &options,
            )
            .unwrap(),
            "read_aiger \"input.aig\"\n&get -n\n&st\n&dch\n&ps\n&dfs -c\n&write -s \"output.aig\"\n"
        );
    }

    #[test]
    fn rejects_empty_libraries_or_multiline_commands() {
        let mut no_libraries = options();
        no_libraries.liberty_files.clear();
        assert_eq!(
            render_abc_choice_optimization_script(
                Path::new("input.aig"),
                Path::new("output.aig"),
                &no_libraries,
            )
            .unwrap_err(),
            "ABC choice optimization requires at least one Liberty file"
        );

        let mut multiline = options();
        multiline.flow.synthesis_command = "&syn2\n&nf".to_string();
        assert_eq!(
            render_abc_choice_optimization_script(
                Path::new("input.aig"),
                Path::new("output.aig"),
                &multiline,
            )
            .unwrap_err(),
            "ABC optimization commands must fit on one line"
        );
    }

    #[test]
    fn recognizes_abc_failures_even_when_the_process_exits_successfully() {
        assert!(abc_reported_failure(
            "Error: Abc_FrameUpdateGia(): Transformation has failed\n"
        ));
        assert!(abc_reported_failure("abc: unknown command '&missing'\n"));
        assert!(abc_reported_failure("Error: unknown option '-m'\n"));
        assert!(!abc_reported_failure(
            "Warning: Using approximate matching\nnetwork: and = 12\n"
        ));
    }

    #[test]
    fn reads_and_validates_complete_aiger_interfaces() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("sample.aig");
        fs::write(&path, "aig 7 3 1 2 3\n").unwrap();
        assert_eq!(
            read_aiger_interface(&path).unwrap(),
            AbcAigerInterface {
                inputs: 3,
                latches: 1,
                outputs: 2,
                and_nodes: 3,
            }
        );
    }
}
