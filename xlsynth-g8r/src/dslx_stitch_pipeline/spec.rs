// SPDX-License-Identifier: Apache-2.0

//! Reusable DSLX source and interface description for a stitched pipeline.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use xlsynth::ir_package::IrPackage;
use xlsynth::{DslxConvertOptions, XlsynthError, convert_dslx_to_ir, dslx};

use super::names::{collect_stage_name_infos, validate_stitch_pipeline_names};
use super::{
    resolve_stage_ir_names, select_stage_names, verify_stage_port_widths, verify_stage_signatures,
};
use crate::aig::{ClockPort, SequentialPipelineOptions};

/// Source selection, register boundaries and external control ports shared by
/// both emitters.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StitchPipelineSpec {
    pub dslx_input_file: PathBuf,
    pub dslx_top: Option<String>,
    pub stages: Option<Vec<String>>,
    pub output_module_name: Option<String>,
    #[serde(default)]
    pub dslx_path: Vec<PathBuf>,
    pub dslx_stdlib_path: Option<PathBuf>,
    #[serde(default = "default_clock_name")]
    pub clock_name: String,
    #[serde(default = "default_flop_boundary")]
    pub flop_inputs: bool,
    #[serde(default = "default_flop_boundary")]
    pub flop_outputs: bool,
    pub input_valid_signal: Option<String>,
    pub output_valid_signal: Option<String>,
    pub reset: Option<String>,
    #[serde(default)]
    pub reset_active_low: bool,
}

fn default_clock_name() -> String {
    "clk".into()
}

fn default_flop_boundary() -> bool {
    true
}

impl StitchPipelineSpec {
    /// Reads a TOML specification and resolves all source and import paths
    /// relative to it.
    pub fn load(path: &Path) -> Result<Self, XlsynthError> {
        let contents = fs::read_to_string(path).map_err(|error| {
            XlsynthError(format!(
                "could not read stitch specification {}: {error}",
                path.display()
            ))
        })?;
        let mut spec: Self = toml::from_str(&contents).map_err(|error| {
            XlsynthError(format!(
                "invalid stitch specification {}: {error}",
                path.display()
            ))
        })?;
        spec.validate()?;
        let absolute = std::path::absolute(path)
            .map_err(|error| XlsynthError(format!("invalid specification path: {error}")))?;
        let base = absolute
            .parent()
            .ok_or_else(|| XlsynthError("specification path has no parent directory".into()))?;
        spec.dslx_input_file = base.join(&spec.dslx_input_file);
        spec.dslx_stdlib_path = spec.dslx_stdlib_path.map(|dir| base.join(dir));
        for dir in &mut spec.dslx_path {
            *dir = base.join(&*dir);
        }
        Ok(spec)
    }

    /// Checks manifest field dependencies and returns the enclosing design
    /// name.
    fn validate(&self) -> Result<&str, XlsynthError> {
        if self.input_valid_signal.is_none() {
            for (field, value) in [
                ("output_valid_signal", &self.output_valid_signal),
                ("reset", &self.reset),
            ] {
                if let Some(signal) = value {
                    return Err(XlsynthError(format!(
                        "{field} = {signal:?} requires input_valid_signal in the stitch specification"
                    )));
                }
            }
        }
        if self.reset_active_low && self.reset.is_none() {
            return Err(XlsynthError(
                "reset_active_low = true requires a reset signal in the stitch specification"
                    .into(),
            ));
        }
        match (self.dslx_top.as_deref(), self.stages.as_deref()) {
            (Some(top), None) => Ok(self.output_module_name.as_deref().unwrap_or(top)),
            (None, Some([])) => Err(XlsynthError("stages must be nonempty".into())),
            (None, Some(_)) => self.output_module_name.as_deref().ok_or_else(|| {
                XlsynthError("output_module_name is required with explicit stages".into())
            }),
            _ => Err(XlsynthError(
                "specify exactly one of dslx_top and stages".into(),
            )),
        }
    }

    /// Loads and typechecks the selected stages once for subsequent RTL and
    /// proof-model generation.
    pub fn resolve(&self) -> Result<ResolvedStitchPipeline, XlsynthError> {
        let name = self.validate()?;
        let text = fs::read_to_string(&self.dslx_input_file).map_err(|error| {
            XlsynthError(format!(
                "could not read DSLX source {}: {error}",
                self.dslx_input_file.display()
            ))
        })?;
        let search_paths: Vec<&Path> = self.dslx_path.iter().map(PathBuf::as_path).collect();
        let resolved = resolve_stages(
            &text,
            &self.dslx_input_file,
            self.dslx_top.as_deref().unwrap_or(name),
            self.stages.as_deref(),
            self.dslx_stdlib_path.as_deref(),
            &search_paths,
            SequentialPipelineOptions {
                name: name.to_string(),
                clock: ClockPort {
                    name: self.clock_name.clone(),
                },
                flop_inputs: self.flop_inputs,
                flop_outputs: self.flop_outputs,
                input_valid_signal: self.input_valid_signal.clone(),
                output_valid_signal: self.output_valid_signal.clone(),
                reset_signal: self.reset.clone(),
                reset_active_low: self.reset_active_low,
            },
        )?;
        let (stage_names, signatures) = collect_stage_name_infos(&resolved.ir, &resolved.stages)?;
        validate_stitch_pipeline_names(&stage_names, &signatures, &resolved.boundary)?;

        // Native g8r calls the data output output_value. Only the first
        // stage's parameters and the control ports share its external
        // namespace.
        let controls = [
            Some(self.clock_name.as_str()),
            self.reset.as_deref(),
            self.input_valid_signal.as_deref(),
            self.output_valid_signal.as_deref(),
        ];
        if controls.contains(&Some("output_value"))
            || stage_names[0]
                .params
                .iter()
                .any(|name| name == "output_value")
        {
            return Err(XlsynthError(
                "stitch specification port `output_value` collides with the native g8r data output"
                    .into(),
            ));
        }
        Ok(resolved)
    }
}

/// Typechecked IR, ordered stage names and the register boundaries for one
/// pipeline.
pub struct ResolvedStitchPipeline {
    pub(super) ir: IrPackage,
    pub(super) stages: Vec<(String, String)>,
    pub(super) boundary: SequentialPipelineOptions,
    latency: usize,
}

impl ResolvedStitchPipeline {
    pub fn name(&self) -> &str {
        &self.boundary.name
    }

    /// Returns the number of nonempty register layers between the external
    /// interface and the final output.
    pub fn latency(&self) -> usize {
        self.latency
    }

    pub(super) fn validate_wrapper_names(&self) -> Result<(), XlsynthError> {
        let (names, signatures) = collect_stage_name_infos(&self.ir, &self.stages)?;
        validate_stitch_pipeline_names(&names, &signatures, &self.boundary)
    }
}

/// Checks the path representation required by the libxls DSLX C API.
fn dslx_path_str<'a>(path: &'a Path, field: &str) -> Result<&'a str, XlsynthError> {
    let text = path
        .to_str()
        .ok_or_else(|| XlsynthError(format!("invalid {field}: path {path:?} is not UTF-8")))?;
    if text.contains('\0') {
        return Err(XlsynthError(format!(
            "invalid {field}: path contains a NUL byte"
        )));
    }
    Ok(text)
}

/// Typechecks DSLX, selects the stages in order and verifies their packed
/// interfaces.
pub(super) fn resolve_stages(
    text: &str,
    path: &Path,
    top: &str,
    explicit_stages: Option<&[String]>,
    stdlib_path: Option<&Path>,
    search_paths: &[&Path],
    boundary: SequentialPipelineOptions,
) -> Result<ResolvedStitchPipeline, XlsynthError> {
    let module_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| XlsynthError("invalid DSLX module filename".into()))?;
    let path_str = dslx_path_str(path, "dslx_input_file")?;
    if let Some(path) = stdlib_path {
        dslx_path_str(path, "dslx_stdlib_path")?;
    }
    for path in search_paths {
        dslx_path_str(path, "dslx_path")?;
    }
    let mut imports = dslx::ImportData::new(stdlib_path, search_paths);
    let module = dslx::parse_and_typecheck(text, path_str, module_name, &mut imports)?;
    let names = select_stage_names(&module, top, explicit_stages)?;
    let convert = DslxConvertOptions {
        dslx_stdlib_path: stdlib_path,
        additional_search_paths: search_paths.to_vec(),
        enable_warnings: None,
        disable_warnings: None,
        force_implicit_token_calling_convention: false,
    };
    let ir = convert_dslx_to_ir(text, path, &convert)?.ir;
    let stages = resolve_stage_ir_names(&ir, module_name, names)?;
    verify_stage_signatures(&ir, &stages)?;
    verify_stage_port_widths(&ir, &stages)?;
    // Both emitters skip input registers when a nullary stage has no valid
    // signal. Every inter-stage and output boundary carries nonzero-width data.
    let first_stage = ir.get_function(&stages[0].1)?;
    let has_input_layer = boundary.flop_inputs
        && (boundary.input_valid_signal.is_some() || first_stage.get_type()?.param_count() > 0);
    let latency =
        stages.len() - 1 + usize::from(has_input_layer) + usize::from(boundary.flop_outputs);
    Ok(ResolvedStitchPipeline {
        ir,
        stages,
        boundary,
        latency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dslx_stitch_pipeline::StitchRtlOptions;
    use crate::process_ir_path::CanonicalG8rOptions;
    use tempfile::tempdir;

    #[test]
    fn spec_reuses_stage_order_imports_and_boundaries_across_emitters() {
        let dir = tempdir().unwrap();
        let imports = dir.path().join("imports");
        fs::create_dir(&imports).unwrap();
        fs::write(
            imports.join("helper.x"),
            "pub fn inc(x: u8) -> u8 { x + u8:1 }",
        )
        .unwrap();
        fs::write(
            dir.path().join("stages.x"),
            r#"import helper;
fn first(x: u8) -> u8 { helper::inc(x) }
fn second(x: u8) -> u8 { !x }
"#,
        )
        .unwrap();
        let path = dir.path().join("pipeline.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "stages.x"
dslx_path = ["imports"]
stages = ["second", "first"]
output_module_name = "pipe"
clock_name = "tick"
flop_inputs = false
"#,
        )
        .unwrap();
        let spec = StitchPipelineSpec::load(&path).unwrap();
        assert_eq!(spec.dslx_input_file, dir.path().join("stages.x"));
        assert_eq!(spec.dslx_path, [imports]);
        let pipe = spec.resolve().unwrap();
        assert_eq!(pipe.name(), "pipe");
        assert_eq!(pipe.latency(), 2);
        let order = pipe
            .stages
            .iter()
            .map(|stage| stage.0.as_str())
            .collect::<Vec<_>>();
        assert_eq!(order, ["second", "first"]);
        let gates = pipe
            .lower_to_g8r(&CanonicalG8rOptions {
                enable_formal_array_alias_analysis: false,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(gates.clock.unwrap().name, "tick");
        assert_eq!(gates.registers.len(), 2);

        let mut default_clock_spec = spec;
        default_clock_spec.clock_name = "clk".to_string();
        let default_clock_rtl = default_clock_spec
            .resolve()
            .unwrap()
            .emit_rtl(&StitchRtlOptions::default())
            .unwrap();
        assert_eq!(
            pipe.emit_rtl(&StitchRtlOptions::default()).unwrap(),
            default_clock_rtl.replace("clk", "tick")
        );
    }

    #[test]
    fn spec_rejects_conflicting_stage_selectors_and_unknown_fields() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pipeline.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "unused.x"
dslx_top = "pipe"
stages = ["one"]
"#,
        )
        .unwrap();
        assert_eq!(
            StitchPipelineSpec::load(&path).unwrap_err().0,
            "specify exactly one of dslx_top and stages"
        );
        fs::write(
            &path,
            r#"dslx_input_file = "unused.x"
dslx_top = "pipe"
flop_input = false
"#,
        )
        .unwrap();
        assert!(
            StitchPipelineSpec::load(&path)
                .unwrap_err()
                .0
                .contains("unknown field `flop_input`")
        );
    }

    #[test]
    fn explicit_stages_must_name_concrete_source_functions() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("pipe.x"),
            r#"fn helper<N: u32>(x: uN[N]) -> uN[N] { x }
fn first(x: u4) -> u4 { helper(x) }"#,
        )
        .unwrap();
        let path = dir.path().join("pipe.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "pipe.x"
stages = ["first"]
output_module_name = "pipe""#,
        )
        .unwrap();
        let mut spec = StitchPipelineSpec::load(&path).unwrap();
        let pipeline = spec.resolve().unwrap();
        // XLS emitted this IR function to implement first's call to helper.
        assert!(pipeline.ir.get_function("__pipe__helper__4").is_ok());
        spec.stages = Some(vec!["helper__4".into()]);
        assert_eq!(
            spec.resolve().err().unwrap().0,
            r#"selected stage "helper__4" is not a declared DSLX function in module 'pipe'"#
        );
        spec.stages = Some(vec!["helper".into()]);
        assert_eq!(
            spec.resolve().err().unwrap().0,
            "parametric stage function(s) detected that cannot be stitched: helper. Provide a concrete (non-parametric) DSLX wrapper before running the stitch tool."
        );
    }

    #[test]
    fn spec_rejects_control_fields_without_their_dependencies() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pipeline.toml");
        for (control, expected) in [
            (
                r#"output_valid_signal = "vout""#,
                r#"output_valid_signal = "vout" requires input_valid_signal in the stitch specification"#,
            ),
            (
                r#"reset = "rst""#,
                r#"reset = "rst" requires input_valid_signal in the stitch specification"#,
            ),
            (
                "reset_active_low = true",
                "reset_active_low = true requires a reset signal in the stitch specification",
            ),
        ] {
            fs::write(
                &path,
                format!(
                    r#"dslx_input_file = "unused.x"
dslx_top = "pipe"
{control}
"#
                ),
            )
            .unwrap();
            assert_eq!(StitchPipelineSpec::load(&path).unwrap_err().0, expected);
        }
    }

    #[test]
    fn nullary_stage_registers_input_valid_when_configured() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("pipe.x"),
            "fn pipe_cycle0() -> bool { true }",
        )
        .unwrap();
        let path = dir.path().join("pipe.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = true
flop_outputs = false
input_valid_signal = "vin"
output_valid_signal = "vout""#,
        )
        .unwrap();
        let pipeline = StitchPipelineSpec::load(&path).unwrap().resolve().unwrap();
        assert_eq!(pipeline.latency(), 1);
        let gates = pipeline
            .lower_to_g8r(&CanonicalG8rOptions {
                enable_formal_array_alias_analysis: false,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(gates.registers.len(), 1);
        assert_eq!(gates.registers[0].name, "p0_vin");
    }

    #[test]
    fn spec_reports_invalid_import_path_before_calling_libxls() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("pipe.x"),
            "fn pipe_cycle0(x: bool) -> bool { x }",
        )
        .unwrap();
        let path = dir.path().join("pipe.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
dslx_path = ["imports\u0000"]"#,
        )
        .unwrap();
        let spec = StitchPipelineSpec::load(&path).unwrap();
        assert_eq!(
            spec.resolve().err().unwrap().0,
            "invalid dslx_path: path contains a NUL byte"
        );
    }

    #[test]
    fn implicit_stage_selection_rejects_indices_it_cannot_resolve() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("pipe.x");
        let path = dir.path().join("pipe.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "pipe.x"
dslx_top = "pipe""#,
        )
        .unwrap();
        let spec = StitchPipelineSpec::load(&path).unwrap();
        for (index, expected) in [
            (
                "01",
                "stage 'pipe_cycle01' has a leading zero in its index; implicit stage selection expects 'pipe_cycle1'",
            ),
            (
                "999999999999999999999999",
                "stage 'pipe_cycle999999999999999999999999' has an index too large for implicit stage selection",
            ),
        ] {
            fs::write(&source, format!("fn pipe_cycle0(x: bool) -> bool {{ x }}\nfn pipe_cycle{index}(x: bool) -> bool {{ x }}")).unwrap();
            assert_eq!(spec.resolve().err().unwrap().0, expected);
        }
    }

    #[test]
    fn spec_rejects_a_system_verilog_keyword_as_the_clock_name() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("pipe.x"),
            "fn pipe_cycle0(x: bool) -> bool { x }",
        )
        .unwrap();
        let path = dir.path().join("pipe.toml");
        fs::write(
            &path,
            r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
clock_name = "checker""#,
        )
        .unwrap();
        let spec = StitchPipelineSpec::load(&path).unwrap();
        assert_eq!(
            spec.resolve().err().unwrap().0,
            "name validation failed: clock control port `checker` cannot be emitted as-is: it is a SystemVerilog keyword"
        );
    }

    #[test]
    fn spec_reserves_native_result_only_on_the_external_interface() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("stages.x");
        fs::write(
            &source,
            r#"fn pipe_cycle0(x: u8) -> u8 { x }
fn pipe_cycle1(output_value: u8) -> u8 { output_value }
"#,
        )
        .unwrap();
        let path = dir.path().join("pipeline.toml");
        let base = r#"dslx_input_file = "stages.x"
dslx_top = "pipe"
"#;
        fs::write(&path, base).unwrap();
        let spec = StitchPipelineSpec::load(&path).unwrap();
        spec.resolve().unwrap();
        let expected =
            "stitch specification port `output_value` collides with the native g8r data output";
        for controls in [
            r#"clock_name = "output_value""#,
            r#"input_valid_signal = "output_value""#,
            r#"input_valid_signal = "in_valid"
output_valid_signal = "output_value""#,
            r#"input_valid_signal = "in_valid"
reset = "output_value""#,
        ] {
            fs::write(&path, format!("{base}{controls}")).unwrap();
            let with_controls = StitchPipelineSpec::load(&path).unwrap();
            assert_eq!(
                with_controls.resolve().err().unwrap().0,
                expected,
                "{controls}"
            );
        }
        fs::write(
            &source,
            "fn pipe_cycle0(output_value: u8) -> u8 { output_value }",
        )
        .unwrap();
        assert_eq!(spec.resolve().err().unwrap().0, expected);
    }
}
