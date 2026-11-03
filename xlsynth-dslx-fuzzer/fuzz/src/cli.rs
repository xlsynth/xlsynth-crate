// SPDX-License-Identifier: Apache-2.0

//! Checks the standalone converter's stdout against the independent reference.

use std::path::Path;
use std::process::Command;
use xlsynth_dslx_fuzzer::{PreparedSample, Sample};
use xlsynth_prover::prover::types::EquivResult;

use crate::{check_prepared, native::NativeChecker};

/// Proves the standalone CLI output before running in-process native feedback.
pub fn check_cli(
    sample: &Sample,
    converter: &Path,
    stdlib: &Path,
    profile: Option<&Path>,
    native: Option<&NativeChecker>,
) -> Result<EquivResult, String> {
    let modules = sample.sources()?;
    let expected_ir = sample.expected_ir()?.to_string();
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    for module in &modules {
        std::fs::write(
            directory.path().join(format!("{}.x", module.name)),
            &module.text,
        )
        .map_err(|error| error.to_string())?;
    }
    let mut command = Command::new(converter);
    command
        .current_dir(directory.path())
        .arg(format!("--dslx_stdlib_path={}", stdlib.display()))
        .arg("--dslx_path=.")
        .arg("--warnings_as_errors=false")
        .arg("sample.x");
    if let Some(profile) = profile {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    let output = command
        .output()
        .map_err(|error| format!("could not execute {}: {error}", converter.display()))?;
    if !output.status.success() {
        return Err(format!(
            "ir_converter_main failed ({}): {}\n{modules:#?}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let prepared = PreparedSample {
        modules,
        expected_ir,
        converted_ir: String::from_utf8(output.stdout).map_err(|error| error.to_string())?,
        converted_top: xlsynth::mangle_dslx_name("sample", "main")
            .map_err(|error| error.to_string())?,
    };
    let result = check_prepared(&prepared)?;

    // Run the same source through the DSO to supply in-process C++ counters.
    // The CLI's IR has already been checked, even if the DSO fails separately.
    let entry = prepared.modules.last().ok_or("missing entry module")?;
    let options = xlsynth::DslxConvertOptions {
        dslx_stdlib_path: Some(stdlib),
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    xlsynth::convert_dslx_to_ir_text(&entry.text, Path::new("sample.x"), &options)
        .map_err(|error| format!("in-process conversion failed: {error}\n{prepared:#?}"))?;
    if let Some(native) = native {
        native
            .check(&prepared.modules)
            .map_err(|error| format!("native DSLX check failed: {error}\n{prepared:#?}"))?;
    }
    Ok(result)
}
