// SPDX-License-Identifier: Apache-2.0

//! Checks the standalone converter's stdout against the independent reference.

use std::path::{Component, Path};
use std::process::Command;
use xlsynth_dslx_fuzzer::converter_contracts::ContractCase;
use xlsynth_dslx_fuzzer::{Grammar, PreparedSample, Sample, SourceModule};
use xlsynth_pir::ir::{NaryOp, NodePayload};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::AssertionSemantics;
use xlsynth_prover::prover::types::EquivResult;

use crate::{CheckFailure, classify_prepared, classify_with_assertions, native::NativeChecker};

/// Proves the standalone CLI output before running in-process native feedback.
pub fn check_cli(
    sample: &Sample,
    grammar: Grammar,
    converter: &Path,
    stdlib: &Path,
    profile: Option<&Path>,
    native: Option<&NativeChecker>,
) -> Result<EquivResult, String> {
    classify_cli(sample, grammar, converter, stdlib, profile, native)
        .map_err(CheckFailure::into_message)
}

/// Distinguishes production converter findings from checker and execution
/// errors.
pub fn classify_cli(
    sample: &Sample,
    grammar: Grammar,
    converter: &Path,
    stdlib: &Path,
    profile: Option<&Path>,
    native: Option<&NativeChecker>,
) -> Result<EquivResult, CheckFailure> {
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let prepared = convert_modules(
        sample.sources()?,
        sample.expected_ir()?.to_string(),
        converter,
        stdlib,
        directory.path(),
        profile,
        false,
    )?;
    let result = classify_prepared(&prepared, grammar)?;
    native_feedback(&prepared, stdlib, directory.path(), native)?;
    Ok(result)
}

/// Compares a focused program with its independent reference, including the
/// conditions for assertion failures and any case-specific IR size bound.
pub fn check_contract(
    case: ContractCase,
    converter: &Path,
    stdlib: &Path,
) -> Result<EquivResult, CheckFailure> {
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let prepared = convert_modules(
        case.modules,
        case.expected_ir,
        converter,
        stdlib,
        directory.path(),
        None,
        true,
    )?;
    xlsynth::IrPackage::parse_ir(&prepared.converted_ir, None).map_err(|error| {
        CheckFailure::InvalidOutput(format!(
            "C++ IR parser rejected converter output: {error}\n{prepared:#?}"
        ))
    })?;
    let result = classify_with_assertions(&prepared, AssertionSemantics::Same);
    // CLI rejections are returned before this point. Successful conversions,
    // including wrong-value cases, can feed in-process native counters.
    native_feedback(&prepared, stdlib, directory.path(), None)?;
    let outcome = result?;
    if let (EquivResult::Proved, Some(limit)) = (&outcome, case.max_and_nodes) {
        let package = Parser::new(&prepared.converted_ir)
            .parse_package()
            .map_err(|error| error.to_string())?;
        let actual = package
            .members
            .iter()
            .flat_map(|member| &member.graph().nodes)
            .filter(|node| matches!(node.payload, NodePayload::Nary(NaryOp::And, _)))
            .count();
        if actual > limit {
            return Err(CheckFailure::StructuralGrowth(format!(
                "Value and assertion behavior proved equal, but emitted {actual} AND nodes with a limit of {limit}\n{prepared:#?}"
            )));
        }
    }
    Ok(outcome)
}

fn convert_modules(
    modules: Vec<SourceModule>,
    expected_ir: String,
    converter: &Path,
    stdlib: &Path,
    directory: &Path,
    profile: Option<&Path>,
    emit_assert: bool,
) -> Result<PreparedSample, CheckFailure> {
    for module in &modules {
        let stem = Path::new(&module.name);
        if !stem
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
            || stem.as_os_str().is_empty()
        {
            return Err(CheckFailure::Checker(format!(
                "invalid generated module name: {:?}",
                module.name
            )));
        }
        let file = directory.join(format!("{}.x", module.name));
        std::fs::create_dir_all(file.parent().expect("generated path has a parent"))
            .map_err(|error| error.to_string())?;
        std::fs::write(file, &module.text).map_err(|error| error.to_string())?;
    }
    let mut command = Command::new(converter);
    command
        .current_dir(directory)
        .arg(format!("--dslx_stdlib_path={}", stdlib.display()))
        .arg("--dslx_path=.")
        .arg("--warnings_as_errors=false");
    if emit_assert {
        command.arg("--emit_assert=true");
    }
    command.arg("sample.x");
    if let Some(profile) = profile {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    let output = command
        .output()
        .map_err(|error| format!("could not execute {}: {error}", converter.display()))?;
    if !output.status.success() {
        return Err(CheckFailure::ConverterRejected(format!(
            "ir_converter_main failed ({}): {}\n{modules:#?}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(PreparedSample {
        modules,
        expected_ir,
        converted_ir: String::from_utf8(output.stdout).map_err(|error| error.to_string())?,
        converted_top: xlsynth::mangle_dslx_name("sample", "main")
            .map_err(|error| error.to_string())?,
    })
}

fn native_feedback(
    prepared: &PreparedSample,
    stdlib: &Path,
    directory: &Path,
    native: Option<&NativeChecker>,
) -> Result<(), CheckFailure> {
    // Run the same source through the DSO to supply in-process C++ counters.
    // The CLI's IR has already been checked, even if the DSO fails separately.
    let entry = prepared
        .modules
        .last()
        .ok_or_else(|| String::from("missing entry module"))?;
    let options = xlsynth::DslxConvertOptions {
        dslx_stdlib_path: Some(stdlib),
        additional_search_paths: vec![directory],
        ..Default::default()
    };
    xlsynth::convert_dslx_to_ir_text(&entry.text, Path::new("sample.x"), &options)
        .map_err(|error| format!("in-process conversion failed: {error}\n{prepared:#?}"))?;
    if let Some(native) = native {
        native
            .check(&prepared.modules)
            .map_err(|error| format!("native DSLX check failed: {error}\n{prepared:#?}"))?;
    }
    Ok(())
}
