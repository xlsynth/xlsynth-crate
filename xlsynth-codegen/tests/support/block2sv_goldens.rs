// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use pretty_assertions::assert_eq;
use xlsynth_codegen::emit_system_verilog;

#[path = "golden_fixture.rs"]
mod golden_fixture;
pub(super) use golden_fixture::GoldenFixture;
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir_parser::Parser;

/// Flattens a small bits or tuple result in SystemVerilog port packing order.
pub(super) fn flatten_reference_value(value: &xlsynth_pir::IrValue, ty: &Type) -> u64 {
    match ty {
        Type::Bits(_) => value.to_u64().expect("reference bits value fits in u64"),
        Type::Tuple(elements) => elements.iter().enumerate().fold(0, |packed, (index, ty)| {
            let value = value
                .get_element(index)
                .expect("reference tuple element exists");
            (packed << ty.bit_count()) | flatten_reference_value(&value, ty)
        }),
        Type::Array(_) | Type::Token => panic!("unexpected extension result type {ty:?}"),
    }
}

/// Recursively discovers combined golden fixtures without filesystem-order
/// bias.
pub(super) fn collect_golden_fixtures(
    directory: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read directory entry: {error}"))?;
        let path = entry.path();
        if path.is_dir() {
            collect_golden_fixtures(&path, output)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".golden.ir"))
        {
            output.push(path);
        }
    }
    Ok(())
}

/// Parses the library's typed fixture format with path-specific diagnostics.
pub(super) fn parse_golden_fixture(path: &Path, text: &str) -> Result<GoldenFixture, String> {
    GoldenFixture::parse(text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Calls the emitter in-process, keeping success/error expectations enforced.
pub(super) fn execute_golden_fixture(fixture: &GoldenFixture) -> Result<String, String> {
    let actual = Parser::new(&fixture.source)
        .parse_package()
        .map_err(|error| format!("failed to parse block IR: {error}"))
        .and_then(|package| {
            emit_system_verilog(&package, &fixture.options)
                .map(|output| output.system_verilog)
                .map_err(|error| error.to_string())
        });
    match (fixture.expects_error(), actual) {
        (false, Ok(source)) => Ok(source),
        (true, Err(error)) => Ok(format!("{error}\n")),
        (false, Err(error)) => Err(format!("expected successful code generation: {error}")),
        (true, Ok(source)) => Err(format!(
            "expected code generation to fail, but emitted:\n{source}"
        )),
    }
}

/// Compares or updates native golden expectations.
pub(super) fn run_golden_fixture(path: &Path) -> Result<(), String> {
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let fixture = parse_golden_fixture(path, &contents)?;
    let actual = execute_golden_fixture(&fixture)?;
    if std::env::var_os("XLSYNTH_UPDATE_GOLDEN").is_some() {
        fs::write(path, fixture.with_expected_output(&actual)?)
            .map_err(|error| format!("update {}: {error}", path.display()))?;
    } else {
        assert_eq!(
            actual,
            fixture.expected_output(),
            "golden mismatch for {}; run with XLSYNTH_UPDATE_GOLDEN=1 to update",
            path.display()
        );
    }
    Ok(())
}
