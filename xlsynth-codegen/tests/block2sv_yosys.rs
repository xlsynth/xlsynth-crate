// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "support/block2sv_goldens.rs"]
mod fixtures;
use fixtures::*;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use xlsynth::external_tool::{resolve_executable, run_checked};

/// Parses every successful generated fixture as synthesis-mode SystemVerilog.
#[test]
fn block2sv_generated_fixtures_parse_with_yosys() {
    let yosys = std::env::var_os("XLSYNTH_YOSYS_PATH").unwrap_or_else(|| "yosys".into());
    let yosys = resolve_executable(Path::new(&yosys)).expect("resolve required Yosys executable");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let mut paths = Vec::new();
    collect_golden_fixtures(&directory, &mut paths).unwrap();
    paths.sort();

    let temporary_directory = tempfile::tempdir().expect("create Yosys input directory");
    let verilog_path = temporary_directory.path().join("generated.sv");
    let support_types =
        "package types; typedef logic [7:0] input_t; typedef logic [7:0] output_t; endpackage\n";
    let mut failures = Vec::new();
    for path in paths {
        let contents = fs::read_to_string(&path).unwrap();
        let fixture = parse_golden_fixture(&path, &contents).unwrap();
        if fixture.expects_error() {
            continue;
        }
        let output = match execute_golden_fixture(&fixture) {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        fs::write(&verilog_path, format!("{support_types}{output}"))
            .expect("write generated SystemVerilog");
        if let Err(error) = run_checked(
            Command::new(&yosys)
                .current_dir(temporary_directory.path())
                .args(["-q", "-p", "read_verilog -sv -DSYNTHESIS generated.sv"]),
            temporary_directory.path(),
            "yosys-fixture",
            Duration::from_secs(120),
        ) {
            failures.push(format!("{}: {error}", path.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} generated SystemVerilog fixtures failed to parse:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
