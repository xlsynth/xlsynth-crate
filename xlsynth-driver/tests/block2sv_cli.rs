// SPDX-License-Identifier: Apache-2.0

//! CLI-boundary checks; backend goldens and semantic tests live in codegen.

use std::path::Path;
use std::process::{Command, Output};

use xlsynth::vast_helpers_options::CodegenOptions;
// Share test support as source, without adding a crate dependency back-edge.
#[allow(dead_code)]
#[path = "../../xlsynth-codegen/tests/support/golden_fixture.rs"]
mod golden_fixture;
use golden_fixture::{GoldenFixture, default_golden_fixture_root};

/// Loads a shared codegen fixture without depending on another crate's tests.
fn fixture(relative: &str) -> GoldenFixture {
    let text = std::fs::read_to_string(default_golden_fixture_root().join(relative)).unwrap();
    GoldenFixture::parse(&text).unwrap()
}

/// Executes the real CLI with a known input and caller-controlled arguments.
fn invoke(directory: &Path, arguments: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .arg("block2sv")
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn defaults_match_the_library_and_stdout_has_a_final_newline() {
    let fixture = fixture("arrays/dynamic_index_checked.golden.ir");
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("input.ir"), &fixture.source).unwrap();
    let output = invoke(directory.path(), &["input.ir".into()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        fixture.expected_output()
    );
}

#[test]
fn underscore_and_hyphen_flags_forward_typed_codegen_options() {
    for relative in [
        "structure/explicit_top_override.golden.ir",
        "structure/module_name_override.golden.ir",
        "structure/separate_lines.golden.ir",
        "structure/systemverilog_types_disabled.golden.ir",
        "debug/assertion_disabled.golden.ir",
        "arrays/ignore_custom_typed_array.golden.ir",
        "registers/two_stage_pipeline.golden.ir",
        "registers/custom_reset_and_enable_template.golden.ir",
    ] {
        let fixture = fixture(relative);
        let options = &fixture.options;
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("input.ir"), &fixture.source).unwrap();
        let mut flags = vec![
            (
                "layout",
                if options.layout == xlsynth_codegen::Layout::Pipeline {
                    "pipeline"
                } else {
                    "none"
                }
                .to_owned(),
            ),
            (
                "array_index_bounds_checking",
                options.array_index_bounds_checking.to_string(),
            ),
            ("separate_lines", options.separate_lines.to_string()),
            ("max_inline_depth", options.max_inline_depth.to_string()),
            ("emit_sv_types", options.emit_sv_types.to_string()),
            (
                "add_invariant_assertions",
                options.add_invariant_assertions.to_string(),
            ),
        ];
        if let Some(top) = &options.top {
            flags.push(("top", top.clone()));
        }
        if let Some(name) = &options.module_name {
            flags.push(("module_name", name.clone()));
        }
        if let Some(registers) = &options.register_codegen_options {
            std::fs::write(
                directory.path().join("registers.toml"),
                toml::to_string(registers).unwrap(),
            )
            .unwrap();
            flags.push(("register_codegen_options", "registers.toml".into()));
        }
        for hyphenated in [false, true] {
            let mut arguments = flags
                .iter()
                .map(|(name, value)| {
                    let name = if hyphenated {
                        name.replace('_', "-")
                    } else {
                        name.to_string()
                    };
                    format!("--{name}={value}")
                })
                .collect::<Vec<_>>();
            arguments.push("input.ir".into());
            let output = invoke(directory.path(), &arguments);
            assert!(
                output.status.success(),
                "{relative}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                fixture.expected_output(),
                "{relative}, hyphenated={hyphenated}"
            );
        }
    }
}

#[test]
fn backend_error_has_cli_prefix_nonzero_exit_and_empty_stdout() {
    let fixture = fixture("errors/missing_top.golden.ir");
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("input.ir"), &fixture.source).unwrap();
    let output = invoke(
        directory.path(),
        &["input.ir".into(), "--top=missing".into()],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        format!("error: {}", fixture.expected_output())
    );
}

#[test]
fn invalid_flags_fail_before_reading_the_input() {
    let directory = tempfile::tempdir().unwrap();
    for flag in [
        "--layout=invalid",
        "--separate-lines=not-a-boolean",
        "--max-inline-depth=-1",
        "--unknown-flag=true",
    ] {
        let output = invoke(directory.path(), &["missing.ir".into(), flag.into()]);
        assert_eq!(output.status.code(), Some(2), "{flag}");
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.starts_with("error:"), "{stderr}");
        assert!(!stderr.contains("failed to read block IR"), "{stderr}");
    }
}

#[test]
fn missing_input_and_configuration_files_report_their_paths() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = fixture("registers/simple_register.golden.ir");
    std::fs::write(directory.path().join("input.ir"), fixture.source).unwrap();
    for (arguments, path, description) in [
        (vec!["missing.ir".into()], "missing.ir", "block IR file"),
        (
            vec![
                "input.ir".into(),
                "--register-codegen-options=missing.toml".into(),
            ],
            "missing.toml",
            "register codegen options",
        ),
    ] {
        let io_error = std::fs::read_to_string(directory.path().join(path)).unwrap_err();
        let output = invoke(directory.path(), &arguments);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            format!("error: failed to read {description} '{path}': {io_error}\n")
        );
    }
}

#[test]
fn malformed_register_configuration_reports_cli_context() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = fixture("registers/simple_register.golden.ir");
    std::fs::write(directory.path().join("input.ir"), fixture.source).unwrap();
    let config = r#"reg_template = "always_ff @(posedge {{clock}}) {{reg}} <= input_data;""#;
    std::fs::write(directory.path().join("registers.toml"), config).unwrap();
    let parse_error = toml::from_str::<CodegenOptions>(config).unwrap_err();
    let output = invoke(
        directory.path(),
        &[
            "input.ir".into(),
            "--register_codegen_options=registers.toml".into(),
        ],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        format!(
            "error: failed to parse register codegen options 'registers.toml': {parse_error}\n"
        )
    );
}
