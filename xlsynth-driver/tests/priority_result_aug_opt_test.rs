// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_pir::ir_parser::Parser;

#[test]
fn aug_opt_priority_fusion_defaults_on_and_preserves_external_amount_sharing() {
    for (source, activates) in [
        (
            include_str!(
                "../../xlsynth-aug-opt/tests/fixtures/mffc_regressions/priority_historical.ir"
            ),
            true,
        ),
        (
            include_str!(
                "../../xlsynth-aug-opt/tests/fixtures/mffc_regressions/priority_shared_amount63.ir"
            ),
            false,
        ),
    ] {
        let package = Parser::new(source).parse_and_validate_package().unwrap();
        let top = &package.get_top_fn().unwrap().name;
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), source).unwrap();
        for rounds in [1, 3] {
            let optimize = |flag: Option<bool>| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
                command
                    .arg("ir2opt")
                    .arg(file.path())
                    .arg("--top")
                    .arg(top)
                    .arg("--aug-opt=true")
                    .arg(format!("--aug-opt-rounds={rounds}"));
                if let Some(enabled) = flag {
                    command.arg(format!("--aug-opt-fuse-priority-results={enabled}"));
                }
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                output.stdout
            };
            let normal = optimize(None);
            assert_eq!(normal, optimize(Some(true)));
            assert_eq!(normal != optimize(Some(false)), activates);
        }
    }
}

/// Covers each command's default separately from its explicit ablation flag.
#[test]
fn dslx_and_codegen_commands_default_priority_fusion_on() {
    let dir = tempfile::tempdir().unwrap();
    let ir_path = dir.path().join("priority.ir");
    std::fs::write(
        &ir_path,
        include_str!("../../xlsynth-aug-opt/tests/fixtures/mffc_regressions/priority_u8.ir"),
    )
    .unwrap();
    let dslx_path = dir.path().join("priority.x");
    std::fs::write(
        &dslx_path,
        "import std;\npub fn main(x: u32) -> u32 { std::next_pow2(x) }\n",
    )
    .unwrap();
    let tool_path = std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS must be set");
    let toolchain = dir.path().join("toolchain.toml");
    std::fs::write(
        &toolchain,
        format!("[toolchain]\ntool_path = {tool_path:?}\n"),
    )
    .unwrap();

    for subcommand in ["dslx2ir", "ir2combo", "ir2pipeline"] {
        for rounds in [1, 3] {
            let run = |fusion: Option<bool>| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
                if subcommand != "dslx2ir" {
                    command.arg("--toolchain").arg(&toolchain);
                }
                command.arg(subcommand);
                if subcommand == "dslx2ir" {
                    command
                        .arg("--dslx_input_file")
                        .arg(&dslx_path)
                        .arg("--dslx_top=main");
                } else {
                    command
                        .arg(&ir_path)
                        .args(["--top=main", "--delay_model=unit"]);
                }
                if subcommand == "ir2pipeline" {
                    command.arg("--pipeline_stages=1");
                }
                command
                    .args(["--opt=true", "--aug-opt=true"])
                    .arg(format!("--aug-opt-rounds={rounds}"));
                if let Some(fusion) = fusion {
                    command.arg(format!("--aug-opt-fuse-priority-results={fusion}"));
                }
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "{subcommand}, rounds={rounds}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                output.stdout
            };
            let normal = run(None);
            assert_eq!(normal, run(Some(true)), "{subcommand}, rounds={rounds}");
            assert_ne!(normal, run(Some(false)), "{subcommand}, rounds={rounds}");
        }
    }
}
