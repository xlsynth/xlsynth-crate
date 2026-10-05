// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_pir::ir_parser::Parser;

#[test]
fn aug_opt_priority_fusion_defaults_off_and_preserves_external_amount_sharing() {
    for (source, activates) in [
        (
            include_str!(
                "../../xlsynth-aug-opt/tests/testdata/priority_result/historical-priority.ir"
            ),
            true,
        ),
        (
            include_str!("../../xlsynth-aug-opt/tests/testdata/priority_result/shared-amount63.ir"),
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
            assert_eq!(normal, optimize(Some(false)));
            assert_eq!(normal != optimize(Some(true)), activates);
        }
    }
}
