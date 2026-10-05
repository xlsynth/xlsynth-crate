// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn priority_result_flag_defaults_off_and_rejects_external_amount_sharing() {
    for (source, activates) in [
        (
            include_str!("../../xlsynth-g8r/tests/testdata/priority_result/historical-priority.ir"),
            true,
        ),
        (
            include_str!("../../xlsynth-g8r/tests/testdata/priority_result/shared-amount63.ir"),
            false,
        ),
    ] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), source).unwrap();
        let prepare = |flag: Option<bool>| {
            let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
            command
                .arg("ir-prep-for-gates")
                .arg(file.path())
                .arg("--enable-formal-array-alias-analysis=false");
            if let Some(enabled) = flag {
                command.arg(format!("--experimental-priority-result={enabled}"));
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            output.stdout
        };
        let normal = prepare(None);
        assert_eq!(normal, prepare(Some(false)));
        assert_eq!(normal != prepare(Some(true)), activates);
    }
}
