// SPDX-License-Identifier: Apache-2.0

//! Driver-facing checks for portable ABC choice-generation orchestration.

#[cfg(unix)]
mod unix_tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::process::{Command, Output};

    use tempfile::TempDir;
    use xlsynth_g8r::aig_serdes::load_abc_choice_aiger::load_abc_choice_aiger_auto_from_path;

    const TWO_INPUT_AND: &str = "aag 3 2 0 1 1\n2\n4\n6\n6 2 4\n";

    /// Creates an executable fixture that copies the requested input AIGER.
    fn write_fake_abc(path: &Path) {
        fs::write(
            path,
            r#"#!/bin/sh
set -eu
script="$2"
input=$(sed -n 's/^read_aiger "\(.*\)"$/\1/p' "$script")
output=$(sed -n 's/^&write -s "\(.*\)"$/\1/p' "$script")
cp "$script" "$0.script"
cp "$input" "$output"
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    /// Runs the CLI against an isolated synthetic AIG, Liberty, and ABC.
    fn run_command(temporary: &TempDir, additional: &[&str]) -> Output {
        let input = temporary.path().join("input.aig");
        let output = temporary.path().join("output.aig");
        let liberty = temporary.path().join("cells.lib");
        let executable = temporary.path().join("fake-abc");
        fs::write(&input, TWO_INPUT_AND).unwrap();
        fs::write(&liberty, "library (test) {}\n").unwrap();
        write_fake_abc(&executable);

        Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-optimize-choices")
            .arg(&input)
            .args(["--abc", executable.to_str().unwrap()])
            .args(["--liberty", liberty.to_str().unwrap()])
            .args(["--aiger-out", output.to_str().unwrap()])
            .args(additional)
            .output()
            .unwrap()
    }

    #[test]
    fn exports_validated_aig_and_reports_complete_interface() {
        let temporary = TempDir::new().unwrap();
        let output = run_command(&temporary, &["--rounds", "2"]);
        assert!(output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-optimize-choices: inputs=2, latches=0, outputs=1, ANDs=1 -> 1, choices=0\n"
        );
        let choices = load_abc_choice_aiger_auto_from_path(&temporary.path().join("output.aig"))
            .expect("the fake ABC output remains a valid complete-interface AIG");
        assert_eq!(choices.graph().inputs.len(), 2);
        assert_eq!(choices.graph().outputs.len(), 1);
        let captured = fs::read_to_string(temporary.path().join("fake-abc.script"))
            .expect("capture the exact ABC commands issued by the driver");
        let synthesis = captured
            .lines()
            .filter(|line| *line == "&resyn3" || *line == "&syn2")
            .collect::<Vec<_>>();
        assert_eq!(synthesis, vec!["&resyn3", "&resyn3"]);
    }

    #[test]
    fn rejects_multiline_abc_commands_before_invocation() {
        let temporary = TempDir::new().unwrap();
        let output = run_command(&temporary, &["--syn-command", "&syn2\n&nf"]);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-optimize-choices error: ABC choice optimization failed: ABC optimization commands must fit on one line\n"
        );
        assert!(!temporary.path().join("output.aig").exists());
    }
}
