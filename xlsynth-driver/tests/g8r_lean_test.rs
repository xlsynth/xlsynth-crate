// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_lean::prepare_equivalence_target;

#[test]
fn proof_json_preserves_diagnostics_and_the_named_verifiers_audit_count() {
    let directory = tempfile::tempdir().unwrap();
    let mut builder = GateBuilder::new("identity".to_string(), GateBuilderOptions::no_opt());
    let input = builder.add_input("a".to_string(), 1);
    builder.add_output("result".to_string(), input);
    let graph = builder.build();
    let target = directory.path().join("target");
    prepare_equivalence_target(&graph, &graph)
        .unwrap()
        .write_to(&target)
        .unwrap();
    let proof = directory.path().join("Body.lean");
    std::fs::write(&proof, "by rfl\n").unwrap();

    let lean = directory.path().join("lean");
    std::fs::write(
        &lean,
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  printf 'Lean test version\n'
  exit 0
fi
if [ "$1" = "--src-deps" ]; then
  exit 0
fi
if [ "$1" = "--run" ]; then
  [ "$4" = named ] && [ "$5" = Proof.equivalent ] || exit 1
  printf '3\n' > native-checks-replayed.json
  printf 'xlsynth: verified UserProof.equivalent\n'
  exit 0
fi
lean_source=
for lean_argument do
  lean_source=$lean_argument
done
case "$lean_source" in
  */Target.lean)
    printf 'target stdout\n'
    printf 'target stderr\n' >&2
    ;;
  */UserProof.lean)
    if [ -n "$MOCK_LEAN_FAIL_COMPILATION" ]; then
      printf 'proof stdout\n'
      printf 'proof stderr\n' >&2
      exit 1
    fi
    ;;
esac
: > "$4" # Stand in for the compiled module at the path following -o.
exit 0
"#,
    )
    .unwrap();
    std::fs::set_permissions(&lean, std::fs::Permissions::from_mode(0o755)).unwrap();

    let report_path = directory.path().join("result.json");
    let check = |proof| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command
            .arg("g8r-lean-check")
            .arg("--target-dir")
            .arg(&target)
            .arg("--proof")
            .arg(proof)
            .arg("--lean")
            .arg(&lean)
            .arg("--output-json")
            .arg(&report_path);
        command
    };
    let output = check(&proof)
        .env("MOCK_LEAN_FAIL_COMPILATION", "1")
        .output()
        .unwrap();
    assert!(!output.status.success());

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["audit_performed"], false);
    assert!(report.get("native_checks_replayed").is_none());
    assert_eq!(report["lean_version"], "Lean test version");
    assert_eq!(report["stdout"], "target stdout\nproof stdout\n");
    assert_eq!(report["stderr"], "target stderr\nproof stderr\n");
    assert!(report["error"].is_string());

    let module = directory.path().join("Proof.lean");
    std::fs::write(
        &module,
        "import Target\ntheorem Proof.equivalent : Target.Equivalent := by intro inputs; rfl\n",
    )
    .unwrap();
    let output = check(&module)
        .args(["--proof-theorem", "Proof.equivalent"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["status"], "accepted_by_lean");
    assert_eq!(report["audit_performed"], true);
    assert_eq!(report["native_checks_replayed"], 3);
}
