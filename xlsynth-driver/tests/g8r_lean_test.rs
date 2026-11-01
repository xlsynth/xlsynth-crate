// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_lean::prepare_equivalence_target;

#[test]
fn failed_proof_json_preserves_version_and_process_diagnostics() {
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
    printf 'proof stdout\n'
    printf 'proof stderr\n' >&2
    exit 1
    ;;
esac
exit 0
"#,
    )
    .unwrap();
    std::fs::set_permissions(&lean, std::fs::Permissions::from_mode(0o755)).unwrap();

    let report_path = directory.path().join("result.json");
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("g8r-lean-check")
        .arg("--target-dir")
        .arg(&target)
        .arg("--proof")
        .arg(&proof)
        .arg("--lean")
        .arg(&lean)
        .arg("--output-json")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(!output.status.success());

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["lean_version"], "Lean test version");
    assert_eq!(report["stdout"], "target stdout\nproof stdout\n");
    assert_eq!(report["stderr"], "target stderr\nproof stderr\n");
    assert!(report["error"].is_string());
}
