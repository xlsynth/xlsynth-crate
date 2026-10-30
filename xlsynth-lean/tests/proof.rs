// SPDX-License-Identifier: Apache-2.0

mod common;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use xlsynth_g8r::aig::{AigBitVector, AigOperand, AigRef, GateFn};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_lean::{
    InterfaceDirection, LeanError, LeanProofSources, TESTED_LEAN_VERSION, check_equivalence_proof,
    prepare_equivalence_target, run_equivalence_proof,
};

/// Creates `a && (b && c)` with the same interface as `common::and3`.
fn right_associated_and3() -> GateFn {
    let mut builder = GateBuilder::new("and3_right".to_string(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
    let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
    let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
    let bc = builder.add_and_binary(b, c);
    let abc = builder.add_and_binary(a, bc);
    builder.add_output("result".to_string(), AigBitVector::from_bit(abc));
    builder.build()
}

/// Returns the explicitly configured Lean executable for external tests.
fn lean_executable() -> Option<PathBuf> {
    std::env::var_os("XLSYNTH_LEAN").map(PathBuf::from)
}

/// Returns the checked-in proof source for the two three-input AND graphs.
fn proof_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/and3_equivalent.lean")
}

#[test]
fn prepared_target_uses_aag_files_and_the_fixed_root_contract() {
    let directory = tempfile::tempdir().unwrap();
    let target_root = directory.path().join("target");
    prepare_equivalence_target(&common::and3(), &right_associated_and3())
        .unwrap()
        .write_to(&target_root)
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(target_root.join("Target.lean")).unwrap(),
        include_str!("goldens/target.lean")
    );
    assert_eq!(
        std::fs::read_to_string(target_root.join("Check.lean")).unwrap(),
        include_str!("goldens/check.lean")
    );
    assert_eq!(
        std::fs::read_to_string(target_root.join("Lhs.aag")).unwrap(),
        include_str!("goldens/and3.aag")
    );
    assert!(target_root.join("Rhs.aag").is_file());
    assert!(!target_root.join("Generated").exists());
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn aig_decide_proves_ordered_outputs_and_replays_an_inequivalent_input() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/aig_decide");
    let sources = LeanProofSources {
        body: root.join("Body.lean"),
        modules: vec![root.join("EdgeCases.lean")],
        root,
    };
    let mut lhs = common::and3();
    let mut rhs = right_associated_and3();
    for graph in [&mut lhs, &mut rhs] {
        let and = *graph.outputs[0].bit_vector.get_lsb(0);
        graph.outputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(&[and.negate(), and]);
    }
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("target");
    prepare_equivalence_target(&lhs, &rhs)
        .unwrap()
        .write_to(&target)
        .unwrap();
    run_equivalence_proof(&lean, &target, &sources, &directory.path().join("work")).unwrap();

    // Only 111 distinguishes output bit 1 from a stuck-at-zero RHS.
    let low_bit = *rhs.outputs[0].bit_vector.get_lsb(0);
    let zero = AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    };
    rhs.outputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(&[low_bit, zero]);
    let error = check_equivalence_proof(&lean, &lhs, &rhs, &sources.body).unwrap_err();
    assert!(
        error.stdout.contains(
            "aig_decide found a counterexample:\ninput atoms set to true: [0, 1, 2] (all others false)\noutput bit 1: lhs=true, rhs=false"
        ),
        "unexpected proof failure: {error}"
    );
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn decomposed_proof_closes_the_generated_multi_output_target() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decomposition");
    let sources = LeanProofSources {
        body: root.join("Body.lean"),
        modules: [
            "Partitioned",
            "RewriteExamples",
            "DomainExamples",
            "CutExamples",
            "SparseExamples",
        ]
        .map(|name| root.join(format!("{name}.lean")))
        .to_vec(),
        root,
    };
    let mut lhs = common::and3();
    let mut rhs = right_associated_and3();
    for graph in [&mut lhs, &mut rhs] {
        let result = *graph.outputs[0].bit_vector.get_lsb(0);
        let bypass = *graph.inputs[0].bit_vector.get_lsb(0);
        graph.outputs[0].bit_vector =
            AigBitVector::from_lsb_is_index_0(&[result, result.negate(), bypass]);
    }
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("target");
    prepare_equivalence_target(&lhs, &rhs)
        .unwrap()
        .write_to(&target)
        .unwrap();
    run_equivalence_proof(&lean, &target, &sources, &directory.path().join("work")).unwrap();
}

#[cfg(unix)]
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn aig_decide_rejects_an_invalid_unsat_certificate() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let directory = tempfile::tempdir().unwrap();
    let solver = directory.path().join("false-unsat");
    // Syntactically valid LRAT claiming the empty clause follows from clause 1
    // alone.
    std::fs::write(
        &solver,
        "#!/bin/sh\nprintf '1000 0 1 0\\n' > \"$2\"\nprintf 's UNSATISFIABLE\\n'\nexit 20\n",
    )
    .unwrap();
    std::fs::set_permissions(&solver, std::fs::Permissions::from_mode(0o755)).unwrap();
    let body = directory.path().join("Body.lean");
    let solver_literal = serde_json::to_string(solver.to_str().unwrap()).unwrap();
    std::fs::write(
        &body,
        format!("by\n  apply XlsynthLean.Proof.equivalent_of_rootValid\n  set_option sat.solver {solver_literal} in aig_decide\n"),
    )
    .unwrap();
    let lhs = common::and3();
    let mut rhs = right_associated_and3();
    rhs.outputs[0].bit_vector =
        AigBitVector::from_bit(rhs.outputs[0].bit_vector.get_lsb(0).negate());
    let error = check_equivalence_proof(&lean, &lhs, &rhs, &body).unwrap_err();
    assert!(
        error
            .stdout
            .contains("LRAT certificate failed verification"),
        "unexpected proof failure: {error}"
    );
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn prepared_target_accepts_helpers_and_bv_decide() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let directory = tempfile::tempdir().unwrap();
    let target_root = directory.path().join("target");
    prepare_equivalence_target(&common::and3(), &right_associated_and3())
        .unwrap()
        .write_to(&target_root)
        .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/multifile");
    let sources = LeanProofSources {
        root: root.clone(),
        body: root.join("Body.lean"),
        // Deliberately list the dependent module first.
        modules: vec![
            root.join("Helpers/Proof.lean"),
            root.join("Helpers/Lemmas.lean"),
        ],
    };
    run_equivalence_proof(
        &lean,
        &target_root,
        &sources,
        &directory.path().join("work"),
    )
    .unwrap();
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn equivalent_graphs_succeed_and_return_lean_diagnostics() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let directory = tempfile::tempdir().unwrap();
    let proof_file = directory.path().join("and3_with_diagnostic.lean");
    let proof_source = include_str!("fixtures/and3_equivalent.lean").replacen(
        "by\n",
        "by\n  trace \"proof-diagnostic-marker\"\n",
        1,
    );
    std::fs::write(&proof_file, proof_source).unwrap();

    let output = check_equivalence_proof(
        &lean,
        &common::and3(),
        &right_associated_and3(),
        &proof_file,
    )
    .unwrap();
    assert!(
        output.lean_version.contains(TESTED_LEAN_VERSION),
        "unexpected Lean version: {}",
        output.lean_version
    );
    assert!(
        output.stdout.contains("proof-diagnostic-marker"),
        "stdout:\n{}\nstderr:\n{}",
        output.stdout,
        output.stderr
    );
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn changed_rhs_rejects_the_same_proof() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let mut rhs = right_associated_and3();
    let result = rhs.outputs[0].bit_vector.get_lsb(0).negate();
    rhs.outputs[0].bit_vector = AigBitVector::from_bit(result);

    let error =
        check_equivalence_proof(&lean, &common::and3(), &rhs, &proof_fixture()).unwrap_err();
    assert!(
        matches!(
            error.error.as_ref(),
            LeanError::ProcessFailed {
                operation: "user proof compilation",
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn empty_or_wrong_type_proof_body_is_rejected() {
    let lean = lean_executable().expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let directory = tempfile::tempdir().unwrap();
    let cases = [("empty", ""), ("wrong_type", "True.intro")];
    for (name, source) in cases {
        let proof_file = directory.path().join(format!("{name}.lean"));
        std::fs::write(&proof_file, source).unwrap();
        let error = check_equivalence_proof(
            &lean,
            &common::and3(),
            &right_associated_and3(),
            &proof_file,
        )
        .unwrap_err();
        assert!(
            matches!(error.error.as_ref(), LeanError::ProcessFailed { .. }),
            "{name}: unexpected error: {error:?}"
        );
    }
}

#[test]
fn interface_mismatches_return_structured_errors_before_invoking_lean() {
    let lhs = common::and3();
    let check = |rhs: &GateFn| prepare_equivalence_target(&lhs, rhs);

    let error = check(&common::interface_graph()).unwrap_err();
    assert!(matches!(
        error,
        LeanError::InterfacePortCountMismatch {
            direction: InterfaceDirection::Input,
            lhs: 3,
            rhs: 4,
        }
    ));

    let mut rhs = right_associated_and3();
    rhs.outputs.clear();
    let error = check(&rhs).unwrap_err();
    assert!(matches!(
        error,
        LeanError::InterfacePortCountMismatch {
            direction: InterfaceDirection::Output,
            lhs: 1,
            rhs: 0,
        }
    ));

    let mut rhs = right_associated_and3();
    rhs.inputs[1].name = "renamed".to_string();
    let error = check(&rhs).unwrap_err();
    assert!(matches!(
        error,
        LeanError::InterfacePortNameMismatch {
            direction: InterfaceDirection::Input,
            port_index: 1,
            lhs,
            rhs,
        } if lhs == "b" && rhs == "renamed"
    ));

    let mut rhs = right_associated_and3();
    rhs.outputs[0].name = "renamed".to_string();
    let error = check(&rhs).unwrap_err();
    assert!(matches!(
        error,
        LeanError::InterfacePortNameMismatch {
            direction: InterfaceDirection::Output,
            port_index: 0,
            lhs,
            rhs,
        } if lhs == "result" && rhs == "renamed"
    ));

    let mut rhs = right_associated_and3();
    let a = *rhs.inputs[0].bit_vector.get_lsb(0);
    let b = *rhs.inputs[1].bit_vector.get_lsb(0);
    rhs.inputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(&[a, b]);
    rhs.inputs[1].bit_vector = AigBitVector::from_lsb_is_index_0(&[]);
    let error = check(&rhs).unwrap_err();
    assert!(matches!(
        error,
        LeanError::InterfacePortWidthMismatch {
            direction: InterfaceDirection::Input,
            port_index: 0,
            lhs: 1,
            rhs: 2,
        }
    ));

    let mut rhs = right_associated_and3();
    let result = *rhs.outputs[0].bit_vector.get_lsb(0);
    rhs.outputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(&[result, result.negate()]);
    let error = check(&rhs).unwrap_err();
    assert!(matches!(
        error,
        LeanError::InterfacePortWidthMismatch {
            direction: InterfaceDirection::Output,
            port_index: 0,
            lhs: 1,
            rhs: 2,
        }
    ));
}
