// SPDX-License-Identifier: Apache-2.0

#![cfg(all(unix, feature = "has-bitwuzla"))]

use std::os::unix::fs::PermissionsExt;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::converter_contracts::ContractCase;
use xlsynth_dslx_fuzzer::{Grammar, Sample, SourceModule, audit_contracts};
use xlsynth_dslx_fuzzer_fuzz::CheckFailure;
use xlsynth_dslx_fuzzer_fuzz::cli::{check_audit, check_cli, check_contract, classify_cli};
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn standalone_output_is_the_program_under_proof() -> Result<(), Box<dyn std::error::Error>> {
    let mut lane = Body::new(BitsType {
        signed: false,
        width: Width::Concrete(8),
    })?;
    lane.push(Fragment::Binary {
        op: BinaryOp::Add,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })?;
    let sample = Sample::new(vec![], vec![lane])?;
    let dir = tempfile::tempdir()?;
    let converter = dir.path().join("converter");
    // This executable substitutes subtraction for the generated DSLX addition.
    std::fs::write(
        &converter,
        r#"#!/bin/sh
cat <<'IR'
package sample

top fn __sample__main(x: bits[8] id=1, y: bits[8] id=2) -> (bits[8]) {
  difference: bits[8] = sub(x, y, id=3)
  ret result: (bits[8]) = tuple(difference, id=4)
}
IR
"#,
    )?;
    std::fs::set_permissions(&converter, std::fs::Permissions::from_mode(0o755))?;
    let error = check_cli(
        &sample,
        Grammar::Mission,
        &converter,
        xlsynth::default_dslx_stdlib_path(),
        None,
        None,
    )
    .expect_err("wrong standalone output must fail even when the DSO is correct");
    assert!(error.contains("Disproved"), "{error}");
    let classify = || {
        classify_cli(
            &sample,
            Grammar::Mission,
            &converter,
            xlsynth::default_dslx_stdlib_path(),
            None,
            None,
        )
    };
    assert!(matches!(classify(), Err(CheckFailure::Disproved(_))));
    // Both alternatives are valid IR. A wrong input width prevents an SMT
    // comparison, whereas flattening hides the missing result tuple.
    for wrong_top in [
        r#"top fn __sample__main(x: bits[2] id=1, y: bits[8] id=2) -> (bits[8]) {
  ret result: (bits[8]) = tuple(y, id=3)
}"#,
        r#"top fn __sample__main(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret result: bits[8] = add(x, y, id=3)
}"#,
    ] {
        std::fs::write(
            &converter,
            format!("#!/bin/sh\ncat <<'IR'\npackage sample\n\n{wrong_top}\nIR\n"),
        )?;
        assert!(matches!(classify(), Err(CheckFailure::InvalidOutput(_))));
    }
    std::fs::write(&converter, "#!/bin/sh\nexit 1\n")?;
    assert!(matches!(
        classify(),
        Err(CheckFailure::ConverterRejected(_))
    ));
    std::fs::remove_file(&converter)?;
    assert!(matches!(classify(), Err(CheckFailure::Checker(_))));
    Ok(())
}

#[test]
fn failing_constexpr_requires_a_semantic_rejection() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let converter = dir.path().join("converter");
    let stdlib = xlsynth::default_dslx_stdlib_path();
    for (script, expected) in [
        ("exit 0", "accepted"),
        (
            "echo 'No constexpr value found for node (StatementBlock)' >&2; exit 1",
            "proved",
        ),
        ("echo 'bad DSLX syntax' >&2; exit 1", "rejected"),
        (
            "echo 'No constexpr value found for node (StatementBlock)' >&2; kill -TERM $$",
            "rejected",
        ),
    ] {
        std::fs::write(&converter, format!("#!/bin/sh\n{script}\n"))?;
        std::fs::set_permissions(&converter, std::fs::Permissions::from_mode(0o755))?;
        let result = check_audit(audit_contracts::generate(&[3, 0, 0])?, &converter, stdlib);
        match expected {
            "accepted" => assert!(matches!(result, Err(CheckFailure::InvalidOutput(_)))),
            "proved" => assert!(matches!(result, Ok(EquivResult::Proved))),
            _ => assert!(matches!(result, Err(CheckFailure::ConverterRejected(_)))),
        }
    }
    Ok(())
}

#[test]
fn correct_trace_counts_do_not_mask_wrong_text_on_either_input()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let converter = dir.path().join("converter");
    // The generated s2 inputs are -1 and 1, rendered as 3 and 1 with {:u}.
    for (when_false, when_true, correct) in [(3, 1, true), (0, 1, false), (3, 0, false)] {
        std::fs::write(
            &converter,
            format!(
                r#"#!/bin/sh
cat <<'IR'
package sample
top fn __sample__main(x: bits[1] id=1) -> bits[1] {{
  token0: token = after_all(id=2)
  enabled: bits[1] = literal(value=1, id=3)
  low: bits[2] = literal(value={when_false}, id=4)
  high: bits[2] = literal(value={when_true}, id=5)
  n: bits[2] = sel(x, cases=[low, high], id=6)
  event: token = trace(token0, enabled, format="VALUE {{:u}}", data_operands=[n], id=7)
  ret x: bits[1] = param(name=x, id=1)
}}
IR
"#
            ),
        )?;
        std::fs::set_permissions(&converter, std::fs::Permissions::from_mode(0o755))?;
        let case = audit_contracts::generate_v2(&[5, 2])?;
        let result = check_audit(case, &converter, xlsynth::default_dslx_stdlib_path());
        if correct {
            assert!(matches!(result, Ok(EquivResult::Proved)), "{result:?}");
        } else {
            assert!(
                matches!(result, Err(CheckFailure::Disproved(_))),
                "{result:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn secondary_dso_failure_preserves_cli_signature_finding() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let converter = dir.path().join("converter");
    std::fs::write(
        &converter,
        r#"#!/bin/sh
cat <<'IR'
package sample
top fn __sample__main(x: bits[8] id=1) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}
IR
"#,
    )?;
    std::fs::set_permissions(&converter, std::fs::Permissions::from_mode(0o755))?;
    let case = ContractCase {
        family: "test",
        // The substituted CLI emits valid IR, but the DSO cannot parse the source.
        modules: vec![SourceModule {
            name: "sample".into(),
            text: "pub fn".into(),
        }],
        expected_ir: r#"package reference
top fn main(x: bits[1] id=1) -> bits[1] {
  ret x: bits[1] = param(name=x, id=1)
}
"#
        .into(),
        max_and_nodes: None,
    };
    assert!(matches!(
        check_contract(case, &converter, xlsynth::default_dslx_stdlib_path()),
        Err(CheckFailure::InvalidOutput(_))
    ));
    Ok(())
}
