// SPDX-License-Identifier: Apache-2.0

#![cfg(all(unix, feature = "has-bitwuzla"))]

use std::os::unix::fs::PermissionsExt;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::{Grammar, Sample};
use xlsynth_dslx_fuzzer_fuzz::CheckFailure;
use xlsynth_dslx_fuzzer_fuzz::cli::{check_cli, classify_cli};

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
