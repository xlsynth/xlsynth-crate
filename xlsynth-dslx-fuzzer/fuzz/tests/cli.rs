// SPDX-License-Identifier: Apache-2.0

#![cfg(all(unix, feature = "has-bitwuzla"))]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::audit_contracts::AuditCase;
use xlsynth_dslx_fuzzer::converter_contracts::ContractCase;
use xlsynth_dslx_fuzzer::{Grammar, Sample, SourceModule, audit_contracts};
use xlsynth_dslx_fuzzer_fuzz::CheckFailure;
use xlsynth_dslx_fuzzer_fuzz::cli::{check_audit, check_cli, check_contract, classify_cli};
use xlsynth_prover::prover::types::EquivResult;

fn pattern_case() -> AuditCase {
    AuditCase::PatternAcceptance {
        modules: vec![SourceModule {
            name: "sample".into(),
            text: "pub fn main(x: (u8, u8)) -> bool { let ((a), b) = x; true }".into(),
        }],
    }
}

fn write_tool(path: &Path, script: &str) -> std::io::Result<()> {
    std::fs::write(path, format!("#!/bin/sh\n{script}\n"))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}

#[test]
fn parametric_splat_retains_the_inline_source_when_the_converter_stalls()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    write_tool(
        &converter,
        r#"set -eu
grep -Fq '..make(x)' sample.x
echo 'checking inline base' >&2
sleep 4"#,
    )?;
    let case = audit_contracts::generate_v15(&[3])?;
    let result = check_audit(case, &converter, xlsynth::default_dslx_stdlib_path());
    assert!(
        matches!(result, Err(CheckFailure::ConverterRejected(ref message))
            if message.contains("ir_converter_main exceeded 2s")
                && message.contains("checking inline base")
                && message.contains("..make(x)")),
        "{result:?}"
    );
    Ok(())
}

#[test]
fn nested_struct_parametric_requires_a_clean_range_rejection()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let run = |data| -> Result<_, String> {
        let case = audit_contracts::generate_v16(data)?;
        Ok(check_audit(
            case,
            &converter,
            xlsynth::default_dslx_stdlib_path(),
        ))
    };
    let diagnostic = r#"echo '~~~^ TypeInferenceError: size mismatch: s4 vs. s3' >&2
echo 'Error: INVALID_ARGUMENT: TypeInferenceError: size mismatch: s4 vs. s3' >&2"#;
    write_tool(
        &converter,
        &format!(
            r#"if grep -Fq 'Outer<u64:4>' sample.x; then
  {diagnostic}
  exit 1
fi
grep -Fq 'Outer<uN[65]:4>' sample.x || exit 2
echo 'ret K: bits[8] = literal(value=252)'"#
        ),
    )?;
    assert!(matches!(run(&[6, 1])?, Ok(EquivResult::Proved)));
    let result = run(&[6])?;
    assert!(
        matches!(result, Err(CheckFailure::InvalidOutput(ref message))
            if message.contains("Outer<uN[65]:4>") && message.contains("literal(value=252)")),
        "{result:?}"
    );
    for script in [
        format!("{diagnostic}\nkill -ABRT $$"),
        String::from("echo 'Error: INTERNAL: missing parametric binding' >&2; exit 1"),
        String::from("echo 'Error: INVALID_ARGUMENT: ParseError: missing struct' >&2; exit 1"),
    ] {
        write_tool(&converter, &script)?;
        assert!(matches!(
            run(&[6])?,
            Err(CheckFailure::ConverterRejected(_))
        ));
    }
    Ok(())
}

#[test]
fn only_a_clean_pattern_type_error_satisfies_the_frontend_contract()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let typechecker = converter.with_file_name("typecheck_main");
    let run = || {
        check_audit(
            pattern_case(),
            &converter,
            xlsynth::default_dslx_stdlib_path(),
        )
    };
    assert!(matches!(run(), Err(CheckFailure::Checker(_))));
    // The converter is intentionally absent. A genuine type rejection ends
    // the check before conversion, even with independent binding/RHS shapes.
    let diagnostic = r#"echo '~~~~^ TypeInferenceError: Cannot match a 3-element tuple to 2 values.' >&2
echo 'Error: INVALID_ARGUMENT: An error occurred during parsing / typechecking.' >&2"#;
    write_tool(&typechecker, &format!("{diagnostic}\nexit 1"))?;
    assert!(matches!(run(), Ok(EquivResult::Proved)));
    for script in [
        format!("{diagnostic}\nkill -ABRT $$"),
        String::from("echo 'Error: INTERNAL: tuple type missing' >&2; exit 1"),
        String::from(
            "echo '~~~~^ ParseError: missing tuple' >&2; echo 'Error: INVALID_ARGUMENT: An error occurred during parsing / typechecking.' >&2; exit 1",
        ),
    ] {
        write_tool(&typechecker, &script)?;
        let result = run();
        assert!(
            matches!(result, Err(CheckFailure::ConverterRejected(ref message)) if message.contains("typecheck_main failed")),
            "{result:?}"
        );
    }
    Ok(())
}

#[test]
fn self_referential_parametric_retains_a_crash_with_a_source_diagnostic()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let typechecker = converter.with_file_name("typecheck_main");
    let run = |seed| -> Result<_, String> {
        Ok(check_audit(
            audit_contracts::generate_v19(seed)?,
            &converter,
            xlsynth::default_dslx_stdlib_path(),
        ))
    };
    let diagnostic = r#"echo '~~~^ NotConstantError: expr is not constexpr.' >&2
echo 'Error: INVALID_ARGUMENT: An error occurred during parsing / typechecking.' >&2"#;
    write_tool(&typechecker, &format!("{diagnostic}\nexit 1"))?;
    assert!(matches!(run(&[4, 1])?, Ok(EquivResult::Proved)));
    assert!(matches!(
        run(&[4, 2])?,
        Err(CheckFailure::ConverterRejected(_))
    ));
    write_tool(&typechecker, &format!("{diagnostic}\nkill -SEGV $$"))?;
    let result = run(&[4, 4])?;
    assert!(
        matches!(result, Err(CheckFailure::ConverterRejected(ref message))
            if message.contains("typecheck_main failed")
                && message.contains("SIGSEGV")
                && message.contains("Word<N: uN[N]>")),
        "{result:?}"
    );
    Ok(())
}

#[test]
fn accepted_pattern_requires_successful_conversion_to_a_boolean_function()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    write_tool(
        &converter.with_file_name("typecheck_main"),
        "test -s sample.x",
    )?;
    let run = || {
        check_audit(
            pattern_case(),
            &converter,
            xlsynth::default_dslx_stdlib_path(),
        )
    };
    assert!(matches!(run(), Err(CheckFailure::Checker(_))));
    for script in ["echo 'bits vs. tuple' >&2; kill -ABRT $$", "exit 1"] {
        write_tool(&converter, script)?;
        let result = run();
        assert!(
            matches!(result, Err(CheckFailure::ConverterRejected(ref message)) if message.contains("ir_converter_main failed")),
            "{result:?}"
        );
    }
    for output in [
        "broken IR",
        "package sample",
        "package sample\nfn __sample__main() -> bits[8] { ret x: bits[8] = literal(value=1) }",
    ] {
        write_tool(&converter, &format!("cat <<'IR'\n{output}\nIR"))?;
        assert!(matches!(run(), Err(CheckFailure::InvalidOutput(_))));
    }
    write_tool(
        &converter,
        r#"cat <<'IR'
package sample
fn __sample__main(x: (bits[8], bits[8])) -> bits[1] {
  ret t: bits[1] = literal(value=1)
}
IR"#,
    )?;
    assert!(matches!(run(), Ok(EquivResult::Proved)));
    Ok(())
}

#[test]
fn tuple_literal_checks_the_uncasted_cli_and_separate_bytecode_source()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let typechecker = converter.with_file_name("typecheck_main");
    let interpreter = converter.with_file_name("dslx_interpreter_main");
    let run = || {
        let case = audit_contracts::generate_v5(&[3])?;
        check_audit(case, &converter, xlsynth::default_dslx_stdlib_path())
    };
    let original = "pub fn main(x: u8) -> u8 { let (a, _) = (1, x); a }";
    let check_original = format!("set -eu\ntest \"$(cat sample.x)\" = '{original}'");
    write_tool(&typechecker, &format!("{check_original}\ntouch types.pb"))?;
    write_tool(
        &interpreter,
        r#"set -eu
test ! -e types.pb
grep -Fxq 'pub fn main(x: u8) -> u8 { let (a, _) = (1, x); a as u8 }' sample.x
grep -Fxq 'fn literal_runtime() {' sample.x
for flag in --evaluator=dslx-interpreter --compare=none --lower_to_ir=false --test_filter=literal_runtime; do
  printf '%s\n' "$@" | grep -Fxq -- "$flag"
done
echo ran >> "$(dirname "$0")/runtime-runs"
echo '[ RUN UNITTEST  ] literal_runtime' >&2
echo '[            OK ]' >&2"#,
    )?;
    for (width, value, expected) in [(1, 1, 'X'), (8, 0, 'D'), (8, 1, 'P')] {
        write_tool(
            &converter,
            &format!(
                r#"{check_original}
test -f types.pb
cat <<'IR'
package sample
top fn __sample__main(x: bits[8] id=1) -> bits[{width}] {{
  ret result: bits[{width}] = literal(value={value}, id=2)
}}
IR"#
            ),
        )?;
        let result = run();
        assert!(
            matches!(
                (expected, &result),
                ('X', Err(CheckFailure::InvalidOutput(_)))
                    | ('D', Err(CheckFailure::Disproved(_)))
                    | ('P', Ok(EquivResult::Proved))
            ),
            "{result:?}"
        );
    }
    write_tool(&converter, &format!("{check_original}\nkill -ABRT $$"))?;
    assert!(
        matches!(run(), Err(CheckFailure::ConverterRejected(message)) if message.contains("ir_converter_main"))
    );
    assert_eq!(
        std::fs::read_to_string(directory.path().join("runtime-runs"))?,
        "ran\nran\nran\nran\n"
    );
    std::fs::remove_file(converter.with_file_name("dslx_interpreter_main"))?;
    assert!(matches!(run(), Err(CheckFailure::Checker(_))));
    write_tool(&typechecker, "echo 'invalid source' >&2; exit 1")?;
    assert!(
        matches!(run(), Err(CheckFailure::ConverterRejected(message)) if message.contains("typecheck_main"))
    );
    Ok(())
}

#[test]
fn tuple_literal_requires_a_completed_passing_runtime_comparison()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let interpreter = converter.with_file_name("dslx_interpreter_main");
    write_tool(&converter.with_file_name("typecheck_main"), "exit 0")?;
    write_tool(
        &converter,
        r#"cat <<'IR'
package sample
top fn __sample__main(x: bits[8] id=1) -> bits[8] {
  ret result: bits[8] = literal(value=1, id=2)
}
IR"#,
    )?;
    let run = || {
        let case = audit_contracts::generate_v5(&[3, 2])?;
        check_audit(case, &converter, xlsynth::default_dslx_stdlib_path())
    };
    write_tool(
        &interpreter,
        "echo '[===============] 0 test(s) ran; 0 failed; 0 skipped.' >&2",
    )?;
    assert!(matches!(run(), Err(CheckFailure::Checker(_))));
    write_tool(
        &interpreter,
        r#"echo '[ RUN UNITTEST  ] literal_runtime' >&2
echo '~~~~^ FailureError: lhs and rhs were not equal' >&2
echo '[        FAILED ] literal_runtime' >&2
exit 1"#,
    )?;
    assert!(matches!(run(), Err(CheckFailure::Disproved(_))));
    write_tool(&interpreter, "kill -ABRT $$")?;
    assert!(
        matches!(run(), Err(CheckFailure::ConverterRejected(message)) if message.contains("dslx_interpreter_main"))
    );
    Ok(())
}

#[test]
fn mixed_literal_checks_both_bytecode_results_and_preserves_ir_findings()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let tools = directory.path();
    let converter = tools.join("ir_converter_main");
    let original = "pub fn main() -> bool { let (lo, hi) = (0x80, -127); lo < hi }";
    let check_original = format!("set -eu\ntest \"$(cat sample.x)\" = '{original}'");
    write_tool(
        &converter.with_file_name("typecheck_main"),
        &format!("{check_original}\ntouch types.pb"),
    )?;
    write_tool(
        &converter.with_file_name("dslx_interpreter_main"),
        r#"set -eu
test ! -e types.pb
grep -Fxq 'fn literal_runtime() { assert_eq(main(), true); }' sample.x
if grep -Fq '(lo as s8) < (hi as s8)' sample.x; then kind=cast; else kind=raw; fi
tools=$(dirname "$0")
echo "$kind" >> "$tools/runs"
case "$(cat "$tools/$kind")" in
  skip) exit 0 ;;
  fail) echo '[ RUN UNITTEST  ] literal_runtime' >&2
        echo '~~~~^ FailureError: lhs and rhs were not equal' >&2
        exit 1 ;;
  reject) echo '[ RUN UNITTEST  ] literal_runtime' >&2
          echo 'internal error: Same tag is required for a comparison operation' >&2
          exit 1 ;;
  pass) echo '[ RUN UNITTEST  ] literal_runtime' >&2
        echo '[            OK ]' >&2 ;;
esac"#,
    )?;
    for (value, raw, cast, outcome) in [
        ("1", "pass", "pass", "proved"),
        ("1", "fail", "pass", "raw mismatch"),
        ("1", "reject", "pass", "raw rejection"),
        ("1", "pass", "fail", "cast mismatch"),
        ("0", "fail", "pass", "IR mismatch"),
        ("reject", "fail", "pass", "IR rejection"),
        ("0", "fail", "skip", "missing cast control"),
    ] {
        let output = if value == "reject" {
            "exit 1".into()
        } else {
            format!(
                "cat <<'IR'\npackage sample\ntop fn __sample__main() -> bits[1] {{\n  ret t: bits[1] = literal(value={value}, id=1)\n}}\nIR"
            )
        };
        write_tool(
            &converter,
            &format!("{check_original}\ntest -e types.pb\n{output}"),
        )?;
        std::fs::write(tools.join("raw"), raw)?;
        std::fs::write(tools.join("cast"), cast)?;
        std::fs::write(tools.join("runs"), "")?;
        let case = audit_contracts::generate_v6(&[3, 0, 0, 0, 2, 0x7f, 126])?;
        let result = check_audit(case, &converter, xlsynth::default_dslx_stdlib_path());
        let matches = match (outcome, &result) {
            ("proved", Ok(EquivResult::Proved)) => true,
            ("raw mismatch", Err(CheckFailure::Disproved(message))) => {
                message.starts_with("Uncast DSLX runtime comparison failed")
            }
            ("raw rejection", Err(CheckFailure::ConverterRejected(message))) => {
                message.starts_with("dslx_interpreter_main failed")
                    && message.ends_with("Uncast DSLX runtime comparison")
            }
            ("cast mismatch", Err(CheckFailure::Disproved(message))) => {
                message.starts_with("Casted DSLX runtime comparison failed")
            }
            ("IR mismatch", Err(CheckFailure::Disproved(message))) => {
                message.starts_with("equivalence check failed")
            }
            ("IR rejection", Err(CheckFailure::ConverterRejected(message))) => {
                message.starts_with("ir_converter_main failed")
            }
            ("missing cast control", Err(CheckFailure::Checker(_))) => true,
            _ => false,
        };
        assert!(matches, "{outcome}: {result:?}");
        assert_eq!(std::fs::read_to_string(tools.join("runs"))?, "raw\ncast\n");
    }
    Ok(())
}

#[test]
fn self_dimension_can_be_diagnosed_or_checked_at_the_declared_width()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let typechecker = converter.with_file_name("typecheck_main");
    let run = |bytes: &[u8]| {
        check_audit(
            audit_contracts::generate_v10(bytes)?,
            &converter,
            xlsynth::default_dslx_stdlib_path(),
        )
    };
    write_tool(
        &typechecker,
        r#"echo '~~~^ TypeInferenceError: self-referential dimension N' >&2
echo 'Error: INVALID_ARGUMENT: An error occurred during parsing / typechecking.' >&2
exit 1"#,
    )?;
    assert!(matches!(run(&[6]), Ok(EquivResult::Proved)));
    assert!(matches!(
        run(&[6, 1]),
        Err(CheckFailure::ConverterRejected(_))
    ));
    for script in ["kill -SEGV $$", "echo 'Error: INTERNAL: cycle' >&2; exit 1"] {
        write_tool(&typechecker, script)?;
        assert!(matches!(run(&[6]), Err(CheckFailure::ConverterRejected(_))));
    }
    write_tool(&typechecker, "touch types.pb")?;
    write_tool(
        &converter,
        r#"set -eu
test -f types.pb
cat <<'IR'
package sample
top fn __sample__main(x: bits[7] id=1) -> bits[7] {
  ret same: bits[7] = identity(x, id=2)
}
IR"#,
    )?;
    assert!(matches!(run(&[6]), Err(CheckFailure::InvalidOutput(_))));
    Ok(())
}

#[test]
fn folded_lambda_still_requires_a_parseable_helper() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    write_tool(
        &converter,
        r#"cat <<'IR'
package sample
top fn __sample__main() -> bits[8] {
  ret result: bits[8] = literal(value=2, id=1)
}
fn __call__xN__dim_bool:0__dim_8(x: bits[8] id=2) -> bits[8] {
  ret same: bits[8] = identity(x, id=3)
}
IR"#,
    )?;
    let verdict = check_audit(
        audit_contracts::generate_v10(&[7])?,
        &converter,
        xlsynth::default_dslx_stdlib_path(),
    );
    assert!(
        matches!(verdict, Err(CheckFailure::InvalidOutput(ref message))
        if message.contains("C++ IR parser rejected converter output")),
        "{verdict:?}"
    );
    Ok(())
}

#[test]
fn optimized_contract_retains_optimizer_failures_and_proves_its_output()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    let optimizer = converter.with_file_name("opt_main");
    write_tool(
        &converter,
        r#"cat <<'IR'
package sample
top fn __sample__main() -> bits[1] {
  a: bits[8][0] = array(id=1)
  ret result: bits[1] = eq(a, a, id=2)
}
IR"#,
    )?;
    let require_input = r#"set -eu
test "$1" = --top=__sample__main
grep -Fq 'bits[8][0] = array(' "$2""#;
    let run = || {
        check_audit(
            audit_contracts::generate_v11(&[6])?,
            &converter,
            xlsynth::default_dslx_stdlib_path(),
        )
    };
    write_tool(
        &optimizer,
        &format!(
            "{require_input}\necho 'Empty array Values are not supported.' >&2\nkill -ABRT $$"
        ),
    )?;
    let verdict = run();
    assert!(
        matches!(verdict, Err(CheckFailure::ConverterRejected(ref message))
            if message.starts_with("opt_main failed")
                && message.contains("Empty array Values are not supported.")
                && message.contains("bits[8][0] = array(")),
        "{verdict:?}"
    );
    for value in [0, 1] {
        write_tool(
            &optimizer,
            &format!(
                "{require_input}\ncat <<'IR'\npackage sample\ntop fn __sample__main() -> bits[1] {{\n  ret result: bits[1] = literal(value={value}, id=1)\n}}\nIR"
            ),
        )?;
        let verdict = run();
        assert!(
            matches!(
                (value, &verdict),
                (0, Err(CheckFailure::Disproved(_))) | (1, Ok(EquivResult::Proved))
            ),
            "{verdict:?}"
        );
    }
    Ok(())
}

#[test]
fn signed_bound_call_reports_a_bytecode_disagreement_when_the_ir_is_correct()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let converter = directory.path().join("ir_converter_main");
    write_tool(
        &converter.with_file_name("typecheck_main"),
        "touch types.pb",
    )?;
    write_tool(
        &converter,
        r#"set -eu
test -f types.pb
cat <<'IR'
package sample
top fn __sample__main() -> bits[32] {
  ret x: bits[32] = literal(value=0xfffffff5, id=1)
}
IR"#,
    )?;
    write_tool(
        &converter.with_file_name("dslx_interpreter_main"),
        r#"set -eu
grep -Fq 'let x = 0x15; widen<5>(x)' sample.x
echo '[ RUN UNITTEST  ] literal_runtime' >&2
echo '~~~^ FailureError: 21 vs 4294967285' >&2
exit 1"#,
    )?;
    let outcome = check_audit(
        audit_contracts::generate_v9(&[7])?,
        &converter,
        xlsynth::default_dslx_stdlib_path(),
    );
    assert!(
        matches!(outcome, Err(CheckFailure::Disproved(ref message))
        if message.starts_with("Original DSLX runtime comparison failed")),
        "{outcome:?}"
    );
    Ok(())
}

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
