// SPDX-License-Identifier: Apache-2.0

//! Prove native BX function lowering against the generator and distinguish
//! discrepancies that are shared with native DSLX.

use std::path::Path;
use xlsynth_bx::{ResetBehavior, check_module, empty_package, lower_module, parse_module};
use xlsynth_dslx_fuzzer::audit_contracts::AuditCase;
use xlsynth_dslx_fuzzer::{PreparedSample, SourceModule};
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult};

use crate::{CheckFailure, classify_with_assertions, cli};

/// Distinguishes reference proofs, discrepancies shared with DSLX, and
/// inputs that never reached lowering.
#[derive(Debug)]
pub enum BxOutcome {
    Checked(EquivResult),
    SharedWithDslx,
    FrontendRejected(String),
    OutsideScope,
}

/// Check the original source of a self-contained function. Only report a BX
/// discrepancy if the standalone DSLX converter provides distinct behavior.
pub fn check_audit(
    case: AuditCase,
    converter: &Path,
    stdlib: &Path,
) -> Result<BxOutcome, CheckFailure> {
    let case = match case {
        AuditCase::Semantic(case)
        | AuditCase::TupleLiteral { case, .. }
        | AuditCase::BoundCall { case, .. }
        | AuditCase::DimensionScope { case, .. }
        | AuditCase::MixedLiteral { case, .. } => case,
        // These contracts observe behavior outside the emitted function's value.
        AuditCase::Optimized(_)
        | AuditCase::Traces { .. }
        | AuditCase::TraceMessages { .. }
        | AuditCase::ConstexprFailure { .. }
        | AuditCase::LiteralRuntime { .. }
        | AuditCase::PatternAcceptance { .. } => return Ok(BxOutcome::OutsideScope),
    };
    // Restrict this target to a single native BX source. Linking generated
    // DSLX modules would also exercise the native DSLX converter.
    let [source] = case.modules.as_slice() else {
        return Ok(BxOutcome::OutsideScope);
    };
    // The shared grammar includes syntax outside BX. Keep the diagnostic and
    // count the rejection separately from programs whose values were proved.
    let module = match parse_module(&source.text) {
        Ok(module) => module,
        Err(message) => return Ok(BxOutcome::FrontendRejected(message)),
    };
    // Resolve no external dependencies. In particular, BX can call into DSLX
    // when checking an imported function, before lower_module runs.
    if !module.uses.is_empty() || !module.externs.is_empty() || !module.extern_blocks.is_empty() {
        return Ok(BxOutcome::OutsideScope);
    }
    // The source parser and typechecker report unsupported programs normally;
    // errors after this point concern an accepted BX program.
    let checked = match check_module(&module) {
        Ok(checked) => checked,
        Err(message) => return Ok(BxOutcome::FrontendRejected(message)),
    };
    let ir = match lower_module(&checked, empty_package("sample"), ResetBehavior::default())
        .and_then(|package| validate_top(package.to_string(), "main"))
    {
        Ok(ir) => ir,
        Err(error) => {
            let failure = CheckFailure::InvalidOutput(format!(
                "BX failed to emit a valid main for its checked source: {error}\n{source:#?}"
            ));
            return compare_with_dslx(case.modules, None, failure, converter, stdlib);
        }
    };
    let prepared = PreparedSample {
        modules: case.modules,
        expected_ir: case.expected_ir,
        converted_ir: ir,
        converted_top: "main".to_owned(),
    };
    match classify_with_assertions(&prepared, AssertionSemantics::Same) {
        Err(failure @ (CheckFailure::Disproved(_) | CheckFailure::InvalidOutput(_))) => {
            compare_with_dslx(
                prepared.modules,
                Some(prepared.converted_ir),
                failure,
                converter,
                stdlib,
            )
        }
        result => result.map(BxOutcome::Checked),
    }
}

/// Require the production IR parser to accept the package and resolve its entry
/// point.
fn validate_top(ir: String, name: &str) -> Result<String, String> {
    xlsynth::IrPackage::parse_ir(&ir, None)
        .and_then(|package| package.get_function(name))
        .map_err(|error| format!("Invalid IR or missing {name}: {error}\n{ir}"))?;
    Ok(ir)
}

/// Compare the two native lowerings on every input after BX has failed its
/// reference check.
fn compare_with_dslx(
    modules: Vec<SourceModule>,
    bx_ir: Option<String>,
    reference_failure: CheckFailure,
    converter: &Path,
    stdlib: &Path,
) -> Result<BxOutcome, CheckFailure> {
    if !stdlib.is_dir() {
        return Err(CheckFailure::Checker(format!(
            "DSLX stdlib directory is unavailable: {}",
            stdlib.display()
        )));
    }
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let output = cli::run_converter(&modules, converter, stdlib, directory.path(), None, true)?;
    // A DSLX rejection or crash leaves no second function to compare with BX.
    if !output.status.success() {
        return Ok(BxOutcome::Checked(EquivResult::Inconclusive(format!(
            "DSLX did not produce IR ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ))));
    }
    let dslx_top =
        xlsynth::mangle_dslx_name("sample", "main").map_err(|error| error.to_string())?;
    // Malformed DSLX output also leaves no function whose behavior can be
    // established.
    let dslx_ir = match String::from_utf8(output.stdout)
        .map_err(|error| error.to_string())
        .and_then(|ir| validate_top(ir, &dslx_top))
    {
        Ok(ir) => ir,
        Err(error) => return Ok(BxOutcome::Checked(EquivResult::Inconclusive(error))),
    };
    // If BX could not emit valid IR, DSLX's valid main establishes a lowering
    // discrepancy without a pair of functions available for a value proof.
    let Some(bx_ir) = bx_ir else {
        return Err(CheckFailure::InvalidOutput(format!(
            "Compiler disagreement: DSLX produced a valid {dslx_top}, but BX did not.\nDSLX IR:\n{dslx_ir}\nBX lowering diagnostic:\n{}",
            reference_failure.into_message()
        )));
    };
    let native = PreparedSample {
        modules,
        expected_ir: bx_ir,
        converted_ir: dslx_ir,
        converted_top: dslx_top,
    };
    let diagnostic = |difference| {
        format!(
            "Compiler disagreement between BX (lhs) and DSLX (rhs):\n{difference}\nIndependent reference (lhs) versus BX (rhs):\n{}",
            reference_failure.into_message()
        )
    };
    match classify_with_assertions(&native, AssertionSemantics::Same) {
        Ok(EquivResult::Proved) => Ok(BxOutcome::SharedWithDslx),
        Err(CheckFailure::Disproved(message)) => Err(CheckFailure::Disproved(diagnostic(message))),
        Err(CheckFailure::InvalidOutput(message)) => {
            Err(CheckFailure::InvalidOutput(diagnostic(message)))
        }
        result => result.map(BxOutcome::Checked),
    }
}

#[cfg(all(test, feature = "has-bitwuzla"))]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use xlsynth_dslx_fuzzer::SourceModule;
    use xlsynth_dslx_fuzzer::converter_contracts::ContractCase;

    const IDENTITY: &str = r#"package ref_pkg
top fn main(x: bits[8] id=1) -> bits[8] {
  ret result: bits[8] = identity(x, id=2)
}
"#;

    fn case(source: &str) -> AuditCase {
        AuditCase::Semantic(ContractCase {
            family: "test",
            modules: vec![SourceModule {
                name: "sample".into(),
                text: source.into(),
            }],
            expected_ir: IDENTITY.into(),
            max_and_nodes: None,
        })
    }

    #[test]
    fn proves_correct_bx_without_running_dslx() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        assert!(matches!(
            check_audit(
                case("fn main(x: u8) -> u8 { x }"),
                &dir.path().join("absent"),
                xlsynth::default_dslx_stdlib_path(),
            ),
            Ok(BxOutcome::Checked(EquivResult::Proved))
        ));
        Ok(())
    }

    #[cfg(unix)]
    fn write_converter(path: &Path, script: &str) -> std::io::Result<()> {
        std::fs::write(path, format!("#!/bin/sh\n{script}\n"))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
    }

    #[cfg(unix)]
    #[test]
    fn suppresses_only_all_input_agreement_on_values_and_signature() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let converter = dir.path().join("ir_converter_main");
        write_converter(
            &converter,
            r#"set -eu
fixture=$(dirname "$0")
cmp sample.x "$fixture/expected.x"
cat "$fixture/result.ir""#,
        )?;
        let bx_zero = "fn main(x: u8) -> u8 { u8:0 }";
        let bx_tuple = "fn main(x: u8) -> (u8,) { (x,) }";
        let dslx_zero = "bits[8] { ret z: bits[8] = literal(value=0, id=2) }";
        let dslx_tuple = "(bits[8]) { ret t: (bits[8]) = tuple(x, id=2) }";
        // With BX returning zero, both disagree with the identity oracle on
        // x=1. Their outputs differ only on x=231.
        let rare_difference = r#"bits[8] {
  z: bits[8] = literal(value=0, id=2)
  special: bits[8] = literal(value=231, id=3)
  hit: bits[1] = eq(x, special, id=4)
  ret result: bits[8] = sel(hit, cases=[z, x], id=5)
}"#;
        let dslx_fails_with_zero = r#"bits[8] {
  tok: token = after_all(id=2)
  never: bits[1] = literal(value=0, id=3)
  failed: token = assert(tok, never, message="failed", label="check", id=4)
  ret z: bits[8] = literal(value=0, id=5)
}"#;
        let dslx_zero_tuple = r#"(bits[8]) {
  z: bits[8] = literal(value=0, id=2)
  ret t: (bits[8]) = tuple(z, id=3)
}"#;
        for (source, body, expected) in [
            (bx_zero, dslx_zero, "shared"),
            (bx_zero, rare_difference, "disproved"),
            (bx_zero, dslx_fails_with_zero, "disproved"),
            (bx_zero, dslx_tuple, "different signatures"),
            (bx_tuple, dslx_tuple, "shared"),
            (bx_tuple, dslx_zero_tuple, "disproved"),
            (
                bx_tuple,
                "bits[8] { ret r: bits[8] = identity(x, id=2) }",
                "different signatures",
            ),
        ] {
            std::fs::write(dir.path().join("expected.x"), source)?;
            let ir = format!("package sample\ntop fn __sample__main(x: bits[8] id=1) -> {body}\n");
            std::fs::write(dir.path().join("result.ir"), ir)?;
            let result = check_audit(
                case(source),
                &converter,
                xlsynth::default_dslx_stdlib_path(),
            );
            assert!(
                matches!(
                    (expected, &result),
                    ("shared", Ok(BxOutcome::SharedWithDslx))
                        | ("disproved", Err(CheckFailure::Disproved(_)))
                        | ("different signatures", Err(CheckFailure::InvalidOutput(_)))
                ),
                "{source}\n{body}\nexpected {expected}, got {result:?}"
            );
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn missing_dslx_evidence_cannot_classify_a_bx_disproof() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let converter = dir.path().join("ir_converter_main");
        let compare = || {
            check_audit(
                case("fn main(x: u8) -> u8 { u8:0 }"),
                &converter,
                xlsynth::default_dslx_stdlib_path(),
            )
        };
        assert!(matches!(compare(), Err(CheckFailure::Checker(_))));
        for script in ["exit 1", "echo malformed"] {
            write_converter(&converter, script)?;
            let result = compare();
            assert!(
                matches!(result, Ok(BxOutcome::Checked(EquivResult::Inconclusive(_)))),
                "{result:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn separates_rejections_and_imports_from_completed_value_checks() {
        let check = |case| {
            check_audit(
                case,
                Path::new("absent"),
                xlsynth::default_dslx_stdlib_path(),
            )
        };
        assert!(matches!(
            check(case("fn main(x: u8) -> u8 { missing }")),
            Ok(BxOutcome::FrontendRejected(_))
        ));
        assert!(matches!(
            check(case("use unavailable; fn main(x: u8) -> u8 { x }")),
            Ok(BxOutcome::OutsideScope)
        ));
        assert!(matches!(
            check(AuditCase::PatternAcceptance { modules: vec![] }),
            Ok(BxOutcome::OutsideScope)
        ));
    }
}
