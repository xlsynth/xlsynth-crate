// SPDX-License-Identifier: Apache-2.0

//! Prove native BX function lowering against the audit generator's reference
//! IR.

use xlsynth_bx::{ResetBehavior, check_module, empty_package, lower_module, parse_module};
use xlsynth_dslx_fuzzer::PreparedSample;
use xlsynth_dslx_fuzzer::audit_contracts::AuditCase;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult};

use crate::{CheckFailure, classify_with_assertions};

/// Distinguishes completed value checks from inputs that never reached
/// lowering.
#[derive(Debug)]
pub enum BxOutcome {
    Checked(EquivResult),
    FrontendRejected(String),
    OutsideScope,
}

/// Check the original source of a self-contained function. The generator's
/// independent IR remains the oracle even if the DSLX converter shares a bug.
pub fn check_audit(case: AuditCase) -> Result<BxOutcome, CheckFailure> {
    let case = match case {
        AuditCase::Semantic(case)
        | AuditCase::TupleLiteral { case, .. }
        | AuditCase::MixedLiteral { case, .. } => case,
        // Event, negative constexpr, and acceptance-only contracts require
        // observations beyond comparing the value of an emitted function.
        AuditCase::Traces { .. }
        | AuditCase::TraceMessages { .. }
        | AuditCase::ConstexprFailure { .. }
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
    let package = lower_module(&checked, empty_package("sample"), ResetBehavior::default())
        .map_err(|error| {
            CheckFailure::InvalidOutput(format!(
                "BX failed to lower its checked source: {error}\n{source:#?}"
            ))
        })?;
    let ir = package.to_string();
    xlsynth::IrPackage::parse_ir(&ir, None)
        .and_then(|pkg| pkg.get_function("main"))
        .map_err(|error| {
            CheckFailure::InvalidOutput(format!(
                "BX emitted invalid IR or omitted main: {error}\n{source:#?}\n{ir}"
            ))
        })?;
    let prepared = PreparedSample {
        modules: case.modules,
        expected_ir: case.expected_ir,
        converted_ir: ir,
        converted_top: "main".to_owned(),
    };
    classify_with_assertions(&prepared, AssertionSemantics::Same).map(BxOutcome::Checked)
}

#[cfg(all(test, feature = "has-bitwuzla"))]
mod tests {
    use super::*;
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
    fn checks_all_inputs_and_exact_signature() {
        assert!(matches!(
            check_audit(case("fn main(x: u8) -> u8 { x }")),
            Ok(BxOutcome::Checked(EquivResult::Proved))
        ));
        assert!(matches!(
            check_audit(case(
                "fn main(x: u8) -> u8 { if x == u8:231 { u8:0 } else { x } }"
            )),
            Err(CheckFailure::Disproved(_))
        ));
        assert!(matches!(
            check_audit(case("fn main(x: u8) -> (u8,) { (x,) }")),
            Err(CheckFailure::InvalidOutput(_))
        ));
    }

    #[test]
    fn separates_rejections_and_imports_from_completed_value_checks() {
        assert!(matches!(
            check_audit(case("fn main(x: u8) -> u8 { missing }")),
            Ok(BxOutcome::FrontendRejected(_))
        ));
        assert!(matches!(
            check_audit(case("use unavailable; fn main(x: u8) -> u8 { x }")),
            Ok(BxOutcome::OutsideScope)
        ));
        assert!(matches!(
            check_audit(AuditCase::PatternAcceptance { modules: vec![] }),
            Ok(BxOutcome::OutsideScope)
        ));
    }
}
