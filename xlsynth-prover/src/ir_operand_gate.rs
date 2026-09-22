// SPDX-License-Identifier: Apache-2.0

//! Solver-backed proof of conditional clamps at exact IR operand sites.

use xlsynth_pir::ir::{self, MemberType, PackageMember};
use xlsynth_pir::ir_operand_gate::{
    OperandGateSite, gate_operands_in_package, predicate_is_false_property_in_package,
};
use xlsynth_pir::ir_parser::Parser;

use crate::prover::types::{
    AssertionSemantics, BoolPropertyResult, EquivParallelism, EquivResult, ProverFn,
    QuickCheckAssertionSemantics,
};
use crate::prover::{SolverChoice, SolverLimits, prover_for_choice_with_limits};

/// Whether the original predicate can be true for some input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateReachability {
    Reachable,
    Unreachable,
    Unknown,
}

/// Results for a single simultaneous operand-gate proof.
#[derive(Debug)]
pub struct OperandGateProof {
    pub function: String,
    pub result: EquivResult,
    pub predicate_reachability: PredicateReachability,
}

/// Chooses a function without silently taking the first of several candidates.
pub fn select_function<'a>(
    package: &'a ir::Package,
    top: Option<&str>,
) -> Result<&'a ir::Fn, String> {
    if let Some(top) = top {
        return package
            .get_fn(top)
            .ok_or_else(|| format!("function {top:?} not found in package"));
    }
    if let Some((name, kind)) = &package.top {
        return match kind {
            MemberType::Function => package
                .get_fn(name)
                .ok_or_else(|| format!("top function {name:?} not found")),
            MemberType::Block => {
                Err("package top is a block; select a function with --top".to_string())
            }
        };
    }
    let mut functions = package.members.iter().filter_map(|member| match member {
        PackageMember::Function(function) => Some(function),
        PackageMember::Block(_) => None,
    });
    match (functions.next(), functions.next()) {
        (Some(function), None) => Ok(function),
        (None, _) => Err("package has no functions".to_string()),
        (Some(_), Some(_)) => {
            Err("package has multiple functions; select one with --top".to_string())
        }
    }
}

/// Checks whether simultaneous conditional operand clamps preserve every return
/// bit.
pub fn prove_operand_gate(
    source: &str,
    top: Option<&str>,
    when: &str,
    sites: &[OperandGateSite],
    limits: SolverLimits,
) -> Result<OperandGateProof, String> {
    let package = Parser::new(source)
        .parse_and_validate_package()
        .map_err(|error| format!("invalid IR package: {error}"))?;
    let original = select_function(&package, top)?;
    let transformed = gate_operands_in_package(original, &package, when, sites)?;
    let predicate_property = predicate_is_false_property_in_package(original, &package, when)?;
    let prover = prover_for_choice_with_limits(SolverChoice::Bitwuzla, None, limits);
    let reachability = prover.prove_ir_quickcheck(
        &ProverFn::new(&predicate_property, Some(&package)),
        QuickCheckAssertionSemantics::Ignore,
        None,
    );
    let predicate_reachability = match reachability {
        BoolPropertyResult::Proved => PredicateReachability::Unreachable,
        BoolPropertyResult::Disproved { .. } => PredicateReachability::Reachable,
        BoolPropertyResult::Inconclusive(_) => PredicateReachability::Unknown,
        BoolPropertyResult::Error(message) | BoolPropertyResult::ToolchainDisproved(message) => {
            return Err(format!("predicate reachability check failed: {message}"));
        }
    };
    let result = prover.prove_ir_equiv(
        &ProverFn::new(original, Some(&package)),
        &ProverFn::new(&transformed.function, Some(&package)),
        EquivParallelism::SingleThreaded,
        AssertionSemantics::Ignore,
        None,
        false,
    );
    match result {
        EquivResult::Error(message) | EquivResult::ToolchainDisproved(message) => {
            Err(format!("operand gate proof failed: {message}"))
        }
        result => Ok(OperandGateProof {
            function: original.name.clone(),
            result,
            predicate_reachability,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::select_function;
    #[cfg(feature = "has-bitwuzla")]
    use super::{PredicateReachability, prove_operand_gate};
    #[cfg(feature = "has-bitwuzla")]
    use crate::prover::SolverLimits;
    #[cfg(feature = "has-bitwuzla")]
    use crate::prover::types::EquivResult;
    #[cfg(feature = "has-bitwuzla")]
    use xlsynth_pir::ir_operand_gate::OperandGateSite;
    use xlsynth_pir::ir_parser::Parser;

    #[cfg(feature = "has-bitwuzla")]
    fn site(operand: usize) -> OperandGateSite {
        OperandGateSite {
            consumer: "difference".to_string(),
            operand,
            start: None,
            width: None,
            clamp: "0".to_string(),
        }
    }

    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn proves_simultaneous_clamps_and_disproves_individual_clamp() {
        let source = r#"package gated

top fn main(x: bits[8] id=1, p: bits[1] id=2) -> bits[8] {
  ret difference: bits[8] = xor(x, x, id=3)
}"#;
        let combined = prove_operand_gate(
            source,
            None,
            "p",
            &[site(0), site(1)],
            SolverLimits::default(),
        )
        .unwrap();
        assert_eq!(
            combined.predicate_reachability,
            PredicateReachability::Reachable
        );
        assert!(matches!(combined.result, EquivResult::Proved));
        let individual =
            prove_operand_gate(source, None, "p", &[site(0)], SolverLimits::default()).unwrap();
        assert!(matches!(individual.result, EquivResult::Disproved { .. }));
    }

    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn disproves_clamp_with_model_and_flags_vacuous_proof() {
        let source = r#"package gated

top fn main(x: bits[8] id=1, p: bits[1] id=2) -> bits[8] {
  ret difference: bits[8] = add(x, x, id=3)
}"#;
        let proof =
            prove_operand_gate(source, None, "p", &[site(0)], SolverLimits::default()).unwrap();
        match proof.result {
            EquivResult::Disproved {
                lhs_inputs,
                lhs_output,
                rhs_output,
                ..
            } => {
                assert_eq!(lhs_inputs[1].name, "p");
                assert!(lhs_inputs[1].value.bits_equals_u64_value(1));
                assert_ne!(lhs_output.value, rhs_output.value);
            }
            other => panic!("expected a counterexample, got {other:?}"),
        }
        let source = r#"package gated

top fn main(x: bits[8] id=1) -> bits[8] {
  never: bits[1] = literal(value=0, id=2)
  ret difference: bits[8] = add(x, x, id=3)
}"#;
        let proof =
            prove_operand_gate(source, None, "never", &[site(0)], SolverLimits::default()).unwrap();
        assert!(matches!(proof.result, EquivResult::Proved));
        assert_eq!(
            proof.predicate_reachability,
            PredicateReachability::Unreachable
        );
    }

    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn proves_package_function_containing_an_invoke() {
        let source = r#"package gated

fn helper(x: bits[8] id=1) -> bits[8] {
  ret inverted: bits[8] = not(x, id=2)
}

top fn main(x: bits[8] id=3, p: bits[1] id=4) -> bits[8] {
  call: bits[8] = invoke(x, to_apply=helper, id=5)
  ret difference: bits[8] = xor(call, call, id=6)
}"#;
        let site = OperandGateSite {
            consumer: "call".to_string(),
            operand: 0,
            start: None,
            width: None,
            clamp: "0".to_string(),
        };
        let proof =
            prove_operand_gate(source, None, "p", &[site], SolverLimits::default()).unwrap();
        assert!(matches!(proof.result, EquivResult::Proved));
    }

    #[test]
    fn selects_only_unambiguous_function() {
        let source = r#"package gated

fn main(x: bits[1] id=1) -> bits[1] { ret neg_x: bits[1] = not(x, id=2) }
fn helper(y: bits[1] id=3) -> bits[1] { ret y2: bits[1] = not(y, id=4) }
"#;
        let package = Parser::new(source).parse_and_validate_package().unwrap();
        assert!(
            select_function(&package, None)
                .unwrap_err()
                .contains("multiple")
        );
        assert_eq!(
            select_function(&package, Some("helper")).unwrap().name,
            "helper"
        );
    }
}
