// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::{aig::GateFn, aig_serdes::gate2ir};
use xlsynth_pir::ir;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice};

pub fn check_equivalence(orig_package: &str, gate_package: &str) -> Result<(), String> {
    check_equivalence_with_top_and_solver(
        orig_package,
        gate_package,
        None,
        SolverChoice::Bitwuzla,
        None,
    )
}

/// Checks package equivalence using an explicitly selected solver backend.
pub fn check_equivalence_with_top_and_solver(
    orig_package: &str,
    gate_package: &str,
    top_fn_name: Option<&str>,
    solver: SolverChoice,
    tool_path: Option<&Path>,
) -> Result<(), String> {
    let prover = prover_for_choice(solver, tool_path);
    match prover.prove_ir_pkg_text_equiv(orig_package, gate_package, top_fn_name) {
        EquivResult::Proved => Ok(()),
        EquivResult::Inconclusive(msg) => Err(format!("inconclusive: {msg}")),
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => Err(format!(
            "not equivalent: lhs_inputs={lhs_inputs:?} rhs_inputs={rhs_inputs:?} \
             lhs_output={lhs_output:?} rhs_output={rhs_output:?}"
        )),
        EquivResult::ToolchainDisproved(msg) | EquivResult::Error(msg) => Err(msg),
    }
}

/// Checks package equivalence with the explicitly selected XLS toolchain
/// oracle.
pub fn check_equivalence_with_top_via_toolchain(
    orig_package: &str,
    gate_package: &str,
    top_fn_name: Option<&str>,
) -> Result<(), String> {
    check_equivalence_with_top_and_solver(
        orig_package,
        gate_package,
        top_fn_name,
        SolverChoice::Toolchain,
        None,
    )
}

/// Checks package equivalence with the explicitly selected XLS toolchain
/// oracle.
pub fn check_equivalence_via_toolchain(
    orig_package: &str,
    gate_package: &str,
) -> Result<(), String> {
    check_equivalence_with_top_via_toolchain(orig_package, gate_package, None)
}

fn get_fn_signature(f: &ir::Fn) -> String {
    let mut signature = String::new();
    signature.push_str("fn ");
    signature.push_str(&f.name);
    signature.push_str("(");
    for (i, param) in f.params.iter().enumerate() {
        signature.push_str(&format!("{}: {}", param.name, param.ty));
        if i + 1 != f.params.len() {
            signature.push_str(", ");
        }
    }
    signature.push_str(")");
    if !f.ret_ty.is_nil() {
        signature.push_str(&format!(" -> {}", f.ret_ty));
    }
    signature
}

pub fn validate_same_signature(orig_fn: &ir::Fn, gate_fn: &GateFn) -> Result<(), String> {
    let gate_signature = gate_fn.get_signature();
    let orig_signature = get_fn_signature(orig_fn);
    if orig_signature != gate_signature {
        return Err(format!(
            "signature mismatch: original fn: `{}` != gate fn: `{}`",
            orig_signature, gate_signature
        ));
    }
    Ok(())
}

/// Note: if the original IR function has a different signature than the gate
/// function (because gate functions generally have flattened signatures into a
/// bit vector for each parameter / result tuple element) then we adjust in the
/// conversion from "gate IR" to "XLS IR" to use the original function's
/// signature so we can check XLS IR equivalence directly.
pub fn validate_same_fn(orig_fn: &ir::Fn, gate_fn: &GateFn) -> Result<(), String> {
    validate_same_fn_with_solver(orig_fn, gate_fn, SolverChoice::Bitwuzla, None)
}

/// Checks a PIR function and a GateFn using an explicitly selected solver.
pub fn validate_same_fn_with_solver(
    orig_fn: &ir::Fn,
    gate_fn: &GateFn,
    solver: SolverChoice,
    tool_path: Option<&Path>,
) -> Result<(), String> {
    let orig_ir_fn_text: String = orig_fn.to_string();
    let xlsynth_package_ir: String =
        gate2ir::gate_fn_to_xlsynth_ir(gate_fn, "gate", &orig_fn.get_type())
            .unwrap()
            .to_string();
    log::info!("xlsynth_package_ir:\n{}", xlsynth_package_ir);
    let orig_ir_pkg_text: String = format!("package orig\n\ntop {}", orig_ir_fn_text);
    check_equivalence_with_top_and_solver(
        &orig_ir_pkg_text,
        &xlsynth_package_ir,
        None,
        solver,
        tool_path,
    )
}

/// Checks a PIR function and GateFn with the explicitly selected XLS oracle.
pub fn validate_same_fn_via_toolchain(orig_fn: &ir::Fn, gate_fn: &GateFn) -> Result<(), String> {
    validate_same_fn_with_solver(orig_fn, gate_fn, SolverChoice::Toolchain, None)
}

/// Structured result for IR-level equivalence checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrCheckResult {
    /// The two GateFns were proven equivalent.
    Equivalent,
    /// A concrete counter-example was found – they are *not* equivalent.
    NotEquivalent,
    /// The external solver timed out or was interrupted (SIGINT, etc.).
    TimedOutOrInterrupted,
    /// Some other infrastructure failure (I/O, missing tool etc.). Payload is
    /// explanatory message.
    OtherProcessError(String),
}

impl IrCheckResult {
    pub fn is_equivalent(&self) -> bool {
        matches!(self, IrCheckResult::Equivalent)
    }
}

fn map_equiv_result(result: EquivResult) -> IrCheckResult {
    match result {
        EquivResult::Proved => IrCheckResult::Equivalent,
        EquivResult::Inconclusive(_) => IrCheckResult::TimedOutOrInterrupted,
        EquivResult::Disproved { .. } | EquivResult::ToolchainDisproved(_) => {
            IrCheckResult::NotEquivalent
        }
        EquivResult::Error(msg) => IrCheckResult::OtherProcessError(msg),
    }
}

/// Structured variant of `prove_same_gate_fn_via_ir`.
pub fn prove_same_gate_fn_via_ir_status(lhs: &GateFn, rhs: &GateFn) -> IrCheckResult {
    prove_same_gate_fn_via_ir_status_with_solver(lhs, rhs, SolverChoice::Bitwuzla, None)
}

/// Checks two GateFns using an explicitly selected IR solver backend.
pub fn prove_same_gate_fn_via_ir_status_with_solver(
    lhs: &GateFn,
    rhs: &GateFn,
    solver: SolverChoice,
    tool_path: Option<&Path>,
) -> IrCheckResult {
    let lhs_type = lhs.get_flat_type();
    let rhs_type = rhs.get_flat_type();
    if lhs_type != rhs_type {
        return IrCheckResult::OtherProcessError("type mismatch".to_string());
    }
    let lhs_ir: xlsynth::IrPackage = gate2ir::gate_fn_to_xlsynth_ir(lhs, "lhs", &lhs_type).unwrap();
    let rhs_ir: xlsynth::IrPackage = gate2ir::gate_fn_to_xlsynth_ir(rhs, "rhs", &rhs_type).unwrap();

    let prover = prover_for_choice(solver, tool_path);
    map_equiv_result(prover.prove_ir_pkg_text_equiv(&lhs_ir.to_string(), &rhs_ir.to_string(), None))
}

/// Checks two GateFns with the explicitly selected XLS toolchain oracle.
pub fn prove_same_gate_fn_via_ir_status_via_toolchain(lhs: &GateFn, rhs: &GateFn) -> IrCheckResult {
    prove_same_gate_fn_via_ir_status_with_solver(lhs, rhs, SolverChoice::Toolchain, None)
}

/// Backwards-compatibility wrapper that preserves the original
/// Result<(),String> behaviour used elsewhere in the codebase.
pub fn prove_same_gate_fn_via_ir(lhs: &GateFn, rhs: &GateFn) -> Result<(), String> {
    match prove_same_gate_fn_via_ir_status(lhs, rhs) {
        IrCheckResult::Equivalent => Ok(()),
        IrCheckResult::NotEquivalent => Err("not equivalent".to_string()),
        IrCheckResult::TimedOutOrInterrupted => Err("TimedOutOrInterrupted".to_string()),
        IrCheckResult::OtherProcessError(msg) => Err(msg),
    }
}

/// Checks two GateFns with the explicitly selected XLS toolchain oracle.
pub fn prove_same_gate_fn_via_ir_via_toolchain(lhs: &GateFn, rhs: &GateFn) -> Result<(), String> {
    match prove_same_gate_fn_via_ir_status_via_toolchain(lhs, rhs) {
        IrCheckResult::Equivalent => Ok(()),
        IrCheckResult::NotEquivalent => Err("not equivalent".to_string()),
        IrCheckResult::TimedOutOrInterrupted => Err("TimedOutOrInterrupted".to_string()),
        IrCheckResult::OtherProcessError(msg) => Err(msg),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        aig::gate::AigBitVector,
        gate_builder::{GateBuilder, GateBuilderOptions},
    };
    use xlsynth_pir::ir_parser;

    use super::*;

    #[test]
    fn test_validate_same_signature_simple_one_bit() {
        let simple_xor_ir = "package simple_xor
 top fn my_xor(a: bits[1], b: bits[1]) -> bits[1] {
     ret xor.3: bits[1] = xor(a, b, id=3)
 }
";
        let mut parser = ir_parser::Parser::new(simple_xor_ir);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_top = ir_package.get_top_fn().unwrap();

        // Now we make a simple one bit gate fn.
        let mut gate_builder = GateBuilder::new("my_xor".to_string(), GateBuilderOptions::opt());
        let a = gate_builder
            .add_input("a".to_string(), 1)
            .get_lsb(0)
            .clone();
        let b = gate_builder
            .add_input("b".to_string(), 1)
            .get_lsb(0)
            .clone();
        let xor = gate_builder.add_xor_binary(a, b);
        gate_builder.add_output("output_value".to_string(), AigBitVector::from_bit(xor));
        let gate_fn = gate_builder.build();

        validate_same_signature(&ir_top, &gate_fn).unwrap();
    }
}
