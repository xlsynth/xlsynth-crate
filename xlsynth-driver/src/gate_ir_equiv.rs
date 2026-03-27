// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;
use xlsynth_prover::prover::SolverChoice;

use crate::ir_equiv::{dispatch_ir_equiv, EquivOutcome, IrEquivRequest, IrModule};

/// Proves equivalence between two gate functions by lifting both sides to IR
/// and then dispatching through the normal IR equivalence flow.
pub fn prove_gate_fns_equiv_via_ir(
    lhs: &GateFn,
    rhs: &GateFn,
    solver: Option<SolverChoice>,
    tool_path: Option<&Path>,
    subcommand: &str,
) -> Result<EquivOutcome, String> {
    let lhs_type = lhs.get_flat_type();
    let rhs_type = rhs.get_flat_type();
    if lhs_type != rhs_type {
        return Err(format!(
            "gate function signatures do not match\nlhs: {}\nrhs: {}",
            lhs.get_signature(),
            rhs.get_signature()
        ));
    }

    let lhs_ir = gate_fn_to_xlsynth_ir(lhs, "lhs", &lhs_type)
        .map_err(|e| format!("failed to convert LHS GateFn to XLS IR: {}", e))?;
    let rhs_ir = gate_fn_to_xlsynth_ir(rhs, "rhs", &rhs_type)
        .map_err(|e| format!("failed to convert RHS GateFn to XLS IR: {}", e))?;
    let lhs_ir_text = lhs_ir.to_string();
    let rhs_ir_text = rhs_ir.to_string();
    let lhs_top = lhs.name.clone();
    let rhs_top = rhs.name.clone();

    let request = IrEquivRequest::new(
        IrModule::new(&lhs_ir_text).with_top(Some(lhs_top.as_str())),
        IrModule::new(&rhs_ir_text).with_top(Some(rhs_top.as_str())),
    )
    .with_solver(solver)
    .with_tool_path(tool_path);

    Ok(dispatch_ir_equiv(&request, subcommand))
}
