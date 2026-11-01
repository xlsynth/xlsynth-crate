// SPDX-License-Identifier: Apache-2.0

//! Conditional independence of selected return bits from selected input bits.

use super::translate::{get_fn_inputs, ir_to_smt};
use super::types::{
    EquivResult, FnInput, FnInputs, FnOutput, IrTypedBitVec, ProverFn, SmtFn, UfRegistry,
};
use super::{SolverLimits, default_prover_with_limits};
use crate::ir_operand_gate::PredicateReachability;
use crate::solver::{Response, Solver};
use std::collections::HashMap;
use std::ops::Range;
use xlsynth_pir::{ir, ir_verify};

/// A half-open, LSB-numbered bit range in a bits leaf of a value.
/// Each path component indexes a tuple member or array element. An empty path
/// selects a scalar root. Aggregate roots and tokens are not bits leaves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitSelection {
    pub path: Vec<usize>,
    pub bits: Range<usize>,
}

/// Bits of a positional function parameter to replace with arbitrary values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputBitSelection {
    pub parameter: usize,
    pub selection: BitSelection,
}

/// A proof result plus coverage of the original-input guard.
#[derive(Debug, Clone, PartialEq)]
pub struct IndependenceResult {
    /// Disproved carries both complete parameter assignments and full outputs.
    /// Only selected output bits are required to differ. Errors are returned
    /// separately; ToolchainDisproved is not produced by this API.
    pub result: EquivResult,
    pub guard_reachability: PredicateReachability,
}

/// Proves when(x) => observe(f(x)) == observe(f(x')) for all x and x',
/// where x' shares every bit of x except the union of the vary selections.
///
/// The guard is an ordinary PIR function returning bits[1], with the same
/// parameter types/order as function (names may differ). It is evaluated on x
/// ONLY: replacement inputs need not satisfy the guard. Assertions are ignored;
/// only return bits are observed. Both functions must be valid, and package
/// context is required for calls. Domains, UFs, and fixed implicit activation
/// are rejected rather than silently adding assumptions.
///
/// Assertions in either function (including callees) do not constrain the
/// quantified input domain, and assertion failures are not observed. Token
/// parameters and results are supported as zero-width values, but cannot be
/// selected as bits leaves. Explicit activation bits remain ordinary inputs.
/// Trace, cover, and token-ordering effects are not observed either: a proof
/// establishes only the selected return-bit property, not effect equivalence.
/// Block ports and state/register operations are outside this function API.
///
/// Empty vary or observe proves equality, but guard reachability is still
/// checked. Empty ranges, including 0..0 in bits[0], select no bits.
/// Overlapping selections mean their union. Tuple/array indices must be in
/// bounds. Unknown/resource-limited checks are inconclusive, never proofs of
/// vacuity. Limits apply separately to each solver check, as in the existing
/// prover.
///
/// See also Prover::prove_conditional_independence to select another backend.
///
/// ```no_run
/// use xlsynth_prover::prover::conditional_independence::{
///     prove_conditional_independence, BitSelection, InputBitSelection,
/// };
/// use xlsynth_prover::prover::{SolverLimits, types::ProverFn};
/// # fn example(f: &ProverFn<'_>, disabled: &ProverFn<'_>) -> Result<(), String> {
/// // f(enable: bits[1], payload: bits[8]) -> bits[8];
/// // disabled has the same inputs and returns !enable.
/// let proof = prove_conditional_independence(
///     f, disabled,
///     &[InputBitSelection {
///         parameter: 1,
///         selection: BitSelection { path: vec![], bits: 0..8 },
///     }],
///     &[BitSelection { path: vec![], bits: 0..8 }],
///     SolverLimits::default(),
/// )?;
/// # Ok(()) }
/// ```
pub fn prove_conditional_independence(
    function: &ProverFn<'_>,
    when: &ProverFn<'_>,
    vary: &[InputBitSelection],
    observe: &[BitSelection],
    limits: SolverLimits,
) -> Result<IndependenceResult, String> {
    default_prover_with_limits(limits).prove_conditional_independence(function, when, vary, observe)
}

/// Resolves a leaf selection using the existing SMT aggregate layout.
fn flat_range(ty: &ir::Type, selection: &BitSelection) -> Result<Range<usize>, String> {
    let mut ty = ty;
    let mut offset = 0;
    for &index in &selection.path {
        match ty {
            ir::Type::Tuple(members) if index < members.len() => {
                offset += ty.tuple_get_flat_bit_slice_for_index(index).unwrap().start;
                ty = &members[index];
            }
            ir::Type::Array(array) if index < array.element_count => {
                offset += index * array.element_type.bit_count();
                ty = &array.element_type;
            }
            _ => {
                return Err(format!(
                    "invalid selection path {:?} for {ty:?}",
                    selection.path
                ));
            }
        }
    }
    match ty {
        ir::Type::Bits(width)
            if selection.bits.start <= selection.bits.end && selection.bits.end <= *width =>
        {
            Ok(offset + selection.bits.start..offset + selection.bits.end)
        }
        _ => Err(format!(
            "invalid bits range {:?} for leaf {ty:?}",
            selection.bits
        )),
    }
}

/// Validates functions before invoking the existing SMT translator.
fn validate(function: &ProverFn<'_>) -> Result<(), String> {
    if function.has_domains() || !function.uf_map.is_empty() || function.fixed_implicit_activation {
        return Err(
            "conditional independence does not support domains, UFs, or fixed activation".into(),
        );
    }
    match function.pkg_ref {
        Some(pkg) => {
            ir_verify::verify_package(pkg).map_err(|e| e.to_string())?;
            ir_verify::verify_function_in_package(function.fn_ref, pkg)
        }
        None => ir_verify::verify_function(function.fn_ref),
    }
    .map_err(|e| e.to_string())?;
    if function.pkg_ref.is_none()
        && function.fn_ref.nodes.iter().any(|n| {
            matches!(
                n.payload,
                ir::NodePayload::Invoke { .. } | ir::NodePayload::CountedFor { .. }
            )
        })
    {
        return Err("calls require package context".into());
    }
    Ok(())
}

/// Runs the miter on the existing solver abstraction, sharing all unvaried
/// bits.
pub(crate) fn prove_with_solver<S: Solver>(
    config: &S::Config,
    function: &ProverFn<'_>,
    when: &ProverFn<'_>,
    vary: &[InputBitSelection],
    observe: &[BitSelection],
) -> Result<IndependenceResult, String> {
    validate(function)?;
    validate(when)?;
    if when.fn_ref.ret_ty != ir::Type::Bits(1)
        || !function
            .params()
            .map(|p| &p.ty)
            .eq(when.params().map(|p| &p.ty))
    {
        return Err(
            "guard must return bits[1] and match function parameter types and order".into(),
        );
    }
    let params: Vec<_> = function.params().collect();
    let mut ranges = vec![Vec::new(); params.len()];
    for selection in vary {
        let param = params
            .get(selection.parameter)
            .ok_or_else(|| format!("invalid parameter index {}", selection.parameter))?;
        ranges[selection.parameter].push(flat_range(&param.ty, &selection.selection)?);
    }
    let outputs = observe
        .iter()
        .map(|s| flat_range(&function.fn_ref.ret_ty, s))
        .collect::<Result<Vec<_>, _>>()?;
    let run = || -> std::io::Result<IndependenceResult> {
        let mut solver = S::new(config)?;
        let original = get_fn_inputs(&mut solver, function.clone(), Some("original"));
        let mut replacement = original.clone();
        // Splice ranges instead of materializing a term for every shared bit.
        // Overlaps are harmless: the last fresh slice still varies exactly the
        // union.
        for (param, ranges) in params.iter().zip(&ranges) {
            let value = replacement.inputs.get_mut(param.param_name()).unwrap();
            let width = value.bitvec.get_width();
            for range in ranges.iter().filter(|r| !r.is_empty()) {
                let fresh = solver.declare_fresh("replacement", range.len())?;
                let low = solver.slice(&value.bitvec, 0, range.start);
                let high = solver.slice(&value.bitvec, range.end, width - range.end);
                let low = solver.concat(&fresh, &low);
                value.bitvec = solver.concat(&high, &low);
            }
        }
        let guard_inputs = FnInputs {
            prover_fn: when.clone(),
            inputs: when
                .params()
                .zip(params.iter())
                .map(|(guard, param)| {
                    (
                        guard.param_name().to_string(),
                        IrTypedBitVec {
                            ir_type: &guard.ty,
                            bitvec: original.inputs[param.param_name()].bitvec.clone(),
                        },
                    )
                })
                .collect(),
        };
        let uf_map = HashMap::new();
        let registry = UfRegistry {
            ufs: HashMap::new(),
        };
        let guard = ir_to_smt(&mut solver, &guard_inputs, &uf_map, &registry);
        solver.assert(&guard.output.bitvec)?;
        let guard_reachability = match solver.check()? {
            Response::Sat => PredicateReachability::Reachable,
            Response::Unsat => PredicateReachability::Unreachable,
            Response::Unknown => PredicateReachability::Unknown,
        };
        if guard_reachability == PredicateReachability::Unreachable {
            return Ok(IndependenceResult {
                result: EquivResult::Proved,
                guard_reachability,
            });
        }
        let lhs = ir_to_smt(&mut solver, &original, &uf_map, &registry);
        let rhs = ir_to_smt(&mut solver, &replacement, &uf_map, &registry);
        let mut differs = solver.false_bv();
        for range in outputs.iter().filter(|r| !r.is_empty()) {
            let left = solver.slice(&lhs.output.bitvec, range.start, range.len());
            let right = solver.slice(&rhs.output.bitvec, range.start, range.len());
            let diff = solver.ne(&left, &right);
            differs = solver.or(&differs, &diff);
        }
        solver.assert(&differs)?;
        let result = match solver.check()? {
            Response::Unsat => EquivResult::Proved,
            Response::Unknown => EquivResult::Inconclusive(
                "conditional independence solver returned unknown (possibly a resource limit)"
                    .into(),
            ),
            Response::Sat => EquivResult::Disproved {
                lhs_inputs: model_inputs(&mut solver, &lhs)?,
                rhs_inputs: model_inputs(&mut solver, &rhs)?,
                lhs_output: FnOutput {
                    value: solver.get_value(&lhs.output.bitvec, lhs.output.ir_type)?,
                    assertion_violation: None,
                },
                rhs_output: FnOutput {
                    value: solver.get_value(&rhs.output.bitvec, rhs.output.ir_type)?,
                    assertion_violation: None,
                },
            },
        };
        // A counterexample establishes reachability even if the first check
        // timed out.
        let guard_reachability = if matches!(result, EquivResult::Disproved { .. }) {
            PredicateReachability::Reachable
        } else {
            guard_reachability
        };
        Ok(IndependenceResult {
            result,
            guard_reachability,
        })
    };
    run().map_err(|e| format!("conditional independence solver error: {e}"))
}

/// Extracts complete input assignments in declaration order for witness replay.
fn model_inputs<S: Solver>(
    solver: &mut S,
    function: &SmtFn<'_, S::Term>,
) -> std::io::Result<Vec<FnInput>> {
    function
        .fn_ref
        .param_nodes()
        .zip(&function.inputs)
        .map(|(param, term)| {
            Ok(FnInput {
                name: param.param_name().to_string(),
                value: solver.get_value(&term.bitvec, term.ir_type)?,
            })
        })
        .collect()
}
