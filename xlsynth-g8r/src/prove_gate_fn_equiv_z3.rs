// SPDX-License-Identifier: Apache-2.0

//! Gate-level equivalence checking using the Z3 SMT solver.
//!
//! This implementation mirrors the SAT‐based checker in
//! `prove_gate_fn_equiv_varisat.rs`, but uses the Z3 SMT solver instead of
//! Varisat.  By sharing the same public interface we can swap solvers by
//! choosing the appropriate module at the call-site.

use std::collections::HashMap;

use crate::gate::{AigNode, AigRef, GateFn};
use crate::prove_gate_fn_equiv_common::EquivResult;
use z3::{
    ast::{Ast, Bool},
    Config, Context, SatResult, Solver,
};

use xlsynth::IrBits;

/// Context object that allows the caller to reuse a Z3 configuration across
/// calls.  Currently this is a light-weight placeholder – we allocate a fresh
/// Z3 context/solver per invocation because Z3 does not support resetting a
/// solver to an empty state.  The struct is kept so the public signature is
/// identical to the Varisat implementation.
#[derive(Default)]
pub struct Ctx;

impl Ctx {
    pub fn new() -> Self {
        Self
    }

    pub fn reset(&mut self) {
        // Nothing to do.  We construct a fresh solver for each query.
    }
}

/// Builds Z3 Boolean expressions for every gate in `gate_fn`.
fn build_gate_fn<'ctx>(
    ctx: &'ctx Context,
    gate_fn: &GateFn,
    input_map: &HashMap<AigRef, Bool<'ctx>>,
) -> HashMap<AigRef, Bool<'ctx>> {
    let mut map: HashMap<AigRef, Bool<'ctx>> = HashMap::new();

    // Seed the map with primary inputs.
    for (k, v) in input_map.iter() {
        map.insert(*k, v.clone());
    }

    // Post-order ensures operands are available before use.
    for aig_ref in gate_fn.post_order_refs() {
        if map.contains_key(&aig_ref) {
            continue;
        }
        let expr = match &gate_fn.gates[aig_ref.id] {
            AigNode::Literal(v) => Bool::from_bool(ctx, *v),
            AigNode::Input { .. } => input_map
                .get(&aig_ref)
                .expect("input Bool not in map")
                .clone(),
            AigNode::And2 { a, b, .. } => {
                let a_bool = map.get(&a.node).expect("operand a not yet built").clone();
                let b_bool = map.get(&b.node).expect("operand b not yet built").clone();
                let a_bool = if a.negated { a_bool.not() } else { a_bool };
                let b_bool = if b.negated { b_bool.not() } else { b_bool };
                z3::ast::Bool::and(ctx, &[&a_bool, &b_bool])
            }
        };
        map.insert(aig_ref, expr);
    }
    map
}

/// Extracts a concrete counter-example assignment from the Z3 model.
fn model_to_cex<'ctx>(
    model: &z3::Model<'ctx>,
    input_map: &HashMap<AigRef, Bool<'ctx>>,
    gate_fn: &GateFn,
) -> Vec<IrBits> {
    let mut assignment = HashMap::new();
    for (aig_ref, var) in input_map.iter() {
        let val_ast = model.eval(var, true).expect("model does not assign var");
        let bit_val = val_ast.as_bool().expect("expected Bool value in model");
        assignment.insert(*aig_ref, bit_val);
    }
    gate_fn.map_to_inputs(assignment)
}

/// Proves equivalence between `a` and `b` using Z3.
///
/// If outputs are identical for every input assignment the function returns
/// `EquivResult::Proved`; otherwise a concrete counter-example is returned via
/// `EquivResult::Disproved`.
pub fn prove_gate_fn_equiv<'a>(a: &GateFn, b: &GateFn, _ctx: &mut Ctx) -> EquivResult {
    assert_eq!(a.inputs.len(), b.inputs.len(), "Input count mismatch");
    assert_eq!(a.outputs.len(), b.outputs.len(), "Output count mismatch");

    // Create a fresh Z3 context and solver for this query.
    let cfg = Config::new();
    let z3ctx = Context::new(&cfg);
    let solver = Solver::new(&z3ctx);

    // Build shared primary inputs (Bool variables) for both functions.
    let mut input_map: HashMap<AigRef, Bool> = HashMap::new();
    for (inp_idx, input) in a.inputs.iter().enumerate() {
        let other_input = &b.inputs[inp_idx];
        assert_eq!(
            input.get_bit_count(),
            other_input.get_bit_count(),
            "Bit-width mismatch on input {}",
            inp_idx
        );
        for bit_idx in 0..input.get_bit_count() {
            let name = format!("in_{}_{}", inp_idx, bit_idx);
            let var = Bool::new_const(&z3ctx, name);
            // Inputs should not be negated.
            let a_op = input.bit_vector.get_lsb(bit_idx);
            let b_op = other_input.bit_vector.get_lsb(bit_idx);
            assert_eq!(a_op.node, b_op.node, "Input ordering mismatch");
            assert!(
                !a_op.negated && !b_op.negated,
                "Primary input operands should not be negated"
            );
            input_map.insert(a_op.node, var);
        }
    }

    // Build Boolean expressions for both gate functions.
    let map_a = build_gate_fn(&z3ctx, a, &input_map);
    let map_b = build_gate_fn(&z3ctx, b, &input_map);

    // Collect output Bools (respecting any explicit negations on the
    // connections to the outputs).
    let mut diff_terms = Vec::new();
    for (out_a, out_b) in a.outputs.iter().zip(&b.outputs) {
        assert_eq!(
            out_a.get_bit_count(),
            out_b.get_bit_count(),
            "Output width mismatch"
        );
        for bit_idx in 0..out_a.get_bit_count() {
            let op_a = out_a.bit_vector.get_lsb(bit_idx);
            let op_b = out_b.bit_vector.get_lsb(bit_idx);
            let expr_a = map_a.get(&op_a.node).unwrap().clone();
            let expr_b = map_b.get(&op_b.node).unwrap().clone();
            let expr_a = if op_a.negated { expr_a.not() } else { expr_a };
            let expr_b = if op_b.negated { expr_b.not() } else { expr_b };
            // a != b  <=>  (a XOR b) === !(a == b)
            diff_terms.push(expr_a._eq(&expr_b).not());
        }
    }

    if diff_terms.is_empty() {
        return EquivResult::Proved;
    }

    // Assert that at least one output bit differs.
    let any_diff = Bool::or(&z3ctx, &diff_terms.iter().collect::<Vec<_>>());
    solver.assert(&any_diff);

    match solver.check() {
        SatResult::Unsat => EquivResult::Proved,
        SatResult::Unknown => panic!("Z3 returned Unknown for gate equivalence check"),
        SatResult::Sat => {
            let model = solver.get_model().expect("model expected on SAT");
            let cex = model_to_cex(&model, &input_map, a);
            EquivResult::Disproved(cex)
        }
    }
}
