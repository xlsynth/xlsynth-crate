use std::collections::HashMap;

use crate::gate::{AigNode, AigRef, GateFn};
use crate::topo::extract_cone;
use crate::validate_equiv::{Ctx, EquivResult};
use z3::{ast::Bool, Config, Context, SatResult, Solver};

/// Build a mapping from AIG refs to Z3 Bool expressions and collect the output
/// expressions in evaluation order.
fn build_gate_fn<'ctx>(
    ctx: &'ctx Context,
    gate_fn: &GateFn,
    input_bools: &[Vec<Bool<'ctx>>],
) -> (HashMap<AigRef, Bool<'ctx>>, Vec<Bool<'ctx>>) {
    // Map primary input node -> Bool variable.
    let mut input_map = HashMap::new();
    for (i, inp) in gate_fn.inputs.iter().enumerate() {
        for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
            input_map.insert(op.node, input_bools[i][j].clone());
        }
    }

    // Determine cone we actually need.
    let output_refs: Vec<AigRef> = gate_fn
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
        .map(|op| op.node)
        .collect();
    let (cone_gates, cone_inputs) = extract_cone(&output_refs, &gate_fn.gates);

    // Build a set for quick membership test.
    let cone_set: std::collections::HashSet<AigRef> = cone_gates.iter().cloned().collect();

    // Iterate gates in a global topological order so that operands are built first.
    let global_topo = crate::topo::topo_sort_refs(&gate_fn.gates);

    let mut map: HashMap<AigRef, Bool<'ctx>> = HashMap::new();

    // Populate known primary inputs first.
    for input in &cone_inputs {
        let bool_var = input_map.get(input).expect("missing primary input").clone();
        map.insert(*input, bool_var);
    }

    // Evaluate gates in topological order.
    for g in global_topo.iter().filter(|r| cone_set.contains(r)) {
        if map.contains_key(g) {
            continue;
        }
        let expr = match &gate_fn.gates[g.id] {
            AigNode::Literal(v) => {
                if *v {
                    Bool::from_bool(ctx, true)
                } else {
                    Bool::from_bool(ctx, false)
                }
            }
            AigNode::And2 { a, b, .. } => {
                let a_expr = {
                    let base = map.get(&a.node).expect("a missing").clone();
                    if a.negated {
                        base.not()
                    } else {
                        base
                    }
                };
                let b_expr = {
                    let base = map.get(&b.node).expect("b missing").clone();
                    if b.negated {
                        base.not()
                    } else {
                        base
                    }
                };
                Bool::and(ctx, &[&a_expr, &b_expr])
            }
            AigNode::Input { .. } => {
                unreachable!("primary inputs already handled")
            }
        };
        map.insert(*g, expr);
    }

    // Collect outputs.
    let mut outputs = Vec::new();
    for out in &gate_fn.outputs {
        for bit in out.bit_vector.iter_lsb_to_msb() {
            let base = map.get(&bit.node).expect("output missing").clone();
            outputs.push(if bit.negated { base.not() } else { base });
        }
    }

    (map, outputs)
}

/// Z3-based equivalence checker.
#[allow(clippy::needless_lifetimes)]
pub fn check_equiv<'a>(a: &GateFn, b: &GateFn, _ctx: &mut Ctx<'a>) -> EquivResult {
    assert_eq!(a.inputs.len(), b.inputs.len());
    assert_eq!(a.outputs.len(), b.outputs.len());

    let mut cfg = Config::new();
    cfg.set_param_value("model", "true");
    let z3_ctx = Context::new(&cfg);
    let solver = Solver::new(&z3_ctx);

    // Build primary input variables.
    let mut input_bools: Vec<Vec<Bool>> = Vec::new();
    for (idx, inp) in a.inputs.iter().enumerate() {
        assert_eq!(inp.get_bit_count(), b.inputs[idx].get_bit_count());
        let mut vec_bits = Vec::new();
        for bit_idx in 0..inp.get_bit_count() {
            let name = format!("in_{}_{}", idx, bit_idx);
            let var = Bool::new_const(&z3_ctx, name);
            vec_bits.push(var);
        }
        input_bools.push(vec_bits);
    }

    let (_, outputs_a) = build_gate_fn(&z3_ctx, a, &input_bools);
    let (_, outputs_b) = build_gate_fn(&z3_ctx, b, &input_bools);

    // Build XOR miters
    let mut miters = Vec::new();
    for (la, lb) in outputs_a.iter().zip(outputs_b.iter()) {
        miters.push(la.xor(lb));
    }

    // diff = OR(miters)
    let diff = Bool::or(&z3_ctx, &miters.iter().collect::<Vec<&Bool>>());
    solver.assert(&diff);

    match solver.check() {
        SatResult::Unsat => EquivResult::Proved,
        SatResult::Sat => {
            let model = solver.get_model().expect("model expected");
            let mut map = HashMap::new();
            for (i, inp) in a.inputs.iter().enumerate() {
                for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
                    let bv = &input_bools[i][j];
                    let val = model.eval(bv, true).unwrap().as_bool().unwrap();
                    map.insert(op.node, val);
                }
            }
            let cex = a.map_to_inputs(map);
            EquivResult::Disproved(cex)
        }
        SatResult::Unknown => panic!("Z3 returned unknown"),
    }
}
