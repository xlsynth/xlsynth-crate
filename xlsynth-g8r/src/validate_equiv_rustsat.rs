use std::collections::HashMap;
use std::env;

use crate::gate::{AigNode, AigRef, GateFn};
use crate::topo::extract_cone;
use crate::validate_equiv::{Ctx, EquivResult};

use rustsat::instances::SatInstance;
use rustsat::solvers::{Solve, SolveIncremental, SolverResult};
use rustsat::types::{Assignment, Lit, TernaryVal};
// use rustsat_batsat::BasicSolver;
// use rustsat_cadical::{CaDiCaL, Config, Limit};
// use rustsat_minisat::core::Minisat;

// Tseitin clauses for: output <=> a AND b
fn add_tseitsin_and(inst: &mut SatInstance, a: Lit, b: Lit, output: Lit) {
    // (x ∨ ¬A ∨ ¬B)
    inst.add_ternary(!a, !b, output);
    // (¬x ∨ A)
    inst.add_binary(!output, a);
    // (¬x ∨ B)
    inst.add_binary(!output, b);
}

// Tseitin clauses for m = a XOR b
fn add_tseitsin_xor(inst: &mut SatInstance, a: Lit, b: Lit, output: Lit) {
    // (!a | !b | !m)
    inst.add_ternary(!a, !b, !output);
    // (a | b | !m)
    inst.add_ternary(a, b, !output);
    // (a | !b | m)
    inst.add_ternary(a, !b, output);
    // (!a | b | m)
    inst.add_ternary(!a, b, output);
}

fn build_gate_fn(
    inst: &mut SatInstance,
    gate_fn: &GateFn,
    input_lits: &[Vec<Lit>],
) -> (HashMap<AigRef, Lit>, Vec<Lit>) {
    let mut input_map = HashMap::new();
    for (i, inp) in gate_fn.inputs.iter().enumerate() {
        for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
            input_map.insert(op.node, input_lits[i][j]);
        }
    }

    let output_refs: Vec<AigRef> = gate_fn
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
        .map(|op| op.node)
        .collect();
    let (cone_gates, cone_inputs) = extract_cone(&output_refs, &gate_fn.gates);

    let mut map: HashMap<AigRef, Lit> = HashMap::new();

    // fresh literals for cone gates
    for g in &cone_gates {
        let lit = inst.new_lit();
        map.insert(*g, lit);
    }

    // map primary inputs
    for input in &cone_inputs {
        let lit = *input_map
            .get(input)
            .expect("cone input should be in primary input map");
        map.insert(*input, lit);
    }

    // structural clauses
    for g in &cone_gates {
        let out_lit = map[g];
        match &gate_fn.gates[g.id] {
            AigNode::Literal(v) => {
                if *v {
                    inst.add_unit(out_lit);
                } else {
                    inst.add_unit(!out_lit);
                }
            }
            AigNode::And2 { a, b, .. } => {
                let a_lit = if a.negated {
                    !map[&a.node]
                } else {
                    map[&a.node]
                };
                let b_lit = if b.negated {
                    !map[&b.node]
                } else {
                    map[&b.node]
                };
                add_tseitsin_and(inst, a_lit, b_lit, out_lit);
            }
            AigNode::Input { .. } => {}
        }
    }

    // collect outputs
    let mut outputs = Vec::new();
    for out in &gate_fn.outputs {
        for bit in out.bit_vector.iter_lsb_to_msb() {
            let base = map[&bit.node];
            outputs.push(if bit.negated { !base } else { base });
        }
    }

    (map, outputs)
}

/// RustSAT/CaDiCaL-based equivalence checker mirroring the Varisat version.
/// Note: the provided `ctx` is ignored as the implementation relies on its own
/// solver instance.
#[allow(clippy::needless_lifetimes)]
pub fn check_equiv<'a>(a: &GateFn, b: &GateFn, _ctx: &mut Ctx<'a>) -> EquivResult {
    assert_eq!(a.inputs.len(), b.inputs.len());
    assert_eq!(a.outputs.len(), b.outputs.len());

    // Build SAT instance.
    let mut instance = SatInstance::new();

    // Primary input literals for both functions.
    let mut input_lits: Vec<Vec<Lit>> = Vec::new();
    for (ia, ib) in a.inputs.iter().zip(b.inputs.iter()) {
        assert_eq!(ia.get_bit_count(), ib.get_bit_count());
        let mut vec_bits = Vec::new();
        for _ in 0..ia.get_bit_count() {
            vec_bits.push(instance.new_lit());
        }
        input_lits.push(vec_bits);
    }

    let (_, outputs_a) = build_gate_fn(&mut instance, a, &input_lits);
    let (_, outputs_b) = build_gate_fn(&mut instance, b, &input_lits);

    // XOR miters
    let mut miters = Vec::new();
    for (la, lb) in outputs_a.iter().zip(outputs_b.iter()) {
        let m = instance.new_lit();
        add_tseitsin_xor(&mut instance, *la, *lb, m);
        miters.push(m);
    }

    // diff literal: true if any miter is true
    let diff = instance.new_lit();
    let mut clause_lits = Vec::with_capacity(miters.len() + 1);
    clause_lits.push(!diff);
    clause_lits.extend(miters.iter().copied());
    instance.add_clause(rustsat::types::Clause::from(clause_lits.as_slice()));

    // Solve under assumption diff
    let (cnf, _vmanager) = instance.into_cnf();

    // Decide which backend to use.
    //   • default: CaDiCaL ("cadical")
    //   • XLSYNTH_ORACLE_SOLVER=batsat  → pure-Rust BatSat (fast, no deps)
    //   • XLSYNTH_ORACLE_SOLVER=minisat → MiniSat C++ backend (often faster)
    //   • XLSYNTH_ORACLE_SOLVER=cadical → explicit CaDiCaL
    let chosen = env::var("XLSYNTH_ORACLE_SOLVER").unwrap_or_else(|_| "cadical".into());

    let (sat_result, model_opt): (SolverResult, Option<Assignment>) = todo!();
    /*match chosen.as_str() {
        "batsat" => {
            let mut solver = BasicSolver::default();
            solver.add_cnf(cnf).expect("add cnf");
            let res = solver.solve_assumps(&[diff]).expect("solver");
            let model = if matches!(res, SolverResult::Sat) {
                Some(solver.full_solution().expect("model"))
            } else {
                None
            };
            (res, model)
        }
        "minisat" => {
            let mut solver = Minisat::default();
            solver.add_cnf(cnf).expect("add cnf");
            let res = solver.solve_assumps(&[diff]).expect("solver");
            let model = if matches!(res, SolverResult::Sat) {
                Some(solver.full_solution().expect("model"))
            } else {
                None
            };
            (res, model)
        }
        "cadical" | _ => {
            let mut solver = CaDiCaL::default();
            let _ = solver.set_configuration(Config::Plain);
            let _ = solver.set_limit(Limit::Preprocessing(0));
            solver.add_cnf(cnf).expect("add cnf");
            let res = solver.solve_assumps(&[diff]).expect("solver");
            let model = if matches!(res, SolverResult::Sat) {
                Some(solver.full_solution().expect("model"))
            } else {
                None
            };
            (res, model)
        }
    };*/

    match sat_result {
        SolverResult::Unsat => EquivResult::Proved,
        SolverResult::Sat => {
            let sol = model_opt.expect("model missing");
            // Map assignment to counterexample
            let mut map = HashMap::new();
            for (i, inp) in a.inputs.iter().enumerate() {
                for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
                    let lit = input_lits[i][j];
                    let val = matches!(sol[lit.var()], TernaryVal::True);
                    map.insert(op.node, val);
                }
            }
            let cex = a.map_to_inputs(map);
            EquivResult::Disproved(cex)
        }
        SolverResult::Interrupted => panic!("solver interrupted"),
    }
}
