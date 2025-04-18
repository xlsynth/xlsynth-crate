// SPDX-License-Identifier: Apache-2.0

//! Validates equivalence classes proposed by `propose_equiv`.
//!
//! For a given equivalence class we will either get confirmation that they are
//! all equivalent or a counterexample that demonstrates a case in which they
//! are not equivalent.
//!
//! We use varisat for this because it supports incrementality via assume/solve
//! and add_clause.

use std::collections::{HashMap, HashSet};

use crate::gate::{extract_cone, AigNode, AigRef, GateFn};
use xlsynth::IrBits;

pub struct ValidationResult {
    /// Sets that were proven equivalent, i.e. any value in set i can be
    /// substituted for any other value in set i.
    pub proven_equiv_sets: Vec<Vec<AigRef>>,

    /// Input values that showed counterexamples in the equivalence sets, so
    /// that these can be used as concrete stimulus for distinguishing proposals
    /// in subsequent iterations.
    pub cex_inputs: Vec<Vec<IrBits>>,
}

#[derive(Debug)]
pub enum ValidationError {
    SolverError(varisat::solver::SolverError),
}

// Tseitin clauses for: output_lit <=> lit_a AND lit_b
// The Tseitsin clauses are a way of encoding the result of the AND gate in
// terms of a fresh literal, which in our case is the `output_literal`.
// The expansion is that `x ↔ A ∧ B` becomes:
// (x ∨ ¬A ∨ ¬B)
// (¬x ∨ A)
// (¬x ∨ B)
fn add_tseitsin_and(
    solver: &mut impl varisat::ExtendFormula,
    a: varisat::Lit,
    b: varisat::Lit,
    output: varisat::Lit,
) {
    solver.add_clause(&[!a, !b, output]);
    solver.add_clause(&[a, !output]);
    solver.add_clause(&[b, !output]);
}

// Clauses for m = a XOR b are:
// (!a | !b | !m) & (a | b | !m) & (a | !b | m) & (!a | b | m)
fn add_tseitsin_xor(
    solver: &mut impl varisat::ExtendFormula,
    a: varisat::Lit,
    b: varisat::Lit,
    output: varisat::Lit,
) {
    solver.add_clause(&[!a, !b, !output]);
    solver.add_clause(&[a, b, !output]);
    solver.add_clause(&[a, !b, output]);
    solver.add_clause(&[!a, b, output]);
}

/// Returns a mapping from each AigRef in the cone to its corresponding SAT
/// literal.
fn build_sat_clauses(
    solver: &mut impl varisat::ExtendFormula,
    cone_gates: &[AigRef],
    cone_inputs: &HashSet<AigRef>,
    gates: &[AigNode],
) -> HashMap<AigRef, varisat::Lit> {
    let mut aig_ref_to_lit: HashMap<AigRef, varisat::Lit> = HashMap::new();

    // Create literals for all gates in the cone.
    for aig_ref in cone_gates {
        let lit = solver.new_lit();
        aig_ref_to_lit.insert(*aig_ref, lit);
    }

    // Create literals for all inputs to the cone.
    for input in cone_inputs {
        let lit = solver.new_lit();
        aig_ref_to_lit.insert(*input, lit);
    }

    // For each gate add correpsonding structural clauses.
    for aig_ref in cone_gates {
        let output_lit = aig_ref_to_lit[aig_ref];
        let gate = &gates[aig_ref.id];
        match gate {
            AigNode::Literal(value) => {
                if *value {
                    solver.add_clause(&[output_lit]);
                } else {
                    solver.add_clause(&[!output_lit]);
                }
            }
            AigNode::And2 { a, b, .. } => {
                let a_node_lit = aig_ref_to_lit[&a.node];
                let b_node_lit = aig_ref_to_lit[&b.node];
                let a_lit = if a.negated { !a_node_lit } else { a_node_lit };
                let b_lit = if b.negated { !b_node_lit } else { b_node_lit };
                add_tseitsin_and(solver, a_lit, b_lit, output_lit);
            }
            AigNode::Input { .. } => {
                // Nothing to do for this.
            }
        }
    }

    aig_ref_to_lit
}

fn add_miters(
    solver: &mut impl varisat::ExtendFormula,
    aig_ref_to_lit: &HashMap<AigRef, varisat::Lit>,
    known_equiv: &[AigRef],
    candidate: AigRef,
) -> Vec<varisat::Lit> {
    let mut miters: Vec<varisat::Lit> = Vec::with_capacity(known_equiv.len());
    for lhs in known_equiv {
        let xor_miter = solver.new_lit();
        let a = aig_ref_to_lit[lhs];
        let b = aig_ref_to_lit[&candidate];
        add_tseitsin_xor(solver, a, b, xor_miter);
        miters.push(xor_miter);
    }
    miters
}

fn solver_model_to_cex(
    model: &[varisat::Lit],
    cone_inputs: &HashSet<AigRef>,
    aig_ref_to_lit: &HashMap<AigRef, varisat::Lit>,
    gate_fn: &GateFn,
) -> Vec<IrBits> {
    let model_set: HashSet<varisat::Lit> = model.iter().cloned().collect();

    // We could really collect this as a tristate value and then flatten it with
    // some X policy, but to keep it simple we just emit zero.
    let mut inputs_map: HashMap<AigRef, bool> = HashMap::new();
    for input_aig_ref in cone_inputs.iter() {
        let input_lit: varisat::Lit = aig_ref_to_lit[&input_aig_ref];
        let not_input_lit: varisat::Lit = !input_lit;
        if model_set.contains(&input_lit) {
            inputs_map.insert(*input_aig_ref, true);
        } else if model_set.contains(&not_input_lit) {
            inputs_map.insert(*input_aig_ref, false);
        } else {
            inputs_map.insert(*input_aig_ref, false);
        }
    }
    let cex = gate_fn.map_to_inputs(inputs_map);
    cex
}

pub fn validate_equiv(
    gate_fn: &GateFn,
    equiv_classes: &[&[AigRef]],
) -> Result<ValidationResult, ValidationError> {
    // Extract the combined cone for all of the references we're trying to determine
    // equivalence for.
    let mut frontier: Vec<AigRef> = vec![];
    for equiv_class in equiv_classes {
        for aig_ref in equiv_class.iter() {
            frontier.push(*aig_ref);
        }
    }

    let (cone_gates, cone_inputs) = extract_cone(&frontier, &gate_fn.gates);

    let mut solver = varisat::Solver::new();

    // Build the SAT clauses for the cone -- we're going to add miters on top of
    // this structure.
    let aig_ref_to_lit = build_sat_clauses(&mut solver, &cone_gates, &cone_inputs, &gate_fn.gates);

    let mut validation_result = ValidationResult {
        proven_equiv_sets: Vec::new(),
        cex_inputs: Vec::new(),
    };

    // Now iterate through the equivalence classes -- for each equivalence class
    // we'll advance a miter that checks whether the next value in the
    // equivalence class is equivalent to the previous value(s) which are
    // determined to be equivalent. By using multiple miters I'm suspecting we can
    // potentially find counterexamples faster.
    for equiv_class in equiv_classes {
        let mut known_equiv = vec![equiv_class[0]];
        for &candidate in &equiv_class[1..] {
            // Create a miter between this candidate and all known equivalent values.
            let miters = add_miters(&mut solver, &aig_ref_to_lit, &known_equiv, candidate);

            // Assume the miters outputs are true, which is stating "the candidate is
            // unequal to the known_equiv values".
            solver.assume(&miters);
            let solve_result = solver.solve();
            match solve_result {
                Ok(false) => {
                    // No counterexample found, expand the known equivalent set.
                    known_equiv.push(candidate);
                }
                Ok(true) => {
                    // Counterexample found, extract it from the model.
                    let model = solver.model();
                    assert!(
                        model.is_some(),
                        "counterexample found, should be able to extract model"
                    );
                    let cex = solver_model_to_cex(
                        &model.unwrap(),
                        &cone_inputs,
                        &aig_ref_to_lit,
                        gate_fn,
                    );
                    validation_result.cex_inputs.push(cex);
                    break;
                }
                Err(e) => {
                    return Err(ValidationError::SolverError(e));
                }
            }
        }
        if known_equiv.len() > 1 {
            validation_result
                .proven_equiv_sets
                .push(equiv_class.to_vec())
        }
    }

    Ok(validation_result)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::SeedableRng;

    use crate::{
        gate::AigRef,
        propose_equiv::propose_equiv,
        test_utils::{load_bf16_mul_sample, setup_graph_with_redundancies, Opt},
    };

    use super::validate_equiv;

    #[test]
    fn test_validate_equiv_graph_with_redundancies() {
        let setup = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let equiv_classes = propose_equiv(&setup.g, 16, &mut seeded_rng);
        let classes: Vec<&[AigRef]> = equiv_classes
            .values()
            .map(|nodes| nodes.as_slice())
            .collect();
        let validation_result = validate_equiv(&setup.g, &classes).unwrap();
        assert_eq!(validation_result.proven_equiv_sets.len(), 2);
    }

    #[test]
    fn test_validate_equiv_bf16_mul() {
        let setup = load_bf16_mul_sample(Opt::Yes);
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let propose_start = Instant::now();
        let equiv_classes = propose_equiv(&setup.gate_fn, 256, &mut seeded_rng);
        let propose_end = Instant::now();
        let propose_duration = propose_end - propose_start;
        eprintln!(
            "Proposed {} equiv classes in {:?}",
            equiv_classes.len(),
            propose_duration
        );
        assert_eq!(equiv_classes.len(), 1036);

        let classes: Vec<&[AigRef]> = equiv_classes
            .values()
            .map(|nodes| nodes.as_slice())
            .collect();
        let validate_start = Instant::now();
        let validation_result = validate_equiv(&setup.gate_fn, &classes).unwrap();
        let validate_end = Instant::now();
        let validate_duration = validate_end - validate_start;
        eprintln!(
            "Validated to {} proven sets {} counterexamples in {:?}",
            validation_result.proven_equiv_sets.len(),
            validation_result.cex_inputs.len(),
            validate_duration
        );
        assert_eq!(validation_result.proven_equiv_sets.len(), 19);
    }
}
