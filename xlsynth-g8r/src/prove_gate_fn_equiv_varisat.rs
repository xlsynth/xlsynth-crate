// SPDX-License-Identifier: Apache-2.0

//! Validates equivalence classes proposed by `propose_equiv`.
//!
//! For a given equivalence class we will either get confirmation that they are
//! all equivalent or a counterexample that demonstrates a case in which they
//! are not equivalent.
//!
//! The default backend is Varisat because it supports incrementality via
//! assume/solve and add_clause. CaDiCaL is also available for profiling and
//! benchmarking the validation path.

use std::collections::{HashMap, HashSet};
use std::env;
use std::ops::Not;

use crate::aig::gate::{AigNode, AigRef, GateFn};
use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::topo::extract_cone;
use crate::propose_equiv::EquivNode;
pub use crate::prove_gate_fn_equiv_common::EquivResult;
use varisat::ExtendFormula;
use xlsynth::IrBits;

/// Context holding a SAT solver so clause memory can be reused across calls.
pub struct Ctx<'a> {
    pub(crate) solver: varisat::Solver<'a>,
}

impl<'a> Ctx<'a> {
    pub fn new() -> Self {
        Self {
            solver: varisat::Solver::new(),
        }
    }

    pub fn reset(&mut self) {
        self.solver = varisat::Solver::new();
    }
}

pub struct ValidationResult {
    /// Sets that were proven equivalent, i.e. any value in set i can be
    /// substituted for any other value in set i.
    pub proven_equiv_sets: Vec<Vec<EquivNode>>,

    /// Input values that showed counterexamples in the equivalence sets, so
    /// that these can be used as concrete stimulus for distinguishing proposals
    /// in subsequent iterations.
    pub cex_inputs: Vec<Vec<IrBits>>,
}

#[derive(Debug)]
pub enum ValidationError {
    VarisatSolverError(varisat::solver::SolverError),
    CadicalConfigError(String),
    CadicalSolveInterrupted,
    ModelUnavailable,
    UnknownBackend(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ValidationError {}

pub const VALIDATION_BACKEND_ENV: &str = "XLSYNTH_G8R_VALIDATION_BACKEND";
pub const CADICAL_CONFIG_ENV: &str = "XLSYNTH_G8R_CADICAL_CONFIG";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationBackend {
    Varisat,
    Cadical,
}

impl ValidationBackend {
    fn from_env() -> Result<Self, ValidationError> {
        match env::var(VALIDATION_BACKEND_ENV) {
            Ok(value) => Self::parse(&value),
            Err(env::VarError::NotPresent) => Ok(Self::Varisat),
            Err(env::VarError::NotUnicode(value)) => Err(ValidationError::UnknownBackend(
                value.to_string_lossy().to_string(),
            )),
        }
    }

    fn parse(value: &str) -> Result<Self, ValidationError> {
        match value {
            "varisat" | "Varisat" | "VARISAT" => Ok(Self::Varisat),
            "cadical" | "CaDiCaL" | "CADICAL" => Ok(Self::Cadical),
            other => Err(ValidationError::UnknownBackend(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SatSolveResult {
    Sat,
    Unsat,
}

trait SatModel<Lit: Copy> {
    fn lit_value(&self, lit: Lit) -> bool;
}

trait IncrementalSat {
    type Lit: Copy + Eq + std::hash::Hash + Not<Output = Self::Lit>;
    type Model: SatModel<Self::Lit> + Clone;

    fn sat_new_lit(&mut self) -> Self::Lit;
    fn sat_add_clause(&mut self, clause: &[Self::Lit]);
    fn sat_solve_assuming(
        &mut self,
        assumptions: &[Self::Lit],
    ) -> Result<SatSolveResult, ValidationError>;
    fn sat_model(&self) -> Result<Self::Model, ValidationError>;
}

#[derive(Clone)]
struct VarisatModel {
    true_lits: HashSet<varisat::Lit>,
}

impl SatModel<varisat::Lit> for VarisatModel {
    fn lit_value(&self, lit: varisat::Lit) -> bool {
        self.true_lits.contains(&lit)
    }
}

impl<'a> IncrementalSat for varisat::Solver<'a> {
    type Lit = varisat::Lit;
    type Model = VarisatModel;

    fn sat_new_lit(&mut self) -> Self::Lit {
        self.new_lit()
    }

    fn sat_add_clause(&mut self, clause: &[Self::Lit]) {
        self.add_clause(clause);
    }

    fn sat_solve_assuming(
        &mut self,
        assumptions: &[Self::Lit],
    ) -> Result<SatSolveResult, ValidationError> {
        self.assume(assumptions);
        self.solve()
            .map(|sat| {
                if sat {
                    SatSolveResult::Sat
                } else {
                    SatSolveResult::Unsat
                }
            })
            .map_err(ValidationError::VarisatSolverError)
    }

    fn sat_model(&self) -> Result<Self::Model, ValidationError> {
        let model = self.model().ok_or(ValidationError::ModelUnavailable)?;
        Ok(VarisatModel {
            true_lits: model.into_iter().collect(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CadicalLit(i32);

impl Not for CadicalLit {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(-self.0)
    }
}

struct CadicalSat {
    solver: cadical::Solver,
    next_var: i32,
}

impl CadicalSat {
    fn new() -> Result<Self, ValidationError> {
        let solver = match env::var(CADICAL_CONFIG_ENV) {
            Ok(config) => cadical::Solver::with_config(&config)
                .map_err(|e| ValidationError::CadicalConfigError(format!("{config}: {e}")))?,
            Err(env::VarError::NotPresent) => cadical::Solver::new(),
            Err(env::VarError::NotUnicode(value)) => {
                return Err(ValidationError::CadicalConfigError(
                    value.to_string_lossy().to_string(),
                ));
            }
        };
        Ok(Self {
            solver,
            next_var: 1,
        })
    }
}

#[derive(Clone)]
struct CadicalModel {
    values: Vec<bool>,
}

impl SatModel<CadicalLit> for CadicalModel {
    fn lit_value(&self, lit: CadicalLit) -> bool {
        let var = lit.0.unsigned_abs() as usize;
        let value = self.values.get(var).copied().unwrap_or(false);
        if lit.0 < 0 { !value } else { value }
    }
}

impl IncrementalSat for CadicalSat {
    type Lit = CadicalLit;
    type Model = CadicalModel;

    fn sat_new_lit(&mut self) -> Self::Lit {
        let lit = CadicalLit(self.next_var);
        self.next_var += 1;
        lit
    }

    fn sat_add_clause(&mut self, clause: &[Self::Lit]) {
        self.solver.add_clause(clause.iter().map(|lit| lit.0));
    }

    fn sat_solve_assuming(
        &mut self,
        assumptions: &[Self::Lit],
    ) -> Result<SatSolveResult, ValidationError> {
        match self.solver.solve_with(assumptions.iter().map(|lit| lit.0)) {
            Some(true) => Ok(SatSolveResult::Sat),
            Some(false) => Ok(SatSolveResult::Unsat),
            None => Err(ValidationError::CadicalSolveInterrupted),
        }
    }

    fn sat_model(&self) -> Result<Self::Model, ValidationError> {
        let max_var = (self.next_var - 1) as usize;
        let mut values = vec![false; max_var + 1];
        for (var, slot) in values.iter_mut().enumerate().skip(1) {
            *slot = self.solver.value(var as i32).unwrap_or(false);
        }
        Ok(CadicalModel { values })
    }
}

// Tseitin clauses for: output_lit <=> lit_a AND lit_b
// The Tseitsin clauses are a way of encoding the result of the AND gate in
// terms of a fresh literal, which in our case is the `output_literal`.
// The expansion is that `x ↔ A ∧ B` becomes:
// (x ∨ ¬A ∨ ¬B)
// (¬x ∨ A)
// (¬x ∨ B)
fn add_tseitsin_and<S: IncrementalSat>(solver: &mut S, a: S::Lit, b: S::Lit, output: S::Lit) {
    solver.sat_add_clause(&[!a, !b, output]);
    solver.sat_add_clause(&[a, !output]);
    solver.sat_add_clause(&[b, !output]);
}

// Clauses for m = a XOR b are:
// (!a | !b | !m) & (a | b | !m) & (a | !b | m) & (!a | b | m)
fn add_tseitsin_xor<S: IncrementalSat>(solver: &mut S, a: S::Lit, b: S::Lit, output: S::Lit) {
    solver.sat_add_clause(&[!a, !b, !output]);
    solver.sat_add_clause(&[a, b, !output]);
    solver.sat_add_clause(&[a, !b, output]);
    solver.sat_add_clause(&[!a, b, output]);
}

/// Returns a mapping from each AigRef in the cone to its corresponding SAT
/// literal.
fn build_sat_clauses<S: IncrementalSat>(
    solver: &mut S,
    cone_gates: &[AigRef],
    cone_inputs: &HashSet<AigRef>,
    gates: &[AigNode],
) -> HashMap<AigRef, S::Lit> {
    let mut aig_ref_to_lit: HashMap<AigRef, S::Lit> = HashMap::new();

    // Create literals for all gates in the cone.
    for aig_ref in cone_gates {
        let lit = solver.sat_new_lit();
        aig_ref_to_lit.insert(*aig_ref, lit);
    }

    // Create literals for all inputs to the cone in deterministic order. SAT
    // model choices feed FRAIG counterexamples, so literal numbering should
    // not depend on HashSet iteration order.
    let mut sorted_cone_inputs: Vec<AigRef> = cone_inputs.iter().copied().collect();
    sorted_cone_inputs.sort_unstable_by_key(|input| input.id);
    for input in sorted_cone_inputs {
        let lit = solver.sat_new_lit();
        aig_ref_to_lit.insert(input, lit);
    }

    // For each gate add correpsonding structural clauses.
    for aig_ref in cone_gates {
        let output_lit = aig_ref_to_lit[aig_ref];
        let gate = &gates[aig_ref.id];
        match gate {
            AigNode::Literal { value, .. } => {
                if *value {
                    solver.sat_add_clause(&[output_lit]);
                } else {
                    solver.sat_add_clause(&[!output_lit]);
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

fn add_miter<S: IncrementalSat>(
    solver: &mut S,
    aig_ref_to_lit: &HashMap<AigRef, S::Lit>,
    lhs_node: EquivNode,
    candidate: EquivNode,
) -> S::Lit {
    let xor_miter = solver.sat_new_lit();

    // Get SAT literals for the underlying AIG nodes.
    let a_lit = aig_ref_to_lit[&lhs_node.aig_ref()];
    let b_lit = aig_ref_to_lit[&candidate.aig_ref()];

    // Check if the relationship is inverted (Normal vs Inverted).
    match (lhs_node, candidate) {
        (EquivNode::Normal(_), EquivNode::Normal(_))
        | (EquivNode::Inverted(_), EquivNode::Inverted(_)) => {
            // Same type: Check for equivalence (a XOR b == 0). Miter output is
            // true if they are different.
            add_tseitsin_xor(solver, a_lit, b_lit, xor_miter);
        }
        (EquivNode::Normal(_), EquivNode::Inverted(_))
        | (EquivNode::Inverted(_), EquivNode::Normal(_)) => {
            // Different type: Check for inverse equivalence (a XNOR b == 0, or
            // a XOR !b == 0). Miter output is true if they are different (i.e.,
            // not inverses). We compute a XOR (NOT b) for the miter.
            add_tseitsin_xor(solver, a_lit, !b_lit, xor_miter);
        }
    }

    xor_miter
}

fn model_value_for_equiv_node<Lit: Copy>(
    model: &impl SatModel<Lit>,
    aig_ref_to_lit: &HashMap<AigRef, Lit>,
    equiv_node: EquivNode,
) -> bool {
    let base_value = model.lit_value(aig_ref_to_lit[&equiv_node.aig_ref()]);
    if equiv_node.is_inverted() {
        !base_value
    } else {
        base_value
    }
}

fn split_bucket_by_model<Lit: Copy>(
    nodes: &[EquivNode],
    model: &impl SatModel<Lit>,
    aig_ref_to_lit: &HashMap<AigRef, Lit>,
) -> Vec<Vec<EquivNode>> {
    let mut false_values = Vec::new();
    let mut true_values = Vec::new();

    for &node in nodes {
        if model_value_for_equiv_node(model, aig_ref_to_lit, node) {
            true_values.push(node);
        } else {
            false_values.push(node);
        }
    }

    if false_values.is_empty() || true_values.is_empty() {
        return vec![nodes.to_vec()];
    }

    [false_values, true_values]
        .into_iter()
        .filter(|bucket| bucket.len() > 1)
        .collect()
}

fn presplit_by_counterexample_models<Lit: Copy, Model: SatModel<Lit>>(
    nodes: Vec<EquivNode>,
    counterexample_models: &[Model],
    aig_ref_to_lit: &HashMap<AigRef, Lit>,
) -> Vec<Vec<EquivNode>> {
    let mut buckets = vec![nodes];
    for model in counterexample_models {
        buckets = buckets
            .into_iter()
            .flat_map(|bucket| split_bucket_by_model(&bucket, model, aig_ref_to_lit))
            .collect();
        if buckets.is_empty() {
            break;
        }
    }
    buckets
}

fn equiv_node_depth_key(
    ref_to_depth: &HashMap<AigRef, usize>,
    equiv_node: EquivNode,
) -> (usize, bool, usize) {
    let aig_ref = equiv_node.aig_ref();
    (ref_to_depth[&aig_ref], equiv_node.is_inverted(), aig_ref.id)
}

fn sorted_equiv_class(
    equiv_class: &[EquivNode],
    ref_to_depth: &HashMap<AigRef, usize>,
) -> Vec<EquivNode> {
    let mut nodes = equiv_class.to_vec();
    nodes.sort_unstable_by_key(|node| equiv_node_depth_key(ref_to_depth, *node));
    nodes
}

fn solver_model_to_cex<Lit: Copy, Model: SatModel<Lit>>(
    model: &Model,
    all_inputs: &HashSet<AigRef>,
    aig_ref_to_lit: &HashMap<AigRef, Lit>,
    gate_fn: &GateFn,
) -> Vec<IrBits> {
    let mut inputs_map: HashMap<AigRef, bool> = HashMap::new();
    for input_aig_ref in all_inputs {
        if let Some(input_lit) = aig_ref_to_lit.get(input_aig_ref) {
            // Input was part of the cone, check model
            if model.lit_value(*input_lit) {
                inputs_map.insert(*input_aig_ref, true);
            } else {
                // Default to false if not explicitly true in the model
                inputs_map.insert(*input_aig_ref, false);
            }
        } else {
            // Input was NOT part of the cone, default to false
            inputs_map.insert(*input_aig_ref, false);
        }
    }

    // Now map_to_inputs should receive a map covering all expected inputs
    let cex = gate_fn.map_to_inputs(inputs_map);
    cex
}

fn build_gate_fn<S: IncrementalSat>(
    solver: &mut S,
    gate_fn: &GateFn,
    input_lits: &[Vec<S::Lit>],
) -> (HashMap<AigRef, S::Lit>, Vec<S::Lit>) {
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

    let mut map = HashMap::new();

    for g in &cone_gates {
        let lit = solver.sat_new_lit();
        map.insert(*g, lit);
    }

    for input in &cone_inputs {
        let lit = *input_map
            .get(input)
            .expect("cone input should be in primary input map");
        map.insert(*input, lit);
    }

    for g in &cone_gates {
        let out_lit = map[g];
        match &gate_fn.gates[g.id] {
            AigNode::Literal { value: v, .. } => {
                if *v {
                    solver.sat_add_clause(&[out_lit]);
                } else {
                    solver.sat_add_clause(&[!out_lit]);
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
                add_tseitsin_and(solver, a_lit, b_lit, out_lit);
            }
            AigNode::Input { .. } => {}
        }
    }

    let mut outputs = Vec::new();
    for out in &gate_fn.outputs {
        for bit in out.bit_vector.iter_lsb_to_msb() {
            let base = map[&bit.node];
            outputs.push(if bit.negated { !base } else { base });
        }
    }

    (map, outputs)
}

/// Checks equivalence of two gate functions using a SAT solver.
pub fn prove_gate_fn_equiv<'a>(a: &GateFn, b: &GateFn, ctx: &mut Ctx<'a>) -> EquivResult {
    assert_eq!(a.inputs.len(), b.inputs.len());
    assert_eq!(a.outputs.len(), b.outputs.len());

    let mut input_lits = Vec::new();
    for (ia, ib) in a.inputs.iter().zip(b.inputs.iter()) {
        assert_eq!(ia.get_bit_count(), ib.get_bit_count());
        let mut lits = Vec::new();
        for _ in 0..ia.get_bit_count() {
            lits.push(ctx.solver.new_lit());
        }
        input_lits.push(lits);
    }

    let (_map_a, outputs_a) = build_gate_fn(&mut ctx.solver, a, &input_lits);
    let (_map_b, outputs_b) = build_gate_fn(&mut ctx.solver, b, &input_lits);

    // Build XOR miters for each corresponding output bit.
    let mut miters = Vec::new();
    for (la, lb) in outputs_a.iter().zip(outputs_b.iter()) {
        let m = ctx.solver.new_lit();
        add_tseitsin_xor(&mut ctx.solver, *la, *lb, m);
        miters.push(m);
    }

    // Fresh literal that stands for "outputs differ in *some* bit".
    let diff = ctx.solver.new_lit();
    // diff -> OR(miters)  === (!diff OR m1 OR m2 ...)
    let mut clause = Vec::with_capacity(miters.len() + 1);
    clause.push(!diff);
    clause.extend(miters.iter().cloned());
    ctx.solver.add_clause(&clause);

    // Ask the solver to find an assignment where outputs differ.
    ctx.solver.assume(&[diff]);
    match ctx.solver.solve() {
        Ok(false) => EquivResult::Proved, // UNSAT ⇒ no way for outputs to differ.
        Ok(true) => {
            let model = ctx.solver.model().expect("model available when SAT");
            let model_set: HashSet<varisat::Lit> = model.iter().cloned().collect();
            let mut map = HashMap::new();
            for (i, inp) in a.inputs.iter().enumerate() {
                for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
                    let lit = input_lits[i][j];
                    map.insert(op.node, model_set.contains(&lit));
                }
            }
            let cex = a.map_to_inputs(map);
            EquivResult::Disproved(cex)
        }
        Err(e) => panic!("Solver error: {:?}", e),
    }
}

pub fn validate_equivalence_classes(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
) -> Result<ValidationResult, ValidationError> {
    validate_equivalence_classes_with_backend(
        gate_fn,
        equiv_classes,
        ValidationBackend::from_env()?,
    )
}

/// Validates classes whose members and class order have already been
/// depth-sorted.
pub fn validate_equivalence_classes_presorted(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
) -> Result<ValidationResult, ValidationError> {
    validate_equivalence_classes_presorted_with_backend(
        gate_fn,
        equiv_classes,
        ValidationBackend::from_env()?,
    )
}

pub fn validate_equivalence_classes_with_backend(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: ValidationBackend,
) -> Result<ValidationResult, ValidationError> {
    match backend {
        ValidationBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ false,
            )
        }
        ValidationBackend::Cadical => {
            let mut solver = CadicalSat::new()?;
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ false,
            )
        }
    }
}

fn validate_equivalence_classes_presorted_with_backend(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: ValidationBackend,
) -> Result<ValidationResult, ValidationError> {
    match backend {
        ValidationBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ true,
            )
        }
        ValidationBackend::Cadical => {
            let mut solver = CadicalSat::new()?;
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ true,
            )
        }
    }
}

fn validate_equivalence_classes_with_solver<S: IncrementalSat>(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    solver: &mut S,
    classes_are_depth_sorted: bool,
) -> Result<ValidationResult, ValidationError> {
    // Extract the combined cone for all of the references we're trying to determine
    // equivalence for.
    let mut frontier: Vec<AigRef> = vec![];
    for equiv_class in equiv_classes {
        for equiv_node in equiv_class.iter() {
            frontier.push(equiv_node.aig_ref());
        }
    }

    // Collect all primary input refs
    let all_primary_inputs: HashSet<AigRef> = gate_fn
        .inputs
        .iter()
        .flat_map(|input_vec| input_vec.bit_vector.iter_lsb_to_msb())
        .map(|op| op.node)
        .collect();
    let (cone_gates, cone_inputs) = extract_cone(&frontier, &gate_fn.gates);

    // Build the SAT clauses for the cone -- we're going to add miters on top of
    // this structure.
    let aig_ref_to_lit = build_sat_clauses(solver, &cone_gates, &cone_inputs, &gate_fn.gates);

    let mut validation_result = ValidationResult {
        proven_equiv_sets: Vec::new(),
        cex_inputs: Vec::new(),
    };
    let sorted_equiv_classes: Vec<Vec<EquivNode>> = if classes_are_depth_sorted {
        equiv_classes
            .iter()
            .map(|equiv_class| equiv_class.to_vec())
            .collect()
    } else {
        let all_nodes: Vec<AigRef> = gate_fn
            .gates
            .iter()
            .enumerate()
            .map(|(id, _)| AigRef { id })
            .collect();
        let depth_stats = get_gate_depth(gate_fn, &all_nodes);
        let mut sorted_equiv_classes: Vec<Vec<EquivNode>> = equiv_classes
            .iter()
            .map(|equiv_class| sorted_equiv_class(equiv_class, &depth_stats.ref_to_depth))
            .collect();
        sorted_equiv_classes.sort_unstable_by_key(|equiv_class| {
            let representative = equiv_class[0];
            (
                equiv_node_depth_key(&depth_stats.ref_to_depth, representative),
                equiv_class.len(),
            )
        });
        sorted_equiv_classes
    };
    let mut counterexample_models: Vec<S::Model> = Vec::new();

    // Now iterate through the equivalence classes -- for each equivalence class
    // we'll advance a representative and check each next value against it.
    // Values already in `known_equiv` have been proven equivalent to the
    // representative, so a representative-only check is sufficient by
    // transitivity and avoids adding redundant miter clauses as the bucket
    // grows. Before spending SAT work on a class, split it by counterexamples
    // found in earlier classes. A new counterexample still stops the current
    // original class, preserving the old per-class proof budget.
    for equiv_class in sorted_equiv_classes {
        let buckets =
            presplit_by_counterexample_models(equiv_class, &counterexample_models, &aig_ref_to_lit);
        let mut stop_class_after_new_counterexample = false;
        for bucket in buckets {
            let mut known_equiv = vec![bucket[0]];
            for &candidate in &bucket[1..] {
                // Create a miter between this candidate and the class representative.
                let representative = known_equiv[0];
                let miter = add_miter(solver, &aig_ref_to_lit, representative, candidate);

                // Assume the miter output is true, which asks for a counterexample where
                // the candidate is unequal to the representative.
                match solver.sat_solve_assuming(&[miter])? {
                    SatSolveResult::Unsat => {
                        // No counterexample found, expand the known equivalent set.
                        known_equiv.push(candidate);
                    }
                    SatSolveResult::Sat => {
                        // Counterexample found, extract it from the model.
                        let model = solver.sat_model()?;
                        let cex = solver_model_to_cex(
                            &model,
                            &all_primary_inputs,
                            &aig_ref_to_lit,
                            gate_fn,
                        );
                        validation_result.cex_inputs.push(cex);
                        counterexample_models.push(model);
                        stop_class_after_new_counterexample = true;
                        break;
                    }
                }
            }

            if known_equiv.len() > 1 {
                validation_result.proven_equiv_sets.push(known_equiv);
            }
            if stop_class_after_new_counterexample {
                break;
            }
        }
    }

    Ok(validation_result)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::SeedableRng;

    use crate::{
        propose_equiv::{EquivNode, propose_equivalence_classes},
        test_utils::{setup_graph_with_redundancies, setup_partially_equiv_graph},
    };

    use super::{
        ValidationBackend, ValidationResult, validate_equivalence_classes,
        validate_equivalence_classes_with_backend,
    };
    #[allow(unused_imports)]
    use crate::assert_within;

    fn canonical_proven_sets(result: &ValidationResult) -> Vec<Vec<EquivNode>> {
        let mut sets = result.proven_equiv_sets.clone();
        for set in &mut sets {
            set.sort_unstable();
        }
        sets.sort_unstable();
        sets
    }

    #[test]
    fn test_validate_equiv_graph_with_redundancies() {
        let setup = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let counterexamples = HashSet::new();
        let equiv_classes =
            propose_equivalence_classes(&setup.g, 16, &mut seeded_rng, &counterexamples);
        let classes: Vec<&[EquivNode]> = equiv_classes
            .values()
            .map(|nodes| nodes.as_slice())
            .collect();
        let validation_result = validate_equivalence_classes(&setup.g, &classes).unwrap();
        // There are 2 redundancies and they have inverted pairs.
        assert_eq!(validation_result.proven_equiv_sets.len(), 4);
    }

    #[test]
    fn test_cadical_matches_varisat_on_redundant_graph() {
        let setup = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let counterexamples = HashSet::new();
        let equiv_classes =
            propose_equivalence_classes(&setup.g, 16, &mut seeded_rng, &counterexamples);
        let classes: Vec<&[EquivNode]> = equiv_classes
            .values()
            .map(|nodes| nodes.as_slice())
            .collect();

        let varisat = validate_equivalence_classes_with_backend(
            &setup.g,
            &classes,
            ValidationBackend::Varisat,
        )
        .unwrap();
        let cadical = validate_equivalence_classes_with_backend(
            &setup.g,
            &classes,
            ValidationBackend::Cadical,
        )
        .unwrap();

        assert_eq!(
            canonical_proven_sets(&cadical),
            canonical_proven_sets(&varisat)
        );
        assert_eq!(cadical.cex_inputs.len(), varisat.cex_inputs.len());
    }

    #[test]
    fn test_validate_partial_equivalence() {
        let setup = setup_partially_equiv_graph();

        // Propose a class where a and b are equivalent, but c is not.
        let proposed_class = &[
            EquivNode::Normal(setup.a.node),
            EquivNode::Normal(setup.b.node),
            EquivNode::Normal(setup.c.node),
        ];

        let validation_result = validate_equivalence_classes(&setup.g, &[proposed_class]).unwrap();

        // Expect one proven set containing only a and b.
        assert_eq!(
            validation_result.proven_equiv_sets.len(),
            1,
            "Should find exactly one proven set"
        );
        assert_eq!(
            validation_result.proven_equiv_sets[0].len(),
            2,
            "Proven set should contain 2 elements (a, b)"
        );

        // Sort for consistent comparison
        let mut proven_set = validation_result.proven_equiv_sets[0].clone();
        proven_set.sort_unstable();
        let mut expected_proven = vec![
            EquivNode::Normal(setup.a.node),
            EquivNode::Normal(setup.b.node),
        ];
        expected_proven.sort_unstable();
        assert_eq!(
            proven_set, expected_proven,
            "Proven set should contain nodes a and b"
        );

        // Expect one counterexample (for c vs a/b).
        assert_eq!(
            validation_result.cex_inputs.len(),
            1,
            "Should find exactly one counterexample"
        );
    }

    #[test]
    fn test_cadical_matches_varisat_on_partial_equivalence() {
        let setup = setup_partially_equiv_graph();
        let proposed_class = &[
            EquivNode::Normal(setup.a.node),
            EquivNode::Normal(setup.b.node),
            EquivNode::Normal(setup.c.node),
        ];

        let varisat = validate_equivalence_classes_with_backend(
            &setup.g,
            &[proposed_class],
            ValidationBackend::Varisat,
        )
        .unwrap();
        let cadical = validate_equivalence_classes_with_backend(
            &setup.g,
            &[proposed_class],
            ValidationBackend::Cadical,
        )
        .unwrap();

        assert_eq!(
            canonical_proven_sets(&cadical),
            canonical_proven_sets(&varisat)
        );
        assert_eq!(cadical.cex_inputs.len(), varisat.cex_inputs.len());
    }

    #[test]
    fn test_validate_reuses_counterexample_for_later_class() {
        let setup = setup_partially_equiv_graph();

        let proposed_class = &[
            EquivNode::Normal(setup.c.node),
            EquivNode::Normal(setup.a.node),
            EquivNode::Normal(setup.b.node),
        ];

        let validation_result =
            validate_equivalence_classes(&setup.g, &[proposed_class, proposed_class]).unwrap();

        assert_eq!(
            validation_result.proven_equiv_sets.len(),
            2,
            "Both duplicate classes should still prove the a == b pair"
        );
        assert_eq!(
            validation_result.cex_inputs.len(),
            1,
            "The second class should be split by the first class's counterexample"
        );
    }
}
