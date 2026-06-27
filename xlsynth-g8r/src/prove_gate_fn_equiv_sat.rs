// SPDX-License-Identifier: Apache-2.0

//! SAT-backed gate-function equivalence and equivalence-class validation.
//!
//! For a given equivalence class we will either get confirmation that they are
//! all equivalent or a counterexample that demonstrates a case in which they
//! are not equivalent.
//!
//! The default backend is CaDiCaL. Varisat remains available for comparison and
//! context-reuse testing; Z3 and IR backends are dispatched through the common
//! gate-formal backend API where supported.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::ops::Not;
use std::time::{Duration, Instant};

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Output};
use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::topo::extract_cone;
use crate::aig_sim::gate_sim;
use crate::propose_equiv::EquivNode;
pub use crate::prove_gate_fn_equiv_common::{EquivResult, GateFormalBackend};
use varisat::ExtendFormula;
use xlsynth::IrBits;

/// Context holding a SAT solver so clause memory can be reused across calls.
pub struct VarisatCtx<'a> {
    pub(crate) solver: varisat::Solver<'a>,
}

impl<'a> VarisatCtx<'a> {
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

    /// Number of formal proof queries issued while validating the classes.
    pub proof_query_count: usize,

    /// Number of proof queries that exhausted their deterministic resource
    /// limit and were conservatively left unresolved.
    pub interrupted_proof_count: usize,
}

/// Result of proving normalized simulation-constant candidates in groups.
pub struct ConstantGroupValidationResult {
    /// Sets proven equal to false; each set contains the false literal followed
    /// by every candidate proved by one grouped query.
    pub proven_equiv_sets: Vec<Vec<EquivNode>>,

    /// Counterexamples that disproved at least one candidate in a group.
    pub cex_inputs: Vec<Vec<IrBits>>,

    /// Number of grouped formal queries issued.
    pub proof_query_count: usize,

    /// Number of grouped queries left unresolved by the resource limit.
    pub interrupted_group_count: usize,

    /// Number of normalized candidates proved constant.
    pub proven_candidate_count: usize,

    /// Number of normalized candidates disproved by a SAT model.
    pub disproven_candidate_count: usize,

    /// Number of candidates left unresolved, including a final singleton.
    pub unresolved_candidate_count: usize,
}

/// Statistics for a single-solver FRAIG validation run.
#[derive(Debug, Clone)]
pub struct FullGraphFraigValidationStat {
    pub solver_build_seconds: f64,
    pub constant_seconds: f64,
    pub equivalence_seconds: f64,
    pub constant_candidate_count: usize,
    pub constant_group_query_count: usize,
    pub constant_interrupted_group_count: usize,
    pub constant_deferred_candidate_count: usize,
    pub constant_proven_candidate_count: usize,
    pub constant_disproven_candidate_count: usize,
    pub constant_counterexample_count: usize,
    pub equivalence_class_visit_count: usize,
    pub equivalence_proof_query_count: usize,
    pub equivalence_interrupted_proof_count: usize,
    pub interrupted_relation_skip_count: usize,
    pub virtual_proof_count: usize,
    pub equivalence_counterexample_count: usize,
    pub initial_class_histogram: BTreeMap<usize, usize>,
    pub post_constant_class_histogram: BTreeMap<usize, usize>,
}

/// Result of grouped constant proving and ordinary FRAIG in one SAT session.
pub struct FullGraphFraigValidationResult {
    pub validation: ValidationResult,
    pub stat: FullGraphFraigValidationStat,
}

#[derive(Debug)]
pub enum ValidationError {
    VarisatSolverError(varisat::solver::SolverError),
    CadicalConfigError(String),
    CadicalSolveInterrupted,
    ModelUnavailable,
    UnsupportedBackend {
        backend: GateFormalBackend,
        operation: &'static str,
    },
    IrEquivalenceError(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ValidationError {}

pub const CADICAL_CONFIG_ENV: &str = "XLSYNTH_G8R_CADICAL_CONFIG";
pub const DEFAULT_CADICAL_TERMINATE_LIMIT: u32 = 100;

/// Optional resource limits for gate-level formal proof backends.
#[derive(Debug, Clone, Copy, Default)]
pub struct GateFormalOptions {
    pub cadical_timeout: Option<Duration>,
    pub cadical_terminate_limit: Option<u32>,
}

impl GateFormalOptions {
    /// Applies a timeout to each CaDiCaL solve call.
    pub fn with_cadical_timeout(mut self, timeout: Duration) -> Self {
        self.cadical_timeout = Some(timeout);
        self
    }

    /// Applies a deterministic internal termination-check limit to each
    /// CaDiCaL solve call. A value of zero disables the limit.
    pub fn with_cadical_terminate_limit(mut self, limit: u32) -> Self {
        self.cadical_terminate_limit = (limit != 0).then_some(limit);
        self
    }
}

fn resolve_equivalence_class_backend(
    backend: GateFormalBackend,
) -> Result<GateFormalBackend, ValidationError> {
    match backend {
        GateFormalBackend::Cadical => Ok(GateFormalBackend::Cadical),
        GateFormalBackend::Varisat => Ok(GateFormalBackend::Varisat),
        GateFormalBackend::Z3 | GateFormalBackend::Ir => Ok(backend),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SatSolveResult {
    Sat,
    Unsat,
}

pub(crate) trait SatModel<Lit: Copy> {
    fn lit_value(&self, lit: Lit) -> bool;
}

pub(crate) trait IncrementalSat {
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
pub(crate) struct VarisatModel {
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
pub(crate) struct CadicalLit(i32);

impl Not for CadicalLit {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(-self.0)
    }
}

pub(crate) struct CadicalSat {
    solver: cadical::Solver,
    next_var: i32,
    terminate_limit: Option<u32>,
}

impl CadicalSat {
    pub(crate) fn new() -> Result<Self, ValidationError> {
        Self::new_with_options(GateFormalOptions::default())
    }

    pub(crate) fn new_with_options(options: GateFormalOptions) -> Result<Self, ValidationError> {
        let mut solver = match env::var(CADICAL_CONFIG_ENV) {
            Ok(config) => cadical::Solver::with_config(&config)
                .map_err(|e| ValidationError::CadicalConfigError(format!("{config}: {e}")))?,
            Err(env::VarError::NotPresent) => cadical::Solver::new(),
            Err(env::VarError::NotUnicode(value)) => {
                return Err(ValidationError::CadicalConfigError(
                    value.to_string_lossy().to_string(),
                ));
            }
        };
        if let Some(timeout) = options.cadical_timeout {
            solver.set_callbacks(Some(cadical::Timeout::new(timeout.as_secs_f32())));
        }
        Ok(Self {
            solver,
            next_var: 1,
            terminate_limit: options.cadical_terminate_limit,
        })
    }
}

#[derive(Clone)]
pub(crate) struct CadicalModel {
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
        if let Some(limit) = self.terminate_limit {
            let limit = i32::try_from(limit).map_err(|_| {
                ValidationError::CadicalConfigError(format!(
                    "terminate limit {limit} exceeds CaDiCaL's i32 range"
                ))
            })?;
            // CaDiCaL resets limits after each solve, so apply this before every
            // incremental query.
            self.solver
                .set_limit("terminate", limit)
                .map_err(|e| ValidationError::CadicalConfigError(e.to_string()))?;
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VirtualExpression {
    Operand(AigOperand),
    And2 {
        a: AigOperand,
        b: AigOperand,
        output_negated: bool,
    },
}

/// Tracks proven relationships and recognizes downstream structural congruence.
struct VirtualEquivalence {
    parent: Vec<usize>,
    parity_to_parent: Vec<bool>,
    depth: Vec<usize>,
}

impl VirtualEquivalence {
    fn new(gate_fn: &GateFn) -> Self {
        let all_nodes: Vec<AigRef> = gate_fn
            .gates
            .iter()
            .enumerate()
            .map(|(id, _)| AigRef { id })
            .collect();
        let depth_stats = get_gate_depth(gate_fn, &all_nodes);
        let depth = all_nodes
            .iter()
            .map(|node| depth_stats.ref_to_depth[node])
            .collect();
        Self {
            parent: (0..gate_fn.gates.len()).collect(),
            parity_to_parent: vec![false; gate_fn.gates.len()],
            depth,
        }
    }

    /// Returns the representative and whether `node` is its logical inverse.
    fn find(&mut self, node: AigRef) -> (AigRef, bool) {
        let mut current = node.id;
        let mut parity = false;
        let mut path = Vec::new();
        while self.parent[current] != current {
            path.push((current, parity));
            parity ^= self.parity_to_parent[current];
            current = self.parent[current];
        }
        for (path_node, parity_from_start) in path {
            self.parent[path_node] = current;
            self.parity_to_parent[path_node] = parity ^ parity_from_start;
        }
        (AigRef { id: current }, parity)
    }

    fn canonical_operand(&mut self, operand: AigOperand) -> AigOperand {
        let (root, parity) = self.find(operand.node);
        AigOperand {
            node: root,
            negated: operand.negated ^ parity,
        }
    }

    fn canonical_equiv_node(&mut self, node: EquivNode) -> AigOperand {
        self.canonical_operand(AigOperand {
            node: node.aig_ref(),
            negated: node.is_inverted(),
        })
    }

    /// Records a proven equality between the values represented by two nodes.
    fn union_equiv_nodes(&mut self, lhs: EquivNode, rhs: EquivNode) -> bool {
        let (lhs_root, lhs_parity) = self.find(lhs.aig_ref());
        let (rhs_root, rhs_parity) = self.find(rhs.aig_ref());
        let root_parity = lhs_parity ^ lhs.is_inverted() ^ rhs_parity ^ rhs.is_inverted();
        if lhs_root == rhs_root {
            debug_assert!(!root_parity, "inconsistent proven AIG relationship");
            return false;
        }

        let lhs_key = (self.depth[lhs_root.id], lhs_root.id);
        let rhs_key = (self.depth[rhs_root.id], rhs_root.id);
        if lhs_key <= rhs_key {
            self.parent[rhs_root.id] = lhs_root.id;
            self.parity_to_parent[rhs_root.id] = root_parity;
        } else {
            self.parent[lhs_root.id] = rhs_root.id;
            self.parity_to_parent[lhs_root.id] = root_parity;
        }
        true
    }

    fn are_equivalent(&mut self, lhs: EquivNode, rhs: EquivNode) -> bool {
        self.canonical_equiv_node(lhs) == self.canonical_equiv_node(rhs)
    }

    fn folded_and(lhs: AigOperand, rhs: AigOperand) -> Option<AigOperand> {
        let false_operand = AigOperand::from(AigRef { id: 0 });
        let true_operand = false_operand.negate();
        if lhs.node == rhs.node {
            return if lhs.negated == rhs.negated {
                Some(lhs)
            } else {
                Some(false_operand)
            };
        }
        if lhs == false_operand || rhs == false_operand {
            return Some(false_operand);
        }
        if lhs == true_operand {
            return Some(rhs);
        }
        if rhs == true_operand {
            return Some(lhs);
        }
        None
    }

    /// Describes a node after applying all relationships proven so far.
    fn virtual_expression(&mut self, gate_fn: &GateFn, node: EquivNode) -> VirtualExpression {
        let canonical = self.canonical_equiv_node(node);
        match gate_fn.gates[canonical.node.id] {
            AigNode::Literal { value, .. } => {
                let constant = AigOperand {
                    node: AigRef { id: 0 },
                    negated: value ^ canonical.negated,
                };
                let constant_node = if constant.negated {
                    EquivNode::Inverted(constant.node)
                } else {
                    EquivNode::Normal(constant.node)
                };
                self.union_equiv_nodes(node, constant_node);
                VirtualExpression::Operand(self.canonical_equiv_node(node))
            }
            AigNode::Input { .. } => VirtualExpression::Operand(canonical),
            AigNode::And2 { a, b, .. } => {
                let a = self.canonical_operand(a);
                let b = self.canonical_operand(b);
                if let Some(mut folded) = Self::folded_and(a, b) {
                    folded.negated ^= canonical.negated;
                    let folded_node = if folded.negated {
                        EquivNode::Inverted(folded.node)
                    } else {
                        EquivNode::Normal(folded.node)
                    };
                    self.union_equiv_nodes(node, folded_node);
                    return VirtualExpression::Operand(self.canonical_equiv_node(node));
                }
                let (a, b) = if a <= b { (a, b) } else { (b, a) };
                VirtualExpression::And2 {
                    a,
                    b,
                    output_negated: canonical.negated,
                }
            }
        }
    }

    /// Returns true when prior proofs make this query structurally redundant.
    fn prove_by_structure(&mut self, gate_fn: &GateFn, lhs: EquivNode, rhs: EquivNode) -> bool {
        if self.are_equivalent(lhs, rhs) {
            return true;
        }
        let lhs_expression = self.virtual_expression(gate_fn, lhs);
        let rhs_expression = self.virtual_expression(gate_fn, rhs);
        if self.are_equivalent(lhs, rhs) || lhs_expression == rhs_expression {
            self.union_equiv_nodes(lhs, rhs);
            return true;
        }
        false
    }

    /// Returns one normalized equality set for every non-singleton component.
    fn into_proven_equiv_sets(mut self) -> Vec<Vec<EquivNode>> {
        let mut groups: BTreeMap<AigRef, Vec<EquivNode>> = BTreeMap::new();
        for id in 0..self.parent.len() {
            let node = AigRef { id };
            let (root, parity) = self.find(node);
            groups.entry(root).or_default().push(if parity {
                EquivNode::Inverted(node)
            } else {
                EquivNode::Normal(node)
            });
        }
        groups
            .into_values()
            .filter(|group| group.len() > 1)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CanonicalEquivRelation {
    lower_node_id: usize,
    upper_node_id: usize,
    inversion_parity: bool,
}

impl CanonicalEquivRelation {
    /// Canonicalizes operand order and collapses `a == b` with `!a == !b`.
    fn new(lhs: EquivNode, rhs: EquivNode) -> Self {
        let lhs_id = lhs.aig_ref().id;
        let rhs_id = rhs.aig_ref().id;
        let (lower_node_id, upper_node_id) = if lhs_id <= rhs_id {
            (lhs_id, rhs_id)
        } else {
            (rhs_id, lhs_id)
        };
        Self {
            lower_node_id,
            upper_node_id,
            inversion_parity: lhs.is_inverted() ^ rhs.is_inverted(),
        }
    }
}

#[derive(Debug, Clone)]
struct MutableEquivalenceClass {
    nodes: Vec<EquivNode>,
    is_normalized_constant: bool,
}

fn class_size_histogram(
    equiv_classes: impl IntoIterator<Item = impl AsRef<[EquivNode]>>,
) -> BTreeMap<usize, usize> {
    let mut histogram = BTreeMap::new();
    for equiv_class in equiv_classes {
        *histogram.entry(equiv_class.as_ref().len()).or_default() += 1;
    }
    histogram
}

fn split_mutable_class_by_model<Lit: Copy>(
    equiv_class: MutableEquivalenceClass,
    model: &impl SatModel<Lit>,
    aig_ref_to_lit: &HashMap<AigRef, Lit>,
) -> Vec<MutableEquivalenceClass> {
    let mut false_values = Vec::new();
    let mut true_values = Vec::new();
    for node in equiv_class.nodes {
        if model_value_for_equiv_node(model, aig_ref_to_lit, node) {
            true_values.push(node);
        } else {
            false_values.push(node);
        }
    }

    let mut split = Vec::new();
    if false_values.len() > 1 {
        split.push(MutableEquivalenceClass {
            nodes: false_values,
            is_normalized_constant: equiv_class.is_normalized_constant,
        });
    }
    if true_values.len() > 1 {
        split.push(MutableEquivalenceClass {
            nodes: true_values,
            // A true result under a counterexample removes this child from the
            // normalized zero class, while preserving it as an ordinary class.
            is_normalized_constant: false,
        });
    }
    split
}

fn refine_mutable_classes_by_model<Lit: Copy>(
    equiv_classes: VecDeque<MutableEquivalenceClass>,
    model: &impl SatModel<Lit>,
    aig_ref_to_lit: &HashMap<AigRef, Lit>,
) -> VecDeque<MutableEquivalenceClass> {
    equiv_classes
        .into_iter()
        .flat_map(|equiv_class| split_mutable_class_by_model(equiv_class, model, aig_ref_to_lit))
        .collect()
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

/// Proves groups of normalized candidates that simulation predicts are false.
fn validate_constant_groups_with_solver<S: IncrementalSat>(
    gate_fn: &GateFn,
    normalized_candidates: &[EquivNode],
    group_size: usize,
    solver: &mut S,
) -> Result<ConstantGroupValidationResult, ValidationError> {
    assert!(group_size > 1);
    let mut unique_refs = HashSet::new();
    debug_assert!(
        normalized_candidates
            .iter()
            .all(|candidate| candidate.aig_ref().id != 0)
    );
    debug_assert!(
        normalized_candidates
            .iter()
            .all(|candidate| unique_refs.insert(candidate.aig_ref()))
    );

    if normalized_candidates.is_empty() {
        return Ok(ConstantGroupValidationResult {
            proven_equiv_sets: Vec::new(),
            cex_inputs: Vec::new(),
            proof_query_count: 0,
            interrupted_group_count: 0,
            proven_candidate_count: 0,
            disproven_candidate_count: 0,
            unresolved_candidate_count: 0,
        });
    }

    let frontier: Vec<AigRef> = normalized_candidates
        .iter()
        .map(EquivNode::aig_ref)
        .collect();
    let (cone_gates, cone_inputs) = extract_cone(&frontier, &gate_fn.gates);
    let aig_ref_to_lit = build_sat_clauses(solver, &cone_gates, &cone_inputs, &gate_fn.gates);
    let all_primary_inputs: HashSet<AigRef> = gate_fn
        .inputs
        .iter()
        .flat_map(|input_vec| input_vec.bit_vector.iter_lsb_to_msb())
        .map(|operand| operand.node)
        .collect();
    let false_literal = EquivNode::Normal(AigRef { id: 0 });
    let mut pending: VecDeque<EquivNode> = normalized_candidates.iter().copied().collect();
    let mut result = ConstantGroupValidationResult {
        proven_equiv_sets: Vec::new(),
        cex_inputs: Vec::new(),
        proof_query_count: 0,
        interrupted_group_count: 0,
        proven_candidate_count: 0,
        disproven_candidate_count: 0,
        unresolved_candidate_count: 0,
    };

    while pending.len() > 1 {
        let query_size = group_size.min(pending.len());
        let query_group: Vec<EquivNode> = pending.drain(..query_size).collect();
        let activation = solver.sat_new_lit();
        let mut clause = Vec::with_capacity(query_group.len() + 1);
        clause.push(!activation);
        clause.extend(query_group.iter().map(|candidate| {
            let base_lit = aig_ref_to_lit[&candidate.aig_ref()];
            if candidate.is_inverted() {
                !base_lit
            } else {
                base_lit
            }
        }));
        solver.sat_add_clause(&clause);
        result.proof_query_count += 1;

        match solver.sat_solve_assuming(&[activation]) {
            Ok(SatSolveResult::Unsat) => {
                result.proven_candidate_count += query_group.len();
                let mut proven_set = Vec::with_capacity(query_group.len() + 1);
                proven_set.push(false_literal);
                proven_set.extend(query_group);
                result.proven_equiv_sets.push(proven_set);
            }
            Ok(SatSolveResult::Sat) => {
                let model = solver.sat_model()?;
                let cex =
                    solver_model_to_cex(&model, &all_primary_inputs, &aig_ref_to_lit, gate_fn);
                result.cex_inputs.push(cex);

                let mut refined = VecDeque::new();
                for candidate in query_group.into_iter().chain(pending.drain(..)) {
                    if model_value_for_equiv_node(&model, &aig_ref_to_lit, candidate) {
                        result.disproven_candidate_count += 1;
                    } else {
                        refined.push_back(candidate);
                    }
                }
                pending = refined;
            }
            Err(ValidationError::CadicalSolveInterrupted) => {
                // A grouped interruption remains unresolved as a group. It is
                // deliberately never decomposed into individual retries.
                result.interrupted_group_count += 1;
                result.unresolved_candidate_count += query_group.len();
            }
            Err(error) => return Err(error),
        }
    }
    result.unresolved_candidate_count += pending.len();
    Ok(result)
}

/// Implements grouped constant queries for non-incremental formal backends.
fn validate_constant_groups_with_pairwise_backend(
    gate_fn: &GateFn,
    normalized_candidates: &[EquivNode],
    group_size: usize,
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<ConstantGroupValidationResult, ValidationError> {
    assert!(group_size > 1);
    let false_literal = EquivNode::Normal(AigRef { id: 0 });
    let all_candidates_fn =
        gate_fn_with_equiv_outputs(gate_fn, normalized_candidates, "constant_candidates");
    let mut pending: VecDeque<EquivNode> = normalized_candidates.iter().copied().collect();
    let mut result = ConstantGroupValidationResult {
        proven_equiv_sets: Vec::new(),
        cex_inputs: Vec::new(),
        proof_query_count: 0,
        interrupted_group_count: 0,
        proven_candidate_count: 0,
        disproven_candidate_count: 0,
        unresolved_candidate_count: 0,
    };

    while pending.len() > 1 {
        let query_size = group_size.min(pending.len());
        let query_group: Vec<EquivNode> = pending.drain(..query_size).collect();
        let candidate_fn = gate_fn_with_equiv_outputs(gate_fn, &query_group, "candidates");
        let false_candidates = vec![false_literal; query_group.len()];
        let false_fn = gate_fn_with_equiv_outputs(gate_fn, &false_candidates, "false_values");
        result.proof_query_count += 1;
        match prove_gate_fn_equiv_with_backend_and_options(
            &candidate_fn,
            &false_fn,
            backend,
            options,
        )? {
            EquivResult::Proved => {
                result.proven_candidate_count += query_group.len();
                let mut proven_set = Vec::with_capacity(query_group.len() + 1);
                proven_set.push(false_literal);
                proven_set.extend(query_group);
                result.proven_equiv_sets.push(proven_set);
            }
            EquivResult::Disproved(cex) if cex.is_empty() => {
                // Some backends report inequivalence without a model. The
                // group is unresolved because no member can be singled out.
                result.unresolved_candidate_count += query_group.len();
            }
            EquivResult::Disproved(cex) => {
                let simulation = gate_sim::eval(&all_candidates_fn, &cex, gate_sim::Collect::None);
                let values = &simulation.outputs[0];
                let value_by_candidate: HashMap<EquivNode, bool> = normalized_candidates
                    .iter()
                    .enumerate()
                    .map(|(index, &candidate)| (candidate, values.get_bit(index).unwrap()))
                    .collect();
                result.cex_inputs.push(cex);
                let mut refined = VecDeque::new();
                for candidate in query_group.into_iter().chain(pending.drain(..)) {
                    if value_by_candidate[&candidate] {
                        result.disproven_candidate_count += 1;
                    } else {
                        refined.push_back(candidate);
                    }
                }
                pending = refined;
            }
        }
    }
    result.unresolved_candidate_count += pending.len();
    Ok(result)
}

/// Proves normalized simulation-constant candidates in groups with one shared
/// SAT encoding of their combined cone.
pub fn validate_constant_groups_with_backend_and_options(
    gate_fn: &GateFn,
    normalized_candidates: &[EquivNode],
    group_size: usize,
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<ConstantGroupValidationResult, ValidationError> {
    if normalized_candidates.is_empty() {
        return Ok(ConstantGroupValidationResult {
            proven_equiv_sets: Vec::new(),
            cex_inputs: Vec::new(),
            proof_query_count: 0,
            interrupted_group_count: 0,
            proven_candidate_count: 0,
            disproven_candidate_count: 0,
            unresolved_candidate_count: 0,
        });
    }
    match resolve_equivalence_class_backend(backend)? {
        GateFormalBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_constant_groups_with_solver(
                gate_fn,
                normalized_candidates,
                group_size,
                &mut solver,
            )
        }
        GateFormalBackend::Cadical => {
            let mut solver = CadicalSat::new_with_options(options)?;
            validate_constant_groups_with_solver(
                gate_fn,
                normalized_candidates,
                group_size,
                &mut solver,
            )
        }
        GateFormalBackend::Z3 | GateFormalBackend::Ir => {
            validate_constant_groups_with_pairwise_backend(
                gate_fn,
                normalized_candidates,
                group_size,
                backend,
                options,
            )
        }
    }
}

fn validate_full_graph_fraig_with_grouped_constants_with_solver<S: IncrementalSat>(
    gate_fn: &GateFn,
    normalized_constant_candidates: &[EquivNode],
    ordinary_equiv_classes: &[Vec<EquivNode>],
    constant_group_size: usize,
    solver: &mut S,
) -> Result<FullGraphFraigValidationResult, ValidationError> {
    assert!(constant_group_size > 1);
    let solver_build_start = Instant::now();
    let mut frontier: Vec<AigRef> = gate_fn
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .map(|operand| operand.node)
        .collect();
    frontier.push(AigRef { id: 0 });
    let (cone_gates, cone_inputs) = extract_cone(&frontier, &gate_fn.gates);
    let aig_ref_to_lit = build_sat_clauses(solver, &cone_gates, &cone_inputs, &gate_fn.gates);
    let solver_build_seconds = solver_build_start.elapsed().as_secs_f64();
    let all_primary_inputs: HashSet<AigRef> = gate_fn
        .inputs
        .iter()
        .flat_map(|input_vec| input_vec.bit_vector.iter_lsb_to_msb())
        .map(|operand| operand.node)
        .collect();

    let mut classes = VecDeque::new();
    if normalized_constant_candidates.len() > 1 {
        classes.push_back(MutableEquivalenceClass {
            nodes: normalized_constant_candidates.to_vec(),
            is_normalized_constant: true,
        });
    }
    classes.extend(
        ordinary_equiv_classes
            .iter()
            .filter(|equiv_class| equiv_class.len() > 1)
            .cloned()
            .map(|nodes| MutableEquivalenceClass {
                nodes,
                is_normalized_constant: false,
            }),
    );
    let initial_class_histogram = class_size_histogram(
        classes
            .iter()
            .map(|equiv_class| equiv_class.nodes.as_slice()),
    );

    let false_literal = EquivNode::Normal(AigRef { id: 0 });
    let mut virtual_equivalence = VirtualEquivalence::new(gate_fn);
    let mut proven_constant_refs = HashSet::new();
    let mut constant_pending: VecDeque<EquivNode> =
        normalized_constant_candidates.iter().copied().collect();
    let mut all_counterexamples = Vec::new();
    let mut counterexample_seen = HashSet::new();
    let mut constant_group_query_count = 0usize;
    let mut constant_interrupted_group_count = 0usize;
    let mut constant_deferred_candidate_count = 0usize;
    let mut constant_proven_candidate_count = 0usize;
    let mut constant_disproven_candidate_count = 0usize;
    let mut constant_counterexample_count = 0usize;
    let constant_start = Instant::now();

    while constant_pending.len() > 1 {
        let query_size = constant_group_size.min(constant_pending.len());
        let query_group: Vec<EquivNode> = constant_pending.drain(..query_size).collect();
        let activation = solver.sat_new_lit();
        let mut clause = Vec::with_capacity(query_group.len() + 1);
        clause.push(!activation);
        clause.extend(query_group.iter().map(|candidate| {
            let base_lit = aig_ref_to_lit[&candidate.aig_ref()];
            if candidate.is_inverted() {
                !base_lit
            } else {
                base_lit
            }
        }));
        solver.sat_add_clause(&clause);
        constant_group_query_count += 1;

        match solver.sat_solve_assuming(&[activation]) {
            Ok(SatSolveResult::Unsat) => {
                constant_proven_candidate_count += query_group.len();
                for candidate in query_group {
                    proven_constant_refs.insert(candidate.aig_ref());
                    virtual_equivalence.union_equiv_nodes(false_literal, candidate);
                }
            }
            Ok(SatSolveResult::Sat) => {
                let model = solver.sat_model()?;
                let counterexample =
                    solver_model_to_cex(&model, &all_primary_inputs, &aig_ref_to_lit, gate_fn);
                if counterexample_seen.insert(counterexample.clone()) {
                    all_counterexamples.push(counterexample);
                    constant_counterexample_count += 1;
                }
                classes = refine_mutable_classes_by_model(classes, &model, &aig_ref_to_lit);

                let mut refined_pending = VecDeque::new();
                for candidate in query_group.into_iter().chain(constant_pending.drain(..)) {
                    if model_value_for_equiv_node(&model, &aig_ref_to_lit, candidate) {
                        constant_disproven_candidate_count += 1;
                    } else {
                        refined_pending.push_back(candidate);
                    }
                }
                constant_pending = refined_pending;
            }
            Err(ValidationError::CadicalSolveInterrupted) => {
                // An interrupted grouped query is deferred wholesale. Its
                // members remain eligible for node-to-node FRAIG, but are not
                // retried against the constant literal.
                constant_interrupted_group_count += 1;
                constant_deferred_candidate_count += query_group.len();
            }
            Err(error) => return Err(error),
        }
    }
    constant_deferred_candidate_count += constant_pending.len();
    let constant_seconds = constant_start.elapsed().as_secs_f64();

    classes = classes
        .into_iter()
        .filter_map(|mut equiv_class| {
            equiv_class
                .nodes
                .retain(|node| !proven_constant_refs.contains(&node.aig_ref()));
            (equiv_class.nodes.len() > 1).then_some(equiv_class)
        })
        .collect();
    let all_nodes: Vec<AigRef> = gate_fn
        .gates
        .iter()
        .enumerate()
        .map(|(id, _)| AigRef { id })
        .collect();
    let depth_stats = get_gate_depth(gate_fn, &all_nodes);
    let mut classes: Vec<MutableEquivalenceClass> = classes.into();
    classes.sort_unstable_by_key(|equiv_class| {
        let representative = equiv_class.nodes[0];
        (
            !equiv_class.is_normalized_constant,
            equiv_node_depth_key(&depth_stats.ref_to_depth, representative),
            equiv_class.nodes.len(),
        )
    });
    let post_constant_class_histogram = class_size_histogram(
        classes
            .iter()
            .map(|equiv_class| equiv_class.nodes.as_slice()),
    );
    let mut pending: VecDeque<MutableEquivalenceClass> = classes.into();

    let equivalence_start = Instant::now();
    let mut interrupted_relations = HashSet::new();
    let mut equivalence_class_visit_count = 0usize;
    let mut equivalence_proof_query_count = 0usize;
    let mut equivalence_interrupted_proof_count = 0usize;
    let mut interrupted_relation_skip_count = 0usize;
    let mut virtual_proof_count = 0usize;
    let mut equivalence_counterexample_count = 0usize;

    while let Some(equiv_class) = pending.pop_front() {
        equivalence_class_visit_count += 1;
        let representative = equiv_class.nodes[0];
        let mut found_counterexample = false;
        for &candidate in &equiv_class.nodes[1..] {
            if virtual_equivalence.prove_by_structure(gate_fn, representative, candidate) {
                virtual_proof_count += 1;
                continue;
            }
            let relation = CanonicalEquivRelation::new(representative, candidate);
            if interrupted_relations.contains(&relation) {
                interrupted_relation_skip_count += 1;
                continue;
            }

            let miter = add_miter(solver, &aig_ref_to_lit, representative, candidate);
            equivalence_proof_query_count += 1;
            match solver.sat_solve_assuming(&[miter]) {
                Ok(SatSolveResult::Unsat) => {
                    virtual_equivalence.union_equiv_nodes(representative, candidate);
                }
                Ok(SatSolveResult::Sat) => {
                    let model = solver.sat_model()?;
                    let counterexample =
                        solver_model_to_cex(&model, &all_primary_inputs, &aig_ref_to_lit, gate_fn);
                    if counterexample_seen.insert(counterexample.clone()) {
                        all_counterexamples.push(counterexample);
                        equivalence_counterexample_count += 1;
                    }

                    let mut classes_to_refine = VecDeque::new();
                    classes_to_refine.push_back(equiv_class.clone());
                    classes_to_refine.append(&mut pending);
                    pending =
                        refine_mutable_classes_by_model(classes_to_refine, &model, &aig_ref_to_lit);
                    found_counterexample = true;
                    break;
                }
                Err(ValidationError::CadicalSolveInterrupted) => {
                    interrupted_relations.insert(relation);
                    equivalence_interrupted_proof_count += 1;
                }
                Err(error) => return Err(error),
            }
        }
        if found_counterexample {
            continue;
        }
    }
    let equivalence_seconds = equivalence_start.elapsed().as_secs_f64();

    let validation = ValidationResult {
        proven_equiv_sets: virtual_equivalence.into_proven_equiv_sets(),
        cex_inputs: all_counterexamples,
        proof_query_count: constant_group_query_count + equivalence_proof_query_count,
        interrupted_proof_count: constant_interrupted_group_count
            + equivalence_interrupted_proof_count,
    };
    let stat = FullGraphFraigValidationStat {
        solver_build_seconds,
        constant_seconds,
        equivalence_seconds,
        constant_candidate_count: normalized_constant_candidates.len(),
        constant_group_query_count,
        constant_interrupted_group_count,
        constant_deferred_candidate_count,
        constant_proven_candidate_count,
        constant_disproven_candidate_count,
        constant_counterexample_count,
        equivalence_class_visit_count,
        equivalence_proof_query_count,
        equivalence_interrupted_proof_count,
        interrupted_relation_skip_count,
        virtual_proof_count,
        equivalence_counterexample_count,
        initial_class_histogram,
        post_constant_class_histogram,
    };
    Ok(FullGraphFraigValidationResult { validation, stat })
}

/// Runs grouped constant proving followed by ordinary FRAIG using one shared
/// full-live-graph incremental SAT instance.
pub fn validate_full_graph_fraig_with_grouped_constants_and_options(
    gate_fn: &GateFn,
    normalized_constant_candidates: &[EquivNode],
    ordinary_equiv_classes: &[Vec<EquivNode>],
    constant_group_size: usize,
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<FullGraphFraigValidationResult, ValidationError> {
    match resolve_equivalence_class_backend(backend)? {
        GateFormalBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_full_graph_fraig_with_grouped_constants_with_solver(
                gate_fn,
                normalized_constant_candidates,
                ordinary_equiv_classes,
                constant_group_size,
                &mut solver,
            )
        }
        GateFormalBackend::Cadical => {
            let mut solver = CadicalSat::new_with_options(options)?;
            validate_full_graph_fraig_with_grouped_constants_with_solver(
                gate_fn,
                normalized_constant_candidates,
                ordinary_equiv_classes,
                constant_group_size,
                &mut solver,
            )
        }
        GateFormalBackend::Z3 | GateFormalBackend::Ir => Err(ValidationError::UnsupportedBackend {
            backend,
            operation: "single-session full-graph FRAIG",
        }),
    }
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

fn prove_gate_fn_equiv_with_solver<S: IncrementalSat>(
    a: &GateFn,
    b: &GateFn,
    solver: &mut S,
) -> Result<EquivResult, ValidationError> {
    assert_eq!(a.inputs.len(), b.inputs.len());
    assert_eq!(a.outputs.len(), b.outputs.len());

    let mut input_lits = Vec::new();
    for (ia, ib) in a.inputs.iter().zip(b.inputs.iter()) {
        assert_eq!(ia.get_bit_count(), ib.get_bit_count());
        let mut lits = Vec::new();
        for _ in 0..ia.get_bit_count() {
            lits.push(solver.sat_new_lit());
        }
        input_lits.push(lits);
    }

    let (_map_a, outputs_a) = build_gate_fn(solver, a, &input_lits);
    let (_map_b, outputs_b) = build_gate_fn(solver, b, &input_lits);

    // Build XOR miters for each corresponding output bit.
    let mut miters = Vec::new();
    for (la, lb) in outputs_a.iter().zip(outputs_b.iter()) {
        let m = solver.sat_new_lit();
        add_tseitsin_xor(solver, *la, *lb, m);
        miters.push(m);
    }

    // Fresh literal that stands for "outputs differ in *some* bit".
    let diff = solver.sat_new_lit();
    // diff -> OR(miters)  === (!diff OR m1 OR m2 ...)
    let mut clause = Vec::with_capacity(miters.len() + 1);
    clause.push(!diff);
    clause.extend(miters.iter().cloned());
    solver.sat_add_clause(&clause);

    // Ask the solver to find an assignment where outputs differ.
    match solver.sat_solve_assuming(&[diff])? {
        SatSolveResult::Unsat => Ok(EquivResult::Proved), // UNSAT => no way for outputs to differ.
        SatSolveResult::Sat => {
            let model = solver.sat_model()?;
            let mut map = HashMap::new();
            for (i, inp) in a.inputs.iter().enumerate() {
                for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
                    let lit = input_lits[i][j];
                    map.insert(op.node, model.lit_value(lit));
                }
            }
            let cex = a.map_to_inputs(map);
            Ok(EquivResult::Disproved(cex))
        }
    }
}

/// Checks equivalence of two gate functions using a selected backend.
pub fn prove_gate_fn_equiv_with_backend(
    a: &GateFn,
    b: &GateFn,
    backend: GateFormalBackend,
) -> Result<EquivResult, ValidationError> {
    prove_gate_fn_equiv_with_backend_and_options(a, b, backend, GateFormalOptions::default())
}

/// Checks equivalence with backend-specific resource limits.
pub fn prove_gate_fn_equiv_with_backend_and_options(
    a: &GateFn,
    b: &GateFn,
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<EquivResult, ValidationError> {
    match backend {
        GateFormalBackend::Cadical => {
            let mut solver = CadicalSat::new_with_options(options)?;
            prove_gate_fn_equiv_with_solver(a, b, &mut solver)
        }
        GateFormalBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            prove_gate_fn_equiv_with_solver(a, b, &mut solver)
        }
        GateFormalBackend::Z3 => {
            #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
            {
                let mut ctx = crate::prove_gate_fn_equiv_z3::Ctx::new();
                Ok(crate::prove_gate_fn_equiv_z3::prove_gate_fn_equiv(
                    a, b, &mut ctx,
                ))
            }

            #[cfg(not(any(feature = "with-z3-system", feature = "with-z3-built")))]
            {
                Err(ValidationError::UnsupportedBackend {
                    backend,
                    operation: "pairwise gate function equivalence",
                })
            }
        }
        GateFormalBackend::Ir => {
            use crate::check_equivalence::{IrCheckResult, prove_same_gate_fn_via_ir_status};

            match prove_same_gate_fn_via_ir_status(a, b) {
                IrCheckResult::Equivalent => Ok(EquivResult::Proved),
                IrCheckResult::NotEquivalent => Ok(EquivResult::Disproved(Vec::new())),
                other => Err(ValidationError::IrEquivalenceError(format!("{other:?}"))),
            }
        }
    }
}

/// Checks equivalence of two gate functions using a Varisat context.
pub fn prove_gate_fn_equiv_varisat<'a>(
    a: &GateFn,
    b: &GateFn,
    ctx: &mut VarisatCtx<'a>,
) -> EquivResult {
    prove_gate_fn_equiv_with_solver(a, b, &mut ctx.solver).expect("solver error")
}

pub fn validate_equivalence_classes(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
) -> Result<ValidationResult, ValidationError> {
    validate_equivalence_classes_with_backend(gate_fn, equiv_classes, GateFormalBackend::default())
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
        GateFormalBackend::default(),
    )
}

pub fn validate_equivalence_classes_with_backend(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: GateFormalBackend,
) -> Result<ValidationResult, ValidationError> {
    validate_equivalence_classes_with_backend_and_options(
        gate_fn,
        equiv_classes,
        backend,
        GateFormalOptions::default(),
    )
}

/// Validates classes with backend-specific resource limits.
pub fn validate_equivalence_classes_with_backend_and_options(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<ValidationResult, ValidationError> {
    match resolve_equivalence_class_backend(backend)? {
        GateFormalBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ false,
                /* use_virtual_rewrite= */ false,
            )
        }
        GateFormalBackend::Cadical => {
            let mut solver = CadicalSat::new_with_options(options)?;
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ false,
                /* use_virtual_rewrite= */ false,
            )
        }
        GateFormalBackend::Z3 | GateFormalBackend::Ir => {
            validate_equivalence_classes_pairwise_with_backend(
                gate_fn,
                equiv_classes,
                backend,
                /* classes_are_depth_sorted= */ false,
                options,
            )
        }
    }
}

pub fn validate_equivalence_classes_presorted_with_backend(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: GateFormalBackend,
) -> Result<ValidationResult, ValidationError> {
    validate_equivalence_classes_presorted_with_backend_and_options(
        gate_fn,
        equiv_classes,
        backend,
        GateFormalOptions::default(),
    )
}

/// Validates depth-sorted classes with backend-specific resource limits.
pub fn validate_equivalence_classes_presorted_with_backend_and_options(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<ValidationResult, ValidationError> {
    match resolve_equivalence_class_backend(backend)? {
        GateFormalBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ true,
                /* use_virtual_rewrite= */ false,
            )
        }
        GateFormalBackend::Cadical => {
            let mut solver = CadicalSat::new_with_options(options)?;
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ true,
                /* use_virtual_rewrite= */ false,
            )
        }
        GateFormalBackend::Z3 | GateFormalBackend::Ir => {
            validate_equivalence_classes_pairwise_with_backend(
                gate_fn,
                equiv_classes,
                backend,
                /* classes_are_depth_sorted= */ true,
                options,
            )
        }
    }
}

/// Validates depth-sorted classes while carrying proven equivalences forward
/// through a virtual structural rewrite of later AIG nodes.
pub fn validate_equivalence_classes_presorted_with_virtual_rewrite_and_options(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: GateFormalBackend,
    options: GateFormalOptions,
) -> Result<ValidationResult, ValidationError> {
    match resolve_equivalence_class_backend(backend)? {
        GateFormalBackend::Varisat => {
            let mut solver = varisat::Solver::new();
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ true,
                /* use_virtual_rewrite= */ true,
            )
        }
        GateFormalBackend::Cadical => {
            let mut solver = CadicalSat::new_with_options(options)?;
            validate_equivalence_classes_with_solver(
                gate_fn,
                equiv_classes,
                &mut solver,
                /* classes_are_depth_sorted= */ true,
                /* use_virtual_rewrite= */ true,
            )
        }
        GateFormalBackend::Z3 | GateFormalBackend::Ir => {
            validate_equivalence_classes_pairwise_with_backend(
                gate_fn,
                equiv_classes,
                backend,
                /* classes_are_depth_sorted= */ true,
                options,
            )
        }
    }
}

fn gate_fn_with_single_output(
    gate_fn: &GateFn,
    equiv_node: EquivNode,
    output_name: &str,
) -> GateFn {
    gate_fn_with_equiv_outputs(gate_fn, &[equiv_node], output_name)
}

fn gate_fn_with_equiv_outputs(
    gate_fn: &GateFn,
    equiv_nodes: &[EquivNode],
    output_name: &str,
) -> GateFn {
    let operands: Vec<AigOperand> = equiv_nodes
        .iter()
        .map(|equiv_node| {
            let operand = AigOperand::from(equiv_node.aig_ref());
            if equiv_node.is_inverted() {
                operand.negate()
            } else {
                operand
            }
        })
        .collect();
    let mut out = gate_fn.clone();
    out.outputs = vec![Output {
        name: output_name.to_string(),
        bit_vector: AigBitVector::from_lsb_is_index_0(&operands),
    }];
    out
}

fn validate_equivalence_classes_pairwise_with_backend(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    backend: GateFormalBackend,
    classes_are_depth_sorted: bool,
    options: GateFormalOptions,
) -> Result<ValidationResult, ValidationError> {
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

    let mut validation_result = ValidationResult {
        proven_equiv_sets: Vec::new(),
        cex_inputs: Vec::new(),
        proof_query_count: 0,
        interrupted_proof_count: 0,
    };
    for equiv_class in sorted_equiv_classes {
        let mut known_equiv = vec![equiv_class[0]];
        for &candidate in &equiv_class[1..] {
            let representative = known_equiv[0];
            let representative_fn =
                gate_fn_with_single_output(gate_fn, representative, "representative");
            let candidate_fn = gate_fn_with_single_output(gate_fn, candidate, "candidate");
            validation_result.proof_query_count += 1;
            match prove_gate_fn_equiv_with_backend_and_options(
                &representative_fn,
                &candidate_fn,
                backend,
                options,
            )? {
                EquivResult::Proved => known_equiv.push(candidate),
                EquivResult::Disproved(cex) => {
                    if !cex.is_empty() {
                        validation_result.cex_inputs.push(cex);
                    }
                    break;
                }
            }
        }

        if known_equiv.len() > 1 {
            validation_result.proven_equiv_sets.push(known_equiv);
        }
    }
    Ok(validation_result)
}

fn validate_equivalence_classes_with_solver<S: IncrementalSat>(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    solver: &mut S,
    classes_are_depth_sorted: bool,
    use_virtual_rewrite: bool,
) -> Result<ValidationResult, ValidationError> {
    let mut virtual_equivalence = use_virtual_rewrite.then(|| VirtualEquivalence::new(gate_fn));
    let mut virtual_proof_count = 0usize;
    let mut validation_result = validate_equivalence_classes_with_solver_and_virtual_state(
        gate_fn,
        equiv_classes,
        solver,
        classes_are_depth_sorted,
        virtual_equivalence.as_mut(),
        &mut virtual_proof_count,
    )?;
    if let Some(equivalence) = virtual_equivalence {
        validation_result.proven_equiv_sets = equivalence.into_proven_equiv_sets();
        log::info!(
            "virtual FRAIG rewrite avoided {} SAT proof queries",
            virtual_proof_count
        );
    }
    Ok(validation_result)
}

fn validate_equivalence_classes_with_solver_and_virtual_state<S: IncrementalSat>(
    gate_fn: &GateFn,
    equiv_classes: &[&[EquivNode]],
    solver: &mut S,
    classes_are_depth_sorted: bool,
    mut virtual_equivalence: Option<&mut VirtualEquivalence>,
    virtual_proof_count: &mut usize,
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
        proof_query_count: 0,
        interrupted_proof_count: 0,
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
    // found in earlier classes. When a solve finds a new counterexample,
    // re-partition the current bucket with that model and continue with each
    // non-singleton partition. This avoids losing valid equivalences that
    // happened to follow a mismatching candidate in the simulation bucket.
    for equiv_class in sorted_equiv_classes {
        let mut buckets: VecDeque<Vec<EquivNode>> =
            presplit_by_counterexample_models(equiv_class, &counterexample_models, &aig_ref_to_lit)
                .into();
        while let Some(bucket) = buckets.pop_front() {
            let mut known_equiv = vec![bucket[0]];
            let mut split_bucket = false;
            for &candidate in &bucket[1..] {
                // Create a miter between this candidate and the class representative.
                let representative = known_equiv[0];
                if virtual_equivalence
                    .as_mut()
                    .map(|equivalence| {
                        equivalence.prove_by_structure(gate_fn, representative, candidate)
                    })
                    .unwrap_or(false)
                {
                    known_equiv.push(candidate);
                    *virtual_proof_count += 1;
                    continue;
                }
                let miter = add_miter(solver, &aig_ref_to_lit, representative, candidate);

                // Assume the miter output is true, which asks for a counterexample where
                // the candidate is unequal to the representative.
                validation_result.proof_query_count += 1;
                match solver.sat_solve_assuming(&[miter]) {
                    Ok(SatSolveResult::Unsat) => {
                        // No counterexample found, expand the known equivalent set.
                        if let Some(equivalence) = virtual_equivalence.as_mut() {
                            equivalence.union_equiv_nodes(representative, candidate);
                        }
                        known_equiv.push(candidate);
                    }
                    Ok(SatSolveResult::Sat) => {
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
                        buckets.extend(split_bucket_by_model(
                            &bucket,
                            counterexample_models.last().unwrap(),
                            &aig_ref_to_lit,
                        ));
                        split_bucket = true;
                        break;
                    }
                    Err(ValidationError::CadicalSolveInterrupted) => {
                        // A resource-limited proof is inconclusive. Leaving the
                        // candidate out of the proven set preserves correctness.
                        validation_result.interrupted_proof_count += 1;
                    }
                    Err(e) => return Err(e),
                }
            }

            if !split_bucket && known_equiv.len() > 1 {
                validation_result.proven_equiv_sets.push(known_equiv);
            }
        }
    }

    if validation_result.interrupted_proof_count != 0 {
        log::info!(
            "equivalence-class validation skipped {} resource-limited candidate proofs as unproven",
            validation_result.interrupted_proof_count
        );
    }
    Ok(validation_result)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rand::SeedableRng;

    use crate::{
        aig::AigBitVector,
        gate_builder::{GateBuilder, GateBuilderOptions},
        propose_equiv::{EquivNode, propose_equivalence_classes},
        test_utils::{setup_graph_with_redundancies, setup_partially_equiv_graph},
    };

    use super::{
        CadicalSat, CanonicalEquivRelation, GateFormalBackend, GateFormalOptions, IncrementalSat,
        SatModel, SatSolveResult, ValidationError, ValidationResult,
        validate_constant_groups_with_backend_and_options, validate_constant_groups_with_solver,
        validate_equivalence_classes,
        validate_equivalence_classes_presorted_with_virtual_rewrite_and_options,
        validate_equivalence_classes_with_backend, validate_equivalence_classes_with_solver,
        validate_full_graph_fraig_with_grouped_constants_and_options,
        validate_full_graph_fraig_with_grouped_constants_with_solver,
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
    fn test_cadical_timeout_is_reported_as_interrupted() {
        let mut solver = CadicalSat::new_with_options(
            GateFormalOptions::default().with_cadical_timeout(Duration::ZERO),
        )
        .unwrap();
        let lit = solver.sat_new_lit();
        solver.sat_add_clause(&[lit]);
        assert!(matches!(
            solver.sat_solve_assuming(&[lit]),
            Err(ValidationError::CadicalSolveInterrupted)
        ));
    }

    #[test]
    fn test_cadical_terminate_limit_configuration() {
        let options = GateFormalOptions::default().with_cadical_terminate_limit(100);
        assert_eq!(options.cadical_terminate_limit, Some(100));
        let solver = CadicalSat::new_with_options(options).unwrap();
        assert_eq!(solver.terminate_limit, Some(100));

        let unlimited = options.with_cadical_terminate_limit(0);
        assert_eq!(unlimited.cadical_terminate_limit, None);
    }

    #[derive(Clone)]
    struct InterruptingModel;

    impl SatModel<usize> for InterruptingModel {
        fn lit_value(&self, _lit: usize) -> bool {
            false
        }
    }

    struct InterruptingSat {
        next_lit: usize,
    }

    impl IncrementalSat for InterruptingSat {
        type Lit = usize;
        type Model = InterruptingModel;

        fn sat_new_lit(&mut self) -> Self::Lit {
            let lit = self.next_lit;
            self.next_lit += 1;
            lit
        }

        fn sat_add_clause(&mut self, _clause: &[Self::Lit]) {}

        fn sat_solve_assuming(
            &mut self,
            _assumptions: &[Self::Lit],
        ) -> Result<SatSolveResult, ValidationError> {
            Err(ValidationError::CadicalSolveInterrupted)
        }

        fn sat_model(&self) -> Result<Self::Model, ValidationError> {
            Ok(InterruptingModel)
        }
    }

    #[test]
    fn test_interrupted_class_proof_is_treated_as_unproven() {
        let setup = setup_graph_with_redundancies();
        let proposed_class = &[
            EquivNode::Normal(setup.inner0.node),
            EquivNode::Normal(setup.inner1.node),
        ];
        let mut solver = InterruptingSat { next_lit: 0 };

        let result = validate_equivalence_classes_with_solver(
            &setup.g,
            &[proposed_class],
            &mut solver,
            true,
            false,
        )
        .unwrap();

        assert!(result.proven_equiv_sets.is_empty());
        assert!(result.cex_inputs.is_empty());
    }

    #[test]
    fn test_grouped_constant_proof_refines_all_remaining_candidates() {
        let mut builder = GateBuilder::new(
            "grouped_constants".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let input = *builder.add_input("input".to_string(), 1).get_lsb(0);
        let constant_zero = builder.add_and_binary(input, input.negate());
        let constant_one = builder.add_and_binary(constant_zero.negate(), constant_zero.negate());
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[input, constant_zero, constant_one]),
        );
        let gate_fn = builder.build();
        let candidates = [
            EquivNode::Normal(constant_zero.node),
            EquivNode::Inverted(constant_one.node),
            EquivNode::Normal(input.node),
        ];

        let result = validate_constant_groups_with_backend_and_options(
            &gate_fn,
            &candidates,
            8,
            GateFormalBackend::Varisat,
            GateFormalOptions::default(),
        )
        .unwrap();

        assert_eq!(result.proof_query_count, 2);
        assert_eq!(result.proven_candidate_count, 2);
        assert_eq!(result.disproven_candidate_count, 1);
        assert_eq!(result.unresolved_candidate_count, 0);
        assert_eq!(result.cex_inputs.len(), 1);
        assert_eq!(result.proven_equiv_sets.len(), 1);
        assert_eq!(result.proven_equiv_sets[0].len(), 3);
    }

    #[test]
    fn test_grouped_constant_interruption_is_not_retried_individually() {
        let setup = setup_graph_with_redundancies();
        let candidates = [
            EquivNode::Normal(setup.inner0.node),
            EquivNode::Normal(setup.inner1.node),
            EquivNode::Normal(setup.outer0.node),
        ];
        let mut solver = InterruptingSat { next_lit: 0 };

        let result =
            validate_constant_groups_with_solver(&setup.g, &candidates, 8, &mut solver).unwrap();

        assert_eq!(result.proof_query_count, 1);
        assert_eq!(result.interrupted_group_count, 1);
        assert_eq!(result.unresolved_candidate_count, candidates.len());
        assert!(result.proven_equiv_sets.is_empty());
        assert!(result.cex_inputs.is_empty());
    }

    #[test]
    fn canonical_relation_collapses_inverse_and_operand_order() {
        let a = crate::aig::AigRef { id: 3 };
        let b = crate::aig::AigRef { id: 7 };
        let normal = CanonicalEquivRelation::new(EquivNode::Normal(a), EquivNode::Normal(b));

        assert_eq!(
            normal,
            CanonicalEquivRelation::new(EquivNode::Inverted(a), EquivNode::Inverted(b))
        );
        assert_eq!(
            normal,
            CanonicalEquivRelation::new(EquivNode::Normal(b), EquivNode::Normal(a))
        );
        assert_ne!(
            normal,
            CanonicalEquivRelation::new(EquivNode::Normal(a), EquivNode::Inverted(b))
        );
    }

    #[test]
    fn full_graph_constant_counterexample_refines_ordinary_classes() {
        let mut builder = GateBuilder::new(
            "full_graph_global_refinement".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let constant_zero_a = builder.add_and_binary(a, a.negate());
        let constant_zero_b = builder.add_and_binary(b, b.negate());
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[a, b, constant_zero_a, constant_zero_b]),
        );
        let gate_fn = builder.build();
        let constant_candidates = [
            EquivNode::Normal(constant_zero_a.node),
            EquivNode::Normal(constant_zero_b.node),
            EquivNode::Normal(a.node),
        ];
        let ordinary_classes = vec![vec![EquivNode::Normal(a.node), EquivNode::Inverted(a.node)]];

        let result = validate_full_graph_fraig_with_grouped_constants_and_options(
            &gate_fn,
            &constant_candidates,
            &ordinary_classes,
            8,
            GateFormalBackend::Varisat,
            GateFormalOptions::default(),
        )
        .unwrap();

        assert_eq!(result.stat.constant_group_query_count, 2);
        assert_eq!(result.stat.constant_counterexample_count, 1);
        assert_eq!(result.stat.constant_proven_candidate_count, 2);
        assert_eq!(result.stat.constant_disproven_candidate_count, 1);
        assert_eq!(result.stat.equivalence_proof_query_count, 0);
        assert!(result.stat.post_constant_class_histogram.is_empty());
    }

    #[test]
    fn full_graph_fraig_does_not_retry_interrupted_inverse_relation() {
        let mut builder = GateBuilder::new(
            "full_graph_no_retry".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[a, b]),
        );
        let gate_fn = builder.build();
        let ordinary_classes = vec![
            vec![EquivNode::Normal(a.node), EquivNode::Normal(b.node)],
            vec![EquivNode::Inverted(a.node), EquivNode::Inverted(b.node)],
        ];
        let mut solver = InterruptingSat { next_lit: 0 };

        let result = validate_full_graph_fraig_with_grouped_constants_with_solver(
            &gate_fn,
            &[],
            &ordinary_classes,
            8,
            &mut solver,
        )
        .unwrap();

        assert_eq!(result.stat.equivalence_proof_query_count, 1);
        assert_eq!(result.stat.equivalence_interrupted_proof_count, 1);
        assert_eq!(result.stat.interrupted_relation_skip_count, 1);
    }

    #[test]
    fn full_graph_fraig_defers_an_interrupted_constant_group_wholesale() {
        let setup = setup_graph_with_redundancies();
        let candidates = [
            EquivNode::Normal(setup.inner0.node),
            EquivNode::Normal(setup.inner1.node),
            EquivNode::Normal(setup.outer0.node),
        ];
        let mut solver = InterruptingSat { next_lit: 0 };

        let result = validate_full_graph_fraig_with_grouped_constants_with_solver(
            &setup.g,
            &candidates,
            &[],
            8,
            &mut solver,
        )
        .unwrap();

        assert_eq!(result.stat.constant_group_query_count, 1);
        assert_eq!(result.stat.constant_interrupted_group_count, 1);
        assert_eq!(result.stat.constant_deferred_candidate_count, 3);
        // The deferred candidates remain eligible for ordinary node-to-node
        // FRAIG, but no candidate-to-literal singleton query is issued.
        assert_eq!(result.stat.equivalence_proof_query_count, 1);
    }

    #[test]
    fn test_validate_equiv_graph_with_redundancies() {
        let setup = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let counterexamples = Vec::new();
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
    fn test_validate_repartitions_current_class_after_counterexample() {
        let setup = setup_graph_with_redundancies();
        let proposed_class = &[
            EquivNode::Normal(setup.inner0.node),
            EquivNode::Normal(setup.inner1.node),
            EquivNode::Normal(setup.outer0.node),
            EquivNode::Normal(setup.outer1.node),
        ];

        let result = validate_equivalence_classes(&setup.g, &[proposed_class]).unwrap();
        assert_eq!(
            canonical_proven_sets(&result),
            vec![
                vec![
                    EquivNode::Normal(setup.inner0.node),
                    EquivNode::Normal(setup.inner1.node),
                ],
                vec![
                    EquivNode::Normal(setup.outer0.node),
                    EquivNode::Normal(setup.outer1.node),
                ],
            ]
        );
        assert_eq!(result.cex_inputs.len(), 1);
    }

    #[test]
    fn test_cadical_matches_varisat_on_redundant_graph() {
        let setup = setup_graph_with_redundancies();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(0);
        let counterexamples = Vec::new();
        let equiv_classes =
            propose_equivalence_classes(&setup.g, 16, &mut seeded_rng, &counterexamples);
        let classes: Vec<&[EquivNode]> = equiv_classes
            .values()
            .map(|nodes| nodes.as_slice())
            .collect();

        let varisat = validate_equivalence_classes_with_backend(
            &setup.g,
            &classes,
            GateFormalBackend::Varisat,
        )
        .unwrap();
        let cadical = validate_equivalence_classes_with_backend(
            &setup.g,
            &classes,
            GateFormalBackend::Cadical,
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
            GateFormalBackend::Varisat,
        )
        .unwrap();
        let cadical = validate_equivalence_classes_with_backend(
            &setup.g,
            &[proposed_class],
            GateFormalBackend::Cadical,
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

    #[test]
    fn test_virtual_rewrite_avoids_downstream_sat_query() {
        let mut builder =
            GateBuilder::new("virtual_rewrite".to_string(), GateBuilderOptions::no_opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let z = *builder.add_input("z".to_string(), 1).get_lsb(0);
        let x1 = builder.add_and_binary(a, b);
        let x2 = builder.add_and_binary(x1, a);
        let y1 = builder.add_and_binary(x1, z);
        let y2 = builder.add_and_binary(x2, z);
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[y1, y2]),
        );
        let gate_fn = builder.build();
        let x_class = [EquivNode::Normal(x1.node), EquivNode::Normal(x2.node)];
        let y_class = [EquivNode::Normal(y1.node), EquivNode::Normal(y2.node)];

        let result = validate_equivalence_classes_presorted_with_virtual_rewrite_and_options(
            &gate_fn,
            &[&x_class, &y_class],
            GateFormalBackend::Varisat,
            GateFormalOptions::default(),
        )
        .unwrap();

        assert_eq!(result.proof_query_count, 1);
        assert!(canonical_proven_sets(&result).contains(&vec![
            EquivNode::Normal(y1.node),
            EquivNode::Normal(y2.node),
        ]));
    }
}
