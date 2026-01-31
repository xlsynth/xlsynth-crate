// SPDX-License-Identifier: Apache-2.0

//! Candidate discovery for mapping PIR node output bits to AIG node outputs.
//!
//! This module intentionally starts with a simulation-only heuristic:
//! - Run PIR interpretation over random inputs and record per-node bit
//!   histories.
//! - Run GateFn simulation over the same inputs and record per-node bit
//!   histories.
//! - Match identical or inverted histories to propose candidate
//!   correspondences.
//!
//! A later step can use a solver to confirm or refute each candidate.

use std::collections::HashMap;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::rand_core::SeedableRng;

use xlsynth::IrBits;
use xlsynth::IrValue;

use xlsynth_pir::ir;
use xlsynth_pir::ir_eval;
use xlsynth_pir::ir_utils::is_structural_payload;

use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::aig::topo::topo_sort_refs;
use crate::gatify::ir2gate::{GatifyOptions, gatify};
use crate::ir2gate_utils::AdderMapping;
use varisat::ExtendFormula;
use varisat::Solver;

#[derive(Debug, Clone)]
pub struct IrAigSharingOptions {
    pub sample_count: usize,
    pub sample_seed: u64,
    /// When true, omit PIR nodes whose payload is classified as "structural".
    pub exclude_structural_pir_nodes: bool,
}

impl Default for IrAigSharingOptions {
    fn default() -> Self {
        Self {
            sample_count: 256,
            sample_seed: 0,
            exclude_structural_pir_nodes: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrAigEquivalenceCandidate {
    pub pir_node_ref: ir::NodeRef,
    pub pir_node_text_id: usize,
    pub bit_index: usize,
    pub rhs: IrAigCandidateRhs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrAigCandidateRhs {
    AigOperand(AigOperand),
    Const(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateProofResult {
    /// Proved that the PIR node bit equals the given AIG operand for all
    /// inputs.
    Proved,
    /// Disproved with a concrete counterexample input assignment.
    Disproved { counterexample_inputs: Vec<IrBits> },
    /// Skipped proof attempt (e.g. missing lowering map entry).
    Skipped { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateProof {
    pub candidate: IrAigEquivalenceCandidate,
    pub result: CandidateProofResult,
}

#[derive(Debug)]
struct PirNodeHistory {
    node_ref: ir::NodeRef,
    node_text_id: usize,
    bit_count: usize,
    /// `bit_histories[i][k]` is bit i (LSB=0) at sample k.
    bit_histories: Vec<Vec<bool>>,
}

#[derive(Debug, Default)]
struct PirHistories {
    by_text_id: HashMap<usize, PirNodeHistory>,
}

impl PirHistories {
    fn push_node_value(
        &mut self,
        f: &ir::Fn,
        node_ref: ir::NodeRef,
        node_text_id: usize,
        value: &IrValue,
    ) -> Result<(), String> {
        let ty = f.get_node_ty(node_ref);
        let mut flat_bits: Vec<bool> = Vec::with_capacity(ty.bit_count());
        flatten_ir_value_to_lsb0_bits(value, ty, &mut flat_bits)?;

        let bit_count = ty.bit_count();
        if flat_bits.len() != bit_count {
            return Err(format!(
                "internal error: flattened bit count mismatch for node_text_id={}: got {} expected {}",
                node_text_id,
                flat_bits.len(),
                bit_count
            ));
        }

        let entry = self.by_text_id.entry(node_text_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut bit_histories = Vec::with_capacity(bit_count);
                for bit in &flat_bits {
                    bit_histories.push(vec![*bit]);
                }
                v.insert(PirNodeHistory {
                    node_ref,
                    node_text_id,
                    bit_count,
                    bit_histories,
                });
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let hist = o.get_mut();
                if hist.bit_count != bit_count {
                    return Err(format!(
                        "PIR node bit_count changed across samples for node_text_id={}: {} -> {}",
                        node_text_id, hist.bit_count, bit_count
                    ));
                }
                for (i, bit) in flat_bits.iter().enumerate() {
                    hist.bit_histories[i].push(*bit);
                }
            }
        }

        Ok(())
    }
}

struct PirHistoryObserver<'a> {
    f: &'a ir::Fn,
    histories: &'a mut PirHistories,
    exclude_structural: bool,
}

impl ir_eval::EvalObserver for PirHistoryObserver<'_> {
    fn on_select(&mut self, _ev: ir_eval::SelectEvent) {}

    fn on_node_value(&mut self, node_ref: ir::NodeRef, node_text_id: usize, value: &IrValue) {
        if self.exclude_structural {
            let node = self.f.get_node(node_ref);
            if is_structural_payload(&node.payload) {
                return;
            }
        }

        // Any error here indicates an internal mismatch (type/value mismatch),
        // so we surface it via panic for now. This can be refined later.
        self.histories
            .push_node_value(self.f, node_ref, node_text_id, value)
            .unwrap();
    }
}

pub fn get_equivalences(
    pir_pkg: &ir::Package,
    pir_fn: &ir::Fn,
    gate_fn: &GateFn,
    options: &IrAigSharingOptions,
) -> Result<Vec<IrAigEquivalenceCandidate>, String> {
    if gate_fn.inputs.len() != pir_fn.params.len() {
        return Err(format!(
            "input arity mismatch: PIR params={} vs GateFn inputs={}",
            pir_fn.params.len(),
            gate_fn.inputs.len()
        ));
    }
    for (i, (p, g_in)) in pir_fn.params.iter().zip(gate_fn.inputs.iter()).enumerate() {
        let pir_w = p.ty.bit_count();
        let gate_w = g_in.get_bit_count();
        if pir_w != gate_w {
            return Err(format!(
                "input width mismatch at index {} ('{}'): PIR bits[{}] vs GateFn bits[{}]",
                i, p.name, pir_w, gate_w
            ));
        }
    }

    if options.sample_count == 0 {
        return Ok(Vec::new());
    }

    // Candidate discovery via shared-input simulation.
    //
    // Important: we must drive PIR and GateFn with the *same* bit patterns in
    // the same flattened order (LSB=0), otherwise history matching is meaningless
    // for tuple/array parameters.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(options.sample_seed);

    // PIR node histories keyed by node_text_id.
    let mut pir_histories = PirHistories::default();

    // GateFn per-node histories keyed by AigRef.id (positive polarity).
    let gate_count = gate_fn.gates.len();
    let mut gate_node_histories: Vec<Vec<bool>> = (0..gate_count)
        .map(|_| Vec::with_capacity(options.sample_count))
        .collect();

    for _sample_idx in 0..options.sample_count {
        let (pir_args, gate_inputs) = make_random_args_for_both(pir_fn, gate_fn, &mut rng)?;

        let mut observer = PirHistoryObserver {
            f: pir_fn,
            histories: &mut pir_histories,
            exclude_structural: options.exclude_structural_pir_nodes,
        };
        match ir_eval::eval_fn_in_package_with_observer(
            pir_pkg,
            pir_fn,
            &pir_args,
            Some(&mut observer),
        ) {
            ir_eval::FnEvalResult::Success(_) => {}
            ir_eval::FnEvalResult::Failure(fail) => {
                return Err(format!(
                    "PIR evaluation failed while sampling: assertion_failures={} trace_messages={}",
                    fail.assertion_failures.len(),
                    fail.trace_messages.len()
                ));
            }
        }

        let values = eval_gate_fn_all_node_values_positive(gate_fn, &gate_inputs)?;
        for (id, v) in values.into_iter().enumerate() {
            gate_node_histories[id].push(v);
        }
    }

    // Build signature->operands index from GateFn histories.
    let mut sig_to_operands: HashMap<Vec<u8>, Vec<AigOperand>> = HashMap::new();
    for (node_id, hist) in gate_node_histories.iter().enumerate() {
        let node = AigRef { id: node_id };
        let sig = pack_bools_to_bytes(hist);
        sig_to_operands
            .entry(sig.clone())
            .or_default()
            .push(AigOperand {
                node,
                negated: false,
            });
        let inv = invert_packed_signature(&sig, options.sample_count);
        sig_to_operands.entry(inv).or_default().push(AigOperand {
            node,
            negated: true,
        });
    }

    // --- 3) Match PIR bit histories against the GateFn index.
    let mut candidates: Vec<IrAigEquivalenceCandidate> = Vec::new();
    // Deterministic iteration for deterministic output.
    let mut pir_text_ids: Vec<usize> = pir_histories.by_text_id.keys().copied().collect();
    pir_text_ids.sort_unstable();
    for tid in pir_text_ids {
        let hist = pir_histories
            .by_text_id
            .get(&tid)
            .expect("text_id collected from keys must exist");
        for (bit_index, bit_hist) in hist.bit_histories.iter().enumerate() {
            if bit_hist.len() != options.sample_count {
                return Err(format!(
                    "internal error: PIR history length mismatch for node_text_id={} bit_index={}: got {} expected {}",
                    hist.node_text_id,
                    bit_index,
                    bit_hist.len(),
                    options.sample_count
                ));
            }

            // If the observed history is constant, prefer generating a single
            // "prove it's a constant" candidate over matching to arbitrary AIG
            // nodes that merely *appear* constant over this sample set.
            if let Some(const_value) = is_all_same_bool(bit_hist) {
                candidates.push(IrAigEquivalenceCandidate {
                    pir_node_ref: hist.node_ref,
                    pir_node_text_id: hist.node_text_id,
                    bit_index,
                    rhs: IrAigCandidateRhs::Const(const_value),
                });
                continue;
            }

            let sig = pack_bools_to_bytes(bit_hist);
            if let Some(ops) = sig_to_operands.get(&sig) {
                for op in ops {
                    candidates.push(IrAigEquivalenceCandidate {
                        pir_node_ref: hist.node_ref,
                        pir_node_text_id: hist.node_text_id,
                        bit_index,
                        rhs: IrAigCandidateRhs::AigOperand(*op),
                    });
                }
            }
        }
    }

    candidates.sort_by(|a, b| {
        let rhs_key = |rhs: &IrAigCandidateRhs| match rhs {
            IrAigCandidateRhs::Const(false) => (0u8, 0usize, false),
            IrAigCandidateRhs::Const(true) => (0u8, 0usize, true),
            IrAigCandidateRhs::AigOperand(op) => (1u8, op.node.id, op.negated),
        };
        (a.pir_node_ref.index, a.bit_index, rhs_key(&a.rhs)).cmp(&(
            b.pir_node_ref.index,
            b.bit_index,
            rhs_key(&b.rhs),
        ))
    });

    Ok(candidates)
}

pub fn confirm_or_deny_candidate_equivalence(
    pir_fn: &ir::Fn,
    gate_fn: &GateFn,
    candidate: &IrAigEquivalenceCandidate,
) -> Result<bool, String> {
    let opts = GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::default(),
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
    };
    let proofs =
        prove_equivalence_candidates_varisat(pir_fn, gate_fn, &[candidate.clone()], &opts)?;
    match &proofs[0].result {
        CandidateProofResult::Proved => Ok(true),
        CandidateProofResult::Disproved { .. } => Ok(false),
        CandidateProofResult::Skipped { reason } => Err(reason.clone()),
    }
}

/// Proves (or disproves) equivalence for all provided candidates using Varisat.
///
/// This is intended to be the "bulk" confirmation step after simulation-based
/// candidate discovery: we encode both circuits once under shared inputs, then
/// query each candidate via an XOR miter under an assumption.
pub fn prove_equivalence_candidates_varisat(
    pir_fn: &ir::Fn,
    gate_fn: &GateFn,
    candidates: &[IrAigEquivalenceCandidate],
    gatify_options: &GatifyOptions,
) -> Result<Vec<CandidateProof>, String> {
    prove_equivalence_candidates_varisat_streaming(
        pir_fn,
        gate_fn,
        candidates,
        gatify_options,
        |_p| {},
    )
}

/// Streaming variant of `prove_equivalence_candidates_varisat`.
///
/// Calls `on_proof` as each candidate is proved/disproved/skipped, in the same
/// order as `candidates`.
pub fn prove_equivalence_candidates_varisat_streaming<F>(
    pir_fn: &ir::Fn,
    gate_fn: &GateFn,
    candidates: &[IrAigEquivalenceCandidate],
    gatify_options: &GatifyOptions,
    mut on_proof: F,
) -> Result<Vec<CandidateProof>, String>
where
    F: FnMut(&CandidateProof),
{
    // Gatify PIR once to get a GateFn and a per-node lowering map.
    let gatify_output = gatify(pir_fn, gatify_options.clone())?;
    let pir_gate_fn = gatify_output.gate_fn;
    let lowering_map = gatify_output.lowering_map;

    // Ensure input shapes match so a single shared input vector drives both.
    if pir_gate_fn.inputs.len() != gate_fn.inputs.len() {
        return Err(format!(
            "gate input arity mismatch: pir_gate_fn has {} inputs but gate_fn has {}",
            pir_gate_fn.inputs.len(),
            gate_fn.inputs.len()
        ));
    }
    for (i, (a, b)) in pir_gate_fn
        .inputs
        .iter()
        .zip(gate_fn.inputs.iter())
        .enumerate()
    {
        if a.get_bit_count() != b.get_bit_count() {
            return Err(format!(
                "gate input width mismatch at input {}: pir_gate_fn bits[{}] vs gate_fn bits[{}]",
                i,
                a.get_bit_count(),
                b.get_bit_count()
            ));
        }
    }

    let mut solver = Solver::new();

    // Dedicated constant literals (so we can prove "PIR bit is constant 0/1").
    let const_true = solver.new_lit();
    solver.add_clause(&[const_true]);
    let const_false = solver.new_lit();
    solver.add_clause(&[!const_false]);

    // Shared input literals: [input_port][bit_index_lsb0]
    let mut input_lits: Vec<Vec<varisat::Lit>> = Vec::with_capacity(gate_fn.inputs.len());
    for inp in gate_fn.inputs.iter() {
        let mut bits: Vec<varisat::Lit> = Vec::with_capacity(inp.get_bit_count());
        for _ in 0..inp.get_bit_count() {
            bits.push(solver.new_lit());
        }
        input_lits.push(bits);
    }

    let pir_lits = encode_gate_fn_all_nodes(&mut solver, &pir_gate_fn, &input_lits)?;
    let gate_lits = encode_gate_fn_all_nodes(&mut solver, gate_fn, &input_lits)?;

    // GateFn primary input refs for counterexample reconstruction.
    let gate_primary_inputs: Vec<AigRef> = gate_fn
        .inputs
        .iter()
        .flat_map(|inp| inp.bit_vector.iter_lsb_to_msb())
        .map(|op| op.node)
        .collect();

    let mut results: Vec<CandidateProof> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let Some(pir_bv) = lowering_map.get(&cand.pir_node_ref) else {
            let proof = CandidateProof {
                candidate: cand.clone(),
                result: CandidateProofResult::Skipped {
                    reason: format!(
                        "missing lowering_map entry for pir_node_text_id={}",
                        cand.pir_node_text_id
                    ),
                },
            };
            on_proof(&proof);
            results.push(proof);
            continue;
        };
        if cand.bit_index >= pir_bv.get_bit_count() {
            let proof = CandidateProof {
                candidate: cand.clone(),
                result: CandidateProofResult::Skipped {
                    reason: format!(
                        "bit_index {} out of range for pir node bits[{}] (node_text_id={})",
                        cand.bit_index,
                        pir_bv.get_bit_count(),
                        cand.pir_node_text_id
                    ),
                },
            };
            on_proof(&proof);
            results.push(proof);
            continue;
        }

        let pir_op: AigOperand = *pir_bv.get_lsb(cand.bit_index);
        let pir_lit = lit_for_operand(&pir_lits, pir_op)?;
        let gate_lit = match cand.rhs {
            IrAigCandidateRhs::AigOperand(op) => lit_for_operand(&gate_lits, op)?,
            IrAigCandidateRhs::Const(true) => const_true,
            IrAigCandidateRhs::Const(false) => const_false,
        };

        // Ask the solver for a counterexample: (pir != gate).
        let diff = solver.new_lit();
        add_tseitsin_xor(&mut solver, pir_lit, gate_lit, diff);
        solver.assume(&[diff]);
        let sat = solver
            .solve()
            .map_err(|e| format!("varisat solve error: {e:?}"))?;
        if !sat {
            let proof = CandidateProof {
                candidate: cand.clone(),
                result: CandidateProofResult::Proved,
            };
            on_proof(&proof);
            results.push(proof);
            continue;
        }

        let model = solver
            .model()
            .ok_or_else(|| "expected model when SAT".to_string())?;
        let model_set: std::collections::HashSet<varisat::Lit> = model.iter().cloned().collect();

        let mut input_assignment: HashMap<AigRef, bool> = HashMap::new();
        for aig_ref in &gate_primary_inputs {
            let lit = gate_lits
                .get(aig_ref)
                .ok_or_else(|| format!("missing lit for gate primary input {:?}", aig_ref))?;
            input_assignment.insert(*aig_ref, model_set.contains(lit));
        }

        let cex_inputs: Vec<IrBits> = gate_fn.map_to_inputs(input_assignment);
        let proof = CandidateProof {
            candidate: cand.clone(),
            result: CandidateProofResult::Disproved {
                counterexample_inputs: cex_inputs,
            },
        };
        on_proof(&proof);
        results.push(proof);
    }

    Ok(results)
}

fn is_all_same_bool(bits: &[bool]) -> Option<bool> {
    let first = *bits.first()?;
    if bits.iter().all(|b| *b == first) {
        Some(first)
    } else {
        None
    }
}

fn encode_gate_fn_all_nodes(
    solver: &mut Solver,
    gate_fn: &GateFn,
    input_lits: &[Vec<varisat::Lit>],
) -> Result<HashMap<AigRef, varisat::Lit>, String> {
    if gate_fn.inputs.len() != input_lits.len() {
        return Err(format!(
            "input_lits arity mismatch: got {} expected {}",
            input_lits.len(),
            gate_fn.inputs.len()
        ));
    }

    let mut map: HashMap<AigRef, varisat::Lit> = HashMap::new();

    // Seed primary inputs using the shared literals (one per input bit).
    for (i, inp) in gate_fn.inputs.iter().enumerate() {
        if inp.get_bit_count() != input_lits[i].len() {
            return Err(format!(
                "input_lits width mismatch for input {} ('{}'): got {} expected {}",
                i,
                inp.name,
                input_lits[i].len(),
                inp.get_bit_count()
            ));
        }
        for (j, op) in inp.bit_vector.iter_lsb_to_msb().enumerate() {
            if op.negated {
                return Err("gate_fn primary input operands should not be negated".to_string());
            }
            map.insert(op.node, input_lits[i][j]);
        }
    }

    // Allocate literals for all non-input nodes.
    for (id, node) in gate_fn.gates.iter().enumerate() {
        let r = AigRef { id };
        if matches!(node, AigNode::Input { .. }) {
            if !map.contains_key(&r) {
                return Err(format!(
                    "AigNode::Input id={} not present in gate_fn.inputs mapping",
                    id
                ));
            }
            continue;
        }
        map.entry(r).or_insert_with(|| solver.new_lit());
    }

    // Add structural clauses for all nodes.
    for (id, node) in gate_fn.gates.iter().enumerate() {
        let r = AigRef { id };
        let out = *map.get(&r).expect("lit allocated for node");
        match node {
            AigNode::Input { .. } => {}
            AigNode::Literal(v) => {
                if *v {
                    solver.add_clause(&[out]);
                } else {
                    solver.add_clause(&[!out]);
                }
            }
            AigNode::And2 { a, b, .. } => {
                let a_lit = lit_for_operand(&map, *a)?;
                let b_lit = lit_for_operand(&map, *b)?;
                add_tseitsin_and(solver, a_lit, b_lit, out);
            }
        }
    }

    Ok(map)
}

fn lit_for_operand(
    map: &HashMap<AigRef, varisat::Lit>,
    op: AigOperand,
) -> Result<varisat::Lit, String> {
    let base = *map
        .get(&op.node)
        .ok_or_else(|| format!("missing lit for AigRef id={}", op.node.id))?;
    Ok(if op.negated { !base } else { base })
}

fn add_tseitsin_and(
    solver: &mut impl ExtendFormula,
    a: varisat::Lit,
    b: varisat::Lit,
    output: varisat::Lit,
) {
    solver.add_clause(&[!a, !b, output]);
    solver.add_clause(&[a, !output]);
    solver.add_clause(&[b, !output]);
}

fn add_tseitsin_xor(
    solver: &mut impl ExtendFormula,
    a: varisat::Lit,
    b: varisat::Lit,
    output: varisat::Lit,
) {
    solver.add_clause(&[!a, !b, !output]);
    solver.add_clause(&[a, b, !output]);
    solver.add_clause(&[a, !b, output]);
    solver.add_clause(&[!a, b, output]);
}

fn eval_gate_fn_all_node_values_positive(
    gate_fn: &GateFn,
    inputs: &[IrBits],
) -> Result<Vec<bool>, String> {
    if inputs.len() != gate_fn.inputs.len() {
        return Err(format!(
            "gate sim input arity mismatch: got {} expected {}",
            inputs.len(),
            gate_fn.inputs.len()
        ));
    }

    let gate_count = gate_fn.gates.len();
    let mut values: Vec<Option<bool>> = vec![None; gate_count];

    // Seed primary inputs.
    for (inp_value, inp_decl) in inputs.iter().zip(gate_fn.inputs.iter()) {
        for (bit_index, op) in inp_decl.bit_vector.iter_lsb_to_msb().enumerate() {
            if op.negated {
                return Err("GateFn input operands should not be negated".to_string());
            }
            let b = inp_value
                .get_bit(bit_index)
                .map_err(|e| format!("failed to read input bit {}: {}", bit_index, e))?;
            values[op.node.id] = Some(b);
        }
    }

    let topo = topo_sort_refs(&gate_fn.gates);
    for r in topo {
        match &gate_fn.gates[r.id] {
            AigNode::Input { .. } => {
                // Should already be seeded.
                if values[r.id].is_none() {
                    values[r.id] = Some(false);
                }
            }
            AigNode::Literal(v) => {
                values[r.id] = Some(*v);
            }
            AigNode::And2 { a, b, .. } => {
                let a_base = values[a.node.id].ok_or_else(|| {
                    format!("unseeded AIG value for a operand node {}", a.node.id)
                })?;
                let b_base = values[b.node.id].ok_or_else(|| {
                    format!("unseeded AIG value for b operand node {}", b.node.id)
                })?;
                let a_val = if a.negated { !a_base } else { a_base };
                let b_val = if b.negated { !b_base } else { b_base };
                values[r.id] = Some(a_val && b_val);
            }
        }
    }

    Ok(values
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.ok_or_else(|| format!("missing gate sim value for node {}", i)))
        .collect::<Result<Vec<bool>, String>>()?)
}

fn make_random_args_for_both(
    pir_fn: &ir::Fn,
    gate_fn: &GateFn,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(Vec<IrValue>, Vec<IrBits>), String> {
    let mut pir_args: Vec<IrValue> = Vec::with_capacity(pir_fn.params.len());
    let mut gate_inputs: Vec<IrBits> = Vec::with_capacity(gate_fn.inputs.len());

    for (param, gate_input) in pir_fn.params.iter().zip(gate_fn.inputs.iter()) {
        // Generate the *flat* bitvector first, then unflatten into an IrValue.
        // This ensures the PIR evaluator and the GateFn see identical bit patterns
        // in the same flattened order for tuple/array parameters.
        let flat_bits = random_bool_vec(rng, param.ty.bit_count());
        let v = unflatten_ir_value_from_lsb0_bits(&param.ty, &flat_bits)?;
        if flat_bits.len() != gate_input.get_bit_count() {
            return Err(format!(
                "flattened arg width mismatch for param '{}': got {} expected {}",
                param.name,
                flat_bits.len(),
                gate_input.get_bit_count()
            ));
        }
        pir_args.push(v);
        gate_inputs.push(IrBits::from_lsb_is_0(&flat_bits));
    }

    Ok((pir_args, gate_inputs))
}

fn random_bool_vec(rng: &mut impl RngCore, width: usize) -> Vec<bool> {
    let mut bits: Vec<bool> = Vec::with_capacity(width);
    for _ in 0..width {
        bits.push((rng.next_u32() & 1) != 0);
    }
    bits
}

fn unflatten_ir_value_from_lsb0_bits(ty: &ir::Type, flat_bits: &[bool]) -> Result<IrValue, String> {
    let (v, used) = unflatten_ir_value_from_lsb0_bits_at(ty, flat_bits, 0)?;
    if used != flat_bits.len() {
        return Err(format!(
            "unflatten did not consume all bits: used {} of {}",
            used,
            flat_bits.len()
        ));
    }
    Ok(v)
}

fn unflatten_ir_value_from_lsb0_bits_at(
    ty: &ir::Type,
    flat_bits: &[bool],
    mut offset: usize,
) -> Result<(IrValue, usize), String> {
    match ty {
        ir::Type::Token => Ok((IrValue::make_token(), offset)),
        ir::Type::Bits(w) => {
            let w = *w;
            if offset + w > flat_bits.len() {
                return Err("not enough bits to unflatten bits value".to_string());
            }
            let slice = &flat_bits[offset..offset + w];
            offset += w;
            Ok((IrValue::from_bits(&IrBits::from_lsb_is_0(slice)), offset))
        }
        ir::Type::Tuple(types) => {
            // Flattening places the *last* tuple element at the least-significant bits.
            // So when unflattening, consume elements from the tail first.
            let mut elems_rev: Vec<IrValue> = Vec::with_capacity(types.len());
            for t in types.iter().rev() {
                let (v, next) = unflatten_ir_value_from_lsb0_bits_at(t, flat_bits, offset)?;
                offset = next;
                elems_rev.push(v);
            }
            elems_rev.reverse();
            Ok((IrValue::make_tuple(&elems_rev), offset))
        }
        ir::Type::Array(ir::ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let mut elems_rev: Vec<IrValue> = Vec::with_capacity(*element_count);
            for _ in 0..*element_count {
                let (v, next) =
                    unflatten_ir_value_from_lsb0_bits_at(element_type, flat_bits, offset)?;
                offset = next;
                elems_rev.push(v);
            }
            elems_rev.reverse();
            let arr = IrValue::make_array(&elems_rev).map_err(|e| e.to_string())?;
            Ok((arr, offset))
        }
    }
}

fn flatten_ir_value_to_lsb0_bits(
    v: &IrValue,
    ty: &ir::Type,
    out: &mut Vec<bool>,
) -> Result<(), String> {
    match ty {
        ir::Type::Token => Ok(()),
        ir::Type::Bits(w) => {
            let bits = v.to_bits().map_err(|e| e.to_string())?;
            if bits.get_bit_count() != *w {
                return Err(format!(
                    "bits width mismatch: value has bits[{}] but type is bits[{}]",
                    bits.get_bit_count(),
                    w
                ));
            }
            for i in 0..*w {
                out.push(bits.get_bit(i).map_err(|e| e.to_string())?);
            }
            Ok(())
        }
        ir::Type::Tuple(types) => {
            let elems = v.get_elements().map_err(|e| e.to_string())?;
            if elems.len() != types.len() {
                return Err(format!(
                    "tuple arity mismatch: value has {} elems but type expects {}",
                    elems.len(),
                    types.len()
                ));
            }
            // PIR/XLS tuple flattening: last element occupies least-significant bits.
            for (elem, elem_ty) in elems.iter().rev().zip(types.iter().rev()) {
                flatten_ir_value_to_lsb0_bits(elem, elem_ty, out)?;
            }
            Ok(())
        }
        ir::Type::Array(ir::ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let got_count = v.get_element_count().map_err(|e| e.to_string())?;
            if got_count != *element_count {
                return Err(format!(
                    "array length mismatch: value has {} elems but type expects {}",
                    got_count, element_count
                ));
            }
            // Flatten arrays similarly: element (N-1) occupies least-significant bits.
            for i in (0..*element_count).rev() {
                let elem = v.get_element(i).map_err(|e| e.to_string())?;
                flatten_ir_value_to_lsb0_bits(&elem, element_type, out)?;
            }
            Ok(())
        }
    }
}

fn pack_bools_to_bytes(bits: &[bool]) -> Vec<u8> {
    let byte_len = (bits.len() + 7) / 8;
    let mut out = vec![0u8; byte_len];
    for (i, b) in bits.iter().enumerate() {
        if *b {
            out[i / 8] |= 1u8 << (i % 8);
        }
    }
    out
}

fn invert_packed_signature(sig: &[u8], bit_len: usize) -> Vec<u8> {
    let mut out: Vec<u8> = sig.iter().map(|b| !b).collect();
    let rem = bit_len % 8;
    if rem != 0 {
        let mask = (1u8 << rem) - 1;
        if let Some(last) = out.last_mut() {
            *last &= mask;
        }
    }
    out
}
