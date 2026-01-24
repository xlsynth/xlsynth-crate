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
    pub aig_operand: AigOperand,
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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(options.sample_seed);

    // --- 1) PIR simulation and node histories.
    let mut pir_histories = PirHistories::default();
    for _sample_idx in 0..options.sample_count {
        let (pir_args, gate_inputs) = make_random_args_for_both(pir_fn, gate_fn, &mut rng)?;

        let mut observer = PirHistoryObserver {
            f: pir_fn,
            histories: &mut pir_histories,
            exclude_structural: options.exclude_structural_pir_nodes,
        };
        match ir_eval::eval_fn_with_observer(pir_fn, &pir_args, Some(&mut observer)) {
            ir_eval::FnEvalResult::Success(_) => {}
            ir_eval::FnEvalResult::Failure(fail) => {
                return Err(format!(
                    "PIR evaluation failed while sampling: assertion_failures={} trace_messages={}",
                    fail.assertion_failures.len(),
                    fail.trace_messages.len()
                ));
            }
        }

        // GateFn inputs are only needed for GateFn simulation; keep them for the
        // second pass by re-generating the same inputs above. (We already have
        // them here; just consume below.)
        drop(gate_inputs);
    }

    // --- 2) GateFn simulation and node histories (positive AigRef values).
    let gate_node_histories =
        simulate_gate_fn_node_histories(gate_fn, options.sample_count, options.sample_seed)?;

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
    for hist in pir_histories.by_text_id.values() {
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
            let sig = pack_bools_to_bytes(bit_hist);
            if let Some(ops) = sig_to_operands.get(&sig) {
                for op in ops {
                    candidates.push(IrAigEquivalenceCandidate {
                        pir_node_ref: hist.node_ref,
                        pir_node_text_id: hist.node_text_id,
                        bit_index,
                        aig_operand: *op,
                    });
                }
            }
        }
    }

    Ok(candidates)
}

pub fn confirm_or_deny_candidate_equivalence(
    _pir_fn: &ir::Fn,
    _gate_fn: &GateFn,
    _candidate: &IrAigEquivalenceCandidate,
) -> Result<bool, String> {
    todo!("confirm candidate equivalence using a shared solver context")
}

fn simulate_gate_fn_node_histories(
    gate_fn: &GateFn,
    sample_count: usize,
    sample_seed: u64,
) -> Result<Vec<Vec<bool>>, String> {
    let gate_count = gate_fn.gates.len();
    let mut histories: Vec<Vec<bool>> = (0..gate_count)
        .map(|_| Vec::with_capacity(sample_count))
        .collect();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(sample_seed);
    for _ in 0..sample_count {
        let mut inputs: Vec<IrBits> = Vec::with_capacity(gate_fn.inputs.len());
        for inp in &gate_fn.inputs {
            let w = inp.get_bit_count();
            inputs.push(random_irbits(&mut rng, w));
        }
        let values = eval_gate_fn_all_node_values_positive(gate_fn, &inputs)?;
        for (id, v) in values.into_iter().enumerate() {
            histories[id].push(v);
        }
    }
    Ok(histories)
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
        let v = random_ir_value_for_type(rng, &param.ty)?;
        let flat_bits = ir_value_to_flat_lsb0_bits(&v, &param.ty)?;
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

fn random_ir_value_for_type(rng: &mut impl RngCore, ty: &ir::Type) -> Result<IrValue, String> {
    match ty {
        ir::Type::Token => Ok(IrValue::make_token()),
        ir::Type::Bits(w) => Ok(IrValue::from_bits(&random_irbits(rng, *w))),
        ir::Type::Tuple(types) => {
            let mut elems: Vec<IrValue> = Vec::with_capacity(types.len());
            for t in types {
                elems.push(random_ir_value_for_type(rng, t)?);
            }
            Ok(IrValue::make_tuple(&elems))
        }
        ir::Type::Array(ir::ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let mut elems: Vec<IrValue> = Vec::with_capacity(*element_count);
            for _ in 0..*element_count {
                elems.push(random_ir_value_for_type(rng, element_type)?);
            }
            IrValue::make_array(&elems).map_err(|e| e.to_string())
        }
    }
}

fn random_irbits(rng: &mut impl RngCore, width: usize) -> IrBits {
    if width == 0 {
        return IrBits::make_ubits(0, 0).expect("u0");
    }
    let mut bits: Vec<bool> = Vec::with_capacity(width);
    for _ in 0..width {
        bits.push((rng.next_u32() & 1) != 0);
    }
    IrBits::from_lsb_is_0(&bits)
}

fn ir_value_to_flat_lsb0_bits(v: &IrValue, ty: &ir::Type) -> Result<Vec<bool>, String> {
    let mut out = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits(v, ty, &mut out)?;
    Ok(out)
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
