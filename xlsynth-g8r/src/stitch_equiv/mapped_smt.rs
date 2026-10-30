// SPDX-License-Identifier: Apache-2.0

//! Bit-vector formulas for mapped Liberty logic and register captures.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use xlsynth_prover::solver::{BitVec, Solver};

use crate::aig::topo::postorder_for_aig_refs_node_only;
use crate::aig::{AigNode, AigOperand, AigRef, GateFn, SequentialGateFn};

use super::correspondence::RegisterBitPair;
use super::output_name;
use super::stage_graph::validate_stage_relation;

/// The shared external input or arbitrary state word supplying an AIG input.
#[derive(Clone, Copy)]
enum Word<'a> {
    Port(&'a str),
    Register(usize),
}

/// A mapped input node can consume a permuted or complemented state bit.
struct InputBit<'a> {
    word: Word<'a>,
    bit: usize,
    inverted: bool,
}

/// Physical input bindings and the ordered capture/output roots of each stage.
pub(super) struct MappedStages<'a> {
    graph: &'a GateFn,
    bindings: HashMap<AigRef, InputBit<'a>>,
    roots: Vec<Vec<AigOperand>>,
}

impl<'a> MappedStages<'a> {
    /// Aligns effective D and external outputs with the behavioral obligation.
    pub(super) fn new(
        reference: &SequentialGateFn,
        mapped: &'a SequentialGateFn,
        pairs: &[RegisterBitPair],
        register_stages: &[usize],
        stage_count: usize,
    ) -> Result<Self, String> {
        validate_stage_relation(reference, mapped, pairs, register_stages, stage_count)?;
        let graph = &mapped.transition;
        let mut bindings = external_bindings(mapped);
        let mut roots = vec![Vec::new(); stage_count];
        let mut ordered = pairs.to_vec();
        ordered.sort_unstable_by_key(|p| (p.reference_register, p.reference_bit));
        for p in ordered {
            let reg = &mapped.registers[p.mapped_register];
            let q = graph.inputs[reg.q.index()].bit_vector.get_lsb(p.mapped_bit);
            bindings.insert(
                q.node,
                InputBit {
                    word: Word::Register(p.reference_register),
                    bit: p.reference_bit,
                    inverted: p.inverted ^ q.negated,
                },
            );
            let mut d = *graph.outputs[reg.d.index()]
                .bit_vector
                .get_lsb(p.mapped_bit);
            d.negated ^= p.inverted;
            roots[register_stages[p.reference_register]].push(d);
        }
        let mapped_outputs: BTreeMap<_, _> = mapped
            .outputs
            .iter()
            .map(|id| {
                let output = &graph.outputs[id.index()];
                (output_name(&output.name), &output.bit_vector)
            })
            .collect();
        for id in &reference.outputs {
            let name = output_name(&reference.transition.outputs[id.index()].name);
            roots[stage_count - 1].extend(mapped_outputs[name].iter_lsb_to_msb());
        }
        Ok(Self {
            graph,
            bindings,
            roots,
        })
    }

    pub(super) fn width(&self, stage: usize) -> usize {
        self.roots[stage].len()
    }

    /// Translates just this proof leaf's physical cone in its worker's solver.
    pub(super) fn evaluate<S: Solver>(
        &self,
        solver: &mut S,
        inputs: &BTreeMap<String, BitVec<S::Term>>,
        logical_q: &[BitVec<S::Term>],
        stage: usize,
        bits: Range<usize>,
    ) -> Result<BitVec<S::Term>, String> {
        let roots = self
            .roots
            .get(stage)
            .and_then(|roots| roots.get(bits))
            .ok_or_else(|| format!("invalid mapped output selection for stage {stage}"))?;
        let order = cone_order(self.graph, roots.iter());
        let values = evaluate_cone(
            solver,
            self.graph,
            &order,
            &self.bindings,
            inputs,
            logical_q,
        )?;
        Ok(pack_bits(solver, roots.iter(), &values))
    }
}

/// Binds named external input words by their packed port bit positions.
fn external_bindings(design: &SequentialGateFn) -> HashMap<AigRef, InputBit<'_>> {
    let mut bindings = HashMap::new();
    for id in &design.inputs {
        let port = &design.transition.inputs[id.index()];
        for (bit, operand) in port.bit_vector.iter_lsb_to_msb().enumerate() {
            bindings.insert(
                operand.node,
                InputBit {
                    word: Word::Port(&port.name),
                    bit,
                    inverted: operand.negated,
                },
            );
        }
    }
    bindings
}

/// Orders each distinct reachable AIG node after its operands.
fn cone_order<'a>(graph: &GateFn, bits: impl Iterator<Item = &'a AigOperand>) -> Vec<AigRef> {
    let roots = bits.map(|b| b.node).collect::<Vec<_>>();
    postorder_for_aig_refs_node_only(&roots, &graph.gates, &HashMap::<AigRef, ()>::new())
}

/// Retrieves a previously translated AIG node with the edge's polarity.
fn operand_term<S: Solver>(
    solver: &mut S,
    operand: AigOperand,
    values: &HashMap<AigRef, BitVec<S::Term>>,
) -> BitVec<S::Term> {
    let term = &values[&operand.node];
    if operand.negated {
        solver.not(term)
    } else {
        term.clone()
    }
}

/// Concatenates bits supplied in register/port LSB order into one word.
fn pack_bits<'a, S: Solver>(
    solver: &mut S,
    bits: impl DoubleEndedIterator<Item = &'a AigOperand>,
    values: &HashMap<AigRef, BitVec<S::Term>>,
) -> BitVec<S::Term> {
    let terms: Vec<_> = bits
        .rev()
        .map(|b| operand_term(solver, *b, values))
        .collect();
    solver.concat_many(terms.iter().collect())
}

/// Translates each node of the selected cone once, slicing only live inputs.
fn evaluate_cone<S: Solver>(
    solver: &mut S,
    graph: &GateFn,
    order: &[AigRef],
    bindings: &HashMap<AigRef, InputBit<'_>>,
    inputs: &BTreeMap<String, BitVec<S::Term>>,
    q: &[BitVec<S::Term>],
) -> Result<HashMap<AigRef, BitVec<S::Term>>, String> {
    let mut values = HashMap::with_capacity(order.len());
    for &node in order {
        let term = match graph.get(node) {
            AigNode::Input { name, .. } => {
                let binding = bindings
                    .get(&node)
                    .ok_or_else(|| format!("unbound mapped input '{name}'"))?;
                let word = match binding.word {
                    Word::Port(port) => inputs
                        .get(port)
                        .ok_or_else(|| format!("missing external input '{port}'"))?,
                    Word::Register(reg) => q
                        .get(reg)
                        .ok_or_else(|| format!("missing state for register {reg}"))?,
                };
                if binding.bit >= word.get_width() {
                    return Err(format!(
                        "mapped input '{name}' requires bit {} of a {}-bit binding",
                        binding.bit,
                        word.get_width()
                    ));
                }
                let bit = solver.slice(word, binding.bit, 1);
                if binding.inverted {
                    solver.not(&bit)
                } else {
                    bit
                }
            }
            AigNode::Literal { value, .. } => solver.numerical(1, u64::from(*value)),
            AigNode::And2 { a, b, .. } => {
                let a = operand_term(solver, *a, &values);
                let b = operand_term(solver, *b, &values);
                solver.and(&a, &b)
            }
        };
        values.insert(node, term);
    }
    Ok(values)
}

/// Advances independent arbitrary physical state, observing all named outputs
/// on the final input frame. Every earlier edge samples all effective Ds.
pub(super) fn unroll_mapped<S: Solver>(
    solver: &mut S,
    mapped: &SequentialGateFn,
    frames: &[BTreeMap<String, BitVec<S::Term>>],
    initial_q: &[BitVec<S::Term>],
) -> Result<BTreeMap<String, BitVec<S::Term>>, String> {
    let (observation, advances) = frames
        .split_last()
        .ok_or("unrolling needs an observation frame")?;
    let graph = &mapped.transition;
    let mut bindings = external_bindings(mapped);
    for (index, reg) in mapped.registers.iter().enumerate() {
        for (bit, operand) in graph.inputs[reg.q.index()]
            .bit_vector
            .iter_lsb_to_msb()
            .enumerate()
        {
            bindings.insert(
                operand.node,
                InputBit {
                    word: Word::Register(index),
                    bit,
                    inverted: operand.negated,
                },
            );
        }
    }
    let captures: Vec<_> = mapped
        .registers
        .iter()
        .map(|r| &graph.outputs[r.d.index()].bit_vector)
        .collect();
    let capture_order = cone_order(graph, captures.iter().flat_map(|d| d.iter_lsb_to_msb()));
    let mut q = initial_q.to_vec();
    for frame in advances {
        let values = evaluate_cone(solver, graph, &capture_order, &bindings, frame, &q)?;
        q = captures
            .iter()
            .map(|d| pack_bits(solver, d.iter_lsb_to_msb(), &values))
            .collect();
    }
    let outputs: Vec<_> = mapped
        .outputs
        .iter()
        .map(|id| &graph.outputs[id.index()])
        .collect();
    let observe_order = cone_order(
        graph,
        outputs.iter().flat_map(|o| o.bit_vector.iter_lsb_to_msb()),
    );
    let values = evaluate_cone(solver, graph, &observe_order, &bindings, observation, &q)?;
    Ok(outputs
        .into_iter()
        .map(|o| {
            (
                output_name(&o.name).to_owned(),
                pack_bits(solver, o.bit_vector.iter_lsb_to_msb(), &values),
            )
        })
        .collect())
}
