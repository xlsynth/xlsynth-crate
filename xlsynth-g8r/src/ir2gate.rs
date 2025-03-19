// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting an IR function into a gate function via
//! `gatify`.

use crate::check_equivalence;
use crate::gate::{AigBitVector, AigOperand, GateBuilder, GateFn};
use crate::ir;
use crate::ir::StartAndLimit;
use crate::ir_utils;
use std::collections::HashMap;

use crate::ir2gate_utils::{
    gatify_add_ripple_carry, gatify_barrel_shifter, gatify_one_hot, gatify_one_hot_select,
    Direction,
};

#[derive(Debug)]
enum GateOrVec {
    Gate(AigOperand),
    BitVector(AigBitVector),
}

struct GateEnv {
    ir_to_g8: HashMap<ir::NodeRef, GateOrVec>,
}

impl GateEnv {
    fn new() -> Self {
        Self {
            ir_to_g8: HashMap::new(),
        }
    }

    pub fn contains(&self, ir_node_ref: ir::NodeRef) -> bool {
        self.ir_to_g8.contains_key(&ir_node_ref)
    }

    pub fn add(&mut self, ir_node_ref: ir::NodeRef, gate_or_vec: GateOrVec) {
        log::debug!(
            "add; ir_node_ref: {:?}; gate_or_vec: {:?}",
            ir_node_ref,
            gate_or_vec
        );
        match self.ir_to_g8.insert(ir_node_ref, gate_or_vec) {
            Some(_) => {
                panic!("Duplicate gate reference for IR node {:?}", ir_node_ref);
            }
            None => {}
        }
    }

    pub fn get_bit_vector(&self, ir_node_ref: ir::NodeRef) -> Result<AigBitVector, String> {
        match self.ir_to_g8.get(&ir_node_ref) {
            Some(GateOrVec::BitVector(bv)) => Ok(bv.clone()),
            Some(GateOrVec::Gate(gate_ref)) => Ok(AigBitVector::from_bit(*gate_ref)),
            None => Err(format!(
                "No gate data present for IR node {:?}",
                ir_node_ref
            )),
        }
    }
}

fn gatify_priority_sel(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    selector_bits: AigBitVector,
    cases: &[AigBitVector],
    default_bits: Option<AigBitVector>,
) -> AigBitVector {
    assert_eq!(
        selector_bits.get_bit_count(),
        cases.len(),
        "priority select selector bit width {} does not match number of cases {}",
        selector_bits.get_bit_count(),
        cases.len()
    );

    let mut masked_cases = vec![];
    // As we process cases we track whether any prior case had been selected.
    let mut any_prior_selected = gb.get_false();
    for (i, case_bits) in cases.iter().enumerate() {
        assert_eq!(case_bits.get_bit_count(), output_bit_count, "all cases of the priority select must have the same bit count which is the same as the output bit count");
        let this_wants_selected = selector_bits.get_lsb(i).clone();
        let no_prior_selected = gb.add_not(any_prior_selected);
        let this_selected = gb.add_and_binary(this_wants_selected, no_prior_selected);
        any_prior_selected = gb.add_or_binary(any_prior_selected, this_selected);

        let mask = gb.replicate(this_selected, output_bit_count);
        let masked = gb.add_and_vec(&mask, &case_bits);
        masked_cases.push(masked);
    }

    if let Some(default_bits) = default_bits {
        let no_prior_selected = gb.add_not(any_prior_selected);
        let mask = gb.replicate(no_prior_selected, output_bit_count);
        let masked = gb.add_and_vec(&mask, &default_bits);
        masked_cases.push(masked);
    }
    gb.add_or_vec_nary(&masked_cases)
}

fn gatify_array_index(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    index_bits: &AigBitVector,
) -> AigBitVector {
    let array_element_count = array_ty.element_count;
    let index_decoded = gatify_decode(gb, array_element_count, index_bits);
    let oob = gb.add_ez(&index_decoded);
    let one_hot_selector = AigBitVector::concat(oob.into(), index_decoded);

    // An array index selection is effectively a one hot selection of the elements
    // into a single element result.
    let element_bit_count = array_ty.element_type.bit_count();
    let mut cases = Vec::new();
    for i in (0..array_element_count).rev() {
        let case_bits = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
        cases.push(case_bits);
    }
    cases.push(cases.last().unwrap().clone());
    let result = gatify_one_hot_select(gb, &one_hot_selector, &cases);
    result
}

fn gatify_sel(
    gb: &mut GateBuilder,
    selector_bits: &AigBitVector,
    cases: &[AigBitVector],
    default_bits: Option<AigBitVector>,
) -> AigBitVector {
    let case_count = cases.len();
    let index_decoded = gatify_decode(gb, case_count, selector_bits);

    let mut ohs_cases: Vec<AigBitVector> = Vec::new();
    for case in cases {
        ohs_cases.push(case.clone());
    }

    if let Some(default_bits) = default_bits {
        // This is the scenario where the select has an OOB case.
        ohs_cases.push(default_bits.clone());
        let oob = gb.add_ez(&index_decoded);
        let one_hot_selector = AigBitVector::concat(oob.into(), index_decoded);
        gatify_one_hot_select(gb, &one_hot_selector, &ohs_cases)
    } else {
        // This is the scenario where there is no OOB case so we can just OHS using the
        // decoded value.
        gatify_one_hot_select(gb, &index_decoded, &ohs_cases)
    }
}

fn gatify_concat(args: &[AigBitVector]) -> AigBitVector {
    let mut bits = Vec::new();
    for arg in args.iter().rev() {
        bits.extend(arg.iter_lsb_to_msb().cloned());
    }
    AigBitVector::from_lsb_is_index_0(&bits)
}

fn gatify_zero_ext(new_bit_count: usize, arg_bits: &AigBitVector) -> AigBitVector {
    let zero_count = new_bit_count - arg_bits.get_bit_count();
    let zeros = AigBitVector::zeros(zero_count);
    AigBitVector::concat(zeros, arg_bits.clone())
}

fn gatify_ugt(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    // ugt(a, b) iff a - b is > 0
    let b_complement = gb.add_not_vec(&b_bits);
    let (carry_out, sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_complement,
        gb.get_true(),
        Some(&format!("ugt_{}", text_id)),
        gb,
    );
    let sub_result_is_zero = gb.add_ez(&sub_result);
    let sub_result_is_nonzero = gb.add_not(sub_result_is_zero);
    // The carry_out represents that `a >= b`.
    let sub_result_is_positive = carry_out;
    gb.add_and_binary(sub_result_is_positive, sub_result_is_nonzero)
}

fn gatify_uge(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    let b_complement = gb.add_not_vec(&b_bits);
    let (carry_out, _sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_complement,
        gb.get_true(),
        Some(&format!("uge_{}", text_id)),
        gb,
    );
    // The carry_out represents that `a >= b`.
    carry_out
}

fn gatify_ult(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    let b_inverted = gb.add_not_vec(&b_bits);
    let (c_out, _sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_inverted,
        gb.get_true(),
        Some(&format!("ult_{}", text_id)),
        gb,
    );
    gb.add_not(c_out)
}

fn gatify_ule(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    let b_inverted = gb.add_not_vec(&b_bits);
    let (c_out, _sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_inverted,
        gb.get_true(),
        Some(&format!("ule_{}", text_id)),
        gb,
    );
    // Note: there's a choice here of whether to test after subtraction or equality
    // before subtraction.
    let is_lt = gb.add_not(c_out);
    let is_eq = gb.add_eq_vec(&a_bits, &b_bits);
    gb.add_or_binary(is_lt, is_eq)
}

fn gatify_sign_ext(
    gb: &mut GateBuilder,
    text_id: usize,
    new_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    let msb = arg_bits.get_msb(0);
    gb.add_tag(msb.node, format!("sign_ext_{}_msb", text_id));
    let input_bit_count = arg_bits.get_bit_count();
    assert!(new_bit_count >= input_bit_count);
    let replicated_msb = gb.replicate(*msb, new_bit_count - input_bit_count);

    // Concatenate the replicated msb with the gates[1..]
    let result = AigBitVector::concat(replicated_msb, arg_bits.clone());

    assert_eq!(result.get_bit_count(), new_bit_count);
    result
}

/// The `decode` operation tests whether the input matches a particular value,
/// and if so sets that corresponding bit in the output to be 1.
///
/// Since the input bits can only take on at most one value at a time,
/// this output vector is at most one hot. If the width of the output
/// is the full `output_bits = 2^input_bits`, then the output is
/// guaranteed to be a one-hot bit vector.
fn gatify_decode(
    gb: &mut GateBuilder,
    output_width: usize,
    input_bits: &AigBitVector,
) -> AigBitVector {
    let input_bit_count = input_bits.get_bit_count();
    log::info!(
        "gatify_decode; input_bit_count: {}; width: {}",
        input_bit_count,
        output_width
    );
    let mut bits = Vec::new();
    for i in 0..output_width {
        let literal_bits =
            gb.add_literal(&xlsynth::IrBits::make_ubits(input_bit_count, i as u64).unwrap());
        let is_selected = gb.add_eq_vec(&input_bits, &literal_bits);
        bits.push(is_selected);
    }
    AigBitVector::from_lsb_is_index_0(&bits)
}

fn gatify_encode(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    let input_bit_count = arg_bits.get_bit_count();
    let mut to_or_reduce = Vec::new();
    for i in 0..input_bit_count {
        let gate_i_set = arg_bits.get_lsb(i);
        let gate_i_mask = gb.replicate(*gate_i_set, output_bit_count);
        let on_selected =
            gb.add_literal(&xlsynth::IrBits::make_ubits(output_bit_count, i as u64).unwrap());
        let masked = gb.add_and_vec(&gate_i_mask, &on_selected);
        to_or_reduce.push(masked);
    }
    let to_or_reduce_slices: Vec<AigBitVector> = to_or_reduce.iter().map(|v| v.clone()).collect();
    let or_reduced = gb.add_or_vec_nary(&to_or_reduce_slices);
    assert_eq!(or_reduced.get_bit_count(), output_bit_count);
    or_reduced
}

/// Converts the contents of the given IR function to our "g8" representation
/// which has gates and vectors of gates.
fn gatify_internal(f: &ir::Fn, g8_builder: &mut GateBuilder, env: &mut GateEnv) {
    log::info!("gatify_internal; f.name: {}", f.name);
    log::debug!("gatify; f:\n{}", f.to_string());

    // First we place all the inputs into the G8 structure and environment.
    for (i, param) in f.params.iter().enumerate() {
        assert!(f.nodes[i + 1].payload == ir::NodePayload::GetParam(param.id));
        log::debug!("Gatifying param {:?}", param);
        let gate_ref_vec = g8_builder.add_input(param.name.clone(), param.ty.bit_count());
        env.add(
            ir::NodeRef { index: i + 1 },
            GateOrVec::BitVector(gate_ref_vec),
        );
    }

    for node_ref in ir_utils::get_topological(f) {
        let node = &f.get_node(node_ref);
        let payload = &node.payload;
        log::debug!(
            "Gatifying node {:?} type: {:?} payload: {:?}",
            node_ref,
            node.ty,
            payload
        );
        match payload {
            ir::NodePayload::GetParam(param_index) => {
                if env.contains(node_ref) {
                    continue; // Handled above.
                }
                let param_ir_node_ref = ir::NodeRef {
                    index: *param_index,
                };
                let entry = env.get_bit_vector(param_ir_node_ref).unwrap();
                env.add(node_ref, GateOrVec::BitVector(entry));
            }
            ir::NodePayload::ArrayIndex { array, indices } => {
                if indices.len() != 1 {
                    todo!();
                }
                let index = indices[0];
                let array_ty = match f.get_node_ty(*array) {
                    ir::Type::Array(array_ty_data) => array_ty_data,
                    other => panic!("Expected array type for array_index, got {:?}", other),
                };
                let array_bits = env.get_bit_vector(*array).unwrap();
                let index_bits = env.get_bit_vector(index).unwrap();
                let result = gatify_array_index(g8_builder, array_ty, &array_bits, &index_bits);
                env.add(node_ref, GateOrVec::BitVector(result));
            }
            ir::NodePayload::TupleIndex { tuple, index } => {
                // We have to figure out what bit range the index indicates from the original
                // tuple's flat bits.
                let tuple_bits = env.get_bit_vector(*tuple).unwrap();
                let tuple_ty = f.get_node_ty(*tuple);
                let StartAndLimit { start, limit } =
                    tuple_ty.tuple_get_flat_bit_slice_for_index(*index).unwrap();
                let member_bits = tuple_bits
                    .iter_lsb_to_msb()
                    .skip(start)
                    .take(limit - start)
                    .cloned()
                    .collect::<Vec<_>>();
                let bit_vector = AigBitVector::from_lsb_is_index_0(&member_bits);
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::Tuple(args) => {
                // Tuples, similar to arrays, need to answer the question: "which member is
                // least significant when flattened?"
                //
                // When we perform: `tuple(a, b, c)` does `a` go in the lower bits or does `c`?
                //
                // For concat we have `concat(a, b, c)` place `a` in the upper bits such that
                // the result is: `a_msb, ..., a_lsb, b_msb, ..., b_lsb, c_msb,
                // ..., c_lsb`. So we take the same approach for tuples -- we
                // iterate the arguments in reverse order to make sure c's lsb
                // comes first and build from lsb to msb.

                let mut lsb_to_msb = Vec::new();
                for arg in args.iter().rev() {
                    let arg_gates = env
                        .get_bit_vector(*arg)
                        .expect("tuple arg should be present");
                    lsb_to_msb.extend(arg_gates.iter_lsb_to_msb().cloned());
                }
                let bit_vector = AigBitVector::from_lsb_is_index_0(&lsb_to_msb);
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                // Note: sel is basically an array index into the cases where we pick default if
                // the selector value is OOB.
                let selector_bits = env
                    .get_bit_vector(*selector)
                    .expect("selector should be present");
                let cases: Vec<AigBitVector> = cases
                    .iter()
                    .map(|c| env.get_bit_vector(*c).expect("case should be present"))
                    .collect();
                let default_bits =
                    default.map(|d| env.get_bit_vector(d).expect("default should be present"));

                let gates = gatify_sel(g8_builder, &selector_bits, &cases, default_bits);

                // Tag the result bits
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(gate.node, format!("sel_{}_output_bit_{}", node.text_id, i));
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Literal(literal) => {
                let literal_bits = match literal.to_bits() {
                    Ok(bits) => bits,
                    Err(e) => {
                        panic!("Literal {:?} is not a bits value: {:?}", literal, e);
                    }
                };

                let gate_refs = g8_builder.add_literal(&literal_bits);
                assert_eq!(gate_refs.get_bit_count(), literal_bits.get_bit_count());
                env.add(node_ref, GateOrVec::BitVector(gate_refs));
            }

            // -- unary operations
            ir::NodePayload::Unop(ir::Unop::Not, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gates: AigBitVector = g8_builder.add_not_vec(&arg_gates);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Unop(ir::Unop::Neg, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let not_arg = g8_builder.add_not_vec(&arg_gates);
                let zero = g8_builder.add_literal(
                    &xlsynth::IrBits::make_ubits(arg_gates.get_bit_count(), 0).unwrap(),
                );
                let (_, result) = gatify_add_ripple_carry(
                    &not_arg,
                    &zero,
                    g8_builder.get_true(),
                    Some(&format!("neg_{}", node.text_id)),
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(result));
            }
            ir::NodePayload::Unop(ir::Unop::Identity, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                env.add(node_ref, GateOrVec::BitVector(arg_gates));
            }
            ir::NodePayload::Unop(ir::Unop::Reverse, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let result_gates: Vec<AigOperand> = arg_gates
                    .iter_lsb_to_msb()
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>();
                env.add(
                    node_ref,
                    GateOrVec::BitVector(AigBitVector::from_lsb_is_index_0(&result_gates)),
                );
            }

            // -- bitwise reductions
            ir::NodePayload::Unop(ir::Unop::OrReduce, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gate: AigOperand = g8_builder.add_or_reduce(&arg_gates);
                g8_builder.add_tag(gate.node, format!("or_reduce_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Unop(ir::Unop::AndReduce, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gate: AigOperand = g8_builder.add_and_reduce(&arg_gates);
                g8_builder.add_tag(gate.node, format!("and_reduce_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Unop(ir::Unop::XorReduce, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gate: AigOperand = g8_builder.add_xor_reduce(&arg_gates);
                g8_builder.add_tag(gate.node, format!("xor_reduce_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }

            // -- bitwise binary operations
            ir::NodePayload::Binop(ir::Binop::Xor, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("xor lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("xor rhs should be present");
                let gates: AigBitVector = g8_builder.add_xor_vec(&a_bits, &b_bits);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Or, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("or lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("or rhs should be present");
                let gates: AigBitVector = g8_builder.add_or_vec(&a_gate_refs, &b_gate_refs);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Nand, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("nand lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("nand rhs should be present");
                let gates: AigBitVector = g8_builder.add_and_vec(&a_gate_refs, &b_gate_refs);
                let gates = g8_builder.add_not_vec(&gates);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }

            // -- binary operations
            ir::NodePayload::Binop(ir::Binop::Eq, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("eq lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("eq rhs should be present");
                assert_eq!(a_bits.get_bit_count(), b_bits.get_bit_count());
                let gate: AigOperand = g8_builder.add_eq_vec(&a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("eq_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ne, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("ne lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("ne rhs should be present");
                log::info!(
                    "ne lhs bits[{}] rhs bits[{}]",
                    a_gate_refs.get_bit_count(),
                    b_gate_refs.get_bit_count()
                );
                let gate: AigOperand = g8_builder.add_ne_vec(&a_gate_refs, &b_gate_refs);
                g8_builder.add_tag(gate.node, format!("ne_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ult, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("ult lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("ult rhs should be present");
                let gate = gatify_ult(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("ult_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ugt, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("ugt lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("ugt rhs should be present");
                let gate: AigOperand = gatify_ugt(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("ugt_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Uge, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("uge lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("uge rhs should be present");
                let gate: AigOperand = gatify_uge(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("uge_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ule, a, b) => {
                // a <= b when:
                // * the subtraction gives indication of negative
                // * the subtraction gives back zero
                let a_bits = env.get_bit_vector(*a).expect("ule lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("ule rhs should be present");
                let gate = gatify_ule(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("ule_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }

            // -- nary operations
            ir::NodePayload::Nary(ir::NaryOp::And, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| env.get_bit_vector(*arg).expect("and arg should be present"))
                    .collect();
                let gates: AigBitVector = g8_builder.add_and_vec_nary(arg_gates.as_slice());
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Nor, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| env.get_bit_vector(*arg).expect("nor arg should be present"))
                    .collect();
                let gates: AigBitVector = g8_builder.add_or_vec_nary(&arg_gates);
                let gates = g8_builder.add_not_vec(&gates);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Concat, args) => {
                let arg_bits: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| {
                        env.get_bit_vector(*arg)
                            .expect("concat arg should be present")
                    })
                    .collect();

                let bits = gatify_concat(&arg_bits);

                let output_bit_count = node.ty.bit_count();
                assert_eq!(bits.get_bit_count(), output_bit_count);
                for (i, bit) in bits.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        bit.node,
                        format!("concat_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(bits));
            }
            ir::NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let output_bit_count = node.ty.bit_count();
                let selector_bits = env
                    .get_bit_vector(*selector)
                    .expect("selector should be present");
                let cases: Vec<AigBitVector> = cases
                    .iter()
                    .map(|c| env.get_bit_vector(*c).expect("case should be present"))
                    .collect::<Vec<AigBitVector>>();
                let default_bits =
                    default.map(|d| env.get_bit_vector(d).expect("default should be present"));

                let gates = gatify_priority_sel(
                    g8_builder,
                    output_bit_count,
                    selector_bits,
                    cases.as_slice(),
                    default_bits,
                );
                // Tag the result.
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("priority_sel_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::OneHotSel { selector, cases } => {
                let selector_bits = env
                    .get_bit_vector(*selector)
                    .expect("selector should be present");
                let cases: Vec<AigBitVector> = cases
                    .iter()
                    .map(|c| env.get_bit_vector(*c).expect("case should be present"))
                    .collect::<Vec<AigBitVector>>();
                let gates = gatify_one_hot_select(g8_builder, &selector_bits, &cases);
                // Tag the result.
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("one_hot_select_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Add, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("add lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("add rhs should be present");
                assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
                let (_c_out, gates) = gatify_add_ripple_carry(
                    &a_gate_refs,
                    &b_gate_refs,
                    g8_builder.get_false(),
                    Some(&format!("add_{}", node.text_id)),
                    g8_builder,
                );
                assert_eq!(gates.get_bit_count(), a_gate_refs.get_bit_count());
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Sub, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("sub lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("sub rhs should be present");
                assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
                let b_complement = g8_builder.add_not_vec(&b_gate_refs);
                let (_c_out, gates) = gatify_add_ripple_carry(
                    &a_gate_refs,
                    &b_complement,
                    g8_builder.get_true(),
                    Some(&format!("sub_{}", node.text_id)),
                    g8_builder,
                );
                let output_bit_count = node.ty.bit_count();
                assert_eq!(gates.get_bit_count(), output_bit_count);
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(gate.node, format!("sub_{}_output_bit_{}", node.text_id, i));
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::BitSlice { arg, start, width } => {
                let value_gates = env
                    .get_bit_vector(*arg)
                    .expect("bit_slice value should be present");
                let slice_gates = value_gates.get_lsb_slice(*start, *width);
                assert_eq!(slice_gates.get_bit_count(), *width);
                env.add(node_ref, GateOrVec::BitVector(slice_gates));
            }
            ir::NodePayload::ZeroExt { arg, new_bit_count } => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("zero_ext value should be present");
                let result_bits = gatify_zero_ext(*new_bit_count, &arg_bits);
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::SignExt { arg, new_bit_count } => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("sign_ext value should be present");
                let result_bits =
                    gatify_sign_ext(g8_builder, node.text_id, *new_bit_count, &arg_bits);
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::Decode { arg, width } => {
                assert_eq!(*width, node.ty.bit_count());
                let input_bits = env
                    .get_bit_vector(*arg)
                    .expect("decode arg should be present");
                let bits = gatify_decode(g8_builder, *width, &input_bits);
                assert_eq!(bits.get_bit_count(), *width);
                for (i, bit) in bits.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        bit.node,
                        format!("decode_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(bits));
            }
            ir::NodePayload::Unop(ir::Unop::Encode, arg) => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("encode arg should be present");
                let result_bits = gatify_encode(g8_builder, node.ty.bit_count(), &arg_bits);
                for (i, gate) in result_bits.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("encode_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::Binop(ir::Binop::Shrl, arg, amount) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("shrl arg should be present");
                let amount_gates = env
                    .get_bit_vector(*amount)
                    .expect("shrl amount should be present");
                let result_gates = gatify_barrel_shifter(
                    &arg_gates,
                    &amount_gates,
                    Direction::Right,
                    &format!("shrl_{}", node.text_id),
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(result_gates));
            }
            ir::NodePayload::Binop(ir::Binop::Shll, arg, amount) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("shll arg should be present");
                let amount_gates = env
                    .get_bit_vector(*amount)
                    .expect("shll amount should be present");
                let result_gates = gatify_barrel_shifter(
                    &arg_gates,
                    &amount_gates,
                    Direction::Left,
                    &format!("shll_{}", node.text_id),
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(result_gates));
            }
            ir::NodePayload::OneHot { arg, lsb_prio } => {
                let bits = env
                    .get_bit_vector(*arg)
                    .expect("one_hot arg should be present");
                let bit_vector = gatify_one_hot(g8_builder, &bits, *lsb_prio);
                for (lsb_i, gate) in bit_vector.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("one_hot_{}_output_bit_{}", node.text_id, lsb_i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::Assert { .. }
            | ir::NodePayload::AfterAll(..)
            | ir::NodePayload::Trace { .. }
            | ir::NodePayload::Nil => {
                // No incarnation in gates.
            }
            _ => {
                todo!("Unsupported node payload {:?}", payload);
            }
        }
    }
    // Resolve the outputs and place them into the builder.
    let ret_node_ref = match f.ret_node_ref {
        Some(ret_node_ref) => ret_node_ref,
        None => {
            return;
        }
    };
    let gate_refs = env
        .get_bit_vector(ret_node_ref)
        .expect("return node should be present");
    g8_builder.add_output("output_value".to_string(), gate_refs);
}

pub struct GatifyOptions {
    pub fold: bool,
    pub check_equivalence: bool,
}

pub fn gatify(f: &ir::Fn, options: GatifyOptions) -> Result<GateFn, String> {
    let mut g8_builder = GateBuilder::new(f.name.clone(), options.fold);
    let mut env = GateEnv::new();
    gatify_internal(f, &mut g8_builder, &mut env);
    let gate_fn = g8_builder.build();
    log::info!(
        "converted IR function to gate function:\n{}",
        gate_fn.to_string()
    );

    // If we're told we should do so, we check equivalance between the original IR
    // function and the gate function that we converted it to.
    if options.check_equivalence {
        log::info!("checking equivalence of IR function and gate function...");
        check_equivalence::validate_same_fn(f, &gate_fn)?;
    }
    Ok(gate_fn)
}
