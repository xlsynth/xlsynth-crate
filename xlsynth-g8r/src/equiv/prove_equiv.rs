// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use xlsynth::IrValue;

use crate::{
    equiv::solver_interface::{BitVec, Response, Solver},
    xls_ir::{
        ir::{self, NaryOp, NodePayload, NodeRef, Unop},
        ir_utils::get_topological,
    },
};

#[derive(Clone)]
pub struct IrTypedBitVec<'a, R> {
    pub ir_type: &'a ir::Type,
    pub bitvec: BitVec<R>,
}

impl<'a, R: std::fmt::Debug> std::fmt::Debug for IrTypedBitVec<'a, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IrTypedBitVec {{ ir_type: {:?}, bitvec: {:?} }}",
            self.ir_type, self.bitvec
        )
    }
}

pub fn ir_value_to_bv<'a, S: Solver>(
    solver: &mut S,
    ir_value: &IrValue,
    ir_type: &'a ir::Type,
) -> IrTypedBitVec<'a, S::Term> {
    fn gather_bits(v: &IrValue, ty: &ir::Type, bits: &mut Vec<bool>) {
        match ty {
            ir::Type::Bits(width) => {
                let b = v.to_bits().expect("Expected bits literal");
                assert_eq!(b.get_bit_count(), *width);
                for i in 0..*width {
                    bits.push(b.get_bit(i).unwrap()); // LSB first
                }
            }
            ir::Type::Array(arr) => {
                let elems = v.get_elements().expect("Array literal elements");
                assert_eq!(elems.len(), arr.element_count);
                for elem in elems.iter() {
                    gather_bits(elem, &arr.element_type, bits);
                }
            }
            ir::Type::Tuple(members) => {
                let elems = v.get_elements().expect("Tuple literal elements");
                assert_eq!(elems.len(), members.len());
                for (elem, elem_ty) in elems.iter().rev().zip(members.iter().rev()) {
                    gather_bits(elem, elem_ty, bits);
                }
            }
            ir::Type::Token => {
                // Tokens are zero-width; nothing to gather.
            }
        }
    }
    let mut bits_vec: Vec<bool> = Vec::new();
    gather_bits(ir_value, ir_type, &mut bits_vec);
    let width = bits_vec.len();
    if width == 0 {
        return IrTypedBitVec {
            ir_type,
            bitvec: BitVec::ZeroWidth,
        };
    }
    let mut s = String::from("#b");
    // SMT bit-vector constant expects MSB first.
    for bit in bits_vec.iter().rev() {
        s.push(if *bit { '1' } else { '0' });
    }
    IrTypedBitVec {
        ir_type,
        bitvec: solver.from_raw_str(width, &s),
    }
}

#[derive(Debug, Clone)]
pub struct FnInputs<'a, R> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: HashMap<String, IrTypedBitVec<'a, R>>,
}

pub fn get_fn_inputs<'a, S: Solver>(
    solver: &mut S,
    fn_ref: &'a ir::Fn,
    name_prefix: Option<&str>,
) -> FnInputs<'a, S::Term> {
    let mut inputs = HashMap::new();
    for p in fn_ref.params.iter() {
        let name = match name_prefix {
            Some(prefix) => format!("__{}__{}", prefix, p.name),
            None => p.name.clone(),
        };
        let bv = solver.declare(&name, p.ty.bit_count() as usize).unwrap();
        inputs.insert(
            p.name.clone(),
            IrTypedBitVec {
                ir_type: &p.ty,
                bitvec: bv,
            },
        );
    }
    FnInputs { fn_ref, inputs }
}

impl<'a, R> FnInputs<'a, R> {
    pub fn total_width(&self) -> usize {
        self.inputs.values().map(|b| b.bitvec.get_width()).sum()
    }
}

pub struct AlignedFnInputs<'a, R> {
    pub lhs: FnInputs<'a, R>,
    pub rhs: FnInputs<'a, R>,
    pub flattened: BitVec<R>,
}

/// Given the bit-vector representation of the two functions' inputs, this
/// function generates a flattened bit-vector, and use slices of the flattened
/// bit-vector for inputs to both functions.
///
/// We need this is to enhance term sharing. For example, when two functions
/// shares the same prelude, they will be compiled into exactly the same SMT
/// terms with the aligned inputs. The solver will then be able to just reason
/// about the behavior of the remaining parts of the functions.
pub fn align_fn_inputs<'a, S: Solver>(
    solver: &mut S,
    lhs_inputs: &FnInputs<'a, S::Term>,
    rhs_inputs: &FnInputs<'a, S::Term>,
    allow_flatten: bool,
) -> AlignedFnInputs<'a, S::Term> {
    let lhs_inputs_total_width = lhs_inputs.total_width();
    let rhs_inputs_total_width = rhs_inputs.total_width();
    assert_eq!(
        lhs_inputs_total_width, rhs_inputs_total_width,
        "LHS and RHS must have the same number of bits"
    );
    if !allow_flatten {
        assert_eq!(
            lhs_inputs.fn_ref.params.len(),
            rhs_inputs.fn_ref.params.len(),
            "LHS and RHS must have the same number of inputs"
        );
        for (l, r) in lhs_inputs
            .fn_ref
            .params
            .iter()
            .zip(rhs_inputs.fn_ref.params.iter())
        {
            assert_eq!(
                l.ty, r.ty,
                "Input type mismatch for {} vs {}: {:?} vs {:?}",
                l.name, r.name, l.ty, r.ty
            );
        }
    }
    if lhs_inputs_total_width == 0 {
        return AlignedFnInputs {
            lhs: lhs_inputs.clone(),
            rhs: rhs_inputs.clone(),
            flattened: BitVec::ZeroWidth,
        };
    }
    let params_name = format!(
        "__flattened_params__{}__{}",
        lhs_inputs.fn_ref.name, rhs_inputs.fn_ref.name
    );
    let flattened = solver
        .declare(&params_name, lhs_inputs_total_width)
        .unwrap();
    // Split into individual param symbols
    let mut split_map =
        |inputs: &FnInputs<'a, S::Term>| -> HashMap<String, IrTypedBitVec<'a, S::Term>> {
            let mut m = HashMap::new();
            let mut offset = 0;
            for n in inputs.fn_ref.params.iter() {
                let existing_bitvec = inputs.inputs.get(&n.name).unwrap();
                let new_bitvec = {
                    let w = existing_bitvec.bitvec.get_width();
                    let h = (offset + w - 1) as i32;
                    let l = offset as i32;
                    offset += w;
                    let extracted = solver.extract(&flattened, h, l);
                    let eq_expr = solver.eq(&extracted, &existing_bitvec.bitvec);
                    solver.assert(&eq_expr).unwrap();
                    extracted
                };
                m.insert(
                    n.name.clone(),
                    IrTypedBitVec {
                        ir_type: &n.ty,
                        bitvec: new_bitvec,
                    },
                );
            }
            m
        };
    let lhs_params = split_map(&lhs_inputs);
    let rhs_params = split_map(&rhs_inputs);

    AlignedFnInputs {
        lhs: FnInputs {
            fn_ref: lhs_inputs.fn_ref,
            inputs: lhs_params,
        },
        rhs: FnInputs {
            fn_ref: rhs_inputs.fn_ref,
            inputs: rhs_params,
        },
        flattened,
    }
}

pub struct SmtFn<'a, R> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: Vec<IrTypedBitVec<'a, R>>,
    pub output: IrTypedBitVec<'a, R>,
}

pub fn ir_to_smt<'a, S: Solver>(
    solver: &mut S,
    inputs: &'a FnInputs<'a, S::Term>,
) -> SmtFn<'a, S::Term> {
    let topo = get_topological(inputs.fn_ref);
    let mut env: HashMap<NodeRef, IrTypedBitVec<'a, S::Term>> = HashMap::new();
    for nr in topo {
        let node = &inputs.fn_ref.nodes[nr.index];
        fn get_num_indexable_elements<'a, S: Solver>(
            index: &IrTypedBitVec<'a, S::Term>,
            num_elements: usize,
        ) -> usize {
            assert!(num_elements > 0, "select: no values to select from");
            let index_width = index.bitvec.get_width();
            let max_indexable_elements = if index_width >= usize::BITS as usize {
                usize::MAX
            } else {
                1 << index_width
            };
            max_indexable_elements.min(num_elements)
        }
        let exp: IrTypedBitVec<'a, S::Term> = match &node.payload {
            NodePayload::Nil => continue,
            NodePayload::GetParam(pid) => {
                let p = inputs.fn_ref.params.iter().find(|p| p.id == *pid).unwrap();
                if let Some(sym) = inputs.inputs.get(&p.name) {
                    sym.clone()
                } else {
                    panic!("Param not found: {}", p.name);
                }
            }
            NodePayload::Tuple(elems) => {
                let mut bv = BitVec::ZeroWidth;
                for elem in elems {
                    let elem_bv = env.get(elem).expect("Tuple element must be present");
                    bv = solver.concat(&bv, &elem_bv.bitvec);
                }
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: bv,
                }
            }
            NodePayload::Array(elems) => {
                let mut bv = BitVec::ZeroWidth;
                for elem in elems {
                    let elem_bv = env.get(elem).expect("Array element must be present");
                    bv = solver.concat(&elem_bv.bitvec, &bv);
                }
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: bv,
                }
            }
            NodePayload::TupleIndex { tuple, index } => {
                let tuple_bv = env.get(tuple).expect("Tuple operand must be present");
                let tuple_ty = inputs.fn_ref.get_node_ty(*tuple);
                let slice = tuple_ty
                    .tuple_get_flat_bit_slice_for_index(*index)
                    .expect("TupleIndex: not a tuple type");
                let width = slice.limit - slice.start;
                assert!(width > 0, "TupleIndex: width must be > 0");
                let high = (slice.limit - 1) as i32;
                let low = slice.start as i32;
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.extract(&tuple_bv.bitvec, high, low),
                }
            }
            NodePayload::Binop(op, lhs, rhs) => {
                let l = env.get(lhs).unwrap().clone();
                let r = env.get(rhs).unwrap().clone();
                let result_width = node.ty.bit_count();
                let rep = match op {
                    ir::Binop::Add => solver.add(&l.bitvec, &r.bitvec),
                    ir::Binop::Sub => solver.sub(&l.bitvec, &r.bitvec),
                    ir::Binop::Shll => solver.xls_shll(&l.bitvec, &r.bitvec),
                    ir::Binop::Shrl => solver.xls_shrl(&l.bitvec, &r.bitvec),
                    ir::Binop::Shra => solver.xls_shra(&l.bitvec, &r.bitvec),
                    ir::Binop::Eq => solver.eq(&l.bitvec, &r.bitvec),
                    ir::Binop::Ne => solver.ne(&l.bitvec, &r.bitvec),
                    ir::Binop::Uge => solver.uge(&l.bitvec, &r.bitvec),
                    ir::Binop::Ugt => solver.ugt(&l.bitvec, &r.bitvec),
                    ir::Binop::Ult => solver.ult(&l.bitvec, &r.bitvec),
                    ir::Binop::Ule => solver.ule(&l.bitvec, &r.bitvec),
                    ir::Binop::Sgt => solver.sgt(&l.bitvec, &r.bitvec),
                    ir::Binop::Sge => solver.sge(&l.bitvec, &r.bitvec),
                    ir::Binop::Slt => solver.slt(&l.bitvec, &r.bitvec),
                    ir::Binop::Sle => solver.sle(&l.bitvec, &r.bitvec),
                    ir::Binop::Umul => {
                        solver.xls_arbitrary_width_umul(&l.bitvec, &r.bitvec, result_width)
                    }
                    ir::Binop::Smul => {
                        solver.xls_arbitrary_width_smul(&l.bitvec, &r.bitvec, result_width)
                    }
                    ir::Binop::Sdiv => solver.xls_sdiv(&l.bitvec, &r.bitvec),
                    ir::Binop::Udiv => solver.xls_udiv(&l.bitvec, &r.bitvec),
                    ir::Binop::Umod => solver.xls_umod(&l.bitvec, &r.bitvec),
                    ir::Binop::Smod => solver.xls_smod(&l.bitvec, &r.bitvec),
                    _ => panic!("Unsupported binop: {:?}", op),
                };
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: rep,
                }
            }
            NodePayload::Unop(op, arg) => {
                let a = env.get(&arg).unwrap().clone();
                let rep = match op {
                    Unop::Not => solver.not(&a.bitvec),
                    Unop::Neg => solver.neg(&a.bitvec),
                    Unop::OrReduce => solver.or_reduce(&a.bitvec),
                    Unop::AndReduce => solver.and_reduce(&a.bitvec),
                    Unop::XorReduce => solver.xor_reduce(&a.bitvec),
                    Unop::Identity => a.bitvec,
                    Unop::Reverse => solver.reverse(&a.bitvec),
                };
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: rep,
                }
            }
            NodePayload::Literal(v) => ir_value_to_bv(solver, v, &node.ty),
            NodePayload::ZeroExt { arg, new_bit_count }
            | NodePayload::SignExt { arg, new_bit_count } => {
                let arg_bv = env
                    .get(arg)
                    .expect("ZeroExt/SignExt argument must be present");
                let to_width = *new_bit_count;
                let signed = matches!(node.payload, NodePayload::SignExt { .. });
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.extend_to(&arg_bv.bitvec, to_width, signed),
                }
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } => {
                // Recursively build an updated array value that reflects
                // `array[indices] = value` semantics.

                fn array_update_recursive<'a, S: Solver>(
                    solver: &mut S,
                    array_val: &IrTypedBitVec<'a, S::Term>,
                    indices: &[&IrTypedBitVec<'a, S::Term>],
                    new_value: &IrTypedBitVec<'a, S::Term>,
                ) -> IrTypedBitVec<'a, S::Term> {
                    // If we have consumed all indices, replace the entire value.
                    if indices.is_empty() {
                        return new_value.clone();
                    }

                    // Expect an array at this level.
                    let (elem_ty, elem_cnt) = match array_val.ir_type {
                        ir::Type::Array(arr) => (&arr.element_type, arr.element_count),
                        _ => panic!(
                            "ArrayUpdate: expected array type, found {:?}",
                            array_val.ir_type
                        ),
                    };

                    // Width for extracting each element slice.
                    let elem_bits = elem_ty.bit_count() as i32;

                    // Precompute index comparison helpers.
                    let index_bv = indices[0];
                    let index_width = index_bv.bitvec.get_width();

                    // Iterate over elements, build updated slice.
                    let mut concatenated = BitVec::ZeroWidth;
                    for i in (0..elem_cnt as i32).rev() {
                        // Extract original slice.
                        let high = (i + 1) * elem_bits - 1;
                        let low = i * elem_bits;
                        let orig_slice = solver.extract(&array_val.bitvec, high, low);

                        // Recursively update child if this is the selected index.
                        let updated_child = array_update_recursive(
                            solver,
                            &IrTypedBitVec {
                                ir_type: elem_ty,
                                bitvec: orig_slice.clone(),
                            },
                            &indices[1..],
                            new_value,
                        );

                        // cond = (indices[0] == i)
                        let idx_val = solver.numerical(index_width, i as u64);
                        let cond = solver.eq(&index_bv.bitvec, &idx_val);

                        let selected_slice = solver.ite(&cond, &updated_child.bitvec, &orig_slice);

                        concatenated = solver.concat(&concatenated, &selected_slice);
                    }

                    IrTypedBitVec {
                        ir_type: array_val.ir_type,
                        bitvec: concatenated,
                    }
                }

                let base_array = env.get(array).expect("Array BV must be present");
                let new_val = env.get(value).expect("Update value BV must be present");
                let index_bvs: Vec<&IrTypedBitVec<'_, S::Term>> = indices
                    .iter()
                    .map(|i| env.get(i).expect("Index BV must be present"))
                    .collect();

                array_update_recursive(solver, base_array, &index_bvs, new_val)
            }
            NodePayload::ArrayIndex { array, indices, .. } => {
                /// Build a value that corresponds to
                /// `array_val[indices...]`.
                fn array_index_recursive<'a, S: Solver>(
                    solver: &mut S,
                    array_val: &IrTypedBitVec<'a, S::Term>,
                    indices: &[&IrTypedBitVec<'a, S::Term>],
                ) -> IrTypedBitVec<'a, S::Term> {
                    // Base-case: no further indices → return the current value.
                    if indices.is_empty() {
                        return array_val.clone();
                    }

                    // The value must be an array; retrieve element type / count.
                    let (elem_ty, elem_cnt) = match array_val.ir_type {
                        ir::Type::Array(arr) => (&arr.element_type, arr.element_count),
                        _ => panic!(
                            "ArrayIndex: expected array type, found {:?}",
                            array_val.ir_type
                        ),
                    };

                    // Recursively compute each element after applying the *tail* indices.
                    let elem_bit_width = elem_ty.bit_count() as i32;
                    let children: Vec<IrTypedBitVec<'a, S::Term>> = (0..elem_cnt as i32)
                        .map(|i| {
                            let high = (i + 1) * elem_bit_width - 1;
                            let low = i * elem_bit_width;
                            let slice = solver.extract(&array_val.bitvec, high, low);
                            array_index_recursive(
                                solver,
                                &IrTypedBitVec {
                                    ir_type: elem_ty,
                                    bitvec: slice,
                                },
                                &indices[1..],
                            )
                        })
                        .collect();

                    // Build a chain of `ite` expressions that selects the requested
                    // element, falling back to the last element for OOB indices (XLS
                    // semantics).
                    let index_bv = indices[0];
                    let max_selectable = get_num_indexable_elements::<S>(index_bv, children.len());
                    let index_width = index_bv.bitvec.get_width();

                    let mut result = children[max_selectable - 1].bitvec.clone();
                    for i in (0..max_selectable - 1).rev() {
                        let idx_val = solver.numerical(index_width, i as u64);
                        let cond = solver.eq(&index_bv.bitvec, &idx_val);
                        result = solver.ite(&cond, &children[i].bitvec, &result);
                    }

                    IrTypedBitVec {
                        ir_type: elem_ty,
                        bitvec: result,
                    }
                }

                // Gather the index bit-vectors in evaluation order.
                let base_array = env.get(array).expect("Array BV must be present");
                let index_bvs: Vec<&IrTypedBitVec<'_, S::Term>> = indices
                    .iter()
                    .map(|i| env.get(i).expect("Index BV must be present"))
                    .collect();

                array_index_recursive(solver, base_array, &index_bvs)
            }
            NodePayload::DynamicBitSlice { arg, start, width } => {
                let arg_bv = env.get(arg).expect("DynamicBitSlice arg must be present");
                let start_bv = env
                    .get(start)
                    .expect("DynamicBitSlice start must be present");
                assert!(*width > 0, "DynamicBitSlice: width must be > 0");
                assert!(
                    *width <= arg_bv.bitvec.get_width(),
                    "DynamicBitSlice: width must be <= arg width"
                );

                let shifted = solver.xls_shrl(&arg_bv.bitvec, &start_bv.bitvec);
                let shifted_ext = if *width > shifted.get_width() {
                    solver.extend_to(&shifted, *width, false)
                } else {
                    shifted
                };
                let extracted = solver.slice(&shifted_ext, 0, *width);
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: extracted,
                }
            }
            NodePayload::BitSlice { arg, start, width } => {
                let arg_bv = env.get(arg).expect("BitSlice arg must be present");
                assert!(
                    *start + *width <= arg_bv.bitvec.get_width(),
                    "BitSlice: start + width must be <= arg width"
                );
                let extracted = solver.slice(&arg_bv.bitvec, *start, *width);
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: extracted,
                }
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                let arg_bv = env.get(arg).expect("BitSliceUpdate arg must be present");
                let start_bv = env
                    .get(start)
                    .expect("BitSliceUpdate start must be present");
                let update_bv = env
                    .get(update_value)
                    .expect("BitSliceUpdate update_value must be present");
                let arg_width = arg_bv.bitvec.get_width();
                let update_width = update_bv.bitvec.get_width();
                let max_width = arg_width.max(update_width);
                let arg_ext = solver.extend_to(&arg_bv.bitvec, max_width, false);
                let update_ext = solver.extend_to(&update_bv.bitvec, max_width, false);
                let ones = solver.all_ones(update_width);
                let ones_ext = solver.extend_to(&ones, max_width, false);
                let update_shifted = solver.xls_shll(&update_ext, &start_bv.bitvec);
                let ones_shifted = solver.xls_shll(&ones_ext, &start_bv.bitvec);
                let negated_ones_shifted = solver.not(&ones_shifted);
                let cleared_orig = solver.and(&arg_ext, &negated_ones_shifted);
                let combined = solver.or(&cleared_orig, &update_shifted);
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.slice(&combined, 0, arg_width),
                }
            }
            NodePayload::Assert { .. } => {
                // TODO: Turn Assert into a proof objective in Boolector.
                // For now, treat as a no-op (like Nil/AfterAll).
                continue;
            }
            NodePayload::Trace { .. } => {
                // Trace has no effect on value computation
                continue;
            }
            NodePayload::AfterAll(_) => {
                // AfterAll is a no-op for Boolector; do not insert a BV (like Nil)
                continue;
            }
            NodePayload::Nary(op, elems) => {
                let elems_bvs: Vec<&BitVec<S::Term>> = elems
                    .iter()
                    .map(|e| &(env.get(e).expect("Nary operand must be present").bitvec))
                    .collect();
                let bvs = match op {
                    NaryOp::Concat => solver.concat_many(elems_bvs),
                    NaryOp::And => solver.and_many(elems_bvs),
                    NaryOp::Nor => solver.nor_many(elems_bvs),
                    NaryOp::Or => solver.or_many(elems_bvs),
                    NaryOp::Xor => solver.xor_many(elems_bvs),
                    NaryOp::Nand => solver.nand_many(elems_bvs),
                };
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: bvs,
                }
            }
            NodePayload::Invoke { .. } => {
                // TODO: add support for Invoke
                panic!("Invoke not supported in SMT conversion");
            }
            NodePayload::OneHot { arg, lsb_prio } => {
                let a = env.get(arg).expect("OneHot arg must be present").clone();
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.xls_one_hot(&a.bitvec, *lsb_prio),
                }
            }
            NodePayload::Decode { arg, width } => {
                let a = env.get(arg).expect("Decode arg must be present").clone();
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.xls_decode(&a.bitvec, *width),
                }
            }
            NodePayload::Encode { arg } => {
                let a = env.get(arg).expect("Encode arg must be present").clone();
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.xls_encode(&a.bitvec),
                }
            }
            NodePayload::Cover { .. } => {
                // Cover statements do not contribute to value computation
                continue;
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                let sel_bv = env.get(selector).expect("Sel selector BV");
                let width = sel_bv.bitvec.get_width();
                let addr_space = if width >= usize::BITS as usize {
                    usize::MAX
                } else {
                    1usize << width
                };

                // ---------------- Validation -------------------------------------------------
                assert!(!cases.is_empty(), "Sel: must have at least one case");
                let case_count = cases.len();
                assert!(
                    case_count <= addr_space,
                    "Sel: too many cases for selector width"
                );

                // XLS rule: default iff selector width does NOT cover all cases.
                if case_count == addr_space {
                    assert!(
                        default.is_none(),
                        "Sel: default forbidden when selector covers all cases"
                    );
                } else {
                    assert!(
                        default.is_some(),
                        "Sel: default required when selector may overflow"
                    );
                }

                // ---------------- Initial accumulator & loop ------------------------------
                // Gather all case bit-vectors once.
                let mut case_bvs: Vec<_> = cases
                    .iter()
                    .map(|r| env.get(r).unwrap().bitvec.clone())
                    .collect();

                // Determine starting accumulator.
                let mut result_bv = if let Some(def_ref) = default {
                    env.get(def_ref).unwrap().bitvec.clone()
                } else {
                    assert!(
                        case_bvs.len() >= 2,
                        "Sel without default must have ≥2 cases"
                    );
                    case_bvs.pop().unwrap() // O(1) pop from back
                };

                // Iterate remaining cases from highest to 0, building the ITE chain.
                for idx in (0..case_bvs.len()).rev() {
                    let case_bv = case_bvs[idx].clone();
                    let idx_const = solver.numerical(width, idx as u64);
                    let cond = solver.eq(&sel_bv.bitvec, &idx_const);
                    result_bv = solver.ite(&cond, &case_bv, &result_bv);
                }

                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: result_bv,
                }
            }
            NodePayload::OneHotSel { selector, cases } => {
                let sel_bv = env.get(selector).expect("OneHotSel selector");
                let zero_t = solver.zero(env.get(&cases[0]).unwrap().bitvec.get_width());
                let result_bv = cases.iter().enumerate().fold(
                    solver.zero(zero_t.get_width()),
                    |acc, (idx, case_ref)| {
                        let bit = solver.slice(&sel_bv.bitvec, idx, 1);
                        let case_bv = env.get(case_ref).unwrap().bitvec.clone();
                        let selected = solver.ite(&bit, &case_bv, &zero_t);
                        solver.or(&acc, &selected)
                    },
                );
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: result_bv,
                }
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let sel_bv = env.get(selector).expect("PrioritySel selector");
                let def_bv = env
                    .get(
                        default
                            .as_ref()
                            .expect("PrioritySel currently requires default"),
                    )
                    .unwrap()
                    .bitvec
                    .clone();
                let result_bv = cases
                    .iter()
                    .enumerate()
                    .rev() // highest idx first, so last evaluated has higher priority
                    .fold(def_bv, |acc, (idx, case_ref)| {
                        let bit = solver.slice(&sel_bv.bitvec, idx, 1);
                        let case_bv = env.get(case_ref).unwrap().bitvec.clone();
                        solver.ite(&bit, &case_bv, &acc)
                    });
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: result_bv,
                }
            }
        };
        env.insert(nr, exp);
    }
    let ret = inputs.fn_ref.ret_node_ref.unwrap();
    SmtFn {
        fn_ref: inputs.fn_ref,
        inputs: inputs.inputs.values().cloned().collect(),
        output: env.remove(&ret).unwrap(),
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum EquivResult {
    Proved,
    Disproved {
        inputs: Vec<IrValue>,
        outputs: (IrValue, IrValue),
    },
}

fn check_aligned_fn_equiv_internal<'a, S: Solver>(
    solver: &mut S,
    lhs: &SmtFn<'a, S::Term>,
    rhs: &SmtFn<'a, S::Term>,
) -> EquivResult {
    let diff = solver.ne(&lhs.output.bitvec, &rhs.output.bitvec);
    solver.assert(&diff).unwrap();

    match solver.check().unwrap() {
        Response::Sat => {
            let mut get_value = |i: &IrTypedBitVec<'a, S::Term>| -> IrValue {
                solver.get_value(&i.bitvec, &i.ir_type).unwrap()
            };
            EquivResult::Disproved {
                inputs: lhs.inputs.iter().map(|i| get_value(i)).collect(),
                outputs: (get_value(&lhs.output), get_value(&rhs.output)),
            }
        }
        Response::Unsat => EquivResult::Proved,
        Response::Unknown => panic!("Solver returned unknown result"),
    }
}

fn check_fn_equiv_internal<'a, S: Solver>(
    solver: &mut S,
    lhs: &ir::Fn,
    rhs: &ir::Fn,
    allow_flatten: bool,
) -> EquivResult {
    let fn_inputs_1 = get_fn_inputs(solver, lhs, Some("lhs"));
    let fn_inputs_2 = get_fn_inputs(solver, rhs, Some("rhs"));
    let aligned_fn_inputs = align_fn_inputs(solver, &fn_inputs_1, &fn_inputs_2, allow_flatten);
    let smt_fn_1 = ir_to_smt(solver, &aligned_fn_inputs.lhs);
    let smt_fn_2 = ir_to_smt(solver, &aligned_fn_inputs.rhs);
    check_aligned_fn_equiv_internal(solver, &smt_fn_1, &smt_fn_2)
}

pub fn prove_ir_fn_equiv<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &ir::Fn,
    rhs: &ir::Fn,
    allow_flatten: bool,
) -> EquivResult {
    let mut solver = S::new(solver_config).unwrap();
    check_fn_equiv_internal(&mut solver, lhs, rhs, allow_flatten)
}

// Add parallel equivalence-checking strategies that were previously only
// implemented in the Boolector-specific backend.  These generic versions work
// with any `Solver` implementation by accepting a factory closure that can
// create fresh solver instances on demand.

/// Helper: create a variant of `f` that returns only the single output bit at
/// position `bit`.
fn make_bit_fn(f: &ir::Fn, bit: usize) -> ir::Fn {
    use crate::xls_ir::ir::{Node, NodePayload, NodeRef, Type};

    let mut nf = f.clone();
    let ret_ref = nf.ret_node_ref.expect("Function must have a return node");
    let slice_ref = NodeRef {
        index: nf.nodes.len(),
    };
    nf.nodes.push(Node {
        text_id: nf.nodes.len(),
        name: None,
        ty: Type::Bits(1),
        payload: NodePayload::BitSlice {
            arg: ret_ref,
            start: bit,
            width: 1,
        },
        pos: None,
    });
    nf.ret_node_ref = Some(slice_ref);
    nf.ret_ty = Type::Bits(1);
    nf
}

/// Prove equivalence by splitting on each output bit in parallel.
/// `solver_factory` must be able to create an independent solver instance for
/// each spawned thread.
pub fn prove_ir_fn_equiv_output_bits_parallel<S: Solver>(
    solver_config: &S::Config,
    lhs: &ir::Fn,
    rhs: &ir::Fn,
    allow_flatten: bool,
) -> EquivResult {
    let width = lhs.ret_ty.bit_count();
    assert_eq!(width, rhs.ret_ty.bit_count(), "Return widths must match");
    if width == 0 {
        // Zero-width values – fall back to the standard equivalence prover because
        // there is no bit to split on.
        return prove_ir_fn_equiv::<S>(solver_config, lhs, rhs, allow_flatten);
    }

    let found = Arc::new(AtomicBool::new(false));
    let counterexample: Arc<Mutex<Option<EquivResult>>> = Arc::new(Mutex::new(None));
    let next_bit = Arc::new(AtomicUsize::new(0));

    let thread_cnt = std::cmp::min(width, num_cpus::get());

    std::thread::scope(|scope| {
        for _ in 0..thread_cnt {
            let lhs_cl = lhs.clone();
            let rhs_cl = rhs.clone();
            let found_cl = found.clone();
            let cex_cl = counterexample.clone();
            let next_cl = next_bit.clone();

            scope.spawn(move || {
                loop {
                    if found_cl.load(Ordering::SeqCst) {
                        break;
                    }
                    let idx = next_cl.fetch_add(1, Ordering::SeqCst);
                    if idx >= width {
                        break;
                    }

                    let lf = make_bit_fn(&lhs_cl, idx);
                    let rf = make_bit_fn(&rhs_cl, idx);
                    let res = prove_ir_fn_equiv::<S>(solver_config, &lf, &rf, allow_flatten);
                    if let EquivResult::Disproved { .. } = &res {
                        let mut guard = cex_cl.lock().unwrap();
                        *guard = Some(res.clone());
                        found_cl.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            });
        }
    });

    let maybe_cex = {
        let mut guard = counterexample.lock().unwrap();
        guard.take()
    };
    if let Some(res) = maybe_cex {
        res
    } else {
        EquivResult::Proved
    }
}

/// Prove equivalence by case-splitting on a single input bit value (0 / 1).
/// `split_input_index` selects the parameter to split on and
/// `split_input_bit_index` selects the bit inside that parameter.
pub fn prove_ir_fn_equiv_split_input_bit<S: Solver>(
    solver_config: &S::Config,
    lhs: &ir::Fn,
    rhs: &ir::Fn,
    split_input_index: usize,
    split_input_bit_index: usize,
    allow_flatten: bool,
) -> EquivResult {
    if lhs.params.is_empty() || rhs.params.is_empty() {
        return prove_ir_fn_equiv::<S>(solver_config, lhs, rhs, allow_flatten);
    }

    assert_eq!(
        lhs.params.len(),
        rhs.params.len(),
        "Parameter count mismatch"
    );
    assert!(
        split_input_index < lhs.params.len(),
        "split_input_index out of bounds"
    );
    assert!(
        split_input_bit_index < lhs.params[split_input_index].ty.bit_count(),
        "split_input_bit_index out of bounds"
    );

    for bit_val in 0..=1u64 {
        let mut solver = S::new(solver_config).unwrap();
        // Build aligned SMT representations first so we can assert the bit-constraint.
        let fn_inputs_lhs = get_fn_inputs(&mut solver, lhs, Some("lhs"));
        let fn_inputs_rhs = get_fn_inputs(&mut solver, rhs, Some("rhs"));
        let aligned = align_fn_inputs(&mut solver, &fn_inputs_lhs, &fn_inputs_rhs, allow_flatten);
        let smt_lhs = ir_to_smt(&mut solver, &aligned.lhs);
        let smt_rhs = ir_to_smt(&mut solver, &aligned.rhs);

        // Locate the chosen parameter on the LHS side.
        let param_name = &lhs.params[split_input_index].name;
        let param_bv = aligned
            .lhs
            .inputs
            .get(param_name)
            .expect("Parameter BV missing")
            .bitvec
            .clone();
        let bit_bv = solver.slice(&param_bv, split_input_bit_index, 1);
        let val_bv = solver.numerical(1, bit_val);
        let eq_bv = solver.eq(&bit_bv, &val_bv);
        solver.assert(&eq_bv).unwrap();

        // Assert outputs differ.
        let diff = solver.ne(&smt_lhs.output.bitvec, &smt_rhs.output.bitvec);
        solver.assert(&diff).unwrap();

        match solver.check().unwrap() {
            Response::Sat => {
                let mut get_val = |irbv: &IrTypedBitVec<'_, S::Term>| -> IrValue {
                    solver.get_value(&irbv.bitvec, &irbv.ir_type).unwrap()
                };
                return EquivResult::Disproved {
                    inputs: smt_lhs.inputs.iter().map(|i| get_val(i)).collect(),
                    outputs: (get_val(&smt_lhs.output), get_val(&smt_rhs.output)),
                };
            }
            Response::Unsat => {
                // Continue to next bit value
            }
            Response::Unknown => panic!("Solver returned unknown result"),
        }
    }

    EquivResult::Proved
}

mod test_utils {
    #[macro_export]
    macro_rules! assert_smt_fn_eq {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $ir_text:expr, $expected:expr) => {
            #[test]
            fn $fn_name() {
                let mut parser = crate::xls_ir::ir_parser::Parser::new(&$ir_text);
                let f = parser.parse_fn().unwrap();
                let mut solver = <$solver_type>::new($solver_config).unwrap();
                let fn_inputs = get_fn_inputs(&mut solver, &f, None);
                let smt_fn = ir_to_smt(&mut solver, &fn_inputs);
                let expected = $expected(&mut solver, &fn_inputs);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &smt_fn.output.bitvec,
                    &expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! assert_ir_value_to_bv_eq {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $ir_value:expr, $ir_type:expr, $expected:expr) => {
            #[test]
            fn $fn_name() {
                let mut solver = <$solver_type>::new($solver_config).unwrap();
                let ir_value = $ir_value;
                let ir_type = $ir_type;
                let bv = ir_value_to_bv(&mut solver, &ir_value, &ir_type);
                assert_eq!(bv.ir_type, &ir_type);
                let expected = $expected(&mut solver);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &bv.bitvec,
                    &expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! test_ir_fn_equiv {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $ir_text_1:expr, $ir_text_2:expr, $allow_flatten:expr, $result:pat) => {
            #[test]
            fn $fn_name() {
                let mut parser = crate::xls_ir::ir_parser::Parser::new($ir_text_1);
                let f1 = parser.parse_fn().unwrap();
                let mut parser = crate::xls_ir::ir_parser::Parser::new($ir_text_2);
                let f2 = parser.parse_fn().unwrap();
                let result =
                    prove_ir_fn_equiv::<$solver_type>($solver_config, &f1, &f2, $allow_flatten);
                assert!(matches!(result, $result));
            }
        };
    }

    #[macro_export]
    macro_rules! test_assert_fn_equiv_to_self {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $ir_text:expr) => {
            crate::test_ir_fn_equiv!(
                $fn_name,
                $solver_type,
                $solver_config,
                $ir_text,
                $ir_text,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_assert_fn_inequiv {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $ir_text_1:expr, $ir_text_2:expr, $allow_flatten:expr) => {
            crate::test_ir_fn_equiv!(
                $fn_name,
                $solver_type,
                $solver_config,
                $ir_text_1,
                $ir_text_2,
                $allow_flatten,
                EquivResult::Disproved { .. }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_bits {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_bits,
                $solver_type,
                $solver_config,
                IrValue::u32(0x12345678),
                ir::Type::Bits(32),
                |solver: &mut $solver_type| solver.from_raw_str(32, "#x12345678")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_bits {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_bits,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[32] {
                    ret literal.1: bits[32] = literal(value=0x12345678, id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Term>| solver
                    .from_raw_str(32, "#x12345678")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_array {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_array,
                $solver_type,
                $solver_config,
                IrValue::make_array(&[
                    IrValue::make_array(&[
                        IrValue::make_ubits(8, 0x12).unwrap(),
                        IrValue::make_ubits(8, 0x34).unwrap(),
                    ])
                    .unwrap(),
                    IrValue::make_array(&[
                        IrValue::make_ubits(8, 0x56).unwrap(),
                        IrValue::make_ubits(8, 0x78).unwrap(),
                    ])
                    .unwrap(),
                    IrValue::make_array(&[
                        IrValue::make_ubits(8, 0x9a).unwrap(),
                        IrValue::make_ubits(8, 0xbc).unwrap(),
                    ])
                    .unwrap(),
                ])
                .unwrap(),
                ir::Type::Array(ir::ArrayTypeData {
                    element_type: Box::new(ir::Type::Array(ir::ArrayTypeData {
                        element_type: Box::new(ir::Type::Bits(8)),
                        element_count: 2,
                    })),
                    element_count: 3,
                }),
                |solver: &mut $solver_type| solver.from_raw_str(48, "#xbc9a78563412")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_array {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_array,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[8][2][3] {
                    ret literal.1: bits[8][2][3] = literal(value=[[0x12, 0x34], [0x56, 0x78], [0x9a, 0xbc]], id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Term>| solver
                    .from_raw_str(48, "#xbc9a78563412")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_tuple {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_tuple,
                $solver_type,
                $solver_config,
                IrValue::make_tuple(&[
                    IrValue::make_ubits(8, 0x12).unwrap(),
                    IrValue::make_ubits(4, 0x4).unwrap(),
                ]),
                ir::Type::Tuple(vec![
                    Box::new(ir::Type::Bits(8)),
                    Box::new(ir::Type::Bits(4)),
                ]),
                |solver: &mut $solver_type| solver.from_raw_str(12, "#x124")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_tuple {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_tuple,
                $solver_type,
                $solver_config,
                r#"fn f() -> (bits[8], bits[4]) {
                   ret literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Term>| solver
                    .from_raw_str(12, "#x124")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_token {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_token,
                $solver_type,
                $solver_config,
                IrValue::make_ubits(0, 0).unwrap(),
                ir::Type::Token,
                |_: &mut $solver_type| BitVec::ZeroWidth
            );
        };
    }

    #[macro_export]
    macro_rules! test_unop_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $unop_xls_name:expr, $unop:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                r#"fn f(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = "#
                    .to_string()
                    + $unop_xls_name
                    + r#"(x, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    $unop(solver, &inputs.inputs.get("x").unwrap().bitvec)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_all_unops {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_unop_base!(test_not, $solver_type, $solver_config, "not", Solver::not);
            crate::test_unop_base!(test_neg, $solver_type, $solver_config, "neg", Solver::neg);
            crate::test_unop_base!(
                test_or_reduce,
                $solver_type,
                $solver_config,
                "or_reduce",
                Solver::or_reduce
            );
            crate::test_unop_base!(
                test_and_reduce,
                $solver_type,
                $solver_config,
                "and_reduce",
                Solver::and_reduce
            );
            crate::test_unop_base!(
                test_xor_reduce,
                $solver_type,
                $solver_config,
                "xor_reduce",
                Solver::xor_reduce
            );
            crate::test_unop_base!(
                test_identity,
                $solver_type,
                $solver_config,
                "identity",
                |_, a: &BitVec<<$solver_type as Solver>::Term>| a.clone()
            );
            crate::test_unop_base!(
                test_reverse,
                $solver_type,
                $solver_config,
                "reverse",
                Solver::reverse
            );
        };
    }

    #[macro_export]
    macro_rules! test_binop_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $binop_xls_name:expr,
            $expected:expr, $lhs_width:expr, $rhs_width:expr, $result_width:expr) => {
            #[test]
            fn $fn_name() {
                let lhs_width_str = $lhs_width.to_string();
                let rhs_width_str = $rhs_width.to_string();
                let ir_text = format!(
                    r#"fn f(x: bits[{}], y: bits[{}]) -> bits[{}] {{
                    ret get_param.1: bits[{}] = {}(x, y, id=1)
                }}"#,
                    lhs_width_str, rhs_width_str, $result_width, $result_width, $binop_xls_name
                );
                let mut parser = crate::xls_ir::ir_parser::Parser::new(&ir_text);
                let f = parser.parse_fn().unwrap();
                let mut solver = <$solver_type>::new($solver_config).unwrap();
                let fn_inputs = get_fn_inputs(&mut solver, &f, None);
                let smt_fn = ir_to_smt(&mut solver, &fn_inputs);
                let x = fn_inputs.inputs.get("x").unwrap().bitvec.clone();
                let y = fn_inputs.inputs.get("y").unwrap().bitvec.clone();
                let expected = $expected(&mut solver, x, y);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &smt_fn.output.bitvec,
                    &expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! test_binop_simple {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $binop_xls_name:expr, $binop:ident,
            $lhs_width:expr, $rhs_width:expr, $result_width:expr) => {
            crate::test_binop_base!(
                $fn_name,
                $solver_type,
                $solver_config,
                $binop_xls_name,
                |solver: &mut $solver_type, x, y| { solver.$binop(&x, &y) },
                $lhs_width,
                $rhs_width,
                $result_width
            );
        };
    }

    #[macro_export]
    macro_rules! test_all_binops {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_binop_simple!(test_add, $solver_type, $solver_config, "add", add, 8, 8, 8);
            crate::test_binop_simple!(test_sub, $solver_type, $solver_config, "sub", sub, 8, 8, 8);
            crate::test_binop_simple!(test_eq, $solver_type, $solver_config, "eq", eq, 8, 8, 1);
            crate::test_binop_simple!(test_ne, $solver_type, $solver_config, "ne", ne, 8, 8, 1);
            crate::test_binop_simple!(test_uge, $solver_type, $solver_config, "uge", uge, 8, 8, 1);
            crate::test_binop_simple!(test_ugt, $solver_type, $solver_config, "ugt", ugt, 8, 8, 1);
            crate::test_binop_simple!(test_ult, $solver_type, $solver_config, "ult", ult, 8, 8, 1);
            crate::test_binop_simple!(test_ule, $solver_type, $solver_config, "ule", ule, 8, 8, 1);
            crate::test_binop_simple!(test_sgt, $solver_type, $solver_config, "sgt", sgt, 8, 8, 1);
            crate::test_binop_simple!(test_sge, $solver_type, $solver_config, "sge", sge, 8, 8, 1);
            crate::test_binop_simple!(test_slt, $solver_type, $solver_config, "slt", slt, 8, 8, 1);
            crate::test_binop_simple!(test_sle, $solver_type, $solver_config, "sle", sle, 8, 8, 1);
            crate::test_binop_simple!(
                test_xls_shll,
                $solver_type,
                $solver_config,
                "shll",
                xls_shll,
                8,
                4,
                8
            );
            crate::test_binop_simple!(
                test_xls_shrl,
                $solver_type,
                $solver_config,
                "shrl",
                xls_shrl,
                8,
                4,
                8
            );
            crate::test_binop_simple!(
                test_xls_shra,
                $solver_type,
                $solver_config,
                "shra",
                xls_shra,
                8,
                4,
                8
            );
            crate::test_binop_base!(
                test_xls_umul,
                $solver_type,
                $solver_config,
                "umul",
                |solver: &mut $solver_type, x, y| { solver.xls_arbitrary_width_umul(&x, &y, 16) },
                8,
                12,
                16
            );
            crate::test_binop_base!(
                test_xls_smul,
                $solver_type,
                $solver_config,
                "smul",
                |solver: &mut $solver_type, x, y| { solver.xls_arbitrary_width_smul(&x, &y, 16) },
                8,
                12,
                16
            );
            crate::test_binop_simple!(
                test_xls_udiv,
                $solver_type,
                $solver_config,
                "udiv",
                xls_udiv,
                8,
                8,
                8
            );
            crate::test_binop_simple!(
                test_xls_sdiv,
                $solver_type,
                $solver_config,
                "sdiv",
                xls_sdiv,
                8,
                8,
                8
            );
            crate::test_binop_simple!(
                test_xls_umod,
                $solver_type,
                $solver_config,
                "umod",
                xls_umod,
                8,
                8,
                8
            );
            crate::test_binop_simple!(
                test_xls_smod,
                $solver_type,
                $solver_config,
                "smod",
                xls_smod,
                8,
                8,
                8
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_tuple_index {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_tuple_index,
                $solver_type,
                $solver_config,
                r#"fn f(input: (bits[8], bits[4])) -> bits[8] {
                    ret tuple_index.1: bits[8] = tuple_index(input, index=0, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let tuple = inputs.inputs.get("input").unwrap().bitvec.clone();
                    solver.extract(&tuple, 11, 4)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_tuple_index_literal {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_tuple_index_literal,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[8] {
                    literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                    ret tuple_index.1: bits[8] = tuple_index(literal.1, index=0, id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Term>| {
                    solver.from_raw_str(8, "#x12")
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_tuple_reverse {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_tuple_reverse,
                $solver_type,
                $solver_config,
                r#"fn f(a: (bits[8], bits[4])) -> (bits[4], bits[8]) {
                    tuple_index.3: bits[4] = tuple_index(a, index=1, id=3)
                    tuple_index.5: bits[8] = tuple_index(a, index=0, id=5)
                    ret tuple.6: (bits[4], bits[8]) = tuple(tuple_index.3, tuple_index.5, id=6)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let tuple = inputs.inputs.get("a").unwrap().bitvec.clone();
                    let tuple_0 = solver.extract(&tuple, 11, 4);
                    let tuple_1 = solver.extract(&tuple, 3, 0);
                    solver.concat(&tuple_1, &tuple_0)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_tuple_flattened {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_ir_tuple_flattened,
                $solver_type,
                $solver_config,
                r#"fn f() -> (bits[8], bits[4]) {
                    ret tuple.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                }"#,
                r#"fn g() -> bits[12] {
                    ret tuple.1: bits[12] = literal(value=0x124, id=1)
                }"#,
                true,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_tuple_literal_vs_constructed {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_tuple_literal_vs_constructed,
                $solver_type,
                $solver_config,
                r#"fn lhs() -> (bits[8], bits[4]) {
                    ret lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                }"#,
                r#"fn rhs() -> (bits[8], bits[4]) {
                    lit0: bits[8] = literal(value=0x12, id=1)
                    lit1: bits[4] = literal(value=0x4, id=2)
                    ret tup: (bits[8], bits[4]) = tuple(lit0, lit1, id=3)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_tuple_index_on_literal {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_tuple_index_on_literal,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[8] {
                    lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                    ret idx0: bits[8] = tuple_index(lit_tuple, index=0, id=2)
                }"#,
                r#"fn g() -> bits[8] {
                    ret lit: bits[8] = literal(value=0x12, id=1)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_array_index_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $index:expr, $expected_low:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8][4] id=1) -> bits[8] {
                    literal.4: bits[3] = literal(value="#.to_string() + $index + r#", id=4)
                    ret array_index.5: bits[8] = array_index(input, indices=[literal.4], assumed_in_bounds=true, id=5)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                    solver.extract(&array, $expected_low + 7, $expected_low)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_array_index {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_array_index_base!(
                test_ir_array_index_0,
                $solver_type,
                $solver_config,
                "0",
                0
            );
            crate::test_ir_array_index_base!(
                test_ir_array_index_1,
                $solver_type,
                $solver_config,
                "1",
                8
            );
            crate::test_ir_array_index_base!(
                test_ir_array_index_2,
                $solver_type,
                $solver_config,
                "2",
                16
            );
            crate::test_ir_array_index_base!(
                test_ir_array_index_3,
                $solver_type,
                $solver_config,
                "3",
                24
            );
            crate::test_ir_array_index_base!(
                test_ir_array_index_out_of_bounds,
                $solver_type,
                $solver_config,
                "4",
                24
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_array_index_multi_level {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_array_index_multi_level,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8][4][2] id=1) -> bits[8] {
                    literal.4: bits[2] = literal(value=1, id=4)
                    ret array_index.6: bits[8] = array_index(input, indices=[literal.4], assumed_in_bounds=true, id=6)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                    solver.extract(&array, 63, 32)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_array_index_deep_multi_level {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_ir_array_index_deep_multi_level,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8][4][2] id=1) -> bits[8] {
                    literal.4: bits[2] = literal(value=1, id=4)
                    literal.5: bits[2] = literal(value=0, id=5)
                    ret array_index.6: bits[8] = array_index(input, indices=[literal.4, literal.5], assumed_in_bounds=true, id=6)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                    solver.extract(&array, 39, 32)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_array_update_inbound_value {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_array_update_elem0_value,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[4][4] id=1, val: bits[4] id=2) -> bits[4][4] {
                    lit: bits[2] = literal(value=1, id=3)
                    ret upd: bits[4][4] = array_update(input, val, indices=[lit], id=4)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                    let pre = solver.extract(&array, 3, 0);
                    let with_mid = solver.concat(&val, &pre);
                    let post = solver.extract(&array, 15, 8);
                    solver.concat(&post, &with_mid)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_array_update_nested {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_array_update_nested,
                $solver_type,
                $solver_config,
                r#"fn lhs(input: bits[8][4][4] id=1, val: bits[8][4] id=2) -> bits[8][4][4] {
                    idx0: bits[2] = literal(value=1, id=3)
                    ret upd: bits[8][4][4] = array_update(input, val, indices=[idx0], id=5)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                    let pre = solver.extract(&array, 31, 0);
                    let with_mid = solver.concat(&val, &pre);
                    let post = solver.extract(&array, 127, 64);
                    solver.concat(&post, &with_mid)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_array_update_deep_nested {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_array_update_deep_nested,
                $solver_type,
                $solver_config,
                r#"fn lhs(input: bits[8][2][2] id=1, val: bits[8] id=2) -> bits[8][2][2] {
                    idx0: bits[2] = literal(value=1, id=3)
                    idx1: bits[2] = literal(value=0, id=4)
                    ret upd: bits[8][2][2] = array_update(input, val, indices=[idx0, idx1], id=5)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                    let pre = solver.extract(&array, 15, 0);
                    let with_mid = solver.concat(&val, &pre);
                    let post = solver.extract(&array, 31, 24);
                    solver.concat(&post, &with_mid)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_extend {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $extend_width:expr, $signed:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                format!(
                    r#"fn f(x: bits[8]) -> bits[{}] {{
                    ret get_param.1: bits[{}] = {}(x, new_bit_count={}, id=1)
                }}"#,
                    $extend_width,
                    $extend_width,
                    if $signed { "sign_ext" } else { "zero_ext" },
                    $extend_width,
                ),
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                    solver.extend_to(&x, $extend_width, $signed)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_dynamic_bit_slice_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $start:expr, $width:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                format!(r#"fn f(input: bits[8]) -> bits[{}] {{
                    start: bits[4] = literal(value={}, id=2)
                    ret get_param.1: bits[{}] = dynamic_bit_slice(input, start, width={}, id=1)
                }}"#, $width, $start, $width, $width),
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let needed_width = $start + $width;
                    let input_ext = if needed_width > input.get_width() {
                        solver.extend_to(&input, needed_width, false)
                    } else {
                        input
                    };
                    solver.slice(&input_ext, $start, $width)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_dynamic_bit_slice {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_dynamic_bit_slice_base!(
                test_dynamic_bit_slice_0_4,
                $solver_type,
                $solver_config,
                0,
                4
            );
            crate::test_dynamic_bit_slice_base!(
                test_dynamic_bit_slice_5_4,
                $solver_type,
                $solver_config,
                5,
                4
            );
            crate::test_dynamic_bit_slice_base!(
                test_dynamic_bit_slice_0_8,
                $solver_type,
                $solver_config,
                0,
                8
            );
            crate::test_dynamic_bit_slice_base!(
                test_dynamic_bit_slice_5_8,
                $solver_type,
                $solver_config,
                5,
                8
            );
            crate::test_dynamic_bit_slice_base!(
                test_dynamic_bit_slice_10_4,
                $solver_type,
                $solver_config,
                10,
                4
            );
            crate::test_dynamic_bit_slice_base!(
                test_dynamic_bit_slice_10_8,
                $solver_type,
                $solver_config,
                10,
                8
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $start:expr, $width:expr) => {
            crate::test_ir_fn_equiv!(
                $fn_name,
                $solver_type,
                $solver_config,
                &format!(r#"fn f(input: bits[8]) -> bits[{}] {{
                    ret get_param.1: bits[{}] = bit_slice(input, start={}, width={}, id=1)
                }}"#, $width, $width, $start, $width),
                &format!(r#"fn f(input: bits[8]) -> bits[{}] {{
                    start: bits[4] = literal(value={}, id=2)
                    ret get_param.1: bits[{}] = dynamic_bit_slice(input, start, width={}, id=1)
                }}"#, $width, $start, $width, $width),
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_bit_slice_base!(test_bit_slice_0_4, $solver_type, $solver_config, 0, 4);
            crate::test_bit_slice_base!(test_bit_slice_5_4, $solver_type, $solver_config, 5, 3);
            crate::test_bit_slice_base!(test_bit_slice_0_8, $solver_type, $solver_config, 0, 8);
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_update_zero {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_bit_slice_update_zero,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=0, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                    let input_upper = solver.slice(&input, 4, 4);
                    let updated = solver.concat(&input_upper, &slice);
                    updated
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_update_middle {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_bit_slice_update_middle,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=1, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                    let input_lower = solver.slice(&input, 0, 1);
                    let input_upper = solver.slice(&input, 5, 3);
                    let updated = solver.concat(&slice, &input_lower);
                    let updated = solver.concat(&input_upper, &updated);
                    updated
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_update_end {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_bit_slice_update_end,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=4, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                    let input_lower = solver.slice(&input, 0, 4);
                    let updated = solver.concat(&slice, &input_lower);
                    updated
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_update_beyond_end {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_bit_slice_update_beyond_end,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[8], slice: bits[10]) -> bits[8] {
                    start: bits[4] = literal(value=4, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                    let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                    let input_lower = solver.slice(&input, 0, 4);
                    let slice_extracted = solver.slice(&slice, 0, 4);
                    let updated = solver.concat(&slice_extracted, &input_lower);
                    updated
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_update_wider_update_value {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_bit_slice_update_wider_update_value,
                $solver_type,
                $solver_config,
                r#"fn f(input: bits[7] id=1) -> bits[5] {
                    slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
                    ret upd.3: bits[5] = bit_slice_update(slice.2, input, input, id=3)
                }"#,
                r#"fn f(input: bits[7] id=1) -> bits[5] {
                    slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
                    ret upd.3: bits[5] = bit_slice_update(slice.2, input, input, id=3)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_bit_slice_update_large_update_value {
        ($solver_type:ty, $solver_config:expr) => {
            crate::assert_smt_fn_eq!(
                test_bit_slice_update_large_update_value,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[32] {
                    operand: bits[32] = literal(value=0xABCD1234, id=1)
                    start: bits[5] = literal(value=4, id=2)
                    upd_val: bits[80] = literal(value=0xFFFFFFFFFFFFFFFFFFF, id=3)
                    ret r: bits[32] = bit_slice_update(operand, start, upd_val, id=4)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Term>| {
                    solver.from_raw_str(32, "#xFFFFFFF4")
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_fuzz_dynamic_bit_slice_shrink_panics {
        ($solver_type:ty, $solver_config:expr) => {
            #[test]
            fn test_fuzz_dynamic_bit_slice_shrink_panics() {
                let ir = r#"fn bad(input: bits[32]) -> bits[16] {
                    start: bits[4] = literal(value=4, id=2)
                    ret r: bits[16] = dynamic_bit_slice(input, start, width=16, id=1)
                }"#;
                let f = crate::xls_ir::ir_parser::Parser::new(ir)
                    .parse_fn()
                    .unwrap();
                let mut solver = <$solver_type>::new($solver_config).unwrap();
                let inputs = get_fn_inputs(&mut solver, &f, None);
                // This call should not panic
                let _ = ir_to_smt(&mut solver, &inputs);
            }
        };
    }

    #[macro_export]
    macro_rules! test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob,
                $solver_type,
                $solver_config,
                r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
                    literal_255: bits[8] = literal(value=255, id=3)
                    bsu1: bits[8] = bit_slice_update(input, input, input, id=2)
                    ret bsu2: bits[8] = bit_slice_update(literal_255, literal_255, bsu1, id=4)
                }"#,
                r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
                    ret literal_255: bits[8] = literal(value=255, id=3)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_prove_fn_equiv {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_assert_fn_equiv_to_self!(
                test_prove_fn_equiv,
                $solver_type,
                $solver_config,
                r#"fn f(x: bits[8], y: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = identity(x, id=1)
                }"#
            );
        };
    }

    #[macro_export]
    macro_rules! test_prove_fn_inequiv {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_prove_fn_inequiv,
                $solver_type,
                $solver_config,
                r#"fn f(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = identity(x, id=1)
                }"#,
                r#"fn g(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = not(x, id=1)
                }"#,
                false,
                EquivResult::Disproved { .. }
            );
        };
    }

    // --------------------------------------------------------------
    // Nary operation test helpers (Concat/And/Or/Xor/Nor/Nand)
    // --------------------------------------------------------------
    #[macro_export]
    macro_rules! test_nary_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $op_name:expr, $builder:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                concat!(
                    "fn f(a: bits[4], b: bits[4], c: bits[4]) -> bits[4] {\n    ret nary.1: bits[4] = ",
                    $op_name,
                    "(a, b, c, id=1)\n}"
                ),
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let a = inputs.inputs.get("a").unwrap().bitvec.clone();
                    let b = inputs.inputs.get("b").unwrap().bitvec.clone();
                    let c = inputs.inputs.get("c").unwrap().bitvec.clone();
                    $builder(solver, a, b, c)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_all_nary {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_nary_base!(
                test_concat_nary,
                $solver_type,
                $solver_config,
                "concat",
                |s: &mut $solver_type, a, b, c| { s.concat_many(vec![&a, &b, &c]) }
            );
            crate::test_nary_base!(
                test_and_nary,
                $solver_type,
                $solver_config,
                "and",
                |s: &mut $solver_type, a, b, c| { s.and_many(vec![&a, &b, &c]) }
            );
            crate::test_nary_base!(
                test_or_nary,
                $solver_type,
                $solver_config,
                "or",
                |s: &mut $solver_type, a, b, c| { s.or_many(vec![&a, &b, &c]) }
            );
            crate::test_nary_base!(
                test_xor_nary,
                $solver_type,
                $solver_config,
                "xor",
                |s: &mut $solver_type, a, b, c| { s.xor_many(vec![&a, &b, &c]) }
            );
            crate::test_nary_base!(
                test_nor_nary,
                $solver_type,
                $solver_config,
                "nor",
                |s: &mut $solver_type, a, b, c| { s.nor_many(vec![&a, &b, &c]) }
            );
            crate::test_nary_base!(
                test_nand_nary,
                $solver_type,
                $solver_config,
                "nand",
                |s: &mut $solver_type, a, b, c| { s.nand_many(vec![&a, &b, &c]) }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_decode_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $in_width:expr, $out_width:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                format!(
                    r#"fn f(x: bits[{iw}]) -> bits[{ow}] {{
    ret d.1: bits[{ow}] = decode(x, width={ow}, id=1)
}}"#,
                    iw = $in_width,
                    ow = $out_width
                ),
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                    solver.xls_decode(&x, $out_width)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_encode_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $in_width:expr, $out_width:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                format!(
                    r#"fn f(x: bits[{iw}]) -> bits[{ow}] {{
                        ret e.1: bits[{ow}] = encode(x, id=1)
                    }}"#,
                    iw = $in_width,
                    ow = $out_width
                ),
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                    solver.xls_encode(&x)
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_one_hot_base {
        ($fn_name:ident, $solver_type:ty, $solver_config:expr, $prio:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver_type,
                $solver_config,
                if $prio {
                    r#"fn f(x: bits[16]) -> bits[16] {
                        ret oh.1: bits[16] = one_hot(x, lsb_prio=true, id=1)
                    }"#
                } else {
                    r#"fn f(x: bits[16]) -> bits[16] {
                        ret oh.1: bits[16] = one_hot(x, lsb_prio=false, id=1)
                    }"#
                },
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Term>| {
                    let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                    if $prio {
                        solver.xls_one_hot_lsb_prio(&x)
                    } else {
                        solver.xls_one_hot_msb_prio(&x)
                    }
                }
            );
        };
    }

    // ----------------------------
    // Sel / OneHotSel / PrioritySel tests
    // ----------------------------

    // sel basic: selector in range (bits[2]=1) -> second case
    #[macro_export]
    macro_rules! test_sel_basic {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_sel_basic,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[4] {
                    selidx: bits[2] = literal(value=1, id=10)
                    a: bits[4] = literal(value=10, id=11)
                    b: bits[4] = literal(value=11, id=12)
                    c: bits[4] = literal(value=12, id=13)
                    d: bits[4] = literal(value=13, id=14)
                    ret s: bits[4] = sel(selidx, cases=[a, b, c, d], id=15)
                }"#,
                r#"fn g() -> bits[4] {
                    ret k: bits[4] = literal(value=11, id=1)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    // sel default path: selector out of range chooses default
    #[macro_export]
    macro_rules! test_sel_default {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_sel_default,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[4] {
                    selidx: bits[2] = literal(value=3, id=10)
                    a: bits[4] = literal(value=10, id=11)
                    b: bits[4] = literal(value=11, id=12)
                    c: bits[4] = literal(value=12, id=13)
                    def: bits[4] = literal(value=15, id=14)
                    ret s: bits[4] = sel(selidx, cases=[a, b, c], default=def, id=15)
                }"#,
                r#"fn g() -> bits[4] {
                    ret k: bits[4] = literal(value=15, id=1)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_sel_missing_default_panics {
        ($solver_type:ty, $solver_config:expr) => {
            // Invalid Sel spec tests (expect panic)
            #[should_panic]
            #[test]
            fn test_sel_missing_default_panics() {
                let ir = r#"fn bad() -> bits[4] {
                    idx: bits[2] = literal(value=3, id=1)
                    a: bits[4] = literal(value=1, id=2)
                    b: bits[4] = literal(value=2, id=3)
                    c: bits[4] = literal(value=3, id=4)
                    ret s: bits[4] = sel(idx, cases=[a, b, c], id=5)
                }"#;
                let f = crate::xls_ir::ir_parser::Parser::new(ir)
                    .parse_fn()
                    .unwrap();
                let mut solver = <$solver_type>::new($solver_config).unwrap();
                let inputs = get_fn_inputs(&mut solver, &f, None);
                // Should panic during conversion due to missing default
                let _ = ir_to_smt(&mut solver, &inputs);
            }
        };
    }

    #[macro_export]
    macro_rules! test_sel_unexpected_default_panics {
        ($solver_type:ty, $solver_config:expr) => {
            #[should_panic]
            #[test]
            fn test_sel_unexpected_default_panics() {
                let ir = r#"fn bad() -> bits[4] {
                    idx: bits[2] = literal(value=1, id=1)
                    a: bits[4] = literal(value=1, id=2)
                    b: bits[4] = literal(value=2, id=3)
                    c: bits[4] = literal(value=3, id=4)
                    d: bits[4] = literal(value=4, id=5)
                    def: bits[4] = literal(value=5, id=6)
                    ret s: bits[4] = sel(idx, cases=[a,b,c,d], default=def, id=7)
                }"#;
                let f = crate::xls_ir::ir_parser::Parser::new(ir)
                    .parse_fn()
                    .unwrap();
                let mut solver = <$solver_type>::new($solver_config).unwrap();
                let inputs = get_fn_inputs(&mut solver, &f, None);
                let _ = ir_to_smt(&mut solver, &inputs);
            }
        };
    }

    // one_hot_sel: multiple bits
    #[macro_export]
    macro_rules! test_one_hot_sel_multi {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_one_hot_sel_multi,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[4] {
                    sel: bits[3] = literal(value=3, id=1)
                    c0: bits[4] = literal(value=1, id=2)
                    c1: bits[4] = literal(value=2, id=3)
                    c2: bits[4] = literal(value=4, id=4)
                    ret o: bits[4] = one_hot_sel(sel, cases=[c0, c1, c2], id=5)
                }"#,
                r#"fn g() -> bits[4] {
                    ret lit: bits[4] = literal(value=3, id=1)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    // priority_sel: multiple bits -> lowest index wins
    #[macro_export]
    macro_rules! test_priority_sel_multi {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_priority_sel_multi,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[4] {
                    sel: bits[3] = literal(value=5, id=1)
                    c0: bits[4] = literal(value=1, id=2)
                    c1: bits[4] = literal(value=2, id=3)
                    c2: bits[4] = literal(value=4, id=4)
                    def: bits[4] = literal(value=8, id=5)
                    ret o: bits[4] = priority_sel(sel, cases=[c0, c1, c2], default=def, id=6)
                }"#,
                r#"fn g() -> bits[4] {
                    ret lit: bits[4] = literal(value=1, id=1)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }

    // priority_sel: no bits set selects default
    #[macro_export]
    macro_rules! test_priority_sel_default {
        ($solver_type:ty, $solver_config:expr) => {
            crate::test_ir_fn_equiv!(
                test_priority_sel_default,
                $solver_type,
                $solver_config,
                r#"fn f() -> bits[4] {
                    sel: bits[3] = literal(value=0, id=1)
                    c0: bits[4] = literal(value=1, id=2)
                    c1: bits[4] = literal(value=2, id=3)
                    c2: bits[4] = literal(value=4, id=4)
                    def: bits[4] = literal(value=8, id=5)
                    ret o: bits[4] = priority_sel(sel, cases=[c0, c1, c2], default=def, id=6)
                }"#,
                r#"fn g() -> bits[4] {
                    ret lit: bits[4] = literal(value=8, id=1)
                }"#,
                false,
                EquivResult::Proved
            );
        };
    }
}

macro_rules! test_with_solver {
    ($mod_ident:ident, $solver_type:ty, $solver_config:expr) => {
        #[cfg(test)]
        mod $mod_ident {
            use super::*;

            crate::test_ir_value_bits!($solver_type, $solver_config);
            crate::test_ir_bits!($solver_type, $solver_config);
            crate::test_ir_value_array!($solver_type, $solver_config);
            crate::test_ir_array!($solver_type, $solver_config);
            crate::test_ir_value_tuple!($solver_type, $solver_config);
            crate::test_ir_tuple!($solver_type, $solver_config);
            crate::test_ir_value_token!($solver_type, $solver_config);
            crate::test_all_binops!($solver_type, $solver_config);
            crate::test_all_unops!($solver_type, $solver_config);
            crate::test_ir_tuple_index!($solver_type, $solver_config);
            crate::test_ir_tuple_index_literal!($solver_type, $solver_config);
            crate::test_ir_tuple_reverse!($solver_type, $solver_config);
            crate::test_ir_tuple_flattened!($solver_type, $solver_config);
            crate::test_tuple_literal_vs_constructed!($solver_type, $solver_config);
            crate::test_tuple_index_on_literal!($solver_type, $solver_config);
            crate::test_ir_array_index!($solver_type, $solver_config);
            crate::test_ir_array_index_multi_level!($solver_type, $solver_config);
            crate::test_ir_array_index_deep_multi_level!($solver_type, $solver_config);
            crate::test_array_update_inbound_value!($solver_type, $solver_config);
            crate::test_array_update_nested!($solver_type, $solver_config);
            crate::test_array_update_deep_nested!($solver_type, $solver_config);
            crate::test_prove_fn_equiv!($solver_type, $solver_config);
            crate::test_prove_fn_inequiv!($solver_type, $solver_config);
            crate::test_extend!(test_extend_zero, $solver_type, $solver_config, 16, false);
            crate::test_extend!(test_extend_sign, $solver_type, $solver_config, 16, true);
            crate::test_dynamic_bit_slice!($solver_type, $solver_config);
            crate::test_bit_slice!($solver_type, $solver_config);
            crate::test_bit_slice_update_zero!($solver_type, $solver_config);
            crate::test_bit_slice_update_middle!($solver_type, $solver_config);
            crate::test_bit_slice_update_end!($solver_type, $solver_config);
            crate::test_bit_slice_update_beyond_end!($solver_type, $solver_config);
            crate::test_bit_slice_update_wider_update_value!($solver_type, $solver_config);
            crate::test_bit_slice_update_large_update_value!($solver_type, $solver_config);
            crate::test_fuzz_dynamic_bit_slice_shrink_panics!($solver_type, $solver_config);
            crate::test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob!(
                $solver_type,
                $solver_config
            );
            crate::test_all_nary!($solver_type, $solver_config);
            crate::test_ir_decode_base!(test_ir_decode_0, $solver_type, $solver_config, 8, 8);
            crate::test_ir_decode_base!(test_ir_decode_1, $solver_type, $solver_config, 8, 16);
            crate::test_ir_encode_base!(test_ir_encode_0, $solver_type, $solver_config, 8, 3);
            crate::test_ir_encode_base!(test_ir_encode_1, $solver_type, $solver_config, 16, 4);
            crate::test_ir_one_hot_base!(test_ir_one_hot_true, $solver_type, $solver_config, true);
            crate::test_ir_one_hot_base!(
                test_ir_one_hot_false,
                $solver_type,
                $solver_config,
                false
            );
            crate::test_sel_basic!($solver_type, $solver_config);
            crate::test_sel_default!($solver_type, $solver_config);
            crate::test_sel_missing_default_panics!($solver_type, $solver_config);
            crate::test_sel_unexpected_default_panics!($solver_type, $solver_config);
            crate::test_one_hot_sel_multi!($solver_type, $solver_config);
            crate::test_priority_sel_multi!($solver_type, $solver_config);
            crate::test_priority_sel_default!($solver_type, $solver_config);
        }
    };
}

#[cfg(feature = "with-bitwuzla-binary-test")]
test_with_solver!(
    bitwuzla_tests,
    crate::equiv::easy_smt_backend::EasySmtSolver,
    &crate::equiv::easy_smt_backend::EasySmtConfig::bitwuzla()
);

#[cfg(feature = "with-boolector-binary-test")]
test_with_solver!(
    boolector_tests,
    crate::equiv::easy_smt_backend::EasySmtSolver,
    &crate::equiv::easy_smt_backend::EasySmtConfig::boolector()
);

#[cfg(feature = "with-z3-binary-test")]
test_with_solver!(
    z3_tests,
    crate::equiv::easy_smt_backend::EasySmtSolver,
    &crate::equiv::easy_smt_backend::EasySmtConfig::z3()
);

#[cfg(feature = "with-bitwuzla-built")]
test_with_solver!(
    bitwuzla_built_tests,
    crate::equiv::bitwuzla_backend::Bitwuzla,
    &crate::equiv::bitwuzla_backend::BitwuzlaOptions::new()
);
