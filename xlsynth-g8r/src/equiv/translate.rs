// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use xlsynth::IrValue;

use crate::equiv::{
    solver_interface::{BitVec, Solver},
    types::{Assertion, FnInputs, IrFn, IrTypedBitVec, SmtFn, UfRegistry},
};
use xlsynth_pir::{
    ir::{self, NaryOp, NodePayload, NodeRef, Unop},
    ir_utils::get_topological,
};

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

#[inline]
const fn min_bits_u128(n: u128) -> usize {
    (u128::BITS - n.leading_zeros()) as usize
}

pub fn get_fn_inputs<'a, S: Solver>(
    solver: &mut S,
    ir_fn: &'a IrFn<'a>,
    name_prefix: Option<&str>,
) -> FnInputs<'a, S::Term> {
    let mut params_iter = ir_fn.fn_ref.params.iter();
    let mut inputs = HashMap::new();
    let prefix_name = |name: &str| match name_prefix {
        Some(prefix) => format!("_{}__{}", prefix, name),
        None => name.to_string(),
    };
    if ir_fn.fixed_implicit_activation {
        let itok = params_iter.next().unwrap();
        assert_eq!(itok.ty, ir::Type::Token);
        inputs.insert(
            itok.name.clone(),
            IrTypedBitVec {
                ir_type: &itok.ty,
                bitvec: solver.zero_width(),
            },
        );
        let iact = params_iter.next().unwrap();
        assert_eq!(iact.ty, ir::Type::Bits(1));
        inputs.insert(
            prefix_name(&iact.name),
            IrTypedBitVec {
                ir_type: &iact.ty,
                bitvec: solver.one(1),
            },
        );
    }
    for p in params_iter {
        let name = prefix_name(&p.name);
        let bv = solver
            .declare_fresh(&name, p.ty.bit_count() as usize)
            .unwrap();
        inputs.insert(
            p.name.clone(),
            IrTypedBitVec {
                ir_type: &p.ty,
                bitvec: bv,
            },
        );
    }
    FnInputs { ir_fn, inputs }
}

pub fn ir_to_smt<'a, S: Solver>(
    solver: &mut S,
    inputs: &'a FnInputs<'a, S::Term>,
    uf_map: &HashMap<String, String>,
    uf_registry: &UfRegistry<S>,
) -> SmtFn<'a, S::Term> {
    let topo = get_topological(inputs.ir_fn.fn_ref);
    let mut env: HashMap<NodeRef, IrTypedBitVec<'a, S::Term>> = HashMap::new();
    let mut assertions = Vec::new();
    for nr in topo {
        let node = &inputs.ir_fn.fn_ref.nodes[nr.index];
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
                let p = inputs.params().iter().find(|p| p.id == *pid).unwrap();
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
            NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
                invariant_args,
            } => {
                // Evaluate the initial accumulator value.
                let mut acc = env
                    .get(init)
                    .expect("CountedFor init must be present")
                    .clone();

                // Resolve the body function via shared helper.
                let callee = inputs.get_fn(body);

                // Body params must be: (i, loop_carry, [invariant_args...])
                assert_eq!(
                    callee.params.len(),
                    2 + invariant_args.len(),
                    "CountedFor body expects 2 + |invariant_args| params; have {} but invariant_args has {}",
                    callee.params.len(),
                    invariant_args.len()
                );

                let ind_ty = &callee.params[0].ty;
                let ind_width = ind_ty.bit_count();
                // TODO: Make an independent verify_ir function that verifies a package to have
                // all these invariants hold.
                //
                // Statically ensure the induction variable cannot overflow given
                // trip_count/stride. Minimal bits needed is ceil(log2(max_i +
                // 1)), where max_i = (trip_count-1)*stride.
                let max_i: u128 = (*trip_count as u128)
                    .saturating_sub(1)
                    .saturating_mul(*stride as u128);
                let min_i_bits: usize = min_bits_u128(max_i);
                assert!(
                    matches!(ind_ty, ir::Type::Bits(w) if *w >= min_i_bits),
                    "CountedFor induction variable too narrow: have {:?}, need >= {} for trip_count={} stride={}",
                    ind_ty,
                    min_i_bits,
                    trip_count,
                    stride
                );

                // Prepare callee wrapper and induction-constant builder.
                let callee_ir_fn = IrFn {
                    fn_ref: callee,
                    pkg_ref: inputs.ir_fn.pkg_ref,
                    fixed_implicit_activation: false,
                };
                let mk_ind_var_bv = |solver: &mut S, k: usize| -> BitVec<S::Term> {
                    let val = (k as u128) * (*stride as u128);
                    solver.numerical_u128(ind_width, val)
                };

                // Pre-fetch invariant arg BVs once; they do not change across iterations.
                let invariant_bvs: Vec<IrTypedBitVec<'_, S::Term>> = invariant_args
                    .iter()
                    .map(|nr| {
                        env.get(nr)
                            .expect("Invariant arg BV must be present")
                            .clone()
                    })
                    .collect();

                for k in 0..*trip_count {
                    // Build the input vector in callee param order.
                    let mut actual_bvs: Vec<BitVec<S::Term>> =
                        Vec::with_capacity(callee.params.len());
                    actual_bvs.push(mk_ind_var_bv(solver, k));
                    actual_bvs.push(acc.bitvec.clone());
                    for iv in &invariant_bvs {
                        actual_bvs.push(iv.bitvec.clone());
                    }

                    // Map by param name with types from callee signature.
                    let callee_input_map =
                        HashMap::from_iter(callee.params.iter().zip(actual_bvs.into_iter()).map(
                            |(p, bv)| {
                                (
                                    p.name.clone(),
                                    IrTypedBitVec {
                                        ir_type: &p.ty,
                                        bitvec: bv,
                                    },
                                )
                            },
                        ));

                    let callee_inputs = FnInputs {
                        ir_fn: &callee_ir_fn,
                        inputs: callee_input_map,
                    };
                    let callee_smt = ir_to_smt(solver, &callee_inputs, uf_map, uf_registry);
                    acc = IrTypedBitVec {
                        ir_type: &node.ty,
                        bitvec: callee_smt.output.bitvec,
                    };
                }

                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: acc.bitvec,
                }
            }
            NodePayload::TupleIndex { tuple, index } => {
                let tuple_bv = env.get(tuple).expect("Tuple operand must be present");
                let tuple_ty = inputs.ir_fn.fn_ref.get_node_ty(*tuple);
                let slice = tuple_ty
                    .tuple_get_flat_bit_slice_for_index(*index)
                    .expect("TupleIndex: not a tuple type");
                let width = slice.limit - slice.start;
                if width == 0 {
                    IrTypedBitVec {
                        ir_type: &node.ty,
                        bitvec: BitVec::ZeroWidth,
                    }
                } else {
                    let high = (slice.limit - 1) as i32;
                    let low = slice.start as i32;
                    IrTypedBitVec {
                        ir_type: &node.ty,
                        bitvec: solver.extract(&tuple_bv.bitvec, high, low),
                    }
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
            NodePayload::ArraySlice {
                array,
                start,
                width,
            } => {
                // Shift-based implementation with padding by last element.
                // Flattened array layout is chunked by element width with index 0 at LSB.
                let base_array = env.get(array).expect("Array BV must be present");
                let start_bv = env.get(start).expect("Start BV must be present");
                let (elem_ty, elem_cnt) = match base_array.ir_type {
                    ir::Type::Array(arr) => (&arr.element_type, arr.element_count),
                    _ => panic!(
                        "ArraySlice: expected array type, found {:?}",
                        base_array.ir_type
                    ),
                };
                let e_bits = elem_ty.bit_count() as i32;

                // Clamp start to the last valid element index to model XLS OOB semantics.
                // This ensures that for any start >= elem_cnt - 1 the slice replicates the
                // last element, matching the piecewise definition (and avoids relying on
                // oversized padding when the start width allows larger values).
                let start_width = start_bv.bitvec.get_width();
                let last_idx_const =
                    solver.numerical(start_width, (elem_cnt.saturating_sub(1)) as u64);
                let start_le_last = solver.ule(&start_bv.bitvec, &last_idx_const);
                let clamped_start = solver.ite(&start_le_last, &start_bv.bitvec, &last_idx_const);

                // 1) Build a padding prefix of (width-1) copies of the last element.
                let last_idx = (elem_cnt as i32) - 1;
                let last_high = (last_idx + 1) * e_bits - 1;
                let last_low = last_idx * e_bits;
                let last_elem = solver.extract(&base_array.bitvec, last_high, last_low);
                let mut pad = BitVec::ZeroWidth;
                if *width > 0 {
                    for _ in 0..(*width - 1) {
                        pad = solver.concat(&last_elem, &pad);
                    }
                }

                // 2) Concatenate pad || base_array to form the extended sequence.
                let extended = solver.concat(&pad, &base_array.bitvec);

                // 3) Compute start_scaled = start * e_bits.
                let e_bits_u = e_bits as u128;
                let extra = min_bits_u128(e_bits_u.saturating_sub(1));
                let start_scaled_w = start_width + extra;
                let start_ext = solver.extend_to(&clamped_start, start_scaled_w, false);
                let e_const = solver.numerical(start_scaled_w, e_bits_u as u64);
                let start_scaled =
                    solver.xls_arbitrary_width_umul(&start_ext, &e_const, start_scaled_w);

                // 4) Shift right by start_scaled and slice low (width * e_bits) bits.
                let shifted = solver.xls_shrl(&extended, &start_scaled);
                let out_width_bits = (*width as usize) * (e_bits as usize);
                let result_bv = solver.slice(&shifted, 0, out_width_bits);

                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: result_bv,
                }
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
            NodePayload::Assert {
                activate,
                message,
                label,
                ..
            } => {
                let active = env.get(activate).expect("Assert activate must be present");
                assertions.push(Assertion {
                    active: active.bitvec.clone(),
                    message: &message,
                    label: &label,
                });
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: BitVec::ZeroWidth,
                }
            }
            NodePayload::Trace { .. } => {
                // Trace has no effect on value computation
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: BitVec::ZeroWidth,
                }
            }
            NodePayload::AfterAll(_) => {
                // AfterAll is a no-op for Boolector; do not insert a BV (like Nil)
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: BitVec::ZeroWidth,
                }
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
            NodePayload::Invoke { to_apply, operands } => {
                let callee = inputs.get_fn(to_apply);

                // Build callee inputs by mapping current env operands to callee params.
                assert_eq!(
                    callee.params.len(),
                    operands.len(),
                    "Invoke operand count does not match callee params"
                );

                let mut callee_input_map: HashMap<String, IrTypedBitVec<'a, S::Term>> =
                    HashMap::new();
                for (param, arg_ref) in callee.params.iter().zip(operands.iter()) {
                    let arg_bv = env
                        .get(arg_ref)
                        .unwrap_or_else(|| panic!("Invoke arg BV missing for {}", param.name));
                    callee_input_map.insert(
                        param.name.clone(),
                        IrTypedBitVec {
                            ir_type: &param.ty,
                            bitvec: arg_bv.bitvec.clone(),
                        },
                    );
                }

                // Kind of hacky as this is just determining whether it is implicit token
                // calling convention based on the name of the function. But
                // this seems to be the only way to determine it from the IR.
                let has_itok_in_name = callee.name.starts_with("__itok");
                let has_itok_calling_convention_in_param = callee.params.len() >= 2
                    && matches!(callee.params[0].ty, ir::Type::Token)
                    && matches!(callee.params[1].ty, ir::Type::Bits(1));
                let has_itok_calling_convention_in_ret = match &callee.ret_ty {
                    ir::Type::Tuple(types) => {
                        types.len() >= 2 && matches!(*types[0], ir::Type::Token)
                    }
                    _ => false,
                };
                let has_itok = has_itok_in_name
                    || has_itok_calling_convention_in_param
                    || has_itok_calling_convention_in_ret;
                if has_itok_in_name && !has_itok {
                    log::warn!(
                        "Warning: function {} has __itok in name but does not conform to implicit token calling convention",
                        callee.name
                    );
                }
                let no_itok_name = if has_itok {
                    callee.name[6..].to_string()
                } else {
                    callee.name.clone()
                };
                // If this callee is configured to be treated as an uninterpreted function,
                // apply the UF instead of inlining the body.
                if let Some(uf_sym) = uf_map.get(&no_itok_name) {
                    let uf = uf_registry
                        .ufs
                        .get(uf_sym)
                        .unwrap_or_else(|| panic!("UF symbol '{}' not declared", uf_sym));

                    if has_itok {
                        log::warn!(
                            "Warning: calling uf {} for {}, any side effects are ignored.",
                            uf_sym,
                            callee.name
                        );
                        assert!(
                            callee.params.len() >= 2,
                            "Implicit token calling convention requires at least 2 params"
                        );
                        let args: Vec<&BitVec<S::Term>> = callee
                            .params
                            .iter()
                            .skip(2)
                            .map(|p| &callee_input_map.get(&p.name).unwrap().bitvec)
                            .collect();
                        let app = solver.apply_uf(uf, &args);
                        // It is okay to use this result as token has zero width
                        IrTypedBitVec {
                            ir_type: &node.ty,
                            bitvec: app,
                        }
                    } else {
                        // Build argument vector in callee param order.
                        let args: Vec<&BitVec<S::Term>> = callee
                            .params
                            .iter()
                            .map(|p| &callee_input_map.get(&p.name).unwrap().bitvec)
                            .collect();
                        let app = solver.apply_uf(uf, &args);
                        IrTypedBitVec {
                            ir_type: &node.ty,
                            bitvec: app,
                        }
                    }
                } else {
                    // Otherwise inline the callee recursively.
                    let callee_ir_fn = IrFn {
                        fn_ref: callee,
                        pkg_ref: inputs.ir_fn.pkg_ref,
                        fixed_implicit_activation: false,
                    };
                    let callee_inputs = FnInputs {
                        ir_fn: &callee_ir_fn,
                        inputs: callee_input_map,
                    };
                    let callee_smt = ir_to_smt(solver, &callee_inputs, uf_map, uf_registry);
                    IrTypedBitVec {
                        ir_type: &node.ty,
                        bitvec: callee_smt.output.bitvec,
                    }
                }
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
                IrTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: BitVec::ZeroWidth,
                }
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
    let ret = inputs.ir_fn.fn_ref.ret_node_ref.unwrap();
    // Collect inputs in the same order as they are declared in the IR function
    let mut ordered_inputs: Vec<IrTypedBitVec<'a, S::Term>> = Vec::new();
    for param in inputs.params() {
        if let Some(bv) = inputs.inputs.get(&param.name) {
            ordered_inputs.push(bv.clone());
        } else {
            panic!("Param {} not found in inputs map", param.name);
        }
    }

    SmtFn {
        fn_ref: inputs.ir_fn.fn_ref,
        inputs: ordered_inputs,
        output: env.remove(&ret).unwrap(),
        assertions,
    }
}
