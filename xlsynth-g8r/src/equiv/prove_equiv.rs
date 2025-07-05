use std::collections::HashMap;

use xlsynth::IrValue;

use crate::{
    equiv::solver_interface::{BitVec, Response, Solver},
    xls_ir::{
        ir::{self, NaryOp, NodePayload, NodeRef, Unop},
        ir_utils::get_topological,
    },
};

#[derive(Clone)]
pub struct IRTypedBitVec<'a, R> {
    pub ir_type: &'a ir::Type,
    pub bitvec: BitVec<R>,
}

impl<'a, R: std::fmt::Debug> std::fmt::Debug for IRTypedBitVec<'a, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IRTypedBitVec {{ ir_type: {:?}, bitvec: {:?} }}",
            self.ir_type, self.bitvec
        )
    }
}

pub fn ir_value_to_bv<'a, S: Solver>(
    solver: &mut S,
    ir_value: &IrValue,
    ir_type: &'a ir::Type,
) -> IRTypedBitVec<'a, S::Rep> {
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
        return IRTypedBitVec {
            ir_type,
            bitvec: BitVec::ZeroWidth,
        };
    }
    let mut s = String::from("#b");
    // SMT bit-vector constant expects MSB first.
    for bit in bits_vec.iter().rev() {
        s.push(if *bit { '1' } else { '0' });
    }
    IRTypedBitVec {
        ir_type,
        bitvec: solver.from_raw_str(width, &s),
    }
}

#[derive(Debug, Clone)]
pub struct FnInputs<'a, R> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: HashMap<String, IRTypedBitVec<'a, R>>,
}

pub fn get_fn_inputs<'a, S: Solver>(
    solver: &mut S,
    fn_ref: &'a ir::Fn,
    name_prefix: Option<&str>,
) -> FnInputs<'a, S::Rep> {
    let mut inputs = HashMap::new();
    for p in fn_ref.params.iter() {
        let name = match name_prefix {
            Some(prefix) => format!("__{}__{}", prefix, p.name),
            None => p.name.clone(),
        };
        let bv = solver.declare(&name, p.ty.bit_count() as usize).unwrap();
        inputs.insert(
            p.name.clone(),
            IRTypedBitVec {
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

pub fn align_fn_inputs<'a, S: Solver>(
    solver: &mut S,
    lhs_inputs: &FnInputs<'a, S::Rep>,
    rhs_inputs: &FnInputs<'a, S::Rep>,
    allow_flatten: bool,
) -> AlignedFnInputs<'a, S::Rep> {
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
        |inputs: &FnInputs<'a, S::Rep>| -> HashMap<String, IRTypedBitVec<'a, S::Rep>> {
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
                    IRTypedBitVec {
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
    pub inputs: Vec<IRTypedBitVec<'a, R>>,
    pub output: IRTypedBitVec<'a, R>,
}

pub fn ir_to_smt<'a, S: Solver>(
    solver: &mut S,
    inputs: &'a FnInputs<'a, S::Rep>,
) -> SmtFn<'a, S::Rep> {
    let topo = get_topological(inputs.fn_ref);
    let mut env: HashMap<NodeRef, IRTypedBitVec<'a, S::Rep>> = HashMap::new();
    for nr in topo {
        let node = &inputs.fn_ref.nodes[nr.index];
        fn get_num_indexable_elements<'a, S: Solver>(
            index: &IRTypedBitVec<'a, S::Rep>,
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
        let exp: IRTypedBitVec<'a, S::Rep> = match &node.payload {
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
                IRTypedBitVec {
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
                IRTypedBitVec {
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
                IRTypedBitVec {
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
                IRTypedBitVec {
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
                IRTypedBitVec {
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
                IRTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: solver.extend_to(&arg_bv.bitvec, to_width, signed),
                }
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
            } => {
                // Recursively build an updated array value that reflects
                // `array[indices] = value` semantics.

                fn array_update_recursive<'a, S: Solver>(
                    solver: &mut S,
                    array_val: &IRTypedBitVec<'a, S::Rep>,
                    indices: &[&IRTypedBitVec<'a, S::Rep>],
                    new_value: &IRTypedBitVec<'a, S::Rep>,
                ) -> IRTypedBitVec<'a, S::Rep> {
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
                            &IRTypedBitVec {
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

                    IRTypedBitVec {
                        ir_type: array_val.ir_type,
                        bitvec: concatenated,
                    }
                }

                let base_array = env.get(array).expect("Array BV must be present");
                let new_val = env.get(value).expect("Update value BV must be present");
                let index_bvs: Vec<&IRTypedBitVec<'_, S::Rep>> = indices
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
                    array_val: &IRTypedBitVec<'a, S::Rep>,
                    indices: &[&IRTypedBitVec<'a, S::Rep>],
                ) -> IRTypedBitVec<'a, S::Rep> {
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
                    let children: Vec<IRTypedBitVec<'a, S::Rep>> = (0..elem_cnt as i32)
                        .map(|i| {
                            let high = (i + 1) * elem_bit_width - 1;
                            let low = i * elem_bit_width;
                            let slice = solver.extract(&array_val.bitvec, high, low);
                            array_index_recursive(
                                solver,
                                &IRTypedBitVec {
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

                    IRTypedBitVec {
                        ir_type: elem_ty,
                        bitvec: result,
                    }
                }

                // Gather the index bit-vectors in evaluation order.
                let base_array = env.get(array).expect("Array BV must be present");
                let index_bvs: Vec<&IRTypedBitVec<'_, S::Rep>> = indices
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
                let shifted_ext = if *width > start_bv.bitvec.get_width() {
                    solver.extend_to(&shifted, *width, false)
                } else {
                    shifted
                };
                let extracted = solver.slice(&shifted_ext, 0, *width);
                IRTypedBitVec {
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
                IRTypedBitVec {
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
                IRTypedBitVec {
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
                let elems_bvs: Vec<&BitVec<S::Rep>> = elems
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
                IRTypedBitVec {
                    ir_type: &node.ty,
                    bitvec: bvs,
                }
            }
            NodePayload::Invoke { .. } => {
                // TODO: add support for Invoke
                panic!("Invoke not supported in Boolector conversion");
            }
            NodePayload::Cover { .. } => {
                // Cover statements do not contribute to value computation
                continue;
            }
            _ => panic!("Not implemented for {:?}", node.payload),
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

#[derive(Debug, PartialEq)]
pub enum EquivResult {
    Proved,
    Disproved {
        inputs: Vec<IrValue>,
        outputs: (IrValue, IrValue),
    },
}

fn check_aligned_fn_equiv_internal<'a, S: Solver>(
    solver: &mut S,
    lhs: &SmtFn<'a, S::Rep>,
    rhs: &SmtFn<'a, S::Rep>,
) -> EquivResult {
    let diff = solver.ne(&lhs.output.bitvec, &rhs.output.bitvec);
    solver.assert(&diff).unwrap();

    match solver.check().unwrap() {
        Response::Sat => {
            let mut get_value = |i: &IRTypedBitVec<'a, S::Rep>| -> IrValue {
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
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Rep>| solver
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
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Rep>| solver
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
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Rep>| solver
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |_, a: &BitVec<<$solver_type as Solver>::Rep>| a.clone()
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, _: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
                |solver: &mut $solver_type, inputs: &FnInputs<<$solver_type as Solver>::Rep>| {
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
            crate::test_all_nary!($solver_type, $solver_config);
        }
    };
}

#[cfg(feature = "with-bitwuzla-binary-test")]
test_with_solver!(
    bitwuzla_tests,
    crate::equiv::easy_smt_backend::EasySMTSolver,
    &crate::equiv::easy_smt_backend::EasySMTConfig::bitwuzla()
);

#[cfg(feature = "with-boolector-binary-test")]
test_with_solver!(
    boolector_tests,
    crate::equiv::easy_smt_backend::EasySMTSolver,
    &crate::equiv::easy_smt_backend::EasySMTConfig::boolector()
);

#[cfg(feature = "with-z3-binary-test")]
test_with_solver!(
    z3_tests,
    crate::equiv::easy_smt_backend::EasySMTSolver,
    &crate::equiv::easy_smt_backend::EasySMTConfig::z3()
);
