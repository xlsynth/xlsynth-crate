// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use xlsynth::IrValue;

use crate::equiv::solver_interface::Uf;
use crate::{
    equiv::solver_interface::{BitVec, Response, Solver},
    xls_ir::{
        ir::{self, NaryOp, NodePayload, NodeRef, Unop},
        ir_utils::get_topological,
    },
};

#[inline]
const fn min_bits_u128(n: u128) -> usize {
    (u128::BITS - n.leading_zeros()) as usize
}

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
pub struct IrFn<'a> {
    pub fn_ref: &'a ir::Fn,
    // This is allowed to be None for IRs without invoke.
    pub pkg_ref: Option<&'a ir::Package>,
    pub fixed_implicit_activation: bool,
}

impl<'a> IrFn<'a> {
    pub fn new(fn_ref: &'a ir::Fn, pkg_ref: Option<&'a ir::Package>) -> Self {
        Self {
            fn_ref,
            pkg_ref: pkg_ref,
            fixed_implicit_activation: false,
        }
    }

    pub fn name(&self) -> &str {
        &self.fn_ref.name
    }

    pub fn params(&self) -> &'a [ir::Param] {
        &self.fn_ref.params
    }
}

#[derive(Debug, Clone)]
pub struct FnInputs<'a, R> {
    pub ir_fn: &'a IrFn<'a>,
    pub inputs: HashMap<String, IrTypedBitVec<'a, R>>,
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

impl<'a, R> FnInputs<'a, R> {
    pub fn total_width(&self) -> usize {
        self.inputs.values().map(|b| b.bitvec.get_width()).sum()
    }

    pub fn total_free_width(&self) -> usize {
        if self.fixed_implicit_activation() {
            self.total_width() - 1
        } else {
            self.total_width()
        }
    }

    pub fn fixed_implicit_activation(&self) -> bool {
        self.ir_fn.fixed_implicit_activation
    }

    pub fn params(&self) -> &'a [ir::Param] {
        self.ir_fn.params()
    }

    pub fn params_len(&self) -> usize {
        self.params().len()
    }

    pub fn free_params_len(&self) -> usize {
        if self.fixed_implicit_activation() {
            self.params_len() - 2
        } else {
            self.params_len()
        }
    }

    pub fn free_params(&self) -> &'a [ir::Param] {
        if self.fixed_implicit_activation() {
            &self.params()[2..]
        } else {
            self.params()
        }
    }

    pub fn name(&self) -> &str {
        self.ir_fn.name()
    }

    pub fn get_fn(&self, name: &str) -> &'a ir::Fn {
        let pkg = self
            .ir_fn
            .pkg_ref
            .expect("fn lookup requires package context");
        pkg.get_fn(name)
            .unwrap_or_else(|| panic!("Function '{}' not found in package", name))
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
    let lhs_inputs_total_width = lhs_inputs.total_free_width();
    let rhs_inputs_total_width = rhs_inputs.total_free_width();
    assert_eq!(
        lhs_inputs_total_width, rhs_inputs_total_width,
        "LHS and RHS must have the same number of bits"
    );
    if !allow_flatten {
        assert_eq!(
            lhs_inputs.free_params_len(),
            rhs_inputs.free_params_len(),
            "LHS and RHS must have the same number of inputs"
        );
        for (l, r) in lhs_inputs
            .free_params()
            .iter()
            .zip(rhs_inputs.free_params().iter())
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
        lhs_inputs.name(),
        rhs_inputs.name()
    );
    let flattened = solver
        .declare_fresh(&params_name, lhs_inputs_total_width)
        .unwrap();
    // Split into individual param symbols
    let mut split_map =
        |inputs: &FnInputs<'a, S::Term>| -> HashMap<String, IrTypedBitVec<'a, S::Term>> {
            let mut m = HashMap::new();
            let mut params_iter = inputs.params().iter();

            if inputs.fixed_implicit_activation() {
                let itok = params_iter.next().unwrap();
                assert_eq!(itok.ty, ir::Type::Token);
                m.insert(
                    itok.name.clone(),
                    IrTypedBitVec {
                        ir_type: &itok.ty,
                        bitvec: solver.zero_width(),
                    },
                );
                let iact = params_iter.next().unwrap();
                assert_eq!(iact.ty, ir::Type::Bits(1));
                m.insert(
                    iact.name.clone(),
                    IrTypedBitVec {
                        ir_type: &iact.ty,
                        bitvec: solver.one(1),
                    },
                );
            }

            let mut offset = 0;
            for n in params_iter {
                let existing_bitvec = inputs.inputs.get(&n.name).unwrap();
                let new_bitvec = {
                    let w = existing_bitvec.bitvec.get_width();
                    let h = offset as i32 + (w as i32) - 1;
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
            ir_fn: lhs_inputs.ir_fn,
            inputs: lhs_params,
        },
        rhs: FnInputs {
            ir_fn: rhs_inputs.ir_fn,
            inputs: rhs_params,
        },
        flattened,
    }
}

pub struct Assertion<'a, R> {
    pub active: BitVec<R>,
    pub message: &'a str,
    pub label: &'a str,
}

pub struct SmtFn<'a, R> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: Vec<IrTypedBitVec<'a, R>>,
    pub output: IrTypedBitVec<'a, R>,
    pub assertions: Vec<Assertion<'a, R>>,
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

#[derive(Debug, PartialEq, Clone)]
pub struct AssertionViolation {
    pub message: String,
    pub label: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnInput {
    pub name: String,
    pub value: IrValue,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnOutput {
    pub value: IrValue,
    pub assertion_violation: Option<AssertionViolation>,
}

impl std::fmt::Display for FnOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(violation) = &self.assertion_violation {
            write!(
                f,
                "Value: {:?}, Assertion violation: {} (label: {})",
                self.value, violation.message, violation.label
            )
        } else {
            write!(f, "Value: {:?}", self.value)
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum EquivResult {
    Proved,
    Disproved {
        lhs_inputs: Vec<FnInput>,
        rhs_inputs: Vec<FnInput>,
        lhs_output: FnOutput,
        rhs_output: FnOutput,
    },
}

/// Semantics for handling `assert` statements when checking functional
/// equivalence.
///
/// Shorthand used in the formulas below:
/// • `r_l` – result of the **l**eft function
/// • `r_r` – result of the **r**ight function
/// • `s_l` – "success" flag of the left (`true` iff no assertion failed)
/// • `s_r` – "success" flag of the right (`true` iff no assertion failed)
///
/// For every variant we list
///  1. **Success condition** – when the equivalence checker should consider the
///     two functions *equivalent*.
///  2. **Failure condition** – negation of the success condition; if *any*
///     model satisfies this predicate, the checker must report a
///     counter-example.
#[derive(Debug, PartialEq, Clone, Copy, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssertionSemantics {
    /// Ignore all assertions.
    ///
    /// 1. Success: `r_l == r_r`
    /// 2. Failure: `r_l != r_r`
    Ignore,

    /// Both sides must succeed and produce the same result – they can **never**
    /// fail.
    ///
    /// 1. Success: `s_l ∧ s_r ∧ (r_l == r_r)`
    /// 2. Failure: `¬s_l ∨ ¬s_r ∨ (r_l != r_r)`
    Never,

    /// The two sides must fail in exactly the same way **or** both succeed with
    /// equal results.
    ///
    /// 1. Success: `(¬s_l ∧ ¬s_r) ∨ (s_l ∧ s_r ∧ (r_l == r_r))`
    /// 2. Failure: `(s_l ⊕ s_r) ∨ (s_l ∧ s_r ∧ (r_l != r_r))`
    Same,

    /// We *assume* both sides do not fail. In other words, we only check that
    /// if they do succeed, their results must be equal.
    ///
    /// 1. Success: `¬(s_l ∧ s_r) ∨ (r_l == r_r)`  (equivalently, `(s_l ∧ s_r) →
    ///    r_l == r_r`)
    /// 2. Failure: `s_l ∧ s_r ∧ (r_l != r_r)`
    Assume,

    /// If the left succeeds, the right must also succeed and match the
    /// result; if the left fails, the right is unconstrained.
    ///
    /// 1. Success: `¬s_l ∨ (s_r ∧ (r_l == r_r))`
    /// 2. Failure: `s_l ∧ (¬s_r ∨ (r_l != r_r))`
    Implies,
}

impl fmt::Display for AssertionSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AssertionSemantics::Ignore => "ignore",
            AssertionSemantics::Never => "never",
            AssertionSemantics::Same => "same",
            AssertionSemantics::Assume => "assume",
            AssertionSemantics::Implies => "implies",
        };
        write!(f, "{}", s)
    }
}

impl std::str::FromStr for AssertionSemantics {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ignore" => Ok(AssertionSemantics::Ignore),
            "never" => Ok(AssertionSemantics::Never),
            "same" => Ok(AssertionSemantics::Same),
            "assume" => Ok(AssertionSemantics::Assume),
            "implies" => Ok(AssertionSemantics::Implies),
            _ => Err(format!("Invalid assertion semantics: {}", s)),
        }
    }
}

fn check_aligned_fn_equiv_internal<'a, S: Solver>(
    solver: &mut S,
    lhs: &SmtFn<'a, S::Term>,
    rhs: &SmtFn<'a, S::Term>,
    assertion_semantics: AssertionSemantics,
) -> EquivResult {
    // --------------------------------------------
    // Helper: build a 1-bit "failed" flag for each
    // side indicating whether **any** assertion is
    // violated (active bit == 0).
    // --------------------------------------------

    // Build a flag that is true iff ALL assertions are active (i.e., no violation).
    let mk_success_flag = |solver: &mut S, asserts: &Vec<Assertion<'a, S::Term>>| {
        if asserts.is_empty() {
            // No assertions → always succeed (true)
            solver.numerical(1, 1)
        } else {
            let mut acc: Option<BitVec<S::Term>> = None;
            for a in asserts {
                acc = Some(match acc {
                    None => a.active.clone(),
                    Some(prev) => solver.and(&prev, &a.active),
                });
            }
            acc.expect("acc populated")
        }
    };

    let lhs_pass = mk_success_flag(solver, &lhs.assertions);
    let rhs_pass = mk_success_flag(solver, &rhs.assertions);

    let lhs_failed = solver.not(&lhs_pass);
    let rhs_failed = solver.not(&rhs_pass);

    // diff of outputs
    let outputs_diff = solver.ne(&lhs.output.bitvec, &rhs.output.bitvec);

    // Build the overall condition to assert based on semantics
    let condition = match assertion_semantics {
        AssertionSemantics::Ignore => outputs_diff.clone(),
        AssertionSemantics::Never => {
            let any_failed = solver.or(&lhs_failed, &rhs_failed);
            solver.or(&outputs_diff, &any_failed)
        }
        AssertionSemantics::Same => {
            // fail status differs OR (both pass AND outputs differ)
            let fail_status_diff = solver.ne(&lhs_failed, &rhs_failed);
            let both_pass = solver.and(&lhs_pass, &rhs_pass);
            let diff_when_pass = solver.and(&outputs_diff, &both_pass);
            solver.or(&fail_status_diff, &diff_when_pass)
        }
        AssertionSemantics::Assume => {
            let both_pass = solver.and(&lhs_pass, &rhs_pass);
            solver.and(&both_pass, &outputs_diff)
        }
        AssertionSemantics::Implies => {
            // lhs passes AND (rhs fails OR outputs differ)
            let rhs_fail_or_diff = solver.or(&rhs_failed, &outputs_diff);
            solver.and(&lhs_pass, &rhs_fail_or_diff)
        }
    };

    solver.assert(&condition).unwrap();

    match solver.check().unwrap() {
        Response::Sat => {
            // Helper to fetch first violated assertion (if any)
            let get_assertion = |solver: &mut S,
                                 asserts: &Vec<Assertion<'a, S::Term>>|
             -> Option<(String, String)> {
                for a in asserts {
                    let val = solver.get_value(&a.active, &ir::Type::Bits(1)).unwrap();
                    let bits = val.to_bits().unwrap();
                    let ok = bits.get_bit(0).unwrap();
                    if !ok {
                        return Some((a.message.to_string(), a.label.to_string()));
                    }
                }
                None
            };

            let lhs_violation = { get_assertion(solver, &lhs.assertions) };
            let rhs_violation = { get_assertion(solver, &rhs.assertions) };

            // Now we can safely create a helper closure that borrows `solver` again
            let get_value = |solver: &mut S, i: &IrTypedBitVec<'a, S::Term>| -> IrValue {
                solver.get_value(&i.bitvec, &i.ir_type).unwrap()
            };

            // Helper to produce FnOutput given optional violation info.
            let get_output =
                |solver: &mut S,
                 vio: Option<(String, String)>,
                 tbv: &IrTypedBitVec<'a, S::Term>| FnOutput {
                    value: get_value(solver, tbv),
                    assertion_violation: vio.map(|(msg, lbl)| AssertionViolation {
                        message: msg,
                        label: lbl,
                    }),
                };

            let build_inputs = |solver: &mut S, smt_fn: &SmtFn<'a, S::Term>| {
                smt_fn
                    .fn_ref
                    .params
                    .iter()
                    .zip(smt_fn.inputs.iter())
                    .map(|(p, i)| FnInput {
                        name: p.name.clone(),
                        value: get_value(solver, i),
                    })
                    .collect()
            };

            let lhs_inputs = build_inputs(solver, lhs);
            let rhs_inputs = build_inputs(solver, rhs);

            let lhs_output = get_output(solver, lhs_violation, &lhs.output);
            let rhs_output = get_output(solver, rhs_violation, &rhs.output);

            EquivResult::Disproved {
                lhs_inputs,
                rhs_inputs,
                lhs_output,
                rhs_output,
            }
        }
        Response::Unsat => EquivResult::Proved,
        Response::Unknown => panic!("Solver returned unknown result"),
    }
}

// Map param name -> allowed IrValues for domain constraints.
pub type ParamDomains = HashMap<String, Vec<IrValue>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UfSignature {
    pub arg_widths: Vec<usize>,
    pub ret_width: usize,
}

pub struct UfRegistry<S: Solver> {
    pub ufs: HashMap<String, Uf<S::Term>>,
}

impl<S: Solver> UfRegistry<S> {
    pub fn from_uf_signatures(
        solver: &mut S,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> Self {
        let mut ufs = HashMap::new();
        for (name, signature) in uf_signatures {
            let uf = solver
                .declare_fresh_uf(&name, &signature.arg_widths, signature.ret_width)
                .unwrap();
            ufs.insert(name.clone(), uf);
        }
        Self { ufs }
    }
}

/// Prove equivalence like `prove_ir_fn_equiv` but constraining parameters that
/// are enums to lie within their defined value sets.
pub fn prove_ir_fn_equiv_with_domains<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &IrFn<'a>,
    rhs: &IrFn<'a>,
    assertion_semantics: AssertionSemantics,
    allow_flatten: bool,
    lhs_domains: Option<&ParamDomains>,
    rhs_domains: Option<&ParamDomains>,
    lhs_uf_map: &HashMap<String, String>,
    rhs_uf_map: &HashMap<String, String>,
    uf_signatures: &HashMap<String, UfSignature>,
) -> EquivResult {
    let mut solver = S::new(solver_config).unwrap();
    let fn_inputs_lhs = get_fn_inputs(&mut solver, lhs, Some("lhs"));
    let fn_inputs_rhs = get_fn_inputs(&mut solver, rhs, Some("rhs"));

    let mut assert_domains = |inputs: &FnInputs<'_, S::Term>, domains: Option<&ParamDomains>| {
        if let Some(dom) = domains {
            for p in inputs.params().iter() {
                if let Some(allowed) = dom.get(&p.name) {
                    if let Some(sym) = inputs.inputs.get(&p.name) {
                        let mut or_chain: Option<BitVec<S::Term>> = None;
                        for v in allowed {
                            let bv = ir_value_to_bv(&mut solver, v, &p.ty).bitvec;
                            let eq = solver.eq(&sym.bitvec, &bv);
                            or_chain = Some(match or_chain {
                                None => eq,
                                Some(prev) => solver.or(&prev, &eq),
                            });
                        }
                        if let Some(expr) = or_chain {
                            solver.assert(&expr).unwrap();
                        }
                    }
                }
            }
        }
    };

    assert_domains(&fn_inputs_lhs, lhs_domains);
    assert_domains(&fn_inputs_rhs, rhs_domains);

    let uf_registry = UfRegistry::from_uf_signatures(&mut solver, uf_signatures);

    let aligned = align_fn_inputs(&mut solver, &fn_inputs_lhs, &fn_inputs_rhs, allow_flatten);
    let smt_lhs = ir_to_smt(&mut solver, &aligned.lhs, &lhs_uf_map, &uf_registry);
    let smt_rhs = ir_to_smt(&mut solver, &aligned.rhs, &rhs_uf_map, &uf_registry);
    check_aligned_fn_equiv_internal(&mut solver, &smt_lhs, &smt_rhs, assertion_semantics)
}

pub fn prove_ir_fn_equiv<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &IrFn<'a>,
    rhs: &IrFn<'a>,
    assertion_semantics: AssertionSemantics,
    allow_flatten: bool,
) -> EquivResult {
    prove_ir_fn_equiv_with_domains::<S>(
        solver_config,
        lhs,
        rhs,
        assertion_semantics,
        allow_flatten,
        None,
        None,
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
    )
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
pub fn prove_ir_fn_equiv_output_bits_parallel<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &IrFn<'a>,
    rhs: &IrFn<'a>,
    assertion_semantics: AssertionSemantics,
    allow_flatten: bool,
) -> EquivResult {
    let width = lhs.fn_ref.ret_ty.bit_count();
    assert_eq!(
        width,
        rhs.fn_ref.ret_ty.bit_count(),
        "Return widths must match"
    );
    if width == 0 {
        // Zero-width values – fall back to the standard equivalence prover because
        // there is no bit to split on.
        return prove_ir_fn_equiv::<S>(solver_config, lhs, rhs, assertion_semantics, allow_flatten);
    };

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

                    let lf = make_bit_fn(&lhs_cl.fn_ref, idx);
                    let rf = make_bit_fn(&rhs_cl.fn_ref, idx);
                    let lf = IrFn {
                        fn_ref: &lf,
                        ..lhs_cl
                    };
                    let rf = IrFn {
                        fn_ref: &rf,
                        ..rhs_cl
                    };
                    let res = prove_ir_fn_equiv::<S>(
                        solver_config,
                        &lf,
                        &rf,
                        assertion_semantics.clone(),
                        allow_flatten,
                    );
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
pub fn prove_ir_fn_equiv_split_input_bit<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &IrFn<'a>,
    rhs: &IrFn<'a>,
    split_input_index: usize,
    split_input_bit_index: usize,
    assertion_semantics: AssertionSemantics,
    allow_flatten: bool,
) -> EquivResult {
    if lhs.fn_ref.params.is_empty() || rhs.fn_ref.params.is_empty() {
        return prove_ir_fn_equiv::<S>(solver_config, lhs, rhs, assertion_semantics, allow_flatten);
    }

    assert_eq!(
        lhs.fn_ref.params.len(),
        rhs.fn_ref.params.len(),
        "Parameter count mismatch"
    );
    assert!(
        split_input_index < lhs.fn_ref.params.len(),
        "split_input_index out of bounds"
    );
    assert!(
        split_input_bit_index < lhs.fn_ref.params[split_input_index].ty.bit_count(),
        "split_input_bit_index out of bounds"
    );

    for bit_val in 0..=1u64 {
        let mut solver = S::new(solver_config).unwrap();
        // Build aligned SMT representations first so we can assert the bit-constraint.
        let fn_inputs_lhs = get_fn_inputs(&mut solver, lhs, Some("lhs"));
        let fn_inputs_rhs = get_fn_inputs(&mut solver, rhs, Some("rhs"));
        let aligned = align_fn_inputs(&mut solver, &fn_inputs_lhs, &fn_inputs_rhs, allow_flatten);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = UfRegistry {
            ufs: HashMap::new(),
        };
        let smt_lhs = ir_to_smt(&mut solver, &aligned.lhs, &empty_map, &empty_registry);
        let smt_rhs = ir_to_smt(&mut solver, &aligned.rhs, &empty_map, &empty_registry);

        // Locate the chosen parameter on the LHS side.
        let param_name = &lhs.fn_ref.params[split_input_index].name;

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

        check_aligned_fn_equiv_internal(&mut solver, &smt_lhs, &smt_rhs, assertion_semantics);
    }

    EquivResult::Proved
}

#[cfg(test)]
pub mod test_utils {

    use std::collections::HashMap;

    use xlsynth::IrValue;

    use crate::{
        equiv::{
            prove_equiv::{
                AssertionSemantics, EquivResult, FnInputs, IrFn, ParamDomains, align_fn_inputs,
                get_fn_inputs, ir_to_smt, ir_value_to_bv, prove_ir_fn_equiv,
                prove_ir_fn_equiv_with_domains,
            },
            solver_interface::{BitVec, Solver, test_utils::assert_solver_eq},
        },
        xls_ir::ir,
    };

    pub fn test_invoke_basic<S: Solver>(solver_config: &S::Config) {
        // Package with a callee that doubles its input and two wrappers:
        // one using invoke, the other inlining the computation.
        let ir_pkg_text = r#"
            package p

            fn g(x: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(x, x, id=1)
            }

            fn call(x: bits[8]) -> bits[8] {
              ret r: bits[8] = invoke(x, to_apply=g, id=1)
            }

            fn inline(x: bits[8]) -> bits[8] {
              ret add.2: bits[8] = add(x, x, id=2)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_fn = pkg.get_fn("call").expect("call not found");
        let inline_fn = pkg.get_fn("inline").expect("inline not found");

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: call_fn,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &IrFn {
                fn_ref: inline_fn,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    /// Uninterpreted-function (UF) handling tests
    ///
    /// Construct a package with two different callees `g` and `h` and two
    /// wrappers that invoke them. Without UF mapping they are not
    /// equivalent; with UF mapping to the same UF symbol and signature they
    /// become equivalent.
    pub fn test_uf_basic_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_uf

            fn g(x: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(x, x, id=1)
            }

            fn h(x: bits[8]) -> bits[8] {
              ret sub.2: bits[8] = sub(x, x, id=2)
            }

            fn call_g(x: bits[8]) -> bits[8] {
              ret r: bits[8] = invoke(x, to_apply=g, id=3)
            }

            fn call_h(x: bits[8]) -> bits[8] {
              ret r: bits[8] = invoke(x, to_apply=h, id=4)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_g = pkg.get_fn("call_g").expect("call_g not found");
        let call_h = pkg.get_fn("call_h").expect("call_h not found");

        // 1) Without UF mapping: should be inequivalent (add(x,x) vs 0)
        let res_no_uf = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &super::IrFn {
                fn_ref: call_g,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &super::IrFn {
                fn_ref: call_h,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            super::AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res_no_uf, super::EquivResult::Disproved { .. }));

        // 2) With UF mapping: map g (LHS) and h (RHS) to the same UF symbol "F".
        let mut lhs_uf_map: HashMap<String, String> = HashMap::new();
        lhs_uf_map.insert("g".to_string(), "F".to_string());
        let mut rhs_uf_map: HashMap<String, String> = HashMap::new();
        rhs_uf_map.insert("h".to_string(), "F".to_string());
        let mut uf_sigs: HashMap<String, super::UfSignature> = HashMap::new();
        uf_sigs.insert(
            "F".to_string(),
            super::UfSignature {
                arg_widths: vec![8],
                ret_width: 8,
            },
        );

        let res_uf = super::prove_ir_fn_equiv_with_domains::<S>(
            solver_config,
            &super::IrFn {
                fn_ref: call_g,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &super::IrFn {
                fn_ref: call_h,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            super::AssertionSemantics::Same,
            false,
            None,
            None,
            &lhs_uf_map,
            &rhs_uf_map,
            &uf_sigs,
        );
        assert!(matches!(res_uf, super::EquivResult::Proved));
    }

    /// Nested invoke case: inner callees differ, but both sides map inner to
    /// the same UF.
    pub fn test_uf_nested_invoke_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_uf_nested

            fn inner_g(x: bits[4]) -> bits[4] {
              ret add.1: bits[4] = add(x, x, id=1)
            }

            fn inner_h(x: bits[4]) -> bits[4] {
              ret lit7.2: bits[4] = literal(value=7, id=2)
            }

            fn mid_g(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=inner_g, id=3)
            }

            fn mid_h(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=inner_h, id=4)
            }

            fn top_g(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=mid_g, id=5)
            }

            fn top_h(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=mid_h, id=6)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let top_g = pkg.get_fn("top_g").expect("top_g not found");
        let top_h = pkg.get_fn("top_h").expect("top_h not found");

        // With UF mapping on inner functions, equality should hold at the top.
        let mut lhs_uf_map: HashMap<String, String> = HashMap::new();
        lhs_uf_map.insert("inner_g".to_string(), "F".to_string());
        let mut rhs_uf_map: HashMap<String, String> = HashMap::new();
        rhs_uf_map.insert("inner_h".to_string(), "F".to_string());
        let mut uf_sigs: HashMap<String, super::UfSignature> = HashMap::new();
        uf_sigs.insert(
            "F".to_string(),
            super::UfSignature {
                arg_widths: vec![4],
                ret_width: 4,
            },
        );

        let res = super::prove_ir_fn_equiv_with_domains::<S>(
            solver_config,
            &super::IrFn {
                fn_ref: top_g,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &super::IrFn {
                fn_ref: top_h,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            super::AssertionSemantics::Same,
            false,
            None,
            None,
            &lhs_uf_map,
            &rhs_uf_map,
            &uf_sigs,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    /// UF + implicit-token with two data args to ensure we slice args after the
    /// first two params.
    pub fn test_uf_implicit_token_two_args_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_uf_itok2

            fn __itokg(__token: token, __activate: bits[1], a: bits[4], b: bits[4]) -> (token, bits[8]) {
              s: bits[8] = umul(a, b, id=1)
              ret t: (token, bits[8]) = tuple(__token, s, id=2)
            }

            fn h(a: bits[4], b: bits[4]) -> bits[8] {
              x: bits[4] = xor(a, b, id=3)
              ret z: bits[8] = zero_ext(x, new_bit_count=8, id=4)
            }

            fn call_g(tok: token, act: bits[1], a: bits[4], b: bits[4]) -> (token, bits[8]) {
              ret r: (token, bits[8]) = invoke(tok, act, a, b, to_apply=__itokg, id=6)
            }

            fn call_h(tok: token, act: bits[1], a: bits[4], b: bits[4]) -> bits[8] {
              ret r: bits[8] = invoke(a, b, to_apply=h, id=7)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_g = pkg.get_fn("call_g").expect("call_g not found");
        let call_h = pkg.get_fn("call_h").expect("call_h not found");

        let res_no_uf = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &super::IrFn {
                fn_ref: call_g,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &super::IrFn {
                fn_ref: call_h,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            super::AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res_no_uf, super::EquivResult::Disproved { .. }));

        let mut lhs_uf_map: HashMap<String, String> = HashMap::new();
        lhs_uf_map.insert("g".to_string(), "F".to_string());
        let mut rhs_uf_map: HashMap<String, String> = HashMap::new();
        rhs_uf_map.insert("h".to_string(), "F".to_string());
        // Two 4-bit data args, 8-bit result.
        let mut uf_sigs: HashMap<String, super::UfSignature> = HashMap::new();
        uf_sigs.insert(
            "F".to_string(),
            super::UfSignature {
                arg_widths: vec![4, 4],
                ret_width: 8,
            },
        );

        let res_uf = super::prove_ir_fn_equiv_with_domains::<S>(
            solver_config,
            &super::IrFn {
                fn_ref: call_g,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &super::IrFn {
                fn_ref: call_h,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            super::AssertionSemantics::Same,
            false,
            None,
            None,
            &lhs_uf_map,
            &rhs_uf_map,
            &uf_sigs,
        );
        assert!(matches!(res_uf, super::EquivResult::Proved));
    }

    pub fn test_invoke_two_args<S: Solver>(solver_config: &S::Config) {
        // Callee adds two 4-bit args. Compare invoke wrapper vs inline.
        let ir_pkg_text = r#"
            package p2

            fn g(a: bits[4], b: bits[4]) -> bits[4] {
              ret add.1: bits[4] = add(a, b, id=1)
            }

            fn call(a: bits[4], b: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(a, b, to_apply=g, id=2)
            }

            fn inline(a: bits[4], b: bits[4]) -> bits[4] {
              ret add.3: bits[4] = add(a, b, id=3)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_fn = pkg.get_fn("call").expect("call not found");
        let inline_fn = pkg.get_fn("inline").expect("inline not found");

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: call_fn,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &IrFn {
                fn_ref: inline_fn,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_basic<S: Solver>(solver_config: &S::Config) {
        // Package with a simple body that accumulates the induction variable into acc.
        // Loop it three times; compare to a manually unrolled implementation.
        let ir_pkg_text = r#"
            package p_cf

            fn body(i: bits[8], acc: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(acc, i, id=1)
            }

            fn looped(init: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=3, stride=1, body=body, id=2)
            }

            fn inline(init: bits[8]) -> bits[8] {
              z: bits[8] = literal(value=0, id=3)
              a0: bits[8] = add(init, z, id=4)
              o1: bits[8] = literal(value=1, id=5)
              a1: bits[8] = add(a0, o1, id=6)
              o2: bits[8] = literal(value=2, id=7)
              ret a2: bits[8] = add(a1, o2, id=8)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: looped,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &IrFn {
                fn_ref: inline,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_stride<S: Solver>(solver_config: &S::Config) {
        // Stride=2 for 3 trips: add 0,2,4
        let ir_pkg_text = r#"
            package p_cf2

            fn body(i: bits[8], acc: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(acc, i, id=1)
            }

            fn looped(init: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=3, stride=2, body=body, id=2)
            }

            fn inline(init: bits[8]) -> bits[8] {
              i0: bits[8] = literal(value=0, id=3)
              a0: bits[8] = add(init, i0, id=4)
              i1: bits[8] = literal(value=2, id=5)
              a1: bits[8] = add(a0, i1, id=6)
              i2: bits[8] = literal(value=4, id=7)
              ret a2: bits[8] = add(a1, i2, id=8)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: looped,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &IrFn {
                fn_ref: inline,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_zero_trip<S: Solver>(solver_config: &S::Config) {
        // Zero trips should produce the init unchanged.
        let ir_pkg_text = r#"
            package p_cf3

            fn body(i: bits[8], acc: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(acc, i, id=1)
            }

            fn looped(init: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=0, stride=1, body=body, id=2)
            }

            fn inline(init: bits[8]) -> bits[8] {
              ret id.1: bits[8] = identity(init, id=1)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: looped,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &IrFn {
                fn_ref: inline,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_invariant_args<S: Solver>(solver_config: &S::Config) {
        // Body uses an invariant offset 'k' added to the induction variable each
        // iteration. Compare counted_for vs inline unrolling.
        let ir_pkg_text = r#"
            package p_cf_inv

            fn body(i: bits[8], acc: bits[8], k: bits[8]) -> bits[8] {
              t0: bits[8] = add(i, k, id=1)
              ret a: bits[8] = add(acc, t0, id=2)
            }

            fn looped(init: bits[8], k: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=3, stride=1, body=body, invariant_args=[k], id=3)
            }

            fn inline(init: bits[8], k: bits[8]) -> bits[8] {
              z: bits[8] = literal(value=0, id=4)
              s0: bits[8] = add(z, k, id=5)
              a0: bits[8] = add(init, s0, id=6)
              one: bits[8] = literal(value=1, id=7)
              s1: bits[8] = add(one, k, id=8)
              a1: bits[8] = add(a0, s1, id=9)
              two: bits[8] = literal(value=2, id=10)
              s2: bits[8] = add(two, k, id=11)
              ret a2: bits[8] = add(a1, s2, id=12)
            }
        "#;

        let pkg = crate::xls_ir::ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: looped,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            &IrFn {
                fn_ref: inline,
                pkg_ref: Some(&pkg),
                fixed_implicit_activation: false,
            },
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }
    pub fn align_zero_width_fn_inputs<S: Solver>(solver_config: &S::Config) {
        let ir_text = r#"
            fn lhs(tok: token) -> token {
                ret t: token = param(name=tok, id=1)
            }
        "#;

        let mut parser = crate::xls_ir::ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        // Must not panic.
        let _ = align_fn_inputs(&mut solver, &fn_inputs, &fn_inputs, false);
    }

    pub fn align_non_zero_width_fn_inputs_first_token<S: Solver>(solver_config: &S::Config) {
        let ir_text = r#"
            fn lhs(tok: token, x: bits[4]) -> token {
                ret t: token = param(name=tok, id=1)
            }
        "#;

        let mut parser = crate::xls_ir::ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        // Must not panic.
        let _ = align_fn_inputs(&mut solver, &fn_inputs, &fn_inputs, false);
    }

    pub fn assert_smt_fn_eq<S: Solver>(
        solver_config: &S::Config,
        ir_text: &str,
        expected: impl Fn(&mut S, &FnInputs<S::Term>) -> BitVec<S::Term>,
    ) {
        let mut parser = crate::xls_ir::ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let smt_fn = ir_to_smt(&mut solver, &fn_inputs, &empty_map, &empty_registry);
        let expected_result = expected(&mut solver, &fn_inputs);
        assert_solver_eq(&mut solver, &smt_fn.output.bitvec, &expected_result);
    }

    pub fn assert_ir_value_to_bv_eq<S: Solver>(
        solver_config: &S::Config,
        ir_value: &IrValue,
        ir_type: &ir::Type,
        expected: impl Fn(&mut S) -> BitVec<S::Term>,
    ) {
        let mut solver = S::new(solver_config).unwrap();
        let bv = ir_value_to_bv(&mut solver, &ir_value, &ir_type);
        assert_eq!(bv.ir_type, ir_type);
        let expected_value = expected(&mut solver);
        crate::equiv::solver_interface::test_utils::assert_solver_eq(
            &mut solver,
            &bv.bitvec,
            &expected_value,
        );
    }

    fn assert_ir_fn_equiv_base_with_implicit_token_policy<S: Solver>(
        solver_config: &S::Config,
        lhs_text: &str,
        rhs_text: &str,
        lhs_fixed_implicit_activation: bool,
        rhs_fixed_implicit_activation: bool,
        allow_flatten: bool,
        assertion_semantics: AssertionSemantics,
        expected_proven: bool,
    ) {
        let mut parser = crate::xls_ir::ir_parser::Parser::new(lhs_text);
        let lhs_ir_fn = parser.parse_fn().unwrap();
        let mut parser = crate::xls_ir::ir_parser::Parser::new(rhs_text);
        let rhs_ir_fn = parser.parse_fn().unwrap();
        let actual = prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn {
                fn_ref: &lhs_ir_fn,
                pkg_ref: None,
                fixed_implicit_activation: lhs_fixed_implicit_activation,
            },
            &IrFn {
                fn_ref: &rhs_ir_fn,
                pkg_ref: None,
                fixed_implicit_activation: rhs_fixed_implicit_activation,
            },
            assertion_semantics,
            allow_flatten,
        );
        if expected_proven {
            assert!(matches!(actual, EquivResult::Proved));
        } else {
            assert!(matches!(actual, EquivResult::Disproved { .. }));
        }
    }

    fn assert_ir_fn_equiv_base<S: Solver>(
        solver_config: &S::Config,
        ir_text_1: &str,
        ir_text_2: &str,
        allow_flatten: bool,
        assertion_semantics: AssertionSemantics,
        expected_proven: bool,
    ) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            ir_text_1,
            ir_text_2,
            false,
            false,
            allow_flatten,
            assertion_semantics,
            expected_proven,
        );
    }

    pub fn assert_ir_fn_equiv<S: Solver>(
        solver_config: &S::Config,
        ir_text_1: &str,
        ir_text_2: &str,
        allow_flatten: bool,
    ) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            ir_text_1,
            ir_text_2,
            allow_flatten,
            AssertionSemantics::Same,
            true,
        );
    }

    pub fn assert_ir_fn_inequiv<S: Solver>(
        solver_config: &S::Config,
        ir_text_1: &str,
        ir_text_2: &str,
        allow_flatten: bool,
    ) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            ir_text_1,
            ir_text_2,
            allow_flatten,
            AssertionSemantics::Same,
            false,
        );
    }

    fn test_ir_fn_equiv_to_self<S: Solver>(solver_config: &S::Config, ir_text: &str) {
        assert_ir_fn_equiv::<S>(solver_config, ir_text, ir_text, false);
    }

    pub fn test_ir_value_bits<S: Solver>(solver_config: &S::Config) {
        assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::u32(0x12345678),
            &ir::Type::Bits(32),
            |solver: &mut S| solver.from_raw_str(32, "#x12345678"),
        );
    }

    pub fn test_ir_bits<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> bits[32] {
                ret literal.1: bits[32] = literal(value=0x12345678, id=1)
            }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(32, "#x12345678"),
        );
    }

    pub fn test_ir_value_array<S: Solver>(solver_config: &S::Config) {
        crate::equiv::prove_equiv::test_utils::assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::make_array(&[
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
            &ir::Type::Array(ir::ArrayTypeData {
                element_type: Box::new(ir::Type::Array(ir::ArrayTypeData {
                    element_type: Box::new(ir::Type::Bits(8)),
                    element_count: 2,
                })),
                element_count: 3,
            }),
            |solver: &mut S| solver.from_raw_str(48, "#xbc9a78563412"),
        );
    }

    pub fn test_ir_array<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> bits[8][2][3] {
                ret literal.1: bits[8][2][3] = literal(value=[[0x12, 0x34], [0x56, 0x78], [0x9a, 0xbc]], id=1)
            }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(48, "#xbc9a78563412"),
        );
    }

    pub fn test_ir_value_tuple<S: Solver>(solver_config: &S::Config) {
        assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::make_tuple(&[
                IrValue::make_ubits(8, 0x12).unwrap(),
                IrValue::make_ubits(4, 0x4).unwrap(),
            ]),
            &ir::Type::Tuple(vec![
                Box::new(ir::Type::Bits(8)),
                Box::new(ir::Type::Bits(4)),
            ]),
            |solver: &mut S| solver.from_raw_str(12, "#x124"),
        );
    }

    pub fn test_ir_tuple<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> (bits[8], bits[4]) {
                ret literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
            }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(12, "#x124"),
        );
    }

    pub fn test_ir_value_token<S: Solver>(solver_config: &S::Config) {
        assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::make_ubits(0, 0).unwrap(),
            &ir::Type::Token,
            |_: &mut S| BitVec::ZeroWidth,
        );
    }

    pub fn test_unop_base<S: Solver>(
        solver_config: &S::Config,
        unop_xls_name: &str,
        unop: impl Fn(&mut S, &BitVec<S::Term>) -> BitVec<S::Term>,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &(format!(
                r#"fn f(x: bits[8]) -> bits[8] {{
                ret get_param.1: bits[8] = {}(x, id=1)
            }}"#,
                unop_xls_name
            )),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                unop(solver, &inputs.inputs.get("x").unwrap().bitvec)
            },
        );
    }

    pub fn test_binop_base<S: Solver>(
        solver_config: &S::Config,
        binop_xls_name: &str,
        binop: impl Fn(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
        lhs_width: usize,
        rhs_width: usize,
        result_width: usize,
    ) {
        let lhs_width_str = lhs_width.to_string();
        let rhs_width_str = rhs_width.to_string();
        let result_width_str = result_width.to_string();
        let ir_text = format!(
            r#"fn f(x: bits[{}], y: bits[{}]) -> bits[{}] {{
                ret get_param.1: bits[{}] = {}(x, y, id=1)
            }}"#,
            lhs_width_str, rhs_width_str, result_width_str, result_width_str, binop_xls_name
        );
        let mut parser = crate::xls_ir::ir_parser::Parser::new(&ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let smt_fn = ir_to_smt(&mut solver, &fn_inputs, &empty_map, &empty_registry);
        let x = fn_inputs.inputs.get("x").unwrap().bitvec.clone();
        let y = fn_inputs.inputs.get("y").unwrap().bitvec.clone();
        let expected = binop(&mut solver, &x, &y);
        crate::equiv::solver_interface::test_utils::assert_solver_eq(
            &mut solver,
            &smt_fn.output.bitvec,
            &expected,
        );
    }

    pub fn test_binop_8_bit<S: Solver>(
        solver_config: &S::Config,
        binop_xls_name: &str,
        binop: impl Fn(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
    ) {
        test_binop_base::<S>(solver_config, binop_xls_name, binop, 8, 8, 8);
    }

    pub fn test_binop_bool<S: Solver>(
        solver_config: &S::Config,
        binop_xls_name: &str,
        binop: impl Fn(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
    ) {
        test_binop_base::<S>(solver_config, binop_xls_name, binop, 8, 8, 1);
    }

    pub fn test_ir_tuple_index<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: (bits[8], bits[4])) -> bits[8] {
                ret tuple_index.1: bits[8] = tuple_index(input, index=0, id=1)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let tuple = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&tuple, 11, 4)
            },
        );
    }

    pub fn test_ir_tuple_index_literal<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(r#"fn f() -> bits[8] {
                    literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                    ret tuple_index.1: bits[8] = tuple_index(literal.1, index=0, id=1)
                    }"#),
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(8, "#x12"),
        );
    }

    pub fn test_ir_tuple_reverse<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(a: (bits[8], bits[4])) -> (bits[4], bits[8]) {
                    tuple_index.3: bits[4] = tuple_index(a, index=1, id=3)
                    tuple_index.5: bits[8] = tuple_index(a, index=0, id=5)
                    ret tuple.6: (bits[4], bits[8]) = tuple(tuple_index.3, tuple_index.5, id=6)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let tuple = inputs.inputs.get("a").unwrap().bitvec.clone();
                let tuple_0 = solver.extract(&tuple, 11, 4);
                let tuple_1 = solver.extract(&tuple, 3, 0);
                solver.concat(&tuple_1, &tuple_0)
            },
        );
    }

    pub fn test_ir_tuple_flattened<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> (bits[8], bits[4]) {
            ret tuple.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
        }"#,
            r#"fn g() -> bits[12] {
            ret tuple.1: bits[12] = literal(value=0x124, id=1)
        }"#,
            true,
        );
    }

    pub fn test_tuple_literal_vs_constructed<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn lhs() -> (bits[8], bits[4]) {
            ret lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
        }"#,
            r#"fn rhs() -> (bits[8], bits[4]) {
            lit0: bits[8] = literal(value=0x12, id=1)
            lit1: bits[4] = literal(value=0x4, id=2)
            ret tup: (bits[8], bits[4]) = tuple(lit0, lit1, id=3)
        }"#,
            false,
        );
    }

    pub fn test_tuple_index_on_literal<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[8] {
            lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
            ret idx0: bits[8] = tuple_index(lit_tuple, index=0, id=2)
        }"#,
            r#"fn g() -> bits[8] {
            ret lit: bits[8] = literal(value=0x12, id=1)
        }"#,
            false,
        );
    }

    pub fn test_ir_array_index_base<S: Solver>(
        solver_config: &S::Config,
        index: &str,
        expected_low: i32,
    ) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(r#"fn f(input: bits[8][4] id=1) -> bits[8] {
                    literal.4: bits[3] = literal(value="#
                .to_string()
                + index
                + r#", id=4)
                    ret array_index.5: bits[8] = array_index(input, indices=[literal.4], assumed_in_bounds=true, id=5)
                }"#),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&array, expected_low + 7, expected_low)
            },
        );
    }

    pub fn test_ir_array_index_multi_level<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4][2] id=1) -> bits[8] {
                literal.4: bits[2] = literal(value=1, id=4)
                ret array_index.6: bits[8] = array_index(input, indices=[literal.4], assumed_in_bounds=true, id=6)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&array, 63, 32)
            },
        );
    }

    pub fn test_ir_array_index_deep_multi_level<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4][2] id=1) -> bits[8] {
                literal.4: bits[2] = literal(value=1, id=4)
                literal.5: bits[2] = literal(value=0, id=5)
                ret array_index.6: bits[8] = array_index(input, indices=[literal.4, literal.5], assumed_in_bounds=true, id=6)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&array, 39, 32)
            },
        );
    }

    pub fn test_array_update_inbound_value<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[4][4] id=1, val: bits[4] id=2) -> bits[4][4] {
                lit: bits[2] = literal(value=1, id=3)
                ret upd: bits[4][4] = array_update(input, val, indices=[lit], id=4)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                let pre = solver.extract(&array, 3, 0);
                let with_mid = solver.concat(&val, &pre);
                let post = solver.extract(&array, 15, 8);
                solver.concat(&post, &with_mid)
            },
        );
    }

    pub fn test_array_update_nested<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4][4] id=1, val: bits[8][4] id=2) -> bits[8][4][4] {
                idx0: bits[2] = literal(value=1, id=3)
                ret upd: bits[8][4][4] = array_update(input, val, indices=[idx0], id=5)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                let pre = solver.extract(&array, 31, 0);
                let with_mid = solver.concat(&val, &pre);
                let post = solver.extract(&array, 127, 64);
                solver.concat(&post, &with_mid)
            },
        );
    }

    pub fn test_array_update_deep_nested<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][2][2] id=1, val: bits[8] id=2) -> bits[8][2][2] {
                idx0: bits[2] = literal(value=1, id=3)
                idx1: bits[2] = literal(value=0, id=4)
                ret upd: bits[8][2][2] = array_update(input, val, indices=[idx0, idx1], id=5)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                let pre = solver.extract(&array, 15, 0);
                let with_mid = solver.concat(&val, &pre);
                let post = solver.extract(&array, 31, 24);
                solver.concat(&post, &with_mid)
            },
        );
    }

    pub fn test_extend_base<S: Solver>(
        solver_config: &S::Config,
        extend_width: usize,
        signed: bool,
    ) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(format!(
                r#"fn f(x: bits[8]) -> bits[{}] {{
                ret get_param.1: bits[{}] = {}(x, new_bit_count={}, id=1)
            }}"#,
                extend_width,
                extend_width,
                if signed { "sign_ext" } else { "zero_ext" },
                extend_width
            )),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                solver.extend_to(&x, extend_width, signed)
            },
        );
    }

    pub fn test_dynamic_bit_slice_base<S: Solver>(
        solver_config: &S::Config,
        start: usize,
        width: usize,
    ) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(format!(
                r#"fn f(input: bits[8]) -> bits[{}] {{
                start: bits[4] = literal(value={}, id=2)
                ret get_param.1: bits[{}] = dynamic_bit_slice(input, start, width={}, id=1)
            }}"#,
                width, start, width, width
            )),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let needed_width = start + width;
                let input_ext = if needed_width > input.get_width() {
                    solver.extend_to(&input, needed_width, false)
                } else {
                    input
                };
                solver.slice(&input_ext, start, width)
            },
        );
    }

    pub fn test_bit_slice_base<S: Solver>(solver_config: &S::Config, start: usize, width: usize) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            &format!(
                r#"fn f(input: bits[8]) -> bits[{}] {{
                ret get_param.1: bits[{}] = bit_slice(input, start={}, width={}, id=1)
            }}"#,
                width, width, start, width
            ),
            &format!(
                r#"fn f(input: bits[8]) -> bits[{}] {{
                start: bits[4] = literal(value={}, id=2)
                ret get_param.1: bits[{}] = dynamic_bit_slice(input, start, width={}, id=1)
            }}"#,
                width, start, width, width
            ),
            false,
        );
    }

    pub fn test_bit_slice_update_zero<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=0, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_upper = solver.slice(&input, 4, 4);
                let updated = solver.concat(&input_upper, &slice);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_middle<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=1, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_lower = solver.slice(&input, 0, 1);
                let input_upper = solver.slice(&input, 5, 3);
                let updated = solver.concat(&slice, &input_lower);
                let updated = solver.concat(&input_upper, &updated);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_end<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=4, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_lower = solver.slice(&input, 0, 4);
                let updated = solver.concat(&slice, &input_lower);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_beyond_end<S: Solver>(solver_config: &S::Config) {
        use crate::equiv::prove_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[10]) -> bits[8] {
                    start: bits[4] = literal(value=4, id=2)
                    ret get_param.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_lower = solver.slice(&input, 0, 4);
                let slice_extracted = solver.slice(&slice, 0, 4);
                let updated = solver.concat(&slice_extracted, &input_lower);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_wider_update_value<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f(input: bits[7] id=1) -> bits[5] {
                    slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
                    ret upd.3: bits[5] = bit_slice_update(slice.2, input, input, id=3)
                }"#,
            r#"fn f(input: bits[7] id=1) -> bits[5] {
                    slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
                    ret upd.3: bits[5] = bit_slice_update(slice.2, input, input, id=3)
                }"#,
            false,
        );
    }

    pub fn test_bit_slice_update_large_update_value<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> bits[32] {
                    operand: bits[32] = literal(value=0xABCD1234, id=1)
                    start: bits[5] = literal(value=4, id=2)
                    upd_val: bits[80] = literal(value=0xFFFFFFFFFFFFFFFFFFF, id=3)
                    ret r: bits[32] = bit_slice_update(operand, start, upd_val, id=4)
                }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(32, "#xFFFFFFF4"),
        );
    }

    pub fn test_fuzz_dynamic_bit_slice_shrink_panics<S: Solver>(solver_config: &S::Config) {
        let ir = r#"fn bad(input: bits[32]) -> bits[16] {
                    start: bits[4] = literal(value=4, id=2)
                    ret r: bits[16] = dynamic_bit_slice(input, start, width=16, id=1)
                }"#;
        let f = crate::xls_ir::ir_parser::Parser::new(ir)
            .parse_fn()
            .unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        // This call should not panic
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let _ = ir_to_smt(&mut solver, &inputs, &empty_map, &empty_registry);
    }

    pub fn test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob<S: Solver>(
        solver_config: &S::Config,
    ) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
                    literal_255: bits[8] = literal(value=255, id=3)
                    bsu1: bits[8] = bit_slice_update(input, input, input, id=2)
                    ret bsu2: bits[8] = bit_slice_update(literal_255, literal_255, bsu1, id=4)
                }"#,
            r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
                    ret literal_255: bits[8] = literal(value=255, id=3)
                }"#,
            false,
        );
    }

    pub fn test_prove_fn_equiv<S: Solver>(solver_config: &S::Config) {
        test_ir_fn_equiv_to_self::<S>(
            solver_config,
            r#"fn f(x: bits[8], y: bits[8]) -> bits[8] {
                ret get_param.1: bits[8] = identity(x, id=1)
            }"#,
        );
    }

    pub fn test_prove_fn_inequiv<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_inequiv::<S>(
            solver_config,
            r#"fn f(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = identity(x, id=1)
                }"#,
            r#"fn g(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = not(x, id=1)
                }"#,
            false,
        );
    }

    // --------------------------------------------------------------
    // Nary operation test helpers (Concat/And/Or/Xor/Nor/Nand)
    // --------------------------------------------------------------
    pub fn test_nary_base<S: Solver>(
        solver_config: &S::Config,
        op_name: &str,
        builder: impl Fn(
            &mut S,
            &BitVec<S::Term>,
            &BitVec<S::Term>,
            &BitVec<S::Term>,
        ) -> BitVec<S::Term>,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &format!(
                r#"fn f(a: bits[4], b: bits[4], c: bits[4]) -> bits[4] {{
                    ret nary.1: bits[4] = {op_name}(a, b, c, id=1)
                }}"#,
            ),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let a = inputs.inputs.get("a").unwrap().bitvec.clone();
                let b = inputs.inputs.get("b").unwrap().bitvec.clone();
                let c = inputs.inputs.get("c").unwrap().bitvec.clone();
                builder(solver, &a, &b, &c)
            },
        );
    }

    pub fn test_ir_decode_base<S: Solver>(
        solver_config: &S::Config,
        in_width: usize,
        out_width: usize,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &format!(
                r#"fn f(x: bits[{iw}]) -> bits[{ow}] {{
    ret d.1: bits[{ow}] = decode(x, width={ow}, id=1)
}}"#,
                iw = in_width,
                ow = out_width
            ),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                solver.xls_decode(&x, out_width)
            },
        );
    }

    pub fn test_ir_encode_base<S: Solver>(
        solver_config: &S::Config,
        in_width: usize,
        out_width: usize,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &format!(
                r#"fn f(x: bits[{iw}]) -> bits[{ow}] {{
                        ret e.1: bits[{ow}] = encode(x, id=1)
                    }}"#,
                iw = in_width,
                ow = out_width
            ),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                solver.xls_encode(&x)
            },
        );
    }

    pub fn test_ir_one_hot_base<S: Solver>(solver_config: &S::Config, lsb_prio: bool) {
        assert_smt_fn_eq::<S>(
            solver_config,
            if lsb_prio {
                r#"fn f(x: bits[16]) -> bits[16] {
                        ret oh.1: bits[16] = one_hot(x, lsb_prio=true, id=1)
                    }"#
            } else {
                r#"fn f(x: bits[16]) -> bits[16] {
                        ret oh.1: bits[16] = one_hot(x, lsb_prio=false, id=1)
                    }"#
            },
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                if lsb_prio {
                    solver.xls_one_hot_lsb_prio(&x)
                } else {
                    solver.xls_one_hot_msb_prio(&x)
                }
            },
        );
    }

    // ----------------------------
    // Sel / OneHotSel / PrioritySel tests
    // ----------------------------

    // sel basic: selector in range (bits[2]=1) -> second case
    pub fn test_sel_basic<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
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
        );
    }

    // sel default path: selector out of range chooses default
    pub fn test_sel_default<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
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
        );
    }

    pub fn test_sel_missing_default_panics<S: Solver>(solver_config: &S::Config) {
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
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        // Should panic during conversion due to missing default
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let _ = ir_to_smt(&mut solver, &inputs, &empty_map, &empty_registry);
    }

    pub fn test_sel_unexpected_default_panics<S: Solver>(solver_config: &S::Config) {
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
        let mut solver = S::new(solver_config).unwrap();
        let ir_fn = IrFn::new(&f, None);
        let inputs = get_fn_inputs(&mut solver, &ir_fn, None);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let _ = ir_to_smt(&mut solver, &inputs, &empty_map, &empty_registry);
    }

    // one_hot_sel: multiple bits
    pub fn test_one_hot_sel_multi<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
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
        );
    }

    // priority_sel: multiple bits -> lowest index wins
    pub fn test_priority_sel_multi<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
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
        );
    }

    // priority_sel: no bits set selects default
    pub fn test_priority_sel_default<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
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
        );
    }

    fn assert_ir_text(is_lt: bool, assert_value: u32, ret_value: u32) -> String {
        format!(
            r#"fn f(__token: token, a: bits[4]) -> (token, bits[4]) {{
            literal.1: bits[4] = literal(value={assert_value}, id=1)
            {op}.2: bits[1] = {op}(a, literal.1, id=2)
            assert.3: token = assert(__token, {op}.2, message="Assertion failure!", label="a", id=3)
            literal.4: bits[4] = literal(value={ret_value}, id=4)
            ret tuple.5: (token, bits[4]) = tuple(assert.3, literal.4, id=4)
        }}"#,
            op = if is_lt { "ult" } else { "uge" },
            ret_value = ret_value,
        )
    }

    // Helper builders leveraging `assert_ir_text` from above.
    fn lt_ir(threshold: u32, ret_val: u32) -> String {
        // Assertion passes when a < threshold
        assert_ir_text(true, threshold, ret_val)
    }

    fn ge_ir(threshold: u32, ret_val: u32) -> String {
        // Assertion passes when a >= threshold
        assert_ir_text(false, threshold, ret_val)
    }

    fn vacuous_ir(ret_val: u32) -> String {
        assert_ir_text(true, 0, ret_val)
    }

    fn no_assertion_ir(ret_val: u32) -> String {
        assert_ir_text(false, 0, ret_val)
    }

    // ----- Same -----
    pub fn test_assert_semantics_same<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lt_ir(1, 2),
            &lt_ir(1, 2),
            false,
            AssertionSemantics::Same,
            true,
        );
    }

    pub fn test_assert_semantics_same_different_assertion<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lt_ir(1, 2),
            &lt_ir(2, 2),
            false,
            AssertionSemantics::Same,
            false,
        );
    }

    pub fn test_assert_semantics_same_different_result<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lt_ir(1, 2),
            &lt_ir(1, 3),
            false,
            AssertionSemantics::Same,
            false,
        );
    }

    // ----- Ignore -----
    pub fn test_assert_semantics_ignore_proved<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(5, 1);
        let rhs = ge_ir(10, 1);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Ignore,
            true,
        );
    }

    pub fn test_assert_semantics_ignore_different_result<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(8, 1);
        let rhs = lt_ir(8, 2);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Ignore,
            false,
        );
    }

    // ----- Never -----
    pub fn test_assert_semantics_never_proved<S: Solver>(solver_config: &S::Config) {
        let lhs = no_assertion_ir(1);
        let rhs = no_assertion_ir(1);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Never,
            true,
        );
    }

    pub fn test_assert_semantics_never_any_fail<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(4, 1);
        let rhs = lt_ir(4, 1);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Never,
            false,
        );
    }

    // ----- Assume -----
    pub fn test_assert_semantics_assume_proved_vacuous<S: Solver>(solver_config: &S::Config) {
        let lhs = vacuous_ir(1);
        let rhs = no_assertion_ir(2);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Assume,
            true,
        );
    }

    pub fn test_assert_semantics_assume_disproved<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(8, 1);
        let rhs = lt_ir(8, 2); // both succeed for same range a<8
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Assume,
            false,
        );
    }

    // ----- Implies -----
    pub fn test_assert_semantics_implies_proved_lhs_fails<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(5, 1); // passes a<5
        let rhs = lt_ir(8, 1); // passes a<8
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Implies,
            true,
        );
    }

    pub fn test_assert_semantics_implies_disproved_rhs_fails<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(8, 1); // passes a<8
        let rhs = lt_ir(5, 1); // passes a<5 (subset) - rhs may fail when lhs passes
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Implies,
            false,
        );
    }

    fn fail_if_deactivated() -> &'static str {
        r#"fn f(__token: token, __activate: bits[1], tok: token, a: bits[4]) -> (token, bits[4]) {
                assert.3: token = assert(__token, __activate, message="Assertion failure!", label="a", id=3)
                literal.4: bits[4] = literal(value=1, id=4)
                ret tuple.5: (token, bits[4]) = tuple(assert.3, literal.4, id=4)
            }"#
    }

    fn fail_if_activated() -> &'static str {
        r#"fn f(__token: token, __activate: bits[1], tok: token, a: bits[4]) -> (token, bits[4]) {
                not.2: bits[1] = not(__activate, id=2)
                assert.3: token = assert(__token, not.2, message="Assertion failure!", label="a", id=3)
                literal.4: bits[4] = literal(value=1, id=4)
                ret tuple.5: (token, bits[4]) = tuple(assert.3, literal.4, id=4)
            }"#
    }

    fn plain_success() -> &'static str {
        r#"fn f(tok: token, a: bits[4]) -> (token, bits[4]) {
                literal.1: bits[4] = literal(value=1, id=1)
                ret tuple.2: (token, bits[4]) = tuple(tok, literal.1, id=2)
            }"#
    }

    fn plain_failure() -> &'static str {
        r#"fn f(tok: token, a: bits[4]) -> (token, bits[4]) {
                literal.1: bits[1] = literal(value=0, id=1)
                assert.2: token = assert(tok, literal.1, message="Assertion failure!", label="a", id=2)
                literal.3: bits[4] = literal(value=1, id=3)
                ret tuple.4: (token, bits[4]) = tuple(tok, literal.3, id=4)
            }"#
    }

    pub fn test_both_implicit_token_no_fixed_implicit_activation<S: Solver>(
        solver_config: &S::Config,
    ) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_deactivated(),
            &fail_if_deactivated(),
            false,
            false,
            false,
            AssertionSemantics::Same,
            true,
        );
    }
    pub fn test_implicit_token_policy_fixed_implicit_activation<S: Solver>(
        solver_config: &S::Config,
    ) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_deactivated(),
            &fail_if_deactivated(),
            true,
            true,
            false,
            AssertionSemantics::Same,
            true,
        );
    }
    pub fn test_single_fixed_implicit_activation_success<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_deactivated(),
            &plain_success(),
            true,
            false,
            false,
            AssertionSemantics::Same,
            true,
        );
    }
    pub fn test_single_fixed_implicit_activation_failure<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_activated(),
            &plain_failure(),
            true,
            false,
            false,
            AssertionSemantics::Same,
            true,
        );
    }

    pub fn test_counterexample_input_order<S: Solver>(solver_config: &S::Config) {
        // IR pair intentionally inequivalent: returns different parameters to force a
        // counterexample.
        let lhs_ir = r#"fn lhs(a: bits[8], b: bits[8]) -> bits[8] {
                ret id.1: bits[8] = identity(a, id=1)
            }"#;
        let rhs_ir = r#"fn rhs(a: bits[8], b: bits[8]) -> bits[8] {
                ret id.1: bits[8] = identity(b, id=1)
            }"#;
        // Parse the IR text into functions.
        let mut parser = crate::xls_ir::ir_parser::Parser::new(lhs_ir);
        let lhs_fn_ir = parser.parse_fn().unwrap();
        let mut parser = crate::xls_ir::ir_parser::Parser::new(rhs_ir);
        let rhs_fn_ir = parser.parse_fn().unwrap();

        // Run equivalence prover – expect a counter-example (Disproved).
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &IrFn::new(&lhs_fn_ir, None),
            &IrFn::new(&rhs_fn_ir, None),
            AssertionSemantics::Same,
            false,
        );

        match res {
            EquivResult::Disproved {
                lhs_inputs,
                rhs_inputs,
                ..
            } => {
                // Verify LHS input ordering & naming.
                assert_eq!(lhs_inputs.len(), lhs_fn_ir.params.len());
                for (idx, param) in lhs_fn_ir.params.iter().enumerate() {
                    assert_eq!(lhs_inputs[idx].name, param.name);
                    assert_eq!(
                        lhs_inputs[idx].value.bit_count().unwrap(),
                        param.ty.bit_count()
                    );
                }
                // Verify RHS input ordering & naming.
                assert_eq!(rhs_inputs.len(), rhs_fn_ir.params.len());
                for (idx, param) in rhs_fn_ir.params.iter().enumerate() {
                    assert_eq!(rhs_inputs[idx].name, param.name);
                    assert_eq!(
                        rhs_inputs[idx].value.bit_count().unwrap(),
                        param.ty.bit_count()
                    );
                }
            }
            _ => panic!("Expected inequivalence with counter-example, but proof succeeded"),
        }
    }

    // New: shared test that exercises prove_ir_fn_equiv_with_domains.
    pub fn test_param_domains_equiv<S: Solver>(solver_config: &S::Config) {
        let lhs_ir = r#"
            fn f(x: bits[2]) -> bits[2] {
                ret id.1: bits[2] = identity(x, id=1)
            }
        "#;
        let rhs_ir = r#"
            fn g(x: bits[2]) -> bits[2] {
                one: bits[2] = literal(value=1, id=1)
                ret and.2: bits[2] = and(x, one, id=2)
            }
        "#;
        let mut parser = crate::xls_ir::ir_parser::Parser::new(lhs_ir);
        let lhs_fn_ir = parser.parse_fn().unwrap();
        let mut parser = crate::xls_ir::ir_parser::Parser::new(rhs_ir);
        let rhs_fn_ir = parser.parse_fn().unwrap();

        let lhs_ir_fn = IrFn::new(&lhs_fn_ir, None);
        let rhs_ir_fn = IrFn::new(&rhs_fn_ir, None);

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_ir_fn,
            &rhs_ir_fn,
            AssertionSemantics::Same,
            false,
        );
        assert!(matches!(res, super::EquivResult::Disproved { .. }));

        let mut doms: ParamDomains = ParamDomains::new();
        doms.insert(
            "x".to_string(),
            vec![
                xlsynth::IrValue::make_ubits(2, 0).unwrap(),
                xlsynth::IrValue::make_ubits(2, 1).unwrap(),
            ],
        );

        let res2 = prove_ir_fn_equiv_with_domains::<S>(
            solver_config,
            &lhs_ir_fn,
            &rhs_ir_fn,
            AssertionSemantics::Same,
            false,
            Some(&doms),
            Some(&doms),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(matches!(res2, super::EquivResult::Proved));
    }
}

#[cfg(test)]
macro_rules! test_with_solver {
    ($mod_ident:ident, $solver_type:ty, $solver_config:expr) => {
        #[cfg(test)]
        mod $mod_ident {
            use super::*;
            use crate::equiv::prove_equiv::test_utils;

            #[test]
            fn test_align_zero_width_fn_inputs() {
                test_utils::align_zero_width_fn_inputs::<$solver_type>($solver_config);
            }
            #[test]
            fn test_align_non_zero_width_fn_inputs_first_token() {
                test_utils::align_non_zero_width_fn_inputs_first_token::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_param_domains_equiv() {
                test_utils::test_param_domains_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_bits() {
                test_utils::test_ir_value_bits::<$solver_type>($solver_config);
            }
            #[test]
            fn test_invoke_basic_equiv() {
                test_utils::test_invoke_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_invoke_two_args_equiv() {
                test_utils::test_invoke_two_args::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_basic() {
                test_utils::test_counted_for_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_stride() {
                test_utils::test_counted_for_stride::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_zero_trip() {
                test_utils::test_counted_for_zero_trip::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_invariant_args() {
                test_utils::test_counted_for_invariant_args::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_bits() {
                test_utils::test_ir_bits::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_array() {
                test_utils::test_ir_value_array::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array() {
                test_utils::test_ir_array::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_tuple() {
                test_utils::test_ir_value_tuple::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_tuple() {
                test_utils::test_ir_tuple::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_token() {
                test_utils::test_ir_value_token::<$solver_type>($solver_config);
            }
            #[test]
            fn test_not() {
                test_utils::test_unop_base::<$solver_type>($solver_config, "not", Solver::not);
            }
            #[test]
            fn test_neg() {
                test_utils::test_unop_base::<$solver_type>($solver_config, "neg", Solver::neg);
            }
            #[test]
            fn test_bit_slice() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "or_reduce",
                    Solver::or_reduce,
                );
            }
            #[test]
            fn test_and_reduce() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "and_reduce",
                    Solver::and_reduce,
                );
            }
            #[test]
            fn test_xor_reduce() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "xor_reduce",
                    Solver::xor_reduce,
                );
            }
            #[test]
            fn test_identity() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "identity",
                    |_solver, x| x.clone(),
                );
            }
            #[test]
            fn test_reverse() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "reverse",
                    Solver::reverse,
                );
            }
            // crate::test_all_binops!($solver_type, $solver_config);
            #[test]
            fn test_add() {
                test_utils::test_binop_8_bit::<$solver_type>($solver_config, "add", Solver::add);
            }
            #[test]
            fn test_sub() {
                test_utils::test_binop_8_bit::<$solver_type>($solver_config, "sub", Solver::sub);
            }
            #[test]
            fn test_eq() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "eq", Solver::eq);
            }
            #[test]
            fn test_ne() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ne", Solver::ne);
            }
            #[test]
            fn test_uge() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "uge", Solver::uge);
            }
            #[test]
            fn test_ugt() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ugt", Solver::ugt);
            }
            #[test]
            fn test_ule() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ule", Solver::ule);
            }
            #[test]
            fn test_ult() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ult", Solver::ult);
            }
            #[test]
            fn test_slt() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "slt", Solver::slt);
            }
            #[test]
            fn test_sle() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "sle", Solver::sle);
            }
            #[test]
            fn test_sgt() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "sgt", Solver::sgt);
            }
            #[test]
            fn test_sge() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "sge", Solver::sge);
            }

            #[test]
            fn test_xls_shll() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "shll",
                    Solver::xls_shll,
                    8,
                    4,
                    8,
                );
            }
            #[test]
            fn test_xls_shrl() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "shrl",
                    Solver::xls_shrl,
                    8,
                    4,
                    8,
                );
            }
            #[test]
            fn test_xls_shra() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "shra",
                    Solver::xls_shra,
                    8,
                    4,
                    8,
                );
            }

            #[test]
            fn test_xls_umul() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "umul",
                    |solver, x, y| Solver::xls_arbitrary_width_umul(solver, x, y, 16),
                    8,
                    12,
                    16,
                );
            }
            #[test]
            fn test_xls_smul() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "smul",
                    |solver, x, y| Solver::xls_arbitrary_width_smul(solver, x, y, 16),
                    8,
                    12,
                    16,
                );
            }
            #[test]
            fn test_xls_udiv() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "udiv",
                    Solver::xls_udiv,
                );
            }

            #[test]
            fn test_xls_sdiv() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "sdiv",
                    Solver::xls_sdiv,
                );
            }
            #[test]
            fn test_xls_umod() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "umod",
                    Solver::xls_umod,
                );
            }
            #[test]
            fn test_xls_smod() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "smod",
                    Solver::xls_smod,
                );
            }

            #[test]
            fn test_ir_tuple_index() {
                test_utils::test_ir_tuple_index::<$solver_type>($solver_config);
            }

            #[test]
            fn test_ir_tuple_index_literal() {
                test_utils::test_ir_tuple_index_literal::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_tuple_reverse() {
                test_utils::test_ir_tuple_reverse::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_tuple_flattened() {
                test_utils::test_ir_tuple_flattened::<$solver_type>($solver_config);
            }
            #[test]
            fn test_tuple_literal_vs_constructed() {
                test_utils::test_tuple_literal_vs_constructed::<$solver_type>($solver_config);
            }
            #[test]
            fn test_tuple_index_on_literal() {
                test_utils::test_tuple_index_on_literal::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array_index_0() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "0", 0);
            }
            #[test]
            fn test_ir_array_index_1() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "1", 8);
            }
            #[test]
            fn test_ir_array_index_2() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "2", 16);
            }
            #[test]
            fn test_ir_array_index_3() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "3", 24);
            }
            #[test]
            fn test_ir_array_index_4() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "4", 24);
            }
            #[test]
            fn test_ir_array_index_multi_level() {
                test_utils::test_ir_array_index_multi_level::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array_index_deep_multi_level() {
                test_utils::test_ir_array_index_deep_multi_level::<$solver_type>($solver_config);
            }
            #[test]
            fn test_array_update_inbound_value() {
                test_utils::test_array_update_inbound_value::<$solver_type>($solver_config);
            }
            #[test]
            fn test_array_update_nested() {
                test_utils::test_array_update_nested::<$solver_type>($solver_config);
            }
            #[test]
            fn test_array_update_deep_nested() {
                test_utils::test_array_update_deep_nested::<$solver_type>($solver_config);
            }
            #[test]
            fn test_prove_fn_equiv() {
                test_utils::test_prove_fn_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_prove_fn_inequiv() {
                test_utils::test_prove_fn_inequiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_extend_zero() {
                test_utils::test_extend_base::<$solver_type>($solver_config, 16, false);
            }
            #[test]
            fn test_extend_sign() {
                test_utils::test_extend_base::<$solver_type>($solver_config, 16, true);
            }
            #[test]
            fn test_dynamic_bit_slice_0_4() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 0, 4);
            }
            #[test]
            fn test_dynamic_bit_slice_5_4() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 5, 4);
            }
            #[test]
            fn test_dynamic_bit_slice_0_8() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 0, 8);
            }
            #[test]
            fn test_dynamic_bit_slice_5_8() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 5, 8);
            }
            #[test]
            fn test_dynamic_bit_slice_10_4() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 10, 4);
            }
            #[test]
            fn test_dynamic_bit_slice_10_8() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 10, 8);
            }
            // crate::test_bit_slice!($solver_type, $solver_config);
            #[test]
            fn test_bit_slice_0_4() {
                test_utils::test_bit_slice_base::<$solver_type>($solver_config, 0, 4);
            }
            #[test]
            fn test_bit_slice_5_3() {
                test_utils::test_bit_slice_base::<$solver_type>($solver_config, 5, 3);
            }
            #[test]
            fn test_bit_slice_0_8() {
                test_utils::test_bit_slice_base::<$solver_type>($solver_config, 0, 8);
            }
            #[test]
            fn test_bit_slice_update_zero() {
                test_utils::test_bit_slice_update_zero::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_middle() {
                test_utils::test_bit_slice_update_middle::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_end() {
                test_utils::test_bit_slice_update_end::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_beyond_end() {
                test_utils::test_bit_slice_update_beyond_end::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_wider_update_value() {
                test_utils::test_bit_slice_update_wider_update_value::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_bit_slice_update_large_update_value() {
                test_utils::test_bit_slice_update_large_update_value::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_fuzz_dynamic_bit_slice_shrink_panics() {
                test_utils::test_fuzz_dynamic_bit_slice_shrink_panics::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob() {
                test_utils::test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_nary_concat() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "concat",
                    |solver, a, b, c| solver.concat_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_and() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "and",
                    |solver, a, b, c| solver.and_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_or() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "or",
                    |solver, a, b, c| solver.or_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_xor() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "xor",
                    |solver, a, b, c| solver.xor_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_nor() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "nor",
                    |solver, a, b, c| solver.nor_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_nand() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "nand",
                    |solver, a, b, c| solver.nand_many([a, b, c].to_vec()),
                );
            }

            #[test]
            fn test_ir_decode_0() {
                test_utils::test_ir_decode_base::<$solver_type>($solver_config, 8, 8);
            }
            #[test]
            fn test_ir_decode_1() {
                test_utils::test_ir_decode_base::<$solver_type>($solver_config, 8, 16);
            }
            #[test]
            fn test_ir_encode_0() {
                test_utils::test_ir_encode_base::<$solver_type>($solver_config, 8, 3);
            }
            #[test]
            fn test_ir_encode_1() {
                test_utils::test_ir_encode_base::<$solver_type>($solver_config, 16, 4);
            }
            #[test]
            fn test_ir_one_hot_true() {
                test_utils::test_ir_one_hot_base::<$solver_type>($solver_config, true);
            }
            #[test]
            fn test_ir_one_hot_false() {
                test_utils::test_ir_one_hot_base::<$solver_type>($solver_config, false);
            }
            #[test]
            fn test_sel_basic() {
                test_utils::test_sel_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_sel_default() {
                test_utils::test_sel_default::<$solver_type>($solver_config);
            }
            #[should_panic]
            #[test]
            fn test_sel_missing_default_panics() {
                test_utils::test_sel_missing_default_panics::<$solver_type>($solver_config);
            }
            #[should_panic]
            #[test]
            fn test_sel_unexpected_default_panics() {
                test_utils::test_sel_unexpected_default_panics::<$solver_type>($solver_config);
            }
            #[test]
            fn test_one_hot_sel_multi() {
                test_utils::test_one_hot_sel_multi::<$solver_type>($solver_config);
            }
            #[test]
            fn test_priority_sel_multi() {
                test_utils::test_priority_sel_multi::<$solver_type>($solver_config);
            }
            #[test]
            fn test_priority_sel_default() {
                test_utils::test_priority_sel_default::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_same() {
                test_utils::test_assert_semantics_same::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_same_different_assertion() {
                test_utils::test_assert_semantics_same_different_assertion::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_same_different_result() {
                test_utils::test_assert_semantics_same_different_result::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_ignore_proved() {
                test_utils::test_assert_semantics_ignore_proved::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_ignore_different_result() {
                test_utils::test_assert_semantics_ignore_different_result::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_never_proved() {
                test_utils::test_assert_semantics_never_proved::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_never_any_fail() {
                test_utils::test_assert_semantics_never_any_fail::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_assume_proved_vacuous() {
                test_utils::test_assert_semantics_assume_proved_vacuous::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_assume_disproved() {
                test_utils::test_assert_semantics_assume_disproved::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_implies_proved_lhs_fails() {
                test_utils::test_assert_semantics_implies_proved_lhs_fails::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_implies_disproved_rhs_fails() {
                test_utils::test_assert_semantics_implies_disproved_rhs_fails::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_both_implicit_token_no_fixed_implicit_activation() {
                test_utils::test_both_implicit_token_no_fixed_implicit_activation::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_implicit_token_policy_fixed_implicit_activation() {
                test_utils::test_implicit_token_policy_fixed_implicit_activation::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_single_fixed_implicit_activation_success() {
                test_utils::test_single_fixed_implicit_activation_success::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_single_fixed_implicit_activation_failure() {
                test_utils::test_single_fixed_implicit_activation_failure::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_counterexample_input_order() {
                test_utils::test_counterexample_input_order::<$solver_type>($solver_config);
            }
            #[test]
            fn test_uf_basic_equiv() {
                test_utils::test_uf_basic_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_uf_nested_invoke_equiv() {
                test_utils::test_uf_nested_invoke_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_uf_implicit_token_two_args_equiv() {
                test_utils::test_uf_implicit_token_two_args_equiv::<$solver_type>($solver_config);
            }
        }
    };
}

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-binary-test")]
test_with_solver!(
    bitwuzla_tests,
    crate::equiv::easy_smt_backend::EasySmtSolver,
    &crate::equiv::easy_smt_backend::EasySmtConfig::bitwuzla()
);

#[cfg(test)]
#[cfg(feature = "with-boolector-binary-test")]
test_with_solver!(
    boolector_tests,
    crate::equiv::easy_smt_backend::EasySmtSolver,
    &crate::equiv::easy_smt_backend::EasySmtConfig::boolector()
);

#[cfg(test)]
#[cfg(feature = "with-z3-binary-test")]
test_with_solver!(
    z3_tests,
    crate::equiv::easy_smt_backend::EasySmtSolver,
    &crate::equiv::easy_smt_backend::EasySmtConfig::z3()
);

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-built")]
test_with_solver!(
    bitwuzla_built_tests,
    crate::equiv::bitwuzla_backend::Bitwuzla,
    &crate::equiv::bitwuzla_backend::BitwuzlaOptions::new()
);
