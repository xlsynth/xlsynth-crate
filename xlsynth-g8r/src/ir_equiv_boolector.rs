// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-boolector")]

use crate::ir_value_utils::ir_value_from_bits_with_type;
use crate::xls_ir::ir::{Fn, NodePayload, NodeRef, Type};
use crate::xls_ir::ir_utils::get_topological;
use boolector::option::{BtorOption, ModelGen};
use boolector::{BV, Btor, SolverResult};
use log::debug;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use xlsynth::IrBits;
use xlsynth::IrValue;

/// Result of converting an XLS IR function to Boolector logic.
pub struct IrFnBoolectorResult {
    pub output: BV<Rc<Btor>>,
    pub inputs: Vec<(String, BV<Rc<Btor>>)>,
}

/// Solver context that can be reused across multiple equivalence checks.
pub struct Ctx {
    btor: Rc<Btor>,
    lhs_params: HashMap<String, BV<Rc<Btor>>>,
    rhs_params: HashMap<String, BV<Rc<Btor>>>,
    flattened_params: Option<BV<Rc<Btor>>>,
}

/// Whether a Boolector solver should be created with incremental (push/pop)
/// capability enabled.
enum Incremental {
    Yes,
    #[allow(dead_code)]
    No,
}

/// Create a fresh `Btor` with model generation always enabled and incremental
/// mode enabled iff `incremental == Incremental::Yes`.
fn new_btor(incremental: Incremental) -> Rc<Btor> {
    let btor = Rc::new(Btor::new());
    btor.set_opt(BtorOption::ModelGen(ModelGen::All));
    if matches!(incremental, Incremental::Yes) {
        btor.set_opt(BtorOption::Incremental(true));
    }
    btor
}

impl Ctx {
    /// Creates a new Boolector solver context with model generation enabled
    /// and incremental solving turned on so clause learning can be reused
    /// across proofs.
    pub fn new(lhs: &Fn, rhs: &Fn) -> Self {
        let btor = new_btor(Incremental::Yes);
        let collect_param_bit_widths = |params: &[crate::xls_ir::ir::Param]| {
            params
                .iter()
                .map(|p| (p.name.clone(), p.ty.bit_count() as u32))
                .collect()
        };
        let lhs_params_bit_widths: Vec<(String, u32)> = collect_param_bit_widths(&lhs.params);
        let rhs_params_bit_widths: Vec<(String, u32)> = collect_param_bit_widths(&rhs.params);
        let lhs_params_total_width = lhs_params_bit_widths.iter().map(|(_, w)| *w).sum::<u32>();
        let rhs_params_total_width = rhs_params_bit_widths.iter().map(|(_, w)| *w).sum::<u32>();
        assert_eq!(
            lhs_params_total_width, rhs_params_total_width,
            "LHS and RHS must have the same number of bits"
        );
        if lhs_params_total_width == 0 {
            return Self {
                btor,
                lhs_params: HashMap::new(),
                rhs_params: HashMap::new(),
                flattened_params: None,
            };
        }
        let flattened_params = BV::new(
            btor.clone(),
            lhs_params_total_width,
            Some("flattened_params"),
        );
        // split the flattened params by the bit widths of the lhs and rhs params
        // Low should be the last high in each iteration
        fn split_params<'a>(
            param_bit_widths: impl IntoIterator<Item = (String, u32)>,
            bv: &BV<Rc<Btor>>,
        ) -> HashMap<String, BV<Rc<Btor>>> {
            let mut params = HashMap::new();
            let mut low = 0;
            for (name, width) in param_bit_widths {
                params.insert(name.clone(), bv.slice(low + width - 1, low));
                low += width;
            }
            params
        }

        let lhs_params = split_params(lhs_params_bit_widths, &flattened_params);
        let rhs_params = split_params(rhs_params_bit_widths, &flattened_params);
        Self {
            btor,
            lhs_params,
            rhs_params,
            flattened_params: Some(flattened_params),
        }
    }
}

/// Converts an XLS IR function to Boolector bitvector logic.
/// If param_bvs is provided, uses those BVs for parameters; otherwise, creates
/// new ones.
pub fn ir_fn_to_boolector(
    btor: Rc<Btor>,
    f: &Fn,
    param_bvs: Option<&HashMap<String, BV<Rc<Btor>>>>,
) -> IrFnBoolectorResult {
    let topo = get_topological(f);
    let mut env: HashMap<NodeRef, BV<Rc<Btor>>> = HashMap::new();
    let mut inputs = Vec::new();
    for node_ref in topo {
        let node = &f.nodes[node_ref.index];
        log::trace!(
            "[ir_fn_to_boolector] Processing node_ref: {:?}, node: {:?}",
            node_ref,
            node
        );
        let bv = match &node.payload {
            NodePayload::Literal(ir_value) => ir_value_to_bv(btor.clone(), ir_value, &node.ty),
            NodePayload::GetParam(param_id) => {
                // Find the parameter by id
                let param = f
                    .params
                    .iter()
                    .find(|p| p.id == *param_id)
                    .expect("Param not found");
                let width = param.ty.bit_count() as u32;
                if let Some(map) = param_bvs {
                    assert!(
                        map.contains_key(&param.name),
                        "param_bvs missing param '{}', all param names: {:?}",
                        param.name,
                        map.keys()
                    );
                    map.get(&param.name)
                        .expect("param_bvs missing param")
                        .clone()
                } else {
                    let bv = BV::new(btor.clone(), width, Some(&param.name));
                    debug!("Adding param BV: name={}, width={}", param.name, width);
                    inputs.push((param.name.clone(), bv.clone()));
                    bv
                }
            }
            NodePayload::TupleIndex { tuple, index } => {
                // Get the tuple BV and type
                let tuple_bv = env.get(tuple).expect("Tuple operand must be present");
                let tuple_ty = f.get_node_ty(*tuple);
                // Get the bit range for the requested index
                let slice = tuple_ty
                    .tuple_get_flat_bit_slice_for_index(*index)
                    .expect("TupleIndex: not a tuple type");
                let width = slice.limit - slice.start;
                assert!(width > 0, "TupleIndex: width must be > 0");
                // Boolector's slice is inclusive: slice(high, low)
                let high = (slice.limit - 1) as u32;
                let low = slice.start as u32;
                tuple_bv.slice(high, low)
            }
            NodePayload::Nil => {
                // Do not insert a BV for Nil nodes
                continue;
            }
            NodePayload::Unop(op, arg) => {
                let arg_bv = env.get(arg).expect("Unop argument must be present");
                match op {
                    crate::xls_ir::ir::Unop::Not => arg_bv.not(),
                    crate::xls_ir::ir::Unop::Neg => arg_bv.neg(),
                    crate::xls_ir::ir::Unop::OrReduce => {
                        let width = arg_bv.get_width();
                        let mut result = arg_bv.slice(0, 0);
                        for i in 1..width {
                            let bit = arg_bv.slice(i, i);
                            result = result.or(&bit);
                        }
                        result
                    }
                    crate::xls_ir::ir::Unop::AndReduce => {
                        let width = arg_bv.get_width();
                        let mut result = arg_bv.slice(0, 0);
                        for i in 1..width {
                            let bit = arg_bv.slice(i, i);
                            result = result.and(&bit);
                        }
                        result
                    }
                    crate::xls_ir::ir::Unop::Identity => arg_bv.clone(),
                    crate::xls_ir::ir::Unop::Reverse => {
                        let width = arg_bv.get_width();
                        if width == 0 {
                            arg_bv.clone()
                        } else {
                            // Iterate from LSB to MSB of the input, and construct the result such
                            // that the input LSB becomes the output MSB.
                            // Boolector's concat A.concat(B) means A is MSB part, B is LSB part.
                            // Start with original LSB as the initial (most significant part of)
                            // result.
                            let mut result = arg_bv.slice(0, 0); // original LSB
                            for i in 1..width {
                                let bit = arg_bv.slice(i, i); // next original bit (towards MSB)
                                result = result.concat(&bit); // current_result_MSBs.concat(new_LSB_bit)
                            }
                            result
                        }
                    }
                    crate::xls_ir::ir::Unop::XorReduce => {
                        let width = arg_bv.get_width();
                        let mut result = arg_bv.slice(0, 0);
                        for i in 1..width {
                            let bit = arg_bv.slice(i, i);
                            result = result.xor(&bit);
                        }
                        result
                    }
                }
            }
            NodePayload::OneHot { arg, lsb_prio } => {
                // Implements XLS semantics:
                //   out[i] = arg[i] & !arg[0..i-1] (if lsb_prio)
                // or out[i] = arg[i] & !arg[i+1..] (if msb priority)
                //   out[N] = (arg == 0)

                let arg_bv = env.get(arg).expect("OneHot argument must be present");
                let width = arg_bv.get_width();
                assert!(width > 0, "OneHot: width must be > 0");

                let btor = arg_bv.get_btor();
                let mut outputs: Vec<BV<Rc<Btor>>> = Vec::with_capacity((width + 1) as usize);

                // prior_clear is 1 when no earlier-priority bits have been seen.
                let mut prior_clear = BV::from_u64(btor.clone(), 1, 1);

                for i in 0..width {
                    let idx = if *lsb_prio { i } else { width - 1 - i };
                    let bit = arg_bv.slice(idx, idx);
                    let out_bit = bit.and(&prior_clear);
                    outputs.push(out_bit);
                    // Update prior_clear = prior_clear & !bit
                    prior_clear = prior_clear.and(&bit.not());
                }

                if !*lsb_prio {
                    outputs.reverse(); // now outputs[0] is bit0
                }

                // Final bit indicates no bits set in input.
                outputs.push(prior_clear.clone()); // index = width

                // Concatenate: start with LSB bit and prepend higher bits to build final value.
                let mut result = outputs[0].clone();
                for b in outputs.iter().skip(1) {
                    result = b.concat(&result);
                }
                result
            }
            NodePayload::Nary(op, elems) => match op {
                crate::xls_ir::ir::NaryOp::Concat => {
                    assert!(elems.len() >= 2, "Concat must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Concat operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Concat operand must be present");
                        acc.concat(next)
                    })
                }
                crate::xls_ir::ir::NaryOp::And => {
                    assert!(elems.len() >= 2, "And must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("And operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("And operand must be present");
                        acc.and(next)
                    })
                }
                crate::xls_ir::ir::NaryOp::Xor => {
                    assert!(elems.len() >= 2, "Xor must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Xor operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Xor operand must be present");
                        acc.xor(next)
                    })
                }
                crate::xls_ir::ir::NaryOp::Nor => {
                    assert!(elems.len() >= 2, "Nor must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Nor operand must be present")
                        .clone();
                    let or_result = it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Nor operand must be present");
                        acc.or(next)
                    });
                    or_result.not()
                }
                crate::xls_ir::ir::NaryOp::Or => {
                    assert!(elems.len() >= 2, "Or must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Or operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Or operand must be present");
                        acc.or(next)
                    })
                }
                crate::xls_ir::ir::NaryOp::Nand => {
                    assert!(elems.len() >= 2, "Nand must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Nand operand must be present")
                        .clone();
                    let and_result = it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Nand operand must be present");
                        acc.and(next)
                    });
                    and_result.not()
                }
            },
            NodePayload::Binop(op, a, b) => {
                let a_bv = env.get(a).expect("Binop lhs must be present");
                let b_bv = env.get(b).expect("Binop rhs must be present");
                use crate::xls_ir::ir::Binop;
                match op {
                    Binop::Add => a_bv.add(b_bv),
                    Binop::Sub => a_bv.sub(b_bv),
                    Binop::Eq => a_bv._eq(b_bv),
                    Binop::Ne => a_bv._ne(b_bv),
                    Binop::Uge => a_bv.ugte(b_bv),
                    Binop::Ugt => a_bv.ugt(b_bv),
                    Binop::Ult => a_bv.ult(b_bv),
                    Binop::Ule => a_bv.ulte(b_bv),
                    Binop::Umul | Binop::Smul => {
                        let expected_width = node.ty.bit_count() as u32;
                        let (a_ext, b_ext) = match op {
                            Binop::Umul => (
                                if a_bv.get_width() < expected_width {
                                    a_bv.uext(expected_width - a_bv.get_width())
                                } else if a_bv.get_width() > expected_width {
                                    a_bv.slice(expected_width - 1, 0)
                                } else {
                                    a_bv.clone()
                                },
                                if b_bv.get_width() < expected_width {
                                    b_bv.uext(expected_width - b_bv.get_width())
                                } else if b_bv.get_width() > expected_width {
                                    b_bv.slice(expected_width - 1, 0)
                                } else {
                                    b_bv.clone()
                                },
                            ),
                            Binop::Smul => (
                                if a_bv.get_width() < expected_width {
                                    a_bv.sext(expected_width - a_bv.get_width())
                                } else if a_bv.get_width() > expected_width {
                                    a_bv.slice(expected_width - 1, 0)
                                } else {
                                    a_bv.clone()
                                },
                                if b_bv.get_width() < expected_width {
                                    b_bv.sext(expected_width - b_bv.get_width())
                                } else if b_bv.get_width() > expected_width {
                                    b_bv.slice(expected_width - 1, 0)
                                } else {
                                    b_bv.clone()
                                },
                            ),
                            _ => unreachable!(),
                        };
                        let prod = a_ext.mul(&b_ext);
                        let prod_width = prod.get_width();
                        if prod_width > expected_width {
                            prod.slice(expected_width - 1, 0)
                        } else if prod_width < expected_width {
                            prod.uext(expected_width - prod_width)
                        } else {
                            prod
                        }
                    }
                    Binop::Umod => {
                        let width = node.ty.bit_count() as u32;
                        let zero = BV::zero(btor.clone(), width);
                        let mod_res = a_bv.urem(b_bv);
                        let rhs_is_zero = b_bv._eq(&zero);
                        // XLS semantics: for unsigned modulus, dividend % 0 == 0.
                        rhs_is_zero.cond_bv(&zero, &mod_res)
                    }
                    Binop::Smod => {
                        let width = node.ty.bit_count() as u32;
                        let zero = BV::zero(btor.clone(), width);
                        let mod_res = a_bv.srem(b_bv);
                        let rhs_is_zero = b_bv._eq(&zero);
                        // XLS semantics: for signed modulus, dividend % 0 == 0.
                        rhs_is_zero.cond_bv(&zero, &mod_res)
                    }
                    Binop::Udiv | Binop::Sdiv => {
                        let btor = a_bv.get_btor();
                        let width = node.ty.bit_count() as u32;
                        let zero = BV::zero(btor.clone(), width);
                        // For signed division the XLS semantics specify:
                        //   if divisor == 0 then result is:
                        //      max_positive  if dividend >= 0
                        //      max_negative  if dividend < 0
                        // For unsigned division the result is all-ones.
                        // Pre-compute these fallback values.  Note that for
                        // width == 1 the "max positive" and "max negative"
                        // are both 1-bit wide and simply 0 and 1 respectively.
                        let all_ones = BV::ones(btor.clone(), width);
                        let (max_pos, max_neg) = if width == 1 {
                            (BV::zero(btor.clone(), 1), BV::one(btor.clone(), 1))
                        } else {
                            let lsb_ones = BV::ones(btor.clone(), width - 1);
                            let zero_msb = BV::zero(btor.clone(), 1);
                            let one_msb = BV::one(btor.clone(), 1);
                            let max_pos = zero_msb.concat(&lsb_ones);
                            let max_neg = one_msb.concat(&BV::zero(btor.clone(), width - 1));
                            (max_pos, max_neg)
                        };
                        let div_res = match op {
                            Binop::Udiv => a_bv.udiv(b_bv),
                            Binop::Sdiv => a_bv.sdiv(b_bv),
                            _ => unreachable!(),
                        };
                        let rhs_is_zero = b_bv._eq(&zero);
                        if matches!(op, Binop::Udiv) {
                            rhs_is_zero.cond_bv(&all_ones, &div_res)
                        } else {
                            // Signed: choose max_neg or max_pos based on sign of dividend.
                            let dividend_neg = a_bv.slice(width - 1, width - 1);
                            let fallback = dividend_neg.cond_bv(&max_neg, &max_pos);
                            rhs_is_zero.cond_bv(&fallback, &div_res)
                        }
                    }
                    Binop::Shll => {
                        let val_bv = env.get(a).expect("Shll arg must be present");
                        let shamt_bv = env.get(b).expect("Shll shamt must be present");
                        let val_width = val_bv.get_width();
                        let shamt_width = shamt_bv.get_width();
                        let btor = val_bv.get_btor();

                        // Determine a comparison width that can represent `val_width`.
                        let mut bits_needed = 0u32;
                        let mut tmp = val_width;
                        while tmp > 0 {
                            bits_needed += 1;
                            tmp >>= 1;
                        }
                        let cmp_width = std::cmp::max(shamt_width, std::cmp::max(bits_needed, 1));

                        // Bring `shamt_bv` to `cmp_width` bits for the comparison.
                        let shamt_cmp = match shamt_width.cmp(&cmp_width) {
                            std::cmp::Ordering::Equal => shamt_bv.clone(),
                            std::cmp::Ordering::Less => shamt_bv.uext(cmp_width - shamt_width),
                            std::cmp::Ordering::Greater => shamt_bv.slice(cmp_width - 1, 0),
                        };

                        let val_width_for_cmp_bv =
                            BV::from_u64(btor.clone(), val_width as u64, cmp_width);
                        let cond_saturate = shamt_cmp.ugte(&val_width_for_cmp_bv);

                        let saturated_val = BV::from_u64(btor.clone(), 0, val_width);
                        let shifted_val_core_logic =
                            shift_boolector(val_bv, shamt_bv, |x, y| x.sll(y));

                        cond_saturate.cond_bv(&saturated_val, &shifted_val_core_logic)
                    }
                    Binop::Shrl => {
                        let val_bv = env.get(a).expect("Shrl arg must be present");
                        let shamt_bv = env.get(b).expect("Shrl shamt must be present");
                        let val_width = val_bv.get_width();
                        let shamt_width = shamt_bv.get_width();
                        let btor = val_bv.get_btor();

                        let mut bits_needed = 0u32;
                        let mut tmp = val_width;
                        while tmp > 0 {
                            bits_needed += 1;
                            tmp >>= 1;
                        }
                        let cmp_width = std::cmp::max(shamt_width, std::cmp::max(bits_needed, 1));

                        let shamt_cmp = match shamt_width.cmp(&cmp_width) {
                            std::cmp::Ordering::Equal => shamt_bv.clone(),
                            std::cmp::Ordering::Less => shamt_bv.uext(cmp_width - shamt_width),
                            std::cmp::Ordering::Greater => shamt_bv.slice(cmp_width - 1, 0),
                        };

                        let val_width_for_cmp_bv =
                            BV::from_u64(btor.clone(), val_width as u64, cmp_width);
                        let cond_saturate = shamt_cmp.ugte(&val_width_for_cmp_bv);

                        let saturated_val = BV::from_u64(btor.clone(), 0, val_width);
                        let shifted_val_core_logic =
                            shift_boolector(val_bv, shamt_bv, |x, y| x.srl(y));

                        cond_saturate.cond_bv(&saturated_val, &shifted_val_core_logic)
                    }
                    Binop::Shra => {
                        let val_bv = env.get(a).expect("Shra arg must be present");
                        let shamt_bv = env.get(b).expect("Shra shamt must be present");
                        let val_width = val_bv.get_width();
                        let shamt_width = shamt_bv.get_width();
                        let btor = val_bv.get_btor();

                        let mut bits_needed = 0u32;
                        let mut tmp = val_width;
                        while tmp > 0 {
                            bits_needed += 1;
                            tmp >>= 1;
                        }
                        let cmp_width = std::cmp::max(shamt_width, std::cmp::max(bits_needed, 1));

                        let shamt_cmp = match shamt_width.cmp(&cmp_width) {
                            std::cmp::Ordering::Equal => shamt_bv.clone(),
                            std::cmp::Ordering::Less => shamt_bv.uext(cmp_width - shamt_width),
                            std::cmp::Ordering::Greater => shamt_bv.slice(cmp_width - 1, 0),
                        };

                        let val_width_for_cmp_bv =
                            BV::from_u64(btor.clone(), val_width as u64, cmp_width);
                        let cond_saturate = shamt_cmp.ugte(&val_width_for_cmp_bv);

                        let saturated_val = if val_width == 0 {
                            BV::from_u64(btor.clone(), 0, 0)
                        } else {
                            val_bv.slice(val_width - 1, val_width - 1).repeat(val_width)
                        };
                        let shifted_val_core_logic = shift_boolector_signed(val_bv, shamt_bv);

                        cond_saturate.cond_bv(&saturated_val, &shifted_val_core_logic)
                    }
                    Binop::Sgt => a_bv.sgt(b_bv),
                    Binop::Slt => a_bv.slt(b_bv),
                    Binop::Sle => a_bv.slte(b_bv),
                    Binop::Sge => a_bv.sgte(b_bv),
                    _ => panic!("Binop {:?} not yet implemented in Boolector conversion", op),
                }
            }
            NodePayload::BitSlice { arg, start, width } => {
                let arg_bv = env.get(arg).expect("BitSlice argument must be present");
                assert!(*width > 0, "BitSlice width must be > 0");
                let high = (*start + *width - 1) as u32;
                let low = *start as u32;
                arg_bv.slice(high, low)
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                // General n-way select with optional default
                assert!(cases.len() > 0, "Sel must have at least one case");
                let sel_bv = env.get(selector).expect("Selector BV must be present");
                let case_bvs: Vec<_> = cases
                    .iter()
                    .map(|c| env.get(c).expect("Case BV must be present"))
                    .collect();
                let width = case_bvs[0].get_width();
                for (_i, case_bv) in case_bvs.iter().enumerate() {
                    assert_eq!(case_bv.get_width(), width, "All case widths must match");
                }
                let sel_width = sel_bv.get_width();
                let mut result = if let Some(default_ref) = default {
                    env.get(default_ref)
                        .expect("Default BV must be present")
                        .clone()
                } else {
                    // If no default, use the last case as the default (mimic array index OOB
                    // behavior)
                    (*case_bvs.last().unwrap()).clone()
                };
                // Build a chain of selects for each possible index value
                for (i, case_bv) in case_bvs.iter().enumerate().rev() {
                    let idx_val = BV::from_u64(btor.clone(), i as u64, sel_width);
                    let is_this = sel_bv._eq(&idx_val);
                    result = is_this.cond_bv(case_bv, &result);
                }
                result
            }
            NodePayload::ZeroExt { arg, new_bit_count } => {
                let arg_bv = env.get(arg).expect("ZeroExt argument must be present");
                let from_width = arg_bv.get_width();
                let to_width = *new_bit_count as u32;
                assert!(
                    to_width >= from_width,
                    "ZeroExt: new_bit_count must be >= argument width"
                );
                arg_bv.uext(to_width - from_width)
            }
            NodePayload::SignExt { arg, new_bit_count } => {
                let arg_bv = env.get(arg).expect("SignExt argument must be present");
                let from_width = arg_bv.get_width();
                let to_width = *new_bit_count as u32;
                assert!(
                    to_width >= from_width,
                    "SignExt: new_bit_count must be >= argument width"
                );
                arg_bv.sext(to_width - from_width)
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                assert!(cases.len() > 0, "PrioritySel must have at least one case");
                let selector_bv = env.get(selector).expect("Selector BV must be present");
                let case_bvs: Vec<_> = cases
                    .iter()
                    .map(|c| env.get(c).expect("Case BV must be present"))
                    .collect();
                let default_bv = if let Some(dref) = default {
                    env.get(dref).expect("Default BV must be present")
                } else {
                    panic!("PrioritySel requires a default value");
                };
                let mut result = default_bv.clone();
                for (i, case_bv) in case_bvs.iter().enumerate().rev() {
                    let bit = selector_bv.slice(i as u32, i as u32);
                    result = bit.cond_bv(case_bv, &result);
                }
                result
            }
            NodePayload::Tuple(elems) => {
                assert!(!elems.is_empty(), "Tuple must have at least one element");
                for nref in elems {
                    log::trace!(
                        "[ir_fn_to_boolector] Tuple element: {:?}, node: {:?}, in env: {}",
                        nref,
                        f.nodes[nref.index],
                        env.contains_key(nref)
                    );
                }
                let mut it = elems.iter();
                let first = env
                    .get(it.next().unwrap())
                    .expect("Tuple element must be present")
                    .clone();
                it.fold(first, |acc, nref| {
                    if !env.contains_key(nref) {
                        log::trace!(
                            "[ir_fn_to_boolector] (fold) Tuple element missing: {:?}, node: {:?}",
                            nref,
                            f.nodes[nref.index]
                        );
                    }
                    let next = env
                        .get(nref)
                        .expect("Tuple element must be present (in fold)");
                    acc.concat(next)
                })
            }
            NodePayload::Array(elems) => {
                assert!(!elems.is_empty(), "Array must have at least one element");
                let mut it = elems.iter().rev();
                let first = env
                    .get(it.next().unwrap())
                    .expect("Array element must be present")
                    .clone();
                it.fold(first, |acc, nref| {
                    let next = env
                        .get(nref)
                        .expect("Array element must be present (in fold)");
                    acc.concat(next)
                })
            }
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds: _assumed_in_bounds,
            } => {
                // Only support single index for now
                assert_eq!(
                    indices.len(),
                    1,
                    "Only single-dimensional array indexing is supported"
                );
                let array_bv = env.get(array).expect("Array BV must be present");
                let index_bv = env.get(&indices[0]).expect("Index BV must be present");
                let array_ty = f.get_node_ty(*array);
                let (element_type, element_count) = match array_ty {
                    crate::xls_ir::ir::Type::Array(arr) => (&arr.element_type, arr.element_count),
                    _ => panic!("ArrayIndex: expected array type"),
                };
                let elem_width = element_type.bit_count() as u32;

                // XLS semantics: If the index is out-of-bounds the last element is returned. We
                // implement this by starting the conditional chain with the *last* element as
                // the default value and then walking the array indices from
                // high to low.
                assert!(
                    element_count > 0,
                    "ArrayIndex: array must have at least one element"
                );
                let mut result: Option<BV<Rc<Btor>>> = None;
                for i in (0..element_count).rev() {
                    let high = ((i + 1) * elem_width as usize - 1) as u32;
                    let low = (i * elem_width as usize) as u32;
                    let elem = array_bv.slice(high, low);
                    if result.is_none() {
                        // First iteration (i == element_count-1): initialise with default (last
                        // element).
                        result = Some(elem);
                        continue;
                    }
                    let idx_val = BV::from_u64(array_bv.get_btor(), i as u64, index_bv.get_width());
                    let is_this = index_bv._eq(&idx_val);
                    // is_this ? elem : acc (acc is the previously accumulated result). We clone
                    // here because BV implements a cheap Rc clone.
                    let acc = result.as_ref().unwrap().clone();
                    result = Some(is_this.cond_bv(&elem, &acc));
                }
                result.expect("ArrayIndex: unable to build result")
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
            } => {
                assert_eq!(
                    indices.len(),
                    1,
                    "Only single-dimensional array updates are supported",
                );
                let array_bv = env.get(array).expect("Array BV must be present");
                let value_bv = env.get(value).expect("Update value BV must be present");
                let index_bv = env.get(&indices[0]).expect("Index BV must be present");
                let array_ty = f.get_node_ty(*array);
                let (elem_ty, elem_count) = match array_ty {
                    crate::xls_ir::ir::Type::Array(arr) => (&arr.element_type, arr.element_count),
                    _ => panic!("ArrayUpdate: expected array type"),
                };
                let elem_width = elem_ty.bit_count() as u32;
                let mut result: Option<BV<Rc<Btor>>> = None;
                for i in (0..elem_count).rev() {
                    let high = ((i + 1) * elem_width as usize - 1) as u32;
                    let low = (i * elem_width as usize) as u32;
                    let orig_elem = array_bv.slice(high, low);
                    let idx_val = BV::from_u64(array_bv.get_btor(), i as u64, index_bv.get_width());
                    let is_this = index_bv._eq(&idx_val);
                    let selected = is_this.cond_bv(value_bv, &orig_elem);
                    // Build the updated array value so that element 0 ends up
                    // in the least-significant slice, matching the layout of
                    // Array literals, Array construction and ArrayIndex.
                    result = Some(if let Some(acc) = result {
                        acc.concat(&selected)
                    } else {
                        selected
                    });
                }
                result.expect("ArrayUpdate: array must have at least one element")
            }
            NodePayload::DynamicBitSlice { arg, start, width } => {
                let arg_bv = env.get(arg).expect("DynamicBitSlice arg must be present");
                let start_bv = env
                    .get(start)
                    .expect("DynamicBitSlice start must be present");

                let arg_width = arg_bv.get_width();

                // Choose index width large enough to represent either start or arg_width-1.
                let needed_for_arg = {
                    let mut bits = 0u32;
                    while (1u32 << bits) <= arg_width {
                        bits += 1;
                    }
                    // ensure non-zero
                    std::cmp::max(bits, 1)
                };
                let idx_width = std::cmp::max(start_bv.get_width() + 1, needed_for_arg);

                let start_ext = if idx_width > start_bv.get_width() {
                    start_bv.uext(idx_width - start_bv.get_width())
                } else {
                    start_bv.clone()
                };

                let mut bits = Vec::with_capacity(*width);
                for i in 0..*width {
                    let btor = arg_bv.get_btor();

                    let idx_const = BV::from_u64(btor.clone(), i as u64, idx_width);
                    let idx = start_ext.add(&idx_const);

                    // in_bounds = idx < arg_width
                    let arg_width_bv = BV::from_u64(btor.clone(), arg_width as u64, idx_width);
                    let in_bounds = idx.ult(&arg_width_bv);

                    // Select the bit at position `idx` if in bounds, else zero.
                    // Boolector lacks variable-index extract, so build a chain of conditional
                    // selects.
                    let mut selected_bit = BV::from_u64(btor.clone(), 0, 1);
                    for j in 0..arg_width {
                        let j_bv = BV::from_u64(btor.clone(), j as u64, idx_width);
                        let is_this = idx._eq(&j_bv);
                        let bit = arg_bv.slice(j, j);
                        selected_bit = is_this.cond_bv(&bit, &selected_bit);
                    }

                    let out_bit =
                        in_bounds.cond_bv(&selected_bit, &BV::from_u64(btor.clone(), 0, 1));
                    bits.push(out_bit);
                }

                // Concatenate bits into a single BV (LSB first): bits[0] is bit0.
                let mut result = bits[0].clone();
                for b in bits.iter().skip(1) {
                    result = b.concat(&result);
                }
                result
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
                    .expect("BitSliceUpdate value must be present");

                // Unify widths to avoid Boolector width-mismatch aborts. The XLS semantics
                // allow `update_value` (M bits) to be either narrower or wider
                // than `operand` (N bits); any portion that falls outside 0..N
                // is simply ignored.  We therefore widen both operands to a
                // common width so the bit-level masking logic works uniformly.
                let arg_width = arg_bv.get_width();
                let upd_width = update_bv.get_width();
                let max_width = arg_width.max(upd_width);

                // Zero-extend operands to the common width.
                let arg_ext = if arg_width < max_width {
                    arg_bv.uext(max_width - arg_width)
                } else {
                    arg_bv.clone()
                };
                let update_ext = if upd_width < max_width {
                    update_bv.uext(max_width - upd_width)
                } else {
                    update_bv.clone()
                };

                // Create a mask of `upd_width` ones and extend to `max_width`.
                // Use BV::ones to support widths > 64 without relying on host integer ranges.
                let ones = BV::ones(btor.clone(), upd_width);
                let ones_ext = if max_width > upd_width {
                    ones.uext(max_width - upd_width)
                } else {
                    ones.clone()
                };

                // Shift mask and update value into position.
                let mask = shift_boolector(&ones_ext, start_bv, |x, y| x.sll(y));
                let update_shifted = shift_boolector(&update_ext, start_bv, |x, y| x.sll(y));

                // Clear the update window in the original arg then OR in shifted update value.
                let cleared = arg_ext.and(&mask.not());
                let inserted = update_shifted.and(&mask);
                let combined = cleared.or(&inserted);

                // Detect start index >= arg_width → no update (spec: ignore out-of-bounds).
                // Build a predicate: start_bv >= arg_width ? 1 : 0.
                let cmp_width = std::cmp::max(start_bv.get_width(), 1);
                let width_const = BV::from_u64(btor.clone(), arg_width as u64, cmp_width);
                let start_ext_for_cmp = match start_bv.get_width().cmp(&cmp_width) {
                    std::cmp::Ordering::Equal => start_bv.clone(),
                    std::cmp::Ordering::Less => start_bv.uext(cmp_width - start_bv.get_width()),
                    std::cmp::Ordering::Greater => start_bv.slice(cmp_width - 1, 0),
                };
                let oob = start_ext_for_cmp.ugte(&width_const); // 1-bit BV

                // Choose arg_bv when out-of-bounds, else combined value.
                let selected = if arg_width == max_width {
                    oob.cond_bv(&arg_bv, &combined)
                } else {
                    // Need operands of same width for cond_bv.
                    let combined_sliced = combined.slice(arg_width - 1, 0);
                    oob.cond_bv(&arg_bv, &combined_sliced)
                };

                selected
            }
            NodePayload::Trace { .. } => {
                // Trace has no effect on value computation
                continue;
            }
            NodePayload::Invoke { .. } => {
                panic!("Invoke not supported in Boolector conversion");
            }
            NodePayload::OneHotSel { selector, cases } => {
                let selector_bv = env.get(selector).expect("Selector BV must be present");
                assert_eq!(selector_bv.get_width() as usize, cases.len());
                let case_bvs: Vec<_> = cases
                    .iter()
                    .map(|c| env.get(c).expect("Case BV must be present"))
                    .collect();
                let width = case_bvs[0].get_width();
                let zero = BV::from_u64(btor.clone(), 0, width);
                let mut result = zero.clone();
                for (i, case) in case_bvs.iter().enumerate() {
                    let bit = selector_bv.slice(i as u32, i as u32);
                    let mask = bit.repeat(width);
                    let masked = mask.and(case);
                    result = result.or(&masked);
                }
                result
            }
            NodePayload::Cover { .. } => {
                // Cover statements do not contribute to value computation
                continue;
            }
            NodePayload::Encode { arg } => {
                let arg_bv = env.get(arg).expect("Encode argument must be present");
                let input_width = arg_bv.get_width(); // N: input bitwidth
                assert!(input_width > 0, "Encode: input_width N must be > 0");

                // M: output bitwidth, taken from the node's type information.
                // XLS IR ensures this is ceil(log2(N)) with a minimum of 1 if N > 0,
                // or 0 if N=0 (though N>0 is asserted above).
                // Specifically, if N=1, M=1 for XLS's typical clog2 definition in types.
                let out_width = node.ty.bit_count() as u32;

                // If out_width is 0, it implies an unusual case (e.g. input_width=0 or IR type
                // mismatch). Given `input_width > 0`, `out_width` should also
                // be > 0 based on XLS semantics for encode. For example,
                // encode(bits[1]) should yield bits[1] (out_width=1).
                // If somehow out_width became 0, the logic would produce a 0-width BV.
                assert!(
                    out_width > 0 || (input_width == 1 && out_width == 0),
                    "Encode: output_width M should generally be > 0 if input_width N > 0. Found M={} for N={}",
                    out_width,
                    input_width
                );

                let btor_ctx = arg_bv.get_btor();

                if out_width == 0 {
                    // This case should ideally not be hit if input_width > 0 and IR types are
                    // consistent with XLS spec (e.g. encode(bits[1]) ->
                    // bits[1]). However, if node.ty.bit_count() is indeed 0,
                    // return a 0-width BV.
                    BV::from_u64(btor_ctx, 0, 0)
                } else {
                    let mut output_bit_bvs = Vec::with_capacity(out_width as usize);

                    // For each output bit j (from 0 for LSB to out_width-1 for MSB)
                    for j in 0..out_width {
                        let mut or_candidates_for_output_bit_j: Vec<BV<Rc<Btor>>> = Vec::new();
                        // For each input bit position i (from 0 to input_width-1)
                        for i in 0..input_width {
                            // Check if the j-th bit of the *index i* is 1.
                            // (i as u64 >> j) extracts the j-th bit of i.
                            if ((i as u64 >> j) & 1) != 0 {
                                // If so, the input_bit[i] (i.e., arg_bv.slice(i,i))
                                // contributes to this output_bit[j].
                                or_candidates_for_output_bit_j.push(arg_bv.slice(i, i));
                            }
                        }

                        let current_output_bit_val = if or_candidates_for_output_bit_j.is_empty() {
                            // If no input bits contribute, this output bit is 0.
                            BV::from_u64(btor_ctx.clone(), 0, 1)
                        } else {
                            // OR all contributing input bits.
                            let mut or_op_result = or_candidates_for_output_bit_j[0].clone();
                            for k_idx in 1..or_candidates_for_output_bit_j.len() {
                                or_op_result =
                                    or_op_result.or(&or_candidates_for_output_bit_j[k_idx]);
                            }
                            or_op_result
                        };
                        output_bit_bvs.push(current_output_bit_val);
                    }

                    // Concatenate the calculated output bits to form the final result.
                    // output_bit_bvs[0] is the LSB, output_bit_bvs[out_width-1] is the MSB.
                    // Boolector's concat is msb.concat(lsb_aggregate).
                    let mut final_result_bv = output_bit_bvs[0].clone(); // Start with LSB
                    for k_idx in 1..(out_width as usize) {
                        final_result_bv = output_bit_bvs[k_idx].concat(&final_result_bv);
                    }
                    final_result_bv
                }
            }
            NodePayload::AfterAll(_) => {
                // AfterAll is a no-op for Boolector; do not insert a BV (like Nil)
                continue;
            }
            NodePayload::Assert { .. } => {
                // TODO: Turn Assert into a proof objective in Boolector.
                // For now, treat as a no-op (like Nil/AfterAll).
                continue;
            }
            NodePayload::Decode { arg, width } => {
                let arg_bv = env.get(arg).expect("Decode arg must be present");
                let in_width = arg_bv.get_width() as usize;
                let out_width = *width;
                assert!(
                    out_width <= (1 << in_width),
                    "Decode output width must be <= 2^input_width"
                );
                // For each output bit i, set to 1 iff arg_bv == i, else 0
                let mut bits = Vec::with_capacity(out_width);
                for i in 0..out_width {
                    let i_bv = BV::from_u64(btor.clone(), i as u64, in_width as u32);
                    let eq = arg_bv._eq(&i_bv);
                    // eq is 1-bit; extend to 1-bit BV
                    bits.push(eq);
                }
                // Concatenate bits into a single BV (LSB first)
                bits.reverse();
                let mut out = bits[0].clone();
                for b in bits.iter().skip(1) {
                    out = out.concat(b);
                }
                out
            }
        };
        // Assert the width matches the node's annotated type
        let expected_width = node.ty.bit_count() as u32;
        let actual_width = bv.get_width();
        assert_eq!(
            actual_width, expected_width,
            "Boolector BV width {} does not match node type width {} for node {:?}",
            actual_width, expected_width, node
        );
        env.insert(node_ref, bv);
    }
    let ret_node_ref = f.ret_node_ref.expect("Function must have a return node");
    if !env.contains_key(&ret_node_ref) {
        let ret_node = &f.nodes[ret_node_ref.index];
        log::trace!(
            "[ir_fn_to_boolector] Return node not found in env! ret_node_ref: {:?}, node: {:?}",
            ret_node_ref,
            ret_node
        );
        log::trace!(
            "[ir_fn_to_boolector] env keys: {:?}",
            env.keys().collect::<Vec<_>>()
        );
    }
    if let Some(bv) = env.remove(&ret_node_ref) {
        IrFnBoolectorResult { output: bv, inputs }
    } else {
        panic!(
            "Cannot return Nil node from ir_fn_to_boolector: Boolector does not support 0-width bitvectors"
        );
    }
}

/// Result of equivalence checking.
#[derive(Debug, PartialEq, Eq)]
pub enum EquivResult {
    Proved,
    /// Counterexample with inputs that distinguish the two functions,
    /// along with the corresponding outputs from the lhs and rhs functions.
    Disproved {
        /// Counterexample input assignments, reconstructed as structured IR
        /// values.
        inputs: Vec<IrValue>,
        /// Counterexample outputs for lhs and rhs functions, reconstructed as
        /// structured IR values.
        outputs: (IrValue, IrValue),
    },
}

/// Helper to flatten a BV output (tuple) to a single BV
fn flatten_bv(bv: &BV<Rc<Btor>>, ty: &crate::xls_ir::ir::Type) -> BV<Rc<Btor>> {
    match ty {
        crate::xls_ir::ir::Type::Bits(_) => bv.clone(),
        crate::xls_ir::ir::Type::Tuple(members) => {
            let mut offset = 0;
            let mut result = None;
            for member in members.iter().rev() {
                let width = member.bit_count() as u32;
                let slice = bv.slice(offset + width - 1, offset);
                offset += width;
                result = Some(if let Some(acc) = result {
                    slice.concat(&acc)
                } else {
                    slice
                });
            }
            result.expect("Tuple must have at least one member")
        }
        crate::xls_ir::ir::Type::Array(arr) => {
            let mut offset = 0;
            let mut result = None;
            for _ in 0..arr.element_count {
                let width = arr.element_type.bit_count() as u32;
                let slice = bv.slice(offset + width - 1, offset);
                offset += width;
                result = Some(if let Some(acc) = result {
                    slice.concat(&acc)
                } else {
                    slice
                });
            }
            result.expect("Array must have at least one element")
        }
        _ => unimplemented!("flatten_bv for {:?}", ty),
    }
}

/// Helper to flatten a type to a single bit width (tuples, arrays, bits)
pub fn flatten_type(ty: &crate::xls_ir::ir::Type) -> usize {
    match ty {
        crate::xls_ir::ir::Type::Bits(width) => *width,
        crate::xls_ir::ir::Type::Tuple(members) => members.iter().map(|t| flatten_type(&**t)).sum(),
        crate::xls_ir::ir::Type::Array(arr) => flatten_type(&arr.element_type) * arr.element_count,
        _ => unimplemented!("flatten_type for {:?}", ty),
    }
}

// Helper: convert a Boolector BV model solution to IrBits in LSB-first (bit0 is
// LSB) order.
fn bv_solution_to_ir_bits(bv: &BV<Rc<Btor>>) -> IrBits {
    // Retrieve the concrete value from the solver model.
    let width = bv.get_width() as usize;
    let solution = bv.get_a_solution();
    let disamb = solution.disambiguate();
    let bitstr = disamb.as_01x_str();
    let bits: Vec<bool> = bitstr.chars().rev().map(|c| c == '1').collect();
    if bits.len() != width {
        log::trace!(
            "[bv_solution_to_ir_bits] Solution width mismatch: expected {}, got {}",
            width,
            bits.len()
        );
    }
    crate::ir_value_utils::ir_bits_from_lsb_is_0(&bits)
}

/// Helper: convert a Boolector BV model solution to a typed IRValue using the
/// given IR type.
fn bv_solution_to_ir_value(bv: &BV<Rc<Btor>>, ty: &Type) -> IrValue {
    let bits = bv_solution_to_ir_bits(bv);
    ir_value_from_bits_with_type(&bits, ty)
}

/// Checks equivalence of two IR functions using Boolector.
/// Only supports literal-only, zero-parameter functions for now.
fn check_equiv_internal_with_btor(
    lhs: &Fn,
    rhs: &Fn,
    flatten_aggregates: bool,
    ctx: &Ctx,
    use_frame: bool, // whether to push/pop a solver frame
) -> EquivResult {
    // Helper to pretty-print a function signature
    fn signature_str(f: &Fn) -> String {
        let params = f
            .params
            .iter()
            .map(|p| format!("{}: {:?}", p.name, p.ty))
            .collect::<Vec<_>>()
            .join(", ");
        format!("fn {}({}) -> {:?}", f.name, params, f.ret_ty)
    }
    log::info!("LHS signature: {}", signature_str(lhs));
    log::info!("RHS signature: {}", signature_str(rhs));
    if !flatten_aggregates {
        // Not flattened, so we need to make sure that return types and parameter lists
        // are identical
        assert_eq!(lhs.ret_ty, rhs.ret_ty, "Return types must match");
        assert_eq!(
            lhs.params.len(),
            rhs.params.len(),
            "Parameter count mismatch"
        );
        for (l, r) in lhs.params.iter().zip(rhs.params.iter()) {
            assert_eq!(
                l.ty, r.ty,
                "Parameter type mismatch for {} vs {}: {:?} vs {:?}",
                l.name, r.name, l.ty, r.ty
            );
        }
    }
    if use_frame {
        ctx.btor.push(1);
    }
    let lhs_result = ir_fn_to_boolector(ctx.btor.clone(), lhs, Some(&ctx.lhs_params));
    let rhs_result = ir_fn_to_boolector(ctx.btor.clone(), rhs, Some(&ctx.rhs_params));
    // Flatten outputs if needed
    let lhs_out = if flatten_aggregates {
        flatten_bv(&lhs_result.output, &lhs.ret_ty)
    } else {
        lhs_result.output.clone()
    };
    let rhs_out = if flatten_aggregates {
        flatten_bv(&rhs_result.output, &rhs.ret_ty)
    } else {
        rhs_result.output.clone()
    };
    // Assert that outputs are not equal (look for a counterexample)
    let diff = lhs_out._ne(&rhs_out);
    diff.assert();
    let sat_result = ctx.btor.sat();
    let res = match sat_result {
        SolverResult::Unsat => EquivResult::Proved,
        SolverResult::Sat => {
            // Extract input assignments from the model and reconstruct typed IR values.
            let mut counterexample = Vec::new();
            for param in &lhs.params {
                let bv = ctx.lhs_params.get(&param.name).unwrap();
                counterexample.push(bv_solution_to_ir_value(bv, &param.ty));
            }
            // Extract outputs for lhs and rhs and reconstruct typed IR values.
            let lhs_val = bv_solution_to_ir_value(&lhs_out, &lhs.ret_ty);
            let rhs_val = bv_solution_to_ir_value(&rhs_out, &lhs.ret_ty);
            EquivResult::Disproved {
                inputs: counterexample,
                outputs: (lhs_val, rhs_val),
            }
        }
        SolverResult::Unknown => panic!("Solver returned unknown result"),
    };
    if use_frame {
        ctx.btor.pop(1);
    }
    res
}

/// Standard equivalence check (no aggregate flattening)
pub fn prove_ir_fn_equiv(lhs: &Fn, rhs: &Fn, flatten_aggregates: bool) -> EquivResult {
    let ctx = Ctx::new(lhs, rhs);
    check_equiv_internal_with_btor(lhs, rhs, flatten_aggregates, &ctx, false)
}

/// Equivalence check reusing the given solver context.
pub fn prove_ir_fn_equiv_with_ctx(
    lhs: &Fn,
    rhs: &Fn,
    flatten_aggregates: bool,
    ctx: &Ctx,
) -> EquivResult {
    check_equiv_internal_with_btor(lhs, rhs, flatten_aggregates, ctx, true)
}

fn shift_boolector<F>(val: &BV<Rc<Btor>>, shamt: &BV<Rc<Btor>>, shift_op: F) -> BV<Rc<Btor>>
where
    F: std::ops::Fn(&BV<Rc<Btor>>, &BV<Rc<Btor>>) -> BV<Rc<Btor>>,
{
    let orig_width = val.get_width();
    // Find the smallest power of two >= orig_width
    let mut pow2 = 1;
    while pow2 < orig_width {
        pow2 *= 2;
    }
    let mut k = (pow2 as f64).log2() as u32;
    if k == 0 {
        // Boolector does not support zero-width bitvectors for shift amounts.
        // Use a minimum width of one to avoid invalid slice operations when the
        // value being shifted is a single bit.
        k = 1;
    }
    // Zero-extend val to pow2 bits if needed
    let val_pow2 = if pow2 == orig_width {
        val.clone()
    } else {
        val.uext(pow2 - orig_width)
    };
    // Adjust shamt to k bits
    let shamt_k = match shamt.get_width().cmp(&k) {
        std::cmp::Ordering::Equal => shamt.clone(),
        std::cmp::Ordering::Less => shamt.uext(k - shamt.get_width()),
        std::cmp::Ordering::Greater => shamt.slice(k - 1, 0),
    };
    let shifted = shift_op(&val_pow2, &shamt_k);
    // Slice result back to original width
    shifted.slice(orig_width - 1, 0)
}

fn irbits_to_binary_str(bits: &xlsynth::IrBits) -> String {
    let width = bits.get_bit_count();
    (0..width)
        .rev() // Boolector expects MSB first in the string
        .map(|i| if bits.get_bit(i).unwrap() { '1' } else { '0' })
        .collect()
}

fn ir_value_to_bv(
    btor: Rc<Btor>,
    ir_value: &xlsynth::IrValue,
    ty: &crate::xls_ir::ir::Type,
) -> BV<Rc<Btor>> {
    match ty {
        crate::xls_ir::ir::Type::Bits(width) => {
            let bits = ir_value.to_bits().expect("Bits literal must be bits");
            if *width <= 64 {
                let mut value = 0u64;
                for i in 0..*width {
                    if bits.get_bit(i).unwrap() {
                        value |= 1 << i;
                    }
                }
                BV::from_u64(btor, value, *width as u32)
            } else {
                let bin_str = irbits_to_binary_str(&bits);
                BV::from_binary_str(btor, &bin_str)
            }
        }
        crate::xls_ir::ir::Type::Array(array_ty) => {
            let elements = ir_value
                .get_elements()
                .expect("Array literal must have elements");
            let mut bvs = Vec::new();
            for elem in elements.iter().rev() {
                // LSB = last element
                let bv = ir_value_to_bv(btor.clone(), elem, &array_ty.element_type);
                bvs.push(bv);
            }
            let mut result = bvs[0].clone();
            for bv in bvs.iter().skip(1) {
                result = result.concat(bv);
            }
            result
        }
        crate::xls_ir::ir::Type::Tuple(types) => {
            // Keep element-0 as the most-significant slice to match the
            // ordering used by the `tuple` IR node and by
            // `tuple_get_flat_bit_slice_for_index`.
            let elements = ir_value
                .get_elements()
                .expect("Tuple literal must have elements");
            assert_eq!(elements.len(), types.len());

            let mut bvs = Vec::with_capacity(elements.len());
            // Iterate in *forward* order so that element-0 is processed first
            // and will end up in the MSB position after successive concats.
            for (elem, elem_ty) in elements.iter().zip(types.iter()) {
                let bv = ir_value_to_bv(btor.clone(), elem, elem_ty);
                bvs.push(bv);
            }

            // Concatenate, keeping accumulated value as MSBs.
            let mut result = bvs[0].clone();
            for bv in bvs.iter().skip(1) {
                result = result.concat(bv);
            }
            result
        }
        crate::xls_ir::ir::Type::Token => {
            // Tokens are zero bits
            BV::from_u64(btor, 0, 0)
        }
    }
}

// Like `shift_boolector` but sign-extends `val` when it needs to grow to the
// next power-of-two width.  This is required for arithmetic-right shifts so the
// sign bit is preserved even after the temporary width increase.
fn shift_boolector_signed(val: &BV<Rc<Btor>>, shamt: &BV<Rc<Btor>>) -> BV<Rc<Btor>> {
    let orig_width = val.get_width();
    // Smallest power-of-two ≥ orig_width.
    let mut pow2 = 1;
    while pow2 < orig_width {
        pow2 *= 2;
    }
    let mut k = (pow2 as f64).log2() as u32;
    if k == 0 {
        k = 1; // Boolector requires at least 1-bit shift amount.
    }

    // Sign-extend `val` to `pow2` bits if necessary so the sign bit occupies
    // the high bit of the temporary vector.
    let val_pow2 = if pow2 == orig_width {
        val.clone()
    } else {
        val.sext(pow2 - orig_width)
    };

    // Bring `shamt` to `k` bits.
    let shamt_k = match shamt.get_width().cmp(&k) {
        std::cmp::Ordering::Equal => shamt.clone(),
        std::cmp::Ordering::Less => shamt.uext(k - shamt.get_width()),
        std::cmp::Ordering::Greater => shamt.slice(k - 1, 0),
    };

    let shifted = val_pow2.sra(&shamt_k);
    // Slice back to original width.
    shifted.slice(orig_width - 1, 0)
}

// Uses a parallel strategy to prove equivalence by splitting on each output
// bit.
pub fn prove_ir_fn_equiv_output_bits_parallel(
    lhs: &Fn,
    rhs: &Fn,
    flatten_aggregates: bool,
) -> EquivResult {
    // Ensure both functions return the same bit width.
    let width = lhs.ret_ty.bit_count();
    assert_eq!(width, rhs.ret_ty.bit_count(), "Return widths must match");

    // Helper to create a variant of `f` that returns just a single bit of the
    // original return value at position `bit`.
    fn make_bit_fn(f: &Fn, bit: usize) -> Fn {
        use crate::xls_ir::ir::{Node, NodePayload, NodeRef, Type};

        let mut nf = f.clone();
        let ret = nf.ret_node_ref.expect("ret node");
        let slice_ref = NodeRef {
            index: nf.nodes.len(),
        };
        nf.nodes.push(Node {
            text_id: nf.nodes.len(),
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::BitSlice {
                arg: ret,
                start: bit,
                width: 1,
            },
            pos: None,
        });
        nf.ret_node_ref = Some(slice_ref);
        nf.ret_ty = Type::Bits(1);
        nf
    }

    let found = Arc::new(AtomicBool::new(false));
    let cex: Arc<Mutex<Option<(Vec<IrValue>, (IrValue, IrValue))>>> = Arc::new(Mutex::new(None));
    let next = Arc::new(AtomicUsize::new(0));
    let threads = num_cpus::get();
    let mut handles = Vec::new();

    for thread_no in 0..std::cmp::min(width, threads) {
        let lhs_cl = lhs.clone();
        let rhs_cl = rhs.clone();
        let found_cl = found.clone();
        let cex_cl = cex.clone();
        let next_cl = next.clone();
        let handle = std::thread::spawn(move || {
            loop {
                if found_cl.load(Ordering::SeqCst) {
                    break;
                }
                let i = next_cl.fetch_add(1, Ordering::SeqCst);
                if i >= width {
                    break;
                }
                log::debug!("thread {} checking bit {}", thread_no, i);
                let lf = make_bit_fn(&lhs_cl, i);
                let rf = make_bit_fn(&rhs_cl, i);
                // Use flattened equivalence for the per-bit functions regardless of the
                // caller-supplied setting. Each per-bit variant always returns a
                // single bits value, but the slice is taken out of a (potentially
                // aggregate) original return value. Flattening guarantees
                // consistent bit ordering between the two functions.
                let res = prove_ir_fn_equiv(&lf, &rf, flatten_aggregates);
                log::debug!("thread {} checking bit {} result: {:?}", thread_no, i, res);
                if let EquivResult::Disproved { inputs, outputs } = res {
                    // Inputs and outputs are already typed IR values.
                    found_cl.store(true, Ordering::SeqCst);
                    *cex_cl.lock().unwrap() = Some((inputs, outputs));
                    break;
                }
            }
        });
        handles.push(handle);
    }

    for (i, h) in handles.into_iter().enumerate() {
        log::debug!("joining on handle {}", i);
        h.join().unwrap();
    }

    let maybe_cex = {
        let mut guard = cex.lock().unwrap();
        guard.take()
    };

    if let Some((inputs, outputs)) = maybe_cex {
        EquivResult::Disproved { inputs, outputs }
    } else {
        EquivResult::Proved
    }
}

/// Prove equivalence by case-splitting on a single input bit (0 / 1) chosen by
/// `split_input_index` and `split_input_bit_index`.
///
/// TODO: pick the maximal-fan-out bit in a wrapper.
/// TODO: divide-and-conquer dynamically on more and more bits.
pub fn prove_ir_fn_equiv_split_input_bit(
    lhs: &Fn,
    rhs: &Fn,
    split_input_index: usize,
    split_input_bit_index: usize,
    flatten_aggregates: bool,
) -> EquivResult {
    // If there are no parameters, fall back to the standard prover. No need to
    // panic here.
    if lhs.params.is_empty() || rhs.params.is_empty() {
        return prove_ir_fn_equiv(lhs, rhs, flatten_aggregates);
    }
    assert_eq!(
        lhs.params.len(),
        rhs.params.len(),
        "Parameter count mismatch"
    );
    assert!(
        split_input_index < lhs.params.len(),
        "split_input_index out of bounds, num params: {}, split_input_index: {}",
        lhs.params.len(),
        split_input_index
    );
    assert!(
        split_input_bit_index < lhs.params[split_input_index].ty.bit_count(),
        "split_input_bit_index out of bounds, param width: {}, split_input_bit_index: {}",
        lhs.params[split_input_index].ty.bit_count(),
        split_input_bit_index
    );

    let split_param = &lhs.params[split_input_index];
    let split_width = split_param.ty.bit_count() as u32;

    log::info!(
        "[input-bit-split] Splitting on parameter '{}' (width {}), bit index {}",
        split_param.name,
        split_width,
        split_input_bit_index
    );

    let ctx = Ctx::new(lhs, rhs);

    // Helper closure to run one branch under an assumption.
    let run_branch = |bit_value: u64| -> EquivResult {
        ctx.btor.push(1);
        let bit_bv = ctx
            .flattened_params
            .as_ref()
            .unwrap()
            .slice(split_input_bit_index as u32, split_input_bit_index as u32);
        let val_bv = BV::from_u64(ctx.btor.clone(), bit_value, 1);
        // In the discussion we said assume, but here we actually use assert as we are
        // proving unsatisfiability. Let F be the formula we are proving
        // unsatisfiability of. We want to make sure that
        // 1. !F /\ (x = 0)
        // 2. !F /\ (x = 1)
        // are both unsatisfiable.
        bit_bv._eq(&val_bv).assert();

        let res = check_equiv_internal_with_btor(
            lhs,
            rhs,
            flatten_aggregates,
            &ctx,
            /* use_frame= */ false,
        );
        ctx.btor.pop(1);
        res
    };

    let res0 = run_branch(0);
    if let EquivResult::Disproved { .. } = res0 {
        return res0;
    }
    let res1 = run_branch(1);
    if let EquivResult::Disproved { .. } = res1 {
        return res1;
    }
    EquivResult::Proved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{Opt, load_bf16_add_sample, load_bf16_mul_sample};
    use crate::xls_ir::ir::{ArrayTypeData, Type};
    use crate::xls_ir::ir_parser;
    use boolector::Btor;
    use std::rc::Rc;

    /// Asserts that the given IR function (as text) is equivalent to itself.
    fn assert_fn_equiv_to_self(ir_text: &str) {
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR");
        let result = prove_ir_fn_equiv(&f, &f, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Function was not equivalent to itself: {:?}",
            result
        );
    }

    #[test]
    fn test_reverse_equiv_to_self() {
        let ir_text = r#"fn reverse4(x: bits[4] id=1) -> bits[4] {
  ret reverse.2: bits[4] = reverse(x, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_onehot_equiv_to_self() {
        let ir_text = r#"fn onehot3(x: bits[3] id=1) -> bits[4] {
  ret one_hot.2: bits[4] = one_hot(x, lsb_prio=true, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_encode_equiv_to_self() {
        let ir_text = r#"fn encode4(x: bits[4] id=1) -> bits[2] {
  ret encode.2: bits[2] = encode(x, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_sgt_equiv_to_self() {
        let ir_text = r#"fn sgt4(x: bits[4] id=1, y: bits[4] id=2) -> bits[1] {
  ret sgt.3: bits[1] = sgt(x, y, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_equiv_different_param_names() {
        let lhs_ir = r#"fn lhs(x: bits[8] id=1) -> bits[8] {
  ret sum: bits[8] = add(x, x, id=2)
}"#;
        let rhs_ir = r#"fn rhs(y: bits[8] id=1) -> bits[8] {
  ret sum: bits[8] = add(y, y, id=2)
}"#;

        let lhs = ir_parser::Parser::new(lhs_ir).parse_fn().unwrap();
        let rhs = ir_parser::Parser::new(rhs_ir).parse_fn().unwrap();

        assert_eq!(prove_ir_fn_equiv(&lhs, &rhs, false), EquivResult::Proved);
    }

    #[test]
    fn test_slt_equiv_to_self() {
        let ir_text = r#"fn slt4(x: bits[4] id=1, y: bits[4] id=2) -> bits[1] {
  ret slt.3: bits[1] = slt(x, y, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_sle_equiv_to_self() {
        let ir_text = r#"fn sle4(x: bits[4] id=1, y: bits[4] id=2) -> bits[1] {
  ret sle.3: bits[1] = sle(x, y, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_check_equiv_counterexample_for_x_eq_42() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ir_text_f = r#"fn f(x: bits[8] id=1) -> bits[1] {
  literal.2: bits[8] = literal(value=42, id=2)
  ret eq.3: bits[1] = eq(x, literal.2, id=3)
}"#;
        let ir_text_g = r#"fn g(x: bits[8] id=1) -> bits[1] {
  ret false.2: bits[1] = literal(value=0, id=2)
}"#;
        let f = ir_parser::Parser::new(ir_text_f).parse_fn().unwrap();
        let g = ir_parser::Parser::new(ir_text_g).parse_fn().unwrap();
        let result = prove_ir_fn_equiv(&f, &g, false);
        match result {
            EquivResult::Disproved {
                inputs: ref cex,
                outputs: _,
            } => {
                assert_eq!(cex.len(), 1);
                let val = &cex[0];
                assert_eq!(val.bit_count().unwrap(), 8);
                // Should be 42
                assert_eq!(val.to_u64().unwrap(), 42);
            }
            _ => panic!("Expected Disproved with counterexample"),
        }
    }

    #[test]
    fn test_nand_equiv_to_self() {
        let ir_text = r#"fn nand4(x: bits[4] id=1, y: bits[4] id=2) -> bits[4] {
  ret nand.3: bits[4] = nand(x, y, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_afterall_noop() {
        let ir_text = r#"fn afterall_noop(x: bits[1] id=1) -> bits[1] {
  afterall.2: token = after_all()
  ret identity.3: bits[1] = identity(x, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_bf16_add_sample_to_boolector() {
        let sample = load_bf16_add_sample(Opt::No);
        let g8r_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let bv = ir_fn_to_boolector(btor, g8r_fn, None);
        // bf16 is 16 bits
        assert_eq!(bv.output.get_width(), 16);
    }

    #[test]
    fn test_bf16_mul_sample_to_boolector() {
        let sample = load_bf16_mul_sample(Opt::No);
        let g8r_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let bv = ir_fn_to_boolector(btor, g8r_fn, None);
        assert_eq!(bv.output.get_width(), 16);
    }

    #[test]
    fn test_reverse_unop() {
        let ir_text = r#"fn reverse4(x: bits[4] id=1) -> bits[4] {
  ret reverse.2: bits[4] = reverse(x, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_onehot_unop() {
        let ir_text = r#"fn onehot3(x: bits[3] id=1) -> bits[4] {
  ret one_hot.2: bits[4] = one_hot(x, lsb_prio=true, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_umod_equiv_to_self() {
        let ir_text = r#"fn umod4(x: bits[4] id=1, y: bits[4] id=2) -> bits[4] {
  ret umod.3: bits[4] = umod(x, y, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_umod_known_case() {
        let ir_text = r#"fn umod_known(x: bits[4] id=1) -> bits[4] {
  literal.2: bits[4] = literal(value=5, id=2)
  ret umod.3: bits[4] = umod(x, literal.2, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR");
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let mut param_bvs = std::collections::HashMap::new();
        // x = 13
        let x_val = 13u64;
        let x_bv = BV::from_u64(btor.clone(), x_val, 4);
        param_bvs.insert("x".to_string(), x_bv);
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        let sat_result = btor.sat();
        assert_eq!(
            sat_result,
            boolector::SolverResult::Sat,
            "Expected SAT before retrieving model, got {:?}",
            sat_result
        );
        let out = result.output.get_a_solution().as_u64().unwrap();
        // 13 mod 5 = 3
        assert_eq!(out, 3);
    }

    #[test]
    fn test_udiv_umod_recompose_equiv_identity() {
        // This test checks that for all x, y (y != 0), the following holds:
        //   let q = x / y;
        //   let r = x % y;
        //   x == q * y + r
        // We encode this as two functions and check equivalence:
        //   (1) identity(x, y) = x
        //   (2) recompose(x, y) = (x / y) * y + (x % y)
        // We restrict y != 0 by using a 3-bit y and adding 1, so y in [1,8].

        let ir_identity = r#"fn identity(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret identity.3: bits[8] = identity(x, id=3)
}"#;

        let ir_recompose = r#"fn recompose(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  one: bits[8] = literal(value=1, id=100)
  y1: bits[8] = or(y, one, id=101)
  q: bits[8] = udiv(x, y1, id=3)
  r: bits[8] = umod(x, y1, id=4)
  prod: bits[8] = umul(q, y1, id=5)
  ret sum: bits[8] = add(prod, r, id=6)
}"#;

        // Use the existing assert_fn_equiv helper to check equivalence
        let f = ir_parser::Parser::new(ir_identity).parse_fn().unwrap();
        let g = ir_parser::Parser::new(ir_recompose).parse_fn().unwrap();
        let result = prove_ir_fn_equiv(&f, &g, false);
        match result {
            EquivResult::Proved => (),
            EquivResult::Disproved {
                inputs: _,
                outputs: _,
            } => panic!("Expected Proved, got Disproved"),
        }
    }

    #[test]
    fn test_non_bits_literal_panics() {
        // This test previously expected a panic for array literals, but now arrays are
        // supported. Instead, we test that an array literal is accepted and
        // produces the expected result.
        let ir_text = r#"fn f() -> bits[4][2] {
  ret literal.1: bits[4][2] = literal(value=[1, 2], id=1)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let result = ir_fn_to_boolector(btor.clone(), &f, None);
        // The expected bit pattern is [1,2] as two 4-bit values, LSB = last element (2)
        // 1 = 0b0001, 2 = 0b0010, so bits = 0b00100001 (LSB = 1, next 4 bits = 2)
        let out = result.output.as_u64().unwrap();
        assert_eq!(out, 0b00100001);
    }

    #[test]
    fn test_array_literal_to_bv() {
        let ir_text = r#"fn f() -> bits[4][2] {
  ret literal.1: bits[4][2] = literal(value=[1, 2], id=1)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let result = ir_fn_to_boolector(btor.clone(), &f, None);
        // The expected bit pattern is [1,2] as two 4-bit values, LSB = last element (2)
        // 1 = 0b0001, 2 = 0b0010, so bits = 0b00100001 (LSB = 1, next 4 bits = 2)
        let out = result.output.as_u64().unwrap();
        assert_eq!(out, 0b00100001);
    }

    #[test]
    fn test_large_bits_literal_to_bv() {
        let ir_text = r#"fn f() -> bits[80] {
  ret literal.1: bits[80] = literal(value=1208925819614629174706175, id=1)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let result = ir_fn_to_boolector(btor.clone(), &f, None);
        // 1208925819614629174706175 = 0xFFFFFFFFFFFFFFFFFFF (80 bits, all ones)
        let out_str = result.output.as_binary_str().unwrap();
        let expected = "1".repeat(80);
        assert_eq!(out_str, expected);
    }

    #[test]
    fn test_decode_equiv_to_self() {
        let ir_text = r#"fn decode4(x: bits[2] id=1) -> bits[4] {
  ret decode.2: bits[4] = decode(x, width=4, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_decode_known_case() {
        let ir_text = r#"fn decode4(x: bits[2] id=1) -> bits[4] {
  ret decode.2: bits[4] = decode(x, width=4, id=2)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        // x = 2
        let x_bv = BV::from_u64(btor.clone(), 2, 2);
        let mut param_bvs = std::collections::HashMap::new();
        param_bvs.insert("x".to_string(), x_bv);
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        let out = result.output.as_u64().unwrap();
        // decode(2) for width=4 is 0b0100 = 4
        assert_eq!(out, 0b0100);
    }

    #[test]
    fn test_sge_equiv_to_self() {
        let ir_text = r#"fn sge4(x: bits[4] id=1, y: bits[4] id=2) -> bits[1] {
  ret sge.3: bits[1] = sge(x, y, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_flatten_type_equivalence_flat_vs_aggregate() {
        // Flat bits[256]
        let flat = Type::Bits(256);

        // Array of 32 tuples, each tuple is (bits[1], bits[4], bits[3])
        let tuple = Type::Tuple(vec![
            Box::new(Type::Bits(1)),
            Box::new(Type::Bits(4)),
            Box::new(Type::Bits(3)),
        ]);
        let array = Type::Array(ArrayTypeData {
            element_type: Box::new(tuple),
            element_count: 32,
        });

        let flat_result = flatten_type(&flat);
        let array_result = flatten_type(&array);

        assert_eq!(flat_result, 256);
        assert_eq!(array_result, 256);
    }

    #[test]
    fn test_shift_single_bit_equiv_to_self() {
        let ir_text = r#"fn shl1(x: bits[1] id=1, s: bits[1] id=2) -> bits[1] {
  ret shll.3: bits[1] = shll(x, s, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_zero_ext_equiv_to_self() {
        let ir_text = r#"fn zext4(x: bits[4] id=1) -> bits[8] {
  ret zero_ext.2: bits[8] = zero_ext(x, new_bit_count=8, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_zero_ext_known_case() {
        let ir_text = r#"fn zext_const() -> bits[8] {
  literal.1: bits[4] = literal(value=10, id=1)
  ret zero_ext.2: bits[8] = zero_ext(literal.1, new_bit_count=8, id=2)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let result = ir_fn_to_boolector(btor.clone(), &f, None);
        let sat_result = btor.sat();
        assert_eq!(sat_result, boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        assert_eq!(out, 10);
    }

    #[test]
    fn test_shift_single_bit_known_case() {
        let ir_text = r#"fn shl1_const(x: bits[1] id=1) -> bits[1] {
  one: bits[1] = literal(value=1, id=2)
  ret shll.3: bits[1] = shll(x, one, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let mut param_bvs = std::collections::HashMap::new();
        // x = 1
        let x_bv = BV::from_u64(btor.clone(), 1, 1);
        param_bvs.insert("x".to_string(), x_bv);
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        let sat_result = btor.sat();
        assert_eq!(sat_result, boolector::SolverResult::Sat);
        // 1 shifted left by 1 with 1-bit width should produce 0
        let out = result.output.get_a_solution().as_u64().unwrap();
        assert_eq!(out, 0);
    }

    #[test]
    fn test_udiv_by_zero_returns_ones() {
        let ir_text = r#"fn f(x: bits[4] id=1) -> bits[4] {
  zero: bits[4] = literal(value=0, id=2)
  ret q: bits[4] = udiv(x, zero, id=3)
 }"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let mut param_bvs = std::collections::HashMap::new();
        param_bvs.insert("x".to_string(), BV::from_u64(btor.clone(), 5, 4));
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        assert_eq!(btor.sat(), boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        assert_eq!(out, 0xF, "udiv by zero should yield all ones");
    }

    #[test]
    fn test_sdiv_by_zero_returns_extrema() {
        let ir_text = r#"fn f(x: bits[4] id=1) -> bits[4] {
  zero: bits[4] = literal(value=0, id=2)
  ret q: bits[4] = sdiv(x, zero, id=3)
 }"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let mut param_bvs = std::collections::HashMap::new();
        // 0xA is negative (-6) for 4-bit signed values.
        param_bvs.insert("x".to_string(), BV::from_u64(btor.clone(), 0xA, 4));
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        assert_eq!(btor.sat(), boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        // Dividend is negative (-6); expect maximal negative value 0b1000 (0x8)
        assert_eq!(
            out, 0x8,
            "sdiv by zero should yield max negative for negative dividend"
        );
    }

    #[test]
    fn test_sdiv_by_zero_positive_dividend() {
        let ir_text = r#"fn f() -> bits[4] {
   dividend: bits[4] = literal(value=6, id=1)
   zero: bits[4] = literal(value=0, id=2)
   ret q: bits[4] = sdiv(dividend, zero, id=3)
 }"#;
        let f = crate::xls_ir::ir_parser::Parser::new(ir_text)
            .parse_fn()
            .unwrap();
        let btor = std::rc::Rc::new(boolector::Btor::new());
        btor.set_opt(boolector::option::BtorOption::ModelGen(
            boolector::option::ModelGen::All,
        ));
        let result = ir_fn_to_boolector(btor.clone(), &f, None);
        assert_eq!(btor.sat(), boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        // Positive dividend → max positive 0b0111 (0x7)
        assert_eq!(
            out, 0x7,
            "sdiv by zero positive dividend should yield max positive"
        );
    }

    #[test]
    fn test_sel_3way_with_default() {
        let ir_text = r#"fn sel3(x: bits[2] id=1, a: bits[4] id=2, b: bits[4] id=3, c: bits[4] id=4, d: bits[4] id=5) -> bits[4] {
  ret sel.6: bits[4] = sel(x, cases=[a, b, c], default=d, id=6)
}"#;
        let mut parser = crate::xls_ir::ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        // Try all selector values 0, 1, 2, 3
        for sel_val in [0, 1, 2, 3].iter().cloned() {
            let btor = Rc::new(Btor::new());
            btor.set_opt(BtorOption::ModelGen(ModelGen::All));
            let mut param_bvs = std::collections::HashMap::new();
            param_bvs.insert("x".to_string(), BV::from_u64(btor.clone(), sel_val, 2));
            param_bvs.insert("a".to_string(), BV::from_u64(btor.clone(), 0xA, 4));
            param_bvs.insert("b".to_string(), BV::from_u64(btor.clone(), 0xB, 4));
            param_bvs.insert("c".to_string(), BV::from_u64(btor.clone(), 0xC, 4));
            param_bvs.insert("d".to_string(), BV::from_u64(btor.clone(), 0xD, 4));
            let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
            let sat_result = btor.sat();
            assert_eq!(
                sat_result,
                boolector::SolverResult::Sat,
                "Expected SAT for sel_val={}",
                sel_val
            );
            let out = result.output.get_a_solution().as_u64().unwrap();
            let expected = match sel_val {
                0 => 0xA,
                1 => 0xB,
                2 => 0xC,
                3 => 0xD,
                _ => unreachable!(),
            };
            assert_eq!(
                out, expected,
                "sel3({}, ...) should yield 0x{:X}",
                sel_val, expected
            );
        }
    }

    #[test]
    fn test_boolector_self_equiv_dynamic_bit_slice() {
        let ir_text = r#"package fuzz_pkg

fn fuzz_test(input: bits[4] id=1) -> bits[1] {
  ret dynamic_bit_slice.2: bits[1] = dynamic_bit_slice(input, input, width=1, id=2)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_fn("fuzz_test").unwrap();
        let result = prove_ir_fn_equiv(f, f, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Boolector should prove self-equivalence for dynamic_bit_slice IR"
        );
    }

    #[test]
    fn test_dynamic_bit_slice_oob_returns_zero() {
        // input width = 3, slice width = 2, start = 7 (0b111) – entirely out of bounds.
        let ir_text = r#"fn slice_oob(input: bits[3] id=1) -> bits[2] {
  ret dynamic_bit_slice.2: bits[2] = dynamic_bit_slice(input, input, width=2, id=2)
}"#;

        let mut parser = crate::xls_ir::ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));

        let mut param_bvs = std::collections::HashMap::new();
        // input = 0b111 = 7
        param_bvs.insert("input".to_string(), BV::from_u64(btor.clone(), 7, 3));

        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        let sat_result = btor.sat();
        assert_eq!(sat_result, boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        assert_eq!(out, 0, "Out-of-bounds dynamic_bit_slice should return zero");
    }

    #[test]
    fn test_dynamic_bit_slice_always_zero_equiv_literal() {
        // The slice is always out-of-bounds for at least one of its bits, and any
        // in-bounds bit is guaranteed to be zero because it comes from a higher
        // index than the bit that carries the set value.  Hence the result is always 0.
        let ir_slice = r#"fn slice_fn(x: bits[3] id=1) -> bits[2] {
  ret dynamic_bit_slice.2: bits[2] = dynamic_bit_slice(x, x, width=2, id=2)
}"#;
        let ir_zero = r#"fn zero_fn(x: bits[3] id=1) -> bits[2] {
  ret zero.2: bits[2] = literal(value=0, id=2)
}"#;
        let slice_f = crate::xls_ir::ir_parser::Parser::new(ir_slice)
            .parse_fn()
            .unwrap();
        let zero_f = crate::xls_ir::ir_parser::Parser::new(ir_zero)
            .parse_fn()
            .unwrap();
        let result = prove_ir_fn_equiv(&slice_f, &zero_f, false);
        assert_eq!(result, EquivResult::Proved);
    }

    #[test]
    fn test_xorreduce_equiv_to_self() {
        let ir_text = r#"fn xor4(x: bits[4] id=1) -> bits[1] {
  ret xor_reduce.2: bits[1] = xor_reduce(x, id=2)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_xorreduce_known_case() {
        let ir_text = r#"fn xor4(x: bits[4] id=1) -> bits[1] {
  ret xor_reduce.2: bits[1] = xor_reduce(x, id=2)
}"#;
        let mut parser = crate::xls_ir::ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        let mut param_bvs = std::collections::HashMap::new();
        // x = 0b1011 (11) -> parity 1 (odd number of ones)
        param_bvs.insert("x".to_string(), BV::from_u64(btor.clone(), 0b1011, 4));
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        let sat_result = btor.sat();
        assert_eq!(sat_result, boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        assert_eq!(out, 1);
    }

    #[test]
    fn test_onehot_msb_priority_constant() {
        // Matches the failing fuzz example: input literal 33 (0b100001) with MSB
        // priority should yield 0b100000 (32)
        let ir_onehot = r#"fn f() -> bits[7] {
  lit: bits[6] = literal(value=33, id=1)
  ret one_hot.2: bits[7] = one_hot(lit, lsb_prio=false, id=2)
}"#;

        let ir_const = r#"fn g() -> bits[7] {
  ret k32: bits[7] = literal(value=32, id=1)
}"#;

        let f = crate::xls_ir::ir_parser::Parser::new(ir_onehot)
            .parse_fn()
            .unwrap();
        let g = crate::xls_ir::ir_parser::Parser::new(ir_const)
            .parse_fn()
            .unwrap();
        let result = prove_ir_fn_equiv(&f, &g, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "OneHot implementation mismatch for MSB priority constant case"
        );
    }

    #[test]
    fn test_dynamic_bit_slice_start_zero_identity() {
        let ir_text = r#"fn f(x: bits[2] id=1) -> bits[2] {
  zero: bits[5] = literal(value=0, id=2)
  ret s: bits[2] = dynamic_bit_slice(x, zero, width=2, id=3)
}"#;
        let ir_id = r#"fn g(x: bits[2] id=1) -> bits[2] {
  ret id: bits[2] = identity(x, id=2)
}"#;

        let f = crate::xls_ir::ir_parser::Parser::new(ir_text)
            .parse_fn()
            .unwrap();
        let g = crate::xls_ir::ir_parser::Parser::new(ir_id)
            .parse_fn()
            .unwrap();
        let res = prove_ir_fn_equiv(&f, &g, false);
        assert_eq!(res, EquivResult::Proved);
    }

    #[test]
    fn test_dynamic_bit_slice_const_start_matches_bit_slice() {
        let ir_dyn = r#"fn f(x: bits[8] id=1) -> bits[2] {
  one: bits[1] = literal(value=1, id=2)
  ret s: bits[2] = dynamic_bit_slice(x, one, width=2, id=3)
}"#;
        let ir_static = r#"fn g(x: bits[8] id=1) -> bits[2] {
  ret bs: bits[2] = bit_slice(x, start=1, width=2, id=2)
}"#;
        let f = crate::xls_ir::ir_parser::Parser::new(ir_dyn)
            .parse_fn()
            .unwrap();
        let g = crate::xls_ir::ir_parser::Parser::new(ir_static)
            .parse_fn()
            .unwrap();
        assert_eq!(prove_ir_fn_equiv(&f, &g, false), EquivResult::Proved);
    }

    #[test]
    fn test_encode_148_equiv_literal_7() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ir_text_encode_148 = r#"
fn func_encode_148() -> bits[3] {
  val_148: bits[8] = literal(value=148, id=1)
  ret result: bits[3] = encode(val_148, id=2)
}
"#;
        let ir_text_literal_7 = r#"
fn func_literal_7() -> bits[3] {
  ret result: bits[3] = literal(value=7, id=1)
}
"#;
        let f_encode = ir_parser::Parser::new(ir_text_encode_148)
            .parse_fn()
            .expect("Failed to parse encode_148 IR");
        let f_literal = ir_parser::Parser::new(ir_text_literal_7)
            .parse_fn()
            .expect("Failed to parse literal_7 IR");

        let result = prove_ir_fn_equiv(&f_encode, &f_literal, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Encode(148) (bits[8]->bits[3]) should be equivalent to literal(7)"
        );
    }

    #[test]
    fn test_one_hot_reverse_fuzz_case() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ir_text_original = r#"
fn original_fn(input: bits[3] id=1) -> bits[5] {
  one_hot.2: bits[4] = one_hot(input, lsb_prio=true, id=2)
  one_hot.3: bits[5] = one_hot(one_hot.2, lsb_prio=true, id=3)
  ret reverse.4: bits[5] = reverse(one_hot.3, id=4)
}
"#;
        let ir_text_optimized = r#"
fn optimized_fn(input: bits[3] id=1) -> bits[5] {
  one_hot.2: bits[4] = one_hot(input, lsb_prio=true, id=2)
  reverse.8: bits[4] = reverse(one_hot.2, id=8)
  literal.12: bits[1] = literal(value=0, id=12)
  ret concat.10: bits[5] = concat(reverse.8, literal.12, id=10)
}
"#;

        let fn_original = ir_parser::Parser::new(ir_text_original)
            .parse_fn()
            .expect("Failed to parse original_fn IR for one_hot_reverse_fuzz_case");
        let fn_optimized = ir_parser::Parser::new(ir_text_optimized)
            .parse_fn()
            .expect("Failed to parse optimized_fn IR for one_hot_reverse_fuzz_case");

        let result = prove_ir_fn_equiv(&fn_original, &fn_optimized, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Original and Optimized IR for one_hot_reverse fuzz case should be equivalent after reverse fix"
        );
    }

    #[test]
    fn test_shrl_saturation_fuzz_case() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ir_text_original = r#"
fn original_shrl_saturation_fn(input: bits[8] id=1) -> bits[1] {
  eq.2: bits[1] = eq(input, input, id=2)
  sign_ext.3: bits[8] = sign_ext(input, new_bit_count=8, id=3)
  ret shrl.4: bits[1] = shrl(eq.2, sign_ext.3, id=4)
}
"#;
        let ir_text_optimized = r#"
fn optimized_shrl_saturation_fn(input: bits[8] id=1) -> bits[1] {
  literal.6: bits[8] = literal(value=0, id=6)
  ret eq.7: bits[1] = eq(input, literal.6, id=7)
}
"#;

        let fn_original = ir_parser::Parser::new(ir_text_original)
            .parse_fn()
            .expect("Failed to parse original_shrl_saturation_fn IR");
        let fn_optimized = ir_parser::Parser::new(ir_text_optimized)
            .parse_fn()
            .expect("Failed to parse optimized_shrl_saturation_fn IR");

        let result = prove_ir_fn_equiv(&fn_original, &fn_optimized, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Original and Optimized IR for shrl saturation fuzz case should be equivalent after shift saturation fix"
        );
    }

    #[test]
    fn test_shift_single_bit_zero_amount_identity() {
        let ir_shl = r#"fn shl1_zero(x: bits[1] id=1) -> bits[1] {
  zero: bits[1] = literal(value=0, id=2)
  ret shll.3: bits[1] = shll(x, zero, id=3)
}"#;
        let ir_id = r#"fn id1(x: bits[1] id=1) -> bits[1] {
  ret id.2: bits[1] = identity(x, id=2)
}"#;
        let f_shl = crate::xls_ir::ir_parser::Parser::new(ir_shl)
            .parse_fn()
            .unwrap();
        let f_id = crate::xls_ir::ir_parser::Parser::new(ir_id)
            .parse_fn()
            .unwrap();
        assert_eq!(prove_ir_fn_equiv(&f_shl, &f_id, false), EquivResult::Proved);
    }

    #[test]
    fn test_shra_single_bit_by_one_identity() {
        // For a 1-bit value, an arithmetic right shift by 1 should replicate the sign
        // bit, which is the value itself. Therefore shra(x, 1) must be
        // equivalent to identity(x).
        let ir_shra = r#"fn shra1_one(x: bits[1] id=1) -> bits[1] {
  one: bits[1] = literal(value=1, id=2)
  ret shra.3: bits[1] = shra(x, one, id=3)
}"#;
        let ir_id = r#"fn id1(x: bits[1] id=1) -> bits[1] {
  ret id.2: bits[1] = identity(x, id=2)
}"#;
        let f_shra = crate::xls_ir::ir_parser::Parser::new(ir_shra)
            .parse_fn()
            .unwrap();
        let f_id = crate::xls_ir::ir_parser::Parser::new(ir_id)
            .parse_fn()
            .unwrap();
        assert_eq!(
            prove_ir_fn_equiv(&f_shra, &f_id, false),
            EquivResult::Proved
        );
    }

    /// From a fuzz_ir_opt_equiv minimized counter-example.
    #[test]
    fn test_fuzz_shra_failing_case_equiv() {
        let ir_original = r#"package fuzz_pkg

fn fuzz_test(input: bits[2] id=1) -> bits[1] {
  shra.2: bits[2] = shra(input, input, id=2)
  sign_ext.5: bits[5] = sign_ext(input, new_bit_count=5, id=5)
  shra.3: bits[2] = shra(input, shra.2, id=3)
  shra.7: bits[5] = shra(sign_ext.5, shra.3, id=7)
  nor.4: bits[2] = nor(shra.2, shra.2, id=4)
  shra.8: bits[5] = shra(shra.7, nor.4, id=8)
  ule.11: bits[1] = ule(sign_ext.5, shra.7, id=11)
  shra.10: bits[5] = shra(shra.8, shra.3, id=10)
  one_hot.6: bits[3] = one_hot(shra.2, lsb_prio=true, id=6)
  one_hot_sel.9: bits[2] = one_hot_sel(shra.2, cases=[shra.2, shra.2], id=9)
  shra.12: bits[2] = shra(shra.3, shra.3, id=12)
  ret shra.13: bits[1] = shra(ule.11, shra.10, id=13)
}
"#;

        let ir_optimized = r#"package fuzz_pkg

top fn fuzz_test(input: bits[2] id=1) -> bits[1] {
  shra.2: bits[2] = shra(input, input, id=2)
  sign_ext.5: bits[5] = sign_ext(input, new_bit_count=5, id=5)
  shra.3: bits[2] = shra(input, shra.2, id=3)
  shra.7: bits[5] = shra(sign_ext.5, shra.3, id=7)
  bit_slice.20: bits[2] = bit_slice(shra.7, start=0, width=2, id=20)
  ret ule.21: bits[1] = ule(input, bit_slice.20, id=21)
}
"#;

        let pkg_orig = crate::xls_ir::ir_parser::Parser::new(ir_original)
            .parse_package()
            .unwrap();
        let pkg_opt = crate::xls_ir::ir_parser::Parser::new(ir_optimized)
            .parse_package()
            .unwrap();

        let f_orig = pkg_orig.get_fn("fuzz_test").unwrap();
        let f_opt = pkg_opt.get_fn("fuzz_test").unwrap();

        let result = prove_ir_fn_equiv(f_orig, f_opt, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Boolector disproved equivalence: {:?}",
            result
        );
    }

    #[test]
    fn test_fuzz_ir_opt_equiv_regression_array_index_oob() {
        // Regression test for array_index out-of-bounds semantics.
        // The original IR performs an unsigned divide. The optimized IR replaces this
        // with a table lookup guarded by an ult comparison along with a
        // sign-extension mask. Prior to fixing ArrayIndex conversion we
        // incorrectly returned the *first* array element on OOB indices whereas
        // XLS semantics require using the *last* element.
        let _ = env_logger::builder().is_test(true).try_init();

        let orig_ir = r#"fn fuzz_test(input: bits[7] id=1) -> bits[7] {
  literal.3: bits[7] = literal(value=113, id=3)
  smul.2: bits[7] = smul(input, input, id=2)
  ret udiv.4: bits[7] = udiv(literal.3, input, id=4)
}"#;

        let opt_ir = r#"fn fuzz_test(input: bits[7] id=1) -> bits[7] {
  literal.12: bits[7] = literal(value=114, id=12)
  literal.7: bits[7][58] = literal(value=[127, 113, 56, 37, 28, 22, 18, 16, 14, 12, 11, 10, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], id=7)
  ult.13: bits[1] = ult(input, literal.12, id=13)
  array_index.8: bits[7] = array_index(literal.7, indices=[input], id=8)
  sign_ext.14: bits[7] = sign_ext(ult.13, new_bit_count=7, id=14)
  ret and.15: bits[7] = and(array_index.8, sign_ext.14, id=15)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(orig_ir)
            .parse_fn()
            .expect("Failed to parse original IR");
        let rhs = crate::xls_ir::ir_parser::Parser::new(opt_ir)
            .parse_fn()
            .expect("Failed to parse optimized IR");

        let result = prove_ir_fn_equiv(&lhs, &rhs, false);
        assert_eq!(
            result,
            EquivResult::Proved,
            "Expected functions to be equivalent after fix, got {:?}",
            result
        );
    }

    #[test]
    fn test_umod_by_zero_returns_zero() {
        // Unsigned modulus by zero should yield 0 regardless of the dividend.
        let ir_text = r#"fn f(x: bits[8] id=1) -> bits[8] {
  zero: bits[8] = literal(value=0, id=2)
  ret r: bits[8] = umod(x, zero, id=3)
}"#;
        let f = crate::xls_ir::ir_parser::Parser::new(ir_text)
            .parse_fn()
            .unwrap();
        // Instantiate Boolector directly to evaluate.
        let btor = std::rc::Rc::new(boolector::Btor::new());
        btor.set_opt(boolector::option::BtorOption::ModelGen(
            boolector::option::ModelGen::All,
        ));
        let mut param_bvs = std::collections::HashMap::new();
        // Pick a non-zero dividend to exercise the x % 0 path.
        let x_val = 37u64;
        let x_bv = boolector::BV::from_u64(btor.clone(), x_val, 8);
        param_bvs.insert("x".to_string(), x_bv);
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        assert_eq!(btor.sat(), boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        // Expect output to be 0.
        assert_eq!(out, 0);
    }

    #[test]
    fn test_smod_by_zero_returns_zero() {
        let ir_text = r#"fn f(x: bits[8] id=1) -> bits[8] {
   ret smod.2: bits[8] = smod(x, x, id=2)
 }"#;
        let f = crate::xls_ir::ir_parser::Parser::new(ir_text)
            .parse_fn()
            .unwrap();
        let btor = std::rc::Rc::new(boolector::Btor::new());
        btor.set_opt(boolector::option::BtorOption::ModelGen(
            boolector::option::ModelGen::All,
        ));
        let mut param_bvs = std::collections::HashMap::new();
        let x_val = 0u64; // divisor zero
        let x_bv = boolector::BV::from_u64(btor.clone(), x_val, 8);
        param_bvs.insert("x".to_string(), x_bv);
        let result = ir_fn_to_boolector(btor.clone(), &f, Some(&param_bvs));
        assert_eq!(btor.sat(), boolector::SolverResult::Sat);
        let out = result.output.get_a_solution().as_u64().unwrap();
        assert_eq!(out, 0);
    }

    #[test]
    fn test_fuzz_ir_opt_equiv_regression_smod_by_zero() {
        // Regression for fuzz case minimized-from-45a97b4d... focusing on smod
        // behaviour.
        let _ = env_logger::builder().is_test(true).try_init();

        let orig_ir = r#"fn fuzz_test(input: bits[7] id=1) -> bits[7] {
   literal.3: bits[7] = literal(value=108, id=3)
   literal.2: bits[1] = literal(value=0, id=2)
   ret smod.4: bits[7] = smod(literal.3, input, id=4)
 }"#;

        let opt_ir = r#"fn fuzz_test(input: bits[7] id=1) -> bits[7] {
   bit_slice.29: bits[1] = bit_slice(input, start=6, width=1, id=29)
   neg.17: bits[7] = neg(input, id=17)
   literal.20: bits[7][12] = literal(value=[0, 108, 118, 122, 123, 124, 125, 126, 126, 126, 126, 127], id=20)
   priority_sel.18: bits[7] = priority_sel(bit_slice.29, cases=[neg.17], default=input, id=18)
   literal.30: bits[7] = literal(value=21, id=30)
   array_index.21: bits[7] = array_index(literal.20, indices=[priority_sel.18], id=21)
   ult.31: bits[1] = ult(priority_sel.18, literal.30, id=31)
   bit_slice.51: bits[6] = bit_slice(array_index.21, start=0, width=6, id=51)
   sign_ext.55: bits[6] = sign_ext(ult.31, new_bit_count=6, id=55)
   and.53: bits[6] = and(bit_slice.51, sign_ext.55, id=53)
   neg.49: bits[6] = neg(and.53, id=49)
   priority_sel.46: bits[6] = priority_sel(bit_slice.29, cases=[neg.49], default=and.53, id=46)
   literal.15: bits[7] = literal(value=0, id=15)
   literal.3: bits[7] = literal(value=108, id=3)
   smul.50: bits[7] = smul(priority_sel.46, input, id=50)
   ne.43: bits[1] = ne(input, literal.15, id=43)
   sub.8: bits[7] = sub(literal.3, smul.50, id=8)
   sign_ext.40: bits[7] = sign_ext(ne.43, new_bit_count=7, id=40)
   ret and.41: bits[7] = and(sub.8, sign_ext.40, id=41)
 }"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(orig_ir)
            .parse_fn()
            .expect("parse lhs");
        let rhs = crate::xls_ir::ir_parser::Parser::new(opt_ir)
            .parse_fn()
            .expect("parse rhs");
        let result = prove_ir_fn_equiv(&lhs, &rhs, false);
        assert_eq!(result, EquivResult::Proved);
    }

    #[test]

    fn test_bit_slice_update_wider_update_value() {
        // Regression test for a fuzz case where the update_value is wider than the
        // slice being updated. Previously this caused Boolector to abort due to a
        // width mismatch. The updated conversion logic should handle the width
        // discrepancy safely and prove self-equivalence.
        let ir_text = r#"fn f(input: bits[7] id=1) -> bits[5] {
  slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
  ret upd.3: bits[5] = bit_slice_update(slice.2, input, input, id=3)
}"#;
        assert_fn_equiv_to_self(ir_text);
    }

    #[test]
    fn test_bit_slice_update_large_update_value() {
        // update_value is 80 bits, operand is 32 bits. Only lower bits should be used.
        let ir_text = r#"fn g() -> bits[32] {
  operand: bits[32] = literal(value=0xABCD1234, id=1)
  start: bits[5] = literal(value=4, id=2)
  upd_val: bits[80] = literal(value=0xFFFFFFFFFFFFFFFFFFF, id=3)
  ret r: bits[32] = bit_slice_update(operand, start, upd_val, id=4)
}"#;

        let f = crate::xls_ir::ir_parser::Parser::new(ir_text)
            .parse_fn()
            .unwrap();
        // Evaluate through Boolector; ensure no panic and width is 32.
        let btor = std::rc::Rc::new(boolector::Btor::new());
        btor.set_opt(boolector::option::BtorOption::ModelGen(
            boolector::option::ModelGen::All,
        ));
        let res = super::ir_fn_to_boolector(btor.clone(), &f, None);
        assert_eq!(res.output.get_width(), 32);
    }

    #[test]
    fn test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob() {
        // Matches minimized-from-e11bf74e... fuzz sample.
        let orig_ir = r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
  literal_255: bits[8] = literal(value=255, id=3)
  bsu1: bits[8] = bit_slice_update(input, input, input, id=2)
  ret bsu2: bits[8] = bit_slice_update(literal_255, literal_255, bsu1, id=4)
}"#;
        let opt_ir = r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
  ret literal_255: bits[8] = literal(value=255, id=3)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(orig_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(opt_ir)
            .parse_fn()
            .unwrap();

        assert_eq!(
            super::prove_ir_fn_equiv(&lhs, &rhs),
            super::EquivResult::Proved

    fn test_tuple_literal_vs_constructed_inconsistent() {
        // Two functions that should be equivalent: one returns a tuple literal, the
        // other constructs the same tuple with the `tuple` operator.  If tuple
        // encoding is consistent the prover should return `Proved`.  The
        // current implementation is inconsistent, therefore we expect
        // `Disproved` which reveals the bug.
        let lhs_ir = r#"fn lhs() -> (bits[8], bits[4]) {
  ret lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
}"#;

        let rhs_ir = r#"fn rhs() -> (bits[8], bits[4]) {
  lit0: bits[8] = literal(value=0x12, id=1)
  lit1: bits[4] = literal(value=0x4, id=2)
  ret tup: (bits[8], bits[4]) = tuple(lit0, lit1, id=3)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, false);
        assert_eq!(
            result,
            super::EquivResult::Proved,
            "Tuple literal vs constructed should be equivalent; failing reveals bug in tuple encoding"
        );
    }

    #[test]
    fn test_tuple_index_on_literal_inconsistent() {
        // The first element of (0x12, 0x4) is 0x12.  With consistent tuple encoding,
        // the two functions below should be equivalent.  Inconsistent encoding
        // causes the prover to find a spurious difference.
        let lhs_ir = r#"fn lhs() -> bits[8] {
  lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
  ret idx0: bits[8] = tuple_index(lit_tuple, index=0, id=2)
}"#;

        let rhs_ir = r#"fn rhs() -> bits[8] {
  ret lit: bits[8] = literal(value=0x12, id=1)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, false);
        assert_eq!(
            result,
            super::EquivResult::Proved,
            "Tuple index on literal should produce equivalent values; failing reveals bug in tuple encoding"
        );
    }

    #[test]
    fn test_array_update_element0_value() {
        // Update element 0 of a 3-element array and then read it back.  Correct
        // semantics should yield the updated value (0xF).
        // Current implementation misorders bits in array_update, so the prover
        // will find a mismatch and return Disproved – the test therefore
        // expects Proved and will fail until the bug is fixed.
        let lhs_ir = r#"fn lhs() -> bits[4] {
  orig: bits[4][3] = literal(value=[1, 2, 4], id=1)
  val_f: bits[4] = literal(value=0xF, id=2)
  idx0: bits[2] = literal(value=0, id=3)
  upd: bits[4][3] = array_update(orig, val_f, indices=[idx0], id=4)
  ret elem0: bits[4] = array_index(upd, indices=[idx0], id=5)
}"#;

        let rhs_ir = r#"fn rhs() -> bits[4] {
  ret lit: bits[4] = literal(value=0xF, id=1)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, false);
        assert!(
            matches!(result, super::EquivResult::Proved),
            "array_update then array_index at 0 should yield updated value; failing reveals bug in array_update ordering"
        );
    }

    #[test]
    fn test_array_update_full_array_equiv() {
        // Replace element 1 in a 3-element array and compare full array to a
        // literal with the expected ordering.  Should be equivalent if ordering
        // is correct.
        let lhs_ir = r#"fn lhs() -> bits[4][3] {
  orig: bits[4][3] = literal(value=[1, 2, 4], id=1)
  val_a: bits[4] = literal(value=0xA, id=2)
  idx1: bits[2] = literal(value=1, id=3)
  upd: bits[4][3] = array_update(orig, val_a, indices=[idx1], id=4)
  ret out: bits[4][3] = identity(upd, id=5)
}"#;

        let rhs_ir = r#"fn rhs() -> bits[4][3] {
  ret expected: bits[4][3] = literal(value=[1, 0xA, 4], id=1)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, false);
        assert!(
            matches!(result, super::EquivResult::Proved),
            "array_update should produce expected full array value; failing reveals ordering bug"
        );
    }

    #[test]
    fn test_array_literal_vs_constructed_equiv() {
        // Literal array should be equivalent to one built via the `array` IR
        // node from the same elements.
        let lhs_ir = r#"fn lhs() -> bits[4][3] {
  ret litarr: bits[4][3] = literal(value=[1, 2, 3], id=1)
}"#;

        let rhs_ir = r#"fn rhs() -> bits[4][3] {
  e0: bits[4] = literal(value=1, id=1)
  e1: bits[4] = literal(value=2, id=2)
  e2: bits[4] = literal(value=3, id=3)
  ret arr: bits[4][3] = array(e0, e1, e2, id=4)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, false);
        assert!(
            matches!(result, super::EquivResult::Proved),
            "Array literal and constructed array should be equivalent; discrepancy indicates ordering bug"
        );
    }

    #[test]
    fn test_array_index_on_literal_elements() {
        // Index 1 of literal [1,2,3] is 2.
        let lhs_ir = r#"fn lhs() -> bits[4] {
  lit: bits[4][3] = literal(value=[1, 2, 3], id=1)
  idx1: bits[2] = literal(value=0, id=2)
  ret elem1: bits[4] = array_index(lit, indices=[idx1], id=3)
}"#;

        let rhs_ir = r#"fn rhs() -> bits[4] {
  ret two: bits[4] = literal(value=1, id=1)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, false);
        assert!(
            matches!(result, super::EquivResult::Proved),
            "array_index on literal should return correct element value; failure indicates ordering bug"
        );
    }

    #[test]
    fn test_array_flatten_equiv_bits() {
        // Verify that an array value and its flattened bits representation are
        // equivalent when flatten_aggregates=true.
        let lhs_ir = r#"fn lhs() -> bits[4][3] {
  ret litarr: bits[4][3] = literal(value=[1, 2, 3], id=1)
}"#;

        let rhs_ir = r#"fn rhs() -> bits[12] {
  ret flat: bits[12] = literal(value=0x321, id=1)
}"#;

        let lhs = crate::xls_ir::ir_parser::Parser::new(lhs_ir)
            .parse_fn()
            .unwrap();
        let rhs = crate::xls_ir::ir_parser::Parser::new(rhs_ir)
            .parse_fn()
            .unwrap();

        let result = super::prove_ir_fn_equiv(&lhs, &rhs, true);
        assert!(
            matches!(result, super::EquivResult::Proved),
            "Flattened bits should be equivalent to array value when flattening is allowed"
        );
    }
}
