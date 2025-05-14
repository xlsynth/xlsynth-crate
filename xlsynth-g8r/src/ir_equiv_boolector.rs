// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-boolector")]

use crate::xls_ir::ir::{Fn, NodePayload, NodeRef};
use crate::xls_ir::ir_parser;
use crate::xls_ir::ir_utils::get_topological;
use crate::{
    test_utils::{load_bf16_add_sample, load_bf16_mul_sample, Opt},
    xls_ir::ir::{Node, Type},
};
use boolector::option::{BtorOption, ModelGen};
use boolector::{Btor, SolverResult, BV};
use log::debug;
use std::collections::HashMap;
use std::rc::Rc;
use xlsynth::IrBits;

/// Result of converting an XLS IR function to Boolector logic.
pub struct IrFnBoolectorResult {
    pub output: BV<Rc<Btor>>,
    pub inputs: Vec<(String, BV<Rc<Btor>>)>,
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
            NodePayload::Literal(ir_value) => {
                let bits = ir_value.to_bits().expect("Literal must be bits");
                let width = bits.get_bit_count() as u32;
                let mut value = 0u64;
                for i in 0..width {
                    if bits.get_bit(i as usize).unwrap() {
                        value |= 1 << i;
                    }
                }
                BV::from_u64(btor.clone(), value, width)
            }
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
                        assert!(width > 0, "Reverse: width must be > 0");
                        let mut result = arg_bv.slice(0, 0);
                        for i in 1..width {
                            let bit = arg_bv.slice(i, i);
                            result = bit.concat(&result);
                        }
                        result
                    }
                    _ => panic!("Unop {:?} not yet implemented in Boolector conversion", op),
                }
            }
            NodePayload::OneHot { arg, lsb_prio } => {
                let arg_bv = env.get(arg).expect("OneHot argument must be present");
                let width = arg_bv.get_width();
                assert!(width > 0, "OneHot: width must be > 0");
                log::info!("[OneHot] arg_bv width: {}", width);
                let mut bits: Vec<BV<Rc<Btor>>> = Vec::with_capacity((width + 1) as usize);
                let mut prior_not: Option<BV<Rc<Btor>>> = None;
                for i in 0..width {
                    let idx = if *lsb_prio { i } else { width - 1 - i };
                    let bit = arg_bv.slice(idx, idx);
                    log::info!(
                        "[OneHot] bit {} (idx={}): width={} is_const={} bin={:?}",
                        i,
                        idx,
                        bit.get_width(),
                        bit.is_const(),
                        bit.as_binary_str()
                    );
                    let this_no_prior = if let Some(prior) = &prior_not {
                        let v = prior.and(&bit.not());
                        log::info!("[OneHot] this_no_prior (i={}) with prior: width={} is_const={} bin={:?}", i, v.get_width(), v.is_const(), v.as_binary_str());
                        v
                    } else {
                        let v = bit.not();
                        log::info!(
                            "[OneHot] this_no_prior (i={}) no prior: width={} is_const={} bin={:?}",
                            i,
                            v.get_width(),
                            v.is_const(),
                            v.as_binary_str()
                        );
                        v
                    };
                    let out_bit = bit.and(&this_no_prior.not());
                    log::info!(
                        "[OneHot] out_bit (i={}): width={} is_const={} bin={:?}",
                        i,
                        out_bit.get_width(),
                        out_bit.is_const(),
                        out_bit.as_binary_str()
                    );
                    bits.push(out_bit.clone());
                    prior_not = Some(if let Some(prior) = prior_not {
                        let v = prior.and(&bit.not());
                        log::info!(
                            "[OneHot] prior_not update (i={}): width={} is_const={} bin={:?}",
                            i,
                            v.get_width(),
                            v.is_const(),
                            v.as_binary_str()
                        );
                        v
                    } else {
                        let v = bit.not();
                        log::info!(
                            "[OneHot] prior_not init (i={}): width={} is_const={} bin={:?}",
                            i,
                            v.get_width(),
                            v.is_const(),
                            v.as_binary_str()
                        );
                        v
                    });
                }
                if !*lsb_prio {
                    bits.reverse();
                }
                if let Some(prior) = prior_not {
                    log::info!(
                        "[OneHot] final prior_not: width={} is_const={} bin={:?}",
                        prior.get_width(),
                        prior.is_const(),
                        prior.as_binary_str()
                    );
                    bits.push(prior);
                } else {
                    let v = arg_bv.slice(0, 0).not();
                    log::info!(
                        "[OneHot] final prior_not (empty): width={} is_const={} bin={:?}",
                        v.get_width(),
                        v.is_const(),
                        v.as_binary_str()
                    );
                    bits.push(v);
                }
                assert_eq!(
                    bits.len(),
                    (width + 1) as usize,
                    "OneHot output width should be input width + 1"
                );
                assert!(!bits.is_empty(), "OneHot: bits vector must not be empty");
                let mut result = bits[0].clone();
                log::info!(
                    "[OneHot] result init: width={} is_const={} bin={:?}",
                    result.get_width(),
                    result.is_const(),
                    result.as_binary_str()
                );
                for (i, b) in bits.iter().enumerate().skip(1) {
                    log::info!(
                        "[OneHot] concat (i={}): lhs width={} rhs width={}",
                        i,
                        b.get_width(),
                        result.get_width()
                    );
                    result = b.concat(&result);
                    log::info!(
                        "[OneHot] after concat (i={}): width={} is_const={} bin={:?}",
                        i,
                        result.get_width(),
                        result.is_const(),
                        result.as_binary_str()
                    );
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
                _ => panic!(
                    "NaryOp {:?} not yet implemented in Boolector conversion",
                    op
                ),
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
                    Binop::Shll => shift_boolector(a_bv, b_bv, |x, y| x.sll(y)),
                    Binop::Shrl => shift_boolector(a_bv, b_bv, |x, y| x.srl(y)),
                    Binop::Shra => shift_boolector(a_bv, b_bv, |x, y| x.sra(y)),
                    Binop::Sgt => a_bv.sgt(b_bv),
                    Binop::Slt => a_bv.slt(b_bv),
                    Binop::Sle => a_bv.slte(b_bv),
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
                // Only support 2-way select for now
                assert!(
                    cases.len() == 2 && default.is_none(),
                    "Only 2-way select without default supported"
                );
                let sel_bv = env.get(selector).expect("Selector BV must be present");
                let case0 = env.get(&cases[0]).expect("Case 0 BV must be present");
                let case1 = env.get(&cases[1]).expect("Case 1 BV must be present");
                assert_eq!(
                    sel_bv.get_width(),
                    1,
                    "Selector BV must be 1 bit for 2-way select"
                );
                assert_eq!(
                    case0.get_width(),
                    case1.get_width(),
                    "Case widths must match"
                );
                sel_bv.cond_bv(case1, case0)
            }
            NodePayload::SignExt { arg, new_bit_count } => {
                let arg_bv = env.get(arg).expect("SignExt argument must be present");
                let from_width = arg_bv.get_width();
                let to_width = *new_bit_count as u32;
                assert!(
                    to_width > from_width,
                    "SignExt: new_bit_count must be greater than argument width"
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
                // Build a chain of selects for each possible index value
                let mut result = None;
                for i in 0..element_count {
                    let high = ((i + 1) * elem_width as usize - 1) as u32;
                    let low = (i * elem_width as usize) as u32;
                    let elem = array_bv.slice(high, low);
                    let idx_val = BV::from_u64(array_bv.get_btor(), i as u64, index_bv.get_width());
                    let is_this = index_bv._eq(&idx_val);
                    result = Some(if let Some(acc) = result {
                        is_this.cond_bv(&elem, &acc)
                    } else {
                        elem
                    });
                }
                result.expect("ArrayIndex: array must have at least one element")
            }
            NodePayload::Encode { arg } => {
                let arg_bv = env.get(arg).expect("Encode argument must be present");
                let width = arg_bv.get_width();
                assert!(width > 0, "Encode: width must be > 0");
                let out_width = (width as f64).log2().ceil() as u32;
                assert_eq!(
                    node.ty.bit_count() as u32,
                    out_width,
                    "Encode: output width must be log2(input width)"
                );
                // Priority encoder: for each bit, if set, output its index
                let mut result = BV::from_u64(btor.clone(), 0, out_width);
                for i in (0..width).rev() {
                    // MSB priority
                    let bit = arg_bv.slice(i, i);
                    let idx_bv = BV::from_u64(btor.clone(), i as u64, out_width);
                    result = bit.cond_bv(&idx_bv, &result);
                }
                result
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
            other => todo!("Boolector conversion for {:?} not yet implemented", other),
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
        panic!("Cannot return Nil node from ir_fn_to_boolector: Boolector does not support 0-width bitvectors");
    }
}

/// Result of equivalence checking.
#[derive(Debug, PartialEq, Eq)]
pub enum EquivResult {
    Proved,
    Disproved(Vec<IrBits>), // Counterexample input
}

/// Checks equivalence of two IR functions using Boolector.
/// Only supports literal-only, zero-parameter functions for now.
pub fn check_equiv(lhs: &Fn, rhs: &Fn) -> EquivResult {
    assert_eq!(lhs.ret_ty, rhs.ret_ty, "Return types must match");
    // Check for duplicate parameter names
    let mut seen = std::collections::HashSet::new();
    for param in &lhs.params {
        assert!(
            seen.insert(&param.name),
            "Duplicate parameter name '{}' in lhs.params",
            param.name
        );
    }
    // Check that parameter lists are identical (name, type, order)
    assert_eq!(
        lhs.params.len(),
        rhs.params.len(),
        "Parameter count mismatch"
    );
    for (l, r) in lhs.params.iter().zip(rhs.params.iter()) {
        assert_eq!(
            l.name, r.name,
            "Parameter name mismatch: {} vs {}",
            l.name, r.name
        );
        assert_eq!(
            l.ty, r.ty,
            "Parameter type mismatch for {}: {:?} vs {:?}",
            l.name, l.ty, r.ty
        );
    }
    assert!(
        lhs.params.len() > 0,
        "check_equiv only supports functions with at least one parameter"
    );
    let btor = Rc::new(Btor::new());
    btor.set_opt(BtorOption::ModelGen(ModelGen::All));
    // Create shared parameter BVs for all parameters (by name)
    let mut param_bvs = HashMap::new();
    for param in &lhs.params {
        let width = param.ty.bit_count() as u32;
        let bv = BV::new(btor.clone(), width, Some(&param.name));
        param_bvs.insert(param.name.clone(), bv);
    }
    let lhs_result = ir_fn_to_boolector(btor.clone(), lhs, Some(&param_bvs));
    let rhs_result = ir_fn_to_boolector(btor.clone(), rhs, Some(&param_bvs));
    // Assert that outputs are not equal (look for a counterexample)
    let diff = lhs_result.output._ne(&rhs_result.output);
    diff.assert();
    match btor.sat() {
        SolverResult::Unsat => EquivResult::Proved,
        SolverResult::Sat => {
            // Extract input assignments from the model
            let mut counterexample = Vec::new();
            for param in &lhs.params {
                let bv = param_bvs.get(&param.name).unwrap();
                let width = bv.get_width() as usize;
                let solution = bv.get_a_solution();
                let disamb = solution.disambiguate();
                let bitstr = disamb.as_01x_str();
                let bits: Vec<bool> = bitstr.chars().rev().map(|c| c == '1').collect();
                if bits.len() != width {
                    log::trace!(
                        "[ir_fn_to_boolector] Solution width mismatch for param: name={}, expected width={}, got {}",
                        param.name, width, bits.len()
                    );
                }
                let ir_bits = crate::ir_value_utils::ir_bits_from_lsb_is_0(&bits);
                counterexample.push(ir_bits);
            }
            EquivResult::Disproved(counterexample)
        }
        SolverResult::Unknown => panic!("Solver returned unknown result"),
    }
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
    let k = (pow2 as f64).log2() as u32;
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

/// Asserts that the given IR function (as text) is equivalent to itself.
fn assert_fn_equiv_to_self(ir_text: &str) {
    let mut parser = ir_parser::Parser::new(ir_text);
    let f = parser.parse_fn().expect("Failed to parse IR");
    let result = check_equiv(&f, &f);
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
    let result = check_equiv(&f, &g);
    match result {
        EquivResult::Disproved(ref cex) => {
            assert_eq!(cex.len(), 1);
            let bits = &cex[0];
            assert_eq!(bits.get_bit_count(), 8);
            // Should be 42
            let mut value = 0u64;
            for i in 0..8 {
                if bits.get_bit(i).unwrap() {
                    value |= 1 << i;
                }
            }
            assert_eq!(value, 42);
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
  afterall.2: () = after_all()
  ret x
}"#;
    assert_fn_equiv_to_self(ir_text);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_utils::{load_bf16_add_sample, load_bf16_mul_sample, Opt},
        xls_ir::ir::{Fn, Node, NodePayload, Type},
    };
    use boolector::{
        option::{BtorOption, ModelGen},
        Btor,
    };
    use std::rc::Rc;
    use xlsynth::IrValue;

    /// Helper to create a Boolector context with model generation enabled,
    /// passing it to the provided closure.
    fn with_btor_model_gen<F, R>(f: F) -> R
    where
        F: FnOnce(Rc<Btor>) -> R,
    {
        let btor = Rc::new(Btor::new());
        btor.set_opt(BtorOption::ModelGen(ModelGen::All));
        f(btor)
    }

    #[test]
    fn test_bf16_add_sample_to_boolector() {
        let sample = load_bf16_add_sample(Opt::No);
        let g8r_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();
        let btor = Rc::new(Btor::new());
        let bv = ir_fn_to_boolector(btor, g8r_fn, None);
        // bf16 is 16 bits
        assert_eq!(bv.output.get_width(), 16);
    }

    #[test]
    fn test_bf16_mul_sample_to_boolector() {
        let sample = load_bf16_mul_sample(Opt::No);
        let g8r_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();
        let btor = Rc::new(Btor::new());
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
}
