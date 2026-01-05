// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::corners::bucket_xor_popcount;
use crate::corners::{
    AddCornerTag, ArrayIndexCornerTag, CompareDistanceCornerTag, CornerEvent, CornerKind,
    DynamicBitSliceCornerTag, FailureEvent, FailureKind, NegCornerTag, ShiftCornerTag,
    ShraCornerTag, SignExtCornerTag,
};
use crate::ir;
use crate::ir::NodePayload as P;
use crate::ir_utils::get_topological;
use xlsynth::{IrBits, IrValue};

fn eval_pure(n: &ir::Node, env: &HashMap<ir::NodeRef, IrValue>) -> IrValue {
    log::trace!("eval_pure: {:?}", n);
    match n.payload {
        ir::NodePayload::Literal(ref ir_value) => ir_value.clone(),
        ir::NodePayload::Binop(binop, ref lhs, ref rhs) => {
            let lhs_value: &IrValue = env.get(lhs).unwrap();
            let rhs_value: &IrValue = env.get(rhs).unwrap();
            match binop {
                ir::Binop::Eq => IrValue::bool(lhs_value == rhs_value),
                ir::Binop::Ne => IrValue::bool(lhs_value != rhs_value),
                ir::Binop::Add => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let r = lhs_bits.add(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Sub => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let r = lhs_bits.sub(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Shll => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
                    let r = lhs_bits.shll(shift);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Shrl => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
                    let r = lhs_bits.shrl(shift);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Shra => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
                    let r = lhs_bits.shra(shift);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Smulp | ir::Binop::Smul => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let r = lhs_bits.smul(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Umulp | ir::Binop::Umul => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    let r = lhs_bits.umul(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Uge => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.uge(&rhs_bits))
                }
                ir::Binop::Ugt => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.ugt(&rhs_bits))
                }
                ir::Binop::Ult => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.ult(&rhs_bits))
                }
                ir::Binop::Ule => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.ule(&rhs_bits))
                }
                ir::Binop::Sgt => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.sgt(&rhs_bits))
                }
                ir::Binop::Sge => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.sge(&rhs_bits))
                }
                ir::Binop::Slt => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.slt(&rhs_bits))
                }
                ir::Binop::Sle => {
                    let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
                    let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                    IrValue::bool(lhs_bits.sle(&rhs_bits))
                }
                ir::Binop::Gate => {
                    // XLS `gate`: when predicate is false, returns all-zero value.
                    // Predicate is expected to be bits[1].
                    let pred = lhs_value.to_bool().expect("gate predicate must be bits[1]");
                    if pred {
                        rhs_value.clone()
                    } else {
                        let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
                        let zeros: Vec<bool> = vec![false; rhs_bits.get_bit_count()];
                        let out = IrBits::from_lsb_is_0(&zeros);
                        IrValue::from_bits(&out)
                    }
                }
                _ => panic!("Unsupported binop: {:?}", binop),
            }
        }
        ir::NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
            let lhs_bits: IrBits = env.get(&lhs).unwrap().to_bits().unwrap();
            let rhs_bits: IrBits = env.get(&rhs).unwrap().to_bits().unwrap();
            let w = lhs_bits.get_bit_count();
            assert_eq!(
                w,
                rhs_bits.get_bit_count(),
                "ExtCarryOut: lhs/rhs width mismatch"
            );
            let c_in_bool = env
                .get(&c_in)
                .unwrap()
                .to_bool()
                .expect("ExtCarryOut c_in must be bits[1]");

            // Reference semantics:
            //   carry = bit_slice(zero_ext(lhs,w+1)+zero_ext(rhs,w+1)+zero_ext(c_in,w+1),
            // start=w, width=1)
            let mut lhs_ext: Vec<bool> = Vec::with_capacity(w + 1);
            let mut rhs_ext: Vec<bool> = Vec::with_capacity(w + 1);
            for i in 0..w {
                lhs_ext.push(lhs_bits.get_bit(i).unwrap());
                rhs_ext.push(rhs_bits.get_bit(i).unwrap());
            }
            lhs_ext.push(false);
            rhs_ext.push(false);
            let lhs_w1 = IrBits::from_lsb_is_0(&lhs_ext);
            let rhs_w1 = IrBits::from_lsb_is_0(&rhs_ext);
            let sum_w1 = lhs_w1.add(&rhs_w1);
            let c_in_w1 = {
                let mut v: Vec<bool> = vec![false; w + 1];
                v[0] = c_in_bool;
                IrBits::from_lsb_is_0(&v)
            };
            let sum_w1_ci = sum_w1.add(&c_in_w1);
            IrValue::bool(sum_w1_ci.get_bit(w).unwrap())
        }
        ir::NodePayload::Unop(unop, ref operand) => {
            let operand_value: &IrValue = env.get(operand).unwrap();
            match unop {
                ir::Unop::Neg => {
                    let operand_bits = operand_value.to_bits().unwrap();

                    let r = operand_bits.negate();
                    IrValue::from_bits(&r)
                }
                ir::Unop::Not => {
                    let operand_bits = operand_value.to_bits().unwrap();
                    let r = operand_bits.not();
                    IrValue::from_bits(&r)
                }
                ir::Unop::Identity => operand_value.clone(),
                ir::Unop::OrReduce => {
                    let operand_bits = operand_value.to_bits().unwrap();
                    let mut result = false;
                    for i in 0..operand_bits.get_bit_count() {
                        if operand_bits.get_bit(i).unwrap() {
                            result = true;
                            break;
                        }
                    }
                    IrValue::bool(result)
                }
                ir::Unop::AndReduce => {
                    let operand_bits = operand_value.to_bits().unwrap();
                    let mut result = true;
                    for i in 0..operand_bits.get_bit_count() {
                        if !operand_bits.get_bit(i).unwrap() {
                            result = false;
                            break;
                        }
                    }
                    IrValue::bool(result)
                }
                ir::Unop::XorReduce => {
                    let operand_bits = operand_value.to_bits().unwrap();
                    let mut result = false;
                    for i in 0..operand_bits.get_bit_count() {
                        if operand_bits.get_bit(i).unwrap() {
                            result = !result;
                        }
                    }
                    IrValue::bool(result)
                }
                ir::Unop::Reverse => {
                    let operand_bits = operand_value.to_bits().unwrap();
                    let w = operand_bits.get_bit_count();
                    let mut outs: Vec<bool> = Vec::with_capacity(w);
                    for i in 0..w {
                        outs.push(operand_bits.get_bit(w - 1 - i).unwrap());
                    }
                    let out_bits = IrBits::from_lsb_is_0(&outs);
                    IrValue::from_bits(&out_bits)
                }
            }
        }
        ir::NodePayload::Tuple(ref elements) => {
            let values: Vec<IrValue> = elements
                .iter()
                .map(|e| env.get(e).unwrap().clone())
                .collect();
            IrValue::make_tuple(&values)
        }
        ir::NodePayload::TupleIndex { tuple, index } => {
            let tuple_value: &IrValue = env.get(&tuple).unwrap();
            tuple_value.get_element(index).unwrap()
        }
        ir::NodePayload::Array(ref elements) => {
            let values: Vec<IrValue> = elements
                .iter()
                .map(|e| env.get(e).unwrap().clone())
                .collect();
            IrValue::make_array(&values).unwrap()
        }
        ir::NodePayload::ArraySlice {
            array,
            start,
            width,
        } => {
            let arr = env.get(&array).unwrap().clone();
            let start_u = env.get(&start).unwrap().to_u64().unwrap() as usize;
            let len = arr.get_element_count().unwrap();
            assert!(len > 0, "ArraySlice: empty array not supported");
            let mut out_elems: Vec<IrValue> = Vec::with_capacity(width);
            for j in 0..width {
                let idx = start_u.saturating_add(j);
                let clamped = if idx >= len { len - 1 } else { idx };
                let v = arr.get_element(clamped).unwrap();
                out_elems.push(v);
            }
            IrValue::make_array(&out_elems).unwrap()
        }
        ir::NodePayload::ArrayUpdate {
            array,
            value,
            ref indices,
            assumed_in_bounds: _,
        } => {
            // Recursively updates the array `base` at the multi-index `idxs`
            // with `new_value`, returning a freshly constructed value.
            fn update_at_indices(base: &IrValue, idxs: &[usize], new_value: &IrValue) -> IrValue {
                if idxs.is_empty() {
                    return new_value.clone();
                }
                let idx = idxs[0];
                let count = base.get_element_count().unwrap();
                let mut elems: Vec<IrValue> = Vec::with_capacity(count);
                for i in 0..count {
                    let elem_i = base.get_element(i).unwrap();
                    if i == idx {
                        elems.push(update_at_indices(&elem_i, &idxs[1..], new_value));
                    } else {
                        elems.push(elem_i);
                    }
                }
                IrValue::make_array(&elems).unwrap()
            }

            let base = env.get(&array).unwrap().clone();
            let new_value = env.get(&value).unwrap().clone();
            let idxs: Vec<usize> = indices
                .iter()
                .map(|r| env.get(r).unwrap().to_u64().unwrap() as usize)
                .collect();
            update_at_indices(&base, &idxs, &new_value)
        }
        ir::NodePayload::ArrayIndex {
            array,
            ref indices,
            assumed_in_bounds: _,
        } => {
            // XLS semantics: out-of-bounds indices are clamped to the maximum in-bounds
            // index for the respective dimension. Note: this helper does not
            // surface `assumed_in_bounds` errors (those are handled in
            // `eval_fn_with_observer`, which can return Failure).
            let mut value = env.get(&array).unwrap().clone();
            for idx_ref in indices {
                let idx = env.get(idx_ref).unwrap().to_u64().unwrap() as usize;
                let count = value.get_element_count().unwrap();
                assert!(count > 0, "ArrayIndex: empty array not supported");
                let clamped = if idx >= count { count - 1 } else { idx };
                value = value.get_element(clamped).unwrap();
            }
            value
        }
        ir::NodePayload::DynamicBitSlice {
            ref arg,
            ref start,
            width,
        } => {
            let arg_bits: IrBits = env.get(arg).unwrap().to_bits().unwrap();
            let start_bits: IrBits = env.get(start).unwrap().to_bits().unwrap();
            let start_u = start_bits.to_u64().unwrap() as usize;
            let bit_count = arg_bits.get_bit_count();
            assert!(
                start_u + width <= bit_count,
                "DynamicBitSlice OOB: start={} width={} arg_width={}",
                start_u,
                width,
                bit_count
            );
            let r = arg_bits.width_slice(start_u as i64, width as i64);
            IrValue::from_bits(&r)
        }
        ir::NodePayload::ZeroExt { arg, new_bit_count } => {
            let arg_bits: IrBits = env.get(&arg).unwrap().to_bits().unwrap();
            let old_w = arg_bits.get_bit_count();
            let new_w = new_bit_count;
            if new_w == old_w {
                env.get(&arg).unwrap().clone()
            } else if new_w < old_w {
                let sliced = arg_bits.width_slice(0, new_w as i64);
                IrValue::from_bits(&sliced)
            } else {
                let mut outs: Vec<bool> = Vec::with_capacity(new_w);
                for i in 0..old_w {
                    outs.push(arg_bits.get_bit(i).unwrap());
                }
                for _ in old_w..new_w {
                    outs.push(false);
                }
                let out_bits = IrBits::from_lsb_is_0(&outs);
                IrValue::from_bits(&out_bits)
            }
        }
        ir::NodePayload::SignExt { arg, new_bit_count } => {
            let arg_bits: IrBits = env.get(&arg).unwrap().to_bits().unwrap();
            let old_w = arg_bits.get_bit_count();
            let new_w = new_bit_count;
            if new_w == old_w {
                env.get(&arg).unwrap().clone()
            } else if new_w < old_w {
                let sliced = arg_bits.width_slice(0, new_w as i64);
                IrValue::from_bits(&sliced)
            } else {
                let msb = if old_w == 0 {
                    false
                } else {
                    arg_bits.get_bit(old_w - 1).unwrap()
                };
                let mut outs: Vec<bool> = Vec::with_capacity(new_w);
                for i in 0..old_w {
                    outs.push(arg_bits.get_bit(i).unwrap());
                }
                for _ in old_w..new_w {
                    outs.push(msb);
                }
                let out_bits = IrBits::from_lsb_is_0(&outs);
                IrValue::from_bits(&out_bits)
            }
        }
        ir::NodePayload::BitSlice {
            ref arg,
            start,
            width,
        } => {
            let arg_bits: IrBits = env.get(arg).unwrap().to_bits().unwrap();
            let bit_count = arg_bits.get_bit_count();
            assert!(
                start + width <= bit_count,
                "BitSlice OOB: start={} width={} arg_width={}",
                start,
                width,
                bit_count
            );
            let r = arg_bits.width_slice(start as i64, width as i64);
            IrValue::from_bits(&r)
        }
        ir::NodePayload::Nary(op, ref operands) => {
            let mut iter = operands.iter();
            let first = env.get(iter.next().unwrap()).unwrap();
            let mut acc = first.to_bits().unwrap();
            match op {
                ir::NaryOp::And => {
                    for operand in iter {
                        let bits = env.get(operand).unwrap().to_bits().unwrap();
                        acc = acc.and(&bits);
                    }
                    IrValue::from_bits(&acc)
                }
                ir::NaryOp::Or => {
                    for operand in iter {
                        let bits = env.get(operand).unwrap().to_bits().unwrap();
                        acc = acc.or(&bits);
                    }
                    IrValue::from_bits(&acc)
                }
                ir::NaryOp::Xor => {
                    for operand in iter {
                        let bits = env.get(operand).unwrap().to_bits().unwrap();
                        acc = acc.xor(&bits);
                    }
                    IrValue::from_bits(&acc)
                }
                ir::NaryOp::Nand => {
                    for operand in iter {
                        let bits = env.get(operand).unwrap().to_bits().unwrap();
                        acc = acc.and(&bits);
                    }
                    let r = acc.not();
                    IrValue::from_bits(&r)
                }
                ir::NaryOp::Nor => {
                    for operand in iter {
                        let bits = env.get(operand).unwrap().to_bits().unwrap();
                        acc = acc.or(&bits);
                    }
                    let r = acc.not();
                    IrValue::from_bits(&r)
                }
                ir::NaryOp::Concat => {
                    // XLS concat semantics: concat(x0, x1, ..., xn-1) places x0 in the MSBs and
                    // xn-1 in the LSBs.
                    //
                    // Our `IrBits` is indexed with LSB at bit 0, so we build the output bits
                    // vector by appending operands in reverse order.
                    let mut out: Vec<bool> = Vec::new();
                    for operand in operands.iter().rev() {
                        let bits: IrBits = env.get(operand).unwrap().to_bits().unwrap();
                        for i in 0..bits.get_bit_count() {
                            out.push(bits.get_bit(i).unwrap());
                        }
                    }
                    let out_bits = IrBits::from_lsb_is_0(&out);
                    IrValue::from_bits(&out_bits)
                }
            }
        }
        ir::NodePayload::PrioritySel {
            selector,
            ref cases,
            default,
        } => {
            // We require a default arm for PrioritySel to be well-defined.
            let default_ref = default.expect("PrioritySel requires a default value");
            let mut result: IrValue = env
                .get(&default_ref)
                .expect("default must be evaluated")
                .clone();

            let sel_bits: IrBits = env
                .get(&selector)
                .expect("PrioritySel selector must be evaluated")
                .to_bits()
                .expect("PrioritySel selector must be bits");
            let sel_w = sel_bits.get_bit_count();

            // Highest index has lowest priority; index 0 has highest priority.
            for (idx, case_ref) in cases.iter().enumerate().rev() {
                if idx < sel_w {
                    let bit_set = sel_bits.get_bit(idx).expect("selector bit in range");
                    if bit_set {
                        let case_v: IrValue =
                            env.get(case_ref).expect("case must be evaluated").clone();
                        result = case_v;
                    }
                }
            }
            result
        }
        ir::NodePayload::OneHot { arg, lsb_prio } => {
            let arg_bits: IrBits = env.get(&arg).unwrap().to_bits().unwrap();
            let w = arg_bits.get_bit_count();
            assert!(w > 0, "OneHot: width must be > 0");
            let mut prior_clear = true;
            let mut outs: Vec<bool> = Vec::with_capacity(w + 1);
            // Generate w one-hot bits according to priority
            for i in 0..w {
                let idx = if lsb_prio { i } else { w - 1 - i };
                let bit = arg_bits.get_bit(idx).unwrap();
                let out_bit = bit && prior_clear;
                outs.push(out_bit);
                prior_clear = prior_clear && !bit;
            }
            if !lsb_prio {
                outs.reverse();
            }
            // Final bit is set when arg == 0
            outs.push(prior_clear);
            let out_bits = IrBits::from_lsb_is_0(&outs);
            IrValue::from_bits(&out_bits)
        }
        ir::NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => {
            let arg_bits: IrBits = env.get(&arg).unwrap().to_bits().unwrap();
            let start_bits: IrBits = env.get(&start).unwrap().to_bits().unwrap();
            let start_i = start_bits.to_u64().unwrap() as usize;
            let upd_bits: IrBits = env.get(&update_value).unwrap().to_bits().unwrap();
            let arg_w = arg_bits.get_bit_count();
            let upd_w = upd_bits.get_bit_count();
            assert!(
                start_i + upd_w <= arg_w,
                "BitSliceUpdate out of bounds: start={}, update_width={}, arg_width={}",
                start_i,
                upd_w,
                arg_w
            );
            let mut out: Vec<bool> = Vec::with_capacity(arg_w);
            for i in 0..arg_w {
                if i >= start_i && i < start_i + upd_w {
                    out.push(upd_bits.get_bit(i - start_i).unwrap());
                } else {
                    out.push(arg_bits.get_bit(i).unwrap());
                }
            }
            let out_bits = IrBits::from_lsb_is_0(&out);
            IrValue::from_bits(&out_bits)
        }
        ir::NodePayload::Sel {
            selector,
            ref cases,
            default,
        } => {
            assert!(!cases.is_empty(), "Sel must have at least one case");
            let sel_bits: IrBits = env.get(&selector).unwrap().to_bits().unwrap();
            let sel_w = sel_bits.get_bit_count();
            // Default result
            let mut result: IrValue = if let Some(dref) = default {
                env.get(&dref).unwrap().clone()
            } else {
                env.get(cases.last().unwrap()).unwrap().clone()
            };
            // Iterate cases in reverse and select when selector == index.
            for (i, case_ref) in cases.iter().enumerate().rev() {
                // Build index bits of selector width and compare.
                let idx_bits = if sel_w == 0 {
                    IrBits::make_ubits(0, 0).unwrap()
                } else {
                    IrBits::make_ubits(sel_w, i as u64).unwrap()
                };
                if sel_bits.equals(&idx_bits) {
                    result = env.get(case_ref).unwrap().clone();
                }
            }
            result
        }
        ir::NodePayload::OneHotSel {
            selector,
            ref cases,
        } => {
            assert!(!cases.is_empty(), "OneHotSel must have at least one case");
            let sel_bits: IrBits = env.get(&selector).unwrap().to_bits().unwrap();
            let sel_w = sel_bits.get_bit_count();
            // Initialize accumulator with zeros of the case width.
            let first_case_bits: IrBits = env.get(&cases[0]).unwrap().to_bits().unwrap();
            let case_w = first_case_bits.get_bit_count();
            let mut acc = IrBits::make_ubits(case_w, 0).unwrap();
            for (i, case_ref) in cases.iter().enumerate() {
                let case_bits: IrBits = env.get(case_ref).unwrap().to_bits().unwrap();
                let bit_set = if i < sel_w {
                    sel_bits.get_bit(i).unwrap()
                } else {
                    false
                };
                if bit_set {
                    acc = acc.or(&case_bits);
                }
            }
            IrValue::from_bits(&acc)
        }
        ir::NodePayload::GetParam(..) | _ => panic!("Cannot evaluate node as pure: {:?}", n),
    }
}

fn observed_type_string_for_expected(expected: &ir::Type, value: &IrValue) -> String {
    match expected {
        ir::Type::Bits(_) => match value.to_bits() {
            Ok(bits) => format!("bits[{}]", bits.get_bit_count()),
            Err(_) => "<non-bits>".to_string(),
        },
        ir::Type::Tuple(member_types) => match value.get_element_count() {
            Ok(count) => {
                let mut parts: Vec<String> = Vec::with_capacity(count);
                for i in 0..count {
                    let elem = value.get_element(i).unwrap();
                    // If we have fewer expected entries than elements, just reuse last expected
                    // kind.
                    let ety = member_types
                        .get(i)
                        .unwrap_or_else(|| member_types.last().unwrap());
                    parts.push(observed_type_string_for_expected(ety, &elem));
                }
                format!("({})", parts.join(", "))
            }
            Err(_) => "(<non-composite>)".to_string(),
        },
        ir::Type::Array(arr) => match value.get_element_count() {
            Ok(count) => {
                if count == 0 {
                    format!(
                        "{}[0]",
                        observed_type_string_for_expected(&arr.element_type, value)
                    )
                } else {
                    let first = value.get_element(0).unwrap();
                    let elem_ty = observed_type_string_for_expected(&arr.element_type, &first);
                    format!("{}[{}]", elem_ty, count)
                }
            }
            Err(_) => format!(
                "{}[?]",
                observed_type_string_for_expected(&arr.element_type, value)
            ),
        },
        ir::Type::Token => "token".to_string(),
    }
}

fn assert_value_conforms_to_type(expected: &ir::Type, value: &IrValue, node: &ir::Node) {
    match expected {
        ir::Type::Bits(width) => {
            let bits = value
                .to_bits()
                .expect("expected bits value to conform to bits type");
            let got_w = bits.get_bit_count();
            assert!(
                got_w == *width,
                "type mismatch at node id={}: expected={}, observed={}",
                node.text_id,
                expected,
                format!("bits[{}]", got_w)
            );
        }
        ir::Type::Tuple(member_types) => {
            let count = value
                .get_element_count()
                .expect("expected tuple value to have elements");
            assert!(
                count == member_types.len(),
                "type mismatch at node id={}: expected={}, observed={}",
                node.text_id,
                expected,
                observed_type_string_for_expected(expected, value)
            );
            for (i, mt) in member_types.iter().enumerate() {
                let elem = value
                    .get_element(i)
                    .expect("tuple element should be accessible");
                assert_value_conforms_to_type(mt, &elem, node);
            }
        }
        ir::Type::Array(arr) => {
            let count = value
                .get_element_count()
                .expect("expected array value to have elements");
            assert!(
                count == arr.element_count,
                "type mismatch at node id={}: expected={}, observed={}",
                node.text_id,
                expected,
                observed_type_string_for_expected(expected, value)
            );
            for i in 0..arr.element_count {
                let elem = value
                    .get_element(i)
                    .expect("array element should be accessible");
                assert_value_conforms_to_type(&arr.element_type, &elem, node);
            }
        }
        ir::Type::Token => {
            // We construct tokens explicitly in this interpreter; no runtime
            // tag to check here.
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceMessage {
    pub message: String,
    pub verbosity: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionFailure {
    pub message: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnEvalSuccess {
    pub value: IrValue,
    pub trace_messages: Vec<TraceMessage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnEvalFailure {
    pub assertion_failures: Vec<AssertionFailure>,
    pub trace_messages: Vec<TraceMessage>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FnEvalResult {
    Success(FnEvalSuccess),
    Failure(FnEvalFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectKind {
    CaseIndex,
    Default,
    NoBitsSet,
    MultiBitsSet,
}

/// Observed selection decision for select-like nodes (`sel`, `priority_sel`,
/// `one_hot_sel`).
///
/// `selected_index` is meaningful only for `CaseIndex`; for other kinds it is
/// set to `usize::MAX` as a sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelectEvent {
    pub node_ref: ir::NodeRef,
    pub node_text_id: usize,
    pub select_kind: SelectKind,
    pub selected_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoolNodeEvent {
    pub node_ref: ir::NodeRef,
    pub node_text_id: usize,
    pub value: bool,
}

pub trait EvalObserver {
    fn on_select(&mut self, ev: SelectEvent);

    fn on_bool_node(&mut self, _ev: BoolNodeEvent) {}

    fn on_node_value(&mut self, _node_ref: ir::NodeRef, _node_text_id: usize, _value: &IrValue) {}

    fn on_corner_event(&mut self, _ev: CornerEvent) {}

    fn on_failure_event(&mut self, _ev: FailureEvent) {}
}

fn observe_corner_like_node(
    f: &ir::Fn,
    nr: ir::NodeRef,
    node: &ir::Node,
    env: &HashMap<ir::NodeRef, IrValue>,
    observer: &mut dyn EvalObserver,
) {
    match &node.payload {
        ir::NodePayload::Binop(binop, lhs, rhs)
            if matches!(binop, ir::Binop::Eq | ir::Binop::Ne) =>
        {
            let lhs_bits = match env.get(lhs).unwrap().to_bits() {
                Ok(v) => v,
                Err(_) => return,
            };
            let rhs_bits = match env.get(rhs).unwrap().to_bits() {
                Ok(v) => v,
                Err(_) => return,
            };
            let w = lhs_bits.get_bit_count();
            if w != rhs_bits.get_bit_count() {
                return;
            }
            let mut d: usize = 0;
            for i in 0..w {
                let a = lhs_bits.get_bit(i).unwrap();
                let b = rhs_bits.get_bit(i).unwrap();
                if a ^ b {
                    d += 1;
                }
            }
            observer.on_corner_event(CornerEvent {
                node_ref: nr,
                node_text_id: node.text_id,
                kind: CornerKind::CompareDistance,
                tag: CompareDistanceCornerTag::try_from(bucket_xor_popcount(d))
                    .expect("bucket_xor_popcount must be 0..=7")
                    .into(),
            });
        }
        ir::NodePayload::Binop(ir::Binop::Add, lhs, rhs) => {
            fn is_literal_operand(f: &ir::Fn, r: ir::NodeRef) -> bool {
                matches!(f.get_node(r).payload, ir::NodePayload::Literal(_))
            }
            fn is_zero_value(v: &IrValue) -> bool {
                let bits = v.to_bits().unwrap();
                for i in 0..bits.get_bit_count() {
                    if bits.get_bit(i).unwrap() {
                        return false;
                    }
                }
                true
            }

            // Add corners: treat lhs==0 and rhs==0 as separate coverpoints.
            //
            // If an operand is a literal, suppress the corresponding coverpoint,
            // because it cannot vary across samples and would be misleading to
            // report as "missed" for the corpus.
            if !is_literal_operand(f, *lhs) {
                if is_zero_value(env.get(lhs).unwrap()) {
                    observer.on_corner_event(CornerEvent {
                        node_ref: nr,
                        node_text_id: node.text_id,
                        kind: CornerKind::Add,
                        tag: AddCornerTag::LhsIsZero.into(),
                    });
                }
            }
            if !is_literal_operand(f, *rhs) {
                if is_zero_value(env.get(rhs).unwrap()) {
                    observer.on_corner_event(CornerEvent {
                        node_ref: nr,
                        node_text_id: node.text_id,
                        kind: CornerKind::Add,
                        tag: AddCornerTag::RhsIsZero.into(),
                    });
                }
            }
        }
        ir::NodePayload::Unop(ir::Unop::Neg, operand) => {
            fn is_literal_operand(f: &ir::Fn, r: ir::NodeRef) -> bool {
                matches!(f.get_node(r).payload, ir::NodePayload::Literal(_))
            }

            // If the operand is a literal, suppress these coverpoints because the
            // operand cannot vary across samples.
            if is_literal_operand(f, *operand) {
                return;
            }

            let bits = env.get(operand).unwrap().to_bits().unwrap();
            let w = bits.get_bit_count();
            if w > 0 {
                let msb = bits.get_bit(w - 1).unwrap();
                if msb {
                    observer.on_corner_event(CornerEvent {
                        node_ref: nr,
                        node_text_id: node.text_id,
                        kind: CornerKind::Neg,
                        tag: NegCornerTag::OperandMsbIsOne.into(),
                    });

                    let mut all_lower_zero = true;
                    for i in 0..(w - 1) {
                        if bits.get_bit(i).unwrap() {
                            all_lower_zero = false;
                            break;
                        }
                    }
                    if all_lower_zero {
                        observer.on_corner_event(CornerEvent {
                            node_ref: nr,
                            node_text_id: node.text_id,
                            kind: CornerKind::Neg,
                            tag: NegCornerTag::OperandIsMinSigned.into(),
                        });
                    }
                }
            }
        }
        ir::NodePayload::SignExt { arg, new_bit_count } => {
            let arg_bits = env.get(arg).unwrap().to_bits().unwrap();
            let old_w = arg_bits.get_bit_count();
            if *new_bit_count > old_w && old_w > 0 {
                let msb = arg_bits.get_bit(old_w - 1).unwrap();
                if !msb {
                    observer.on_corner_event(CornerEvent {
                        node_ref: nr,
                        node_text_id: node.text_id,
                        kind: CornerKind::SignExt,
                        tag: SignExtCornerTag::MsbIsZero.into(),
                    });
                }
            }
        }
        ir::NodePayload::Binop(binop, lhs, rhs)
            if matches!(binop, ir::Binop::Shll | ir::Binop::Shrl) =>
        {
            let lhs_bits = env.get(lhs).unwrap().to_bits().unwrap();
            let rhs_bits = env.get(rhs).unwrap().to_bits().unwrap();
            let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
            let lhs_w = lhs_bits.get_bit_count() as i64;
            if shift == 0 {
                observer.on_corner_event(CornerEvent {
                    node_ref: nr,
                    node_text_id: node.text_id,
                    kind: CornerKind::Shift,
                    tag: ShiftCornerTag::AmtIsZero.into(),
                });
            }
            let tag: u8 = if shift < lhs_w {
                ShiftCornerTag::AmtLtWidth.into()
            } else {
                ShiftCornerTag::AmtGeWidth.into()
            };
            observer.on_corner_event(CornerEvent {
                node_ref: nr,
                node_text_id: node.text_id,
                kind: CornerKind::Shift,
                tag,
            });
        }
        ir::NodePayload::Binop(ir::Binop::Shra, lhs, rhs) => {
            let lhs_bits = env.get(lhs).unwrap().to_bits().unwrap();
            let rhs_bits = env.get(rhs).unwrap().to_bits().unwrap();
            let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
            let lhs_w = lhs_bits.get_bit_count() as i64;
            let amt_lt = shift < lhs_w;
            let msb_is_zero = if lhs_w <= 0 {
                true
            } else {
                !lhs_bits.get_bit((lhs_w - 1) as usize).unwrap()
            };
            let tag: u8 = match (msb_is_zero, amt_lt) {
                (true, true) => ShraCornerTag::Msb0AmtLt.into(),
                (true, false) => ShraCornerTag::Msb0AmtGe.into(),
                (false, true) => ShraCornerTag::Msb1AmtLt.into(),
                (false, false) => ShraCornerTag::Msb1AmtGe.into(),
            };
            observer.on_corner_event(CornerEvent {
                node_ref: nr,
                node_text_id: node.text_id,
                kind: CornerKind::Shra,
                tag,
            });
        }
        _ => {}
    }
}

fn observe_select_like_node(
    nr: ir::NodeRef,
    node: &ir::Node,
    env: &HashMap<ir::NodeRef, IrValue>,
    observer: &mut dyn EvalObserver,
) {
    match &node.payload {
        P::Sel {
            selector,
            cases,
            default,
        } => {
            assert!(!cases.is_empty(), "Sel must have at least one case");
            let sel_bits: IrBits = env
                .get(selector)
                .expect("Sel selector must be evaluated")
                .to_bits()
                .expect("Sel selector must be bits");
            let sel_w = sel_bits.get_bit_count();

            let mut chosen = SelectEvent {
                node_ref: nr,
                node_text_id: node.text_id,
                select_kind: if default.is_some() {
                    SelectKind::Default
                } else {
                    // No explicit default: XLS semantics use the last case as an implicit default.
                    SelectKind::Default
                },
                selected_index: usize::MAX,
            };

            // Iterate cases in reverse and select when selector == index.
            for (i, _case_ref) in cases.iter().enumerate().rev() {
                let idx_bits = if sel_w == 0 {
                    IrBits::make_ubits(0, 0).unwrap()
                } else {
                    IrBits::make_ubits(sel_w, i as u64).unwrap()
                };
                if sel_bits.equals(&idx_bits) {
                    chosen.select_kind = SelectKind::CaseIndex;
                    chosen.selected_index = i;
                }
            }
            observer.on_select(chosen);
        }
        P::PrioritySel {
            selector,
            cases,
            default,
        } => {
            // `eval_pure` requires a default arm; keep the same invariant here.
            default.expect("PrioritySel requires a default value");
            let sel_bits: IrBits = env
                .get(selector)
                .expect("PrioritySel selector must be evaluated")
                .to_bits()
                .expect("PrioritySel selector must be bits");
            let sel_w = sel_bits.get_bit_count();

            let mut chosen = SelectEvent {
                node_ref: nr,
                node_text_id: node.text_id,
                select_kind: SelectKind::Default,
                selected_index: usize::MAX,
            };

            // Highest index has lowest priority; index 0 has highest priority.
            for (idx, _case_ref) in cases.iter().enumerate().rev() {
                if idx < sel_w {
                    let bit_set = sel_bits.get_bit(idx).expect("selector bit in range");
                    if bit_set {
                        chosen.select_kind = SelectKind::CaseIndex;
                        chosen.selected_index = idx;
                    }
                }
            }
            observer.on_select(chosen);
        }
        P::OneHotSel { selector, cases } => {
            assert!(!cases.is_empty(), "OneHotSel must have at least one case");
            let sel_bits: IrBits = env
                .get(selector)
                .expect("OneHotSel selector must be evaluated")
                .to_bits()
                .expect("OneHotSel selector must be bits");
            let sel_w = sel_bits.get_bit_count();

            let mut any = false;
            let mut in_range_set_count: usize = 0;
            for (i, _case_ref) in cases.iter().enumerate() {
                let bit_set = if i < sel_w {
                    sel_bits.get_bit(i).expect("selector bit in range")
                } else {
                    false
                };
                if bit_set {
                    any = true;
                    in_range_set_count += 1;
                    observer.on_select(SelectEvent {
                        node_ref: nr,
                        node_text_id: node.text_id,
                        select_kind: SelectKind::CaseIndex,
                        selected_index: i,
                    });
                }
            }
            if !any {
                observer.on_select(SelectEvent {
                    node_ref: nr,
                    node_text_id: node.text_id,
                    select_kind: SelectKind::NoBitsSet,
                    selected_index: usize::MAX,
                });
            } else if in_range_set_count >= 2 {
                observer.on_select(SelectEvent {
                    node_ref: nr,
                    node_text_id: node.text_id,
                    select_kind: SelectKind::MultiBitsSet,
                    selected_index: usize::MAX,
                });
            }
        }
        _ => {}
    }
}

/// Evaluates an IR function by visiting nodes in topological order.
///
/// - Produces `TraceMessage`s for `trace` nodes whose `activated` predicate is
///   true.
/// - Records `AssertionFailure`s for `assert` nodes whose `activate` predicate
///   is true.
/// - Returns the value of the function's return node on success.
pub fn eval_fn_with_observer(
    f: &ir::Fn,
    args: &[IrValue],
    observer: Option<&mut dyn EvalObserver>,
) -> FnEvalResult {
    let observer = observer.map(|o| o as *mut (dyn EvalObserver + '_));
    eval_fn_impl(None, f, args, observer)
}

pub fn eval_fn_in_package(pkg: &ir::Package, f: &ir::Fn, args: &[IrValue]) -> FnEvalResult {
    eval_fn_impl(Some(pkg), f, args, None)
}

pub fn eval_fn_in_package_with_observer(
    pkg: &ir::Package,
    f: &ir::Fn,
    args: &[IrValue],
    observer: Option<&mut dyn EvalObserver>,
) -> FnEvalResult {
    let observer = observer.map(|o| o as *mut (dyn EvalObserver + '_));
    eval_fn_impl(Some(pkg), f, args, observer)
}

fn eval_fn_impl<'a>(
    pkg: Option<&ir::Package>,
    f: &ir::Fn,
    args: &[IrValue],
    observer: Option<*mut (dyn EvalObserver + 'a)>,
) -> FnEvalResult {
    debug_assert!(
        f.check_pir_layout_invariants().is_ok(),
        "PIR layout invariants violated for function '{}'",
        f.name
    );
    assert_eq!(
        args.len(),
        f.params.len(),
        "argument count must match params"
    );

    // Map ParamId -> argument value.
    let mut param_map: HashMap<ir::ParamId, IrValue> = HashMap::new();
    for (i, p) in f.params.iter().enumerate() {
        // ParamIds start at 1, but we map by the ParamId key directly.
        param_map.insert(p.id, args[i].clone());
    }

    let mut env: HashMap<ir::NodeRef, IrValue> = HashMap::with_capacity(f.nodes.len());
    let mut trace_messages: Vec<TraceMessage> = Vec::new();
    let mut assertion_failures: Vec<AssertionFailure> = Vec::new();

    for nr in get_topological(f) {
        let node = f.get_node(nr);
        let value: IrValue = match &node.payload {
            P::Nil => {
                // Reserved zero node: produce the unit tuple value.
                IrValue::make_tuple(&[])
            }
            P::GetParam(param_id) => {
                // Must exist by construction.
                param_map.get(param_id).expect("param not found").clone()
            }
            P::Assert {
                token,
                activate,
                message,
                label,
            } => {
                let _token_val = env.get(token).cloned().unwrap_or_else(IrValue::make_token);
                let active = env
                    .get(activate)
                    .expect("assert activate operand must be evaluated")
                    .to_bool()
                    .expect("activate must be bits[1]");
                if active {
                    if let Some(observer) = observer {
                        unsafe {
                            (&mut *observer).on_failure_event(FailureEvent {
                                node_ref: nr,
                                node_text_id: node.text_id,
                                kind: FailureKind::AssertTriggered,
                                tag: 0,
                            });
                        }
                    }
                    assertion_failures.push(AssertionFailure {
                        message: message.clone(),
                        label: label.clone(),
                    });
                }
                // The result of assert is a token. Tokens are zero-sized; produce a token.
                IrValue::make_token()
            }
            P::Trace {
                token,
                activated,
                format,
                operands,
            } => {
                let _token_val = env.get(token).cloned().unwrap_or_else(IrValue::make_token);
                let is_active = env
                    .get(activated)
                    .expect("trace activated operand must be evaluated")
                    .to_bool()
                    .expect("activated must be bits[1]");
                if is_active {
                    // Build a message by replacing sequential `{}` occurrences with operand values
                    // in default formatting. If placeholders are fewer than operands, extra
                    // operands are appended in bracketed list for visibility.
                    let mut msg = String::new();
                    let mut parts = format.split("{}");
                    let mut first = true;
                    let mut op_iter = operands.iter();
                    loop {
                        let part = parts.next();
                        if part.is_none() {
                            break;
                        }
                        let part = part.unwrap();
                        if !first {
                            if let Some(op_ref) = op_iter.next() {
                                let v = env.get(op_ref).expect("trace operand must be evaluated");
                                msg.push_str(&v.to_string());
                            }
                        }
                        first = false;
                        msg.push_str(part);
                    }
                    // Append any remaining operands if there were more operands than `{}`.
                    let remaining: Vec<String> = op_iter
                        .map(|r| {
                            env.get(r)
                                .expect("trace operand must be evaluated")
                                .to_string()
                        })
                        .collect();
                    if !remaining.is_empty() {
                        msg.push_str(" [");
                        msg.push_str(&remaining.join(", "));
                        msg.push(']');
                    }
                    trace_messages.push(TraceMessage {
                        message: msg,
                        // XLS trace nodes in this IR do not carry verbosity; use 0.
                        verbosity: 0,
                    });
                }
                // The result of trace is a token.
                IrValue::make_token()
            }
            P::AfterAll(_deps) => {
                // Tokens have no payload; produce a fresh token value.
                IrValue::make_token()
            }
            P::Invoke { to_apply, operands } => {
                let pkg = pkg.expect("Invoke requires package context for callee resolution");
                let callee = pkg
                    .get_fn(to_apply)
                    .expect("Invoke callee must exist in package");
                let callee_args: Vec<IrValue> = operands
                    .iter()
                    .map(|r| {
                        env.get(r)
                            .expect("invoke operand must be evaluated")
                            .clone()
                    })
                    .collect();

                let callee_result = eval_fn_impl(Some(pkg), callee, &callee_args, observer);
                match callee_result {
                    FnEvalResult::Success(success) => {
                        trace_messages.extend(success.trace_messages);
                        success.value
                    }
                    FnEvalResult::Failure(fail) => {
                        assertion_failures.extend(fail.assertion_failures);
                        trace_messages.extend(fail.trace_messages);
                        return FnEvalResult::Failure(FnEvalFailure {
                            assertion_failures,
                            trace_messages,
                        });
                    }
                }
            }
            P::BitSlice { arg, start, width } => {
                // Guard against OOB; return Failure instead of panicking.
                let arg_bits = env
                    .get(arg)
                    .expect("arg must be evaluated")
                    .to_bits()
                    .unwrap();
                let bit_count = arg_bits.get_bit_count();
                if start + width > bit_count {
                    if let Some(observer) = observer {
                        unsafe {
                            (&mut *observer).on_failure_event(FailureEvent {
                                node_ref: nr,
                                node_text_id: node.text_id,
                                kind: FailureKind::BitSliceOob,
                                tag: 0,
                            });
                        }
                    }
                    return FnEvalResult::Failure(FnEvalFailure {
                        assertion_failures,
                        trace_messages,
                    });
                }
                let r = arg_bits.width_slice(*start as i64, *width as i64);
                IrValue::from_bits(&r)
            }
            P::DynamicBitSlice { arg, start, width } => {
                // Guard against OOB; return Failure instead of panicking.
                let arg_bits = env
                    .get(arg)
                    .expect("arg must be evaluated")
                    .to_bits()
                    .unwrap();
                let start_bits = env
                    .get(start)
                    .expect("start must be evaluated")
                    .to_bits()
                    .unwrap();
                let start_u = start_bits.to_u64().unwrap() as usize;
                let bit_count = arg_bits.get_bit_count();
                if start_u + *width > bit_count {
                    if let Some(observer) = observer {
                        unsafe {
                            (&mut *observer).on_corner_event(CornerEvent {
                                node_ref: nr,
                                node_text_id: node.text_id,
                                kind: CornerKind::DynamicBitSlice,
                                tag: DynamicBitSliceCornerTag::OutOfBounds.into(),
                            });
                            (&mut *observer).on_failure_event(FailureEvent {
                                node_ref: nr,
                                node_text_id: node.text_id,
                                kind: FailureKind::DynamicBitSliceOob,
                                tag: 0,
                            });
                        }
                    }
                    return FnEvalResult::Failure(FnEvalFailure {
                        assertion_failures,
                        trace_messages,
                    });
                }
                if let Some(observer) = observer {
                    unsafe {
                        (&mut *observer).on_corner_event(CornerEvent {
                            node_ref: nr,
                            node_text_id: node.text_id,
                            kind: CornerKind::DynamicBitSlice,
                            tag: DynamicBitSliceCornerTag::InBounds.into(),
                        });
                    }
                }
                let r = arg_bits.width_slice(start_u as i64, *width as i64);
                IrValue::from_bits(&r)
            }
            P::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                // Guard against OOB; return Failure instead of panicking.
                let arg_bits = env
                    .get(arg)
                    .expect("arg must be evaluated")
                    .to_bits()
                    .unwrap();
                let start_bits = env
                    .get(start)
                    .expect("start must be evaluated")
                    .to_bits()
                    .unwrap();
                let upd_bits = env
                    .get(update_value)
                    .expect("update_value must be evaluated")
                    .to_bits()
                    .unwrap();
                let start_u = start_bits.to_u64().unwrap() as usize;
                let arg_w = arg_bits.get_bit_count();
                let upd_w = upd_bits.get_bit_count();
                if start_u + upd_w > arg_w {
                    if let Some(observer) = observer {
                        unsafe {
                            (&mut *observer).on_failure_event(FailureEvent {
                                node_ref: nr,
                                node_text_id: node.text_id,
                                kind: FailureKind::BitSliceUpdateOob,
                                tag: 0,
                            });
                        }
                    }
                    return FnEvalResult::Failure(FnEvalFailure {
                        assertion_failures,
                        trace_messages,
                    });
                }
                let mut outs: Vec<bool> = Vec::with_capacity(arg_w);
                for i in 0..arg_w {
                    if i >= start_u && i < start_u + upd_w {
                        outs.push(upd_bits.get_bit(i - start_u).unwrap());
                    } else {
                        outs.push(arg_bits.get_bit(i).unwrap());
                    }
                }
                let out_bits = IrBits::from_lsb_is_0(&outs);
                IrValue::from_bits(&out_bits)
            }
            P::Decode { arg, width: _ } => {
                // Produce a one-hot vector of the node's annotated width, setting bit[arg].
                let arg_bits = env
                    .get(arg)
                    .expect("arg must be evaluated")
                    .to_bits()
                    .unwrap();
                let arg_u = arg_bits.to_u64().unwrap() as usize;
                let expected_w = match node.ty {
                    ir::Type::Bits(w) => w,
                    _ => 0,
                };
                let mut outs: Vec<bool> = vec![false; expected_w];
                if arg_u < expected_w {
                    outs[arg_u] = true;
                }
                let out_bits = IrBits::from_lsb_is_0(&outs);
                IrValue::from_bits(&out_bits)
            }
            P::Encode { arg } => {
                // XLS `encode`: converts a bits-typed argument (commonly one-hot) into a
                // compact bits value (the selected index). The output width is taken from
                // the node's annotated type.
                let arg_bits = env
                    .get(arg)
                    .expect("arg must be evaluated")
                    .to_bits()
                    .unwrap();
                let in_w = arg_bits.get_bit_count();
                let expected_w = match node.ty {
                    ir::Type::Bits(w) => w,
                    _ => 0,
                };
                if expected_w == 0 {
                    IrValue::make_ubits(0, 0).expect("make ubits[0]")
                } else {
                    // XLS encode semantics: bitwise-OR together the indices of all set bits.
                    //
                    // Example: for a 6-bit input, indices 5 (0b101) and 2 (0b010) set yields
                    // output 0b111 (7) in bits[3].
                    let mut encoded: u64 = 0;
                    for i in 0..in_w {
                        if arg_bits.get_bit(i).unwrap() {
                            encoded |= i as u64;
                        }
                    }
                    IrValue::make_ubits(expected_w, encoded).expect("make encoded ubits")
                }
            }
            P::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => {
                // XLS semantics: out-of-bounds indices are clamped to the maximum in-bounds
                // index for the respective dimension. If `assumed_in_bounds` is
                // true, any OOB index is considered a sample failure and we
                // return Failure.
                let mut value = env.get(array).expect("array must be evaluated").clone();
                let mut clamped_any = false;
                for idx_ref in indices.iter() {
                    let idx = env
                        .get(idx_ref)
                        .expect("index must be evaluated")
                        .to_bits()
                        .unwrap()
                        .to_u64()
                        .unwrap() as usize;
                    let count = value.get_element_count().expect("array must have elements");
                    assert!(count > 0, "ArrayIndex: empty array not supported");
                    if idx >= count {
                        if *assumed_in_bounds {
                            if let Some(observer) = observer {
                                unsafe {
                                    (&mut *observer).on_failure_event(FailureEvent {
                                        node_ref: nr,
                                        node_text_id: node.text_id,
                                        kind: FailureKind::ArrayIndexOobAssumedInBounds,
                                        tag: 0,
                                    });
                                }
                            }
                            return FnEvalResult::Failure(FnEvalFailure {
                                assertion_failures,
                                trace_messages,
                            });
                        }
                        clamped_any = true;
                        value = value.get_element(count - 1).unwrap();
                    } else {
                        value = value.get_element(idx).unwrap();
                    }
                }
                if let Some(observer) = observer {
                    unsafe {
                        (&mut *observer).on_corner_event(CornerEvent {
                            node_ref: nr,
                            node_text_id: node.text_id,
                            kind: CornerKind::ArrayIndex,
                            tag: if clamped_any {
                                ArrayIndexCornerTag::Clamped.into()
                            } else {
                                ArrayIndexCornerTag::InBounds.into()
                            },
                        });
                    }
                }
                value
            }
            P::ArrayUpdate {
                array,
                value,
                indices,
                assumed_in_bounds,
            } => {
                // XLS semantics: if any index is out of bounds, the result is identical to the
                // input array, unless `assumed_in_bounds` is true, in which case OOB is an
                // error.
                let arr = env.get(array).expect("array must be evaluated").clone();
                let val = env.get(value).expect("value must be evaluated").clone();
                // Gather concrete indices as usize.
                let mut idxs: Vec<usize> = Vec::with_capacity(indices.len());
                for r in indices.iter() {
                    let u = env
                        .get(r)
                        .expect("index must be evaluated")
                        .to_bits()
                        .unwrap()
                        .to_u64()
                        .unwrap() as usize;
                    idxs.push(u);
                }

                // Recursively update nested arrays; returns None on OOB.
                fn set_at_path(
                    cur: &IrValue,
                    path: &[usize],
                    new_val: &IrValue,
                ) -> Option<IrValue> {
                    if path.is_empty() {
                        return Some(new_val.clone());
                    }
                    let count = cur.get_element_count().ok()?;
                    let idx = path[0];
                    if idx >= count {
                        return None;
                    }
                    let mut elems: Vec<IrValue> = Vec::with_capacity(count);
                    for i in 0..count {
                        let child = cur.get_element(i).ok()?;
                        if i == idx {
                            let updated = set_at_path(&child, &path[1..], new_val)?;
                            elems.push(updated);
                        } else {
                            elems.push(child);
                        }
                    }
                    IrValue::make_array(&elems).ok()
                }

                match set_at_path(&arr, &idxs, &val) {
                    Some(updated) => updated,
                    None => {
                        if *assumed_in_bounds {
                            if let Some(observer) = observer {
                                unsafe {
                                    (&mut *observer).on_failure_event(FailureEvent {
                                        node_ref: nr,
                                        node_text_id: node.text_id,
                                        kind: FailureKind::ArrayUpdateOobAssumedInBounds,
                                        tag: 0,
                                    });
                                }
                            }
                            return FnEvalResult::Failure(FnEvalFailure {
                                assertion_failures,
                                trace_messages,
                            });
                        } else {
                            // OOB but not assumed in-bounds: return the original array unchanged.
                            arr
                        }
                    }
                }
            }
            _ => {
                if let Some(observer) = observer {
                    unsafe {
                        observe_select_like_node(nr, node, &env, &mut *observer);
                        observe_corner_like_node(f, nr, node, &env, &mut *observer);
                    }
                }
                eval_pure(node, &env)
            }
        };
        // Coerce only for specific nodes where a wider internal computation is
        // permitted by semantics but the annotated type narrows (e.g. smul/umul,
        // or decode results). For other nodes we do not coerce.
        let coerced: IrValue = match &node.payload {
            ir::NodePayload::Binop(binop, _, _)
                if matches!(
                    binop,
                    ir::Binop::Smul | ir::Binop::Umul | ir::Binop::Smulp | ir::Binop::Umulp
                ) =>
            {
                match (&node.ty, value.to_bits()) {
                    (ir::Type::Bits(expected_w), Ok(bits)) => {
                        let got_w = bits.get_bit_count();
                        if got_w > *expected_w {
                            let sliced = bits.width_slice(0, *expected_w as i64);
                            IrValue::from_bits(&sliced)
                        } else {
                            value.clone()
                        }
                    }
                    _ => value.clone(),
                }
            }
            ir::NodePayload::Decode { .. } => match (&node.ty, value.to_bits()) {
                (ir::Type::Bits(expected_w), Ok(bits)) => {
                    let got_w = bits.get_bit_count();
                    if got_w > *expected_w {
                        let sliced = bits.width_slice(0, *expected_w as i64);
                        IrValue::from_bits(&sliced)
                    } else {
                        value.clone()
                    }
                }
                _ => value.clone(),
            },
            _ => value.clone(),
        };
        // Verify the computed value conforms to the node's annotated type, include node
        // context.
        assert_value_conforms_to_type(&node.ty, &coerced, node);
        if let Some(observer) = observer {
            let observer = unsafe { &mut *observer };
            let is_param = matches!(node.payload, ir::NodePayload::GetParam(_));
            let is_nil = matches!(node.payload, ir::NodePayload::Nil);
            if !is_param && !is_nil {
                observer.on_node_value(nr, node.text_id, &coerced);
            }
            let is_bool_node = matches!(node.ty, ir::Type::Bits(1));
            if is_bool_node && !is_param {
                observer.on_bool_node(BoolNodeEvent {
                    node_ref: nr,
                    node_text_id: node.text_id,
                    value: coerced
                        .to_bool()
                        .expect("bits[1] nodes must be convertible to bool"),
                });
            }
        }
        env.insert(nr, coerced);
    }

    let ret_ref = f
        .ret_node_ref
        .expect("function must have a designated return node");
    let ret_value = env
        .get(&ret_ref)
        .expect("return value must have been computed")
        .clone();

    if assertion_failures.is_empty() {
        FnEvalResult::Success(FnEvalSuccess {
            value: ret_value,
            trace_messages,
        })
    } else {
        FnEvalResult::Failure(FnEvalFailure {
            assertion_failures,
            trace_messages,
        })
    }
}

pub fn eval_fn(f: &ir::Fn, args: &[IrValue]) -> FnEvalResult {
    eval_fn_with_observer(f, args, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;
    use maplit::hashmap;

    struct RecordingObserver {
        events: Vec<SelectEvent>,
    }

    impl RecordingObserver {
        fn new() -> Self {
            Self { events: Vec::new() }
        }
    }

    impl EvalObserver for RecordingObserver {
        fn on_select(&mut self, ev: SelectEvent) {
            self.events.push(ev);
        }
    }

    struct RecordingBoolObserver {
        bool_events: Vec<(usize, bool)>,
    }

    impl RecordingBoolObserver {
        fn new() -> Self {
            Self {
                bool_events: Vec::new(),
            }
        }
    }

    impl EvalObserver for RecordingBoolObserver {
        fn on_select(&mut self, _ev: SelectEvent) {}

        fn on_bool_node(&mut self, ev: BoolNodeEvent) {
            self.bool_events.push((ev.node_text_id, ev.value));
        }
    }

    struct RecordingNodeValueObserver {
        lines: Vec<String>,
    }

    impl RecordingNodeValueObserver {
        fn new() -> Self {
            Self { lines: Vec::new() }
        }
    }

    impl EvalObserver for RecordingNodeValueObserver {
        fn on_select(&mut self, _ev: SelectEvent) {}

        fn on_node_value(&mut self, _node_ref: ir::NodeRef, node_text_id: usize, value: &IrValue) {
            self.lines
                .push(format!("node_text_id={} value={}", node_text_id, value));
        }
    }

    #[derive(Default)]
    struct RecordingCornerFailureObserver {
        corner_events: Vec<(usize, CornerKind, u8)>,
        failure_events: Vec<(usize, FailureKind)>,
    }

    impl EvalObserver for RecordingCornerFailureObserver {
        fn on_select(&mut self, _ev: SelectEvent) {}

        fn on_corner_event(&mut self, ev: CornerEvent) {
            self.corner_events.push((ev.node_text_id, ev.kind, ev.tag));
        }

        fn on_failure_event(&mut self, ev: FailureEvent) {
            self.failure_events.push((ev.node_text_id, ev.kind));
        }
    }

    #[test]
    fn test_eval_pure_literal() {
        let env = HashMap::new();
        let ir_value = IrValue::make_ubits(32, 1).unwrap();
        let n = ir::Node {
            text_id: 0,
            name: None,
            ty: ir::Type::Bits(32),
            payload: ir::NodePayload::Literal(ir_value.clone()),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, ir_value);
    }

    #[test]
    fn test_eval_pure_binop_add() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(32, 1).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(32, 2).unwrap(),
        );
        let n = ir::Node {
            text_id: 3,
            name: None,
            ty: ir::Type::Bits(32),
            payload: ir::NodePayload::Binop(
                ir::Binop::Add,
                ir::NodeRef { index: 1 },
                ir::NodeRef { index: 2 },
            ),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(32, 3).unwrap());
    }

    #[test]
    fn test_eval_pure_unop_not() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(4, 0b0101).unwrap(),
        );
        let n = ir::Node {
            text_id: 2,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::Unop(ir::Unop::Not, ir::NodeRef { index: 1 }),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b1010).unwrap());
    }

    #[test]
    fn test_eval_pure_unop_reverse_basic() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(4, 0b0101).unwrap(),
        );
        let n = ir::Node {
            text_id: 2001,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::Unop(ir::Unop::Reverse, ir::NodeRef { index: 1 }),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b1010).unwrap());
    }

    #[test]
    fn test_eval_pure_eq_tuple() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => {
                let a = IrValue::make_ubits(2, 0b01).unwrap();
                let b = IrValue::make_ubits(1, 0b1).unwrap();
                IrValue::make_tuple(&[a, b])
            },
            ir::NodeRef { index: 2 } => {
                let a = IrValue::make_ubits(2, 0b01).unwrap();
                let b = IrValue::make_ubits(1, 0b1).unwrap();
                IrValue::make_tuple(&[a, b])
            },
        );
        let n = ir::Node {
            text_id: 2002,
            name: None,
            ty: ir::Type::Bits(1),
            payload: ir::NodePayload::Binop(
                ir::Binop::Eq,
                ir::NodeRef { index: 1 },
                ir::NodeRef { index: 2 },
            ),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::bool(true));
    }

    #[test]
    fn test_eval_pure_ne_tuple() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => {
                let a = IrValue::make_ubits(2, 0b01).unwrap();
                let b = IrValue::make_ubits(1, 0b1).unwrap();
                IrValue::make_tuple(&[a, b])
            },
            ir::NodeRef { index: 2 } => {
                let a = IrValue::make_ubits(2, 0b10).unwrap();
                let b = IrValue::make_ubits(1, 0b1).unwrap();
                IrValue::make_tuple(&[a, b])
            },
        );
        let n = ir::Node {
            text_id: 2003,
            name: None,
            ty: ir::Type::Bits(1),
            payload: ir::NodePayload::Binop(
                ir::Binop::Ne,
                ir::NodeRef { index: 1 },
                ir::NodeRef { index: 2 },
            ),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::bool(true));
    }

    #[test]
    fn test_eval_pure_tuple() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 1).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 2).unwrap(),
        );
        let n = ir::Node {
            text_id: 3,
            name: None,
            ty: ir::Type::Tuple(vec![
                Box::new(ir::Type::Bits(8)),
                Box::new(ir::Type::Bits(8)),
            ]),
            payload: ir::NodePayload::Tuple(vec![
                ir::NodeRef { index: 1 },
                ir::NodeRef { index: 2 },
            ]),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(
            v,
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, 1).unwrap(),
                IrValue::make_ubits(8, 2).unwrap(),
            ])
        );
    }

    #[test]
    fn test_eval_pure_array_index() {
        let array_val = IrValue::make_array(&[
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
        ])
        .unwrap();
        let env = hashmap!(
            ir::NodeRef { index: 1 } => array_val,
            ir::NodeRef { index: 2 } => IrValue::make_ubits(32, 1).unwrap(),
        );
        let n = ir::Node {
            text_id: 4,
            name: None,
            ty: ir::Type::Bits(8),
            payload: ir::NodePayload::ArrayIndex {
                array: ir::NodeRef { index: 1 },
                indices: vec![ir::NodeRef { index: 2 }],
                assumed_in_bounds: true,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(8, 1).unwrap());
    }

    #[test]
    fn test_eval_pure_bit_slice() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 0b11110000).unwrap(),
        );
        let n = ir::Node {
            text_id: 5,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::BitSlice {
                arg: ir::NodeRef { index: 1 },
                start: 4,
                width: 4,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b1111).unwrap());
    }

    #[test]
    fn test_eval_pure_dynamic_bit_slice() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 0b11110000).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 4).unwrap(),
        );
        let n = ir::Node {
            text_id: 6,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::DynamicBitSlice {
                arg: ir::NodeRef { index: 1 },
                start: ir::NodeRef { index: 2 },
                width: 4,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b1111).unwrap());
    }

    #[test]
    fn test_eval_pure_nary_or() {
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(4, 0b0101).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(4, 0b1010).unwrap(),
        );
        let n = ir::Node {
            text_id: 7,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::Nary(
                ir::NaryOp::Or,
                vec![ir::NodeRef { index: 1 }, ir::NodeRef { index: 2 }],
            ),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b1111).unwrap());
    }

    #[test]
    fn test_eval_pure_nary_concat_places_first_operand_in_msbs() {
        // concat(a, b) where a=0b10 (2 bits) and b=0b01 (2 bits) => 0b1001.
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(2, 0b10).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(2, 0b01).unwrap(),
        );
        let n = ir::Node {
            text_id: 8,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::Nary(
                ir::NaryOp::Concat,
                vec![ir::NodeRef { index: 1 }, ir::NodeRef { index: 2 }],
            ),
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b1001).unwrap());
    }

    #[test]
    fn test_eval_pure_priority_sel_basic() {
        // selector: bits[2]:01, cases: [3, 5], default: 7 => select index 1 => 5
        let env = hashmap!(
            ir::NodeRef { index: 0 } => IrValue::make_ubits(2, 0b01).unwrap(),
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 3).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 5).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(8, 7).unwrap(),
        );
        let n = ir::Node {
            text_id: 1,
            name: None,
            ty: ir::Type::Bits(8),
            payload: ir::NodePayload::PrioritySel {
                selector: ir::NodeRef { index: 0 },
                cases: vec![ir::NodeRef { index: 1 }, ir::NodeRef { index: 2 }],
                default: Some(ir::NodeRef { index: 3 }),
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(8, 3).unwrap());
    }

    #[test]
    fn test_eval_pure_one_hot_lsb() {
        // arg: 0b010 -> onehot: [0,1,0,0] => bits[4]:0b0010
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(3, 0b010).unwrap()
        );
        let n = ir::Node {
            text_id: 2,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::OneHot {
                arg: ir::NodeRef { index: 1 },
                lsb_prio: true,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b0010).unwrap());
    }

    #[test]
    fn test_eval_pure_bit_slice_update() {
        // arg: 0b0000, start=1, update=0b11 => 0b1110
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(4, 0b0000).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(2, 0b11).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(32, 1).unwrap(),
        );
        let n = ir::Node {
            text_id: 3,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::BitSliceUpdate {
                arg: ir::NodeRef { index: 1 },
                start: ir::NodeRef { index: 3 },
                update_value: ir::NodeRef { index: 2 },
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, IrValue::make_ubits(4, 0b0110).unwrap());
    }

    #[test]
    fn test_eval_pure_zero_ext_and_sign_ext() {
        // zero extend 0b11 (2 bits) to 4 => 0b0011
        let env_ze = hashmap!( ir::NodeRef { index: 1 } => IrValue::make_ubits(2, 0b11).unwrap());
        let n_ze = ir::Node {
            text_id: 4,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::ZeroExt {
                arg: ir::NodeRef { index: 1 },
                new_bit_count: 4,
            },
            pos: None,
        };
        assert_eq!(
            eval_pure(&n_ze, &env_ze),
            IrValue::make_ubits(4, 0b0011).unwrap()
        );

        // sign extend 0b10 (2 bits, negative) to 4 => 0b1110
        let env_se = hashmap!( ir::NodeRef { index: 1 } => IrValue::make_ubits(2, 0b10).unwrap());
        let n_se = ir::Node {
            text_id: 5,
            name: None,
            ty: ir::Type::Bits(4),
            payload: ir::NodePayload::SignExt {
                arg: ir::NodeRef { index: 1 },
                new_bit_count: 4,
            },
            pos: None,
        };
        assert_eq!(
            eval_pure(&n_se, &env_se),
            IrValue::make_ubits(4, 0b1110).unwrap()
        );
    }

    #[test]
    fn test_eval_pure_sel_basic() {
        // selector: 1, cases [3, 5], default 7 => pick 5
        let env = hashmap!(
            ir::NodeRef { index: 0 } => IrValue::make_ubits(2, 1).unwrap(),
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 3).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 5).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(8, 7).unwrap(),
        );
        let n = ir::Node {
            text_id: 6,
            name: None,
            ty: ir::Type::Bits(8),
            payload: ir::NodePayload::Sel {
                selector: ir::NodeRef { index: 0 },
                cases: vec![ir::NodeRef { index: 1 }, ir::NodeRef { index: 2 }],
                default: Some(ir::NodeRef { index: 3 }),
            },
            pos: None,
        };
        assert_eq!(eval_pure(&n, &env), IrValue::make_ubits(8, 5).unwrap());
    }

    #[test]
    fn test_eval_pure_one_hot_sel_basic() {
        // selector one-hot 0b010 selects the middle case
        let env = hashmap!(
            ir::NodeRef { index: 0 } => IrValue::make_ubits(3, 0b010).unwrap(),
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 0x03).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 0x05).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(8, 0x0A).unwrap(),
        );
        let n = ir::Node {
            text_id: 7,
            name: None,
            ty: ir::Type::Bits(8),
            payload: ir::NodePayload::OneHotSel {
                selector: ir::NodeRef { index: 0 },
                cases: vec![
                    ir::NodeRef { index: 1 },
                    ir::NodeRef { index: 2 },
                    ir::NodeRef { index: 3 },
                ],
            },
            pos: None,
        };
        assert_eq!(eval_pure(&n, &env), IrValue::make_ubits(8, 0x05).unwrap());

        // Multiple bits set (0b101) => OR of case0 | case2 => 0x03 | 0x0A = 0x0B
        let env2 = hashmap!(
            ir::NodeRef { index: 0 } => IrValue::make_ubits(3, 0b101).unwrap(),
            ir::NodeRef { index: 1 } => IrValue::make_ubits(8, 0x03).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 0x05).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(8, 0x0A).unwrap(),
        );
        assert_eq!(eval_pure(&n, &env2), IrValue::make_ubits(8, 0x0B).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_eval_pure_bit_slice_oob_panics() {
        // arg width 3, slice start=3, width=1 => OOB, should panic
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(3, 0b101).unwrap(),
        );
        let n = ir::Node {
            text_id: 8,
            name: None,
            ty: ir::Type::Bits(1),
            payload: ir::NodePayload::BitSlice {
                arg: ir::NodeRef { index: 1 },
                start: 3,
                width: 1,
            },
            pos: None,
        };
        let _ = eval_pure(&n, &env);
    }

    #[test]
    #[should_panic]
    fn test_eval_pure_dynamic_bit_slice_oob_panics() {
        // arg width 3, dynamic start=3, width=1 => OOB, should panic
        let env = hashmap!(
            ir::NodeRef { index: 1 } => IrValue::make_ubits(3, 0b101).unwrap(),
            ir::NodeRef { index: 2 } => IrValue::make_ubits(32, 3).unwrap(),
        );
        let n = ir::Node {
            text_id: 9,
            name: None,
            ty: ir::Type::Bits(1),
            payload: ir::NodePayload::DynamicBitSlice {
                arg: ir::NodeRef { index: 1 },
                start: ir::NodeRef { index: 2 },
                width: 1,
            },
            pos: None,
        };
        let _ = eval_pure(&n, &env);
    }

    #[test]
    fn test_eval_fn_trace_and_assert() {
        let ir_text = r#"package test

fn f(x: bits[1] id=1) -> bits[1] {
  t: token = after_all(id=2)
  _tr: token = trace(t, x, format="x={} done", data_operands=[x], id=3)
  _a: token = assert(t, x, message="boom", label="L", id=4)
  ret literal.5: bits[1] = literal(value=1, id=5)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        // Case 1: x = 0 => no assert triggered, one trace inactive, success
        let x0 = IrValue::make_ubits(1, 0).unwrap();
        let res0 = eval_fn(&f, &[x0]);
        match res0 {
            FnEvalResult::Success(success) => {
                assert_eq!(success.value, IrValue::make_ubits(1, 1).unwrap());
                assert!(success.trace_messages.is_empty());
            }
            other => panic!("unexpected result: {:?}", other),
        }

        // Case 2: x = 1 => assert fires and trace emits
        let x1 = IrValue::make_ubits(1, 1).unwrap();
        let res1 = eval_fn(&f, &[x1]);
        match res1 {
            FnEvalResult::Failure(fail) => {
                assert_eq!(fail.assertion_failures.len(), 1);
                assert_eq!(fail.assertion_failures[0].message, "boom");
                assert_eq!(fail.assertion_failures[0].label, "L");
                assert_eq!(fail.trace_messages.len(), 1);
                assert_eq!(fail.trace_messages[0].message, "x=bits[1]:1 done");
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_fn_with_observer_none_matches_eval_fn() {
        let ir_text = r#"package test

fn f(selidx: bits[2] id=1, prio: bits[2] id=2, oh: bits[3] id=3, a: bits[8] id=4, b: bits[8] id=5, c: bits[8] id=6, d: bits[8] id=7) -> (bits[8], bits[8], bits[8]) {
  s: bits[8] = sel(selidx, cases=[a, b, c], default=d, id=10)
  p: bits[8] = priority_sel(prio, cases=[a, b], default=d, id=11)
  o: bits[8] = one_hot_sel(oh, cases=[a, b, c], id=12)
  ret t: (bits[8], bits[8], bits[8]) = tuple(s, p, o, id=13)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        let args = [
            IrValue::make_ubits(2, 1).unwrap(),     // selidx -> case 1
            IrValue::make_ubits(2, 2).unwrap(),     // prio bit1 set -> case 1
            IrValue::make_ubits(3, 0b101).unwrap(), // onehot bits 0 and 2 set
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 20).unwrap(),
            IrValue::make_ubits(8, 30).unwrap(),
            IrValue::make_ubits(8, 40).unwrap(),
        ];

        let r0 = eval_fn(&f, &args);
        let r1 = eval_fn_with_observer(&f, &args, None);
        assert_eq!(r0, r1);
    }

    #[test]
    fn test_select_observer_emits_expected_events() {
        let ir_text = r#"package test

fn f(selidx: bits[2] id=1, prio: bits[2] id=2, oh: bits[3] id=3, a: bits[8] id=4, b: bits[8] id=5, c: bits[8] id=6, d: bits[8] id=7) -> (bits[8], bits[8], bits[8]) {
  s: bits[8] = sel(selidx, cases=[a, b, c], default=d, id=10)
  p: bits[8] = priority_sel(prio, cases=[a, b], default=d, id=11)
  o: bits[8] = one_hot_sel(oh, cases=[a, b, c], id=12)
  ret t: (bits[8], bits[8], bits[8]) = tuple(s, p, o, id=13)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        let args = [
            IrValue::make_ubits(2, 1).unwrap(),     // selidx -> case 1
            IrValue::make_ubits(2, 2).unwrap(),     // prio bit1 set -> case 1
            IrValue::make_ubits(3, 0b101).unwrap(), // onehot bits 0 and 2 set
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 20).unwrap(),
            IrValue::make_ubits(8, 30).unwrap(),
            IrValue::make_ubits(8, 40).unwrap(),
        ];

        let mut obs = RecordingObserver::new();
        let _ = eval_fn_with_observer(&f, &args, Some(&mut obs));

        let got: Vec<(usize, SelectKind, usize)> = obs
            .events
            .iter()
            .map(|e| (e.node_text_id, e.select_kind, e.selected_index))
            .collect();
        let want: Vec<(usize, SelectKind, usize)> = vec![
            (10, SelectKind::CaseIndex, 1),
            (11, SelectKind::CaseIndex, 1),
            (12, SelectKind::CaseIndex, 0),
            (12, SelectKind::CaseIndex, 2),
            (12, SelectKind::MultiBitsSet, usize::MAX),
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn test_bool_node_observer_excludes_params_and_is_in_topo_order() {
        let ir_text = r#"package test

fn f(x: bits[2] id=1, y: bits[2] id=2) -> bits[1] {
  t: bits[1] = eq(x, y, id=10)
  ret n: bits[1] = not(t, id=11)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let args = vec![
            IrValue::make_ubits(2, 1).unwrap(),
            IrValue::make_ubits(2, 1).unwrap(),
        ];

        let mut obs = RecordingBoolObserver::new();
        let _ = eval_fn_with_observer(&f, &args, Some(&mut obs));

        // Only computed bits[1] nodes (excluding GetParam) should be observed, in topo
        // order: eq first, then not.
        let want = vec![(10, true), (11, false)];
        assert_eq!(obs.bool_events, want);
    }

    #[test]
    fn test_node_value_observer_excludes_params_and_is_in_topo_order() {
        let ir_text = r#"package test

fn f(x: bits[32] id=1) -> bits[32] {
  literal.2: bits[32] = literal(value=1, id=2)
  ret add.3: bits[32] = add(x, literal.2, id=3)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().expect("parse ok");
        let f = pkg.get_fn("f").expect("f").clone();
        let args = vec![IrValue::make_ubits(32, 2).unwrap()];

        let mut obs = RecordingNodeValueObserver::new();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Success(_)));

        // Only computed nodes (excluding GetParam) should be observed, in topo order:
        // literal first, then add.
        let got = obs.lines.join("\n") + "\n";
        let want = "node_text_id=2 value=bits[32]:1\nnode_text_id=3 value=bits[32]:3\n";
        assert_eq!(got, want);
    }

    #[test]
    fn test_eval_fn_in_package_invoke_success() {
        let ir_text = r#"package test

fn g(x: bits[8] id=1) -> bits[8] {
  literal.2: bits[8] = literal(value=1, id=2)
  ret add.3: bits[8] = add(x, literal.2, id=3)
}

fn f(x: bits[8] id=10) -> bits[8] {
  ret invoke.11: bits[8] = invoke(x, to_apply=g, id=11)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = pkg.get_fn("f").expect("f");
        let args = vec![IrValue::make_ubits(8, 5).unwrap()];
        let res = eval_fn_in_package(&pkg, f, &args);
        match res {
            FnEvalResult::Success(success) => {
                assert_eq!(success.value, IrValue::make_ubits(8, 6).unwrap());
                assert!(success.trace_messages.is_empty());
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_fn_in_package_invoke_propagates_trace() {
        let ir_text = r#"package test

fn g(x: bits[8] id=1) -> bits[8] {
  t: token = after_all(id=2)
  literal.3: bits[1] = literal(value=1, id=3)
  trace.4: token = trace(t, literal.3, format="in g x={}", data_operands=[x], id=4)
  ret identity.5: bits[8] = identity(x, id=5)
}

fn f(x: bits[8] id=10) -> bits[8] {
  ret invoke.11: bits[8] = invoke(x, to_apply=g, id=11)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = pkg.get_fn("f").expect("f");
        let args = vec![IrValue::make_ubits(8, 7).unwrap()];
        let res = eval_fn_in_package(&pkg, f, &args);
        match res {
            FnEvalResult::Success(success) => {
                assert_eq!(success.value, IrValue::make_ubits(8, 7).unwrap());
                let want = vec![TraceMessage {
                    message: "in g x=bits[8]:7".to_string(),
                    verbosity: 0,
                }];
                assert_eq!(success.trace_messages, want);
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_encode_matches_xlsynth_interpreter() {
        // Note: package-level validation requires ids to be unique across functions, so
        // we keep a single function here.
        let ir_text = r#"package test

fn f(x: bits[8] id=1) -> bits[3] {
  ret encode.2: bits[3] = encode(x, id=2)
}
"#;

        // Parse via xlsynth (reference) and PIR.
        let xls_pkg = xlsynth::IrPackage::parse_ir(ir_text, None).expect("xls parse");
        let xls_f = xls_pkg.get_function("f").expect("xls f");

        let mut pir_parser = Parser::new(ir_text);
        let pir_pkg = pir_parser
            .parse_and_validate_package()
            .expect("pir parse+validate");
        let pir_f = pir_pkg.get_fn("f").expect("pir f");

        // A handful of inputs, including zero, one-hot, and multi-hot.
        //
        // XLS encode behavior: bitwise-OR the indices of all set bits.
        let cases: &[(u64, u64)] = &[
            (0, 0),
            (1, 0),           // bit 0 -> 0
            (1 << 3, 3),      // bit 3 -> 3
            (1 << 7, 7),      // bit 7 -> 7
            (0b1010, 3),      // bits 1 and 3 -> 1|3=3
            (0b1000_1000, 7), // bits 3 and 7 -> 3|7=7
        ];

        for (x, want) in cases {
            let x = IrValue::make_ubits(8, *x).unwrap();
            let want_v = IrValue::make_ubits(3, *want).unwrap();
            let args = vec![x.clone()];
            let xls_got = xls_f.interpret(&args).expect("xls interpret");
            let pir_got = match eval_fn_in_package(&pir_pkg, pir_f, &args) {
                FnEvalResult::Success(s) => s.value,
                other => panic!("expected success, got {:?}", other),
            };
            assert_eq!(pir_got, xls_got, "x={}", x);
            assert_eq!(pir_got, want_v, "x={}", x);
        }
    }

    #[test]
    fn test_encode_after_umul_matches_xlsynth_interpreter() {
        // This is the minimal CI-discovered shape: `umul` feeding into `encode`.
        //
        // Pick x=6 so umul(x,x) == 36 == 0b100100, which has bits set at indices 5 and
        // 2. XLS encode semantics (and xlsynth interpreter behavior) ORs those
        // indices: 5|2 == 7.
        let ir_text = r#"package test

fn f(x: bits[6] id=1) -> bits[3] {
  umul.2: bits[6] = umul(x, x, id=2)
  ret encode.3: bits[3] = encode(umul.2, id=3)
}
"#;

        let xls_pkg = xlsynth::IrPackage::parse_ir(ir_text, None).expect("xls parse");
        let xls_f = xls_pkg.get_function("f").expect("xls f");

        let mut pir_parser = Parser::new(ir_text);
        let pir_pkg = pir_parser
            .parse_and_validate_package()
            .expect("pir parse+validate");
        let pir_f = pir_pkg.get_fn("f").expect("pir f");

        let args = vec![IrValue::make_ubits(6, 6).unwrap()];
        let xls_got = xls_f.interpret(&args).expect("xls interpret");
        let pir_got = match eval_fn_in_package(&pir_pkg, pir_f, &args) {
            FnEvalResult::Success(s) => s.value,
            other => panic!("expected success, got {:?}", other),
        };
        let want = IrValue::make_ubits(3, 7).unwrap();
        assert_eq!(pir_got, xls_got, "x={}", args[0]);
        assert_eq!(pir_got, want, "x={}", args[0]);
    }

    #[test]
    fn test_eval_fn_oob_bit_slice_early_fail() {
        // OOB static bit_slice: start=3, width=1 on a bits[3] value.
        let ir_text = r#"package test

fn f(x: bits[3] id=1) -> bits[1] {
  ret bit_slice.2: bits[1] = bit_slice(x, start=3, width=1, id=2)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        let x = IrValue::make_ubits(3, 0).unwrap();
        let res = eval_fn(&f, &[x]);
        match res {
            FnEvalResult::Failure(fail) => {
                // No assertion failures expected; this is an early-return guard.
                assert!(fail.assertion_failures.is_empty());
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_pure_array_update_no_indices_replaces_entire_array() {
        // Base array: [0,1,2], new array: [9,9,9], indices=[] => result=new array
        let base = IrValue::make_array(&[
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
        ])
        .unwrap();
        let new_arr = IrValue::make_array(&[
            IrValue::make_ubits(8, 9).unwrap(),
            IrValue::make_ubits(8, 9).unwrap(),
            IrValue::make_ubits(8, 9).unwrap(),
        ])
        .unwrap();
        let env = hashmap!(
            ir::NodeRef { index: 1 } => base,
            ir::NodeRef { index: 2 } => new_arr.clone(),
        );
        let n = ir::Node {
            text_id: 100,
            name: None,
            ty: ir::Type::new_array(ir::Type::Bits(8), 3),
            payload: ir::NodePayload::ArrayUpdate {
                array: ir::NodeRef { index: 1 },
                value: ir::NodeRef { index: 2 },
                indices: vec![],
                assumed_in_bounds: true,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        assert_eq!(v, new_arr);
    }

    #[test]
    fn test_eval_pure_array_update_one_index() {
        // Base array: [0,1,2], update index 1 with 99 => [0,99,2]
        let base = IrValue::make_array(&[
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
        ])
        .unwrap();
        let env = hashmap!(
            ir::NodeRef { index: 1 } => base,
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 99).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(32, 1).unwrap(),
        );
        let n = ir::Node {
            text_id: 101,
            name: None,
            ty: ir::Type::new_array(ir::Type::Bits(8), 3),
            payload: ir::NodePayload::ArrayUpdate {
                array: ir::NodeRef { index: 1 },
                value: ir::NodeRef { index: 2 },
                indices: vec![ir::NodeRef { index: 3 }],
                assumed_in_bounds: true,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        let expected = IrValue::make_array(&[
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(8, 99).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
        ])
        .unwrap();
        assert_eq!(v, expected);
    }

    #[test]
    fn test_eval_pure_array_update_two_indices() {
        // Base 2D: [[0,1],[2,3]], update [1][0] with 55 => [[0,1],[55,3]]
        let row0 = IrValue::make_array(&[
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
        ])
        .unwrap();
        let row1 = IrValue::make_array(&[
            IrValue::make_ubits(8, 2).unwrap(),
            IrValue::make_ubits(8, 3).unwrap(),
        ])
        .unwrap();
        let base2d = IrValue::make_array(&[row0, row1]).unwrap();
        let env = hashmap!(
            ir::NodeRef { index: 1 } => base2d,
            ir::NodeRef { index: 2 } => IrValue::make_ubits(8, 55).unwrap(),
            ir::NodeRef { index: 3 } => IrValue::make_ubits(32, 1).unwrap(),
            ir::NodeRef { index: 4 } => IrValue::make_ubits(32, 0).unwrap(),
        );
        let n = ir::Node {
            text_id: 102,
            name: None,
            ty: ir::Type::new_array(ir::Type::new_array(ir::Type::Bits(8), 2), 2),
            payload: ir::NodePayload::ArrayUpdate {
                array: ir::NodeRef { index: 1 },
                value: ir::NodeRef { index: 2 },
                indices: vec![ir::NodeRef { index: 3 }, ir::NodeRef { index: 4 }],
                assumed_in_bounds: true,
            },
            pos: None,
        };
        let v: IrValue = eval_pure(&n, &env);
        let expected = IrValue::make_array(&[
            IrValue::make_array(&[
                IrValue::make_ubits(8, 0).unwrap(),
                IrValue::make_ubits(8, 1).unwrap(),
            ])
            .unwrap(),
            IrValue::make_array(&[
                IrValue::make_ubits(8, 55).unwrap(),
                IrValue::make_ubits(8, 3).unwrap(),
            ])
            .unwrap(),
        ])
        .unwrap();
        assert_eq!(v, expected);
    }

    #[test]
    fn test_eval_fn_array_update_in_bounds() {
        let ir_text = r#"package test

fn f(a: bits[5][4] id=1, v: bits[5] id=2, i: bits[32] id=3) -> bits[5][4] {
  ret array_update.4: bits[5][4] = array_update(a, v, indices=[i], id=4)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        // a = [1,2,3,4], v = 9, i = 2 (in-bounds)
        let a = IrValue::make_array(&[
            IrValue::make_ubits(5, 1).unwrap(),
            IrValue::make_ubits(5, 2).unwrap(),
            IrValue::make_ubits(5, 3).unwrap(),
            IrValue::make_ubits(5, 4).unwrap(),
        ])
        .unwrap();
        let v = IrValue::make_ubits(5, 9).unwrap();
        let i = IrValue::make_ubits(32, 2).unwrap();

        match eval_fn(&f, &[a.clone(), v.clone(), i]) {
            FnEvalResult::Success(s) => {
                let got = s.value;
                let expected = IrValue::make_array(&[
                    IrValue::make_ubits(5, 1).unwrap(),
                    IrValue::make_ubits(5, 2).unwrap(),
                    v,
                    IrValue::make_ubits(5, 4).unwrap(),
                ])
                .unwrap();
                assert_eq!(got, expected);
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_fn_array_update_oob_returns_original_when_not_assumed() {
        let ir_text = r#"package test

fn f(a: bits[5][4] id=1, v: bits[5] id=2, i: bits[32] id=3) -> bits[5][4] {
  ret array_update.4: bits[5][4] = array_update(a, v, indices=[i], id=4)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        // a = [1,2,3,4], v = 9, i = 5 (OOB) => unchanged when not assumed_in_bounds
        let a = IrValue::make_array(&[
            IrValue::make_ubits(5, 1).unwrap(),
            IrValue::make_ubits(5, 2).unwrap(),
            IrValue::make_ubits(5, 3).unwrap(),
            IrValue::make_ubits(5, 4).unwrap(),
        ])
        .unwrap();
        let v = IrValue::make_ubits(5, 9).unwrap();
        let i = IrValue::make_ubits(32, 5).unwrap();

        match eval_fn(&f, &[a.clone(), v, i]) {
            FnEvalResult::Success(s) => {
                assert_eq!(s.value, a);
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_fn_array_update_oob_failure_when_assumed() {
        let ir_text = r#"package test

fn f(a: bits[5][4] id=1, v: bits[5] id=2, i: bits[32] id=3) -> bits[5][4] {
  ret array_update.4: bits[5][4] = array_update(a, v, indices=[i], assumed_in_bounds=true, id=4)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        // OOB index with assumed_in_bounds=true => Failure
        let a = IrValue::make_array(&[
            IrValue::make_ubits(5, 1).unwrap(),
            IrValue::make_ubits(5, 2).unwrap(),
            IrValue::make_ubits(5, 3).unwrap(),
            IrValue::make_ubits(5, 4).unwrap(),
        ])
        .unwrap();
        let v = IrValue::make_ubits(5, 9).unwrap();
        let i = IrValue::make_ubits(32, 7).unwrap();

        match eval_fn(&f, &[a, v, i]) {
            FnEvalResult::Failure(_fail) => {
                // Early failure as expected
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_fn_array_update_nested_indices() {
        let ir_text = r#"package test

fn f(a: bits[3][2][2] id=1, v: bits[3] id=2, i: bits[32] id=3, j: bits[32] id=4) -> bits[3][2][2] {
  ret array_update.5: bits[3][2][2] = array_update(a, v, indices=[i, j], id=5)
}
"#;
        let mut p = Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse ok");
        let f = match &pkg.members[0] {
            ir::PackageMember::Function(f) => f.clone(),
            _ => unreachable!(),
        };

        // a = [[1,2],[3,4]] (bits[3]); update a[1][0] = 7
        let row0 = IrValue::make_array(&[
            IrValue::make_ubits(3, 1).unwrap(),
            IrValue::make_ubits(3, 2).unwrap(),
        ])
        .unwrap();
        let row1 = IrValue::make_array(&[
            IrValue::make_ubits(3, 3).unwrap(),
            IrValue::make_ubits(3, 4).unwrap(),
        ])
        .unwrap();
        let a = IrValue::make_array(&[row0, row1]).unwrap();
        let v = IrValue::make_ubits(3, 7).unwrap();
        let i = IrValue::make_ubits(32, 1).unwrap();
        let j = IrValue::make_ubits(32, 0).unwrap();

        match eval_fn(&f, &[a, v, i, j]) {
            FnEvalResult::Success(s) => {
                let got = s.value;
                let exp_row0 = IrValue::make_array(&[
                    IrValue::make_ubits(3, 1).unwrap(),
                    IrValue::make_ubits(3, 2).unwrap(),
                ])
                .unwrap();
                let exp_row1 = IrValue::make_array(&[
                    IrValue::make_ubits(3, 7).unwrap(),
                    IrValue::make_ubits(3, 4).unwrap(),
                ])
                .unwrap();
                let expected = IrValue::make_array(&[exp_row0, exp_row1]).unwrap();
                assert_eq!(got, expected);
            }
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_eval_fn_array_index_clamps_oob_when_not_assumed() {
        let ir_text = r#"package test

fn f(a: bits[8][4] id=1, i: bits[3] id=2) -> bits[8] {
  ret out: bits[8] = array_index(a, indices=[i], assumed_in_bounds=false, id=3)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let arr = IrValue::make_array(&[
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 11).unwrap(),
            IrValue::make_ubits(8, 12).unwrap(),
            IrValue::make_ubits(8, 13).unwrap(),
        ])
        .unwrap();
        let args = vec![arr, IrValue::make_ubits(3, 7).unwrap()];
        let r = eval_fn_with_observer(&f, &args, None);
        match r {
            FnEvalResult::Success(s) => {
                assert_eq!(s.value, IrValue::make_ubits(8, 13).unwrap());
            }
            FnEvalResult::Failure(_) => panic!("expected success"),
        }
    }

    #[test]
    fn test_eval_fn_array_index_oob_failure_when_assumed() {
        let ir_text = r#"package test

fn f(a: bits[8][4] id=1, i: bits[3] id=2) -> bits[8] {
  ret out: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=3)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let arr = IrValue::make_array(&[
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 11).unwrap(),
            IrValue::make_ubits(8, 12).unwrap(),
            IrValue::make_ubits(8, 13).unwrap(),
        ])
        .unwrap();
        let args = vec![arr, IrValue::make_ubits(3, 7).unwrap()];
        let r = eval_fn_with_observer(&f, &args, None);
        match r {
            FnEvalResult::Success(_) => panic!("expected failure"),
            FnEvalResult::Failure(_f) => {}
        }
    }

    #[test]
    fn test_corner_add_rhs_is_zero_emits_event() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret z: bits[8] = add(x, y, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let args = vec![
            IrValue::make_ubits(8, 7).unwrap(),
            IrValue::make_ubits(8, 0).unwrap(),
        ];
        let mut obs = RecordingCornerFailureObserver::default();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Success(_)));
        assert_eq!(obs.corner_events, vec![(10, CornerKind::Add, 1)]);
    }

    #[test]
    fn test_corner_neg_min_signed_emits_event() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1) -> bits[8] {
  ret z: bits[8] = neg(x, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let args = vec![IrValue::make_ubits(8, 0x80).unwrap()];
        let mut obs = RecordingCornerFailureObserver::default();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Success(_)));
        assert_eq!(
            obs.corner_events,
            vec![(10, CornerKind::Neg, 1), (10, CornerKind::Neg, 0)]
        );
    }

    #[test]
    fn test_corner_neg_msb_one_emits_event_without_min_signed() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1) -> bits[8] {
  ret z: bits[8] = neg(x, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let args = vec![IrValue::make_ubits(8, 0x81).unwrap()];
        let mut obs = RecordingCornerFailureObserver::default();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Success(_)));
        assert_eq!(obs.corner_events, vec![(10, CornerKind::Neg, 1)]);
    }

    #[test]
    fn test_corner_sign_ext_msb_zero_emits_event() {
        let ir_text = r#"package test

fn f(x: bits[4] id=1) -> bits[8] {
  ret z: bits[8] = sign_ext(x, new_bit_count=8, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let args = vec![IrValue::make_ubits(4, 0b0011).unwrap()];
        let mut obs = RecordingCornerFailureObserver::default();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Success(_)));
        assert_eq!(obs.corner_events, vec![(10, CornerKind::SignExt, 0)]);
    }

    #[test]
    fn test_corner_compare_distance_buckets_xor_popcount() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  ret t: bits[1] = eq(x, y, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();

        let mut obs0 = RecordingCornerFailureObserver::default();
        let r0 = eval_fn_with_observer(
            &f,
            &[
                IrValue::make_ubits(8, 0xAA).unwrap(),
                IrValue::make_ubits(8, 0xAA).unwrap(),
            ],
            Some(&mut obs0),
        );
        assert!(matches!(r0, FnEvalResult::Success(_)));
        assert_eq!(
            obs0.corner_events,
            vec![(10, CornerKind::CompareDistance, 0)]
        );

        let mut obs1 = RecordingCornerFailureObserver::default();
        let r1 = eval_fn_with_observer(
            &f,
            &[
                IrValue::make_ubits(8, 0xAA).unwrap(),
                IrValue::make_ubits(8, 0xAB).unwrap(),
            ],
            Some(&mut obs1),
        );
        assert!(matches!(r1, FnEvalResult::Success(_)));
        assert_eq!(
            obs1.corner_events,
            vec![(10, CornerKind::CompareDistance, 1)]
        );

        let mut obs5 = RecordingCornerFailureObserver::default();
        let r5 = eval_fn_with_observer(
            &f,
            &[
                IrValue::make_ubits(8, 0).unwrap(),
                IrValue::make_ubits(8, 0b0001_1111).unwrap(),
            ],
            Some(&mut obs5),
        );
        assert!(matches!(r5, FnEvalResult::Success(_)));
        assert_eq!(
            obs5.corner_events,
            vec![(10, CornerKind::CompareDistance, 5)]
        );
    }

    #[test]
    fn test_corner_dynamic_bit_slice_in_bounds_vs_oob() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1, s: bits[3] id=2) -> bits[4] {
  ret z: bits[4] = dynamic_bit_slice(x, s, width=4, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();

        // In bounds: start=2, width=4, arg_width=8
        let args_in = vec![
            IrValue::make_ubits(8, 0b1111_0000).unwrap(),
            IrValue::make_ubits(3, 2).unwrap(),
        ];
        let mut obs_in = RecordingCornerFailureObserver::default();
        let r_in = eval_fn_with_observer(&f, &args_in, Some(&mut obs_in));
        assert!(matches!(r_in, FnEvalResult::Success(_)));
        assert_eq!(
            obs_in.corner_events,
            vec![(10, CornerKind::DynamicBitSlice, 0)]
        );
        assert!(obs_in.failure_events.is_empty());

        // OOB: start=6, width=4, arg_width=8
        let args_oob = vec![
            IrValue::make_ubits(8, 0b1111_0000).unwrap(),
            IrValue::make_ubits(3, 6).unwrap(),
        ];
        let mut obs_oob = RecordingCornerFailureObserver::default();
        let r_oob = eval_fn_with_observer(&f, &args_oob, Some(&mut obs_oob));
        assert!(matches!(r_oob, FnEvalResult::Failure(_)));
        assert_eq!(
            obs_oob.corner_events,
            vec![(10, CornerKind::DynamicBitSlice, 1)]
        );
        assert_eq!(
            obs_oob.failure_events,
            vec![(10, FailureKind::DynamicBitSliceOob)]
        );
    }

    #[test]
    fn test_corner_array_index_clamped_and_failure_when_assumed() {
        let ir_text = r#"package test

fn f(a: bits[8][4] id=1, i: bits[3] id=2) -> bits[8] {
  ret out: bits[8] = array_index(a, indices=[i], assumed_in_bounds=false, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let arr = IrValue::make_array(&[
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 11).unwrap(),
            IrValue::make_ubits(8, 12).unwrap(),
            IrValue::make_ubits(8, 13).unwrap(),
        ])
        .unwrap();
        let args = vec![arr.clone(), IrValue::make_ubits(3, 7).unwrap()];
        let mut obs = RecordingCornerFailureObserver::default();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Success(_)));
        assert_eq!(obs.corner_events, vec![(10, CornerKind::ArrayIndex, 1)]);
        assert!(obs.failure_events.is_empty());

        let ir_text_assumed = r#"package test

fn f(a: bits[8][4] id=1, i: bits[3] id=2) -> bits[8] {
  ret out: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=10)
}
"#;
        let mut parser = Parser::new(ir_text_assumed);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap().clone();
        let args = vec![arr, IrValue::make_ubits(3, 7).unwrap()];
        let mut obs = RecordingCornerFailureObserver::default();
        let r = eval_fn_with_observer(&f, &args, Some(&mut obs));
        assert!(matches!(r, FnEvalResult::Failure(_)));
        assert!(obs.corner_events.is_empty());
        assert_eq!(
            obs.failure_events,
            vec![(10, FailureKind::ArrayIndexOobAssumedInBounds)]
        );
    }
}
