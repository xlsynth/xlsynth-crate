// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::ir;
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
                _ => panic!("Unsupported binop: {:?}", binop),
            }
        }
        ir::NodePayload::Unop(unop, ref operand) => {
            let operand_value: &IrValue = env.get(operand).unwrap();
            let operand_bits = operand_value.to_bits().unwrap();
            match unop {
                ir::Unop::Neg => {
                    let r = operand_bits.negate();
                    IrValue::from_bits(&r)
                }
                ir::Unop::Not => {
                    let r = operand_bits.not();
                    IrValue::from_bits(&r)
                }
                ir::Unop::Identity => operand_value.clone(),
                ir::Unop::OrReduce => {
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
                    let mut result = false;
                    for i in 0..operand_bits.get_bit_count() {
                        if operand_bits.get_bit(i).unwrap() {
                            result = !result;
                        }
                    }
                    IrValue::bool(result)
                }
                ir::Unop::Reverse => {
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
        ir::NodePayload::ArrayIndex {
            array,
            ref indices,
            assumed_in_bounds: _,
        } => {
            let mut value = env.get(&array).unwrap().clone();
            for idx_ref in indices {
                let idx = env.get(idx_ref).unwrap().to_u64().unwrap() as usize;
                value = value.get_element(idx).unwrap();
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
                ir::NaryOp::Concat => panic!("Unsupported nary op: concat"),
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

/// Evaluates an IR function by visiting nodes in topological order.
///
/// - Produces `TraceMessage`s for `trace` nodes whose `activated` predicate is
///   true.
/// - Records `AssertionFailure`s for `assert` nodes whose `activate` predicate
///   is true.
/// - Returns the value of the function's return node on success.
pub fn eval_fn(f: &ir::Fn, args: &[IrValue]) -> FnEvalResult {
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
        use ir::NodePayload as P;
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
            P::BitSlice { arg, start, width } => {
                // Guard against OOB; return Failure instead of panicking.
                let arg_bits = env
                    .get(arg)
                    .expect("arg must be evaluated")
                    .to_bits()
                    .unwrap();
                let bit_count = arg_bits.get_bit_count();
                if start + width > bit_count {
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
                    return FnEvalResult::Failure(FnEvalFailure {
                        assertion_failures,
                        trace_messages,
                    });
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
            _ => eval_pure(node, &env),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;
    use maplit::hashmap;

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
  t.2: token = after_all(id=2)
  _tr.3: token = trace(t.2, x, format="x={} done", data_operands=[x], id=3)
  _a.4: token = assert(t.2, x, message="boom", label="L", id=4)
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
}
