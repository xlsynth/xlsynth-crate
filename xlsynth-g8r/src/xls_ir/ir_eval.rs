// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::xls_ir::ir;
use xlsynth::{IrBits, IrValue};

fn eval_pure(n: &ir::Node, env: &HashMap<ir::NodeRef, IrValue>) -> IrValue {
    match n.payload {
        ir::NodePayload::Literal(ref ir_value) => ir_value.clone(),
        ir::NodePayload::Binop(binop, ref lhs, ref rhs) => {
            let lhs_value: &IrValue = env.get(lhs).unwrap();
            let rhs_value: &IrValue = env.get(rhs).unwrap();
            let lhs_bits: IrBits = lhs_value.to_bits().unwrap();
            let rhs_bits: IrBits = rhs_value.to_bits().unwrap();
            match binop {
                ir::Binop::Add => {
                    let r = lhs_bits.add(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Sub => {
                    let r = lhs_bits.sub(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Shll => {
                    let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
                    let r = lhs_bits.shll(shift);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Shrl => {
                    let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
                    let r = lhs_bits.shrl(shift);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Shra => {
                    let shift: i64 = rhs_bits.to_u64().unwrap().try_into().unwrap();
                    let r = lhs_bits.shra(shift);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Smulp | ir::Binop::Smul => {
                    let r = lhs_bits.smul(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Umulp | ir::Binop::Umul => {
                    let r = lhs_bits.umul(&rhs_bits);
                    IrValue::from_bits(&r)
                }
                ir::Binop::Eq => IrValue::bool(lhs_bits.equals(&rhs_bits)),
                ir::Binop::Ne => IrValue::bool(!lhs_bits.equals(&rhs_bits)),
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
                ir::Unop::Reverse => panic!("Unsupported unop: reverse"),
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
            let start_i = start_bits.to_u64().unwrap() as i64;
            let r = arg_bits.width_slice(start_i, width as i64);
            IrValue::from_bits(&r)
        }
        ir::NodePayload::BitSlice {
            ref arg,
            start,
            width,
        } => {
            let arg_bits: IrBits = env.get(arg).unwrap().to_bits().unwrap();
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
        ir::NodePayload::GetParam(..) | _ => panic!("Cannot evaluate node as pure: {:?}", n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maplit::hashmap;

    #[test]
    fn test_eval_pure_literal() {
        let mut env = HashMap::new();
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
}
