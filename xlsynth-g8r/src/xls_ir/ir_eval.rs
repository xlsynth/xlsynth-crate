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
                ir::Binop::Add => lhs_bits.add(&rhs_bits).into(),
                ir::Binop::Sub => lhs_bits.sub(&rhs_bits).into(),
                ir::Binop::Shll => lhs_bits.shll(rhs_bits.to_u64().unwrap()).into(),
                ir::Binop::Shrl => lhs_bits.shrl(rhs_bits.to_u64().unwrap()).into(),
                ir::Binop::Shra => lhs_bits.shra(rhs_bits.to_u64().unwrap()).into(),
                ir::Binop::ArrayConcat => lhs_bits.concat(&rhs_bits).into(),
                ir::Binop::Smulp => lhs_bits.smul(&rhs_bits).into(),
                ir::Binop::Umulp => lhs_bits.umul(&rhs_bits).into(),
                ir::Binop::Eq => lhs_bits.eq(&rhs_bits).into(),
                ir::Binop::Ne => lhs_bits.ne(&rhs_bits).into(),
                ir::Binop::Umul => lhs_bits.umul(&rhs_bits).into(),
                ir::Binop::Smul => lhs_bits.smul(&rhs_bits).into(),
                _ => panic!("Unsupported binop: {:?}", binop),
            }
        }
        ir::NodePayload::GetParam(..) | _ => panic!("Cannot evaluate node as pure: {:?}", n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
